from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Optional, Sequence, Tuple

import torch

from src.evaluate import is_valid_hand
from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, NUM_FACES,
    GameConfig, Observation, Phase, discard_action, legal_action_mask,
)
from src.game.information import EventKind, InformationEvent
from src.model.encoding import collate_observations
from src.model.network import QNetwork, load_checkpoint, select_greedy_action


@dataclass(frozen=True)
class BeliefState:
    unseen_counts: Tuple[int, ...]
    stock_weights: Tuple[int, ...]
    known_held: Tuple[Tuple[int, ...], ...]
    opp_wanted: Tuple[Tuple[float, ...], ...]
    opp_safe: Tuple[Tuple[int, ...], ...]
    take_rate: Tuple[float, ...]
    joker: int = 0

    @classmethod
    def from_observation(cls, obs: Observation) -> "BeliefState":
        count = len(obs.players)
        held = [[0] * NUM_FACES for _ in range(count)]
        safe = [[0] * NUM_FACES for _ in range(count)]
        takes = [0] * count
        passes = [0] * count
        pile = []
        refill_pool = None
        for event in obs.history:
            if event.kind is EventKind.INITIAL:
                pile = [] if event.offered_top is None else [event.offered_top]
            elif event.kind is EventKind.TAKE_DISCARD:
                pile.pop()
                if event.actor != 0:
                    held[event.actor][event.face] += 1
                    takes[event.actor] += 1
            elif event.kind is EventKind.DISCARD:
                pile.append(event.face)
                if event.actor != 0:
                    held[event.actor][event.face] = max(0, held[event.actor][event.face] - 1)
                    safe[event.actor][event.face] += 1
            elif event.kind is EventKind.DRAW_STOCK:
                if refill_pool is not None and event.face is not None:
                    refill_pool[event.face] = max(0, refill_pool[event.face] - 1)
                if event.actor != 0 and event.offered_top is not None:
                    safe[event.actor][event.offered_top] += 1
                    passes[event.actor] += 1
            elif event.kind is EventKind.REFILL:
                refill_pool = [0] * NUM_FACES
                for face in pile[:-1]:
                    refill_pool[face] += 1
                pile = pile[-1:]

        pile_counts = [0] * NUM_FACES
        for face in obs.discard_pile:
            pile_counts[face] += 1
        unseen = tuple(
            max(0, obs.config.num_decks - obs.hand[face] - pile_counts[face]
                - int(face == obs.joker) - sum(seat[face] for seat in held))
            for face in range(NUM_FACES)
        )
        weights = unseen if refill_pool is None else tuple(
            min(refill_pool[face], unseen[face]) for face in range(NUM_FACES)
        )
        if not sum(weights):
            weights = unseen

        wanted = []
        for seat in held:
            row = [0.0] * NUM_FACES
            for known, copies in enumerate(seat):
                if not copies:
                    continue
                for face in range(NUM_FACES):
                    rank, known_rank = face % 13, known % 13
                    if face == known or rank == obs.joker % 13 or known_rank == obs.joker % 13:
                        weight = 1.0
                    elif face // 13 == known // 13:
                        distance = abs(rank - known_rank)
                        if rank == 0 or known_rank == 0:
                            distance = min(distance, 13 - distance)
                        weight = 0.6 if distance == 1 else 0.3 if distance == 2 else 0.0
                    else:
                        weight = 0.5 if rank == known_rank else 0.0
                    row[face] = max(row[face], weight)
            wanted.append(tuple(row))
        return cls(
            unseen, weights, tuple(tuple(seat) for seat in held), tuple(wanted),
            tuple(tuple(seat) for seat in safe),
            tuple((1 + take) / (5 + take + passed) for take, passed in zip(takes, passes)),
            obs.joker,
        )

    def stock_probabilities(self) -> Tuple[Tuple[int, float], ...]:
        total = sum(self.stock_weights)
        return tuple(
            (face, weight / total)
            for face, weight in enumerate(self.stock_weights) if weight > 0
        )

    def feed_risk(self, face: int, seat: int = 1) -> float:
        if seat >= len(self.opp_safe) or self.opp_safe[seat][face] > 0:
            return 0.0
        return self.take_rate[seat] * self.opp_wanted[seat][face]


def can_declare_ub(hand: Sequence[int], joker: int, required_sequences: int) -> bool:
    upper = hand[joker]
    if upper >= required_sequences:
        return True
    for suit in range(4):
        rem = list(hand[suit * 13:(suit + 1) * 13])
        if joker // 13 == suit:
            rem[joker % 13] = 0
        triples = sum(copies // 3 for copies in rem)
        extended = rem + rem[:1]
        triples += sum(min(extended[start:start + 3]) for start in range(12))
        upper += min(sum(rem) // 3, triples)
    return upper >= required_sequences


def winning_discard(
    hand: Sequence[int], joker: int, required_sequences: int = 5, cards_in_hand: int = 21,
) -> Optional[int]:
    if sum(hand) != cards_in_hand + 1 or not can_declare_ub(hand, joker, required_sequences):
        return None
    remaining = list(hand)
    for face, copies in enumerate(hand):
        if copies <= 0:
            continue
        remaining[face] -= 1
        wins = can_declare_ub(remaining, joker, required_sequences) and is_valid_hand(
            remaining, joker, required_sequences, cards_in_hand,
        )
        remaining[face] += 1
        if wins:
            return face
    return None


def outs_1ply(hand: Sequence[int], belief: BeliefState, config: GameConfig) -> float:
    if not can_declare_ub(hand, belief.joker, config.required_sequences - 1):
        return 0.0
    probability = 0.0
    drawn = list(hand)
    for face, chance in belief.stock_probabilities():
        drawn[face] += 1
        if winning_discard(
            drawn, belief.joker, config.required_sequences, config.cards_in_hand,
        ) is not None:
            probability += chance
        drawn[face] -= 1
    return probability


@dataclass(frozen=True)
class SearchConfig:
    exact_wins: bool = True
    draw_lookahead: bool = True
    rollout_depth: int = 2
    rollout_beam: int = 4
    feed_weight: float = 0.03
    outs_weight: float = 0.15

    def __post_init__(self) -> None:
        if type(self.exact_wins) is not bool or type(self.draw_lookahead) is not bool:
            raise ValueError("search switches must be bools")
        if type(self.rollout_depth) is not int or self.rollout_depth < 0:
            raise ValueError("rollout_depth must be a nonnegative integer")
        if type(self.rollout_beam) is not int or self.rollout_beam < 1:
            raise ValueError("rollout_beam must be a positive integer")
        for name in ("feed_weight", "outs_weight"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and nonnegative" % name)


@dataclass(frozen=True)
class SearchResult:
    action: int
    reason: str
    rows: int
    take_value: Optional[float] = None
    stock_value: Optional[float] = None


def _after_draw(obs: Observation, face: int, kind: EventKind) -> Observation:
    hand = list(obs.hand)
    hand[face] += 1
    take = kind is EventKind.TAKE_DISCARD
    stock = obs.stock_remaining - int(not take)
    player = replace(obs.players[0], hand_size=obs.players[0].hand_size + 1)
    event = InformationEvent(
        kind, 0, player.hand_size, obs.turns_remaining, stock,
        player.next_player, face=face, offered_top=obs.discard_top,
    )
    return replace(
        obs, hand=tuple(hand), discard_pile=obs.discard_pile[:-1] if take else obs.discard_pile,
        phase=Phase.DISCARD, stock_remaining=stock, players=(player, *obs.players[1:]),
        history=(*obs.history, event), last_reward=0.0,
    )


@torch.inference_mode()
def search(
    network: QNetwork,
    obs: Observation,
    config: SearchConfig = SearchConfig(),
    *,
    device: Optional[torch.device] = None,
) -> SearchResult:
    network.eval()
    mask = legal_action_mask(obs)
    legal = [action for action, allowed in enumerate(mask) if allowed]
    if not legal:
        raise ValueError("no legal actions")
    if len(legal) == 1:
        return SearchResult(legal[0], "forced", 0)
    if obs.phase is Phase.DISCARD:
        if config.exact_wins:
            face = winning_discard(
                obs.hand, obs.joker, obs.config.required_sequences, obs.config.cards_in_hand,
            )
            if face is not None:
                return SearchResult(discard_action(face), "exact_win", 0)
        if device is None:
            device = next(network.parameters()).device
        belief = BeliefState.from_observation(obs)
        values = network(collate_observations([obs], device))[0, 2:].tolist()
        faces = [action - 2 for action in legal]
        for face in faces:
            values[face] -= config.feed_weight * belief.feed_risk(face, 1)
        rows = 1
        if config.rollout_depth >= 2 and config.outs_weight > 0:
            candidates = sorted(faces, key=lambda face: (-values[face], face))[:config.rollout_beam]
            hand = list(obs.hand)
            for face in candidates:
                hand[face] -= 1
                values[face] += config.outs_weight * outs_1ply(hand, belief, obs.config)
                hand[face] += 1
            if config.rollout_depth >= 3 and len(candidates) >= 2 and obs.turns_remaining > 1:
                probs = belief.stock_probabilities()
                if probs:
                    top_draws = sorted(probs, key=lambda item: (-item[1], item[0]))[:12]
                    norm = sum(p for _, p in top_draws)
                    rollout_children = []
                    beam_cands = candidates[: min(3, len(candidates))]
                    for face in beam_cands:
                        base_hand = list(obs.hand)
                        base_hand[face] -= 1
                        post_disc = replace(
                            obs,
                            hand=tuple(base_hand),
                            discard_pile=(*obs.discard_pile, face),
                            phase=Phase.DRAW,
                            turns_remaining=obs.turns_remaining - 1,
                        )
                        for draw_f, _ in top_draws:
                            rollout_children.append(_after_draw(post_disc, draw_f, EventKind.DRAW_STOCK))
                    r_vals = network(collate_observations(rollout_children, device))[:, 2:]
                    r_allowed = torch.tensor(
                        [legal_action_mask(c)[2:] for c in rollout_children],
                        dtype=torch.bool, device=device,
                    )
                    r_best = r_vals.masked_fill(~r_allowed, -torch.inf).max(dim=1).values.tolist()
                    rows += len(rollout_children)
                    k = len(top_draws)
                    for idx, face in enumerate(beam_cands):
                        ev_next = sum(
                            (top_draws[j][1] / norm) * r_best[idx * k + j]
                            for j in range(k)
                        )
                        values[face] = 0.70 * values[face] + 0.30 * ev_next
        face = max(faces, key=lambda face: values[face])
        return SearchResult(discard_action(face), "discard_search", rows)

    if config.exact_wins and mask[ACTION_TAKE_DISCARD] and obs.discard_top is not None:
        hand = list(obs.hand)
        hand[obs.discard_top] += 1
        if winning_discard(
            hand, obs.joker, obs.config.required_sequences, obs.config.cards_in_hand,
        ) is not None:
            return SearchResult(ACTION_TAKE_DISCARD, "exact_win", 0)
    if (
        config.draw_lookahead and mask[ACTION_DRAW_STOCK]
        and mask[ACTION_TAKE_DISCARD] and obs.discard_top is not None
    ):
        belief = BeliefState.from_observation(obs)
        probs = belief.stock_probabilities()
        if probs:
            if device is None:
                device = next(network.parameters()).device
            children = [_after_draw(obs, obs.discard_top, EventKind.TAKE_DISCARD)] + [
                _after_draw(obs, face, EventKind.DRAW_STOCK) for face, _ in probs
            ]
            values = network(collate_observations(children, device))[:, 2:]
            penalties = torch.tensor(
                [config.feed_weight * belief.feed_risk(face, 1) for face in range(NUM_FACES)],
                dtype=values.dtype, device=device,
            )
            allowed = torch.tensor(
                [legal_action_mask(child)[2:] for child in children],
                dtype=torch.bool, device=device,
            )
            best = (values - penalties).masked_fill(~allowed, -torch.inf).max(dim=1).values
            take_value = float(best[0].item())
            stock_value = sum(
                chance * float(best[index + 1].item())
                for index, (_, chance) in enumerate(probs)
            )
            action = ACTION_TAKE_DISCARD if take_value > stock_value else ACTION_DRAW_STOCK
            return SearchResult(action, "draw_lookahead", len(children), take_value, stock_value)
    return SearchResult(select_greedy_action(network, obs, device=device), "greedy", 1)


def select_search_action(
    network: QNetwork,
    obs: Observation,
    config: SearchConfig = SearchConfig(),
    *,
    device: Optional[torch.device] = None,
) -> int:
    return search(network, obs, config, device=device).action


class SearchPolicy:
    def __init__(
        self, network: QNetwork, game_config: GameConfig,
        config: SearchConfig = SearchConfig(), *, device: Optional[torch.device] = None,
    ) -> None:
        self.network = network
        self.game_config = game_config
        self.config = config
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, path: str, config: SearchConfig = SearchConfig(), *,
        device: Optional[torch.device] = None,
    ) -> "SearchPolicy":
        network, game_config, _ = load_checkpoint(path, device=device)
        return cls(network, game_config, config, device=device)

    def act(self, observation: Observation, rng: random.Random) -> int:
        return select_search_action(self.network, observation, self.config, device=self.device)
