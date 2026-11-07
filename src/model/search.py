from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Optional, Sequence, Tuple

import torch

from src.evaluate import is_valid_hand, minimum_penalty
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


_PEN_CACHE = {}
_DECOMP_CACHE = {}
_PROTO_PAIRS = tuple([(r, r + 1) for r in range(12)] + [(12, 0)] + [(r, r + 2) for r in range(11)] + [(11, 0)])


def _count_residual_protos(rem, joker):
    wild_rank = joker % 13
    r_list = list(rem)
    for s in range(4):
        r_list[s * 13 + wild_rank] = 0
    seq_p = 0
    for s in range(4):
        base = s * 13
        for a, b_idx in _PROTO_PAIRS:
            while r_list[base + a] > 0 and r_list[base + b_idx] > 0:
                r_list[base + a] -= 1
                r_list[base + b_idx] -= 1
                seq_p += 1
    set_p = 0
    for r in range(13):
        if r == wild_rank:
            continue
        suits_present = (r_list[r] > 0) + (r_list[13 + r] > 0) + (r_list[26 + r] > 0) + (r_list[39 + r] > 0)
        if suits_present >= 2:
            set_p += 1
    return seq_p, set_p


def _score_from_parts(q_seq, impure_melds, rem_cards, joker, req, max_impure):
    seq_p, set_p = _count_residual_protos(rem_cards, joker)
    need_pure = max(0, req - q_seq)
    need_imp = max(0, max_impure - impure_melds)
    u_seq = min(need_pure, seq_p)
    u_imp = min(need_imp, max(0, seq_p - u_seq) + set_p)
    return float(q_seq) + 0.55 * float(impure_melds) + 0.22 * float(u_seq) + 0.12 * float(u_imp)


def _decompose_hand(hand_t, joker, req, cards_in_hand):
    wild_rank = joker % 13
    if (hand_t[wild_rank] + hand_t[13 + wild_rank] + hand_t[26 + wild_rank] + hand_t[39 + wild_rank]) > 2:
        return None
    key = (hand_t, joker, req, cards_in_hand)
    cached = _DECOMP_CACHE.get(key)
    if cached is not None:
        return cached
    pen = minimum_penalty(hand_t, joker, req)
    q_seq = min(req, pen.qualifying_sequences)
    max_impure = max(0, (cards_in_hand // 3) - req)
    rem_cards = pen.counted_cards
    if q_seq >= req:
        impure_melds = min(max_impure, max(0, len(pen.exempt_melds) - req))
    elif max_impure > 0 and sum(pen.counted_cards) >= 3:
        pen0 = minimum_penalty(pen.counted_cards, joker, 0)
        impure_melds = min(max_impure, len(pen0.exempt_melds))
        if len(pen0.exempt_melds) <= max_impure:
            rem_cards = pen0.counted_cards
    else:
        impure_melds = 0
    res = (q_seq, impure_melds, max_impure, rem_cards)
    if len(_DECOMP_CACHE) > 32768:
        _DECOMP_CACHE.clear()
    _DECOMP_CACHE[key] = res
    return res


def _cached_meld_val(hand_t, joker, req, cards_in_hand):
    key = (hand_t, joker, req)
    cached = _PEN_CACHE.get(key)
    if cached is not None:
        return cached
    d = _decompose_hand(hand_t, joker, req, cards_in_hand)
    if d is None:
        return 0.0
    q_seq, impure_melds, max_impure, rem_cards = d
    res = _score_from_parts(q_seq, impure_melds, rem_cards, joker, req, max_impure)
    if len(_PEN_CACHE) > 32768:
        _PEN_CACHE.clear()
    _PEN_CACHE[key] = res
    return res


def _score_22_discards(hand22, faces, q_vals, joker, req, cards_in_hand, top_non_u=4):
    d22 = _decompose_hand(tuple(hand22), joker, req, cards_in_hand)
    if d22 is None:
        return {f: 0.0 for f in faces}
    q_seq, impure_melds, max_impure, u22 = d22
    out = {}
    non_u = []
    rem_l = list(u22)
    for f in faces:
        if u22[f] > 0:
            rem_l[f] -= 1
            out[f] = _score_from_parts(q_seq, impure_melds, rem_l, joker, req, max_impure)
            rem_l[f] += 1
        else:
            non_u.append(f)
    non_u_top = sorted(non_u, key=lambda f: (-q_vals[f], f))[:top_non_u]
    h_l = list(hand22)
    min_u_val = min(out.values()) if out else 0.0
    for f in non_u:
        if f in non_u_top:
            h_l[f] -= 1
            out[f] = _cached_meld_val(tuple(h_l), joker, req, cards_in_hand)
            h_l[f] += 1
        else:
            out[f] = max(0.0, min_u_val - 0.55)
    return out


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


def _after_discard(obs: Observation, face: int) -> Observation:
    hand = list(obs.hand)
    hand[face] -= 1
    turns = obs.turns_remaining - 1
    player = replace(
        obs.players[0],
        hand_size=obs.players[0].hand_size - 1,
        turns_remaining=turns,
    )
    event = InformationEvent(
        EventKind.DISCARD, 0, player.hand_size, turns, obs.stock_remaining,
        player.next_player, face=face,
    )
    return replace(
        obs, hand=tuple(hand), discard_pile=(*obs.discard_pile, face),
        phase=Phase.DRAW, turns_remaining=turns, players=(player, *obs.players[1:]),
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
    req = obs.config.required_sequences
    cards = obs.config.cards_in_hand
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
        if (
            len(faces) > 1
            and obs.history
            and obs.history[-1].actor == 0
            and obs.history[-1].kind is EventKind.TAKE_DISCARD
            and obs.history[-1].face in faces
        ):
            faces = [f for f in faces if f != obs.history[-1].face]
        rows = 1
        if (
            config.exact_wins and config.rollout_depth >= 2
            and obs.config.cards_in_hand > 3 and len(faces) > 1
        ):
            after_states = [
                replace(_after_discard(obs, face), discard_pile=obs.discard_pile)
                for face in faces
            ]
            v_after = network(collate_observations(after_states, device))[:, 0].tolist()
            for idx, face in enumerate(faces):
                values[face] = 0.80 * values[face] + 0.20 * v_after[idx]
            rows += len(after_states)
        for face in faces:
            values[face] -= config.feed_weight * belief.feed_risk(face, 1)
        if config.rollout_depth >= 2 and config.outs_weight > 0:
            if config.exact_wins and cards > 3:
                mv_map = _score_22_discards(
                    obs.hand, faces, values, obs.joker, req, cards, top_non_u=4,
                )
                for face in faces:
                    values[face] += 0.06 * mv_map[face]
            candidates = sorted(faces, key=lambda face: (-values[face], face))[:config.rollout_beam]
            hand = list(obs.hand)
            for face in candidates:
                hand[face] -= 1
                values[face] += config.outs_weight * outs_1ply(hand, belief, obs.config)
                hand[face] += 1
            if (
                config.rollout_depth >= 3
                and len(candidates) >= 2
                and obs.turns_remaining > 1
                and obs.stock_remaining > 0
            ):
                probs = belief.stock_probabilities()
                if probs:
                    top_draws = sorted(probs, key=lambda item: (-item[1], item[0]))[:12]
                    norm = sum(p for _, p in top_draws)
                    rollout_children = []
                    beam_cands = candidates[: min(3, len(candidates))]
                    for face in beam_cands:
                        post_disc = _after_discard(obs, face)
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
            children = [_after_draw(obs, obs.discard_top, EventKind.DRAW_STOCK)] + [
                _after_draw(obs, face, EventKind.DRAW_STOCK) for face, _ in probs
            ]
            values = network(collate_observations(children, device))[:, 2:]
            penalties = torch.tensor(
                [config.feed_weight * belief.feed_risk(face, 1) for face in range(NUM_FACES)],
                dtype=values.dtype, device=device,
            )
            take_mask = list(legal_action_mask(children[0])[2:])
            if sum(take_mask) > 1:
                take_mask[obs.discard_top] = False
            allowed = torch.tensor(
                [take_mask, *(legal_action_mask(child)[2:] for child in children[1:])],
                dtype=torch.bool, device=device,
            )
            best_vals, best_faces = (values - penalties).masked_fill(~allowed, -torch.inf).max(dim=1)
            rows = len(children)
            take_value = float(best_vals[0].item())
            stock_value = sum(
                chance * float(best_vals[index + 1].item())
                for index, (_, chance) in enumerate(probs)
            )
            if (
                config.exact_wins and config.rollout_depth >= 2
                and (cards > 3 or config.outs_weight > 0)
            ):
                take_scores = (values[0] - penalties).tolist()
                take_faces = [f for f, ok in enumerate(take_mask) if ok]
                if cards > 3 and config.outs_weight > 0:
                    base_mv = _cached_meld_val(obs.hand, obs.joker, req, cards)
                    mv_take = _score_22_discards(
                        children[0].hand, take_faces, take_scores,
                        obs.joker, req, cards, top_non_u=4,
                    )
                    for face in take_faces:
                        take_scores[face] += 0.06 * (mv_take[face] - base_mv)
                candidates = sorted(take_faces, key=lambda f: (-take_scores[f], f))[:config.rollout_beam]
                if cards > 3:
                    unique_obs = []
                    hand_to_idx = {}
                    for face in candidates:
                        after = replace(
                            _after_discard(children[0], face), discard_pile=obs.discard_pile,
                        )
                        if after.hand not in hand_to_idx:
                            hand_to_idx[after.hand] = len(unique_obs)
                            unique_obs.append(after)
                    stock_hands = []
                    for idx_s, face in enumerate(best_faces[1:].tolist()):
                        after = replace(
                            _after_discard(children[idx_s + 1], face),
                            discard_pile=obs.discard_pile,
                        )
                        stock_hands.append(after.hand)
                        if after.hand not in hand_to_idx:
                            hand_to_idx[after.hand] = len(unique_obs)
                            unique_obs.append(after)
                    v_after_all = network(collate_observations(unique_obs, device))[:, 0].tolist()
                    rows += len(unique_obs)
                    for face in candidates:
                        post_hand = list(children[0].hand)
                        post_hand[face] -= 1
                        raw_q = float((values[0, face] - penalties[face]).item())
                        mv_bonus = take_scores[face] - raw_q
                        take_scores[face] = (
                            0.85 * raw_q
                            + 0.15 * v_after_all[hand_to_idx[tuple(post_hand)]]
                            + mv_bonus
                        )
                    stock_value = sum(
                        chance * (
                            0.85 * float(best_vals[idx_s + 1].item())
                            + 0.15 * v_after_all[hand_to_idx[stock_hands[idx_s]]]
                        )
                        for idx_s, (_, chance) in enumerate(probs)
                    )
                if config.outs_weight > 0:
                    child_hand = list(children[0].hand)
                    for face in candidates:
                        child_hand[face] -= 1
                        take_scores[face] += config.outs_weight * outs_1ply(child_hand, belief, obs.config)
                        child_hand[face] += 1
                    stock_value += config.outs_weight * outs_1ply(obs.hand, belief, obs.config)
                take_value = max(
                    take_scores[f] for f in (candidates if cards > 3 else take_faces)
                )
            action = ACTION_TAKE_DISCARD if take_value > stock_value else ACTION_DRAW_STOCK
            return SearchResult(action, "draw_lookahead", rows, take_value, stock_value)
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
