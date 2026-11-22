from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from random import Random
from struct import Struct
from typing import Callable, Iterable, Iterator, Protocol, Sequence

import torch
from torch import Tensor

from src.evaluate import is_valid_hand, pure_sequence_upper_bound
from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, Observation, PappluEnv, Phase, discard_action,
)
from src.game.information import EventKind, InformationEvent
from src.game.multiplayer import MatchConfig, MatchTelemetry, MultiplayerEnv, TerminalSnapshot
from src.game.outcome import DealOutcome
from src.model.encoding import FIELDS, SCHEMA, Encoded, encode, legal_mask, validate_game
from src.model.network import GameTransformer, Heads, action_mask, load_checkpoint


_DISCARD_CANDIDATES = 6
_VALUE_WEIGHT = 3.0
_UNSEARCHED_PENALTY = 0.15
_QUANTIZED_POLICY = Struct(f"={SCHEMA.num_actions}H")


@dataclass(frozen=True)
class SeatTrace:
    prefix: bytes
    suffix: bytes
    cut: tuple[int, ...]
    legal: tuple[int, ...]
    action: bytes
    behavior_logp: tuple[float, ...]
    target_probs: bytes


@dataclass(frozen=True)
class Turn:
    stream: int
    observation: Observation
    rng: Random
    legal: int


class Actor(Protocol):
    def act(self, turns: Sequence[Turn]) -> Sequence[int]: ...
    def finish(self, stream: int) -> SeatTrace | None: ...


class Policy(Protocol):
    game: GameConfig | None

    def open(self) -> Actor: ...


class _RandomActor:
    def act(self, turns: Sequence[Turn]) -> list[int]:
        return [
            turn.rng.choice([a for a in range(SCHEMA.num_actions) if turn.legal & (1 << a)])
            for turn in turns
        ]

    def finish(self, stream: int) -> None:
        return None


class RandomPolicy:
    game = None

    def open(self) -> Actor:
        return _RandomActor()


class TransformerPolicy:
    def __init__(
        self, network: GameTransformer, game: GameConfig, *, sample: bool = False,
        uniform_mix: float = 0.0, precision: str = "fp32", record: bool = False, search_ply: int = 0,
    ) -> None:
        if network.schema != SCHEMA:
            raise ValueError("policy needs the Papplu token schema")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if not 0 <= uniform_mix <= 1 or (uniform_mix and not sample):
            raise ValueError("uniform_mix must be in [0, 1] and requires sampling")
        if type(search_ply) is not int or search_ply not in (0, 1):
            raise ValueError("search_ply must be 0 or 1")
        self.network, self.game = network, game
        self.sample, self.uniform_mix, self.record, self.precision = sample, uniform_mix, record, precision
        self.search_ply = search_ply

    def open(self) -> Actor:
        return _TransformerActor(self)


class CheckpointPolicy(TransformerPolicy):
    def __init__(
        self, path: str | Path, *, device: str = "cpu", sample: bool = False,
        uniform_mix: float = 0.0, precision: str = "fp32", record: bool = False, search_ply: int = 0,
    ) -> None:
        checkpoint = load_checkpoint(Path(path), SCHEMA, device)
        super().__init__(
            checkpoint.network, GameConfig.from_dict(checkpoint.game), sample=sample,
            uniform_mix=uniform_mix, precision=precision, record=record, search_ply=search_ply,
        )
        self.meta = checkpoint.meta
        self.version = checkpoint.version


@dataclass
class _TraceDraft:
    prefix: bytearray = field(default_factory=bytearray)
    suffix: bytearray = field(default_factory=bytearray)
    cut: list[int] = field(default_factory=list)
    legal: list[int] = field(default_factory=list)
    action: bytearray = field(default_factory=bytearray)
    logp: list[float] = field(default_factory=list)
    target_probs: bytearray = field(default_factory=bytearray)

    def seal(self) -> SeatTrace:
        return SeatTrace(
            bytes(self.prefix), bytes(self.suffix), tuple(self.cut), tuple(self.legal),
            bytes(self.action), tuple(self.logp), bytes(self.target_probs),
        )


@dataclass
class _Stream:
    row: int
    trace: _TraceDraft | None
    events: int = 0


def _edit_hand(hand: tuple[int, ...], face: int, change: int) -> tuple[int, ...]:
    return hand[:face] + (hand[face] + change,) + hand[face + 1:]


def _winning_discards(obs: Observation, hand: tuple[int, ...], legal: int) -> Iterator[int]:
    quota = obs.config.required_sequences
    if quota and pure_sequence_upper_bound(hand, obs.joker) < quota:
        return
    for face, count in enumerate(hand):
        if count and legal & (1 << discard_action(face)) and is_valid_hand(
            _edit_hand(hand, face, -1), obs.joker, quota, obs.config.cards_in_hand,
        ):
            yield discard_action(face)


def _afterstate(obs: Observation, action: int) -> Encoded:
    if obs.phase is Phase.DRAW:
        face = obs.discard_pile[-1]
        hand, pile = _edit_hand(obs.hand, face, 1), obs.discard_pile[:-1]
        phase, kind, remaining = Phase.DISCARD, EventKind.TAKE_DISCARD, obs.turns_remaining
        size = obs.config.cards_in_hand + 1
    else:
        face = action - discard_action(0)
        hand, pile = _edit_hand(obs.hand, face, -1), (*obs.discard_pile, face)
        phase, kind, remaining = Phase.DRAW, EventKind.DISCARD, obs.turns_remaining - 1
        size = obs.config.cards_in_hand
    own = replace(obs.players[0], hand_size=size, turns_remaining=remaining)
    event = InformationEvent(kind, 0, size, remaining, obs.stock_remaining, 0, face, obs.discard_top)
    view = replace(obs, hand=hand, discard_pile=pile, phase=phase, turns_remaining=remaining,
                   players=(own, *obs.players[1:]), history=(event,))
    return encode(view, since=1)


class _TransformerActor:
    def __init__(self, policy: TransformerPolicy) -> None:
        self.policy = policy
        self.device = next(policy.network.parameters()).device
        self.bf16 = policy.precision == "bf16"
        self.cache = policy.network.new_cache(dtype=torch.bfloat16 if self.bf16 else torch.float32)
        self.streams: dict[int, _Stream] = {}
        self.free_rows: list[int] = []
        self.next_row = 0

    def _stream(self, key: int) -> _Stream:
        if key not in self.streams:
            if self.free_rows:
                row = self.free_rows.pop()
            else:
                row = self.next_row
                self.next_row += 1
            self.streams[key] = _Stream(row, _TraceDraft() if self.policy.record else None)
        return self.streams[key]

    def _search(self, turns: Sequence[Turn], rows: Tensor, legal: Tensor, heads: Heads) -> tuple[Tensor, list[int]]:
        prior = heads.log_policy
        logits = prior.clone()
        top = prior.topk(_DISCARD_CANDIDATES, dim=-1).indices.tolist()
        candidates: list[tuple[int, int, Encoded]] = []
        forced = []
        for i, turn in enumerate(turns):
            obs = turn.observation
            if obs.phase is Phase.DISCARD:
                wins = list(_winning_discards(obs, obs.hand, turn.legal))
                if wins:
                    logits[i].fill_(-torch.inf)
                    logits[i, wins] = 0
                    forced.append(i)
                    continue
                logits[i] += _VALUE_WEIGHT * (heads.value[i] - _UNSEARCHED_PENALTY)
                candidates.extend(
                    (i, action, _afterstate(obs, action)) for action in top[i] if turn.legal & (1 << action)
                )
            elif turn.legal == ((1 << ACTION_DRAW_STOCK) | (1 << ACTION_TAKE_DISCARD)):
                face = obs.discard_pile[-1]
                hand = _edit_hand(obs.hand, face, 1)
                discards = sum(1 << discard_action(face) for face, count in enumerate(hand) if count)
                retake = 1 << discard_action(face)
                if discards & ~retake:
                    discards &= ~retake
                if next(_winning_discards(obs, hand, discards), None) is not None:
                    logits[i, ACTION_DRAW_STOCK] = math.log(0.01)
                    logits[i, ACTION_TAKE_DISCARD] = math.log(0.99)
                    continue
                logits[i, ACTION_DRAW_STOCK] += _VALUE_WEIGHT * heads.value[i]
                candidates.append((i, ACTION_TAKE_DISCARD, _afterstate(obs, ACTION_TAKE_DISCARD)))
        if candidates:
            indices, actions, encoded = zip(*candidates)
            index = torch.tensor(indices, device=self.device)
            action = torch.tensor(actions, device=self.device)
            suffix = torch.frombuffer(bytearray(b"".join(row.suffix for row in encoded)), dtype=torch.uint8)
            mask = action_mask(torch.tensor([row.legal for row in encoded], device=self.device), SCHEMA.num_actions)
            scored = self.policy.network.evaluate_candidates(
                self.cache, rows[index], suffix.reshape(-1, SCHEMA.suffix_tokens, FIELDS).to(self.device), mask,
            )
            logits[index, action] = prior[index, action] + _VALUE_WEIGHT * scored.value
        return logits.masked_fill(~legal, -torch.inf).log_softmax(-1), forced

    @torch.inference_mode()
    def act(self, turns: Sequence[Turn]) -> list[int]:
        streams = [self._stream(turn.stream) for turn in turns]
        encoded = [encode(turn.observation, stream.events) for turn, stream in zip(turns, streams)]
        width = max(len(row.prefix) for row in encoded)
        if width:
            data = bytearray(b"".join(row.prefix.ljust(width, b"\x00") for row in encoded))
            prefix = torch.frombuffer(data, dtype=torch.uint8).reshape(len(turns), -1, FIELDS).to(self.device)
        else:
            prefix = torch.empty((len(turns), 0, FIELDS), dtype=torch.uint8, device=self.device)
        suffix = torch.frombuffer(bytearray(b"".join(row.suffix for row in encoded)), dtype=torch.uint8)
        bits = torch.tensor([row.legal for row in encoded], device=self.device)
        legal = action_mask(bits, SCHEMA.num_actions)
        rows = torch.tensor([stream.row for stream in streams], device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            heads = self.policy.network.step(
                self.cache, rows, prefix, suffix.reshape(len(turns), SCHEMA.suffix_tokens, FIELDS).to(self.device),
                legal, append_lengths=torch.tensor([len(row.prefix) // FIELDS for row in encoded], device=self.device),
            )
            log_policy, forced = self._search(turns, rows, legal, heads) if self.policy.search_ply else (heads.log_policy, [])
        target = log_policy.exp() if self.policy.sample or self.policy.record else None
        actions, logps = [], []
        if self.policy.sample:
            mix = self.policy.uniform_mix
            probabilities = (1 - mix) * target + mix * legal / legal.sum(-1, keepdim=True)
            if forced:
                probabilities[forced] = target[forced]
            for turn, row in zip(turns, probabilities.tolist()):
                choices = [a for a, p in enumerate(row) if p > 0]
                weights = [row[a] for a in choices]
                action = turn.rng.choices(choices, weights=weights, k=1)[0]
                actions.append(action)
                logps.append(math.log(row[action] / sum(weights)))
        else:
            actions = log_policy.argmax(-1).tolist()
            logps = [0.0] * len(actions)
        targets = target.mul(65535).round().int().tolist() if self.policy.record else None
        for i, (stream, row, action, logp) in enumerate(zip(streams, encoded, actions, logps)):
            stream.events = row.events
            if stream.trace is not None:
                stream.trace.prefix.extend(row.prefix)
                stream.trace.suffix.extend(row.suffix)
                stream.trace.cut.append(row.cut)
                stream.trace.legal.append(row.legal)
                stream.trace.action.append(action)
                stream.trace.logp.append(logp)
                stream.trace.target_probs.extend(_QUANTIZED_POLICY.pack(*targets[i]))
        return actions

    def finish(self, stream: int) -> SeatTrace | None:
        state = self.streams.pop(stream, None)
        if state is None:
            return None
        self.cache.reset(state.row)
        self.free_rows.append(state.row)
        return state.trace.seal() if state.trace is not None else None


@dataclass(frozen=True)
class Deal:
    seed: int
    seats: tuple[int, ...]
    curriculum: int | None = None

    @property
    def players(self) -> int:
        return len(self.seats)


@dataclass(frozen=True)
class MatchResult:
    deal: Deal
    action_count: int
    seat_turns: tuple[int, ...]
    stock_remaining: int
    telemetry: MatchTelemetry
    outcome: DealOutcome
    traces: tuple[SeatTrace | None, ...]

    @property
    def winner(self) -> int | None:
        return self.outcome.winner

    @property
    def terminal_reason(self) -> str:
        return self.outcome.ending


@dataclass
class _Opened:
    policy: Policy
    actor: Actor
    streams: int = 0


@dataclass
class _Table:
    deal: Deal
    env: PappluEnv | MultiplayerEnv
    actors: tuple[_Opened, ...]
    streams: tuple[int, ...]
    rngs: tuple[Random, ...]
    action_count: int = 0
    solo_draws: list[int] = field(default_factory=lambda: [0, 0])

    @staticmethod
    def start(deal: Deal, game: GameConfig, actors: tuple[_Opened, ...], index: int) -> _Table:
        distance = deal.curriculum
        if distance is not None and (type(distance) is not int or not 0 <= distance <= game.cards_in_hand):
            raise ValueError("curriculum must be None or an integer between zero and the hand size")
        if deal.players == 1:
            env: PappluEnv | MultiplayerEnv = PappluEnv(game)
        else:
            env = MultiplayerEnv(MatchConfig(game, deal.players))
        if distance is None:
            env.reset(deal.seed)
        elif distance:
            env.reset_curriculum(distance, deal.seed)
        elif isinstance(env, PappluEnv):
            env.reset_warm_start(deal.seed)
        else:
            raise ValueError("distance zero is a solo discard puzzle")
        return _Table(
            deal, env, actors, tuple(index * 6 + seat for seat in range(deal.players)),
            tuple(Random(f"{deal.seed}:{seat}") for seat in range(deal.players)),
        )

    def turn(self) -> tuple[_Opened, Turn]:
        seat = self.env.view().current_seat if isinstance(self.env, MultiplayerEnv) else 0
        observation = self.env.observe()
        return self.actors[seat], Turn(self.streams[seat], observation, self.rngs[seat], legal_mask(observation))

    def step(self, turn: Turn, action: int, game: GameConfig) -> MatchResult | None:
        self.action_count += 1
        state = self.env.step(action)
        if isinstance(state, Observation) and turn.observation.phase is Phase.DRAW:
            self.solo_draws[action] += 1
        if not state.done:
            return None
        if isinstance(state, Observation):
            winner = 0 if state.won else None
            ending = "win" if state.won else "turns_exhausted"
            snapshot = TerminalSnapshot((state.hand,), state.joker)
            remaining = state.turns_remaining,
            telemetry = MatchTelemetry((self.solo_draws[0],), (self.solo_draws[1],), ())
        else:
            winner, ending = state.winner, state.terminal_reason
            snapshot = self.env.terminal_snapshot()
            remaining, telemetry = state.turns_remaining, self.env.telemetry()
        outcome = DealOutcome.score(snapshot, winner, ending, game.required_sequences, game.cards_in_hand)
        traces = tuple(actor.actor.finish(stream) for actor, stream in zip(self.actors, self.streams))
        turns = tuple(game.max_turns - count for count in remaining)
        return MatchResult(self.deal, self.action_count, turns, state.stock_remaining, telemetry, outcome, traces)


def run_matches(
    deals: Iterable[Deal], policies: Sequence[Policy] | Callable[[Deal], Sequence[Policy]],
    game: GameConfig, *, tables: int = 64, stop: Callable[[], bool] = lambda: False,
) -> Iterator[MatchResult]:
    if type(tables) is not int or tables < 1:
        raise ValueError("tables must be a positive integer")
    validate_game(game)
    queue = iter(enumerate(deals))
    opened: dict[int, _Opened] = {}
    live: list[_Table] = []
    while not stop():
        while len(live) < tables and not stop():
            item = next(queue, None)
            if item is None:
                break
            index, deal = item
            options = policies(deal) if callable(policies) else policies
            if not 1 <= deal.players <= 6 or any(type(i) is not int or not 0 <= i < len(options) for i in deal.seats):
                raise ValueError("invalid deal seating")
            actors = []
            for seat in deal.seats:
                policy = options[seat]
                key = id(policy)
                if key not in opened:
                    if policy.game is not None and policy.game != game:
                        raise ValueError("policy game config does not match the deal")
                    opened[key] = _Opened(policy, policy.open())
                actor = opened[key]
                actor.streams += 1
                actors.append(actor)
            live.append(_Table.start(deal, game, tuple(actors), index))
        if not live or stop():
            return
        groups: dict[int, list[tuple[_Table, Turn]]] = {}
        for table in live:
            actor, turn = table.turn()
            groups.setdefault(id(actor.policy), []).append((table, turn))
        decisions = []
        for key, group in groups.items():
            turns = [turn for _, turn in group]
            actions = opened[key].actor.act(turns)
            if len(actions) != len(turns):
                raise ValueError("policy must return one action per decision")
            for (table, turn), action in zip(group, actions):
                if type(action) is not int or not 0 <= action < SCHEMA.num_actions or not turn.legal & (1 << action):
                    raise ValueError("policy chose an illegal action")
                decisions.append((table, turn, action))
        live = []
        for table, turn, action in decisions:
            result = table.step(turn, action, game)
            if result is None:
                live.append(table)
                continue
            for actor in table.actors:
                actor.streams -= 1
                if not actor.streams:
                    del opened[id(actor.policy)]
            yield result


@dataclass(frozen=True)
class HeadToHead:
    players: int
    deals: int
    advantage: float
    win_rate: float
    opponent_win_rate: float


def head_to_head(
    candidate: Policy, opponent: Policy, game: GameConfig, players: int, deals: int, seed: int, *,
    tables: int = 64, stop: Callable[[], bool] = lambda: False,
) -> HeadToHead:
    if type(players) is not int or not 1 <= players <= 6 or type(deals) is not int or deals < 1:
        raise ValueError("head-to-head needs a positive deal count and 1..6 players")
    if players == 1:
        queue = (Deal(seed + i, (policy,)) for i in range(deals) for policy in (0, 1))
    else:
        queue = (
            Deal(seed + i, tuple(int(seat != candidate_seat) for seat in range(players)))
            for i in range(deals) for candidate_seat in range(players)
        )
    wins, totals, utility = [0, 0], [0, 0], [0.0, 0.0]
    count = 0
    for result in run_matches(queue, (candidate, opponent), game, tables=tables, stop=stop):
        count += 1
        for seat, (policy, score) in enumerate(zip(result.deal.seats, result.outcome.utilities())):
            totals[policy] += 1
            wins[policy] += int(result.winner == seat)
            utility[policy] += score
    if count != deals * max(2, players):
        raise InterruptedError("head-to-head stopped before completing the paired deals")
    advantage = utility[0] / totals[0]
    if players == 1:
        advantage -= utility[1] / totals[1]
    return HeadToHead(players, count, advantage, wins[0] / totals[0], wins[1] / totals[1])


def resolve_policy(model: str | Path, *, precision: str = "fp32", search_ply: int = 0) -> Policy:
    if model == "random":
        return RandomPolicy()
    if model == "heuristic":
        raise ValueError("heuristic policy is no longer supported; use random or a checkpoint")
    return CheckpointPolicy(model, precision=precision, search_ply=search_ply)


def match_game(policies: Sequence[Policy], max_turns: int | None) -> GameConfig:
    game = next((policy.game for policy in policies if policy.game is not None), GameConfig(max_turns=60))
    return game if max_turns is None else replace(game, max_turns=max_turns)
