from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
import math
from multiprocessing import TimeoutError as PoolTimeout, get_context
from pathlib import Path
from random import Random
from typing import Callable, Iterable, Iterator, Protocol, Sequence

import torch

from src.checkpoints.io import file_sha256
from src.game.environment import GameConfig, Observation, PappluEnv, Phase
from src.game.multiplayer import MatchConfig, MatchTelemetry, MultiplayerEnv, TerminalSnapshot
from src.game.outcome import DealOutcome, hand_potential
from src.model.encoding import FIELDS, SCHEMA, encode, legal_mask, validate_game
from src.model.network import GameTransformer, action_mask, load_checkpoint


@dataclass(frozen=True)
class SeatTrace:
    prefix: bytes
    suffix: bytes
    cut: tuple[int, ...]
    legal: tuple[int, ...]
    action: bytes
    behavior_logp: tuple[float, ...]
    potential: tuple[float, ...]
    behavior_value: tuple[float, ...]


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
        uniform_mix: float = 0.0, precision: str = "fp32", record: bool = False,
    ) -> None:
        if network.schema != SCHEMA:
            raise ValueError("policy needs the Papplu token schema")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if not 0 <= uniform_mix <= 1 or (uniform_mix and not sample):
            raise ValueError("uniform_mix must be in [0, 1] and requires sampling")
        self.network, self.game = network, game
        self.sample, self.uniform_mix, self.record, self.precision = sample, uniform_mix, record, precision

    def open(self) -> Actor:
        return _TransformerActor(self)


class CheckpointPolicy(TransformerPolicy):
    def __init__(
        self, path: str | Path, *, device: str = "cpu", sample: bool = False,
        uniform_mix: float = 0.0, precision: str = "fp32", record: bool = False,
    ) -> None:
        checkpoint = load_checkpoint(Path(path), SCHEMA, device)
        super().__init__(
            checkpoint.network, GameConfig.from_dict(checkpoint.game), sample=sample,
            uniform_mix=uniform_mix, precision=precision, record=record,
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
    potential: array = field(default_factory=lambda: array("f"))
    value: list[float] = field(default_factory=list)

    def seal(self) -> SeatTrace:
        return SeatTrace(
            bytes(self.prefix), bytes(self.suffix), tuple(self.cut), tuple(self.legal),
            bytes(self.action), tuple(self.logp),
            tuple(self.potential), tuple(self.value),
        )


@dataclass
class _Stream:
    row: int
    trace: _TraceDraft | None
    events: int = 0


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
        actions, logps = [], []
        if self.policy.sample:
            mix = self.policy.uniform_mix
            probabilities = (1 - mix) * heads.log_policy.exp() + mix * legal / legal.sum(-1, keepdim=True)
            for turn, row in zip(turns, probabilities.tolist()):
                choices = [a for a, p in enumerate(row) if p > 0]
                weights = [row[a] for a in choices]
                action = turn.rng.choices(choices, weights=weights, k=1)[0]
                actions.append(action)
                logps.append(math.log(row[action] / sum(weights)))
        else:
            actions = heads.log_policy.argmax(-1).tolist()
            logps = [0.0] * len(actions)
        values = heads.value.tolist() if self.policy.record else None
        for i, (turn, stream, row, action, logp) in enumerate(zip(turns, streams, encoded, actions, logps)):
            stream.events = row.events
            trace = stream.trace
            if trace is not None:
                obs = turn.observation
                trace.prefix.extend(row.prefix)
                trace.suffix.extend(row.suffix)
                trace.cut.append(row.cut)
                trace.legal.append(row.legal)
                trace.action.append(action)
                trace.logp.append(logp)
                trace.potential.append(hand_potential(
                    obs.hand, obs.joker, obs.config.required_sequences, obs.config.cards_in_hand,
                ))
                trace.value.append(values[i])
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
                    if policy.game is not None and replace(game, max_turns=policy.game.max_turns) != policy.game:
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
    matches: int
    score: float
    net_points: float
    advantage: float
    win_rate: float
    opponent_win_rate: float
    mean_win_turn: float | None
    losing_deadwood: float | None
    opponent_mean_win_turn: float | None
    opponent_losing_deadwood: float | None
    comparisons: int
    seats_per_policy: int


def head_to_head(
    candidate: Policy, opponent: Policy, game: GameConfig, players: int, deals: int, seed: int, *,
    tables: int = 64, stop: Callable[[], bool] = lambda: False,
) -> HeadToHead:
    if type(players) is not int or not 1 <= players <= 6 or type(deals) is not int or deals < 1:
        raise ValueError("head-to-head needs a positive deal count and 1..6 players")
    if players == 1:
        lineups = ((0,), (1,))
    elif players == 2:
        lineups = ((0, 1), (1, 0))
    else:
        lineups = tuple(
            tuple(policy if seat == focal else 1 - policy for seat in range(players))
            for focal in range(players) for policy in (0, 1)
        )
    queue = (Deal(seed + i, seats) for i in range(deals) for seats in lineups)
    wins, totals, turns, deadwood = [0, 0], [0, 0], [0, 0], [0, 0]
    points, utilities = ([], []), ([], [])
    matches = comparisons = doubled_score = 0
    solo: dict[int, tuple[bool, int]] = {}
    for result in run_matches(queue, (candidate, opponent), game, tables=tables, stop=stop):
        matches += 1
        seats, outcome = result.deal.seats, result.outcome
        penalties = tuple(p.points for p in outcome.penalties)
        for seat, policy in enumerate(seats):
            totals[policy] += 1
            if result.winner == seat:
                wins[policy] += 1
                turns[policy] += result.seat_turns[seat]
            else:
                deadwood[policy] += penalties[seat]
        for policy, net, utility in zip(seats, outcome.net_points(), outcome.utilities()):
            points[policy].append(net)
            utilities[policy].append(utility)
        if players == 1:
            policy = seats[0]
            rank = result.winner is None, penalties[0]
            other = solo.pop(result.deal.seed, None)
            if other is None:
                solo[result.deal.seed] = rank
                continue
            left, right = (rank, other) if policy == 0 else (other, rank)
            doubled_score += 2 * (left < right) + (left == right)
            comparisons += 1
        else:
            ranks = outcome.ranks()
            for a, first in enumerate(seats):
                if first == 0:
                    for b, second in enumerate(seats):
                        if second == 1:
                            doubled_score += 2 * (ranks[a] < ranks[b]) + (ranks[a] == ranks[b])
                            comparisons += 1
    if matches != deals * len(lineups):
        raise InterruptedError("head-to-head stopped before completing the paired deals")
    exposure = totals[0]
    divisor = exposure if players == 1 else 2 * exposure
    losers = [total - win for total, win in zip(totals, wins)]
    return HeadToHead(
        players=players, deals=deals, matches=matches, score=doubled_score / (2 * comparisons),
        net_points=(math.fsum(points[0]) - math.fsum(points[1])) / divisor,
        advantage=(math.fsum(utilities[0]) - math.fsum(utilities[1])) / divisor,
        win_rate=wins[0] / exposure, opponent_win_rate=wins[1] / exposure,
        mean_win_turn=turns[0] / wins[0] if wins[0] else None,
        losing_deadwood=deadwood[0] / losers[0] if losers[0] else None,
        opponent_mean_win_turn=turns[1] / wins[1] if wins[1] else None,
        opponent_losing_deadwood=deadwood[1] / losers[1] if losers[1] else None,
        comparisons=comparisons, seats_per_policy=exposure,
    )


def solve_elo(
    wins_matrix: Sequence[Sequence[float]], *, anchor_index: int = 0, anchor_elo: float = 1000.0,
) -> tuple[float, ...]:
    """Fit Bradley-Terry strengths by coordinate-wise minorization-maximization."""
    size = len(wins_matrix)
    if not size or any(len(row) != size for row in wins_matrix):
        raise ValueError("Elo needs a nonempty square wins matrix")
    if type(anchor_index) is not int or not 0 <= anchor_index < size or not math.isfinite(anchor_elo):
        raise ValueError("Elo needs a valid anchor index and finite anchor rating")
    if any(not math.isfinite(w) or w < 0 for row in wins_matrix for w in row):
        raise ValueError("Elo needs finite nonnegative win counts")
    if size == 1:
        return (float(anchor_elo),)
    wins = [[0.0 if i == j else float(wins_matrix[i][j]) for j in range(size)] for i in range(size)]
    if any(not sum(row) or not any(column) for row, column in zip(wins, zip(*wins))):
        for i, j in combinations(range(size), 2):
            wins[i][j] += 1e-3
            wins[j][i] += 1e-3
    for edges in (wins, tuple(zip(*wins))):
        seen, pending = {anchor_index}, [anchor_index]
        while pending:
            for j, weight in enumerate(edges[pending.pop()]):
                if weight and j not in seen:
                    seen.add(j)
                    pending.append(j)
        if len(seen) != size:
            raise ValueError("Elo needs a strongly connected win graph for a finite fit")
    strengths = [1.0] * size
    for _ in range(50_000):
        change = 0.0
        for i in range(size):
            strength = strengths[i]
            gain = loss = 0.0
            for j, other in enumerate(strengths):
                if i != j:
                    gain += wins[i][j] * (other / (strength + other))
                    loss += wins[j][i] * (strength / (strength + other))
            scale = math.sqrt(gain / loss)
            strengths[i] *= scale
            change = max(change, abs(math.log(scale)))
        anchor = strengths[anchor_index]
        strengths = [strength / anchor for strength in strengths]
        if change < 1e-12:
            return tuple(anchor_elo + 400.0 * math.log10(strength) for strength in strengths)
    raise RuntimeError("Bradley-Terry fit did not converge")


@dataclass
class _LeagueTotals:
    seats: int = 0
    wins: int = 0
    turns: int = 0
    deadwood: int = 0
    points: float = 0.0

    def add(self, match: HeadToHead, candidate: bool) -> None:
        seats = match.seats_per_policy
        rate = match.win_rate if candidate else match.opponent_win_rate
        turn = match.mean_win_turn if candidate else match.opponent_mean_win_turn
        deadwood = match.losing_deadwood if candidate else match.opponent_losing_deadwood
        wins = round(rate * seats)
        self.seats += seats
        self.wins += wins
        self.turns += round((turn or 0) * wins)
        self.deadwood += round((deadwood or 0) * (seats - wins))
        self.points += match.net_points * seats * (1 if candidate else -1)


def _checkpoint_match(
    candidate: Path, opponent: Path, game: GameConfig, players: int, deals: int, seed: int,
    tables: int, precision: str,
) -> HeadToHead:
    return head_to_head(
        CheckpointPolicy(candidate, precision=precision), CheckpointPolicy(opponent, precision=precision),
        game, players, deals, seed, tables=tables,
    )


def rate_checkpoints(
    checkpoints: Sequence[str | Path], *, labels: Sequence[str] | None = None,
    players: Sequence[int] = (2, 3, 4), deals: int = 48, seed: int = 970_000,
    max_turns: int | None = None, tables: int = 64, precision: str = "fp32",
    anchor_elo: float = 1000.0, stop: Callable[[], bool] = lambda: False, workers: int = 1,
) -> dict:
    paths = tuple(map(Path, checkpoints))
    size, counts = len(paths), tuple(players)
    if size < 2:
        raise ValueError("league needs at least two GameTransformer checkpoints")
    if any(str(path).casefold() in ("random", "heuristic") for path in paths):
        raise ValueError("league only accepts GameTransformer checkpoints")
    if not counts or len(set(counts)) != len(counts) or any(type(p) is not int or not 2 <= p <= 6 for p in counts):
        raise ValueError("league player counts must be distinct integers in 2..6")
    if labels is None:
        labels = tuple(path.stem for path in paths)
    if len(labels) != size:
        raise ValueError("league needs one label per checkpoint")
    if type(workers) is not int or workers < 1:
        raise ValueError("league workers must be a positive integer")
    if type(deals) is not int or deals < 1 or not math.isfinite(anchor_elo):
        raise ValueError("league needs a positive deal count and finite anchor rating")
    hashes = tuple(file_sha256(path) for path in paths)
    policies = tuple(CheckpointPolicy(path, precision=precision) for path in paths)
    game = match_game(policies, max_turns)
    for p in counts:
        MatchConfig(game, p)
    schedule = [
        (i, j, p, seed + index * 10_000 + p * 1_000)
        for index, (i, j) in enumerate(combinations(range(size), 2)) for p in counts
    ]
    if workers == 1:
        results = [
            head_to_head(policies[i], policies[j], game, p, deals, pair_seed, tables=tables, stop=stop)
            for i, j, p, pair_seed in schedule
        ]
    else:
        tasks = [(paths[i], paths[j], game, p, deals, pair_seed, tables, precision) for i, j, p, pair_seed in schedule]
        with get_context("spawn").Pool(min(workers, len(tasks)), torch.set_num_threads, (1,)) as pool:
            pending = pool.starmap_async(_checkpoint_match, tasks, chunksize=1)
            while not stop():
                try:
                    results = pending.get(timeout=0.25)
                    break
                except PoolTimeout:
                    continue
            else:
                raise InterruptedError("league stopped before completing the paired deals")
    if tuple(file_sha256(path) for path in paths) != hashes:
        raise ValueError("checkpoint changed during league")
    matrices = {key: [[0.0] * size for _ in paths] for key in (0, *counts)}
    totals = {key: [_LeagueTotals() for _ in paths] for key in matrices}
    pairs = []
    for (i, j, p, pair_seed), match in zip(schedule, results):
        wins = round(2 * match.score * match.comparisons) / 2
        for key in (0, p):
            matrices[key][i][j] += wins
            matrices[key][j][i] += match.comparisons - wins
            totals[key][i].add(match, True)
            totals[key][j].add(match, False)
        pairs.append({"candidate": i, "opponent": j, "seed": pair_seed, **asdict(match)})
    elo = {key: solve_elo(matrix, anchor_elo=anchor_elo) for key, matrix in matrices.items()}
    standings = []
    for i, tally in enumerate(totals[0]):
        won = sum(matrices[0][i])
        comparisons = won + sum(row[i] for row in matrices[0])
        losses = tally.seats - tally.wins
        standings.append({
            "index": i, "label": labels[i], "checkpoint": str(paths[i]), "sha256": hashes[i],
            "elo": elo[0][i], "elo_by_players": {p: elo[p][i] for p in counts},
            "score": won / comparisons, "seats": tally.seats, "win_rate": tally.wins / tally.seats,
            "net_points_per_seat": tally.points / tally.seats,
            "net_points_per_seat_by_players": {p: totals[p][i].points / totals[p][i].seats for p in counts},
            "mean_win_turn": tally.turns / tally.wins if tally.wins else None,
            "losing_deadwood": tally.deadwood / losses if losses else None,
        })
    return {
        "config": {
            "game": game.to_dict(), "players": counts, "deals": deals, "seed": seed, "tables": tables,
            "precision": precision, "workers": workers, "anchor_elo": anchor_elo,
        },
        "matches": sum(match.matches for match in results), "pairs": pairs, "standings": standings,
    }


def resolve_policy(model: str | Path, *, precision: str = "fp32") -> Policy:
    if model == "random":
        return RandomPolicy()
    if model == "heuristic":
        raise ValueError("heuristic policy is no longer supported; use random or a checkpoint")
    return CheckpointPolicy(model, precision=precision)


def match_game(policies: Sequence[Policy], max_turns: int | None) -> GameConfig:
    configs = {policy.game for policy in policies if policy.game is not None}
    if len(configs) > 1:
        raise ValueError("checkpoints must have matching game configs")
    game = next(iter(configs), GameConfig(max_turns=60))
    return game if max_turns is None else replace(game, max_turns=max_turns)
