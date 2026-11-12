from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import random
import time
from typing import List, Optional, Protocol, Sequence, Tuple
import torch

from src.game.environment import GameConfig, Observation
from src.game.multiplayer import MatchConfig, MatchTelemetry, MultiplayerEnv, TerminalSnapshot
from src.game.outcome import DealOutcome
from src.model.encoding import Features, encode, legal_actions
from src.model.network import collate, greedy_actions, load_checkpoint
from src.training.core import Decision


class Policy(Protocol):
    def act(self, observation: Observation, rng: random.Random) -> int:
        ...


class BatchPolicy(Policy, Protocol):
    def act_features(self, features: Sequence[Features], rngs: Sequence[random.Random]) -> list[int]:
        ...


class MatchInterrupted(RuntimeError):
    """Raised when a match reaches its caller-owned monotonic deadline."""


class RandomPolicy:
    def act(self, observation: Observation, rng: random.Random) -> int:
        return self.act_features([encode(observation)], [rng])[0]

    def act_features(self, features: Sequence[Features], rngs: Sequence[random.Random]) -> list[int]:
        if len(features) != len(rngs):
            raise ValueError("each feature row needs its own RNG")
        return [rng.choice(legal_actions(row)) for row, rng in zip(features, rngs)]


class GreedyPolicy:
    def __init__(self, network) -> None:
        self.network = network
        self.game_config = getattr(network, "game_config", None)

    def act(self, observation: Observation, rng: random.Random) -> int:
        return self.act_features([encode(observation)], [rng])[0]

    def act_features(self, features: Sequence[Features], rngs: Sequence[random.Random]) -> list[int]:
        if len(features) != len(rngs):
            raise ValueError("each feature row needs its own RNG")
        if not features:
            return []
        device = next(self.network.parameters()).device
        return greedy_actions(self.network, collate(features, device)).tolist()


class CheckpointPolicy(GreedyPolicy):
    def __init__(self, path: str) -> None:
        if not isinstance(path, str):
            raise TypeError("path must be a str")
        if not path:
            raise ValueError("path must not be empty")
        self.path = path
        self._network = None
        self._game_config: Optional[GameConfig] = None

    @property
    def network(self):
        self._load()
        return self._network

    @property
    def game_config(self) -> GameConfig:
        self._load()
        if self._game_config is None:
            raise RuntimeError("checkpoint did not provide a game config")
        return self._game_config

    def _load(self) -> None:
        if self._network is not None:
            return
        network, game_config, _ = load_checkpoint(self.path)
        network.eval()
        self._network = network
        self._game_config = game_config


class EpsilonPolicy(GreedyPolicy):
    def __init__(self, network, epsilon: float) -> None:
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, Real)
            or not math.isfinite(float(epsilon))
            or not 0.0 <= float(epsilon) <= 1.0
        ):
            raise ValueError("epsilon must be a finite number in [0, 1]")
        self.network = network
        self.epsilon = float(epsilon)
        self.game_config = getattr(network, "game_config", None)

    def act(self, observation: Observation, rng: random.Random) -> int:
        return self.act_features([encode(observation)], [rng])[0]

    def act_features(self, features: Sequence[Features], rngs: Sequence[random.Random]) -> list[int]:
        if len(features) != len(rngs):
            raise ValueError("each feature row needs its own RNG")
        actions, greedy_rows = [0] * len(features), []
        for row, (record, rng) in enumerate(zip(features, rngs)):
            if rng.random() < self.epsilon:
                actions[row] = rng.choice(legal_actions(record))
            else:
                greedy_rows.append(row)
        if greedy_rows:
            device = next(self.network.parameters()).device
            selected = greedy_actions(
                self.network, collate([features[row] for row in greedy_rows], device),
            ).tolist()
            for row, action in zip(greedy_rows, selected):
                actions[row] = action
        return actions


@dataclass(frozen=True)
class Deal:
    seed: int
    seats: tuple[int, ...]
    players: int
    record_seats: frozenset[int] = frozenset()


@dataclass
class _Match:
    env: MultiplayerEnv
    rngs: tuple[random.Random, ...]
    decisions: list[list[Decision]]
    action_count: int = 0


@dataclass(frozen=True)
class MatchResult:
    seed: int
    winner: Optional[int]
    terminal_reason: str
    seat_turns: Tuple[int, ...]
    action_count: int
    stock_remaining: int
    telemetry: MatchTelemetry
    trajectories: Tuple[Tuple[Decision, ...], ...]
    terminal_snapshot: Optional[TerminalSnapshot] = None
    outcome: Optional[DealOutcome] = None


def run_matches(
    deals: Sequence[Deal], policies: Sequence[Policy], game: GameConfig,
    deadline: Optional[float] = None, *, record_terminal: bool = False, score_outcome: bool = True,
) -> list[MatchResult]:
    if not isinstance(game, GameConfig):
        raise TypeError("game must be a GameConfig")
    if type(record_terminal) is not bool:
        raise TypeError("record_terminal must be a boolean")
    if type(score_outcome) is not bool:
        raise TypeError("score_outcome must be a boolean")
    if deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, Real) or not math.isfinite(float(deadline))
    ):
        raise ValueError("deadline must be a finite monotonic timestamp")
    matches = []
    for deal in deals:
        if type(deal.seed) is not int:
            raise TypeError("seed must be an int")
        config = MatchConfig(game=game, players=deal.players)
        if len(deal.seats) != deal.players or any(
            type(index) is not int or not 0 <= index < len(policies) for index in deal.seats
        ):
            raise ValueError("deal seats must select one policy per player")
        if not isinstance(deal.record_seats, frozenset) or any(
            type(seat) is not int or not 0 <= seat < deal.players for seat in deal.record_seats
        ):
            raise ValueError("record_seats must name seats in the deal")
        lineup = [policies[index] for index in deal.seats]
        for policy in lineup:
            if not callable(getattr(policy, "act", None)):
                raise TypeError("policies must define act")
        _validate_policy_configs(lineup, game)
        env = MultiplayerEnv(config)
        env.reset(deal.seed)
        matches.append(_Match(
            env, tuple(
                random.Random(deal.seed ^ ((seat + 1) * 0x9E3779B97F4A7C15))
                for seat in range(deal.players)
            ), [[] for _ in range(deal.players)],
        ))
    while any(not match.env.view().done for match in matches):
        _check_deadline(deadline)
        groups = {}
        for index, (deal, match) in enumerate(zip(deals, matches)):
            view = match.env.view()
            if view.done:
                continue
            seat = view.current_seat
            obs = match.env.observe()
            features = encode(obs)
            policy = policies[deal.seats[seat]]
            groups.setdefault(id(policy), (policy, []))[1].append((index, seat, obs, features))
        chosen = []
        for policy, rows in groups.values():
            rngs = [matches[index].rngs[seat] for index, seat, _, _ in rows]
            if callable(getattr(policy, "act_features", None)):
                actions = policy.act_features([row[3] for row in rows], rngs)
            else:
                actions = [policy.act(row[2], rng) for row, rng in zip(rows, rngs)]
            if len(actions) != len(rows):
                raise ValueError("batched policy returned the wrong number of actions")
            chosen.extend((*row, action) for row, action in zip(rows, actions))
        _check_deadline(deadline)
        for index, seat, obs, features, action in chosen:
            match, deal = matches[index], deals[index]
            if seat in deal.record_seats:
                match.decisions[seat].append(Decision(
                    obs, features, action, match.env.private_hands(seat),
                ))
            match.env.step(action)
            match.action_count += 1
    results = []
    for deal, match in zip(deals, matches):
        _check_deadline(deadline)
        view = match.env.view()
        if view.terminal_reason is None:
            raise RuntimeError("terminal match has no reason")
        snapshot = match.env.terminal_snapshot() if record_terminal or score_outcome else None
        outcome = (
            DealOutcome.score(snapshot, view.winner, view.terminal_reason, game.required_sequences)
            if score_outcome else None
        )
        _check_deadline(deadline)
        results.append(MatchResult(
            seed=deal.seed, winner=view.winner, terminal_reason=view.terminal_reason,
            seat_turns=tuple(game.max_turns - remaining for remaining in view.turns_remaining),
            action_count=match.action_count, stock_remaining=view.stock_remaining,
            telemetry=match.env.telemetry(),
            trajectories=tuple(tuple(decisions) for decisions in match.decisions),
            terminal_snapshot=snapshot if record_terminal else None,
            outcome=outcome,
        ))
    return results


def play_match(
    policies: Sequence[Policy],
    config: MatchConfig,
    seed: int,
    record_trajectories: bool = False,
    deadline: Optional[float] = None,
    *,
    record_terminal: bool = False,
    score_outcome: bool = True,
) -> MatchResult:
    if not isinstance(config, MatchConfig):
        raise TypeError("config must be a MatchConfig")
    if len(policies) != config.players:
        raise ValueError(
            "expected %d policies, got %d" % (config.players, len(policies))
        )
    if type(record_trajectories) is not bool:
        raise TypeError("record_trajectories must be a boolean")
    return run_matches(
        [Deal(seed, tuple(range(config.players)), config.players,
              frozenset(range(config.players)) if record_trajectories else frozenset())],
        policies, config.game, deadline, record_terminal=record_terminal, score_outcome=score_outcome,
    )[0]


def rotated_matches(
    policies: Sequence[Policy],
    config: MatchConfig,
    seed: int,
    record_trajectories: bool = False,
) -> List[MatchResult]:
    """Run one shared seed per cyclic seating.

    Result index ``r`` seats original policy ``(r + seat) % players`` at
    ``seat``. A winning seat maps back to ``(r + winner) % players``.
    """
    lineup = tuple(policies)
    if len(lineup) != config.players:
        raise ValueError(
            "expected %d policies, got %d" % (config.players, len(lineup))
        )
    return run_matches([
        Deal(
            seed, tuple((rotation + seat) % config.players for seat in range(config.players)),
            config.players, frozenset(range(config.players)) if record_trajectories else frozenset(),
        )
        for rotation in range(config.players)
    ], lineup, config.game)


def _validate_policy_configs(
    policies: Sequence[Policy], match_game: GameConfig
) -> None:
    for seat, policy in enumerate(policies):
        policy_config = getattr(policy, "game_config", None)
        if policy_config is None:
            network = getattr(policy, "network", None)
            policy_config = getattr(network, "game_config", None)
        if policy_config is None:
            continue
        if not isinstance(policy_config, GameConfig):
            raise TypeError("policy %d game_config must be a GameConfig" % seat)
        expected = (
            match_game.num_decks,
            match_game.cards_in_hand,
            match_game.required_sequences,
        )
        actual = (
            policy_config.num_decks,
            policy_config.cards_in_hand,
            policy_config.required_sequences,
        )
        if actual != expected:
            raise ValueError(
                "policy %d game config %r does not match match config %r"
                % (seat, actual, expected)
            )


def _check_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise MatchInterrupted("match deadline reached")
