"""Headless multiplayer policy matches and a bounded tournament CLI."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import random
import time
from typing import List, Optional, Protocol, Sequence, Tuple

from src.game.environment import (
    GameConfig,
    Observation,
    Phase,
    encode_observation,
    legal_action_mask,
)
from src.game.multiplayer import MatchConfig, MatchTelemetry, MultiplayerEnv
from src.model.network import (
    load_checkpoint,
    select_action,
    select_greedy_action,
)


EpisodeStep = Tuple[Tuple[float, ...], int, float]


class Policy(Protocol):
    def act(self, observation: Observation, rng: random.Random) -> int:
        ...


class MatchInterrupted(RuntimeError):
    """Raised when a match reaches its caller-owned monotonic deadline."""


class RandomPolicy:
    def act(self, observation: Observation, rng: random.Random) -> int:
        legal = [action for action, allowed in enumerate(legal_action_mask(observation)) if allowed]
        if not legal:
            raise ValueError("No legal actions remain")
        return rng.choice(legal)


class CheckpointPolicy:
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

    def act(self, observation: Observation, rng: random.Random) -> int:
        del rng
        network = self.network
        network.eval()
        return select_greedy_action(network, observation)

    def _load(self) -> None:
        if self._network is not None:
            return
        network, game_config, _ = load_checkpoint(self.path)
        network.eval()
        self._network = network
        self._game_config = game_config


class EpsilonPolicy:
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
        return select_action(
            self.network,
            observation,
            epsilon=self.epsilon,
            rng=rng,
        )


@dataclass(frozen=True)
class MatchResult:
    seed: int
    winner: Optional[int]
    terminal_reason: str
    seat_turns: Tuple[int, ...]
    action_count: int
    stock_remaining: int
    telemetry: MatchTelemetry
    trajectories: Tuple[Tuple[EpisodeStep, ...], ...]


def play_match(
    policies: Sequence[Policy],
    config: MatchConfig,
    seed: int,
    record_trajectories: bool = False,
    deadline: Optional[float] = None,
) -> MatchResult:
    if not isinstance(config, MatchConfig):
        raise TypeError("config must be a MatchConfig")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if len(policies) != config.players:
        raise ValueError(
            "expected %d policies, got %d" % (config.players, len(policies))
        )
    for seat, policy in enumerate(policies):
        if not callable(getattr(policy, "act", None)):
            raise TypeError("policy at seat %d must define act" % seat)
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, Real)
        or not math.isfinite(float(deadline))
    ):
        raise ValueError("deadline must be a finite monotonic timestamp")

    _validate_policy_configs(policies, config.game)
    env = MultiplayerEnv(config)
    view = env.reset(seed)
    seat_rngs = tuple(
        random.Random(seed ^ ((seat + 1) * 0x9E3779B97F4A7C15))
        for seat in range(config.players)
    )
    trajectories: List[List[EpisodeStep]] = [
        [] for _ in range(config.players)
    ]
    action_count = 0

    while not view.done:
        _check_deadline(deadline)
        seat = view.current_seat
        observation = env.observe()
        action = policies[seat].act(observation, seat_rngs[seat])
        _check_deadline(deadline)
        state = encode_observation(observation) if record_trajectories else ()
        next_view = env.step(action)
        reward = float(
            observation.phase is Phase.DISCARD
            and next_view.winner == seat
        )
        if record_trajectories:
            trajectories[seat].append((state, action, reward))
        action_count += 1
        view = next_view

    if view.terminal_reason is None:
        raise RuntimeError("terminal match has no reason")
    seat_turns = tuple(
        config.game.max_turns - remaining
        for remaining in view.turns_remaining
    )
    return MatchResult(
        seed=seed,
        winner=view.winner,
        terminal_reason=view.terminal_reason,
        seat_turns=seat_turns,
        action_count=action_count,
        stock_remaining=view.stock_remaining,
        telemetry=env.telemetry(),
        trajectories=tuple(tuple(steps) for steps in trajectories),
    )


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
    results = []
    for rotation in range(len(lineup)):
        seated = lineup[rotation:] + lineup[:rotation]
        results.append(
            play_match(
                seated,
                config,
                seed,
                record_trajectories=record_trajectories,
            )
        )
    return results


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
