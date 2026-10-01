"""Headless multiplayer policy matches and a bounded tournament CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import random
import time
from typing import List, Optional, Protocol, Sequence, Tuple

from src.environment import (
    GameConfig,
    Observation,
    Phase,
    encode_observation,
    legal_action_mask,
)
from src.multiplayer import MatchConfig, MultiplayerEnv


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
        from src.model import select_greedy_action

        network = self.network
        network.eval()
        return select_greedy_action(network, observation)

    def _load(self) -> None:
        if self._network is not None:
            return
        from src.model import load_checkpoint

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
        from src.model import select_action

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


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="checkpoint path or the sentinel 'random'; repeat once per seat",
    )
    parser.add_argument("--games", type=_positive_int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=_positive_int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    players = len(args.model)
    if players < 2 or players > 6:
        parser.error("--model must be repeated 2 through 6 times")
    policies: Tuple[Policy, ...] = tuple(
        RandomPolicy() if value == "random" else CheckpointPolicy(value)
        for value in args.model
    )
    labels = [
        "random" if value == "random" else Path(value).stem
        for value in args.model
    ]
    config = MatchConfig(
        game=GameConfig(max_turns=args.max_turns),
        players=players,
    )
    games = []
    for index in range(args.games):
        rotation = index % players
        seated = policies[rotation:] + policies[:rotation]
        result = play_match(seated, config, seed=args.seed + index)
        winner_agent = (
            None
            if result.winner is None
            else (rotation + result.winner) % players
        )
        games.append(
            {
                "seed": result.seed,
                "seat_order": [
                    (rotation + seat) % players for seat in range(players)
                ],
                "winner_seat": result.winner,
                "winner_agent": winner_agent,
                "winner_label": (
                    None if winner_agent is None else labels[winner_agent]
                ),
                "terminal_reason": result.terminal_reason,
                "seat_turns": result.seat_turns,
                "action_count": result.action_count,
                "stock_remaining": result.stock_remaining,
            }
        )

    report = {
        "agents": [
            {"index": index, "label": label, "model": model}
            for index, (label, model) in enumerate(zip(labels, args.model))
        ],
        "config": {
            "players": players,
            "game": config.game.to_dict(),
        },
        "games": games,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
