"""Resumable shared-deck league training for one protected challenger."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.environment import ENCODING_VERSION, STATE_DIM, GameConfig
from src.long_train import (
    _atomic_torch_save,
    _dict_with_fields,
    _nonnegative_int,
    _positive_int,
    _validate_finite_tree,
)
from src.train import EpisodeReplay, discounted_returns, _train_batch


LEAGUE_STATE_VERSION = 1
ALGORITHM = "champion_league_monte_carlo_q_regression"
RANDOM_OPPONENT = "random"
TRAINING_SEED_BASE = 1_000_000_000_000


@dataclass(frozen=True)
class LeagueConfig:
    game: GameConfig = field(default_factory=lambda: GameConfig(max_turns=60))
    batch_size: int = 64
    replay_capacity: int = 10_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    updates_per_match: int = 4
    grad_clip: float = 1.0
    epsilon_start: float = 0.2
    epsilon_end: float = 0.05
    epsilon_decay_matches: int = 10_000
    player_counts: Tuple[int, ...] = (2, 3)

    def __post_init__(self) -> None:
        if (
            self.game.num_decks,
            self.game.cards_in_hand,
            self.game.required_sequences,
            self.game.max_turns,
        ) != (3, 21, 5, 60):
            raise ValueError(
                "league training requires 3 decks, 21 cards, "
                "5 pure sequences, and 60 turns"
            )
        for name in (
            "batch_size",
            "replay_capacity",
            "updates_per_match",
            "epsilon_decay_matches",
        ):
            _positive_int(getattr(self, name), name)
        for name in ("learning_rate", "gamma", "grad_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError("%s must be finite and positive" % name)
        if self.gamma > 1:
            raise ValueError("gamma must be in (0, 1]")
        for name in ("epsilon_start", "epsilon_end"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ValueError("%s must be in [0, 1]" % name)
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end cannot exceed epsilon_start")
        if (
            not isinstance(self.player_counts, tuple)
            or not self.player_counts
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 2 <= count <= 6
                for count in self.player_counts
            )
        ):
            raise ValueError("player_counts must be a nonempty tuple in 2..6")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["game"] = self.game.to_dict()
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LeagueConfig":
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("league_config fields do not match this trainer")
        values = dict(raw)
        values["game"] = GameConfig.from_dict(values["game"])
        if not isinstance(values["player_counts"], tuple):
            raise TypeError("player_counts must be a tuple")
        return cls(**values)


@dataclass(frozen=True)
class _Opponent:
    label: str
    sha256: str
    policy: Any


class LeagueSession:
    def __init__(
        self,
        initial_model: str,
        opponent_paths: Sequence[str] = (),
        config: Optional[LeagueConfig] = None,
        seed: int = 41,
    ) -> None:
        import torch
        import torch.nn as nn
        from src.model import load_checkpoint

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self.config = LeagueConfig() if config is None else config
        self.initial_seed = seed
        self.parent_sha256 = _sha256_file(Path(initial_model))
        network, saved_game, _ = load_checkpoint(
            initial_model, device=torch.device("cpu")
        )
        _require_compatible_game(saved_game, self.config.game, "initial model")
        self.network = network
        self.network.train()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = EpisodeReplay(self.config.replay_capacity)

        seed_source = random.Random(seed)
        self.deal_rng = random.Random(seed_source.getrandbits(64))
        self.opponent_rng = random.Random(seed_source.getrandbits(64))
        self.replay_rng = random.Random(seed_source.getrandbits(64))
        torch_rng = torch.Generator(device="cpu")
        torch_rng.manual_seed(seed_source.getrandbits(63))
        self.torch_rng_state = torch_rng.get_state()

        self._parent_path = str(initial_model)
        self._opponents = _load_opponents(
            self._parent_path, opponent_paths, self.config.game
        )
        self.opponent_hashes = tuple(opponent.sha256 for opponent in self._opponents)
        self.opponent_pool_generation = 0

        self.completed_matches = 0
        self.update_matches = 0
        self.total_updates = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.no_action_matches = 0
        self.epsilon_progress = 0
        self.at_match_boundary = True

    @property
    def current_epsilon(self) -> float:
        fraction = min(
            1.0,
            self.epsilon_progress / float(self.config.epsilon_decay_matches),
        )
        return self.config.epsilon_start + (
            self.config.epsilon_end - self.config.epsilon_start
        ) * fraction

    def train_match(self, deadline: Optional[float] = None) -> Dict[str, Any]:
        import torch
        from src.arena import EpsilonPolicy, MatchConfig, MatchInterrupted, play_match

        rng_snapshot = (
            self.deal_rng.getstate(),
            self.opponent_rng.getstate(),
            self.replay_rng.getstate(),
        )
        self.at_match_boundary = False
        match_complete = False
        try:
            players = self.opponent_rng.choice(self.config.player_counts)
            challenger_seat = self.completed_matches % players
            epsilon = self.current_epsilon
            policies = [
                self.opponent_rng.choice(self._opponents).policy
                for _ in range(players)
            ]
            policies[challenger_seat] = EpsilonPolicy(
                self.network, epsilon
            )
            match_seed = TRAINING_SEED_BASE + self.deal_rng.getrandbits(39)
            result = play_match(
                tuple(policies),
                MatchConfig(game=self.config.game, players=players),
                seed=match_seed,
                record_trajectories=True,
                deadline=deadline,
            )
            match_complete = True
        except MatchInterrupted:
            self._restore_rngs(rng_snapshot)
            self.at_match_boundary = True
            raise
        except BaseException:
            if not match_complete:
                self._restore_rngs(rng_snapshot)
                self.at_match_boundary = True
            raise

        trajectory = result.trajectories[challenger_seat]
        terminal_reward = 1.0 if result.winner == challenger_seat else 0.0
        if trajectory:
            rewards = [0.0] * len(trajectory)
            rewards[-1] = terminal_reward
            returns = discounted_returns(rewards, self.config.gamma)
            episode = tuple(
                (_state_tuple(step[0]), _action(step[1]), target)
                for step, target in zip(trajectory, returns)
            )
            self.replay.append(episode)
            previous_torch_state = torch.get_rng_state()
            try:
                torch.set_rng_state(self.torch_rng_state)
                for _ in range(self.config.updates_per_match):
                    batch = self.replay.sample(
                        self.replay_rng,
                        min(self.config.batch_size, len(self.replay)),
                    )
                    _train_batch(
                        self.network,
                        self.optimizer,
                        self.loss_fn,
                        batch,
                        torch.device("cpu"),
                        self.config.grad_clip,
                    )
                    self.total_updates += 1
                self.torch_rng_state = torch.get_rng_state()
            finally:
                torch.set_rng_state(previous_torch_state)
            self.update_matches += 1
        else:
            self.no_action_matches += 1

        self.completed_matches += 1
        self.epsilon_progress += 1
        if result.winner is None:
            self.draws += 1
        elif result.winner == challenger_seat:
            self.wins += 1
        else:
            self.losses += 1
        self.at_match_boundary = True
        return {
            "match": self.completed_matches,
            "seed": result.seed,
            "players": players,
            "challenger_seat": challenger_seat,
            "winner": result.winner,
            "won": result.winner == challenger_seat,
            "terminal_reason": result.terminal_reason,
            "actions": len(trajectory),
            "updated": bool(trajectory),
            "epsilon": epsilon,
        }

    def state_dict(self) -> Dict[str, Any]:
        if not self.at_match_boundary:
            raise RuntimeError("cannot checkpoint a partial match")
        optimizer_state = self.optimizer.state_dict()
        _validate_finite_tree(optimizer_state, "optimizer_state_dict")
        return {
            "league_state_version": LEAGUE_STATE_VERSION,
            "encoding_version": ENCODING_VERSION,
            "algorithm": ALGORITHM,
            "league_config": self.config.to_dict(),
            "initial_seed": self.initial_seed,
            "parent_sha256": self.parent_sha256,
            "opponent_hashes": self.opponent_hashes,
            "opponent_pool_generation": self.opponent_pool_generation,
            "replay_preserved_across_pool_refresh": True,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "replay_state_dict": self.replay.state_dict(),
            "rng_states": {
                "deal": self.deal_rng.getstate(),
                "opponent": self.opponent_rng.getstate(),
                "replay": self.replay_rng.getstate(),
                "torch": self.torch_rng_state,
            },
            "counters": {
                "completed_matches": self.completed_matches,
                "update_matches": self.update_matches,
                "total_updates": self.total_updates,
                "wins": self.wins,
                "losses": self.losses,
                "draws": self.draws,
                "no_action_matches": self.no_action_matches,
                "epsilon_progress": self.epsilon_progress,
            },
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        import torch

        required = {
            "league_state_version",
            "encoding_version",
            "algorithm",
            "league_config",
            "initial_seed",
            "parent_sha256",
            "opponent_hashes",
            "opponent_pool_generation",
            "replay_preserved_across_pool_refresh",
            "network_state_dict",
            "optimizer_state_dict",
            "replay_state_dict",
            "rng_states",
            "counters",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("league checkpoint fields do not match this trainer")
        if payload["league_state_version"] != LEAGUE_STATE_VERSION:
            raise ValueError("league checkpoint version is incompatible")
        if payload["encoding_version"] != ENCODING_VERSION:
            raise ValueError("league checkpoint encoding is incompatible")
        if payload["algorithm"] != ALGORITHM:
            raise ValueError("league checkpoint algorithm is incompatible")
        if LeagueConfig.from_dict(payload["league_config"]) != self.config:
            raise ValueError("league checkpoint config changed")
        if payload["initial_seed"] != self.initial_seed:
            raise ValueError("league checkpoint seed changed")
        if payload["parent_sha256"] != self.parent_sha256:
            raise ValueError("league checkpoint parent changed")
        if payload["opponent_hashes"] != self.opponent_hashes:
            raise ValueError("league checkpoint opponent pool changed")
        pool_generation = _nonnegative_int(
            payload["opponent_pool_generation"],
            "opponent_pool_generation",
        )
        if payload["replay_preserved_across_pool_refresh"] is not True:
            raise ValueError("league checkpoint replay refresh policy changed")
        _validate_finite_tree(
            payload["optimizer_state_dict"], "optimizer_state_dict"
        )

        counters = _dict_with_fields(
            payload["counters"],
            {
                "completed_matches",
                "update_matches",
                "total_updates",
                "wins",
                "losses",
                "draws",
                "no_action_matches",
                "epsilon_progress",
            },
            "counters",
        )
        validated_counters = {
            name: _nonnegative_int(value, name)
            for name, value in counters.items()
        }
        completed = validated_counters["completed_matches"]
        if (
            validated_counters["wins"]
            + validated_counters["losses"]
            + validated_counters["draws"]
            != completed
        ):
            raise ValueError("league outcomes do not sum to completed matches")
        if (
            validated_counters["update_matches"]
            + validated_counters["no_action_matches"]
            != completed
        ):
            raise ValueError("league update counters do not sum to matches")
        if validated_counters["epsilon_progress"] != completed:
            raise ValueError("league epsilon progress must equal completed matches")
        if (
            validated_counters["total_updates"]
            != validated_counters["update_matches"]
            * self.config.updates_per_match
        ):
            raise ValueError("league update count is inconsistent")

        rng_states = _dict_with_fields(
            payload["rng_states"],
            {"deal", "opponent", "replay", "torch"},
            "rng_states",
        )
        candidate_rngs = {}
        for name in ("deal", "opponent", "replay"):
            candidate = random.Random()
            try:
                candidate.setstate(rng_states[name])
            except (TypeError, ValueError) as exc:
                raise ValueError("%s RNG state is invalid" % name) from exc
            candidate_rngs[name] = candidate
        torch_state = rng_states["torch"]
        if (
            not isinstance(torch_state, torch.Tensor)
            or torch_state.dtype != torch.uint8
            or torch_state.ndim != 1
        ):
            raise ValueError("torch RNG state is invalid")

        self.network.load_state_dict(payload["network_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.replay.load_state_dict(payload["replay_state_dict"])
        self.deal_rng.setstate(candidate_rngs["deal"].getstate())
        self.opponent_rng.setstate(candidate_rngs["opponent"].getstate())
        self.replay_rng.setstate(candidate_rngs["replay"].getstate())
        self.torch_rng_state = torch_state.clone()
        self.opponent_pool_generation = pool_generation
        for name, value in validated_counters.items():
            setattr(self, name, value)
        self.network.train()
        self.at_match_boundary = True

    @classmethod
    def load(
        cls,
        path: str,
        initial_model: str,
        opponent_paths: Sequence[str] = (),
        expected_config: Optional[LeagueConfig] = None,
    ) -> "LeagueSession":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("league checkpoint must be a dict")
        config = LeagueConfig.from_dict(payload.get("league_config", {}))
        if expected_config is not None and config != expected_config:
            raise ValueError("resume league_config does not match requested config")
        session = cls(
            initial_model=initial_model,
            opponent_paths=opponent_paths,
            config=config,
            seed=payload.get("initial_seed"),
        )
        session.load_state_dict(payload)
        return session

    def export_model(self, path: Path, role: str = "latest") -> None:
        from src.model import build_checkpoint

        payload = build_checkpoint(
            self.network,
            self.config.game,
            meta={
                "algorithm": ALGORITHM,
                "role": role,
                "matches": self.completed_matches,
                "updates": self.total_updates,
                "parent_sha256": self.parent_sha256,
                "opponent_hashes": self.opponent_hashes,
                "opponent_pool_generation": self.opponent_pool_generation,
                "replay_preserved_across_pool_refresh": True,
            },
        )
        _atomic_torch_save(payload, path)

    def refresh_opponent_pool(self, opponent_paths: Sequence[str]) -> None:
        if not self.at_match_boundary:
            raise RuntimeError("cannot refresh opponents during a match")
        opponents = _load_opponents(
            self._parent_path, opponent_paths, self.config.game
        )
        hashes = tuple(opponent.sha256 for opponent in opponents)
        if hashes == self.opponent_hashes:
            return
        self._opponents = opponents
        self.opponent_hashes = hashes
        self.opponent_pool_generation += 1

    def _restore_rngs(self, states: Tuple[Any, Any, Any]) -> None:
        self.deal_rng.setstate(states[0])
        self.opponent_rng.setstate(states[1])
        self.replay_rng.setstate(states[2])


def save_session_checkpoint(session: LeagueSession, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(session.state_dict(), output_dir / "latest-state.pt")
    session.export_model(output_dir / "latest-model.pt")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model", required=True)
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matches", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, default=82_800)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--resume")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    stop = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        stop[0] = True

    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        from src.arena import MatchInterrupted

        signal.signal(signal.SIGTERM, request_stop)
        _nonnegative_int(args.matches, "matches")
        _positive_int(args.checkpoint_every, "checkpoint_every")
        if (
            isinstance(args.max_seconds, bool)
            or not isinstance(args.max_seconds, (int, float))
            or not math.isfinite(float(args.max_seconds))
            or args.max_seconds < 0
        ):
            raise ValueError("max_seconds must be finite and nonnegative")
        output_dir = Path(args.output_dir)
        session = (
            LeagueSession.load(
                args.resume,
                args.initial_model,
                args.opponent,
                expected_config=LeagueConfig(),
            )
            if args.resume
            else LeagueSession(
                args.initial_model,
                opponent_paths=args.opponent,
                seed=args.seed,
            )
        )
        save_session_checkpoint(session, output_dir)
        target = session.completed_matches + args.matches
        deadline = time.monotonic() + float(args.max_seconds)
        reason = "matches"
        while session.completed_matches < target:
            if stop[0]:
                reason = "stopped"
                break
            if time.monotonic() >= deadline:
                reason = "budget"
                break
            try:
                result = session.train_match(deadline=deadline)
            except MatchInterrupted:
                reason = "budget"
                break
            print(json.dumps(result, sort_keys=True), flush=True)
            if session.completed_matches % args.checkpoint_every == 0:
                save_session_checkpoint(session, output_dir)
        save_session_checkpoint(session, output_dir)
        print(
            json.dumps(
                {
                    "status": reason,
                    "matches": session.completed_matches,
                    "updates": session.total_updates,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _require_compatible_game(
    actual: GameConfig, expected: GameConfig, label: str
) -> None:
    if (
        actual.num_decks,
        actual.cards_in_hand,
        actual.required_sequences,
    ) != (
        expected.num_decks,
        expected.cards_in_hand,
        expected.required_sequences,
    ):
        raise ValueError("%s must use 3 decks, 21 cards, and 5 sequences" % label)


def _load_opponents(
    protected_model: str,
    opponent_paths: Sequence[str],
    game: GameConfig,
) -> Tuple[_Opponent, ...]:
    from src.arena import CheckpointPolicy, RandomPolicy

    opponents = []
    seen = set()
    for path in (protected_model, *opponent_paths, RANDOM_OPPONENT):
        if path == RANDOM_OPPONENT:
            if RANDOM_OPPONENT not in seen:
                opponents.append(
                    _Opponent(RANDOM_OPPONENT, RANDOM_OPPONENT, RandomPolicy())
                )
                seen.add(RANDOM_OPPONENT)
            continue
        digest = _sha256_file(Path(path))
        if digest in seen:
            continue
        policy = CheckpointPolicy(str(path))
        _require_compatible_game(
            policy.game_config, game, "opponent %s" % path
        )
        policy.network.eval()
        opponents.append(_Opponent(str(path), digest, policy))
        seen.add(digest)
    return tuple(opponents)


def _state_tuple(raw: Any) -> Tuple[float, ...]:
    if not isinstance(raw, tuple) or len(raw) != STATE_DIM:
        raise ValueError("arena trajectory state must be a 164-item tuple")
    result = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in result):
        raise ValueError("arena trajectory state must contain finite values")
    return result


def _action(raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < 54:
        raise ValueError("arena trajectory action must be in 0..53")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


if __name__ == "__main__":
    sys.exit(main())
