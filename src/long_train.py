"""Resumable full-hand curriculum training for Papplu."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from src.environment import ENCODING_VERSION, GameConfig, PappluEnv, encode_observation
from src.train import EpisodeReplay, discounted_returns, _train_batch


TRAINING_STATE_VERSION = 1
STATUS_SCHEMA_VERSION = 1
ALGORITHM = "full21_curriculum_monte_carlo_q_regression"
ORDINARY_VALIDATION_SEED = 30_000_000
CURRICULUM_VALIDATION_SEED = 20_000_000
FINAL_CONFIRMATION_SEED = 40_000_000
PUBLISHED_SNAPSHOT_FILES = (
    "latest-state.pt",
    "latest-model.pt",
    "best-model.pt",
    "baseline-model.pt",
    "metrics.jsonl",
)


class ValidationInterrupted(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class TrainingConfig:
    architecture: str = "suit_conv"
    num_decks: int = 3
    cards_in_hand: int = 21
    required_sequences: int = 5
    max_turns: int = 60
    replay_capacity: int = 10_000
    batch_size: int = 64
    updates_per_episode: int = 4
    learning_rate: float = 1e-3
    gamma: float = 0.99
    grad_clip: float = 1.0
    epsilon_start: float = 0.3
    epsilon_end: float = 0.1
    epsilon_decay_episodes: int = 10_000
    curriculum_fraction: float = 0.75
    curriculum_distances: Tuple[int, ...] = (1, 2, 4, 8)
    minimum_stage_episodes: int = 1_000
    promotion_passes: int = 2
    promotion_margin: float = 0.10
    promotion_floor: float = 0.25
    validation_games: int = 64
    validate_every: int = 1_000
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.architecture != "suit_conv":
            raise ValueError("long training requires suit_conv architecture")
        if (
            self.num_decks,
            self.cards_in_hand,
            self.required_sequences,
        ) != (3, 21, 5):
            raise ValueError(
                "long training requires 3 decks, 21 cards, and 5 pure sequences"
            )
        for name in (
            "max_turns",
            "replay_capacity",
            "batch_size",
            "updates_per_episode",
            "epsilon_decay_episodes",
            "minimum_stage_episodes",
            "promotion_passes",
            "validation_games",
            "validate_every",
            "torch_threads",
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
        for name in (
            "epsilon_start",
            "epsilon_end",
            "curriculum_fraction",
            "promotion_margin",
            "promotion_floor",
        ):
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
        if self.curriculum_distances != (1, 2, 4, 8):
            raise ValueError("curriculum_distances must be (1, 2, 4, 8)")

    @property
    def game_config(self) -> GameConfig:
        return GameConfig(
            num_decks=self.num_decks,
            cards_in_hand=self.cards_in_hand,
            required_sequences=self.required_sequences,
            max_turns=self.max_turns,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        if not isinstance(data, dict):
            raise TypeError("training_config must be a dict")
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("training_config fields do not match this trainer")
        values = dict(data)
        distances = values["curriculum_distances"]
        if not isinstance(distances, tuple):
            raise TypeError("curriculum_distances must be a tuple")
        return cls(**values)


class TrainingSession:
    def __init__(
        self,
        config: TrainingConfig,
        seed: int,
        initial_model: Optional[str] = None,
        initial_stage_index: int = 0,
    ) -> None:
        import torch
        import torch.nn as nn
        from src.model import QNetwork, load_checkpoint

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        if (
            isinstance(initial_stage_index, bool)
            or not isinstance(initial_stage_index, int)
            or not 0 <= initial_stage_index <= len(config.curriculum_distances)
        ):
            raise ValueError(
                "initial_stage_index must be in 0..%d"
                % len(config.curriculum_distances)
            )
        self.config = config
        self.initial_seed = seed
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(seed)

        if initial_model is None:
            self.network = QNetwork(architecture=config.architecture)
        else:
            network, saved_game, _ = load_checkpoint(
                initial_model,
                device=torch.device("cpu"),
                expected_architecture=config.architecture,
            )
            expected = config.game_config
            if (
                saved_game.num_decks != expected.num_decks
                or saved_game.cards_in_hand != expected.cards_in_hand
                or saved_game.required_sequences != expected.required_sequences
            ):
                raise ValueError(
                    "initial model must use 3 decks, 21 cards, and 5 pure sequences"
                )
            self.network = network
        self.network.train()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = EpisodeReplay(config.replay_capacity)

        seed_source = random.Random(seed)
        self.deal_rng = random.Random(seed_source.getrandbits(64))
        self.action_rng = random.Random(seed_source.getrandbits(64))
        self.replay_rng = random.Random(seed_source.getrandbits(64))

        self.total_episodes = 0
        self.total_updates = 0
        self.stage_index = initial_stage_index
        self.stage_episodes = 0
        self.epsilon_progress = 0
        self.consecutive_passing_validations = 0
        self.curriculum_episodes = 0
        self.curriculum_wins = 0
        self.random_episodes = 0
        self.random_wins = 0

        self.baseline_validation: Optional[Dict[str, Any]] = None
        self.full_validation: Optional[Dict[str, Any]] = None
        self.curriculum_validation: Optional[Dict[str, Any]] = None
        self.last_validation_episode = -1
        self.best_score = -1.0
        self.best_episode = 0
        self.baseline_model_state = _clone_model_state(self.network.state_dict())
        self.best_model_state = _clone_model_state(self.network.state_dict())
        self.at_episode_boundary = True

    @property
    def current_distance(self) -> Optional[int]:
        if self.stage_index == len(self.config.curriculum_distances):
            return None
        return self.config.curriculum_distances[self.stage_index]

    @property
    def current_epsilon(self) -> float:
        fraction = min(
            1.0,
            self.epsilon_progress / float(self.config.epsilon_decay_episodes),
        )
        return self.config.epsilon_start + (
            self.config.epsilon_end - self.config.epsilon_start
        ) * fraction

    @classmethod
    def load(
        cls,
        path: str,
        expected_config: Optional[TrainingConfig] = None,
    ) -> "TrainingSession":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_state_dict(payload, expected_config=expected_config)

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, Any],
        expected_config: Optional[TrainingConfig] = None,
    ) -> "TrainingSession":
        import torch
        from src.model import CHECKPOINT_VERSION

        if not isinstance(payload, dict):
            raise ValueError("training checkpoint must be a dict")
        required = {
            "training_state_version",
            "encoding_version",
            "model_checkpoint_version",
            "algorithm",
            "training_config",
            "initial_seed",
            "network_state_dict",
            "optimizer_state_dict",
            "replay_state_dict",
            "counters",
            "curriculum",
            "validation",
            "rng_states",
            "baseline_model_state_dict",
            "best_model_state_dict",
            "source_trace",
        }
        if set(payload) != required:
            raise ValueError("training checkpoint fields do not match this trainer")
        if payload["training_state_version"] != TRAINING_STATE_VERSION:
            raise ValueError(
                "unsupported training_state_version %r"
                % (payload["training_state_version"],)
            )
        if payload["encoding_version"] != ENCODING_VERSION:
            raise ValueError("training checkpoint encoding version is incompatible")
        if payload["model_checkpoint_version"] != CHECKPOINT_VERSION:
            raise ValueError("training checkpoint model version is incompatible")
        if payload["algorithm"] != ALGORITHM:
            raise ValueError("training checkpoint algorithm is incompatible")

        config = TrainingConfig.from_dict(payload["training_config"])
        if expected_config is not None and config != expected_config:
            raise ValueError(
                "resume training_config does not match the requested configuration"
            )
        source_trace = _dict_with_fields(
            payload["source_trace"],
            {
                "module",
                "training_state_version",
                "encoding_version",
                "model_checkpoint_version",
                "rules",
            },
            "source_trace",
        )
        expected_trace = {
            "module": "src.long_train",
            "training_state_version": TRAINING_STATE_VERSION,
            "encoding_version": ENCODING_VERSION,
            "model_checkpoint_version": CHECKPOINT_VERSION,
            "rules": config.game_config.to_dict(),
        }
        if source_trace != expected_trace:
            raise ValueError("training checkpoint source_trace is incompatible")
        seed = payload["initial_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("training checkpoint seed must be an int")
        _validate_finite_tree(
            payload["optimizer_state_dict"], "optimizer_state_dict"
        )

        session = cls(config=config, seed=seed)
        session.network.load_state_dict(payload["network_state_dict"], strict=True)
        session.optimizer.load_state_dict(payload["optimizer_state_dict"])
        session.replay.load_state_dict(payload["replay_state_dict"])

        counters = _dict_with_fields(
            payload["counters"],
            {
                "total_episodes",
                "total_updates",
                "curriculum_episodes",
                "curriculum_wins",
                "random_episodes",
                "random_wins",
            },
            "counters",
        )
        for name, value in counters.items():
            setattr(session, name, _nonnegative_int(value, name))
        if session.curriculum_wins > session.curriculum_episodes:
            raise ValueError("curriculum wins exceed curriculum episodes")
        if session.random_wins > session.random_episodes:
            raise ValueError("random wins exceed random episodes")
        if (
            session.curriculum_episodes + session.random_episodes
            != session.total_episodes
        ):
            raise ValueError("episode counters do not sum to total_episodes")
        if (
            session.total_updates
            != session.total_episodes * config.updates_per_episode
        ):
            raise ValueError("update count does not match completed episodes")

        curriculum = _dict_with_fields(
            payload["curriculum"],
            {
                "stage_index",
                "stage_episodes",
                "epsilon_progress",
                "consecutive_passing_validations",
            },
            "curriculum",
        )
        session.stage_index = _nonnegative_int(
            curriculum["stage_index"], "stage_index"
        )
        if session.stage_index > len(config.curriculum_distances):
            raise ValueError("curriculum stage_index is out of range")
        session.stage_episodes = _nonnegative_int(
            curriculum["stage_episodes"], "stage_episodes"
        )
        if session.stage_episodes > session.total_episodes:
            raise ValueError("stage_episodes exceeds total_episodes")
        session.epsilon_progress = _nonnegative_int(
            curriculum["epsilon_progress"], "epsilon_progress"
        )
        if session.epsilon_progress != session.stage_episodes:
            raise ValueError("epsilon progress must equal stage episode count")
        session.consecutive_passing_validations = _nonnegative_int(
            curriculum["consecutive_passing_validations"],
            "consecutive_passing_validations",
        )
        if (
            session.consecutive_passing_validations
            >= config.promotion_passes
        ):
            raise ValueError("saved validation pass streak should have promoted")

        validation = _dict_with_fields(
            payload["validation"],
            {
                "baseline",
                "full",
                "curriculum",
                "last_validation_episode",
                "best_score",
                "best_episode",
            },
            "validation",
        )
        session.baseline_validation = _optional_validation(
            validation["baseline"], "baseline"
        )
        session.full_validation = _optional_validation(validation["full"], "full")
        session.curriculum_validation = _optional_validation(
            validation["curriculum"], "curriculum"
        )
        session.last_validation_episode = _integer(
            validation["last_validation_episode"], "last_validation_episode"
        )
        best_score = validation["best_score"]
        if (
            isinstance(best_score, bool)
            or not isinstance(best_score, (int, float))
            or not math.isfinite(float(best_score))
            or not -1 <= best_score <= 1
        ):
            raise ValueError("best_score must be finite and in [-1, 1]")
        session.best_score = float(best_score)
        session.best_episode = _nonnegative_int(
            validation["best_episode"], "best_episode"
        )
        if session.best_episode > session.total_episodes:
            raise ValueError("best_episode exceeds total_episodes")
        session.baseline_model_state = _validate_model_state(
            payload["baseline_model_state_dict"], session.network.state_dict(),
            "baseline_model_state_dict",
        )
        session.best_model_state = _validate_model_state(
            payload["best_model_state_dict"], session.network.state_dict(),
            "best_model_state_dict",
        )

        rng_states = _dict_with_fields(
            payload["rng_states"], {"deal", "action", "replay", "torch"}, "rng_states"
        )
        for name, rng in (
            ("deal", session.deal_rng),
            ("action", session.action_rng),
            ("replay", session.replay_rng),
        ):
            candidate = random.Random()
            try:
                candidate.setstate(rng_states[name])
            except (TypeError, ValueError) as exc:
                raise ValueError("%s RNG state is invalid" % name) from exc
            rng.setstate(rng_states[name])
        torch_state = rng_states["torch"]
        if (
            not isinstance(torch_state, torch.Tensor)
            or torch_state.dtype != torch.uint8
            or torch_state.ndim != 1
        ):
            raise ValueError("torch RNG state is invalid")
        torch.set_rng_state(torch_state)
        session.network.train()
        session.at_episode_boundary = True
        return session

    def state_dict(self) -> Dict[str, Any]:
        import torch
        from src.model import CHECKPOINT_VERSION

        optimizer_state = self.optimizer.state_dict()
        _validate_finite_tree(optimizer_state, "optimizer_state_dict")
        return {
            "training_state_version": TRAINING_STATE_VERSION,
            "encoding_version": ENCODING_VERSION,
            "model_checkpoint_version": CHECKPOINT_VERSION,
            "algorithm": ALGORITHM,
            "training_config": self.config.to_dict(),
            "initial_seed": self.initial_seed,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "replay_state_dict": self.replay.state_dict(),
            "counters": {
                "total_episodes": self.total_episodes,
                "total_updates": self.total_updates,
                "curriculum_episodes": self.curriculum_episodes,
                "curriculum_wins": self.curriculum_wins,
                "random_episodes": self.random_episodes,
                "random_wins": self.random_wins,
            },
            "curriculum": {
                "stage_index": self.stage_index,
                "stage_episodes": self.stage_episodes,
                "epsilon_progress": self.epsilon_progress,
                "consecutive_passing_validations": (
                    self.consecutive_passing_validations
                ),
            },
            "validation": {
                "baseline": self.baseline_validation,
                "full": self.full_validation,
                "curriculum": self.curriculum_validation,
                "last_validation_episode": self.last_validation_episode,
                "best_score": self.best_score,
                "best_episode": self.best_episode,
            },
            "rng_states": {
                "deal": self.deal_rng.getstate(),
                "action": self.action_rng.getstate(),
                "replay": self.replay_rng.getstate(),
                "torch": torch.get_rng_state(),
            },
            "baseline_model_state_dict": self.baseline_model_state,
            "best_model_state_dict": self.best_model_state,
            "source_trace": {
                "module": "src.long_train",
                "training_state_version": TRAINING_STATE_VERSION,
                "encoding_version": ENCODING_VERSION,
                "model_checkpoint_version": CHECKPOINT_VERSION,
                "rules": self.config.game_config.to_dict(),
            },
        }

    def run_episode(self, deadline: Optional[float] = None) -> Optional[Dict[str, Any]]:
        import torch
        from src.model import select_action

        epsilon = self.current_epsilon
        distance = self.current_distance
        rng_snapshot = {
            "deal": self.deal_rng.getstate(),
            "action": self.action_rng.getstate(),
            "replay": self.replay_rng.getstate(),
            "torch": torch.get_rng_state(),
        }
        self.at_episode_boundary = False
        replay_committed = False
        try:
            use_curriculum = (
                distance is not None
                and self.deal_rng.random() < self.config.curriculum_fraction
            )
            deal_seed = self.deal_rng.getrandbits(63)
            env = PappluEnv(self.config.game_config)
            obs = (
                env.reset_curriculum(distance, seed=deal_seed)
                if use_curriculum
                else env.reset(seed=deal_seed)
            )
            steps = []
            while not obs.done:
                if deadline is not None and time.monotonic() >= deadline:
                    self._restore_rng_snapshot(rng_snapshot)
                    self.at_episode_boundary = True
                    return None
                action = select_action(
                    self.network,
                    obs,
                    epsilon=epsilon,
                    rng=self.action_rng,
                    device=torch.device("cpu"),
                )
                state = encode_observation(obs)
                obs = env.step(action)
                steps.append((state, action, float(obs.last_reward)))

            returns = discounted_returns(
                [reward for _state, _action, reward in steps],
                self.config.gamma,
            )
            episode = tuple(
                (state, action, target)
                for (state, action, _reward), target in zip(steps, returns)
            )
            self.replay.append(episode)
            replay_committed = True
            for _ in range(self.config.updates_per_episode):
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

            self.total_episodes += 1
            self.stage_episodes += 1
            self.epsilon_progress += 1
            if use_curriculum:
                self.curriculum_episodes += 1
                if obs.won:
                    self.curriculum_wins += 1
            else:
                self.random_episodes += 1
                if obs.won:
                    self.random_wins += 1
            self.at_episode_boundary = True
            return {
                "episode": self.total_episodes,
                "source": (
                    "curriculum_distance_%d" % distance
                    if use_curriculum
                    else "ordinary_full21"
                ),
                "won": bool(obs.won),
                "epsilon": epsilon,
                "steps": len(steps),
            }
        except BaseException:
            if not replay_committed:
                self._restore_rng_snapshot(rng_snapshot)
                self.at_episode_boundary = True
            raise

    def initialize_validation(
        self,
        deadline: Optional[float] = None,
        stop_requested: Callable[[], bool] = lambda: False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        full = {
            "seed": ORDINARY_VALIDATION_SEED,
            "distance": None,
            **evaluate_policy_pair(
                self.network,
                self.config.game_config,
                self.config.validation_games,
                ORDINARY_VALIDATION_SEED,
                deadline=deadline,
                stop_requested=stop_requested,
            ),
        }
        curriculum = {
            "seed": CURRICULUM_VALIDATION_SEED,
            "stage": self.stage_index,
            "distance": self.current_distance,
            **evaluate_policy_pair(
                self.network,
                self.config.game_config,
                self.config.validation_games,
                CURRICULUM_VALIDATION_SEED,
                distance=self.current_distance,
                deadline=deadline,
                stop_requested=stop_requested,
            ),
        }
        self.baseline_validation = full
        self.full_validation = full
        self.curriculum_validation = curriculum
        self.last_validation_episode = self.total_episodes
        self.best_score = float(full["greedy"]["win_rate"])
        self.best_episode = self.total_episodes
        self.best_model_state = _clone_model_state(self.network.state_dict())
        return full, curriculum

    def run_validation(
        self,
        deadline: Optional[float] = None,
        stop_requested: Callable[[], bool] = lambda: False,
    ) -> Dict[str, Any]:
        stage = self.stage_index
        distance = self.current_distance
        full = {
            "seed": ORDINARY_VALIDATION_SEED,
            "distance": None,
            **evaluate_policy_pair(
                self.network,
                self.config.game_config,
                self.config.validation_games,
                ORDINARY_VALIDATION_SEED,
                deadline=deadline,
                stop_requested=stop_requested,
            ),
        }
        curriculum_seed = CURRICULUM_VALIDATION_SEED + stage * 10_000
        curriculum = {
            "seed": curriculum_seed,
            "stage": stage,
            "distance": distance,
            **evaluate_policy_pair(
                self.network,
                self.config.game_config,
                self.config.validation_games,
                curriculum_seed,
                distance=distance,
                deadline=deadline,
                stop_requested=stop_requested,
            ),
        }
        self.full_validation = full
        self.curriculum_validation = curriculum
        self.last_validation_episode = self.total_episodes

        full_score = float(full["greedy"]["win_rate"])
        best_changed = full_score > self.best_score
        if best_changed:
            self.best_score = full_score
            self.best_episode = self.total_episodes
            self.best_model_state = _clone_model_state(self.network.state_dict())

        passed = curriculum_passed(curriculum, self.config)
        promoted = False
        if self.stage_index == len(self.config.curriculum_distances):
            self.consecutive_passing_validations = 0
        elif self.stage_episodes < self.config.minimum_stage_episodes:
            self.consecutive_passing_validations = 0
        elif passed:
            self.consecutive_passing_validations += 1
        else:
            self.consecutive_passing_validations = 0
        if (
            self.stage_index < len(self.config.curriculum_distances)
            and self.consecutive_passing_validations >= self.config.promotion_passes
        ):
            self.stage_index += 1
            self.stage_episodes = 0
            self.epsilon_progress = 0
            self.consecutive_passing_validations = 0
            promoted = True

        return {
            "full": full,
            "curriculum": curriculum,
            "validated_stage": stage,
            "validated_distance": distance,
            "passed": passed,
            "promoted": promoted,
            "best_changed": best_changed,
        }

    def _restore_rng_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        import torch

        self.deal_rng.setstate(snapshot["deal"])
        self.action_rng.setstate(snapshot["action"])
        self.replay_rng.setstate(snapshot["replay"])
        torch.set_rng_state(snapshot["torch"])


def curriculum_passed(
    validation: Mapping[str, Any], config: TrainingConfig
) -> bool:
    greedy = float(validation["greedy"]["win_rate"])
    random_rate = float(validation["random"]["win_rate"])
    return greedy >= max(random_rate + config.promotion_margin, config.promotion_floor)


def evaluate_policy_pair(
    network: Any,
    config: GameConfig,
    games: int,
    seed: int,
    distance: Optional[int] = None,
    deadline: Optional[float] = None,
    stop_requested: Callable[[], bool] = lambda: False,
) -> Dict[str, Any]:
    from src.model import select_greedy_action, select_random_legal

    _positive_int(games, "games")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if distance is not None:
        _positive_int(distance, "distance")

    was_training = network.training
    network.eval()
    try:
        results: Dict[str, Any] = {}
        for policy in ("greedy", "random"):
            wins = 0
            for index in range(games):
                _check_validation_budget(deadline, stop_requested)
                env = PappluEnv(config)
                obs = (
                    env.reset(seed=seed + index)
                    if distance is None
                    else env.reset_curriculum(distance, seed=seed + index)
                )
                policy_rng = random.Random(seed + 10_000_000 + index)
                while not obs.done:
                    _check_validation_budget(deadline, stop_requested)
                    if policy == "greedy":
                        action = select_greedy_action(network, obs)
                    else:
                        action = select_random_legal(obs, policy_rng)
                    obs = env.step(action)
                wins += int(obs.won)
            results[policy] = {
                "games": games,
                "wins": wins,
                "win_rate": wins / games,
            }
        return results
    finally:
        network.train(was_training)


def _check_validation_budget(
    deadline: Optional[float], stop_requested: Callable[[], bool]
) -> None:
    if stop_requested():
        raise ValidationInterrupted("stopped")
    if deadline is not None and time.monotonic() >= deadline:
        raise ValidationInterrupted("budget")


def evaluate_final_confirmation(
    session: TrainingSession, games: int = 1_000
) -> Dict[str, Any]:
    return evaluate_policy_pair(
        session.network,
        session.config.game_config,
        games,
        FINAL_CONFIRMATION_SEED,
    )


def save_session_checkpoint(
    session: TrainingSession, output_dir: Path
) -> Dict[str, Dict[str, Any]]:
    if not session.at_episode_boundary:
        raise RuntimeError("cannot checkpoint a partial episode")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.jsonl").touch(exist_ok=True)
    _atomic_torch_save(
        _model_payload(session, session.network.state_dict(), "latest"),
        output_dir / "latest-model.pt",
    )
    _atomic_torch_save(
        _model_payload(session, session.best_model_state, "best"),
        output_dir / "best-model.pt",
    )
    _atomic_torch_save(
        _model_payload(session, session.baseline_model_state, "baseline"),
        output_dir / "baseline-model.pt",
    )
    _atomic_torch_save(session.state_dict(), output_dir / "latest-state.pt")
    return _snapshot_file_manifest(output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial-model")
    source.add_argument("--resume")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=1_000_000)
    parser.add_argument("--max-seconds", type=float, default=82_800)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--validate-every",
        type=int,
        default=None,
        help="Validation cadence. New sessions default to 1000 episodes.",
    )
    parser.add_argument(
        "--validation-games",
        type=int,
        default=None,
        help="Games per policy and validation source. New sessions default to 64.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Full-game turn limit. New sessions default to 60.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optimizer learning rate. New sessions default to 0.001.",
    )
    parser.add_argument(
        "--curriculum-fraction",
        type=float,
        default=None,
        help="Fraction of curriculum deals. New sessions default to 0.75.",
    )
    parser.add_argument(
        "--initial-stage",
        type=int,
        default=None,
        help="Initial curriculum stage 0..4. New sessions default to 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed. New sessions default to 41.",
    )
    return parser


def run(
    session: TrainingSession,
    output_dir: Path,
    max_episodes: int,
    max_seconds: float,
    checkpoint_every: int,
    stop_requested: Callable[[], bool] = lambda: False,
    initialize: bool = False,
) -> str:
    _nonnegative_int(max_episodes, "max_episodes")
    _positive_int(checkpoint_every, "checkpoint_every")
    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not math.isfinite(float(max_seconds))
        or max_seconds < 0
    ):
        raise ValueError("max_seconds must be finite and nonnegative")

    started = time.monotonic()
    deadline = started + float(max_seconds)
    files = save_session_checkpoint(session, output_dir)
    _write_status(output_dir, session, files, "running", None, started)
    if not initialize:
        _append_metric(
            output_dir,
            {
                "event": "resume",
                "episode": session.total_episodes,
                "stage": session.stage_index,
                "distance": session.current_distance,
            },
        )

    reason = "max_episodes"
    if initialize:
        try:
            full, curriculum = session.initialize_validation(deadline, stop_requested)
        except ValidationInterrupted as exc:
            reason = exc.reason
            files = save_session_checkpoint(session, output_dir)
            _write_status(output_dir, session, files, reason, reason, started)
            return reason
        _append_validation_metrics(output_dir, session, full, curriculum, baseline=True)
        files = save_session_checkpoint(session, output_dir)
        _write_status(output_dir, session, files, "running", None, started)
    while True:
        if stop_requested():
            reason = "stopped"
            break
        if time.monotonic() >= deadline:
            reason = "budget"
            break

        validation_due = (
            session.total_episodes > 0
            and session.total_episodes % session.config.validate_every == 0
            and session.last_validation_episode < session.total_episodes
        )
        if validation_due:
            try:
                validation = session.run_validation(deadline, stop_requested)
            except ValidationInterrupted as exc:
                reason = exc.reason
                break
            _append_validation_metrics(
                output_dir,
                session,
                validation["full"],
                validation["curriculum"],
                validated_stage=validation["validated_stage"],
                validated_distance=validation["validated_distance"],
                passed=validation["passed"],
                promoted=validation["promoted"],
            )
            files = save_session_checkpoint(session, output_dir)
            _write_status(output_dir, session, files, "running", None, started)
        if session.total_episodes >= max_episodes:
            break
        result = session.run_episode(deadline=deadline)
        if result is None:
            reason = "budget"
            break
        _append_metric(output_dir, {"event": "episode", **result})
        if session.total_episodes == 1 or session.total_episodes % checkpoint_every == 0:
            files = save_session_checkpoint(session, output_dir)
            _write_status(output_dir, session, files, "running", None, started)

    files = save_session_checkpoint(session, output_dir)
    _write_status(output_dir, session, files, reason, reason, started)
    return reason


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    session: Optional[TrainingSession] = None
    output_dir = Path(args.output_dir)
    started = time.monotonic()
    stop = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        stop[0] = True

    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, request_stop)
        _nonnegative_int(args.max_episodes, "max_episodes")
        _positive_int(args.checkpoint_every, "checkpoint_every")
        if (
            isinstance(args.max_seconds, bool)
            or not isinstance(args.max_seconds, (int, float))
            or not math.isfinite(float(args.max_seconds))
            or args.max_seconds < 0
        ):
            raise ValueError("max_seconds must be finite and nonnegative")
        if args.resume:
            session = TrainingSession.load(args.resume)
            _validate_resume_arguments(session, args)
            initialize = session.baseline_validation is None
        else:
            managed_outputs = (
                "latest-state.pt",
                "latest-model.pt",
                "best-model.pt",
                "baseline-model.pt",
                "status.json",
                "metrics.jsonl",
            )
            if any((output_dir / name).exists() for name in managed_outputs):
                raise ValueError(
                    "output directory already has training outputs; use --resume"
                )
            config = TrainingConfig(
                max_turns=60 if args.max_turns is None else args.max_turns,
                learning_rate=(
                    1e-3 if args.learning_rate is None else args.learning_rate
                ),
                curriculum_fraction=(
                    0.75
                    if args.curriculum_fraction is None
                    else args.curriculum_fraction
                ),
                validation_games=(
                    64 if args.validation_games is None else args.validation_games
                ),
                validate_every=(
                    1_000 if args.validate_every is None else args.validate_every
                ),
            )
            session = TrainingSession(
                config=config,
                seed=41 if args.seed is None else args.seed,
                initial_model=args.initial_model,
                initial_stage_index=(
                    0 if args.initial_stage is None else args.initial_stage
                ),
            )
            initialize = True
        reason = run(
            session,
            output_dir,
            max_episodes=args.max_episodes,
            max_seconds=args.max_seconds,
            checkpoint_every=args.checkpoint_every,
            stop_requested=lambda: stop[0],
            initialize=initialize,
        )
        print(
            json.dumps(
                {
                    "status": reason,
                    "episodes": session.total_episodes,
                    "updates": session.total_updates,
                    "stage": session.stage_index,
                    "distance": session.current_distance,
                    "best_score": session.best_score,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (TypeError, ValueError, OSError, RuntimeError, KeyError) as exc:
        _publish_error(output_dir, session, str(exc), started)
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 2
    except BaseException as exc:
        _publish_error(output_dir, session, str(exc), started)
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _validate_resume_arguments(
    session: TrainingSession, args: argparse.Namespace
) -> None:
    requested = {
        "max_turns": args.max_turns,
        "learning_rate": args.learning_rate,
        "curriculum_fraction": args.curriculum_fraction,
        "validation_games": args.validation_games,
        "validate_every": args.validate_every,
    }
    for name, value in requested.items():
        if value is not None and value != getattr(session.config, name):
            raise ValueError(
                "resume cannot change %s from %r to %r"
                % (name, getattr(session.config, name), value)
            )
    if args.seed is not None and args.seed != session.initial_seed:
        raise ValueError(
            "resume seed %d does not match saved seed %d"
            % (args.seed, session.initial_seed)
        )
    if args.initial_stage is not None:
        raise ValueError("--initial-stage is only valid for a new session")


def _append_validation_metrics(
    output_dir: Path,
    session: TrainingSession,
    full: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    baseline: bool = False,
    validated_stage: Optional[int] = None,
    validated_distance: Optional[int] = None,
    passed: Optional[bool] = None,
    promoted: Optional[bool] = None,
) -> None:
    if baseline:
        _append_metric(
            output_dir,
            {
                "event": "validation",
                "source": "ordinary_full21_baseline",
                "episode": session.total_episodes,
                "stage": session.stage_index,
                "distance": None,
                **full,
            },
        )
    _append_metric(
        output_dir,
        {
            "event": "validation",
            "source": "ordinary_full21",
            "episode": session.total_episodes,
            "stage": session.stage_index,
            "distance": None,
            "primary_score": full["greedy"]["win_rate"],
            **full,
        },
    )
    stage = session.stage_index if validated_stage is None else validated_stage
    distance = (
        session.current_distance
        if validated_stage is None
        else validated_distance
    )
    _append_metric(
        output_dir,
        {
            "event": "validation",
            "source": (
                "curriculum_random_stage"
                if distance is None
                else "curriculum_distance_%d" % distance
            ),
            "episode": session.total_episodes,
            "stage": stage,
            "distance": distance,
            "passed": passed,
            "promoted": promoted,
            **curriculum,
        },
    )


def _model_payload(
    session: TrainingSession,
    state_dict: Mapping[str, Any],
    role: str,
) -> Dict[str, Any]:
    from src.model import build_checkpoint

    payload = build_checkpoint(
        session.network,
        session.config.game_config,
        meta={
            "algorithm": ALGORITHM,
            "role": role,
            "training_state_version": TRAINING_STATE_VERSION,
            "episodes": session.total_episodes,
            "best_score": session.best_score,
            "best_episode": session.best_episode,
            "baseline_validation": session.baseline_validation,
            "latest_full_validation": session.full_validation,
            "training_config": session.config.to_dict(),
        },
    )
    payload["model_state_dict"] = dict(state_dict)
    return payload


def _status_payload(
    session: TrainingSession,
    files: Mapping[str, Mapping[str, Any]],
    status: str,
    stop_reason: Optional[str],
    started: float,
) -> Dict[str, Any]:
    from src.model import CHECKPOINT_VERSION

    return {
        "status_schema_version": STATUS_SCHEMA_VERSION,
        "status": status,
        "stop_reason": stop_reason,
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "completed_episodes": session.total_episodes,
        "cumulative_episodes": session.total_episodes,
        "updates": session.total_updates,
        "stage": session.stage_index,
        "distance": session.current_distance,
        "stage_episodes": session.stage_episodes,
        "current_epsilon": session.current_epsilon,
        "curriculum_episodes": session.curriculum_episodes,
        "curriculum_wins": session.curriculum_wins,
        "random_episodes": session.random_episodes,
        "random_wins": session.random_wins,
        "full_validation": session.full_validation,
        "curriculum_validation": session.curriculum_validation,
        "baseline_validation": session.baseline_validation,
        "best_score": session.best_score,
        "best_episode": session.best_episode,
        "training_state_version": TRAINING_STATE_VERSION,
        "encoding_version": ENCODING_VERSION,
        "model_checkpoint_version": CHECKPOINT_VERSION,
        "algorithm": ALGORITHM,
        "rules": session.config.to_dict(),
        "files": {name: dict(metadata) for name, metadata in files.items()},
    }


def _write_status(
    output_dir: Path,
    session: TrainingSession,
    files: Mapping[str, Mapping[str, Any]],
    status: str,
    stop_reason: Optional[str],
    started: float,
) -> None:
    _atomic_json(
        _status_payload(session, files, status, stop_reason, started),
        output_dir / "status.json",
    )


def _write_error_status(
    output_dir: Path,
    session: TrainingSession,
    files: Mapping[str, Mapping[str, Any]],
    message: str,
    started: float,
) -> None:
    payload = _status_payload(session, files, "error", "error", started)
    payload["error"] = message
    _atomic_json(payload, output_dir / "status.json")


def _publish_error(
    output_dir: Path,
    session: Optional[TrainingSession],
    message: str,
    started: float,
) -> None:
    if session is not None and session.at_episode_boundary:
        try:
            files = save_session_checkpoint(session, output_dir)
            _write_error_status(output_dir, session, files, message, started)
            return
        except (OSError, RuntimeError, ValueError, KeyError) as export_error:
            print("error: checkpoint export failed: %s" % export_error, file=sys.stderr, flush=True)
    try:
        _atomic_json(
            {
                "status": "error",
                "stop_reason": "error",
                "error": message,
                "completed_episodes": (
                    session.total_episodes if session is not None else 0
                ),
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                "training_state_version": TRAINING_STATE_VERSION,
                "encoding_version": ENCODING_VERSION,
            },
            output_dir / "error.json",
        )
    except OSError as diagnostic_error:
        print("error: diagnostic export failed: %s" % diagnostic_error, file=sys.stderr, flush=True)


def _append_metric(output_dir: Path, event: Mapping[str, Any]) -> None:
    from src.model import CHECKPOINT_VERSION

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": ALGORITHM,
        "training_state_version": TRAINING_STATE_VERSION,
        "encoding_version": ENCODING_VERSION,
        "model_checkpoint_version": CHECKPOINT_VERSION,
        **event,
    }
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _fsync_file(temporary)
    os.replace(str(temporary), str(path))


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(dict(payload), temporary)
    _fsync_file(temporary)
    os.replace(str(temporary), str(path))


def _snapshot_file_manifest(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    files: Dict[str, Dict[str, Any]] = {}
    for name in PUBLISHED_SNAPSHOT_FILES:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError("Missing snapshot file: %s" % path)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        files[name] = {"sha256": digest.hexdigest(), "bytes": size}
    return files


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _clone_model_state(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in state_dict.items()}


def _validate_model_state(
    raw: Any, reference: Mapping[str, Any], name: str
) -> Dict[str, Any]:
    import torch

    if not isinstance(raw, dict) or set(raw) != set(reference):
        raise ValueError("%s fields do not match the network" % name)
    result = {}
    for key, expected in reference.items():
        value = raw[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise ValueError("%s tensor %s is incompatible" % (name, key))
        result[key] = value.detach().cpu().clone()
    return result


def _validate_finite_tree(value: Any, name: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError("%s contains a non-finite tensor" % name)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite number" % name)
        return
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_finite_tree(child, name)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_finite_tree(child, name)


def _optional_validation(value: Any, name: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("%s validation must be a dict or None" % name)
    return value


def _dict_with_fields(
    value: Any, fields: set, name: str
) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("%s fields do not match this trainer" % name)
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an int" % name)
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    number = _integer(value, name)
    if number < 0:
        raise ValueError("%s must be nonnegative" % name)
    return number


def _positive_int(value: Any, name: str) -> int:
    number = _integer(value, name)
    if number <= 0:
        raise ValueError("%s must be positive" % name)
    return number


if __name__ == "__main__":
    sys.exit(main())
