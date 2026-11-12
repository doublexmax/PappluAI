"""Resumable full-hand curriculum training for Papplu."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch

from src.checkpoints.io import (
    require_checkpoint_fields,
    require_checkpoint_int,
)
from src.checkpoints.tensor import (
    clone_model_state,
    validate_model_state,
)
from src.evaluation.arena import EpsilonPolicy
from src.evaluation.benchmark import evaluate_games
from src.game.environment import ENCODING_VERSION, GameConfig, PappluEnv
from src.game.multiplayer import TerminalSnapshot
from src.game.outcome import DealOutcome
from src.model.encoding import encode
from src.model.network import (
    CHECKPOINT_VERSION,
    QNetwork,
    load_checkpoint,
)
from src.training.core import Decision, Trajectory
from src.training import curriculum_io
from src.training.curriculum_io import ALGORITHM, TRAINING_STATE_VERSION
from src.training.optimization import Learner, LearnerConfig, LossWeights
from src.model.search import TeacherConfig
from src.model.spec import ModelSpec


ORDINARY_VALIDATION_SEED = 30_000_000
CURRICULUM_VALIDATION_SEED = 20_000_000


class ValidationInterrupted(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class TrainingConfig:
    num_decks: int = 3
    cards_in_hand: int = 21
    required_sequences: int = 5
    max_turns: int = 60
    recycle_discard: bool = True
    replay_capacity: int = 50_000
    batch_size: int = 128
    updates_per_episode: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 1.0
    n_step: int = 5
    target_tau: float = 0.005
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
    losses: LossWeights = LossWeights()
    teacher: TeacherConfig = TeacherConfig()

    def __post_init__(self) -> None:
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
        if type(self.recycle_discard) is not bool:
            raise TypeError("recycle_discard must be a bool")
        self.learner_config

    @property
    def learner_config(self) -> LearnerConfig:
        return LearnerConfig(**{
            name: getattr(self, name) for name in LearnerConfig.__dataclass_fields__
        })

    @property
    def game_config(self) -> GameConfig:
        return GameConfig(
            num_decks=self.num_decks,
            cards_in_hand=self.cards_in_hand,
            required_sequences=self.required_sequences,
            max_turns=self.max_turns,
            recycle_discard=self.recycle_discard,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        if not isinstance(data, dict):
            raise TypeError("training_config must be a dict")
        fields = set(cls.__dataclass_fields__)
        if set(data) == fields:
            values = dict(data)
        else:
            raise ValueError("training_config fields do not match this trainer")
        distances = values["curriculum_distances"]
        if not isinstance(distances, tuple):
            raise TypeError("curriculum_distances must be a tuple")
        for name, kind in (("losses", LossWeights), ("teacher", TeacherConfig)):
            if not isinstance(values[name], dict) or set(values[name]) != set(kind.__dataclass_fields__):
                raise ValueError("training %s fields do not match" % name)
            values[name] = kind(**values[name])
        return cls(**values)


class TrainingSession:
    def __init__(
        self,
        config: TrainingConfig,
        seed: int,
        initial_model: Optional[str] = None,
        initial_stage_index: int = 0,
        model_spec: ModelSpec = ModelSpec(),
    ) -> None:
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
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if initial_model is None:
                network = QNetwork(model_spec)
            else:
                network, saved_game, _ = load_checkpoint(
                    initial_model, device=torch.device("cpu"),
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
        seed_source = random.Random(seed)
        self.deal_rng = random.Random(seed_source.getrandbits(64))
        self.action_rng = random.Random(seed_source.getrandbits(64))
        self.learner = Learner(network, config.learner_config, seed_source.getrandbits(63))

        self.total_episodes = 0
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
        self.baseline_model_state = clone_model_state(self.network.state_dict())
        self.best_model_state = clone_model_state(self.network.state_dict())
        self.at_episode_boundary = True

    @property
    def network(self):
        return self.learner.online

    @property
    def optimizer(self):
        return self.learner.optimizer

    @property
    def replay(self):
        return self.learner.replay

    @property
    def total_updates(self) -> int:
        return self.learner.update_count

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
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_state_dict(payload, expected_config=expected_config)

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, Any],
        expected_config: Optional[TrainingConfig] = None,
    ) -> "TrainingSession":
        previous_rng_state = torch.get_rng_state()
        try:
            return cls._from_state_dict(
                payload,
                expected_config=expected_config,
            )
        finally:
            torch.set_rng_state(previous_rng_state)

    @classmethod
    def _from_state_dict(
        cls,
        payload: Mapping[str, Any],
        expected_config: Optional[TrainingConfig] = None,
    ) -> "TrainingSession":
        if not isinstance(payload, dict):
            raise ValueError("training checkpoint must be a dict")
        required = {
            "training_state_version",
            "encoding_version",
            "model_checkpoint_version",
            "algorithm",
            "training_config",
            "model_spec",
            "initial_seed",
            "learner",
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
        source_trace = require_checkpoint_fields(
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
            "module": "src.training.curriculum",
            "training_state_version": TRAINING_STATE_VERSION,
            "encoding_version": ENCODING_VERSION,
            "model_checkpoint_version": CHECKPOINT_VERSION,
            "rules": config.game_config.to_dict(),
        }
        canonical_trace = dict(source_trace)
        canonical_trace["rules"] = GameConfig.from_dict(
            source_trace["rules"]
        ).to_dict()
        if canonical_trace != expected_trace:
            raise ValueError("training checkpoint source_trace is incompatible")
        seed = require_checkpoint_int(payload["initial_seed"], "initial_seed")
        session = cls(config=config, seed=seed, model_spec=ModelSpec.from_dict(payload["model_spec"]))
        session.learner.load_state_dict(payload["learner"])

        counters = require_checkpoint_fields(
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
            checked = require_checkpoint_int(value, name, minimum=0)
            if name == "total_updates":
                if checked != session.total_updates:
                    raise ValueError("curriculum and learner update counts differ")
            else:
                setattr(session, name, checked)
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

        curriculum = require_checkpoint_fields(
            payload["curriculum"],
            {
                "stage_index",
                "stage_episodes",
                "epsilon_progress",
                "consecutive_passing_validations",
            },
            "curriculum",
        )
        session.stage_index = require_checkpoint_int(
            curriculum["stage_index"],
            "stage_index",
            minimum=0,
            maximum=len(config.curriculum_distances),
        )
        session.stage_episodes = require_checkpoint_int(
            curriculum["stage_episodes"],
            "stage_episodes",
            minimum=0,
            maximum=session.total_episodes,
        )
        session.epsilon_progress = require_checkpoint_int(
            curriculum["epsilon_progress"],
            "epsilon_progress",
            minimum=0,
        )
        if session.epsilon_progress != session.stage_episodes:
            raise ValueError("epsilon progress must equal stage episode count")
        session.consecutive_passing_validations = require_checkpoint_int(
            curriculum["consecutive_passing_validations"],
            "consecutive_passing_validations",
            minimum=0,
        )
        if (
            session.consecutive_passing_validations
            >= config.promotion_passes
        ):
            raise ValueError("saved validation pass streak should have promoted")

        validation = require_checkpoint_fields(
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
        session.last_validation_episode = require_checkpoint_int(
            validation["last_validation_episode"],
            "last_validation_episode",
            minimum=-1,
            maximum=session.total_episodes,
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
        session.best_episode = require_checkpoint_int(
            validation["best_episode"],
            "best_episode",
            minimum=0,
            maximum=session.total_episodes,
        )
        session.baseline_model_state = validate_model_state(
            payload["baseline_model_state_dict"], session.network.state_dict(),
            "baseline_model_state_dict",
        )
        session.best_model_state = validate_model_state(
            payload["best_model_state_dict"], session.network.state_dict(),
            "best_model_state_dict",
        )

        rng_states = require_checkpoint_fields(
            payload["rng_states"], {"deal", "action"}, "rng_states"
        )
        for name, rng in (
            ("deal", session.deal_rng),
            ("action", session.action_rng),
        ):
            candidate = random.Random()
            try:
                candidate.setstate(rng_states[name])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError("%s RNG state is invalid" % name) from exc
            rng.setstate(rng_states[name])
        session.network.train()
        session.at_episode_boundary = True
        return session

    def state_dict(self) -> Dict[str, Any]:
        if not self.at_episode_boundary:
            raise RuntimeError("cannot checkpoint a partial episode")
        return {
            "training_state_version": TRAINING_STATE_VERSION,
            "encoding_version": ENCODING_VERSION,
            "model_checkpoint_version": CHECKPOINT_VERSION,
            "algorithm": ALGORITHM,
            "training_config": self.config.to_dict(),
            "model_spec": self.network.spec.to_dict(),
            "initial_seed": self.initial_seed,
            "learner": self.learner.state_dict(),
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
            },
            "baseline_model_state_dict": self.baseline_model_state,
            "best_model_state_dict": self.best_model_state,
            "source_trace": {
                "module": "src.training.curriculum",
                "training_state_version": TRAINING_STATE_VERSION,
                "encoding_version": ENCODING_VERSION,
                "model_checkpoint_version": CHECKPOINT_VERSION,
                "rules": self.config.game_config.to_dict(),
            },
        }

    def run_episode(self, deadline: Optional[float] = None) -> Optional[Dict[str, Any]]:
        epsilon = self.current_epsilon
        distance = self.current_distance
        rng_snapshot = {
            "deal": self.deal_rng.getstate(),
            "action": self.action_rng.getstate(),
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
            policy = EpsilonPolicy(self.network, epsilon)
            while not obs.done:
                if deadline is not None and time.monotonic() >= deadline:
                    self._restore_rng_snapshot(rng_snapshot)
                    self.at_episode_boundary = True
                    return None
                features = encode(obs)
                action = policy.act_features([features], [self.action_rng])[0]
                steps.append(Decision(
                    obs, features, action, (tuple(obs.hand),) + ((0,) * 52,) * 5,
                ))
                obs = env.step(action)

            if deadline is not None and time.monotonic() >= deadline:
                self._restore_rng_snapshot(rng_snapshot)
                self.at_episode_boundary = True
                return None

            outcome = DealOutcome.score(
                TerminalSnapshot((tuple(obs.hand),), obs.joker),
                0 if obs.won else None, "win" if obs.won else "turns_exhausted",
                self.config.required_sequences,
            ).seat(0)
            self.learner.append([Trajectory(1, tuple(steps), outcome)])
            replay_committed = True
            self.learner.update(self.config.updates_per_episode)

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
        self.best_model_state = clone_model_state(self.network.state_dict())
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
            self.best_model_state = clone_model_state(self.network.state_dict())

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
        self.deal_rng.setstate(snapshot["deal"])
        self.action_rng.setstate(snapshot["action"])


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
    _positive_int(games, "games")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if distance is not None:
        _positive_int(distance, "distance")

    try:
        results: Dict[str, Any] = {}
        for policy in ("greedy", "random"):
            evaluated = evaluate_games(
                network, config, games, seed, policy=policy, curriculum_distance=distance,
                deadline=deadline, stop_requested=stop_requested,
            )
            wins = sum(evaluated.outcomes)
            results[policy] = {
                "games": games,
                "wins": wins,
                "win_rate": wins / games,
            }
        return results
    except InterruptedError as error:
        raise ValidationInterrupted("stopped") from error
    except TimeoutError as error:
        raise ValidationInterrupted("budget") from error


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
    if (output_dir / "metrics.jsonl").exists():
        shutil.copyfile(output_dir / "metrics.jsonl", output_dir / ".active-metrics.jsonl")
    files = curriculum_io.save_session_checkpoint(session, output_dir)
    curriculum_io.write_status(output_dir, session, files, "running", None, started)
    if not initialize:
        curriculum_io.append_metric(
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
            files = curriculum_io.save_session_checkpoint(session, output_dir)
            curriculum_io.write_status(output_dir, session, files, reason, reason, started)
            return reason
        curriculum_io.append_validation_metrics(output_dir, session, full, curriculum, baseline=True)
        files = curriculum_io.save_session_checkpoint(session, output_dir)
        curriculum_io.write_status(output_dir, session, files, "running", None, started)
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
            curriculum_io.append_validation_metrics(
                output_dir,
                session,
                validation["full"],
                validation["curriculum"],
                validated_stage=validation["validated_stage"],
                validated_distance=validation["validated_distance"],
                passed=validation["passed"],
                promoted=validation["promoted"],
            )
            files = curriculum_io.save_session_checkpoint(session, output_dir)
            curriculum_io.write_status(output_dir, session, files, "running", None, started)
        if session.total_episodes >= max_episodes:
            break
        result = session.run_episode(deadline=deadline)
        if result is None:
            reason = "budget"
            break
        curriculum_io.append_metric(output_dir, {"event": "episode", **result})
        if session.total_episodes == 1 or session.total_episodes % checkpoint_every == 0:
            files = curriculum_io.save_session_checkpoint(session, output_dir)
            curriculum_io.write_status(output_dir, session, files, "running", None, started)

    files = curriculum_io.save_session_checkpoint(session, output_dir)
    curriculum_io.write_status(output_dir, session, files, reason, reason, started)
    return reason


def _optional_validation(value: Any, name: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("%s validation must be a dict or None" % name)
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
