from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
import json
import math
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union
import zlib

import torch
from torch import Tensor
import torch.nn.functional as F

from src.checkpoints import evidence
from src.checkpoints.io import atomic_copy, atomic_write_json, file_sha256
from src.checkpoints.registry import ModelRegistry
from src.checkpoints.tensor import (
    atomic_torch_save,
    clone_model_state,
    validate_checkpoint_tree,
    validate_model_state,
)
from src.evaluation.arena import (
    CheckpointPolicy,
    Deal,
    EpsilonPolicy,
    MatchInterrupted,
    Policy,
    RandomPolicy,
    run_matches,
)
from src.evaluation.benchmark import evaluate_games
from src.evaluation.promotion import GateInterrupted, evaluate_gate
from src.game.environment import GameConfig, PappluEnv
from src.game.multiplayer import MatchConfig
from src.game.outcome import SeatOutcome, solo_outcome
from src.model.encoding import FEATURE_SIZE, Features, NUM_FACES, encode, legal_actions
from src.model.network import QNetwork, build_checkpoint, collate, greedy_actions, load_checkpoint
from src.model.spec import ModelSpec


STATE_VERSION = 5
STATUS_SCHEMA_VERSION = 1
ALGORITHM = "unified_nstep_ddqn"
RANDOM_OPPONENT = "random"
ORDINARY_VALIDATION_SEED = 30_000_000
CURRICULUM_VALIDATION_SEED = 20_000_000
SELECTION_SEED_BASE = 60_000_000
SOLO_EXPLORATION_SALT = 0x9E3779B97F4A7C15
FEATURE_BYTES = FEATURE_SIZE * 4
EMPTY_OPPONENT_HANDS = ((0,) * NUM_FACES,) * 5
MAX_OPPONENT_POOL = 8
TUNABLE_ON_RESUME = frozenset({
    "checkpoint_every",
    "gate_workers",
    "torch_threads",
    "device",
})
StopReason = Literal["stopped", "budget", "max_episodes", "max_cycles"]


def _require_int(value: Any, name: str, minimum: Optional[int] = None) -> int:
    if type(value) is not int:
        raise ValueError("%s must be an integer" % name)
    if minimum is not None and value < minimum:
        raise ValueError("%s must be >= %d" % (name, minimum))
    return value


def _require_float(
    value: Any,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    *,
    strict_min: bool = False,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("%s must be a finite number" % name)
    number = float(value)
    if minimum is not None and (number <= minimum if strict_min else number < minimum):
        comparator = ">" if strict_min else ">="
        raise ValueError("%s must be %s %s" % (name, comparator, minimum))
    if maximum is not None and number > maximum:
        raise ValueError("%s must be <= %s" % (name, maximum))
    return number


@dataclass(frozen=True)
class Step:
    """One recorded learner decision without Observation retention."""

    features: Features
    action: int
    opponent_hands: tuple[tuple[int, ...], ...] = EMPTY_OPPONENT_HANDS

    def __post_init__(self) -> None:
        if not isinstance(self.features, Features):
            raise ValueError("step features must be a Features record")
        if type(self.action) is not int or self.action not in legal_actions(self.features):
            raise ValueError("step action must be a legal action index")
        hands = self.opponent_hands
        if not isinstance(hands, tuple) or len(hands) != 5 or any(
            not isinstance(hand, tuple) or len(hand) != NUM_FACES
            or any(type(count) is not int or count < 0 for count in hand)
            for hand in hands
        ):
            raise ValueError("opponent_hands must hold five 52-count tuples")


@dataclass(frozen=True)
class Trajectory:
    """Learner decision sequence and terminal outcome for one completed deal."""

    players: int
    steps: tuple[Step, ...]
    outcome: SeatOutcome

    def __post_init__(self) -> None:
        if type(self.players) is not int or not 1 <= self.players <= 6:
            raise ValueError("trajectory players must lie in 1..6")
        if not isinstance(self.steps, tuple) or not self.steps or any(
            not isinstance(step, Step) for step in self.steps
        ):
            raise ValueError("trajectory steps must be a nonempty tuple of Step")
        outcome = self.outcome
        if (
            not isinstance(outcome, SeatOutcome)
            or type(outcome.utility) not in (int, float)
            or not math.isfinite(outcome.utility)
            or not -1.0 <= outcome.utility <= 1.0
            or type(outcome.won) is not bool
            or type(outcome.penalty) is not int
            or outcome.penalty < 0
        ):
            raise ValueError("trajectory outcome is invalid")


@dataclass(frozen=True)
class Sample:
    step: Step
    outcome: SeatOutcome
    bootstrap: Optional[Features]
    scale: float


class Replay:
    """Player-count-stratified trajectory replay with strict step capacity."""

    STATE_VERSION = 5

    def __init__(self, capacity: int) -> None:
        _require_int(capacity, "capacity", minimum=1)
        self.capacity = capacity
        self._buckets: dict[int, deque[Trajectory]] = {}
        self._counts: dict[int, int] = {}
        self._size = 0
        self._sample_cursor = 0

    def __len__(self) -> int:
        return self._size

    def counts_by_players(self) -> dict[int, int]:
        return {players: count for players, count in sorted(self._counts.items()) if count > 0}

    def append(self, trajectory: Trajectory) -> int:
        if not isinstance(trajectory, Trajectory):
            raise ValueError("replay requires a Trajectory")
        players = trajectory.players
        bucket = self._buckets.setdefault(players, deque())
        bucket.append(trajectory)
        added = len(trajectory.steps)
        self._counts[players] = self._counts.get(players, 0) + added
        self._size += added
        while self._size > self.capacity:
            victim = max(
                (p for p, count in self._counts.items() if count > 0),
                key=lambda p: (self._counts[p], -p),
            )
            queue = self._buckets[victim]
            head = queue[0]
            excess = self._size - self.capacity
            if len(head.steps) <= excess:
                queue.popleft()
                removed = len(head.steps)
                self._counts[victim] -= removed
                self._size -= removed
                if not queue:
                    del self._buckets[victim]
                    del self._counts[victim]
            else:
                queue[0] = Trajectory(head.players, head.steps[excess:], head.outcome)
                self._counts[victim] -= excess
                self._size -= excess
        return added

    def sample(
        self,
        rng: random.Random,
        batch_size: int,
        *,
        n_step: int = 5,
        gamma: float = 0.99,
    ) -> list[Sample]:
        _require_int(batch_size, "batch_size", minimum=1)
        _require_int(n_step, "n_step", minimum=1)
        _require_float(gamma, "gamma", minimum=0.0, maximum=1.0, strict_min=True)
        active = sorted(p for p, count in self._counts.items() if count > 0)
        if not active:
            raise ValueError("cannot sample from an empty replay")
        samples: list[Sample] = []
        for offset in range(batch_size):
            players = active[(self._sample_cursor + offset) % len(active)]
            target = rng.randrange(self._counts[players])
            for trajectory in self._buckets[players]:
                length = len(trajectory.steps)
                if target < length:
                    step = trajectory.steps[target]
                    next_index = target + n_step
                    if next_index < length:
                        bootstrap = trajectory.steps[next_index].features
                        scale = gamma ** n_step
                    else:
                        bootstrap = None
                        scale = gamma ** (length - 1 - target)
                    samples.append(Sample(step, trajectory.outcome, bootstrap, scale))
                    break
                target -= length
        self._sample_cursor = (self._sample_cursor + batch_size) % len(active)
        return samples

    def state_dict(self) -> dict[str, Any]:
        blobs: list[bytes] = []
        records: list[tuple[Any, ...]] = []
        for players in sorted(self._buckets):
            for trajectory in self._buckets[players]:
                for step in trajectory.steps:
                    blobs.append(step.features.data)
                records.append((
                    trajectory.players,
                    tuple(step.action for step in trajectory.steps),
                    tuple(step.opponent_hands for step in trajectory.steps),
                    (
                        float(trajectory.outcome.utility),
                        trajectory.outcome.won,
                        trajectory.outcome.penalty,
                    ),
                ))
        return {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "sample_cursor": self._sample_cursor,
            "features_zlib": zlib.compress(b"".join(blobs), level=1),
            "trajectories": tuple(records),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported replay state version")
        cursor = _require_int(state.get("sample_cursor"), "sample_cursor", minimum=0)
        compressed = state.get("features_zlib")
        records = state.get("trajectories")
        if not isinstance(compressed, bytes) or not isinstance(records, (tuple, list)):
            raise ValueError("invalid replay payload")
        try:
            raw = zlib.decompress(compressed)
        except zlib.error as exc:
            raise ValueError("corrupted replay feature stream") from exc
        total_steps = sum(
            len(entry[1])
            for entry in records
            if isinstance(entry, (tuple, list)) and len(entry) == 4 and isinstance(entry[1], (tuple, list))
        )
        if len(raw) != total_steps * FEATURE_BYTES:
            raise ValueError("replay feature byte length mismatch")
        if total_steps > self.capacity:
            raise ValueError("saved replay exceeds configured capacity")
        buckets: dict[int, deque[Trajectory]] = {}
        counts: dict[int, int] = {}
        offset = 0
        for entry in records:
            if not isinstance(entry, (tuple, list)) or len(entry) != 4:
                raise ValueError("invalid replay trajectory record")
            players, actions, hands_seq, outcome_raw = entry
            if (
                not isinstance(actions, (tuple, list))
                or not isinstance(hands_seq, (tuple, list))
                or len(actions) != len(hands_seq)
                or not isinstance(outcome_raw, (tuple, list))
                or len(outcome_raw) != 3
            ):
                raise ValueError("invalid replay step sequence")
            steps = []
            for action, opp_hands in zip(actions, hands_seq):
                chunk = raw[offset:offset + FEATURE_BYTES]
                offset += FEATURE_BYTES
                normalized_hands = tuple(tuple(hand) for hand in opp_hands)
                steps.append(Step(Features(chunk), action, normalized_hands))
            outcome = SeatOutcome(outcome_raw[0], outcome_raw[1], outcome_raw[2])
            trajectory = Trajectory(players, tuple(steps), outcome)
            buckets.setdefault(players, deque()).append(trajectory)
            counts[players] = counts.get(players, 0) + len(steps)
        self._buckets = buckets
        self._counts = counts
        self._size = total_steps
        self._sample_cursor = cursor % max(1, len(counts))


@dataclass(frozen=True)
class TrainConfig:
    num_decks: int = 3
    cards_in_hand: int = 21
    required_sequences: int = 5
    max_turns: int = 60
    recycle_discard: bool = True
    player_counts: tuple[int, ...] = (1,)
    round_episodes: int = 16
    updates_per_episode: int = 4
    batch_size: int = 128
    replay_capacity: int = 50_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    n_step: int = 5
    target_tau: float = 0.005
    grad_clip: float = 1.0
    win_weight: float = 0.25
    penalty_weight: float = 0.25
    belief_weight: float = 0.10
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 10_000
    warm_start_fraction: float = 0.0
    curriculum_distances: tuple[int, ...] = ()
    curriculum_fraction: float = 0.75
    initial_stage: int = 0
    stage_passes: int = 2
    stage_margin: float = 0.10
    stage_floor: float = 0.25
    validate_every: int = 1_000
    validation_games: int = 64
    pfsp_champion_share: float = 0.35
    pfsp_anchor_share: float = 0.15
    pfsp_exponent: float = 2.0
    pfsp_prior_games: float = 4.0
    promote: bool = False
    cycle_episodes: int = 128
    selection_blocks: int = evidence.MIN_SELECTION_SEED_BLOCKS
    confirmation_blocks: int = evidence.MIN_CONFIRMATION_SEED_BLOCKS
    solo_games: int = evidence.MIN_SOLO_GAMES
    gate_samples: int = 2_000
    gate_workers: int = 1
    rebase_after_rejections: int = 8
    checkpoint_every: int = 64
    seed: int = 41
    torch_threads: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        game = GameConfig(
            num_decks=self.num_decks,
            cards_in_hand=self.cards_in_hand,
            required_sequences=self.required_sequences,
            max_turns=self.max_turns,
            recycle_discard=self.recycle_discard,
        )
        if not isinstance(self.player_counts, (tuple, list)) or not self.player_counts:
            raise ValueError("player_counts must be nonempty")
        counts = tuple(_require_int(c, "player_counts", minimum=1) for c in self.player_counts)
        if any(c > 6 for c in counts) or len(set(counts)) != len(counts):
            raise ValueError("player_counts must contain distinct values in 1..6")
        object.__setattr__(self, "player_counts", tuple(sorted(counts)))
        for count in self.player_counts:
            if count > 1:
                MatchConfig(game=game, players=count)
        for int_name, low in (
            ("round_episodes", 1), ("updates_per_episode", 0), ("batch_size", 1),
            ("replay_capacity", 1), ("n_step", 1), ("epsilon_decay_episodes", 0),
            ("initial_stage", 0), ("stage_passes", 1), ("validate_every", 1),
            ("validation_games", 0), ("cycle_episodes", 1), ("selection_blocks", 1),
            ("confirmation_blocks", 1), ("solo_games", 1), ("gate_samples", 100),
            ("gate_workers", 1), ("rebase_after_rejections", 0), ("checkpoint_every", 1),
            ("torch_threads", 1),
        ):
            _require_int(getattr(self, int_name), int_name, minimum=low)
        _require_int(self.seed, "seed")
        for float_name, low, high, strict in (
            ("learning_rate", 0.0, None, True), ("weight_decay", 0.0, None, False),
            ("gamma", 0.0, 1.0, True), ("target_tau", 0.0, 1.0, True),
            ("grad_clip", 0.0, None, True), ("win_weight", 0.0, None, False),
            ("penalty_weight", 0.0, None, False), ("belief_weight", 0.0, None, False),
            ("epsilon_start", 0.0, 1.0, False), ("epsilon_end", 0.0, 1.0, False),
            ("warm_start_fraction", 0.0, 1.0, False), ("curriculum_fraction", 0.0, 1.0, False),
            ("stage_margin", 0.0, 1.0, False), ("stage_floor", 0.0, 1.0, False),
            ("pfsp_champion_share", 0.0, 1.0, False), ("pfsp_anchor_share", 0.0, 1.0, False),
            ("pfsp_exponent", 0.0, None, True), ("pfsp_prior_games", 0.0, None, True),
        ):
            object.__setattr__(
                self,
                float_name,
                _require_float(getattr(self, float_name), float_name, minimum=low, maximum=high, strict_min=strict),
            )
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must be <= epsilon_start")
        if self.warm_start_fraction > 0.0:
            if 1 not in self.player_counts:
                raise ValueError("warm_start_fraction > 0 requires 1 in player_counts")
            if self.cards_in_hand < 3:
                raise ValueError("warm_start_fraction > 0 requires cards_in_hand >= 3")
        if not isinstance(self.curriculum_distances, (tuple, list)):
            raise ValueError("curriculum_distances must be a sequence")
        distances = tuple(
            _require_int(d, "curriculum_distances", minimum=1)
            for d in self.curriculum_distances
        )
        if any(d > self.cards_in_hand for d in distances):
            raise ValueError("curriculum_distances cannot exceed cards_in_hand")
        object.__setattr__(self, "curriculum_distances", distances)
        if self.initial_stage > len(distances):
            raise ValueError("initial_stage exceeds curriculum_distances length")
        if distances:
            if 1 not in self.player_counts:
                raise ValueError("curriculum_distances requires 1 in player_counts")
            if self.validation_games == 0:
                raise ValueError("curriculum_distances requires validation_games >= 1")
        if self.pfsp_champion_share + self.pfsp_anchor_share > 1.0 + 1e-9:
            raise ValueError("pfsp_champion_share + pfsp_anchor_share must be <= 1.0")
        if type(self.promote) is not bool:
            raise ValueError("promote must be a bool")
        if self.promote:
            if (self.num_decks, self.cards_in_hand, self.required_sequences) != (3, 21, 5):
                raise ValueError("promote requires 3 decks, 21 cards, and 5 pure sequences")
            if not self.gate_counts:
                raise ValueError("promote requires at least one player count in 2, 3, or 4")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a nonempty string")
        torch.device(self.device)

    @property
    def game(self) -> GameConfig:
        return GameConfig(
            num_decks=self.num_decks,
            cards_in_hand=self.cards_in_hand,
            required_sequences=self.required_sequences,
            max_turns=self.max_turns,
            recycle_discard=self.recycle_discard,
        )

    @property
    def gate_counts(self) -> tuple[int, ...]:
        return tuple(count for count in self.player_counts if count in (2, 3, 4))

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TrainConfig:
        if not isinstance(raw, Mapping):
            raise ValueError("config must be a mapping")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError("unknown TrainConfig fields " + ", ".join(unknown))
        converted = dict(raw)
        for seq_name in ("player_counts", "curriculum_distances"):
            if seq_name in converted and isinstance(converted[seq_name], list):
                converted[seq_name] = tuple(converted[seq_name])
        return cls(**converted)

    @classmethod
    def resolve(
        cls,
        saved: Optional[TrainConfig],
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> TrainConfig:
        raw = {} if overrides is None else dict(overrides)
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError("unknown TrainConfig fields " + ", ".join(unknown))
        normalized: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                continue
            if key == "player_counts" and isinstance(value, (list, tuple)):
                normalized[key] = tuple(sorted(set(value)) if len(set(value)) == len(value) else value)
            elif key == "curriculum_distances" and isinstance(value, (list, tuple)):
                normalized[key] = tuple(value)
            else:
                normalized[key] = value
        if saved is None:
            return cls(**normalized)
        candidate = replace(saved, **normalized)
        changed = sorted(
            key
            for key in normalized
            if key not in TUNABLE_ON_RESUME and getattr(candidate, key) != getattr(saved, key)
        )
        if changed:
            raise ValueError(
                "cannot change training config on resume for " + ", ".join(changed),
            )
        return replace(
            saved,
            **{key: normalized[key] for key in normalized if key in TUNABLE_ON_RESUME},
        )


@dataclass(frozen=True)
class LearnStats:
    """Mean loss components across a batch of gradient updates."""

    updates: int = 0
    samples: int = 0
    loss: float = 0.0
    q_loss: float = 0.0
    win_loss: float = 0.0
    penalty_loss: float = 0.0
    belief_loss: float = 0.0


class Learner:
    def __init__(
        self,
        online: QNetwork,
        config: TrainConfig = TrainConfig(),
        *,
        seed: int = 0,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        _require_int(seed, "seed")
        if not isinstance(config, TrainConfig):
            raise ValueError("config must be a TrainConfig")
        resolved_device = torch.device(config.device if device is None else device)
        self.config = config
        self.device = resolved_device
        self.online = online.to(self.device)
        self.target = deepcopy(self.online).to(self.device).eval().requires_grad_(False)
        self.optimizer = self._make_optimizer()
        self.replay = Replay(config.replay_capacity)
        self.rng = random.Random(seed)
        self.step_count = 0
        self.update_count = 0

    def _make_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.online.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def append(self, trajectories: Sequence[Trajectory]) -> int:
        added = 0
        for trajectory in trajectories:
            added += self.replay.append(trajectory)
        self.step_count += added
        return added

    def learn(self, trajectories: Sequence[Trajectory], updates: int = 1) -> LearnStats:
        self.append(trajectories)
        return self.update(updates)

    def update(self, updates: int = 1) -> LearnStats:
        _require_int(updates, "updates", minimum=0)
        if updates == 0 or len(self.replay) == 0:
            return LearnStats()
        self.online.train()
        self.target.eval()
        totals = [0.0, 0.0, 0.0, 0.0, 0.0]
        total_samples = 0
        cfg = self.config
        for _ in range(updates):
            batch = self.replay.sample(
                self.rng,
                min(cfg.batch_size, len(self.replay)),
                n_step=cfg.n_step,
                gamma=cfg.gamma,
            )
            now = collate([item.step.features for item in batch], self.device)
            actions = torch.tensor(
                [item.step.action for item in batch],
                device=self.device,
                dtype=torch.long,
            )
            targets = torch.tensor(
                [item.outcome.utility for item in batch],
                device=self.device,
                dtype=torch.float32,
            )
            boot_rows = [idx for idx, item in enumerate(batch) if item.bootstrap is not None]
            with torch.no_grad():
                if boot_rows:
                    next_batch = collate(
                        [batch[idx].bootstrap for idx in boot_rows if batch[idx].bootstrap is not None],
                        self.device,
                    )
                    best_next = greedy_actions(self.online, next_batch)
                    targets[boot_rows] = self.target(next_batch).gather(1, best_next[:, None]).squeeze(1)
                scales = torch.tensor(
                    [item.scale for item in batch],
                    device=self.device,
                    dtype=torch.float32,
                )
                targets = targets * scales
            won = torch.tensor(
                [1.0 if item.outcome.won else 0.0 for item in batch],
                device=self.device,
                dtype=torch.float32,
            )
            penalties = torch.tensor(
                [item.outcome.penalty / 100.0 for item in batch],
                device=self.device,
                dtype=torch.float32,
            )
            outputs = self.online.outputs(now)
            q_pred = outputs.q.gather(1, actions[:, None]).squeeze(1)
            win_pred = outputs.win_logit.gather(1, actions[:, None]).squeeze(1)
            pen_pred = outputs.penalty.gather(1, actions[:, None]).squeeze(1)
            q_loss = F.smooth_l1_loss(q_pred, targets)
            win_loss = F.binary_cross_entropy_with_logits(win_pred, won)
            penalty_loss = F.smooth_l1_loss(pen_pred, penalties)
            present = now.seat[:, 1:, 0] > 0.5
            belief_loss = outputs.q.new_zeros(())
            if bool(present.any()):
                opp_hands = torch.tensor(
                    [item.step.opponent_hands for item in batch],
                    device=self.device,
                    dtype=torch.float32,
                )
                decks = (now.context[:, 8] * 6.0).round().clamp_min(1.0)
                belief_targets = (opp_hands / decks[:, None, None]).clamp(0.0, 1.0)
                belief_loss = F.binary_cross_entropy_with_logits(
                    outputs.belief[:, 1:][present],
                    belief_targets[present],
                )
            loss = (
                q_loss
                + cfg.win_weight * win_loss
                + cfg.penalty_weight * penalty_loss
                + cfg.belief_weight * belief_loss
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
            self.optimizer.step()
            with torch.no_grad():
                for target_param, online_param in zip(
                    self.target.parameters(),
                    self.online.parameters(),
                ):
                    target_param.lerp_(online_param, cfg.target_tau)
            totals[0] += float(loss.detach().item())
            totals[1] += float(q_loss.detach().item())
            totals[2] += float(win_loss.detach().item())
            totals[3] += float(penalty_loss.detach().item())
            totals[4] += float(belief_loss.detach().item())
            total_samples += len(batch)
            self.update_count += 1
        return LearnStats(
            updates=updates,
            samples=total_samples,
            loss=totals[0] / updates,
            q_loss=totals[1] / updates,
            win_loss=totals[2] / updates,
            penalty_loss=totals[3] / updates,
            belief_loss=totals[4] / updates,
        )

    def adopt(self, model_state: Mapping[str, Tensor]) -> None:
        validated = validate_model_state(model_state, self.online.state_dict(), "adopted_model")
        self.online.load_state_dict(validated, strict=True)
        self.online.to(self.device)
        self.target.load_state_dict(validated, strict=True)
        self.target.to(self.device).eval().requires_grad_(False)
        self.optimizer = self._make_optimizer()

    def state_dict(self) -> dict[str, Any]:
        opt_state = deepcopy(self.optimizer.state_dict())
        for param_state in opt_state.get("state", {}).values():
            for key, val in list(param_state.items()):
                if isinstance(val, Tensor):
                    param_state[key] = val.detach().cpu().clone()
        return {
            "spec": self.online.spec.to_dict(),
            "online": clone_model_state(self.online.state_dict()),
            "target": clone_model_state(self.target.state_dict()),
            "optimizer": opt_state,
            "replay": self.replay.state_dict(),
            "rng": self.rng.getstate(),
            "step_count": self.step_count,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("learner state must be a mapping")
        validate_checkpoint_tree(state, "learner_state")
        if ModelSpec.from_dict(state.get("spec")) != self.online.spec:
            raise ValueError("learner spec mismatch")
        online_state = validate_model_state(state.get("online"), self.online.state_dict(), "online")
        target_state = validate_model_state(state.get("target"), self.target.state_dict(), "target")
        step_count = _require_int(state.get("step_count"), "step_count", minimum=0)
        update_count = _require_int(state.get("update_count"), "update_count", minimum=0)
        replay = Replay(self.config.replay_capacity)
        replay.load_state_dict(state.get("replay"))
        rng = random.Random(0)
        rng.setstate(state["rng"])
        self.online.load_state_dict(online_state, strict=True)
        self.online.to(self.device)
        self.target.load_state_dict(target_state, strict=True)
        self.target.to(self.device).eval().requires_grad_(False)
        optimizer = self._make_optimizer()
        optimizer.load_state_dict(state["optimizer"])
        self.optimizer = optimizer
        self.replay = replay
        self.rng = rng
        self.step_count = step_count
        self.update_count = update_count


def epsilon_at(config: TrainConfig, episodes: int) -> float:
    if config.epsilon_decay_episodes <= 0:
        return config.epsilon_end
    progress = min(1.0, max(0, episodes) / config.epsilon_decay_episodes)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def curriculum_step(
    config: TrainConfig,
    stage: int,
    passes: int,
    greedy_rate: float,
    random_rate: float,
) -> tuple[int, int]:
    if stage >= len(config.curriculum_distances):
        return stage, 0
    threshold = max(random_rate + config.stage_margin, config.stage_floor)
    if greedy_rate + 1e-12 >= threshold:
        if passes + 1 >= config.stage_passes:
            return stage + 1, 0
        return stage, passes + 1
    return stage, 0


def disjoint_seed_gap(config: TrainConfig) -> int:
    span = max(
        4 * config.selection_blocks,
        4 * config.confirmation_blocks,
        config.solo_games,
    )
    return 100_000 + span + 1_000


def pfsp_probabilities(
    config: TrainConfig,
    scores: Mapping[tuple[int, str], tuple[float, int]],
    players: int,
    pool_ids: Sequence[str],
    champion_id: Optional[str],
) -> tuple[float, ...]:
    unique = list(dict.fromkeys(pool_ids))
    if not unique:
        raise ValueError("opponent pool must not be empty")
    if len(unique) == 1:
        return (1.0,)
    weights: dict[str, float] = {}
    remaining = 1.0
    reserved: set[str] = set()
    if champion_id is not None and champion_id in unique and len(unique) > 1 and config.pfsp_champion_share > 0.0:
        weights[champion_id] = config.pfsp_champion_share
        remaining -= config.pfsp_champion_share
        reserved.add(champion_id)
    if RANDOM_OPPONENT in unique and RANDOM_OPPONENT not in reserved and len(unique) > len(reserved) and config.pfsp_anchor_share > 0.0:
        if len(reserved) + 1 == len(unique):
            weights[RANDOM_OPPONENT] = max(0.0, remaining)
            remaining = 0.0
        else:
            weights[RANDOM_OPPONENT] = config.pfsp_anchor_share
            remaining = max(0.0, remaining - config.pfsp_anchor_share)
        reserved.add(RANDOM_OPPONENT)
    others = [opp for opp in unique if opp not in reserved]
    if others:
        raw: list[float] = []
        prior = config.pfsp_prior_games
        for opp in others:
            total, games = scores.get((players, opp), (0.0, 0))
            win_rate = (total + 0.5 * prior) / (games + prior)
            raw.append(max(1e-6, (1.0 - win_rate) ** config.pfsp_exponent))
        raw_sum = sum(raw)
        for opp, value in zip(others, raw):
            weights[opp] = remaining * (value / raw_sum)
    elif remaining > 0.0:
        share = remaining / len(unique)
        for opp in unique:
            weights[opp] = weights.get(opp, 0.0) + share
    probs = [weights.get(opp, 0.0) for opp in pool_ids]
    norm = sum(probs)
    return tuple(p / norm for p in probs)


@dataclass(frozen=True)
class Selecting:
    candidate_id: str
    champion_id: str
    seed: int


@dataclass(frozen=True)
class Confirming:
    candidate_id: str
    champion_id: str
    seed: int
    selection: dict[str, Any]


GatePhase = Union[None, Selecting, Confirming]


@dataclass
class Progress:
    episodes: int = 0
    count_episodes: dict[int, int] = field(default_factory=dict)
    updates: int = 0
    stage: int = 0
    passes: int = 0
    best_score: Optional[float] = None
    pfsp: dict[tuple[int, str], tuple[float, int]] = field(default_factory=dict)
    cycle: int = 0
    cycle_started: int = 0
    rejections: int = 0
    promotions: int = 0
    gate: GatePhase = None
    last_gate: Optional[dict[str, Any]] = None

    @property
    def solo_episodes(self) -> int:
        return self.count_episodes.get(1, 0)

    def to_record(self) -> dict[str, Any]:
        gate_raw: Optional[dict[str, Any]]
        if isinstance(self.gate, Selecting):
            gate_raw = {
                "kind": "selecting",
                "candidate_id": self.gate.candidate_id,
                "champion_id": self.gate.champion_id,
                "seed": self.gate.seed,
            }
        elif isinstance(self.gate, Confirming):
            gate_raw = {
                "kind": "confirming",
                "candidate_id": self.gate.candidate_id,
                "champion_id": self.gate.champion_id,
                "seed": self.gate.seed,
                "selection": dict(self.gate.selection),
            }
        else:
            gate_raw = None
        return {
            "episodes": self.episodes,
            "count_episodes": tuple(sorted(self.count_episodes.items())),
            "updates": self.updates,
            "stage": self.stage,
            "passes": self.passes,
            "best_score": self.best_score,
            "pfsp": tuple(
                (players, opp, float(total), int(games))
                for (players, opp), (total, games) in sorted(self.pfsp.items())
            ),
            "cycle": self.cycle,
            "cycle_started": self.cycle_started,
            "rejections": self.rejections,
            "promotions": self.promotions,
            "gate": gate_raw,
            "last_gate": self.last_gate,
        }

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> Progress:
        if not isinstance(raw, Mapping):
            raise ValueError("progress must be a mapping")
        gate_raw = raw.get("gate")
        gate: GatePhase = None
        if gate_raw is not None:
            if not isinstance(gate_raw, Mapping):
                raise ValueError("invalid gate record")
            kind = gate_raw.get("kind")
            if kind == "selecting":
                gate = Selecting(
                    str(gate_raw["candidate_id"]),
                    str(gate_raw["champion_id"]),
                    _require_int(gate_raw["seed"], "gate.seed", minimum=0),
                )
            elif kind == "confirming":
                selection = gate_raw.get("selection")
                if not isinstance(selection, Mapping):
                    raise ValueError("confirming gate requires selection dict")
                gate = Confirming(
                    str(gate_raw["candidate_id"]),
                    str(gate_raw["champion_id"]),
                    _require_int(gate_raw["seed"], "gate.seed", minimum=0),
                    dict(selection),
                )
            else:
                raise ValueError("unknown gate kind %r" % (kind,))
        best_score = raw.get("best_score")
        if best_score is not None:
            best_score = _require_float(best_score, "best_score", minimum=0.0, maximum=1.0)
        return cls(
            episodes=_require_int(raw["episodes"], "episodes", minimum=0),
            count_episodes={
                _require_int(p, "players", minimum=1): _require_int(c, "count", minimum=0)
                for p, c in raw["count_episodes"]
            },
            updates=_require_int(raw["updates"], "updates", minimum=0),
            stage=_require_int(raw["stage"], "stage", minimum=0),
            passes=_require_int(raw["passes"], "passes", minimum=0),
            best_score=best_score,
            pfsp={
                (_require_int(p, "players", minimum=2), str(opp)): (
                    _require_float(total, "pfsp.total", minimum=0.0),
                    _require_int(games, "pfsp.games", minimum=0),
                )
                for p, opp, total, games in raw["pfsp"]
            },
            cycle=_require_int(raw["cycle"], "cycle", minimum=0),
            cycle_started=_require_int(raw["cycle_started"], "cycle_started", minimum=0),
            rejections=_require_int(raw["rejections"], "rejections", minimum=0),
            promotions=_require_int(raw["promotions"], "promotions", minimum=0),
            gate=gate,
            last_gate=dict(raw["last_gate"]) if isinstance(raw.get("last_gate"), Mapping) else None,
        )


class Trainer:
    def __init__(
        self,
        *,
        config: TrainConfig,
        output_dir: Path,
        learner: Learner,
        rng: random.Random,
        progress: Progress,
        pool: dict[str, Policy],
        pool_order: list[str],
        registry: Optional[ModelRegistry],
        initial_sha256: Optional[str],
        baseline_id: Optional[str],
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.learner = learner
        self.rng = rng
        self.progress = progress
        self.pool = pool
        self.pool_order = pool_order
        self.registry = registry
        self.initial_sha256 = initial_sha256
        self.baseline_id = baseline_id
        self._pending_events: list[dict[str, Any]] = []

    @property
    def network(self) -> QNetwork:
        return self.learner.online

    @property
    def current_distance(self) -> Optional[int]:
        if self.progress.stage < len(self.config.curriculum_distances):
            return self.config.curriculum_distances[self.progress.stage]
        return None

    @property
    def champion_sha256(self) -> Optional[str]:
        if self.registry is not None and self.registry.champion_path.exists():
            return self.registry.champion().sha256
        return self.baseline_id

    @classmethod
    def open(
        cls,
        output_dir: Union[str, Path],
        overrides: Optional[Mapping[str, Any]] = None,
        *,
        initial: Union[str, Path, ModelSpec, None] = None,
        opponents: Sequence[Union[str, Path]] = (),
    ) -> Trainer:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        state_path = root / "latest-state.pt"
        if state_path.exists():
            return cls._resume(root, state_path, overrides, initial=initial, opponents=opponents)
        return cls._create(root, overrides, initial=initial, opponents=opponents)

    @classmethod
    def _create(
        cls,
        root: Path,
        overrides: Optional[Mapping[str, Any]],
        *,
        initial: Union[str, Path, ModelSpec, None],
        opponents: Sequence[Union[str, Path]],
    ) -> Trainer:
        raw_overrides = dict(overrides or {})
        initial_path: Optional[Path] = None
        initial_sha256: Optional[str] = None
        loaded_net: Optional[QNetwork] = None
        loaded_game: Optional[GameConfig] = None
        if isinstance(initial, (str, Path)):
            initial_path = Path(initial).resolve()
            initial_sha256 = file_sha256(initial_path)
            loaded_net, loaded_game, _ = load_checkpoint(str(initial_path))
            for game_field in ("num_decks", "cards_in_hand", "required_sequences", "max_turns", "recycle_discard"):
                if raw_overrides.get(game_field) is None:
                    raw_overrides[game_field] = getattr(loaded_game, game_field)
        config = TrainConfig.resolve(None, raw_overrides)
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        if loaded_net is not None and loaded_game is not None:
            if (
                loaded_game.num_decks,
                loaded_game.cards_in_hand,
                loaded_game.required_sequences,
            ) != (
                config.num_decks,
                config.cards_in_hand,
                config.required_sequences,
            ):
                raise ValueError("initial checkpoint ruleset does not match TrainConfig")
            network = loaded_net
        else:
            spec = initial if isinstance(initial, ModelSpec) else ModelSpec()
            network = QNetwork(spec)
        learner = Learner(network, config, seed=config.seed, device=config.device)
        rng = random.Random(config.seed ^ 0x5DEECE66D)
        progress = Progress(
            count_episodes={count: 0 for count in config.player_counts},
            stage=config.initial_stage,
        )
        pool: dict[str, Policy] = {}
        pool_order: list[str] = []
        registry: Optional[ModelRegistry] = None
        baseline_id: Optional[str] = None
        needs_multiplayer = any(count > 1 for count in config.player_counts) or config.promote
        trainer = cls(
            config=config,
            output_dir=root,
            learner=learner,
            rng=rng,
            progress=progress,
            pool=pool,
            pool_order=pool_order,
            registry=registry,
            initial_sha256=initial_sha256,
            baseline_id=baseline_id,
        )
        trainer.export_model(root / "latest-model.pt", role="latest")
        start_model = initial_path if initial_path is not None else root / "latest-model.pt"
        if needs_multiplayer:
            trainer.pool[RANDOM_OPPONENT] = RandomPolicy()
            baseline_sha = trainer._register_opponent_file(start_model)
            trainer.baseline_id = baseline_sha
            for opp in opponents:
                trainer._register_opponent_file(Path(opp))
            trainer._trim_pool()
        if config.promote:
            trainer.registry = ModelRegistry(root / "registry")
            record = trainer.registry.register(str(start_model), origin="initial")
            trainer.registry.initialize_champion(record.id)
            champ_record = trainer.registry.champion()
            champ_path = trainer.registry.model_path(champ_record.id)
            atomic_copy(champ_path, root / "champion-model.pt")
            trainer._register_opponent_file(champ_path)
            trainer._trim_pool()
        trainer._save(status="initialized", stop_reason=None)
        return trainer

    @classmethod
    def _resume(
        cls,
        root: Path,
        state_path: Path,
        overrides: Optional[Mapping[str, Any]],
        *,
        initial: Union[str, Path, ModelSpec, None],
        opponents: Sequence[Union[str, Path]],
    ) -> Trainer:
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("version") != STATE_VERSION:
            raise ValueError("unsupported trainer state version; warm-start with --initial-model")
        validate_checkpoint_tree(payload, "trainer_state")
        saved_config = TrainConfig.from_dict(payload["config"])
        config = TrainConfig.resolve(saved_config, overrides)
        torch.set_num_threads(config.torch_threads)
        saved_initial = payload.get("initial_sha256")
        learner_state = payload["learner"]
        saved_spec = ModelSpec.from_dict(learner_state["spec"])
        if isinstance(initial, (str, Path)):
            given_sha = file_sha256(Path(initial).resolve())
            if saved_initial is not None and given_sha != saved_initial:
                raise ValueError("initial model checksum does not match saved state")
        elif isinstance(initial, ModelSpec) and initial != saved_spec:
            raise ValueError("ModelSpec does not match saved state")
        network = QNetwork(saved_spec)
        learner = Learner(network, config, seed=config.seed, device=config.device)
        learner.load_state_dict(learner_state)
        progress = Progress.from_record(payload["progress"])
        for count in config.player_counts:
            progress.count_episodes.setdefault(count, 0)
        rng = random.Random(0)
        rng.setstate(payload["rng"])
        torch_rng = payload.get("torch_rng")
        if torch_rng is not None and not isinstance(torch_rng, torch.Tensor):
            raise ValueError("torch_rng must be a Tensor")
        pool: dict[str, Policy] = {}
        pool_order = [str(item) for item in payload.get("pool_order", ())]
        registry = ModelRegistry(root / "registry") if config.promote else None
        trainer = cls(
            config=config,
            output_dir=root,
            learner=learner,
            rng=rng,
            progress=progress,
            pool=pool,
            pool_order=[],
            registry=registry,
            initial_sha256=saved_initial,
            baseline_id=payload.get("baseline_id"),
        )
        if any(count > 1 for count in config.player_counts) or config.promote:
            trainer.pool[RANDOM_OPPONENT] = RandomPolicy()
            opponents_dir = root / "opponents"
            for sha in pool_order:
                candidate = opponents_dir / (sha + ".pt")
                if not candidate.exists():
                    raise ValueError("missing saved opponent checkpoint %s" % candidate)
                trainer._register_opponent_file(candidate)
            for opp in opponents:
                trainer._register_opponent_file(Path(opp))
            if trainer.registry is not None and trainer.registry.champion_path.exists():
                champ = trainer.registry.champion()
                champ_path = trainer.registry.model_path(champ.id)
                atomic_copy(champ_path, root / "champion-model.pt")
                trainer._register_opponent_file(champ_path)
            trainer._trim_pool()
        trainer.export_model(root / "latest-model.pt", role="latest")
        trainer._write_status(status="resumed", stop_reason=None)
        if torch_rng is not None:
            torch.set_rng_state(torch_rng.cpu())
        return trainer

    def _register_opponent_file(self, source: Path) -> str:
        resolved = source.resolve()
        digest = file_sha256(resolved)
        opponents_dir = self.output_dir / "opponents"
        opponents_dir.mkdir(parents=True, exist_ok=True)
        dest = opponents_dir / (digest + ".pt")
        if not dest.exists():
            atomic_copy(resolved, dest)
        elif file_sha256(dest) != digest:
            raise ValueError("corrupted opponent file %s" % dest)
        if digest not in self.pool:
            policy = CheckpointPolicy(str(dest))
            if policy.game_config is not None and (
                policy.game_config.num_decks,
                policy.game_config.cards_in_hand,
                policy.game_config.required_sequences,
            ) != (
                self.config.num_decks,
                self.config.cards_in_hand,
                self.config.required_sequences,
            ):
                raise ValueError("opponent checkpoint ruleset does not match TrainConfig")
            policy.network.eval()
            self.pool[digest] = policy
        if digest not in self.pool_order:
            self.pool_order.append(digest)
        return digest

    def _trim_pool(self) -> None:
        protected = {
            sha for sha in (self.baseline_id, self.champion_sha256) if sha is not None
        }
        if len(self.pool_order) <= MAX_OPPONENT_POOL:
            return
        survivors: list[str] = [sha for sha in self.pool_order if sha in protected]
        tail_slots = max(0, MAX_OPPONENT_POOL - len(survivors))
        others = [sha for sha in self.pool_order if sha not in protected]
        survivors.extend(others[-tail_slots:] if tail_slots else [])
        keep = set(survivors)
        opponents_dir = self.output_dir / "opponents"
        for sha in list(self.pool_order):
            if sha not in keep:
                self.pool.pop(sha, None)
                (opponents_dir / (sha + ".pt")).unlink(missing_ok=True)
        self.pool_order = [sha for sha in self.pool_order if sha in keep]

    def export_model(self, path: Union[str, Path], role: str = "latest") -> None:
        payload = build_checkpoint(
            self.learner.online,
            self.config.game,
            meta={
                "algorithm": ALGORITHM,
                "role": role,
                "episodes": self.progress.episodes,
                "updates": self.progress.updates,
                "cycle": self.progress.cycle,
            },
        )
        atomic_torch_save(payload, Path(path))

    def run(
        self,
        *,
        max_seconds: Optional[float] = None,
        max_episodes: Optional[int] = None,
        max_cycles: Optional[int] = None,
        stop: Callable[[], bool] = lambda: False,
    ) -> StopReason:
        if max_seconds is not None:
            _require_float(max_seconds, "max_seconds", minimum=0.0)
        if max_episodes is not None:
            _require_int(max_episodes, "max_episodes", minimum=0)
        if max_cycles is not None:
            _require_int(max_cycles, "max_cycles", minimum=0)
            if not self.config.promote:
                raise ValueError("max_cycles requires promote=True")
        deadline = None if max_seconds is None else time.monotonic() + float(max_seconds)
        reason: Optional[StopReason] = None
        while True:
            if stop():
                reason = "stopped"
                break
            if max_cycles is not None and self.progress.cycle >= max_cycles:
                reason = "max_cycles"
                break
            if self.progress.gate is None and max_episodes is not None and self.progress.episodes >= max_episodes:
                reason = "max_episodes"
                break
            if deadline is not None and time.monotonic() >= deadline:
                reason = "budget"
                break
            if self.progress.gate is not None:
                if not self._step_gate(deadline):
                    reason = "budget"
                    break
                self._save(status="running", stop_reason=None)
                continue
            limit = self.config.round_episodes
            if max_episodes is not None:
                limit = min(limit, max_episodes - self.progress.episodes)
            if self.config.promote:
                cycle_left = self.config.cycle_episodes - (
                    self.progress.episodes - self.progress.cycle_started
                )
                limit = min(limit, max(1, cycle_left))
            prev_bucket = self.progress.episodes // self.config.checkpoint_every
            self._round(limit)
            new_bucket = self.progress.episodes // self.config.checkpoint_every
            if new_bucket > prev_bucket or self.progress.gate is not None:
                self._save(status="running", stop_reason=None)
        self._save(status="completed" if reason in ("max_episodes", "max_cycles") else "stopped", stop_reason=reason)
        return reason

    def _round(self, count: int) -> None:
        epsilon = epsilon_at(self.config, self.progress.episodes)
        policy = EpsilonPolicy(self.learner.online, epsilon)
        was_training = self.learner.online.training
        self.learner.online.eval()
        solo_specs: list[tuple[int, str, Optional[int]]] = []
        match_specs: list[tuple[int, int, int, tuple[str, ...]]] = []
        pool_ids = [*self.pool_order, RANDOM_OPPONENT] if RANDOM_OPPONENT in self.pool else list(self.pool_order)
        champ_sha = self.champion_sha256
        try:
            for offset in range(count):
                global_index = self.progress.episodes + offset
                players = self.config.player_counts[global_index % len(self.config.player_counts)]
                deal_seed = self.rng.randrange(1, 1 << 60)
                if players == 1:
                    roll = self.rng.random()
                    distance = self.current_distance
                    if roll < self.config.warm_start_fraction:
                        solo_specs.append((deal_seed, "warm", None))
                    elif distance is not None and roll < self.config.warm_start_fraction + self.config.curriculum_fraction:
                        solo_specs.append((deal_seed, "curriculum", distance))
                    else:
                        solo_specs.append((deal_seed, "ordinary", None))
                else:
                    seat = self.rng.randrange(players)
                    probs = pfsp_probabilities(
                        self.config, self.progress.pfsp, players, pool_ids, champ_sha,
                    )
                    opponents = tuple(
                        self.rng.choices(pool_ids, weights=probs, k=players - 1)
                    )
                    match_specs.append((players, deal_seed, seat, opponents))
                self.progress.count_episodes[players] = self.progress.count_episodes.get(players, 0) + 1
            prev_solo = self.progress.solo_episodes - len(solo_specs)
            trajectories: list[Trajectory] = []
            solo_wins = 0
            if solo_specs:
                solo_trajs, solo_wins = self._play_solo(solo_specs, policy)
                trajectories.extend(solo_trajs)
            match_wins = 0
            if match_specs:
                match_trajs, match_wins = self._play_matches(match_specs, policy, pool_ids)
                trajectories.extend(match_trajs)
        finally:
            self.learner.online.train(was_training)
        self.progress.episodes += count
        update_steps = self.config.updates_per_episode * len(trajectories)
        stats = self.learner.learn(trajectories, updates=update_steps)
        self.progress.updates = self.learner.update_count
        self._pending_events.append({
            "event": "round",
            "episodes": self.progress.episodes,
            "round_episodes": count,
            "solo_wins": solo_wins,
            "match_wins": match_wins,
            "updates": self.progress.updates,
            "epsilon": round(epsilon, 6),
            "loss": round(stats.loss, 6),
            "q_loss": round(stats.q_loss, 6),
            "stage": self.progress.stage,
        })
        now_solo = self.progress.solo_episodes
        if (
            self.config.validation_games > 0
            and len(solo_specs) > 0
            and now_solo // self.config.validate_every > prev_solo // self.config.validate_every
        ):
            self._validate()
        if (
            self.config.promote
            and self.progress.episodes - self.progress.cycle_started >= self.config.cycle_episodes
        ):
            self._open_gate()

    def _play_solo(
        self,
        specs: Sequence[tuple[int, str, Optional[int]]],
        policy: EpsilonPolicy,
    ) -> tuple[list[Trajectory], int]:
        game = self.config.game
        envs = [PappluEnv(game) for _ in specs]
        observations = []
        for env, (seed, mode, distance) in zip(envs, specs):
            if mode == "warm":
                observations.append(env.reset_warm_start(seed=seed))
            elif mode == "curriculum" and distance is not None:
                observations.append(env.reset_curriculum(distance, seed=seed))
            else:
                observations.append(env.reset(seed=seed))
        rngs = [random.Random(seed ^ SOLO_EXPLORATION_SALT) for seed, _, _ in specs]
        recorded: list[list[Step]] = [[] for _ in specs]
        while True:
            active = [idx for idx, obs in enumerate(observations) if not obs.done]
            if not active:
                break
            feats = [encode(observations[idx]) for idx in active]
            actions = policy.act_features(feats, [rngs[idx] for idx in active])
            for idx, feat, action in zip(active, feats, actions):
                recorded[idx].append(Step(feat, action))
                observations[idx] = envs[idx].step(action)
        trajectories: list[Trajectory] = []
        wins = 0
        for obs, steps in zip(observations, recorded):
            declared = obs.won
            if declared:
                wins += 1
            if steps:
                outcome = solo_outcome(obs.hand, obs.joker, declared, game.required_sequences)
                trajectories.append(Trajectory(1, tuple(steps), outcome))
        return trajectories, wins

    def _play_matches(
        self,
        specs: Sequence[tuple[int, int, int, tuple[str, ...]]],
        policy: EpsilonPolicy,
        pool_ids: Sequence[str],
    ) -> tuple[list[Trajectory], int]:
        policies: list[Policy] = [policy, *(self.pool[opp] for opp in pool_ids)]
        index_of = {opp: idx + 1 for idx, opp in enumerate(pool_ids)}
        deals: list[Deal] = []
        seat_opponents: list[dict[int, str]] = []
        for players, seed, learner_seat, opponents in specs:
            seats: list[int] = []
            opp_iter = iter(opponents)
            opp_map: dict[int, str] = {}
            for seat in range(players):
                if seat == learner_seat:
                    seats.append(0)
                else:
                    opp_id = next(opp_iter)
                    seats.append(index_of[opp_id])
                    opp_map[seat] = opp_id
            deals.append(Deal(seed=seed, seats=tuple(seats), players=players, record_seats=frozenset((learner_seat,))))
            seat_opponents.append(opp_map)
        results = run_matches(deals, policies, self.config.game, score_outcome=True)
        trajectories: list[Trajectory] = []
        wins = 0
        for (players, _, learner_seat, _), opp_map, result in zip(specs, seat_opponents, results):
            assert result.outcome is not None
            seat_outcome = result.outcome.seat(learner_seat)
            if seat_outcome.won:
                wins += 1
            for other_seat, opp_id in opp_map.items():
                score = 0.5 * (result.outcome.margin(learner_seat, other_seat) + 1.0)
                prev_total, prev_games = self.progress.pfsp.get((players, opp_id), (0.0, 0))
                self.progress.pfsp[(players, opp_id)] = (prev_total + score, prev_games + 1)
            raw_steps = result.trajectories[learner_seat]
            if raw_steps:
                steps = tuple(
                    Step(item.features, item.action, item.hidden_hands[1:])
                    for item in raw_steps
                )
                trajectories.append(Trajectory(players, steps, seat_outcome))
        return trajectories, wins

    def _validate(self) -> None:
        game = self.config.game
        games = self.config.validation_games
        ordinary = evaluate_games(
            self.learner.online,
            game,
            games,
            ORDINARY_VALIDATION_SEED,
            policy="greedy",
        )
        win_rate = ordinary.to_dict()["win_rate"]
        if self.progress.best_score is None or win_rate >= self.progress.best_score:
            self.progress.best_score = win_rate
            self.export_model(self.output_dir / "best-model.pt", role="best")
        distance = self.current_distance
        stage_rate: Optional[float] = None
        random_rate: Optional[float] = None
        if distance is not None:
            seed = CURRICULUM_VALIDATION_SEED + self.progress.stage * 10_000
            greedy_stage = evaluate_games(
                self.learner.online,
                game,
                games,
                seed,
                policy="greedy",
                curriculum_distance=distance,
            )
            random_stage = evaluate_games(
                self.learner.online,
                game,
                games,
                seed,
                policy="random",
                curriculum_distance=distance,
            )
            stage_rate = greedy_stage.to_dict()["win_rate"]
            random_rate = random_stage.to_dict()["win_rate"]
            self.progress.stage, self.progress.passes = curriculum_step(
                self.config,
                self.progress.stage,
                self.progress.passes,
                stage_rate,
                random_rate,
            )
        self._pending_events.append({
            "event": "validation",
            "episodes": self.progress.episodes,
            "ordinary_win_rate": round(win_rate, 6),
            "best_score": round(self.progress.best_score, 6),
            "curriculum_distance": distance,
            "stage_win_rate": None if stage_rate is None else round(stage_rate, 6),
            "stage_random_rate": None if random_rate is None else round(random_rate, 6),
            "stage": self.progress.stage,
            "passes": self.progress.passes,
        })

    def _open_gate(self) -> None:
        assert self.registry is not None
        champion = self.registry.champion()
        with tempfile.NamedTemporaryFile(
            dir=self.output_dir, prefix=".candidate-", suffix=".pt", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            self.export_model(temp_path, role="candidate")
            candidate = self.registry.register(
                str(temp_path), origin="trainer", parents=(champion.id,),
            )
        finally:
            temp_path.unlink(missing_ok=True)
        if candidate.id == champion.id:
            self.progress.last_gate = {
                "accepted": False,
                "phase": "selection",
                "candidate_id": candidate.id,
                "champion_id": champion.id,
                "gate_failures": ["identical_candidate"],
            }
            self._close_cycle(promoted=False, candidate_id=candidate.id)
            return
        gap = disjoint_seed_gap(self.config)
        seed = SELECTION_SEED_BASE + self.progress.cycle * (2 * gap)
        self.progress.gate = Selecting(candidate.id, champion.id, seed)

    def _step_gate(self, deadline: Optional[float]) -> bool:
        assert self.registry is not None and self.progress.gate is not None
        gate = self.progress.gate
        phase_name = "selection" if isinstance(gate, Selecting) else "confirmation"
        blocks = (
            self.config.selection_blocks
            if isinstance(gate, Selecting)
            else self.config.confirmation_blocks
        )
        candidate_path = str(self.registry.model_path(gate.candidate_id))
        champion_path = str(self.registry.model_path(gate.champion_id))
        candidate_sha = self.registry.get(gate.candidate_id).sha256
        champion_sha = self.registry.get(gate.champion_id).sha256
        opponents_dir = self.output_dir / "opponents"
        opponent_paths = [
            str(opponents_dir / (sha + ".pt"))
            for sha in self.pool_order
            if sha not in (candidate_sha, champion_sha) and (opponents_dir / (sha + ".pt")).exists()
        ]
        try:
            result = evaluate_gate(
                candidate_path=candidate_path,
                champion_path=champion_path,
                opponent_paths=opponent_paths,
                seed=gate.seed,
                seed_blocks=blocks,
                solo_games=self.config.solo_games,
                max_turns=self.config.max_turns,
                player_counts=self.config.gate_counts,
                samples=self.config.gate_samples,
                deadline=deadline,
                recycle_discard=self.config.recycle_discard,
                workers=self.config.gate_workers,
                journal_dir=self.output_dir / "registry" / "gates",
                phase=phase_name,
            )
        except (GateInterrupted, MatchInterrupted, TimeoutError):
            return False
        self.progress.last_gate = result
        self._pending_events.append({
            "event": "gate",
            "cycle": self.progress.cycle,
            "phase": phase_name,
            "candidate_id": gate.candidate_id,
            "champion_id": gate.champion_id,
            "accepted": bool(result.get("accepted")),
            "gate_failures": list(result.get("gate_failures", ())),
        })
        if not result.get("accepted"):
            self._close_cycle(promoted=False, candidate_id=gate.candidate_id)
            return True
        if isinstance(gate, Selecting):
            confirm_seed = gate.seed + disjoint_seed_gap(self.config)
            self.progress.gate = Confirming(
                gate.candidate_id, gate.champion_id, confirm_seed, dict(result),
            )
            return True
        if self.registry.champion().id != gate.candidate_id:
            promotion_payload = {
                **result,
                "selection_passed": True,
                "selection_seed": gate.selection["seed"],
                "selection": gate.selection,
            }
            self.registry.promote(
                gate.candidate_id,
                gate.champion_id,
                promotion_payload,
                player_counts=self.config.gate_counts,
            )
        self._close_cycle(promoted=True, candidate_id=gate.candidate_id)
        return True

    def _close_cycle(self, *, promoted: bool, candidate_id: str) -> None:
        assert self.registry is not None
        if promoted:
            champ_path = self.registry.model_path(candidate_id)
            atomic_copy(champ_path, self.output_dir / "champion-model.pt")
            self._register_opponent_file(champ_path)
            self._trim_pool()
            self.progress.promotions += 1
            self.progress.rejections = 0
        else:
            self.progress.rejections += 1
            if (
                self.config.rebase_after_rejections > 0
                and self.progress.rejections >= self.config.rebase_after_rejections
            ):
                champ = self.registry.champion()
                champ_net, _, _ = load_checkpoint(str(self.registry.model_path(champ.id)))
                self.learner.adopt(champ_net.state_dict())
                self.progress.rejections = 0
                self._pending_events.append({
                    "event": "rebase",
                    "cycle": self.progress.cycle,
                    "champion_id": champ.id,
                })
        self.progress.cycle += 1
        self.progress.cycle_started = self.progress.episodes
        self.progress.gate = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "config": self.config.to_dict(),
            "initial_sha256": self.initial_sha256,
            "baseline_id": self.baseline_id,
            "pool_order": tuple(self.pool_order),
            "rng": self.rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "progress": self.progress.to_record(),
            "learner": self.learner.state_dict(),
        }

    def summary(self) -> dict[str, Any]:
        champ_id = (
            self.registry.champion().id
            if self.registry is not None and self.registry.champion_path.exists()
            else None
        )
        return {
            "episodes": self.progress.episodes,
            "count_episodes": {
                str(players): self.progress.count_episodes.get(players, 0)
                for players in self.config.player_counts
            },
            "updates": self.progress.updates,
            "stage": self.progress.stage,
            "curriculum_distance": self.current_distance,
            "best_score": self.progress.best_score,
            "completed_cycles": self.progress.cycle,
            "promotions": self.progress.promotions,
            "rejections": self.progress.rejections,
            "champion_id": champ_id,
            "replay_steps": len(self.learner.replay),
            "replay_by_players": {
                str(k): v for k, v in self.learner.replay.counts_by_players().items()
            },
            "pool_size": len(self.pool),
        }

    def _write_status(self, *, status: str, stop_reason: Optional[str]) -> None:
        payload = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "status": status,
            "stop_reason": stop_reason,
            "rules": self.config.game.to_dict(),
            "last_gate": self.progress.last_gate,
            **self.summary(),
        }
        atomic_write_json(payload, self.output_dir / "status.json")

    def _save(self, *, status: str, stop_reason: Optional[str]) -> None:
        atomic_torch_save(self.state_dict(), self.output_dir / "latest-state.pt")
        self.export_model(self.output_dir / "latest-model.pt", role="latest")
        if self.registry is not None and self.registry.champion_path.exists():
            champ = self.registry.champion()
            atomic_copy(self.registry.model_path(champ.id), self.output_dir / "champion-model.pt")
        self._write_status(status=status, stop_reason=stop_reason)
        if self._pending_events:
            metrics_path = self.output_dir / "metrics.jsonl"
            with metrics_path.open("a", encoding="utf-8") as handle:
                for event in self._pending_events:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
            self._pending_events.clear()
