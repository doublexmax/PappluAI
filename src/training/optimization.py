from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from src.checkpoints.tensor import clone_model_state, validate_checkpoint_tree, validate_model_state
from src.game.lookahead import after_discard
from src.model.network import QNetwork, collate, collate_observations, greedy_actions
from src.model.search import TeacherConfig, teacher_targets
from src.training.core import Trajectory, TrajectoryReplay


@dataclass(frozen=True)
class LossWeights:
    win: float = 0.25
    penalty: float = 0.25
    belief: float = 0.10

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("%s loss weight must be finite and nonnegative" % name)


@dataclass(frozen=True)
class LearnerConfig:
    batch_size: int = 128
    replay_capacity: int = 50_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 1.0
    n_step: int = 5
    target_tau: float = 0.005
    grad_clip: float = 1.0
    losses: LossWeights = LossWeights()
    teacher: TeacherConfig = TeacherConfig()

    def __post_init__(self) -> None:
        for name in ("batch_size", "replay_capacity", "n_step"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        for name in ("learning_rate", "gamma", "target_tau", "grad_clip"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        if self.gamma > 1 or self.target_tau > 1:
            raise ValueError("gamma and target_tau must be in (0, 1]")
        if (
            type(self.weight_decay) not in (int, float)
            or not math.isfinite(self.weight_decay) or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and nonnegative")
        if not isinstance(self.losses, LossWeights) or not isinstance(self.teacher, TeacherConfig):
            raise TypeError("learner needs LossWeights and TeacherConfig")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LearnerConfig":
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("learner config fields do not match")
        values = dict(raw)
        for name, kind in (("losses", LossWeights), ("teacher", TeacherConfig)):
            if not isinstance(values[name], dict) or set(values[name]) != set(kind.__dataclass_fields__):
                raise ValueError("learner %s fields do not match" % name)
            values[name] = kind(**values[name])
        return cls(**values)


@dataclass(frozen=True)
class LearnStats:
    updates: int = 0
    samples: int = 0
    loss: float = 0.0
    q_loss: float = 0.0
    win_loss: float = 0.0
    penalty_loss: float = 0.0
    belief_loss: float = 0.0
    after_loss: float = 0.0
    teacher_loss: float = 0.0


class Learner:
    def __init__(
        self, online: QNetwork, config: LearnerConfig = LearnerConfig(), seed: int = 0,
    ) -> None:
        if type(seed) is not int:
            raise ValueError("learner seed must be an integer")
        self.config = config
        self.online = online
        self.target = deepcopy(online).eval().requires_grad_(False)
        self.optimizer = self._optimizer()
        self.replay = TrajectoryReplay(config.replay_capacity)
        self.rng = random.Random(seed)
        self.torch_rng_state = torch.Generator(device="cpu").manual_seed(seed).get_state()
        self.decision_count = 0
        self.update_count = 0

    def _optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.online.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay,
        )

    def append(self, trajectories: Sequence[Trajectory]) -> int:
        added = 0
        for trajectory in trajectories:
            self.replay.append(trajectory)
            added += len(trajectory.decisions)
        self.decision_count += added
        return added

    def learn(
        self, trajectories: Sequence[Trajectory], updates: int = 1, *,
        device: torch.device = torch.device("cpu"),
    ) -> LearnStats:
        self.append(trajectories)
        return self.update(updates, device=device)

    def update(
        self, updates: int = 1, *, device: torch.device = torch.device("cpu"),
    ) -> LearnStats:
        if type(updates) is not int or updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        if not len(self.replay) or not updates:
            return LearnStats()
        self.online.to(device).train()
        self.target.to(device).eval()
        for state in self.optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, Tensor) and name != "step":
                    state[name] = value.to(device)
        totals = dict.fromkeys((
            "loss", "q_loss", "win_loss", "penalty_loss", "belief_loss", "after_loss", "teacher_loss",
        ), 0.0)
        sample_count = 0
        previous = torch.get_rng_state()
        try:
            torch.set_rng_state(self.torch_rng_state)
            for _ in range(updates):
                samples = self.replay.sample(
                    self.rng, min(self.config.batch_size, len(self.replay)),
                    n_step=self.config.n_step, gamma=self.config.gamma,
                )
                now = collate([sample.decision.features for sample in samples], device)
                actions = torch.tensor(
                    [sample.decision.action for sample in samples], device=device, dtype=torch.long,
                )
                y = torch.tensor([sample.outcome.utility for sample in samples], device=device, dtype=torch.float32)
                boot_rows = [row for row, sample in enumerate(samples) if sample.bootstrap is not None]
                with torch.no_grad():
                    if boot_rows:
                        nb = collate([samples[row].bootstrap.features for row in boot_rows], device)
                        a_star = greedy_actions(self.online, nb)
                        y[boot_rows] = self.target(nb).gather(1, a_star[:, None]).squeeze(1)
                    y *= torch.tensor([sample.scale for sample in samples], device=device, dtype=torch.float32)
                won = torch.tensor([sample.outcome.won for sample in samples], device=device, dtype=torch.float32)
                penalties = torch.tensor(
                    [sample.outcome.penalty / 100.0 for sample in samples], device=device, dtype=torch.float32,
                )
                out = self.online.outputs(now)
                q_loss = F.smooth_l1_loss(out.q.gather(1, actions[:, None]).squeeze(1), y)
                win_loss = F.binary_cross_entropy_with_logits(
                    out.win_logit.gather(1, actions[:, None]).squeeze(1), won,
                )
                penalty_loss = F.smooth_l1_loss(
                    out.penalty.gather(1, actions[:, None]).squeeze(1), penalties,
                )
                present = now.seat[:, 1:, 0] > 0
                belief_loss = out.q.new_zeros(())
                if bool(present.any()):
                    hidden = torch.tensor(
                        [sample.decision.hidden_hands for sample in samples],
                        device=device, dtype=torch.float32,
                    )
                    decks = torch.tensor(
                        [sample.decision.observation.config.num_decks for sample in samples], device=device,
                    )
                    belief_targets = (hidden[:, 1:] / decks[:, None, None]).clamp(0, 1)
                    belief_loss = F.binary_cross_entropy_with_logits(
                        out.belief[:, 1:][present], belief_targets[present],
                    )
                after_loss = out.q.new_zeros(())
                teacher_loss = out.q.new_zeros(())
                teacher = self.config.teacher
                if teacher.fraction > 0 and (self.update_count + 1) % teacher.every_updates == 0:
                    discard_rows = [row for row, sample in enumerate(samples) if sample.decision.action >= 2]
                    if discard_rows:
                        after_obs = [
                            after_discard(samples[row].decision.observation, samples[row].decision.action - 2)
                            for row in discard_rows
                        ]
                        after_loss = F.smooth_l1_loss(
                            self.online.outputs(collate_observations(after_obs, device)).after, y[discard_rows],
                        )
                    targets = teacher_targets(
                        self.target, [sample.decision.observation for sample in samples],
                        teacher, self.rng, self.config.gamma, device,
                    )
                    if bool(targets.mask.any()):
                        teacher_loss = F.smooth_l1_loss(
                            out.q[targets.rows][targets.mask], targets.values[targets.mask],
                        )
                weights = self.config.losses
                loss = (
                    q_loss + weights.win * win_loss + weights.penalty * penalty_loss
                    + weights.belief * belief_loss + teacher.after_weight * after_loss
                    + teacher.teacher_weight * teacher_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.update_count += 1
                with torch.no_grad():
                    for target, online in zip(self.target.parameters(), self.online.parameters()):
                        target.lerp_(online, self.config.target_tau)
                for name, value in (
                    ("loss", loss), ("q_loss", q_loss), ("win_loss", win_loss),
                    ("penalty_loss", penalty_loss), ("belief_loss", belief_loss),
                    ("after_loss", after_loss), ("teacher_loss", teacher_loss),
                ):
                    totals[name] += float(value.detach())
                sample_count += len(samples)
            self.torch_rng_state = torch.get_rng_state()
        finally:
            torch.set_rng_state(previous)
        return LearnStats(updates=updates, samples=sample_count, **{
            name: value / updates for name, value in totals.items()
        })

    def adopt(self, model_state: Mapping[str, Tensor]) -> None:
        validated = validate_model_state(model_state, self.online.state_dict(), "adopt model")
        self.online.load_state_dict(validated)
        self.target.load_state_dict(validated)
        self.target.eval().requires_grad_(False)
        self.optimizer = self._optimizer()

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "online": clone_model_state(self.online.state_dict()),
            "target": clone_model_state(self.target.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "replay": self.replay.state_dict(),
            "rng": self.rng.getstate(),
            "torch_rng_state": self.torch_rng_state.clone(),
            "decision_count": self.decision_count,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != {
            "config", "online", "target", "optimizer", "replay", "rng",
            "torch_rng_state", "decision_count", "update_count",
        }:
            raise ValueError("learner state fields do not match")
        if LearnerConfig.from_dict(state["config"]) != self.config:
            raise ValueError("learner config changed")
        online = validate_model_state(state["online"], self.online.state_dict(), "online")
        target = validate_model_state(state["target"], self.target.state_dict(), "target")
        counters = {}
        for name in ("decision_count", "update_count"):
            value = state[name]
            if type(value) is not int or value < 0:
                raise ValueError("%s must be a nonnegative integer" % name)
            counters[name] = value
        replay = TrajectoryReplay(self.config.replay_capacity)
        replay.load_state_dict(state["replay"])
        if counters["decision_count"] < len(replay):
            raise ValueError("learner decision count is smaller than replay")
        rng = random.Random()
        try:
            rng.setstate(state["rng"])
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError("learner RNG state is invalid") from error
        torch_state = state["torch_rng_state"]
        if not isinstance(torch_state, Tensor) or torch_state.dtype != torch.uint8 or torch_state.ndim != 1:
            raise ValueError("learner torch RNG state is invalid")
        try:
            torch.Generator(device="cpu").set_state(torch_state.cpu())
        except RuntimeError as error:
            raise ValueError("learner torch RNG state is invalid") from error
        validate_checkpoint_tree(state["optimizer"], "optimizer")
        candidate = torch.optim.AdamW(
            self.online.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay,
        )
        expected = candidate.state_dict()
        raw_optimizer = state["optimizer"]
        if not isinstance(raw_optimizer, dict) or set(raw_optimizer) != set(expected):
            raise ValueError("learner optimizer fields do not match AdamW")
        groups = raw_optimizer["param_groups"]
        if not isinstance(groups, list) or len(groups) != 1 or groups[0] != expected["param_groups"][0]:
            raise ValueError("learner optimizer parameter group changed")
        raw_states = raw_optimizer["state"]
        if not isinstance(raw_states, dict):
            raise ValueError("learner optimizer state must be a dict")
        parameters = dict(enumerate(self.online.parameters()))
        for index, values in raw_states.items():
            if type(index) is not int or index not in parameters or not isinstance(values, dict) or set(values) != {
                "step", "exp_avg", "exp_avg_sq",
            }:
                raise ValueError("learner optimizer parameter state is invalid")
            for name in ("exp_avg", "exp_avg_sq"):
                value = values[name]
                if not isinstance(value, Tensor) or value.shape != parameters[index].shape or value.dtype != parameters[index].dtype:
                    raise ValueError("learner optimizer moments are incompatible")
                if name == "exp_avg_sq" and bool((value < 0).any()):
                    raise ValueError("learner optimizer squared moments must be nonnegative")
            step = values["step"]
            if (
                not isinstance(step, Tensor) or step.numel() != 1
                or step.dtype not in (torch.float32, torch.float64)
                or not 0 <= float(step) <= counters["update_count"] or float(step) != int(step)
            ):
                raise ValueError("learner optimizer step is invalid")
        candidate.load_state_dict(deepcopy(raw_optimizer))
        self.online.load_state_dict(online)
        self.target.load_state_dict(target)
        self.target.eval().requires_grad_(False)
        self.optimizer = candidate
        self.replay = replay
        self.rng.setstate(rng.getstate())
        self.torch_rng_state = torch_state.cpu().clone()
        self.decision_count = counters["decision_count"]
        self.update_count = counters["update_count"]
