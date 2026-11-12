from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch
from torch import Tensor

from src.game.belief import BeliefState
from src.game.environment import NUM_ACTIONS, Observation, Phase
from src.game.lookahead import Continues, Declares, can_declare_ub, expand, outs_1ply, winning_discard
from src.model.network import QNetwork, collate_observations


@dataclass(frozen=True)
class TeacherConfig:
    fraction: float = 0.125
    max_children: int = 64
    every_updates: int = 4
    teacher_weight: float = 0.5
    after_weight: float = 0.25

    def __post_init__(self) -> None:
        for name in ("fraction", "teacher_weight", "after_weight"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and nonnegative" % name)
        if self.fraction > 1:
            raise ValueError("teacher fraction must be in [0, 1]")
        for name in ("max_children", "every_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)


@dataclass(frozen=True)
class TeacherTargets:
    rows: Tensor
    mask: Tensor
    values: Tensor


@torch.inference_mode()
def state_values(target: QNetwork, observations: Sequence[Observation], device: torch.device) -> Tensor:
    values = torch.empty(len(observations), dtype=torch.float32, device=device)
    decisions, afterstates = [], []
    for row, obs in enumerate(observations):
        if obs.phase is not Phase.DISCARD:
            afterstates.append(row)
        elif winning_discard(
            obs.hand, obs.joker, obs.config.required_sequences, obs.config.cards_in_hand,
        ) is not None:
            values[row] = 1.0
        else:
            decisions.append(row)
    if decisions:
        batch = collate_observations([observations[row] for row in decisions], device)
        values[decisions] = target(batch).masked_fill(batch.legal == 0, -torch.inf).max(dim=1).values
    if afterstates:
        batch = collate_observations([observations[row] for row in afterstates], device)
        values[afterstates] = target.outputs(batch).after
    return values


@torch.inference_mode()
def teacher_targets(
    target: QNetwork, observations: Sequence[Observation], config: TeacherConfig, rng: random.Random,
    gamma: float, device: torch.device,
) -> TeacherTargets:
    rows, plans, children = [], [], []
    candidates = list(range(len(observations)))
    rng.shuffle(candidates)
    limit = min(len(candidates), max(1, int(config.fraction * len(candidates)))) if config.fraction else 0
    for row in candidates:
        if len(rows) >= limit:
            break
        actions = {}
        for action, branches in expand(observations[row]).items():
            needed = sum(isinstance(branch, Continues) for branch in branches)
            if not branches or len(children) + needed > config.max_children:
                continue
            terms = []
            for branch in branches:
                if isinstance(branch, Declares):
                    terms.append((branch.probability, None))
                else:
                    terms.append((branch.probability, len(children)))
                    children.append(branch.observation)
            actions[action] = terms
        if actions:
            rows.append(row)
            plans.append(actions)
    values = torch.zeros((len(rows), NUM_ACTIONS), dtype=torch.float32, device=device)
    mask = torch.zeros_like(values, dtype=torch.bool)
    child_values = state_values(target, children, device).tolist()
    for position, actions in enumerate(plans):
        for action, terms in actions.items():
            value = sum(probability * (1.0 if child is None else child_values[child])
                        for probability, child in terms)
            values[position, action] = value * (gamma if action < 2 else 1.0)
            mask[position, action] = True
    return TeacherTargets(torch.tensor(rows, dtype=torch.long, device=device), mask, values)
