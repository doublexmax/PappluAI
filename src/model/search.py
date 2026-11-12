from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch
from torch import Tensor

from src.game.belief import BeliefState
from src.game.environment import NUM_ACTIONS, Observation, Phase
from src.game.lookahead import Continues, expand, winning_discard
from src.game.outcome import DECLARATION_UTILITY
from src.model.network import QNetwork, collate_observations


@dataclass(frozen=True)
class TeacherConfig:
    fraction: float = 0.0625
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
            values[row] = DECLARATION_UTILITY
        else:
            decisions.append(row)
    if decisions:
        batch = collate_observations([observations[row] for row in decisions], device)
        values[decisions] = target(batch).masked_fill(batch.legal == 0, -torch.inf).max(dim=1).values
    if afterstates:
        batch = collate_observations([observations[row] for row in afterstates], device)
        values[afterstates] = target.outputs(batch).after
    return values


@torch.no_grad()
def teacher_targets(
    target: QNetwork, observations: Sequence[Observation], config: TeacherConfig, rng: random.Random,
    gamma: float, device: torch.device,
) -> TeacherTargets:
    rows, plans, children = [], [], []
    candidates = list(range(len(observations)))
    rng.shuffle(candidates)
    limit = min(len(candidates), max(1, int(config.fraction * len(candidates)))) if config.fraction else 0
    for row in candidates:
        if len(rows) >= limit or len(children) >= config.max_children:
            break
        obs = observations[row]
        estimate = (
            sum(copies > 0 for copies in obs.hand) if obs.phase is Phase.DISCARD
            else 1 + sum(weight > 0 for weight in BeliefState.from_observation(obs).stock_weights)
        )
        if children and len(children) + estimate > config.max_children:
            continue
        expanded = expand(obs)
        row_needed = sum(
            isinstance(branch, Continues) for branches in expanded.values() for branch in branches
        )
        if not expanded or len(children) + row_needed > config.max_children:
            continue
        actions = {}
        for action, branches in expanded.items():
            terms = []
            for branch in branches:
                if isinstance(branch, Continues):
                    terms.append((branch.probability, len(children), 0.0))
                    children.append(branch.observation)
                else:
                    terms.append((branch.probability, None, branch.value))
            actions[action] = terms
        if actions:
            rows.append(row)
            plans.append(actions)
    values = torch.zeros((len(rows), NUM_ACTIONS), dtype=torch.float32, device=device)
    mask = torch.zeros_like(values, dtype=torch.bool)
    child_values = state_values(target, children, device).tolist()
    for position, actions in enumerate(plans):
        for action, terms in actions.items():
            value = sum(probability * (terminal if child is None else child_values[child])
                        for probability, child, terminal in terms)
            values[position, action] = value * (gamma if action < 2 else 1.0)
            mask[position, action] = True
    return TeacherTargets(torch.tensor(rows, dtype=torch.long, device=device), mask, values)
