from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.checkpoints.io import file_sha256
from src.game.environment import GameConfig, NUM_ACTIONS, Observation, legal_action_mask
from src.model.encoding import ObservationBatch, collate_observations
from src.model.network import QNetwork, load_checkpoint, save_checkpoint
from src.training.core import TerminalShaping


META_KEY = "policy_value"
DEFAULT_DRAW_TEMPERATURE = 0.05
DEFAULT_DISCARD_TEMPERATURE = 0.05


@dataclass(frozen=True)
class PositionEvaluation:
    actions: Tuple[int, ...]
    prior: Tuple[float, ...]
    q_values: Tuple[float, ...]
    position_value: float


@dataclass(frozen=True)
class PolicyValueOutput:
    q: torch.Tensor
    log_policy: torch.Tensor
    position_value: torch.Tensor
    temperature: torch.Tensor


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        qnet: QNetwork,
        game_config: GameConfig = GameConfig(),
        *,
        draw_temperature: float = DEFAULT_DRAW_TEMPERATURE,
        discard_temperature: float = DEFAULT_DISCARD_TEMPERATURE,
        shaping: TerminalShaping = TerminalShaping(),
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        for name, value in (
            ("draw_temperature", draw_temperature),
            ("discard_temperature", discard_temperature),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        if type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not isinstance(game_config, GameConfig) or not isinstance(shaping, TerminalShaping):
            raise TypeError("game_config and shaping must use their domain types")
        self.qnet = qnet
        self.game_config = game_config
        self.shaping = shaping
        self.gamma = float(gamma)
        self.log_temperature = nn.Parameter(torch.tensor(
            [math.log(draw_temperature), math.log(discard_temperature)],
            dtype=torch.float32,
            device=next(qnet.parameters()).device,
        ))

    @classmethod
    def from_checkpoint(
        cls, path: str, *, device: Optional[torch.device] = None,
    ) -> "PolicyValueNet":
        qnet, game_config, meta = load_checkpoint(path, device=device)
        settings = dict(meta.get(META_KEY, {}))
        if "shaping" in settings:
            settings["shaping"] = TerminalShaping.from_dict(settings["shaping"])
        return cls(qnet, game_config, **settings)

    def save(self, path: str, meta: Optional[Mapping[str, Any]] = None) -> str:
        temperatures = self.log_temperature.detach().exp().cpu().tolist()
        metadata = dict(meta or {})
        metadata[META_KEY] = {
            "draw_temperature": temperatures[0],
            "discard_temperature": temperatures[1],
            "gamma": self.gamma,
            "shaping": self.shaping.to_dict(),
        }
        save_checkpoint(path, self.qnet, self.game_config, metadata)
        return file_sha256(Path(path))

    def forward(self, batch: ObservationBatch, mask: torch.Tensor) -> PolicyValueOutput:
        if mask.dtype is not torch.bool or mask.shape != (batch.cards.shape[0], NUM_ACTIONS):
            raise ValueError("mask must be a bool tensor with shape [B, 54]")
        if mask.device != batch.cards.device or not bool(mask.any(dim=1).all()):
            raise ValueError("mask must share the batch device and allow an action in every row")
        q = self.qnet(batch)
        discard = batch.context[:, 1] > 0.5
        temperature = self.log_temperature.exp()[discard.long()].clamp_min(
            torch.finfo(q.dtype).eps,
        )
        log_policy = torch.log_softmax(
            (q / temperature[:, None]).masked_fill(~mask, -torch.inf), dim=1,
        )
        expected_q = (log_policy.exp() * q.masked_fill(~mask, 0.0)).sum(dim=1)
        position_value = torch.where(discard, 0.5 * (q[:, 0] + expected_q), q[:, 0])
        return PolicyValueOutput(q, log_policy, position_value, temperature)

    @torch.no_grad()
    def evaluate_positions(
        self,
        observations: Sequence[Observation],
        device: Optional[torch.device] = None,
    ) -> Tuple[PositionEvaluation, ...]:
        if not observations:
            return ()
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        masks = [legal_action_mask(obs) for obs in observations]
        mask = torch.tensor(masks, dtype=torch.bool, device=device)
        out = self(collate_observations(observations, device), mask)
        q, prior = out.q.cpu().tolist(), out.log_policy.exp().cpu().tolist()
        values = out.position_value.cpu().tolist()
        results = []
        for row, allowed in enumerate(masks):
            actions = tuple(action for action, legal in enumerate(allowed) if legal)
            probabilities = tuple(prior[row][action] for action in actions)
            total = sum(probabilities)
            results.append(PositionEvaluation(
                actions,
                tuple(probability / total for probability in probabilities),
                tuple(q[row][action] for action in actions),
                values[row],
            ))
        return tuple(results)

    def evaluate_position(
        self, observation: Observation, device: Optional[torch.device] = None,
    ) -> PositionEvaluation:
        return self.evaluate_positions((observation,), device=device)[0]
