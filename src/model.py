"""Q-network and checkpoint helpers for episodic Monte Carlo regression.

The network is a 128/64 ReLU MLP mapping encoded observations to 54 action
values. Training targets are discounted full-episode returns, not bootstrapped
DQN targets. No target network.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple
from pathlib import Path
import random

import torch
import torch.nn as nn

from src.environment import (
    ENCODING_VERSION,
    NUM_ACTIONS,
    STATE_DIM,
    GameConfig,
    Observation,
    encode_observation,
    legal_action_mask,
)

CHECKPOINT_VERSION = 1

class QNetwork(nn.Module):
    """Small MLP: state_dim -> 128 -> 64 -> num_actions."""

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def masked_q_values(
    network: "QNetwork",
    state: Sequence[float],
    mask: Sequence[bool],
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    if device is None:
        device = next(network.parameters()).device
    tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q = network(tensor)[0]
    illegal = [i for i, ok in enumerate(mask) if not ok]
    if illegal:
        q = q.clone()
        q[illegal] = -torch.inf
    return q


def select_action(
    network: "QNetwork",
    obs: Observation,
    epsilon: float,
    rng: random.Random,
    device: Optional["torch.device"] = None,
) -> int:
    """Epsilon-greedy over the legal-action mask only."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    mask = legal_action_mask(obs)
    legal = [i for i, ok in enumerate(mask) if ok]
    if not legal:
        raise ValueError("no legal actions")
    if rng.random() < epsilon:
        return rng.choice(legal)
    return select_greedy_action(network, obs, device=device)


def select_greedy_action(
    network: "QNetwork",
    obs: Observation,
    device: Optional["torch.device"] = None,
) -> int:
    mask = legal_action_mask(obs)
    legal = [i for i, ok in enumerate(mask) if ok]
    if not legal:
        raise ValueError("no legal actions")
    state = encode_observation(obs)
    q = masked_q_values(network, state, mask, device=device)
    return int(torch.argmax(q).item())


def select_random_legal(obs: Observation, rng: random.Random) -> int:
    mask = legal_action_mask(obs)
    legal = [i for i, ok in enumerate(mask) if ok]
    if not legal:
        raise ValueError("no legal actions")
    return rng.choice(legal)


def build_checkpoint(
    network: "QNetwork",
    config: GameConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "encoding_version": ENCODING_VERSION,
        "state_dim": network.state_dim,
        "num_actions": network.num_actions,
        "game_config": config.to_dict(),
        "model_state_dict": network.state_dict(),
    }
    if meta:
        payload["meta"] = dict(meta)
    return payload


def save_checkpoint(
    path: str,
    network: "QNetwork",
    config: GameConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    payload = build_checkpoint(network, config, meta=meta)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    device: Optional["torch.device"] = None,
    expected_config: Optional[GameConfig] = None,
    network: Optional["QNetwork"] = None,
) -> Tuple["QNetwork", GameConfig, Dict[str, Any]]:
    """Load network weights and game config. Rejects incompatible versions."""
    if device is None:
        device = torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dict")

    ckpt_ver = payload.get("checkpoint_version")
    enc_ver = payload.get("encoding_version")
    if ckpt_ver != CHECKPOINT_VERSION:
        raise ValueError(
            "unsupported checkpoint_version %r (expected %d)"
            % (ckpt_ver, CHECKPOINT_VERSION)
        )
    if enc_ver != ENCODING_VERSION:
        raise ValueError(
            "incompatible encoding_version %r (expected %d)"
            % (enc_ver, ENCODING_VERSION)
        )

    if "game_config" not in payload or "model_state_dict" not in payload:
        raise ValueError("checkpoint missing game_config or model_state_dict")

    config = GameConfig.from_dict(payload["game_config"])
    if expected_config is not None and config.to_dict() != expected_config.to_dict():
        raise ValueError(
            "checkpoint game_config %s does not match expected %s"
            % (config.to_dict(), expected_config.to_dict())
        )

    state_dim = payload.get("state_dim")
    num_actions = payload.get("num_actions")
    if state_dim != STATE_DIM or num_actions != NUM_ACTIONS:
        raise ValueError(
            "checkpoint dims state=%r actions=%r incompatible with state=%d actions=%d"
            % (state_dim, num_actions, STATE_DIM, NUM_ACTIONS)
        )

    if network is None:
        network = QNetwork(state_dim=state_dim, num_actions=num_actions)
    network.load_state_dict(payload["model_state_dict"])
    network.to(device)
    network.eval()

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("checkpoint meta must be a dict")
    return network, config, meta
