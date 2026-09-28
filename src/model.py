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

CHECKPOINT_VERSION = 2
SUPPORTED_ARCHITECTURES = ("mlp", "wide_mlp", "suit_conv")


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        architecture: str = "mlp",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.architecture = _validate_architecture(architecture)

        if self.architecture == "mlp":
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_actions)
        elif self.architecture == "wide_mlp":
            self.fc1 = nn.Linear(state_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, num_actions)
        else:
            if state_dim != STATE_DIM or num_actions != NUM_ACTIONS:
                raise ValueError(
                    "suit_conv requires state_dim=%d and num_actions=%d"
                    % (STATE_DIM, NUM_ACTIONS)
                )
            channels = 32
            self.suit_conv1 = nn.Conv1d(4, channels, kernel_size=3, padding=1)
            self.suit_conv2 = nn.Conv1d(
                channels, channels, kernel_size=3, padding=1
            )
            self.face_fc1 = nn.Linear(channels * 3 + 8 + 13, 64)
            self.face_fc2 = nn.Linear(64, 1)
            self.draw_fc1 = nn.Linear(channels * 2 + 8, 64)
            self.draw_fc2 = nn.Linear(64, 2)
            self.register_buffer(
                "_rank_one_hot", torch.eye(13, dtype=torch.float32), persistent=False
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if self.architecture == "mlp":
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
        if self.architecture == "wide_mlp":
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            return self.fc4(x)
        return self._forward_suit_conv(x)

    def _forward_suit_conv(self, x: "torch.Tensor") -> "torch.Tensor":
        batch_size = x.shape[0]
        hand = x[:, :52].reshape(batch_size, 4, 13)
        top_discard = x[:, 52:104].reshape(batch_size, 4, 13)
        exact_joker = x[:, 104:156].reshape(batch_size, 4, 13)
        context = x[:, 156:164]

        cards_in_hand = context[:, 6].reshape(batch_size, 1, 1) * 30.0
        num_decks = context[:, 5].reshape(batch_size, 1, 1) * 6.0
        hand_per_deck = hand * cards_in_hand / num_decks.clamp_min(1.0)
        wildcard_rank = exact_joker.amax(dim=1, keepdim=True).expand(-1, 4, -1)
        card_features = torch.stack(
            (hand_per_deck, top_discard, exact_joker, wildcard_rank), dim=2
        )

        suit_rows = card_features.reshape(batch_size * 4, 4, 13)
        suit_rows = torch.cat((suit_rows, suit_rows[:, :, :1]), dim=2)
        local = torch.relu(self.suit_conv1(suit_rows))
        local = torch.relu(self.suit_conv2(local))
        local = local.transpose(1, 2).reshape(batch_size, 4, 14, 32)

        ace = 0.5 * (local[:, :, 0] + local[:, :, 13])
        local = torch.cat((ace.unsqueeze(2), local[:, :, 1:13]), dim=2)
        set_features = local.mean(dim=1, keepdim=True).expand(-1, 4, -1, -1)
        global_features = local.mean(dim=(1, 2))
        top_features = (local * top_discard.unsqueeze(-1)).sum(dim=(1, 2))

        global_faces = global_features[:, None, None, :].expand(-1, 4, 13, -1)
        context_faces = context[:, None, None, :].expand(-1, 4, 13, -1)
        rank_faces = self._rank_one_hot[None, None, :, :].expand(
            batch_size, 4, -1, -1
        )
        face_features = torch.cat(
            (local, set_features, global_faces, context_faces, rank_faces), dim=3
        )
        discard_values = self.face_fc2(
            torch.relu(self.face_fc1(face_features))
        ).reshape(batch_size, 52)

        draw_features = torch.cat((global_features, top_features, context), dim=1)
        draw_values = self.draw_fc2(torch.relu(self.draw_fc1(draw_features)))
        return torch.cat((draw_values, discard_values), dim=1)


def _validate_architecture(architecture: object) -> str:
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "unknown architecture %r (expected one of %s)"
            % (architecture, ", ".join(SUPPORTED_ARCHITECTURES))
        )
    return str(architecture)


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
        "architecture": network.architecture,
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
    expected_architecture: Optional[str] = None,
) -> Tuple["QNetwork", GameConfig, Dict[str, Any]]:
    """Load weights and config from version 1 or version 2 checkpoints."""
    if device is None:
        device = torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dict")

    ckpt_ver = payload.get("checkpoint_version")
    enc_ver = payload.get("encoding_version")
    if ckpt_ver not in (1, CHECKPOINT_VERSION):
        raise ValueError(
            "unsupported checkpoint_version %r (expected 1 or %d)"
            % (ckpt_ver, CHECKPOINT_VERSION)
        )
    if enc_ver != ENCODING_VERSION:
        raise ValueError(
            "incompatible encoding_version %r (expected %d)"
            % (enc_ver, ENCODING_VERSION)
        )

    if "game_config" not in payload or "model_state_dict" not in payload:
        raise ValueError("checkpoint missing game_config or model_state_dict")

    if ckpt_ver == 1:
        architecture = "mlp"
    else:
        if "architecture" not in payload:
            raise ValueError("version 2 checkpoint missing architecture")
        architecture = _validate_architecture(payload["architecture"])
    if expected_architecture is not None:
        _validate_architecture(expected_architecture)
        if architecture != expected_architecture:
            raise ValueError(
                "checkpoint architecture %r does not match expected %r"
                % (architecture, expected_architecture)
            )

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

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("checkpoint meta must be a dict")

    if network is None:
        network = QNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            architecture=architecture,
        )
    elif (
        network.state_dim != state_dim
        or network.num_actions != num_actions
        or network.architecture != architecture
    ):
        raise ValueError(
            "checkpoint model %s/%d/%d does not match network %s/%d/%d"
            % (
                architecture,
                state_dim,
                num_actions,
                network.architecture,
                network.state_dim,
                network.num_actions,
            )
        )
    network.load_state_dict(payload["model_state_dict"])
    network.to(device)
    network.eval()

    return network, config, meta
