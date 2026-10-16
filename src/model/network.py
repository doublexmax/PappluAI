from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.checkpoints.tensor import validate_model_state
from src.game.environment import (
    ENCODING_VERSION, NUM_ACTIONS, NUM_FACES, GameConfig, Observation, legal_action_mask,
)
from src.game.information import EventKind
from src.model.encoding import (
    CONTEXT_FEATURES, EVENT_FEATURES, PLAYER_FEATURES,
    ObservationBatch, collate_observations,
)
from src.model.spec import ARCHITECTURE, CHECKPOINT_VERSION, ModelSpec


class QNetwork(nn.Module):
    architecture = ARCHITECTURE
    num_actions = NUM_ACTIONS

    def __init__(self, spec: ModelSpec = ModelSpec()) -> None:
        super().__init__()
        self.spec = spec
        width = spec.width
        self.suit_conv1 = nn.Conv1d(5, width, kernel_size=3, padding=1)
        self.suit_conv2 = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.card_attention = nn.MultiheadAttention(width, spec.heads, batch_first=True)
        self.card_norm = nn.LayerNorm(width)
        self.history_proj = nn.Linear(5, width)
        self.context_features = nn.Linear(CONTEXT_FEATURES, width)
        self.player_features = nn.Linear(PLAYER_FEATURES, width)
        self.event_summary = nn.Linear(EVENT_FEATURES, width)
        self.turn_relations = nn.Linear(width * 3, width)
        self.register_buffer("_rank_one_hot", torch.eye(13, dtype=torch.float32), persistent=False)
        face_dim = width * 4 + 5 + 13
        draw_dim = width * 4 + 5
        self.draw_head = nn.Sequential(nn.Linear(draw_dim, width), nn.ReLU(), nn.Linear(width, 2))
        with torch.no_grad():
            self.draw_head[-1].bias.copy_(torch.tensor([0.1, -0.1]))
        self.score = nn.Sequential(nn.Linear(face_dim, width), nn.ReLU(), nn.Linear(width, 1))

    def _history_channels(self, batch: ObservationBatch) -> torch.Tensor:
        size = batch.cards.shape[0]
        device = batch.cards.device
        steps = batch.event_kind.shape[1]
        extra = torch.zeros((size, NUM_FACES + 1, 5), dtype=torch.float32, device=device)
        extra[:, :NUM_FACES, 0] = batch.initial_hand - batch.cards[:, :, 0]
        if steps > 0:
            pos = torch.arange(1, steps + 1, dtype=torch.float32, device=device).unsqueeze(0)
            denom = batch.history_lengths.to(device=device, dtype=torch.float32).clamp_min(1.0).unsqueeze(1)
            recency = (0.5 + 0.5 * (pos / denom)) * batch.history_mask.to(torch.float32) / denom
            own = (batch.event_actor == 0) & batch.history_mask
            opp_next = (batch.event_actor == 1) & batch.history_mask
            opp_other = (batch.event_actor > 1) & batch.history_mask
            is_take = batch.event_kind == int(EventKind.TAKE_DISCARD)
            is_disc = batch.event_kind == int(EventKind.DISCARD)
            is_draw = batch.event_kind == int(EventKind.DRAW_STOCK)
            face_idx = batch.event_face.clamp(0, NUM_FACES)
            top_idx = batch.offered_top.clamp(0, NUM_FACES)
            for channel, mask in enumerate((
                own, opp_next & is_take, opp_next & is_disc,
                opp_other & is_take, opp_other & is_disc,
            )):
                extra[:, :, channel].scatter_add_(1, face_idx, recency * mask.to(torch.float32))
            for channel, opponent in ((2, opp_next), (4, opp_other)):
                extra[:, :, channel].scatter_add_(
                    1, top_idx, recency * (opponent & is_draw).to(torch.float32),
                )
        return extra[:, :NUM_FACES, :]

    def forward(self, batch: ObservationBatch) -> torch.Tensor:
        device = batch.cards.device
        width = self.spec.width
        size = batch.cards.shape[0]
        suit_rows = batch.cards.reshape(size * 4, 13, 5).transpose(1, 2)
        suit_rows = torch.cat((suit_rows, suit_rows[:, :, :1]), dim=2)
        local = torch.relu(self.suit_conv1(suit_rows))
        local = torch.relu(self.suit_conv2(local))
        ace = 0.5 * (local[:, :, 0] + local[:, :, 13])
        local_flat = torch.cat((ace[:, :, None], local[:, :, 1:13]), dim=2).transpose(1, 2)
        local_flat = local_flat.reshape(size, NUM_FACES, width)
        interactions, _ = self.card_attention(local_flat, local_flat, local_flat, need_weights=False)
        local_flat = self.card_norm(local_flat + interactions)
        local = local_flat.reshape(size, 4, 13, width)
        set_features = local.mean(dim=1, keepdim=True).expand(-1, 4, -1, -1).reshape(size, NUM_FACES, width)
        global_cards = local_flat.mean(dim=1)

        hist = self.history_proj(self._history_channels(batch))
        players = self.player_features(
            batch.players.masked_fill(~batch.player_mask[:, :, None], 0.0),
        )
        next_players = players.gather(1, batch.next_player[:, :, None].expand(-1, -1, width))
        previous_players = players.gather(1, batch.previous_player[:, :, None].expand(-1, -1, width))
        rel = torch.relu(self.turn_relations(
            torch.cat((players, next_players, previous_players), dim=-1),
        ))
        events = batch.event_features.masked_fill(~batch.history_mask[:, :, None], 0.0)
        pos = torch.arange(1, events.shape[1] + 1, dtype=torch.float32, device=device).unsqueeze(0)
        lengths = batch.history_lengths.to(device=device, dtype=torch.float32).clamp_min(1.0).unsqueeze(1)
        weights = (pos / lengths) * batch.history_mask.to(torch.float32)
        ev_ctx = self.event_summary(
            (events * weights[:, :, None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0),
        )
        table_ctx = self.context_features(batch.context) + rel[:, 0, :] + 0.1 * ev_ctx

        top_idx = batch.top.clamp_max(NUM_FACES - 1)
        has_top = (batch.top < NUM_FACES).to(torch.float32).unsqueeze(-1)
        top_local = local_flat.gather(1, top_idx[:, None, None].expand(-1, 1, width)).squeeze(1) * has_top
        top_set = set_features.gather(1, top_idx[:, None, None].expand(-1, 1, width)).squeeze(1) * has_top
        top_raw = batch.cards.gather(1, top_idx[:, None, None].expand(-1, 1, 5)).squeeze(1) * has_top
        draw_values = self.draw_head(torch.cat((global_cards, top_local, top_set, top_raw, table_ctx + hist.mean(dim=1)), dim=-1))

        ranks = self._rank_one_hot.repeat(4, 1).unsqueeze(0).expand(size, -1, -1)
        face_ctx = table_ctx[:, None, :].expand(-1, NUM_FACES, -1) + 0.1 * hist
        per_face = torch.cat((
            local_flat,
            set_features,
            global_cards[:, None, :].expand(-1, NUM_FACES, -1),
            face_ctx,
            batch.cards,
            ranks,
        ), dim=-1)
        discard_values = self.score(per_face).squeeze(-1)
        return torch.cat((draw_values, discard_values), dim=1)


def masked_q_values(
    network: QNetwork,
    obs: Observation,
    mask: Sequence[bool],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = next(network.parameters()).device
    with torch.no_grad():
        values = network(collate_observations([obs], device))[0]
    return values.masked_fill(~torch.tensor(mask, dtype=torch.bool, device=device), -torch.inf)


def select_action(
    network: QNetwork,
    obs: Observation,
    epsilon: float,
    rng: random.Random,
    device: Optional[torch.device] = None,
) -> int:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    legal = [i for i, allowed in enumerate(legal_action_mask(obs)) if allowed]
    if not legal:
        raise ValueError("no legal actions")
    if rng.random() < epsilon:
        return rng.choice(legal)
    return select_greedy_action(network, obs, device=device)


def select_greedy_action(
    network: QNetwork,
    obs: Observation,
    device: Optional[torch.device] = None,
) -> int:
    mask = legal_action_mask(obs)
    if not any(mask):
        raise ValueError("no legal actions")
    return int(masked_q_values(network, obs, mask, device=device).argmax().item())


def select_random_legal(obs: Observation, rng: random.Random) -> int:
    legal = [i for i, allowed in enumerate(legal_action_mask(obs)) if allowed]
    if not legal:
        raise ValueError("no legal actions")
    return rng.choice(legal)


def build_checkpoint(
    network: QNetwork,
    config: GameConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "encoding_version": ENCODING_VERSION,
        "architecture": ARCHITECTURE,
        "model_spec": network.spec.to_dict(),
        "num_actions": NUM_ACTIONS,
        "game_config": config.to_dict(),
        "model_state_dict": network.state_dict(),
    }
    if meta:
        payload["meta"] = dict(meta)
    return payload


def save_checkpoint(
    path: str,
    network: QNetwork,
    config: GameConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint(network, config, meta), path)


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
    expected_config: Optional[GameConfig] = None,
    network: Optional[QNetwork] = None,
) -> Tuple[QNetwork, GameConfig, Dict[str, Any]]:
    if device is None:
        device = torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dict")
    if type(payload.get("checkpoint_version")) is not int or payload["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint_version %r; fresh dynamic weights are required"
                         % payload.get("checkpoint_version"))
    if type(payload.get("encoding_version")) is not int or payload["encoding_version"] != ENCODING_VERSION:
        raise ValueError("incompatible encoding_version; complete information history is required")
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported architecture; expected dynamic_table")
    required = {
        "checkpoint_version", "encoding_version", "architecture", "model_spec",
        "num_actions", "game_config", "model_state_dict",
    }
    if set(payload) not in (required, required | {"meta"}):
        raise ValueError("checkpoint fields do not match the current model schema")
    if type(payload["num_actions"]) is not int or payload["num_actions"] != NUM_ACTIONS:
        raise ValueError("checkpoint uses an incompatible model input/action schema")
    spec = ModelSpec.from_dict(payload.get("model_spec"))
    config = GameConfig.from_dict(payload.get("game_config"))
    if expected_config is not None and config != expected_config:
        raise ValueError("checkpoint game_config does not match expected game_config")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("checkpoint meta must be a dict")
    if network is None:
        network = QNetwork(spec)
    elif network.spec != spec:
        raise ValueError("checkpoint model_spec does not match network")
    model_state = validate_model_state(
        payload.get("model_state_dict"), network.state_dict(), "checkpoint model_state_dict",
    )
    network.load_state_dict(model_state)
    network.to(device)
    network.eval()
    return network, config, meta
