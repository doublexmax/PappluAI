from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        self.rank_embedding = nn.Embedding(13, width)
        self.suit_embedding = nn.Embedding(4, width)
        self.face_embedding = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.card_features = nn.Linear(5, width)
        self.card_conv = nn.Conv1d(width, width, 3, padding=1)
        self.card_attention = nn.MultiheadAttention(width, spec.heads, batch_first=True)
        self.card_norm = nn.LayerNorm(width)
        self.event_embedding = nn.Embedding(len(EventKind), width)
        self.event_features = nn.Linear(EVENT_FEATURES, width)
        self.event_cards = nn.Linear(width * 2, width)
        self.initial_cards = nn.Linear(width, width)
        self.history_encoder = nn.GRU(width, width, batch_first=True)
        self.context_features = nn.Linear(CONTEXT_FEATURES, width)
        self.player_features = nn.Linear(PLAYER_FEATURES, width)
        self.player_history_attention = nn.MultiheadAttention(width, spec.heads, batch_first=True)
        self.turn_relations = nn.Linear(width * 3, width)
        self.player_attention = nn.MultiheadAttention(width, spec.heads, batch_first=True)
        self.player_norm = nn.LayerNorm(width)
        self.action_embedding = nn.Embedding(3, width)
        self.action_query = nn.Linear(width * 3, width)
        self.action_player_attention = nn.MultiheadAttention(width, spec.heads, batch_first=True)
        self.action_card_attention = nn.MultiheadAttention(width, spec.heads, batch_first=True)
        self.score = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, batch: ObservationBatch) -> torch.Tensor:
        device = batch.cards.device
        width = self.spec.width
        size = batch.cards.shape[0]
        faces = torch.arange(NUM_FACES, device=device)
        base_cards = self.face_embedding(
            self.rank_embedding(faces % 13) + self.suit_embedding(faces // 13)
        )
        visible_faces = torch.cat((base_cards, base_cards.new_zeros(1, width)), dim=0)
        cards = base_cards.unsqueeze(0) + self.card_features(batch.cards)
        suit_rows = cards.reshape(size * 4, 13, width).transpose(1, 2)
        suit_rows = torch.cat((suit_rows, suit_rows[:, :, :1]), dim=2)
        local = torch.relu(self.card_conv(suit_rows)).transpose(1, 2)
        ace = 0.5 * (local[:, 0] + local[:, 13])
        local = torch.cat((ace[:, None, :], local[:, 1:13]), dim=1).reshape(size, NUM_FACES, width)
        cards = cards + local
        interactions, _ = self.card_attention(cards, cards, cards, need_weights=False)
        cards = self.card_norm(cards + interactions)

        event_cards = torch.cat((
            visible_faces[batch.event_face], visible_faces[batch.offered_top],
        ), dim=-1)
        events = (
            self.event_embedding(batch.event_kind)
            + self.event_features(batch.event_features)
            + self.event_cards(event_cards)
        )
        initial = self.initial_cards(batch.initial_hand @ base_cards / 30.0)
        events = events + (
            torch.arange(events.shape[1], device=device)[None, :, None] == 0
        ) * initial[:, None, :]
        packed = pack_padded_sequence(
            events, batch.history_lengths, batch_first=True, enforce_sorted=False,
        )
        packed_history, final = self.history_encoder(packed)
        history, _ = pad_packed_sequence(
            packed_history, batch_first=True, total_length=events.shape[1],
        )
        context = self.context_features(batch.context) + final[0]
        players = self.player_features(batch.players) + context[:, None, :]
        player_history, _ = self.player_history_attention(
            players, history, history, key_padding_mask=~batch.history_mask,
            need_weights=False,
        )
        positions = torch.arange(events.shape[1], device=device)[None, :, None]
        actor_matches = (
            batch.event_actor[:, :, None]
            == torch.arange(players.shape[1], device=device)[None, None, :]
        ) & batch.history_mask[:, :, None]
        last_positions = positions.expand(size, -1, players.shape[1]).masked_fill(
            ~actor_matches, -1,
        ).amax(dim=1)
        last_history = history.gather(
            1, last_positions.clamp_min(0)[:, :, None].expand(-1, -1, width),
        ) * (last_positions >= 0)[:, :, None]
        players = players + player_history + last_history
        next_players = players.gather(1, batch.next_player[:, :, None].expand(-1, -1, width))
        previous_players = players.gather(1, batch.previous_player[:, :, None].expand(-1, -1, width))
        players = players + torch.relu(self.turn_relations(
            torch.cat((players, next_players, previous_players), dim=-1),
        ))
        relations, _ = self.player_attention(
            players, players, players, key_padding_mask=~batch.player_mask,
            need_weights=False,
        )
        players = self.player_norm(players + relations)

        top_cards = cards.gather(
            1, batch.top.clamp_max(NUM_FACES - 1)[:, None, None].expand(-1, 1, width),
        ) * (batch.top < NUM_FACES)[:, None, None]
        action_cards = torch.cat((cards.new_zeros(size, 1, width), top_cards, cards), dim=1)
        action_kinds = torch.tensor([0, 1] + [2] * NUM_FACES, device=device)
        actions = self.action_query(torch.cat((
            action_cards,
            self.action_embedding(action_kinds)[None, :, :].expand(size, -1, -1),
            context[:, None, :].expand(-1, NUM_ACTIONS, -1),
        ), dim=-1))
        player_context, _ = self.action_player_attention(
            actions, players, players, key_padding_mask=~batch.player_mask,
            need_weights=False,
        )
        card_context, _ = self.action_card_attention(actions, cards, cards, need_weights=False)
        return self.score(torch.cat((actions, player_context, card_context), dim=-1)).squeeze(-1)


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
