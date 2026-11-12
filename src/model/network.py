from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.checkpoints.tensor import validate_model_state
from src.game.environment import ENCODING_VERSION, NUM_ACTIONS, NUM_FACES, GameConfig, Observation
from src.model.encoding import (
    CARD_CHANNELS, CONTEXT_CHANNELS, FEATURE_SIZE, HISTORY_CHANNELS, LAYOUT,
    NUM_SEAT_SLOTS, SEAT_CHANNELS, Features, encode, legal_actions,
)
from src.model.spec import ARCHITECTURE, CHECKPOINT_VERSION, ModelSpec


@dataclass(frozen=True)
class ObservationBatch:
    card: Tensor
    history: Tensor
    seat: Tensor
    context: Tensor
    legal: Tensor
    top: Tensor


def collate(features: Sequence[Features], device: torch.device) -> ObservationBatch:
    if not features:
        raise ValueError("model input needs at least one feature record")
    size = len(features)
    data = torch.frombuffer(
        bytearray(b"".join(feature.data for feature in features)), dtype=torch.float32,
    ).reshape(size, FEATURE_SIZE).to(device)
    fields = {
        field.name: data[:, field.offset:field.offset + field.size].reshape(size, *field.shape)
        for field in LAYOUT
    }
    fields["top"] = fields["top"].squeeze(-1).to(torch.long)
    return ObservationBatch(**fields)


def collate_observations(
    observations: Sequence[Observation], device: torch.device,
) -> ObservationBatch:
    return collate([encode(obs) for obs in observations], device)


@dataclass(frozen=True)
class PolicyActivations:
    q: Tensor
    cards: Tensor
    table: Tensor
    faces: Tensor
    draw: Tensor


@dataclass(frozen=True)
class NetworkOutputs:
    q: Tensor
    win_logit: Tensor
    penalty: Tensor
    belief: Tensor
    after: Tensor


class AxialBlock(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        width = spec.width
        self.heads = spec.heads
        self.rank_norm = nn.LayerNorm(width)
        self.rank_conv = nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width)
        self.rank_in = nn.Linear(width, width)
        self.rank_out = nn.Linear(width, width)
        self.suit_norm = nn.LayerNorm(width)
        self.suit_in = nn.Linear(width, width)
        self.suit_out = nn.Linear(width, width)
        self.card_norm = nn.LayerNorm(width)
        self.table_norm = nn.LayerNorm(width)
        self.table_query = nn.Linear(width, width)
        self.table_in = nn.Linear(2 * width, width)
        self.table_out = nn.Linear(width, width)
        self.film = nn.Linear(width, 2 * width)

    def forward(self, cards: Tensor, table: Tensor) -> tuple[Tensor, Tensor]:
        size, _, _, width = cards.shape
        normalized = self.rank_norm(cards)
        ranks = torch.cat((normalized, normalized[:, :, :1]), dim=2)
        ranks = self.rank_conv(ranks.reshape(size * 4, 14, width).transpose(1, 2))
        ranks = ranks.transpose(1, 2).reshape(size, 4, 14, width)
        ranks = torch.cat(((ranks[:, :, :1] + ranks[:, :, 13:]) * 0.5, ranks[:, :, 1:13]), dim=2)
        cards = cards + self.rank_out(torch.relu(self.rank_in(ranks)))
        suits = self.suit_norm(cards).mean(dim=1)
        cards = cards + self.suit_out(torch.relu(self.suit_in(suits)))[:, None]
        head_size = width // self.heads
        keys = self.card_norm(cards).reshape(size, NUM_FACES, self.heads, head_size)
        normalized_table = self.table_norm(table)
        query = self.table_query(normalized_table).reshape(size, self.heads, head_size)
        weights = torch.softmax(torch.einsum("bhd,bfhd->bhf", query, keys) / math.sqrt(head_size), dim=-1)
        pooled = torch.einsum("bhf,bfhd->bhd", weights, keys).reshape(size, width)
        table = table + self.table_out(torch.relu(self.table_in(torch.cat((normalized_table, pooled), dim=-1))))
        gamma, beta = self.film(table).chunk(2, dim=-1)
        cards = cards + (
            keys.reshape(size, NUM_FACES, width) * gamma[:, None] + beta[:, None]
        ).reshape(size, 4, 13, width)
        return cards, table


class PolicyNet(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        width = spec.width
        self.card_embed = nn.Linear(len(CARD_CHANNELS) + NUM_SEAT_SLOTS * len(HISTORY_CHANNELS), width)
        self.table_embed = nn.Linear(len(CONTEXT_CHANNELS) + NUM_SEAT_SLOTS * len(SEAT_CHANNELS), width)
        self.blocks = nn.ModuleList(AxialBlock(spec) for _ in range(spec.blocks))
        self.card_final = nn.LayerNorm(width)
        self.table_final = nn.LayerNorm(width)
        self.face_local = nn.Linear(width, width)
        self.face_set = nn.Linear(width, width, bias=False)
        self.face_table = nn.Linear(width, width, bias=False)
        self.face_out = nn.Linear(width, 1)
        self.draw_table = nn.Linear(width, width)
        self.draw_top = nn.Linear(width, width, bias=False)
        self.draw_set = nn.Linear(width, width, bias=False)
        self.draw_out = nn.Linear(width, 2)
        self.value_hidden = nn.Linear(width, width)
        self.value_out = nn.Linear(width, 1)

    def forward(self, batch: ObservationBatch) -> PolicyActivations:
        size = batch.card.shape[0]
        cards = self.card_embed(torch.cat((batch.card, batch.history.flatten(2)), dim=-1))
        width = cards.shape[-1]
        cards = cards.reshape(size, 4, 13, width)
        table = self.table_embed(torch.cat((batch.context, batch.seat.flatten(1)), dim=-1))
        for block in self.blocks:
            cards, table = block(cards, table)
        cards, table = self.card_final(cards), self.table_final(table)
        sets = cards.mean(dim=1)
        faces = torch.relu(
            self.face_local(cards) + self.face_set(sets)[:, None] + self.face_table(table)[:, None, None],
        )
        index = batch.top.clamp_max(NUM_FACES - 1)
        rows = torch.arange(size, device=cards.device)
        has_top = (batch.top < NUM_FACES).to(cards.dtype)[:, None]
        top_card = cards.reshape(size, NUM_FACES, width)[rows, index] * has_top
        top_set = sets[rows, index % 13] * has_top
        draw = torch.relu(self.draw_table(table) + self.draw_top(top_card) + self.draw_set(top_set))
        advantage = torch.cat((self.draw_out(draw), self.face_out(faces).reshape(size, NUM_FACES)), dim=1)
        mean = (advantage * batch.legal).sum(dim=1, keepdim=True) / batch.legal.sum(
            dim=1, keepdim=True,
        ).clamp_min(1.0)
        q = self.value_out(torch.relu(self.value_hidden(table))) + advantage - mean
        return PolicyActivations(q, cards, table, faces, draw)


class AuxHeads(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        width = spec.width
        self.face_win = nn.Linear(width, 1)
        self.face_penalty = nn.Linear(width, 1)
        self.draw_win = nn.Linear(width, 2)
        self.draw_penalty = nn.Linear(width, 2)
        self.belief = nn.Linear(width, NUM_SEAT_SLOTS)
        self.after_hidden = nn.Linear(width, width)
        self.after_out = nn.Linear(width, 1)

    def forward(self, activations: PolicyActivations) -> NetworkOutputs:
        size = activations.table.shape[0]
        win = torch.cat((
            self.draw_win(activations.draw), self.face_win(activations.faces).reshape(size, NUM_FACES),
        ), dim=1)
        penalty = torch.cat((
            self.draw_penalty(activations.draw), self.face_penalty(activations.faces).reshape(size, NUM_FACES),
        ), dim=1)
        belief = self.belief(activations.cards.flatten(1, 2)).transpose(1, 2)
        after = self.after_out(torch.relu(self.after_hidden(activations.table))).squeeze(-1)
        return NetworkOutputs(activations.q, win, penalty, belief, after)


class QNetwork(nn.Module):
    architecture = ARCHITECTURE
    num_actions = NUM_ACTIONS

    def __init__(self, spec: ModelSpec = ModelSpec()) -> None:
        super().__init__()
        self.spec = spec
        self.policy = PolicyNet(spec)
        self.aux = AuxHeads(spec)

    def forward(self, batch: ObservationBatch) -> Tensor:
        return self.policy(batch).q

    def outputs(self, batch: ObservationBatch) -> NetworkOutputs:
        return self.aux(self.policy(batch))


@torch.inference_mode()
def greedy_actions(network: QNetwork, batch: ObservationBatch) -> Tensor:
    if not bool(batch.legal.any(dim=1).all()):
        raise ValueError("no legal actions")
    return network(batch).masked_fill(batch.legal == 0, -torch.inf).argmax(dim=1)


def masked_q_values(
    network: QNetwork, obs: Observation, mask: Sequence[bool],
    device: Optional[torch.device] = None,
) -> Tensor:
    if device is None:
        device = next(network.parameters()).device
    with torch.no_grad():
        values = network(collate_observations([obs], device))[0]
    return values.masked_fill(~torch.tensor(mask, dtype=torch.bool, device=device), -torch.inf)


def select_action(
    network: QNetwork, obs: Observation, epsilon: float, rng: random.Random,
    device: Optional[torch.device] = None,
) -> int:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    features = encode(obs)
    legal = legal_actions(features)
    if not legal:
        raise ValueError("no legal actions")
    if rng.random() < epsilon:
        return rng.choice(legal)
    if device is None:
        device = next(network.parameters()).device
    return int(greedy_actions(network, collate([features], device))[0].item())


def select_greedy_action(
    network: QNetwork, obs: Observation, device: Optional[torch.device] = None,
) -> int:
    if device is None:
        device = next(network.parameters()).device
    return int(greedy_actions(network, collate_observations([obs], device))[0].item())


def select_random_legal(obs: Observation, rng: random.Random) -> int:
    legal = legal_actions(encode(obs))
    if not legal:
        raise ValueError("no legal actions")
    return rng.choice(legal)


def build_checkpoint(
    network: QNetwork, config: GameConfig, meta: Optional[Mapping[str, Any]] = None,
) -> dict:
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
    path: str, network: QNetwork, config: GameConfig, meta: Optional[Mapping[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint(network, config, meta), path)


def load_checkpoint(
    path: str, device: Optional[torch.device] = None, expected_config: Optional[GameConfig] = None,
    network: Optional[QNetwork] = None,
) -> tuple[QNetwork, GameConfig, dict]:
    if device is None:
        device = torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dict")
    if type(payload.get("checkpoint_version")) is not int or payload["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint_version %r; train fresh axial_table v4 weights"
                         % payload.get("checkpoint_version"))
    if type(payload.get("encoding_version")) is not int or payload["encoding_version"] != ENCODING_VERSION:
        raise ValueError("incompatible encoding_version; complete information history is required")
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported architecture; expected axial_table v4")
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
