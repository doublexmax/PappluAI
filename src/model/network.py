from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.checkpoints.tensor import validate_model_state
from src.game.environment import ENCODING_VERSION, NUM_ACTIONS, NUM_FACES, GameConfig, Observation
from src.model.encoding import (
    CARD_CHANNELS, CONTEXT_CHANNELS, FEATURE_SIZE, LAYOUT,
    NUM_SEAT_SLOTS, SEAT_CHANNELS, Features, encode, legal_actions,
)
from src.model.spec import ARCHITECTURE, CHECKPOINT_VERSION, ModelSpec


@dataclass(frozen=True)
class ObservationBatch:
    card: Tensor
    seat: Tensor
    context: Tensor
    legal: Tensor
    top: Tensor


def collate(features: Sequence[Features], device: torch.device) -> ObservationBatch:
    if not features:
        raise ValueError("model input needs at least one feature record")
    size = len(features)
    data = torch.frombuffer(
        bytearray().join(feature.data for feature in features), dtype=torch.float32,
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


def tensor_meld_channels(card: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
    size = card.shape[0]
    c = card.reshape(size, 4, 13, len(CARD_CHANNELS))
    decks = (context[:, 8] * 6.0).round().clamp_min(1.0)[:, None, None]
    hand_sz = (context[:, 9] * 30.0).round().clamp_min(1.0)
    copies = (c[:, :, :, 0] * decks).round()
    present = (copies >= 1.0).to(card.dtype)
    triple = (copies >= 3.0).to(card.dtype)
    is_joker, is_wild, pts = c[:, :, :, 6], c[:, :, :, 7], c[:, :, :, 8]
    joker_avail = (copies * is_joker).sum(dim=(1, 2), keepdim=True)
    wild_avail = (copies * is_wild).sum(dim=(1, 2), keepdim=True)
    pres14 = torch.cat((present, present[:, :, :1]), dim=2)
    jok14 = torch.cat((is_joker, is_joker[:, :, :1]), dim=2)
    wild14 = torch.cat((is_wild, is_wild[:, :, :1]), dim=2)
    nat_win = pres14[:, :, :12] * pres14[:, :, 1:13] * pres14[:, :, 2:14]
    non_jok14 = pres14 * (1.0 - jok14)
    miss_pure = 3.0 - (non_jok14[:, :, :12] + non_jok14[:, :, 1:13] + non_jok14[:, :, 2:14])
    pure_win = (miss_pure <= joker_avail).to(card.dtype)
    non_wild14 = pres14 * (1.0 - wild14)
    miss_any = 3.0 - (non_wild14[:, :, :12] + non_wild14[:, :, 1:13] + non_wild14[:, :, 2:14])
    any_run_win = (miss_any <= wild_avail).to(card.dtype)

    def scatter_runs(
        win: Tensor, mask14: Tensor,
        flex_avail: Optional[Tensor] = None, flex14: Optional[Tensor] = None,
    ) -> Tensor:
        active = torch.zeros((size, 4, 14), dtype=card.dtype, device=card.device)
        active[:, :, :12] += win
        active[:, :, 1:13] += win
        active[:, :, 2:14] += win
        out14 = ((active > 0.0) & (mask14 > 0.0)).to(card.dtype)
        if flex_avail is not None and flex14 is not None:
            any_win = (win.sum(dim=(1, 2), keepdim=True) > 0.0) & (flex_avail > 0.0)
            flex_hit = (any_win & (pres14 > 0.0) & (flex14 > 0.0)).to(card.dtype)
            out14 = torch.maximum(out14, flex_hit)
        ace = torch.maximum(out14[:, :, :1], out14[:, :, 13:])
        return torch.cat((ace, out14[:, :, 1:13]), dim=2)

    natural = scatter_runs(nat_win, pres14)
    pure = torch.maximum(scatter_runs(pure_win, non_jok14, joker_avail, jok14), triple)
    any_runs = scatter_runs(any_run_win, non_wild14, wild_avail, wild14)
    non_wild = present * (1.0 - is_wild)
    set_non_wild_suits = non_wild.sum(dim=1, keepdim=True)
    miss_set = 3.0 - set_non_wild_suits.clamp_max(3.0)
    valid_set_rank = (miss_set <= wild_avail).to(card.dtype)
    other_non_wild = (set_non_wild_suits - non_wild).clamp_min(0.0)
    miss_set_with_s = 2.0 - other_non_wild.clamp_max(2.0)
    set_member_non_wild = non_wild * (miss_set_with_s <= wild_avail).to(card.dtype)
    any_valid_set = (valid_set_rank.sum(dim=2, keepdim=True) > 0.0) & (wild_avail > 0.0)
    set_member_wild = (any_valid_set & (present > 0.0) & (is_wild > 0.0)).to(card.dtype)
    set_member = torch.maximum(set_member_non_wild, set_member_wild)
    any_meld = torch.maximum(torch.maximum(any_runs, pure), set_member)
    left_adj = torch.cat((present[:, :, 12:13], present[:, :, :12]), dim=2)
    right_adj = torch.cat((present[:, :, 1:13], present[:, :, :1]), dim=2)
    adjacent = (left_adj + right_adj) * 0.5
    set_mates = (present.sum(dim=1, keepdim=True) - present) / 3.0
    unavoidable_dw = (copies * pts * 10.0 * (1.0 - any_meld)).sum(dim=(1, 2)) / (10.0 * hand_sz)
    return torch.stack((adjacent, set_mates, natural, pure, any_meld), dim=-1), unavoidable_dw[:, None]


class CardBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.card_norm = nn.LayerNorm(width)
        self.table_norm = nn.LayerNorm(width)
        self.w_seq = nn.Linear(width, width * 3)
        self.w_rank_set = nn.Linear(width, width, bias=False)
        self.w_suit_run = nn.Linear(width, width, bias=False)
        self.w_table = nn.Linear(width, width, bias=False)
        self.card_out = nn.Linear(width, width)
        self.table_fc1 = nn.Linear(width * 3, width)
        self.table_fc2 = nn.Linear(width, width)

    def forward(self, cards: Tensor, table: Tensor) -> tuple[Tensor, Tensor]:
        c, t = self.card_norm(cards), self.table_norm(table)
        c14 = torch.cat((c, c[:, :, :1, :]), dim=2)
        u_self, u_left, u_right = self.w_seq(c14).chunk(3, dim=-1)
        zeros = torch.zeros_like(u_self[:, :, :1, :])
        left14 = torch.cat((zeros, u_left[:, :, :-1, :]), dim=2)
        right14 = torch.cat((u_right[:, :, 1:, :], zeros), dim=2)
        seq14 = u_self + left14 + right14
        seq = torch.cat((
            (seq14[:, :, :1, :] + seq14[:, :, 13:, :]) * 0.5,
            seq14[:, :, 1:13, :],
        ), dim=2)
        rank_ctx = self.w_rank_set(c.mean(dim=1))[:, None, :, :]
        suit_ctx = self.w_suit_run(c.mean(dim=2))[:, :, None, :]
        tbl_ctx = self.w_table(t)[:, None, None, :]
        cards = cards + self.card_out(F.silu(seq + rank_ctx + suit_ctx + tbl_ctx))
        c_post = self.card_norm(cards)
        t_in = torch.cat((t, c_post.mean(dim=(1, 2)), c_post.amax(dim=(1, 2))), dim=-1)
        table = table + self.table_fc2(F.silu(self.table_fc1(t_in)))
        return cards, table


class PolicyNet(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        width = spec.width
        self.card_embed = nn.Linear(len(CARD_CHANNELS) + 5, width)
        self.table_embed = nn.Linear(NUM_SEAT_SLOTS * len(SEAT_CHANNELS) + len(CONTEXT_CHANNELS) + 1, width)
        self.blocks = nn.ModuleList(CardBlock(width) for _ in range(spec.blocks))
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

    def forward(self, batch: ObservationBatch) -> Tensor:
        size = batch.card.shape[0]
        extra_cards, extra_ctx = tensor_meld_channels(batch.card, batch.context)
        cards = self.card_embed(torch.cat((
            batch.card.reshape(size, 4, 13, len(CARD_CHANNELS)), extra_cards,
        ), dim=-1))
        width = cards.shape[-1]
        table = self.table_embed(torch.cat((batch.seat.flatten(1), batch.context, extra_ctx), dim=-1))
        for block in self.blocks:
            cards, table = block(cards, table)
        cards, table = self.card_final(cards), self.table_final(table)
        sets = cards.mean(dim=1)
        faces = F.silu(
            self.face_local(cards) + self.face_set(sets)[:, None] + self.face_table(table)[:, None, None],
        )
        index = batch.top.clamp(0, NUM_FACES - 1)
        rows = torch.arange(size, device=cards.device)
        has_top = ((batch.top >= 0) & (batch.top < NUM_FACES)).to(cards.dtype)[:, None]
        top_card = cards.reshape(size, NUM_FACES, width)[rows, index] * has_top
        top_set = sets[rows, index % 13] * has_top
        draw = F.silu(self.draw_table(table) + self.draw_top(top_card) + self.draw_set(top_set))
        advantage = torch.cat((self.draw_out(draw), self.face_out(faces).reshape(size, NUM_FACES)), dim=1)
        mean = (advantage * batch.legal).sum(dim=1, keepdim=True) / batch.legal.sum(
            dim=1, keepdim=True,
        ).clamp_min(1.0)
        return self.value_out(F.silu(self.value_hidden(table))) + advantage - mean


class QNetwork(nn.Module):
    architecture = ARCHITECTURE
    num_actions = NUM_ACTIONS

    def __init__(self, spec: ModelSpec = ModelSpec()) -> None:
        super().__init__()
        self.spec = spec
        self.policy = PolicyNet(spec)

    def forward(self, batch: ObservationBatch) -> Tensor:
        return self.policy(batch)


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


def checkpoint_payload(
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
    torch.save(checkpoint_payload(network, config, meta), path)


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
        raise ValueError("unsupported checkpoint_version %r; train fresh card_mlp v5 weights"
                         % payload.get("checkpoint_version"))
    if type(payload.get("encoding_version")) is not int or payload["encoding_version"] != ENCODING_VERSION:
        raise ValueError("incompatible encoding_version; card_mlp v5 requires encoding_version 3")
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported architecture; expected card_mlp v5")
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
