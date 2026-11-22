from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.checkpoints.tensor import atomic_torch_save, validate_model_state
from src.model.spec import ARCHITECTURE, BYTE_VALUES, CHECKPOINT_VERSION, ModelSpec, TokenSchema


@dataclass(frozen=True)
class TokenBatch:
    prefix: Tensor
    suffix: Tensor
    cut: Tensor
    legal: Tensor


@dataclass(frozen=True)
class Heads:
    log_policy: Tensor
    value: Tensor


def action_mask(bits: Tensor, actions: int) -> Tensor:
    return ((bits.unsqueeze(-1) >> torch.arange(actions, device=bits.device)) & 1).bool()


class KVCache:
    def __init__(self, spec: ModelSpec, rows: int, dtype: torch.dtype, device: torch.device) -> None:
        self.length = torch.zeros(rows, dtype=torch.long, device=device)
        shape = (rows, 0, spec.heads, spec.head_dim)
        self.keys = [torch.empty(shape, dtype=dtype, device=device) for _ in range(spec.layers)]
        self.values = [torch.empty_like(key) for key in self.keys]

    def reserve(self, rows: int, tokens: int) -> None:
        old_rows, old_tokens, heads, dim = self.keys[0].shape
        if rows > old_rows:
            length = self.length.new_zeros(max(rows, 2 * old_rows))
            length[:old_rows] = self.length
            self.length = length
        capacity = max(tokens, 2 * old_tokens) if tokens > old_tokens else old_tokens
        if len(self.length) == old_rows and capacity == old_tokens:
            return
        for storage in (self.keys, self.values):
            for layer, tensor in enumerate(storage):
                grown = tensor.new_zeros(len(self.length), capacity, heads, dim)
                grown[:old_rows, :old_tokens] = tensor
                storage[layer] = grown

    @torch.inference_mode()
    def reset(self, row: int) -> None:
        self.length[row] = 0


class _RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + 1e-6) * self.weight).to(x.dtype)


def _rotate(x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = rope
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


class _Block(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.heads, self.head_dim = spec.heads, spec.head_dim
        self.attn_norm = _RMSNorm(spec.width)
        self.ffn_norm = _RMSNorm(spec.width)
        self.qkv = nn.Linear(spec.width, 3 * spec.width, bias=False)
        self.out = nn.Linear(spec.width, spec.width, bias=False)
        self.gate_up = nn.Linear(spec.width, 2 * spec.ffn, bias=False)
        self.down = nn.Linear(spec.ffn, spec.width, bias=False)

    def _project(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        packed = self.qkv(self.attn_norm(x)).reshape(*x.shape[:2], 3, self.heads, self.head_dim)
        q, k, v = packed.permute(2, 0, 3, 1, 4).unbind()
        return _rotate(q, rope), _rotate(k, rope), v

    def _residual(self, x: Tensor, attention: Tensor) -> Tensor:
        x = x + self.out(attention.transpose(1, 2).reshape_as(x))
        gate, up = self.gate_up(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)

    def prefix_causal(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v = self._project(x, rope)
        return self._residual(x, F.scaled_dot_product_attention(q, k, v, is_causal=True)), k, v

    def prefix_append(
        self, x: Tensor, rope: tuple[Tensor, Tensor], cache: KVCache, layer: int,
        rows: Tensor, writes: tuple[Tensor, Tensor], valid: Tensor, mask: Tensor, span: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.size(1):
            q, k, v = self._project(x, rope)
            cache.keys[layer][writes] = k.transpose(1, 2)[valid]
            cache.values[layer][writes] = v.transpose(1, 2)[valid]
        keys = cache.keys[layer][:, :span].index_select(0, rows).transpose(1, 2)
        values = cache.values[layer][:, :span].index_select(0, rows).transpose(1, 2)
        if x.size(1):
            x = self._residual(x, F.scaled_dot_product_attention(q, keys, values, attn_mask=mask))
        return x, keys, values

    def suffix_blocks(
        self, x: Tensor, rope: tuple[Tensor, Tensor], keys: Tensor, values: Tensor,
        prefix_valid: Tensor, suffix_valid: Tensor,
    ) -> Tensor:
        q, k, v = self._project(x, rope)
        rows = keys.size(0)
        blocks = x.size(0) // rows

        def append(prefix: Tensor, suffix: Tensor) -> Tensor:
            prefix = prefix[:, None].expand(-1, blocks, -1, -1, -1)
            return torch.cat((prefix, suffix.unflatten(0, (rows, blocks))), dim=-2).flatten(0, 1)

        mask = torch.cat((prefix_valid, suffix_valid), dim=-1)
        attention = F.scaled_dot_product_attention(q, append(keys, k), append(values, v), attn_mask=mask)
        return self._residual(x, attention)


class GameTransformer(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, spec: ModelSpec, schema: TokenSchema) -> None:
        super().__init__()
        self.spec, self.schema = spec, schema
        self.embedding = nn.Embedding(schema.fields * BYTE_VALUES, spec.width)
        self.blocks = nn.ModuleList(_Block(spec) for _ in range(spec.layers))
        self.final_norm = _RMSNorm(spec.width)
        self.policy = nn.Linear(spec.width, schema.num_actions)
        self.pointer = nn.Linear(spec.width, 1)
        self.value = nn.Linear(spec.width, 1)
        self.register_buffer("field_ids", torch.arange(schema.fields) * BYTE_VALUES, persistent=False)
        powers = torch.arange(0, spec.head_dim, 2, dtype=torch.float32) / spec.head_dim
        self.register_buffer("frequencies", 10_000 ** (-powers), persistent=False)

    def _embed(self, tokens: Tensor) -> Tensor:
        return (self.embedding(tokens.long() + self.field_ids) * (tokens != 0).unsqueeze(-1)).sum(dim=-2)

    def _rope(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        angle = positions[:, None, :, None].float() * self.frequencies
        return angle.cos(), angle.sin()

    def _readout(
        self, cut: Tensor, suffix: Tensor, prefix_positions: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor, Tensor]:
        prefix_valid = prefix_positions < cut.reshape(-1, 1, 1, 1)
        suffix_valid = (suffix[..., 0] != 0)[:, None, None, :]
        return self._embed(suffix), self._rope(cut.reshape(-1, 1)), prefix_valid, suffix_valid

    def _heads(self, suffix: Tensor, suffix_tokens: Tensor, legal: Tensor) -> Heads:
        normed = self.final_norm(suffix)
        readout = normed[:, 0]
        logits = self.policy(readout).float()
        if self.schema.pointer_field is not None:
            pointer = suffix_tokens[..., self.schema.pointer_field].long()
            scores = self.pointer(normed).squeeze(-1).float().masked_fill(pointer == 0, 0)
            logits = logits.scatter_add(-1, (pointer - 1).clamp_min(0), scores)
        logits = logits.reshape_as(legal).masked_fill(~legal, torch.finfo(torch.float32).min)
        return Heads(F.log_softmax(logits, dim=-1), self.value(readout).reshape(legal.shape[:-1]).float())

    def unroll(self, batch: TokenBatch) -> Heads:
        prefix = self._embed(batch.prefix)
        positions = torch.arange(prefix.size(1), device=prefix.device)
        prefix_rope = self._rope(positions[None, :])
        suffix_tokens = batch.suffix.flatten(0, 1)
        suffix, suffix_rope, prefix_valid, suffix_valid = self._readout(batch.cut, suffix_tokens, positions)
        for block in self.blocks:
            prefix, keys, values = block.prefix_causal(prefix, prefix_rope)
            suffix = block.suffix_blocks(suffix, suffix_rope, keys, values, prefix_valid, suffix_valid)
        return self._heads(suffix, suffix_tokens, batch.legal)

    def forward(self, batch: TokenBatch) -> Heads:
        return self.unroll(batch)

    def new_cache(self, rows: int = 0, dtype: torch.dtype = torch.float32) -> KVCache:
        return KVCache(self.spec, rows, dtype, self.field_ids.device)

    @torch.inference_mode()
    def step(
        self, cache: KVCache, rows: Tensor, prefix: Tensor, suffix: Tensor, legal: Tensor, *,
        append_lengths: Tensor | None = None,
    ) -> Heads:
        if append_lengths is None:
            append_lengths = torch.full_like(rows, prefix.size(1))
        cache.reserve(int(rows.max()) + 1, 0)
        start = cache.length[rows]
        cut = start + append_lengths
        span = int(cut.max())
        cache.reserve(len(cache.length), span)
        positions = torch.arange(span, device=rows.device)
        offsets = torch.arange(prefix.size(1), device=rows.device)
        appended = start[:, None] + offsets
        valid = offsets < append_lengths[:, None]
        writes = rows[:, None].expand_as(appended)[valid], appended[valid]
        prefix_mask = (positions <= appended[:, :, None]) & (positions < cut[:, None, None])
        prefix_rope = self._rope(appended)
        x = self._embed(prefix)
        y, suffix_rope, prefix_valid, suffix_valid = self._readout(cut, suffix, positions)
        for layer, block in enumerate(self.blocks):
            x, keys, values = block.prefix_append(
                x, prefix_rope, cache, layer, rows, writes, valid, prefix_mask[:, None], span,
            )
            y = block.suffix_blocks(y, suffix_rope, keys, values, prefix_valid, suffix_valid)
        cache.length[rows] = cut
        return self._heads(y, suffix, legal)

    @torch.inference_mode()
    def evaluate_candidates(
        self, cache: KVCache, rows: Tensor, candidate_suffix: Tensor, candidate_legal: Tensor,
    ) -> Heads:
        cut = cache.length[rows]
        span = int(cut.max())
        positions = torch.arange(span, device=rows.device)
        suffix = candidate_suffix.reshape(-1, *candidate_suffix.shape[-2:])
        cut = cut.repeat_interleave(suffix.size(0) // rows.numel())
        y, rope, prefix_valid, suffix_valid = self._readout(cut, suffix, positions)
        for layer, block in enumerate(self.blocks):
            keys = cache.keys[layer][:, :span].index_select(0, rows).transpose(1, 2)
            values = cache.values[layer][:, :span].index_select(0, rows).transpose(1, 2)
            y = block.suffix_blocks(y, rope, keys, values, prefix_valid, suffix_valid)
        return self._heads(y, suffix, candidate_legal)


@dataclass(frozen=True)
class Checkpoint:
    network: GameTransformer
    game: dict
    meta: dict

    @property
    def version(self) -> int:
        return self.meta["version"]


def _validate_metadata(game: object, meta: object) -> None:
    if not isinstance(game, dict) or not isinstance(meta, dict):
        raise ValueError("checkpoint needs game and metadata dictionaries")
    if type(meta.get("version")) is not int or meta["version"] < 0:
        raise ValueError("checkpoint policy version must be a nonnegative integer")


def save_checkpoint(
    path: Path, network: GameTransformer, game: Mapping[str, object], meta: Mapping[str, object],
) -> None:
    game, meta = dict(game), dict(meta)
    _validate_metadata(game, meta)
    atomic_torch_save({
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE,
        "model_spec": network.spec.to_dict(),
        "schema": network.schema.to_dict(),
        "game": game,
        "meta": meta,
        "model_state_dict": {
            key: tensor.detach().cpu().float() for key, tensor in network.state_dict().items()
        },
    }, Path(path))


def load_checkpoint(path: Path, schema: TokenSchema, device: str | torch.device = "cpu") -> Checkpoint:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, UnpicklingError, EOFError) as exc:
        raise ValueError(f"cannot read model checkpoint {path}") from exc
    if (
        not isinstance(payload, dict) or type(payload.get("checkpoint_version")) is not int
        or payload["checkpoint_version"] != CHECKPOINT_VERSION or payload.get("architecture") != ARCHITECTURE
    ):
        raise ValueError("incompatible checkpoint; train a fresh GameTransformer")
    if set(payload) != {"checkpoint_version", "architecture", "model_spec", "schema", "game", "meta", "model_state_dict"}:
        raise ValueError("invalid checkpoint fields")
    if TokenSchema.from_dict(payload["schema"]) != schema:
        raise ValueError("checkpoint token schema does not match the requested schema")
    _validate_metadata(payload["game"], payload["meta"])
    network = GameTransformer(ModelSpec.from_dict(payload["model_spec"]), schema)
    network.load_state_dict(validate_model_state(payload["model_state_dict"], network.state_dict(), "model"))
    return Checkpoint(network.to(device).eval(), payload["game"], payload["meta"])
