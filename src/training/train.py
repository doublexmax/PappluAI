from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from enum import IntEnum
from io import BytesIO
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from pickle import UnpicklingError
from random import Random
from typing import Callable, Iterator, Mapping, Sequence
import time
from uuid import UUID, uuid4

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from src.checkpoints.io import atomic_write_json
from src.checkpoints.tensor import atomic_torch_save, validate_model_state
from src.evaluation.arena import Deal, MatchResult, RandomPolicy, TransformerPolicy, head_to_head, run_matches
from src.game.environment import GameConfig
from src.game.multiplayer import MatchConfig
from src.model.encoding import HEADER_TOKENS, SCHEMA, validate_game
from src.model.network import GameTransformer, TokenBatch, action_mask, load_checkpoint, save_checkpoint
from src.model.spec import ModelSpec, TokenSchema


STATE_VERSION = 17
ROLLOUT_FORMAT = 5
VALIDATION_SEED = 30_000_001
OUTBOX_LIMIT = 8
LENGTH_BUCKET = 8
DISCOUNT = 0.995
TUNABLE = {
    "actors", "tables", "torch_threads", "actor_threads", "device", "precision", "checkpoint_every",
    "learning_rate", "eta", "sync_every", "plan",
}
TUNABLE_PLAN = ("player_counts", "curriculum_share", "curriculum_max_distance")
MASTERY_LADDER = {
    1: (0.92, 0.45), 2: (0.85, 0.30), 3: (0.75, 0.20), 4: (0.75, 0.20),
    6: (0.60, 0.12), 9: (0.60, 0.12), 14: (0.45, 0.12),
}


@dataclass(frozen=True)
class CollectPlan:
    player_counts: tuple[int, ...] = (1, 2, 3, 4)
    curriculum_share: float = 0.50
    curriculum_max_distance: int = 6
    uniform_mix: float = 0.02
    batch_decisions: int = 8192
    search_ply: int = 1

    def __post_init__(self) -> None:
        counts = self.player_counts
        if (
            not isinstance(counts, tuple) or not counts
            or any(type(n) is not int or not 1 <= n <= 6 for n in counts)
            or len(set(counts)) != len(counts)
        ):
            raise ValueError("player_counts must be distinct integers in 1..6")
        for value in (self.curriculum_share, self.uniform_mix):
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError("curriculum_share and uniform_mix must be in [0, 1]")
        if type(self.curriculum_max_distance) is not int or self.curriculum_max_distance < 1:
            raise ValueError("curriculum_max_distance must be positive")
        if type(self.batch_decisions) is not int or not 1 <= self.batch_decisions <= 262_144:
            raise ValueError("batch_decisions must be in 1..262144")
        if type(self.search_ply) is not int or self.search_ply not in (0, 1):
            raise ValueError("search_ply must be 0 or 1")

    def validate(self, game: GameConfig) -> None:
        validate_game(game)
        for players in self.player_counts:
            if players > 1:
                MatchConfig(game, players)
        if self.curriculum_share:
            if game.cards_in_hand < 3:
                raise ValueError("curriculum needs at least three cards per hand")
            if self.curriculum_max_distance > game.cards_in_hand:
                raise ValueError("curriculum_max_distance exceeds the hand size")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CollectPlan:
        if not isinstance(data, Mapping) or data.keys() != cls.__dataclass_fields__.keys():
            raise ValueError("invalid collection plan fields")
        return cls(**data)

    def deals(self, nonce: str) -> Iterator[Deal]:
        rng = Random(nonce)
        while True:
            for players in self.player_counts:
                start = None
                if rng.random() < self.curriculum_share:
                    low = int(players > 1)
                    start = min(rng.randint(low, self.curriculum_max_distance),
                                rng.randint(low, self.curriculum_max_distance))
                yield Deal(rng.getrandbits(63), (0,) * players, start)


@dataclass(frozen=True)
class TrainConfig:
    game: GameConfig = field(default_factory=lambda: GameConfig(max_turns=60))
    plan: CollectPlan = field(default_factory=CollectPlan)
    seed: int = 41
    actors: int = 2
    tables: int = 64
    torch_threads: int = 4
    actor_threads: int = 2
    device: str = "cpu"
    precision: str = "bf16"
    batch_size: int = 8
    unroll_steps: int = 12
    buffer_decisions: int = 400_000
    min_buffer_decisions: int = 20_000
    max_policy_lag: int = 200
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    value_weight: float = 1.0
    rho_bar: float = 1.0
    c_bar: float = 0.85
    eta: float = 0.01
    entropy_weight: float = 0.0
    anchor_every: int = 2000
    sync_every: int = 1
    publish_every: int = 25
    checkpoint_every: int = 500
    eval_every: int = 2000
    eval_deals: int = 64
    auto_curriculum: bool = False
    promote_margin: float = 0.02

    def __post_init__(self) -> None:
        self.plan.validate(self.game)
        for name in (
            "tables", "torch_threads", "actor_threads", "batch_size", "unroll_steps", "buffer_decisions",
            "min_buffer_decisions", "max_policy_lag", "anchor_every", "sync_every",
            "publish_every", "checkpoint_every", "eval_deals",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "actors", "eval_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "learning_rate", "weight_decay", "grad_clip", "value_weight", "rho_bar",
            "c_bar", "eta", "entropy_weight", "promote_margin",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not self.learning_rate or not self.grad_clip or not 0 < self.c_bar <= min(1, self.rho_bar):
            raise ValueError("learning_rate and grad_clip must be positive, with 0 < c_bar <= min(1, rho_bar)")
        if self.min_buffer_decisions > self.buffer_decisions:
            raise ValueError("min_buffer_decisions exceeds the buffer's capacity")
        if self.publish_every > self.max_policy_lag:
            raise ValueError("publish_every exceeds max_policy_lag")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if type(self.auto_curriculum) is not bool:
            raise ValueError("auto_curriculum must be a boolean")
        if self.auto_curriculum and (
            not self.eval_every or self.game.cards_in_hand < 3
            or self.plan.curriculum_max_distance not in (*MASTERY_LADDER, self.game.cards_in_hand)
            or self.plan.curriculum_max_distance > self.game.cards_in_hand
        ):
            raise ValueError("auto_curriculum requires periodic evaluation and a valid mastery distance")

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        return cls(**{**data, "game": GameConfig.from_dict(data["game"]), "plan": CollectPlan.from_dict(data["plan"])})

    def with_overrides(self, overrides: Mapping[str, object]) -> TrainConfig:
        changes = dict(overrides)
        if "game" in changes:
            changes["game"] = replace(self.game, **changes["game"])
        if "plan" in changes:
            changes["plan"] = replace(self.plan, **changes["plan"])
        return replace(self, **changes)


def _run_id(value: object) -> str:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("expected a canonical run ID")
    return value


def _bytes_tensor(values: bytes, shape: tuple[int, ...], dtype: torch.dtype = torch.uint8) -> Tensor:
    return torch.frombuffer(bytearray(values), dtype=dtype).reshape(shape)


@dataclass(frozen=True)
class Trajectories:
    version: int
    episodes: int
    run_id: str
    players: Tensor
    utility: Tensor
    prefix: Tensor
    prefix_end: Tensor
    suffix: Tensor
    decision_end: Tensor
    cut: Tensor
    legal: Tensor
    action: Tensor
    behavior_logp: Tensor
    target_probs: Tensor
    forced: Tensor
    potential: Tensor
    behavior_value: Tensor

    @property
    def decisions(self) -> int:
        return self.action.numel()

    @classmethod
    def from_results(cls, results: Sequence[MatchResult], version: int, run_id: str) -> Trajectories:
        players, utility, traces = [], [], []
        for result in results:
            for trace, score in zip(result.traces, result.outcome.utilities()):
                if trace is not None:
                    players.append(result.deal.players)
                    utility.append(score)
                    traces.append(trace)
        if not traces:
            raise ValueError("a batch needs at least one recorded seat")
        return cls(
            version, len(results), run_id, torch.tensor(players, dtype=torch.uint8),
            torch.tensor(utility, dtype=torch.float32),
            _bytes_tensor(b"".join(row.prefix for row in traces), (-1, SCHEMA.fields)),
            torch.tensor([len(row.prefix) // SCHEMA.fields for row in traces]).cumsum(0),
            _bytes_tensor(b"".join(row.suffix for row in traces), (-1, SCHEMA.suffix_tokens, SCHEMA.fields)),
            torch.tensor([len(row.action) for row in traces]).cumsum(0),
            torch.tensor([cut for row in traces for cut in row.cut], dtype=torch.int16),
            torch.tensor([bits for row in traces for bits in row.legal], dtype=torch.int64),
            _bytes_tensor(b"".join(row.action for row in traces), (-1,)),
            torch.tensor([logp for row in traces for logp in row.behavior_logp], dtype=torch.float32),
            _bytes_tensor(b"".join(row.target_probs for row in traces), (-1, SCHEMA.num_actions), torch.uint16),
            _bytes_tensor(b"".join(row.forced for row in traces), (-1,)),
            torch.tensor([value for row in traces for value in row.potential], dtype=torch.float32),
            torch.tensor([value for row in traces for value in row.behavior_value], dtype=torch.float32),
        )

    def save(self, path: Path) -> None:
        atomic_torch_save({
            "format": ROLLOUT_FORMAT, "schema": SCHEMA.to_dict(), **vars(self),
            "target_probs": self.target_probs.view(torch.int16),
        }, Path(path))

    @classmethod
    def load(cls, path: Path, schema: TokenSchema = SCHEMA) -> Trajectories:
        path = Path(path)
        if path.stat().st_size > 256 * 1024**2:
            raise ValueError("rollout batch exceeds 256 MiB")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (RuntimeError, UnpicklingError, EOFError) as exc:
            raise ValueError("cannot read rollout batch") from exc
        if not isinstance(payload, dict) or set(payload) != {"format", "schema", *cls.__dataclass_fields__}:
            raise ValueError("invalid rollout batch fields")
        if type(payload["format"]) is not int or payload["format"] != ROLLOUT_FORMAT:
            raise ValueError("incompatible rollout batch format")
        if TokenSchema.from_dict(payload["schema"]) != schema:
            raise ValueError("incompatible rollout token schema")
        payload.pop("format")
        payload.pop("schema")
        targets = payload["target_probs"]
        if not isinstance(targets, Tensor) or targets.layout != torch.strided or targets.dtype != torch.int16:
            raise ValueError("invalid rollout column target_probs")
        payload["target_probs"] = targets.view(torch.uint16)
        batch = cls(**payload)
        batch._validate(schema)
        return batch

    def _validate(self, schema: TokenSchema) -> None:
        _run_id(self.run_id)
        if type(self.version) is not int or self.version < 0 or type(self.episodes) is not int or self.episodes < 1:
            raise ValueError("invalid rollout revision or episode count")
        for name, dtype, trailing in (
            ("players", torch.uint8, ()), ("utility", torch.float32, ()),
            ("prefix", torch.uint8, (schema.fields,)), ("prefix_end", torch.int64, ()),
            ("suffix", torch.uint8, (schema.suffix_tokens, schema.fields)), ("decision_end", torch.int64, ()),
            ("cut", torch.int16, ()), ("legal", torch.int64, ()), ("action", torch.uint8, ()),
            ("behavior_logp", torch.float32, ()), ("target_probs", torch.uint16, (schema.num_actions,)),
            ("forced", torch.uint8, ()), ("potential", torch.float32, ()), ("behavior_value", torch.float32, ()),
        ):
            col = getattr(self, name)
            if (
                not isinstance(col, Tensor) or col.layout != torch.strided or col.device.type != "cpu"
                or col.dtype != dtype or col.ndim != len(trailing) + 1 or col.shape[1:] != trailing
            ):
                raise ValueError(f"invalid rollout column {name}")
        streams, decisions = len(self.players), self.decisions
        if not self.episodes <= streams <= self.episodes * 6 or not bool(((self.players >= 1) & (self.players <= 6)).all()):
            raise ValueError("invalid rollout player counts")
        if any(len(col) != streams for col in (self.utility, self.prefix_end, self.decision_end)):
            raise ValueError("invalid rollout trajectory column lengths")
        if any(len(col) != decisions for col in (
            self.suffix, self.cut, self.legal, self.behavior_logp, self.target_probs, self.forced,
            self.potential, self.behavior_value,
        )):
            raise ValueError("invalid rollout decision column lengths")
        if bool((self.forced > 1).any()):
            raise ValueError("invalid rollout forced flags")
        lengths, prefix_lengths = (
            ends.diff(prepend=ends.new_zeros(1)) for ends in (self.decision_end, self.prefix_end)
        )
        if (
            int(self.decision_end[-1]) != decisions or int(self.prefix_end[-1]) != len(self.prefix)
            or bool((lengths < 1).any() or (prefix_lengths <= HEADER_TOKENS).any())
            or bool((self.decision_end <= 0).any() or (self.prefix_end <= 0).any())
        ):
            raise ValueError("invalid rollout offsets")
        cuts = self.cut.long()
        interior = torch.ones(decisions, dtype=torch.bool)
        interior[self.decision_end - 1] = False
        if (
            bool((cuts <= HEADER_TOKENS).any())
            or not torch.equal(cuts[self.decision_end - 1], prefix_lengths)
            or bool((cuts.diff()[interior[:-1]] < 0).any())
        ):
            raise ValueError("invalid rollout prefix cuts")
        acts = self.action.long()
        if (
            bool((self.legal <= 0).any() or (self.legal >= 1 << schema.num_actions).any())
            or bool((acts >= schema.num_actions).any() or (((self.legal >> acts) & 1) == 0).any())
        ):
            raise ValueError("invalid rollout legal action")
        legal = action_mask(self.legal, schema.num_actions)
        targets = self.target_probs.int()
        if bool(
            targets.masked_fill(legal, 0).any()
            or ((targets.sum(-1) - 65535).abs() > legal.sum(-1) // 2).any()
        ):
            raise ValueError("invalid rollout target probability mass or legality")
        if bool((self.suffix[:, 0, 0] == 0).any()):
            raise ValueError("rollout has a padded readout token")
        if schema.pointer_field is not None and bool((self.suffix[..., schema.pointer_field] > schema.num_actions).any()):
            raise ValueError("rollout has a pointer outside the action space")
        if not bool(((self.behavior_logp >= -104) & (self.behavior_logp <= 0)).all()):
            raise ValueError("invalid rollout behavior log-probabilities")
        if not bool(((self.potential >= 0) & (self.potential <= 0.75)).all()):
            raise ValueError("invalid rollout potential")
        if not bool(self.behavior_value.isfinite().all()):
            raise ValueError("invalid rollout behavior values")
        if not bool((self.utility.abs() <= 1).all()):
            raise ValueError("invalid rollout utilities")


@dataclass
class _Completed:
    deals: list[MatchResult] = field(default_factory=list)
    decisions: int = 0


def collect(
    out_dir: Path, model_path: Path, *, worker: int, tables: int = 64, torch_threads: int = 2,
    precision: str = "bf16", device: str = "cpu", stop: Callable[[], bool] = lambda: False,
    max_batches: int | None = None,
) -> dict:
    if type(worker) is not int or worker < 0:
        raise ValueError("worker must be a nonnegative integer")
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("max_batches must be a positive integer")
    if type(torch_threads) is not int or torch_threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    torch.set_num_threads(torch_threads)
    out_dir, model_path = Path(out_dir), Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    def fingerprint() -> tuple[int, int]:
        stat = model_path.stat()
        return stat.st_mtime_ns, stat.st_size

    stamp = fingerprint()
    checkpoint = load_checkpoint(model_path, SCHEMA, device)
    game = GameConfig.from_dict(checkpoint.game)
    plan = CollectPlan.from_dict(checkpoint.meta.get("plan"))
    plan.validate(game)
    run = _run_id(checkpoint.meta.get("run_id"))
    revision = checkpoint.version
    spec = checkpoint.network.spec
    policy = TransformerPolicy(checkpoint.network, game, sample=True, uniform_mix=plan.uniform_mix,
                               precision=precision, record=True, search_ply=plan.search_ply)
    del checkpoint
    active: Counter[int] = Counter()
    admitted: dict[int, int] = {}
    pending: dict[int, _Completed] = {}
    incarnation = uuid4().hex
    nonce = f"{run}:{worker}:{incarnation}"
    totals = {"batches": 0, "episodes": 0, "decisions": 0}

    def deals() -> Iterator[Deal]:
        nonlocal stamp, revision, policy, plan
        queue = plan.deals(nonce)
        while True:
            if len(active) <= 1:
                new_stamp = fingerprint()
                if new_stamp != stamp:
                    snapshot = load_checkpoint(model_path, SCHEMA, device)
                    updated = CollectPlan.from_dict(snapshot.meta.get("plan"))
                    if (
                        snapshot.meta.get("run_id") != run or snapshot.game != game.to_dict()
                        or snapshot.network.spec != spec or updated != replace(
                            plan, curriculum_share=updated.curriculum_share,
                            curriculum_max_distance=updated.curriculum_max_distance,
                        )
                    ):
                        raise ValueError("collection checkpoint changed run, game, fixed plan fields, or model shape")
                    if snapshot.version > revision:
                        revision = snapshot.version
                        if updated != plan:
                            updated.validate(game)
                            plan = updated
                            queue = plan.deals(f"{nonce}:{revision}")
                        policy = TransformerPolicy(snapshot.network, game, sample=True, uniform_mix=plan.uniform_mix,
                                                   precision=precision, record=True, search_ply=plan.search_ply)
                    stamp = new_stamp
            yield next(queue)

    def admit(deal: Deal) -> tuple[TransformerPolicy, ...]:
        admitted[deal.seed] = revision
        active[revision] += 1
        return (policy,)

    def at_limit() -> bool:
        return max_batches is not None and totals["batches"] >= max_batches

    def flush(version: int) -> None:
        if at_limit():
            return
        pattern = f"batch-w{worker:04d}-*.pt"
        while max_batches is None and not stop() and sum(1 for _ in out_dir.glob(pattern)) >= OUTBOX_LIMIT:
            time.sleep(0.1)
        chunk = pending.pop(version)
        filename = f"batch-w{worker:04d}-{time.time_ns()}-{incarnation}-{totals['batches']:08d}.pt"
        Trajectories.from_results(chunk.deals, version, run).save(out_dir / filename)
        totals["batches"] += 1
        totals["episodes"] += len(chunk.deals)
        totals["decisions"] += chunk.decisions

    stopping = lambda: stop() or at_limit()
    for result in run_matches(deals(), admit, game, tables=tables, stop=stopping):
        version = admitted.pop(result.deal.seed)
        active[version] -= 1
        if not active[version]:
            del active[version]
        chunk = pending.setdefault(version, _Completed())
        chunk.deals.append(result)
        chunk.decisions += result.action_count
        if chunk.decisions >= plan.batch_decisions or (version != revision and version not in active):
            flush(version)
        if stopping():
            break
    for version in tuple(pending):
        flush(version)
    return totals


@dataclass(frozen=True)
class _SeatTrajectory:
    prefix: Tensor
    suffix: Tensor
    cut: Tensor
    legal: Tensor
    action: Tensor
    behavior_logp: Tensor
    utility: Tensor
    target_probs: Tensor
    forced: Tensor
    potential: Tensor
    behavior_value: Tensor


@dataclass(frozen=True)
class StepSample:
    index: Tensor
    weight: Tensor

    def gather(self, column: Tensor) -> Tensor:
        return column[torch.arange(self.index.size(0))[:, None], self.index]

    def tokens(self, tokens: TokenBatch) -> TokenBatch:
        cut = self.gather(tokens.cut)
        return TokenBatch(tokens.prefix[:, :int(cut.max())], self.gather(tokens.suffix), cut, self.gather(tokens.legal))


@dataclass(frozen=True)
class TrajectoryBatch:
    tokens: TokenBatch
    action: Tensor
    behavior_logp: Tensor
    behavior_value: Tensor
    reward: Tensor
    discount: Tensor
    valid: Tensor
    target_probs: Tensor
    forced: Tensor

    @classmethod
    def pack(cls, seats: tuple[_SeatTrajectory, ...]) -> TrajectoryBatch:
        def pack(name: str) -> Tensor:
            return pad_sequence([getattr(seat, name) for seat in seats], batch_first=True)

        action = pack("action").long()
        lengths = torch.tensor([seat.action.numel() for seat in seats])
        steps = torch.arange(action.size(1))
        terminal = lengths[:, None] - 1
        valid = steps <= terminal
        reward = (steps == terminal) * torch.stack([seat.utility for seat in seats])[:, None]
        discount = (steps < terminal).float() * DISCOUNT
        potential = pack("potential")
        reward += discount * torch.roll(potential, -1, dims=1) - potential
        tokens = TokenBatch(pack("prefix"), pack("suffix"), pack("cut").long(),
                            action_mask(pack("legal"), SCHEMA.num_actions))
        targets = pad_sequence([seat.target_probs.float() for seat in seats], batch_first=True)
        targets /= targets.sum(-1, keepdim=True).clamp_min(1)
        return cls(tokens, action, pack("behavior_logp"), pack("behavior_value"),
                   reward, discount, valid, targets, pack("forced").bool())

    def sample_steps(self, limit: int, rng: torch.Generator) -> StepSample:
        lengths = self.valid.sum(1, keepdim=True)
        index = torch.arange(self.valid.size(1)).expand_as(self.valid)
        if limit >= index.size(1):
            return StepSample(index, self.valid / lengths)
        keys = torch.rand(self.valid.shape, generator=rng).masked_fill(~self.valid, -1)
        if limit > 1:
            keys.masked_fill_(index == lengths - 1, 2)
        index = keys.topk(limit, dim=1).indices.sort(dim=1).values
        weight = torch.ones_like(index, dtype=torch.float32)
        if limit > 1:
            count = lengths.clamp(max=limit)
            weight = torch.where(index == lengths - 1, 1, (lengths - 1) / (count - 1).clamp_min(1))
        return StepSample(index, (weight / lengths).masked_fill(index >= lengths, 0))


@dataclass(frozen=True)
class _BatchIndex:
    version: int
    decisions: int
    buckets: dict[tuple[int, int], list[_SeatTrajectory]]

    @classmethod
    def from_batch(cls, batch: Trajectories) -> _BatchIndex:
        buckets: dict[tuple[int, int], list[_SeatTrajectory]] = {}
        start = prefix_start = 0
        offsets = zip(batch.players.tolist(), batch.decision_end.tolist(), batch.prefix_end.tolist())
        for row, (players, end, prefix_end) in enumerate(offsets):
            span = slice(start, end)
            seat = _SeatTrajectory(
                batch.prefix[prefix_start:prefix_end], batch.suffix[span], batch.cut[span],
                batch.legal[span], batch.action[span], batch.behavior_logp[span], batch.utility[row],
                batch.target_probs[span], batch.forced[span],
                batch.potential[span], batch.behavior_value[span],
            )
            buckets.setdefault((players, (end - start - 1) // LENGTH_BUCKET), []).append(seat)
            start, prefix_start = end, prefix_end
        return cls(batch.version, batch.decisions, buckets)


class _Buffer:
    def __init__(self, capacity: int, max_policy_lag: int) -> None:
        self.capacity, self.max_policy_lag = capacity, max_policy_lag
        self.chunks: list[_BatchIndex] = []
        self.decisions = 0

    def __len__(self) -> int:
        return self.decisions

    def prune(self, version: int) -> None:
        kept = []
        for chunk in self.chunks:
            if version - chunk.version <= self.max_policy_lag:
                kept.append(chunk)
            else:
                self.decisions -= chunk.decisions
        self.chunks = kept

    def add(self, batch: Trajectories, version: int) -> bool:
        self.prune(version)
        if not 0 <= version - batch.version <= self.max_policy_lag or batch.decisions > self.capacity:
            return False
        while self.decisions + batch.decisions > self.capacity:
            self.decisions -= self.chunks.pop(0).decisions
        self.chunks.append(_BatchIndex.from_batch(batch))
        self.decisions += batch.decisions
        return True

    def bucket_sizes(self, min_version: int = 0) -> Counter[tuple[int, int]]:
        sizes: Counter[tuple[int, int]] = Counter()
        for chunk in self.chunks:
            if chunk.version >= min_version:
                sizes.update({key: len(seats) for key, seats in chunk.buckets.items()})
        return sizes

    def sample(
        self, rng: Random, batch_size: int, step_version: int, *,
        seed: int = 0, targets: Mapping[tuple[int, int], int] | None = None,
    ) -> TrajectoryBatch:
        self.prune(step_version)
        if not self.chunks:
            raise ValueError("cannot sample an empty buffer")
        if targets is None:
            targets = self.bucket_sizes()
            step_rng = rng
        else:
            step_rng = Random(((seed + 1) * 0x9E3779B1) ^ step_version)
        key = step_rng.choices(tuple(targets), weights=tuple(targets.values()), k=1)[0]
        if not any(key in chunk.buckets for chunk in self.chunks):
            players, length = key
            key = min(self.bucket_sizes(), key=lambda k: (abs(k[0] - players), abs(k[1] - length), k))
        group = [seat for chunk in self.chunks for seat in chunk.buckets.get(key, ())]
        return TrajectoryBatch.pack(tuple(rng.choices(group, k=batch_size)))


@torch.no_grad()
def vtrace(
    log_rho: Tensor, reward: Tensor, discount: Tensor, value: Tensor, bootstrap: Tensor, *,
    rho_bar: float, c_bar: float,
) -> tuple[Tensor, Tensor]:
    log_rho, reward, discount, value, bootstrap = (x.float() for x in (log_rho, reward, discount, value, bootstrap))
    rho = log_rho.clamp(max=math.log(rho_bar)).exp()
    c = log_rho.clamp(max=math.log(c_bar)).exp()
    next_value = torch.cat((value[:, 1:], bootstrap[:, None]), dim=1)
    delta = rho * (reward + discount * next_value - value)
    decay = discount * c
    correction = delta
    stride = 1
    while stride < value.size(1):
        correction = torch.cat((correction[:, :-stride] + decay[:, :-stride] * correction[:, stride:],
                                correction[:, -stride:]), dim=1)
        decay = torch.cat((decay[:, :-stride] * decay[:, stride:], decay[:, -stride:]), dim=1)
        stride *= 2
    vs = value + correction
    next_vs = torch.cat((vs[:, 1:], bootstrap[:, None]), dim=1)
    return vs, rho * (reward + discount * next_vs - value)


@dataclass(frozen=True)
class LearnStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    mean_rho: float
    kl_anchor: float
    grad_norm: float
    trained_steps: int


class _Step(IntEnum):
    UPDATE = 0
    WAIT = 1
    BUDGET = 2
    STOPPED = 3
    MAX_UPDATES = 4
    ACTOR_ERROR = 5
    EVAL_ERROR = 6


@dataclass(frozen=True)
class _LearnerGroup:
    rank: int = 0
    size: int = 1
    owns_backend: bool = False
    use_avg: bool = False

    @classmethod
    def open(cls) -> _LearnerGroup:
        requested = int(os.environ.get("WORLD_SIZE", "1"))
        if not dist.is_available():
            if requested > 1:
                raise RuntimeError("distributed learning requires PyTorch with Gloo")
            return cls()
        owns_backend = requested > 1 and not dist.is_initialized()
        if owns_backend:
            if not dist.is_gloo_available():
                raise RuntimeError("distributed learning requires PyTorch with Gloo")
            dist.init_process_group(backend="gloo", timeout=timedelta(minutes=20))
        try:
            if not dist.is_initialized() or dist.get_world_size() == 1:
                return cls()
            if dist.get_backend() != "gloo":
                raise ValueError("distributed learning requires a Gloo process group")
            use_avg = True
            try:
                dist.all_reduce(torch.zeros(1), op=dist.ReduceOp.AVG)
            except RuntimeError as exc:
                if "ReduceOp.AVG" not in str(exc) or "gloo" not in str(exc).lower():
                    raise
                use_avg = False
            return cls(dist.get_rank(), dist.get_world_size(), owns_backend, use_avg)
        except BaseException:
            if owns_backend:
                dist.destroy_process_group()
            raise

    def close(self) -> None:
        if self.owns_backend and dist.is_initialized():
            dist.destroy_process_group()

    def share(self, value: object):
        if self.size == 1:
            return value
        payload = BytesIO()
        if self.rank == 0:
            torch.save(value, payload)
        length = torch.tensor(payload.tell(), dtype=torch.int64)
        dist.broadcast(length, src=0)
        data = (torch.frombuffer(payload.getbuffer(), dtype=torch.uint8) if self.rank == 0
                else torch.empty(length.item(), dtype=torch.uint8))
        dist.broadcast(data, src=0)
        if self.rank == 0:
            return value
        return torch.load(BytesIO(bytes(data.tolist())), map_location="cpu", weights_only=True)

    def step(self, local: _Step, sizes: Mapping[tuple[int, int], int]) -> tuple[_Step, dict[tuple[int, int], int] | None]:
        if self.size == 1:
            return local, None
        votes = [int(local == step) for step in _Step]
        tally = torch.tensor([*votes, *sizes.values()], dtype=torch.int64)
        dist.all_reduce(tally, op=dist.ReduceOp.SUM)
        votes, counts = tally.split((len(_Step), len(sizes)))
        step = max(step for step, votes in zip(_Step, votes.tolist()) if votes)
        if step != _Step.UPDATE:
            return step, None
        pooled = dict(zip(sizes, counts.tolist()))
        return step, {key: count for key, count in pooled.items() if count > 0}

    def _average(self, flat: Tensor) -> None:
        dist.all_reduce(flat, op=dist.ReduceOp.AVG if self.use_avg else dist.ReduceOp.SUM)
        if not self.use_avg:
            flat.div_(self.size)

    def average_gradients(self, parameters: tuple[Tensor, ...]) -> None:
        if self.size == 1:
            return
        gradients = [parameter.grad for parameter in parameters]
        if any(grad is None for grad in gradients):
            raise RuntimeError("distributed learning requires a gradient for every parameter")
        flat = torch.cat([grad.reshape(-1) for grad in gradients])
        self._average(flat)
        for parameter, grad in zip(parameters, flat.split([p.numel() for p in parameters])):
            parameter.grad = grad.view_as(parameter)

    @torch.no_grad()
    def average_parameters(self, model: GameTransformer) -> None:
        parameters = tuple(model.parameters())
        flat = torch.cat([parameter.reshape(-1) for parameter in parameters])
        if flat.dtype != torch.float32:
            raise RuntimeError("parameter averaging requires fp32 weights")
        self._average(flat)
        for parameter, average in zip(parameters, flat.split([p.numel() for p in parameters])):
            parameter.copy_(average.view_as(parameter))


class Learner:
    def __init__(self, online: GameTransformer, config: TrainConfig, group: _LearnerGroup = _LearnerGroup()) -> None:
        self.config = config
        self.group = group
        self.device = torch.device(config.device)
        if group.size > 1 and self.device.type != "cpu":
            raise ValueError("distributed learning requires a CPU learner")
        self.online = online.to(self.device, dtype=torch.float32)
        self.anchor = copy.deepcopy(self.online).eval().requires_grad_(False)
        self._step_rng = torch.Generator()
        self._parameters = tuple(self.online.parameters())
        self.optimizer = torch.optim.AdamW(
            self._parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        self.version = 0

    @torch.no_grad()
    def synchronize(self) -> None:
        if self.group.size > 1:
            for model in (self.online, self.anchor):
                for parameter in model.parameters():
                    dist.broadcast(parameter, src=0)

    def refresh_anchor(self) -> None:
        self.anchor.load_state_dict(self.online.state_dict())

    def update(self, batch: TrajectoryBatch, *, skip_grad_sync: bool = False) -> LearnStats:
        config, device = self.config, self.device
        self._step_rng.manual_seed(config.seed + self.version * self.group.size + self.group.rank)
        sample = batch.sample_steps(config.unroll_steps, self._step_rng)
        tokens = TokenBatch(*(tensor.to(device) for tensor in vars(sample.tokens(batch.tokens)).values()))
        action, behavior, expert, forced = (
            sample.gather(tensor).to(device)
            for tensor in (batch.action, batch.behavior_logp, batch.target_probs, batch.forced)
        )
        weights = sample.weight.to(device)
        value = batch.behavior_value
        targets, advantage = vtrace(
            torch.zeros_like(value), batch.reward, batch.discount,
            value, value.new_zeros(value.size(0)), rho_bar=1.0, c_bar=config.c_bar,
        )
        targets, advantage = (sample.gather(tensor).to(device) for tensor in (targets, advantage))
        self.online.train()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
            heads = self.online.unroll(tokens)
            with torch.no_grad():
                log_anchor = self.anchor.unroll(tokens).log_policy
        legal, log_pi = tokens.legal, heads.log_policy
        taken = log_pi.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        ratio = taken.detach() - behavior
        rho = ratio.clamp(max=math.log(config.rho_bar)).exp()
        probabilities = log_pi.exp()
        divergence = (probabilities * (log_pi - log_anchor).masked_fill(~legal, 0)).sum(-1)
        masked_log_pi = log_pi.masked_fill(~legal, 0)
        entropy = -(probabilities * masked_log_pi).sum(-1)

        def per_trajectory(x: Tensor) -> Tensor:
            return (x.masked_fill(weights == 0, 0) * weights).sum(dim=1).mean()

        kl = per_trajectory(divergence)
        entropy = per_trajectory(entropy)
        ce_step = -(expert * masked_log_pi).sum(-1)
        pg_step = -(rho * advantage * taken)
        policy_loss = per_trajectory(torch.where(forced, ce_step, pg_step))
        value_loss = 0.5 * per_trajectory((targets - heads.value).square())
        loss = policy_loss + config.value_weight * value_loss + config.eta * kl - config.entropy_weight * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not skip_grad_sync:
            self.group.average_gradients(self._parameters)
        norm = torch.nn.utils.clip_grad_norm_(self._parameters, config.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.version += 1
        if not skip_grad_sync and self.version % config.anchor_every == 0:
            self.refresh_anchor()
        return LearnStats(
            float(loss.detach()), float(policy_loss.detach()), float(value_loss.detach()),
            float(entropy.detach()), float(per_trajectory(ratio.double().exp())), float(kl.detach()), float(norm),
            int(sample.weight.count_nonzero()),
        )

    def state_dict(self) -> dict:
        return {
            "version": self.version, "online": self.online.state_dict(),
            "anchor": self.anchor.state_dict(), "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or set(state) != {"version", "online", "anchor", "optimizer"}:
            raise ValueError("invalid learner state fields")
        if type(state["version"]) is not int or state["version"] < 0:
            raise ValueError("invalid learner version")
        for name, model in (("online", self.online), ("anchor", self.anchor)):
            model.load_state_dict(validate_model_state(state[name], model.state_dict(), name))
        self.optimizer.load_state_dict(state["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        self.version = state["version"]


class Trainer:
    def __init__(self, root: Path, config: TrainConfig, learner: Learner, run_id: str) -> None:
        self.root, self.config, self.learner, self.run_id = root, config, learner, run_id
        self.group = learner.group
        self.rng = Random(config.seed + self.group.rank)
        self.actor_root = root / f"rank-{self.group.rank}" if self.group.size > 1 else root
        self.inbox = self.actor_root / "inbox"
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.model_path = (root if self.group.rank == 0 else self.actor_root) / "latest-model.pt"
        self.buffer = _Buffer(config.buffer_decisions, config.max_policy_lag)
        self.episodes, self.decisions, self.trained_decisions, self.dropped_batches = 0, 0, 0, 0
        self._group_progress: dict[str, int] = {}
        self.validation, self.last_update = {}, {}
        self.curriculum_history: list[dict] = []
        self._start = time.monotonic()
        self._base_version, self._base_decisions = learner.version, 0
        self._base_trained_decisions = 0

    @classmethod
    def open(
        cls, root: Path, overrides: Mapping[str, object] | None = None, *,
        spec: ModelSpec | None = None, initial: Path | None = None,
    ) -> Trainer:
        group = _LearnerGroup.open()
        try:
            return cls._open(Path(root), overrides or {}, spec, initial, group)
        except BaseException:
            group.close()
            raise

    @classmethod
    def _open(
        cls, root: Path, overrides: Mapping[str, object], spec: ModelSpec | None, initial: Path | None,
        group: _LearnerGroup,
    ) -> Trainer:
        path = root / "latest-state.pt"
        state = checkpoint = None
        if group.rank == 0:
            if path.exists():
                state = torch.load(path, map_location="cpu", weights_only=True)
            elif initial is not None:
                checkpoint = load_checkpoint(initial, SCHEMA)
        bootstrap = None
        if group.rank == 0:
            initial_info = (checkpoint.game, checkpoint.network.spec.to_dict()) if checkpoint else None
            bootstrap = (str(uuid4()), state, initial_info)
        run, state, initial_info = group.share(bootstrap)
        if state is not None:
            if (
                not isinstance(state, dict) or state.get("format") != STATE_VERSION
                or state.get("schema") != SCHEMA.to_dict()
            ):
                raise ValueError("incompatible trainer state; start a fresh GameTransformer run")
            if spec is not None or initial is not None:
                raise ValueError("model shape and initial weights only apply to fresh runs")
            saved = TrainConfig.from_dict(state["config"])
            config = saved.with_overrides(overrides)
            if any(getattr(config, k) != getattr(saved, k) for k in overrides.keys() - TUNABLE):
                raise ValueError("cannot change training config on resume")
            if config.plan != replace(saved.plan, **{key: getattr(config.plan, key) for key in TUNABLE_PLAN}):
                raise ValueError("cannot change the collection plan on resume except players, curriculum share, and distance")
            spec = ModelSpec.from_dict(state["spec"])
            run = _run_id(state["run_id"])
        else:
            if spec is not None and initial_info is not None:
                raise ValueError("use an initial model or a model shape, not both")
            if initial_info is not None:
                initial_game, initial_spec = initial_info
                game = GameConfig.from_dict(initial_game)
                if replace(game, **overrides.get("game", {})) != game:
                    raise ValueError("initial model game config mismatch")
                overrides = {**overrides, "game": game.to_dict()}
                spec = ModelSpec.from_dict(initial_spec)
            if overrides.get("auto_curriculum"):
                plan = {"curriculum_share": MASTERY_LADDER[1][0], "curriculum_max_distance": 1}
                overrides = {**overrides, "plan": {**plan, **overrides.get("plan", {})}}
            config = TrainConfig().with_overrides(overrides)
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        online = checkpoint.network if checkpoint is not None else GameTransformer(spec or ModelSpec(), SCHEMA)
        trainer = cls(root, config, Learner(online, config, group), run)
        if state is not None:
            trainer.learner.load_state_dict(state["learner"])
            if group.rank == 0:
                if not (root / "best-model.pt").is_file():
                    raise ValueError("missing best-model.pt on resume")
                trainer.rng.setstate(state["rng"])
                trainer.episodes, trainer.decisions = state["episodes"], state["decisions"]
                trainer.trained_decisions = state["trained_decisions"]
                trainer.dropped_batches, trainer.validation = state["dropped_batches"], state["validation"]
                trainer.curriculum_history = state["curriculum_history"]
        trainer.learner.synchronize()
        trainer._sync_progress()
        if state is None and group.rank == 0:
            trainer.export_model(root / "best-model.pt")
            trainer._save()
        trainer.export_model(trainer.model_path)
        if group.rank == 0:
            trainer._status("ready")
        return trainer

    def export_model(self, path: Path) -> None:
        save_checkpoint(path, self.learner.online, self.config.game.to_dict(), {
            "version": self.learner.version, "plan": self.config.plan.to_dict(), "run_id": self.run_id,
        })

    def _save(self) -> None:
        progress = self._group_progress or self._local_progress()
        atomic_torch_save({
            "format": STATE_VERSION, "schema": SCHEMA.to_dict(), "run_id": self.run_id,
            "spec": self.learner.online.spec.to_dict(),
            "config": asdict(self.config), "learner": self.learner.state_dict(), "rng": self.rng.getstate(),
            **{name: progress[name] for name in ("episodes", "decisions", "trained_decisions", "dropped_batches")},
            "validation": self.validation, "curriculum_history": self.curriculum_history,
        }, self.root / "latest-state.pt")

    def evaluate(self, *, stop: Callable[[], bool] = lambda: False) -> dict:
        cfg = self.config
        candidate = TransformerPolicy(self.learner.online, cfg.game, precision=cfg.precision)
        if cfg.auto_curriculum:
            return self._evaluate_curriculum(candidate, stop)
        checkpoint = load_checkpoint(self.root / "best-model.pt", SCHEMA, self.learner.device)
        best = TransformerPolicy(checkpoint.network, GameConfig.from_dict(checkpoint.game), precision=cfg.precision)
        validation = {}
        for name, opponent in (("best", best), ("random", RandomPolicy())):
            validation[name] = [
                asdict(head_to_head(
                    candidate, opponent, cfg.game, players, cfg.eval_deals, VALIDATION_SEED + players * cfg.eval_deals,
                    tables=cfg.tables, stop=stop,
                ))
                for players in cfg.plan.player_counts
            ]
        scores = [row["advantage"] for row in validation["best"]]
        promote = sum(scores) / len(scores) >= cfg.promote_margin and min(scores) >= -cfg.promote_margin
        self.validation = {**validation, "promoted": promote, "version": self.learner.version}
        if promote:
            self._save()
            self.export_model(self.root / "best-model.pt")
        return self.validation

    def _evaluate_curriculum(self, candidate: TransformerPolicy, stop: Callable[[], bool]) -> dict:
        cfg, plan = self.config, self.config.plan
        distance = plan.curriculum_max_distance
        cold = distance == cfg.game.cards_in_hand
        threshold = None if cold else MASTERY_LADDER[distance][1]
        seed = VALIDATION_SEED + distance * 100_000
        deals = [Deal(seed + i, (0,), None if cold else distance) for i in range(48)]
        results = list(run_matches(deals, (candidate,), cfg.game, tables=cfg.tables, stop=stop))
        if len(results) != len(deals):
            raise InterruptedError("mastery evaluation stopped before completing the frontier deals")
        wins = sum(result.winner == 0 for result in results)
        rate = wins / len(deals)
        mastered = threshold is not None and rate >= threshold
        next_plan = plan
        if mastered:
            next_distance = min((step for step in MASTERY_LADDER if distance < step < cfg.game.cards_in_hand),
                                default=cfg.game.cards_in_hand)
            share = MASTERY_LADDER[next_distance][0] if next_distance < cfg.game.cards_in_hand else 0.45
            next_plan = replace(plan, curriculum_max_distance=next_distance, curriculum_share=share)
        check = {
            "distance": distance, "cold": cold, "wins": wins, "deals": len(deals), "seed": seed,
            "win_rate": rate, "threshold": threshold, "search_ply": candidate.search_ply,
            "mean_qualifying_sequences": sum(
                result.outcome.penalties[0].qualifying_sequences for result in results
            ) / len(results),
            "next_distance": next_plan.curriculum_max_distance, "next_share": next_plan.curriculum_share,
            "version": self.learner.version,
        }
        self.validation = {"curriculum": check, "promoted": mastered, "version": self.learner.version}
        if mastered:
            self.config = replace(cfg, plan=next_plan)
            self.curriculum_history.append(check)
            self.export_model(self.root / "best-model.pt")
        return self.validation

    def _ingest(self) -> None:
        version = self.learner.version
        self.buffer.prune(version)
        files = sorted(self.inbox.glob("batch-*.pt"), key=lambda path: path.stat().st_mtime_ns)
        for path in files[:max(1, self.config.actors)]:
            try:
                batch = Trajectories.load(path)
                lengths = batch.decision_end.diff(prepend=batch.decision_end.new_zeros(1))
                if (
                    batch.run_id != self.run_id or int(lengths.max()) > 2 * self.config.game.max_turns
                    or any(players not in self.config.plan.player_counts for players in batch.players.unique().tolist())
                ):
                    raise ValueError("batch does not belong to this run's collection plan")
            except ValueError:
                rejected = self.actor_root / "rejected"
                rejected.mkdir(exist_ok=True)
                os.replace(path, rejected / path.name)
                self.dropped_batches += 1
                continue
            if self.buffer.add(batch, version):
                self.episodes += batch.episodes
                self.decisions += batch.decisions
            else:
                self.dropped_batches += 1
            path.unlink()

    def _local_progress(self) -> dict[str, int]:
        return {
            "episodes": self.episodes, "decisions": self.decisions, "trained_decisions": self.trained_decisions,
            "dropped_batches": self.dropped_batches, "buffer_decisions": len(self.buffer),
            "inbox_files": sum(1 for _ in self.inbox.glob("batch-*.pt")),
        }

    def _sync_progress(self) -> None:
        if self.group.size > 1:
            progress = self._local_progress()
            totals = torch.tensor(list(progress.values()), dtype=torch.int64)
            dist.all_reduce(totals)
            self._group_progress = dict(zip(progress, totals.tolist()))

    def summary(self) -> dict:
        progress = self._group_progress or self._local_progress()
        elapsed = max(time.monotonic() - self._start, 1e-6)
        version = self.learner.version
        return {
            "version": version, "run_id": self.run_id, **progress,
            "decisions_per_sec": (progress["decisions"] - self._base_decisions) / elapsed,
            "trained_decisions_per_sec": (progress["trained_decisions"] - self._base_trained_decisions) / elapsed,
            "updates_per_sec": (version - self._base_version) / elapsed, "elapsed_seconds": elapsed,
            "validation": self.validation, "world_size": self.group.size,
            "curriculum_history": self.curriculum_history, "plan": self.config.plan.to_dict(),
            "actors": self.config.actors, "tables": self.config.tables, "torch_threads": self.config.torch_threads,
            "actor_threads": self.config.actor_threads, "cpu_count": os.cpu_count(), "sync_every": self.config.sync_every,
            "device": self.config.device, "precision": self.config.precision,
            "mean_rho": None, "kl_anchor": None, **self.last_update,
        }

    def _status(self, status: str) -> None:
        atomic_write_json({"status": status, **self.summary()}, self.root / "status.json")

    def run(
        self, *, stop: Callable[[], bool] = lambda: False, max_updates: int | None = None,
        max_seconds: float | None = None,
    ) -> str:
        try:
            return self._run(stop, max_updates, max_seconds)
        finally:
            self.group.close()

    def _run(self, stop: Callable[[], bool], max_updates: int | None, max_seconds: float | None) -> str:
        if max_updates is not None and (type(max_updates) is not int or max_updates < 0):
            raise ValueError("max_updates must be a nonnegative integer")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite")
        cfg = self.config
        primary = self.group.rank == 0
        local_steps = cfg.sync_every if self.group.size > 1 else 1
        post_local = local_steps > 1
        if local_steps > cfg.max_policy_lag + 1:
            raise ValueError("sync_every cannot exceed max_policy_lag + 1 for distributed learning")
        progress = self._group_progress or self._local_progress()
        self._start = time.monotonic()
        self._base_version, self._base_decisions = self.learner.version, progress["decisions"]
        self._base_trained_decisions = progress["trained_decisions"]
        deadline = math.inf if max_seconds is None else self._start + max_seconds
        context = mp.get_context("spawn")
        done = context.Event()
        workers = []
        metrics = None
        reason = "error"
        eval_error = None
        max_bucket = (2 * cfg.game.max_turns - 1) // LENGTH_BUCKET
        bucket_keys = tuple((players, length) for players in cfg.plan.player_counts for length in range(max_bucket + 1))

        def stopping() -> bool:
            return stop() or time.monotonic() >= deadline

        def next_step() -> _Step:
            if eval_error is not None:
                return _Step.EVAL_ERROR
            if max_updates is not None and self.learner.version >= max_updates:
                return _Step.MAX_UPDATES
            if stop():
                return _Step.STOPPED
            if time.monotonic() >= deadline:
                return _Step.BUDGET
            if any(worker.exitcode is not None for worker in workers):
                return _Step.ACTOR_ERROR
            self._ingest()
            return _Step.UPDATE if len(self.buffer) >= cfg.min_buffer_decisions else _Step.WAIT

        try:
            if not stopping() and (max_updates is None or self.learner.version < max_updates):
                for worker in range(cfg.actors):
                    process = context.Process(
                        target=_actor_process,
                        args=(self.inbox, self.model_path, worker, cfg.tables, cfg.actor_threads, cfg.precision, done),
                    )
                    process.start()
                    workers.append(process)
            heartbeat = 0.0
            if primary:
                metrics = (self.root / "metrics.jsonl").open("a", encoding="utf-8")
            while True:
                local = next_step()
                before = self.learner.version
                steps = local_steps if max_updates is None else min(local_steps, max_updates - before)
                sizes = {}
                if self.group.size > 1:
                    min_version = before + steps - 1 - cfg.max_policy_lag
                    available = self.buffer.bucket_sizes(min_version) if local == _Step.UPDATE else {}
                    starting = workers and self.learner.version == self._base_version
                    if local == _Step.UPDATE and (
                        not available or (starting and {players for players, _ in available} != set(cfg.plan.player_counts))
                    ):
                        local = _Step.WAIT
                    sizes = {key: available.get(key, 0) for key in bucket_keys}
                step, targets = self.group.step(local, sizes)
                if step == _Step.EVAL_ERROR:
                    raise RuntimeError("evaluation failed on rank 0") from eval_error
                if step == _Step.ACTOR_ERROR:
                    raise RuntimeError("an actor worker exited before the learner stopped")
                if step > _Step.WAIT:
                    self._sync_progress()
                    reason = step.name.lower()
                    break
                if step == _Step.UPDATE:
                    for version in range(before, before + steps):
                        rng = Random(f"{cfg.seed}:{self.group.rank}:{version}") if post_local else self.rng
                        batch = self.buffer.sample(rng, cfg.batch_size, version, seed=cfg.seed, targets=targets)
                        stats = self.learner.update(batch, skip_grad_sync=post_local)
                        self.last_update = asdict(stats)
                        self.trained_decisions += stats.trained_steps
                    version = self.learner.version
                    self.buffer.prune(version)
                    if post_local:
                        self.group.average_parameters(self.learner.online)
                        if version // cfg.anchor_every > before // cfg.anchor_every:
                            self.learner.refresh_anchor()
                self._sync_progress()
                if step == _Step.UPDATE:
                    publish = version // cfg.publish_every > before // cfg.publish_every
                    previous_plan = self.config.plan
                    if cfg.eval_every and version // cfg.eval_every > before // cfg.eval_every:
                        if self.group.size > 1:
                            dist.barrier()
                        if primary and not stopping():
                            try:
                                self.evaluate(stop=stopping)
                            except Exception as exc:
                                if not isinstance(exc, InterruptedError) or not stopping():
                                    eval_error = exc
                        if cfg.auto_curriculum:
                            shared = self.group.share(self.config.plan.to_dict() if primary else None)
                            if not primary:
                                self.config = replace(self.config, plan=CollectPlan.from_dict(shared))
                        elif self.group.size > 1:
                            dist.barrier()
                    publish |= self.config.plan != previous_plan
                    if publish:
                        self.export_model(self.model_path)
                    if primary:
                        if publish or version // cfg.checkpoint_every > before // cfg.checkpoint_every:
                            self._save()
                        metrics.write(json.dumps(self.summary(), allow_nan=False) + "\n")
                        metrics.flush()
                if primary and time.monotonic() >= heartbeat:
                    self._status("running" if step == _Step.UPDATE else "waiting_for_actors")
                    heartbeat = time.monotonic() + 1
                if step == _Step.WAIT:
                    time.sleep(0.05)
        finally:
            if metrics is not None:
                metrics.close()
            done.set()
            grace = time.monotonic() + 5.0
            for worker in workers:
                worker.join(max(0.0, grace - time.monotonic()))
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                worker.close()
            if reason != "error":
                self.export_model(self.model_path)
                if primary:
                    self._save()
            if primary:
                self._status(reason)
        return reason


def _actor_process(
    inbox: Path, model: Path, worker: int, tables: int, threads: int, precision: str, done: object,
) -> None:
    collect(
        inbox, model, worker=worker, tables=tables, torch_threads=threads, precision=precision, stop=done.is_set,
    )
