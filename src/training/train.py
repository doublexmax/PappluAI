from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass, field, replace
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


STATE_VERSION = 14
ROLLOUT_FORMAT = 2
VALIDATION_SEED = 30_000_001
OUTBOX_LIMIT = 8
LENGTH_BUCKET = 8
DISCOUNT = 0.995
TUNABLE = {"actors", "tables", "torch_threads", "actor_threads", "device", "precision", "checkpoint_every"}


@dataclass(frozen=True)
class CollectPlan:
    player_counts: tuple[int, ...] = (1, 2, 3, 4)
    curriculum_share: float = 0.50
    curriculum_max_distance: int = 6
    uniform_mix: float = 0.02
    batch_decisions: int = 8192

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

    def validate(self, game: GameConfig) -> None:
        validate_game(game)
        for players in self.player_counts:
            if players > 1:
                MatchConfig(game, players)
        if 1 in self.player_counts and self.curriculum_share and self.curriculum_max_distance > game.cards_in_hand:
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
                start = rng.randint(1, self.curriculum_max_distance) if players == 1 and rng.random() < self.curriculum_share else 0
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
    batch_size: int = 16
    buffer_decisions: int = 400_000
    min_buffer_decisions: int = 20_000
    max_policy_lag: int = 200
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    value_weight: float = 0.5
    rho_bar: float = 1.0
    c_bar: float = 1.0
    eta: float = 0.05
    entropy_weight: float = 0.003
    anchor_every: int = 2000
    publish_every: int = 25
    checkpoint_every: int = 500
    eval_every: int = 2000
    eval_deals: int = 64
    promote_margin: float = 0.02

    def __post_init__(self) -> None:
        self.plan.validate(self.game)
        for name in (
            "tables", "torch_threads", "actor_threads", "batch_size", "buffer_decisions",
            "min_buffer_decisions", "max_policy_lag", "anchor_every", "publish_every", "checkpoint_every", "eval_deals",
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


def _bytes_tensor(values: bytes, shape: tuple[int, ...]) -> Tensor:
    return torch.frombuffer(bytearray(values), dtype=torch.uint8).reshape(shape)


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
        )

    def save(self, path: Path) -> None:
        atomic_torch_save({"format": ROLLOUT_FORMAT, "schema": SCHEMA.to_dict(), **vars(self)}, Path(path))

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
            ("behavior_logp", torch.float32, ()),
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
        if any(len(col) != decisions for col in (self.suffix, self.cut, self.legal, self.behavior_logp)):
            raise ValueError("invalid rollout decision column lengths")
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
        if bool((self.suffix[:, 0, 0] == 0).any()):
            raise ValueError("rollout has a padded readout token")
        if not bool(((self.behavior_logp >= -104) & (self.behavior_logp <= 0)).all()):
            raise ValueError("invalid rollout behavior log-probabilities")
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
                               precision=precision, record=True)
    del checkpoint
    active: Counter[int] = Counter()
    admitted: dict[int, int] = {}
    pending: dict[int, _Completed] = {}
    incarnation = uuid4().hex
    totals = {"batches": 0, "episodes": 0, "decisions": 0}

    def admit(deal: Deal) -> tuple[TransformerPolicy, ...]:
        nonlocal stamp, revision, policy
        if len(active) <= 1:
            new_stamp = fingerprint()
            if new_stamp != stamp:
                snapshot = load_checkpoint(model_path, SCHEMA, device)
                if (
                    snapshot.meta.get("run_id") != run or snapshot.game != game.to_dict()
                    or CollectPlan.from_dict(snapshot.meta.get("plan")) != plan or snapshot.network.spec != spec
                ):
                    raise ValueError("collection checkpoint changed run, game, plan, or model shape")
                if snapshot.version > revision:
                    revision = snapshot.version
                    policy = TransformerPolicy(snapshot.network, game, sample=True, uniform_mix=plan.uniform_mix,
                                               precision=precision, record=True)
                stamp = new_stamp
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

    queue = plan.deals(f"{run}:{worker}:{incarnation}")
    stopping = lambda: stop() or at_limit()
    for result in run_matches(queue, admit, game, tables=tables, stop=stopping):
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


@dataclass(eq=False)
class _SeatTrajectory:
    prefix: Tensor
    suffix: Tensor
    cut: Tensor
    legal: Tensor
    action: Tensor
    behavior_logp: Tensor
    utility: Tensor
    anchor: tuple[object, Tensor] | None = field(default=None, init=False)


@dataclass(frozen=True)
class TrajectoryBatch:
    seats: tuple[_SeatTrajectory, ...]
    tokens: TokenBatch
    action: Tensor
    behavior_logp: Tensor
    reward: Tensor
    discount: Tensor
    valid: Tensor

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
        tokens = TokenBatch(pack("prefix"), pack("suffix"), pack("cut").long(),
                            action_mask(pack("legal"), SCHEMA.num_actions))
        return cls(seats, tokens, action, pack("behavior_logp"), reward, discount, valid)


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

    def sample(self, rng: Random, batch_size: int, version: int) -> TrajectoryBatch:
        self.prune(version)
        buckets: dict[tuple[int, int], list[_SeatTrajectory]] = {}
        for chunk in self.chunks:
            for key, seats in chunk.buckets.items():
                buckets.setdefault(key, []).extend(seats)
        if not buckets:
            raise ValueError("cannot sample an empty buffer")
        groups = list(buckets.values())
        group = rng.choices(groups, weights=[len(seats) for seats in groups], k=1)[0]
        return TrajectoryBatch.pack(tuple(rng.choices(group, k=batch_size)))


def regularized_reward(reward: Tensor, log_pi: Tensor, log_anchor: Tensor, eta: float) -> Tensor:
    return reward.detach().float() - eta * (log_pi.detach().float() - log_anchor.detach().float())


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
    correction = torch.zeros_like(bootstrap)
    vs = torch.empty_like(value)
    for t in range(value.size(1) - 1, -1, -1):
        correction = delta[:, t] + discount[:, t] * c[:, t] * correction
        vs[:, t] = value[:, t] + correction
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


class Learner:
    def __init__(self, online: GameTransformer, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.online = online.to(self.device, dtype=torch.float32)
        self.anchor = copy.deepcopy(self.online).eval().requires_grad_(False)
        self._anchor_version = object()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        self.version = 0

    @torch.no_grad()
    def _anchor_log_policy(self, batch: TrajectoryBatch, tokens: TokenBatch) -> Tensor:
        missing: dict[_SeatTrajectory, int] = {}
        for i, seat in enumerate(batch.seats):
            if seat.anchor is None or seat.anchor[0] is not self._anchor_version:
                missing.setdefault(seat, i)
        if missing:
            if len(missing) != len(batch.seats):
                indices = torch.tensor(list(missing.values()), device=self.device)
                tokens = TokenBatch(*(column.index_select(0, indices) for column in vars(tokens).values()))
            ref = self.anchor.unroll(tokens).log_policy
            for seat, logp in zip(missing, ref):
                seat.anchor = (self._anchor_version, logp[:seat.action.numel()].clone())
        return pad_sequence([seat.anchor[1] for seat in batch.seats], batch_first=True)

    def update(self, batch: TrajectoryBatch) -> LearnStats:
        config, device = self.config, self.device
        tokens = TokenBatch(*(tensor.to(device) for tensor in vars(batch.tokens).values()))
        action, behavior, reward, discount, valid = (
            tensor.to(device) for tensor in (batch.action, batch.behavior_logp, batch.reward, batch.discount, batch.valid)
        )
        self.online.train()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
            heads = self.online.unroll(tokens)
            log_anchor = self._anchor_log_policy(batch, tokens)
        legal, log_pi = tokens.legal, heads.log_policy
        taken = log_pi.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        anchor_taken = log_anchor.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        ratio = taken.detach() - behavior
        values = heads.value
        targets, advantage = vtrace(
            ratio, regularized_reward(reward, taken, anchor_taken, config.eta), discount,
            values, values.new_zeros(values.size(0)), rho_bar=config.rho_bar, c_bar=config.c_bar,
        )
        probabilities = log_pi.exp()
        divergence = (probabilities * (log_pi - log_anchor).masked_fill(~legal, 0)).sum(-1)
        entropy = -(probabilities * log_pi.masked_fill(~legal, 0)).sum(-1)
        kl = divergence[valid].mean()
        entropy = entropy[valid].mean()
        policy_loss = -(advantage * taken)[valid].mean()
        value_loss = 0.5 * (targets - values).square()[valid].mean()
        loss = policy_loss + config.value_weight * value_loss + config.eta * kl - config.entropy_weight * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), config.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.version += 1
        if self.version % config.anchor_every == 0:
            self.anchor.load_state_dict(self.online.state_dict())
            self._anchor_version = object()
        return LearnStats(
            float(loss.detach()), float(policy_loss.detach()), float(value_loss.detach()),
            float(entropy.detach()), float(ratio[valid].double().exp().mean()), float(kl.detach()), float(norm),
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
        self.version = state["version"]
        self._anchor_version = object()


class Trainer:
    def __init__(self, root: Path, config: TrainConfig, learner: Learner, run_id: str) -> None:
        self.root, self.config, self.learner, self.run_id = root, config, learner, run_id
        self.rng = Random(config.seed)
        self.inbox = root / "inbox"
        self.inbox.mkdir(exist_ok=True)
        self.buffer = _Buffer(config.buffer_decisions, config.max_policy_lag)
        self.episodes, self.decisions, self.trained_decisions, self.dropped_batches = 0, 0, 0, 0
        self.validation, self.last_update = {}, {}
        self._start = time.monotonic()
        self._base_version, self._base_decisions = learner.version, 0
        self._base_trained_decisions = 0

    @classmethod
    def open(
        cls, root: Path, overrides: Mapping[str, object] | None = None, *,
        spec: ModelSpec | None = None, initial: Path | None = None,
    ) -> Trainer:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "latest-state.pt"
        state = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
        overrides = overrides or {}
        if state is not None:
            if not isinstance(state, dict) or state.get("format") != STATE_VERSION:
                raise ValueError("incompatible trainer state; start a fresh GameTransformer run")
            if spec is not None or initial is not None:
                raise ValueError("model shape and initial weights only apply to fresh runs")
            saved = TrainConfig.from_dict(state["config"])
            config = saved.with_overrides(overrides)
            if any(getattr(config, k) != getattr(saved, k) for k in overrides.keys() - TUNABLE):
                raise ValueError("cannot change training config on resume")
            spec = ModelSpec.from_dict(state["spec"])
            run = _run_id(state["run_id"])
        else:
            if spec is not None and initial is not None:
                raise ValueError("use an initial model or a model shape, not both")
            run = str(uuid4())
            if initial is not None:
                checkpoint = load_checkpoint(initial, SCHEMA)
                game = GameConfig.from_dict(checkpoint.game)
                if replace(game, **overrides.get("game", {})) != game:
                    raise ValueError("initial model game config mismatch")
                overrides = {**overrides, "game": game.to_dict()}
            config = TrainConfig().with_overrides(overrides)
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        online = checkpoint.network if initial is not None else GameTransformer(spec or ModelSpec(), SCHEMA)
        trainer = cls(root, config, Learner(online, config), run)
        if state is not None:
            trainer.learner.load_state_dict(state["learner"])
            trainer.rng.setstate(state["rng"])
            trainer.episodes, trainer.decisions = state["episodes"], state["decisions"]
            trainer.trained_decisions = state["trained_decisions"]
            trainer.dropped_batches, trainer.validation = state["dropped_batches"], state["validation"]
            trainer._base_version, trainer._base_decisions = trainer.learner.version, trainer.decisions
            if not (root / "best-model.pt").is_file():
                raise ValueError("missing best-model.pt on resume")
        else:
            trainer.export_model(root / "best-model.pt")
            trainer._save()
        trainer.export_model(root / "latest-model.pt")
        trainer._status("ready")
        return trainer

    def export_model(self, path: Path) -> None:
        save_checkpoint(path, self.learner.online, self.config.game.to_dict(), {
            "version": self.learner.version, "plan": self.config.plan.to_dict(), "run_id": self.run_id,
        })

    def _save(self) -> None:
        atomic_torch_save({
            "format": STATE_VERSION, "run_id": self.run_id, "spec": self.learner.online.spec.to_dict(),
            "config": asdict(self.config), "learner": self.learner.state_dict(), "rng": self.rng.getstate(),
            "episodes": self.episodes, "decisions": self.decisions,
            "trained_decisions": self.trained_decisions,
            "dropped_batches": self.dropped_batches, "validation": self.validation,
        }, self.root / "latest-state.pt")

    def evaluate(self, *, stop: Callable[[], bool] = lambda: False) -> dict:
        cfg = self.config
        candidate = TransformerPolicy(self.learner.online, cfg.game, precision=cfg.precision)
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
                rejected = self.root / "rejected"
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

    def summary(self) -> dict:
        elapsed = max(time.monotonic() - self._start, 1e-6)
        version = self.learner.version
        return {
            "version": version, "run_id": self.run_id, "episodes": self.episodes, "decisions": self.decisions,
            "decisions_per_sec": (self.decisions - self._base_decisions) / elapsed,
            "trained_decisions": self.trained_decisions,
            "trained_decisions_per_sec": (self.trained_decisions - self._base_trained_decisions) / elapsed,
            "updates_per_sec": (version - self._base_version) / elapsed, "elapsed_seconds": elapsed,
            "buffer_decisions": len(self.buffer), "inbox_files": sum(1 for _ in self.inbox.glob("batch-*.pt")),
            "dropped_batches": self.dropped_batches, "validation": self.validation,
            "actors": self.config.actors, "tables": self.config.tables, "torch_threads": self.config.torch_threads,
            "actor_threads": self.config.actor_threads, "cpu_count": os.cpu_count(),
            "device": self.config.device, "precision": self.config.precision,
            "mean_rho": None, "kl_anchor": None, **self.last_update,
        }

    def _status(self, status: str) -> None:
        atomic_write_json({"status": status, **self.summary()}, self.root / "status.json")

    def run(
        self, *, stop: Callable[[], bool] = lambda: False, max_updates: int | None = None,
        max_seconds: float | None = None,
    ) -> str:
        if max_updates is not None and (type(max_updates) is not int or max_updates < 0):
            raise ValueError("max_updates must be a nonnegative integer")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite")
        cfg = self.config
        self._start = time.monotonic()
        self._base_version, self._base_decisions = self.learner.version, self.decisions
        self._base_trained_decisions = self.trained_decisions
        deadline = math.inf if max_seconds is None else self._start + max_seconds
        context = mp.get_context("spawn")
        done = context.Event()
        workers = []
        reason = "error"

        def stopping() -> bool:
            return stop() or time.monotonic() >= deadline

        try:
            if not stopping() and (max_updates is None or self.learner.version < max_updates):
                for worker in range(cfg.actors):
                    process = context.Process(
                        target=_actor_process,
                        args=(self.root, worker, cfg.tables, cfg.actor_threads, cfg.precision, done),
                    )
                    process.start()
                    workers.append(process)
            heartbeat = 0.0
            with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
                while not stopping():
                    if max_updates is not None and self.learner.version >= max_updates:
                        reason = "max_updates"
                        break
                    if any(worker.exitcode is not None for worker in workers):
                        raise RuntimeError("an actor worker exited before the learner stopped")
                    self._ingest()
                    if len(self.buffer) < cfg.min_buffer_decisions:
                        if time.monotonic() >= heartbeat:
                            self._status("waiting_for_actors")
                            heartbeat = time.monotonic() + 1
                        time.sleep(0.05)
                        continue
                    batch = self.buffer.sample(self.rng, cfg.batch_size, self.learner.version)
                    self.last_update = asdict(self.learner.update(batch))
                    self.trained_decisions += int(batch.valid.sum())
                    version = self.learner.version
                    self.buffer.prune(version)
                    if version % cfg.checkpoint_every == 0 or version % cfg.publish_every == 0:
                        self._save()
                    if version % cfg.publish_every == 0:
                        self.export_model(self.root / "latest-model.pt")
                    if cfg.eval_every and version % cfg.eval_every == 0 and not stopping():
                        self.evaluate(stop=stopping)
                    metrics.write(json.dumps(self.summary(), allow_nan=False) + "\n")
                    metrics.flush()
                    if time.monotonic() >= heartbeat:
                        self._status("running")
                        heartbeat = time.monotonic() + 1
        except InterruptedError:
            if not stopping():
                raise
        finally:
            if reason == "error" and stopping():
                reason = "stopped" if stop() else "budget"
            done.set()
            grace = time.monotonic() + 5.0
            for worker in workers:
                worker.join(max(0.0, grace - time.monotonic()))
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
                worker.close()
            self._save()
            self.export_model(self.root / "latest-model.pt")
            self._status(reason)
        return reason


def _actor_process(root: Path, worker: int, tables: int, threads: int, precision: str, done: object) -> None:
    collect(
        root / "inbox", root / "latest-model.pt", worker=worker, tables=tables,
        torch_threads=threads, precision=precision, stop=done.is_set,
    )
