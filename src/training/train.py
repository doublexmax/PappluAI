from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import ctypes
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Callable, NamedTuple, Sequence
import zlib

import torch
import torch.nn.functional as F

from src.checkpoints.io import atomic_copy, atomic_write_json
from src.checkpoints.tensor import atomic_torch_save, clone_model_state, validate_model_state
from src.evaluation.arena import CheckpointPolicy, Deal, EpsilonPolicy, GreedyPolicy, HeuristicPolicy, RandomPolicy, run_matches
from src.evaluation.benchmark import evaluate_games
from src.game.environment import GameConfig, PappluEnv
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.game.outcome import DealOutcome, solo_outcome
from src.model.encoding import FEATURE_SIZE, LAYOUT, Features, encode
from src.model.network import ObservationBatch, QNetwork, checkpoint_payload, load_checkpoint
from src.model.spec import ModelSpec


STATE_VERSION = 12
VALIDATION_SEED = 30_000_001
ZLIB_CHUNK = 1 << 26


class Step(NamedTuple):
    features: Features
    action: int
    greedy: bool = True


class Trajectory(NamedTuple):
    steps: tuple[Step, ...]
    utility: float


def save_rollout_batch(path: Path, episodes: int, trajectories: Sequence[Trajectory]) -> None:
    atomic_torch_save({
        "version": STATE_VERSION, "episodes": episodes,
        "features": zlib.compress(bytearray().join(step.features.data for t in trajectories for step in t.steps), level=1),
        "records": tuple((tuple(s.action for s in t.steps), tuple(s.greedy for s in t.steps), float(t.utility))
                         for t in trajectories),
    }, path)


def load_rollout_batch(path: Path) -> tuple[int, list[Trajectory]]:
    batch = torch.load(path, map_location="cpu", weights_only=True)
    episodes, records = batch["episodes"], batch["records"]
    if batch.get("version") != STATE_VERSION or type(episodes) is not int or episodes < 1:
        raise ValueError("invalid rollout batch version or episode count")
    if any(len(actions) != len(greedy) for actions, greedy, _ in records):
        raise ValueError("rollout action and greedy lengths mismatch")
    try:
        raw = zlib.decompress(batch["features"])
    except zlib.error as exc:
        raise ValueError("corrupted rollout features") from exc
    size = FEATURE_SIZE * 4
    if len(raw) != sum(len(actions) for actions, _, _ in records) * size:
        raise ValueError("rollout byte length mismatch")
    features = iter(Features(raw[offset:offset + size]) for offset in range(0, len(raw), size))
    return episodes, [
        Trajectory(tuple(Step(next(features), action, flag) for action, flag in zip(actions, greedy)), utility)
        for actions, greedy, utility in records
    ]


@dataclass(frozen=True)
class TrainConfig:
    num_decks: int = 3
    cards_in_hand: int = 21
    required_sequences: int = 5
    max_turns: int = 60
    recycle_discard: bool = True
    player_counts: tuple[int, ...] = (1, 2, 3, 4)
    round_episodes: int = 32
    updates_per_episode: int = 2
    batch_size: int = 256
    replay_capacity: int = 60_000
    learning_rate: float = 4e-4
    lr_decay_updates: int = 20_000
    weight_decay: float = 1e-4
    gamma: float = 0.99
    margin_weight: float = 0.35
    grad_clip: float = 1.0
    epsilons: tuple[float, ...] = (0.01, 0.02, 0.05)
    actors: int = 1
    self_play_share: float = 0.35
    max_pool: int = 8
    validate_every: int = 1_000
    validation_games: int = 64
    checkpoint_every: int = 64
    seed: int = 41
    torch_threads: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        game = self.game
        counts = tuple(self.player_counts)
        if not counts or len(set(counts)) != len(counts) or any(type(c) is not int or not 1 <= c <= 6 for c in counts):
            raise ValueError("player_counts must contain distinct integers in 1..6")
        object.__setattr__(self, "player_counts", counts)
        for count in counts:
            if count > 1:
                MatchConfig(game, count)
        for name, low in (
            ("round_episodes", 1), ("updates_per_episode", 0), ("batch_size", 1),
            ("replay_capacity", 1), ("lr_decay_updates", 0), ("actors", 1),
            ("max_pool", 1), ("validate_every", 1), ("validation_games", 0),
            ("checkpoint_every", 1), ("seed", 0), ("torch_threads", 1),
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < low:
                raise ValueError("%s must be an integer >= %d" % (name, low))
        if not isinstance(self.epsilons, (tuple, list)) or not self.epsilons or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in self.epsilons
        ):
            raise ValueError("epsilons must contain finite numbers in [0, 1]")
        object.__setattr__(self, "epsilons", tuple(float(value) for value in self.epsilons))
        for name in ("learning_rate", "weight_decay", "gamma", "margin_weight", "grad_clip", "self_play_share"):
            value = getattr(self, name)
            high = 1.0 if name in ("gamma", "self_play_share") else math.inf
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= high:
                raise ValueError("%s is out of range" % name)
        if min(self.learning_rate, self.gamma, self.grad_clip) <= 0:
            raise ValueError("learning_rate, gamma and grad_clip must be positive")
        torch.device(self.device)

    @property
    def game(self) -> GameConfig:
        return GameConfig(self.num_decks, self.cards_in_hand, self.required_sequences, self.max_turns, self.recycle_discard)


def _compress_tensor(tensor: torch.Tensor) -> bytes:
    ptr, total = tensor.data_ptr(), tensor.numel() * tensor.element_size()
    compressor = zlib.compressobj(level=1)
    chunks = [compressor.compress(ctypes.string_at(ptr + offset, min(ZLIB_CHUNK, total - offset)))
              for offset in range(0, total, ZLIB_CHUNK)]
    chunks.append(compressor.flush())
    return b"".join(chunks)


def _decompress_tensor(payload: object, target: torch.Tensor) -> None:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("corrupted replay data")
    ptr, expected = target.data_ptr(), target.numel() * target.element_size()
    decompressor = zlib.decompressobj()
    written = 0

    def write(piece: bytes) -> None:
        nonlocal written
        size = len(piece)
        if written + size > expected:
            raise ValueError("replay byte length mismatch")
        if size:
            ctypes.memmove(ptr + written, piece, size)
            written += size

    try:
        data = memoryview(payload).cast("B")
        for offset in range(0, len(data), ZLIB_CHUNK):
            if decompressor.eof:
                raise ValueError("corrupted replay data")
            chunk: bytes | memoryview = data[offset:offset + ZLIB_CHUNK]
            while True:
                piece = decompressor.decompress(chunk, max_length=ZLIB_CHUNK)
                write(piece)
                if decompressor.eof or (not decompressor.unconsumed_tail and len(piece) < ZLIB_CHUNK):
                    break
                chunk = decompressor.unconsumed_tail
        write(decompressor.flush(ZLIB_CHUNK))
    except (TypeError, zlib.error) as exc:
        raise ValueError("corrupted replay data") from exc
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("corrupted replay data")
    if written != expected:
        raise ValueError("replay byte length mismatch")


class Replay:
    def __init__(self, capacity: int, gamma: float = 0.99) -> None:
        if type(capacity) is not int or capacity < 1 or not 0 < gamma <= 1:
            raise ValueError("replay needs positive capacity and gamma in (0, 1]")
        self.capacity, self.gamma, self.size, self.cursor = capacity, gamma, 0, 0
        self.features = torch.empty((capacity, FEATURE_SIZE), dtype=torch.float32, device="cpu")
        self.actions = torch.empty(capacity, dtype=torch.int64, device="cpu")
        self.returns = torch.empty(capacity, dtype=torch.float32, device="cpu")
        self.greedy = torch.empty(capacity, dtype=torch.bool, device="cpu")

    def __len__(self) -> int:
        return self.size

    def append(self, trajectory: Trajectory) -> int:
        steps, utility = trajectory
        length = len(steps)
        if not length:
            return 0
        kept = steps[-self.capacity:]
        data = torch.frombuffer(bytearray().join(step.features.data for step in kept), dtype=torch.float32)
        actions = torch.tensor([step.action for step in kept], dtype=torch.int64, device="cpu")
        returns = torch.tensor(
            [utility * self.gamma ** offset if utility > 0 else float(utility)
             for offset in range(len(kept) - 1, -1, -1)],
            dtype=torch.float32, device="cpu",
        )
        greedy = torch.tensor([step.greedy for step in kept], dtype=torch.bool, device="cpu")
        start = (self.cursor + length - len(kept)) % self.capacity
        first = min(len(kept), self.capacity - start)
        for target, source in (
            (self.features, data.reshape(-1, FEATURE_SIZE)), (self.actions, actions),
            (self.returns, returns), (self.greedy, greedy),
        ):
            target[start:start + first].copy_(source[:first])
            target[:len(kept) - first].copy_(source[first:])
        self.cursor = (self.cursor + length) % self.capacity
        self.size = min(self.capacity, self.size + length)
        return length

    def state_dict(self) -> dict:
        state = {"capacity": self.capacity, "gamma": self.gamma, "size": self.size, "cursor": self.cursor}
        for name in ("features", "actions", "returns", "greedy"):
            state[name] = _compress_tensor(getattr(self, name)[:self.size])
        return state

    def load_state_dict(self, state: dict) -> None:
        size, cursor = state["size"], state["cursor"]
        if state["capacity"] != self.capacity or state["gamma"] != self.gamma:
            raise ValueError("replay configuration mismatch")
        if type(size) is not int or not 0 <= size <= self.capacity or type(cursor) is not int or not 0 <= cursor < self.capacity:
            raise ValueError("invalid replay size or cursor")
        for name in ("features", "actions", "returns", "greedy"):
            _decompress_tensor(state[name], getattr(self, name)[:size])
        self.size, self.cursor = size, cursor


@dataclass(frozen=True)
class LearnStats:
    updates: int = 0
    samples: int = 0
    loss: float = 0.0


class Learner:
    def __init__(self, online: QNetwork, config: TrainConfig, *, seed: int = 0) -> None:
        self.config, self.device = config, torch.device(config.device)
        self.online = online.to(self.device)
        self.optimizer = torch.optim.AdamW(online.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.replay = Replay(config.replay_capacity, config.gamma)
        self.rng = random.Random(seed)
        self.step_count = self.update_count = 0

    def learn(self, trajectories: Sequence[Trajectory], updates: int = 1) -> LearnStats:
        self.step_count += sum(self.replay.append(trajectory) for trajectory in trajectories)
        return self.update(updates)

    def update(self, updates: int = 1) -> LearnStats:
        if type(updates) is not int or updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        if not len(self.replay) or not updates:
            return LearnStats()
        self.online.train()
        total, count = 0.0, min(self.config.batch_size, len(self.replay))
        for _ in range(updates):
            learning_rate = max(
                self.config.learning_rate * 0.15,
                self.config.learning_rate / math.sqrt(1.0 + self.update_count / self.config.lr_decay_updates),
            ) if self.config.lr_decay_updates else self.config.learning_rate
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            idx = torch.tensor(self.rng.choices(range(len(self.replay)), k=count), dtype=torch.long, device="cpu")
            data = self.replay.features[idx].to(self.device)
            fields = {
                field.name: data[:, field.offset:field.offset + field.size].reshape(count, *field.shape)
                for field in LAYOUT
            }
            fields["top"] = fields["top"].squeeze(-1).long()
            actions = self.replay.actions[idx].to(self.device)
            targets = self.replay.returns[idx].to(self.device)
            greedy = self.replay.greedy[idx].to(self.device)
            q_all = self.online(ObservationBatch(**fields))
            q_taken = q_all.gather(1, actions[:, None]).squeeze(1)
            mc_loss = F.smooth_l1_loss(q_taken, targets)
            if self.config.margin_weight > 0:
                margin = torch.where(fields["legal"] > 0, 0.10, -1e9)
                margin.scatter_(1, actions[:, None], 0.0)
                win_mask = ((targets > 0) & greedy).to(q_all.dtype)
                margin_loss = (((q_all + margin).amax(dim=1) - q_taken) * win_mask).sum() / win_mask.sum().clamp_min(1.0)
                loss = mc_loss + self.config.margin_weight * margin_loss
            else:
                loss = mc_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            total += loss.item()
            self.update_count += 1
        return LearnStats(updates, updates * count, total / updates)

    def state_dict(self) -> dict:
        return {
            "online": clone_model_state(self.online.state_dict()), "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(), "rng": self.rng.getstate(),
            "step_count": self.step_count, "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(validate_model_state(state["online"], self.online.state_dict(), "online"))
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.rng.setstate(state["rng"])
        self.step_count, self.update_count = state["step_count"], state["update_count"]


class _Rollout(NamedTuple):
    env: PappluEnv | MultiplayerEnv
    deal: Deal
    rngs: tuple[random.Random, ...]
    epsilons: tuple[float, ...]
    steps: list[list[Step]]


def collect_round(
    config: TrainConfig, online: QNetwork, best_path: str, pool: Sequence[str], start: int, count: int,
) -> list[Trajectory]:
    learner = EpsilonPolicy(online, config.epsilons[0])
    policies = [learner, CheckpointPolicy(best_path), HeuristicPolicy(), RandomPolicy(),
                *(CheckpointPolicy(path) for path in pool[-config.max_pool:])]
    rollouts, trajectories = [], []
    for episode in range(start, start + count):
        players = config.player_counts[episode % len(config.player_counts)]
        seed = 2 * (config.seed * 10_000_000 + episode)
        rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
        seats = [0] * players
        if players > 1 and rng.random() >= config.self_play_share:
            seat = rng.randrange(players)
            for other in range(players):
                if other == seat:
                    continue
                draw = rng.random()
                seats[other] = (
                    1 if draw < 0.45 else 2 if draw < 0.65 else 3 if draw < 0.70
                    else rng.randrange(4, len(policies)) if len(policies) > 4 else 1
                )
        deal = Deal(seed, tuple(seats), players, frozenset(range(players)))
        rngs = tuple(random.Random(seed ^ ((seat + 1) * 0x9E3779B97F4A7C15)) for seat in range(players))
        epsilons = tuple(seat_rng.choice(config.epsilons) if policy == 0 else 0.0
                         for policy, seat_rng in zip(seats, rngs))
        env = PappluEnv(config.game) if players == 1 else MultiplayerEnv(MatchConfig(config.game, players))
        env.reset(seed)
        rollouts.append(_Rollout(env, deal, rngs, epsilons, [[] for _ in range(players)]))
    while True:
        groups = {}
        for rollout in rollouts:
            obs = rollout.env.observe()
            if obs.done:
                continue
            seat = 0 if rollout.deal.players == 1 else rollout.env.view().current_seat
            groups.setdefault(rollout.deal.seats[seat], []).append((rollout, seat, encode(obs)))
        if not groups:
            break
        for policy, rows in groups.items():
            features = [feature for _, _, feature in rows]
            rngs = [rollout.rngs[seat] for rollout, seat, _ in rows]
            if policy == 0:
                actions = learner.act_features(features, rngs, [rollout.epsilons[seat] for rollout, seat, _ in rows])
                greedy = learner.last_greedy
            else:
                actions = policies[policy].act_features(features, rngs)
                greedy = [policy != 3] * len(rows)
            for (rollout, seat, feature), action, was_greedy in zip(rows, actions, greedy):
                rollout.steps[seat].append(Step(feature, action, was_greedy))
                rollout.env.step(action)
    for rollout in rollouts:
        deal, env = rollout.deal, rollout.env
        if deal.players == 1:
            obs = env.observe()
            trajectories.append(Trajectory(tuple(rollout.steps[0]), solo_outcome(
                obs.hand, obs.joker, obs.won, config.required_sequences,
            ).utility))
            continue
        view = env.view()
        outcome = DealOutcome.score(env.terminal_snapshot(), view.winner, view.terminal_reason, config.required_sequences)
        for seat in sorted(deal.record_seats):
            steps = tuple(rollout.steps[seat])
            if steps:
                trajectories.append(Trajectory(steps, outcome.seat(seat).utility))
    return trajectories


def _actor_init() -> None:
    torch.set_num_threads(1)


def _actor_job(job: tuple) -> list[Trajectory]:
    config, spec, state, best_path, pool, start, count = job
    online = QNetwork(spec).eval()
    online.load_state_dict(state)
    return collect_round(config, online, best_path, pool, start, count)


def _validation_job(job: tuple) -> tuple[str, float]:
    cfg, spec, state, key, opponent, deals = job
    online = QNetwork(spec).eval()
    online.load_state_dict(state)
    players = deals[0].players
    if players == 1:
        return key, sum(evaluate_games(online, cfg.game, len(deals), deals[0].seed, seed_stride=2).outcomes)
    results = run_matches(deals, [GreedyPolicy(online), opponent], cfg.game)
    return key, sum(
        (1 + result.outcome.margin(deal.seats.index(0), other)) / 2
        for deal, result in zip(deals, results) for other in range(players) if deal.seats[other] != 0
    )


class Trainer:
    def __init__(self, root: Path, config: TrainConfig, online: QNetwork) -> None:
        self.root, self.config = root, config
        self.learner = Learner(online, config, seed=config.seed)
        self.episodes, self.best_score, self.validation = 0, -1.0, {}
        self.snapshot_pool: list[str] = []
        self._executor = None

    @classmethod
    def open(cls, output_dir: Path, overrides: dict | None = None, *, initial: Path | ModelSpec | None = None, opponents: Sequence[Path] = ()) -> Trainer:
        root, overrides = Path(output_dir).resolve(), dict(overrides or {})
        root.mkdir(parents=True, exist_ok=True)
        state_path = root / "latest-state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True) if state_path.exists() else None
        if state is not None:
            if state.get("version") != STATE_VERSION:
                raise ValueError("incompatible trainer state; start a fresh DMC run")
            if initial is not None or opponents:
                raise ValueError("initial models and opponents only apply to fresh runs")
            config = TrainConfig(**{**state["config"], **overrides})
            saved = TrainConfig(**state["config"])
            tunable = {"actors", "device", "torch_threads", "checkpoint_every"}
            if any(getattr(config, key) != getattr(saved, key) for key in overrides if key not in tunable):
                raise ValueError("cannot change training config on resume")
            spec = ModelSpec.from_dict(state["spec"])
        else:
            if isinstance(initial, (str, Path)):
                online, game, _ = load_checkpoint(str(initial))
                overrides = {**game.to_dict(), **overrides}
                if TrainConfig(**overrides).game != game:
                    raise ValueError("initial model game_config mismatch")
            config = TrainConfig(**overrides)
            spec = initial if isinstance(initial, ModelSpec) else ModelSpec()
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        if state is not None or not isinstance(initial, (str, Path)):
            online = QNetwork(spec)
        trainer = cls(root, config, online)
        if state is not None:
            trainer.learner.load_state_dict(state["learner"])
            trainer.episodes, trainer.best_score, trainer.validation = state["episodes"], state["best_score"], state["validation"]
            trainer.snapshot_pool = state["snapshot_pool"]
            for path in ["initial-model.pt", "best-model.pt", *trainer.snapshot_pool]:
                if not (root / path).is_file():
                    raise ValueError("missing opponent checkpoint " + path)
        else:
            trainer.export_model(root / "initial-model.pt")
            trainer.export_model(root / "best-model.pt")
            trainer.snapshot_pool = ["initial-model.pt"]
            for index, path in enumerate(opponents):
                load_checkpoint(str(path), expected_config=config.game)
                name = "opponents/seed-%03d.pt" % index
                atomic_copy(Path(path), root / name)
                trainer.snapshot_pool.append(name)
            trainer.snapshot_pool = trainer.snapshot_pool[-config.max_pool:]
            trainer.save("ready")
        return trainer

    def export_model(self, path: Path) -> None:
        atomic_torch_save(checkpoint_payload(self.learner.online, self.config.game, {
            "algorithm": "dmc", "gamma": self.config.gamma, "episodes": self.episodes,
        }), path)

    def validate(self) -> dict:
        cfg, online = self.config, self.learner.online.eval()
        if cfg.actors > 1 and self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=cfg.actors, initializer=_actor_init)
        opponents = (("heuristic", HeuristicPolicy()), ("best", CheckpointPolicy(str(self.root / "best-model.pt"))))
        cases = [("solo", 1, None)] if 1 in cfg.player_counts else []
        cases += [("%dp_%s" % (players, name), players, path)
                  for players in cfg.player_counts if players > 1 for name, path in opponents]
        jobs, state = [], clone_model_state(online.state_dict())
        for key, players, opponent in cases:
            seed = VALIDATION_SEED + (2 * players * cfg.validation_games if players > 1 else 0)
            deals = [Deal(seed + 2 * block, tuple(0 if s == seat else 1 for s in range(players)), players)
                     for block in range(cfg.validation_games) for seat in range(players)]
            jobs.extend((cfg, online.spec, state, key, opponent, deals[start:start + 8 * players])
                        for start in range(0, len(deals), 8 * players))
        results = list(map(_validation_job, jobs) if cfg.actors == 1 else self._executor.map(_validation_job, jobs))
        metrics = {key: sum(credit for name, credit in results if name == key)
                   / (cfg.validation_games * players * max(1, players - 1)) for key, players, _ in cases}
        anchors = [value for key, value in metrics.items() if not key.endswith("_best")]
        score = sum(anchors) / len(anchors)
        best = [value for key, value in metrics.items() if key.endswith("_best")]
        if best:
            metrics["vs_best"] = sum(best) / len(best)
        promoted = self.best_score < 0 or (
            metrics.get("vs_best", 0) >= 0.51 and score >= self.best_score - 0.02
        ) or (score > self.best_score + 0.02 and (not best or metrics["vs_best"] >= 0.49))
        if promoted:
            self.best_score = max(self.best_score, score) if best else score
            self.export_model(self.root / "best-model.pt")
        if promoted or metrics.get("vs_best", 0) >= 0.45:
            name = "opponents/%s-%012d.pt" % ("best" if promoted else "snap", self.episodes)
            if promoted:
                atomic_copy(self.root / "best-model.pt", self.root / name)
            else:
                self.export_model(self.root / name)
            self.snapshot_pool = [*self.snapshot_pool, name][-cfg.max_pool:]
        self.validation = {"score": score, **metrics, "promoted": promoted}
        return self.validation

    def run(self, *, max_episodes: int | None = None, max_seconds: float | None = None, stop: Callable[[], bool] = lambda: False) -> str:
        if max_episodes is not None and (type(max_episodes) is not int or max_episodes < 0):
            raise ValueError("max_episodes must be a nonnegative integer")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite")
        cfg, reason = self.config, "error"
        deadline = math.inf if max_seconds is None else time.monotonic() + max_seconds
        try:
            while True:
                if stop():
                    reason = "stopped"
                    break
                if max_episodes is not None and self.episodes >= max_episodes:
                    reason = "max_episodes"
                    break
                if time.monotonic() >= deadline:
                    reason = "budget"
                    break
                count = min(cfg.round_episodes, max_episodes - self.episodes) if max_episodes is not None else cfg.round_episodes
                start = self.episodes
                extra_episodes, extra_trajs = 0, []
                for path in sorted((self.root / "inbox").glob("batch-*.pt"))[:2]:
                    episodes, batch = load_rollout_batch(path)
                    if max_episodes is not None and start + extra_episodes + episodes > max_episodes:
                        continue
                    extra_episodes += episodes
                    extra_trajs.extend(batch)
                    path.unlink()
                count = max(0, count - extra_episodes)
                rollout_start = start + extra_episodes
                best_path = str(self.root / "best-model.pt")
                pool = [str(self.root / path) for path in self.snapshot_pool]
                if not count:
                    trajectories = []
                elif cfg.actors == 1:
                    trajectories = collect_round(cfg, self.learner.online, best_path, pool, rollout_start, count)
                else:
                    if self._executor is None:
                        self._executor = ProcessPoolExecutor(max_workers=cfg.actors, initializer=_actor_init)
                    chunk, state = math.ceil(count / cfg.actors), clone_model_state(self.learner.online.state_dict())
                    jobs = [(cfg, self.learner.online.spec, state, best_path, pool, offset, min(chunk, rollout_start + count - offset))
                            for offset in range(rollout_start, rollout_start + count, chunk)]
                    trajectories = [t for batch in self._executor.map(_actor_job, jobs) for t in batch]
                trajectories = extra_trajs + trajectories
                count += extra_episodes
                stats = self.learner.learn(trajectories, cfg.updates_per_episode * count)
                self.episodes += count
                validation = self.validate() if cfg.validation_games and self.episodes // cfg.validate_every > start // cfg.validate_every else None
                self.export_model(self.root / "latest-model.pt")
                with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"episodes": self.episodes, "epsilons": cfg.epsilons, "trajectories": len(trajectories),
                                             **asdict(stats), "validation": validation}, allow_nan=False) + "\n")
                if self.episodes // cfg.checkpoint_every > start // cfg.checkpoint_every:
                    self.save("running")
        finally:
            try:
                self.save(reason)
            finally:
                self.close()
        return reason

    def summary(self) -> dict:
        return {
            "episodes": self.episodes, "updates": self.learner.update_count, "steps": self.learner.step_count,
            "replay_size": len(self.learner.replay), "best_score": None if self.best_score < 0 else self.best_score,
            "model_spec": self.learner.online.spec.to_dict(), "validation": self.validation,
            "snapshot_pool": self.snapshot_pool,
        }

    def save(self, status: str) -> None:
        self.export_model(self.root / "latest-model.pt")
        atomic_torch_save({
            "version": STATE_VERSION, "config": asdict(self.config), "spec": self.learner.online.spec.to_dict(),
            "learner": self.learner.state_dict(), "episodes": self.episodes, "best_score": self.best_score,
            "validation": self.validation, "snapshot_pool": self.snapshot_pool,
        }, self.root / "latest-state.pt")
        atomic_write_json({"status": status, **self.summary()}, self.root / "status.json")

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
