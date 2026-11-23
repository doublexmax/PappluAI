from __future__ import annotations

import argparse
import cProfile
from itertools import islice
import json
from pathlib import Path
from pstats import SortKey, Stats
from random import Random
import resource
from statistics import median
from time import perf_counter
from uuid import UUID

import torch

from src.evaluation.arena import CheckpointPolicy, Deal, run_matches
from src.game.environment import GameConfig, PappluEnv
from src.model.encoding import SCHEMA
from src.model.network import load_checkpoint
from src.training.train import Learner, TrainConfig, Trajectories, TrajectoryBatch, _BatchIndex, vtrace


def prepare(checkpoint: Path, output: Path) -> None:
    policy = CheckpointPolicy(checkpoint, sample=True, precision="bf16", record=True)
    matches = run_matches((Deal(seed, (0,)) for seed in range(720_000, 720_160)), (policy,), policy.game, tables=32)
    results = list(islice((result for result in matches if result.action_count == 120), 48))
    assert len(results) == 48
    Trajectories.from_results(results, 0, str(UUID(int=1))).save(output)


def batch_from(rollouts: Path) -> tuple[TrajectoryBatch, TrajectoryBatch]:
    payload = torch.load(rollouts, map_location="cpu", weights_only=True)
    payload["target_probs"] = payload["target_probs"].view(torch.uint16)
    batch = Trajectories(**{name: payload[name] for name in Trajectories.__dataclass_fields__})
    seats = next(iter(_BatchIndex.from_batch(batch).buckets.values()))
    assert len(seats) == 48 and all(seat.action.numel() == 120 for seat in seats)
    return TrajectoryBatch.pack(tuple(seats[24:])), TrajectoryBatch.pack(tuple(seats[:24]))


def update(args: argparse.Namespace) -> dict:
    config = {"precision": "bf16", "batch_size": 24, "torch_threads": 6, "c_bar": 0.85}
    if "unroll_steps" in TrainConfig.__dataclass_fields__:
        config["unroll_steps"] = args.steps
    learner = Learner(load_checkpoint(args.checkpoint, SCHEMA).network, TrainConfig(**config))
    batch, warmup = batch_from(args.rollouts)
    learner.update(warmup)
    profiler = cProfile.Profile() if args.profile else None
    start = perf_counter()
    if profiler:
        profiler.enable()
    stats = learner.update(batch)
    if profiler:
        profiler.disable()
    elapsed = perf_counter() - start
    if profiler:
        with args.profile.open("w", encoding="utf-8") as stream:
            Stats(profiler, stream=stream).sort_stats(SortKey.CUMULATIVE).print_stats(30)
    assert learner.version == 2 and stats.grad_norm > 0
    assert {int(state["step"]) for state in learner.optimizer.state.values()} == {2}
    ran = getattr(stats, "trained_steps", 24 * 120)
    assert ran == 24 * min(args.steps, 120)
    target_us = None
    if hasattr(batch, "behavior_value"):
        value = batch.behavior_value
        inputs = (torch.zeros_like(value), batch.reward, batch.discount, value, torch.zeros(value.size(0)))
        timings = []
        for _ in range(128):
            tick = perf_counter()
            vtrace(*inputs, rho_bar=1.0, c_bar=0.85)
            timings.append(1e6 * (perf_counter() - tick))
        target_us = median(timings)
    return {
        "seconds": elapsed, "updates": 1, "trajectories": 24, "available_steps": 24 * 120, "trained_steps": ran,
        "loss": stats.loss, "grad_norm": stats.grad_norm, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "target_median_us": target_us, "errors": 0,
    }


def potential(checkpoint: Path) -> dict:
    from src.game.outcome import hand_potential

    game = load_checkpoint(checkpoint, SCHEMA).game
    env = PappluEnv(GameConfig.from_dict(game))
    rng = Random(108)
    workloads = {}
    for label, reset in (("cold", env.reset), ("distance_1", env.reset_curriculum), ("distance_0", env.reset_warm_start)):
        samples = [reset(1, seed=seed) if label == "distance_1" else reset(seed) for seed in range(750_000, 760_000)]
        rng.shuffle(samples)
        timings = []
        for offset in range(0, len(samples), 2000):
            start = perf_counter()
            scores = [hand_potential(obs.hand, obs.joker, 5, 21) for obs in samples[offset:offset + 2000]]
            timings.append(1e6 * (perf_counter() - start) / len(scores))
            assert all(0 <= score <= 0.75 for score in scores)
            if label == "distance_0":
                assert min(scores) == 0.75
        workloads[label] = {"calls": len(samples), "us_per_call_by_shard": timings, "median_us": median(timings)}
    return {"errors": 0, "workloads": workloads}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "update", "potential"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6)
    if args.mode == "prepare":
        prepare(args.checkpoint, args.rollouts)
    else:
        print(json.dumps(update(args) if args.mode == "update" else potential(args.checkpoint), allow_nan=False))


if __name__ == "__main__":
    main()
