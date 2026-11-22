import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.game.environment import GameConfig
from src.model.encoding import HEADER_TOKENS, SCHEMA
from src.model.spec import ModelSpec

try:
    import torch
    import torch.distributed as dist
except ImportError:
    torch = dist = None

if torch is not None:
    from src.model.network import GameTransformer, load_checkpoint, save_checkpoint
    from src.training.train import Learner, Trainer, Trajectories, TrajectoryBatch, _SeatTrajectory, _Step


def _seat(rank):
    return _SeatTrajectory(
        torch.full((HEADER_TOKENS + 2, SCHEMA.fields), 1 + rank, dtype=torch.uint8),
        torch.full((2, SCHEMA.suffix_tokens, SCHEMA.fields), 20 + rank, dtype=torch.uint8),
        torch.tensor([HEADER_TOKENS + 1, HEADER_TOKENS + 2], dtype=torch.int16), torch.full((2,), 12),
        torch.full((2,), 2 + rank, dtype=torch.uint8), torch.full((2,), -math.log(2)),
        torch.tensor(1.0 if rank == 0 else -0.5),
    )


def _train_rank(base):
    rank = int(os.environ["RANK"])
    root = base / f"host-{rank}"
    seed = root / "initial.pt"
    trainer = Trainer.open(root, {
        "actors": 0, "torch_threads": 1, "batch_size": 1, "min_buffer_decisions": 2,
        "publish_every": 1, "anchor_every": 2, "eval_every": 1, "eval_deals": 1, "precision": "fp32",
    }, initial=seed)
    step, buckets = trainer.group.step(_Step.UPDATE, {(1, 0): 1 + rank * 4, (1, 14): 5, (2, 0): rank})
    assert (step, buckets) == (_Step.UPDATE, {(1, 0): 6, (1, 14): 10}), (step, buckets)
    original = load_checkpoint(trainer.model_path, SCHEMA).network
    for model in (trainer.learner.online, trainer.learner.anchor):
        for actual, expected in zip(model.parameters(), original.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    reference = Learner(copy.deepcopy(original), trainer.config)
    combined = TrajectoryBatch.pack((_seat(0), _seat(1)))
    for _ in range(2):
        reference.update(combined)
    seat = _seat(rank)
    Trajectories(
        0, 1, trainer.run_id, torch.ones(1, dtype=torch.uint8), seat.utility[None],
        seat.prefix, torch.tensor([len(seat.prefix)]), seat.suffix, torch.tensor([2]), seat.cut,
        seat.legal, seat.action, seat.behavior_logp,
    ).save(trainer.inbox / "batch-test.pt")
    assert trainer.run(stop=lambda: rank == 1 and trainer.learner.version == 2, max_updates=3, max_seconds=20) == "stopped"
    assert not dist.is_initialized()
    for actual, expected in zip(trainer.learner.online.parameters(), reference.online.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)


def _train_post_local_rank(base):
    rank = int(os.environ["RANK"])
    root = base / f"host-{rank}"
    trainer = Trainer.open(root, {
        "actors": 0, "torch_threads": 1, "batch_size": 1, "min_buffer_decisions": 2,
        "sync_every": 4, "publish_every": 3, "anchor_every": 3, "eval_every": 0, "precision": "fp32",
    }, initial=root / "initial.pt")
    seat = _seat(rank)
    Trajectories(
        0, 1, trainer.run_id, torch.ones(1, dtype=torch.uint8), seat.utility[None],
        seat.prefix, torch.tensor([len(seat.prefix)]), seat.suffix, torch.tensor([2]), seat.cut,
        seat.legal, seat.action, seat.behavior_logp,
    ).save(trainer.inbox / "batch-test.pt")
    references = [
        Learner(copy.deepcopy(trainer.learner.online), replace(trainer.config, anchor_every=100))
        for _ in range(2)
    ]
    batches = [TrajectoryBatch.pack((_seat(peer),)) for peer in range(2)]
    for steps in (4, 2):
        for model, batch in zip(references, batches):
            for _ in range(steps):
                model.update(batch)
        first, second = (model.online.state_dict() for model in references)
        average = {key: (first[key] + second[key]) / 2 for key in first}
        for model in references:
            model.online.load_state_dict(average)
            model.refresh_anchor()
    assert trainer.run(max_updates=6, max_seconds=20) == "max_updates"
    assert not dist.is_initialized()
    online, anchor = trainer.learner.online.state_dict(), trainer.learner.anchor.state_dict()
    for key, expected in average.items():
        torch.testing.assert_close(online[key], expected, rtol=0, atol=0)
        torch.testing.assert_close(anchor[key], expected, rtol=0, atol=0)
    torch.save(anchor, root / "anchor.pt")


@unittest.skipUnless(dist is not None and dist.is_available() and dist.is_gloo_available(), "Gloo not installed")
class TestDistributedLearner(unittest.TestCase):
    def test_two_ranks_match_one_update_stream_over_both_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            primary, peer = base / "host-0", base / "host-1"
            primary.mkdir()
            torch.manual_seed(991)
            seed = GameTransformer(ModelSpec(16, 2, 2, 32), SCHEMA)
            save_checkpoint(primary / "initial.pt", seed, GameConfig(max_turns=2).to_dict(), {"version": 9})
            run = subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                 "-m", "tests.test_distributed", str(base)],
                capture_output=True, text=True, timeout=60, env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            first = load_checkpoint(primary / "latest-model.pt", SCHEMA)
            second = load_checkpoint(peer / "rank-1" / "latest-model.pt", SCHEMA)
            self.assertEqual(first.meta, second.meta)
            self.assertEqual(first.version, 2)
            for name, value in first.network.state_dict().items():
                torch.testing.assert_close(value, second.network.state_dict()[name], rtol=0, atol=0)
            self.assertFalse(torch.equal(first.network.policy.weight, seed.policy.weight))
            self.assertEqual(sorted(path.name for path in peer.iterdir()), ["rank-1"])
            self.assertTrue((primary / "rank-0" / "inbox").is_dir())
            status = json.loads((primary / "status.json").read_text())
            self.assertEqual((status["status"], status["validation"]["version"]), ("stopped", 2))
            metrics = [json.loads(row) for row in (primary / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([(row["version"], row["trained_decisions"]) for row in metrics], [(1, 4), (2, 8)])

    def test_post_local_adamw_averages_online_and_anchor_at_full_and_partial_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            primary, peer = base / "host-0", base / "host-1"
            primary.mkdir()
            torch.manual_seed(991)
            seed = GameTransformer(ModelSpec(16, 2, 2, 32), SCHEMA)
            save_checkpoint(primary / "initial.pt", seed, GameConfig(max_turns=2).to_dict(), {"version": 9})
            run = subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                 "-m", "tests.test_distributed", str(base), "post_local"],
                capture_output=True, text=True, timeout=60, env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            first = load_checkpoint(primary / "latest-model.pt", SCHEMA)
            second = load_checkpoint(peer / "rank-1" / "latest-model.pt", SCHEMA)
            self.assertEqual((first.version, second.version), (6, 6))
            self.assertFalse(torch.equal(first.network.policy.weight, seed.policy.weight))
            first_anchor = torch.load(primary / "anchor.pt", weights_only=True)
            second_anchor = torch.load(peer / "anchor.pt", weights_only=True)
            for name, value in first.network.state_dict().items():
                for other in (second.network.state_dict()[name], first_anchor[name], second_anchor[name]):
                    self.assertTrue(torch.equal(value.view(torch.int32), other.view(torch.int32)), name)
            metrics = [json.loads(row) for row in (primary / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([(row["version"], row["trained_decisions"]) for row in metrics], [(4, 16), (6, 24)])
            before = torch.load(primary / "latest-state.pt", weights_only=True)
            resumed = Trainer.open(primary, {
                "learning_rate": 2e-4, "eta": 0.02, "sync_every": 1,
                "plan": {"curriculum_share": 0.25, "curriculum_max_distance": 12},
            })
            self.assertEqual(resumed.learner.optimizer.param_groups[0]["lr"], 2e-4)
            self.assertEqual((resumed.config.eta, resumed.config.sync_every), (0.02, 1))
            self.assertEqual((resumed.config.plan.curriculum_share, resumed.config.plan.curriculum_max_distance),
                             (0.25, 12))
            self.assertEqual(resumed.run_id, before["run_id"])
            self.assertEqual(resumed.learner.version, 6)
            moments = resumed.learner.optimizer.state_dict()["state"]
            torch.testing.assert_close(moments, before["learner"]["optimizer"]["state"], rtol=0, atol=0)
            published = load_checkpoint(resumed.model_path, SCHEMA)
            self.assertEqual(published.meta["plan"], resumed.config.plan.to_dict())


if __name__ == "__main__":
    run_rank = _train_post_local_rank if sys.argv[2:] == ["post_local"] else _train_rank
    run_rank(Path(sys.argv[1]))
