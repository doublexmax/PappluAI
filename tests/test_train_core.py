from collections import Counter
from dataclasses import replace
from itertools import islice
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import UUID

from src.game.environment import GameConfig
from src.model.encoding import SCHEMA
from src.model.spec import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.evaluation.arena import Deal, TransformerPolicy, run_matches
    from src.model.network import GameTransformer, load_checkpoint, save_checkpoint
    from src.training.train import CollectPlan, Trajectories, Trainer, _Buffer, collect


RUN_ID = str(UUID(int=93))
TINY = ModelSpec(width=16, layers=1, heads=2, ffn=32)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestTrainingCore(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = GameConfig(max_turns=1)

    def test_columnar_roundtrip_rejects_corrupt_rollouts_and_targets(self):
        network = GameTransformer(TINY, SCHEMA)
        policy = TransformerPolicy(network, self.game, sample=True, record=True)
        results = list(run_matches((Deal(9, (0,)), Deal(11, (0, 0))), (policy,), self.game))
        batch = Trajectories.from_results(results, 4, RUN_ID)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "batch-001.pt"
            batch.save(path)
            loaded = Trajectories.load(path)
            self.assertEqual((loaded.version, loaded.episodes, loaded.decisions), (4, 2, 6))
            self.assertEqual(loaded.suffix.shape, (6, 30, 8))
            for name in ("prefix", "suffix", "decision_end", "cut", "action", "behavior_logp", "utility", "legal",
                         "target_probs", "forced"):
                torch.testing.assert_close(getattr(batch, name), getattr(loaded, name))
            self.assertEqual((loaded.target_probs.dtype, loaded.target_probs.shape), (torch.uint16, (6, 54)))
            torch.testing.assert_close(loaded.forced, torch.zeros(6, dtype=torch.uint8))
            payload = torch.load(path, weights_only=True)
            payload["schema"]["version"] = 6
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "incompatible rollout token schema"):
                Trajectories.load(path)
            for changes, text in (
                ({"cut": batch.cut + 1}, "prefix cuts"),
                ({"decision_end": batch.decision_end + 1}, "offsets"),
                ({"behavior_logp": torch.full_like(batch.behavior_logp, float("nan"))}, "behavior"),
                ({"target_probs": batch.target_probs[:, :53]}, "target_probs"),
                ({"target_probs": torch.zeros_like(batch.target_probs)}, "probability mass"),
                ({"forced": torch.full_like(batch.forced, 2)}, "forced flags"),
                ({"forced": batch.forced[:-1]}, "decision column lengths"),
            ):
                with self.subTest(text=text):
                    replace(batch, **changes).save(path)
                    with self.assertRaisesRegex(ValueError, text):
                        Trajectories.load(path)
            targets = batch.target_probs.int()
            targets[0, 2] = 1
            replace(batch, target_probs=targets.to(torch.uint16)).save(path)
            with self.assertRaisesRegex(ValueError, "target probability"):
                Trajectories.load(path)
            replace(batch, action=torch.full_like(batch.action, 54)).save(path)
            with self.assertRaisesRegex(ValueError, "legal action"):
                Trajectories.load(path)
            suffix = batch.suffix.clone()
            suffix[0, -1, 7] = 55
            replace(batch, suffix=suffix).save(path)
            with self.assertRaisesRegex(ValueError, "pointer outside the action space"):
                Trajectories.load(path)

    def test_terminal_win_flag_survives_rollout_and_replay(self):
        model = GameTransformer(TINY, SCHEMA)
        policy = TransformerPolicy(model, self.game, search_ply=1, sample=True, record=True)
        runs = list(run_matches((Deal(9, (0,), curriculum=0),), (policy,), self.game))
        self.assertEqual((runs[0].winner, runs[0].action_count), (0, 1))
        batch = Trajectories.from_results(runs, 0, RUN_ID)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "batch-win.pt"
            batch.save(path)
            batch = Trajectories.load(path)
        self.assertEqual(batch.forced.tolist(), [1])
        buffer = _Buffer(4, 3)
        self.assertTrue(buffer.add(batch, 0))
        sample = buffer.sample(Random(55), 1, 0)
        self.assertEqual((sample.forced.tolist(), sample.reward.tolist()), ([[True]], [[1.0]]))

    def test_collection_ladder_includes_solo_puzzles_but_keeps_cold_deals_separate(self):
        plan = CollectPlan(curriculum_share=1.0, curriculum_max_distance=6)
        deals = list(islice(plan.deals("mastery"), 4096))
        self.assertEqual(deals, list(islice(plan.deals("mastery"), 4096)))
        self.assertEqual([deal.players for deal in deals[:8]], [1, 2, 3, 4, 1, 2, 3, 4])
        self.assertEqual({deal.curriculum for deal in deals if deal.players == 1}, set(range(7)))
        self.assertGreater(sum(deal.players == 1 and deal.curriculum <= 1 for deal in deals), 420)
        for players in (2, 3, 4):
            self.assertEqual({deal.curriculum for deal in deals if deal.players == players}, set(range(1, 7)))
        cold = list(islice(replace(plan, curriculum_share=0).deals("mastery"), 8))
        self.assertEqual([deal.curriculum for deal in cold], [None] * 8)

    def test_buffer_keeps_terminal_reward_and_rejects_future_or_stale_batches(self):
        model = GameTransformer(TINY, SCHEMA)
        policy = TransformerPolicy(model, self.game, sample=True, record=True)
        runs = list(run_matches([Deal(11, (0,))], [policy], self.game))
        batch = Trajectories.from_results(runs, 7, RUN_ID)
        buffer = _Buffer(4, 3)
        self.assertFalse(buffer.add(batch, 6))
        self.assertTrue(buffer.add(batch, 10))
        sample = buffer.sample(Random(55), 1, 10)
        self.assertEqual(sample.valid.tolist(), [[True, True]])
        torch.testing.assert_close(sample.discount, torch.tensor([[0.995, 0.0]]))
        self.assertEqual(sample.tokens.cut.tolist(), [[30, 31]])
        self.assertAlmostEqual(sample.reward[0, 1].item(), runs[0].outcome.utilities()[0])
        self.assertEqual(sample.reward[0, 0], 0.0)
        torch.testing.assert_close(sample.target_probs.sum(-1), torch.ones(1, 2))
        self.assertEqual(sample.target_probs.masked_select(~sample.tokens.legal).count_nonzero(), 0)
        self.assertEqual(sample.forced.tolist(), [[False, False]])
        buffer.prune(11)
        self.assertEqual(len(buffer), 0)
        self.assertFalse(buffer.add(batch, 11))

    def test_buckets_keep_short_trajectories_separate_and_only_pad_after_terminal(self):
        lengths = (1, 8, 120)
        decisions = sum(lengths)
        targets = torch.zeros(decisions, SCHEMA.num_actions, dtype=torch.int32)
        targets[:, 1] = 65535
        batch = Trajectories(
            0, 3, RUN_ID, torch.ones(3, dtype=torch.uint8), torch.tensor([1.0, 0.5, -1.0]),
            torch.ones(decisions, SCHEMA.fields, dtype=torch.uint8), torch.tensor(lengths).cumsum(0),
            torch.ones(decisions, SCHEMA.suffix_tokens, SCHEMA.fields, dtype=torch.uint8),
            torch.tensor(lengths).cumsum(0), torch.cat([torch.arange(1, n + 1) for n in lengths]),
            torch.full((decisions,), 2), torch.ones(decisions, dtype=torch.uint8), torch.zeros(decisions),
            targets.to(torch.uint16), torch.zeros(decisions, dtype=torch.uint8),
        )
        buffer = _Buffer(decisions, 3)
        self.assertTrue(buffer.add(batch, 0))
        rng = Random(21)
        observed = set()
        for _ in range(12):
            sample = buffer.sample(rng, 4, 0)
            seen = set(sample.valid.sum(dim=1).tolist())
            self.assertTrue(seen <= {1, 8} or seen == {120}, seen)
            observed.update(seen)
            self.assertEqual(sample.action.size(1), max(seen))
            for index, valid in enumerate(sample.valid):
                length = int(valid.sum())
                torch.testing.assert_close(sample.tokens.cut[index, :length], torch.arange(1, length + 1))
                torch.testing.assert_close(sample.discount[index, :length - 1], torch.full((length - 1,), .995))
                self.assertFalse(sample.discount[index, length - 1:].any())
                self.assertEqual(sample.reward[index].nonzero().flatten().tolist(), [length - 1])
                self.assertEqual(sample.reward[index, length - 1].item(), {1: 1.0, 8: 0.5, 120: -1.0}[length])
        self.assertEqual(observed, set(lengths))

        frequencies = Counter(int(buffer.sample(rng, 1, 0).valid.sum()) for _ in range(900))
        for length in lengths:
            self.assertGreater(frequencies[length], 240, frequencies)
            self.assertLess(frequencies[length], 360, frequencies)

        peer = _Buffer(decisions * 2, 20)
        peer.add(batch, 0)
        peer.add(replace(batch, players=torch.tensor([2, 2, 1], dtype=torch.uint8)), 0)
        shared = {(1, 0): 2, (1, 14): 1}
        observed_buckets = set()
        for version in range(4):
            left = buffer.sample(Random(55), 4, version, seed=41, targets=shared)
            right = peer.sample(Random(73), 4, version, seed=41, targets=shared)
            self.assertEqual(left.valid.size(1) > 8, right.valid.size(1) > 8)
            observed_buckets.add(left.valid.size(1) > 8)
        self.assertEqual(observed_buckets, {False, True})

        fallback = peer.sample(Random(55), 4, 4, seed=41, targets={(2, 14): 1})
        self.assertLessEqual(fallback.valid.size(1), 8)

    def test_published_revision_does_not_change_live_stream_actions(self):
        plan = CollectPlan(player_counts=(1, 2), curriculum_share=0, batch_decisions=1, uniform_mix=0, search_ply=0)
        old, new = (GameTransformer(TINY, SCHEMA) for _ in range(2))
        for action, network in enumerate((old, new)):
            with torch.no_grad():
                for param in network.parameters():
                    param.zero_()
                network.policy.bias[action] = 100
        emitted = []
        persist = Trajectories.save
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "latest-model.pt"

            def publish(network, version):
                save_checkpoint(path, network, self.game.to_dict(), {
                    "version": version, "run_id": RUN_ID, "plan": plan.to_dict(),
                })

            def on_save(batch, destination):
                persist(batch, destination)
                emitted.append(Trajectories.load(destination))
                if len(emitted) == 1:
                    publish(new, 1)

            publish(old, 0)
            with patch.object(Trajectories, "save", on_save):
                report = collect(root / "inbox", path, worker=2, tables=2, max_batches=3)
        self.assertEqual(report, {"batches": 3, "episodes": 3, "decisions": 8})
        self.assertEqual([row.version for row in emitted], [0, 0, 1])
        self.assertEqual([row.players.tolist() for row in emitted], [[1], [2, 2], [1]])
        self.assertEqual([row.action[0].item() for row in emitted], [0, 0, 1])
        self.assertEqual(emitted[1].action[2].item(), 0)

    def test_external_collection_and_cold_evaluation_survive_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = Trainer.open(root, {
                "game": self.game.to_dict(), "plan": {"batch_decisions": 6, "curriculum_share": 0.0},
                "actors": 0, "tables": 2, "torch_threads": 1, "batch_size": 2,
                "min_buffer_decisions": 2, "eval_every": 1, "eval_deals": 1, "anchor_every": 2,
            }, spec=TINY)
            for version in (1, 2):
                result = collect(trainer.inbox, root / "latest-model.pt", worker=35, tables=2, max_batches=1)
                self.assertEqual(result["batches"], 1)
                self.assertEqual(trainer.run(max_updates=version, max_seconds=15), "max_updates")
                state = json.loads((root / "status.json").read_text())
                self.assertEqual((state["status"], state["version"]), ("max_updates", version))
                self.assertEqual({row["players"] for row in state["validation"]["random"]}, {1, 2, 3, 4})
                self.assertGreater(state["updates_per_sec"], 0)
                self.assertEqual(state["trained_decisions"], version * 4)
                checkpoint = load_checkpoint(root / "latest-model.pt", SCHEMA)
                self.assertEqual((checkpoint.version, checkpoint.meta["run_id"]), (version, trainer.run_id))
                self.assertEqual(checkpoint.meta["plan"]["curriculum_share"], 0.0)
                trainer = Trainer.open(root)
                self.assertEqual(trainer.learner.version, version)
                self.assertEqual(trainer.run_id, checkpoint.meta["run_id"])
            with self.assertRaisesRegex(ValueError, "cannot change"):
                Trainer.open(root, {"plan": {"uniform_mix": 0.5}})
