import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Training dependencies not installed")
class TestStockRuleMigration(unittest.TestCase):
    def test_preserves_weights_counters_and_champion_but_retires_old_rule_experience(self):
        from src.game.environment import GameConfig
        from src.training.improve import ImproveConfig, ImprovementController
        from src.model.network import QNetwork, save_checkpoint
        from src.checkpoints.rule_migration import migrate

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.pt"
            save_checkpoint(
                str(model), QNetwork(architecture="suit_conv"),
                GameConfig(max_turns=60, recycle_discard=False),
            )
            old = ImprovementController(
                str(model), root / "old",
                config=ImproveConfig(
                    max_cycles=1, league_matches_per_cycle=1,
                    research_episodes_per_cycle=1, selection_blocks=2,
                    confirmation_blocks=2, solo_games=2, recycle_discard=False,
                ),
            )
            old.league.train_match()
            old.research.run_episode()
            old.league_in_cycle = old.research_in_cycle = 1
            old._prepare_candidates()
            old.save_checkpoint()
            before = old.state_dict()
            source_state_before = (
                old.output_dir / "latest-state.pt"
            ).read_bytes()
            champion = old.registry.champion()
            event = migrate(old.output_dir, root / "new")
            self.assertEqual(
                (old.output_dir / "latest-state.pt").read_bytes(),
                source_state_before,
            )
            new = ImprovementController.load(str(root / "new" / "latest-state.pt"), root / "new")
            self.assertEqual(new.registry.champion().sha256, champion.sha256)
            self.assertEqual(new.league.completed_matches, old.league.completed_matches)
            self.assertEqual(new.research.total_episodes, old.research.total_episodes)
            self.assertTrue(new.league.config.game.recycle_discard)
            self.assertTrue(new.research.config.recycle_discard)
            self.assertEqual(len(new.league.replay), 0)
            self.assertEqual(len(new.research.replay), 0)
            self.assertEqual(new.league.optimizer.state_dict()["state"], {})
            self.assertEqual(new.research.optimizer.state_dict()["state"], {})
            self.assertEqual(new.phase, "training")
            self.assertEqual(new.candidate_ids, {})
            self.assertEqual(new.selection_results, {})
            self.assertEqual((new.league_in_cycle, new.research_in_cycle), (0, 0))
            for role in ("league", "research"):
                session = getattr(new, role)
                for key, value in session.network.state_dict().items():
                    self.assertTrue(torch.equal(value, before[role + "_state"]["network_state_dict"][key]))
            self.assertTrue(event["old_evaluations_invalidated"])
            status = json.loads((root / "new" / "status.json").read_text())
            self.assertTrue(status["rules"]["recycle_discard"])
            audit = json.loads(
                (root / "new" / "rule-migration.json").read_text()
            )
            self.assertEqual(audit["event_id"], event["event_id"])
            self.assertTrue(
                (root / "new" / "old-rule-metrics.jsonl").is_file()
            )
            self.assertFalse(
                (root / "new" / ".rule-migration-state.pt").exists()
            )
            (root / "new" / "status.json").unlink()
            recovered = migrate(old.output_dir, root / "new")
            self.assertEqual(recovered["event_id"], event["event_id"])
            self.assertTrue((root / "new" / "status.json").is_file())
            self.assertEqual(
                ImprovementController.load(
                    str(root / "new" / "latest-state.pt"),
                    root / "new",
                ).registry.champion().sha256,
                champion.sha256,
            )
            before_repeat = (root / "new" / "latest-state.pt").read_bytes()
            repeated = migrate(root / "new", root / "new")
            self.assertEqual(repeated["status"], "already_migrated")
            self.assertEqual((root / "new" / "latest-state.pt").read_bytes(), before_repeat)
            repeated_from_old_source = migrate(old.output_dir, root / "new")
            self.assertEqual(repeated_from_old_source["status"], "already_migrated")
            self.assertEqual((root / "new" / "latest-state.pt").read_bytes(), before_repeat)


if __name__ == "__main__":
    unittest.main()
