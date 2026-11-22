from pathlib import Path
import tempfile
import unittest

from src.game.environment import GameConfig
from src.model.encoding import SCHEMA
from src.model.spec import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.evaluation.arena import RandomPolicy
    from src.evaluation.benchmark import evaluate_checkpoint, evaluate_games, paired_comparison
    from src.model.network import GameTransformer, save_checkpoint


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestBenchmark(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = GameConfig(max_turns=1)

    def test_result_order_and_draw_totals_do_not_depend_on_table_count(self):
        one = evaluate_games(RandomPolicy(), self.game, 7, 47, tables=1)
        many = evaluate_games(RandomPolicy(), self.game, 7, 47, tables=3)
        self.assertEqual(one, many)
        self.assertEqual(one.stock_draws + one.discard_draws, 7)
        self.assertEqual(one.outcomes, (0,) * 7)

    def test_checkpoint_report_keeps_puzzles_separate_and_has_a_positive_control(self):
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, self.game.to_dict(), {"version": 1})
            report = evaluate_checkpoint(str(path), games=3, warm_games=8, sensitivity=True, tables=2)
        self.assertEqual((report["architecture"], report["checkpoint_version"], report["encoding_version"]),
                         ("game_transformer", 6, 8))
        self.assertEqual(report["greedy"]["outcomes"], (0, 0, 0))
        self.assertEqual(report["warm_oracle"]["outcomes"], (1,) * 8)
        self.assertEqual(report["warm_random"]["stock_draws"], 0)
        self.assertEqual(report["parameter_count"], sum(t.numel() for t in model.parameters()))

    def test_paired_bootstrap_keeps_the_matched_deal_effect(self):
        result = paired_comparison(((1, 1), (1, 1)), ((0, 0), (0, 0)), samples=100)
        self.assertEqual((result["absolute_gain"], result["ci95"]), (1, [1, 1]))
