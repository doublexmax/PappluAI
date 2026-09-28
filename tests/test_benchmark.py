import random
import json
from pathlib import Path
import tempfile
import unittest

from src.benchmark import evaluate_games, paired_comparison
from src.environment import GameConfig
from src.compare import run_comparison

try:
    import torch
except ImportError:
    torch = None


class TestPairedComparison(unittest.TestCase):
    def test_identical_outcomes_have_no_gain(self):
        outcomes = [[0, 1, 0, 1], [1, 0, 0, 1]]
        result = paired_comparison(outcomes, outcomes, samples=100)
        self.assertEqual(result["absolute_gain"], 0.0)
        self.assertEqual(result["ci95"], [0.0, 0.0])

    def test_known_complete_improvement(self):
        result = paired_comparison([[1] * 10] * 3, [[0] * 10] * 3, samples=100)
        self.assertEqual(result["absolute_gain"], 1.0)
        self.assertEqual(result["ci95"], [1.0, 1.0])

    def test_repeatable_without_global_rng_change(self):
        before = random.getstate()
        args = ([[0, 1, 1], [1, 1, 0]], [[0, 0, 1], [1, 0, 0]])
        self.assertEqual(paired_comparison(*args, samples=100), paired_comparison(*args, samples=100))
        self.assertEqual(before, random.getstate())

    def test_rejects_unpaired_results(self):
        for args in (([], []), ([[1]], []), ([[1, 0]], [[0]]), ([[2]], [[0]])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                paired_comparison(*args, samples=100)


class TestComparisonPlan(unittest.TestCase):
    def test_rejects_existing_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "comparison.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_comparison({}, output)

    def test_rejects_overlapping_selection_and_test_deals(self):
        plan_path = Path(__file__).resolve().parents[1] / "experiments" / "networks.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["screen"]["test_seed"] = plan["screen"]["validation_seed"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                run_comparison(plan, Path(directory))


@unittest.skipIf(torch is None, "Training dependencies not installed")
class TestEvaluation(unittest.TestCase):
    def test_puzzle_oracle_proves_sensitivity(self):
        from src.model import QNetwork

        config = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=3)
        network = QNetwork()
        oracle = evaluate_games(network, config, 20, 600000, "oracle", True)
        random_policy = evaluate_games(network, config, 20, 600000, "random", True)
        self.assertEqual(sum(oracle.outcomes), 20)
        self.assertLess(sum(random_policy.outcomes), 19)

    def test_evaluation_preserves_network_and_rng(self):
        from src.model import QNetwork

        network = QNetwork()
        config = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=2)
        before = [parameter.detach().clone() for parameter in network.parameters()]
        rng_state = torch.get_rng_state().clone()
        a = evaluate_games(network, config, 5, 900000)
        b = evaluate_games(network, config, 5, 900000)
        self.assertEqual(a, b)
        self.assertTrue(network.training)
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        for old, new in zip(before, network.parameters()):
            self.assertTrue(torch.equal(old, new))

    def test_validation_rejects_empty_games(self):
        from src.model import QNetwork

        with self.assertRaises(ValueError):
            evaluate_games(QNetwork(), GameConfig(), 0, 0)


if __name__ == "__main__":
    unittest.main()
