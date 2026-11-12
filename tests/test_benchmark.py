import random
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.evaluation.benchmark import (
    PolicyResults,
    evaluate_checkpoint,
    evaluate_games,
    paired_comparison,
)
from src.game.environment import GameConfig

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
        self.assertEqual(
            paired_comparison(*args, samples=100),
            paired_comparison(*args, samples=100),
        )
        self.assertEqual(before, random.getstate())

    def test_rejects_unpaired_results(self):
        for args in (([], []), ([[1]], []), ([[1, 0]], [[0]]), ([[2]], [[0]])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                paired_comparison(*args, samples=100)


@unittest.skipIf(torch is None, "Training dependencies not installed")
class TestEvaluation(unittest.TestCase):
    def test_curriculum_games_batch_without_changing_deal_order_or_random_policy_streams(self):
        from src.model.network import QNetwork
        from src.model.spec import ModelSpec

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                network = QNetwork(ModelSpec(8, 1, 1))
            config = GameConfig(max_turns=1)
            with mock.patch.object(network, "forward", wraps=network.forward) as forward:
                greedy = evaluate_games(network, config, 35, 600000, curriculum_distance=1)
            self.assertEqual(greedy, PolicyResults((
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
            ), 35, 0))
            self.assertEqual([call.args[0].card.shape[0] for call in forward.call_args_list], [32, 32, 3, 3])
            self.assertEqual(
                evaluate_games(network, config, 35, 600000, policy="random", curriculum_distance=1),
                PolicyResults((0,) * 35, 14, 21),
            )
            self.assertTrue(network.training)
        finally:
            torch.set_num_threads(previous_threads)

    def test_lockstep_batches_preserve_scalar_deal_order_outcomes_and_draws(self):
        from src.model.network import QNetwork
        from src.model.spec import ModelSpec

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                network = QNetwork(ModelSpec(8, 1, 1))
            config = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=3)
            greedy = evaluate_games(network, config, 35, 600000, warm_start=True)
            self.assertEqual(greedy, PolicyResults((
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
            ), 0, 0))
            random_policy = evaluate_games(network, config, 35, 600000, "random", True)
            self.assertEqual(random_policy, PolicyResults((
                0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
            ), 0, 0))
            ordinary = evaluate_games(network, config, 35, 600000)
            self.assertEqual(ordinary, PolicyResults((
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            ), 103, 0))
        finally:
            torch.set_num_threads(previous_threads)

    def test_checkpoint_evaluation_restores_torch_threads(self):
        from src.model.network import QNetwork, save_checkpoint

        before = torch.get_num_threads()
        outcomes = PolicyResults((0,), 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(str(path), QNetwork(), GameConfig())
            with mock.patch("src.evaluation.benchmark.evaluate_games", return_value=outcomes):
                result = evaluate_checkpoint(str(path), games=1, warm_games=0)
            self.assertEqual(result["greedy"]["outcomes"], (0,))
            self.assertEqual(torch.get_num_threads(), before)
            with mock.patch("src.evaluation.benchmark.load_checkpoint", side_effect=RuntimeError("load failed")):
                with self.assertRaisesRegex(RuntimeError, "load failed"):
                    evaluate_checkpoint(str(path), games=1, warm_games=0)
            self.assertEqual(torch.get_num_threads(), before)

    def test_puzzle_oracle_proves_sensitivity(self):
        from src.model.network import QNetwork

        config = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=3)
        network = QNetwork()
        oracle = evaluate_games(network, config, 20, 600000, "oracle", True)
        random_policy = evaluate_games(network, config, 20, 600000, "random", True)
        self.assertEqual(sum(oracle.outcomes), 20)
        self.assertLess(sum(random_policy.outcomes), 19)

    def test_evaluation_preserves_network_and_rng(self):
        from src.model.network import QNetwork

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
        from src.model.network import QNetwork

        with self.assertRaises(ValueError):
            evaluate_games(QNetwork(), GameConfig(), 0, 0)
        for distance in (0, -1, True, 1.5, "1"):
            with self.subTest(distance=distance), self.assertRaisesRegex(ValueError, "curriculum_distance"):
                evaluate_games(QNetwork(), GameConfig(), 1, 0, curriculum_distance=distance)
        with self.assertRaisesRegex(ValueError, "separate evaluation sources"):
            evaluate_games(QNetwork(), GameConfig(), 1, 0, warm_start=True, curriculum_distance=1)


if __name__ == "__main__":
    unittest.main()
