import random
import unittest
from unittest import mock

from src.checkpoints.evidence import (
    balanced_seat_orders,
    clustered_interval,
)
from src.evaluation.promotion import (
    evaluate_gate,
)

try:
    import torch
except ImportError:
    torch = None


class TestSeatClusterBootstrap(unittest.TestCase):
    def test_identical_contenders_cannot_gain_from_following_a_weak_opponent(self):
        orders = balanced_seat_orders(3)
        scores = []
        for order in orders:
            winner_seat = (order.index(2) + 1) % 3
            winner = order[winner_seat]
            scores.append(1 if winner == 0 else -1 if winner == 1 else 0)
        self.assertEqual(len(orders), 6)
        self.assertEqual(sum(scores), 0)
        self.assertEqual(set(orders), {
            (0, 1, 2), (1, 2, 0), (2, 0, 1),
            (1, 0, 2), (0, 2, 1), (2, 1, 0),
        })

    def test_constant_margin_has_exact_interval(self):
        for value, blocks in ((0.0, 128), (0.5, 64)):
            with self.subTest(value=value):
                self.assertEqual(
                    clustered_interval(
                        [value] * blocks,
                        samples=100,
                    ),
                    [value, value],
                )

    def test_bootstrap_preserves_global_random_state(self):
        before = random.getstate()
        values = [0, 0.5, -0.5, 1] * 32
        self.assertEqual(clustered_interval(values, samples=100), clustered_interval(values, samples=100))
        self.assertEqual(before, random.getstate())

    def test_invalid_or_nonfinite_scores_are_rejected(self):
        for values in (
            [],
            [float("nan")],
            [float("inf")],
            [2.0],
            [True],
            ["0.5"],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                clustered_interval(values, samples=100)

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_gate_restores_torch_threads_when_checkpoint_loading_fails(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with mock.patch(
                "src.evaluation.promotion.CheckpointPolicy",
                side_effect=RuntimeError("bad checkpoint"),
            ), self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
                evaluate_gate(
                    "candidate.pt",
                    "champion.pt",
                    (),
                    seed=1,
                    seed_blocks=32,
                    solo_games=256,
                )
            self.assertEqual(torch.get_num_threads(), 2)
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
