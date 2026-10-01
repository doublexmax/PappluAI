import random
import unittest

from src.promotion import clustered_interval, balanced_seat_orders


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

    def test_identical_policies_have_zero_margin(self):
        self.assertEqual(clustered_interval([0.0] * 128, samples=100), [0.0, 0.0])

    def test_consistently_superior_policy_has_positive_bound(self):
        lower, upper = clustered_interval([0.5] * 64, samples=100)
        self.assertEqual((lower, upper), (0.5, 0.5))

    def test_bootstrap_preserves_global_random_state(self):
        before = random.getstate()
        values = [0, 0.5, -0.5, 1] * 32
        self.assertEqual(clustered_interval(values, samples=100), clustered_interval(values, samples=100))
        self.assertEqual(before, random.getstate())

    def test_invalid_or_nonfinite_scores_are_rejected(self):
        for values in ([], [float("nan")], [float("inf")], [2.0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                clustered_interval(values, samples=100)


if __name__ == "__main__":
    unittest.main()
