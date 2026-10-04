import unittest
from unittest import mock

from src.evaluate import hand_reward
from src.game.reward_cache import RewardCache


class TestExactRewardCache(unittest.TestCase):
    def test_repeated_validated_hand_is_evaluated_once_without_changing_reward(self):
        cache = RewardCache()
        hand = [0] * 52
        hand[0] = hand[1] = hand[2] = 1
        evaluator = mock.Mock(wraps=hand_reward)
        for _ in range(60):
            self.assertEqual(cache.score(hand, 20, 1, 3, evaluator), 1.0)
        self.assertEqual(evaluator.call_count, 1)
        self.assertEqual(sum(hand), 3)

    def test_errors_do_not_become_cached_losing_verdicts(self):
        cache = RewardCache()
        evaluator = mock.Mock(side_effect=[RuntimeError("failed"), 1.0])
        with self.assertRaises(RuntimeError):
            cache.score([0] * 52, 20, 1, 3, evaluator)
        self.assertEqual(cache.score([0] * 52, 20, 1, 3, evaluator), 1.0)
        self.assertEqual(evaluator.call_count, 2)

    def test_boolean_count_cannot_alias_an_existing_integer_cache_key(self):
        cache = RewardCache()
        hand = [0] * 52
        hand[0] = hand[1] = hand[2] = 1
        self.assertEqual(cache.score(hand, 20, 1, 3, hand_reward), 1.0)
        hand[0] = True
        with self.assertRaises(TypeError):
            cache.score(hand, 20, 1, 3, hand_reward)

    def test_different_rules_and_evaluator_never_reuse_wrong_results(self):
        cache = RewardCache()
        hand = [0] * 52
        first = mock.Mock(return_value=0.0)
        second = mock.Mock(return_value=1.0)
        self.assertEqual(cache.score(hand, 20, 1, 3, first), 0.0)
        self.assertEqual(cache.score(hand, 20, 1, 3, second), 1.0)
        cache.score(hand, 21, 1, 3, first)
        cache.score(hand, 21, 0, 3, first)
        self.assertEqual(first.call_count, 3)

    def test_capacity_is_bounded_and_oldest_result_is_recomputed(self):
        cache = RewardCache(capacity=1)
        evaluator = mock.Mock(return_value=0.0)
        cache.score([0] * 52, 20, 1, 3, evaluator)
        cache.score([0] * 52, 21, 1, 3, evaluator)
        cache.score([0] * 52, 20, 1, 3, evaluator)
        self.assertEqual(len(cache._results), 1)
        self.assertEqual(evaluator.call_count, 3)


if __name__ == "__main__":
    unittest.main()
