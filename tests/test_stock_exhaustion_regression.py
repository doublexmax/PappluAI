import unittest

from src.environment import GameConfig, PappluEnv, legal_action_mask


class TestStockExhaustionRegression(unittest.TestCase):
    def test_stock_is_refilled_before_draw_budget_is_exhausted(self):
        environment = PappluEnv(GameConfig(
            num_decks=1, cards_in_hand=3, required_sequences=1, max_turns=60,
        ))
        observation = environment.reset(seed=0)
        for count in range(60):
            before = observation.hand
            observation = environment.step(0)
            drawn = next(face for face in range(52) if observation.hand[face] > before[face])
            observation = environment.step(drawn + 2)
            if count == 46:
                self.assertEqual(observation.stock_remaining, 47)
                self.assertEqual(len(observation.discard_pile), 1)
                self.assertEqual(observation.turns_remaining, 13)
                self.assertTrue(legal_action_mask(observation)[0])
        self.assertTrue(observation.done)
        self.assertFalse(observation.won)
        self.assertEqual(observation.turns_remaining, 0)
        self.assertEqual(environment.total_cards_in_play(), 51)


if __name__ == "__main__":
    unittest.main()
