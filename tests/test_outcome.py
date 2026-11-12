from __future__ import annotations

import random
import unittest

from src.evaluate import HandPenalty, card_points
from src.game.multiplayer import MatchConfig, MultiplayerEnv, TerminalSnapshot
from src.game.outcome import DealOutcome, placements


class TestDealOutcome(unittest.TestCase):
    @staticmethod
    def outcome(points, winner=None):
        return DealOutcome(winner, "win" if winner is not None else "turns_exhausted", tuple(
            HandPenalty(point, (), (0,) * 52) for point in points
        ))

    def test_declarer_wins_and_tied_losers_split_pairwise_utility(self):
        outcome = self.outcome((0, 20, 20, 130), 0)
        self.assertEqual(outcome.ranks(), (1, 2, 2, 4))
        self.assertEqual(tuple(outcome.seat(seat).utility for seat in range(4)), (1.0, 0.0, 0.0, -1.0))
        self.assertEqual(outcome.margin(0, 3), 1)
        self.assertEqual(outcome.margin(3, 0), -1)
        self.assertEqual(outcome.margin(1, 2), 0)
        self.assertTrue(outcome.seat(0).won)
        self.assertEqual(outcome.seat(3).penalty, 130)
        self.assertEqual(placements((0, 0, 20), 0, "win", "points"), (1, 2, 3))

    def test_multiplayer_utility_is_zero_sum_for_every_tie_pattern(self):
        rng = random.Random(13)
        for players in range(2, 7):
            for _ in range(50):
                points = [rng.choice((0, 20, 20, 80, 130)) for _ in range(players)]
                for winner in (None, rng.randrange(players)):
                    if winner is not None:
                        points[winner] = 0
                    outcome = self.outcome(points, winner)
                    self.assertAlmostEqual(sum(outcome.seat(seat).utility for seat in range(players)), 0.0, places=12)

    def test_solo_utility_tracks_declaration_instead_of_penalty(self):
        self.assertEqual(self.outcome((0,), 0).seat(0).utility, 1.0)
        self.assertEqual(self.outcome((0,)).seat(0).utility, -1.0)
        self.assertEqual(self.outcome((0,)).ranks(), (1,))

    def test_score_uses_minimum_penalty_for_each_final_hand(self):
        hands = (
            tuple(int(face in (0, 1, 2)) for face in range(52)),
            tuple(int(face in (0, 1, 12)) for face in range(52)),
        )
        outcome = DealOutcome.score(TerminalSnapshot(hands, 7), 0, "win", 1)
        self.assertEqual(tuple(penalty.points for penalty in outcome.penalties), (0, 22))
        self.assertEqual(tuple(outcome.seat(seat).utility for seat in range(2)), (1.0, -1.0))

    def test_card_points_validate_both_faces(self):
        self.assertEqual(tuple(card_points(face, 20) for face in range(13)), (10, 2, 3, 4, 5, 6, 7, 0, 9, 10, 10, 10, 10))
        for face, joker in ((-1, 7), (52, 7), (0, -1), (0, 52)):
            with self.subTest(face=face, joker=joker), self.assertRaises(ValueError):
                card_points(face, joker)
        with self.assertRaises(TypeError):
            card_points(True, 7)

    def test_private_targets_rotate_and_pad_without_changing_observation(self):
        env = MultiplayerEnv(MatchConfig(players=4))
        env.reset(29)
        before = env.observe()
        hands = env.private_hands(0)
        self.assertEqual(hands[0], before.hand)
        self.assertEqual(tuple(sum(hand) for hand in hands), (21, 21, 21, 21, 0, 0))
        self.assertEqual(env.private_hands(2), (hands[2], hands[3], hands[0], hands[1], (0,) * 52, (0,) * 52))
        self.assertEqual(env.observe(), before)
        self.assertNotIn("hidden_hands", before.to_dict())
        with self.assertRaises(ValueError):
            env.private_hands(4)


if __name__ == "__main__":
    unittest.main()
