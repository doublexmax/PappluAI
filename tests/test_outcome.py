from __future__ import annotations

import random
from itertools import product
import unittest

from src.evaluate import HandPenalty, card_points
from src.game.multiplayer import MatchConfig, MultiplayerEnv, TerminalSnapshot
from src.game.outcome import DealOutcome, placements, solo_outcome


class TestDealOutcome(unittest.TestCase):
    @staticmethod
    def outcome(points, winner=None):
        return DealOutcome(winner, "win" if winner is not None else "turns_exhausted", tuple(
            HandPenalty(point, (), (0,) * 52) for point in points
        ))

    def test_declarer_wins_and_tied_losers_split_pairwise_utility(self):
        outcome = self.outcome((0, 20, 20, 130), 0)
        self.assertEqual(outcome.ranks(), (1, 2, 2, 4))
        self.assertEqual(
            tuple(outcome.seat(seat).utility for seat in range(4)),
            (1.0, -1 / 6, -1 / 6, -2 / 3),
        )
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

    def test_solo_losses_order_deadwood_below_every_declaration(self):
        self.assertEqual(self.outcome((0,), 0).seat(0).utility, 1.0)
        self.assertEqual(self.outcome((0,)).seat(0).utility, -0.5)
        self.assertEqual(self.outcome((0,)).ranks(), (1,))
        losses = [self.outcome((points,)).seat(0) for points in (3, 40, 140)]
        for result, expected in zip(losses, (-23 / 43, -0.75, -8 / 9)):
            self.assertAlmostEqual(result.utility, expected, places=12)
            self.assertFalse(result.won)
        self.assertEqual([result.penalty for result in losses], [3, 40, 140])
        self.assertGreater(losses[0].utility, losses[1].utility)
        self.assertGreater(losses[1].utility, losses[2].utility)
        self.assertTrue(all(-1.0 < result.utility <= -0.5 for result in losses))

    def test_deadwood_winner_gets_half_credit_without_changing_margins(self):
        outcome = self.outcome((3, 21, 42, 63))
        self.assertEqual(outcome.ranks(), (1, 2, 3, 4))
        self.assertEqual(
            tuple(outcome.seat(seat).utility for seat in range(4)),
            (0.5, 1 / 6, -1 / 6, -0.5),
        )
        self.assertEqual(tuple(outcome.margin(0, other) for other in (1, 2, 3)), (1, 1, 1))
        self.assertFalse(outcome.seat(0).won)

    def test_an_opponent_declaration_never_improves_a_non_declarers_utility(self):
        for count in (2, 3, 4):
            for points in product((0, 3, 40, 140), repeat=count):
                exhausted = self.outcome(points)
                for winner in range(count):
                    declared_points = list(points)
                    declared_points[winner] = 0
                    declared = self.outcome(declared_points, winner)
                    self.assertEqual(declared.seat(winner).utility, 1.0)
                    for seat in range(count):
                        result = declared.seat(seat)
                        self.assertTrue(-1.0 <= result.utility <= 1.0)
                        if seat != winner:
                            self.assertLess(result.utility, exhausted.seat(seat).utility)

    def test_score_uses_minimum_penalty_for_each_final_hand(self):
        hands = (
            tuple(int(face in (0, 1, 2)) for face in range(52)),
            tuple(int(face in (0, 1, 12)) for face in range(52)),
        )
        outcome = DealOutcome.score(TerminalSnapshot(hands, 7), 0, "win", 1)
        self.assertEqual(tuple(penalty.points for penalty in outcome.penalties), (0, 22))
        self.assertEqual(tuple(outcome.seat(seat).utility for seat in range(2)), (1.0, -1.0))

    def test_solo_helper_keeps_zero_penalty_losses_distinct_from_declarations(self):
        natural = tuple(int(face in (0, 1, 2)) for face in range(52))
        wildcards = tuple(3 if face == 7 else 0 for face in range(52))
        declared = solo_outcome(natural, 7, True, 1)
        incomplete = solo_outcome(wildcards, 7, False, 1)
        self.assertEqual((declared.utility, declared.won, declared.penalty), (1.0, True, 0))
        self.assertEqual((incomplete.utility, incomplete.won, incomplete.penalty), (-0.5, False, 0))

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
