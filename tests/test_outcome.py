import unittest

from src.evaluate import HandPenalty
from src.game.multiplayer import TerminalSnapshot
from src.game.outcome import DealOutcome, SeatOutcome, placements, solo_outcome


def hand(*counts):
    return tuple(dict(counts).get(face, 0) for face in range(52))


class TestOutcomes(unittest.TestCase):
    def test_every_point_counts_above_the_old_cap(self):
        pairs = tuple((face, 2) for face in (0, 9, 10, 12, 13, 22, 23, 25, 26, 35))
        maximum = hand(*pairs, (36, 1))
        one_less = hand(*pairs, (47, 1))
        self.assertEqual(solo_outcome(maximum, 1, False, 5), SeatOutcome(-1.0, False, 210))
        self.assertEqual(solo_outcome(one_less, 1, False, 5), SeatOutcome(-209 / 210, False, 209))
        score = DealOutcome.score(TerminalSnapshot((maximum, one_less), 1), None, "turns_exhausted", 5)
        self.assertEqual(score.max_points, 210)
        self.assertEqual(score.utilities(), (-1 / 420, 1 / 420))
        self.assertEqual(score.ranks(), (2, 1))
        self.assertEqual(score.margin(1, 0), 1)

    def test_zero_sum_and_placements_on_declaration(self):
        score = DealOutcome(0, "win", tuple(HandPenalty(p, (), ()) for p in (0, 0, 210, 210)), 210)
        self.assertEqual(score.utilities(), (5 / 6, 1 / 6, -0.5, -0.5))
        self.assertAlmostEqual(sum(score.utilities()), 0)
        self.assertEqual(score.ranks(), (1, 2, 3, 3))
        self.assertEqual(score.margin(2, 3), 0)
        self.assertEqual(score.seat(2), SeatOutcome(-0.5, False, 210))
        self.assertEqual(placements((25, 30, 25), None, "turns_exhausted"), (1, 3, 1))

    def test_solo_declaration_and_invalid_terminal_hands(self):
        winning = hand((0, 1), (1, 1), (2, 1))
        self.assertEqual(solo_outcome(winning, 51, True, 1), SeatOutcome(1, True, 0))
        with self.assertRaisesRegex(ValueError, "final hands"):
            DealOutcome.score(TerminalSnapshot((winning,), 51), None, "turns_exhausted", 1, 4)
        with self.assertRaisesRegex(ValueError, "winning seat"):
            solo_outcome(hand((0, 1), (4, 1), (8, 1)), 51, True, 1)
