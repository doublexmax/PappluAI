import unittest

from src.evaluate import HandPenalty
from src.game.multiplayer import TerminalSnapshot
from src.game.outcome import DealOutcome, SeatOutcome, placements, solo_outcome


def hand(*counts):
    return tuple(dict(counts).get(face, 0) for face in range(52))


class TestOutcomes(unittest.TestCase):
    def test_timeout_keeps_exact_deadwood_but_does_not_reward_it(self):
        pairs = tuple((face, 2) for face in (0, 9, 10, 12, 13, 22, 23, 25, 26, 35))
        maximum = hand(*pairs, (36, 1))
        one_less = hand(*pairs, (47, 1))
        self.assertEqual(solo_outcome(maximum, 1, False, 5), SeatOutcome(-1.0, False, 210))
        self.assertEqual(solo_outcome(one_less, 1, False, 5), SeatOutcome(-1.0, False, 209))
        score = DealOutcome.score(TerminalSnapshot((maximum, one_less), 1), None, "turns_exhausted", 5)
        self.assertEqual(score.max_points, 210)
        self.assertEqual(score.utilities(), (-0.5, -0.5))
        self.assertEqual(score.net_points(), (-0.5, 0.5))
        self.assertEqual(score.ranks(), (2, 1))
        self.assertEqual(score.margin(1, 0), 1)

    def test_timeout_rewards_qualifying_sequences_not_card_rank(self):
        high = hand((11, 1), (12, 1), (0, 1))
        low = hand((1, 1), (2, 1), (3, 1))
        unmatched = hand((1, 1), (14, 1), (27, 1))
        four = hand(*((offset + rank, 1) for offset in range(0, 52, 13) for rank in (1, 2, 3)))
        self.assertAlmostEqual(solo_outcome(high, 8, False, 5).utility, -0.88)
        self.assertAlmostEqual(solo_outcome(low, 8, False, 5).utility, -0.88)
        self.assertAlmostEqual(solo_outcome(unmatched, 8, False, 5).utility, -1.0)
        self.assertAlmostEqual(solo_outcome(four, 8, False, 5).utility, -0.52)
        self.assertAlmostEqual(solo_outcome(unmatched, 8, False, 0).utility, -0.40)
        score = DealOutcome.score(TerminalSnapshot((high, low, unmatched), 8), None, "turns_exhausted", 1)
        self.assertEqual(score.required_sequences, 1)
        self.assertEqual(score.utilities(), (-0.35, -0.35, -0.8))
        self.assertAlmostEqual(sum(score.utilities()), -1.5)

    def test_zero_sum_and_placements_on_declaration(self):
        score = DealOutcome(0, "win", tuple(HandPenalty(p, (), ()) for p in (0, 0, 210, 210)), 210)
        self.assertEqual(score.utilities(), (1.0, -1 / 3, -1 / 3, -1 / 3))
        self.assertEqual(score.net_points(), (420.0, 0.0, -210.0, -210.0))
        self.assertEqual(sum(score.net_points()), 0.0)
        self.assertAlmostEqual(sum(score.utilities()), 0)
        self.assertEqual(score.ranks(), (1, 2, 3, 3))
        self.assertEqual(score.margin(2, 3), 0)
        self.assertEqual(score.seat(2), SeatOutcome(-1 / 3, False, 210))
        self.assertEqual(placements((25, 30, 25), None, "turns_exhausted"), (1, 3, 1))

    def test_payoffs_with_different_seat_counts_and_deadwood(self):
        losses = (-1.0, -0.5, -1 / 3, -0.25, -0.2)
        for players, loss in enumerate(losses, start=2):
            with self.subTest(players=players):
                points = (42,) + (105,) * (players - 2) + (0,)
                penalties = tuple(HandPenalty(p, (), ()) for p in points)
                declared = DealOutcome(players - 1, "win", penalties, 210)
                expired = DealOutcome(None, "turns_exhausted", penalties, 210)
                self.assertEqual(declared.utilities(), (loss,) * (players - 1) + (1.0,))
                self.assertAlmostEqual(sum(declared.utilities()), 0)
                self.assertEqual(expired.utilities(), (-0.5,) * players)
                payouts = expired.net_points()
                self.assertEqual(sum(payouts), 0.0)
                for payout, points_at_seat in zip(payouts, points):
                    self.assertAlmostEqual(payout, sum(points) / players - points_at_seat)

    def test_solo_net_points_and_fractional_multiplayer_draw(self):
        win = DealOutcome(0, "win", (HandPenalty(0, (), ()),), 210)
        loss = DealOutcome(None, "turns_exhausted", (HandPenalty(26, (), ()),), 210)
        draw = DealOutcome(None, "turns_exhausted", tuple(HandPenalty(p, (), ()) for p in (1, 4, 17)), 210)
        self.assertEqual((win.net_points(), loss.net_points()), ((0.0,), (-26.0,)))
        self.assertEqual(sum(draw.net_points()), 0.0)
        for actual, expected in zip(draw.net_points(), (19 / 3, 10 / 3, -29 / 3)):
            self.assertAlmostEqual(actual, expected)

    def test_solo_declaration_and_invalid_terminal_hands(self):
        winning = hand((0, 1), (1, 1), (2, 1))
        self.assertEqual(solo_outcome(winning, 51, True, 1), SeatOutcome(1, True, 0))
        with self.assertRaisesRegex(ValueError, "final hands"):
            DealOutcome.score(TerminalSnapshot((winning,), 51), None, "turns_exhausted", 1, 4)
        with self.assertRaisesRegex(ValueError, "winning seat"):
            solo_outcome(hand((0, 1), (4, 1), (8, 1)), 51, True, 1)
