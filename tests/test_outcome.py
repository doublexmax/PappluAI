import unittest

from src.evaluate import HandPenalty, minimum_penalty
from src.game.multiplayer import TerminalSnapshot
from src.game.outcome import DealOutcome, SeatOutcome, placements, solo_outcome


def hand(*counts):
    return tuple(dict(counts).get(face, 0) for face in range(52))


class TestOutcomes(unittest.TestCase):
    def test_solo_timeout_ignores_deadwood_and_multiplayer_timeout_rewards_it(self):
        pairs = tuple((face, 2) for face in (0, 9, 10, 12, 13, 22, 23, 25, 26, 35))
        maximum = hand(*pairs, (36, 1))
        one_less = hand(*pairs, (47, 1))
        self.assertEqual(solo_outcome(maximum, 1, False, 5), SeatOutcome(-1.0, False, 210))
        self.assertEqual(solo_outcome(one_less, 1, False, 5), SeatOutcome(-1.0, False, 209))
        score = DealOutcome.score(TerminalSnapshot((maximum, one_less), 1), None, "turns_exhausted", 5)
        self.assertEqual(score.max_points, 210)
        low, high = score.utilities()
        self.assertAlmostEqual(low, -0.35 / 420)
        self.assertEqual(high, -low)
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
        for actual, expected in zip(score.utilities(), (0.15 + 0.35 / 30, 0.15 + 0.35 / 30, -0.3 - 0.35 / 15)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(sum(score.utilities()), 0.0)

    def test_zero_sum_and_placements_on_declaration(self):
        score = DealOutcome(0, "win", tuple(HandPenalty(p, (), ()) for p in (0, 0, 210, 210)), 210)
        for actual, expected in zip(score.utilities(), (0.65 + 0.35 * 2 / 3, -0.65 / 3, -1 / 3, -1 / 3)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(score.net_points(), (420.0, 0.0, -210.0, -210.0))
        self.assertEqual(sum(score.net_points()), 0.0)
        self.assertEqual(sum(score.utilities()), 0.0)
        self.assertEqual(score.ranks(), (1, 2, 3, 3))
        self.assertEqual(score.margin(2, 3), 0)
        self.assertEqual(score.seat(2), SeatOutcome(score.utilities()[2], False, 210))
        self.assertEqual(placements((25, 30, 25), None, "turns_exhausted"), (1, 3, 1))

    def test_payoffs_with_different_seat_counts_and_deadwood(self):
        for players in range(2, 7):
            with self.subTest(players=players):
                clean = minimum_penalty(hand((0, 1), (1, 1), (2, 1)), 51, 1)
                full = HandPenalty(210, (), ())
                partial = HandPenalty(43, (), ())
                draw = DealOutcome(None, "turns_exhausted", (partial,) * players, 210, 1)
                self.assertEqual(draw.utilities(), (0.0,) * players)
                fractional_draw = DealOutcome(None, "turns_exhausted", (clean,) * players, 210, 7)
                self.assertEqual(fractional_draw.utilities(), (0.0,) * players)
                patterns = ((clean,) * (players - 1), (full,) * (players - 1), ((partial, full) * players)[:players - 1])
                for winner in range(players):
                    for losers in patterns:
                        hands = (*losers[:winner], clean, *losers[winner:])
                        with self.subTest(winner=winner, hands=hands):
                            win = DealOutcome(winner, "win", hands, 210, 1).utilities()
                            expiry = DealOutcome(None, "turns_exhausted", hands, 210, 1).utilities()
                            self.assertEqual(sum(win), 0.0)
                            self.assertEqual(sum(expiry), 0.0)
                            self.assertTrue(all(-0.475 <= value <= 0.475 for value in expiry))
                            self.assertTrue(0.65 <= win[winner] <= 1.0)
                            self.assertGreater(win[winner], max(expiry))
                            self.assertTrue(all(-1.0 <= value <= -0.65 / (players - 1)
                                                for seat, value in enumerate(win) if seat != winner))
                            self.assertTrue(all(draw.utilities()[seat] > win[seat]
                                                for seat in range(players) if seat != winner))
                soft = DealOutcome(0, "win", (clean, partial, *((full,) * (players - 2))), 210, 1).utilities()
                hard = DealOutcome(0, "win", (clean, *((full,) * (players - 1))), 210, 1).utilities()
                self.assertGreater(soft[1], hard[1])
                self.assertLess(soft[0], hard[0])

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
