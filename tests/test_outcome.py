from __future__ import annotations

import unittest

from src.evaluate import HandPenalty, Meld, card_points
from src.game.multiplayer import MatchConfig, MultiplayerEnv, TerminalSnapshot
from src.game.outcome import DealOutcome, placements, solo_outcome


class TestDealOutcome(unittest.TestCase):
    @staticmethod
    def outcome(points, winner=None, sequences=None, required_sequences=5):
        sequences = sequences if sequences is not None else (0,) * len(points)
        sequence = Meld("sequence", (0, 1, 2), (0, 1, 2), True)
        return DealOutcome(winner, "win" if winner is not None else "turns_exhausted", tuple(
            HandPenalty(point, (sequence,) * count, (0,) * 52)
            for point, count in zip(points, sequences)
        ), required_sequences)

    def test_declarer_wins_without_changing_ranks_or_margins(self):
        outcome = self.outcome((0, 20, 20, 130), 0)
        self.assertEqual(outcome.ranks(), (1, 2, 2, 4))
        self.assertEqual(
            tuple(round(outcome.seat(seat).utility, 12) for seat in range(4)),
            (1.0, -0.45, -0.45, -0.85),
        )
        self.assertEqual(outcome.margin(0, 3), 1)
        self.assertEqual(outcome.margin(3, 0), -1)
        self.assertEqual(outcome.margin(1, 2), 0)
        self.assertTrue(outcome.seat(0).won)
        self.assertEqual(outcome.seat(3).penalty, 130)
        self.assertEqual(placements((0, 0, 20), 0, "win"), (1, 2, 3))

    def test_turn_cap_ranks_points_without_counting_declaration_wins(self):
        for points, ranks in (
            ((40, 0), (2, 1)),
            ((40, 0, 0, 100), (3, 1, 1, 4)),
            ((20, 20), (1, 1)),
        ):
            with self.subTest(points=points):
                outcome = self.outcome(points)
                self.assertEqual(outcome.ranks(), ranks)
                self.assertEqual(tuple(outcome.seat(seat).won for seat in range(len(points))), (False,) * len(points))

    def test_opponent_declaration_penalties_are_monotonic_and_capped(self):
        for players in (2, 3, 4):
            for winner in range(players):
                utilities = []
                for points, expected in ((0, -0.35), (3, -0.365), (40, -0.55), (100, -0.85), (140, -0.85)):
                    penalties = [points] * players
                    penalties[winner] = 0
                    outcome = self.outcome(penalties, winner, (5,) * players)
                    self.assertEqual(outcome.seat(winner).utility, 1.0)
                    for seat in range(players):
                        if seat == winner:
                            continue
                        with self.subTest(players=players, winner=winner, seat=seat, points=points):
                            result = outcome.seat(seat)
                            self.assertAlmostEqual(result.utility, expected, places=12)
                            self.assertLessEqual(result.utility, -0.35)
                            self.assertFalse(result.won)
                            self.assertEqual(result.penalty, points)
                    utilities.append(outcome.seat((winner + 1) % players).utility)
                self.assertEqual(utilities, sorted(utilities, reverse=True))
                self.assertEqual(utilities[-2:], [-0.85, -0.85])

    def test_exhausted_utilities_are_strictly_negative_and_capped(self):
        for players in (1, 2, 3, 4):
            for sequences, expected in (
                (0, (-0.8, -0.88, -1.0, -1.0)),
                (1, (-0.76, -0.84, -0.96, -0.96)),
                (5, (-0.6, -0.68, -0.8, -0.8)),
                (6, (-0.6, -0.68, -0.8, -0.8)),
            ):
                for points, utility in zip((0, 40, 100, 140), expected):
                    outcome = self.outcome((points,) * players, sequences=(sequences,) * players)
                    for seat in range(players):
                        with self.subTest(players=players, sequences=sequences, points=points, seat=seat):
                            result = outcome.seat(seat)
                            self.assertAlmostEqual(result.utility, utility, places=12)
                            self.assertTrue(-1.0 <= round(result.utility, 12) <= -0.6)
                            self.assertLessEqual(result.utility, -0.6)
                            self.assertFalse(result.won)
                            self.assertEqual(result.penalty, points)
        self.assertEqual(self.outcome((0,)).ranks(), (1,))

    def test_sequences_beat_low_point_hoarding_on_exhaustion(self):
        outcome = self.outcome((0, 80), sequences=(0, 5))
        self.assertAlmostEqual(outcome.seat(0).utility, -0.8, places=12)
        self.assertAlmostEqual(outcome.seat(1).utility, -0.76, places=12)
        self.assertGreater(outcome.seat(1).utility, outcome.seat(0).utility)
        self.assertEqual(outcome.ranks(), (1, 2))
        self.assertEqual(outcome.margin(0, 1), 1)
        self.assertFalse(outcome.seat(0).won)

    def test_exhausted_utility_uses_required_sequences(self):
        outcome = self.outcome((0, 0, 0, 0), sequences=(0, 1, 2, 3), required_sequences=2)
        for seat, expected in enumerate((-0.8, -0.7, -0.6, -0.6)):
            self.assertAlmostEqual(outcome.seat(seat).utility, expected, places=12)
        default = DealOutcome(None, "turns_exhausted", outcome.penalties)
        self.assertAlmostEqual(default.seat(1).utility, -0.76, places=12)

    def test_zero_required_sequences_gives_full_sequence_credit(self):
        outcome = self.outcome((0, 40, 100, 140), sequences=(0, 1, 5, 6), required_sequences=0)
        for seat, expected in enumerate((-0.6, -0.68, -0.8, -0.8)):
            self.assertAlmostEqual(outcome.seat(seat).utility, expected, places=12)

    def test_zero_point_concession_and_timeout_have_distinct_negative_utilities(self):
        exhausted = self.outcome((0, 0), sequences=(0, 5))
        declared = self.outcome((0, 0), winner=0, sequences=(0, 5))
        self.assertAlmostEqual(exhausted.seat(0).utility, -0.8, places=12)
        self.assertAlmostEqual(exhausted.seat(1).utility, -0.6, places=12)
        self.assertEqual(declared.seat(0).utility, 1.0)
        self.assertAlmostEqual(declared.seat(1).utility, -0.35, places=12)

    def test_score_uses_minimum_penalty_for_each_final_hand(self):
        hands = (
            tuple(int(face in (0, 1, 2)) for face in range(52)),
            tuple(int(face in (0, 1, 12)) for face in range(52)),
        )
        snapshot = TerminalSnapshot(hands, 7)
        outcome = DealOutcome.score(snapshot, 0, "win", 1)
        self.assertEqual(tuple(penalty.points for penalty in outcome.penalties), (0, 22))
        self.assertEqual(outcome.required_sequences, 1)
        self.assertEqual(tuple(round(outcome.seat(seat).utility, 12) for seat in range(2)), (1.0, -0.46))
        exhausted = DealOutcome.score(snapshot, None, "turns_exhausted", 1)
        self.assertAlmostEqual(exhausted.seat(0).utility, -0.6, places=12)
        self.assertAlmostEqual(exhausted.seat(1).utility, -0.844, places=12)

    def test_solo_helper_matches_the_deal_objective(self):
        natural = tuple(int(face in (0, 1, 2)) for face in range(52))
        wildcard = tuple(int(face == 7) for face in range(52))
        deadwood = tuple(int(face in (0, 1, 12)) for face in range(52))
        for hand, declared, required, utility, points in (
            (natural, True, 1, 1.0, 0),
            (natural, False, 1, -0.6, 0),
            (natural, False, 2, -0.7, 0),
            (natural, False, 5, -0.76, 0),
            (wildcard, False, 1, -0.8, 0),
            (wildcard, False, 0, -0.6, 0),
            (deadwood, False, 1, -0.844, 22),
            (deadwood, False, 0, -0.644, 22),
        ):
            with self.subTest(declared=declared, required=required, utility=utility):
                solo = solo_outcome(hand, 7, declared, required)
                self.assertAlmostEqual(solo.utility, utility, places=12)
                self.assertEqual((solo.won, solo.penalty), (declared, points))
                outcome = DealOutcome.score(
                    TerminalSnapshot((hand,), 7), 0 if declared else None,
                    "win" if declared else "turns_exhausted", required,
                )
                self.assertEqual(solo, outcome.seat(0))
                for players in (2, 3, 4):
                    multiplayer = DealOutcome(
                        outcome.winner, outcome.ending, outcome.penalties * players, outcome.required_sequences,
                    )
                    self.assertEqual(solo, multiplayer.seat(0))

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
