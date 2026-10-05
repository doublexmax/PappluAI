from __future__ import annotations

import itertools
import random
import unittest

from src.evaluate import evaluate_hand, minimum_penalty
from tests.test_evaluate_oracle import group_quality


SUITS = "shdc"
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")


def card(name):
    return SUITS.index(name[-1]) * 13 + RANKS.index(name[:-1])


def counts(faces):
    hand = [0] * 52
    for face in faces:
        hand[face] += 1
    return hand


def score(names, joker, required):
    return minimum_penalty(counts(map(card, names.split())), card(joker), required)


def card_points(face, joker):
    rank = face % 13
    if rank == joker % 13:
        return 0
    return 10 if rank == 0 else min(rank + 1, 10)


def brute_force_penalty(faces, joker, required):
    def groupings(remaining):
        if not remaining:
            yield ()
            return
        first, rest = remaining[0], remaining[1:]
        yield from groupings(rest)
        for size in range(2, len(rest) + 1):
            for partners in itertools.combinations(rest, size):
                group = (first,) + partners
                quality = group_quality(tuple(sorted(faces[i] for i in group)), joker)
                if quality < 0:
                    continue
                left = tuple(i for i in rest if i not in partners)
                for tail in groupings(left):
                    yield ((group, quality),) + tail

    best = best_meeting_quota = None
    for grouping in groupings(tuple(range(len(faces)))):
        quota_met = sum(quality for _, quality in grouping) >= required
        exempt = {i for group, quality in grouping if quota_met or quality for i in group}
        points = sum(
            card_points(face, joker) for i, face in enumerate(faces) if i not in exempt
        )
        best = points if best is None else min(best, points)
        if quota_met:
            best_meeting_quota = (
                points if best_meeting_quota is None else min(best_meeting_quota, points)
            )
    return best, best_meeting_quota == best


def near_complete_hand(randomizer, replaced, wild_bias):
    joker = randomizer.randrange(52)
    order = list(range(13)) + [0]
    faces = []
    for _ in range(5):
        suit = randomizer.randrange(4)
        start = randomizer.randrange(12)
        faces.extend(suit * 13 + rank for rank in order[start:start + 3])
    for _ in range(2):
        rank = randomizer.randrange(13)
        faces.extend(suit * 13 + rank for suit in randomizer.sample(range(4), 3))
    wilds = [suit * 13 + joker % 13 for suit in range(4)]
    for index in randomizer.sample(range(21), replaced):
        if randomizer.random() < wild_bias:
            faces[index] = randomizer.choice(wilds)
        else:
            faces[index] = randomizer.randrange(52)
    return counts(faces), joker


def dealt_hand(randomizer):
    shoe = [face for face in range(52) for _ in range(3)]
    randomizer.shuffle(shoe)
    joker = shoe.pop()
    return counts(shoe[:21]), joker


class PenaltyTests(unittest.TestCase):
    def assert_witness(self, hand, joker, required, penalty):
        covered = [0] * 52
        for meld in penalty.exempt_melds:
            quality = group_quality(tuple(sorted(meld.cards)), joker)
            self.assertGreaterEqual(quality, 0, meld)
            self.assertEqual(meld.is_pure, quality == 1, meld)
            for face in meld.cards:
                covered[face] += 1
        self.assertEqual(
            [melded + counted for melded, counted in zip(covered, penalty.counted_cards)],
            list(hand),
        )
        self.assertEqual(
            penalty.points,
            sum(card_points(face, joker) * n for face, n in enumerate(penalty.counted_cards)),
        )
        if penalty.qualifying_sequences < required:
            self.assertTrue(all(meld.is_pure for meld in penalty.exempt_melds), penalty)

    def assert_matches_brute_force(self, faces, joker, required):
        hand = counts(faces)
        penalty = minimum_penalty(hand, joker, required)
        points, optimum_meets_quota = brute_force_penalty(faces, joker, required)
        self.assertEqual(penalty.points, points, (faces, joker, required))
        self.assertEqual(
            penalty.qualifying_sequences >= required, optimum_meets_quota,
            (faces, joker, required),
        )
        self.assert_witness(hand, joker, required, penalty)

    def test_set_of_two_sevens_and_a_wildcard_counts_until_the_quota_is_met(self):
        cases = (
            ("2s 3s 4s 2h 3h 4h 7s 7h Kd", 14),
            ("2s 3s 4s 2h 3h 4h 2d 3d 4d 7s 7h Kd", 0),
            ("2s 3s 4s 2h 3h 4h 7s 7h 7d", 21),
            ("2s 3s 4s 2h 3h 4h 2d 3d 4d 7s 7h 7d", 0),
        )
        for names, points in cases:
            with self.subTest(names=names):
                self.assertEqual(score(names, "Kc", 3).points, points)

        below = score("2s 3s 4s 2h 3h 4h 7s 7h Kd", "Kc", 3)
        self.assertEqual(below.qualifying_sequences, 2)
        self.assertEqual(
            sorted(sorted(meld.cards) for meld in below.exempt_melds),
            [[card("2s"), card("3s"), card("4s")], [card("2h"), card("3h"), card("4h")]],
        )
        self.assertEqual(below.counted_cards, tuple(counts(map(card, ("7s", "7h", "Kd")))))

        met = score("2s 3s 4s 2h 3h 4h 2d 3d 4d 7s 7h Kd", "Kc", 3)
        self.assertEqual(met.qualifying_sequences, 3)
        self.assertIn(
            sorted((card("7s"), card("7h"), card("Kd"))),
            [sorted(meld.cards) for meld in met.exempt_melds],
        )
        self.assertEqual(met.counted_cards, (0,) * 52)

    def test_card_points(self):
        cases = (
            ("As", "Kc", 10), ("2h", "Kc", 2), ("3d", "Kc", 3), ("4c", "Kc", 4),
            ("5s", "Kc", 5), ("6h", "Kc", 6), ("7d", "Kc", 7), ("8c", "Kc", 8),
            ("9s", "Kc", 9), ("10h", "Kc", 10), ("Jd", "Kc", 10), ("Qc", "Kc", 10),
            ("Ks", "2c", 10), ("Kd", "Kc", 0), ("Kc", "Kc", 0), ("Ah", "As", 0),
        )
        for name, joker, points in cases:
            with self.subTest(card=name, joker=joker):
                self.assertEqual(score(name, joker, 5).points, points)
        self.assertEqual(score("As Kh Qd Jc 10s 2h 8d", "8c", 0).points, 52)

    def test_global_optimum_over_overlapping_melds(self):
        self.assertEqual(score("4h 5h 6h 7h 7s 7d", "Kc", 1).points, 0)
        self.assertEqual(score("3h 4h 5h 6h 7h 3s 3d 7s 7d", "Kc", 1).points, 0)
        self.assertEqual(score("9s 10s Js 9h 9d", "Kc", 0).points, 18)

    def test_duplicate_faces(self):
        self.assertEqual(score("5h 5h 6h 6h 7h 7h", "Kc", 2).points, 0)
        self.assertEqual(score("7s 7s 8d", "8s", 0).points, 14)
        self.assertEqual(score("4s 4s 5s 6s", "Kc", 1).points, 4)

    def test_only_the_exact_joker_substitutes_in_a_qualifying_sequence(self):
        self.assertEqual(score("4h 5h 8d Kc Kd Ks", "8s", 1).points, 39)
        self.assertEqual(score("4h 5h 8s Kc Kd Ks", "8s", 1).points, 0)
        self.assertEqual(score("7d 8d 9d Kc Kd Ks", "8s", 1).points, 0)

    def test_ace_is_high_or_low_without_wrapping(self):
        self.assertEqual(score("Qh Kh Ah", "8c", 1).points, 0)
        self.assertEqual(score("Ah 2h 3h", "8c", 1).points, 0)
        self.assertEqual(score("Kh Ah 2h", "8c", 1).points, 22)

    def test_zero_quota_exempts_every_meld(self):
        self.assertEqual(score("7s 7h 7d", "Kc", 0).points, 0)
        self.assertEqual(score("7s 7h 7d", "Kc", 1).points, 21)
        self.assertEqual(score("4h 5h 8d", "8s", 0).points, 0)
        self.assertEqual(score("4h 5h 8d", "8s", 1).points, 9)

    def test_long_run_counts_as_several_sequences(self):
        self.assertEqual(score("3s 4s 5s 6s 7s 8s 9h 9d 9c", "Kc", 2).points, 0)
        self.assertEqual(score("3s 4s 5s 6s 7s 8s 9h 9d 9c", "Kc", 3).points, 27)

    def test_zero_points_without_a_declaration(self):
        names = "As 2s 3s 4s 5s 9h 10h Jh Qh Kh 2d 3d 4d 5d 6d 9c 10c Jc Qc Kc 8d"
        hand = counts(map(card, names.split()))
        penalty = minimum_penalty(hand, card("8s"))
        self.assertFalse(evaluate_hand(hand, card("8s")).is_valid)
        self.assertEqual((penalty.points, penalty.qualifying_sequences), (0, 4))
        self.assertEqual(penalty.counted_cards, tuple(counts([card("8d")])))

        hand = counts(map(card, "4h 5h 6h 8h 8d 8c".split()))
        self.assertFalse(
            evaluate_hand(hand, card("8s"), required_sequences=2, cards_in_hand=6).is_valid
        )
        self.assertEqual(minimum_penalty(hand, card("8s"), 2).points, 0)

    def test_a_tie_reports_the_grouping_that_meets_the_quota(self):
        penalty = score("4h 5h 6h 8s 8s 8s", "8s", 2)
        self.assertEqual((penalty.points, penalty.qualifying_sequences), (0, 2))
        penalty = score("2s 2s 2s 2d 3d 4d 8d 8d Jd Qd", "2s", 3)
        self.assertEqual((penalty.points, penalty.qualifying_sequences), (8, 3))

    def test_malformed_arguments_raise(self):
        hand = counts([card("As")])
        with self.assertRaises(ValueError):
            minimum_penalty([0] * 51, 0)
        with self.assertRaises(ValueError):
            minimum_penalty(hand, 52)
        with self.assertRaises(TypeError):
            minimum_penalty(hand, True)
        with self.assertRaises(ValueError):
            minimum_penalty(hand, 0, -1)
        with self.assertRaises(TypeError):
            minimum_penalty(hand, 0, 1.5)
        hand[0] = -1
        with self.assertRaises(ValueError):
            minimum_penalty(hand, 0)

    def test_small_multisets_match_brute_force(self):
        pools = (
            ("8s", "4h 5h 6h 8s 8d 4s"),
            ("8s", "7s 7h 7d 8s 8d 9d"),
            ("8c", "Qh Kh Ah Kd Ks 8d"),
        )
        for joker_name, pool_names in pools:
            joker = card(joker_name)
            pool = [card(name) for name in pool_names.split()]
            for faces in itertools.combinations_with_replacement(pool, 5):
                for required in (0, 1, 2):
                    self.assert_matches_brute_force(faces, joker, required)

    def test_seeded_small_hands_match_brute_force(self):
        randomizer = random.Random(20261004)
        order = list(range(13)) + [0]
        for _ in range(240):
            joker = randomizer.randrange(52)
            suit = randomizer.randrange(4)
            start = randomizer.randrange(10)
            window = [suit * 13 + rank for rank in order[start:start + 5]]
            rank = randomizer.choice(order[start:start + 5])
            pool = (
                window
                + [other * 13 + rank for other in range(4)]
                + [other * 13 + joker % 13 for other in range(4)]
                + [joker]
            )
            faces = [randomizer.choice(pool) for _ in range(randomizer.choice((3, 5, 7, 8)))]
            required = randomizer.randrange(len(faces) // 3 + 2)
            self.assert_matches_brute_force(faces, joker, required)

    def test_full_hands_have_conserving_legal_witnesses(self):
        randomizer = random.Random(7031)
        for case in range(90):
            if case % 3 == 0:
                hand, joker = dealt_hand(randomizer)
                replaced = None
            else:
                replaced = case // 3 % 9
                hand, joker = near_complete_hand(
                    randomizer, replaced, wild_bias=0.5 if case % 3 == 2 else 0.0,
                )
            with self.subTest(case=case, hand=hand, joker=joker):
                penalty = minimum_penalty(hand, joker)
                self.assert_witness(hand, joker, 5, penalty)
                declared = evaluate_hand(hand, joker).is_valid
                if replaced == 0:
                    self.assertTrue(declared)
                if declared:
                    self.assertEqual(penalty.points, 0)
                    self.assertGreaterEqual(penalty.qualifying_sequences, 5)
                if not any(penalty.counted_cards) and penalty.qualifying_sequences >= 5:
                    self.assertTrue(declared)


if __name__ == "__main__":
    unittest.main()
