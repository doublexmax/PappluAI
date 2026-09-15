import itertools
import random
import unittest
from collections import Counter
from functools import lru_cache

from src.evaluate import evaluate_hand


def counts(cards):
    hand = [0] * 52
    for card in cards:
        hand[card] += 1
    return hand


def sequence_possible(cards, joker, pure):
    fixed = [
        card for card in cards
        if (card != joker if pure else card % 13 != joker % 13)
    ]
    if not 3 <= len(cards) <= 13 or len(set(fixed)) != len(fixed):
        return False
    if len({card // 13 for card in fixed}) > 1:
        return False
    ranks = {card % 13 for card in fixed}
    for order in (tuple(range(13)), tuple(range(1, 13)) + (0,)):
        for start in range(14 - len(cards)):
            if ranks <= set(order[start:start + len(cards)]):
                return True
    return False


@lru_cache(maxsize=None)
def group_quality(cards, joker):
    if sequence_possible(cards, joker, pure=True):
        return 1
    if sequence_possible(cards, joker, pure=False):
        return 0
    fixed = [card for card in cards if card % 13 != joker % 13]
    if (
        len(cards) in (3, 4)
        and len({card % 13 for card in fixed}) <= 1
        and len({card // 13 for card in fixed}) == len(fixed)
    ):
        return 0
    return -1


def oracle(cards, joker, required):
    cards = tuple(cards)

    @lru_cache(maxsize=None)
    def partition(remaining, needed):
        if not remaining:
            return needed == 0
        if len(remaining) < 3 or len(remaining) < needed * 3:
            return False
        for size in range(3, min(13, len(remaining)) + 1):
            for tail in itertools.combinations(range(1, len(remaining)), size - 1):
                chosen = (0,) + tail
                quality = group_quality(
                    tuple(sorted(cards[remaining[i]] for i in chosen)), joker
                )
                if quality < 0:
                    continue
                rest = tuple(
                    card for i, card in enumerate(remaining) if i not in chosen
                )
                if partition(rest, max(0, needed - quality)):
                    return True
        return False

    return partition(tuple(range(len(cards))), required)


class OracleTests(unittest.TestCase):
    def assert_witness(self, cards, joker, required, result):
        if not result.is_valid:
            self.assertEqual(result.melds, ())
            return
        self.assertEqual(
            Counter(card for meld in result.melds for card in meld.cards),
            Counter(cards),
        )
        self.assertGreaterEqual(sum(meld.is_pure for meld in result.melds), required)
        for meld in result.melds:
            actual = meld.cards
            represented = meld.represented_cards
            self.assertEqual(len(actual), len(represented))
            self.assertEqual(len(set(represented)), len(represented))
            for card, target in zip(actual, represented):
                self.assertTrue(0 <= target < 52)
                self.assertTrue(card == target or card % 13 == joker % 13)
            if meld.kind == "sequence":
                self.assertTrue(sequence_possible(represented, -1, pure=True))
                qualifies = all(
                    card == target or card == joker
                    for card, target in zip(actual, represented)
                )
                self.assertEqual(meld.is_pure, qualifies)
            else:
                self.assertEqual(meld.kind, "set")
                self.assertIn(len(actual), (3, 4))
                self.assertEqual(len({card % 13 for card in represented}), 1)
                self.assertFalse(meld.is_pure)

    def compare(self, cards, joker, required):
        expected = oracle(cards, joker, required)
        result = evaluate_hand(
            counts(cards), joker, required_sequences=required,
            cards_in_hand=len(cards),
        )
        self.assertEqual(
            result.is_valid, expected,
            (cards, joker, required),
        )
        self.assert_witness(cards, joker, required, result)

    def test_exhaustive_six_card_multisets(self):
        pools = (
            (16, 17, 18, 7, 33, 3),
            (6, 19, 32, 7, 33, 34),
        )
        for pool in pools:
            for cards in itertools.combinations_with_replacement(pool, 6):
                for required in (0, 1, 2):
                    self.compare(cards, 7, required)

    def test_seeded_small_hands(self):
        randomizer = random.Random(20260913)
        for case in range(180):
            joker = randomizer.randrange(52)
            rank = joker % 13
            size = randomizer.choice((3, 4, 6, 7, 9))
            if case % 2:
                pool = list(range(52))
            else:
                suit = randomizer.randrange(4)
                start = randomizer.randrange(9)
                pool = [suit * 13 + r for r in range(start, start + 5)]
            pool += [suit * 13 + rank for suit in range(4)] * 2
            cards = [randomizer.choice(pool) for _ in range(size)]
            self.compare(cards, joker, randomizer.randrange(size // 3 + 1))

    def test_constructed_nine_card_hands(self):
        randomizer = random.Random(1741)
        for _ in range(100):
            joker = randomizer.randrange(52)
            cards = []
            for _ in range(3):
                if randomizer.randrange(2):
                    suit = randomizer.randrange(4)
                    order = list(range(13)) + [0]
                    start = randomizer.randrange(12)
                    group = [suit * 13 + r for r in order[start:start + 3]]
                else:
                    rank = randomizer.randrange(13)
                    group = [s * 13 + rank for s in randomizer.sample(range(4), 3)]
                for i in range(3):
                    if randomizer.randrange(4) == 0:
                        group[i] = randomizer.randrange(4) * 13 + joker % 13
                cards.extend(group)
            self.compare(cards, joker, randomizer.randrange(4))

    def test_documented_default_hand(self):
        cards = (
            2, 3, 4,
            18, 19, 20,
            34, 35, 36,
            39, 40, 41,
            10, 11, 12,
            1, 14, 27,
            17, 30, 43,
        )
        result = evaluate_hand(counts(cards), 7)
        self.assertTrue(result.is_valid)
        self.assert_witness(cards, 7, 5, result)

    def test_constructed_default_hands_and_suit_permutations(self):
        randomizer = random.Random(215)
        for case in range(24):
            joker = randomizer.randrange(52)
            cards = []
            for _ in range(5):
                suit = randomizer.randrange(4)
                start = randomizer.randrange(12)
                order = list(range(13)) + [0]
                group = [suit * 13 + r for r in order[start:start + 3]]
                if randomizer.randrange(3) == 0:
                    group[randomizer.randrange(3)] = joker
                cards.extend(group)
            for _ in range(2):
                rank = randomizer.randrange(13)
                group = [s * 13 + rank for s in randomizer.sample(range(4), 3)]
                if randomizer.randrange(2):
                    group[randomizer.randrange(3)] = (
                        randomizer.randrange(4) * 13 + joker % 13
                    )
                cards.extend(group)
            for permutation in ((0, 1, 2, 3), (2, 0, 3, 1)):
                remapped = [permutation[c // 13] * 13 + c % 13 for c in cards]
                selected = permutation[joker // 13] * 13 + joker % 13
                with self.subTest(case=case, permutation=permutation):
                    result = evaluate_hand(counts(remapped), selected)
                    self.assertTrue(result.is_valid)
                    self.assert_witness(remapped, selected, 5, result)


if __name__ == "__main__":
    unittest.main()
