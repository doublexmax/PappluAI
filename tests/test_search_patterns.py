from __future__ import annotations

from collections import Counter
import unittest

from src.evaluate import evaluate_hand


class TestLongSequenceDeclarations(unittest.TestCase):
    def test_two_exact_wildcards_can_fill_one_sequence_piece(self):
        counts = [0] * 52
        for face in (16, 7, 7, 19, 20, 21):
            counts[face] += 1
        result = evaluate_hand(counts, joker=7, required_sequences=2, cards_in_hand=6)
        self.assertTrue(result.is_valid)
        self.assertEqual(sum(meld.is_pure for meld in result.melds), 2)

    def test_long_runs_remain_complete_declarations_with_their_full_sequence_quota(self):
        for length in range(6, 14):
            patterns = [
                tuple(range(start, start + length))
                for start in range(14 - length)
            ]
            patterns.append(tuple(range(14 - length, 13)) + (0,))
            for suit in range(4):
                joker = ((suit + 1) % 4) * 13 + 9
                for pattern in patterns:
                    for substitute in (False, True):
                        with self.subTest(length=length, suit=suit, pattern=pattern, substitute=substitute):
                            cards = [suit * 13 + rank for rank in pattern]
                            if substitute:
                                cards[0] = joker
                                cards[-1] = joker
                            counts = [0] * 52
                            for face in cards:
                                counts[face] += 1
                            result = evaluate_hand(
                                counts, joker,
                                required_sequences=length // 3,
                                cards_in_hand=length,
                            )
                            self.assertTrue(result.is_valid)
                            self.assertGreaterEqual(
                                sum(meld.is_pure for meld in result.melds),
                                length // 3,
                            )
                            self.assertEqual(
                                Counter(face for meld in result.melds for face in meld.cards),
                                Counter(cards),
                            )


if __name__ == "__main__":
    unittest.main()
