from __future__ import annotations

import itertools
import unittest

from src.evaluate import (
    NUM_FACES,
    HandEvaluation,
    Meld,
    evaluate_hand,
    hand_reward,
    is_valid_hand,
)


def face(suit, rank):
    return suit * 13 + rank


def empty_hand():
    return [0] * NUM_FACES


def add(hand, card, n=1):
    hand[card] += n
    return hand


def from_cards(cards):
    hand = empty_hand()
    for card in cards:
        add(hand, card)
    return hand


JOKER_8S = face(0, 7)
JOKER_8H = face(1, 7)
JOKER_8D = face(2, 7)
JOKER_8C = face(3, 7)


def classic_valid_21():
    cards = []
    for suit in range(3):
        cards.extend([face(suit, 0), face(suit, 1), face(suit, 2)])
    cards.extend([face(0, 3), face(0, 4), face(0, 5)])
    cards.extend([face(1, 6), face(1, 7), face(1, 8)])
    for suit in range(3):
        cards.append(face(suit, 9))
    for suit in range(3):
        cards.append(face(suit, 10))
    return from_cards(cards)


class TestEvaluateValid(unittest.TestCase):
    def test_classic_21_five_pure(self):
        hand = classic_valid_21()
        self.assertEqual(sum(hand), 21)
        ev = evaluate_hand(hand, JOKER_8S)
        self.assertTrue(ev.is_valid)
        self.assertEqual(hand_reward(hand, JOKER_8S), 1.0)
        self.assertTrue(is_valid_hand(hand, JOKER_8S))
        pure = [m for m in ev.melds if m.is_pure]
        self.assertGreaterEqual(len(pure), 5)
        self._assert_witness(hand, ev)

    def test_close_negative_broken_run(self):
        hand = classic_valid_21()
        hand[face(0, 5)] -= 1
        hand[face(3, 12)] += 1
        self.assertEqual(sum(hand), 21)
        self.assertFalse(is_valid_hand(hand, JOKER_8S))
        self.assertEqual(hand_reward(hand, JOKER_8S), 0.0)
        ev = evaluate_hand(hand, JOKER_8S)
        self.assertFalse(ev.is_valid)
        self.assertEqual(ev.melds, ())

    def test_exact_joker_in_pure_sequence(self):
        cards = [
            face(1, 3), face(1, 4), JOKER_8S,
            face(0, 0), face(0, 1), face(0, 2),
            face(2, 0), face(2, 1), face(2, 2),
            face(3, 0), face(3, 1), face(3, 2),
            face(0, 9), face(1, 9), face(2, 9),
            face(0, 10), face(1, 10), face(2, 10),
            face(0, 3), face(0, 4), face(0, 5),
        ]
        hand = from_cards(cards)
        self.assertEqual(sum(hand), 21)
        ev = evaluate_hand(hand, JOKER_8S)
        self.assertTrue(ev.is_valid, ev)
        pure = [m for m in ev.melds if m.is_pure]
        self.assertGreaterEqual(len(pure), 5)
        self._assert_witness(hand, ev)

    def test_off_suit_rank_joker_cannot_fill_pure(self):
        cards = [
            face(1, 3), face(1, 4), JOKER_8D,
            face(0, 0), face(0, 1), face(0, 2),
            face(2, 0), face(2, 1), face(2, 2),
            face(3, 0), face(3, 1), face(3, 2),
            face(0, 9), face(1, 9), face(2, 9),
            face(0, 10), face(1, 10), face(2, 10),
            face(0, 3), face(0, 4), face(0, 5),
        ]
        hand = from_cards(cards)
        self.assertFalse(is_valid_hand(hand, JOKER_8S))

    def test_off_suit_joker_natural_in_required_sequence(self):
        cards = [
            face(2, 6), face(2, 7), face(2, 8),
            face(0, 0), face(0, 1), face(0, 2),
            face(1, 0), face(1, 1), face(1, 2),
            face(3, 0), face(3, 1), face(3, 2),
            face(0, 9), face(1, 9), face(2, 9),
            face(0, 10), face(1, 10), face(2, 10),
            face(0, 3), face(0, 4), face(0, 5),
        ]
        hand = from_cards(cards)
        self.assertTrue(is_valid_hand(hand, JOKER_8S))
        self._assert_witness(hand, evaluate_hand(hand, JOKER_8S))

    def test_rank_joker_in_residual_set(self):
        cards = []
        for suit in range(3):
            cards.extend([face(suit, 0), face(suit, 1), face(suit, 2)])
        cards.extend([face(0, 3), face(0, 4), face(0, 5)])
        cards.extend([face(1, 6), face(1, 7), face(1, 8)])
        cards.extend([face(0, 9), face(1, 9), JOKER_8D])
        cards.extend([face(0, 10), face(1, 10), face(2, 10)])
        hand = from_cards(cards)
        self.assertEqual(sum(hand), 21)
        ev = evaluate_hand(hand, JOKER_8S)
        self.assertTrue(ev.is_valid)
        self.assertTrue(any(m.kind == "set" and not m.is_pure for m in ev.melds))
        self._assert_witness(hand, ev)

    def test_duplicate_natural_faces_in_set_rejected(self):
        hand = empty_hand()
        add(hand, face(0, 6), 2)
        add(hand, face(0, 10), 1)
        self.assertFalse(is_valid_hand(hand, JOKER_8S, required_sequences=0, cards_in_hand=3))

    def test_duplicate_natural_with_wild_in_set_rejected(self):
        hand = empty_hand()
        add(hand, face(0, 6), 2)
        add(hand, JOKER_8H, 1)
        self.assertFalse(is_valid_hand(hand, JOKER_8S, required_sequences=0, cards_in_hand=3))

    def test_legitimate_duplicate_across_groups(self):
        hand = empty_hand()
        for c in (face(0, 2), face(0, 3), face(0, 4), face(0, 4), face(0, 5), face(0, 6)):
            add(hand, c)
        self.assertTrue(
            is_valid_hand(hand, JOKER_8S, required_sequences=2, cards_in_hand=6)
        )

    def test_all_suit_triplets_as_sets(self):
        for suits in itertools.combinations(range(4), 3):
            with self.subTest(suits=suits):
                hand = from_cards([face(suit, 9) for suit in suits])
                result = evaluate_hand(
                    hand, JOKER_8S, required_sequences=0, cards_in_hand=3,
                )
                self.assertTrue(result.is_valid)
                self.assertEqual(result.melds[0].kind, "set")

    def test_four_card_set(self):
        hand = empty_hand()
        for suit in range(4):
            add(hand, face(suit, 9))
        self.assertTrue(is_valid_hand(hand, JOKER_8S, required_sequences=0, cards_in_hand=4))
        ev = evaluate_hand(hand, JOKER_8S, required_sequences=0, cards_in_hand=4)
        self.assertEqual(len(ev.melds), 1)
        self.assertEqual(ev.melds[0].kind, "set")
        self.assertEqual(len(ev.melds[0].cards), 4)

    def test_ace_low(self):
        hand = from_cards([face(0, 0), face(0, 1), face(0, 2)])
        self.assertTrue(is_valid_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=3))

    def test_ace_high_qka(self):
        hand = from_cards([face(1, 11), face(1, 12), face(1, 0)])
        self.assertTrue(is_valid_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=3))

    def test_ace_no_wrap_ka2(self):
        hand = from_cards([face(0, 12), face(0, 0), face(0, 1)])
        self.assertFalse(is_valid_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=3))

    def test_suit_boundary_not_adjacent(self):
        hand = from_cards([face(0, 12), face(1, 0), face(1, 1)])
        self.assertFalse(is_valid_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=3))

    def test_longer_sequence_splitting(self):
        hand = from_cards([face(0, r) for r in range(6)])
        self.assertTrue(is_valid_hand(hand, JOKER_8S, required_sequences=2, cards_in_hand=6))
        ev = evaluate_hand(hand, JOKER_8S, required_sequences=2, cards_in_hand=6)
        self.assertEqual(sum(1 for m in ev.melds if m.is_pure), 2)

    def test_leftovers_not_ignored(self):
        hand = from_cards([face(0, 0), face(0, 1), face(0, 2), face(1, 12)])
        self.assertFalse(is_valid_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=4))

    def test_wrong_cardinality_is_invalid_not_error(self):
        hand = classic_valid_21()
        self.assertFalse(is_valid_hand(hand, JOKER_8S, cards_in_hand=20))
        self.assertEqual(hand_reward(hand, JOKER_8S, cards_in_hand=20), 0.0)

    def test_impossible_quota_is_losing(self):
        hand = classic_valid_21()
        self.assertFalse(is_valid_hand(hand, JOKER_8S, required_sequences=8))
        self.assertEqual(hand_reward(hand, JOKER_8S, required_sequences=8), 0.0)

    def test_defaults_match_21_and_5(self):
        hand = classic_valid_21()
        self.assertEqual(
            is_valid_hand(hand, JOKER_8S),
            is_valid_hand(hand, JOKER_8S, required_sequences=5, cards_in_hand=21),
        )

    def test_all_joker_pure_sequence(self):
        hand = empty_hand()
        add(hand, JOKER_8S, 3)
        self.assertTrue(is_valid_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=3))
        ev = evaluate_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=3)
        self.assertTrue(ev.melds[0].is_pure)
        self._assert_witness(hand, ev)

    def test_twenty_one_exact_joker_copies(self):
        hand = empty_hand()
        add(hand, JOKER_8S, 21)
        ev = evaluate_hand(hand, JOKER_8S)
        self.assertTrue(ev.is_valid)
        self.assertEqual(hand_reward(hand, JOKER_8S), 1.0)
        self.assertGreaterEqual(sum(1 for m in ev.melds if m.is_pure), 5)
        self.assertEqual(sum(len(m.cards) for m in ev.melds), 21)
        self._assert_witness(hand, ev)

    def test_twenty_one_off_suit_wildcard_copies_invalid(self):
        hand = empty_hand()
        add(hand, JOKER_8H, 21)
        self.assertFalse(is_valid_hand(hand, JOKER_8S))
        self.assertEqual(hand_reward(hand, JOKER_8S), 0.0)
        self.assertEqual(evaluate_hand(hand, JOKER_8S).melds, ())

    def test_three_exact_plus_eighteen_off_suit_invalid_for_default_quota(self):
        hand = empty_hand()
        add(hand, JOKER_8S, 3)
        add(hand, JOKER_8H, 18)
        self.assertFalse(is_valid_hand(hand, JOKER_8S))
        self.assertEqual(hand_reward(hand, JOKER_8S), 0.0)

    def test_twenty_one_non_joker_copies_invalid(self):
        hand = empty_hand()
        add(hand, face(0, 0), 21)
        self.assertFalse(is_valid_hand(hand, JOKER_8S))
        self.assertEqual(hand_reward(hand, JOKER_8S), 0.0)

    def test_residual_natural_sequence_stays_pure_after_quota(self):
        cards = [
            face(0, 0), face(0, 1), face(0, 2),
            face(1, 3), face(1, 4), face(1, 5),
        ]
        hand = from_cards(cards)
        ev = evaluate_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=6)
        self.assertTrue(ev.is_valid)
        self.assertEqual(len(ev.melds), 2)
        self.assertTrue(all(m.is_pure for m in ev.melds))
        self.assertTrue(all(
            a == r or a == JOKER_8S
            for m in ev.melds
            for a, r in zip(m.cards, m.represented_cards)
        ))
        self._assert_witness(hand, ev)

    def test_rank_jokers_in_impure_sequence_after_quota(self):
        cards = [
            face(0, 0), face(0, 1), face(0, 2),
            face(1, 3), face(1, 4), JOKER_8D,
        ]
        hand = from_cards(cards)
        self.assertTrue(is_valid_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=6))

    def test_overlap_forces_alternate_allocation(self):
        hand = from_cards([
            face(0, 2), face(0, 3), face(0, 4), face(0, 5),
            face(1, 2), face(2, 2),
        ])
        result = evaluate_hand(
            hand, JOKER_8S, required_sequences=1, cards_in_hand=6,
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(
            {frozenset(meld.cards) for meld in result.melds},
            {frozenset((3, 4, 5)), frozenset((2, 15, 28))},
        )

    def test_joker_not_required_in_hand(self):
        hand = from_cards([face(0, 0), face(0, 1), face(0, 2)])
        self.assertTrue(is_valid_hand(hand, face(3, 12), required_sequences=1, cards_in_hand=3))

    def test_no_input_mutation(self):
        hand = classic_valid_21()
        snapshot = list(hand)
        evaluate_hand(hand, JOKER_8S)
        self.assertEqual(hand, snapshot)

    def test_reject_bool_counts(self):
        hand = empty_hand()
        hand[0] = True
        with self.assertRaises(TypeError):
            evaluate_hand(hand, JOKER_8S, required_sequences=0, cards_in_hand=1)

    def test_reject_float_counts(self):
        hand = empty_hand()
        hand[0] = 1.0
        with self.assertRaises(TypeError):
            evaluate_hand(hand, JOKER_8S, required_sequences=0, cards_in_hand=1)

    def test_reject_bad_length(self):
        with self.assertRaises(ValueError):
            evaluate_hand([0] * 51, JOKER_8S)

    def test_reject_negative_count(self):
        hand = empty_hand()
        hand[0] = -1
        with self.assertRaises(ValueError):
            evaluate_hand(hand, JOKER_8S, required_sequences=0, cards_in_hand=1)

    def test_reject_bad_joker(self):
        hand = empty_hand()
        hand[0] = 1
        with self.assertRaises(ValueError):
            evaluate_hand(hand, 52, required_sequences=0, cards_in_hand=1)
        with self.assertRaises(TypeError):
            evaluate_hand(hand, True, required_sequences=0, cards_in_hand=1)

    def test_reject_bad_config(self):
        hand = empty_hand()
        hand[0] = 1
        with self.assertRaises(ValueError):
            evaluate_hand(hand, JOKER_8S, required_sequences=-1, cards_in_hand=1)
        with self.assertRaises(ValueError):
            evaluate_hand(hand, JOKER_8S, required_sequences=0, cards_in_hand=0)
        with self.assertRaises(TypeError):
            evaluate_hand(hand, JOKER_8S, required_sequences=1.5, cards_in_hand=1)

    def test_invalid_has_no_partial_witness(self):
        hand = from_cards([face(0, 0), face(0, 1), face(0, 2), face(1, 5)])
        ev = evaluate_hand(hand, JOKER_8S, required_sequences=1, cards_in_hand=4)
        self.assertIsInstance(ev, HandEvaluation)
        self.assertFalse(ev.is_valid)
        self.assertEqual(ev.melds, ())

    def test_witness_assignment_legality_pure_and_impure(self):
        hand = classic_valid_21()
        ev = evaluate_hand(hand, JOKER_8S)
        self._assert_witness(hand, ev)
        for meld in ev.melds:
            self.assertIsInstance(meld, Meld)
            self.assertIn(meld.kind, ("sequence", "set"))
            if meld.kind == "set":
                self.assertFalse(meld.is_pure)

    def _assert_witness(self, hand, ev):
        self.assertTrue(ev.is_valid)
        used = empty_hand()
        for meld in ev.melds:
            self.assertEqual(len(meld.cards), len(meld.represented_cards))
            self.assertGreaterEqual(len(meld.cards), 3)
            for actual, represented in zip(meld.cards, meld.represented_cards):
                used[actual] += 1
                if meld.is_pure:
                    self.assertTrue(
                        actual == represented or actual == JOKER_8S,
                        msg=(actual, represented, meld),
                    )
                else:
                    joker_rank = JOKER_8S % 13
                    self.assertTrue(
                        actual == represented or actual % 13 == joker_rank,
                        msg=(actual, represented, meld),
                    )
            if meld.kind == "sequence":
                suits = {r // 13 for r in meld.represented_cards}
                self.assertEqual(len(suits), 1)
                ranks = [r % 13 for r in meld.represented_cards]
                self.assertEqual(len(ranks), len(set(ranks)))
            if meld.kind == "set":
                ranks = {r % 13 for r in meld.represented_cards}
                self.assertEqual(len(ranks), 1)
                suits = [r // 13 for r in meld.represented_cards]
                self.assertEqual(len(suits), len(set(suits)))
                self.assertIn(len(suits), (3, 4))
        self.assertEqual(used, list(hand))


if __name__ == "__main__":
    unittest.main()
