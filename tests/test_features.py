from __future__ import annotations

from dataclasses import replace
import math
import unittest
from unittest import mock

from src.game.environment import GameConfig, PappluEnv, Phase, discard_action
from src.game.information import EventKind, InformationEvent
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.model.encoding import (
    CARD_CHANNELS, CONTEXT_CHANNELS, FEATURE_SIZE, Features,
    disjoint_pack, encode, encoding_layout, encoding_metadata, feature_values,
    legal_actions, unpack,
)


def field_values(features, name):
    return unpack(features)[name]


class TestFeatures(unittest.TestCase):
    def test_exact_copy_counts_do_not_turn_pairs_and_triplets_into_singletons(self):
        obs = PappluEnv().reset(17)
        hand = tuple({0: 1, 5: 2, 18: 3}.get(face, 0) for face in range(52))
        cards = unpack(encode(replace(obs, hand=hand, joker=7)))["card"]
        self.assertEqual(cards[1:4], (1.0, 0.0, 0.0))
        self.assertAlmostEqual(cards[5 * 25], 2 / 3, places=7)
        self.assertEqual(cards[5 * 25 + 1:5 * 25 + 4], (0.0, 1.0, 0.0))
        self.assertEqual(cards[18 * 25:18 * 25 + 4], (1.0, 0.0, 0.0, 1.0))

    def test_unseen_subtracts_the_indicator_pile_and_every_publicly_held_copy(self):
        cfg = GameConfig(cards_in_hand=3, required_sequences=1)
        env = MultiplayerEnv(MatchConfig(cfg, 3))
        env.reset(17)
        obs = env.observe()
        hand = tuple(int(face == 5) for face in range(52))
        history = (*obs.history,
                   InformationEvent(EventKind.TAKE_DISCARD, 0, 4, 60, 140, 1, face=5),
                   InformationEvent(EventKind.TAKE_DISCARD, 1, 4, 60, 140, 2, face=5),
                   InformationEvent(EventKind.TAKE_DISCARD, 2, 4, 60, 140, 0, face=8),
                   InformationEvent(EventKind.DISCARD, 1, 3, 59, 140, 2, face=5))
        features = unpack(encode(replace(obs, hand=hand, joker=7, discard_pile=(5,), history=history)))
        card = lambda face, name: features["card"][face * 25 + CARD_CHANNELS.index(name)]
        self.assertEqual(card(5, "unseen"), 0.0)
        self.assertEqual(card(5, "next_held"), 0.0)
        self.assertAlmostEqual(card(8, "unseen"), 2 / 3, places=7)
        self.assertAlmostEqual(card(8, "other_held"), 1 / 3, places=7)
        self.assertAlmostEqual(card(7, "unseen"), 2 / 3, places=7)
        self.assertEqual(card(9, "unseen"), 1.0)

    def test_fixed_record_is_finite_for_solo_hexa_and_terminal(self):
        observations = [PappluEnv().reset(1), PappluEnv().reset_warm_start(3)]
        env = MultiplayerEnv(MatchConfig(GameConfig(max_turns=1), 6))
        env.reset(19)
        while not env.view().done:
            observations.append(env.observe())
            env.step(1)
            obs = env.observe()
            env.step(2 + next(face for face, copies in enumerate(obs.hand) if copies))
        observations.append(env.observe())
        for obs in observations:
            features = encode(obs)
            self.assertEqual(len(feature_values(features)), 1433)
            self.assertEqual(len(features.data), 5732)
            self.assertTrue(all(math.isfinite(value) for value in feature_values(features)))
        self.assertEqual(legal_actions(encode(observations[-1])), ())
        solo = encode(observations[0])
        self.assertEqual(field_values(solo, "seat")[9:], (0.0,) * 45)
        self.assertEqual(field_values(solo, "context")[CONTEXT_CHANNELS.index("to_act")], 1.0)
        self.assertEqual(legal_actions(solo), (0, 1))
        warm = encode(observations[1])
        self.assertEqual(field_values(warm, "top"), (52.0,))
        self.assertEqual(field_values(warm, "context")[CONTEXT_CHANNELS.index("discard")], 1.0)
        with self.assertRaises(ValueError):
            Features(b"short")

    def test_afterstate_has_no_legal_actions_when_another_seat_acts(self):
        env = MultiplayerEnv(MatchConfig(GameConfig(cards_in_hand=3, required_sequences=1), 4))
        env.reset(2)
        obs = env.observe()
        after = replace(obs, history=(*obs.history, InformationEvent(
            EventKind.DISCARD, 0, 3, 59, obs.stock_remaining, 1, face=0,
        )))
        self.assertEqual(after.phase, Phase.DRAW)
        self.assertEqual(after.players[0].next_player, 1)
        features = encode(after)
        self.assertEqual(legal_actions(features), ())
        self.assertEqual(field_values(features, "context")[CONTEXT_CHANNELS.index("to_act")], 0.0)
        self.assertEqual(field_values(features, "context")[-4:], (0.0, 0.0, 0.0, 0.0))

    def test_raw_points_and_wild_counts_do_not_require_exact_meld_search(self):
        obs = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17)
        hand = tuple(int(face in (0, 1, 7, 12, 20)) for face in range(52))
        with mock.patch(
            "src.evaluate.pure_sequence_upper_bound", side_effect=AssertionError("sequence search"),
        ):
            features = encode(replace(obs, hand=hand, joker=7))
        context = field_values(features, "context")
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("raw_points")], 0.22, places=7)
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("wilds_held")], 2 / 12, places=7)
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("jokers_held")], 1 / 3, places=7)
        self.assertEqual(FEATURE_SIZE * 4, len(features.data))
        cards = field_values(features, "card")
        self.assertEqual(cards[CARD_CHANNELS.index("own_eq1")], 1.0)
        self.assertEqual(cards[7 * 25 + CARD_CHANNELS.index("points")], 0.0)
        self.assertEqual(cards[20 * 25 + CARD_CHANNELS.index("wild")], 1.0)

    def test_layout_and_metadata_describe_the_encoded_segments(self):
        features = encode(PappluEnv().reset(17))
        segments = unpack(features)
        layout = encoding_layout()
        self.assertEqual(
            [(field.name, field.offset, len(segments[field.name])) for field in layout],
            [("card", 0, 1300), ("seat", 1300, 54), ("context", 1354, 24),
             ("legal", 1378, 54), ("top", 1432, 1)],
        )
        self.assertEqual(tuple(value for field in layout for value in segments[field.name]), feature_values(features))
        metadata = encoding_metadata()
        self.assertEqual(metadata["feature_bytes"], len(features.data))
        self.assertEqual(metadata["feature_size"], len(feature_values(features)))
        self.assertEqual([row["name"] for row in metadata["layout"]], list(segments))
        self.assertEqual(set(metadata["channels"]), {"card", "seat", "context"})

    def test_public_takes_passes_and_discards_are_encoded_without_beliefs(self):
        cfg = GameConfig(num_decks=1, cards_in_hand=3, required_sequences=1, max_turns=60)
        env = MultiplayerEnv(MatchConfig(cfg, 2))
        env.reset(19)
        initial = unpack(encode(env.observe()))
        self.assertEqual(initial["seat"][6:9], (0.0, 0.0, 0.0))
        taken, discarded = [], []
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for _ in range(2):
                taken.append(env.observe().discard_top)
                env.step(1)
                action = legal_actions(encode(env.observe()))[0]
                discarded.append(action - 2)
                env.step(action)
            for seat in range(2):
                env.step(0)
                obs = env.observe()
                action = next(a for a in legal_actions(encode(obs)) if a != discard_action(taken[seat]))
                env.step(action)
        values = unpack(encode(env.observe()))
        card = lambda face, channel: values["card"][face * 25 + CARD_CHANNELS.index(channel)]
        self.assertEqual(card(taken[1], "next_held"), 1.0)
        self.assertEqual(card(discarded[1], "opp_discarded"), 1.0)
        self.assertEqual(card(taken[1], "other_held"), 0.0)
        for seat in range(2):
            row = values["seat"][seat * 9:(seat + 1) * 9]
            self.assertAlmostEqual(row[6], 1 / 60, places=7)
            self.assertAlmostEqual(row[7], 1 / 60, places=7)
            self.assertAlmostEqual(row[8], 1 / 3, places=7)

    def test_taken_face_cannot_be_repeated_unless_it_is_the_only_face(self):
        for players in (1, 2, 4):
            game = GameConfig(cards_in_hand=3, required_sequences=1)
            env = PappluEnv(game) if players == 1 else MultiplayerEnv(MatchConfig(game, players))
            env.reset(17)
            env.step(1)
            obs = env.observe()
            taken = obs.history[-1].face
            legal = legal_actions(encode(obs))
            self.assertNotIn(discard_action(taken), legal)
            self.assertTrue(legal)
            one_face = replace(obs, hand=tuple(4 if face == taken else 0 for face in range(52)))
            self.assertEqual(legal_actions(encode(one_face)), (discard_action(taken),))
            from_stock = replace(obs, history=(*obs.history[:-1], replace(
                obs.history[-1], kind=EventKind.DRAW_STOCK,
            )))
            self.assertIn(discard_action(taken), legal_actions(encode(from_stock)))

    def test_draw_mask_excludes_previous_own_discard_only_while_stock_remains(self):
        for players, expected in ((1, (0,)), (2, (0, 1)), (4, (0, 1))):
            with self.subTest(players=players):
                game = GameConfig(cards_in_hand=3, required_sequences=1)
                env = PappluEnv(game) if players == 1 else MultiplayerEnv(MatchConfig(game, players))
                env.reset(17)
                env.step(0)
                env.step(discard_action(next(face for face, copies in enumerate(env.observe().hand) if copies)))
                obs = env.observe()
                self.assertIs(obs.phase, Phase.DRAW)
                self.assertEqual(legal_actions(encode(obs)), expected)
                self.assertEqual(legal_actions(encode(replace(obs, stock_remaining=0))), (1,))


class TestDisjointFeatures(unittest.TestCase):
    @staticmethod
    def counts(cards):
        return tuple(cards.count(face) for face in range(52))

    def observation(self, cards, joker, required, phase=Phase.DRAW, top=None):
        cfg = GameConfig(cards_in_hand=len(cards) - int(phase is Phase.DISCARD), required_sequences=required)
        env = PappluEnv(cfg)
        obs = env.reset(17)
        if phase is Phase.DISCARD:
            obs = env.step(0)
        return replace(obs, hand=self.counts(cards), joker=joker, discard_pile=() if top is None else (top,))

    def test_packing_keeps_sets_gated_until_the_pure_sequence_quota(self):
        hand = self.counts((1, 2, 3, 14, 15, 16, 6, 19, 32))
        packed = disjoint_pack(hand, 51, 3)
        self.assertEqual(packed[:4], (2, 21, 0, 0))
        self.assertEqual(packed[4], [0] * 52)
        self.assertEqual(packed[5], list(self.counts((6, 19, 32))))
        self.assertEqual(hand, self.counts((1, 2, 3, 14, 15, 16, 6, 19, 32)))
        complete = self.counts((1, 2, 3, 14, 15, 16, 27, 28, 29, 6, 19, 32))
        self.assertEqual(disjoint_pack(complete, 51, 3)[:4], (3, 0, 0, 0))

    def test_spare_fractions_distinguish_pure_sequences_from_sets_and_duplicate_copies(self):
        obs = self.observation((1, 1, 2, 3, 6, 19, 32), 51, 2)
        values = unpack(encode(obs))
        card = lambda face, name: values["card"][face * 25 + CARD_CHANNELS.index(name)]
        self.assertEqual((card(1, "spare_pure"), card(1, "spare_meld")), (0.5, 0.5))
        self.assertEqual((card(6, "spare_pure"), card(6, "spare_meld")), (1.0, 0.0))
        self.assertEqual((card(2, "spare_pure"), card(2, "spare_meld")), (0.0, 0.0))
        self.assertEqual((card(0, "spare_pure"), card(0, "spare_meld")), (0.0, 0.0))
        self.assertAlmostEqual(values["context"][CONTEXT_CHANNELS.index("disjoint_gated_pts")], 0.23, places=7)

    def test_exact_joker_qualifies_pure_pairs_but_an_off_suit_wild_does_not(self):
        for joker_card, pure in ((7, 1.0), (20, 0.0)):
            with self.subTest(joker_card=joker_card):
                obs = self.observation((16, 17, joker_card), 7, 1)
                values = unpack(encode(obs))
                for face in (16, 17, joker_card):
                    row = values["card"][face * 25:(face + 1) * 25]
                    self.assertEqual(tuple(row[CARD_CHANNELS.index(name)] for name in ("natural", "pure", "any_meld")),
                                     (0.0, pure, 1.0))
                self.assertEqual(values["context"][CONTEXT_CHANNELS.index("disjoint_pseq_frac")], pure)
                self.assertAlmostEqual(values["context"][CONTEXT_CHANNELS.index("disjoint_gated_pts")],
                                       0.0 if pure else 0.09, places=7)

    def test_discard_delta_keeps_gated_and_meld_points_and_zeros_the_just_taken_face(self):
        obs = self.observation((1, 2, 3, 14, 15, 16, 6, 19, 32, 49), 51, 3, Phase.DISCARD)
        offset = 6 * 25 + CARD_CHANNELS.index("disc_pseq_delta")
        deltas = unpack(encode(obs))["card"][offset:offset + 4]
        self.assertEqual((deltas[0], deltas[3]), (0.0, 0.0))
        self.assertAlmostEqual(deltas[1], 0.7, places=7)
        self.assertAlmostEqual(deltas[2], -1.4, places=7)
        taken = replace(obs, history=(*obs.history[:-1], replace(
            obs.history[-1], kind=EventKind.TAKE_DISCARD, face=6,
        )))
        self.assertEqual(unpack(encode(taken))["card"][offset:offset + 4], (0.0, 0.0, 0.0, 0.0))
        terminal = replace(obs, phase=Phase.TERMINAL)
        values = unpack(encode(terminal))
        self.assertTrue(all(
            values["card"][face * 25 + CARD_CHANNELS.index("disc_pseq_delta"):face * 25 + 25] == (0.0, 0.0, 0.0, 0.0)
            for face in range(52)
        ))

    def test_king_set_preservation_is_visible_below_five_pure_sequences(self):
        cards = (1, 14, 27, 40) * 3 + (12, 25, 38, 4, 4, 17, 17, 8, 8, 47)
        obs = self.observation(cards, 6, 5, Phase.DISCARD)
        self.assertEqual(disjoint_pack(obs.hand, obs.joker, 5)[:4], (4, 77, 47, 0))
        values = unpack(encode(obs))["card"]
        for face, expected in ((12, (0.0, 1.0, -2.0, 0.0)), (47, (0.0, 0.9, 0.9, 0.0))):
            with self.subTest(face=face):
                offset = face * 25 + CARD_CHANNELS.index("disc_pseq_delta")
                for actual, wanted in zip(values[offset:offset + 4], expected):
                    self.assertAlmostEqual(actual, wanted, places=7)

    def test_top_swap_can_complete_a_set_without_meeting_the_pure_sequence_quota(self):
        cards = (1, 14, 27, 40) * 3 + (12, 25, 4, 4, 17, 17, 8, 8, 47)
        obs = self.observation(cards, 6, 5, top=38)
        self.assertEqual(disjoint_pack(obs.hand, obs.joker, 5)[:4], (4, 67, 67, 0))
        for actual, wanted in zip(unpack(encode(obs))["context"][-4:], (0.0, -0.05, 1.45, 0.0)):
            self.assertAlmostEqual(actual, wanted, places=7)

    def test_top_swap_balances_gated_and_meld_points_before_sequence_pairs(self):
        cases = (
            ((3, 7, 10, 11, 16, 18, 23, 26, 28, 29, 31, 32, 33, 35, 36, 37, 40, 41, 46, 47, 50),
             32, 46, (0.0, 0.2, 1.4, -2 / 3)),
            ((1, 5, 6, 10, 13, 15, 20, 21, 24, 27, 28, 30, 33, 35, 37, 39, 41, 44, 48, 49, 51),
             30, 9, (0.0, 0.0, 1.5, 0.0)),
        )
        for cards, joker, top, expected in cases:
            with self.subTest(joker=joker):
                obs = self.observation(cards, joker, 5, top=top)
                for actual, wanted in zip(unpack(encode(obs))["context"][-4:], expected):
                    self.assertAlmostEqual(actual, wanted, places=7)
                no_top = replace(obs, discard_pile=())
                self.assertEqual(unpack(encode(no_top))["context"][-4:], (0.0, 0.0, 0.0, 0.0))

    def test_top_swap_zeros_a_blocked_own_discard_but_scores_a_forced_take(self):
        obs = self.observation((0, 1, 9), 7, 1, top=2)
        obs = replace(obs, history=(*obs.history, InformationEvent(
            EventKind.DISCARD, 0, 3, 59, obs.stock_remaining, 0, face=2,
        )))
        self.assertEqual(unpack(encode(obs))["context"][-4:], (0.0, 0.0, 0.0, 0.0))
        forced = encode(replace(obs, stock_remaining=0))
        self.assertEqual(legal_actions(forced), (1,))
        self.assertEqual(unpack(forced)["context"][CONTEXT_CHANNELS.index("top_gain_pseq")], 1.0)

    def test_zero_sequence_quota_exempts_sets_without_dividing_by_zero(self):
        obs = self.observation((6, 19, 32), 51, 0)
        values = unpack(encode(obs))
        self.assertEqual(values["context"][CONTEXT_CHANNELS.index("disjoint_pseq_frac")], 0.0)
        self.assertEqual(values["context"][CONTEXT_CHANNELS.index("disjoint_gated_pts")], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in feature_values(encode(obs))))


if __name__ == "__main__":
    unittest.main()
