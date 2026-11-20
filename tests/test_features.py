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
    _meld_membership, disjoint_pack, encode, encoding_layout, encoding_metadata, feature_values,
    legal_actions, unpack,
)


def field_values(features, name):
    return unpack(features)[name]


class TestFeatures(unittest.TestCase):
    def membership(self, cards, joker=7):
        hand = tuple(cards.count(face) for face in range(52))
        return dict(zip(("natural", "pure", "any_meld"), _meld_membership(hand, joker)))

    def test_ace_high_and_low_runs_do_not_allow_king_ace_two(self):
        for cards in ((0, 1, 2), (11, 12, 0)):
            channels = self.membership(cards)
            for name in ("natural", "pure", "any_meld"):
                self.assertEqual(tuple(face for face, value in enumerate(channels[name]) if value), tuple(sorted(cards)))
        channels = self.membership((12, 0, 1))
        for name in ("natural", "pure", "any_meld"):
            self.assertEqual(sum(channels[name]), 0.0)

    def test_exact_joker_substitution_is_pure_but_same_rank_wild_is_not(self):
        pure = self.membership((16, 17, 7))
        impure = self.membership((16, 17, 33))
        self.assertEqual(sum(pure["natural"]), 0.0)
        self.assertEqual(sum(pure["pure"]), 3.0)
        self.assertEqual(sum(pure["any_meld"]), 3.0)
        self.assertEqual(sum(impure["pure"]), 0.0)
        self.assertEqual(sum(impure["any_meld"]), 3.0)
        natural_wild = self.membership((32, 33, 34))
        self.assertEqual(sum(natural_wild["natural"]), 3.0)
        self.assertEqual(sum(natural_wild["pure"]), 3.0)

    def test_triplets_and_sets_use_physical_copies(self):
        triplet = self.membership((5, 5, 5))
        self.assertEqual(triplet["pure"][5], 1.0)
        self.assertEqual(triplet["natural"][5], 0.0)
        distinct = self.membership((5, 18, 31))
        self.assertEqual(sum(distinct["pure"]), 0.0)
        self.assertEqual(sum(distinct["any_meld"]), 3.0)
        duplicate = self.membership((5, 5, 7))
        self.assertEqual(sum(duplicate["pure"]), 0.0)
        self.assertEqual(sum(duplicate["any_meld"]), 0.0)

    def test_disjoint_pack_spends_a_shared_joker_once_and_gates_sets(self):
        cards = (0, 1, 13, 14, 26, 27, 39, 40, 9, 10, 22, 23, 7)
        hand = tuple(cards.count(face) for face in range(52))
        packed = disjoint_pack(hand, 7, 5)
        self.assertEqual(packed[:3], (1, 68, 20))
        self.assertEqual(packed[4][7], 0)
        self.assertTrue(all(0 <= remaining <= copies for remaining, copies in zip(packed[4], hand)))
        obs = replace(PappluEnv().reset(17), hand=hand, joker=7)
        context = unpack(encode(obs))["context"]
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("disjoint_pseq_frac")], 0.2, places=7)
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("disjoint_gated_pts")], 0.68, places=7)

    def test_disjoint_pack_preserves_two_after_high_ace_and_splits_six_card_runs(self):
        hand = tuple(int(face in (11, 12, 0, 1)) for face in range(52))
        packed = disjoint_pack(hand, 7, 1)
        self.assertEqual(packed[:3], (1, 2, 2))
        self.assertEqual(packed[4][1], 1)
        hand = tuple(int(face < 6) for face in range(52))
        self.assertEqual(disjoint_pack(hand, 7, 2)[:3], (2, 0, 0))
        hand = tuple(int(face in (12, 0, 1)) for face in range(52))
        self.assertEqual(disjoint_pack(hand, 7, 1)[:3], (0, 22, 22))

    def test_disjoint_discard_and_top_gains_describe_the_legal_exchange(self):
        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1))
        obs = env.reset_warm_start(17)
        hand = tuple(int(face in (0, 1, 2, 9)) for face in range(52))
        values = unpack(encode(replace(obs, hand=hand, joker=7)))["card"]
        self.assertEqual(values[9 * 22 + 18:9 * 22 + 22], (1.0, 0.0, 1.0, 0.0))
        self.assertEqual(values[19], -1.0)
        obs = env.reset(17)
        hand = tuple(int(face in (0, 1, 30)) for face in range(52))
        context = unpack(encode(replace(obs, hand=hand, joker=7, discard_pile=(2,))))["context"]
        self.assertEqual(context[20], 1.0)
        self.assertAlmostEqual(context[21], 0.85, places=7)

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
            self.assertEqual(len(feature_values(features)), 1275)
            self.assertEqual(len(features.data), 5100)
            self.assertTrue(all(math.isfinite(value) for value in feature_values(features)))
        self.assertEqual(legal_actions(encode(observations[-1])), ())
        solo = encode(observations[0])
        self.assertEqual(field_values(solo, "seat")[9:], (0.0,) * 45)
        self.assertEqual(field_values(solo, "context")[CONTEXT_CHANNELS.index("to_act")], 1.0)
        self.assertEqual(legal_actions(solo), (0, 1))
        warm = encode(observations[1])
        self.assertEqual(field_values(warm, "top"), (52.0,))
        self.assertEqual(field_values(warm, "context")[CONTEXT_CHANNELS.index("warm_start")], 1.0)
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

    def test_raw_points_and_wild_counts_do_not_require_meld_search(self):
        obs = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17)
        hand = tuple(int(face in (0, 1, 7, 12, 20)) for face in range(52))
        with mock.patch("src.model.encoding._meld_membership", wraps=_meld_membership) as membership, mock.patch(
            "src.evaluate.pure_sequence_upper_bound", side_effect=AssertionError("sequence search"),
        ):
            features = encode(replace(obs, hand=hand, joker=7))
        self.assertEqual(membership.call_count, 1)
        context = field_values(features, "context")
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("deadwood")], 0.22, places=7)
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("wilds")], 2 / 12, places=7)
        self.assertEqual(FEATURE_SIZE * 4, len(features.data))
        cards = field_values(features, "card")
        self.assertEqual(cards[CARD_CHANNELS.index("own_ge1")], 1.0)
        self.assertEqual(cards[7 * 22 + CARD_CHANNELS.index("points")], 0.0)
        self.assertEqual(cards[20 * 22 + CARD_CHANNELS.index("wild")], 1.0)

    def test_layout_and_metadata_describe_the_encoded_segments(self):
        features = encode(PappluEnv().reset(17))
        segments = unpack(features)
        layout = encoding_layout()
        self.assertEqual(
            [(field.name, field.offset, len(segments[field.name])) for field in layout],
            [("card", 0, 1144), ("seat", 1144, 54), ("context", 1198, 22),
             ("legal", 1220, 54), ("top", 1274, 1)],
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
        card = lambda face, channel: values["card"][face * 22 + CARD_CHANNELS.index(channel)]
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


if __name__ == "__main__":
    unittest.main()
