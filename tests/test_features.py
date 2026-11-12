from __future__ import annotations

from dataclasses import replace
import math
import unittest

from src.game.environment import GameConfig, PappluEnv, Phase
from src.game.information import EventKind
from src.game.lookahead import after_discard, after_draw
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.model.encoding import (
    CARD_CHANNELS, CONTEXT_CHANNELS, FEATURE_SIZE, LAYOUT, Features, encode,
    feature_values, legal_actions,
)


def field_values(features, name):
    field = next(field for field in LAYOUT if field.name == name)
    return feature_values(features)[field.offset:field.offset + field.size]


class TestFeatures(unittest.TestCase):
    def card_channels(self, cards, joker=7):
        obs = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17)
        hand = tuple(cards.count(face) for face in range(52))
        values = field_values(encode(replace(obs, hand=hand, joker=joker)), "card")
        return {
            channel: tuple(values[face * 17 + index] for face in range(52))
            for index, channel in enumerate(CARD_CHANNELS)
        }

    def test_ace_high_and_low_runs_do_not_allow_king_ace_two(self):
        for cards in ((0, 1, 2), (11, 12, 0)):
            channels = self.card_channels(cards)
            for name in ("natural_run_member", "pure_meld_member", "any_meld_member"):
                self.assertEqual(tuple(face for face, value in enumerate(channels[name]) if value), tuple(sorted(cards)))
        channels = self.card_channels((12, 0, 1))
        for name in ("natural_run_member", "pure_meld_member", "any_meld_member"):
            self.assertEqual(sum(channels[name]), 0.0)

    def test_exact_joker_substitution_is_pure_but_same_rank_wild_is_not(self):
        pure = self.card_channels((16, 17, 7))
        impure = self.card_channels((16, 17, 33))
        self.assertEqual(sum(pure["natural_run_member"]), 0.0)
        self.assertEqual(sum(pure["pure_meld_member"]), 3.0)
        self.assertEqual(sum(pure["any_meld_member"]), 3.0)
        self.assertEqual(sum(impure["pure_meld_member"]), 0.0)
        self.assertEqual(sum(impure["any_meld_member"]), 3.0)
        natural_wild = self.card_channels((32, 33, 34))
        self.assertEqual(sum(natural_wild["natural_run_member"]), 3.0)
        self.assertEqual(sum(natural_wild["pure_meld_member"]), 3.0)

    def test_triplets_and_sets_use_physical_copies(self):
        triplet = self.card_channels((5, 5, 5))
        self.assertEqual(triplet["pure_meld_member"][5], 1.0)
        self.assertEqual(triplet["natural_run_member"][5], 0.0)
        distinct = self.card_channels((5, 18, 31))
        self.assertEqual(sum(distinct["pure_meld_member"]), 0.0)
        self.assertEqual(sum(distinct["any_meld_member"]), 3.0)
        duplicate = self.card_channels((5, 5, 7))
        self.assertEqual(sum(duplicate["pure_meld_member"]), 0.0)
        self.assertEqual(sum(duplicate["any_meld_member"]), 0.0)

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
            self.assertEqual(len(feature_values(features)), 3208)
            self.assertEqual(len(features.data), 12832)
            self.assertTrue(all(math.isfinite(value) for value in feature_values(features)))
        self.assertEqual(legal_actions(encode(observations[-1])), ())
        solo = encode(observations[0])
        self.assertEqual(field_values(solo, "seat")[11:], (0.0,) * 55)
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
        drawn = after_draw(env.observe(), env.observe().discard_top, EventKind.TAKE_DISCARD)
        after = after_discard(drawn, next(face for face, copies in enumerate(drawn.hand) if copies))
        self.assertEqual(after.phase, Phase.DRAW)
        self.assertEqual(after.players[0].next_player, 1)
        features = encode(after)
        self.assertEqual(legal_actions(features), ())
        self.assertEqual(field_values(features, "context")[CONTEXT_CHANNELS.index("to_act")], 0.0)

    def test_deficit_and_unavoidable_deadwood_are_exact_lower_bounds(self):
        obs = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17)
        hand = tuple(int(face in (0, 1, 12)) for face in range(52))
        features = encode(replace(obs, hand=hand, joker=7))
        context = field_values(features, "context")
        self.assertEqual(context[CONTEXT_CHANNELS.index("pure_quota_deficit")], 1.0)
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("unavoidable_deadwood")], 22 / 30, places=7)
        self.assertEqual(FEATURE_SIZE * 4, len(features.data))


if __name__ == "__main__":
    unittest.main()
