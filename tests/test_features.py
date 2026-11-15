from __future__ import annotations

from dataclasses import replace
import math
import unittest
from unittest import mock

from src.game.belief import BeliefState
from src.game.environment import GameConfig, PappluEnv, Phase, discard_action
from src.game.information import EventKind
from src.game.lookahead import after_discard, after_draw
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.model.encoding import (
    CARD_CHANNELS, CONTEXT_CHANNELS, FEATURE_SIZE, SEAT_CHANNELS, Features,
    _meld_membership, encode, encoding_layout, encoding_metadata, feature_values,
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
            self.assertEqual(len(feature_values(features)), 802)
            self.assertEqual(len(features.data), 3208)
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
        drawn = after_draw(env.observe(), env.observe().discard_top, EventKind.TAKE_DISCARD)
        after = after_discard(drawn, next(face for face, copies in enumerate(drawn.hand) if copies))
        self.assertEqual(after.phase, Phase.DRAW)
        self.assertEqual(after.players[0].next_player, 1)
        features = encode(after)
        self.assertEqual(legal_actions(features), ())
        self.assertEqual(field_values(features, "context")[CONTEXT_CHANNELS.index("to_act")], 0.0)

    def test_raw_points_and_wild_counts_do_not_require_meld_search(self):
        obs = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17)
        hand = tuple(int(face in (0, 1, 7, 12, 20)) for face in range(52))
        with mock.patch("src.model.encoding._meld_membership", side_effect=AssertionError("meld search")), mock.patch(
            "src.game.belief.BeliefState.from_observation", side_effect=AssertionError("belief construction"),
        ), mock.patch("src.evaluate.pure_sequence_upper_bound", side_effect=AssertionError("sequence search")):
            features = encode(replace(obs, hand=hand, joker=7))
        context = field_values(features, "context")
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("deadwood")], 0.22, places=7)
        self.assertAlmostEqual(context[CONTEXT_CHANNELS.index("wilds")], 2 / 12, places=7)
        self.assertEqual(FEATURE_SIZE * 4, len(features.data))
        cards = field_values(features, "card")
        self.assertEqual(cards[CARD_CHANNELS.index("own_ge1")], 1.0)
        self.assertEqual(cards[7 * 13 + CARD_CHANNELS.index("points")], 0.0)
        self.assertEqual(cards[20 * 13 + CARD_CHANNELS.index("wild")], 1.0)

    def test_layout_and_metadata_describe_the_encoded_segments(self):
        features = encode(PappluEnv().reset(17))
        segments = unpack(features)
        layout = encoding_layout()
        self.assertEqual(
            [(field.name, field.offset, len(segments[field.name])) for field in layout],
            [("card", 0, 676), ("seat", 676, 54), ("context", 730, 17),
             ("legal", 747, 54), ("top", 801, 1)],
        )
        self.assertEqual(tuple(value for field in layout for value in segments[field.name]), feature_values(features))
        metadata = encoding_metadata()
        self.assertEqual(metadata["feature_bytes"], len(features.data))
        self.assertEqual(metadata["feature_size"], len(feature_values(features)))
        self.assertEqual([row["name"] for row in metadata["layout"]], list(segments))
        self.assertEqual(set(metadata["channels"]), {"card", "seat", "context"})

    def assert_ledger_channels(self, obs):
        ledger = BeliefState.from_observation(obs).ledger
        values = unpack(encode(obs))
        count, cfg = len(obs.players), obs.config
        next_opp = obs.players[0].next_player if count > 1 and obs.players[0].next_player > 0 else -1
        for face in range(52):
            held = ledger.seats[next_opp].held[face] if next_opp > 0 else 0
            passed = ledger.seats[next_opp].passed[face] if next_opp > 0 else 0
            expected = {
                "opp_discarded": sum(row.discarded[face] for row in ledger.seats[1:]),
                "next_held": held,
                "other_held": sum(row.held[face] for row in ledger.seats[1:]) - held,
                "next_passed": passed,
            }
            for channel, value in expected.items():
                self.assertAlmostEqual(
                    values["card"][face * len(CARD_CHANNELS) + CARD_CHANNELS.index(channel)],
                    value / cfg.num_decks, places=6,
                )
        for seat, row in enumerate(ledger.seats):
            for channel, value, denominator in (
                ("takes", row.takes, cfg.max_turns),
                ("passes", row.passes, cfg.max_turns),
                ("held_total", sum(row.held), cfg.cards_in_hand),
            ):
                self.assertAlmostEqual(
                    values["seat"][seat * len(SEAT_CHANNELS) + SEAT_CHANNELS.index(channel)],
                    value / denominator, places=6,
                )
        self.assertAlmostEqual(values["context"][7], min(ledger.refills, 3) / 3.0, places=6)
        return ledger

    def test_sparse_history_channels_match_the_full_ledger_for_multiseat_deals(self):
        cfg = GameConfig(num_decks=1, cards_in_hand=3, required_sequences=1, max_turns=60)
        for players in (2, 4, 6):
            for seed in (19, 47):
                with self.subTest(players=players, seed=seed), mock.patch(
                    "src.game.multiplayer.hand_reward", return_value=0.0,
                ):
                    env = MultiplayerEnv(MatchConfig(cfg, players))
                    env.reset(seed)
                    turn = 0
                    while not env.view().done:
                        obs = env.observe()
                        self.assert_ledger_channels(obs)
                        take = turn % 3 == 0 or not obs.stock_remaining
                        env.step(int(take))
                        obs = env.observe()
                        self.assert_ledger_channels(obs)
                        held = [face for face, copies in enumerate(obs.hand) if copies]
                        taken = obs.history[-1].face
                        choices = [face for face in held if face != taken] if take and turn % 2 else held
                        env.step(discard_action((choices or held)[-1]))
                        turn += 1
                    ledger = self.assert_ledger_channels(env.observe())
                    self.assertGreater(ledger.refills, 0)
                    self.assertGreater(sum(row.takes for row in ledger.seats), 0)
                    self.assertGreater(sum(row.passes for row in ledger.seats), 0)


if __name__ == "__main__":
    unittest.main()
