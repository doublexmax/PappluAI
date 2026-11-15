import copy
import unittest
from unittest import mock

from src.game.environment import (
    GameConfig, Observation, PappluEnv, discard_action,
)
from src.game.information import EventKind
from src.game.multiplayer import MatchConfig, MultiplayerEnv


class TestInformationHistory(unittest.TestCase):
    def make_match(self, players=4, turns=12):
        env = MultiplayerEnv(MatchConfig(
            GameConfig(cards_in_hand=3, required_sequences=1, max_turns=turns), players,
        ))
        env.reset(73)
        return env

    def test_stock_face_is_private_without_private_chronology_gaps(self):
        env = self.make_match()
        initial = env.observe()
        drawn_face = env._stock[-1]
        env.step(0)
        own = env.observe()
        env.step(discard_action(drawn_face))
        peer = env.observe()
        self.assertEqual(own.history[0].initial_hand, initial.hand)
        self.assertEqual(own.history[-1].face, drawn_face)
        self.assertEqual(own.history[-1].offered_top, initial.discard_top)
        self.assertEqual(peer.history[1].kind, EventKind.DRAW_STOCK)
        self.assertIsNone(peer.history[1].face)
        self.assertEqual(peer.history[1].actor, 3)
        self.assertEqual(peer.history[1].hand_size, 4)
        self.assertEqual(peer.history[1].stock_remaining, initial.stock_remaining - 1)
        self.assertEqual(peer.history[2].face, drawn_face)
        self.assertEqual(len(peer.history), 3)
        self.assertEqual(len(own.history), 2)
        self.assertEqual(peer.history[0].initial_hand, peer.hand)

    def test_hidden_stock_and_peer_cards_do_not_change_visible_observation(self):
        env = self.make_match()
        other = copy.deepcopy(env)
        other._stock.reverse()
        other._hands[1], other._hands[2] = other._hands[2], other._hands[1]
        self.assertEqual(env.observe(), other.observe())
        env.step(0)
        other.step(0)
        env._current_seat = other._current_seat = 1
        self.assertEqual(env._journal.observe(1)[1], other._journal.observe(1)[1])

    def test_discard_pickup_and_passed_top_remain_public(self):
        env = self.make_match(players=3)
        top = env.observe().discard_top
        env.step(1)
        env.step(discard_action(top))
        peer = env.observe()
        self.assertEqual(peer.history[1].kind, EventKind.TAKE_DISCARD)
        self.assertEqual(peer.history[1].face, top)
        self.assertEqual(peer.history[1].offered_top, top)
        self.assertEqual(peer.history[2].face, top)
        self.assertEqual(peer.history[1].actor, 2)

    def test_history_retains_own_old_draw_and_full_prefix_past_sixteen_events(self):
        env = self.make_match(turns=20)
        initial = env.observe()
        private_face = env._stock[-1]
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for _ in range(20):
                env.step(0)
                face = next(i for i, n in enumerate(env.observe().hand) if n)
                env.step(discard_action(face))
        observation = env.observe()
        self.assertEqual(len(observation.history), 41)
        self.assertEqual(observation.history[0].initial_hand, initial.hand)
        self.assertEqual(observation.history[1].face, private_face)
        self.assertEqual(len(initial.history), 1)
        self.assertEqual(initial.history[0].initial_hand, initial.hand)

    def test_relative_projection_survives_cyclic_absolute_seat_relabeling(self):
        original = self.make_match()
        original._current_seat = 2
        rotated = copy.deepcopy(original)
        rotated._hands = [hand.copy() for hand in original._hands[2:] + original._hands[:2]]
        rotated._turns_remaining = original._turns_remaining[2:] + original._turns_remaining[:2]
        rotated._current_seat = 0
        rotated._journal.start(
            rotated._hands, rotated._turns_remaining, len(rotated._stock),
            rotated._discard[-1],
        )
        self.assertEqual(original.observe(), rotated.observe())
        original.step(0)
        rotated.step(0)
        self.assertEqual(original.observe(), rotated.observe())

    def test_player_views_encode_skipped_next_and_previous_players(self):
        env = self.make_match()
        env._turns_remaining[1] = 0
        players = env.observe().players
        self.assertEqual(players[0].next_player, 2)
        self.assertEqual(players[2].previous_player, 0)
        self.assertFalse(players[1].active)
        self.assertEqual(players[3].next_player, 0)
        self.assertEqual(tuple(p.hand_size for p in players), (3, 3, 3, 3))

    def test_constructed_starts_record_only_the_actual_initial_hand(self):
        env = PappluEnv(GameConfig(max_turns=5))
        for reset in (env.reset, env.reset_warm_start, lambda seed: env.reset_curriculum(4, seed)):
            obs = reset(seed=17)
            self.assertEqual(obs.history[0].initial_hand, obs.hand)
            self.assertEqual(len(obs.history), 1)
            self.assertEqual(obs.players[0].hand_size, sum(obs.hand))
            self.assertEqual(Observation.from_dict(obs.to_dict()), obs)

    def test_initial_history_rejects_consistently_corrupted_reset_state(self):
        original = self.make_match(players=2).observe().to_dict()
        wrong_stock = copy.deepcopy(original)
        wrong_stock["stock_remaining"] -= 1
        wrong_stock["history"][0]["stock_remaining"] -= 1
        reduced_turns = copy.deepcopy(original)
        reduced_turns["turns_remaining"] -= 1
        reduced_turns["players"][0]["turns_remaining"] -= 1
        reduced_turns["history"][0]["turns_remaining"] -= 1
        extra_card = copy.deepcopy(original)
        cards = list(extra_card["hand"])
        cards[next(face for face, count in enumerate(cards) if count == 0)] += 1
        extra_card["hand"] = extra_card["history"][0]["initial_hand"] = tuple(cards)
        extra_card["players"][0]["hand_size"] += 1
        extra_card["history"][0]["hand_size"] += 1
        extra_card["discard_pile"] = ()
        extra_card["history"][0]["offered_top"] = None
        extra_card["phase"] = 1
        for name, corrupted in (
            ("stock", wrong_stock), ("turns", reduced_turns), ("multiplayer warm start", extra_card),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "initial"):
                Observation.from_dict(corrupted)

    def test_initial_discard_shape_matches_the_start_type(self):
        env = PappluEnv(GameConfig(max_turns=5))
        ordinary = env.reset(17).to_dict()
        ordinary["discard_pile"] = ()
        ordinary["history"][0]["offered_top"] = None
        warm = env.reset_warm_start(17).to_dict()
        warm["discard_pile"] = (0,)
        warm["history"][0]["offered_top"] = 0
        for corrupted in (ordinary, warm):
            with self.subTest(warm=corrupted["warm_start"]), self.assertRaisesRegex(ValueError, "initial"):
                Observation.from_dict(corrupted)

    def test_ordinary_start_with_zero_stock_roundtrips(self):
        env = MultiplayerEnv(MatchConfig(
            GameConfig(num_decks=1, cards_in_hand=25, required_sequences=1), 2,
        ))
        env.reset(73)
        observation = env.observe()
        self.assertEqual(observation.stock_remaining, 0)
        self.assertIsNotNone(observation.discard_top)
        self.assertEqual(Observation.from_dict(observation.to_dict()), observation)

    def test_primitive_load_rejects_privileged_and_missing_history(self):
        env = self.make_match()
        env.step(0)
        env._current_seat = 1
        obs = env.observe()
        raw = obs.to_dict()
        bad = copy.deepcopy(raw)
        bad["history"][1]["face"] = 17
        with self.assertRaisesRegex(ValueError, "opponent stock"):
            Observation.from_dict(bad)
        with self.assertRaisesRegex(ValueError, "complete initial history"):
            Observation.from_dict({**raw, "history": ()})
        with self.assertRaisesRegex(ValueError, "current information schema"):
            Observation.from_dict({"hand": obs.hand})

    def test_primitive_load_rejects_inconsistent_own_card_history(self):
        env = PappluEnv(GameConfig(max_turns=5))
        original = env.reset(17)
        observation = env.step(0)
        self.assertEqual(Observation.from_dict(observation.to_dict()), observation)
        wrong_face = observation.to_dict()
        wrong_face["history"][-1]["face"] = (
            wrong_face["history"][-1]["face"] + 1
        ) % 52
        with self.assertRaisesRegex(ValueError, "own card history"):
            Observation.from_dict(wrong_face)
        wrong_count = original.to_dict()
        wrong_count["history"][0]["hand_size"] = 0
        with self.assertRaisesRegex(ValueError, "own card history"):
            Observation.from_dict(wrong_count)
        missing_draw = observation.to_dict()
        missing_draw["history"] = missing_draw["history"][:-1]
        with self.assertRaisesRegex(ValueError, "own card history"):
            Observation.from_dict(missing_draw)
        wrong_turns = observation.to_dict()
        wrong_turns["players"][0]["turns_remaining"] -= 1
        with self.assertRaisesRegex(ValueError, "own player turns"):
            Observation.from_dict(wrong_turns)

    def test_snapshot_outcome_and_start_flags_match_history(self):
        env = PappluEnv(GameConfig(max_turns=1))
        initial = env.reset(17)
        for change in (
            {"won": True}, {"last_reward": 1.0}, {"warm_start": True},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Observation.from_dict({**initial.to_dict(), **change})
        env.step(0)
        terminal = env.step(discard_action(next(i for i, n in enumerate(env.observe().hand) if n)))
        self.assertEqual(Observation.from_dict(terminal.to_dict()), terminal)
        warm = env.reset_warm_start(17)
        self.assertEqual(Observation.from_dict(warm.to_dict()), warm)
        with self.assertRaisesRegex(ValueError, "warm_start"):
            Observation.from_dict({**warm.to_dict(), "warm_start": False})


if __name__ == "__main__":
    unittest.main()
