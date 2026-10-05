from __future__ import annotations

import unittest
from unittest import mock

from src.game.environment import (
    ACTION_DRAW_STOCK,
    ACTION_TAKE_DISCARD,
    GameConfig,
    discard_action,
)
from src.game.multiplayer import MatchConfig, MultiplayerEnv


class TestMatchTelemetry(unittest.TestCase):
    def make_match(self, turns=2):
        config = MatchConfig(
            game=GameConfig(
                num_decks=1,
                cards_in_hand=3,
                required_sequences=1,
                max_turns=turns,
            ),
            players=2,
        )
        env = MultiplayerEnv(config)
        env.reset(seed=301)
        return env

    def discard_first(self, env):
        face = next(
            face for face, count in enumerate(env.observe().hand) if count
        )
        return env.step(discard_action(face))

    def test_draw_sources_are_counted_by_seat_without_mutating_snapshots(self):
        env = self.make_match()
        before = env.telemetry()
        env.step(ACTION_DRAW_STOCK)
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            self.discard_first(env)
            env.step(ACTION_TAKE_DISCARD)
            self.discard_first(env)

        self.assertEqual(before.stock_draws, (0, 0))
        self.assertEqual(before.discard_draws, (0, 0))
        self.assertEqual(env.telemetry().stock_draws, (1, 0))
        self.assertEqual(env.telemetry().discard_draws, (0, 1))
        self.assertEqual(env.telemetry().refill_turns, ())

    def test_actual_refills_record_completed_turns_and_do_not_reset_budgets(self):
        env = self.make_match()
        env._stock = [7]
        env._discard = [8, 9]

        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for index in range(4):
                env.step(ACTION_DRAW_STOCK)
                view = self.discard_first(env)
                if index == 0:
                    self.assertEqual(view.turns_remaining, (1, 2))
                    self.assertEqual(env.telemetry().refill_turns, (1,))
                    self.assertEqual(len(env._discard), 1)

        self.assertTrue(view.done)
        self.assertEqual(view.terminal_reason, "turns_exhausted")
        self.assertEqual(view.turns_remaining, (0, 0))
        self.assertEqual(env.telemetry().stock_draws, (2, 2))
        self.assertEqual(env.telemetry().discard_draws, (0, 0))
        self.assertEqual(env.telemetry().refill_turns, (1, 3))

    def test_a_winning_discard_does_not_count_an_unused_refill(self):
        env = self.make_match()
        env._joker = 40
        env._stock = [7]
        env._discard = [8, 9]
        env._hands[0] = [0] * 52
        for face in (0, 1, 2):
            env._hands[0][face] = 1

        env.step(ACTION_DRAW_STOCK)
        view = env.step(discard_action(7))

        self.assertEqual(view.winner, 0)
        self.assertTrue(view.done)
        self.assertEqual(view.stock_remaining, 0)
        self.assertEqual(env.telemetry().stock_draws, (1, 0))
        self.assertEqual(env.telemetry().refill_turns, ())

    def test_reset_clears_all_match_telemetry(self):
        env = self.make_match()
        env._stock = [7]
        env._discard = [8, 9]
        env.step(ACTION_DRAW_STOCK)
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            self.discard_first(env)
        self.assertEqual(env.telemetry().refill_turns, (1,))

        env.reset(seed=301)

        self.assertEqual(env.telemetry().stock_draws, (0, 0))
        self.assertEqual(env.telemetry().discard_draws, (0, 0))
        self.assertEqual(env.telemetry().refill_turns, ())

    def test_private_terminal_snapshot_is_unavailable_during_play(self):
        env = self.make_match()
        with self.assertRaisesRegex(ValueError, "only after"):
            env.terminal_snapshot()
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for _ in range(4):
                env.step(ACTION_DRAW_STOCK)
                self.discard_first(env)
        snapshot = env.terminal_snapshot()
        expected = tuple(tuple(hand) for hand in env._hands)
        self.assertEqual(snapshot.hands, expected)
        self.assertEqual(snapshot.joker, env._joker)
        self.assertFalse(hasattr(env.observe(), "opponent_hands"))
        env.reset(seed=999)
        self.assertEqual(snapshot.hands, expected)


if __name__ == "__main__":
    unittest.main()
