from __future__ import annotations

from collections import Counter
import copy
import operator
import unittest
from unittest import mock

from src.environment import (
    ACTION_DRAW_STOCK,
    ACTION_TAKE_DISCARD,
    GameConfig,
    Phase,
    discard_action,
    legal_action_mask,
)
from src.multiplayer import MatchConfig, MatchView, MultiplayerEnv


def snapshot(env: MultiplayerEnv):
    return (
        env.view(),
        tuple(tuple(hand) for hand in env._hands),
        tuple(env._stock),
        tuple(env._discard),
        env._joker,
        env._rng.getstate(),
    )


def first_discard(env: MultiplayerEnv) -> int:
    observation = env.observe()
    return discard_action(
        next(face for face, count in enumerate(observation.hand) if count)
    )


class TestMatchConfig(unittest.TestCase):
    def test_defaults_are_production_rules_with_sixty_turns(self):
        config = MatchConfig()
        self.assertEqual(config.players, 2)
        self.assertEqual(config.game.num_decks, 3)
        self.assertEqual(config.game.cards_in_hand, 21)
        self.assertEqual(config.game.required_sequences, 5)
        self.assertEqual(config.game.max_turns, 60)
        self.assertTrue(config.game.recycle_discard)

    def test_accepts_two_through_six_players(self):
        for players in range(2, 7):
            with self.subTest(players=players):
                self.assertEqual(MatchConfig(players=players).players, players)
        for players in (1, 7, True, 2.5):
            with self.subTest(players=players), self.assertRaises(
                (TypeError, ValueError)
            ):
                MatchConfig(players=players)

    def test_rejects_multiplayer_supply_shortage(self):
        game = GameConfig(
            num_decks=1,
            cards_in_hand=21,
            required_sequences=5,
            max_turns=1,
        )
        with self.assertRaisesRegex(ValueError, "deck supply"):
            MatchConfig(game=game, players=3)


class TestMultiplayerDeal(unittest.TestCase):
    def test_every_face_is_conserved_for_two_through_six_players(self):
        for players in range(2, 7):
            with self.subTest(players=players):
                config = MatchConfig(players=players)
                env = MultiplayerEnv(config)
                view = env.reset(seed=100 + players)
                counts = Counter(env._stock)
                counts.update(env._discard)
                counts[env._joker] += 1
                for hand in env._hands:
                    counts.update(
                        {
                            face: count
                            for face, count in enumerate(hand)
                            if count
                        }
                    )
                    self.assertEqual(sum(hand), config.game.cards_in_hand)
                self.assertEqual(
                    counts,
                    Counter(
                        {
                            face: config.game.num_decks
                            for face in range(52)
                        }
                    ),
                )
                self.assertEqual(
                    view.stock_remaining,
                    config.game.num_decks * 52
                    - players * config.game.cards_in_hand
                    - 2,
                )

    def test_indicator_is_removed_from_stock_discard_and_hands(self):
        env = MultiplayerEnv(MatchConfig(players=6))
        env.reset(seed=9)
        tracked = (
            sum(map(sum, env._hands))
            + len(env._stock)
            + len(env._discard)
        )
        self.assertEqual(tracked, env.config.game.num_decks * 52 - 1)

    def test_seed_repeats_complete_private_deal(self):
        config = MatchConfig(players=4)
        first = MultiplayerEnv(config)
        second = MultiplayerEnv(config)
        first.reset(seed=444)
        second.reset(seed=444)
        self.assertEqual(snapshot(first), snapshot(second))
        first.reset(seed=444)
        self.assertEqual(snapshot(first), snapshot(second))

    def test_public_view_contains_no_hands_or_stock_order(self):
        env = MultiplayerEnv(MatchConfig(players=3))
        view = env.reset(seed=5)
        self.assertIsInstance(view, MatchView)
        self.assertEqual(
            set(vars(view)),
            {
                "current_seat",
                "phase",
                "winner",
                "done",
                "terminal_reason",
                "turns_remaining",
                "stock_remaining",
            },
        )
        self.assertNotIn("hand", vars(view))
        self.assertNotIn("stock", vars(view))

    def test_observation_is_current_seat_private_snapshot(self):
        env = MultiplayerEnv(MatchConfig(players=2))
        env.reset(seed=12)
        observation = env.observe()
        self.assertEqual(
            observation.hand,
            tuple(env._hands[0]),
        )
        with self.assertRaises(TypeError):
            operator.setitem(observation.hand, 0, 99)
        before = observation.hand
        env.step(ACTION_DRAW_STOCK)
        self.assertEqual(observation.hand, before)


class TestMultiplayerTurns(unittest.TestCase):
    def setUp(self):
        game = GameConfig(
            num_decks=1,
            cards_in_hand=3,
            required_sequences=1,
            max_turns=3,
        )
        self.env = MultiplayerEnv(MatchConfig(game=game, players=3))
        self.env.reset(seed=21)

    def assert_faces_conserved(self):
        counts = Counter(self.env._stock)
        counts.update(self.env._discard)
        counts[self.env._joker] += 1
        for hand in self.env._hands:
            counts.update(
                {
                    face: count
                    for face, count in enumerate(hand)
                    if count
                }
            )
        self.assertEqual(
            counts,
            Counter(
                {
                    face: self.env.config.game.num_decks
                    for face in range(52)
                }
            ),
        )

    def test_each_action_preserves_every_physical_face(self):
        self.assert_faces_conserved()
        self.env.step(ACTION_DRAW_STOCK)
        self.assert_faces_conserved()
        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
            self.env.step(first_discard(self.env))
        self.assert_faces_conserved()

    def test_phase_and_action_mask_follow_draw_then_discard(self):
        draw = self.env.observe()
        draw_mask = legal_action_mask(draw)
        self.assertEqual(draw.phase, Phase.DRAW)
        self.assertTrue(draw_mask[ACTION_DRAW_STOCK])
        self.assertTrue(draw_mask[ACTION_TAKE_DISCARD])
        self.assertFalse(any(draw_mask[2:]))

        first = self.env.view().current_seat
        self.env.step(ACTION_DRAW_STOCK)
        discard = self.env.observe()
        discard_mask = legal_action_mask(discard)
        self.assertEqual(discard.phase, Phase.DISCARD)
        self.assertEqual(self.env.view().current_seat, first)
        self.assertFalse(discard_mask[ACTION_DRAW_STOCK])
        self.assertFalse(discard_mask[ACTION_TAKE_DISCARD])
        self.assertTrue(any(discard_mask[2:]))

        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
            view = self.env.step(first_discard(self.env))
        self.assertEqual(view.current_seat, 1)
        self.assertEqual(view.phase, Phase.DRAW)
        self.assertEqual(view.turns_remaining, (2, 3, 3))

    def test_exhausted_seat_is_skipped(self):
        self.env._turns_remaining = [2, 0, 2]
        self.env.step(ACTION_DRAW_STOCK)
        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
            view = self.env.step(first_discard(self.env))
        self.assertEqual(view.current_seat, 2)
        self.assertEqual(view.turns_remaining, (1, 0, 2))

    def test_two_player_last_round_reaches_both_seats(self):
        game = GameConfig(
            num_decks=1,
            cards_in_hand=3,
            required_sequences=1,
            max_turns=1,
        )
        env = MultiplayerEnv(MatchConfig(game=game, players=2))
        env.reset(seed=8)
        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
            env.step(ACTION_DRAW_STOCK)
            first = env.step(first_discard(env))
            self.assertFalse(first.done)
            self.assertEqual(first.current_seat, 1)
            env.step(ACTION_DRAW_STOCK)
            final = env.step(first_discard(env))
        self.assertTrue(final.done)
        self.assertEqual(final.terminal_reason, "turns_exhausted")
        self.assertEqual(final.turns_remaining, (0, 0))
        self.assertIsNone(final.winner)

    def test_first_verified_win_terminates_match(self):
        self.env._phase = Phase.DISCARD
        self.env._current_seat = 0
        self.env._hands[0] = [0] * 52
        self.env._hands[0][0] = 1
        self.env._hands[0][1] = 1
        self.env._hands[0][2] = 1
        self.env._hands[0][20] = 1
        self.env._joker = 40
        view = self.env.step(discard_action(20))
        self.assertTrue(view.done)
        self.assertEqual(view.winner, 0)
        self.assertEqual(view.terminal_reason, "win")
        self.assertEqual(view.turns_remaining[0], 2)

    def test_legacy_rule_does_not_recycle_discard(self):
        game = GameConfig(
            num_decks=1,
            cards_in_hand=3,
            required_sequences=1,
            max_turns=3,
            recycle_discard=False,
        )
        self.env = MultiplayerEnv(MatchConfig(game=game, players=3))
        self.env.reset(seed=21)
        self.env._stock = [7]
        self.env._discard = [8]
        self.env.step(ACTION_DRAW_STOCK)
        self.assertEqual(self.env._stock, [])
        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
            self.env.step(discard_action(7))
        observation = self.env.observe()
        mask = legal_action_mask(observation)
        self.assertFalse(mask[ACTION_DRAW_STOCK])
        self.assertTrue(mask[ACTION_TAKE_DISCARD])
        self.assertEqual(observation.stock_remaining, 0)

    def test_nonterminal_discard_refills_before_next_seat_draw(self):
        self.env._stock = [7]
        self.env._discard = [8, 9]
        self.env._hands[0] = [0] * 52
        self.env._hands[0][0] = 1
        self.env._hands[0][1] = 1
        self.env._hands[0][2] = 1
        self.env.step(ACTION_DRAW_STOCK)
        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
            view = self.env.step(discard_action(7))
        self.assertEqual(view.current_seat, 1)
        self.assertEqual(view.phase, Phase.DRAW)
        self.assertEqual(view.turns_remaining, (2, 3, 3))
        self.assertEqual(view.stock_remaining, 2)
        self.assertEqual(self.env._discard, [7])
        self.assertTrue(legal_action_mask(self.env.observe())[ACTION_DRAW_STOCK])

    def test_manual_draw_state_refills_before_stock_pop(self):
        self.env._stock = []
        self.env._discard = [7, 8, 9]
        self.env._hands[0] = [0] * 52
        self.env._hands[0][0] = 1
        self.env._hands[0][1] = 1
        self.env._hands[0][2] = 1
        self.env._phase = Phase.DRAW
        self.assertTrue(legal_action_mask(self.env.observe())[ACTION_DRAW_STOCK])
        view = self.env.step(ACTION_DRAW_STOCK)
        self.assertEqual(view.phase, Phase.DISCARD)
        self.assertEqual(view.stock_remaining, 1)
        self.assertEqual(self.env._discard, [9])
        self.assertEqual(sum(self.env._hands[0]), 4)

    def test_terminal_round_does_not_refill_or_reset_budget(self):
        game = GameConfig(
            num_decks=1,
            cards_in_hand=3,
            required_sequences=1,
            max_turns=1,
        )
        env = MultiplayerEnv(MatchConfig(game=game, players=2))
        env.reset(seed=22)
        env._turns_remaining = [0, 1]
        env._current_seat = 1
        env._stock = [7]
        env._discard = [8, 9]
        env._hands[1] = [0] * 52
        env._hands[1][0] = 1
        env._hands[1][1] = 1
        env._hands[1][2] = 1
        env.step(ACTION_DRAW_STOCK)
        rng_before = env._rng.getstate()
        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
            view = env.step(discard_action(7))
        self.assertTrue(view.done)
        self.assertEqual(view.terminal_reason, "turns_exhausted")
        self.assertEqual(view.turns_remaining, (0, 0))
        self.assertEqual(view.stock_remaining, 0)
        self.assertEqual(env._discard, [8, 9, 7])
        self.assertEqual(env._rng.getstate(), rng_before)


class TestMultiplayerRecycling(unittest.TestCase):
    def test_seeded_refill_conserves_every_face_for_two_through_six_players(self):
        for players in range(2, 7):
            with self.subTest(players=players):
                config = MatchConfig(players=players)
                first = MultiplayerEnv(config)
                second = MultiplayerEnv(config)
                first.reset(seed=200 + players)
                second.reset(seed=200 + players)
                for env in (first, second):
                    env._discard = env._stock + env._discard
                    env._stock = []
                    hand_before = env.observe().hand
                    env.step(ACTION_DRAW_STOCK)
                    drawn = next(
                        face
                        for face, (before, after) in enumerate(
                            zip(hand_before, env.observe().hand)
                        )
                        if after == before + 1
                    )
                    with mock.patch(
                        "src.multiplayer.hand_reward",
                        return_value=0.0,
                    ):
                        env.step(discard_action(drawn))

                self.assertEqual(snapshot(first), snapshot(second))
                counts = Counter(first._stock)
                counts.update(first._discard)
                for hand in first._hands:
                    counts.update(
                        {
                            face: count
                            for face, count in enumerate(hand)
                            if count
                        }
                    )
                self.assertEqual(sum(counts.values()), 155)
                counts[first._joker] += 1
                self.assertEqual(
                    counts,
                    Counter({face: 3 for face in range(52)}),
                )
                self.assertEqual(len(first._hands[0]), 52)
                self.assertTrue(
                    all(
                        isinstance(value, int)
                        for hand in first._hands
                        for value in hand
                    )
                )
                self.assertTrue(
                    all(isinstance(face, int) for face in first._stock)
                )
                self.assertTrue(
                    all(isinstance(face, int) for face in first._discard)
                )


class TestMultiplayerAtomicity(unittest.TestCase):
    def setUp(self):
        game = GameConfig(
            num_decks=1,
            cards_in_hand=3,
            required_sequences=1,
            max_turns=2,
        )
        self.env = MultiplayerEnv(MatchConfig(game=game, players=2))
        self.env.reset(seed=91)

    def test_illegal_and_bad_type_actions_do_not_mutate(self):
        for action, error in (
            (True, TypeError),
            (99, ValueError),
            (discard_action(0), ValueError),
        ):
            with self.subTest(action=action):
                before = copy.deepcopy(snapshot(self.env))
                with self.assertRaises(error):
                    self.env.step(action)
                self.assertEqual(snapshot(self.env), before)

    def test_evaluator_error_preserves_pending_discard(self):
        self.env.step(ACTION_DRAW_STOCK)
        action = first_discard(self.env)
        self.env._stock = []
        self.env._discard = [7, 8]
        before = copy.deepcopy(snapshot(self.env))
        with mock.patch(
            "src.multiplayer.hand_reward",
            side_effect=RuntimeError("evaluation failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                self.env.step(action)
        self.assertEqual(snapshot(self.env), before)
        self.assertEqual(self.env.observe().phase, Phase.DISCARD)

    def test_terminal_action_rejection_does_not_mutate(self):
        self.env._phase = Phase.TERMINAL
        self.env._terminal_reason = "turns_exhausted"
        before = copy.deepcopy(snapshot(self.env))
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.env.step(ACTION_DRAW_STOCK)
        self.assertEqual(snapshot(self.env), before)

    def test_invalid_stock_source_preserves_rng_and_cards(self):
        self.env._stock = []
        self.env._discard = [7]
        before = copy.deepcopy(snapshot(self.env))
        with self.assertRaisesRegex(ValueError, "illegal action"):
            self.env.step(ACTION_DRAW_STOCK)
        self.assertEqual(snapshot(self.env), before)


if __name__ == "__main__":
    unittest.main()
