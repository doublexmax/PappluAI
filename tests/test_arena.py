from __future__ import annotations

from dataclasses import replace
import json
import os
import random
import tempfile
import unittest
from unittest import mock

from src.evaluation.arena import (
    CheckpointPolicy,
    EpsilonPolicy,
    MatchInterrupted,
    MatchResult,
    RandomPolicy,
    play_match,
    rotated_matches,
    run_matches,
)
from src.cli.arena import main
from src.game.environment import (
    GameConfig,
    legal_action_mask,
)
from src.game.multiplayer import MatchConfig, MatchTelemetry, MultiplayerEnv

try:
    import torch
except ImportError:
    torch = None


class FirstLegalPolicy:
    def __init__(self, game_config=None):
        self.game_config = game_config
        self.observations = []
        self.calls = 0

    def act(self, observation, rng):
        del rng
        self.calls += 1
        self.observations.append(observation)
        mask = legal_action_mask(observation)
        return next(action for action, legal in enumerate(mask) if legal)


class RandomLegalPolicy:
    def act(self, observation, rng):
        legal = [
            action
            for action, allowed in enumerate(legal_action_mask(observation))
            if allowed
        ]
        return rng.choice(legal)


class TestArenaMatches(unittest.TestCase):
    def setUp(self):
        self.game = GameConfig(
            num_decks=1,
            cards_in_hand=3,
            required_sequences=1,
            max_turns=2,
        )

    def test_draw_records_complete_cached_decisions(self):
        config = MatchConfig(game=self.game, players=3)
        policies = [FirstLegalPolicy() for _ in range(3)]
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            result = play_match(
                policies,
                config,
                seed=70,
                record_trajectories=True,
            )
        self.assertIsNone(result.winner)
        self.assertEqual(result.terminal_reason, "turns_exhausted")
        self.assertEqual(result.seat_turns, (2, 2, 2))
        self.assertEqual(result.action_count, 12)
        self.assertEqual(result.telemetry.stock_draws, (2, 2, 2))
        self.assertEqual(result.telemetry.discard_draws, (0, 0, 0))
        self.assertEqual(result.telemetry.refill_turns, ())
        self.assertEqual(tuple(map(len, result.trajectories)), (4, 4, 4))
        for trajectory in result.trajectories:
            for decision in trajectory:
                state, action = decision.observation, decision.action
                self.assertEqual(len(state.players), 3)
                self.assertEqual(state.history[0].initial_hand, trajectory[0].observation.hand)
                self.assertTrue(legal_action_mask(state)[action])
                self.assertEqual(decision.hidden_hands[0], state.hand)
        for policy, trajectory in zip(policies, result.trajectories):
            self.assertEqual(policy.observations, [step.observation for step in trajectory])

    def test_winning_discard_stops_recording_before_the_other_seat_acts(self):
        config = MatchConfig(game=self.game, players=2)
        with mock.patch("src.game.multiplayer.hand_reward", return_value=1.0):
            result = play_match(
                [FirstLegalPolicy(), FirstLegalPolicy()],
                config,
                seed=71,
                record_trajectories=True,
            )
        self.assertEqual(result.winner, 0)
        self.assertEqual(result.terminal_reason, "win")
        self.assertEqual(result.action_count, 2)
        self.assertEqual(result.trajectories[0][0].action, 0)
        self.assertGreaterEqual(result.trajectories[0][1].action, 2)
        self.assertEqual(result.outcome.winner, 0)
        self.assertEqual(result.trajectories[1], ())
        self.assertIsNone(result.terminal_snapshot)

    def test_terminal_capture_is_opt_in_and_keeps_every_hand(self):
        config = MatchConfig(game=self.game, players=2)
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            result = play_match(
                [FirstLegalPolicy(), FirstLegalPolicy()],
                config, seed=711, record_terminal=True,
            )
        self.assertIsNotNone(result.terminal_snapshot)
        self.assertEqual(len(result.terminal_snapshot.hands), 2)
        self.assertEqual(tuple(sum(hand) for hand in result.terminal_snapshot.hands), (3, 3))

    def test_policies_receive_only_their_own_hand_and_public_state(self):
        game = GameConfig(
            num_decks=1,
            cards_in_hand=3,
            required_sequences=1,
            max_turns=1,
        )
        config = MatchConfig(game=game, players=2)
        policies = [FirstLegalPolicy(), FirstLegalPolicy()]
        expected = MultiplayerEnv(config)
        expected.reset(seed=72)
        expected_hands = tuple(
            tuple(hand) for hand in expected._hands
        )

        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            play_match(policies, config, seed=72)

        for seat, policy in enumerate(policies):
            self.assertEqual(policy.observations[0].hand, expected_hands[seat])
            for observation in policy.observations:
                self.assertFalse(hasattr(observation, "opponent_hands"))
                self.assertFalse(hasattr(observation, "stock"))
                self.assertIsInstance(observation.hand, tuple)
                self.assertIsInstance(observation.discard_pile, tuple)

    def test_seed_repeats_deck_policy_rng_and_trajectories(self):
        config = MatchConfig(game=self.game, players=2)
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            first = play_match(
                [RandomLegalPolicy(), RandomLegalPolicy()],
                config,
                seed=73,
                record_trajectories=True,
            )
            second = play_match(
                [RandomLegalPolicy(), RandomLegalPolicy()],
                config,
                seed=73,
                record_trajectories=True,
            )
        self.assertEqual(first, second)

    def test_deadline_after_policy_does_not_create_a_false_verdict(self):
        policy = FirstLegalPolicy()
        config = MatchConfig(game=self.game, players=2)
        with mock.patch(
            "src.evaluation.arena.time.monotonic",
            side_effect=(0.0, 2.0),
        ), mock.patch("src.game.multiplayer.hand_reward") as evaluator:
            with self.assertRaises(MatchInterrupted):
                play_match(
                    [policy, FirstLegalPolicy()],
                    config,
                    seed=74,
                    deadline=1.0,
                )
        self.assertEqual(policy.calls, 1)
        evaluator.assert_not_called()

    def test_policy_errors_propagate(self):
        class BrokenPolicy:
            def act(self, observation, rng):
                del observation, rng
                raise RuntimeError("policy failed")

        with self.assertRaisesRegex(RuntimeError, "policy failed"):
            play_match(
                [BrokenPolicy(), FirstLegalPolicy()],
                MatchConfig(game=self.game, players=2),
                seed=75,
            )

    def test_mismatched_policy_rules_reject_before_first_action(self):
        policy = FirstLegalPolicy(
            GameConfig(
                num_decks=1,
                cards_in_hand=6,
                required_sequences=1,
                max_turns=2,
            )
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            play_match(
                [policy, FirstLegalPolicy()],
                MatchConfig(game=self.game, players=2),
                seed=76,
            )
        self.assertEqual(policy.calls, 0)

    def test_checkpoint_max_turn_difference_is_compatible(self):
        policy_config = GameConfig(
            num_decks=1,
            cards_in_hand=3,
            required_sequences=1,
            max_turns=99,
        )
        policy = FirstLegalPolicy(policy_config)
        config = MatchConfig(game=self.game, players=2)
        with mock.patch("src.game.multiplayer.hand_reward", return_value=1.0):
            result = play_match(
                [policy, FirstLegalPolicy(policy_config)],
                config,
                seed=77,
            )
        self.assertEqual(result.winner, 0)
        self.assertEqual(policy.observations[0].config.max_turns, 2)

    def test_rotations_keep_seed_and_have_explicit_seat_order(self):
        policies = tuple(FirstLegalPolicy() for _ in range(3))
        result = MatchResult(
            seed=78,
            winner=None,
            terminal_reason="turns_exhausted",
            seat_turns=(2, 2, 2),
            action_count=12,
            stock_remaining=30,
            telemetry=MatchTelemetry((2, 2, 2), (0, 0, 0), ()),
            trajectories=((), (), ()),
        )
        with mock.patch("src.evaluation.arena.run_matches", return_value=[result] * 3) as play:
            results = rotated_matches(
                policies,
                MatchConfig(game=self.game, players=3),
                seed=78,
            )
        self.assertEqual(results, [result, result, result])
        self.assertEqual(
            [deal.seats for deal in play.call_args.args[0]],
            [(0, 1, 2), (1, 2, 0), (2, 0, 1)],
        )
        self.assertEqual(
            [deal.seed for deal in play.call_args.args[0]],
            [78, 78, 78],
        )


class TestArenaPolicies(unittest.TestCase):
    def test_invalid_epsilon_is_rejected(self):
        network = object()
        for epsilon in (-0.01, 1.01, float("nan"), True, "0.5"):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                EpsilonPolicy(network, epsilon)


class TestArenaCli(unittest.TestCase):
    def test_cli_infers_players_rotates_seats_and_omits_trajectories(self):
        fake = MatchResult(
            seed=0,
            winner=0,
            terminal_reason="win",
            seat_turns=(1, 0),
            action_count=2,
            stock_remaining=110,
            telemetry=MatchTelemetry((1, 0), (0, 0), ()),
            trajectories=((), ()),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "matches.json")
            with mock.patch(
                "src.evaluation.arena.run_matches",
                side_effect=lambda deals, *_args, **_kwargs: [replace(fake, seed=deal.seed) for deal in deals],
            ) as play:
                code = main(
                    [
                        "--model",
                        "random",
                        "--model",
                        "random",
                        "--games",
                        "2",
                        "--seed",
                        "800",
                        "--max-turns",
                        "7",
                        "--output",
                        output,
                    ]
                )
            with open(output, encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertEqual(code, 0)
        self.assertEqual(report["config"]["players"], 2)
        self.assertEqual(report["config"]["game"]["max_turns"], 7)
        self.assertEqual(
            [game["seat_order"] for game in report["games"]],
            [[0, 1], [1, 0]],
        )
        self.assertEqual(
            [game["winner_agent"] for game in report["games"]],
            [0, 1],
        )
        self.assertNotIn("trajectories", report["games"][0])
        self.assertEqual(
            [deal.seed for deal in play.call_args.args[0]],
            [800, 801],
        )
        self.assertEqual([game["seed"] for game in report["games"]], [800, 801])
        self.assertFalse(play.call_args.kwargs["score_outcome"])

    def test_cli_batches_across_the_chunk_boundary_without_scoring_unused_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "matches.json")
            with mock.patch("src.evaluation.arena.run_matches", wraps=run_matches) as runner, mock.patch(
                "src.evaluation.arena.DealOutcome.score", side_effect=AssertionError("unused outcome scoring"),
            ):
                code = main([
                    "--model", "random", "--model", "random", "--games", "33",
                    "--seed", "800", "--max-turns", "1", "--output", output,
                ])
            with open(output, encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertEqual(code, 0)
        self.assertEqual([len(call.args[0]) for call in runner.call_args_list], [32, 1])
        self.assertEqual([game["seed"] for game in report["games"]], list(range(800, 833)))
        self.assertEqual(
            [game["seat_order"] for game in report["games"]],
            [[0, 1] if index % 2 == 0 else [1, 0] for index in range(33)],
        )
        self.assertEqual([game["action_count"] for game in report["games"]], [4] * 33)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestArenaCheckpoints(unittest.TestCase):
    def test_current_checkpoint_loads_lazily_in_eval_mode(self):
        from src.model.network import QNetwork, save_checkpoint

        checkpoint_config = GameConfig(max_turns=40)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "current.pt")
            save_checkpoint(path, QNetwork(), checkpoint_config)
            policy = CheckpointPolicy(path)
            self.assertIsNone(policy._network)
            self.assertEqual(policy.game_config, checkpoint_config)
            self.assertEqual(policy.network.architecture, "axial_table")
            self.assertFalse(policy.network.training)

    def test_current_checkpoint_plays_all_supported_full21_tables(self):
        from src.model.network import QNetwork, save_checkpoint

        checkpoint_config = GameConfig(max_turns=40)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "current.pt")
            save_checkpoint(path, QNetwork(), checkpoint_config)
            for players in (2, 3, 4):
                with self.subTest(players=players):
                    with mock.patch(
                        "src.game.multiplayer.hand_reward",
                        side_effect=[0.0] * (players - 1) + [1.0],
                    ):
                        result = play_match(
                            [RandomPolicy()] * (players - 1) + [CheckpointPolicy(path)],
                            MatchConfig(GameConfig(max_turns=60), players),
                            seed=900,
                            record_trajectories=True,
                        )
                    self.assertEqual(result.winner, players - 1)
                    self.assertEqual(result.action_count, players * 2)
                    observation = result.trajectories[-1][0].observation
                    self.assertEqual(len(observation.players), players)
                    self.assertEqual(len(observation.history), 2 * players - 1)

    def test_epsilon_boundaries_select_legal_actions_without_changing_mode(self):
        from src.model.network import QNetwork

        for epsilon in (0.0, 1.0):
            with self.subTest(epsilon=epsilon):
                network = QNetwork()
                network.train()
                env = MultiplayerEnv(
                    MatchConfig(
                        game=GameConfig(
                            num_decks=1,
                            cards_in_hand=3,
                            required_sequences=1,
                            max_turns=1,
                        ),
                        players=2,
                    )
                )
                env.reset(seed=901)
                action = EpsilonPolicy(network, epsilon).act(
                    env.observe(),
                    random.Random(1),
                )
                self.assertTrue(
                    legal_action_mask(env.observe())[action]
                )
                self.assertTrue(network.training)


if __name__ == "__main__":
    unittest.main()
