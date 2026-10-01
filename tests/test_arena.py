from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from unittest import mock

from src.arena import (
    CheckpointPolicy,
    EpsilonPolicy,
    MatchInterrupted,
    MatchResult,
    RandomPolicy,
    main,
    play_match,
    rotated_matches,
)
from src.environment import (
    GameConfig,
    legal_action_mask,
)
from src.multiplayer import MatchConfig, MultiplayerEnv

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

    def test_draw_records_complete_zero_reward_trajectories(self):
        config = MatchConfig(game=self.game, players=3)
        policies = [FirstLegalPolicy() for _ in range(3)]
        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
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
        self.assertEqual(tuple(map(len, result.trajectories)), (4, 4, 4))
        for trajectory in result.trajectories:
            for state, action, reward in trajectory:
                self.assertIsInstance(state, tuple)
                self.assertEqual(len(state), 164)
                self.assertIsInstance(action, int)
                self.assertEqual(reward, 0.0)

    def test_only_winning_discard_receives_reward(self):
        config = MatchConfig(game=self.game, players=2)
        with mock.patch("src.multiplayer.hand_reward", return_value=1.0):
            result = play_match(
                [FirstLegalPolicy(), FirstLegalPolicy()],
                config,
                seed=71,
                record_trajectories=True,
            )
        self.assertEqual(result.winner, 0)
        self.assertEqual(result.terminal_reason, "win")
        self.assertEqual(result.action_count, 2)
        self.assertEqual(
            [step[2] for step in result.trajectories[0]],
            [0.0, 1.0],
        )
        self.assertEqual(result.trajectories[1], ())

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

        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
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
        with mock.patch("src.multiplayer.hand_reward", return_value=0.0):
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
            "src.arena.time.monotonic",
            side_effect=(0.0, 2.0),
        ), mock.patch("src.multiplayer.hand_reward") as evaluator:
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
        with mock.patch("src.multiplayer.hand_reward", return_value=1.0):
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
            trajectories=((), (), ()),
        )
        with mock.patch("src.arena.play_match", return_value=result) as play:
            results = rotated_matches(
                policies,
                MatchConfig(game=self.game, players=3),
                seed=78,
            )
        self.assertEqual(results, [result, result, result])
        self.assertEqual(
            [tuple(call.args[0]) for call in play.call_args_list],
            [
                policies,
                policies[1:] + policies[:1],
                policies[2:] + policies[:2],
            ],
        )
        self.assertEqual(
            [call.args[2] for call in play.call_args_list],
            [78, 78, 78],
        )


class TestArenaPolicies(unittest.TestCase):
    def test_epsilon_boundaries(self):
        network = object()
        self.assertEqual(EpsilonPolicy(network, 0.0).epsilon, 0.0)
        self.assertEqual(EpsilonPolicy(network, 1.0).epsilon, 1.0)
        for epsilon in (-0.01, 1.01, float("nan"), True, "0.5"):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                EpsilonPolicy(network, epsilon)

    def test_checkpoint_policy_is_lazy(self):
        policy = CheckpointPolicy("model.pt")
        self.assertIsNone(policy._network)
        self.assertIsNone(policy._game_config)


class TestArenaCli(unittest.TestCase):
    def test_cli_infers_players_rotates_seats_and_omits_trajectories(self):
        fake = MatchResult(
            seed=0,
            winner=0,
            terminal_reason="win",
            seat_turns=(1, 0),
            action_count=2,
            stock_remaining=110,
            trajectories=((), ()),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "matches.json")
            with mock.patch("src.arena.play_match", return_value=fake) as play:
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
            [call.kwargs["seed"] for call in play.call_args_list],
            [800, 801],
        )


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestArenaCheckpoints(unittest.TestCase):
    def test_v1_v2_mlp_wide_and_cnn_checkpoints_load_in_eval_mode(self):
        from src.model import QNetwork, build_checkpoint, save_checkpoint

        checkpoint_config = GameConfig(max_turns=40)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            v1_path = os.path.join(tmp, "v1.pt")
            v1_payload = build_checkpoint(QNetwork(), checkpoint_config)
            v1_payload["checkpoint_version"] = 1
            del v1_payload["architecture"]
            torch.save(v1_payload, v1_path)
            paths.append((v1_path, "mlp"))
            for architecture in ("mlp", "wide_mlp", "suit_conv"):
                path = os.path.join(tmp, architecture + ".pt")
                save_checkpoint(
                    path,
                    QNetwork(architecture=architecture),
                    checkpoint_config,
                )
                paths.append((path, architecture))

            for path, architecture in paths:
                with self.subTest(path=path):
                    policy = CheckpointPolicy(path)
                    self.assertIsNone(policy._network)
                    self.assertEqual(
                        policy.game_config.to_dict(),
                        checkpoint_config.to_dict(),
                    )
                    self.assertEqual(policy.network.architecture, architecture)
                    self.assertFalse(policy.network.training)

    def test_mlp_and_cnn_checkpoints_play_matched_full21_random_smoke(self):
        from src.model import QNetwork, save_checkpoint

        checkpoint_config = GameConfig(max_turns=40)
        match_config = MatchConfig(
            game=GameConfig(max_turns=60),
            players=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            for architecture in ("mlp", "suit_conv"):
                with self.subTest(architecture=architecture):
                    path = os.path.join(tmp, architecture + ".pt")
                    save_checkpoint(
                        path,
                        QNetwork(architecture=architecture),
                        checkpoint_config,
                    )
                    with mock.patch(
                        "src.multiplayer.hand_reward",
                        side_effect=(0.0, 1.0),
                    ):
                        result = play_match(
                            [RandomPolicy(), CheckpointPolicy(path)],
                            match_config,
                            seed=900,
                            record_trajectories=True,
                        )
                    self.assertEqual(result.winner, 1)
                    self.assertEqual(result.action_count, 4)
                    self.assertEqual(
                        len(result.trajectories[1][0][0]),
                        164,
                    )

    def test_epsilon_policy_does_not_change_network_training_mode(self):
        from src.model import QNetwork

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
        action = EpsilonPolicy(network, 0.0).act(
            env.observe(),
            random.Random(1),
        )
        self.assertTrue(legal_action_mask(env.observe())[action])
        self.assertTrue(network.training)


if __name__ == "__main__":
    unittest.main()
