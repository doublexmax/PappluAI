import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from src.cli.solo import build_arg_parser, main as train_main
from src.game.environment import GameConfig, PappluEnv, Phase, discard_action
from src.model.network import QNetwork, collate_observations, load_checkpoint
from src.model.search import TeacherConfig
from src.model.spec import ModelSpec
from src.training.core import TrajectoryReplay
from src.training.optimization import Learner
from src.training.solo import evaluate_policy, run_episode, train


class TestSoloCli(unittest.TestCase):
    def test_help_without_training_dependencies_includes_model_dimensions(self):
        script = """
import builtins, runpy, sys
real_import = builtins.__import__
def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("blocked for CLI help test", name="torch")
    return real_import(name, *args, **kwargs)
builtins.__import__ = import_without_torch
sys.argv = ["src.cli.solo", "--help"]
runpy.run_module("src.cli.solo", run_name="__main__")
"""
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for flag in (
            "--episodes", "--max-turns", "--cards-in-hand", "--required-sequences", "--num-decks",
            "--seed", "--warm-start-fraction", "--checkpoint", "--load", "--eval-episodes",
            "--eval-seed", "--width", "--heads", "--blocks", "--n-step",
        ):
            self.assertIn(flag, completed.stdout)
        self.assertIn("Double DQN", completed.stdout)

    def test_removed_architecture_and_replay_mode_flags_are_rejected(self):
        parser = build_arg_parser()
        for arguments in (["--architecture", "mlp"], ["--replay-sampling", "episode"]):
            with self.subTest(arguments=arguments), mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    parser.parse_args(arguments)

    def test_wide_model_can_be_initialized_and_smoke_trained_from_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "wide.pt")
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                code = train_main([
                    "--episodes", "1", "--width", "128", "--heads", "8", "--blocks", "3",
                    "--cards-in-hand", "3", "--required-sequences", "1", "--num-decks", "2",
                    "--max-turns", "1", "--warm-start-fraction", "1", "--batch-size", "2",
                    "--updates-per-episode", "1", "--log-every", "0", "--checkpoint", path,
                ])
            self.assertEqual(code, 0)
            network, _, _ = load_checkpoint(path)
            self.assertEqual(network.spec.to_dict(), {"width": 128, "heads": 8, "blocks": 3})
            self.assertEqual(json.loads(output.getvalue())["train_updates"], 1)
            with mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                self.assertEqual(train_main(["--load", path, "--episodes", "0", "--width", "64"]), 2)
            self.assertIn("does not match model", error.getvalue())


class TestTrainLearning(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = GameConfig(num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3)

    def test_solo_losses_use_deadwood_credit_without_replacing_the_exact_penalty(self):
        learners = []

        def capture_learner(*args, **kwargs):
            learner = Learner(*args, **kwargs)
            learners.append(learner)
            return learner

        with mock.patch("src.training.solo.Learner", side_effect=capture_learner):
            summary = train(
                episodes=1, seed=19, updates_per_episode=1, log_every=0,
                config=GameConfig(cards_in_hand=3, required_sequences=1, max_turns=1),
                model_spec=ModelSpec(8, 1, 1), teacher=TeacherConfig(fraction=0),
            )
        self.assertEqual(summary["reward_avg"], -0.6875)
        self.assertEqual(summary["return_avg"], -0.6875)
        self.assertEqual(summary["win_rate"], 0.0)
        self.assertEqual(learners[0].replay.state_dict()["trajectories"][0]["outcome"], (-0.6875, False, 24))

    def test_old_replay_requires_a_model_warm_start_without_mutating_live_state(self):
        replay = TrajectoryReplay(8)
        state = replay.state_dict()
        with self.assertRaisesRegex(ValueError, r"v3.*--initial-model"):
            replay.load_state_dict({**state, "version": 3})
        self.assertEqual(replay.state_dict(), state)

    def test_collection_rngs_are_independent_of_replay_and_update_budget(self):
        captured, sources = [], []

        def capture_episode(*args, **kwargs):
            result = run_episode(*args, **kwargs)
            captured.append(result)
            sources.append(kwargs["warm_start"])
            return result

        settings = dict(
            episodes=3, seed=29, gamma=0.9, batch_size=2, replay_capacity=10, log_every=0,
            warm_start_fraction=0.5, epsilon_start=1.0, epsilon_end=1.0, config=self.game,
            teacher=TeacherConfig(fraction=0),
        )
        with mock.patch("src.training.solo.run_episode", side_effect=capture_episode):
            first = train(**settings, updates_per_episode=1)
            second = train(**settings, updates_per_episode=3)
        self.assertEqual(captured[:3], captured[3:])
        seed_source = random.Random(29)
        source_rng = random.Random(seed_source.getrandbits(64))
        self.assertEqual(sources[:3], [source_rng.random() < 0.5 for _ in range(3)])
        self.assertEqual((first["train_updates"], second["train_updates"]), (3, 9))
        self.assertEqual(first["replay_sampling"], "decision")

    def test_current_model_summary_and_loaded_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "current.pt")
            summary = train(episodes=0, seed=7, config=self.game, checkpoint_path=path)
            loaded, _, _ = load_checkpoint(path)
            self.assertEqual(summary["architecture"], loaded.architecture)
            self.assertEqual(summary["param_count"], sum(p.numel() for p in loaded.parameters()))
            inferred = train(episodes=0, seed=7, load_path=path)
            self.assertEqual(inferred["game_config"], self.game.to_dict())
            self.assertEqual(summary["algorithm"], "solo_nstep_ddqn")

    def test_old_checkpoint_is_a_clean_training_error(self):
        from src.model.network import build_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "old.pt")
            payload = build_checkpoint(QNetwork(), self.game)
            payload["checkpoint_version"] = 2
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "fresh axial_table v4 weights"):
                train(episodes=0, load_path=path)
            with mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                self.assertEqual(train_main(["--load", path, "--episodes", "0"]), 2)
            self.assertIn("fresh axial_table v4 weights", error.getvalue())

    def test_controlled_discard_task_learns_a_winning_action(self):
        from src.evaluate import hand_reward
        from src.model.network import select_greedy_action

        env = PappluEnv(self.game)
        obs = env.reset_warm_start(seed=99)
        self.assertEqual(obs.phase, Phase.DISCARD)
        winning, losing = [], []
        for face, count in enumerate(obs.hand):
            if count:
                trial = list(obs.hand)
                trial[face] -= 1
                (winning if hand_reward(trial, obs.joker, 1, 3) else losing).append(face)
        self.assertEqual(len(winning), 1)
        self.assertGreater(len(losing), 0)

        class FixedPuzzleEnv(PappluEnv):
            def reset_warm_start(self, seed=None):
                return super().reset_warm_start(seed=99)

        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "model.pt")
            train(episodes=0, seed=42, config=self.game, checkpoint_path=path)
            initial, _, _ = load_checkpoint(path)
            with mock.patch("src.training.solo.PappluEnv", FixedPuzzleEnv):
                summary = train(
                    episodes=80, seed=42, config=self.game, lr=1e-3,
                    warm_start_fraction=1.0, epsilon_decay_episodes=60,
                    checkpoint_path=path, log_every=0,
                )
            network, _, _ = load_checkpoint(path)
        batch = collate_observations([obs], torch.device("cpu"))
        with torch.no_grad():
            before, after = initial(batch)[0], network(batch)[0]
        action = discard_action(winning[0])
        self.assertGreater(summary["warm_wins"], 0)
        self.assertGreater(float(after[action]), float(before[action]))
        self.assertGreater(float(after[action]), float(after[discard_action(losing[0])]))
        self.assertEqual(env.step(select_greedy_action(network, obs)).last_reward, 1.0)

    def test_seeded_train_reproduces_metrics(self):
        settings = dict(
            episodes=3, seed=123, gamma=0.95, lr=1e-3, batch_size=4,
            warm_start_fraction=0.5, updates_per_episode=2, config=self.game,
            log_every=0, epsilon_decay_episodes=3,
        )
        self.assertEqual(train(**settings), train(**settings))

    def test_heldout_evaluation_never_mutates_parameters_or_rng(self):
        network = QNetwork()
        before = [p.detach().clone() for p in network.parameters()]
        rng = torch.get_rng_state().clone()
        first = evaluate_policy(network, config=self.game, games=3, seed=50)
        second = evaluate_policy(network, config=self.game, games=3, seed=50)
        self.assertEqual(first, second)
        self.assertEqual(first["games"], 3.0)
        self.assertTrue(network.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for saved, current in zip(before, network.parameters()):
            self.assertTrue(torch.equal(saved, current))

    def test_checkpoint_can_be_loaded_for_further_solo_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "model.pt")
            summary = train(
                episodes=2, seed=0, config=self.game, checkpoint_path=path,
                warm_start_fraction=1.0, log_every=0, updates_per_episode=2,
            )
            _, game, meta = load_checkpoint(path)
            self.assertEqual(game, self.game)
            self.assertEqual(summary["train_updates"], 4)
            self.assertEqual(meta["algorithm"], "solo_nstep_ddqn")
            self.assertEqual(meta["summary"]["replay_sampling"], "decision")
            continued = train(episodes=1, seed=1, load_path=path, warm_start_fraction=1.0, log_every=0)
            self.assertEqual(continued["episodes"], 1)
            self.assertEqual(continued["game_config"], self.game.to_dict())

    def test_cli_short_smoke_and_config_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "checkpoints", "cli.pt")
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(train_main([
                    "--episodes", "2", "--cards-in-hand", "3", "--required-sequences", "1",
                    "--num-decks", "2", "--max-turns", "3", "--warm-start-fraction", "1",
                    "--log-every", "0", "--checkpoint", path, "--seed", "0",
                ]), 0)
            self.assertEqual(json.loads(output.getvalue())["train_updates"], 8)
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(train_main([
                    "--load", path, "--episodes", "1", "--warm-start-fraction", "1",
                    "--max-turns", "3", "--checkpoint", path,
                ]), 0)
            self.assertEqual(json.loads(output.getvalue())["replay_sampling"], "decision")
            with mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                self.assertEqual(train_main(["--load", path, "--episodes", "0", "--cards-in-hand", "6"]), 2)
            self.assertIn("does not match", error.getvalue())
        with mock.patch("sys.stderr", new_callable=io.StringIO) as error:
            self.assertEqual(train_main(["--episodes", "1", "--cards-in-hand", "3", "--required-sequences", "9"]), 2)
        self.assertIn("error:", error.getvalue())

    def test_invalid_hyperparameters_are_rejected_before_collection(self):
        for name, value in (
            ("episodes", -1), ("episodes", True), ("batch_size", 0), ("replay_capacity", 0),
            ("gamma", float("nan")), ("gamma", 0), ("lr", 0), ("lr", float("inf")),
            ("epsilon_start", 1.1), ("epsilon_end", -0.1), ("epsilon_decay_episodes", -1),
            ("warm_start_fraction", float("nan")), ("grad_clip", -1), ("updates_per_episode", 0),
            ("eval_episodes", -1), ("log_every", -1), ("torch_threads", 0), ("n_step", 0),
        ):
            with self.subTest(name=name, value=value), mock.patch("src.training.solo.PappluEnv") as env:
                with self.assertRaises(ValueError):
                    train(**{name: value})
                env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
