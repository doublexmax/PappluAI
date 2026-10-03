from __future__ import annotations

from collections import deque
import io
import os
import random
import tempfile
import unittest
from unittest import mock

from src.environment import (
    GameConfig,
    PappluEnv,
    Phase,
    STATE_DIM,
    discard_action,
    encode_observation,
)
from src.train import main as train_main
from src.training_core import EpisodeReplay, discounted_returns

try:
    import torch
except ImportError:
    torch = None

TORCH_REASON = "PyTorch not installed (see requirements-training.txt)"


class TestDiscountedReturns(unittest.TestCase):
    def test_known_returns(self):
        rewards = [0.0, 0.0, 1.0]
        got = discounted_returns(rewards, gamma=0.5)
        self.assertEqual(got, [0.25, 0.5, 1.0])

    def test_empty(self):
        self.assertEqual(discounted_returns([], 0.9), [])

    def test_bad_gamma(self):
        for gamma in (-0.1, 1.5, float("nan"), float("inf")):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                discounted_returns([1.0], gamma=gamma)

    def test_help_without_training_dependencies(self):
        real_import = __import__

        def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("blocked for CLI help test", name="torch")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=import_without_torch):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertEqual(train_main(["--help"]), 0)
        for flag in (
            "--episodes",
            "--max-turns",
            "--cards-in-hand",
            "--required-sequences",
            "--num-decks",
            "--seed",
            "--warm-start-fraction",
            "--checkpoint",
            "--load",
            "--architecture",
            "--replay-sampling",
            "--eval-episodes",
            "--eval-seed",
        ):
            self.assertIn(flag, out.getvalue())
        self.assertIn("fine-tune", out.getvalue().lower())

    def test_architecture_argument_defaults_to_inference_and_has_choices(self):
        from src.train import build_arg_parser

        parser = build_arg_parser()
        self.assertIsNone(parser.parse_args([]).architecture)
        for architecture in ("mlp", "wide_mlp", "suit_conv"):
            self.assertEqual(
                parser.parse_args(["--architecture", architecture]).architecture,
                architecture,
            )
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--architecture", "unknown"])

    def test_replay_sampling_argument_contract(self):
        from src.train import build_arg_parser

        parser = build_arg_parser()
        self.assertEqual(parser.parse_args([]).replay_sampling, "transition")
        for sampling in ("transition", "episode"):
            self.assertEqual(
                parser.parse_args(["--replay-sampling", sampling]).replay_sampling,
                sampling,
            )
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--replay-sampling", "unknown"])


class TestEpisodeReplay(unittest.TestCase):
    @staticmethod
    def transition(index):
        state = (float(index),) + (0.0,) * (STATE_DIM - 1)
        return (state, index, float(index))

    def test_rejects_empty_episode_and_invalid_batch(self):
        replay = EpisodeReplay(capacity=3)
        with self.assertRaises(ValueError):
            replay.append(())
        for size in (0, -1, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                replay.sample(random.Random(0), size)

    def test_capacity_trims_oldest_episode_prefix(self):
        replay = EpisodeReplay(capacity=5)
        replay.append(tuple(self.transition(index) for index in range(3)))
        replay.append(tuple(self.transition(index) for index in range(3, 7)))
        self.assertEqual(len(replay), 5)

        rng = mock.Mock()
        rng.randrange.side_effect = [0, 0, 1, 0, 1, 1, 1, 2, 1, 3]
        sampled = replay.sample(rng, 5)
        self.assertEqual([transition[1] for transition in sampled], [2, 3, 4, 5, 6])

    def test_oversized_episode_keeps_last_capacity_transitions(self):
        replay = EpisodeReplay(capacity=3)
        replay.append(tuple(self.transition(index) for index in range(8)))
        self.assertEqual(len(replay), 3)

        rng = mock.Mock()
        rng.randrange.side_effect = [0, 0, 0, 1, 0, 2]
        sampled = replay.sample(rng, 3)
        self.assertEqual([transition[1] for transition in sampled], [5, 6, 7])

    def test_sampling_is_seeded_and_repeatable(self):
        replay = EpisodeReplay(capacity=20)
        replay.append(tuple(self.transition(index) for index in range(3)))
        replay.append(tuple(self.transition(index) for index in range(3, 10)))
        first = replay.sample(random.Random(41), 100)
        second = replay.sample(random.Random(41), 100)
        self.assertEqual(first, second)

    def test_sampling_weights_episodes_equally(self):
        replay = EpisodeReplay(capacity=61)
        replay.append((self.transition(0),))
        replay.append(tuple(self.transition(1) for _ in range(60)))
        sampled = replay.sample(random.Random(73), 2000)
        short_episode_draws = sum(transition[1] == 0 for transition in sampled)
        self.assertGreater(short_episode_draws, 900)
        self.assertLess(short_episode_draws, 1100)


@unittest.skipIf(torch is None, TORCH_REASON)
class TestTrainLearning(unittest.TestCase):
    def test_transition_sampling_uses_separate_seeded_streams(self):
        from src.train import train

        def state(index):
            return (float(index),) + (0.0,) * (STATE_DIM - 1)

        episode_steps = (
            (
                (state(0), 0, 0.0),
                (state(1), 1, 1.0),
            ),
            (
                (state(2), 2, 0.0),
                (state(3), 3, 0.0),
                (state(4), 4, 1.0),
            ),
        )
        captured = []
        action_draws = []
        warm_sources = []
        episodes = iter(episode_steps)

        def run_seeded_episode(
            _env,
            _network,
            epsilon,
            rng,
            warm_start,
            device,
        ):
            del epsilon, device
            action_draws.append(rng.random())
            warm_sources.append(warm_start)
            return list(next(episodes)), mock.Mock(won=False)

        def capture_batch(_network, _optimizer, _loss, batch, _device, _clip):
            captured.append(tuple(batch))
            return 0.0

        with mock.patch(
            "src.train.run_episode",
            side_effect=run_seeded_episode,
        ), mock.patch(
            "src.training_core.train_batch",
            side_effect=capture_batch,
        ):
            summary = train(
                episodes=2,
                seed=29,
                gamma=0.9,
                batch_size=2,
                replay_capacity=10,
                updates_per_episode=2,
                log_every=0,
                warm_start_fraction=0.5,
            )

        seed_source = random.Random(29)
        source_rng = random.Random(seed_source.getrandbits(64))
        action_rng = random.Random(seed_source.getrandbits(64))
        replay_rng = random.Random(seed_source.getrandbits(64))
        expected = []
        replay = deque(maxlen=10)
        for steps in episode_steps:
            returns = discounted_returns([step[2] for step in steps], 0.9)
            for (state, action, _reward), ret in zip(steps, returns):
                replay.append((state, action, ret))
            for _ in range(2):
                expected.append(tuple(
                    replay_rng.sample(
                        list(replay),
                        min(2, len(replay)),
                    )
                ))

        self.assertEqual(
            warm_sources,
            [source_rng.random() < 0.5 for _ in episode_steps],
        )
        self.assertEqual(
            action_draws,
            [action_rng.random() for _ in episode_steps],
        )
        self.assertEqual(captured, expected)
        self.assertEqual(summary["replay_sampling"], "transition")

    def test_architecture_selection_summary_and_load_inference(self):
        from src.model import load_checkpoint
        from src.train import train

        cfg = GameConfig(
            num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3
        )
        with tempfile.TemporaryDirectory() as tmp:
            for architecture in ("mlp", "wide_mlp", "suit_conv"):
                with self.subTest(architecture=architecture):
                    path = os.path.join(tmp, architecture + ".pt")
                    summary = train(
                        episodes=0,
                        seed=7,
                        config=cfg,
                        architecture=architecture,
                        checkpoint_path=path,
                    )
                    loaded, _, _ = load_checkpoint(path)
                    self.assertEqual(summary["architecture"], architecture)
                    self.assertEqual(loaded.architecture, architecture)
                    self.assertEqual(
                        summary["param_count"],
                        sum(parameter.numel() for parameter in loaded.parameters()),
                    )
                    inferred = train(episodes=0, seed=7, load_path=path)
                    self.assertEqual(inferred["architecture"], architecture)

    def test_default_architecture_remains_mlp(self):
        from src.train import train

        summary = train(episodes=0)
        self.assertEqual(summary["architecture"], "mlp")

    def test_explicit_architecture_mismatch_is_clean_error(self):
        from src.train import train

        cfg = GameConfig(
            num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wide.pt")
            train(
                episodes=0,
                config=cfg,
                architecture="wide_mlp",
                checkpoint_path=path,
            )
            with self.assertRaisesRegex(ValueError, "architecture"):
                train(episodes=0, load_path=path, architecture="mlp")
            with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                code = train_main(
                    [
                        "--load",
                        path,
                        "--episodes",
                        "0",
                        "--architecture",
                        "suit_conv",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("architecture", err.getvalue())

    def test_controlled_discard_task_raises_chosen_q(self):
        """Repeated warm-start 3-card puzzle: correct discard value should rise."""
        from src.model import load_checkpoint, select_greedy_action
        from src.train import train

        cfg = GameConfig(
            num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3
        )
        env = PappluEnv(cfg)
        obs = env.reset_warm_start(seed=99)
        self.assertEqual(obs.phase, Phase.DISCARD)
        win_face = None
        lose_face = None
        from src.evaluate import hand_reward

        for face, count in enumerate(obs.hand):
            if count <= 0:
                continue
            trial = list(obs.hand)
            trial[face] -= 1
            r = hand_reward(trial, obs.joker, required_sequences=1, cards_in_hand=3)
            if r == 1.0 and win_face is None:
                win_face = face
            if r == 0.0 and lose_face is None:
                lose_face = face
        self.assertIsNotNone(win_face)
        self.assertIsNotNone(lose_face)

        class FixedPuzzleEnv(PappluEnv):
            def reset_warm_start(self, seed=None):
                return super().reset_warm_start(seed=99)

        state_t = torch.tensor(encode_observation(obs), dtype=torch.float32).unsqueeze(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            train(episodes=0, seed=42, config=cfg, checkpoint_path=path)
            initial, _, _ = load_checkpoint(path)
            with mock.patch("src.train.PappluEnv", FixedPuzzleEnv):
                summary = train(
                    episodes=80, seed=42, config=cfg,
                    warm_start_fraction=1.0, epsilon_decay_episodes=60,
                    checkpoint_path=path, log_every=0,
                )
            net, _, _ = load_checkpoint(path)
        with torch.no_grad():
            q0 = initial(state_t)[0]
            q1 = net(state_t)[0]
        win_action = discard_action(win_face)
        lose_action = discard_action(lose_face)
        self.assertGreater(summary["warm_wins"], 0)
        self.assertGreater(float(q1[win_action]), float(q0[win_action]))
        self.assertGreater(float(q1[win_action]), float(q1[lose_action]))
        self.assertEqual(env.step(select_greedy_action(net, obs)).last_reward, 1.0)

    def test_seeded_train_reproducible_metrics(self):
        from src.train import train

        cfg = GameConfig(
            num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3
        )
        kwargs = dict(
            episodes=3,
            seed=123,
            gamma=0.95,
            lr=1e-3,
            batch_size=4,
            warm_start_fraction=0.5,
            updates_per_episode=2,
            config=cfg,
            log_every=0,
            epsilon_decay_episodes=3,
        )
        a = train(**kwargs)
        b = train(**kwargs)
        self.assertEqual(a, b)

    def test_heldout_eval_no_param_mutation(self):
        from src.model import QNetwork
        from src.train import evaluate_policy

        cfg = GameConfig(
            num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3
        )
        net = QNetwork()
        before = [p.detach().clone() for p in net.parameters()]
        rng_before = torch.get_rng_state().clone()
        first = evaluate_policy(net, config=cfg, games=3, seed=50)
        second = evaluate_policy(net, config=cfg, games=3, seed=50)
        self.assertEqual(first, second)
        self.assertTrue(net.training)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        for a, b in zip(before, net.parameters()):
            self.assertTrue(torch.allclose(a, b))

    def test_checkpoint_save_load_cli_smoke(self):
        from src.train import train
        from src.model import load_checkpoint

        cfg = GameConfig(
            num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pt")
            summary = train(
                episodes=2,
                seed=0,
                config=cfg,
                checkpoint_path=path,
                warm_start_fraction=1.0,
                log_every=0,
                updates_per_episode=2,
                replay_sampling="episode",
            )
            net, loaded_cfg, meta = load_checkpoint(path)
            self.assertEqual(summary["replay_sampling"], "episode")
            self.assertEqual(loaded_cfg.to_dict(), cfg.to_dict())
            self.assertEqual(meta.get("algorithm"), "episodic_monte_carlo_q_regression")
            self.assertEqual(meta["summary"]["replay_sampling"], "episode")

            summary = train(
                episodes=1,
                seed=1,
                config=cfg,
                load_path=path,
                warm_start_fraction=1.0,
                log_every=0,
            )
            self.assertEqual(summary["episodes"], 1)
            self.assertEqual(summary["replay_sampling"], "transition")
            inherited = train(episodes=0, load_path=path)
            self.assertEqual(inherited["game_config"], cfg.to_dict())


    def test_cli_rejects_bad_config(self):
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            code = train_main(
                [
                    "--episodes",
                    "1",
                    "--cards-in-hand",
                    "3",
                    "--required-sequences",
                    "9",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("error:", err.getvalue())

    def test_cli_short_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoints", "cli.pt")
            code = train_main(
                [
                    "--episodes",
                    "2",
                    "--cards-in-hand",
                    "3",
                    "--required-sequences",
                    "1",
                    "--num-decks",
                    "2",
                    "--max-turns",
                    "3",
                    "--warm-start-fraction",
                    "1",
                    "--replay-sampling",
                    "episode",
                    "--log-every",
                    "0",
                    "--checkpoint",
                    path,
                    "--seed",
                    "0",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(os.path.isfile(path))
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(train_main([
                    "--load", path, "--episodes", "1", "--warm-start-fraction", "1",
                    "--max-turns", "3", "--checkpoint", path,
                ]), 0)
            with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                self.assertEqual(train_main([
                    "--load", path, "--episodes", "0", "--cards-in-hand", "6",
                ]), 2)
                self.assertIn("does not match", err.getvalue())

    def test_rejects_invalid_hyperparameters_before_any_game(self):
        from src.train import train

        for name, value in (
            ("episodes", -1), ("episodes", True), ("batch_size", 0),
            ("replay_capacity", 0), ("gamma", float("nan")),
            ("lr", 0), ("lr", float("inf")), ("epsilon_start", 1.1),
            ("epsilon_end", -0.1), ("epsilon_decay_episodes", -1),
            ("warm_start_fraction", float("nan")), ("grad_clip", -1),
            ("updates_per_episode", 0), ("eval_episodes", -1),
            ("log_every", -1), ("torch_threads", 0),
            ("replay_sampling", "unknown"),
        ):
            with self.subTest(name=name, value=value):
                with mock.patch("src.train.PappluEnv") as env:
                    with self.assertRaises(ValueError):
                        train(**{name: value})
                    env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
