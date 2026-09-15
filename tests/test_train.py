from __future__ import annotations

import io
import os
import tempfile
import unittest
from unittest import mock

from src.environment import (
    GameConfig,
    PappluEnv,
    Phase,
    discard_action,
    encode_observation,
)
from src.train import discounted_returns, main as train_main

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
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(train_main(["--help"]), 0)
        self.assertIn("--checkpoint", out.getvalue())


@unittest.skipIf(torch is None, TORCH_REASON)
class TestTrainLearning(unittest.TestCase):
    def test_episode_changes_weights_with_signal(self):
        from src.model import load_checkpoint
        from src.train import train

        cfg = GameConfig(
            num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=4
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            train(episodes=0, seed=1, config=cfg, checkpoint_path=path)
            initial, _, _ = load_checkpoint(path)
            summary = train(
                episodes=30, seed=1, config=cfg, checkpoint_path=path,
                warm_start_fraction=1.0, log_every=0,
            )
            trained, _, _ = load_checkpoint(path)
        self.assertGreater(summary["warm_wins"], 0)
        self.assertGreater(summary["train_updates"], 0)
        self.assertTrue(any(
            not torch.equal(a, b)
            for a, b in zip(initial.parameters(), trained.parameters())
        ))

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
            train(
                episodes=2,
                seed=0,
                config=cfg,
                checkpoint_path=path,
                warm_start_fraction=1.0,
                log_every=0,
                updates_per_episode=2,
            )
            net, loaded_cfg, meta = load_checkpoint(path)
            self.assertEqual(loaded_cfg.to_dict(), cfg.to_dict())
            self.assertEqual(meta.get("algorithm"), "episodic_monte_carlo_q_regression")

            summary = train(
                episodes=1,
                seed=1,
                config=cfg,
                load_path=path,
                warm_start_fraction=1.0,
                log_every=0,
            )
            self.assertEqual(summary["episodes"], 1)
            inherited = train(episodes=0, load_path=path)
            self.assertEqual(inherited["game_config"], cfg.to_dict())


    def test_cli_help_lists_contract_flags(self):
        out = io.StringIO()
        err = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
            code = train_main(["--help"])
        self.assertEqual(code, 0)
        text = out.getvalue() + err.getvalue()
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
            "--eval-episodes",
            "--eval-seed",
        ):
            self.assertIn(flag, text)
        self.assertIn("fine-tune", text.lower())

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
        ):
            with self.subTest(name=name, value=value):
                with mock.patch("src.train.PappluEnv") as env:
                    with self.assertRaises(ValueError):
                        train(**{name: value})
                    env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
