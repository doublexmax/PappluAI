from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from src.environment import GameConfig, PappluEnv
from src.long_train import (
    STATUS_SCHEMA_VERSION,
    TRAINING_STATE_VERSION,
    TrainingConfig,
    TrainingSession,
    build_arg_parser,
    main,
    run,
    save_session_checkpoint,
)
from src.train import EpisodeReplay

try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "PyTorch not installed (see requirements-training.txt)"


def validation(greedy, random_rate):
    return {
        "greedy": {"games": 10, "wins": int(greedy * 10), "win_rate": greedy},
        "random": {
            "games": 10,
            "wins": int(random_rate * 10),
            "win_rate": random_rate,
        },
    }


class TestLongTrainCli(unittest.TestCase):
    def test_help_does_not_import_torch(self):
        real_import = __import__

        def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("blocked", name="torch")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=import_without_torch):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertEqual(main(["--help"]), 0)
        text = out.getvalue()
        for flag in (
            "--initial-model",
            "--resume",
            "--output-dir",
            "--max-episodes",
            "--max-seconds",
            "--checkpoint-every",
            "--validate-every",
            "--validation-games",
            "--max-turns",
            "--seed",
        ):
            self.assertIn(flag, text)

    def test_source_arguments_are_mutually_exclusive_and_required(self):
        parser = build_arg_parser()
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--output-dir", "out"])
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--initial-model",
                        "a.pt",
                        "--resume",
                        "b.pt",
                        "--output-dir",
                        "out",
                    ]
                )

    def test_production_defaults_are_full21(self):
        config = TrainingConfig()
        self.assertEqual(config.game_config, GameConfig(max_turns=60))
        self.assertEqual(config.architecture, "suit_conv")
        self.assertEqual(config.curriculum_distances, (1, 2, 4, 8))
        self.assertEqual(config.replay_capacity, 10_000)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.updates_per_episode, 4)


class TestEpisodeReplayState(unittest.TestCase):
    def test_roundtrip_preserves_episode_sampling(self):
        replay = EpisodeReplay(capacity=8)
        replay.append((((0.0,), 1, 0.5), ((1.0,), 2, 1.0)))
        replay.append((((2.0,), 3, 0.25),))
        restored = EpisodeReplay(capacity=8)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(
            replay.sample(random.Random(91), 50),
            restored.sample(random.Random(91), 50),
        )
        self.assertEqual(replay.state_dict(), restored.state_dict())

    def test_load_validation_is_atomic(self):
        replay = EpisodeReplay(capacity=4)
        replay.append((((0.0,), 1, 0.5),))
        before = replay.state_dict()
        bad_states = (
            {**before, "version": 999},
            {**before, "capacity": 5},
            {**before, "episodes": ((((float("nan"),), 1, 0.0),),)},
            {**before, "episodes": ((((0.0,), True, 0.0),),)},
        )
        for state in bad_states:
            with self.subTest(state=state), self.assertRaises(
                (TypeError, ValueError)
            ):
                replay.load_state_dict(state)
            self.assertEqual(replay.state_dict(), before)


@unittest.skipIf(torch is None, TORCH_REASON)
class TestTrainingSession(unittest.TestCase):
    @staticmethod
    def small_config(**overrides):
        values = {
            "max_turns": 1,
            "replay_capacity": 32,
            "batch_size": 2,
            "updates_per_episode": 1,
            "validation_games": 1,
            "validate_every": 100,
            "minimum_stage_episodes": 1,
        }
        values.update(overrides)
        return TrainingConfig(**values)

    def assert_model_equal(self, left, right):
        self.assertEqual(set(left.state_dict()), set(right.state_dict()))
        for name, value in left.state_dict().items():
            self.assertTrue(
                torch.equal(value, right.state_dict()[name]),
                msg=name,
            )

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_initial_model_allows_turn_budget_change_only(self):
        from src.model import QNetwork, save_checkpoint

        config = self.small_config()
        with tempfile.TemporaryDirectory() as tmp:
            compatible = os.path.join(tmp, "compatible.pt")
            save_checkpoint(
                compatible,
                QNetwork(architecture="suit_conv"),
                GameConfig(max_turns=30),
            )
            session = TrainingSession(config, seed=41, initial_model=compatible)
            self.assertEqual(session.config.max_turns, 1)
            self.assertEqual(session.network.architecture, "suit_conv")

            wrong_hand = os.path.join(tmp, "wrong-hand.pt")
            save_checkpoint(
                wrong_hand,
                QNetwork(architecture="suit_conv"),
                GameConfig(cards_in_hand=18, required_sequences=5),
            )
            with self.assertRaisesRegex(ValueError, "3 decks, 21 cards"):
                TrainingSession(config, seed=41, initial_model=wrong_hand)

            wrong_architecture = os.path.join(tmp, "wrong-architecture.pt")
            save_checkpoint(
                wrong_architecture,
                QNetwork(architecture="mlp"),
                GameConfig(max_turns=30),
            )
            with self.assertRaisesRegex(ValueError, "architecture"):
                TrainingSession(
                    config, seed=41, initial_model=wrong_architecture
                )

    def test_resume_reproduces_next_action_batch_rng_and_update(self):
        from src.model import select_action

        config = self.small_config()
        control = TrainingSession(config, seed=41)
        split = TrainingSession(config, seed=41)
        control.stage_index = len(config.curriculum_distances)
        split.stage_index = len(config.curriculum_distances)
        control.run_episode()
        split.run_episode()
        self.assert_model_equal(control.network, split.network)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            save_session_checkpoint(split, output)
            resumed = TrainingSession.load(str(output / "latest-state.pt"), config)

            obs = PappluEnv(config.game_config).reset(seed=1234)
            action_rng = random.Random()
            action_rng.setstate(control.action_rng.getstate())
            resumed_action_rng = random.Random()
            resumed_action_rng.setstate(resumed.action_rng.getstate())
            self.assertEqual(
                select_action(
                    control.network,
                    obs,
                    epsilon=control.current_epsilon,
                    rng=action_rng,
                ),
                select_action(
                    resumed.network,
                    obs,
                    epsilon=resumed.current_epsilon,
                    rng=resumed_action_rng,
                ),
            )

            replay_rng = random.Random()
            replay_rng.setstate(control.replay_rng.getstate())
            resumed_replay_rng = random.Random()
            resumed_replay_rng.setstate(resumed.replay_rng.getstate())
            self.assertEqual(
                control.replay.sample(replay_rng, 8),
                resumed.replay.sample(resumed_replay_rng, 8),
            )
            deal_rng = random.Random()
            deal_rng.setstate(control.deal_rng.getstate())
            resumed_deal_rng = random.Random()
            resumed_deal_rng.setstate(resumed.deal_rng.getstate())
            self.assertEqual(
                [deal_rng.random() for _ in range(5)],
                [resumed_deal_rng.random() for _ in range(5)],
            )

            control.run_episode()
            resumed.run_episode()
            self.assert_model_equal(control.network, resumed.network)
            self.assert_nested_equal(
                control.optimizer.state_dict(),
                resumed.optimizer.state_dict(),
            )
            self.assertEqual(
                control.replay.state_dict(),
                resumed.replay.state_dict(),
            )
            self.assertEqual(
                control.state_dict()["counters"],
                resumed.state_dict()["counters"],
            )
            self.assertEqual(
                control.state_dict()["curriculum"],
                resumed.state_dict()["curriculum"],
            )
            for name in ("deal", "action", "replay"):
                self.assertEqual(
                    control.state_dict()["rng_states"][name],
                    resumed.state_dict()["rng_states"][name],
                )

    def test_resume_rejects_version_and_learning_config_changes(self):
        config = self.small_config()
        session = TrainingSession(config, seed=41)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            save_session_checkpoint(session, output)
            path = output / "latest-state.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["training_state_version"] = 999
            bad = output / "bad-version.pt"
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, "training_state_version"):
                TrainingSession.load(str(bad))

            changed = self.small_config(max_turns=2)
            with self.assertRaisesRegex(ValueError, "training_config"):
                TrainingSession.load(str(path), expected_config=changed)

    def test_promotion_needs_two_heldout_passes(self):
        config = self.small_config()
        session = TrainingSession(config, seed=41)
        session.stage_episodes = config.minimum_stage_episodes
        session.epsilon_progress = session.stage_episodes
        session.best_score = 0.0

        full = validation(0.2, 0.3)
        curriculum = validation(0.4, 0.1)
        with mock.patch(
            "src.long_train.evaluate_policy_pair",
            side_effect=[full, curriculum, full, curriculum],
        ):
            first = session.run_validation()
            self.assertTrue(first["passed"])
            self.assertFalse(first["promoted"])
            self.assertEqual(session.consecutive_passing_validations, 1)
            self.assertEqual(session.stage_index, 0)

            second = session.run_validation()
            self.assertTrue(second["promoted"])
            self.assertEqual(session.stage_index, 1)
            self.assertEqual(session.stage_episodes, 0)
            self.assertEqual(session.epsilon_progress, 0)
            self.assertEqual(session.consecutive_passing_validations, 0)

        self.assertEqual(session.best_score, 0.2)
        self.assertEqual(session.best_episode, 0)

    def test_curriculum_score_never_selects_best_model(self):
        session = TrainingSession(self.small_config(), seed=41)
        session.best_score = 0.5
        session.stage_episodes = session.config.minimum_stage_episodes
        full = validation(0.4, 0.0)
        curriculum = validation(1.0, 0.0)
        before = {
            name: value.clone()
            for name, value in session.best_model_state.items()
        }
        with mock.patch(
            "src.long_train.evaluate_policy_pair",
            side_effect=[full, curriculum],
        ):
            result = session.run_validation()
        self.assertFalse(result["best_changed"])
        self.assertEqual(session.best_score, 0.5)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, session.best_model_state[name]))

    def test_validation_and_baseline_do_not_consume_training_rngs(self):
        session = TrainingSession(self.small_config(), seed=41)
        before = session.state_dict()["rng_states"]
        session.initialize_validation()
        after = session.state_dict()["rng_states"]
        for name in ("deal", "action", "replay"):
            self.assertEqual(before[name], after[name])
        self.assertTrue(torch.equal(before["torch"], after["torch"]))
        self.assertEqual(session.baseline_validation, session.full_validation)

    def test_budget_abort_rewinds_partial_trajectory(self):
        config = self.small_config(max_turns=2)
        before_action = TrainingSession(config, seed=41)
        mid_trajectory = TrainingSession(config, seed=41)
        expected = before_action.state_dict()

        with mock.patch("src.long_train.time.monotonic", return_value=2.0):
            self.assertIsNone(before_action.run_episode(deadline=1.0))
        with mock.patch(
            "src.long_train.time.monotonic",
            side_effect=[0.0, 2.0],
        ):
            self.assertIsNone(mid_trajectory.run_episode(deadline=1.0))

        for session in (before_action, mid_trajectory):
            self.assertEqual(session.total_episodes, 0)
            self.assertEqual(len(session.replay), 0)
            self.assert_model_equal(
                TrainingSession(config, seed=41).network,
                session.network,
            )
            current = session.state_dict()
            self.assertEqual(current["counters"], expected["counters"])
            for name in ("deal", "action", "replay"):
                self.assertEqual(
                    current["rng_states"][name],
                    expected["rng_states"][name],
                )

    def test_resume_completes_validation_pending_at_checkpoint_boundary(self):
        config = self.small_config(validate_every=2)
        session = TrainingSession(config, seed=41)
        with mock.patch("src.long_train.evaluate_policy_pair", return_value=validation(0.1, 0.0)):
            session.initialize_validation()
        session.run_episode()
        session.run_episode()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_session_checkpoint(session, output)
            resumed = TrainingSession.load(str(output / "latest-state.pt"))
            self.assertEqual(resumed.last_validation_episode, 0)
            with mock.patch("src.long_train.evaluate_policy_pair", return_value=validation(0.2, 0.0)):
                with mock.patch.object(resumed, "run_episode", side_effect=AssertionError("unexpected new episode")):
                    reason = run(resumed, output, 2, 60, 1)
            self.assertEqual(reason, "max_episodes")
            self.assertEqual(resumed.last_validation_episode, 2)
            self.assertEqual(resumed.total_episodes, 2)
            self.assertEqual(resumed.best_score, 0.2)

    def test_validation_budget_does_not_publish_partial_scores(self):
        from src.long_train import ValidationInterrupted

        session = TrainingSession(self.small_config(), seed=41)
        before = session.state_dict()["rng_states"]
        with self.assertRaises(ValidationInterrupted) as raised:
            session.run_validation(deadline=-1.0)
        self.assertEqual(raised.exception.reason, "budget")
        self.assertIsNone(session.full_validation)
        self.assertIsNone(session.curriculum_validation)
        self.assertTrue(session.network.training)
        after = session.state_dict()["rng_states"]
        for name in ("deal", "action", "replay"):
            self.assertEqual(before[name], after[name])

    def test_atomic_checkpoint_has_resumable_and_standard_models(self):
        from src.model import load_checkpoint

        session = TrainingSession(self.small_config(), seed=41)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            real_replace = os.replace
            replacements = []

            def capture_replace(source, destination):
                replacements.append((Path(source), Path(destination)))
                real_replace(source, destination)

            with mock.patch(
                "src.long_train.os.replace",
                side_effect=capture_replace,
            ):
                manifest = save_session_checkpoint(session, output)

            expected = {
                "latest-state.pt",
                "latest-model.pt",
                "best-model.pt",
                "baseline-model.pt",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected | {"metrics.jsonl"})
            self.assertEqual(
                {destination.name for _source, destination in replacements},
                expected,
            )
            self.assertTrue(
                all(source.name.endswith(".tmp") for source, _ in replacements)
            )
            self.assertEqual(
                set(manifest),
                {"latest-state.pt", "latest-model.pt", "best-model.pt", "baseline-model.pt", "metrics.jsonl"},
            )
            for name, metadata in manifest.items():
                contents = (output / name).read_bytes()
                self.assertEqual(metadata["bytes"], len(contents))
                self.assertEqual(
                    metadata["sha256"],
                    hashlib.sha256(contents).hexdigest(),
                )
            state = torch.load(
                output / "latest-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                state["training_state_version"],
                TRAINING_STATE_VERSION,
            )
            self.assertIn("optimizer_state_dict", state)
            self.assertIn("replay_state_dict", state)
            self.assertIn("rng_states", state)
            for name in ("latest-model.pt", "best-model.pt", "baseline-model.pt"):
                network, game, _ = load_checkpoint(str(output / name))
                self.assertEqual(network.architecture, "suit_conv")
                self.assertEqual(game, session.config.game_config)

    def test_live_metric_append_preserves_the_published_checkpoint_hash(self):
        from src.long_train import _append_metric

        session = TrainingSession(self.small_config(), seed=41)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = save_session_checkpoint(session, output)
            published = (output / "metrics.jsonl").read_bytes()
            _append_metric(output, {"event": "new-episode", "episode": 1})
            self.assertEqual((output / "metrics.jsonl").read_bytes(), published)
            self.assertEqual(hashlib.sha256(published).hexdigest(), manifest["metrics.jsonl"]["sha256"])
            updated = save_session_checkpoint(session, output)
            self.assertIn(b"new-episode", (output / "metrics.jsonl").read_bytes())
            self.assertEqual(
                hashlib.sha256((output / "metrics.jsonl").read_bytes()).hexdigest(),
                updated["metrics.jsonl"]["sha256"],
            )

    def test_run_status_distinguishes_stop_reasons(self):
        config = self.small_config()
        cases = (
            ("max_episodes", 0, 10.0, lambda: False),
            ("budget", 1, 0.0, lambda: False),
            ("stopped", 1, 10.0, lambda: True),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (expected, episodes, seconds, stop) in enumerate(cases):
                with self.subTest(expected=expected):
                    session = TrainingSession(config, seed=41)
                    output = root / str(index)
                    reason = run(
                        session,
                        output,
                        max_episodes=episodes,
                        max_seconds=seconds,
                        checkpoint_every=1,
                        stop_requested=stop,
                    )
                    self.assertEqual(reason, expected)
                    status = json.loads(
                        (output / "status.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(status["status"], expected)
                    self.assertEqual(status["stop_reason"], expected)
                    self.assertEqual(
                        status["status_schema_version"],
                        STATUS_SCHEMA_VERSION,
                    )
                    self.assertEqual(status["completed_episodes"], 0)
                    self.assertEqual(
                        set(status["files"]),
                        {
                            "latest-state.pt",
                            "latest-model.pt",
                            "best-model.pt",
                            "baseline-model.pt",
                            "metrics.jsonl",
                        },
                    )

    def test_remote_two_episode_full21_smoke_contract(self):
        config = TrainingConfig(
            max_turns=2,
            validation_games=2,
        )
        session = TrainingSession(config, seed=41)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            reason = run(
                session,
                output,
                max_episodes=2,
                max_seconds=120,
                checkpoint_every=1,
                initialize=True,
            )
            self.assertEqual(reason, "max_episodes")
            self.assertEqual(session.total_episodes, 2)
            status = json.loads(
                (output / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["completed_episodes"], 2)
            self.assertEqual(status["rules"]["max_turns"], 2)
            self.assertEqual(status["rules"]["validation_games"], 2)
            self.assertEqual(status["rules"]["validate_every"], 1_000)
            for name, metadata in status["files"].items():
                contents = (output / name).read_bytes()
                self.assertEqual(metadata["bytes"], len(contents))
                self.assertEqual(
                    metadata["sha256"],
                    hashlib.sha256(contents).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
