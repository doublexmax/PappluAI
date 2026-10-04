from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.game.environment import (
    ACTION_DRAW_STOCK,
    NUM_ACTIONS,
    STATE_DIM,
    GameConfig,
    PappluEnv,
    Phase,
    discard_action,
    encode_observation,
)
from src.training.curriculum import (
    STATUS_SCHEMA_VERSION,
    TRAINING_STATE_VERSION,
    TrainingConfig,
    TrainingSession,
    run,
    save_session_checkpoint,
)
from src.cli.curriculum import build_arg_parser
from src.training.core import EpisodeReplay

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
        script = """
import builtins
import runpy
import sys
real_import = builtins.__import__
def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("blocked for CLI help test", name="torch")
    return real_import(name, *args, **kwargs)
builtins.__import__ = import_without_torch
sys.argv = ["src.cli.curriculum", "--help"]
runpy.run_module("src.cli.curriculum", run_name="__main__")
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        text = completed.stdout
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
            "--recycle-discard",
            "--no-recycle-discard",
            "--seed",
        ):
            self.assertIn(flag, text)

    def test_source_arguments_are_mutually_exclusive_and_optional(self):
        parser = build_arg_parser()
        self.assertIsNone(
            parser.parse_args(["--output-dir", "out"]).initial_model
        )
        with mock.patch("sys.stderr", new_callable=io.StringIO):
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
        self.assertTrue(config.recycle_discard)
        self.assertTrue(config.to_dict()["recycle_discard"])
        self.assertEqual(config.architecture, "suit_conv")
        self.assertEqual(config.curriculum_distances, (1, 2, 4, 8))
        self.assertEqual(config.replay_capacity, 10_000)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.updates_per_episode, 4)

    def test_legacy_config_shape_restores_historical_rule(self):
        current = TrainingConfig().to_dict()
        self.assertTrue(TrainingConfig.from_dict(current).recycle_discard)
        legacy = dict(current)
        del legacy["recycle_discard"]
        restored = TrainingConfig.from_dict(legacy)
        self.assertFalse(restored.recycle_discard)
        self.assertFalse(restored.game_config.recycle_discard)
        self.assertFalse(restored.to_dict()["recycle_discard"])

        missing_other_field = dict(legacy)
        del missing_other_field["max_turns"]
        with self.assertRaisesRegex(ValueError, "fields"):
            TrainingConfig.from_dict(missing_other_field)
        for value in (0, 1, None, "true"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                TrainingConfig(recycle_discard=value)

    def test_cli_rule_override_is_explicit(self):
        parser = build_arg_parser()
        base = ["--initial-model", "model.pt", "--output-dir", "out"]
        self.assertIsNone(parser.parse_args(base).recycle_discard)
        self.assertTrue(
            parser.parse_args(base + ["--recycle-discard"]).recycle_discard
        )
        self.assertFalse(
            parser.parse_args(base + ["--no-recycle-discard"]).recycle_discard
        )


class TestEpisodeReplayState(unittest.TestCase):
    @staticmethod
    def transition(index, action, target):
        state = (float(index),) + (0.0,) * (STATE_DIM - 1)
        return (state, action, target)

    def test_roundtrip_preserves_episode_sampling(self):
        replay = EpisodeReplay(capacity=8)
        replay.append(
            (
                self.transition(0, 1, 0.5),
                self.transition(1, 2, 1.0),
            )
        )
        replay.append((self.transition(2, 3, 0.25),))
        restored = EpisodeReplay(capacity=8)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(
            replay.sample(random.Random(91), 50),
            restored.sample(random.Random(91), 50),
        )
        self.assertEqual(replay.state_dict(), restored.state_dict())

    def test_load_validation_is_atomic(self):
        replay = EpisodeReplay(capacity=4)
        replay.append((self.transition(0, 1, 0.5),))
        before = replay.state_dict()
        short = (0.0,)
        wrong_length = (0.0, 1.0)
        nonfinite = (float("nan"),) + (0.0,) * (STATE_DIM - 1)
        valid = (0.0,) * STATE_DIM
        bad_states = (
            {**before, "version": 999},
            {**before, "capacity": 5},
            {**before, "episodes": (((short, 1, 0.0),),)},
            {**before, "episodes": (((wrong_length, 1, 0.0),),)},
            {**before, "episodes": (((nonfinite, 1, 0.0),),)},
            {**before, "episodes": (((valid, True, 0.0),),)},
            {**before, "episodes": (((valid, NUM_ACTIONS, 0.0),),)},
            {**before, "episodes": (((valid, 999, 0.0),),)},
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
        from src.model.network import QNetwork, save_checkpoint

        config = self.small_config()
        with tempfile.TemporaryDirectory() as tmp:
            compatible = os.path.join(tmp, "compatible.pt")
            save_checkpoint(
                compatible,
                QNetwork(architecture="suit_conv"),
                GameConfig(max_turns=30, recycle_discard=False),
            )
            session = TrainingSession(config, seed=41, initial_model=compatible)
            self.assertEqual(session.config.max_turns, 1)
            self.assertTrue(session.config.recycle_discard)
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
        from src.model.network import select_action

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

    def test_full21_episode_updates_q_toward_monte_carlo_target(self):
        from src.evaluate import hand_reward

        class Full21FixtureEnv(PappluEnv):
            def reset(self, seed=None):
                del seed
                observation = self.reset_warm_start(seed=2026)
                winning_discard = next(
                    face
                    for face, count in enumerate(observation.hand)
                    if count
                    and hand_reward(
                        [
                            value - int(index == face)
                            for index, value in enumerate(observation.hand)
                        ],
                        observation.joker,
                        required_sequences=5,
                        cards_in_hand=21,
                    )
                    == 1.0
                )
                self._hand[winning_discard] -= 1
                self._discard = [self._stock.pop(0)]
                self._stock.append(winning_discard)
                self._phase = Phase.DRAW
                self._turns_remaining = self.config.max_turns
                self._last_reward = 0.0
                self._won = False
                self._warm_start = False
                return self.observe()

        config = self.small_config(
            max_turns=2,
            curriculum_fraction=0.0,
            batch_size=2,
        )
        session = TrainingSession(config, seed=41)
        before_parameters = [
            parameter.detach().clone()
            for parameter in session.network.parameters()
        ]
        captured = {}

        def winning_action(network, observation, epsilon, rng, device):
            del epsilon, rng, device
            if observation.phase is Phase.DRAW:
                return ACTION_DRAW_STOCK
            action = next(
                discard_action(face)
                for face, count in enumerate(observation.hand)
                if count
                and hand_reward(
                    [
                        value - int(index == face)
                        for index, value in enumerate(observation.hand)
                    ],
                    observation.joker,
                    required_sequences=5,
                    cards_in_hand=21,
                )
                == 1.0
            )
            state = encode_observation(observation)
            with torch.no_grad():
                value = network(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                )[0, action]
            captured.update(state=state, action=action, before=float(value))
            return action

        def final_transition(_rng, _batch_size):
            episode = session.replay.state_dict()["episodes"][0]
            return [episode[-1]]

        with mock.patch(
            "src.training.curriculum.PappluEnv",
            Full21FixtureEnv,
        ), mock.patch(
            "src.training.curriculum.select_action",
            side_effect=winning_action,
        ), mock.patch.object(
            session.replay,
            "sample",
            side_effect=final_transition,
        ):
            result = session.run_episode()

        with torch.no_grad():
            after = session.network(
                torch.tensor(
                    captured["state"],
                    dtype=torch.float32,
                ).unsqueeze(0)
            )[0, captured["action"]]
        transitions = session.replay.state_dict()["episodes"][0]
        self.assertEqual(result["source"], "ordinary_full21")
        self.assertTrue(result["won"])
        self.assertEqual(transitions[-1][2], 1.0)
        self.assertLess(
            abs(float(after) - 1.0),
            abs(captured["before"] - 1.0),
        )
        self.assertTrue(session.optimizer.state_dict()["state"])
        self.assertTrue(
            any(
                not torch.equal(before, after_parameter)
                for before, after_parameter in zip(
                    before_parameters,
                    session.network.parameters(),
                )
            )
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

    def test_resume_rejects_nonfinite_model_states(self):
        session = TrainingSession(self.small_config(), seed=41)
        for field in (
            "network_state_dict",
            "baseline_model_state_dict",
            "best_model_state_dict",
        ):
            payload = copy.deepcopy(session.state_dict())
            tensor = next(iter(payload[field].values()))
            tensor.reshape(-1)[0] = float("nan")
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError,
                field,
            ):
                TrainingSession.from_state_dict(payload)

    def test_failed_resume_preserves_torch_rng_and_rejects_invalid_replay(self):
        session = TrainingSession(self.small_config(), seed=41)
        cases = (
            ((0.0,), 0, "must contain 164 values"),
            ((0.0,) * STATE_DIM, NUM_ACTIONS, "must be in 0..53"),
        )
        for state, action, message in cases:
            payload = copy.deepcopy(session.state_dict())
            payload["replay_state_dict"]["episodes"] = (
                ((state, action, 0.0),),
            )
            torch.manual_seed(8128)
            before = torch.get_rng_state().clone()
            with self.subTest(action=action), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                TrainingSession.from_state_dict(payload)
            self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_legacy_state_shape_roundtrips_as_nonrecycling(self):
        config = self.small_config(recycle_discard=False)
        session = TrainingSession(config, seed=41)
        payload = session.state_dict()
        del payload["training_config"]["recycle_discard"]
        del payload["source_trace"]["rules"]["recycle_discard"]

        restored = TrainingSession.from_state_dict(
            payload,
            expected_config=config,
        )
        self.assertFalse(restored.config.recycle_discard)
        rewritten = restored.state_dict()
        self.assertFalse(rewritten["training_config"]["recycle_discard"])
        self.assertFalse(
            rewritten["source_trace"]["rules"]["recycle_discard"]
        )

    def test_new_state_metadata_names_recycling_rule(self):
        session = TrainingSession(self.small_config(), seed=41)
        payload = session.state_dict()
        self.assertTrue(payload["training_config"]["recycle_discard"])
        self.assertTrue(
            payload["source_trace"]["rules"]["recycle_discard"]
        )

    def test_promotion_needs_two_heldout_passes(self):
        config = self.small_config()
        session = TrainingSession(config, seed=41)
        session.stage_episodes = config.minimum_stage_episodes
        session.epsilon_progress = session.stage_episodes
        session.best_score = 0.0

        full = validation(0.2, 0.3)
        curriculum = validation(0.4, 0.1)
        with mock.patch(
            "src.training.curriculum.evaluate_policy_pair",
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
            "src.training.curriculum.evaluate_policy_pair",
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
        final_step_config = self.small_config(max_turns=1)
        final_step = TrainingSession(final_step_config, seed=41)
        expected = before_action.state_dict()
        final_expected = final_step.state_dict()

        with mock.patch("src.training.curriculum.time.monotonic", return_value=2.0):
            self.assertIsNone(before_action.run_episode(deadline=1.0))
        with mock.patch(
            "src.training.curriculum.time.monotonic",
            side_effect=[0.0, 2.0],
        ):
            self.assertIsNone(mid_trajectory.run_episode(deadline=1.0))
        with mock.patch(
            "src.training.curriculum.time.monotonic",
            side_effect=[0.0, 0.0, 2.0],
        ):
            self.assertIsNone(final_step.run_episode(deadline=1.0))

        for session, initial, expected_state in (
            (
                before_action,
                TrainingSession(config, seed=41),
                expected,
            ),
            (
                mid_trajectory,
                TrainingSession(config, seed=41),
                expected,
            ),
            (
                final_step,
                TrainingSession(final_step_config, seed=41),
                final_expected,
            ),
        ):
            self.assertEqual(session.total_episodes, 0)
            self.assertEqual(len(session.replay), 0)
            self.assertEqual(session.optimizer.state_dict()["state"], {})
            self.assert_model_equal(initial.network, session.network)
            current = session.state_dict()
            self.assertEqual(
                current["counters"],
                expected_state["counters"],
            )
            for name in ("deal", "action", "replay"):
                self.assertEqual(
                    current["rng_states"][name],
                    expected_state["rng_states"][name],
                )
            self.assertTrue(
                torch.equal(
                    current["rng_states"]["torch"],
                    expected_state["rng_states"]["torch"],
                )
            )

    def test_resume_completes_validation_pending_at_checkpoint_boundary(self):
        config = self.small_config(validate_every=2)
        session = TrainingSession(config, seed=41)
        with mock.patch(
            "src.training.curriculum.evaluate_policy_pair",
            return_value=validation(0.1, 0.0),
        ):
            session.initialize_validation()
        session.run_episode()
        session.run_episode()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_session_checkpoint(session, output)
            resumed = TrainingSession.load(str(output / "latest-state.pt"))
            self.assertEqual(resumed.last_validation_episode, 0)
            with mock.patch(
                "src.training.curriculum.evaluate_policy_pair",
                return_value=validation(0.2, 0.0),
            ):
                with mock.patch.object(
                    resumed,
                    "run_episode",
                    side_effect=AssertionError("unexpected new episode"),
                ):
                    reason = run(resumed, output, 2, 60, 1)
            self.assertEqual(reason, "max_episodes")
            self.assertEqual(resumed.last_validation_episode, 2)
            self.assertEqual(resumed.total_episodes, 2)
            self.assertEqual(resumed.best_score, 0.2)

    def test_validation_budget_does_not_publish_partial_scores(self):
        from src.training.curriculum import ValidationInterrupted

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
        from src.model.network import load_checkpoint

        session = TrainingSession(self.small_config(), seed=41)
        session.total_episodes = 6
        session.total_updates = 6
        session.random_episodes = 6
        session.stage_episodes = 6
        session.epsilon_progress = 6
        session.best_episode = 2
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            real_replace = os.replace
            replacements = []

            def capture_replace(source, destination):
                replacements.append((Path(source), Path(destination)))
                real_replace(source, destination)

            with mock.patch(
                "src.checkpoints.tensor.os.replace",
                side_effect=capture_replace,
            ):
                manifest = save_session_checkpoint(session, output)

            expected = {
                "latest-state.pt",
                "latest-model.pt",
                "best-model.pt",
                "baseline-model.pt",
            }
            self.assertEqual(
                {path.name for path in output.iterdir()},
                expected | {"metrics.jsonl"},
            )
            self.assertEqual(
                {destination.name for _source, destination in replacements},
                expected,
            )
            self.assertTrue(
                all(source.name.endswith(".tmp") for source, _ in replacements)
            )
            self.assertEqual(
                set(manifest),
                {
                    "latest-state.pt",
                    "latest-model.pt",
                    "best-model.pt",
                    "baseline-model.pt",
                    "metrics.jsonl",
                },
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
            expected_episodes = {
                "latest-model.pt": 6,
                "best-model.pt": 2,
                "baseline-model.pt": 0,
            }
            for name, episodes in expected_episodes.items():
                network, game, meta = load_checkpoint(str(output / name))
                self.assertEqual(network.architecture, "suit_conv")
                self.assertEqual(game, session.config.game_config)
                self.assertEqual(meta["role"], name.removesuffix("-model.pt"))
                self.assertEqual(meta["episodes"], episodes)
                self.assertEqual(meta["checkpoint_episodes"], episodes)

    def test_live_metric_append_preserves_the_published_checkpoint_hash(self):
        from src.training.curriculum import _append_metric

        session = TrainingSession(self.small_config(), seed=41)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = save_session_checkpoint(session, output)
            published = (output / "metrics.jsonl").read_bytes()
            _append_metric(output, {"event": "new-episode", "episode": 1})
            self.assertEqual((output / "metrics.jsonl").read_bytes(), published)
            self.assertEqual(
                hashlib.sha256(published).hexdigest(),
                manifest["metrics.jsonl"]["sha256"],
            )
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
