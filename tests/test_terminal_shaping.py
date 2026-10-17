from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from src.evaluate import minimum_penalty
from src.game.environment import GameConfig, PappluEnv, discard_action
from src.game.multiplayer import MatchConfig, MultiplayerEnv, TerminalSnapshot
from src.training.core import TerminalShaping, terminal_reward

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.model.network import QNetwork, save_checkpoint
    from src.model.search import winning_discard
    from src.training.curriculum import TrainingConfig, TrainingSession
    from src.training.improve import ImproveConfig, ImprovementController
    from src.training.league import LeagueConfig, LeagueSession


class TestTerminalShaping(unittest.TestCase):
    @staticmethod
    def hand(*faces):
        counts = [0] * 52
        for face in faces:
            counts[face] += 1
        return counts

    def reward(self, shaping, *, won=False, hand=None, turns=30):
        return terminal_reward(
            shaping, won=won, turns_remaining=turns, max_turns=60,
            hand=self.hand(0, 1, 4) if hand is None else hand,
            joker=51, required_sequences=1,
        )

    def test_wins_reward_earlier_declarations_without_scoring_deadwood(self):
        with mock.patch("src.training.core.minimum_penalty", side_effect=AssertionError("unused")):
            self.assertEqual(self.reward(TerminalShaping(), won=True), 1.0)
            self.assertEqual(
                self.reward(TerminalShaping(speed=0.35, margin=0.25), won=True), 1.175,
            )
            self.assertEqual(
                self.reward(TerminalShaping(speed=0.35), won=True, turns=0), 1.0,
            )
            self.assertEqual(self.reward(TerminalShaping()), 0.0)

    def test_loss_margin_uses_sequence_gated_deadwood_and_caps_points(self):
        shaping = TerminalShaping(margin=0.25)
        self.assertAlmostEqual(self.reward(shaping), 0.2075)
        self.assertEqual(self.reward(shaping, hand=self.hand(0, 1, 2, 16)), 0.24)
        self.assertEqual(self.reward(replace(shaping, points_cap=10)), 0.0)
        self.assertEqual(self.reward(shaping, hand=self.hand(0, 1, 2)), 0.25)

    def test_validation_rejects_bools_nonfinite_values_and_invalid_ranges(self):
        for field, values in (
            ("speed", (True, -0.1, float("nan"), float("inf"), "0.1")),
            ("margin", (False, -0.1, 1.0, float("nan"), float("inf"))),
            ("points_cap", (True, 0, -1, 1.0, float("inf"))),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    TerminalShaping(**{field: value})
        settings = TerminalShaping(speed=0.35, margin=0.25, points_cap=80)
        restored = TerminalShaping.from_dict(settings.to_dict())
        self.assertEqual(self.reward(restored, won=True), 1.175)
        self.assertAlmostEqual(self.reward(restored), 0.196875)
        for raw in ({}, {"speed": 0.1}, {**settings.to_dict(), "extra": 1}, None):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                TerminalShaping.from_dict(raw)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestShapedTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)

    def test_training_configs_read_old_dicts_and_round_trip_shaped_dicts(self):
        for config in (TrainingConfig(), LeagueConfig()):
            old = config.to_dict()
            self.assertEqual(type(config).from_dict(old), config)
            shaped = replace(config, shaping=TerminalShaping(speed=0.4, margin=0.2, points_cap=80))
            serialized = shaped.to_dict()
            self.assertEqual(serialized["shaping"], {"speed": 0.4, "margin": 0.2, "points_cap": 80})
            self.assertEqual(type(config).from_dict(serialized), shaped)
            self.assertEqual(
                type(config).from_dict({**old, "shaping": TerminalShaping().to_dict()}), config,
            )
            with self.assertRaises(ValueError):
                type(config).from_dict({**serialized, "shaping": {"margin": 0.2}})
            with self.assertRaises(TypeError):
                replace(config, shaping={})

    def test_curriculum_replay_uses_post_discard_win_speed_reward(self):
        class WinningEnv(PappluEnv):
            def reset(self, seed=None):
                return self.reset_warm_start(seed)

        config = TrainingConfig(
            max_turns=3, curriculum_fraction=0.0, batch_size=1, updates_per_episode=1,
            shaping=TerminalShaping(speed=0.3, margin=0.25),
        )
        session = TrainingSession(config, seed=41)

        def winning_action(network, obs, **kwargs):
            return discard_action(winning_discard(
                obs.hand, obs.joker, obs.config.required_sequences, obs.config.cards_in_hand,
            ))

        with (
            mock.patch("src.training.curriculum.PappluEnv", WinningEnv),
            mock.patch("src.training.curriculum.select_action", side_effect=winning_action),
        ):
            result = session.run_episode()
        self.assertTrue(result["won"])
        transitions = session.replay.state_dict()["episodes"][0]["transitions"]
        self.assertEqual([step[-1] for step in transitions], [1.2])
        restored = TrainingSession.from_state_dict(session.state_dict())
        self.assertEqual(restored.config.shaping, config.shaping)
        self.assertEqual(restored.replay.state_dict(), session.replay.state_dict())

    def test_curriculum_loss_targets_use_final_hand_and_discounting(self):
        final = []

        class RecordingEnv(PappluEnv):
            def step(self, action):
                observation = super().step(action)
                if observation.done:
                    final.append(observation)
                return observation

        config = TrainingConfig(
            max_turns=1, curriculum_fraction=0.0, batch_size=1, updates_per_episode=1,
            gamma=0.5, shaping=TerminalShaping(margin=0.25),
        )
        session = TrainingSession(config, seed=41)
        with mock.patch("src.training.curriculum.PappluEnv", RecordingEnv):
            result = session.run_episode()
        self.assertFalse(result["won"])
        observation = final[0]
        points = minimum_penalty(observation.hand, observation.joker, 5).points
        expected = 0.25 * (1 - min(points, 100) / 100)
        transitions = session.replay.state_dict()["episodes"][0]["transitions"]
        self.assertEqual([step[-1] for step in transitions], [0.5 * expected, expected])

    def test_league_uses_terminal_snapshot_and_challenger_turn_budget(self):
        hand = [0] * 52
        for face in (0, 3, 6, 9, 13, 16):
            hand[face] = 3
        for face in (30, 34, 40):
            hand[face] = 1
        self.assertEqual(minimum_penalty(hand, 51, 5).points, 16)
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "model.pt")
            save_checkpoint(path, QNetwork(), GameConfig())
            config = LeagueConfig(
                batch_size=1, updates_per_match=1, player_counts=(2,),
                shaping=TerminalShaping(speed=0.3, margin=0.25),
            )
            for winner, expected in ((0, 1.15), (1, 0.21), (None, 0.21)):
                with self.subTest(winner=winner):
                    session = LeagueSession(path, config=config)
                    env = MultiplayerEnv(MatchConfig(game=config.game, players=2))
                    env.reset(7)
                    obs = env.observe()
                    result = SimpleNamespace(
                        seed=7, winner=winner, seat_turns=(30, 1), terminal_reason="win",
                        trajectories=(((obs, 0, 0.0),), ()),
                        terminal_snapshot=TerminalSnapshot((tuple(hand), obs.hand), 51),
                    )
                    with mock.patch("src.training.league.play_match", return_value=result) as play:
                        session.train_match()
                    self.assertTrue(play.call_args.kwargs["record_terminal"])
                    transition = session.replay.state_dict()["episodes"][0]["transitions"][0]
                    self.assertAlmostEqual(transition[-1], expected)

    def test_improvement_applies_shaping_to_both_learners(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "initial.pt"
            save_checkpoint(str(model), QNetwork(), GameConfig())
            controller = ImprovementController(
                str(model), root / "run", config=ImproveConfig(max_cycles=1),
            )
            for learner in (controller.league, controller.research):
                self.assertEqual(
                    learner.config.shaping, TerminalShaping(speed=0.35, margin=0.25),
                )


if __name__ == "__main__":
    unittest.main()
