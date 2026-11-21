import copy
from dataclasses import replace
import math
import unittest

from src.game.environment import GameConfig, PappluEnv
from src.model.encoding import FEATURE_SIZE, encode, feature_values
from src.model.spec import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.model.network import QNetwork, collate
    from src.training.train import Learner, Replay, Step, TrainConfig, Trajectory


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestTensorReplay(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.feature = encode(PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17))

    def trajectory(self, actions, utility):
        return Trajectory(tuple(Step(self.feature, action, index % 2 == 0) for index, action in enumerate(actions)), utility)

    def test_ring_retains_latest_steps_and_discounts_only_positive_returns(self):
        replay = Replay(5, gamma=0.5)
        self.assertEqual(replay.append(self.trajectory((0, 1, 0, 1), -0.5)), 4)
        self.assertEqual(replay.returns[:4].tolist(), [-0.5, -0.5, -0.5, -0.5])
        self.assertEqual(replay.greedy[:4].tolist(), [True, False, True, False])
        replay.append(self.trajectory((1, 0, 1), 1.0))
        self.assertEqual((len(replay), replay.cursor), (5, 2))
        self.assertEqual(replay.actions.tolist(), [0, 1, 0, 1, 1])
        self.assertEqual(replay.returns.tolist(), [0.5, 1.0, -0.5, -0.5, 0.25])
        self.assertEqual(replay.greedy.tolist(), [False, True, True, False, True])
        replay.append(self.trajectory((0,) * 8, 1.0))
        self.assertEqual((len(replay), replay.cursor), (5, 0))
        self.assertEqual(replay.returns.tolist(), [0.0625, 0.125, 0.25, 0.5, 1.0])
        self.assertEqual(replay.greedy.tolist(), [False, True, False, True, False])
        self.assertEqual(replay.features.shape, (5, FEATURE_SIZE))
        self.assertEqual(replay.features.dtype, torch.float32)
        self.assertEqual(replay.actions.dtype, torch.int64)
        self.assertEqual(replay.returns.dtype, torch.float32)
        self.assertEqual(replay.greedy.dtype, torch.bool)
        self.assertTrue(replay.features.is_contiguous())
        self.assertEqual(replay.features[0].tolist(), list(feature_values(self.feature)))

    def test_compressed_raw_bytes_roundtrip_empty_partial_and_wrapped_replays(self):
        for count in (0, 2, 8):
            with self.subTest(count=count):
                replay = Replay(5)
                replay.append(self.trajectory((0, 1) * count, -0.75))
                state = replay.state_dict()
                self.assertTrue(all(isinstance(state[name], bytes) for name in ("features", "actions", "returns", "greedy")))
                restored = Replay(5)
                restored.load_state_dict(state)
                self.assertEqual((len(restored), restored.cursor), (len(replay), replay.cursor))
                for name in ("features", "actions", "returns", "greedy"):
                    torch.testing.assert_close(getattr(restored, name)[:len(replay)], getattr(replay, name)[:len(replay)], rtol=0, atol=0)
                self.assertEqual(restored.state_dict(), state)

    def test_mismatched_and_corrupt_replay_states_are_rejected(self):
        replay = Replay(5)
        replay.append(self.trajectory((0, 1), 1.0))
        state = replay.state_dict()
        for change in ({"capacity": 4}, {"gamma": 0.5}, {"size": 6}, {"cursor": -1},
                       {"features": b"not compressed"}, {"greedy": b"not compressed"}, {"size": 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Replay(5).load_state_dict({**state, **change})

    def test_long_losses_and_zero_returns_never_shrink_with_turn_count(self):
        for utility in (-0.8, 0.0):
            with self.subTest(utility=utility):
                replay = Replay(120, gamma=0.5)
                replay.append(self.trajectory((0,) * 120, utility))
                torch.testing.assert_close(replay.returns, torch.full((120,), utility), rtol=0, atol=0)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestDMCLearner(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.config = TrainConfig(
            player_counts=(1,), cards_in_hand=3, required_sequences=1,
            replay_capacity=128, batch_size=16, learning_rate=1e-3, weight_decay=0,
        )
        self.network = QNetwork(ModelSpec(16, 1))
        self.features = [encode(PappluEnv(self.config.game).reset(seed)) for seed in range(8)]
        self.trajectories = [Trajectory((Step(feature, 0, index % 2 == 0),), 0.75)
                             for index, feature in enumerate(self.features)]

    def test_updates_reduce_selected_action_error_on_real_observations(self):
        learner = Learner(self.network, self.config, seed=19)
        batch = collate(self.features, torch.device("cpu"))
        with torch.no_grad():
            before = ((self.network(batch)[:, 0] - 0.75) ** 2).mean().item()
        stats = learner.learn(self.trajectories, updates=32)
        with torch.no_grad():
            after = ((self.network(batch)[:, 0] - 0.75) ** 2).mean().item()
        self.assertLess(after, before / 10)
        self.assertEqual((stats.updates, stats.samples, learner.step_count, learner.update_count), (32, 256, 8, 32))
        self.assertGreaterEqual(stats.loss, 0)
        self.assertEqual(learner.replay.features[:8].tolist(), [list(feature_values(f)) for f in self.features])

    def test_winning_margin_ranks_taken_actions_above_only_legal_alternatives(self):
        class ActionValues(torch.nn.Module):
            def __init__(self, target):
                super().__init__()
                self.q = torch.nn.Parameter(torch.tensor([target, target - 0.05] + [100.0] * 52))

            def forward(self, batch):
                return self.q.expand(batch.legal.shape[0], -1)

        for utility in (1.0, -1.0, 0.0):
            for greedy in (True, False):
                for weight in (0.0, 0.5):
                    with self.subTest(utility=utility, greedy=greedy, weight=weight):
                        network = ActionValues(utility)
                        before = network.q.detach().clone()
                        learner = Learner(network, replace(self.config, margin_weight=weight))
                        stats = learner.learn([Trajectory((Step(self.features[0], 0, greedy),), utility)])
                        expected = 0.05 * weight if utility > 0 and greedy else 0.0
                        self.assertAlmostEqual(stats.loss, expected, places=7)
                        if expected:
                            self.assertGreater(network.q[0].item(), utility)
                            self.assertLess(network.q[1].item(), before[1].item())
                            torch.testing.assert_close(network.q[2:], before[2:], rtol=0, atol=0)
                        else:
                            torch.testing.assert_close(network.q, before, rtol=0, atol=0)

    def test_inverse_sqrt_learning_rate_uses_completed_updates_and_a_floor(self):
        config = replace(self.config, learning_rate=0.004, lr_decay_updates=2)
        learner = Learner(self.network, config)
        learner.learn(self.trajectories)
        self.assertEqual(learner.optimizer.param_groups[0]["lr"], 0.004)
        learner.update()
        self.assertAlmostEqual(learner.optimizer.param_groups[0]["lr"], 0.004 / math.sqrt(1.5), places=12)
        learner.update()
        self.assertAlmostEqual(learner.optimizer.param_groups[0]["lr"], 0.004 / math.sqrt(2.0), places=12)
        learner.update_count = 100
        learner.update()
        self.assertAlmostEqual(learner.optimizer.param_groups[0]["lr"], 0.0006, places=12)

    def test_zero_decay_keeps_learning_rate_fixed_after_many_updates(self):
        learner = Learner(self.network, replace(self.config, learning_rate=0.004, lr_decay_updates=0))
        learner.update_count = 100_000
        learner.learn(self.trajectories)
        self.assertEqual(learner.optimizer.param_groups[0]["lr"], 0.004)

    def test_state_restores_optimizer_replay_and_sampling_for_exact_continuation(self):
        learner = Learner(self.network, self.config, seed=19)
        learner.learn(self.trajectories, updates=4)
        state = copy.deepcopy(learner.state_dict())
        restored = Learner(QNetwork(ModelSpec(16, 1)), self.config, seed=99)
        restored.load_state_dict(state)
        expected = learner.update(3)
        actual = restored.update(3)
        self.assertEqual(actual, expected)
        self.assertEqual(restored.rng.getstate(), learner.rng.getstate())
        self.assertEqual(restored.replay.state_dict(), learner.replay.state_dict())
        self.assertEqual((restored.step_count, restored.update_count), (8, 7))
        for name, value in learner.online.state_dict().items():
            torch.testing.assert_close(value, restored.online.state_dict()[name], rtol=0, atol=0)

    def test_zero_updates_keeps_samples_without_changing_weights(self):
        learner = Learner(self.network, self.config)
        before = copy.deepcopy(self.network.state_dict())
        stats = learner.learn(self.trajectories, updates=0)
        self.assertEqual((stats.updates, learner.step_count, len(learner.replay)), (0, 8, 8))
        for name, value in self.network.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
