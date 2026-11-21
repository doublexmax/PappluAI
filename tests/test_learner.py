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
    from src.model.network import QNetwork, collate, greedy_actions
    from src.training.train import Learner, Replay, Step, TrainConfig, Trajectory


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestTensorReplay(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.feature = encode(PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17))

    def trajectory(self, actions, utility):
        return Trajectory(tuple(Step(self.feature, action) for action in actions), utility)

    def test_ring_retains_latest_steps_and_precomputed_discounted_returns(self):
        replay = Replay(5, gamma=0.5)
        self.assertEqual(replay.append(self.trajectory((0, 1, 0, 1), -0.5)), 4)
        self.assertEqual(replay.returns[:4].tolist(), [-0.5, -0.5, -0.5, -0.5])
        replay.append(self.trajectory((1, 0, 1), 1.0))
        self.assertEqual((len(replay), replay.cursor), (5, 2))
        self.assertEqual(replay.actions.tolist(), [0, 1, 0, 1, 1])
        self.assertEqual(replay.returns.tolist(), [0.5, 1.0, -0.5, -0.5, 0.25])
        replay.append(self.trajectory((0,) * 8, 1.0))
        self.assertEqual((len(replay), replay.cursor), (5, 0))
        self.assertEqual(replay.returns.tolist(), [0.0625, 0.125, 0.25, 0.5, 1.0])
        self.assertEqual(replay.features.shape, (5, FEATURE_SIZE))
        self.assertEqual(replay.features.dtype, torch.float32)
        self.assertEqual(replay.actions.dtype, torch.int64)
        self.assertEqual(replay.returns.dtype, torch.float32)
        self.assertTrue(replay.features.is_contiguous())
        self.assertEqual(replay.features[0].tolist(), list(feature_values(self.feature)))

    def test_only_positive_returns_are_discounted(self):
        for utility, expected in (
            (1.0, [0.125, 0.25, 0.5, 1.0]),
            (0.5, [0.0625, 0.125, 0.25, 0.5]),
            (-0.5, [-0.5, -0.5, -0.5, -0.5]),
            (0.0, [0.0, 0.0, 0.0, 0.0]),
        ):
            with self.subTest(utility=utility):
                replay = Replay(4, gamma=0.5)
                replay.append(self.trajectory((0, 1, 0, 1), utility))
                self.assertEqual(replay.returns.tolist(), expected)

    def test_timeout_penalties_are_constant_for_every_step_at_any_length(self):
        for length in (1, 4, 120):
            for utility in (-1.0, -0.22, 0.0):
                with self.subTest(length=length, utility=utility):
                    replay = Replay(length, gamma=0.99)
                    replay.append(self.trajectory((0,) * length, utility))
                    torch.testing.assert_close(
                        replay.returns, torch.full((length,), utility, dtype=torch.float32),
                        rtol=0, atol=0,
                    )

    def test_compressed_raw_bytes_roundtrip_empty_partial_and_wrapped_replays(self):
        for count in (0, 2, 8):
            with self.subTest(count=count):
                replay = Replay(5)
                replay.append(self.trajectory((0, 1) * count, -0.75))
                state = replay.state_dict()
                self.assertTrue(all(isinstance(state[name], bytes) for name in ("features", "actions", "returns")))
                restored = Replay(5)
                restored.load_state_dict(state)
                self.assertEqual((len(restored), restored.cursor), (len(replay), replay.cursor))
                for name in ("features", "actions", "returns"):
                    torch.testing.assert_close(getattr(restored, name)[:len(replay)], getattr(replay, name)[:len(replay)], rtol=0, atol=0)
                self.assertEqual(restored.state_dict(), state)

    def test_mismatched_and_corrupt_replay_states_are_rejected(self):
        replay = Replay(5)
        replay.append(self.trajectory((0, 1), 1.0))
        state = replay.state_dict()
        for change in ({"capacity": 4}, {"gamma": 0.5}, {"size": 6}, {"cursor": -1},
                       {"features": b"not compressed"}, {"size": 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Replay(5).load_state_dict({**state, **change})


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
        self.trajectories = [Trajectory((Step(feature, 0),), 0.75) for feature in self.features]

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

    def test_monte_carlo_loss_changes_only_the_taken_action_for_every_utility(self):
        class ActionValues(torch.nn.Module):
            def __init__(self, target):
                super().__init__()
                self.q = torch.nn.Parameter(torch.tensor([target + 0.5, target + 10] + [100.0] * 52))

            def forward(self, batch):
                return self.q.expand(batch.legal.shape[0], -1)

        for utility in (1.0, -1.0, 0.0):
            with self.subTest(utility=utility):
                network = ActionValues(utility)
                before = network.q.detach().clone()
                learner = Learner(network, self.config)
                stats = learner.learn([Trajectory((Step(self.features[0], 0),), utility)])
                self.assertAlmostEqual(stats.loss, 0.125, places=7)
                self.assertLess(network.q[0].item(), before[0].item())
                torch.testing.assert_close(network.q[1:], before[1:], rtol=0, atol=0)

    def test_a_winning_exploratory_action_does_not_rerank_greedy_alternatives(self):
        class ActionValues(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Parameter(torch.tensor([100.0] * 2 + [1.0, 1.2, 1.2, 1.2] + [100.0] * 48))

            def forward(self, batch):
                return self.q.expand(batch.legal.shape[0], -1)

        env = PappluEnv(self.config.game)
        env.reset(17)
        obs = replace(env.step(0), hand=(1, 1, 1, 1) + (0,) * 48, joker=7)
        feature = encode(obs)
        config = replace(self.config, batch_size=1, learning_rate=0.02, lr_decay_updates=0)
        network = ActionValues()
        before = network.q.detach().clone()
        learner = Learner(network, config)
        stats = learner.learn([Trajectory((Step(feature, 2),), 1.0)], updates=32)
        self.assertEqual(stats.loss, 0.0)
        torch.testing.assert_close(network.q, before, rtol=0, atol=0)
        self.assertEqual(greedy_actions(network, collate([feature], torch.device("cpu"))).tolist(), [3])

    def test_learning_rate_decays_before_updates_and_has_a_fixed_floor(self):
        learner = Learner(self.network, replace(self.config, lr_decay_updates=1))
        learner.learn(self.trajectories, updates=0)
        for update, expected in ((0, 1e-3), (1, 1e-3 / math.sqrt(2)), (10_000, 1.5e-4)):
            learner.update_count = update
            learner.update(1)
            self.assertEqual(learner.update_count, update + 1)
            for group in learner.optimizer.param_groups:
                self.assertAlmostEqual(group["lr"], expected)
        for disabled in (0, -1):
            learner = Learner(QNetwork(ModelSpec(16, 1)), replace(self.config, lr_decay_updates=disabled))
            learner.update_count = 10_000
            learner.learn(self.trajectories)
            self.assertEqual(learner.optimizer.param_groups[0]["lr"], 1e-3)

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
