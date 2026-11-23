import copy
import math
import unittest

from src.model.encoding import SCHEMA
from src.model.spec import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.model.network import GameTransformer
    from src.training.train import Learner, TrainConfig, TrajectoryBatch, _SeatTrajectory, vtrace


def _expert(n, action=2, mass=65535):
    row = torch.zeros(n, SCHEMA.num_actions, dtype=torch.int32)
    row[:, action] = mass
    if mass != 65535:
        row[:, 2] = 65535 - mass
    return row.to(torch.uint16)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestLearner(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(86)

    def test_vtrace_matches_clipped_recurrence_and_terminal_discount(self):
        args = (
            torch.tensor([[math.log(0.5), 1000.0]]), torch.tensor([[0.0, 1.0]]),
            torch.tensor([[1.0, 0.0]]), torch.tensor([[0.2, 0.4]]), torch.tensor([98.0]),
        )
        values, advantages = vtrace(*args, rho_bar=1.0, c_bar=0.75)
        torch.testing.assert_close(values, torch.tensor([[0.6, 1.0]]))
        torch.testing.assert_close(advantages, torch.tensor([[0.4, 0.6]]))
        values, _ = vtrace(
            torch.zeros((1, 1)), torch.tensor([[0.3]]), torch.ones((1, 1)),
            torch.tensor([[0.1]]), torch.tensor([0.8]), rho_bar=1, c_bar=1,
        )
        torch.testing.assert_close(values, torch.tensor([[1.1]]))

    def test_distillation_fits_search_distribution_instead_of_the_sampled_move(self):
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        learner = Learner(model, TrainConfig(precision="fp32", grad_clip=1000))
        row = _SeatTrajectory(
            torch.ones(1, SCHEMA.fields, dtype=torch.uint8),
            torch.ones(1, SCHEMA.suffix_tokens, SCHEMA.fields, dtype=torch.uint8),
            torch.tensor([1]), torch.tensor([12]), torch.tensor([2]),
            torch.tensor([-math.log(2)]), torch.tensor(1.0), _expert(1, 3, 49151),
            torch.ones(1, dtype=torch.uint8),
            torch.zeros(1), torch.zeros(1),
        )
        stats = learner.update(TrajectoryBatch.pack((row,)))
        self.assertAlmostEqual(stats.policy_loss, math.log(2), places=6)
        self.assertAlmostEqual(stats.value_loss, 0.5)
        self.assertAlmostEqual(stats.loss, math.log(2) + 0.5, places=6)
        torch.testing.assert_close(model.policy.bias.grad[2:4], torch.tensor([0.25, -0.25]), atol=1e-5, rtol=0)
        self.assertGreater(float(model.policy.bias[3]), float(model.policy.bias[2]))
        self.assertGreater(float(model.value.bias), 0.0)

    def test_unforced_action_gets_credit_for_a_later_forced_win(self):
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        learner = Learner(model, TrainConfig(precision="fp32", entropy_weight=0.002, grad_clip=1000))
        targets = torch.zeros(2, SCHEMA.num_actions, dtype=torch.int32)
        targets[0, 3] = targets[1, 5] = 65535
        row = _SeatTrajectory(
            torch.ones(2, SCHEMA.fields, dtype=torch.uint8),
            torch.ones(2, SCHEMA.suffix_tokens, SCHEMA.fields, dtype=torch.uint8),
            torch.tensor([1, 2]), torch.tensor([12, 48]), torch.tensor([2, 5]),
            torch.tensor([-math.log(2), 0]), torch.tensor(1.0), targets.to(torch.uint16),
            torch.tensor([0, 1], dtype=torch.uint8),
            torch.zeros(2), torch.zeros(2),
        )
        stats = learner.update(TrajectoryBatch.pack((row,)))
        credit = 0.995
        self.assertAlmostEqual(stats.policy_loss, (credit + 1) * math.log(2) / 2, places=6)
        self.assertAlmostEqual(stats.loss, stats.policy_loss + stats.value_loss - 0.002 * math.log(2), places=6)
        torch.testing.assert_close(
            model.policy.bias.grad[2:6], torch.tensor([-credit, credit, 1.0, -1.0]) / 4,
        )

    def test_on_policy_terminal_reward_reaches_the_first_decision(self):
        values, _ = vtrace(
            torch.zeros((2, 4)), torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, -1.0, 0.0, 0.0]]),
            torch.tensor([[0.995, 0.995, 0.995, 0.0], [0.995, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.2, 0.4, -0.8, 0.9], [0.3, -0.2, 17.0, -32.0]]),
            torch.zeros(2), rho_bar=1, c_bar=1,
        )
        torch.testing.assert_close(values, torch.tensor([
            [0.995**3, 0.995**2, 0.995, 1.0], [-0.995, -1.0, 0.0, 0.0],
        ]))

    def test_shaped_rewards_telescope_with_ragged_terminal_steps(self):
        seats = tuple(
            _SeatTrajectory(
                torch.ones(n, SCHEMA.fields, dtype=torch.uint8),
                torch.ones(n, SCHEMA.suffix_tokens, SCHEMA.fields, dtype=torch.uint8),
                torch.arange(1, n + 1), torch.full((n,), 12), torch.full((n,), 2),
                torch.full((n,), -math.log(2)), torch.tensor(utility), _expert(n),
                torch.zeros(n, dtype=torch.uint8), torch.tensor(phi), torch.zeros(n),
            )
            for n, utility, phi in ((1, 1.0, [0.75]), (4, -0.4, [0.1, 0.3, 0.2, 0.7]))
        )
        batch = TrajectoryBatch.pack(seats)
        torch.testing.assert_close(batch.reward, torch.tensor([
            [0.25, 0, 0, 0], [0.1985, -0.101, 0.4965, -1.1],
        ]))
        torch.testing.assert_close(
            (batch.reward * 0.995 ** torch.arange(4)).sum(1),
            torch.tensor([0.25, -0.4 * 0.995**3 - 0.1]),
        )

    def test_subsample_keeps_terminal_and_weights_the_other_decisions(self):
        def seat(n):
            return _SeatTrajectory(
                torch.ones(n, SCHEMA.fields, dtype=torch.uint8),
                torch.ones(n, SCHEMA.suffix_tokens, SCHEMA.fields, dtype=torch.uint8),
                torch.arange(1, n + 1), torch.full((n,), 12), torch.full((n,), 2),
                torch.zeros(n), torch.tensor(1.0), _expert(n), torch.zeros(n, dtype=torch.uint8),
                torch.zeros(n), torch.zeros(n),
            )

        batch = TrajectoryBatch.pack((seat(2), seat(4)))
        values = torch.tensor([[1.0, 3.0, 0.0, 0.0], [1.0, 2.0, 3.0, 40.0]])
        observed = {}
        rng = torch.Generator().manual_seed(983)
        for _ in range(32):
            sample = batch.sample_steps(2, rng)
            self.assertEqual(sample.index[:, 1].tolist(), [1, 3])
            self.assertEqual(sample.weight.tolist(), [[0.5, 0.5], [0.75, 0.25]])
            estimate = (sample.gather(values) * sample.weight).sum(1)
            self.assertEqual(float(estimate[0]), 2.0)
            observed[int(sample.index[1, 0])] = float(estimate[1])
        self.assertEqual(observed, {0: 10.75, 1: 11.5, 2: 12.25})
        self.assertEqual(sum(observed.values()) / 3, float(values[1].mean()))

    def test_mixed_length_update_weights_each_complete_trajectory_equally(self):
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        config = TrainConfig(precision="fp32", grad_clip=1000)
        seats = tuple(
            _SeatTrajectory(
                torch.ones(n, SCHEMA.fields, dtype=torch.uint8),
                torch.ones(n, SCHEMA.suffix_tokens, SCHEMA.fields, dtype=torch.uint8),
                torch.arange(1, n + 1), torch.full((n,), legal), torch.full((n,), 2),
                torch.full((n,), -math.log(2)), torch.tensor(utility), _expert(n),
                torch.zeros(n, dtype=torch.uint8),
                torch.zeros(n), torch.zeros(n),
            )
            for n, legal, utility in ((1, 4, 1.0), (8, 12, -1.0))
        )

        def update(rows):
            learner = Learner(copy.deepcopy(model), config)
            with torch.no_grad():
                learner.anchor.policy.bias[2] += 1
            stats = learner.update(TrajectoryBatch.pack(rows))
            return stats, learner.online.policy.bias.grad, learner.online.value.bias.grad

        mixed, mixed_policy_grad, mixed_value_grad = update(seats)
        short, short_policy_grad, short_value_grad = update(seats[:1])
        long, long_policy_grad, long_value_grad = update(seats[1:])
        for name in ("policy_loss", "value_loss", "kl_anchor", "entropy"):
            with self.subTest(component=name):
                self.assertAlmostEqual(getattr(mixed, name), (getattr(short, name) + getattr(long, name)) / 2, places=6)
        torch.testing.assert_close(mixed_policy_grad, (short_policy_grad + long_policy_grad) / 2)
        torch.testing.assert_close(mixed_value_grad, (short_value_grad + long_value_grad) / 2)

    def test_bf16_updates_fp32_weights_and_refreshes_the_frozen_anchor(self):
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        learner = Learner(model, TrainConfig(anchor_every=2, learning_rate=0.01))
        first, second = (
            _SeatTrajectory(
                torch.ones(n, SCHEMA.fields, dtype=torch.uint8),
                torch.ones(n, SCHEMA.suffix_tokens, SCHEMA.fields, dtype=torch.uint8),
                torch.arange(1, n + 1), torch.full((n,), 12), torch.full((n,), 2),
                torch.full((n,), -math.log(2)), torch.tensor(1.0), _expert(n),
                torch.ones(n, dtype=torch.uint8),
                torch.zeros(n), torch.zeros(n),
            )
            for n in (1, 2)
        )
        batch = TrajectoryBatch.pack((first, first, second))
        initial = learner.anchor.policy.weight.clone()
        stats = learner.update(batch)
        self.assertAlmostEqual(stats.kl_anchor, 0.0)
        self.assertTrue(math.isfinite(stats.loss))
        self.assertFalse(torch.equal(learner.online.policy.weight, initial))
        torch.testing.assert_close(learner.anchor.policy.weight, initial)
        self.assertGreater(learner.update(batch).kl_anchor, 0)
        self.assertEqual(learner.version, 2)
        torch.testing.assert_close(learner.online.policy.weight, learner.anchor.policy.weight)
        self.assertAlmostEqual(learner.update(batch).kl_anchor, 0)
        self.assertTrue(all(p.dtype == torch.float32 and not p.requires_grad for p in learner.anchor.parameters()))
