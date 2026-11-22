import copy
import math
import unittest
from unittest.mock import patch

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
        )
        stats = learner.update(TrajectoryBatch.pack((row,)))
        self.assertAlmostEqual(stats.policy_loss, math.log(2), places=6)
        self.assertAlmostEqual(stats.value_loss, 0.5)
        self.assertAlmostEqual(stats.loss, math.log(2) + 0.5, places=6)
        torch.testing.assert_close(model.policy.bias.grad[2:4], torch.tensor([0.25, -0.25]), atol=1e-5, rtol=0)
        self.assertGreater(float(model.policy.bias[3]), float(model.policy.bias[2]))
        self.assertGreater(float(model.value.bias), 0.0)

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

    def test_mixed_length_update_weights_each_complete_trajectory_equally(self):
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        config = TrainConfig(precision="fp32", grad_clip=1000)
        seats = tuple(
            _SeatTrajectory(
                torch.ones(n, SCHEMA.fields, dtype=torch.uint8),
                torch.ones(n, SCHEMA.suffix_tokens, SCHEMA.fields, dtype=torch.uint8),
                torch.arange(1, n + 1), torch.full((n,), legal), torch.full((n,), 2),
                torch.full((n,), -math.log(2)), torch.tensor(utility), _expert(n),
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
            )
            for n in (1, 2)
        )
        batch = TrajectoryBatch.pack((first, first, second))
        initial = learner.anchor.policy.weight.clone()
        with patch.object(learner.anchor, "unroll", wraps=learner.anchor.unroll) as anchor_unroll:
            stats = learner.update(batch)
            self.assertAlmostEqual(stats.kl_anchor, 0.0)
            self.assertTrue(math.isfinite(stats.loss))
            self.assertFalse(torch.equal(learner.online.policy.weight, initial))
            torch.testing.assert_close(learner.anchor.policy.weight, initial)
            self.assertEqual(anchor_unroll.call_args.args[0].prefix.size(0), 2)
            stats = learner.update(batch)
            self.assertGreater(stats.kl_anchor, 0)
            self.assertEqual(anchor_unroll.call_count, 1)
            self.assertEqual(learner.version, 2)
            torch.testing.assert_close(learner.online.policy.weight, learner.anchor.policy.weight)
            self.assertAlmostEqual(learner.update(batch).kl_anchor, 0)
            self.assertEqual(anchor_unroll.call_count, 2)
        self.assertTrue(all(p.dtype == torch.float32 and not p.requires_grad for p in learner.anchor.parameters()))
