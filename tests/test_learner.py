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
    from src.training.train import Learner, TrainConfig, TrajectoryBatch, _SeatTrajectory, regularized_reward, vtrace


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

    def test_regularized_rewards_are_fp32_and_detached(self):
        online = torch.tensor([[-2.0, -0.5]], dtype=torch.bfloat16, requires_grad=True)
        anchor = torch.tensor([[-1.0, -1.0]], dtype=torch.bfloat16, requires_grad=True)
        result = regularized_reward(torch.tensor([[0.0, 1.0]]), online, anchor, 0.2)
        torch.testing.assert_close(result, torch.tensor([[0.2, 0.9]]))
        self.assertFalse(result.requires_grad)

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

    def test_bf16_updates_fp32_weights_and_refreshes_the_frozen_anchor(self):
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        learner = Learner(model, TrainConfig(anchor_every=2, learning_rate=0.01))
        first, second = (
            _SeatTrajectory(
                torch.ones(n, SCHEMA.fields, dtype=torch.uint8),
                torch.ones(n, SCHEMA.suffix_tokens, SCHEMA.fields, dtype=torch.uint8),
                torch.arange(1, n + 1), torch.full((n,), 12), torch.full((n,), 2),
                torch.full((n,), -math.log(2)), torch.tensor(1.0),
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
