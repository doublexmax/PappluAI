from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from src.model.spec import ModelSpec, ModelSpecError, TokenSchema

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.model.network import GameTransformer, TokenBatch, load_checkpoint, save_checkpoint


class TestSpec(unittest.TestCase):
    def test_rejects_bad_rope_shape_and_incompatible_schema_fields(self):
        for config in ({"width": 30, "heads": 4}, {"layers": True}, {"ffn": 0}):
            with self.subTest(config=config), self.assertRaises(ModelSpecError):
                ModelSpec(**config)
        with self.assertRaises(ModelSpecError):
            TokenSchema("other", 1, 2, 4, 64)
        with self.assertRaises(ModelSpecError):
            ModelSpec.from_dict({"width": 16, "blocks": 1})


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestGameTransformer(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(55)
        self.schema = TokenSchema("not-papplu", 1, 3, 5, 7)
        self.model = GameTransformer(ModelSpec(width=32, layers=2, heads=4, ffn=64), self.schema)

    def test_batched_unroll_matches_step_across_ragged_chunks_and_cache_reuse(self):
        model = self.model
        prefix = torch.randint(1, 256, (2, 41, 3), dtype=torch.uint8)
        prefix[0, 3:6] = 0
        prefix[1, 30:] = 0
        suffix = torch.randint(1, 256, (2, 4, 5, 3), dtype=torch.uint8)
        suffix[0, :, -2:] = 0
        suffix[1, :, -1:] = 0
        cuts = torch.tensor([[8, 25, 25, 41], [3, 3, 19, 30]])
        legal = torch.tensor([True, False, True, False, False, True, True]).expand(2, 4, 7)
        batch = TokenBatch(prefix, suffix, cuts, legal)
        reference = model.unroll(batch)
        rows = torch.tensor([1, 0])
        cache = model.new_cache()
        starts = [0, 0]
        for t, ends in enumerate(cuts.T.tolist()):
            lengths = torch.tensor([end - start for start, end in zip(starts, ends)])
            added = torch.zeros(2, int(lengths.max()), 3, dtype=torch.uint8)
            for i, (start, end) in enumerate(zip(starts, ends)):
                added[i, :end - start] = prefix[i, start:end]
            pred = model.step(cache, rows, added, suffix[:, t], legal[:, t], append_lengths=lengths)
            torch.testing.assert_close(pred.log_policy, reference.log_policy[:, t], atol=1e-5, rtol=0)
            torch.testing.assert_close(pred.value, reference.value[:, t], atol=1e-5, rtol=0)
            starts = ends
        cache.reset(1)
        replay = model.step(cache, rows[:1], prefix[:1, :8], suffix[:1, 0], legal[:1, 0])
        torch.testing.assert_close(replay.value, reference.value[:1, 0], atol=1e-5, rtol=0)
        self.assertTrue((reference.log_policy.exp()[~legal] == 0).all())
        torch.testing.assert_close(reference.log_policy.exp().sum(-1), torch.ones(2, 4))

    def test_earlier_readout_is_independent_of_future_prefix_and_other_suffix(self):
        prefix = torch.ones((1, 9, 3), dtype=torch.uint8)
        suffix = torch.ones((1, 2, 5, 3), dtype=torch.uint8)
        legal = torch.ones((1, 2, 7), dtype=torch.bool)
        original = self.model.unroll(TokenBatch(prefix, suffix, torch.tensor([[4, 9]]), legal))
        prefix[:, 4:] = 82
        suffix[:, 1] = 42
        changed = self.model.unroll(TokenBatch(prefix, suffix, torch.tensor([[4, 9]]), legal))
        torch.testing.assert_close(original.log_policy[:, 0], changed.log_policy[:, 0], atol=1e-6, rtol=0)
        torch.testing.assert_close(original.value[:, 0], changed.value[:, 0], atol=1e-6, rtol=0)
        self.assertFalse(torch.allclose(original.log_policy[:, 1], changed.log_policy[:, 1]))

    def test_checkpoints_bind_schema_and_validate_finite_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model, {"game": "opaque"}, {"version": 4})
            checkpoint = load_checkpoint(path, self.schema)
            self.assertEqual((checkpoint.game, checkpoint.version), ({"game": "opaque"}, 4))
            torch.testing.assert_close(checkpoint.network.policy.weight, self.model.policy.weight)
            with self.assertRaisesRegex(ValueError, "token schema"):
                load_checkpoint(path, replace(self.schema, version=2))
            payload = torch.load(path, weights_only=True)
            payload["model_state_dict"]["policy.bias"][0] = float("nan")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_checkpoint(path, self.schema)
            torch.save({"checkpoint_version": 5, "architecture": "card_mlp"}, path)
            with self.assertRaisesRegex(ValueError, "train a fresh"):
                load_checkpoint(path, self.schema)
