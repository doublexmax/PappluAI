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

    def test_pointer_field_round_trips_and_must_index_a_byte_field(self):
        for field in (None, 0, 2):
            with self.subTest(field=field):
                schema = TokenSchema("other", 1, 3, 5, 7, pointer_field=field)
                self.assertEqual(schema.to_dict()["pointer_field"], field)
                self.assertEqual(TokenSchema.from_dict(schema.to_dict()), schema)
        for field in (-1, 3, True, 1.0):
            with self.subTest(field=field), self.assertRaisesRegex(ModelSpecError, "pointer_field"):
                TokenSchema("other", 1, 3, 5, 7, pointer_field=field)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestGameTransformer(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(55)
        self.schema = TokenSchema("not-papplu", 1, 3, 5, 7)
        self.model = GameTransformer(ModelSpec(width=32, layers=2, heads=4, ffn=64), self.schema)

    def test_unused_fields_have_zero_output_and_zero_gradient(self):
        tokens = torch.tensor([[0, 0, 0], [1, 0, 1], [2, 1, 0]], dtype=torch.uint8)
        with torch.no_grad():
            self.model.embedding.weight[::256].fill_(123)
        embedded = self.model._embed(tokens)
        table = self.model.embedding.weight
        expected = torch.stack((torch.zeros(32), table[1] + table[513], table[2] + table[257]))
        torch.testing.assert_close(embedded, expected, atol=0, rtol=0)
        embedded.sum().backward()
        torch.testing.assert_close(table.grad[::256], torch.zeros(3, 32), atol=0, rtol=0)

    def test_batched_unroll_matches_step_across_ragged_chunks_and_cache_reuse(self):
        model = GameTransformer(self.model.spec, replace(self.schema, pointer_field=2))
        prefix = torch.randint(1, 256, (2, 41, 3), dtype=torch.uint8)
        prefix[0, 3:6] = 0
        prefix[1, 30:] = 0
        suffix = torch.randint(1, 256, (2, 4, 5, 3), dtype=torch.uint8)
        suffix[..., 2] %= self.schema.num_actions + 1
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

    def test_candidate_suffixes_match_unroll_without_changing_the_root_cache(self):
        model = GameTransformer(self.model.spec, replace(self.schema, pointer_field=2))
        prefix = torch.randint(1, 256, (2, 11, 3), dtype=torch.uint8)
        suffix = torch.randint(1, 256, (2, 3, 5, 3), dtype=torch.uint8)
        suffix[..., 2] %= self.schema.num_actions + 1
        suffix[1, 2, -2:] = 0
        legal = torch.rand(2, 3, 7) > 0.4
        legal[..., 0] = True
        lengths, rows = torch.tensor([4, 11]), torch.tensor([2, 0])
        cache = model.new_cache()
        model.step(cache, rows, prefix, suffix[:, 0], legal[:, 0], append_lengths=lengths)
        before = tuple(t.clone() for t in (cache.length, *cache.keys, *cache.values))
        reference = model.unroll(TokenBatch(prefix, suffix, lengths[:, None].expand(2, 3), legal))
        for indices, tokens, mask, shape in (
            (rows, suffix, legal, (2, 3)),
            (rows.repeat_interleave(3), suffix.flatten(0, 1), legal.flatten(0, 1), (6,)),
        ):
            actual = model.evaluate_candidates(cache, indices, tokens, mask)
            self.assertEqual(actual.value.shape, shape)
            torch.testing.assert_close(actual.value.reshape(2, 3), reference.value, rtol=0, atol=1e-5)
            torch.testing.assert_close(actual.log_policy.reshape_as(legal), reference.log_policy, rtol=0, atol=1e-5)
        for unchanged, original in zip((cache.length, *cache.keys, *cache.values), before):
            self.assertTrue(torch.equal(unchanged, original))
        tail = torch.ones((2, 1, 3), dtype=torch.uint8)
        continued = model.step(cache, rows, tail, suffix[:, 0], legal[:, 0])
        fresh = model.new_cache()
        extended = torch.cat((prefix, tail), dim=1)
        extended[0, 4] = tail[0, 0]
        replay = model.step(fresh, rows, extended, suffix[:, 0], legal[:, 0], append_lengths=lengths + 1)
        torch.testing.assert_close(continued.value, replay.value, rtol=0, atol=1e-5)

    def test_entities_add_scores_to_the_named_action_before_legal_masking(self):
        model = GameTransformer(self.model.spec, replace(self.schema, pointer_field=1))
        with torch.no_grad():
            for weight in model.parameters():
                weight.zero_()
            model.final_norm.weight.fill_(1)
            model.embedding.weight[1:3].fill_(1)
            model.pointer.weight[0, 0] = 1
        prefix = torch.tensor([[[1, 0, 0]]], dtype=torch.uint8)
        suffix = torch.tensor([[[[1, 0, 0], [2, 3, 0], [2, 3, 0], [2, 4, 0], [2, 5, 0]]]], dtype=torch.uint8)
        legal = torch.tensor([[[True, True, True, True, False, False, False]]])
        logp = model.unroll(TokenBatch(prefix, suffix, torch.tensor([[1]]), legal)).log_policy
        torch.testing.assert_close(
            logp[0, 0, :4] - logp[0, 0, 0], torch.tensor([0.0, 0.0, 2.0, 1.0]), atol=1e-5, rtol=0,
        )
        self.assertEqual(logp.exp()[~legal].tolist(), [0.0, 0.0, 0.0])
        (-logp[0, 0, 2]).backward()
        self.assertLess(float(model.pointer.weight.grad[0, 0]), 0)

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

    def test_suffix_attention_matches_joint_softmax_for_every_token(self):
        block = self.model.blocks[-1]
        x = torch.randn(6, 5, 32)
        suffix_valid = torch.arange(5) < torch.tensor([4, 2, 1, 3, 5, 0])[:, None, None, None]
        for span in (0, 7):
            with self.subTest(span=span):
                cuts = torch.tensor([[1, 3, 7], [2, 5, 0]]).clamp_max(span)
                rope = self.model._rope(cuts.reshape(-1, 1))
                prefix_valid = torch.arange(span) < cuts.reshape(-1, 1, 1, 1)
                k_pre, v_pre = (torch.randn(2, 4, span, 8) for _ in range(2))
                q, k, v = block._project(x, rope)
                mask = torch.cat((prefix_valid, suffix_valid), dim=-1)
                attention = torch.nn.functional.scaled_dot_product_attention(
                    q, torch.cat((k_pre.repeat_interleave(3, dim=0), k), dim=-2),
                    torch.cat((v_pre.repeat_interleave(3, dim=0), v), dim=-2), attn_mask=mask,
                )
                expected = block._residual(x, attention)
                actual = block.suffix_blocks(x, rope, k_pre, v_pre, prefix_valid, suffix_valid)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_checkpoints_bind_schema_and_validate_finite_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model, {"game": "opaque"}, {"version": 4})
            checkpoint = load_checkpoint(path, self.schema)
            self.assertEqual((checkpoint.game, checkpoint.version), ({"game": "opaque"}, 4))
            torch.testing.assert_close(checkpoint.network.policy.weight, self.model.policy.weight)
            with self.assertRaisesRegex(ValueError, "token schema"):
                load_checkpoint(path, replace(self.schema, version=2))
            with self.assertRaisesRegex(ValueError, "token schema"):
                load_checkpoint(path, replace(self.schema, pointer_field=2))
            payload = torch.load(path, weights_only=True)
            payload["model_state_dict"]["policy.bias"][0] = float("nan")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_checkpoint(path, self.schema)
            torch.save({"checkpoint_version": 5, "architecture": "card_mlp"}, path)
            with self.assertRaisesRegex(ValueError, "train a fresh"):
                load_checkpoint(path, self.schema)
