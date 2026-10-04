from __future__ import annotations

import os
import tempfile
import unittest

from src.environment import (
    ENCODING_VERSION,
    NUM_ACTIONS,
    STATE_DIM,
    GameConfig,
    PappluEnv,
    encode_observation,
    legal_action_mask,
)

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.model import CHECKPOINT_VERSION

TORCH_REASON = "PyTorch not installed (see requirements-training.txt)"
ARCHITECTURES = ("mlp", "wide_mlp", "suit_conv")


@unittest.skipIf(torch is None, TORCH_REASON)
class TestQNetwork(unittest.TestCase):
    def test_forward_shape_and_finite_gradients(self):
        from src.model import QNetwork

        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                net = QNetwork(architecture=architecture)
                x = torch.zeros(2, STATE_DIM)
                y = net(x)
                self.assertEqual(tuple(y.shape), (2, NUM_ACTIONS))
                self.assertTrue(torch.isfinite(y).all())
                y.sum().backward()
                self.assertTrue(all(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    for parameter in net.parameters()
                ))

    def test_architecture_contract_and_validation(self):
        from src.model import QNetwork

        for architecture in ARCHITECTURES:
            net = QNetwork(architecture=architecture)
            self.assertEqual(net.architecture, architecture)
            self.assertEqual(net.state_dim, STATE_DIM)
            self.assertEqual(net.num_actions, NUM_ACTIONS)
        with self.assertRaisesRegex(ValueError, "unknown architecture"):
            QNetwork(architecture="transformer")
        with self.assertRaisesRegex(ValueError, "suit_conv requires"):
            QNetwork(state_dim=10, architecture="suit_conv")

    def test_masking_is_unchanged_for_all_architectures(self):
        from src.model import QNetwork, masked_q_values

        state = [0.0] * STATE_DIM
        mask = tuple(index % 3 == 0 for index in range(NUM_ACTIONS))
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                q_values = masked_q_values(
                    QNetwork(architecture=architecture), state, mask
                )
                for index, legal in enumerate(mask):
                    if legal:
                        self.assertTrue(torch.isfinite(q_values[index]))
                    else:
                        self.assertTrue(torch.isneginf(q_values[index]))

    def test_masked_greedy_respects_mask(self):
        from src.model import QNetwork, select_greedy_action, select_action
        import random

        net = QNetwork()
        with torch.no_grad():
            net.fc3.bias.zero_()
            net.fc3.bias[0] = 50.0

        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1, max_turns=5))
        obs = env.reset(seed=1)
        obs = env.step(0)
        mask = legal_action_mask(obs)
        self.assertFalse(mask[0])
        action = select_greedy_action(net, obs)
        self.assertTrue(mask[action])
        action_e = select_action(net, obs, epsilon=0.0, rng=random.Random(0))
        self.assertTrue(mask[action_e])

    def test_epsilon_explores_legal_only(self):
        from src.model import QNetwork, select_action
        import random

        net = QNetwork()
        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1, max_turns=5))
        obs = env.reset(seed=2)
        mask = legal_action_mask(obs)
        legal = {i for i, ok in enumerate(mask) if ok}
        seen = set()
        rng = random.Random(0)
        for _ in range(40):
            a = select_action(net, obs, epsilon=1.0, rng=rng)
            self.assertIn(a, legal)
            seen.add(a)
        self.assertTrue(seen <= legal)

    def test_checkpoint_roundtrip_prediction_all_architectures(self):
        from src.model import QNetwork, save_checkpoint, load_checkpoint

        cfg = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=5)
        env = PappluEnv(cfg)
        obs = env.reset(seed=3)
        state = torch.tensor(encode_observation(obs), dtype=torch.float32).unsqueeze(0)

        with tempfile.TemporaryDirectory() as tmp:
            for architecture in ARCHITECTURES:
                with self.subTest(architecture=architecture):
                    net = QNetwork(architecture=architecture)
                    with torch.no_grad():
                        for parameter in net.parameters():
                            parameter.add_(0.1)
                    before = net(state).detach().clone()
                    path = os.path.join(
                        tmp, "new-directory", architecture + ".pt"
                    )
                    save_checkpoint(path, net, cfg, meta={"note": "test"})
                    loaded, loaded_cfg, meta = load_checkpoint(path)
                    self.assertEqual(loaded.architecture, architecture)
                    self.assertEqual(loaded_cfg.to_dict(), cfg.to_dict())
                    self.assertEqual(meta.get("note"), "test")
                    after = loaded(state).detach()
                    self.assertTrue(torch.allclose(before, after))

    def test_generated_version_one_checkpoint_loads_as_mlp(self):
        from src.model import QNetwork, build_checkpoint, load_checkpoint

        cfg = GameConfig()
        payload = build_checkpoint(QNetwork(), cfg)
        payload["checkpoint_version"] = 1
        del payload["architecture"]
        del payload["game_config"]["recycle_discard"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v1.pt")
            torch.save(payload, path)
            loaded, loaded_cfg, _ = load_checkpoint(path)
        self.assertEqual(loaded.architecture, "mlp")
        self.assertEqual(
            loaded_cfg.to_dict(),
            {**cfg.to_dict(), "recycle_discard": False},
        )

    def test_version_two_checkpoint_requires_known_architecture(self):
        from src.model import QNetwork, build_checkpoint, load_checkpoint

        cfg = GameConfig()
        for architecture in (None, "attention"):
            payload = build_checkpoint(QNetwork(), cfg)
            if architecture is None:
                del payload["architecture"]
                message = "missing architecture"
            else:
                payload["architecture"] = architecture
                message = "unknown architecture"
            with self.subTest(architecture=architecture):
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "bad.pt")
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, message):
                        load_checkpoint(path)

    def test_checkpoint_rejects_expected_architecture_mismatch(self):
        from src.model import QNetwork, save_checkpoint, load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wide.pt")
            save_checkpoint(
                path, QNetwork(architecture="wide_mlp"), GameConfig()
            )
            with self.assertRaisesRegex(ValueError, "does not match expected"):
                load_checkpoint(path, expected_architecture="mlp")

    def test_checkpoint_rejects_network_mismatch_before_mutation(self):
        from src.model import QNetwork, save_checkpoint, load_checkpoint

        target = QNetwork()
        before = {
            name: value.detach().clone()
            for name, value in target.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wide.pt")
            save_checkpoint(
                path, QNetwork(architecture="wide_mlp"), GameConfig()
            )
            with self.assertRaisesRegex(ValueError, "does not match network"):
                load_checkpoint(path, network=target)
        for name, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_checkpoint_rejects_late_bad_tensor_before_mutation(self):
        from src.model import QNetwork, build_checkpoint, load_checkpoint

        target = QNetwork()
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.fill_(7.0)
        before = {
            name: value.detach().clone()
            for name, value in target.state_dict().items()
        }
        payload = build_checkpoint(QNetwork(), GameConfig())
        last_name = next(reversed(payload["model_state_dict"]))
        payload["model_state_dict"][last_name] = torch.zeros(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad-late-tensor.pt")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_checkpoint(path, network=target)
        for name, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), msg=name)

    def test_suit_conv_is_suit_equivariant(self):
        from src.model import QNetwork

        network = QNetwork(architecture="suit_conv")
        state = torch.zeros(2, STATE_DIM)
        state[:, :52] = torch.rand(2, 52)
        state[0, 52 + 4] = 1.0
        state[1, 52 + 35] = 1.0
        state[0, 104 + 27] = 1.0
        state[1, 104 + 11] = 1.0
        state[:, 156:164] = torch.rand(2, 8)
        state[:, 161] = 0.5
        state[:, 162] = 0.7
        permutation = [2, 0, 3, 1]
        permuted = state.clone()
        for offset in (0, 52, 104):
            block = state[:, offset:offset + 52].reshape(2, 4, 13)
            permuted[:, offset:offset + 52] = block[:, permutation].reshape(2, 52)

        original_values = network(state)
        permuted_values = network(permuted)
        torch.testing.assert_close(permuted_values[:, :2], original_values[:, :2])
        expected_discards = original_values[:, 2:].reshape(2, 4, 13)[
            :, permutation
        ]
        torch.testing.assert_close(
            permuted_values[:, 2:].reshape(2, 4, 13),
            expected_discards,
        )

    def test_suit_conv_appends_high_ace_without_circular_padding(self):
        from src.model import QNetwork

        network = QNetwork(architecture="suit_conv")
        self.assertEqual(network.suit_conv1.padding_mode, "zeros")
        self.assertEqual(network.suit_conv2.padding_mode, "zeros")
        state = torch.zeros(1, STATE_DIM)
        expected = torch.empty(4, 13)
        for suit in range(4):
            ranks = torch.arange(13, dtype=torch.float32) + suit * 20
            expected[suit] = ranks
            state[0, suit * 13:(suit + 1) * 13] = ranks
        state[0, 161] = 1.0 / 6.0
        state[0, 162] = 1.0 / 30.0
        captured = []
        handle = network.suit_conv1.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            network(state)
        finally:
            handle.remove()
        conv_input = captured[0].reshape(1, 4, 4, 14)[0, :, 0]
        torch.testing.assert_close(conv_input[:, :13], expected)
        torch.testing.assert_close(conv_input[:, 13], expected[:, 0])

    def test_checkpoint_rejects_bad_version(self):
        from src.model import QNetwork, build_checkpoint, load_checkpoint

        cfg = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=5)
        net = QNetwork()
        payload = build_checkpoint(net, cfg)
        payload["checkpoint_version"] = 999
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.pt")
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_checkpoint_rejects_config_mismatch(self):
        from src.model import QNetwork, save_checkpoint, load_checkpoint

        cfg = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=5)
        other = GameConfig(cards_in_hand=6, required_sequences=1, max_turns=5)
        net = QNetwork()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(path, net, cfg)
            with self.assertRaises(ValueError):
                load_checkpoint(path, expected_config=other)

    def test_versions_exported(self):
        self.assertEqual(CHECKPOINT_VERSION, 2)
        self.assertEqual(ENCODING_VERSION, 1)

    def test_checkpoint_rejects_missing_encoding_dimensions(self):
        from src.model import QNetwork, build_checkpoint, load_checkpoint

        payload = build_checkpoint(QNetwork(), GameConfig())
        del payload["state_dim"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.pt")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_checkpoint(path)

    def test_checkpoint_rejects_nonfinite_imported_model(self):
        from src.model import QNetwork, build_checkpoint, load_checkpoint

        payload = build_checkpoint(QNetwork(), GameConfig())
        payload["model_state_dict"]["fc1.weight"][0, 0] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nonfinite.pt")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_checkpoint(path)

    def test_mask_excludes_actions_below_old_sentinel(self):
        from src.model import QNetwork, select_greedy_action

        net = QNetwork()
        with torch.no_grad():
            for param in net.parameters():
                param.zero_()
            net.fc3.bias.fill_(-1e10)
        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1))
        env.reset(seed=0)
        obs = env.step(0)
        self.assertTrue(legal_action_mask(obs)[select_greedy_action(net, obs)])


if __name__ == "__main__":
    unittest.main()
