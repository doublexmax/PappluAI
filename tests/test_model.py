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


@unittest.skipIf(torch is None, TORCH_REASON)
class TestQNetwork(unittest.TestCase):
    def test_forward_shape(self):
        from src.model import QNetwork

        net = QNetwork()
        x = torch.zeros(2, STATE_DIM)
        y = net(x)
        self.assertEqual(tuple(y.shape), (2, NUM_ACTIONS))

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

    def test_checkpoint_roundtrip_prediction(self):
        from src.model import QNetwork, save_checkpoint, load_checkpoint

        cfg = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=5)
        net = QNetwork()
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.1)

        env = PappluEnv(cfg)
        obs = env.reset(seed=3)
        state = torch.tensor(encode_observation(obs), dtype=torch.float32).unsqueeze(0)
        before = net(state).detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "new-directory", "ckpt.pt")
            save_checkpoint(path, net, cfg, meta={"note": "test"})
            loaded, loaded_cfg, meta = load_checkpoint(path)
            self.assertEqual(loaded_cfg.to_dict(), cfg.to_dict())
            self.assertEqual(meta.get("note"), "test")
            after = loaded(state).detach()
            self.assertTrue(torch.allclose(before, after))

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
        self.assertEqual(CHECKPOINT_VERSION, 1)
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
