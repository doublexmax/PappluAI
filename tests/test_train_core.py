import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zlib

from src.game.environment import GameConfig, PappluEnv
from src.model.encoding import encode
from src.model.spec import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.training.train import Replay, Step, Trainer, Trajectory, collect_round, save_rollout_batch


@unittest.skipIf(torch is None, "PyTorch not installed")
@mock.patch("src.training.train.ZLIB_CHUNK", 128)
class TestReplayStreams(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1))
        first, second = (encode(env.reset(seed)) for seed in (17, 23))
        self.raw = first.data + second.data
        self.replay = Replay(3, gamma=0.5)
        self.replay.append(Trajectory((Step(first, 1), Step(second, 53, False)), 1.0))

    def test_round_trip_accepts_chunked_and_legacy_zlib_streams(self):
        state = self.replay.state_dict()
        self.assertGreater(len(state["features"]), 128)
        self.assertEqual(zlib.decompress(state["features"]), self.raw)
        legacy = zlib.compress(self.raw, level=1)
        for payload in (state["features"], bytearray(legacy), memoryview(legacy)):
            with self.subTest(payload_type=type(payload)):
                restored = Replay(3, gamma=0.5)
                restored.load_state_dict({**state, "features": payload})
                self.assertEqual((len(restored), restored.cursor), (2, 2))
                torch.testing.assert_close(restored.features[:2], self.replay.features[:2], rtol=0, atol=0)
                self.assertEqual(restored.actions[:2].tolist(), [1, 53])
                self.assertEqual(restored.returns[:2].tolist(), [0.5, 1.0])
                self.assertEqual(restored.greedy[:2].tolist(), [True, False])

    def test_invalid_streams_and_byte_lengths_do_not_overrun_the_replay(self):
        state = self.replay.state_dict()
        compressed = state["features"]
        for case, payload, message in (
            ("invalid header", b"not zlib", "corrupted replay data"),
            ("truncated", compressed[:-1], "corrupted replay data"),
            ("trailing data", compressed + b"\x00", "corrupted replay data"),
            ("not bytes-like", None, "corrupted replay data"),
            ("oversized", zlib.compress(self.raw + b"\xff"), "replay byte length mismatch"),
            ("undersized", zlib.compress(self.raw[:-1]), "replay byte length mismatch"),
        ):
            with self.subTest(case=case):
                restored = Replay(3, gamma=0.5)
                restored.features[2].fill_(42)
                with self.assertRaisesRegex(ValueError, message):
                    restored.load_state_dict({**state, "features": payload})
                torch.testing.assert_close(restored.features[2], torch.full_like(restored.features[2], 42), rtol=0, atol=0)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestRoundExports(unittest.TestCase):
    def test_each_round_exports_current_weights_and_consumes_at_most_two_sorted_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Trainer.open(Path(temporary), {
                "num_decks": 1, "cards_in_hand": 3, "required_sequences": 1, "max_turns": 1,
                "player_counts": (1,), "round_episodes": 1, "updates_per_episode": 1,
                "replay_capacity": 16, "batch_size": 4, "validation_games": 0, "checkpoint_every": 100,
            }, initial=ModelSpec(9, 1))
            trajectories = collect_round(trainer.config, trainer.learner.online, str(trainer.root / "best-model.pt"), [], 0, 1)
            inbox = trainer.root / "inbox"
            for index in range(5, 0, -1):
                save_rollout_batch(inbox / ("batch-%03d.pt" % index), episodes=1, trajectories=trajectories)
            rounds = []

            def stop():
                if not trainer.episodes:
                    return False
                latest = torch.load(trainer.root / "latest-model.pt", weights_only=True)
                state = torch.load(trainer.root / "latest-state.pt", weights_only=True)
                self.assertEqual(latest["meta"]["episodes"], trainer.episodes)
                torch.testing.assert_close(latest["model_state_dict"], trainer.learner.online.state_dict(), rtol=0, atol=0)
                rounds.append((trainer.episodes, state["episodes"], sorted(path.name for path in inbox.glob("batch-*.pt"))))
                return len(rounds) == 2

            self.assertEqual(trainer.run(max_episodes=5, stop=stop), "stopped")
            self.assertEqual(rounds, [
                (2, 0, ["batch-003.pt", "batch-004.pt", "batch-005.pt"]),
                (4, 0, ["batch-005.pt"]),
            ])
            lines = (trainer.root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([(row["episodes"], row["updates"]) for row in map(json.loads, lines)], [(2, 2), (4, 2)])
            self.assertEqual(trainer.run(max_episodes=5), "max_episodes")
            self.assertEqual(trainer.episodes, 5)
            self.assertFalse((inbox / "batch-005.pt").exists())
