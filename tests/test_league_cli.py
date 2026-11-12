import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.cli.league import main
from src.game.environment import GameConfig

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.model.network import QNetwork, save_checkpoint


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestFreshLeagueOutputs(unittest.TestCase):
    def test_round_flag_batches_matches_without_skipping_checkpoint_boundaries(self):
        from src.model.spec import ModelSpec
        from src.training.league import LeagueSession

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, output = root / "initial.pt", root / "league"
            save_checkpoint(str(model), QNetwork(ModelSpec(8, 1, 1)), GameConfig())
            real_round = LeagueSession.train_round
            calls = []

            def train_round(session, matches, deadline=None):
                calls.append(matches)
                return real_round(session, matches, deadline=deadline)

            with mock.patch.object(LeagueSession, "train_round", train_round), mock.patch(
                "sys.stdout", new_callable=io.StringIO,
            ) as stdout:
                code = main([
                    "--initial-model", str(model), "--output-dir", str(output),
                    "--matches", "5", "--matches-per-round", "4", "--checkpoint-every", "3",
                    "--max-turns", "1", "--batch-size", "2", "--updates-per-match", "1",
                ])
            self.assertEqual(code, 0)
            self.assertEqual(calls, [3, 2])
            self.assertEqual(json.loads(stdout.getvalue().splitlines()[-1])["matches"], 5)
            state = torch.load(output / "latest-state.pt", weights_only=True)
            self.assertEqual(state["counters"]["total_updates"], 5)
            self.assertEqual(state["league_config"]["matches_per_round"], 4)

    def test_fresh_run_rejects_existing_outputs_without_overwriting_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "initial.pt"
            save_checkpoint(str(model), QNetwork(), GameConfig())
            output = root / "league"
            output.mkdir()
            arguments = [
                "--initial-model", str(model),
                "--output-dir", str(output),
                "--matches", "0", "--max-seconds", "0",
            ]
            for name in ("latest-state.pt", "latest-model.pt"):
                existing = output / name
                existing.write_bytes(b"existing training evidence")
                with self.subTest(name=name), mock.patch(
                    "sys.stderr", new_callable=io.StringIO,
                ) as error:
                    self.assertEqual(main(arguments), 2)
                    self.assertIn("use --resume", error.getvalue())
                self.assertEqual(existing.read_bytes(), b"existing training evidence")
                self.assertEqual(
                    {path.name for path in output.iterdir()}, {name},
                )
                existing.unlink()
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(main(arguments), 0)
            self.assertEqual(json.loads(stdout.getvalue())["count_matches"], {"2": 0, "3": 0, "4": 0})
            state = torch.load(output / "latest-state.pt", weights_only=True)
            self.assertEqual(state["counters"]["completed_matches"], 0)


if __name__ == "__main__":
    unittest.main()
