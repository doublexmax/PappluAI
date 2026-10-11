import io
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
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main(arguments), 0)
            state = torch.load(output / "latest-state.pt", weights_only=True)
            self.assertEqual(state["counters"]["completed_matches"], 0)
            self.assertEqual(state["encoding_version"], 2)


if __name__ == "__main__":
    unittest.main()
