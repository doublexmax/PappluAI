import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.cli._runtime import MissingTrainingDependency
from src.cli.train import build_arg_parser, main


ROOT = Path(__file__).resolve().parents[1]


class TestTrainCli(unittest.TestCase):
    def test_collection_flags_and_legacy_options_fail_before_importing_training(self):
        cases = (
            ([], "--output-dir"),
            (["--collect", "out"], "requires --model"),
            (["--collect", "out", "--model", "file.pt", "--worker", "0", "--width", "32"], "learner flags"),
            (["--output-dir", "unused", "--model", "file.pt"], "require --collect"),
            (["--output-dir", "unused", "--round-episodes", "4"], "unrecognized arguments"),
        )
        for flags, message in cases:
            with self.subTest(flags=flags), mock.patch(
                "src.cli.train.load_runtime", side_effect=AssertionError("runtime loaded"),
            ), mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                self.assertEqual(main(flags), 2)
                self.assertIn(message, error.getvalue())
        args = build_arg_parser().parse_args([
            "--output-dir", "run", "--players", "1", "3", "--heads", "2",
            "--champion-share", "0.5", "--warmup-steps", "12",
        ])
        self.assertEqual((args.output_dir, args.player_counts, args.heads), (Path("run"), [1, 3], 2))
        self.assertEqual((args.champion_share, args.warmup_steps), (0.5, 12))

    def test_missing_dependency_does_not_create_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "src.cli.train.load_runtime", side_effect=MissingTrainingDependency("PyTorch is missing"),
        ), mock.patch("sys.stderr", new_callable=io.StringIO) as error:
            path = Path(temporary) / "run"
            self.assertEqual(main(["--output-dir", str(path)]), 2)
            self.assertIn("PyTorch is missing", error.getvalue())
            self.assertFalse(path.exists())

    def test_command_help_with_dependencies_blocked(self):
        script = """
import builtins, runpy, sys
real = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy'):
        raise AssertionError('help imported ' + name)
    return real(name, *args, **kwargs)
builtins.__import__ = blocked
module = sys.argv[1]
sys.argv = [module, '--help']
runpy.run_module(module, run_name='__main__')
"""
        for module, text in (("train", "--collect"), ("evaluate", "--checkpoint"), ("arena", "--league")):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-c", script, f"src.cli.{module}"], cwd=ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(text, result.stdout)
