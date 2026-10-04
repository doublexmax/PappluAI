from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import torch

from src.evaluation.ratings import RatingStore
from src.evaluation.tournament import _initialize_worker, choose_lineup, run_tournament
from src.game.environment import GameConfig
from src.model.network import QNetwork, save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def snapshot(directory, seed):
    torch.manual_seed(seed)
    temporary = directory / ("seed-%d.pt" % seed)
    save_checkpoint(str(temporary), QNetwork(architecture="suit_conv"), GameConfig(max_turns=1))
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.rename(directory / ("model-" + digest + ".pt"))
    return digest


class TestTournament(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.models = self.directory / "snapshots"
        self.models.mkdir()
        self.reference = snapshot(self.models, 81)
        self.root = self.directory / "run"

    def tearDown(self):
        self.temporary.cleanup()

    def run_blocks(self, blocks):
        return run_tournament(
            self.root, (self.models,), {2: 1}, self.reference,
            workers=1, blocks=blocks, match_seconds=30, cap_policy="weighted-points",
        )

    def test_real_matches_are_scored_once_and_new_snapshots_join(self):
        result = self.run_blocks(2)
        self.assertEqual(result["phase"], "completed")
        self.assertEqual(result["scored_games"], 4)
        self.assertEqual(result["failed_jobs"], 0)
        with RatingStore(self.root / "ratings.sqlite3", read_only=True) as store:
            before = store.jobs()
            self.assertTrue(all(job["raw"]["terminal_snapshot"] for job in before))
            self.assertTrue(all(job["status"] == "complete" for job in before))
        resumed = self.run_blocks(2)
        self.assertEqual(resumed["scored_games"], 4)
        with RatingStore(self.root / "ratings.sqlite3", read_only=True) as store:
            self.assertEqual(before, store.jobs())
        newcomer = snapshot(self.models, 82)
        extended = self.run_blocks(3)
        self.assertEqual(extended["scored_games"], 6)
        with RatingStore(self.root / "ratings.sqlite3", read_only=True) as store:
            profile = next(iter(store.protocols()))
            self.assertIn(newcomer, store.blocks(profile)[-1]["lineup"])
            report = store.report(profile, self.reference, "weighted-points")
            self.assertEqual(report["completed_blocks"], 3)
            self.assertFalse(report["cap_weight_validated"])

    def test_abrupt_coordinator_restart_recovers_without_duplicate_results(self):
        command = [
            sys.executable, "-m", "src.cli.tournament", "--root", str(self.root),
            "--snapshots", str(self.models), "--cap", "2:1", "--reference", self.reference,
            "--continuous", "--workers", "1", "--match-seconds", "30",
        ]
        log_path = self.directory / "continuous.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(log_path.read_text(encoding="utf-8"))
                path = self.root / "standings.json"
                if path.exists():
                    reports = json.loads(path.read_text(encoding="utf-8"))["reports"]
                    if reports and reports[0]["completed_blocks"] >= 1:
                        break
                time.sleep(0.05)
            else:
                self.fail("continuous tournament did not publish a complete block")
        finally:
            process.terminate()
            process.wait(timeout=5)
            log.close()
        with RatingStore(self.root / "ratings.sqlite3", read_only=True) as store:
            profile = next(iter(store.protocols()))
            target = len(store.blocks(profile)) + 1
        result = self.run_blocks(target)
        self.assertEqual(result["phase"], "completed")
        with RatingStore(self.root / "ratings.sqlite3", read_only=True) as store:
            jobs = store.jobs()
            self.assertEqual(len(jobs), target * 2)
            self.assertEqual(len({job["id"] for job in jobs}), len(jobs))
            self.assertTrue(all(job["status"] == "complete" for job in jobs))

    def test_scheduling_uses_coverage_rather_than_ratings(self):
        identities = tuple(character * 64 for character in "abcde")
        history = []
        for ordinal in range(10):
            lineup = choose_lineup(identities, history, 3, str(ordinal))
            self.assertEqual(len(set(lineup)), 3)
            self.assertEqual(lineup, choose_lineup(identities, history, 3, str(ordinal)))
            history.append({"lineup": list(lineup)})
        counts = {identity: sum(identity in block["lineup"] for block in history) for identity in identities}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 2)

    def test_worker_refuses_a_different_source_fingerprint(self):
        with self.assertRaisesRegex(RuntimeError, "worker source changed"):
            _initialize_worker("0" * 64)


if __name__ == "__main__":
    unittest.main()
