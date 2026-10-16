from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import torch

from src.checkpoints.io import file_sha256
from src.checkpoints.registry import ModelRegistry
from src.evaluation.ratings import RatingStore
from src.evaluation.tournament import _execute_job, _initialize_worker, choose_lineup, discover_snapshots, run_tournament
from src.game.environment import GameConfig
from src.model.network import QNetwork, save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def snapshot(directory, seed):
    torch.manual_seed(seed)
    temporary = directory / ("seed-%d.pt" % seed)
    save_checkpoint(str(temporary), QNetwork(), GameConfig(max_turns=1))
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.rename(directory / ("model-" + digest + ".pt"))
    return digest


class TestSnapshotDiscovery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.sources = self.directory / "snapshots"
        self.sources.mkdir()
        self.identity = snapshot(self.sources, 81)
        self.source = self.sources / ("model-" + self.identity + ".pt")
        self.registry = ModelRegistry(self.directory / "owned")
        self.store = RatingStore(self.directory / "ratings.sqlite3").__enter__()

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_rediscovery_keeps_owned_bytes_and_survives_source_deletion(self):
        discover_snapshots((self.sources,), self.registry, self.store)
        before = self.store.competitors()
        owned = self.registry.root / self.source.name
        discover_snapshots((self.sources,), self.registry, self.store)
        self.assertEqual(self.store.competitors(), before)
        self.assertEqual(owned.read_bytes(), self.source.read_bytes())
        self.source.unlink()
        discover_snapshots((self.sources,), self.registry, self.store)
        self.assertEqual(self.store.competitors(), before)
        self.assertEqual(file_sha256(self.registry.model_path(self.identity[:20])), self.identity)
        newcomer = snapshot(self.sources, 82)
        discover_snapshots((self.sources,), self.registry, self.store)
        self.assertEqual(set(self.store.competitors()), {"random-v1", self.identity, newcomer})
        self.assertEqual(file_sha256(self.registry.model_path(newcomer[:20])), newcomer)

    def test_duplicate_new_sources_admit_one_verified_snapshot(self):
        duplicate = self.directory / "duplicate"
        duplicate.mkdir()
        shutil.copyfile(self.source, duplicate / self.source.name)
        discover_snapshots((self.sources, duplicate, self.sources), self.registry, self.store)
        self.assertEqual(set(self.store.competitors()), {"random-v1", self.identity})
        self.assertEqual(
            set(json.loads(self.registry.path.read_text(encoding="utf-8"))["models"]),
            {self.identity[:20]},
        )
        self.assertEqual(self.registry.model_path(self.identity[:20]).read_bytes(), self.source.read_bytes())

    def test_corrupt_or_missing_owned_files_are_not_repaired_from_the_source(self):
        discover_snapshots((self.sources,), self.registry, self.store)
        before = self.store.competitors()
        owned = self.registry.root / self.source.name
        original = owned.read_bytes()
        owned.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum"):
            discover_snapshots((self.sources,), self.registry, self.store)
        self.assertEqual(owned.read_bytes(), b"corrupted")
        self.assertEqual(self.store.competitors(), before)
        owned.unlink()
        for source_present in (True, False):
            if not source_present:
                self.source.unlink()
            with self.subTest(source_present=source_present), self.assertRaises(FileNotFoundError):
                discover_snapshots((self.sources,), self.registry, self.store)
            self.assertEqual(self.store.competitors(), before)
        owned.write_bytes(original)
        discover_snapshots((self.sources,), self.registry, self.store)
        self.assertEqual(file_sha256(owned), self.identity)

    def test_new_invalid_filenames_or_checksums_do_not_mutate_ratings(self):
        before = self.store.competitors()
        revision = self.store.connection.execute("SELECT revision FROM state").fetchone()[0]
        for name in ("model-short.pt", "model-" + "0" * 64 + ".pt"):
            path = self.sources / name
            path.write_bytes(b"not the named checkpoint")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "filename|checksum"):
                discover_snapshots((self.sources,), self.registry, self.store)
            self.assertEqual(self.store.competitors(), before)
            self.assertEqual(
                self.store.connection.execute("SELECT revision FROM state").fetchone()[0], revision,
            )
            path.unlink()
        discover_snapshots((self.sources,), self.registry, self.store)
        self.assertEqual(set(self.store.competitors()), {"random-v1", self.identity})

    def test_loading_an_owned_policy_rechecks_its_checksum(self):
        discover_snapshots((self.sources,), self.registry, self.store)
        owned = self.registry.root / self.source.name
        original = owned.read_bytes()
        job = {
            "raw": None, "participants": [self.identity, "random-v1"],
            "config": {"game": GameConfig(max_turns=1).to_dict()}, "seed": 123,
        }
        with mock.patch.dict("src.evaluation.tournament._POLICIES", clear=True):
            owned.write_bytes(b"corrupted after admission")
            with self.assertRaisesRegex(ValueError, "tournament-owned model checksum"):
                _execute_job(job, str(self.registry.root), 30, "0" * 64)
            owned.write_bytes(original)
            outcome = _execute_job(job, str(self.registry.root), 30, "0" * 64)
        self.assertEqual(outcome["phase"], "play")
        self.assertEqual(outcome["raw"]["seed"], 123)
        self.assertEqual(outcome["raw"]["seat_turns"], (1, 1))


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
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(log_path.read_text(encoding="utf-8"))
                path = self.root / "ratings.sqlite3"
                if path.exists():
                    with closing(sqlite3.connect(
                        path.resolve().as_uri() + "?mode=ro", uri=True,
                    )) as database:
                        initialized = database.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs'"
                        ).fetchone()
                        if initialized and database.execute(
                            "SELECT COUNT(*) FROM jobs WHERE block=0 AND status='complete'"
                        ).fetchone()[0] == 2:
                            break
                time.sleep(0.05)
            else:
                self.fail("continuous tournament did not persist a complete block")
        finally:
            try:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            check=True, capture_output=True, timeout=10,
                        )
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            finally:
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
