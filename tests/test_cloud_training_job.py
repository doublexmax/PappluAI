import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.cloud_training_job import Publisher, restore_snapshot, training_required, learner_contract


class MemoryStore:
    def __init__(self):
        self.blobs = {}
        self.writes = []
        self.fail_name = None

    def request(self, name, data=None, optional=False):
        if data is None:
            return self.blobs.get(name) if optional else self.blobs[name]
        if name == self.fail_name:
            raise RuntimeError("simulated upload failure")
        self.writes.append(name)
        self.blobs[name] = data
        return b""

    def put_json(self, name, value):
        self.request(name, json.dumps(value).encode())

    def get_json(self, name):
        value = self.blobs.get(name)
        return None if value is None else json.loads(value)


def write_snapshot(directory, episode):
    files = {}
    for name in ("latest-state.pt", "latest-model.pt", "best-model.pt", "baseline-model.pt", "metrics.jsonl"):
        data = ("%s:%d" % (name, episode)).encode()
        (directory / name).write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (directory / "status.json").write_text(
        json.dumps({"status_schema_version": 1, "completed_episodes": episode, "files": files}), encoding="utf-8"
    )


class TestCloudSnapshots(unittest.TestCase):
    def test_pointer_commits_after_complete_uploads(self):
        store = MemoryStore()
        publisher = Publisher(store, "run", "source")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_snapshot(output, 1)
            self.assertTrue(publisher.sync(output))
            self.assertEqual(store.writes[-1], "latest.json")
            self.assertEqual(store.get_json("latest.json")["training_status"]["completed_episodes"], 1)
            writes = len(store.writes)
            self.assertTrue(publisher.sync(output))
            self.assertEqual(len(store.writes), writes)

    def test_interrupted_publication_preserves_previous_generation(self):
        store = MemoryStore()
        publisher = Publisher(store, "run", "source")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            write_snapshot(output, 1)
            publisher.sync(output)
            first = store.blobs["latest.json"]
            write_snapshot(output, 2)
            store.fail_name = "checkpoints/1/latest-model.pt"
            with self.assertRaises(RuntimeError):
                publisher.sync(output)
            self.assertEqual(store.blobs["latest.json"], first)
            restore = Path(temporary) / "restore"
            state = restore_snapshot(store, store.get_json("latest.json"), restore)
            self.assertEqual(state.read_bytes(), b"latest-state.pt:1")
            self.assertEqual((restore / "metrics.jsonl").read_bytes(), b"metrics.jsonl:1")
            self.assertTrue((restore / "status.json").is_file())
            store.fail_name = None
            self.assertTrue(publisher.sync(output))
            self.assertEqual(store.get_json("latest.json")["slot"], 1)

    def test_uncommitted_local_file_is_not_published(self):
        store = MemoryStore()
        publisher = Publisher(store, "run", "source")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_snapshot(output, 1)
            publisher.sync(output)
            first = store.blobs["latest.json"]
            (output / "latest-state.pt").write_bytes(b"next generation is not ready")
            self.assertFalse(publisher.sync(output))
            self.assertEqual(store.blobs["latest.json"], first)

    def test_corrupt_remote_checkpoint_is_rejected(self):
        store = MemoryStore()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            write_snapshot(output, 1)
            Publisher(store, "run", "source").sync(output)
            pointer = store.get_json("latest.json")
            store.blobs[pointer["files"]["latest-state.pt"]["blob"]] = b"corrupt"
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                restore_snapshot(store, pointer, Path(temporary) / "restore")

    def test_other_run_cannot_reuse_snapshot(self):
        store = MemoryStore()
        store.put_json("latest.json", {"run_id": "first", "source_sha256": "source"})
        with self.assertRaises(ValueError):
            Publisher(store, "second", "source")

    def test_reentry_does_not_start_training_after_its_window(self):
        pointer = {"training_status": {"completed_episodes": 10, "status": "budget"}}
        self.assertFalse(training_required(pointer, 1000, deadline=100, now=101))
        self.assertFalse(training_required(pointer, 10, deadline=100, now=50))
        self.assertTrue(training_required(pointer, 1000, deadline=100, now=50))
        self.assertTrue(training_required(None, 1000, deadline=100, now=50))

    def test_heartbeat_failure_does_not_cancel_committed_training(self):
        store = MemoryStore()
        store.fail_name = "heartbeat.json"
        publisher = Publisher(store, "run", "source")
        publisher.heartbeat("training")

    def test_cycle_controller_contract_does_not_count_training_episodes_as_cycles(self):
        self.assertEqual(learner_contract("src.improve"), ("--max-cycles", "completed_cycles"))
        self.assertEqual(learner_contract("src.long_train"), ("--max-episodes", "completed_episodes"))
        pointer = {"training_status": {"completed_cycles": 2, "completed_episodes": 1000}}
        self.assertTrue(training_required(pointer, 4, 100, 50, "completed_cycles"))
        self.assertFalse(training_required(pointer, 2, 100, 50, "completed_cycles"))
        with self.assertRaises(ValueError):
            learner_contract("unapproved.module")


if __name__ == "__main__":
    unittest.main()
