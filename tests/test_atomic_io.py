from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from src.checkpoints.io import atomic_copy, atomic_write_json


class TestAtomicPublication(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows sharing semantics")
    def test_publication_waits_for_a_short_lived_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            atomic_write_json({"generation": 0}, path)
            opened = threading.Event()

            def read_briefly():
                with path.open("rb") as reader:
                    self.assertTrue(reader.read())
                    opened.set()
                    time.sleep(0.15)

            thread = threading.Thread(target=read_briefly)
            thread.start()
            self.assertTrue(opened.wait(5))
            atomic_write_json({"generation": 1}, path)
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(json.loads(path.read_text()), {"generation": 1})

    def test_unrelated_permission_errors_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            with mock.patch("src.checkpoints.io.os.replace", side_effect=PermissionError("denied")) as replace:
                with self.assertRaises(PermissionError):
                    atomic_write_json({"generation": 1}, path)
            self.assertEqual(replace.call_count, 1)

    def test_atomic_copy_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            destination = Path(directory) / "copy"
            source.write_bytes(b"immutable checkpoint")
            atomic_copy(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()
