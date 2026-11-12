import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestPortableExport(unittest.TestCase):
    def test_bootstrap_runs_with_torch_and_numpy_imports_blocked(self):
        script = """
import builtins, runpy, sys
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy'):
        raise AssertionError('bootstrap imported ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
sys.argv = ['tools/weights_export.py', '--bootstrap', '--output', sys.argv[1], '--golden', sys.argv[2]]
runpy.run_path('tools/weights_export.py', run_name='__main__')
"""
        with tempfile.TemporaryDirectory() as temporary:
            model, golden = Path(temporary) / "model.json", Path(temporary) / "golden.json"
            result = subprocess.run(
                [sys.executable, "-c", script, str(model), str(golden)],
                cwd=ROOT, capture_output=True, text=True, timeout=180, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            document = json.loads(model.read_text())
            fixture = json.loads(golden.read_text())
            self.assertEqual(document["architecture"], "axial_table")
            self.assertEqual(document["source"]["kind"], "bootstrap")
            self.assertEqual(fixture["oracle"], "python-reference")
            self.assertEqual(len(fixture["cases"]), 15)
            self.assertTrue(all(name.startswith("policy.") for name in document["tensors"]))
            self.assertEqual(document["source"]["tensorsSha256"], fixture["tensorsSha256"])
            self.assertEqual(len(fixture["cases"][0]["features"]), 3208)


if __name__ == "__main__":
    unittest.main()
