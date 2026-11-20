import ast
import unittest

from tools.check_training_architecture import ROOT, check_cycles, check_import_placement


class TestRuntimeImportCycles(unittest.TestCase):
    def cycle_errors(self, first):
        errors = []
        check_cycles(
            {
                ROOT / "src" / "first.py": ast.parse(first),
                ROOT / "src" / "second.py": ast.parse("import src.first"),
            },
            errors,
        )
        return errors

    def test_function_imports_have_no_registry_exception(self):
        errors = []
        check_import_placement({
            ROOT / "src/checkpoints/registry.py": ast.parse(
                "class ModelRegistry:\n def register(self):\n  import torch\n"
            ),
            ROOT / "src/training/train.py": ast.parse("import torch\n\ndef run():\n return 1\n"),
        }, errors)
        self.assertEqual(errors, [
            "function-local import at src/checkpoints/registry.py:3 in ModelRegistry.register",
        ])

    def test_type_only_reference_is_not_a_runtime_cycle(self):
        for source in (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n import src.second",
            "from typing import TYPE_CHECKING as TC\nif TC:\n import src.second",
            "import typing as t\nif t.TYPE_CHECKING:\n import src.second",
        ):
            with self.subTest(source=source):
                self.assertEqual(self.cycle_errors(source), [])
        self.assertEqual(
            self.cycle_errors("import src.second"),
            ["source import cycle: src.first -> src.second -> src.first"],
        )

    def test_runtime_else_branch_and_unbound_names_still_detect_cycles(self):
        for source in (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n pass\nelse:\n import src.second",
            "TYPE_CHECKING = True\nif TYPE_CHECKING:\n import src.second",
            "from typing import TYPE_CHECKING\nTYPE_CHECKING = True\nif TYPE_CHECKING:\n import src.second",
            "import typing as t\nt.TYPE_CHECKING = True\nif t.TYPE_CHECKING:\n import src.second",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    self.cycle_errors(source),
                    ["source import cycle: src.first -> src.second -> src.first"],
                )


if __name__ == "__main__":
    unittest.main()
