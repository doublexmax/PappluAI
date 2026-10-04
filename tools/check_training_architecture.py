from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PACKAGES = (
    "src",
    "src/game",
    "src/model",
    "src/training",
    "src/evaluation",
    "src/checkpoints",
    "src/cli",
)

REMOVED_MODULES = (
    "arena.py",
    "benchmark.py",
    "checkpoints.py",
    "compare.py",
    "environment.py",
    "improve.py",
    "league_train.py",
    "long_train.py",
    "main.py",
    "model.py",
    "multiplayer.py",
    "promotion.py",
    "registry.py",
    "reward_cache.py",
    "rule_migration.py",
    "stock.py",
    "train.py",
    "training_core.py",
)

PURE_MODULES = (
    "src",
    "src.game",
    "src.game.environment",
    "src.game.multiplayer",
    "src.game.reward_cache",
    "src.game.stock",
    "src.training",
    "src.training.core",
    "src.checkpoints",
    "src.checkpoints.evidence",
    "src.checkpoints.io",
    "src.checkpoints.registry",
)

CLI_MODULES = (
    "arena",
    "benchmark",
    "compare",
    "curriculum",
    "improve",
    "league",
    "rule_migration",
    "solo",
)

CLI_RUNTIMES = {
    "arena": "src.evaluation.arena",
    "benchmark": "src.evaluation.benchmark",
    "compare": "src.evaluation.compare",
    "curriculum": "src.training.curriculum",
    "improve": "src.training.improve",
    "league": "src.training.league",
    "rule_migration": "src.checkpoints.rule_migration",
    "solo": "src.training.solo",
}

SERIALIZED_CONSTANTS = {
    "src/model/network.py": {
        "CHECKPOINT_VERSION": 2,
    },
    "src/training/core.py": {
        "STATE_VERSION": 1,
    },
    "src/training/curriculum.py": {
        "TRAINING_STATE_VERSION": 1,
        "ALGORITHM": "full21_curriculum_monte_carlo_q_regression",
    },
    "src/training/league.py": {
        "LEAGUE_STATE_VERSION": 1,
        "ALGORITHM": "champion_league_monte_carlo_q_regression",
    },
    "src/training/improve.py": {
        "IMPROVEMENT_STATE_VERSION": 1,
        "ALGORITHM": "champion_league_mc",
    },
}

LEGACY_IMPORTS = {
    "src.arena",
    "src.benchmark",
    "src.compare",
    "src.environment",
    "src.improve",
    "src.league_train",
    "src.long_train",
    "src.model",
    "src.multiplayer",
    "src.promotion",
    "src.registry",
    "src.reward_cache",
    "src.rule_migration",
    "src.stock",
    "src.train",
    "src.training_core",
}


def module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_module(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom) or node.level:
        return []
    if node.module is None:
        return []
    return [node.module]


def source_trees() -> dict[Path, ast.Module]:
    trees = {}
    for path in sorted(SRC.rglob("*.py")):
        trees[path] = ast.parse(path.read_text(encoding="utf-8"), path)
    return trees


def check_layout(errors: list[str]) -> None:
    for package in PACKAGES:
        initializer = ROOT / package / "__init__.py"
        if not initializer.is_file():
            errors.append("missing package initializer: %s" % initializer)
        elif initializer.read_text(encoding="utf-8").strip():
            errors.append("package initializer must be empty: %s" % initializer)
    for filename in REMOVED_MODULES:
        path = SRC / filename
        if path.exists():
            errors.append("removed flat module still exists: %s" % path)
    for name in CLI_MODULES:
        if not (SRC / "cli" / (name + ".py")).is_file():
            errors.append("missing CLI module: src.cli.%s" % name)


def check_cli_boundaries(
    trees: dict[Path, ast.Module],
    errors: list[str],
) -> None:
    for name, runtime in CLI_RUNTIMES.items():
        path = SRC / "cli" / (name + ".py")
        tree = trees[path]
        for node in ast.walk(tree):
            for imported in imported_module(node):
                if imported.startswith("src.") and imported != "src.cli._runtime":
                    errors.append(
                        "CLI imports implementation before parsing at %s:%d: %s"
                        % (path.relative_to(ROOT), node.lineno, imported)
                    )
        text = path.read_text(encoding="utf-8")
        expected = 'load_runtime("%s")' % runtime
        if text.count(expected) != 1:
            errors.append(
                "%s must load exactly one runtime after parsing: %s"
                % (path.relative_to(ROOT), runtime)
            )


class LocalImportVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, errors: list[str]) -> None:
        self.path = path
        self.errors = errors
        self.scope: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        self._check(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._check(node)

    def _check(self, node: ast.AST) -> None:
        if not self.scope:
            return
        relative = self.path.relative_to(ROOT).as_posix()
        allowed = (
            relative == "src/checkpoints/registry.py"
            and self.scope[-2:] == ["ModelRegistry", "register"]
        )
        if not allowed:
            self.errors.append(
                "function-local import at %s:%d in %s"
                % (relative, node.lineno, ".".join(self.scope))
            )


def check_import_placement(
    trees: dict[Path, ast.Module],
    errors: list[str],
) -> None:
    for path, tree in trees.items():
        LocalImportVisitor(path, errors).visit(tree)


def check_migrated_consumers(errors: list[str]) -> None:
    malformed = (
        "src.training.soloing",
        "src.model.network.network",
        "src.checkpoints.io.io",
    )
    for root in (SRC, ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for value in malformed:
                if value in text:
                    errors.append(
                        "malformed migrated path in %s: %s"
                        % (path.relative_to(ROOT), value)
                    )
            tree = ast.parse(text, path)
            for node in ast.walk(tree):
                for imported in imported_module(node):
                    if imported in LEGACY_IMPORTS:
                        errors.append(
                            "legacy import at %s:%d: %s"
                            % (
                                path.relative_to(ROOT),
                                node.lineno,
                                imported,
                            )
                        )

    command_pattern = re.compile(
        r"-m[\"',\s]+src\.(?:training|evaluation|checkpoints)\."
    )
    for path in (
        ROOT / "README.md",
        ROOT / "TRAINING.md",
        ROOT / ".github" / "workflows" / "training-ci.yml",
    ):
        if command_pattern.search(path.read_text(encoding="utf-8")):
            errors.append(
                "runtime module used as a command in %s"
                % path.relative_to(ROOT)
            )


def check_cycles(
    trees: dict[Path, ast.Module],
    errors: list[str],
) -> None:
    known = {module_name(path) for path in trees}
    graph: dict[str, set[str]] = defaultdict(set)
    for path, tree in trees.items():
        source = module_name(path)
        for node in ast.walk(tree):
            for imported in imported_module(node):
                if imported in known and imported != source:
                    graph[source].add(imported)

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            start = visiting.index(node)
            cycle = visiting[start:] + [node]
            errors.append("source import cycle: %s" % " -> ".join(cycle))
            return
        if node in visited:
            return
        visiting.append(node)
        for target in sorted(graph[node]):
            visit(target)
        visiting.pop()
        visited.add(node)

    for node in sorted(known):
        visit(node)


def assignments(tree: ast.Module) -> dict[str, object]:
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "EpisodeReplay":
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "STATE_VERSION"
            ):
                values["STATE_VERSION"] = ast.literal_eval(statement.value)
    return values


def check_serialized_contracts(
    trees: dict[Path, ast.Module],
    errors: list[str],
) -> None:
    by_relative = {
        path.relative_to(ROOT).as_posix(): tree
        for path, tree in trees.items()
    }
    for relative, expected in SERIALIZED_CONSTANTS.items():
        actual = assignments(by_relative[relative])
        for name, value in expected.items():
            if actual.get(name) != value:
                errors.append(
                    "%s must keep %s=%r, found %r"
                    % (relative, name, value, actual.get(name))
                )
    curriculum = (
        ROOT / "src" / "training" / "curriculum.py"
    ).read_text(encoding="utf-8")
    if curriculum.count('"module": "src.long_train"') != 2:
        errors.append(
            "curriculum source_trace.module must remain 'src.long_train'"
        )


def check_pure_imports(errors: list[str]) -> None:
    script = (
        "import builtins,importlib,sys;"
        "real=builtins.__import__;"
        "builtins.__import__=lambda name,*a,**k: "
        "(_ for _ in ()).throw(ModuleNotFoundError("
        "'blocked optional dependency',name='torch')) "
        "if name=='torch' or name.startswith('torch.') else real(name,*a,**k);"
        "importlib.import_module(sys.argv[1])"
    )
    for module in PURE_MODULES:
        result = subprocess.run(
            [sys.executable, "-c", script, module],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            errors.append(
                "pure module imports torch or fails to import: %s\n%s"
                % (module, result.stderr.strip())
            )


def check_dimensions(errors: list[str]) -> None:
    script = (
        "from dataclasses import fields;"
        "from src.game.environment import "
        "ENCODING_VERSION,GameConfig,NUM_ACTIONS,STATE_DIM;"
        "assert ENCODING_VERSION==1;"
        "assert STATE_DIM==164;"
        "assert NUM_ACTIONS==54;"
        "assert len(fields(GameConfig))==5"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        errors.append(
            "game serialization dimensions changed\n%s"
            % result.stderr.strip()
        )


def main() -> int:
    errors: list[str] = []
    trees = source_trees()
    check_layout(errors)
    check_cli_boundaries(trees, errors)
    check_import_placement(trees, errors)
    check_migrated_consumers(errors)
    check_cycles(trees, errors)
    check_serialized_contracts(trees, errors)
    check_pure_imports(errors)
    check_dimensions(errors)
    if errors:
        for error in errors:
            print("error: %s" % error, file=sys.stderr)
        return 1
    print(
        "training architecture: %d modules, no cycles, pure boundary intact"
        % len(trees)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
