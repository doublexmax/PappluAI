from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def source_trees() -> dict[Path, ast.Module]:
    return {
        path: ast.parse(path.read_text(encoding="utf-8"), path)
        for path in sorted(SRC.rglob("*.py"))
    }


def imported_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        return [node.module]
    return []


class LocalImportVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, errors: list[str]) -> None:
        self.path = path
        self.errors = errors
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.allowed = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        self._check(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._check(node)

    def _check(self, node: ast.AST) -> None:
        if not self.functions:
            return
        relative = self.path.relative_to(ROOT).as_posix()
        if (
            relative == "src/checkpoints/registry.py"
            and self.classes[-1:] == ["ModelRegistry"]
            and self.functions[-1:] == ["register"]
        ):
            self.allowed += 1
            return
        scope = ".".join((*self.classes, *self.functions))
        self.errors.append(
            "function-local import at %s:%d in %s"
            % (relative, node.lineno, scope)
        )


def check_import_placement(
    trees: dict[Path, ast.Module],
    errors: list[str],
) -> int:
    allowed = 0
    for path, tree in trees.items():
        visitor = LocalImportVisitor(path, errors)
        visitor.visit(tree)
        allowed += visitor.allowed
    return allowed


def check_cli_boundaries(
    trees: dict[Path, ast.Module],
    errors: list[str],
) -> None:
    for path in sorted((SRC / "cli").glob("*.py")):
        if path.name in {"__init__.py", "_runtime.py"}:
            continue
        tree = trees[path]
        runtime_calls = []
        parse_lines = []
        for node in ast.walk(tree):
            for imported in imported_names(node):
                if (
                    imported.startswith("src.")
                    and imported != "src.cli._runtime"
                ):
                    errors.append(
                        "CLI imports implementation before parsing at %s:%d"
                        % (path.relative_to(ROOT), node.lineno)
                    )
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if name == "parse_args":
                parse_lines.append(node.lineno)
            if name == "load_runtime":
                runtime_calls.append(node)
        if len(runtime_calls) != 1:
            errors.append(
                "%s must load exactly one runtime"
                % path.relative_to(ROOT)
            )
            continue
        call = runtime_calls[0]
        if (
            not call.args
            or not isinstance(call.args[0], ast.Constant)
            or not isinstance(call.args[0].value, str)
            or not call.args[0].value.startswith("src.")
        ):
            errors.append(
                "%s must name its runtime explicitly"
                % path.relative_to(ROOT)
            )
        if not parse_lines or call.lineno <= min(parse_lines):
            errors.append(
                "%s must parse arguments before loading its runtime"
                % path.relative_to(ROOT)
            )


class ModuleImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[ast.AST] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.nodes.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.nodes.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef


def check_cycles(
    trees: dict[Path, ast.Module],
    errors: list[str],
) -> None:
    known = {module_name(path) for path in trees}
    graph: dict[str, set[str]] = defaultdict(set)
    for path, tree in trees.items():
        source = module_name(path)
        visitor = ModuleImportVisitor()
        visitor.visit(tree)
        for node in visitor.nodes:
            for imported in imported_names(node):
                candidates = [imported]
                if isinstance(node, ast.ImportFrom):
                    candidates.extend(
                        imported + "." + alias.name
                        for alias in node.names
                    )
                graph[source].update(
                    candidate
                    for candidate in candidates
                    if candidate in known and candidate != source
                )

    active: list[str] = []
    complete: set[str] = set()

    def visit(module: str) -> None:
        if module in active:
            cycle = active[active.index(module):] + [module]
            errors.append("source import cycle: %s" % " -> ".join(cycle))
            return
        if module in complete:
            return
        active.append(module)
        for imported in sorted(graph[module]):
            visit(imported)
        active.pop()
        complete.add(module)

    for module in sorted(known):
        visit(module)


def pure_modules(trees: dict[Path, ast.Module]) -> list[str]:
    modules = set()
    for path in trees:
        relative = path.relative_to(SRC)
        if path.name == "__init__.py" or relative.parts[0] == "game":
            modules.add(module_name(path))
    for relative in (
        "training/core.py",
        "checkpoints/evidence.py",
        "checkpoints/io.py",
        "checkpoints/registry.py",
    ):
        path = SRC / relative
        if path in trees:
            modules.add(module_name(path))
    return sorted(modules)


def check_pure_imports(
    trees: dict[Path, ast.Module],
    errors: list[str],
) -> None:
    blocker = (
        "import builtins,importlib,sys;"
        "real=builtins.__import__;"
        "builtins.__import__=lambda name,*a,**k: "
        "(_ for _ in ()).throw(ModuleNotFoundError("
        "'blocked optional dependency',name='torch')) "
        "if name=='torch' or name.startswith('torch.') else real(name,*a,**k);"
        "importlib.import_module(sys.argv[1])"
    )
    for module in pure_modules(trees):
        result = subprocess.run(
            [sys.executable, "-c", blocker, module],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            errors.append(
                "Torch-free module failed to import: %s\n%s"
                % (module, result.stderr.strip())
            )


def main() -> int:
    errors: list[str] = []
    trees = source_trees()
    deferred_imports = check_import_placement(trees, errors)
    check_cli_boundaries(trees, errors)
    check_cycles(trees, errors)
    check_pure_imports(trees, errors)
    if errors:
        for error in errors:
            print("error: %s" % error, file=sys.stderr)
        return 1
    print(
        "training architecture: %d modules, no cycles, "
        "%d documented deferred import"
        % (len(trees), deferred_imports)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
