from __future__ import annotations

import importlib
from types import ModuleType


class MissingTrainingDependency(RuntimeError):
    pass


def load_runtime(module: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name not in ("torch", "openskill"):
            raise
        dependency = "PyTorch" if exc.name == "torch" else "OpenSkill"
        raise MissingTrainingDependency(
            dependency + " is required. Run python -m pip install "
            "-r requirements-training.txt"
        ) from exc
