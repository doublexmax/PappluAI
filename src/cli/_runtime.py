from __future__ import annotations

import importlib
from types import ModuleType


class MissingTrainingDependency(RuntimeError):
    pass


def load_runtime(module: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise MissingTrainingDependency(
            "PyTorch is required. Run python -m pip install "
            "-r requirements-training.txt"
        ) from exc
