from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
from typing import AbstractSet, Any, Dict, Mapping, Optional

__all__ = (
    "atomic_copy",
    "atomic_torch_save",
    "atomic_write_json",
    "clone_model_state",
    "require_checkpoint_fields",
    "require_checkpoint_int",
    "validate_checkpoint_tree",
    "validate_model_state",
)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    _fsync_file(temporary)
    os.replace(temporary, destination)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _fsync_file(temporary)
    os.replace(temporary, path)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(dict(payload), temporary)
    _fsync_file(temporary)
    os.replace(temporary, path)


def clone_model_state(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
    }


def validate_model_state(
    value: Any,
    reference: Mapping[str, Any],
    name: str,
) -> Dict[str, Any]:
    import torch

    if not isinstance(value, dict) or set(value) != set(reference):
        raise ValueError("%s fields do not match the network" % name)
    result = {}
    for key, expected in reference.items():
        tensor = value[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected.shape
            or tensor.dtype != expected.dtype
        ):
            raise ValueError("%s tensor %s is incompatible" % (name, key))
        if (
            (tensor.is_floating_point() or tensor.is_complex())
            and not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(
                "%s tensor %s contains non-finite values" % (name, key)
            )
        result[key] = tensor.detach().cpu().clone()
    return result


def validate_checkpoint_tree(value: Any, name: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError("%s contains a non-finite tensor" % name)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite number" % name)
        return
    if isinstance(value, Mapping):
        for child in value.values():
            validate_checkpoint_tree(child, name)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            validate_checkpoint_tree(child, name)
        return
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return
    raise ValueError(
        "%s contains unsupported value %s" % (name, type(value).__name__)
    )


def require_checkpoint_fields(
    value: Any,
    fields: AbstractSet[str],
    name: str,
) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("%s fields do not match this checkpoint" % name)
    return value


def require_checkpoint_int(
    value: Any,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an int" % name)
    if minimum is not None and value < minimum:
        raise ValueError("%s must be at least %d" % (name, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("%s must be at most %d" % (name, maximum))
    return value


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())
