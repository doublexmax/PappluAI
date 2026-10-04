from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    torch.save(dict(payload), temporary)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
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
