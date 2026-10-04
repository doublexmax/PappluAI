from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import hashlib
from typing import AbstractSet, Any, Dict, Mapping, Optional

__all__ = (
    "atomic_copy",
    "atomic_write_json",
    "file_metadata",
    "file_sha256",
    "require_checkpoint_fields",
    "require_checkpoint_int",
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
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _fsync_file(temporary)
    os.replace(temporary, path)


def file_metadata(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def file_sha256(path: Path) -> str:
    return str(file_metadata(path)["sha256"])


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
