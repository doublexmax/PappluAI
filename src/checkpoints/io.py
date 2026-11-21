from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import hashlib
import time
from typing import Any, Dict, Mapping

__all__ = (
    "atomic_copy",
    "atomic_replace",
    "atomic_write_json",
    "file_metadata",
    "file_sha256",
)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    atomic_replace(temporary, destination)


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
    atomic_replace(temporary, path)


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


def atomic_replace(stage: Path, target: Path) -> None:
    with stage.open("rb+") as stream:
        os.fsync(stream.fileno())
    deadline = time.monotonic() + 2
    delay = 0.005
    while True:
        try:
            os.replace(stage, target)
            return
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.05)
