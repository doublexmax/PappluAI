from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

from src.checkpoints.io import atomic_copy, atomic_write_json
from src.checkpoints.tensor import atomic_torch_save
from src.game.environment import ENCODING_VERSION
from src.model.network import CHECKPOINT_VERSION, build_checkpoint

if TYPE_CHECKING:
    from src.training.curriculum import TrainingSession


TRAINING_STATE_VERSION = 3
STATUS_SCHEMA_VERSION = 1
ALGORITHM = "full21_curriculum_nstep_ddqn"
PUBLISHED_SNAPSHOT_FILES = (
    "latest-state.pt", "latest-model.pt", "best-model.pt", "baseline-model.pt", "metrics.jsonl",
)


def save_session_checkpoint(
    session: TrainingSession, output_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    if not session.at_episode_boundary:
        raise RuntimeError("cannot checkpoint a partial episode")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.jsonl").touch(exist_ok=True)
    staged_state = output_dir / ".staged-latest-state.pt"
    staged_latest = output_dir / ".staged-latest-model.pt"
    staged_best = output_dir / ".staged-best-model.pt"
    staged_base = output_dir / ".staged-baseline-model.pt"
    atomic_torch_save(session.state_dict(), staged_state)
    atomic_torch_save(
        _model_payload(session, session.network.state_dict(), "latest"), staged_latest,
    )
    atomic_torch_save(_model_payload(session, session.best_model_state, "best"), staged_best)
    atomic_torch_save(_model_payload(session, session.baseline_model_state, "baseline"), staged_base)
    os.replace(str(staged_state), str(output_dir / "latest-state.pt"))
    os.replace(str(staged_latest), str(output_dir / "latest-model.pt"))
    os.replace(str(staged_best), str(output_dir / "best-model.pt"))
    os.replace(str(staged_base), str(output_dir / "baseline-model.pt"))
    active_metrics = output_dir / ".active-metrics.jsonl"
    if active_metrics.exists():
        atomic_copy(active_metrics, output_dir / "metrics.jsonl")
    return _snapshot_file_manifest(output_dir)


def append_validation_metrics(
    output_dir: Path,
    session: TrainingSession,
    full: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    baseline: bool = False,
    validated_stage: Optional[int] = None,
    validated_distance: Optional[int] = None,
    passed: Optional[bool] = None,
    promoted: Optional[bool] = None,
) -> None:
    if baseline:
        append_metric(output_dir, {
            "event": "validation", "source": "ordinary_full21_baseline",
            "episode": session.total_episodes, "stage": session.stage_index,
            "distance": None, **full,
        })
    append_metric(output_dir, {
        "event": "validation", "source": "ordinary_full21",
        "episode": session.total_episodes, "stage": session.stage_index,
        "distance": None, "primary_score": full["greedy"]["win_rate"], **full,
    })
    stage = session.stage_index if validated_stage is None else validated_stage
    distance = session.current_distance if validated_stage is None else validated_distance
    append_metric(output_dir, {
        "event": "validation",
        "source": "curriculum_random_stage" if distance is None else "curriculum_distance_%d" % distance,
        "episode": session.total_episodes, "stage": stage, "distance": distance,
        "passed": passed, "promoted": promoted, **curriculum,
    })


def _model_payload(
    session: TrainingSession, state_dict: Mapping[str, Any], role: str,
) -> Dict[str, Any]:
    checkpoint_episodes = {
        "latest": session.total_episodes, "best": session.best_episode, "baseline": 0,
    }[role]
    payload = build_checkpoint(
        session.network, session.config.game_config,
        meta={
            "algorithm": ALGORITHM, "role": role, "training_state_version": TRAINING_STATE_VERSION,
            "episodes": checkpoint_episodes, "checkpoint_episodes": checkpoint_episodes,
            "best_score": session.best_score, "best_episode": session.best_episode,
            "baseline_validation": session.baseline_validation,
            "latest_full_validation": session.full_validation,
            "training_config": session.config.to_dict(),
        },
    )
    payload["model_state_dict"] = dict(state_dict)
    return payload


def _status_payload(
    session: TrainingSession,
    files: Mapping[str, Mapping[str, Any]],
    status: str,
    stop_reason: Optional[str],
    started: float,
) -> Dict[str, Any]:
    return {
        "status_schema_version": STATUS_SCHEMA_VERSION,
        "status": status, "stop_reason": stop_reason,
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "completed_episodes": session.total_episodes,
        "cumulative_episodes": session.total_episodes,
        "updates": session.total_updates,
        "stage": session.stage_index, "distance": session.current_distance,
        "stage_episodes": session.stage_episodes, "current_epsilon": session.current_epsilon,
        "curriculum_episodes": session.curriculum_episodes, "curriculum_wins": session.curriculum_wins,
        "random_episodes": session.random_episodes, "random_wins": session.random_wins,
        "full_validation": session.full_validation, "curriculum_validation": session.curriculum_validation,
        "baseline_validation": session.baseline_validation,
        "best_score": session.best_score, "best_episode": session.best_episode,
        "training_state_version": TRAINING_STATE_VERSION, "encoding_version": ENCODING_VERSION,
        "model_checkpoint_version": CHECKPOINT_VERSION, "algorithm": ALGORITHM,
        "rules": session.config.to_dict(),
        "files": {name: dict(metadata) for name, metadata in files.items()},
    }


def write_status(
    output_dir: Path,
    session: TrainingSession,
    files: Mapping[str, Mapping[str, Any]],
    status: str,
    stop_reason: Optional[str],
    started: float,
) -> None:
    atomic_write_json(
        _status_payload(session, files, status, stop_reason, started), output_dir / "status.json",
    )


def publish_error(
    output_dir: Path, session: Optional[TrainingSession], message: str, started: float,
) -> None:
    if session is not None and session.at_episode_boundary:
        try:
            files = save_session_checkpoint(session, output_dir)
            payload = _status_payload(session, files, "error", "error", started)
            payload["error"] = message
            atomic_write_json(payload, output_dir / "status.json")
            return
        except (OSError, RuntimeError, ValueError, KeyError) as export_error:
            print("error: checkpoint export failed: %s" % export_error, file=sys.stderr, flush=True)
    try:
        atomic_write_json({
            "status": "error", "stop_reason": "error", "error": message,
            "completed_episodes": session.total_episodes if session is not None else 0,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
            "training_state_version": TRAINING_STATE_VERSION, "encoding_version": ENCODING_VERSION,
        }, output_dir / "error.json")
    except OSError as diagnostic_error:
        print("error: diagnostic export failed: %s" % diagnostic_error, file=sys.stderr, flush=True)


def append_metric(output_dir: Path, event: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": ALGORITHM, "training_state_version": TRAINING_STATE_VERSION,
        "encoding_version": ENCODING_VERSION, "model_checkpoint_version": CHECKPOINT_VERSION,
        **event,
    }
    with (output_dir / ".active-metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _snapshot_file_manifest(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    files: Dict[str, Dict[str, Any]] = {}
    for name in PUBLISHED_SNAPSHOT_FILES:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError("Missing snapshot file: %s" % path)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        files[name] = {"sha256": digest.hexdigest(), "bytes": size}
    return files
