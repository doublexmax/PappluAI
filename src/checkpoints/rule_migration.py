"""Preserve policy weights while starting a new discard-recycling learning regime."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from src.checkpoints.io import (
    atomic_copy,
    atomic_write_json,
)
from src.checkpoints.tensor import atomic_torch_save
from src.training.curriculum import TrainingConfig
from src.training.improve import ImprovementController, IMPROVEMENT_STATE_VERSION
from src.training.league import LeagueConfig


def migrate(source: Path, output: Path) -> dict:
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "latest-state.pt"
    staged_state = output / ".rule-migration-state.pt"
    if existing.exists():
        saved = torch.load(existing, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict):
            raise ValueError("Unsupported improvement checkpoint")
        if saved.get("config", {}).get("recycle_discard") is True:
            return _finish_interrupted_migration(
                output,
                existing,
                staged_state,
            )
    if source.resolve() != output.resolve():
        for path in source.iterdir():
            if path.is_file() and not path.name.startswith("."):
                atomic_copy(path, output / path.name)
    state_path = output / "latest-state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("improvement_state_version")
        != IMPROVEMENT_STATE_VERSION
    ):
        raise ValueError("Unsupported improvement checkpoint")
    if payload["config"].get("recycle_discard") is True:
        return _finish_interrupted_migration(
            output,
            state_path,
            staged_state,
        )
    before = {
        "league_matches": payload["league_state"]["counters"]["completed_matches"],
        "research_episodes": payload["research_state"]["counters"]["total_episodes"],
        "cycle": payload["cycle"],
        "phase": payload["phase"],
    }
    payload["config"]["recycle_discard"] = True
    league = payload["league_state"]
    league["league_config"]["game"]["recycle_discard"] = True
    LeagueConfig.from_dict(league["league_config"])
    research = payload["research_state"]
    research["training_config"]["recycle_discard"] = True
    research_config = TrainingConfig.from_dict(research["training_config"])
    research["source_trace"]["rules"] = research_config.game_config.to_dict()
    for learner in (league, research):
        replay = learner["replay_state_dict"]
        replay["episodes"] = ()
        learner["optimizer_state_dict"]["state"] = {}
    research["validation"] = {
        "baseline": None, "full": None, "curriculum": None,
        "last_validation_episode": -1, "best_score": -1.0, "best_episode": 0,
    }
    research["curriculum"]["consecutive_passing_validations"] = 0
    payload["phase"] = "training"
    payload["cursors"] = {"league": 0, "research": 0, "next_learner": "league"}
    payload["evaluation"] = {
        "candidate_ids": {}, "selection_results": {}, "nominee_origin": None,
        "nominee_id": None, "confirmation_result": None,
        "champion_at_cycle_id": None, "pending_gate_seed": None,
    }
    payload["last_gate"] = None
    payload["best_score"] = None
    event = {
        "event_id": "discard-recycling-rule-migration",
        "event": "rule_migration",
        "previous": before,
        "recycle_discard": True,
        "top_discard_retained": True,
        "replay_cleared": True,
        "optimizer_moments_reset": True,
        "weights_counters_and_rng_preserved": True,
        "old_evaluations_invalidated": True,
    }
    atomic_torch_save(payload, staged_state)
    controller = ImprovementController.load(str(staged_state), output)
    if (
        controller.league.completed_matches != before["league_matches"]
        or controller.research.total_episodes != before["research_episodes"]
        or len(controller.league.replay) != 0
        or len(controller.research.replay) != 0
    ):
        raise RuntimeError("Migration failed to preserve counters or clear old-rule experience")
    history = output / "metrics.jsonl"
    retired_history = output / "old-rule-metrics.jsonl"
    if history.exists() and not retired_history.exists():
        atomic_copy(history, retired_history)
    _atomic_write_event(history, event)
    _atomic_write_event(output / ".active-metrics.jsonl", event)
    controller.metric_ids = {event["event_id"]}
    atomic_write_json(event, output / "rule-migration.json")
    controller.save_checkpoint("running", None)
    staged_state.unlink(missing_ok=True)
    return event


def _atomic_write_event(path: Path, event: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(event, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _finish_interrupted_migration(
    output: Path,
    state_path: Path,
    staged_state: Path,
) -> dict:
    audit_path = output / "rule-migration.json"
    if (
        not audit_path.exists()
        or _published_migration_complete(output)
    ):
        staged_state.unlink(missing_ok=True)
        return {
            "event": "rule_migration",
            "status": "already_migrated",
        }
    event = json.loads(audit_path.read_text(encoding="utf-8"))
    controller = ImprovementController.load(str(state_path), output)
    _atomic_write_event(output / "metrics.jsonl", event)
    _atomic_write_event(output / ".active-metrics.jsonl", event)
    controller.metric_ids = {event["event_id"]}
    controller.save_checkpoint("running", None)
    staged_state.unlink(missing_ok=True)
    return event


def _published_migration_complete(output: Path) -> bool:
    try:
        status = json.loads(
            (output / "status.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    rules = status.get("rules")
    files = status.get("files")
    return (
        isinstance(rules, dict)
        and rules.get("recycle_discard") is True
        and isinstance(files, dict)
        and "latest-state.pt" in files
        and "rule-migration.json" in files
    )
