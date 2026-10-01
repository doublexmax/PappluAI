"""Preserve policy weights while starting a new discard-recycling learning regime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def migrate(source: Path, output: Path) -> dict:
    import torch
    from src.improve import ImprovementController, IMPROVEMENT_STATE_VERSION
    from src.league_train import LeagueConfig
    from src.long_train import TrainingConfig
    from src.environment import GameConfig

    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and not path.name.startswith("."):
            (output / path.name).write_bytes(path.read_bytes())
    state_path = output / "latest-state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    if payload.get("improvement_state_version") != IMPROVEMENT_STATE_VERSION:
        raise ValueError("Unsupported improvement checkpoint")
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
    history = output / "metrics.jsonl"
    if history.exists():
        (output / "old-rule-metrics.jsonl").write_bytes(history.read_bytes())
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
    history.write_text(json.dumps(event) + "\n", encoding="utf-8")
    torch.save(payload, state_path)
    controller = ImprovementController.load(str(state_path), output)
    if (
        controller.league.completed_matches != before["league_matches"]
        or controller.research.total_episodes != before["research_episodes"]
        or len(controller.league.replay) != 0
        or len(controller.research.replay) != 0
    ):
        raise RuntimeError("Migration failed to preserve counters or clear old-rule experience")
    controller.save_checkpoint("running", None)
    (output / "rule-migration.json").write_text(json.dumps(event, indent=2), encoding="utf-8")
    controller.save_checkpoint("running", None)
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(Path(args.source), Path(args.output)), sort_keys=True))


if __name__ == "__main__":
    main()
