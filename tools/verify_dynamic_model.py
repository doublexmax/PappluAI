from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.checkpoints.io import atomic_write_json, file_sha256
from src.evaluation.arena import CheckpointPolicy, play_match
from src.evaluation.promotion import evaluate_gate
from src.evaluation.ratings import RatingStore
from src.game.environment import (
    ACTION_TAKE_DISCARD, GameConfig, Observation, discard_action, legal_action_mask,
)
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.game.outcome import DealOutcome
from src.model.encoding import encode
from src.model.network import collate_observations
from src.training.train import Learner, Replay, Step, TrainConfig, Trainer, Trajectory


def normalized(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": hashlib.sha256(
                bytes(tensor.reshape(-1).view(torch.uint8).tolist()),
            ).hexdigest(),
        }
    if isinstance(value, bytes):
        return {
            "bytes_len": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): normalized(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(item) for item in value]
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        normalized(value), sort_keys=True, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def persisted_tournament(path: Path) -> dict:
    with RatingStore(path, read_only=True) as store:
        profiles = store.protocols()
        results = store.connection.execute(
            "SELECT COUNT(*) FROM results WHERE points IS NOT NULL AND proofs IS NOT NULL"
        ).fetchone()[0]
        return {
            "revision": store.connection.execute("SELECT revision FROM state WHERE id=1").fetchone()[0],
            "scored_results": results,
            "blocks": {str(config["players"]): len(store.blocks(profile))
                       for profile, config in profiles.items()},
        }


def verify(output: Path) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output.resolve() / run_id
    output.mkdir(parents=True)
    evidence = {
        "status": "RUNNING",
        "output_dir": str(output),
        "commands": [],
        "strength": "Fresh random weights and tiny CPU updates. No competitive strength claim.",
    }
    evidence_path = output / "evidence.json"

    def cli(name: str, *arguments: object, timeout: int = 180) -> None:
        command = [sys.executable, "-m", "src.cli." + name, *map(str, arguments)]
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False,
        )
        evidence["commands"].append({
            "command": command, "exit_code": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
        })
        atomic_write_json(evidence, evidence_path)
        if completed.returncode:
            raise RuntimeError("%s failed\n%s\n%s" % (name, completed.stdout, completed.stderr))

    try:
        torch.set_num_threads(1)
        evidence.update({
            "python": sys.version, "torch": str(torch.__version__),
            "openskill": version("openskill"),
            "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
        })
        evidence["head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        evidence["source_changes"] = subprocess.check_output(
            ["git", "--no-pager", "status", "--short"], cwd=ROOT, text=True,
        ).splitlines()
        player_counts = (2, 3, 4)
        snapshots = output / "snapshots"
        snapshots.mkdir()
        paths = []
        for seed in (11, 22, 33):
            path = output / ("fresh-%d.pt" % seed)
            cli("train", "--max-episodes", 0, "--seed", seed, "--max-turns", 60,
                "--width", 16, "--blocks", 1,
                "--checkpoint", path)
            frozen = snapshots / ("model-" + file_sha256(path) + ".pt")
            frozen.write_bytes(path.read_bytes())
            paths.append(frozen)
        evidence["snapshot_sha256"] = [file_sha256(path) for path in paths]
        checkpoint = torch.load(paths[0], weights_only=True)
        assert checkpoint["checkpoint_version"] == 5
        assert checkpoint["architecture"] == "card_mlp"
        assert checkpoint["model_spec"] == {"width": 16, "blocks": 1}
        evidence["model"] = {
            "checkpoint_version": checkpoint["checkpoint_version"],
            "architecture": checkpoint["architecture"], "model_spec": checkpoint["model_spec"],
        }
        policy = CheckpointPolicy(str(paths[0]))
        before = digest(policy.network.state_dict())
        full_play = []
        for players in player_counts:
            result = play_match(
                [policy] * players, MatchConfig(GameConfig(max_turns=60), players),
                seed=4000 + players, record_trajectories=True, record_terminal=True,
                deadline=time.monotonic() + 120,
            )
            observations = [step.observation for trajectory in result.trajectories for step in trajectory]
            for trajectory in result.trajectories:
                for decision in trajectory:
                    obs, action = decision.observation, decision.action
                    assert len(obs.players) == players
                    assert legal_action_mask(obs)[action]
                    assert obs.history[0].initial_hand == trajectory[0].observation.hand
                    assert decision.hidden_hands[0] == obs.hand
                    assert decision.features == encode(obs)
            assert result.action_count == sum(map(len, result.trajectories))
            assert result.action_count == 2 * sum(result.seat_turns)
            assert result.terminal_reason in ("win", "turns_exhausted")
            if result.winner is None:
                assert result.seat_turns == (60,) * players
            full_play.append({
                "players": players, "seed": result.seed, "winner": result.winner,
                "terminal_reason": result.terminal_reason, "seat_turns": result.seat_turns,
                "actions": result.action_count, "max_history": max(len(obs.history) for obs in observations),
            })
            arena_path = output / ("arena-%d.json" % players)
            models = [value for _ in range(players) for value in ("--model", paths[0])]
            cli("arena", *models, "--games", 1, "--seed", 5000 + players,
                "--max-turns", 1, "--output", arena_path)
            arena = json.loads(arena_path.read_text(encoding="utf-8"))
            assert arena["config"]["players"] == players
            assert arena["config"]["game"]["cards_in_hand"] == 21
            game = arena["games"][0]
            assert game["action_count"] == 2 * sum(game["seat_turns"])
            assert all(0 <= turns <= arena["config"]["game"]["max_turns"] for turns in game["seat_turns"])
        assert digest(policy.network.state_dict()) == before
        evidence["full60_play"] = full_play

        history_match = MultiplayerEnv(MatchConfig(GameConfig(max_turns=60), max(player_counts)))
        history_match.reset(4004)
        completed_turns = history_match.config.players * history_match.config.game.max_turns
        for _ in range(completed_turns):
            face = history_match.observe().discard_top
            history_match.step(ACTION_TAKE_DISCARD)
            representative = history_match.observe()
            representative_seat = history_match.view().current_seat
            history_match.step(discard_action(face))
        assert history_match.view().terminal_reason == "turns_exhausted"
        assert len(representative.history) == 2 * completed_turns
        assert Observation.from_dict(representative.to_dict()) == representative
        evidence["history_retention"] = {
            "completed_turns": completed_turns,
            "retained_events": len(representative.history),
        }

        started = time.perf_counter()
        with torch.no_grad():
            batch = collate_observations([representative], torch.device("cpu"))
            reloaded = collate_observations([Observation.from_dict(representative.to_dict())], torch.device("cpu"))
            for field_name in ("card", "seat", "context", "legal", "top"):
                assert torch.equal(getattr(batch, field_name), getattr(reloaded, field_name))
            assert len(encode(representative).data) == 3208
            values = policy.network(batch)
        forward_seconds = time.perf_counter() - started
        assert torch.isfinite(values).all()
        learner = Learner(
            copy.deepcopy(policy.network),
            TrainConfig(batch_size=1, learning_rate=1e-4),
        )
        learner_before = digest(learner.online.state_dict())
        action = next(i for i, allowed in enumerate(legal_action_mask(representative)) if allowed)
        outcome = DealOutcome.score(
            history_match.terminal_snapshot(), None, "turns_exhausted",
            history_match.config.game.required_sequences,
        )
        started = time.perf_counter()
        stats = learner.learn([Trajectory(
            len(representative.players),
            (Step(encode(representative), action),),
            outcome.seat(representative_seat),
        )])
        update_seconds = time.perf_counter() - started
        assert digest(learner.online.state_dict()) != learner_before
        evidence["bounded_cpu_cost"] = {
            "history_events": len(representative.history), "players": len(representative.players), "batch_size": 1,
            "model_spec": learner.online.spec.to_dict(), "forward_seconds": [forward_seconds],
            "update_seconds": update_seconds, "loss": stats.loss,
            "errors": 0, "completed_forwards": 1, "completed_updates": 1,
            "scope": "Single diagnostic samples with full collation and encoded replay step. No speedup or production throughput claim.",
        }
        curriculum = output / "curriculum"
        cli("train", "--output-dir", curriculum, "--players", 1,
            "--curriculum-distances", 3, 6, "--max-episodes", 1,
            "--max-seconds", 120, "--checkpoint-every", 1, "--validation-games", 1,
            "--validate-every", 100, "--max-turns", 1, "--seed", 61,
            "--learning-rate", 0.0001, "--epsilon-start", 0.08, "--epsilon-end", 0.02,
            "--epsilon-decay-episodes", 77,
            "--width", 16, "--blocks", 1)
        cli("train", "--output-dir", curriculum,
            "--max-episodes", 2, "--max-seconds", 120, "--checkpoint-every", 1)
        solo_state = torch.load(curriculum / "latest-state.pt", weights_only=True)
        assert solo_state["progress"]["episodes"] == 2
        assert solo_state["learner"]["replay"]["version"] == Replay.STATE_VERSION
        assert tuple(solo_state["config"][name] for name in (
            "learning_rate", "epsilon_start", "epsilon_end", "epsilon_decay_episodes",
        )) == (0.0001, 0.08, 0.02, 77)
        cli("benchmark", "--checkpoint", curriculum / "latest-model.pt",
            "--output", output / "benchmark.json", "--games", 1, "--warm-games", 0)

        league = output / "league"
        cli("train", "--initial-model", paths[0], "--output-dir", league, "--max-episodes", 6,
            "--players", 2, 3, 4, "--max-turns", 1, "--batch-size", 2,
            "--updates-per-episode", 1, "--round-episodes", 3,
            "--learning-rate", 0.00015, "--epsilon-start", 0.12, "--epsilon-end", 0.03,
            "--epsilon-decay-episodes", 77,
            "--max-seconds", 120, "--checkpoint-every", 3, "--seed", 62)
        state_path = league / "latest-state.pt"
        first = torch.load(state_path, weights_only=True)
        league_summary = json.loads(evidence["commands"][-1]["stdout"].splitlines()[-1])
        assert league_summary["count_episodes"] == {str(players): 2 for players in player_counts}
        control = Trainer.open(
            output / "control-league",
            {
                "player_counts": (2, 3, 4),
                "max_turns": 1,
                "batch_size": 2,
                "updates_per_episode": 1,
                "round_episodes": 3,
                "learning_rate": 0.00015,
                "epsilon_start": 0.12,
                "epsilon_end": 0.03,
                "epsilon_decay_episodes": 77,
                "checkpoint_every": 3,
                "seed": 62,
            },
            initial=paths[0],
        )
        control.run(max_episodes=6)
        assert digest(control.state_dict()) == digest(first)
        resumed = Trainer.open(league, initial=paths[0])
        control.run(max_episodes=7)
        resumed.run(max_episodes=7)
        assert digest(control.state_dict()) == digest(resumed.state_dict())
        exact_digest = digest(resumed.state_dict())
        final = torch.load(state_path, weights_only=True)
        assert digest(final) == exact_digest
        counts = Counter(
            entry[0]
            for entry in final["learner"]["replay"]["trajectories"]
        )
        assert set(counts) == {2, 3, 4}
        assert final["learner"]["replay"]["version"] == Replay.STATE_VERSION
        assert tuple(final["config"][name] for name in (
            "learning_rate", "epsilon_start", "epsilon_end", "epsilon_decay_episodes",
        )) == (0.00015, 0.12, 0.03, 77)
        evidence["resume"] = {
            "exact_state_sha256": exact_digest,
            "episodes": final["progress"]["episodes"],
            "count_episodes": dict(final["progress"]["count_episodes"]),
            "replay_trajectories_by_count": dict(counts),
            "curriculum_episodes": solo_state["progress"]["episodes"],
            "uninterrupted_control_episodes": control.progress.episodes,
            "comparison": "Uninterrupted actual play and updates against restored CLI state.",
        }
        improve = output / "improve"
        cli("train", "--initial-model", paths[0], "--output-dir", improve,
            "--players", 2, 3, 4, "--promote",
            "--max-seconds", 120, "--max-cycles", 1, "--max-turns", 1,
            "--cycle-episodes", 1, "--round-episodes", 2,
            "--selection-blocks", 1, "--confirmation-blocks", 1, "--solo-games", 1,
            "--gate-samples", 100,
            "--learning-rate", 0.00015, "--epsilon-start", 0.12, "--epsilon-end", 0.03,
            "--epsilon-decay-episodes", 77,
            "--checkpoint-every", 1, "--gate-workers", 2)
        cli("train", "--output-dir", improve,
            "--max-seconds", 0, "--gate-workers", 1)
        improvement_status = json.loads((improve / "status.json").read_text())
        assert improvement_status["promotions"] == 0
        assert improvement_status["completed_cycles"] == 1
        assert improvement_status["last_gate"]["accepted"] is False
        assert improvement_status["last_gate"]["gate_schema_version"] == 5
        assert improvement_status["last_gate"]["phase"] == "selection"
        assert "rules" in improvement_status["last_gate"]["gate_failures"]
        assert improvement_status["last_gate"]["architecture"] == "card_mlp"
        assert [
            (raw["blocks"], raw["games"])
            for raw in improvement_status["last_gate"]["multiplayer"].values()
        ] == [(4, 8), (2, 12), (1, 24)]
        assert improvement_status["episodes"] == 1
        assert improvement_status["updates"] == 4
        improvement_state = torch.load(improve / "latest-state.pt", weights_only=True)
        assert improvement_state["version"] == 6
        assert tuple(improvement_state["config"][name] for name in (
            "learning_rate", "epsilon_start", "epsilon_end", "epsilon_decay_episodes",
        )) == (0.00015, 0.12, 0.03, 77)
        assert improvement_state["config"]["gate_workers"] == 1
        improvement_metrics = [
            json.loads(line) for line in (improve / "metrics.jsonl").read_text().splitlines()
        ]
        assert [event["event"] for event in improvement_metrics] == ["round", "gate"]
        assert improvement_metrics[-1]["phase"] == "selection"
        evidence["improvement"] = {
            "cycles": improvement_status["completed_cycles"],
            "promotions": improvement_status["promotions"],
            "cap": improvement_status["rules"]["max_turns"],
            "gate_schema_version": improvement_status["last_gate"]["gate_schema_version"],
        }

        gate_arguments = {
            "candidate_path": str(paths[1]), "champion_path": str(paths[0]),
            "opponent_paths": (str(paths[2]),), "seed": 70_000_000,
            "seed_blocks": 1, "solo_games": 35, "max_turns": 1, "samples": 100,
            "phase": "selection",
        }
        journal_dir = output / "gates"
        serial_gate = evaluate_gate(**gate_arguments, workers=1, journal_dir=journal_dir)
        parallel_gate = evaluate_gate(**gate_arguments, workers=2)
        resumed_gate = evaluate_gate(**gate_arguments, workers=2, journal_dir=journal_dir)
        assert json.dumps(serial_gate) == json.dumps(parallel_gate) == json.dumps(resumed_gate)
        journal, = journal_dir.glob("*.jsonl")
        completed_jobs = [json.loads(line) for line in journal.read_text().splitlines()]
        assert len(completed_jobs) == 9
        assert [record["job"] for record in completed_jobs[-2:]] == [["solo", 0, 32], ["solo", 32, 3]]
        assert serial_gate["gate_schema_version"] == 5
        assert serial_gate["phase"] == "selection"
        assert "rules" in serial_gate["gate_failures"]
        assert serial_gate["accepted"] is False
        evidence["gate"] = {
            "gate_schema_version": serial_gate["gate_schema_version"],
            "result_sha256": digest(serial_gate), "completed_jobs": len(completed_jobs),
            "workers_compared": [1, 2], "journal": str(journal),
            "scope": "One-turn worker-count and journal-resume equality. Not promotion evidence.",
        }

        tournament = output / "tournament"
        tournament_args = (
            "--root", tournament, "--snapshots", snapshots,
            "--cap", "2:1", "--cap", "3:1", "--cap", "4:1",
            "--reference", "random-v1", "--blocks", 1, "--workers", 1,
            "--max-seconds", 120, "--match-seconds", 30, "--cap-policy", "points",
        )
        cli("tournament", *tournament_args)
        persisted = persisted_tournament(tournament / "ratings.sqlite3")
        assert persisted["scored_results"] == sum(math.factorial(players) for players in player_counts)
        assert persisted["blocks"] == {str(players): 1 for players in player_counts}
        cli("tournament", *tournament_args)
        assert persisted_tournament(tournament / "ratings.sqlite3") == persisted
        report_path = output / "rating-report.json"
        cli("ratings", "--database", tournament / "ratings.sqlite3", "--output", report_path,
            "--reference", "random-v1", "--cap-policy", "points", "--samples", 100)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert {entry["config"]["players"] for entry in report["reports"]} == {2, 3, 4}
        evidence["tournament"] = persisted
        evidence["artifacts"] = {
            "league_state": str(state_path), "curriculum_state": str(curriculum / "latest-state.pt"),
            "rating_database": str(tournament / "ratings.sqlite3"), "rating_report": str(report_path),
        }
        evidence["status"] = "PASS"
    except BaseException as error:
        evidence["status"] = "FAIL"
        evidence["error"] = repr(error)
        raise
    finally:
        atomic_write_json(evidence, evidence_path)
        print(json.dumps({"status": evidence["status"], "evidence": str(evidence_path)}), flush=True)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify card_mlp v5 play, finetune flags, gate v5 journals, resume, and ratings.")
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "papplu-dynamic-verification")
    args = parser.parse_args()
    verify(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
