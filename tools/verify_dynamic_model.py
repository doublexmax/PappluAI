from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.checkpoints.io import atomic_write_json, file_sha256
from src.evaluation.arena import CheckpointPolicy, play_match
from src.evaluation.ratings import RatingStore
from src.game.environment import GameConfig, legal_action_mask
from src.game.multiplayer import MatchConfig
from src.model.encoding import collate_observations
from src.training.league import LeagueConfig, LeagueSession
from src.training.optimization import train_batch


def normalized(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": hashlib.sha256(
                bytes(tensor.reshape(-1).view(torch.uint8).tolist()),
            ).hexdigest(),
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
            "encodings": [config["encoding_version"] for config in profiles.values()],
        }


def verify(output: Path) -> dict:
    torch.set_num_threads(1)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output.resolve() / run_id
    output.mkdir(parents=True)
    evidence = {
        "status": "RUNNING",
        "output_dir": str(output),
        "python": sys.version, "torch": str(torch.__version__),
        "openskill": version("openskill"),
        "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_changes": subprocess.check_output(
            ["git", "--no-pager", "status", "--short"], cwd=ROOT, text=True,
        ).splitlines(),
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
        snapshots = output / "snapshots"
        snapshots.mkdir()
        paths = []
        for seed in (11, 22, 33):
            path = output / ("fresh-%d.pt" % seed)
            cli("solo", "--episodes", 0, "--seed", seed, "--max-turns", 60,
                "--log-every", 0, "--checkpoint", path)
            payload = torch.load(path, weights_only=True)
            assert payload["checkpoint_version"] == 3
            assert payload["encoding_version"] == 2
            assert payload["architecture"] == "dynamic_table"
            frozen = snapshots / ("model-" + file_sha256(path) + ".pt")
            frozen.write_bytes(path.read_bytes())
            paths.append(frozen)
        evidence["snapshot_sha256"] = [file_sha256(path) for path in paths]
        policy = CheckpointPolicy(str(paths[0]))
        before = digest(policy.network.state_dict())
        full_play = []
        representative = None
        for players in (2, 3, 4):
            result = play_match(
                [policy] * players, MatchConfig(GameConfig(max_turns=60), players),
                seed=4000 + players, record_trajectories=True, record_terminal=True,
                deadline=time.monotonic() + 120,
            )
            observations = [step[0] for trajectory in result.trajectories for step in trajectory]
            for trajectory in result.trajectories:
                for obs, action, _ in trajectory:
                    assert len(obs.players) == players
                    assert legal_action_mask(obs)[action]
                    assert obs.history[0].initial_hand == trajectory[0][0].hand
            assert result.action_count == sum(map(len, result.trajectories))
            assert result.terminal_reason in ("win", "turns_exhausted")
            if result.winner is None:
                assert result.seat_turns == (60,) * players
            full_play.append({
                "players": players, "seed": result.seed, "winner": result.winner,
                "terminal_reason": result.terminal_reason, "seat_turns": result.seat_turns,
                "actions": result.action_count, "max_history": max(len(obs.history) for obs in observations),
            })
            if players == 4:
                representative = max(observations, key=lambda obs: len(obs.history))
            arena_path = output / ("arena-%d.json" % players)
            models = [value for _ in range(players) for value in ("--model", paths[0])]
            cli("arena", *models, "--games", 1, "--seed", 5000 + players,
                "--max-turns", 1, "--output", arena_path)
            arena = json.loads(arena_path.read_text(encoding="utf-8"))
            assert arena["config"]["players"] == players
            assert arena["config"]["game"]["cards_in_hand"] == 21
            assert arena["games"][0]["action_count"] <= 2 * players
        assert digest(policy.network.state_dict()) == before
        evidence["full60_play"] = full_play
        assert representative is not None and len(representative.history) > 16

        timings = []
        for _ in range(3):
            started = time.perf_counter()
            with torch.no_grad():
                values = policy.network(collate_observations([representative], torch.device("cpu")))
            timings.append(time.perf_counter() - started)
            assert torch.isfinite(values).all()
        learner = copy.deepcopy(policy.network)
        learner_before = digest(learner.state_dict())
        optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)
        action = next(i for i, allowed in enumerate(legal_action_mask(representative)) if allowed)
        started = time.perf_counter()
        loss = train_batch(
            learner, optimizer, torch.nn.MSELoss(), [(representative, action, 1.0)],
            torch.device("cpu"), 1.0,
        )
        update_seconds = time.perf_counter() - started
        assert digest(learner.state_dict()) != learner_before
        evidence["bounded_cpu_cost"] = {
            "history_events": len(representative.history), "players": 4, "batch_size": 1,
            "model_spec": learner.spec.to_dict(), "forward_seconds": timings,
            "update_seconds": update_seconds, "loss": loss,
            "errors": 0, "completed_forwards": 3, "completed_updates": 1,
            "scope": "One local run, full collation and history reconstruction. No speedup or production throughput claim.",
        }
        curriculum = output / "curriculum"
        cli("curriculum", "--output-dir", curriculum, "--max-episodes", 1,
            "--max-seconds", 120, "--checkpoint-every", 1, "--validation-games", 1,
            "--validate-every", 100, "--max-turns", 1, "--seed", 61)
        cli("curriculum", "--resume", curriculum / "latest-state.pt", "--output-dir", curriculum,
            "--max-episodes", 2, "--max-seconds", 120, "--checkpoint-every", 1)
        solo_state = torch.load(curriculum / "latest-state.pt", weights_only=True)
        assert solo_state["counters"]["total_episodes"] == 2
        assert solo_state["encoding_version"] == 2
        cli("benchmark", "--checkpoint", curriculum / "latest-model.pt",
            "--output", output / "benchmark.json", "--games", 1, "--warm-games", 0)

        league = output / "league"
        cli("league", "--initial-model", paths[0], "--output-dir", league, "--matches", 6,
            "--players", 2, 3, 4, "--max-turns", 1, "--batch-size", 2,
            "--updates-per-match", 1, "--max-seconds", 120, "--checkpoint-every", 1, "--seed", 62)
        state_path = league / "latest-state.pt"
        first = torch.load(state_path, weights_only=True)
        assert first["count_matches"] == {2: 2, 3: 2, 4: 2}
        control = LeagueSession(
            str(paths[0]),
            config=LeagueConfig(
                game=GameConfig(max_turns=1), batch_size=2, updates_per_match=1,
            ),
            seed=62,
        )
        for _ in range(6):
            control.train_match()
        assert digest(control.state_dict()) == digest(first)
        resumed = LeagueSession.load(str(state_path), str(paths[0]))
        assert control.train_match() == resumed.train_match()
        assert digest(control.state_dict()) == digest(resumed.state_dict())
        exact_digest = digest(resumed.state_dict())
        cli("league", "--initial-model", paths[0], "--output-dir", league,
            "--resume", state_path, "--matches", 1, "--max-seconds", 120, "--checkpoint-every", 1)
        final = torch.load(state_path, weights_only=True)
        assert digest(final) == exact_digest
        counts = Counter(
            episode["player_count"]
            for episode in final["replay_state_dict"]["episodes"]
        )
        assert set(counts) == {2, 3, 4}
        evidence["resume"] = {
            "exact_state_sha256": exact_digest,
            "matches": final["counters"]["completed_matches"],
            "count_matches": final["count_matches"], "replay_episodes_by_count": dict(counts),
            "curriculum_episodes": solo_state["counters"]["total_episodes"],
            "uninterrupted_control_matches": control.completed_matches,
            "comparison": "Uninterrupted actual play and updates against restored CLI state.",
        }
        improve = output / "improve"
        cli("improve", "--initial-model", paths[0], "--output-dir", improve,
            "--max-seconds", 120, "--max-cycles", 1, "--max-turns", 1,
            "--league-matches", 1, "--research-episodes", 1,
            "--selection-blocks", 1, "--confirmation-blocks", 1, "--solo-games", 1,
            "--checkpoint-every", 1)
        cli("improve", "--resume", improve / "latest-state.pt", "--output-dir", improve,
            "--max-seconds", 0)
        assert json.loads((improve / "status.json").read_text())["promotions"] == 0
        improvement_status = json.loads((improve / "status.json").read_text())
        assert improvement_status["completed_cycles"] == 1
        assert improvement_status["last_gate"]["accepted"] is False
        evidence["improvement"] = {
            "cycles": improvement_status["completed_cycles"],
            "promotions": improvement_status["promotions"],
            "cap": improvement_status["rules"]["max_turns"],
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
        assert persisted["scored_results"] == 32
        assert persisted["blocks"] == {"2": 1, "3": 1, "4": 1}
        assert persisted["encodings"] == [2, 2, 2]
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
    parser = argparse.ArgumentParser(description="Verify the current dynamic model with bounded CPU play, training, resume, and ratings.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
