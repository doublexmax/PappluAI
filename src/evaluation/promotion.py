from __future__ import annotations

from contextlib import ExitStack, closing
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Generator, Optional, Sequence, Tuple, Union

import torch

from src.checkpoints import evidence
from src.checkpoints.io import file_sha256
from src.evaluation.arena import CheckpointPolicy, Deal, MatchInterrupted, RandomPolicy, run_matches
from src.evaluation.benchmark import evaluate_games
from src.evaluation.execution import BoundedWorkers, WorkerFailure, source_fingerprint
from src.game.environment import ENCODING_VERSION, GameConfig
from src.model.spec import ARCHITECTURE


class GateInterrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class _MultiplayerJob:
    players: int
    block_index: int


@dataclass(frozen=True)
class _SoloJob:
    start: int
    games: int


_GateJob = Union[_MultiplayerJob, _SoloJob]


class _GateRunner:
    def __init__(
        self, candidate_path: str, champion_path: str, opponent_paths: Sequence[str],
        rules: GameConfig, seed: int,
    ) -> None:
        self.candidate = CheckpointPolicy(candidate_path)
        self.champion = CheckpointPolicy(champion_path)
        frozen = [CheckpointPolicy(path) for path in opponent_paths]
        for policy in (self.candidate, self.champion, *frozen):
            policy.network.eval()
        self.opponents = [*frozen, RandomPolicy()]
        self.rules = rules
        self.seed = seed

    def run(self, job: _GateJob, deadline: Optional[float] = None) -> dict:
        if isinstance(job, _SoloJob):
            paired = [
                evaluate_games(
                    policy.network, self.rules, job.games,
                    self.seed + 100_000 + job.start, deadline=deadline,
                )
                for policy in (self.candidate, self.champion)
            ]
            return {
                "candidate_outcomes": list(paired[0].outcomes),
                "champion_outcomes": list(paired[1].outcomes),
            }
        lineup = [
            self.candidate, self.champion,
            *(self.opponents[(job.block_index + seat) % len(self.opponents)]
              for seat in range(job.players - 2)),
        ]
        orders = evidence.balanced_seat_orders(job.players)
        matches = run_matches(
            [Deal(self.seed + job.block_index, order, job.players) for order in orders],
            lineup, self.rules, deadline=deadline,
        )
        placement, declaration = 0, 0
        for order, match in zip(orders, matches):
            candidate_seat, champion_seat = order.index(0), order.index(1)
            if match.outcome is None:
                raise RuntimeError("Gate matches require scored placement outcomes")
            placement += match.outcome.margin(candidate_seat, champion_seat)
            declaration += int(match.winner == candidate_seat) - int(match.winner == champion_seat)
        return {
            "placement_margin": placement / len(orders),
            "declaration_margin": declaration / len(orders),
        }


_GATE_WORKER: Optional[_GateRunner] = None


def _init_gate_worker(
    *, expected_source: str, candidate_path: str, champion_path: str,
    opponent_paths: Sequence[str], rules: GameConfig, seed: int,
) -> None:
    if source_fingerprint() != expected_source:
        raise RuntimeError("Gate worker source differs from the coordinator")
    torch.set_num_threads(1)
    global _GATE_WORKER
    _GATE_WORKER = _GateRunner(candidate_path, champion_path, opponent_paths, rules, seed)


def _run_gate_worker(job: _GateJob) -> dict:
    if _GATE_WORKER is None:
        raise RuntimeError("Gate worker was not initialized")
    return _GATE_WORKER.run(job)


def _check_deadline(deadline: Optional[float]) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise GateInterrupted("Promotion evaluation reached its time limit")


def _execute_jobs(
    jobs: Sequence[_GateJob], candidate_path: str, champion_path: str,
    opponent_paths: Sequence[str], rules: GameConfig, seed: int,
    workers: int, task_seconds: float, deadline: Optional[float],
) -> Generator[tuple[_GateJob, dict], None, None]:
    if not jobs:
        return
    _check_deadline(deadline)
    if workers == 1:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            runner = _GateRunner(candidate_path, champion_path, opponent_paths, rules, seed)
            for job in jobs:
                _check_deadline(deadline)
                job_deadline = time.monotonic() + task_seconds
                if deadline is not None:
                    job_deadline = min(job_deadline, deadline)
                try:
                    result = runner.run(job, job_deadline)
                except (MatchInterrupted, TimeoutError) as error:
                    raise GateInterrupted("Promotion evaluation reached its time limit") from error
                _check_deadline(job_deadline)
                yield job, result
        finally:
            torch.set_num_threads(previous_threads)
        return
    initializer = partial(
        _init_gate_worker, expected_source=source_fingerprint(),
        candidate_path=candidate_path, champion_path=champion_path,
        opponent_paths=opponent_paths, rules=rules, seed=seed,
    )
    with BoundedWorkers(
        _run_gate_worker, workers=workers, task_seconds=task_seconds, initializer=initializer,
    ) as pool:
        for result in pool.run(jobs, on_tick=lambda _active: _check_deadline(deadline)):
            if isinstance(result, WorkerFailure):
                if result.kind == "timeout":
                    raise GateInterrupted("Gate job timed out. " + result.detail)
                raise RuntimeError("Gate worker failed. " + result.detail)
            yield result.job, result.value


def _job_key(job: _GateJob) -> tuple:
    if isinstance(job, _MultiplayerJob):
        return ("multiplayer", job.players, job.block_index)
    return ("solo", job.start, job.games)


def _job_result(job: _GateJob, raw: object) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Gate job result must be a dict")
    if isinstance(job, _MultiplayerJob):
        fields = {"placement_margin", "declaration_margin"}
        if set(raw) != fields or any(
            type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1
            for value in raw.values()
        ):
            raise ValueError("Gate block margins must be finite and in [-1, 1]")
        return {name: float(raw[name]) for name in ("placement_margin", "declaration_margin")}
    fields = {"candidate_outcomes", "champion_outcomes"}
    if set(raw) != fields or any(
        not isinstance(values, (list, tuple)) or len(values) != job.games
        or any(type(value) is not int or value not in (0, 1) for value in values)
        for values in raw.values()
    ):
        raise ValueError("Gate solo slices require one binary outcome per game")
    return {name: list(raw[name]) for name in ("candidate_outcomes", "champion_outcomes")}


def _read_journal(stream, gate_id: str, jobs: Sequence[_GateJob]) -> dict:
    expected = {_job_key(job): job for job in jobs}
    completed = {}
    stream.seek(0)
    last_line = ""
    for line in stream:
        last_line = line
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("gate_id") != gate_id:
                continue
            key = record["job"]
            if (
                not isinstance(key, list) or len(key) != 3 or not isinstance(key[0], str)
                or any(type(value) is not int for value in key[1:])
            ):
                continue
            key = tuple(key)
            result = _job_result(expected[key], record["result"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if key in completed and completed[key] != result:
            raise ValueError("Conflicting completed gate job in journal")
        completed[key] = result
    stream.seek(0, os.SEEK_END)
    if last_line and not last_line.endswith("\n"):
        stream.write("\n")
        stream.flush()
    return completed


def evaluate_gate(
    candidate_path: str,
    champion_path: str,
    opponent_paths: Sequence[str],
    seed: int,
    seed_blocks: int,
    solo_games: int,
    max_turns: int = 60,
    player_counts: Tuple[int, ...] = (2, 3, 4),
    samples: int = 2000,
    deadline: Optional[float] = None,
    recycle_discard: bool = True,
    workers: int = 1,
    task_seconds: float = 180.0,
    journal_dir: Optional[Path] = None,
) -> dict:
    for name, value in (
        ("seed_blocks", seed_blocks), ("solo_games", solo_games),
        ("max_turns", max_turns), ("workers", workers),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError("%s must be a positive integer" % name)
    if type(samples) is not int or samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    if (
        type(task_seconds) not in (int, float)
        or not math.isfinite(task_seconds) or task_seconds <= 0
    ):
        raise ValueError("task_seconds must be finite and positive")
    if deadline is not None and (
        type(deadline) not in (int, float) or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be a finite monotonic timestamp")
    if not isinstance(player_counts, (list, tuple)):
        raise TypeError("player_counts must be a list or tuple")
    counts = evidence.require_player_counts(tuple(player_counts))
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    rules = GameConfig(max_turns=max_turns, recycle_discard=recycle_discard)
    paths = (str(candidate_path), str(champion_path), *(str(path) for path in opponent_paths))
    hashes = [file_sha256(Path(path)) for path in paths]
    gate_id = hashlib.sha256(json.dumps((
        evidence.GATE_SCHEMA_VERSION, ENCODING_VERSION, ARCHITECTURE,
        hashes[0], hashes[1], hashes[2:], rules.to_dict(), seed, seed_blocks, solo_games, counts,
    ), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    jobs: list[_GateJob] = [
        _MultiplayerJob(players, index)
        for players in counts
        for index in range(evidence.blocks_for_count(seed_blocks, players))
    ] + [_SoloJob(start, min(32, solo_games - start)) for start in range(0, solo_games, 32)]
    with ExitStack() as stack:
        journal = None
        completed = {}
        if journal_dir is not None:
            directory = Path(journal_dir)
            directory.mkdir(parents=True, exist_ok=True)
            journal = stack.enter_context((directory / (gate_id + ".jsonl")).open(
                "a+", encoding="utf-8", errors="replace",
            ))
            completed = _read_journal(journal, gate_id, jobs)
        pending = [job for job in jobs if _job_key(job) not in completed]
        results = _execute_jobs(
            pending, paths[0], paths[1], paths[2:], rules, seed,
            workers, task_seconds, deadline,
        )
        with closing(results):
            for job, raw in results:
                key, value = _job_key(job), _job_result(job, raw)
                if journal is not None:
                    journal.write(json.dumps(
                        {"gate_id": gate_id, "job": key, "result": value},
                        sort_keys=True, allow_nan=False,
                    ) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                completed[key] = value
    by_player_count = {}
    for players in counts:
        count_blocks = evidence.blocks_for_count(seed_blocks, players)
        ordered = [completed[_job_key(_MultiplayerJob(players, index))] for index in range(count_blocks)]
        placement = [result["placement_margin"] for result in ordered]
        declaration = [result["declaration_margin"] for result in ordered]
        by_player_count[str(players)] = {
            "blocks": count_blocks,
            "games": count_blocks * len(evidence.balanced_seat_orders(players)),
            "placement_margins": placement,
            "declaration_margins": declaration,
            "candidate_minus_champion": sum(placement) / count_blocks,
            "declaration_delta": sum(declaration) / count_blocks,
        }
    slices = [completed[_job_key(job)] for job in jobs if isinstance(job, _SoloJob)]
    candidate = [value for result in slices for value in result["candidate_outcomes"]]
    champion = [value for result in slices for value in result["champion_outcomes"]]
    solo = {
        "seed": seed + 100_000, "games": solo_games,
        "candidate_outcomes": candidate, "champion_outcomes": champion,
        "candidate_wins": sum(candidate), "champion_wins": sum(champion),
    }
    metrics = evidence.evidence_metrics(
        by_player_count, solo, seed_blocks, seed, samples, player_counts=counts,
    )
    for path, digest in zip(paths, hashes):
        if file_sha256(Path(path)) != digest:
            raise ValueError("Checkpoint changed during gate evaluation")
    accepted = max_turns == 60 and recycle_discard and evidence.gate_accepted(
        by_player_count, metrics, seed_blocks, solo_games, player_counts=counts,
    )
    return {
        "gate_schema_version": evidence.GATE_SCHEMA_VERSION,
        "encoding_version": ENCODING_VERSION,
        "architecture": ARCHITECTURE,
        "candidate_sha256": hashes[0],
        "champion_sha256": hashes[1],
        "opponent_sha256": hashes[2:],
        "accepted": accepted,
        "seed": seed,
        "seed_blocks": seed_blocks,
        "bootstrap_samples": samples,
        "seating": "Cyclic/mirrored 2/3 seats and all 24 four-player permutations",
        "player_counts": list(counts),
        "max_turns": max_turns,
        **metrics,
        "multiplayer": by_player_count,
        "solo": solo,
        "rules": rules.to_dict(),
        "scope": "Per-count placement gate with declaration and ordinary full21 solo retention.",
    }
