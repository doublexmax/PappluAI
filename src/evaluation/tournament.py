from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from functools import partial
import hashlib
from itertools import combinations
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Optional, Sequence

import torch

from src.checkpoints.io import atomic_write_json, file_sha256
from src.checkpoints.registry import ModelRegistry
from src.evaluate import SCORING_VERSION, minimum_penalty
from src.evaluation.arena import CheckpointPolicy, Policy, RandomPolicy, play_match
from src.evaluation.execution import BoundedWorkers, WorkerFailure
from src.evaluation.ratings import CAP_POLICIES, RatingStore
from src.game.environment import ENCODING_VERSION, GameConfig
from src.game.multiplayer import MatchConfig


_POLICIES: dict[str, Policy] = {}


class TournamentPaused(RuntimeError):
    pass


def _initialize_worker() -> None:
    torch.set_num_threads(1)
    _POLICIES.clear()


def _execute_job(job: dict, models_dir: str, match_seconds: float, source_sha256: str) -> dict:
    if job["raw"] is not None:
        raw = job["raw"]
        snapshot = raw["terminal_snapshot"]
        penalties = []
        for hand in snapshot["hands"]:
            penalty = minimum_penalty(hand, snapshot["joker"], job["config"]["game"]["required_sequences"])
            penalties.append({
                **asdict(penalty), "qualifying_sequences": penalty.qualifying_sequences,
                "scorer_source_sha256": source_sha256,
            })
        return {"phase": "score", "penalties": penalties, "version": SCORING_VERSION}
    policies = []
    for identity in job["participants"]:
        if identity not in _POLICIES:
            if identity == "random-v1":
                _POLICIES[identity] = RandomPolicy()
            else:
                path = Path(models_dir) / ("model-" + identity + ".pt")
                if file_sha256(path) != identity:
                    raise ValueError("tournament-owned model checksum mismatch")
                _POLICIES[identity] = CheckpointPolicy(str(path))
        policies.append(_POLICIES[identity])
    result = play_match(
        policies,
        MatchConfig(game=GameConfig.from_dict(job["config"]["game"]), players=len(policies)),
        job["seed"], deadline=time.monotonic() + match_seconds, record_terminal=True,
    )
    raw = asdict(result)
    del raw["trajectories"]
    if raw["terminal_snapshot"] is None:
        raise RuntimeError("tournament match omitted its terminal snapshot")
    raw["source_sha256"] = source_sha256
    return {"phase": "play", "raw": raw}


def _source_fingerprint() -> str:
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def discover_snapshots(sources: Sequence[Path], model_registry: ModelRegistry, store: RatingStore) -> None:
    store.register_competitor("random-v1", "Random legal", {"kind": "random", "version": 1})
    paths = set(model_registry.root.glob("model-*.pt"))
    for source in sources:
        source = Path(source)
        if not source.is_dir():
            raise FileNotFoundError(source)
        paths.update(source.glob("model-*.pt"))
    for path in sorted(paths):
        digest = file_sha256(path)
        if path.name != "model-" + digest + ".pt":
            raise ValueError("snapshot filename does not match its checksum: %s" % path)
        record = model_registry.register(str(path), origin="rating")
        store.register_competitor(
            record.sha256,
            "%s %s" % (record.metadata["architecture"], record.sha256[:10]),
            {"filename": record.filename, **record.metadata},
        )


def choose_lineup(identities: Sequence[str], blocks: list[dict], players: int, salt: str) -> tuple[str, ...]:
    if len(identities) < players:
        raise ValueError("not enough distinct frozen competitors for this table")
    counts = Counter()
    pairs = Counter()
    for block in blocks:
        counts.update(block["lineup"])
        pairs.update(tuple(sorted(pair)) for pair in combinations(block["lineup"], 2))

    def tie_break(identity: str) -> bytes:
        return hashlib.sha256((salt + ":" + identity).encode("ascii")).digest()

    chosen = [min(identities, key=lambda identity: (counts[identity], tie_break(identity)))]
    while len(chosen) < players:
        candidate = min(
            (identity for identity in identities if identity not in chosen),
            key=lambda identity: (
                sum(pairs[tuple(sorted((identity, other)))] for other in chosen),
                counts[identity], tie_break(identity),
            ),
        )
        chosen.append(candidate)
    return tuple(chosen)


def _reference(prefix: str, identities: Sequence[str]) -> str:
    matches = [identity for identity in identities if identity == prefix or identity.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("reference must uniquely identify a registered competitor")
    return matches[0]


def _publish_standings(store: RatingStore, profiles: Sequence[str], reference: str, root: Path, cap_policy: str, cap_weight: float) -> None:
    result = {
        "version": 1, "generated_unix_time": time.time(),
        "reports": [store.report(profile, reference, cap_policy, cap_weight=cap_weight) for profile in profiles],
    }
    atomic_write_json(result, root / "standings.json")
    if cap_policy != "exclude":
        atomic_write_json({
            "version": 1, "generated_unix_time": time.time(),
            "reports": [store.report(profile, reference, "exclude") for profile in profiles],
        }, root / "standings-completed-only.json")


def run_tournament(
    root: Path,
    sources: Sequence[Path],
    caps: dict[int, int],
    reference: str,
    workers: int = 4,
    blocks: Optional[int] = 16,
    max_seconds: Optional[float] = None,
    match_seconds: float = 300,
    poll_seconds: float = 60,
    seed: int = 9_041_100_000,
    cap_policy: str = "undecided",
    cap_weight: float = 0.25,
) -> dict:
    if not sources or not caps:
        raise ValueError("snapshot sources and explicit table caps are required")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(reference, str) or not reference:
        raise ValueError("reference is required")
    if blocks is not None and (type(blocks) is not int or blocks < 1):
        raise ValueError("blocks must be positive or None for continuous execution")
    if cap_policy not in CAP_POLICIES or isinstance(cap_weight, bool) or not isinstance(cap_weight, (int, float)) or not math.isfinite(cap_weight) or not 0 <= cap_weight <= 1:
        raise ValueError("invalid capped-game evidence policy")
    for name, value in (("poll_seconds", poll_seconds), ("match_seconds", match_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("%s must be finite and positive" % name)
    if max_seconds is not None and (
        isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
        or not math.isfinite(max_seconds) or max_seconds <= 0
    ):
        raise ValueError("max_seconds must be finite and positive")
    if type(seed) is not int or not 0 <= seed < 2 ** 52:
        raise ValueError("invalid tournament seed")
    for players, cap in caps.items():
        if type(cap) is not int or cap < 1:
            raise ValueError("each cap must be a positive integer")
        MatchConfig(game=GameConfig(max_turns=cap), players=players)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    phase, error = "failed", None
    final = {}
    with RatingStore(root / "ratings.sqlite3") as store:
        model_registry = ModelRegistry(root / "models")
        discover_snapshots(sources, model_registry, store)
        identities = tuple(store.competitors())
        anchor = _reference(reference, identities)
        profiles = [
            store.ensure_protocol({
                "players": players, "game": GameConfig(max_turns=cap).to_dict(),
                "encoding_version": ENCODING_VERSION, "scoring_version": SCORING_VERSION,
                "policy": "greedy-v1", "seating_design": "all-permutations-v1", "seed": seed,
            })
            for players, cap in sorted(caps.items())
        ]
        source_sha256 = _source_fingerprint()
        atomic_write_json({
            "source_sha256": source_sha256, "python_version": sys.version,
            "torch_version": str(torch.__version__), "started_unix_time": time.time(),
            "workers": workers, "match_seconds": match_seconds, "profiles": profiles,
            "reference": anchor, "cap_policy": cap_policy, "cap_weight": cap_weight,
        }, root / "executions" / ("%d.json" % time.time_ns()))
        last_status = last_discovery = 0.0

        def tick(active: int) -> None:
            nonlocal last_status
            if deadline is not None and time.monotonic() >= deadline:
                raise TournamentPaused("tournament runtime budget reached")
            if time.monotonic() - last_status >= 5:
                atomic_write_json({
                    "phase": "running", "pid": os.getpid(), "unix_time": time.time(),
                    "active_jobs": active, "reference": anchor, "cap_policy": cap_policy,
                    "cap_weight": cap_weight, **store.progress(profiles),
                }, root / "status.json")
                last_status = time.monotonic()

        try:
            with BoundedWorkers(
                partial(
                    _execute_job, models_dir=str(model_registry.root),
                    match_seconds=match_seconds, source_sha256=source_sha256,
                ),
                workers=workers, task_seconds=match_seconds, initializer=_initialize_worker,
            ) as pool:
                while True:
                    tick(0)
                    if time.monotonic() - last_discovery >= poll_seconds:
                        discover_snapshots(sources, model_registry, store)
                        last_discovery = time.monotonic()
                    pending = [job for job in store.jobs(pending_only=True) if job["protocol"] in profiles]
                    if not pending:
                        identities = tuple(store.competitors())
                        for profile in profiles:
                            history = store.blocks(profile)
                            if blocks is not None and len(history) >= blocks:
                                continue
                            config = store.protocols()[profile]
                            ordinal = len(history)
                            lineup = choose_lineup(identities, history, config["players"], profile + ":" + str(ordinal))
                            block_seed = seed + ordinal * 8 + config["players"]
                            store.schedule_block(profile, ordinal, block_seed, lineup)
                        pending = [job for job in store.jobs(pending_only=True) if job["protocol"] in profiles]
                    if not pending:
                        phase = "failed" if store.progress(profiles)["failed_jobs"] else "completed"
                        if phase == "failed":
                            error = "some tournament jobs exhausted their retry limit"
                        break

                    def attempts():
                        for job in pending:
                            store.start_attempt(job["id"])
                            yield job

                    for outcome in pool.run(attempts(), on_tick=tick):
                        job = outcome.job
                        if isinstance(outcome, WorkerFailure):
                            terminal = store.failure_count(job["id"]) + 1 >= 3
                            store.fail_job(job["id"], outcome.kind + ": " + outcome.detail, terminal)
                            continue
                        result = outcome.value
                        if result["phase"] == "play":
                            store.record_raw(job["id"], result["raw"])
                        elif result["phase"] == "score":
                            store.record_scores(job["id"], result["penalties"], result["version"])
                        else:
                            raise RuntimeError("worker returned an unknown tournament phase")
                    _publish_standings(store, profiles, anchor, root, cap_policy, cap_weight)
        except (TournamentPaused, KeyboardInterrupt):
            phase = "paused"
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as exception:
            phase, error = "failed", str(exception)
            raise
        finally:
            if phase == "failed" and error is None and sys.exc_info()[1] is not None:
                error = str(sys.exc_info()[1])
            _publish_standings(store, profiles, anchor, root, cap_policy, cap_weight)
            final = {
                "phase": phase, "error": error, "pid": os.getpid(),
                "unix_time": time.time(), "active_jobs": 0, "reference": anchor,
                "cap_policy": cap_policy, "cap_weight": cap_weight, **store.progress(profiles),
            }
            atomic_write_json(final, root / "status.json")
    return final
