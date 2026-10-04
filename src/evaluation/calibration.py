from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Tuple
import uuid

import torch

from src.checkpoints.io import atomic_write_json, file_sha256
from src.checkpoints.locking import exclusive_writer
from src.checkpoints.registry import ModelRegistry
from src.evaluation.arena import CheckpointPolicy, MatchInterrupted, Policy, RandomPolicy, play_match
from src.evaluation.execution import BoundedWorkers, WorkerFailure
from src.game.environment import GameConfig
from src.game.multiplayer import MatchConfig


@dataclass(frozen=True)
class CalibrationConfig:
    caps: Tuple[int, ...] = (30, 60, 120)
    player_counts: Tuple[int, ...] = (2, 3, 4)
    blocks: int = 32
    seed: int = 9_040_500_000
    workers: int = 4
    match_seconds: float = 120.0
    max_extra_completion_rate: float = 0.05

    def __post_init__(self) -> None:
        for name in ("blocks", "workers"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("%s must be a positive integer" % name)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("caps", "player_counts"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple) or not values
                or any(type(value) is not int or value < 1 for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("%s must contain distinct positive integers" % name)
        if any(count < 2 or count > 6 for count in self.player_counts):
            raise ValueError("player counts must be in 2..6")
        if (
            isinstance(self.match_seconds, bool)
            or not isinstance(self.match_seconds, (int, float))
            or not math.isfinite(self.match_seconds)
            or self.match_seconds <= 0
        ):
            raise ValueError("match_seconds must be finite and positive")
        if (
            isinstance(self.max_extra_completion_rate, bool)
            or not isinstance(self.max_extra_completion_rate, (int, float))
            or not 0 < self.max_extra_completion_rate < 1
        ):
            raise ValueError("max_extra_completion_rate must be in (0, 1)")


@dataclass(frozen=True)
class CalibrationCase:
    block: int
    players: int
    cap: int
    seed: int
    policies: Tuple[str, ...]
    seat_order: Tuple[int, ...]

    @property
    def id(self) -> str:
        return "%d-%d-%d-%s" % (
            self.players, self.block, self.cap,
            "".join(str(seat) for seat in self.seat_order),
        )

    def payload(self) -> dict:
        return {
            "id": self.id,
            "block": self.block,
            "players": self.players,
            "cap": self.cap,
            "seed": self.seed,
            "policies": list(self.policies),
            "seat_order": list(self.seat_order),
        }


def calibration_cases(config: CalibrationConfig, identities: Tuple[str, ...]) -> list[CalibrationCase]:
    if len(set(identities)) != len(identities):
        raise ValueError("competitor identities must be unique")
    if len(identities) < max(config.player_counts):
        raise ValueError("not enough distinct competitors for the requested table sizes")
    cases = []
    for block in range(config.blocks):
        for players in config.player_counts:
            seed = config.seed + block * 8 + players
            design_seed = hashlib.sha256(
                ("lineup:%d:%d:%d" % (config.seed, players, block)).encode("ascii")
            ).digest()
            rng = random.Random(design_seed)
            lineup = tuple(rng.sample(identities, players))
            canonical = tuple(sorted(lineup))
            order = tuple(canonical.index(identity) for identity in lineup)
            caps = list(config.caps)
            rng.shuffle(caps)
            for cap in caps:
                cases.append(CalibrationCase(block, players, cap, seed, lineup, order))
    return cases


_POLICIES: dict[str, Policy] = {}


def _initialize_worker() -> None:
    torch.set_num_threads(1)
    _POLICIES.clear()


def _play_case(case: CalibrationCase, snapshots: str, match_seconds: float) -> dict:
    policies = []
    for identity in case.policies:
        if identity not in _POLICIES:
            _POLICIES[identity] = (
                RandomPolicy() if identity == "random-v1"
                else CheckpointPolicy(str(Path(snapshots) / ("model-" + identity + ".pt")))
            )
        policies.append(_POLICIES[identity])
    started = time.monotonic()
    try:
        result = play_match(
            policies,
            MatchConfig(game=GameConfig(max_turns=case.cap), players=case.players),
            case.seed,
            deadline=started + match_seconds,
        )
    except MatchInterrupted as error:
        return {
            "case": case.payload(), "status": "interrupted",
            "error": str(error), "elapsed_seconds": time.monotonic() - started,
        }
    data = asdict(result)
    del data["trajectories"]
    return {
        "case": case.payload(), "status": "completed", "result": data,
        "elapsed_seconds": time.monotonic() - started,
    }


def validate_case_result(record: dict, case: CalibrationCase) -> None:
    if record.get("case") != case.payload() or record.get("status") != "completed":
        raise ValueError("calibration result does not match its requested case")
    result = record["result"]
    if (
        type(result["seed"]) is not int
        or type(result["action_count"]) is not int
        or type(result["stock_remaining"]) is not int
        or not 0 <= result["stock_remaining"] <= 156
    ):
        raise ValueError("invalid scalar calibration counters")
    turns = result["seat_turns"]
    telemetry = result["telemetry"]
    for values in (turns, telemetry["stock_draws"], telemetry["discard_draws"]):
        if len(values) != case.players or any(type(value) is not int or value < 0 for value in values):
            raise ValueError("invalid per-seat calibration counters")
    if any(turn > case.cap for turn in turns):
        raise ValueError("a calibration match exceeded its draw budget")
    if any(
        stock + discard != turns[seat]
        for seat, (stock, discard) in enumerate(zip(
            telemetry["stock_draws"], telemetry["discard_draws"],
        ))
    ):
        raise ValueError("draw counters do not match completed turns")
    if result["seed"] != case.seed or result["action_count"] != 2 * sum(turns):
        raise ValueError("calibration result has an invalid seed or action count")
    refills = telemetry["refill_turns"]
    if (
        any(type(turn) is not int or not 0 < turn < sum(turns) for turn in refills)
        or list(refills) != sorted(set(refills))
    ):
        raise ValueError("invalid refill timing")
    winner = result["winner"]
    reason = result["terminal_reason"]
    if reason == "win":
        if type(winner) is not int or not 0 <= winner < case.players:
            raise ValueError("a declaration must identify its winning seat")
    elif reason == "turns_exhausted":
        if winner is not None or any(turn != case.cap for turn in turns):
            raise ValueError("a capped match must exhaust every seat without a winner")
    else:
        raise ValueError("unknown completed-match ending")


def upper_event_probability(events: int, trials: int, comparisons: int) -> float:
    if not 0 <= events <= trials:
        raise ValueError("event count must be between zero and the trial count")
    if trials <= 0:
        return 1.0
    alpha = 0.05 / max(1, comparisons)
    if events == 0:
        return -math.expm1(math.log(alpha) / trials)
    if events == trials:
        return 1.0
    coefficients = [
        math.lgamma(trials + 1) - math.lgamma(index + 1) - math.lgamma(trials - index + 1)
        for index in range(events + 1)
    ]
    lower, upper = events / trials, 1.0
    for _ in range(60):
        probability = (lower + upper) / 2
        log_terms = [
            coefficient + index * math.log(probability)
            + (trials - index) * math.log1p(-probability)
            for index, coefficient in enumerate(coefficients)
        ]
        largest = max(log_terms)
        cdf = math.exp(largest) * sum(math.exp(term - largest) for term in log_terms)
        if cdf <= alpha:
            upper = probability
        else:
            lower = probability
    return upper


def summarize_calibration(records: list[dict], config: CalibrationConfig) -> dict:
    grouped = defaultdict(list)
    indexed = {}
    for record in records:
        case = record["case"]
        grouped[(case["players"], case["cap"])].append(record)
        key = (case["players"], case["cap"], case["block"])
        if key in indexed:
            raise ValueError("a calibration deal must have one independently randomized seating per cap")
        indexed[key] = record
    rows = []
    decisions = []
    guard = max(config.caps)
    comparisons = len(config.player_counts) * max(1, len(config.caps) - 1)
    for players in config.player_counts:
        eligible = []
        for cap in sorted(config.caps):
            games = grouped[(players, cap)]
            declarations = [record for record in games if record["result"]["winner"] is not None]
            rounds = sorted(max(record["result"]["seat_turns"]) for record in declarations)
            rows.append({
                "players": players, "cap": cap, "completed_games": len(games),
                "expected_games": config.blocks,
                "declarations": len(declarations),
                "capped_games": len(games) - len(declarations),
                "refill_games": sum(bool(record["result"]["telemetry"]["refill_turns"]) for record in games),
                "declaration_rounds_median": statistics.median(rounds) if rounds else None,
                "declaration_rounds_p95": rounds[math.ceil(0.95 * len(rounds)) - 1] if rounds else None,
                "complete_paired_blocks": 0,
                "extra_completion_blocks": 0,
                "extra_completion_probability_upper": None,
            })
            if cap == guard:
                continue
            events = 0
            complete = 0
            for block in range(config.blocks):
                if not all((players, limit, block) in indexed for limit in (cap, guard)):
                    continue
                complete += 1
                if (
                    indexed[(players, cap, block)]["result"]["winner"] is None
                    and indexed[(players, guard, block)]["result"]["winner"] is not None
                ):
                    events += 1
            upper = upper_event_probability(events, complete, comparisons)
            rows[-1].update({
                "complete_paired_blocks": complete,
                "extra_completion_blocks": events,
                "extra_completion_probability_upper": upper,
            })
            if complete == config.blocks and upper <= config.max_extra_completion_rate:
                eligible.append(cap)
        full_cohort = all(
            len(grouped[(players, cap)]) == config.blocks
            for cap in config.caps
        )
        decisions.append({
            "players": players,
            "cap": min(eligible) if eligible and full_cohort else None,
            "guard_cap": guard,
            "status": (
                "incomplete_experiment" if not full_cohort
                else "supported_for_recorded_pool" if eligible else "insufficient_evidence"
            ),
        })
    return {
        "version": 2, "kind": "draw_limit_calibration",
        "requested_blocks": config.blocks, "completed_games": len(records),
        "selection_rule": {
            "event": "a paired match gains a declaration under the guard when the candidate cap did not",
            "max_probability": config.max_extra_completion_rate,
            "familywise_confidence": 0.95,
            "method": "one-sided exact binomial upper bound with Bonferroni correction",
            "uncertainty_unit": "independent deal with a randomized lineup and seating",
            "guard_is_not_a_recommendation": True,
        },
        "groups": rows, "recommendations": decisions,
    }


def _engine_fingerprint() -> str:
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in (
        "evaluate.py", "game/environment.py", "game/multiplayer.py",
        "game/stock.py", "game/reward_cache.py", "model/network.py",
        "evaluation/arena.py", "evaluation/calibration.py",
        "evaluation/execution.py",
    ):
        digest.update(name.encode("ascii"))
        digest.update((source / name).read_bytes())
    return digest.hexdigest()


def _run_calibration(models_dir: Path, output_dir: Path, config: CalibrationConfig) -> dict:
    sources = sorted(Path(models_dir).glob("model-*.pt"))
    if not sources:
        raise ValueError("models_dir must contain frozen model-<sha256>.pt files")
    identities = []
    for source in sources:
        digest = file_sha256(source)
        if source.name != "model-" + digest + ".pt":
            raise ValueError("frozen model filename does not match its content: %s" % source)
        identities.append(digest)
    identities.append("random-v1")
    cases = calibration_cases(config, tuple(identities))
    output_dir = Path(output_dir)
    protocol_path = output_dir / "protocol.json"
    protocol = {
        "version": 2, "kind": "draw_limit_calibration",
        "sampling": "one-randomized-seating-per-independent-deal-v1",
        "caps": list(config.caps), "player_counts": list(config.player_counts),
        "blocks": config.blocks, "seed": config.seed, "policies": identities,
        "rules": GameConfig().to_dict(),
        "engine_sha256": _engine_fingerprint(), "torch_version": str(torch.__version__),
        "max_extra_completion_rate": config.max_extra_completion_rate,
    }
    del protocol["rules"]["max_turns"]
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing calibration protocol differs from this invocation")
    elif output_dir.exists() and any(path.name != ".writer.lock" for path in output_dir.iterdir()):
        raise ValueError("output_dir must be empty or an existing compatible calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(protocol, protocol_path)
    snapshots = output_dir / "models"
    registry = ModelRegistry(snapshots)
    for source in sources:
        registry.register(str(source), origin="calibration")
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(exist_ok=True)
    by_id = {case.id: case for case in cases}
    records = []
    completed = set()
    for path in results_dir.glob("*.json"):
        if path.stem not in by_id:
            raise ValueError("existing results are outside the requested block range")
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_case_result(record, by_id[path.stem])
        records.append(record)
        completed.add(path.stem)
    pending = iter(case for case in cases if case.id not in completed)
    failure_count = 0
    interrupted = False
    last_status = 0.0
    execution_error = None

    def publish_status(active_jobs: int) -> None:
        nonlocal last_status
        if time.monotonic() - last_status >= 5:
            atomic_write_json({
                "phase": "running", "completed_games": len(completed),
                "requested_games": len(cases), "failed_attempts": failure_count,
                "active_jobs": active_jobs, "unix_time": time.time(),
            }, output_dir / "status.json")
            last_status = time.monotonic()

    try:
        with BoundedWorkers(
            partial(_play_case, snapshots=str(snapshots), match_seconds=config.match_seconds),
            workers=config.workers, task_seconds=config.match_seconds,
            initializer=_initialize_worker,
        ) as pool:
            for outcome in pool.run(pending, on_tick=publish_status):
                case = outcome.job
                record = (
                    {
                        "case": case.payload(), "status": "interrupted",
                        "error": outcome.detail, "error_kind": outcome.kind,
                        "elapsed_seconds": outcome.elapsed_seconds,
                    }
                    if isinstance(outcome, WorkerFailure) else outcome.value
                )
                if record["status"] == "completed":
                    validate_case_result(record, case)
                    atomic_write_json(record, results_dir / (case.id + ".json"))
                    records.append(record)
                    completed.add(case.id)
                else:
                    failure_count += 1
                    atomic_write_json(record, failures_dir / (case.id + "-" + uuid.uuid4().hex + ".json"))
                    raise RuntimeError("calibration case %s failed: %s" % (case.id, record["error"]))
    except KeyboardInterrupt:
        interrupted = True
        raise
    except (RuntimeError, ValueError, OSError) as error:
        execution_error = str(error)
        raise
    finally:
        report = summarize_calibration(records, config)
        report["failed_attempts_this_invocation"] = failure_count
        report["requested_games"] = len(cases)
        report["execution_complete"] = len(completed) == len(cases) and execution_error is None and not interrupted
        report["execution_error"] = execution_error
        atomic_write_json(report, output_dir / "summary.json")
        atomic_write_json({
            "phase": "completed" if report["execution_complete"] else "interrupted" if interrupted else "failed",
            "completed_games": len(completed), "requested_games": len(cases),
            "failed_attempts": failure_count, "error": execution_error,
            "active_jobs": 0, "unix_time": time.time(),
        }, output_dir / "status.json")
    return report


def run_calibration(models_dir: Path, output_dir: Path, config: CalibrationConfig) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_writer(output_dir / ".writer.lock"):
        return _run_calibration(models_dir, output_dir, config)
