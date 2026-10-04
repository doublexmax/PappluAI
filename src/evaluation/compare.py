"""Run a fixed-budget architecture comparison from an experiment plan."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from src.evaluation.benchmark import paired_comparison


def run_comparison(plan: dict, output: Path) -> dict:
    if (output / "comparison.json").exists():
        raise FileExistsError("comparison already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    architectures = plan["architectures"]
    seeds = plan["training_seeds"]
    if "mlp" not in architectures or len(set(architectures)) != len(architectures):
        raise ValueError("architectures must be distinct and include the mlp control")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct training seeds are required")
    if len(architectures) < plan["promotion_target"]["minimum_structures"]:
        raise ValueError("plan requires more architecture candidates")
    screen = plan["screen"]
    for name in ("episodes", "validation_games", "test_games", "warm_validation_games"):
        value = screen[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("%s must be a positive integer" % name)
    ranges = [
        (screen["validation_seed"], screen["validation_seed"] + screen["validation_games"]),
        (screen["test_seed"], screen["test_seed"] + screen["test_games"]),
        (screen["warm_validation_seed"], screen["warm_validation_seed"] + screen["warm_validation_games"]),
    ]
    for index, (start, end) in enumerate(ranges):
        if any(start < other_end and other_start < end for other_start, other_end in ranges[index + 1:]):
            raise ValueError("validation, test, and warm-validation seed ranges must not overlap")
    started = time.monotonic()
    report = {
        "plan": plan, "status": "running", "runs": [], "stages": [],
        "scope": "reduced-game architecture screen; not full 21-card validation",
    }
    report_path = output / "comparison.json"

    def persist():
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        temporary = output / "comparison.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)

    def run(name, command, timeout=900):
        stage = {"name": name, "command": command, "status": "running"}
        report["stages"].append(stage)
        persist()
        print(json.dumps({"stage": name, "status": "running"}), flush=True)
        before = time.monotonic()
        try:
            with (output / (name + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout, check=False,
                )
            stage["exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError("%s failed with exit code %d" % (name, result.returncode))
            stage["status"] = "success"
        except (subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
            stage["status"] = "timed_out" if isinstance(exc, subprocess.TimeoutExpired) else "failed"
            report["status"] = "failed"
            raise
        finally:
            stage["seconds"] = round(time.monotonic() - before, 2)
            persist()
        print(json.dumps({"stage": name, "status": stage["status"], "seconds": stage["seconds"]}), flush=True)

    def evaluate(name, checkpoint, games, seed, warm_games=0, sensitivity=False):
        destination = output / (name + ".json")
        command = [
            sys.executable, "-m", "src.cli.benchmark", "--checkpoint", str(checkpoint),
            "--output", str(destination), "--games", str(games), "--seed", str(seed),
            "--warm-games", str(warm_games),
            "--warm-seed", str(screen["warm_validation_seed"]),
        ]
        if sensitivity:
            command.append("--sensitivity")
        run(name, command)
        return json.loads(destination.read_text(encoding="utf-8"))

    run("regression", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], 180)
    rules = [
        "--cards-in-hand", str(screen["cards_in_hand"]),
        "--required-sequences", str(screen["required_sequences"]),
        "--max-turns", str(screen["max_turns"]),
    ]
    sensitivity_checkpoint = output / "sensitivity.pt"
    run("sensitivity-model", [
        sys.executable, "-m", "src.cli.solo", "--episodes", "0",
        "--seed", "0", "--architecture", "mlp",
        "--checkpoint", str(sensitivity_checkpoint), *rules,
    ])
    report["sensitivity"] = evaluate(
        "sensitivity", sensitivity_checkpoint, 20,
        screen["validation_seed"], 100, True,
    )
    persist()

    for architecture in architectures:
        for seed in seeds:
            name = "%s-seed-%d" % (architecture, seed)
            checkpoint = output / (name + ".pt")
            command = [
                sys.executable, "-u", "-m", "src.cli.solo",
                "--architecture", architecture, "--seed", str(seed),
                "--checkpoint", str(checkpoint), "--log-every", "500", *rules,
            ]
            for argument in (
                "episodes", "warm_start_fraction", "batch_size", "replay_capacity",
                "updates_per_episode", "epsilon_decay_episodes",
            ):
                command.extend(["--" + argument.replace("_", "-"), str(screen[argument])])
            run(name, command)
            validation = evaluate(
                name + "-validation", checkpoint, screen["validation_games"],
                screen["validation_seed"], screen["warm_validation_games"],
            )
            report["runs"].append({
                "architecture": architecture, "seed": seed,
                "checkpoint": str(checkpoint),
                "parameter_count": validation["parameter_count"],
                "validation": validation,
            })
            persist()

    validation_means = {
        architecture: sum(
            trial["validation"]["greedy"]["win_rate"]
            for trial in report["runs"] if trial["architecture"] == architecture
        ) / len(seeds)
        for architecture in architectures
    }
    selected = max(architectures, key=lambda name: validation_means[name])
    report.update(selected_architecture=selected, validation_win_rates=validation_means)
    persist()
    for trial in report["runs"]:
        if trial["architecture"] in ("mlp", selected):
            name = "%s-seed-%d-test" % (trial["architecture"], trial["seed"])
            trial["test"] = evaluate(
                name, trial["checkpoint"], screen["test_games"], screen["test_seed"],
            )
            persist()

    selected_runs = [trial for trial in report["runs"] if trial["architecture"] == selected]
    baseline_runs = [trial for trial in report["runs"] if trial["architecture"] == "mlp"]
    outcomes = lambda runs, policy: [trial["test"][policy]["outcomes"] for trial in runs]
    report["versus_mlp"] = paired_comparison(outcomes(selected_runs, "greedy"), outcomes(baseline_runs, "greedy"))
    report["versus_random"] = paired_comparison(outcomes(selected_runs, "greedy"), outcomes(selected_runs, "random"))
    target = plan["promotion_target"]["absolute_win_rate_gain"]
    report["promotion_passed"] = selected != "mlp" and all(
        report[name]["absolute_gain"] >= target and report[name]["ci95"][0] > 0
        for name in ("versus_mlp", "versus_random")
    )
    report["status"] = "success"
    persist()
    print(json.dumps({
        key: report[key] for key in (
            "selected_architecture", "validation_win_rates", "versus_mlp",
            "versus_random", "promotion_passed", "elapsed_seconds",
        )
    }), flush=True)
    return report
