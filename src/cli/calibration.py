from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate multiplayer draw limits using frozen models and paired deal blocks."
    )
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--caps", type=int, nargs="+", default=[30, 60, 120])
    parser.add_argument("--players", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9_040_500_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--match-seconds", type=float, default=120)
    parser.add_argument("--max-extra-completion-rate", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        runtime = load_runtime("src.evaluation.calibration")
    except MissingTrainingDependency as error:
        print(error, file=sys.stderr)
        return 2
    config = runtime.CalibrationConfig(
        caps=tuple(args.caps), player_counts=tuple(args.players),
        blocks=args.blocks, seed=args.seed, workers=args.workers,
        match_seconds=args.match_seconds,
        max_extra_completion_rate=args.max_extra_completion_rate,
    )
    report = runtime.run_calibration(args.models_dir, args.output_dir, config)
    for row in report["groups"]:
        print(
            "%d players, cap %d: %d/%d declarations, %d capped, %d games with refills"
            % (row["players"], row["cap"], row["declarations"],
               row["completed_games"], row["capped_games"], row["refill_games"])
        )
    for recommendation in report["recommendations"]:
        print(
            "%d players: %s, cap=%s"
            % (recommendation["players"], recommendation["status"], recommendation["cap"])
        )
    return 0 if report["execution_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
