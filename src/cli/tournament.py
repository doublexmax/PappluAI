from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def _cap(value: str) -> tuple[int, int]:
    try:
        players, turns = (int(part) for part in value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("cap must be PLAYERS:TURNS") from error
    if not 2 <= players <= 6 or turns < 1:
        raise argparse.ArgumentTypeError("cap requires 2..6 players and positive turns")
    return players, turns


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Continuously evaluate frozen Papplu models and persist full placement evidence.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, action="append", type=Path)
    parser.add_argument("--cap", required=True, action="append", type=_cap)
    parser.add_argument("--reference", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--blocks", type=int, default=16)
    mode.add_argument("--continuous", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--match-seconds", type=float, default=300)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=9_041_100_000)
    parser.add_argument("--cap-policy", choices=("undecided", "points", "weighted-points", "tie", "exclude"), default="undecided")
    parser.add_argument("--cap-weight", type=float, default=0.25)
    args = parser.parse_args(argv)
    if len(dict(args.cap)) != len(args.cap):
        parser.error("each player count must have exactly one cap")
    try:
        runtime = load_runtime("src.evaluation.tournament")
    except MissingTrainingDependency as error:
        print(error, file=sys.stderr)
        return 2
    result = runtime.run_tournament(
        args.root, args.snapshots, dict(args.cap), args.reference,
        workers=args.workers, blocks=None if args.continuous else args.blocks,
        max_seconds=args.max_seconds, match_seconds=args.match_seconds,
        poll_seconds=args.poll_seconds, seed=args.seed,
        cap_policy=args.cap_policy, cap_weight=args.cap_weight,
    )
    print("%s: %d scored games, %d awaiting scores, %d failed jobs" % (
        result["phase"], result["scored_games"], result["awaiting_scores"], result["failed_jobs"],
    ))
    return 0 if result["phase"] in ("completed", "paused") else 2


if __name__ == "__main__":
    raise SystemExit(main())
