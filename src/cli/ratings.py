from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report reproducible multiplayer ratings from persisted match evidence.")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--cap-policy", choices=("undecided", "points", "weighted-points", "tie", "exclude"), default="undecided")
    parser.add_argument("--cap-weight", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=81)
    args = parser.parse_args(argv)
    try:
        runtime = load_runtime("src.evaluation.ratings")
    except MissingTrainingDependency as error:
        print(error, file=sys.stderr)
        return 2
    result = runtime.write_rating_report(
        args.database, args.output, args.reference, args.cap_policy,
        args.protocol, args.samples, args.seed, args.cap_weight,
    )
    for report in result["reports"]:
        print("%d players, cap %d, %s" % (
            report["config"]["players"], report["config"]["game"]["max_turns"], report["scope"],
        ))
        for row in report["rows"]:
            rating = "unrated" if row["rating"] is None else "%.1f" % row["rating"]
            interval = "pending" if row["interval95"] is None else "[%.1f, %.1f]" % tuple(row["interval95"])
            print("%s  %s  %s  games=%d blocks=%d %s" % (
                row["label"], rating, interval, row["games"], row["rating_blocks"], row["status"],
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
