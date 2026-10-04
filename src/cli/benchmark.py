from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired game evaluations with separate discard puzzles."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--warm-games", type=int, default=300)
    parser.add_argument("--warm-seed", type=int, default=600000)
    parser.add_argument("--sensitivity", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_arg_parser().parse_args(
            list(argv) if argv is not None else None
        )
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    try:
        runtime = load_runtime("src.evaluation.benchmark")
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    result = runtime.evaluate_checkpoint(
        args.checkpoint,
        args.games,
        args.seed,
        args.warm_games,
        args.warm_seed,
        args.sensitivity,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in (
                    "greedy",
                    "random",
                    "warm_greedy",
                    "warm_random",
                    "warm_oracle",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
