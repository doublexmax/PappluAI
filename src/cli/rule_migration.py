from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a discard-recycling learning regime."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_arg_parser().parse_args(
            list(argv) if argv is not None else None
        )
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    try:
        runtime = load_runtime("src.checkpoints.rule_migration")
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    print(
        json.dumps(
            runtime.migrate(Path(args.source), Path(args.output)),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
