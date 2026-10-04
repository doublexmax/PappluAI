from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-budget architecture comparison."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_arg_parser().parse_args(
            list(argv) if argv is not None else None
        )
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    try:
        runtime = load_runtime("src.evaluation.compare")
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    runtime.run_comparison(plan, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
