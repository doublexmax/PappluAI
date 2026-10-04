from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time
from typing import Any, Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable shared-deck league training against frozen opponents."
    )
    parser.add_argument("--initial-model", required=True)
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--matches", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, default=82_800)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--resume")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    try:
        runtime = load_runtime("src.training.league")
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 2

    stop = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        stop[0] = True

    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, request_stop)
        runtime.require_checkpoint_int(args.matches, "matches", minimum=0)
        runtime.require_checkpoint_int(
            args.checkpoint_every,
            "checkpoint_every",
            minimum=1,
        )
        if (
            isinstance(args.max_seconds, bool)
            or not isinstance(args.max_seconds, (int, float))
            or not math.isfinite(float(args.max_seconds))
            or args.max_seconds < 0
        ):
            raise ValueError("max_seconds must be finite and nonnegative")
        output_dir = Path(args.output_dir)
        session = (
            runtime.LeagueSession.load(
                args.resume,
                args.initial_model,
                args.opponent,
                expected_config=runtime.LeagueConfig(),
            )
            if args.resume
            else runtime.LeagueSession(
                args.initial_model,
                opponent_paths=args.opponent,
                seed=args.seed,
            )
        )
        runtime.save_session_checkpoint(session, output_dir)
        target = session.completed_matches + args.matches
        deadline = time.monotonic() + float(args.max_seconds)
        reason = "matches"
        while session.completed_matches < target:
            if stop[0]:
                reason = "stopped"
                break
            if time.monotonic() >= deadline:
                reason = "budget"
                break
            try:
                result = session.train_match(deadline=deadline)
            except runtime.MatchInterrupted:
                reason = "budget"
                break
            print(json.dumps(result, sort_keys=True), flush=True)
            if session.completed_matches % args.checkpoint_every == 0:
                runtime.save_session_checkpoint(session, output_dir)
        runtime.save_session_checkpoint(session, output_dir)
        print(
            json.dumps(
                {
                    "status": reason,
                    "matches": session.completed_matches,
                    "updates": session.total_updates,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
