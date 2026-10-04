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
        description="Resumable full-hand curriculum training for Papplu."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--initial-model",
        help="Start from model weights. Omit with --resume to train from scratch.",
    )
    source.add_argument("--resume")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=1_000_000)
    parser.add_argument("--max-seconds", type=float, default=82_800)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--validate-every",
        type=int,
        default=None,
        help="Validation cadence. New sessions default to 1000 episodes.",
    )
    parser.add_argument(
        "--validation-games",
        type=int,
        default=None,
        help="Games per policy and validation source. New sessions default to 64.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Full-game turn limit. New sessions default to 60.",
    )
    parser.add_argument(
        "--recycle-discard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recycle older discards into stock. New sessions default to enabled.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optimizer learning rate. New sessions default to 0.001.",
    )
    parser.add_argument(
        "--curriculum-fraction",
        type=float,
        default=None,
        help="Fraction of curriculum deals. New sessions default to 0.75.",
    )
    parser.add_argument(
        "--initial-stage",
        type=int,
        default=None,
        help="Initial curriculum stage 0..4. New sessions default to 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed. New sessions default to 41.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    try:
        runtime = load_runtime("src.training.curriculum")
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 2

    session = None
    output_dir = Path(args.output_dir)
    started = time.monotonic()
    stop = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        stop[0] = True

    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, request_stop)
        runtime._nonnegative_int(args.max_episodes, "max_episodes")
        runtime._positive_int(args.checkpoint_every, "checkpoint_every")
        if (
            isinstance(args.max_seconds, bool)
            or not isinstance(args.max_seconds, (int, float))
            or not math.isfinite(float(args.max_seconds))
            or args.max_seconds < 0
        ):
            raise ValueError("max_seconds must be finite and nonnegative")
        if args.resume:
            session = runtime.TrainingSession.load(args.resume)
            _validate_resume_arguments(runtime, session, args)
            initialize = session.baseline_validation is None
        else:
            managed_outputs = (
                "latest-state.pt",
                "latest-model.pt",
                "best-model.pt",
                "baseline-model.pt",
                "status.json",
                "metrics.jsonl",
            )
            if any((output_dir / name).exists() for name in managed_outputs):
                raise ValueError(
                    "output directory already has training outputs; use --resume"
                )
            config = runtime.TrainingConfig(
                max_turns=60 if args.max_turns is None else args.max_turns,
                recycle_discard=(
                    True
                    if args.recycle_discard is None
                    else args.recycle_discard
                ),
                learning_rate=(
                    1e-3 if args.learning_rate is None else args.learning_rate
                ),
                curriculum_fraction=(
                    0.75
                    if args.curriculum_fraction is None
                    else args.curriculum_fraction
                ),
                validation_games=(
                    64 if args.validation_games is None else args.validation_games
                ),
                validate_every=(
                    1_000 if args.validate_every is None else args.validate_every
                ),
            )
            session = runtime.TrainingSession(
                config=config,
                seed=41 if args.seed is None else args.seed,
                initial_model=args.initial_model,
                initial_stage_index=(
                    0 if args.initial_stage is None else args.initial_stage
                ),
            )
            initialize = True
        reason = runtime.run(
            session,
            output_dir,
            max_episodes=args.max_episodes,
            max_seconds=args.max_seconds,
            checkpoint_every=args.checkpoint_every,
            stop_requested=lambda: stop[0],
            initialize=initialize,
        )
        print(
            json.dumps(
                {
                    "status": reason,
                    "episodes": session.total_episodes,
                    "updates": session.total_updates,
                    "stage": session.stage_index,
                    "distance": session.current_distance,
                    "best_score": session.best_score,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (TypeError, ValueError, OSError, RuntimeError, KeyError) as exc:
        runtime._publish_error(output_dir, session, str(exc), started)
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 2
    except BaseException as exc:
        runtime._publish_error(output_dir, session, str(exc), started)
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _validate_resume_arguments(
    runtime,
    session,
    args: argparse.Namespace,
) -> None:
    requested = {
        "max_turns": args.max_turns,
        "recycle_discard": args.recycle_discard,
        "learning_rate": args.learning_rate,
        "curriculum_fraction": args.curriculum_fraction,
        "validation_games": args.validation_games,
        "validate_every": args.validate_every,
    }
    for name, value in requested.items():
        if value is not None and value != getattr(session.config, name):
            raise ValueError(
                "resume cannot change %s from %r to %r"
                % (name, getattr(session.config, name), value)
            )
    if args.seed is not None and args.seed != session.initial_seed:
        raise ValueError(
            "resume seed %d does not match saved seed %d"
            % (args.seed, session.initial_seed)
        )
    if args.initial_stage is not None:
        raise ValueError("--initial-stage is only valid for a new session")


if __name__ == "__main__":
    raise SystemExit(main())
