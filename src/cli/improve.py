from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import sys
from typing import Any, Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finite champion-league improvement cycles with guarded promotion."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial-model")
    source.add_argument("--resume")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--max-cycles", type=int, default=64)
    parser.add_argument(
        "--league-matches",
        "--league-matches-per-cycle",
        dest="league_matches_per_cycle",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--research-episodes",
        "--research-episodes-per-cycle",
        dest="research_episodes_per_cycle",
        type=int,
        default=128,
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--selection-blocks", type=int, default=32)
    parser.add_argument("--confirmation-blocks", type=int, default=128)
    parser.add_argument("--solo-games", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    supplied_arguments = list(argv) if argv is not None else sys.argv[1:]
    try:
        args = parser.parse_args(supplied_arguments)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    try:
        runtime = load_runtime("src.training.improve")
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 2

    output_dir = Path(args.output_dir)
    stop = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        stop[0] = True

    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, request_stop)
        if args.resume:
            controller = runtime.ImprovementController.load(
                args.resume,
                output_dir,
            )
            _validate_resume_options(
                runtime,
                controller,
                args,
                supplied_arguments,
            )
        else:
            controller = runtime.ImprovementController(
                args.initial_model,
                output_dir,
                opponent_paths=args.opponent,
                config=runtime.ImproveConfig(
                    max_cycles=args.max_cycles,
                    league_matches_per_cycle=args.league_matches_per_cycle,
                    research_episodes_per_cycle=args.research_episodes_per_cycle,
                    seed=args.seed,
                    selection_blocks=args.selection_blocks,
                    confirmation_blocks=args.confirmation_blocks,
                    solo_games=args.solo_games,
                    checkpoint_every=args.checkpoint_every,
                ),
            )
        reason = controller.run(
            args.max_seconds,
            stop_requested=lambda: stop[0],
        )
        print(
            json.dumps(
                {
                    "status": reason,
                    "cycle": controller.cycle_index,
                    "champion_id": controller.registry.champion().id,
                    "league_matches": controller.league.completed_matches,
                    "research_episodes": controller.research.total_episodes,
                    "promotions": controller.promotions,
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


def _validate_resume_options(
    runtime,
    controller,
    args,
    supplied_arguments,
) -> None:
    explicit = {
        value.split("=", 1)[0]
        for value in supplied_arguments
        if value.startswith("--")
    }
    learning_options = {
        "--league-matches": "league_matches_per_cycle",
        "--league-matches-per-cycle": "league_matches_per_cycle",
        "--research-episodes": "research_episodes_per_cycle",
        "--research-episodes-per-cycle": "research_episodes_per_cycle",
        "--selection-blocks": "selection_blocks",
        "--confirmation-blocks": "confirmation_blocks",
        "--solo-games": "solo_games",
        "--seed": "seed",
    }
    for option, field in learning_options.items():
        if option in explicit and getattr(args, field) != getattr(controller.config, field):
            raise ValueError("Resume cannot change %s" % option)
    for path in args.opponent:
        if path == runtime.RANDOM_OPPONENT:
            continue
        digest = runtime.file_sha256(Path(path))
        record = controller.registry.get(digest[:20])
        if record.sha256 != digest or (
            record.origin != "frozen_opponent"
            and record.id != controller.baseline_id
        ):
            raise ValueError(
                "Resume cannot add or replace the initial opponent pool"
            )
    changes = {}
    if "--max-cycles" in explicit:
        runtime.require_checkpoint_int(
            args.max_cycles,
            "max_cycles",
            minimum=1,
        )
        if args.max_cycles < controller.cycle_index:
            raise ValueError("max_cycles cannot be below completed cycles")
        changes["max_cycles"] = args.max_cycles
    if "--checkpoint-every" in explicit:
        runtime.require_checkpoint_int(
            args.checkpoint_every,
            "checkpoint_every",
            minimum=1,
        )
        changes["checkpoint_every"] = args.checkpoint_every
    if changes:
        controller.config = replace(controller.config, **changes)


if __name__ == "__main__":
    raise SystemExit(main())
