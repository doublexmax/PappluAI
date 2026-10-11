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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume")
    parser.add_argument("--players", type=int, nargs="+", choices=(2, 3, 4), default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--updates-per-match", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    try:
        runtime = load_runtime("src.training.league")
        runtime.torch.set_num_threads(1)
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
        settings = {
            name: getattr(args, name)
            for name in ("batch_size", "updates_per_match")
            if getattr(args, name) is not None
        }
        if args.players is not None:
            settings["player_counts"] = tuple(args.players)
        if args.resume:
            session = runtime.LeagueSession.load(
                args.resume,
                args.initial_model,
                args.opponent,
            )
            for name, value in settings.items():
                if value != getattr(session.config, name):
                    raise ValueError("resume cannot change %s" % name)
            if args.max_turns is not None and args.max_turns != session.config.game.max_turns:
                raise ValueError("resume cannot change max_turns")
            if args.seed is not None and args.seed != session.initial_seed:
                raise ValueError("resume cannot change seed")
        else:
            if any(
                (output_dir / name).exists()
                for name in ("latest-state.pt", "latest-model.pt")
            ):
                raise ValueError(
                    "output directory already has training outputs; use --resume"
                )
            if args.max_turns is not None:
                settings["game"] = runtime.GameConfig(max_turns=args.max_turns)
            session = runtime.LeagueSession(
                args.initial_model,
                opponent_paths=args.opponent,
                config=runtime.LeagueConfig(**settings),
                seed=41 if args.seed is None else args.seed,
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
