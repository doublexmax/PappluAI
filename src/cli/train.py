from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import tempfile
from typing import Any, Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


CONFIG_FLAGS: tuple[tuple[str, type], ...] = (
    ("--num-decks", int),
    ("--cards-in-hand", int),
    ("--required-sequences", int),
    ("--max-turns", int),
    ("--round-episodes", int),
    ("--updates-per-episode", int),
    ("--batch-size", int),
    ("--replay-capacity", int),
    ("--learning-rate", float),
    ("--weight-decay", float),
    ("--gamma", float),
    ("--grad-clip", float),
    ("--epsilon-start", float),
    ("--epsilon-end", float),
    ("--epsilon-decay-episodes", int),
    ("--warm-start-fraction", float),
    ("--curriculum-fraction", float),
    ("--initial-stage", int),
    ("--stage-passes", int),
    ("--stage-margin", float),
    ("--stage-floor", float),
    ("--validate-every", int),
    ("--validation-games", int),
    ("--pfsp-champion-share", float),
    ("--pfsp-anchor-share", float),
    ("--pfsp-exponent", float),
    ("--pfsp-prior-games", float),
    ("--cycle-episodes", int),
    ("--selection-blocks", int),
    ("--confirmation-blocks", int),
    ("--solo-games", int),
    ("--gate-samples", int),
    ("--gate-workers", int),
    ("--rebase-after-rejections", int),
    ("--checkpoint-every", int),
    ("--seed", int),
    ("--torch-threads", int),
    ("--device", str),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.train",
        description="Train card_mlp v5 with Q-only DMC across solo, curriculum, PFSP league, and promotion gates.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--initial-model", type=str, default=None)
    parser.add_argument("--opponent", dest="opponents", action="append", default=[])
    parser.add_argument("--players", dest="player_counts", type=int, nargs="+", default=None)
    parser.add_argument(
        "--curriculum-distances",
        dest="curriculum_distances",
        type=int,
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--recycle-discard",
        dest="recycle_discard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--promote", dest="promote", action="store_true", default=None)
    parser.add_argument("--width", type=int, default=None, help="Model width for a fresh run (default: 64).")
    parser.add_argument("--blocks", type=int, default=None, help="Card blocks for a fresh run (default: 2).")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    for flag, flag_type in CONFIG_FLAGS:
        parser.add_argument(flag, type=flag_type, default=None)
    return parser


def _resolve_initial(args: argparse.Namespace, runtime: Any) -> Any:
    spec_fields = {
        name: getattr(args, name)
        for name in ("width", "blocks")
        if getattr(args, name) is not None
    }
    if args.initial_model is not None and spec_fields:
        raise ValueError("cannot combine --initial-model with --width or --blocks")
    if args.initial_model is not None:
        return Path(args.initial_model)
    if spec_fields:
        defaults = runtime.ModelSpec()
        return runtime.ModelSpec(
            width=spec_fields.get("width", defaults.width),
            blocks=spec_fields.get("blocks", defaults.blocks),
        )
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_exc:
        return int(exit_exc.code) if isinstance(exit_exc.code, int) else 1

    if args.output_dir is None and args.checkpoint is None:
        print("error: supply --output-dir or --checkpoint", file=sys.stderr)
        return 2

    try:
        runtime = load_runtime("src.training.train")
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    config_names = set(runtime.TrainConfig.__dataclass_fields__)
    overrides = {
        name: getattr(args, name)
        for name in config_names
        if getattr(args, name, None) is not None
    }
    stopping = [False]
    previous_handlers: dict[int, Any] = {}

    def _request_stop(_signum: int, _frame: Any) -> None:
        stopping[0] = True

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                previous_handlers[sig] = signal.signal(sig, _request_stop)
            except ValueError:
                pass

    try:
        initial = _resolve_initial(args, runtime)
        opponents = [Path(path) for path in (args.opponents or ())]
        if args.output_dir is not None:
            trainer = runtime.Trainer.open(
                Path(args.output_dir),
                overrides,
                initial=initial,
                opponents=opponents,
            )
            max_episodes = args.max_episodes
            if (
                max_episodes is None
                and args.max_seconds is None
                and args.max_cycles is None
            ):
                max_episodes = trainer.progress.episodes + trainer.config.round_episodes
            reason = trainer.run(
                max_seconds=args.max_seconds,
                max_episodes=max_episodes,
                max_cycles=args.max_cycles,
                stop=lambda: stopping[0],
            )
            if args.checkpoint is not None:
                trainer.export_model(Path(args.checkpoint), role="checkpoint")
            summary = {"status": reason, **trainer.summary()}
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                trainer = runtime.Trainer.open(
                    Path(temp_dir),
                    overrides,
                    initial=initial,
                    opponents=opponents,
                )
                max_episodes = (
                    0
                    if (args.max_episodes is None and args.max_seconds is None and args.max_cycles is None)
                    else args.max_episodes
                )
                reason = trainer.run(
                    max_seconds=args.max_seconds,
                    max_episodes=max_episodes,
                    max_cycles=args.max_cycles,
                    stop=lambda: stopping[0],
                )
                trainer.export_model(Path(args.checkpoint), role="checkpoint")
                summary = {
                    "status": reason,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    **trainer.summary(),
                }
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
