from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys

from src.cli._runtime import MissingTrainingDependency, load_runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train card_mlp with v8 encoding and champion-anchored fictitious self-play DMC. Reuse the output directory to resume.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--opponent", type=Path, action="append", default=[])
    parser.add_argument("--players", dest="player_counts", type=int, nargs="+")
    parser.add_argument("--actors", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--round-episodes", type=int)
    parser.add_argument("--updates-per-episode", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-decay-updates", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--self-play-share", type=float)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--validation-games", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-seconds", type=float)
    return parser


def main(argv=None) -> int:
    try:
        args = build_arg_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        runtime = load_runtime("src.training.train")
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    stopping = [False]

    def request_stop(_signum, _frame):
        stopping[0] = True

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.initial_model is not None and (args.width is not None or args.blocks is not None):
            raise ValueError("--initial-model cannot be combined with --width or --blocks")
        initial = args.initial_model
        if args.width is not None or args.blocks is not None:
            default = runtime.ModelSpec()
            initial = runtime.ModelSpec(default.width if args.width is None else args.width,
                                        default.blocks if args.blocks is None else args.blocks)
        overrides = {name: getattr(args, name) for name in runtime.TrainConfig.__dataclass_fields__
                     if getattr(args, name, None) is not None}
        with runtime.Trainer.open(args.output_dir, overrides, initial=initial, opponents=args.opponent) as trainer:
            episodes = args.max_episodes
            if episodes is None and args.max_seconds is None:
                episodes = trainer.episodes + trainer.config.round_episodes
            reason = trainer.run(max_episodes=episodes, max_seconds=args.max_seconds, stop=lambda: stopping[0])
            print(json.dumps({"status": reason, **trainer.summary()}, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
