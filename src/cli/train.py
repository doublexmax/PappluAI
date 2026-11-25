from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

from src.cli._runtime import load_runtime


_SHAPE = ("width", "layers", "heads", "ffn")
_GAME = ("num_decks", "cards_in_hand", "required_sequences", "max_turns", "recycle_discard")
_PLAN = ("player_counts", "curriculum_share", "curriculum_max_distance", "champion_share", "uniform_mix", "batch_decisions")
_HOST = ("tables", "torch_threads", "precision", "device")
_INTS = (
    "actors", "actor_threads", "batch_size", "unroll_steps", "min_buffer_decisions", "buffer_decisions",
    "max_policy_lag", "anchor_every", "sync_every", "publish_every", "checkpoint_every", "eval_every",
    "eval_deals", "seed", "warmup_steps",
)
_FLOATS = ("learning_rate", "weight_decay", "grad_clip", "value_weight", "eta",
           "entropy_weight", "rho_bar", "c_bar", "promote_margin")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GameTransformer with V-trace self-play.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path, help="start or resume a learner (supports torchrun on CPU)")
    mode.add_argument("--collect", type=Path, help="write completed batches to this inbox")
    parser.add_argument("--model", type=Path, help="latest-model.pt for collection")
    parser.add_argument("--worker", type=int, help="nonnegative actor ID for collection")
    parser.add_argument("--initial-model", type=Path, help="seed a fresh learner from compatible weights")
    parser.add_argument("--players", dest="player_counts", type=int, nargs="+")
    parser.add_argument("--recycle-discard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto-curriculum", action=argparse.BooleanOptionalAction, default=None,
                        help="advance the collection plan on 0-ply frontier mastery")
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--max-seconds", type=float)
    for name in (
        *_SHAPE, *_INTS, *_GAME[:-1], "tables", "torch_threads",
        "curriculum_max_distance", "batch_decisions", "max_updates", "max_batches",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in (*_FLOATS, "curriculum_share", "champion_share", "uniform_mix"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    return parser


def _selected(args: argparse.Namespace, names: tuple[str, ...]) -> dict:
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def main(argv=None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        learner = _selected(args, (*_SHAPE, *_GAME, *_PLAN, *_INTS, *_FLOATS,
                                   "auto_curriculum", "initial_model", "max_updates"))
        if args.collect is not None:
            if args.model is None or args.worker is None or args.worker < 0:
                parser.error("--collect requires --model and a nonnegative --worker")
            if learner:
                parser.error("learner flags cannot be used with --collect")
            if args.max_batches is not None and args.max_batches < 1:
                parser.error("--max-batches must be positive")
        else:
            if any(value is not None for value in (args.model, args.worker, args.max_batches)):
                parser.error("--model, --worker, and --max-batches require --collect")
            if args.initial_model is not None and _selected(args, _SHAPE):
                parser.error("--initial-model cannot be combined with model shape flags")
            if args.max_updates is not None and args.max_updates < 0:
                parser.error("--max-updates must be nonnegative")
        if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
            parser.error("--max-seconds must be positive and finite")
    except SystemExit as exc:
        return int(exc.code)
    try:
        runtime = load_runtime("src.training.train")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        host = _selected(args, _HOST)
        if args.collect is not None:
            deadline = math.inf if args.max_seconds is None else time.monotonic() + args.max_seconds
            result = runtime.collect(
                args.collect, args.model, worker=args.worker, max_batches=args.max_batches,
                stop=lambda: stopping or time.monotonic() >= deadline, **host,
            )
        else:
            game, plan = _selected(args, _GAME), _selected(args, _PLAN)
            if "player_counts" in plan:
                plan["player_counts"] = tuple(plan["player_counts"])
            overrides = {**host, **_selected(args, (*_INTS, *_FLOATS, "auto_curriculum"))}
            if game:
                overrides["game"] = game
            if plan:
                overrides["plan"] = plan
            shape = _selected(args, _SHAPE)
            trainer = runtime.Trainer.open(
                args.output_dir, overrides, spec=runtime.ModelSpec(**shape) if shape else None, initial=args.initial_model,
            )
            reason = trainer.run(stop=lambda: stopping, max_updates=args.max_updates, max_seconds=args.max_seconds)
            if trainer.group.rank != 0:
                return 0
            result = {"status": reason, **trainer.summary()}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
