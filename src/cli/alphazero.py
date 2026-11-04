from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from src.cli._runtime import load_runtime


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a policy/value position evaluator with PUCT self-play.",
    )
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--load", default="", help="Warm-start a dynamic_table model.")
    source.add_argument("--resume", default="", help="Resume a saved AlphaZero session.")
    parser.add_argument("--checkpoint", default="", help="Write model, optimizer, RNGs, and replay.")
    parser.add_argument("--export", default="", help="Write a standard dynamic_table model.")
    parser.add_argument("--opponent", action="append", default=[], help="Frozen opponent checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--num-decks", type=int)
    parser.add_argument("--cards-in-hand", type=int)
    parser.add_argument("--required-sequences", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--stock-outcomes", type=int, default=12)
    parser.add_argument("--opponent-outcomes", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--exploration-turns", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=5000)
    parser.add_argument("--updates-per-match", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--bootstrap-blend", type=float, default=0.65)
    parser.add_argument("--bootstrap-blend-end", type=float, default=0.20)
    parser.add_argument("--bootstrap-decay-matches", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        runtime = load_runtime("src.training.alphazero")
    except RuntimeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    return runtime.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
