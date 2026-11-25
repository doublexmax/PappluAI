from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run headless policy matches or rate a league of GameTransformer checkpoints."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="repeat once per seat, or once per checkpoint with --league; 'random' is only for single matches",
    )
    parser.add_argument("--league", action="store_true", help="rate all checkpoint pairs on mirror-duplicate deals")
    parser.add_argument("--label", action="append", help="league label; repeat in checkpoint order")
    parser.add_argument("--players", nargs="+", type=int, choices=range(2, 7), default=(2, 3, 4))
    parser.add_argument("--anchor-elo", type=float, default=1000.0, help="league rating of the first checkpoint")
    parser.add_argument("--workers", type=_positive_int, default=1, help="parallel league matchups")
    parser.add_argument("--games", type=_positive_int, help="deals per matchup; defaults to 48 in league mode")
    parser.add_argument("--seed", type=int, help="defaults to 970000 in league mode and 0 otherwise")
    parser.add_argument("--max-turns", type=_positive_int)
    parser.add_argument("--tables", type=_positive_int, default=64)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _single_matches(runtime: ModuleType, args: argparse.Namespace) -> dict:
    players = len(args.model)
    policies = tuple(runtime.resolve_policy(value, precision=args.precision) for value in args.model)
    labels = [Path(value).stem for value in args.model]
    game = runtime.match_game(policies, args.max_turns)
    seed = args.seed if args.seed is not None else 0
    deals = (
        runtime.Deal(seed + i, tuple((i + seat) % players for seat in range(players)))
        for i in range(args.games)
    )
    games = []
    results = runtime.run_matches(deals, policies, game, tables=args.tables)
    for result in sorted(results, key=lambda row: row.deal.seed):
        deal = result.deal
        winner_agent = None if result.winner is None else deal.seats[result.winner]
        games.append({
            "seed": deal.seed,
            "seat_order": list(deal.seats),
            "winner_seat": result.winner,
            "winner_agent": winner_agent,
            "winner_label": None if winner_agent is None else labels[winner_agent],
            "terminal_reason": result.terminal_reason,
            "seat_turns": result.seat_turns,
            "action_count": result.action_count,
            "stock_remaining": result.stock_remaining,
            "telemetry": asdict(result.telemetry),
            "utilities": result.outcome.utilities(),
            "net_points": result.outcome.net_points(),
        })
    return {
        "agents": [
            {"index": index, "label": label, "model": model}
            for index, (label, model) in enumerate(zip(labels, args.model))
        ],
        "config": {"players": players, "game": game.to_dict()},
        "games": games,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        if args.league:
            if len(args.model) < 2:
                parser.error("--league requires at least two --model checkpoints")
            if args.label is not None and len(args.label) != len(args.model):
                parser.error("--league requires one --label per --model when labels are supplied")
        else:
            if not 1 <= len(args.model) <= 6:
                parser.error("--model must be repeated 1 through 6 times")
            if args.games is None:
                parser.error("--games is required without --league")
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    try:
        runtime = load_runtime("src.evaluation.arena")
        runtime.torch.set_num_threads(1)
        if args.league:
            report = runtime.rate_checkpoints(
                args.model, labels=args.label, players=args.players, deals=args.games or 48,
                seed=args.seed if args.seed is not None else 970_000, max_turns=args.max_turns,
                tables=args.tables, precision=args.precision, anchor_elo=args.anchor_elo, workers=args.workers,
            )
        else:
            report = _single_matches(runtime, args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (MissingTrainingDependency, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
