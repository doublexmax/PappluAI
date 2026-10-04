from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

from src.cli._runtime import MissingTrainingDependency, load_runtime


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run headless multiplayer policy matches."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="checkpoint path or the sentinel 'random'; repeat once per seat",
    )
    parser.add_argument("--games", type=_positive_int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=_positive_int, default=60)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    players = len(args.model)
    if players < 2 or players > 6:
        parser.error("--model must be repeated 2 through 6 times")
    try:
        runtime = load_runtime("src.evaluation.arena")
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    policies: Tuple[runtime.Policy, ...] = tuple(
        runtime.RandomPolicy()
        if value == "random"
        else runtime.CheckpointPolicy(value)
        for value in args.model
    )
    labels = [
        "random" if value == "random" else Path(value).stem
        for value in args.model
    ]
    config = runtime.MatchConfig(
        game=runtime.GameConfig(max_turns=args.max_turns),
        players=players,
    )
    games = []
    for index in range(args.games):
        rotation = index % players
        seated = policies[rotation:] + policies[:rotation]
        result = runtime.play_match(
            seated,
            config,
            seed=args.seed + index,
        )
        winner_agent = (
            None
            if result.winner is None
            else (rotation + result.winner) % players
        )
        games.append(
            {
                "seed": result.seed,
                "seat_order": [
                    (rotation + seat) % players
                    for seat in range(players)
                ],
                "winner_seat": result.winner,
                "winner_agent": winner_agent,
                "winner_label": (
                    None if winner_agent is None else labels[winner_agent]
                ),
                "terminal_reason": result.terminal_reason,
                "seat_turns": result.seat_turns,
                "action_count": result.action_count,
                "stock_remaining": result.stock_remaining,
            }
        )

    report = {
        "agents": [
            {"index": index, "label": label, "model": model}
            for index, (label, model) in enumerate(
                zip(labels, args.model)
            )
        ],
        "config": {
            "players": players,
            "game": config.game.to_dict(),
        },
        "games": games,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
