from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from src.cli._runtime import MissingTrainingDependency, load_runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Papplu Q-network with episodic Monte Carlo return regression "
            "(not DQN). Rewards come from src.evaluate.hand_reward."
        )
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=5000)
    parser.add_argument(
        "--replay-sampling",
        choices=("transition", "episode"),
        default="transition",
        help="Replay sampling unit (default transition).",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=200)
    parser.add_argument(
        "--warm-start-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of episodes that start as winning-hand-plus-extra discard "
            "puzzles (default 0 = random deals)."
        ),
    )
    parser.add_argument("--updates-per-episode", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads for the small network (default 1).",
    )
    parser.add_argument(
        "--architecture",
        choices=("mlp", "wide_mlp", "suit_conv"),
        default=None,
        help="Model architecture. A loaded checkpoint supplies the default.",
    )
    parser.add_argument(
        "--num-decks",
        type=int,
        help="Decks (default 3, or loaded checkpoint).",
    )
    parser.add_argument(
        "--cards-in-hand",
        type=int,
        help="Hand size (default 21, or loaded checkpoint).",
    )
    parser.add_argument(
        "--required-sequences",
        type=int,
        help="Required pure sequences (default 5, or loaded checkpoint).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Turn budget (default 40, or loaded checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to write the trained checkpoint after training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Load weights and game settings to fine-tune. Replay and optimizer "
            "start fresh. Explicit game settings must match the checkpoint."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Held-out greedy vs random games after training (0 skips).",
    )
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1

    try:
        runtime = load_runtime("src.training.solo")
        game_settings = {
            name: getattr(args, name)
            for name in (
                "num_decks",
                "cards_in_hand",
                "required_sequences",
                "max_turns",
            )
            if getattr(args, name) is not None
        }
        if args.load and game_settings:
            _, saved_config, _ = runtime.load_checkpoint(
                args.load,
                expected_architecture=args.architecture,
            )
            config = runtime.GameConfig(
                **{**saved_config.to_dict(), **game_settings}
            )
        else:
            config = (
                runtime.GameConfig(**game_settings)
                if not args.load
                else None
            )
        summary = runtime.train(
            episodes=args.episodes,
            seed=args.seed,
            gamma=args.gamma,
            lr=args.lr,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            replay_sampling=args.replay_sampling,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
            warm_start_fraction=args.warm_start_fraction,
            updates_per_episode=args.updates_per_episode,
            grad_clip=args.grad_clip,
            device=args.device,
            torch_threads=args.torch_threads,
            config=config,
            checkpoint_path=args.checkpoint or None,
            load_path=args.load or None,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            log_every=args.log_every,
            architecture=args.architecture,
        )
    except MissingTrainingDependency as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except (TypeError, ValueError, OSError, RuntimeError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
