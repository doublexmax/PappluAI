from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys

from src.cli._runtime import MissingTrainingDependency, load_runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train card_mlp v5 with DMC and winning-trajectory margins. Reuse the output directory to resume.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--best-model", type=Path)
    parser.add_argument("--collect-batch", type=Path)
    parser.add_argument("--start-episode", type=int, default=0)
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
    parser.add_argument("--margin-weight", type=float)
    parser.add_argument("--epsilons", type=float, nargs="+")
    parser.add_argument("--self-play-share", type=float)
    parser.add_argument("--max-pool", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--validation-games", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-seconds", type=float)
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        if args.collect_batch is None and args.output_dir is None:
            parser.error("--output-dir is required unless --collect-batch is supplied")
        if args.collect_batch is not None and args.initial_model is None:
            parser.error("--collect-batch requires --initial-model")
        if args.start_episode < 0:
            parser.error("--start-episode must be nonnegative")
        if args.collect_batch is None and (args.best_model is not None or args.start_episode):
            parser.error("--best-model and --start-episode require --collect-batch")
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
            defaults = runtime.ModelSpec()
            initial = runtime.ModelSpec(defaults.width if args.width is None else args.width,
                                        defaults.blocks if args.blocks is None else args.blocks)
        overrides = {name: getattr(args, name) for name in runtime.TrainConfig.__dataclass_fields__
                     if getattr(args, name, None) is not None}
        if args.collect_batch is not None:
            online, game, _ = runtime.load_checkpoint(str(args.initial_model))
            config = runtime.TrainConfig(**{**game.to_dict(), **overrides})
            if config.game != game:
                raise ValueError("initial model game_config mismatch")
            best_path = str(args.best_model or args.initial_model)
            pool = [str(path) for path in args.opponent]
            for path in [best_path, *pool]:
                runtime.load_checkpoint(path, expected_config=config.game)
            runtime.torch.set_num_threads(config.torch_threads)
            start, count = args.start_episode, config.round_episodes
            if config.actors == 1:
                trajectories = runtime.collect_round(config, online, best_path, pool, start, count)
            else:
                chunk = (count + config.actors - 1) // config.actors
                state = runtime.clone_model_state(online.state_dict())
                jobs = [(config, online.spec, state, best_path, pool, offset, min(chunk, start + count - offset))
                        for offset in range(start, start + count, chunk)]
                with runtime.ProcessPoolExecutor(max_workers=config.actors, initializer=runtime._actor_init) as executor:
                    trajectories = [t for batch in executor.map(runtime._actor_job, jobs) for t in batch]
            runtime.save_rollout_batch(args.collect_batch, episodes=count, trajectories=trajectories)
            print(json.dumps({"episodes": count, "trajectories": len(trajectories)}))
            return 0
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
