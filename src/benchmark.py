"""Paired game evaluations, with discard puzzles reported separately."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import List, Optional, Sequence, Tuple

from src.environment import GameConfig, PappluEnv, Phase, discard_action
from src.evaluate import hand_reward


@dataclass(frozen=True)
class PolicyResults:
    outcomes: Tuple[int, ...]
    stock_draws: int
    discard_draws: int

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "games": len(self.outcomes),
            "wins": sum(self.outcomes),
            "win_rate": sum(self.outcomes) / len(self.outcomes),
        }


def evaluate_games(
    network,
    config: GameConfig,
    games: int,
    seed: int,
    policy: str = "greedy",
    warm_start: bool = False,
) -> PolicyResults:
    from src.model import select_greedy_action, select_random_legal

    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise ValueError("games must be a positive integer")
    if policy not in ("greedy", "random", "oracle"):
        raise ValueError("policy must be greedy, random, or oracle")
    if policy == "oracle" and not warm_start:
        raise ValueError("oracle is only a discard-puzzle sensitivity check")
    env = PappluEnv(config)
    outcomes: List[int] = []
    draws = [0, 0]
    was_training = network.training
    network.eval()
    try:
        for index in range(games):
            obs = (
                env.reset_warm_start(seed=seed + index)
                if warm_start else env.reset(seed=seed + index)
            )
            rng = random.Random(seed + index + 10_000_000)
            while not obs.done:
                if policy == "greedy":
                    action = select_greedy_action(network, obs)
                elif policy == "random":
                    action = select_random_legal(obs, rng)
                else:
                    action = None
                    for face, count in enumerate(obs.hand):
                        if count:
                            hand = list(obs.hand)
                            hand[face] -= 1
                            if hand_reward(
                                hand, obs.joker, config.required_sequences,
                                config.cards_in_hand,
                            ):
                                action = discard_action(face)
                                break
                    if action is None:
                        raise RuntimeError("warm-start puzzle has no winning discard")
                if obs.phase is Phase.DRAW:
                    draws[action] += 1
                obs = env.step(action)
            outcomes.append(int(obs.won))
    finally:
        network.train(was_training)
    return PolicyResults(tuple(outcomes), draws[0], draws[1])


def evaluate_checkpoint(
    path: str,
    games: int = 300,
    seed: int = 300000,
    warm_games: int = 300,
    warm_seed: int = 600000,
    sensitivity: bool = False,
) -> dict:
    import torch
    from src.model import load_checkpoint

    if isinstance(warm_games, bool) or not isinstance(warm_games, int) or warm_games < 0:
        raise ValueError("warm_games must be a nonnegative integer")
    if sensitivity and not warm_games:
        raise ValueError("sensitivity requires positive warm_games")
    torch.set_num_threads(1)
    network, config, _ = load_checkpoint(path)
    result = {
        "checkpoint": path,
        "game_config": config.to_dict(),
        "architecture": network.architecture,
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
        "game_seed": seed,
        "warm_seed": warm_seed,
        "greedy": evaluate_games(network, config, games, seed).to_dict(),
        "random": evaluate_games(network, config, games, seed, policy="random").to_dict(),
    }
    if warm_games:
        result["warm_greedy"] = evaluate_games(
            network, config, warm_games, warm_seed, warm_start=True,
        ).to_dict()
        result["warm_random"] = evaluate_games(
            network, config, warm_games, warm_seed, policy="random", warm_start=True,
        ).to_dict()
        if sensitivity:
            result["warm_oracle"] = evaluate_games(
                network, config, warm_games, warm_seed, policy="oracle", warm_start=True,
            ).to_dict()
            if result["warm_oracle"]["win_rate"] != 1.0:
                raise RuntimeError("discard-puzzle ruler failed its positive control")
            if result["warm_random"]["win_rate"] >= 0.95:
                raise RuntimeError("discard-puzzle ruler does not separate random play")
    return result


def paired_comparison(
    candidate: Sequence[Sequence[int]],
    control: Sequence[Sequence[int]],
    samples: int = 2000,
    seed: int = 91827,
) -> dict:
    """Bootstrap both training seeds and shared evaluation deals."""
    if not candidate or len(candidate) != len(control):
        raise ValueError("paired comparison requires equal, nonempty seed groups")
    games = len(candidate[0])
    if games == 0 or any(len(row) != games for row in (*candidate, *control)):
        raise ValueError("paired outcomes must have the same positive game count")
    if any(value not in (0, 1) for row in (*candidate, *control) for value in row):
        raise ValueError("outcomes must be binary")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise ValueError("samples must be an integer >= 100")
    differences = [
        [a - b for a, b in zip(left, right)]
        for left, right in zip(candidate, control)
    ]
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        rows = rng.choices(differences, k=len(differences))
        indices = rng.choices(range(games), k=games)
        deltas.append(
            sum(sum(row[index] for index in indices) for row in rows)
            / (len(rows) * games)
        )
    deltas.sort()
    total = len(candidate) * games
    candidate_rate = sum(map(sum, candidate)) / total
    control_rate = sum(map(sum, control)) / total
    return {
        "candidate_win_rate": candidate_rate,
        "control_win_rate": control_rate,
        "absolute_gain": candidate_rate - control_rate,
        "ci95": [deltas[int(samples * 0.025)], deltas[int(samples * 0.975)]],
        "training_seeds": len(candidate),
        "games_per_seed": games,
        "method": "paired bootstrap over training seeds and shared deals",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--warm-games", type=int, default=300)
    parser.add_argument("--warm-seed", type=int, default=600000)
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(
        args.checkpoint, args.games, args.seed, args.warm_games,
        args.warm_seed, args.sensitivity,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in ("greedy", "random", "warm_greedy", "warm_random", "warm_oracle")
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
