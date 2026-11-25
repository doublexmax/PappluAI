from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Sequence

import torch

from src.checkpoints.io import file_sha256
from src.evaluation.arena import CheckpointPolicy, Deal, Policy, RandomPolicy, Turn, run_matches
from src.evaluate import hand_reward
from src.game.environment import ENCODING_VERSION, GameConfig, discard_action
from src.model.spec import CHECKPOINT_VERSION


@dataclass(frozen=True)
class PolicyResults:
    outcomes: tuple[int, ...]
    mean_utility: float
    stock_draws: int
    discard_draws: int

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "games": len(self.outcomes),
            "wins": sum(self.outcomes),
            "win_rate": sum(self.outcomes) / len(self.outcomes),
        }


class _DiscardOracle:
    game = None

    def open(self) -> _DiscardOracle:
        return self

    def finish(self, stream: int) -> None:
        return None

    def act(self, turns: Sequence[Turn]) -> list[int]:
        actions = []
        for turn in turns:
            obs = turn.observation
            for face, count in enumerate(obs.hand):
                if count:
                    hand = list(obs.hand)
                    hand[face] -= 1
                    if hand_reward(hand, obs.joker, obs.config.required_sequences, obs.config.cards_in_hand):
                        actions.append(discard_action(face))
                        break
            else:
                raise RuntimeError("discard puzzle has no winning discard")
        return actions


def evaluate_games(
    policy: Policy, config: GameConfig, games: int, seed: int, *,
    warm_start: bool = False, tables: int = 64,
) -> PolicyResults:
    if type(games) is not int or games <= 0:
        raise ValueError("games must be a positive integer")
    deals = (Deal(seed + i, (0,), curriculum=0 if warm_start else None) for i in range(games))
    results = sorted(run_matches(deals, (policy,), config, tables=tables), key=lambda row: row.deal.seed)
    return PolicyResults(
        tuple(int(row.winner == 0) for row in results),
        sum(row.outcome.utilities()[0] for row in results) / games,
        sum(row.telemetry.stock_draws[0] for row in results),
        sum(row.telemetry.discard_draws[0] for row in results),
    )


def evaluate_checkpoint(
    path: str,
    games: int = 300,
    seed: int = 300000,
    warm_games: int = 300,
    warm_seed: int = 600000,
    sensitivity: bool = False, *,
    tables: int = 64, precision: str = "fp32",
) -> dict:
    if isinstance(warm_games, bool) or not isinstance(warm_games, int) or warm_games < 0:
        raise ValueError("warm_games must be a nonnegative integer")
    if sensitivity and not warm_games:
        raise ValueError("sensitivity requires positive warm_games")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoint_sha256 = file_sha256(Path(path))
        policy = CheckpointPolicy(path, precision=precision)
        network, config = policy.network, policy.game
        random_policy = RandomPolicy()
        result = {
            "checkpoint": path,
            "checkpoint_sha256": checkpoint_sha256,
            "game_config": config.to_dict(),
            "architecture": network.architecture,
            "checkpoint_version": CHECKPOINT_VERSION,
            "encoding_version": ENCODING_VERSION,
            "model_spec": network.spec.to_dict(),
            "parameter_count": sum(
                parameter.numel() for parameter in network.parameters()
            ),
            "game_seed": seed,
            "warm_seed": warm_seed,
            "greedy": evaluate_games(policy, config, games, seed, tables=tables).to_dict(),
            "random": evaluate_games(random_policy, config, games, seed, tables=tables).to_dict(),
        }
        if warm_games:
            result["warm_greedy"] = evaluate_games(
                policy, config, warm_games, warm_seed, warm_start=True, tables=tables,
            ).to_dict()
            result["warm_random"] = evaluate_games(
                random_policy, config, warm_games, warm_seed, warm_start=True, tables=tables,
            ).to_dict()
            if sensitivity:
                result["warm_oracle"] = evaluate_games(
                    _DiscardOracle(), config, warm_games, warm_seed, warm_start=True, tables=tables,
                ).to_dict()
                if result["warm_oracle"]["win_rate"] != 1.0:
                    raise RuntimeError(
                        "discard-puzzle ruler failed its positive control"
                    )
                if result["warm_random"]["win_rate"] >= 0.95:
                    raise RuntimeError(
                        "discard-puzzle ruler does not separate random play"
                    )
        if file_sha256(Path(path)) != checkpoint_sha256:
            raise ValueError("checkpoint changed during evaluation")
        return result
    finally:
        torch.set_num_threads(previous_threads)


def paired_comparison(
    candidate: Sequence[Sequence[int]],
    control: Sequence[Sequence[int]],
    samples: int = 2000,
    seed: int = 91827,
) -> dict:
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
