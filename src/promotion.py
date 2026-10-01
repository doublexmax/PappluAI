from __future__ import annotations

from pathlib import Path
import random
import time
import math
from typing import Sequence, Tuple

from src.registry import file_sha256


class GateInterrupted(RuntimeError):
    pass


def balanced_seat_orders(players: int) -> tuple:
    if isinstance(players, bool) or not isinstance(players, int) or not 2 <= players <= 6:
        raise ValueError("players must be an integer in 2..6")
    lineup = tuple(range(players))
    lineups = (lineup,) if players == 2 else (lineup, (1, 0, *lineup[2:]))
    return tuple(
        order[rotation:] + order[:rotation]
        for order in lineups
        for rotation in range(players)
    )


def clustered_interval(values: Sequence[float], samples: int = 2000, seed: int = 81) -> list:
    if not values or isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise ValueError("Nonempty seed-block scores and at least 100 bootstrap samples are required")
    if any(not -1 <= value <= 1 for value in values):
        raise ValueError("Seed-block scores must be finite and in [-1, 1]")
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(samples)
    )
    return [bootstrap[int(samples * 0.025)], bootstrap[int(samples * 0.975)]]


def evidence_metrics(multiplayer: dict, solo: dict, blocks: int, seed: int, samples: int) -> dict:
    if not isinstance(multiplayer, dict) or set(multiplayer) != {"2", "3"}:
        raise ValueError("Both player-count result sets are required")
    margins = {}
    for count in (2, 3):
        raw = multiplayer[str(count)]
        values = raw.get("seed_block_margins")
        if not isinstance(values, (list, tuple)) or len(values) != blocks:
            raise ValueError("Each multiplayer result requires one margin per independent seed block")
        if raw.get("games") != blocks * len(balanced_seat_orders(count)):
            raise ValueError("Game count does not match the balanced seating schedule")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not -1 <= value <= 1
            for value in values
        ):
            raise ValueError("Invalid seed-block margin")
        mean = sum(values) / blocks
        declared = raw.get("candidate_minus_champion")
        if (
            isinstance(declared, bool) or not isinstance(declared, (int, float))
            or not math.isclose(mean, declared, abs_tol=1e-12)
        ):
            raise ValueError("Player-count summary disagrees with its raw margins")
        margins[count] = values
    if not isinstance(solo, dict):
        raise ValueError("Paired solo results are required")
    games = solo.get("games")
    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise ValueError("Solo game count must be positive")
    candidate = solo.get("candidate_outcomes")
    champion = solo.get("champion_outcomes")
    for outcomes in (candidate, champion):
        if (
            not isinstance(outcomes, (list, tuple)) or len(outcomes) != games
            or any(type(value) is not int or value not in (0, 1) for value in outcomes)
        ):
            raise ValueError("Solo outcomes must contain one binary verdict per game")
    if solo.get("candidate_wins") != sum(candidate) or solo.get("champion_wins") != sum(champion):
        raise ValueError("Solo win counts disagree with their raw outcomes")
    combined = [(left + right) / 2 for left, right in zip(margins[2], margins[3])]
    differences = [left - right for left, right in zip(candidate, champion)]
    return {
        "multiplayer_gain": sum(combined) / blocks,
        "multiplayer_ci95": clustered_interval(combined, samples, seed + 91827),
        "solo_gain": sum(differences) / games,
        "solo_ci95": clustered_interval(differences, samples, seed + 88181),
    }


def evaluate_gate(
    candidate_path: str,
    champion_path: str,
    opponent_paths: Sequence[str],
    seed: int,
    seed_blocks: int,
    solo_games: int,
    max_turns: int = 60,
    player_counts: Tuple[int, ...] = (2, 3),
    samples: int = 2000,
    deadline=None,
) -> dict:
    import torch
    from src.arena import CheckpointPolicy, RandomPolicy, play_match
    from src.benchmark import evaluate_games
    from src.environment import GameConfig
    from src.multiplayer import MatchConfig

    for name, value in (("seed_blocks", seed_blocks), ("solo_games", solo_games), ("max_turns", max_turns)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("%s must be a positive integer" % name)
    if tuple(player_counts) != (2, 3):
        raise ValueError("The current promotion gate covers two- and three-player matches")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    torch.set_num_threads(1)
    rules = GameConfig(max_turns=max_turns)
    candidate = CheckpointPolicy(str(candidate_path))
    champion = CheckpointPolicy(str(champion_path))
    opponents = [CheckpointPolicy(str(path)) for path in opponent_paths]
    opponents.append(RandomPolicy())
    by_player_count = {}
    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise GateInterrupted("Promotion evaluation reached its time limit")

    for players in player_counts:
        raw = []
        for index in range(seed_blocks):
            check_deadline()
            fillers = [opponents[(index + seat) % len(opponents)] for seat in range(players - 2)]
            lineup = [candidate, champion, *fillers]
            results = []
            orders = balanced_seat_orders(players)
            for order in orders:
                policies = [lineup[index] for index in order]
                match = play_match(
                    policies, MatchConfig(game=rules, players=players), seed + index,
                    deadline=deadline,
                )
                winner = None if match.winner is None else order[match.winner]
                results.append(1 if winner == 0 else -1 if winner == 1 else 0)
            raw.append(sum(results) / len(orders))
        by_player_count[str(players)] = {
            "seed_block_margins": raw,
            "candidate_minus_champion": sum(raw) / len(raw),
            "games": seed_blocks * len(balanced_seat_orders(players)),
        }
    solo_seed = seed + 100_000
    check_deadline()
    candidate_solo = evaluate_games(candidate.network, rules, solo_games, solo_seed, deadline=deadline)
    champion_solo = evaluate_games(champion.network, rules, solo_games, solo_seed, deadline=deadline)
    solo = {
        "seed": solo_seed,
        "games": solo_games,
        "candidate_outcomes": list(candidate_solo.outcomes),
        "champion_outcomes": list(champion_solo.outcomes),
        "candidate_wins": sum(candidate_solo.outcomes),
        "champion_wins": sum(champion_solo.outcomes),
    }
    metrics = evidence_metrics(by_player_count, solo, seed_blocks, seed, samples)
    accepted = (
        seed_blocks >= 32 and solo_games >= 256
        and metrics["multiplayer_gain"] >= 0.05 and metrics["multiplayer_ci95"][0] > 0
        and all(value["candidate_minus_champion"] >= 0 for value in by_player_count.values())
        and metrics["solo_gain"] >= 0 and metrics["solo_ci95"][0] >= -0.03
    )
    return {
        "candidate_sha256": file_sha256(Path(candidate_path)),
        "champion_sha256": file_sha256(Path(champion_path)),
        "opponent_sha256": [file_sha256(Path(path)) for path in opponent_paths],
        "accepted": accepted,
        "seed": seed,
        "seed_blocks": seed_blocks,
        "bootstrap_samples": samples,
        "seating": "Cyclic rotations with mirrored contender order",
        "player_counts": list(player_counts),
        "max_turns": max_turns,
        **metrics,
        "multiplayer": by_player_count,
        "solo": solo,
        "rules": rules.to_dict(),
        "scope": "Two-/three-player seat-clustered league gate with ordinary full21 solo retention.",
    }
