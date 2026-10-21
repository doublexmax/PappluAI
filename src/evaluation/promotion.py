from __future__ import annotations

from pathlib import Path
import time
from typing import Sequence, Tuple

import torch

from src.checkpoints import evidence
from src.checkpoints.io import file_sha256
from src.evaluation.arena import CheckpointPolicy, RandomPolicy, play_match
from src.evaluation.benchmark import evaluate_games
from src.game.environment import ENCODING_VERSION, GameConfig
from src.game.multiplayer import MatchConfig
from src.model.spec import ARCHITECTURE


class GateInterrupted(RuntimeError):
    pass


def evaluate_gate(
    candidate_path: str,
    champion_path: str,
    opponent_paths: Sequence[str],
    seed: int,
    seed_blocks: int,
    solo_games: int,
    max_turns: int = 60,
    player_counts: Tuple[int, ...] = (2, 3, 4),
    samples: int = 2000,
    deadline=None,
    recycle_discard: bool = True,
) -> dict:
    for name, value in (
        ("seed_blocks", seed_blocks),
        ("solo_games", solo_games),
        ("max_turns", max_turns),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("%s must be a positive integer" % name)
    if not isinstance(player_counts, (list, tuple)):
        raise TypeError("player_counts must be a list or tuple")
    counts = evidence.require_player_counts(tuple(player_counts))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rules = GameConfig(
            max_turns=max_turns,
            recycle_discard=recycle_discard,
        )
        candidate = CheckpointPolicy(str(candidate_path))
        champion = CheckpointPolicy(str(champion_path))
        opponents = [
            CheckpointPolicy(str(path)) for path in opponent_paths
        ]
        opponents.append(RandomPolicy())
        by_player_count = {}

        def check_deadline():
            if deadline is not None and time.monotonic() >= deadline:
                raise GateInterrupted(
                    "Promotion evaluation reached its time limit"
                )

        for players in counts:
            raw = []
            for index in range(seed_blocks):
                check_deadline()
                fillers = [
                    opponents[(index + seat) % len(opponents)]
                    for seat in range(players - 2)
                ]
                lineup = [candidate, champion, *fillers]
                results = []
                orders = evidence.balanced_seat_orders(players)
                for order in orders:
                    policies = [lineup[index] for index in order]
                    match = play_match(
                        policies,
                        MatchConfig(game=rules, players=players),
                        seed + index,
                        deadline=deadline,
                    )
                    winner = (
                        None
                        if match.winner is None
                        else order[match.winner]
                    )
                    results.append(
                        1 if winner == 0 else -1 if winner == 1 else 0
                    )
                raw.append(sum(results) / len(orders))
            by_player_count[str(players)] = {
                "seed_block_margins": raw,
                "candidate_minus_champion": sum(raw) / len(raw),
                "games": seed_blocks * len(evidence.balanced_seat_orders(players)),
            }
        solo_seed = seed + 100_000
        check_deadline()
        candidate_solo = evaluate_games(
            candidate.network,
            rules,
            solo_games,
            solo_seed,
            deadline=deadline,
        )
        champion_solo = evaluate_games(
            champion.network,
            rules,
            solo_games,
            solo_seed,
            deadline=deadline,
        )
        solo = {
            "seed": solo_seed,
            "games": solo_games,
            "candidate_outcomes": list(candidate_solo.outcomes),
            "champion_outcomes": list(champion_solo.outcomes),
            "candidate_wins": sum(candidate_solo.outcomes),
            "champion_wins": sum(champion_solo.outcomes),
        }
        metrics = evidence.evidence_metrics(
            by_player_count,
            solo,
            seed_blocks,
            seed,
            samples,
            player_counts=counts,
        )
        accepted = max_turns == 60 and recycle_discard and evidence.gate_accepted(
            by_player_count,
            metrics,
            seed_blocks,
            solo_games,
            player_counts=counts,
        )
        return {
            "gate_schema_version": evidence.GATE_SCHEMA_VERSION,
            "encoding_version": ENCODING_VERSION,
            "architecture": ARCHITECTURE,
            "candidate_sha256": file_sha256(Path(candidate_path)),
            "champion_sha256": file_sha256(Path(champion_path)),
            "opponent_sha256": [
                file_sha256(Path(path)) for path in opponent_paths
            ],
            "accepted": accepted,
            "seed": seed,
            "seed_blocks": seed_blocks,
            "bootstrap_samples": samples,
            "seating": "Cyclic/mirrored 2/3 seats and all 24 four-player permutations",
            "player_counts": list(counts),
            "max_turns": max_turns,
            **metrics,
            "multiplayer": by_player_count,
            "solo": solo,
            "rules": rules.to_dict(),
            "scope": (
                "Configured player-count seat-clustered league gate with "
                "ordinary full21 solo retention."
            ),
        }
    finally:
        torch.set_num_threads(previous_threads)
