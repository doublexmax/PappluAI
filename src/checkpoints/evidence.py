from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import random
from typing import Sequence


MIN_SELECTION_SEED_BLOCKS = 32
MIN_CONFIRMATION_SEED_BLOCKS = 128
MIN_SOLO_GAMES = 256


@dataclass(frozen=True)
class ValidatedGateEvidence:
    seed_blocks: int
    solo_games: int
    seed: int
    rules: dict


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


def clustered_interval(
    values: Sequence[float],
    samples: int = 2000,
    seed: int = 81,
) -> list:
    if not values or isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise ValueError("Nonempty seed-block scores and at least 100 bootstrap samples are required")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not -1 <= float(value) <= 1
        for value in values
    ):
        raise ValueError("Seed-block scores must be finite and in [-1, 1]")
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(samples)
    )
    return [bootstrap[int(samples * 0.025)], bootstrap[int(samples * 0.975)]]


def evidence_metrics(
    multiplayer: dict,
    solo: dict,
    blocks: int,
    seed: int,
    samples: int,
) -> dict:
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


def gate_accepted(
    multiplayer: dict,
    metrics: dict,
    seed_blocks: int,
    solo_games: int,
) -> bool:
    return (
        seed_blocks >= MIN_SELECTION_SEED_BLOCKS
        and solo_games >= MIN_SOLO_GAMES
        and metrics["multiplayer_gain"] >= 0.05
        and metrics["multiplayer_ci95"][0] > 0
        and all(
            value["candidate_minus_champion"] >= 0
            for value in multiplayer.values()
        )
        and metrics["solo_gain"] >= 0
        and metrics["solo_ci95"][0] >= -0.03
    )


def validate_gate_evidence(
    evidence: dict,
    phase: str,
    candidate_sha256: str,
    champion_sha256: str,
    minimum_seed_blocks: int,
) -> ValidatedGateEvidence:
    if (
        not isinstance(evidence, dict)
        or evidence.get("accepted") is not True
        or evidence.get("phase") != phase
        or evidence.get("candidate_sha256") != candidate_sha256
        or evidence.get("champion_sha256") != champion_sha256
    ):
        raise ValueError(
            "%s evidence must accept these exact models" % phase
        )
    if tuple(evidence.get("player_counts", ())) != (2, 3):
        raise ValueError(
            "%s evidence requires two- and three-player results" % phase
        )
    rules = evidence.get("rules")
    if not isinstance(rules, dict) or rules.get("recycle_discard") is not True:
        raise ValueError(
            "%s evidence must use discard-recycling rules" % phase
        )
    multiplayer = evidence.get("multiplayer")
    solo = evidence.get("solo")
    if (
        not isinstance(multiplayer, dict)
        or set(multiplayer) != {"2", "3"}
        or not isinstance(solo, dict)
    ):
        raise ValueError(
            "%s evidence requires multiplayer and solo results" % phase
        )
    seed_blocks = evidence.get("seed_blocks")
    solo_games = solo.get("games")
    seed = evidence.get("seed")
    samples = evidence.get("bootstrap_samples")
    if (
        isinstance(seed_blocks, bool)
        or not isinstance(seed_blocks, int)
        or seed_blocks < minimum_seed_blocks
        or isinstance(solo_games, bool)
        or not isinstance(solo_games, int)
        or solo_games < MIN_SOLO_GAMES
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < 100
    ):
        raise ValueError(
            "%s requires at least %d seed blocks, %d solo games, "
            "and 100 bootstrap samples"
            % (
                phase,
                minimum_seed_blocks,
                MIN_SOLO_GAMES,
            )
        )
    for name in ("multiplayer_gain", "solo_gain"):
        value = evidence.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                "%s evidence lacks finite %s" % (phase, name)
            )
    for name in ("multiplayer_ci95", "solo_ci95"):
        values = evidence.get(name)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in values
            )
            or values[0] > values[1]
        ):
            raise ValueError(
                "%s evidence has invalid %s" % (phase, name)
            )
    recomputed = evidence_metrics(
        multiplayer,
        solo,
        seed_blocks,
        seed,
        samples,
    )
    for name, computed in recomputed.items():
        declared = evidence[name]
        pairs = (
            zip(declared, computed)
            if isinstance(computed, list)
            else ((declared, computed),)
        )
        if any(
            not math.isclose(left, right, abs_tol=1e-12)
            for left, right in pairs
        ):
            raise ValueError(
                "%s summary disagrees with raw evidence: %s"
                % (phase, name)
            )
    if not gate_accepted(
        multiplayer,
        recomputed,
        seed_blocks,
        solo_games,
    ):
        raise ValueError("%s gate did not pass" % phase)
    return ValidatedGateEvidence(
        seed_blocks=seed_blocks,
        solo_games=solo_games,
        seed=seed,
        rules=dict(rules),
    )
