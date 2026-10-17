from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import permutations
from numbers import Real
import random
from typing import Optional, Sequence

from src.game.environment import ENCODING_VERSION, GameConfig
from src.model.spec import ARCHITECTURE


MIN_SELECTION_SEED_BLOCKS = 32
MIN_CONFIRMATION_SEED_BLOCKS = 128
MIN_SOLO_GAMES = 256
GATE_SCHEMA_VERSION = 2
PLAYER_COUNTS = (2, 3, 4)


@dataclass(frozen=True)
class ValidatedGateEvidence:
    seed_blocks: int
    solo_games: int
    seed: int
    rules: dict
    player_counts: tuple[int, ...] = PLAYER_COUNTS


def require_player_counts(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("player_counts must be a tuple")
    if not value or any(
        isinstance(count, bool) or not isinstance(count, int) or count not in PLAYER_COUNTS
        for count in value
    ):
        raise ValueError("player_counts must be a nonempty tuple drawn from 2/3/4")
    if tuple(sorted(set(value))) != value:
        raise ValueError("player_counts must be distinct and sorted")
    return value


def balanced_seat_orders(players: int) -> tuple:
    if isinstance(players, bool) or not isinstance(players, int) or not 2 <= players <= 6:
        raise ValueError("players must be an integer in 2..6")
    lineup = tuple(range(players))
    if players == 4:
        return tuple(permutations(lineup))
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
    player_counts: Sequence[int] = PLAYER_COUNTS,
) -> dict:
    counts = require_player_counts(
        tuple(player_counts) if isinstance(player_counts, (list, tuple)) else player_counts
    )
    if not isinstance(multiplayer, dict) or set(multiplayer) != {str(count) for count in counts}:
        raise ValueError("Multiplayer results must match player_counts")
    margins = {}
    for count in counts:
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
    combined = [
        sum(values) / len(counts)
        for values in zip(*(margins[count] for count in counts))
    ]
    differences = [left - right for left, right in zip(candidate, champion)]
    return {
        "multiplayer_gain": sum(combined) / blocks,
        "multiplayer_ci95": clustered_interval(combined, samples, seed + 91827),
        "multiplayer_ci95_by_count": {
            str(count): clustered_interval(margins[count], samples, seed + 91827 + count)
            for count in counts
        },
        "solo_gain": sum(differences) / games,
        "solo_ci95": clustered_interval(differences, samples, seed + 88181),
    }


def gate_accepted(
    multiplayer: dict,
    metrics: dict,
    seed_blocks: int,
    solo_games: int,
    player_counts: Sequence[int] = PLAYER_COUNTS,
) -> bool:
    counts = require_player_counts(tuple(player_counts))
    return (
        seed_blocks >= MIN_SELECTION_SEED_BLOCKS
        and solo_games >= MIN_SOLO_GAMES
        and metrics["multiplayer_gain"] >= 0.05
        and metrics["multiplayer_ci95"][0] > 0
        and all(
            value["candidate_minus_champion"] > 0
            for value in multiplayer.values()
        )
        and set(metrics["multiplayer_ci95_by_count"]) == {str(count) for count in counts}
        and all(interval[0] > 0 for interval in metrics["multiplayer_ci95_by_count"].values())
        and metrics["solo_gain"] >= 0
        and metrics["solo_ci95"][0] >= -0.03
    )


def validate_gate_evidence(
    evidence: dict,
    phase: str,
    candidate_sha256: str,
    champion_sha256: str,
    minimum_seed_blocks: int,
    player_counts: Optional[Sequence[int]] = None,
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
    if (
        evidence.get("gate_schema_version") != GATE_SCHEMA_VERSION
        or evidence.get("encoding_version") != ENCODING_VERSION
        or evidence.get("architecture") != ARCHITECTURE
    ):
        raise ValueError("%s evidence requires the current dynamic gate schema" % phase)
    raw_counts = evidence.get("player_counts")
    if not isinstance(raw_counts, (list, tuple)):
        raise ValueError("%s evidence requires player_counts as a list or tuple" % phase)
    counts = require_player_counts(tuple(raw_counts))
    if player_counts is not None and counts != require_player_counts(tuple(player_counts)):
        raise ValueError("%s evidence player_counts do not match the promotion scope" % phase)
    rules = evidence.get("rules")
    if not isinstance(rules, dict) or rules.get("recycle_discard") is not True:
        raise ValueError(
            "%s evidence must use discard-recycling rules" % phase
        )
    game = GameConfig.from_dict(rules)
    if (game.num_decks, game.cards_in_hand, game.required_sequences, game.max_turns) != (3, 21, 5, 60):
        raise ValueError("%s evidence requires full21 60-turn rules" % phase)
    multiplayer = evidence.get("multiplayer")
    solo = evidence.get("solo")
    if (
        not isinstance(multiplayer, dict)
        or set(multiplayer) != {str(count) for count in counts}
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
        player_counts=counts,
    )
    pairs_to_check = [
        (name, evidence.get(name), computed) for name, computed in recomputed.items()
        if name != "multiplayer_ci95_by_count"
    ]
    declared_counts = evidence.get("multiplayer_ci95_by_count")
    if (
        not isinstance(declared_counts, dict)
        or set(declared_counts) != {str(count) for count in counts}
    ):
        raise ValueError("%s evidence needs independent player-count intervals" % phase)
    pairs_to_check.extend(
        ("multiplayer_ci95_by_count.%s" % count, declared_counts[count], interval)
        for count, interval in recomputed["multiplayer_ci95_by_count"].items()
    )
    for name, declared, computed in pairs_to_check:
        if isinstance(computed, list) and (
            not isinstance(declared, (list, tuple)) or len(declared) != 2
        ):
            raise ValueError("%s evidence has invalid %s" % (phase, name))
        pairs = (
            zip(declared, computed)
            if isinstance(computed, list)
            else ((declared, computed),)
        )
        if any(
            type(left) not in (int, float) or not math.isfinite(left)
            or not math.isclose(left, right, abs_tol=1e-12)
            for left, right in pairs
        ):
            raise ValueError(
                "%s summary disagrees with raw evidence for %s"
                % (phase, name)
            )
    if not gate_accepted(
        multiplayer,
        recomputed,
        seed_blocks,
        solo_games,
        player_counts=counts,
    ):
        raise ValueError("%s gate did not pass" % phase)
    return ValidatedGateEvidence(
        seed_blocks=seed_blocks,
        solo_games=solo_games,
        seed=seed,
        rules=game.to_dict(),
        player_counts=counts,
    )
