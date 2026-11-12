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
GATE_SCHEMA_VERSION = 4
PLAYER_COUNTS = (2, 3, 4)
GATE_PHASES = ("selection", "confirmation")


@dataclass(frozen=True)
class ValidatedGateEvidence:
    phase: str
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


def replicas_for_count(players: int) -> int:
    if type(players) is not int or players not in PLAYER_COUNTS:
        raise ValueError("players must be 2, 3, or 4")
    return {2: 4, 3: 2, 4: 1}[players]


def blocks_for_count(seed_blocks: int, players: int) -> int:
    if type(seed_blocks) is not int or seed_blocks <= 0:
        raise ValueError("seed_blocks must be a positive integer")
    return replicas_for_count(players) * seed_blocks


def _require_margins(values: Sequence[float]) -> None:
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not -1 <= float(value) <= 1
        for value in values
    ):
        raise ValueError("Seed-block scores must be nonempty, finite, and in [-1, 1]")


def _require_samples(samples: int) -> None:
    if type(samples) is not int or samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")


def clustered_interval(
    values: Sequence[float],
    samples: int = 2000,
    seed: int = 81,
) -> list[float]:
    _require_margins(values)
    _require_samples(samples)
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
    declarations = {}
    for count in counts:
        raw = multiplayer[str(count)]
        count_blocks = blocks_for_count(blocks, count)
        if not isinstance(raw, dict) or type(raw.get("blocks")) is not int or raw["blocks"] != count_blocks:
            raise ValueError("Block count does not match the player-count schedule")
        if raw.get("games") != count_blocks * len(balanced_seat_orders(count)):
            raise ValueError("Game count does not match the balanced seating schedule")
        for field, summary, target in (
            ("placement_margins", "candidate_minus_champion", margins),
            ("declaration_margins", "declaration_delta", declarations),
        ):
            values = raw.get(field)
            if not isinstance(values, (list, tuple)) or len(values) != count_blocks:
                raise ValueError("Each multiplayer result requires one %s value per seed block" % field)
            _require_margins(values)
            declared = raw.get(summary)
            if (
                isinstance(declared, bool) or not isinstance(declared, Real)
                or not math.isfinite(float(declared))
                or not math.isclose(sum(values) / count_blocks, declared, rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError("Player-count summary disagrees with its raw %s" % field)
            replicas = replicas_for_count(count)
            target[count] = [
                sum(values[block * replicas:(block + 1) * replicas]) / replicas
                for block in range(blocks)
            ]
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
    differences = [left - right for left, right in zip(candidate, champion)]
    placement = [sum(margins[count][block] for count in counts) / len(counts) for block in range(blocks)]
    declaration = [sum(declarations[count][block] for count in counts) / len(counts) for block in range(blocks)]
    return {
        "multiplayer_gain": sum(placement) / blocks,
        "declaration_gain": sum(declaration) / blocks,
        "multiplayer_ci95": clustered_interval(placement, samples, seed + 91827),
        "declaration_ci95": clustered_interval(declaration, samples, seed + 93113),
        "multiplayer_ci95_by_count": {
            str(count): clustered_interval(margins[count], samples, seed + 91827 + count)
            for count in counts
        },
        "solo_gain": sum(differences) / games,
        "solo_ci95": clustered_interval(differences, samples, seed + 88181),
    }


def assess_gate(
    phase: str,
    rules: GameConfig,
    multiplayer: dict,
    metrics: dict,
    seed_blocks: int,
    solo_games: int,
    player_counts: Sequence[int] = PLAYER_COUNTS,
) -> tuple[bool, list[str]]:
    if phase not in GATE_PHASES:
        raise ValueError("gate phase must be selection or confirmation")
    counts = require_player_counts(tuple(player_counts))
    minimum = MIN_SELECTION_SEED_BLOCKS if phase == "selection" else MIN_CONFIRMATION_SEED_BLOCKS
    checks = [
        ("rules", (rules.num_decks, rules.cards_in_hand, rules.required_sequences, rules.max_turns) != (3, 21, 5, 60)
         or rules.recycle_discard is not True),
        ("seed_blocks", seed_blocks < minimum),
        ("solo_games", solo_games < MIN_SOLO_GAMES),
        ("placement", metrics["multiplayer_gain"] <= 0.0
         or (phase == "confirmation" and metrics["multiplayer_ci95"][0] <= 0.0)),
    ]
    for count in counts:
        mean = multiplayer[str(count)]["candidate_minus_champion"]
        checks.append((
            "placement_%dp" % count,
            mean < -0.05 if phase == "selection"
            else mean < -0.02 or metrics["multiplayer_ci95_by_count"][str(count)][0] < -0.06,
        ))
    checks.extend((
        ("declaration", metrics["declaration_gain"] < -0.02 if phase == "selection"
         else metrics["declaration_gain"] < -0.01 or metrics["declaration_ci95"][0] < -0.05),
        ("solo", metrics["solo_gain"] < -0.02 if phase == "selection"
         else metrics["solo_gain"] < -0.01 or metrics["solo_ci95"][0] < -0.05),
    ))
    failures = [name for name, failed in checks if failed]
    return not failures, failures


def validate_gate_evidence(
    evidence: dict,
    phase: str,
    candidate_sha256: str,
    champion_sha256: str,
    player_counts: Optional[Sequence[int]] = None,
) -> ValidatedGateEvidence:
    if not isinstance(evidence, dict):
        raise ValueError("%s evidence must accept these exact models" % phase)
    version = evidence.get("gate_schema_version")
    if type(version) is not int or version != GATE_SCHEMA_VERSION:
        raise ValueError("%s evidence uses unsupported gate schema v%r; re-run gates under schema v4" % (phase, version))
    if (
        type(evidence.get("accepted")) is not bool
        or evidence.get("phase") != phase
        or evidence.get("candidate_sha256") != candidate_sha256
        or evidence.get("champion_sha256") != champion_sha256
    ):
        raise ValueError(
            "%s evidence must accept these exact models" % phase
        )
    if (
        evidence.get("encoding_version") != ENCODING_VERSION
        or evidence.get("architecture") != ARCHITECTURE
    ):
        raise ValueError("%s evidence requires axial_table gate schema v4" % phase)
    raw_counts = evidence.get("player_counts")
    if not isinstance(raw_counts, (list, tuple)):
        raise ValueError("%s evidence requires player_counts as a list or tuple" % phase)
    counts = require_player_counts(tuple(raw_counts))
    if player_counts is not None and counts != require_player_counts(tuple(player_counts)):
        raise ValueError("%s evidence player_counts do not match the promotion scope" % phase)
    rules = evidence.get("rules")
    game = GameConfig.from_dict(rules)
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
        or seed_blocks <= 0
        or isinstance(solo_games, bool)
        or not isinstance(solo_games, int)
        or solo_games <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < 100
    ):
        raise ValueError(
            "%s evidence requires positive seed_blocks and solo_games, an integer seed, and 100 bootstrap samples" % phase
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
        raise ValueError("%s evidence needs joint-block player-count intervals" % phase)
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
            or not math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
            for left, right in pairs
        ):
            raise ValueError(
                "%s summary disagrees with raw evidence for %s"
                % (phase, name)
            )
    accepted, failures = assess_gate(
        phase,
        game,
        multiplayer,
        recomputed,
        seed_blocks,
        solo_games,
        player_counts=counts,
    )
    if (
        evidence["accepted"] != accepted
        or type(evidence.get("gate_failures")) is not list
        or evidence["gate_failures"] != failures
    ):
        raise ValueError("%s verdict disagrees with raw evidence; gate_failures=%r" % (phase, failures))
    if not accepted:
        raise ValueError("%s gate did not pass: %s" % (phase, ", ".join(failures)))
    return ValidatedGateEvidence(
        phase=phase,
        seed_blocks=seed_blocks,
        solo_games=solo_games,
        seed=seed,
        rules=game.to_dict(),
        player_counts=counts,
    )
