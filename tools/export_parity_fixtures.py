from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from src.evaluate import (
    NUM_FACES, SCORING_VERSION, evaluate_hand, hand_reward, is_valid_hand, minimum_penalty,
)


OUTPUT = ROOT / "tests" / "fixtures" / "evaluate-parity.json"
SEED = 1729
RANDOM_CASES = 400
NEAR_WIN_CASES = 200
SUITS = "shdc"
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SOURCES = ("src/evaluate.py", "src/model/search.py", "tools/export_parity_fixtures.py")


def load_search_functions():
    path = ROOT / "src" / "model" / "search.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    names = {"can_declare_ub", "winning_discard"}
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise RuntimeError("search parity functions are missing")
    namespace = {"is_valid_hand": is_valid_hand, "Sequence": Sequence, "Optional": Optional}
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["can_declare_ub"], namespace["winning_discard"]


can_declare_ub, winning_discard = load_search_functions()


def face(name):
    return SUITS.index(name[-1]) * 13 + RANKS.index(name[:-1])


def counts(cards):
    if isinstance(cards, str):
        cards = map(face, cards.split())
    hand = [0] * NUM_FACES
    for card in cards:
        hand[card] += 1
    return hand


def record(entry, field, function, *arguments):
    try:
        result = function(*arguments)
    except (TypeError, ValueError) as error:
        entry.setdefault("errors", {})[field] = {
            "type": type(error).__name__, "message": str(error),
        }
        return None
    entry[field] = asdict(result)
    if field == "penalty":
        entry[field]["qualifying_sequences"] = result.qualifying_sequences
    return result


def parity_case(name, group, hand, joker, required=5, size=21):
    entry = {
        "name": name, "group": group, "hand": hand, "joker": joker,
        "required_sequences": required, "cards_in_hand": size,
    }
    evaluation = record(entry, "evaluation", evaluate_hand, hand, joker, required, size)
    record(entry, "penalty", minimum_penalty, hand, joker, required)
    if evaluation is not None:
        entry["is_valid_hand"] = is_valid_hand(hand, joker, required, size)
        entry["hand_reward"] = hand_reward(hand, joker, required, size)
    if "errors" not in entry:
        entry["can_declare_ub"] = can_declare_ub(hand, joker, required)
        entry["winning_discard"] = winning_discard(hand, joker, required, size)
    return entry


def random_hand(randomizer, size, decks):
    shoe = [card for card in range(NUM_FACES) for _ in range(decks)]
    return counts(randomizer.sample(shoe, size))


def run_cards(randomizer, length):
    patterns = [list(range(start, start + length)) for start in range(14 - length)]
    patterns.append(list(range(14 - length, 13)) + [0])
    suit = randomizer.randrange(4)
    return [suit * 13 + rank for rank in randomizer.choice(patterns)]


def winning_hand(randomizer, joker, required=5, size=21):
    four_counts = [
        number for number in range(min(required, size - required * 3) + 1)
        if size - required * 3 - number not in (1, 2)
    ]
    fours = set(randomizer.sample(range(required), randomizer.choice(four_counts)))
    cards = []
    for index in range(required):
        group = run_cards(randomizer, 4 if index in fours else 3)
        if randomizer.random() < 0.3:
            group[randomizer.randrange(len(group))] = joker
        cards.extend(group)
    while len(cards) < size:
        remaining = size - len(cards)
        length = randomizer.choice([
            length for length in range(3, min(5, remaining) + 1)
            if remaining - length not in (1, 2)
        ])
        kind = randomizer.randrange(3)
        if kind == 0 and length <= 4:
            rank = randomizer.randrange(13)
            group = [suit * 13 + rank for suit in randomizer.sample(range(4), length)]
        else:
            group = run_cards(randomizer, length)
            if kind == 1:
                slot = randomizer.randrange(length)
                wilds = [
                    suit * 13 + joker % 13 for suit in range(4)
                    if suit * 13 + joker % 13 not in (joker, group[slot])
                ]
                group[slot] = randomizer.choice(wilds)
        cards.extend(group)
    hand = counts(cards)
    if not is_valid_hand(hand, joker, required, size):
        raise AssertionError("constructed winning hand does not declare")
    return hand


def near_win_variants(randomizer, hand, joker):
    cards = [card for card, copies in enumerate(hand) for _ in range(copies)]
    removed = hand.copy()
    removed[randomizer.choice(cards)] -= 1
    yield "removed", removed
    added = hand.copy()
    added[randomizer.randrange(NUM_FACES)] += 1
    yield "added", added
    swapped = hand.copy()
    old = randomizer.choice(cards)
    swapped[old] -= 1
    swapped[(old + 1 + randomizer.randrange(NUM_FACES - 1)) % NUM_FACES] += 1
    yield "swapped", swapped
    wild = hand.copy()
    natural = [card for card in cards if card % 13 != joker % 13]
    wild[randomizer.choice(natural or cards)] -= 1
    wild[randomizer.randrange(4) * 13 + joker % 13] += 1
    yield "wild", wild


def edge_cases():
    specs = (
        ("classic_21", "As 2s 3s Ah 2h 3h Ad 2d 3d 4s 5s 6s 7h 8h 9h 10s 10h 10d Js Jh Jd", "8s", 5, 21),
        ("broken_classic", "As 2s 3s Ah 2h 3h Ad 2d 3d 4s 5s Kc 7h 8h 9h 10s 10h 10d Js Jh Jd", "8s", 5, 21),
        ("exact_joker_pure", "4h 5h 8s", "8s", 1, 3),
        ("rank_joker_not_pure", "4h 5h 8d", "8s", 1, 3),
        ("rank_joker_residual", "As 2s 3s 4h 5h 8d", "8s", 1, 6),
        ("rank_joker_natural", "7d 8d 9d", "8s", 1, 3),
        ("rank_joker_residual_set", "As 2s 3s Ah 2h 3h Ad 2d 3d 4s 5s 6s 7h 8h 9h 10s 10h 8d Js Jh Jd", "8s", 5, 21),
        ("duplicate_natural_set", "7s 7s Js", "8s", 0, 3),
        ("duplicate_with_wild_set", "7s 7s 8h", "8s", 0, 3),
        ("duplicate_across_groups", "3s 4s 5s 5s 6s 7s", "8s", 2, 6),
        ("four_card_set", "10s 10h 10d 10c", "8s", 0, 4),
        ("ace_low", "As 2s 3s", "8s", 1, 3),
        ("ace_high", "Qh Kh Ah", "8s", 1, 3),
        ("ace_does_not_wrap", "Ks As 2s", "8s", 1, 3),
        ("suit_boundary", "Ks Ah 2h", "8s", 1, 3),
        ("leftover_not_ignored", "As 2s 3s Kh", "8s", 1, 4),
        ("tunnela_natural", "5d 5d 5d", "8s", 1, 3),
        ("tunnela_exact_joker", "8s 8s 8s", "8s", 1, 3),
        ("two_copies_plus_wild", "5d 5d 8s", "8s", 0, 3),
        ("two_copies_plus_wild_quota", "5d 5d 8s", "8s", 1, 3),
        ("mixed_wild_not_pure", "8h 8h 8d", "8s", 1, 3),
        ("mixed_wild_zero_quota", "8h 8h 8d", "8s", 0, 3),
        ("all_four_wild_faces", "8s 8h 8d 8c", "8s", 0, 4),
        ("residual_natural_stays_pure", "As 2s 3s 4h 5h 6h", "8s", 1, 6),
        ("overlap_alternate_allocation", "3s 4s 5s 6s 3h 3d", "8s", 1, 6),
        ("joker_absent", "As 2s 3s", "Kc", 1, 3),
        ("six_identical_copies", "5d 5d 5d 5d 5d 5d", "8s", 2, 6),
        ("two_exact_jokers_in_one_run", "4h 8s 8s 7h 8h 9h", "8s", 2, 6),
        ("empty_three", "", "8s", 0, 3),
        ("empty_default", "", "8s", 5, 21),
        ("size_two_not_error", "As 2s", "8s", 0, 2),
        ("one_card_below_target", "As 2s 3s", "8s", 1, 4),
        ("one_card_above_target", "As 2s 3s Kh", "8s", 1, 3),
    )
    for name, cards, joker, required, size in specs:
        yield name, counts(cards), face(joker), required, size
    classic = counts(specs[0][1])
    yield "wrong_cardinality", classic, face("8s"), 5, 20
    yield "impossible_quota", classic, face("8s"), 10, 21
    yield "twenty_one_exact_jokers", counts([face("8s")] * 21), face("8s"), 5, 21
    yield "twenty_one_off_suit_jokers", counts([face("8h")] * 14 + [face("8d")] * 7), face("8s"), 5, 21
    yield "thirty_cards_ten_sequences", counts([card for card in range(10) for _ in range(3)]), face("Kc"), 10, 30
    yield "zero_is_winning_discard", counts([face("8s")] * 21 + [0]), face("8s"), 5, 21
    yield "user_tunnela_deal", counts([
        19, 20, 21, 22, 30, 30, 30, 36, 37, 38, 26, 42, 43, 23, 46, 47, 48, 36, 23, 35, 34,
    ]), face("Js"), 5, 21
    for index, suits in enumerate(itertools.combinations(range(4), 3)):
        yield "set_suit_combo_%d" % index, counts([suit * 13 + 9 for suit in suits]), face("8s"), 0, 3
    for length in range(6, 14):
        for suit in range(4):
            joker = ((suit + 1) % 4) * 13 + 9
            patterns = (
                list(range(length)),
                list(range(14 - length, 13)) + [0],
            )
            for high, pattern in enumerate(patterns):
                cards = [suit * 13 + rank for rank in pattern]
                for substitute in (False, True):
                    actuals = cards.copy()
                    if substitute:
                        actuals[0] = actuals[-1] = joker
                    name = "long_run_%d_%d_%d_%d" % (length, suit, high, substitute)
                    yield name, counts(actuals), joker, length // 3, length


def penalty_cases():
    specs = (
        ("wild_set_below_quota", "2s 3s 4s 2h 3h 4h 7s 7h Kd", "Kc", 3),
        ("wild_set_meets_quota", "2s 3s 4s 2h 3h 4h 2d 3d 4d 7s 7h Kd", "Kc", 3),
        ("natural_set_below_quota", "2s 3s 4s 2h 3h 4h 7s 7h 7d", "Kc", 3),
        ("natural_set_meets_quota", "2s 3s 4s 2h 3h 4h 2d 3d 4d 7s 7h 7d", "Kc", 3),
        ("point_sum", "As Kh Qd Jc 10s 2h 8d", "8c", 0),
        ("overlap_four_card_run", "4h 5h 6h 7h 7s 7d", "Kc", 1),
        ("overlap_two_sets", "3h 4h 5h 6h 7h 3s 3d 7s 7d", "Kc", 1),
        ("overlap_choose_run", "9s 10s Js 9h 9d", "Kc", 0),
        ("duplicate_runs", "5h 5h 6h 6h 7h 7h", "Kc", 2),
        ("duplicate_set_with_wild", "7s 7s 8d", "8s", 0),
        ("duplicate_unmatched", "4s 4s 5s 6s", "Kc", 1),
        ("off_suit_joker_not_qualifying", "4h 5h 8d Kc Kd Ks", "8s", 1),
        ("exact_joker_qualifying", "4h 5h 8s Kc Kd Ks", "8s", 1),
        ("off_suit_joker_natural_qualifying", "7d 8d 9d Kc Kd Ks", "8s", 1),
        ("ace_high_penalty", "Qh Kh Ah", "8c", 1),
        ("ace_low_penalty", "Ah 2h 3h", "8c", 1),
        ("no_wrap_penalty", "Kh Ah 2h", "8c", 1),
        ("zero_quota_set", "7s 7h 7d", "Kc", 0),
        ("unmet_quota_set", "7s 7h 7d", "Kc", 1),
        ("zero_quota_impure_run", "4h 5h 8d", "8s", 0),
        ("unmet_quota_impure_run", "4h 5h 8d", "8s", 1),
        ("long_run_two_sequences", "3s 4s 5s 6s 7s 8s 9h 9d 9c", "Kc", 2),
        ("long_run_insufficient_quota", "3s 4s 5s 6s 7s 8s 9h 9d 9c", "Kc", 3),
        ("zero_points_not_declaration", "As 2s 3s 4s 5s 9h 10h Jh Qh Kh 2d 3d 4d 5d 6d 9c 10c Jc Qc Kc 8d", "8s", 5),
        ("zero_points_unmet_quota", "4h 5h 6h 8h 8d 8c", "8s", 2),
        ("winner_leaves_zero_point_cards", "2s 3s 4s Kd Kd Kd", "Kc", 1),
        ("zero_point_quota_tie", "4h 5h 6h 8s 8s 8s", "8s", 2),
        ("positive_point_quota_tie", "2s 2s 2s 2d 3d 4d 8d 8d Jd Qd", "2s", 3),
    )
    for name, cards, joker, required in specs:
        hand = counts(cards)
        yield name, hand, face(joker), required, len(cards.split())
    for rank in range(13):
        yield "rank_points_%d" % rank, counts([rank]), face("Kc" if rank != 12 else "2c"), 5, 1
    for suit in range(4):
        yield "wild_rank_points_%d" % suit, counts([suit * 13 + 7]), face("8s"), 5, 1


def error_cases():
    hand = [0] * NUM_FACES
    for length in (0, 51, 53):
        yield "length_%d" % length, [0] * length, 7, 5, 21
    for label, value in (
        ("null", None), ("number", 52), ("boolean", True), ("object", {}),
        ("short_string", "hand"), ("long_string", "0" * 52),
    ):
        yield "hand_%s" % label, value, 7, 5, 21
    for index in (0, 23, 51):
        for label, value in (
            ("negative", -1), ("boolean", True), ("fraction", 1.5),
            ("null", None), ("string", "1"), ("object", {}),
        ):
            malformed = hand.copy()
            malformed[index] = value
            yield "count_%d_%s" % (index, label), malformed, 7, 5, 21
    for label, value in (
        ("negative", -1), ("too_large", 52), ("boolean", True),
        ("fraction", 1.5), ("null", None), ("string", "7"),
    ):
        yield "joker_%s" % label, hand, value, 5, 21
    for label, value in (
        ("negative", -1), ("boolean", True), ("fraction", 1.5),
        ("null", None), ("string", "5"),
    ):
        yield "quota_%s" % label, hand, 7, value, 21
    for label, value in (
        ("zero", 0), ("negative", -1), ("boolean", True),
        ("fraction", 1.5), ("null", None), ("string", "21"),
    ):
        yield "size_%s" % label, hand, 7, 5, value
    yield "length_precedes_joker", [0] * 51, -1, -1, 0
    yield "count_precedes_joker", [-1] + [0] * 51, -1, -1, 0
    yield "joker_precedes_quota", hand, -1, -1, 0
    yield "quota_precedes_size", hand, 7, -1, 0


def build_document():
    randomizer = random.Random(SEED)
    cases = []
    for index in range(RANDOM_CASES):
        size = randomizer.choice((3, 7, 13, 21, 21, 21, 30))
        decks = randomizer.randint(1, 6)
        required = randomizer.choice((0, 1, 3, 5, 5, 5, 10))
        hand = random_hand(randomizer, size, decks)
        cases.append(parity_case(
            "random_%03d" % index, "random", hand, randomizer.randrange(NUM_FACES), required, size,
        ))
    for index in range(NEAR_WIN_CASES):
        joker = randomizer.randrange(NUM_FACES)
        hand = winning_hand(randomizer, joker)
        for variant, candidate in near_win_variants(randomizer, hand, joker):
            cases.append(parity_case(
                "near_win_%03d_%s" % (index, variant), "near_win", candidate, joker,
            ))
    for group, source in (("edge", edge_cases()), ("penalty", penalty_cases()), ("error", error_cases())):
        for name, hand, joker, required, size in source:
            cases.append(parity_case(name, group, hand, joker, required, size))
    return {
        "scoring_version": SCORING_VERSION,
        "seed": SEED,
        "sources": {
            path: hashlib.sha256((ROOT / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for path in SOURCES
        },
        "cases": cases,
    }


def main():
    document = build_document()
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(payload.encode("utf-8"))
    print("Wrote %d cases to %s" % (len(document["cases"]), OUTPUT.relative_to(ROOT).as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
