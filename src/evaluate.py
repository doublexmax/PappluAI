"""Papplu hand validity and binary declaration reward.

Card encoding is a length-52 count vector. Index = suit * 13 + rank_offset with
suit order s, h, d, c (0..3) and ranks A,2,3,4,5,6,7,8,9,10,J,Q,K (offset 0..12).

The selected joker is an exact face index 0..51. That face may substitute in any
meld, including required pure sequences. Every other card of the same rank may
substitute only in melds that do not count toward the pure-sequence quota.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Iterator, List, Literal, Optional, Sequence, Tuple


NUM_SUITS = 4
NUM_RANKS = 13
NUM_FACES = NUM_SUITS * NUM_RANKS
MIN_MELD = 3

SUITS = ("s", "h", "d", "c")
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")


@dataclass(frozen=True)
class Meld:
    kind: Literal["sequence", "set"]
    cards: Tuple[int, ...]
    represented_cards: Tuple[int, ...]
    is_pure: bool


@dataclass(frozen=True)
class HandEvaluation:
    is_valid: bool
    melds: Tuple[Meld, ...] = ()


def _require_integral(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("%s must be an integral number, got %r" % (name, type(value).__name__))
    return int(value)


def _require_nonnegative_integral(value: object, name: str) -> int:
    number = _require_integral(value, name)
    if number < 0:
        raise ValueError("%s must be nonnegative, got %d" % (name, number))
    return number


def _require_positive_integral(value: object, name: str) -> int:
    number = _require_integral(value, name)
    if number <= 0:
        raise ValueError("%s must be positive, got %d" % (name, number))
    return number


def _parse_hand(hand: Sequence[int]) -> Tuple[int, ...]:
    try:
        length = len(hand)
    except TypeError as exc:
        raise TypeError("hand must be a sequence of length %d" % NUM_FACES) from exc
    if length != NUM_FACES:
        raise ValueError("hand must have length %d, got %d" % (NUM_FACES, length))

    counts = []
    for index, raw in enumerate(hand):
        counts.append(_require_nonnegative_integral(raw, "hand[%d]" % index))
    return tuple(counts)


def _parse_joker(joker: object) -> int:
    number = _require_integral(joker, "joker")
    if number < 0 or number >= NUM_FACES:
        raise ValueError("joker must be in 0..%d, got %d" % (NUM_FACES - 1, number))
    return number


def _face(suit: int, rank: int) -> int:
    return suit * NUM_RANKS + rank


def _rank_patterns() -> Tuple[Tuple[int, ...], ...]:
    patterns = []
    for start in range(NUM_RANKS):
        for end in range(start + MIN_MELD - 1, NUM_RANKS):
            patterns.append(tuple(range(start, end + 1)))
    for start in range(1, NUM_RANKS - 1):
        # Ace-high runs: start..K,A (QKA, JQKA, ...). No KA2 wrap.
        patterns.append(tuple(range(start, NUM_RANKS)) + (0,))
    return tuple(patterns)


RANK_PATTERNS = _rank_patterns()


def _subtract(counts: Tuple[int, ...], used: Sequence[int]) -> Tuple[int, ...]:
    next_counts = list(counts)
    for face in used:
        next_counts[face] -= 1
        if next_counts[face] < 0:
            raise RuntimeError("internal count underflow")
    return tuple(next_counts)


def _wild_faces(joker: int, pure: bool) -> Tuple[int, ...]:
    if pure:
        return (joker,)
    rank = joker % NUM_RANKS
    return tuple(_face(suit, rank) for suit in range(NUM_SUITS))


def _iter_allocations(
    counts: Tuple[int, ...],
    represented: Sequence[int],
    joker: int,
    pure: bool,
    require_actual: Optional[int] = None,
) -> Iterator[Tuple[int, ...]]:
    wilds = _wild_faces(joker, pure)
    n = len(represented)
    actuals = [0] * n
    pool = list(counts)

    def place(slot: int, used_required: bool) -> Iterator[Tuple[int, ...]]:
        if slot == n:
            if require_actual is None or used_required:
                yield tuple(actuals)
            return

        target = represented[slot]
        candidates = []
        if pool[target] > 0:
            candidates.append(target)
        for wild in wilds:
            if wild != target and pool[wild] > 0:
                candidates.append(wild)

        for card in candidates:
            pool[card] -= 1
            actuals[slot] = card
            next_used = used_required or (require_actual is not None and card == require_actual)
            yield from place(slot + 1, next_used)
            pool[card] += 1

    return place(0, False)


def _meld_from(
    kind: Literal["sequence", "set"],
    actuals: Tuple[int, ...],
    represented: Tuple[int, ...],
    joker: int,
) -> Meld:
    if kind == "set":
        is_pure = False
    else:
        is_pure = all(a == r or a == joker for a, r in zip(actuals, represented))
    return Meld(kind=kind, cards=actuals, represented_cards=represented, is_pure=is_pure)


def _iter_sequences(
    counts: Tuple[int, ...],
    joker: int,
    pure_only: bool,
    anchor: Optional[int] = None,
) -> Iterator[Tuple[Meld, Tuple[int, ...]]]:
    total = sum(counts)
    anchor_is_wild = anchor in _wild_faces(joker, pure_only)
    for suit in range(NUM_SUITS):
        for pattern in RANK_PATTERNS:
            if len(pattern) > total:
                continue
            represented = tuple(_face(suit, rank) for rank in pattern)
            if anchor is not None and anchor not in represented and not anchor_is_wild:
                continue
            for actuals in _iter_allocations(
                counts, represented, joker, pure_only, require_actual=anchor,
            ):
                meld = _meld_from("sequence", actuals, represented, joker)
                yield meld, _subtract(counts, actuals)


def _iter_sets(
    counts: Tuple[int, ...], joker: int, anchor: int,
) -> Iterator[Tuple[Meld, Tuple[int, ...]]]:
    wilds = set(_wild_faces(joker, pure=False))
    anchor_rank = anchor % NUM_RANKS
    anchor_is_wild = anchor in wilds

    ranks: Sequence[int]
    if anchor_is_wild:
        ranks = range(NUM_RANKS)
    else:
        ranks = (anchor_rank,)

    suit_combos = (
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
        (0, 1, 2, 3),
    )

    for rank in ranks:
        for suits in suit_combos:
            represented = tuple(_face(suit, rank) for suit in suits)
            if not anchor_is_wild and anchor not in represented:
                continue
            for actuals in _iter_allocations(counts, represented, joker, pure=False, require_actual=anchor):
                meld = _meld_from("set", actuals, represented, joker)
                yield meld, _subtract(counts, actuals)


def _collapse_candidates(
    items: Iterator[Tuple[Meld, Tuple[int, ...]]],
) -> List[Tuple[Meld, Tuple[int, ...]]]:
    best = {}
    for meld, remaining in items:
        key = (remaining, meld.is_pure)
        best.setdefault(key, (meld, remaining))
    ordered = list(best.values())
    ordered.sort(key=lambda item: (len(item[0].cards), item[0].kind, item[0].represented_cards))
    return ordered


def _search(
    counts: Tuple[int, ...],
    pure_needed: int,
    joker: int,
    memo: Dict[Tuple[Tuple[int, ...], int], Optional[Tuple[Meld, ...]]],
) -> Optional[Tuple[Meld, ...]]:
    key = (counts, pure_needed)
    if key in memo:
        return memo[key]

    total = sum(counts)
    if total == 0:
        result: Optional[Tuple[Meld, ...]] = () if pure_needed == 0 else None
        memo[key] = result
        return result
    if total < MIN_MELD:
        memo[key] = None
        return None
    if pure_needed * MIN_MELD > total:
        memo[key] = None
        return None

    if pure_needed > 0:
        # Do not pin the global lowest card: it may belong only in a later set.
        raw = _iter_sequences(counts, joker, pure_only=True, anchor=None)
    else:
        anchor = next(face for face, count in enumerate(counts) if count)

        def all_melds() -> Iterator[Tuple[Meld, Tuple[int, ...]]]:
            yield from _iter_sequences(counts, joker, pure_only=False, anchor=anchor)
            yield from _iter_sets(counts, joker, anchor=anchor)

        raw = all_melds()

    for meld, remaining in _collapse_candidates(raw):
        next_pure = max(0, pure_needed - meld.is_pure)
        suffix = _search(remaining, next_pure, joker, memo)
        if suffix is not None:
            result = (meld,) + suffix
            memo[key] = result
            return result

    memo[key] = None
    return None


def evaluate_hand(
    hand: Sequence[int],
    joker: int,
    required_sequences: int = 5,
    cards_in_hand: int = 21,
) -> HandEvaluation:
    """Return whether hand is a winning declaration and a full meld witness.

    Invalid hands return is_valid False with an empty melds tuple. Wrong total
    card count is a losing declaration, not a validation error. Malformed
    arguments raise TypeError or ValueError.
    """
    counts = _parse_hand(hand)
    joker_face = _parse_joker(joker)
    pure_needed = _require_nonnegative_integral(required_sequences, "required_sequences")
    target_cards = _require_positive_integral(cards_in_hand, "cards_in_hand")

    if sum(counts) != target_cards:
        return HandEvaluation(is_valid=False)

    melds = _search(counts, pure_needed, joker_face, {})
    if melds is None:
        return HandEvaluation(is_valid=False)
    return HandEvaluation(is_valid=True, melds=melds)


def is_valid_hand(
    hand: Sequence[int],
    joker: int,
    required_sequences: int = 5,
    cards_in_hand: int = 21,
) -> bool:
    return evaluate_hand(hand, joker, required_sequences, cards_in_hand).is_valid


def hand_reward(
    hand: Sequence[int],
    joker: int,
    required_sequences: int = 5,
    cards_in_hand: int = 21,
) -> float:
    return 1.0 if is_valid_hand(hand, joker, required_sequences, cards_in_hand) else 0.0
