"""Papplu hand validity, binary declaration reward, and minimum penalty scoring.

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
_SEARCH_SEQUENCES = tuple(
    tuple(_face(suit, rank) for rank in pattern)
    for suit in range(NUM_SUITS)
    for pattern in RANK_PATTERNS
    if len(pattern) <= 5
)


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
    wilds = _wild_faces(joker, pure_only)
    wild_count = sum(counts[face] for face in wilds)
    anchor_is_wild = anchor in wilds
    natural = tuple(count > 0 and face not in wilds for face, count in enumerate(counts))
    # Longer runs split into legal pieces of three to five cards without losing purity.
    for represented in _SEARCH_SEQUENCES:
        if len(represented) > total:
            continue
        if anchor is not None and anchor not in represented and not anchor_is_wild:
            continue
        if len(represented) - sum(natural[face] for face in represented) > wild_count:
            continue
        for actuals in _iter_allocations(
            counts, represented, joker, pure_only, require_actual=anchor,
        ):
            meld = _meld_from("sequence", actuals, represented, joker)
            yield meld, _subtract(counts, actuals)


def _iter_sets(
    counts: Tuple[int, ...], joker: int, anchor: Optional[int] = None,
) -> Iterator[Tuple[Meld, Tuple[int, ...]]]:
    wilds = set(_wild_faces(joker, pure=False))
    anchor_is_wild = anchor in wilds

    ranks: Sequence[int]
    if anchor is None or anchor_is_wild:
        ranks = range(NUM_RANKS)
    else:
        ranks = (anchor % NUM_RANKS,)

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
            if anchor is not None and not anchor_is_wild and anchor not in represented:
                continue
            for actuals in _iter_allocations(counts, represented, joker, pure=False, require_actual=anchor):
                meld = _meld_from("set", actuals, represented, joker)
                yield meld, _subtract(counts, actuals)


def _search(
    counts: Tuple[int, ...],
    pure_needed: int,
    joker: int,
    memo: Dict[Tuple[Tuple[int, ...], int], Optional[Tuple[Meld, ...]]],
) -> Optional[Tuple[Meld, ...]]:
    if sum(counts) < pure_needed * MIN_MELD:
        return None
    faces, options = _compile_meld_options(counts, joker)
    initial = tuple(counts[face] for face in faces)

    def visit(state: Tuple[int, ...], needed: int) -> Optional[Tuple[Meld, ...]]:
        key = (state, needed)
        if key in memo:
            return memo[key]
        total = sum(state)
        if not total:
            result: Optional[Tuple[Meld, ...]] = () if needed == 0 else None
            memo[key] = result
            return result
        if total < MIN_MELD or total < needed * MIN_MELD:
            memo[key] = None
            return None
        anchor = next(index for index, count in enumerate(state) if count)
        for option in options[anchor]:
            if any(state[position] < copies for position, copies in option.take):
                continue
            remaining = list(state)
            for position, copies in option.take:
                remaining[position] -= copies
            suffix = visit(tuple(remaining), max(0, needed - option.meld.is_pure))
            if suffix is not None:
                result = (option.meld,) + suffix
                memo[key] = result
                return result
        memo[key] = None
        return None

    return visit(initial, pure_needed)


@dataclass(frozen=True)
class _MeldOption:
    meld: Meld
    take: Tuple[Tuple[int, int], ...]


def _compile_meld_options(
    counts: Tuple[int, ...], joker: int,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[_MeldOption, ...], ...]]:
    wilds = _wild_faces(joker, pure=False)
    faces = tuple(sorted(
        (face for face, count in enumerate(counts) if count),
        key=lambda face: (face in wilds, face),
    ))
    positions = {face: index for index, face in enumerate(faces)}
    best: Dict[Tuple[Tuple[int, int], ...], Meld] = {}

    def candidates() -> Iterator[Meld]:
        for meld, _ in _iter_sequences(counts, joker, pure_only=False):
            yield meld
        for meld, _ in _iter_sets(counts, joker):
            yield meld

    for meld in candidates():
        used: Dict[int, int] = {}
        for face in meld.cards:
            position = positions[face]
            used[position] = used.get(position, 0) + 1
        take = tuple(sorted(used.items()))
        previous = best.get(take)
        if previous is None or meld.is_pure and not previous.is_pure:
            best[take] = meld
    by_anchor: List[List[_MeldOption]] = [[] for _ in faces]
    for take, meld in best.items():
        by_anchor[take[0][0]].append(_MeldOption(meld, take))
    for options in by_anchor:
        options.sort(key=lambda option: (
            not option.meld.is_pure, len(option.meld.cards),
            option.meld.kind, option.meld.represented_cards, option.take,
        ))
    return faces, tuple(tuple(options) for options in by_anchor)


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


SCORING_VERSION = 1

_RANK_POINTS = (10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)
_UNREACHABLE = float("inf")


@dataclass(frozen=True)
class HandPenalty:
    """Lowest sequence-gated penalty of a hand and one grouping that attains it.

    Every card is in exactly one exempt meld or in counted_cards, a 52-count
    vector like the hand. points is the card points of counted_cards. When the
    grouping holds fewer qualifying sequences than required, every exempt meld
    is a qualifying sequence.
    """

    points: int
    exempt_melds: Tuple[Meld, ...]
    counted_cards: Tuple[int, ...]

    @property
    def qualifying_sequences(self) -> int:
        return sum(meld.kind == "sequence" and meld.is_pure for meld in self.exempt_melds)


def minimum_penalty(
    hand: Sequence[int], joker: int, required_sequences: int = 5,
) -> HandPenalty:
    """Return the lowest unmatched-card penalty over every legal grouping of hand.

    Cards 2 through 10 score their face value, A, J, Q, and K score 10, and
    every card of the joker's rank scores 0. Qualifying sequences are the pure
    sequences that count toward required_sequences, and they always exempt their
    cards. Other melds exempt theirs only in a grouping that also holds the
    required qualifying sequences. The returned grouping meets the quota
    whenever some optimal grouping does. Any hand size is scored. Malformed
    arguments raise like evaluate_hand.
    """
    counts = _parse_hand(hand)
    joker_face = _parse_joker(joker)
    quota = _require_nonnegative_integral(required_sequences, "required_sequences")
    faces, options = _compile_meld_options(counts, joker_face)
    start = tuple(counts[face] for face in faces)
    points = tuple(_card_points(face, joker_face) for face in faces)

    # A grouping below the quota scores the same as its qualifying sequences
    # alone. The minimum is therefore the lower of an all-meld search that must
    # meet the quota and a qualifying-sequence search without one.
    search, need = _PenaltySearch(points, options), quota
    best = search.cost(start, need)
    if best:
        low = _PenaltySearch(points, tuple(
            tuple(option for option in anchored if option.meld.is_pure)
            for anchored in options
        ))
        low_best = low.cost(start, 0)
        if low_best < best:
            search, need, best = low, 0, low_best

    melds = search.witness(start, need)
    counted = counts
    for meld in melds:
        counted = _subtract(counted, meld.cards)
    total = sum(_card_points(face, joker_face) * count for face, count in enumerate(counted))
    if total != best:
        raise RuntimeError("internal penalty witness mismatch")
    return HandPenalty(points=total, exempt_melds=melds, counted_cards=counted)


def _card_points(face: int, joker: int) -> int:
    rank = face % NUM_RANKS
    return 0 if rank == joker % NUM_RANKS else _RANK_POINTS[rank]


class _PenaltySearch:
    """Exact minimum counted points, memoized on (remaining copies, unmet quota).

    The first position holding a card is the anchor. Its card either stays
    counted or joins an option filed under that position.
    """

    def __init__(
        self,
        points: Tuple[int, ...],
        options: Tuple[Tuple[_MeldOption, ...], ...],
    ) -> None:
        self._points = points
        self._scoring = tuple(position for position, value in enumerate(points) if value)
        self._options = options
        self._memo: Dict[Tuple[Tuple[int, ...], int], float] = {}

    def cost(self, state: Tuple[int, ...], need: int) -> float:
        key = (state, need)
        if key not in self._memo:
            self._memo[key] = self._solve(state, need)
        return self._memo[key]

    def witness(self, state: Tuple[int, ...], need: int) -> Tuple[Meld, ...]:
        melds: List[Meld] = []
        while not self._settled(state, need):
            target = self.cost(state, need)
            for meld, rest, rest_need, points in self._branches(state, need):
                if points + self.cost(rest, rest_need) == target:
                    break
            else:
                raise RuntimeError("internal penalty witness mismatch")
            if meld is not None:
                melds.append(meld)
            state, need = rest, rest_need
        return tuple(melds)

    def _settled(self, state: Tuple[int, ...], need: int) -> bool:
        return need == 0 and not any(state[position] for position in self._scoring)

    def _solve(self, state: Tuple[int, ...], need: int) -> float:
        if self._settled(state, need):
            return 0
        if sum(state) < need * MIN_MELD:
            return _UNREACHABLE
        best = _UNREACHABLE
        for _, rest, rest_need, points in self._branches(state, need):
            best = min(best, points + self.cost(rest, rest_need))
            if not best:
                break
        return best

    def _branches(
        self, state: Tuple[int, ...], need: int,
    ) -> Iterator[Tuple[Optional[Meld], Tuple[int, ...], int, int]]:
        anchor = next(position for position, count in enumerate(state) if count)
        for option in self._options[anchor]:
            if all(state[position] >= copies for position, copies in option.take):
                rest = list(state)
                for position, copies in option.take:
                    rest[position] -= copies
                yield option.meld, tuple(rest), max(0, need - option.meld.is_pure), 0
        rest = list(state)
        rest[anchor] -= 1
        yield None, tuple(rest), need, self._points[anchor]
