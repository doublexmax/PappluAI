from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Iterator, Optional, Sequence

from src.evaluate import HandPenalty, MIN_MELD, NUM_FACES, NUM_RANKS, RANK_PATTERNS, minimum_penalty
from src.game.multiplayer import TerminalSnapshot


_RANK_ONES = int.from_bytes(b"\x01" * NUM_RANKS, "little")
_RANK_HIGH = _RANK_ONES << 7
_SUIT_BITS = 8 * NUM_RANKS
_HAND_HIGH = int.from_bytes(b"\x80" * NUM_FACES, "little")
_ACE_HIGH_SHIFTS = (8 * (NUM_RANKS - 2), 8 * (NUM_RANKS - 1))
_ACE_HIGH = 1 | sum(1 << shift for shift in _ACE_HIGH_SHIFTS)
_PAIR_RADIX = 8192
_TRIPLE_RANKS = frozenset(sum(1 << rank for rank in row) for row in RANK_PATTERNS if len(row) == 3)
_PAIR_RANKS = frozenset(
    mask ^ (1 << rank) for mask in _TRIPLE_RANKS for rank in range(NUM_RANKS) if mask & (1 << rank)
)


def _low_bits(bits: int) -> Iterator[int]:
    while bits:
        bit = bits & -bits
        yield bit
        bits ^= bit


def _linear_pairs(cards: int) -> int:
    pairs = 0
    while cards:
        cards >>= ((cards & -cards).bit_length() - 1) & -8
        copies = cards & 255
        cards ^= copies
        pairs += copies // 2
        neighbors = cards & 0xffff00
        if copies & 1 and neighbors:
            cards -= 1 << (((neighbors & -neighbors).bit_length() - 1) & -8)
            pairs += 1
    return pairs


@lru_cache(maxsize=32_768)
def _suit_pairs(cards: int) -> int:
    best = _linear_pairs(cards)
    if cards & 255:
        for shift in _ACE_HIGH_SHIFTS:
            bit = 1 << shift
            if cards & (255 << shift):
                best = max(best, 1 + _suit_pairs(cards - bit - 1))
    return best


def _suit_triples(cards: int, jokers: int) -> Iterator[tuple[int, int]]:
    present = (((cards | _RANK_HIGH) - _RANK_ONES | cards) & _RANK_HIGH) >> 7
    for bit in _low_bits(present & (present >> 8) & (present >> 16)):
        yield bit * 0x10101, 0
    if present & _ACE_HIGH == _ACE_HIGH:
        yield _ACE_HIGH, 0
    repeated = (((cards | _RANK_HIGH) - 3 * _RANK_ONES | cards) & _RANK_HIGH) >> 7
    for bit in _low_bits(repeated):
        yield 3 * bit, 0
    if jokers:
        for span in (8, 16):
            for bit in _low_bits(present & (present >> span)):
                yield bit | (bit << span), 1
        if present & 1:
            for bit in _low_bits(present & (_ACE_HIGH ^ 1)):
                yield 1 | bit, 1
        if jokers >= 2:
            for bit in _low_bits(present):
                yield bit, 2
        if jokers >= 3:
            yield 0, 3


@lru_cache(maxsize=65_536)
def _suit_potential(cards: int, jokers: int) -> int:
    best = _suit_pairs(cards)
    for take, spent in _suit_triples(cards, jokers):
        best = max(best, _PAIR_RADIX + _suit_potential(cards - take, jokers - spent))
    return best


def _extra_triple(cards: tuple[int, int, int], wild_rank: int) -> bool:
    a, b, c = cards
    if a == c:
        return True
    r, s, t = a % NUM_RANKS, b % NUM_RANKS, c % NUM_RANKS
    if r == wild_rank:
        a, r, b, s = b, s, c, t
    elif s == wild_rank:
        b, s = c, t
    elif t != wild_rank:
        return a != b != c and (
            r == s == t or a // NUM_RANKS == c // NUM_RANKS and
            ((1 << r) | (1 << s) | (1 << t)) in _TRIPLE_RANKS
        )
    return r == wild_rank or s == wild_rank or (
        a != b and (r == s or a // NUM_RANKS == b // NUM_RANKS and ((1 << r) | (1 << s)) in _PAIR_RANKS)
    )


def _extra_potential(suits: tuple[int, ...], joker: int, jokers: int, quota: int, limit: int) -> int:
    cards = sum(counts << (suit * _SUIT_BITS) for suit, counts in enumerate(suits))
    joker_bit = 1 << (8 * joker)
    wild_rank = joker % NUM_RANKS
    pure = tuple(dict.fromkeys(
        (take << (suit * _SUIT_BITS), spent)
        for suit, counts in enumerate(suits) for take, spent in _suit_triples(counts, jokers)
    ))

    @lru_cache(maxsize=None)
    def extra(remaining: int, unused: int, need: int, first: int) -> int:
        if not need:
            faces = tuple(
                face for face, copies in enumerate((remaining + unused * joker_bit).to_bytes(NUM_FACES, "little"))
                for _ in range(copies)
            )
            triples = []
            for i, j, k in combinations(range(len(faces)), 3):
                if _extra_triple((faces[i], faces[j], faces[k]), wild_rank):
                    if limit == 1:
                        return 1
                    mask = (1 << i) | (1 << j) | (1 << k)
                    if limit == 2 and any(not mask & prev for prev in triples):
                        return 2
                    triples.append(mask)
            if limit <= 2 or not triples:
                return int(bool(triples))

            def pack(used: int, start: int, left: int) -> int:
                best = 0
                for index in range(start, len(triples)):
                    take = triples[index]
                    if not used & take:
                        best = max(best, 1 if left == 1 else 1 + pack(used | take, index + 1, left - 1))
                        if best == left:
                            break
                return best

            return pack(0, 0, limit)
        best = 0
        for index in range(first, len(pure)):
            take, spent = pure[index]
            fits = (((remaining | _HAND_HIGH) - take | remaining) & _HAND_HIGH) == _HAND_HIGH
            if spent <= unused and fits:
                best = max(best, extra(remaining - take, unused - spent, need - 1, index))
                if best == limit:
                    break
        return best

    return extra(cards, jokers, quota, 0)


def hand_potential(
    hand: Sequence[int], joker: int, required_sequences: int = 5, cards_in_hand: int = 21,
) -> float:
    """Score disjoint triples, then pairs below quota or extra melds at quota, without point values."""
    counts = bytes(hand)
    jokers = counts[joker]
    suits = tuple(
        int.from_bytes(counts[i:i + NUM_RANKS], "little")
        for i in range(0, NUM_FACES, NUM_RANKS)
    )
    if not jokers:
        score = sum(_suit_potential(cards, 0) for cards in suits)
    else:
        offset = joker // NUM_RANKS
        suits = (*suits[:offset], suits[offset] - (jokers << (8 * (joker % NUM_RANKS))), *suits[offset + 1:])
        best = [0] * (jokers + 1)
        for cards in suits:
            row = [_suit_potential(cards, k) for k in range(jokers + 1)]
            best = [max(best[k - n] + row[n] for n in range(k + 1)) for k in range(jokers + 1)]
        score = best[-1]
    sequences, pairs = divmod(score, _PAIR_RADIX)
    extra_limit = max(0, (cards_in_hand - MIN_MELD * required_sequences) // MIN_MELD)
    ceiling = 20 * required_sequences + 4 * extra_limit
    if not ceiling:
        return 0.0
    if sequences < required_sequences:
        points = 20 * sequences + 5 * min(pairs, required_sequences - sequences)
    else:
        extra = extra_limit if sequences >= required_sequences + extra_limit else _extra_potential(
            suits, joker, jokers, required_sequences, extra_limit,
        )
        points = 20 * required_sequences + 4 * extra
    return 0.75 * points / ceiling


def placements(points: Sequence[int], winner: Optional[int], ending: str) -> tuple[int, ...]:
    if len(points) < 2 or any(type(point) is not int or point < 0 for point in points):
        raise ValueError("penalties must contain nonnegative integer points for every player")
    if ending == "win":
        if type(winner) is not int or not 0 <= winner < len(points) or points[winner] != 0:
            raise ValueError("a declaration needs a winning seat with zero penalty")
        return tuple(
            1 if seat == winner else 2 + sum(
                other != winner and points[other] < mine
                for other in range(len(points))
            )
            for seat, mine in enumerate(points)
        )
    if ending != "turns_exhausted" or winner is not None:
        raise ValueError("only completed declarations and draw-limit endings have placements")
    return tuple(1 + sum(other < mine for other in points) for mine in points)


@dataclass(frozen=True)
class SeatOutcome:
    utility: float
    won: bool
    penalty: int


@dataclass(frozen=True)
class DealOutcome:
    winner: Optional[int]
    ending: str
    penalties: tuple[HandPenalty, ...]
    max_points: int
    required_sequences: int = 5

    def __post_init__(self) -> None:
        if not 1 <= len(self.penalties) <= 6 or type(self.max_points) is not int or self.max_points < 1:
            raise ValueError("outcome needs one to six seats and a positive point bound")
        if any(not 0 <= penalty.points <= self.max_points for penalty in self.penalties):
            raise ValueError("penalty exceeds the full hand's point bound")
        if type(self.required_sequences) is not int or self.required_sequences < 0:
            raise ValueError("required_sequences must be a nonnegative integer")
        if self.ending == "win":
            if (
                type(self.winner) is not int
                or not 0 <= self.winner < len(self.penalties)
                or self.penalties[self.winner].points != 0
            ):
                raise ValueError("a declaration needs a winning seat with zero penalty")
        elif self.ending != "turns_exhausted" or self.winner is not None:
            raise ValueError("outcome needs a declaration or expiry")

    @classmethod
    def score(
        cls, snapshot: TerminalSnapshot, winner: Optional[int], ending: str,
        required_sequences: int, cards_in_hand: Optional[int] = None,
    ) -> "DealOutcome":
        if not snapshot.hands:
            raise ValueError("outcome needs at least one hand")
        if cards_in_hand is None:
            cards_in_hand = sum(snapshot.hands[0])
        if type(cards_in_hand) is not int or any(sum(hand) != cards_in_hand for hand in snapshot.hands):
            raise ValueError("final hands must match cards_in_hand")
        return cls(winner, ending, tuple(
            minimum_penalty(hand, snapshot.joker, required_sequences)
            for hand in snapshot.hands
        ), cards_in_hand * 10, required_sequences)

    def ranks(self) -> tuple[int, ...]:
        if len(self.penalties) == 1:
            return (1,)
        return placements(
            tuple(penalty.points for penalty in self.penalties),
            self.winner, self.ending,
        )

    def margin(self, seat: int, other: int) -> int:
        ranks = self.ranks()
        return int(ranks[seat] < ranks[other]) - int(ranks[seat] > ranks[other])

    def seat(self, seat: int) -> SeatOutcome:
        return SeatOutcome(self.utilities()[seat], self.winner == seat, self.penalties[seat].points)

    def utilities(self) -> tuple[float, ...]:
        players = len(self.penalties)
        if self.ending == "win":
            if players == 1:
                return (1.0,)
            return tuple(1.0 if seat == self.winner else -1.0 / (players - 1) for seat in range(players))
        quota = self.required_sequences
        fractions = tuple(
            min(penalty.qualifying_sequences, quota) / quota if quota else 1.0
            for penalty in self.penalties
        )
        if players == 1:
            return (-1.0 + 0.60 * fractions[0],)
        total = sum(fractions)
        return tuple(-0.50 + 0.30 * (fraction - (total - fraction) / (players - 1)) for fraction in fractions)


def solo_outcome(
    hand: Sequence[int], joker: int, declared: bool, required_sequences: int,
) -> SeatOutcome:
    return DealOutcome.score(
        TerminalSnapshot((tuple(hand),), joker),
        0 if declared else None, "win" if declared else "turns_exhausted",
        required_sequences,
    ).seat(0)
