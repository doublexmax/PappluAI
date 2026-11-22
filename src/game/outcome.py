from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.evaluate import HandPenalty, minimum_penalty
from src.game.multiplayer import TerminalSnapshot


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
