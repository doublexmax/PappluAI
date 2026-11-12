from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.evaluate import HandPenalty, minimum_penalty
from src.game.multiplayer import TerminalSnapshot


CAP_POLICIES = ("undecided", "points", "weighted-points", "tie", "exclude")
DECLARATION_UTILITY = 1.0
DEADWOOD_WEIGHT = 0.5
SOLO_HALF_CREDIT_POINTS = 40


def placements(points: Sequence[int], winner: Optional[int], ending: str, cap_policy: str) -> Optional[tuple[int, ...]]:
    if cap_policy not in CAP_POLICIES:
        raise ValueError("unknown cap policy")
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
    if cap_policy in ("points", "weighted-points"):
        return tuple(1 + sum(other < mine for other in points) for mine in points)
    if cap_policy == "tie":
        return (1,) * len(points)
    return None


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

    @classmethod
    def score(
        cls, snapshot: TerminalSnapshot, winner: Optional[int], ending: str,
        required_sequences: int,
    ) -> "DealOutcome":
        return cls(winner, ending, tuple(
            minimum_penalty(hand, snapshot.joker, required_sequences)
            for hand in snapshot.hands
        ))

    def ranks(self) -> tuple[int, ...]:
        if len(self.penalties) == 1:
            return (1,)
        return placements(
            tuple(penalty.points for penalty in self.penalties),
            self.winner, self.ending, "points",
        )

    def margin(self, seat: int, other: int) -> int:
        ranks = self.ranks()
        return int(ranks[seat] < ranks[other]) - int(ranks[seat] > ranks[other])

    def seat(self, seat: int) -> SeatOutcome:
        count = len(self.penalties)
        points = self.penalties[seat].points
        utility = (
            sum(
                (DECLARATION_UTILITY if self.winner in (seat, other) else DEADWOOD_WEIGHT)
                * self.margin(seat, other)
                for other in range(count) if other != seat
            ) / (count - 1)
            if count > 1 else DECLARATION_UTILITY if self.winner == 0
            else -1.0 + DEADWOOD_WEIGHT * SOLO_HALF_CREDIT_POINTS / (SOLO_HALF_CREDIT_POINTS + points)
        )
        return SeatOutcome(utility, self.winner == seat, points)


def solo_outcome(
    hand: Sequence[int], joker: int, declared: bool, required_sequences: int,
) -> SeatOutcome:
    return DealOutcome.score(
        TerminalSnapshot((tuple(hand),), joker),
        0 if declared else None, "win" if declared else "turns_exhausted",
        required_sequences,
    ).seat(0)
