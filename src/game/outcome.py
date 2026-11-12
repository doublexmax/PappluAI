from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.evaluate import HandPenalty, minimum_penalty
from src.game.multiplayer import TerminalSnapshot


CAP_POLICIES = ("undecided", "points", "weighted-points", "tie", "exclude")


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
        utility = (
            sum(self.margin(seat, other) for other in range(count) if other != seat) / (count - 1)
            if count > 1 else 1.0 if self.winner == 0 else -1.0
        )
        return SeatOutcome(utility, self.winner == seat, self.penalties[seat].points)
