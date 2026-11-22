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
    required_sequences: int = 5

    @classmethod
    def score(
        cls, snapshot: TerminalSnapshot, winner: Optional[int], ending: str,
        required_sequences: int,
    ) -> "DealOutcome":
        return cls(winner, ending, tuple(
            minimum_penalty(hand, snapshot.joker, required_sequences)
            for hand in snapshot.hands
        ), required_sequences)

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
        penalty = self.penalties[seat]
        points = min(penalty.points, 100)
        if self.winner == seat:
            utility = 1.0
        elif self.winner is not None:
            utility = -0.35 - 0.50 * points / 100.0
        else:
            required = max(1, self.required_sequences)
            seq_frac = (
                min(penalty.qualifying_sequences, required) / required
                if self.required_sequences > 0 else 1.0
            )
            utility = -0.80 + 0.20 * seq_frac - 0.20 * points / 100.0
        return SeatOutcome(utility, self.winner == seat, penalty.points)


def solo_outcome(
    hand: Sequence[int], joker: int, declared: bool, required_sequences: int,
) -> SeatOutcome:
    return DealOutcome.score(
        TerminalSnapshot((tuple(hand),), joker),
        0 if declared else None, "win" if declared else "turns_exhausted",
        required_sequences,
    ).seat(0)
