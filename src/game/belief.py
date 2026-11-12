from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.evaluate import NUM_FACES
from src.game.environment import Observation
from src.game.information import EventKind, InformationEvent


@dataclass(frozen=True)
class SeatLedger:
    took: tuple[int, ...]
    held: tuple[int, ...]
    discarded: tuple[int, ...]
    passed: tuple[int, ...]
    last_take: tuple[int, ...]
    last_discard: tuple[int, ...]
    last_pass: tuple[int, ...]
    takes: int
    passes: int


@dataclass(frozen=True)
class TableLedger:
    seats: tuple[SeatLedger, ...]
    pile: tuple[int, ...]
    refill_pool: Optional[tuple[int, ...]]
    refills: int

    @classmethod
    def from_history(
        cls, history: Sequence[InformationEvent], player_count: int,
    ) -> "TableLedger":
        seats = [[[0] * NUM_FACES for _ in range(7)] for _ in range(player_count)]
        takes, passes = [0] * player_count, [0] * player_count
        pile, refill_pool, refills = [], None, 0
        for step, event in enumerate(history, 1):
            took, held, discarded, passed, last_take, last_discard, last_pass = seats[event.actor]
            if event.kind is EventKind.INITIAL:
                pile = [] if event.offered_top is None else [event.offered_top]
            elif event.kind is EventKind.TAKE_DISCARD:
                pile.pop()
                took[event.face] += 1
                held[event.face] += 1
                last_take[event.face] = step
                takes[event.actor] += 1
            elif event.kind is EventKind.DISCARD:
                pile.append(event.face)
                held[event.face] = max(0, held[event.face] - 1)
                discarded[event.face] += 1
                last_discard[event.face] = step
            elif event.kind is EventKind.DRAW_STOCK:
                if refill_pool is not None and event.face is not None:
                    refill_pool[event.face] = max(0, refill_pool[event.face] - 1)
                if event.offered_top is not None:
                    passed[event.offered_top] += 1
                    last_pass[event.offered_top] = step
                    passes[event.actor] += 1
            elif event.kind is EventKind.REFILL:
                refill_pool = [0] * NUM_FACES
                for face in pile[:-1]:
                    refill_pool[face] += 1
                pile = pile[-1:]
                refills += 1
        return cls(
            tuple(SeatLedger(
                *(tuple(channel) for channel in channels), takes[seat], passes[seat],
            ) for seat, channels in enumerate(seats)),
            tuple(pile), None if refill_pool is None else tuple(refill_pool), refills,
        )


@dataclass(frozen=True)
class BeliefState:
    ledger: TableLedger
    joker: int
    unseen: tuple[int, ...]
    stock_weights: tuple[int, ...]

    @classmethod
    def from_observation(cls, obs: Observation) -> "BeliefState":
        ledger = TableLedger.from_history(obs.history, len(obs.players))
        pile_counts = [0] * NUM_FACES
        for face in obs.discard_pile:
            pile_counts[face] += 1
        unseen = tuple(
            max(0, obs.config.num_decks - obs.hand[face] - pile_counts[face]
                - int(face == obs.joker) - sum(seat.held[face] for seat in ledger.seats[1:]))
            for face in range(NUM_FACES)
        )
        weights = unseen if ledger.refill_pool is None else tuple(
            min(ledger.refill_pool[face], unseen[face]) for face in range(NUM_FACES)
        )
        return cls(ledger, obs.joker, unseen, weights if sum(weights) else unseen)

    def stock_probabilities(self) -> tuple[tuple[int, float], ...]:
        total = sum(self.stock_weights)
        return tuple(
            (face, weight / total) for face, weight in enumerate(self.stock_weights) if weight
        )

    def take_rate(self, seat: int) -> float:
        ledger = self.ledger.seats[seat]
        return (1 + ledger.takes) / (5 + ledger.takes + ledger.passes)

    def feed_risk(self, face: int, seat: int = 1) -> float:
        if not 0 <= seat < len(self.ledger.seats):
            return 0.0
        ledger = self.ledger.seats[seat]
        if ledger.discarded[face] or ledger.passed[face]:
            return 0.0
        wanted = 0.0
        rank = face % 13
        for known, copies in enumerate(ledger.held):
            if not copies:
                continue
            known_rank = known % 13
            if face == known or rank == self.joker % 13 or known_rank == self.joker % 13:
                weight = 1.0
            elif face // 13 == known // 13:
                distance = abs(rank - known_rank)
                if rank == 0 or known_rank == 0:
                    distance = min(distance, 13 - distance)
                weight = 0.6 if distance == 1 else 0.3 if distance == 2 else 0.0
            else:
                weight = 0.5 if rank == known_rank else 0.0
            wanted = max(wanted, weight)
        return self.take_rate(seat) * wanted
