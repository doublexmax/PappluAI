from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import random
from typing import Any, Optional

from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, Observation,
    discard_action, legal_action_mask,
)
from src.game.information import EventKind, InformationHistory
from src.game.outcome import SeatOutcome
from src.model.encoding import Features, encode, legal_actions


@dataclass(frozen=True)
class Decision:
    observation: Observation
    features: Features
    action: int
    hidden_hands: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.action) is not int or self.action not in legal_actions(self.features):
            raise ValueError("decision must choose a legal action")
        if not legal_action_mask(self.observation)[self.action]:
            raise ValueError("decision action does not match its observation")
        hands, obs = self.hidden_hands, self.observation
        if not isinstance(hands, tuple) or len(hands) != 6 or any(
            not isinstance(hand, tuple) or len(hand) != 52
            or any(type(value) is not int or not 0 <= value <= obs.config.num_decks for value in hand)
            for hand in hands
        ):
            raise ValueError("hidden_hands must contain six tuples of 52 card counts")
        if hands[0] != tuple(obs.hand):
            raise ValueError("hidden_hands observer does not match observation")
        if any(any(hand) for hand in hands[len(obs.players):]):
            raise ValueError("hidden_hands absent seats must be empty")
        if any(sum(hands[seat]) != player.hand_size for seat, player in enumerate(obs.players)):
            raise ValueError("hidden_hands sizes do not match player views")


@dataclass(frozen=True)
class Trajectory:
    players: int
    decisions: tuple[Decision, ...]
    outcome: SeatOutcome

    def __post_init__(self) -> None:
        if type(self.players) is not int or not 1 <= self.players <= 6:
            raise ValueError("trajectory players must be in 1..6")
        if not isinstance(self.decisions, tuple) or not self.decisions:
            raise ValueError("trajectory decisions must be a nonempty tuple")
        if any(len(decision.observation.players) != self.players for decision in self.decisions):
            raise ValueError("trajectory changes player count")
        outcome = self.outcome
        if (
            not isinstance(outcome, SeatOutcome)
            or type(outcome.utility) not in (int, float)
            or not math.isfinite(outcome.utility) or not -1 <= outcome.utility <= 1
            or type(outcome.won) is not bool
            or type(outcome.penalty) is not int or outcome.penalty < 0
        ):
            raise ValueError("trajectory outcome is invalid")


@dataclass(frozen=True)
class Sample:
    decision: Decision
    outcome: SeatOutcome
    bootstrap: Optional[Decision]
    scale: float


class TrajectoryReplay:
    STATE_VERSION = 3

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._trajectories: deque[Trajectory] = deque()
        self._slots: dict[int, list[tuple[Trajectory, int]]] = {}
        self._starts: dict[int, int] = {}
        self._size = 0
        self._sample_cursor = 0

    def __len__(self) -> int:
        return self._size

    def append(self, trajectory: Trajectory) -> None:
        stored = replace(trajectory, decisions=trajectory.decisions[-self.capacity:])
        self._trajectories.append(stored)
        slots = self._slots.setdefault(stored.players, [])
        self._starts.setdefault(stored.players, 0)
        slots.extend((stored, index) for index in range(len(stored.decisions)))
        self._size += len(stored.decisions)
        while self._size > self.capacity:
            oldest = self._trajectories[0]
            removed = min(self._size - self.capacity, len(oldest.decisions))
            self._starts[oldest.players] += removed
            self._size -= removed
            if removed == len(oldest.decisions):
                self._trajectories.popleft()
            else:
                self._trajectories[0] = replace(oldest, decisions=oldest.decisions[removed:])
        for count in tuple(self._slots):
            start = self._starts[count]
            if start == len(self._slots[count]):
                del self._slots[count]
                del self._starts[count]
            elif start > len(self._slots[count]) // 2:
                self._slots[count] = self._slots[count][start:]
                self._starts[count] = 0

    def sample(
        self, rng: random.Random, batch_size: int, *, n_step: int = 5, gamma: float = 1.0,
    ) -> list[Sample]:
        for name, value in (("batch_size", batch_size), ("n_step", n_step)):
            if type(value) is not int or value < 1:
                raise ValueError("%s must be a positive integer" % name)
        if type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not self._size:
            raise ValueError("cannot sample empty replay")
        counts = sorted(self._slots)
        samples = []
        for row in range(batch_size):
            count = counts[(self._sample_cursor + row) % len(counts)]
            slots, start = self._slots[count], self._starts[count]
            trajectory, index = slots[rng.randrange(start, len(slots))]
            decisions = trajectory.decisions
            boot = index + n_step
            samples.append(Sample(
                decisions[index], trajectory.outcome,
                decisions[boot] if boot < len(decisions) else None,
                gamma ** (n_step if boot < len(decisions) else len(decisions) - 1 - index),
            ))
        self._sample_cursor += batch_size
        return samples

    def state_dict(self) -> dict[str, Any]:
        trajectories = []
        for trajectory in self._trajectories:
            final = trajectory.decisions[-1].observation
            history = final.history
            decisions = []
            previous = 0
            for decision in trajectory.decisions:
                obs = decision.observation
                prefix = len(obs.history)
                if (
                    obs.config != final.config or obs.joker != final.joker
                    or obs.warm_start != final.warm_start or len(obs.players) != trajectory.players
                ):
                    raise ValueError("replay trajectory changes journal configuration")
                if prefix < previous or obs.history != history[:prefix]:
                    raise ValueError("replay trajectory mixes seat or deal histories")
                decisions.append((
                    obs.to_snapshot(), prefix, decision.action,
                    tuple(value for hand in decision.hidden_hands for value in hand),
                ))
                previous = prefix
            outcome = trajectory.outcome
            trajectories.append({
                "config": final.config.to_dict(),
                "joker": final.joker,
                "player_count": trajectory.players,
                "warm_start": final.warm_start,
                "history": tuple(event.to_dict() for event in history),
                "decisions": tuple(decisions),
                "outcome": (outcome.utility, outcome.won, outcome.penalty),
            })
        return {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "sample_cursor": self._sample_cursor,
            "trajectories": tuple(trajectories),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != {
            "version", "capacity", "sample_cursor", "trajectories",
        }:
            raise ValueError("replay state fields do not match the journal schema")
        if type(state["version"]) is not int or state["version"] != self.STATE_VERSION:
            raise ValueError("unsupported replay state version %r" % state["version"])
        if type(state["capacity"]) is not int or state["capacity"] != self.capacity:
            raise ValueError("replay capacity does not match")
        cursor = state["sample_cursor"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError("replay sample_cursor must be a nonnegative integer")
        if not isinstance(state["trajectories"], tuple):
            raise ValueError("replay trajectories must be a tuple")
        candidate = TrajectoryReplay(self.capacity)
        for journal in state["trajectories"]:
            if not isinstance(journal, dict) or set(journal) != {
                "config", "joker", "player_count", "warm_start", "history", "decisions", "outcome",
            }:
                raise ValueError("replay trajectory fields do not match the journal schema")
            config = GameConfig.from_dict(journal["config"])
            history = InformationHistory.from_dict(journal["history"], config, journal["player_count"])
            if type(journal["warm_start"]) is not bool:
                raise ValueError("replay warm_start must be a boolean")
            if not isinstance(journal["decisions"], tuple) or not journal["decisions"]:
                raise ValueError("replay decisions must be a nonempty tuple")
            decisions, previous = [], 0
            for raw in journal["decisions"]:
                if not isinstance(raw, tuple) or len(raw) != 4:
                    raise ValueError("replay decision must be a four-item tuple")
                snapshot, prefix, action, flat = raw
                obs = Observation.from_snapshot(
                    snapshot, history=history, prefix=prefix, joker=journal["joker"],
                )
                if prefix < previous or obs.warm_start != journal["warm_start"]:
                    raise ValueError("replay history prefixes or warm_start are inconsistent")
                previous = prefix
                if not isinstance(flat, tuple) or len(flat) != 312:
                    raise ValueError("replay hidden_hands must contain 312 counts")
                decision = Decision(
                    obs, encode(obs), action,
                    tuple(flat[seat * 52:(seat + 1) * 52] for seat in range(6)),
                )
                if prefix < len(history.events):
                    event = history.events[prefix]
                    expected = (
                        ACTION_DRAW_STOCK if event.kind is EventKind.DRAW_STOCK
                        else ACTION_TAKE_DISCARD if event.kind is EventKind.TAKE_DISCARD
                        else discard_action(event.face) if event.kind is EventKind.DISCARD
                        else None
                    )
                    if event.actor != 0 or action != expected:
                        raise ValueError("replay action does not match the next history event")
                decisions.append(decision)
            raw_outcome = journal["outcome"]
            if not isinstance(raw_outcome, tuple) or len(raw_outcome) != 3:
                raise ValueError("replay outcome must contain utility, won, and penalty")
            trajectory = Trajectory(journal["player_count"], tuple(decisions), SeatOutcome(*raw_outcome))
            if len(candidate) + len(decisions) > self.capacity:
                raise ValueError("replay contains more decisions than its capacity")
            candidate.append(trajectory)
        self._trajectories = candidate._trajectories
        self._slots, self._starts = candidate._slots, candidate._starts
        self._size, self._sample_cursor = len(candidate), cursor
