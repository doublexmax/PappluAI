from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import List, Optional, Tuple

from src.game.environment import (
    ACTION_DRAW_STOCK,
    NUM_ACTIONS,
    GameConfig,
    Observation,
    Phase,
    action_to_discard_face,
    legal_action_mask,
)
from src.evaluate import NUM_FACES, hand_reward
from src.game.reward_cache import RewardCache
from src.game.stock import refill_stock


@dataclass(frozen=True)
class MatchConfig:
    game: GameConfig = field(default_factory=lambda: GameConfig(max_turns=60))
    players: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.game, GameConfig):
            raise TypeError("game must be a GameConfig")
        if isinstance(self.players, bool) or not isinstance(self.players, int):
            raise TypeError("players must be an int")
        if self.players < 2 or self.players > 6:
            raise ValueError("players must be in 2..6")
        required = self.game.cards_in_hand * self.players + 2
        available = self.game.num_decks * NUM_FACES
        if available < required:
            raise ValueError(
                "deck supply %d cannot deal %d hands of %d plus indicator and discard"
                % (available, self.players, self.game.cards_in_hand)
            )


@dataclass(frozen=True)
class MatchView:
    current_seat: int
    phase: Phase
    winner: Optional[int]
    done: bool
    terminal_reason: Optional[str]
    turns_remaining: Tuple[int, ...]
    stock_remaining: int


@dataclass(frozen=True)
class MatchTelemetry:
    stock_draws: Tuple[int, ...]
    discard_draws: Tuple[int, ...]
    refill_turns: Tuple[int, ...]


@dataclass(frozen=True)
class TerminalSnapshot:
    hands: Tuple[Tuple[int, ...], ...]
    joker: int


class MultiplayerEnv:
    """Mutable match state with immutable public observations and views."""

    def __init__(self, config: MatchConfig) -> None:
        if not isinstance(config, MatchConfig):
            raise TypeError("config must be a MatchConfig")
        self.config = config
        self._rng = random.Random()
        self._hands: List[List[int]] = [
            [0] * NUM_FACES for _ in range(self.config.players)
        ]
        self._stock: List[int] = []
        self._discard: List[int] = []
        self._joker = 0
        self._phase = Phase.TERMINAL
        self._current_seat = 0
        self._winner: Optional[int] = None
        self._terminal_reason: Optional[str] = None
        self._turns_remaining = [0] * self.config.players
        self._stock_draws = [0] * self.config.players
        self._discard_draws = [0] * self.config.players
        self._refill_turns: List[int] = []
        self._last_reward = 0.0
        self._reward_cache = RewardCache()

    def reset(self, seed: int) -> MatchView:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self._rng.seed(seed)
        deck = [
            face
            for face in range(NUM_FACES)
            for _ in range(self.config.game.num_decks)
        ]
        self._rng.shuffle(deck)

        self._hands = [
            [0] * NUM_FACES for _ in range(self.config.players)
        ]
        for _ in range(self.config.game.cards_in_hand):
            for hand in self._hands:
                hand[deck.pop()] += 1

        self._joker = deck.pop()
        self._discard = [deck.pop()]
        self._stock = deck
        self._phase = Phase.DRAW
        self._current_seat = 0
        self._winner = None
        self._terminal_reason = None
        self._turns_remaining = [
            self.config.game.max_turns
            for _ in range(self.config.players)
        ]
        self._stock_draws = [0] * self.config.players
        self._discard_draws = [0] * self.config.players
        self._refill_turns = []
        self._last_reward = 0.0
        return self.view()

    def observe(self) -> Observation:
        seat = self._current_seat
        return Observation(
            hand=tuple(self._hands[seat]),
            discard_pile=tuple(self._discard),
            joker=self._joker,
            phase=self._phase,
            turns_remaining=self._turns_remaining[seat],
            stock_remaining=len(self._stock),
            config=self.config.game,
            last_reward=self._last_reward,
            won=self._winner == seat,
            warm_start=False,
        )

    def step(self, action: int) -> MatchView:
        if self._phase is Phase.TERMINAL:
            raise ValueError("match is terminal; call reset")
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action must be an int, got %r" % type(action).__name__)
        if action < 0 or action >= NUM_ACTIONS:
            raise ValueError(
                "action must be in 0..%d, got %d" % (NUM_ACTIONS - 1, action)
            )
        observation = self.observe()
        if not legal_action_mask(observation)[action]:
            raise ValueError(
                "illegal action %d in phase %s" % (action, self._phase.name)
            )

        hand = self._hands[self._current_seat]
        if self._phase is Phase.DRAW:
            if action == ACTION_DRAW_STOCK:
                self._refill_stock()
                card = self._stock.pop()
                self._stock_draws[self._current_seat] += 1
            else:
                card = self._discard.pop()
                self._discard_draws[self._current_seat] += 1
            hand[card] += 1
            self._phase = Phase.DISCARD
            self._last_reward = 0.0
            return self.view()

        face = action_to_discard_face(action)
        next_hand = hand.copy()
        next_hand[face] -= 1
        reward = self._reward_cache.score(
            next_hand,
            self._joker,
            required_sequences=self.config.game.required_sequences,
            cards_in_hand=self.config.game.cards_in_hand,
            evaluator=hand_reward,
        )

        seat = self._current_seat
        self._hands[seat] = next_hand
        self._discard.append(face)
        self._turns_remaining[seat] -= 1
        self._last_reward = float(reward)
        if reward == 1.0:
            self._winner = seat
            self._terminal_reason = "win"
            self._phase = Phase.TERMINAL
            return self.view()

        next_seat = self._next_active_seat(seat)
        if next_seat is None:
            self._terminal_reason = "turns_exhausted"
            self._phase = Phase.TERMINAL
        else:
            self._refill_stock()
            self._current_seat = next_seat
            self._phase = Phase.DRAW
            self._last_reward = 0.0
        return self.view()

    def view(self) -> MatchView:
        return MatchView(
            current_seat=self._current_seat,
            phase=self._phase,
            winner=self._winner,
            done=self._phase is Phase.TERMINAL,
            terminal_reason=self._terminal_reason,
            turns_remaining=tuple(self._turns_remaining),
            stock_remaining=len(self._stock),
        )

    def telemetry(self) -> MatchTelemetry:
        return MatchTelemetry(
            stock_draws=tuple(self._stock_draws),
            discard_draws=tuple(self._discard_draws),
            refill_turns=tuple(self._refill_turns),
        )

    def terminal_snapshot(self) -> TerminalSnapshot:
        if self._phase is not Phase.TERMINAL or self._terminal_reason is None:
            raise ValueError("terminal hands are available only after a completed match")
        return TerminalSnapshot(tuple(tuple(hand) for hand in self._hands), self._joker)

    def _refill_stock(self) -> None:
        if refill_stock(
            self._stock,
            self._discard,
            self._rng,
            self.config.game.recycle_discard,
        ):
            self._refill_turns.append(
                sum(
                    self.config.game.max_turns - remaining
                    for remaining in self._turns_remaining
                )
            )

    def _next_active_seat(self, seat: int) -> Optional[int]:
        for offset in range(1, self.config.players + 1):
            candidate = (seat + offset) % self.config.players
            if self._turns_remaining[candidate] > 0:
                return candidate
        return None
