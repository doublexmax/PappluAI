"""Solo finite-deck Papplu environment for Monte Carlo Q training.

Phases are DRAW -> DISCARD -> (DRAW | TERMINAL). Actions are a fixed 54-slot
mask: 0 draw stock, 1 take discard top, 2..53 discard face 0..51. Terminal
reward is the evaluator binary hand_reward on the post-discard hand of size
cards_in_hand. Illegal actions and bad config raise without mutating state.
Older discards refill an empty stock after a nonterminal discard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
import math
import random
from typing import List, Optional, Tuple

from src.evaluate import NUM_FACES, NUM_RANKS, NUM_SUITS, RANK_PATTERNS, hand_reward, is_valid_hand
from src.game.reward_cache import RewardCache
from src.game.stock import refill_stock
from src.game.information import (
    EventKind,
    InformationEvent,
    InformationHistory,
    InformationJournal,
    PlayerView,
    player_views,
)

ACTION_DRAW_STOCK = 0
ACTION_TAKE_DISCARD = 1
NUM_DISCARD_ACTIONS = NUM_FACES
NUM_ACTIONS = 2 + NUM_DISCARD_ACTIONS

ENCODING_VERSION = 2


class Phase(Enum):
    DRAW = 0
    DISCARD = 1
    TERMINAL = 2


@dataclass(frozen=True)
class GameConfig:
    num_decks: int = 3
    cards_in_hand: int = 21
    required_sequences: int = 5
    max_turns: int = 40
    recycle_discard: bool = True

    def __post_init__(self) -> None:
        _require_positive_int(self.num_decks, "num_decks")
        _require_positive_int(self.cards_in_hand, "cards_in_hand")
        if isinstance(self.required_sequences, bool) or not isinstance(
            self.required_sequences, int
        ):
            raise TypeError("required_sequences must be an int")
        if self.required_sequences < 0:
            raise ValueError("required_sequences must be nonnegative")
        _require_positive_int(self.max_turns, "max_turns")
        if type(self.recycle_discard) is not bool:
            raise TypeError("recycle_discard must be a bool")
        total = self.num_decks * NUM_FACES
        if total < self.cards_in_hand + 2:
            raise ValueError(
                "deck supply %d cannot deal hand %d plus indicator and discard"
                % (total, self.cards_in_hand)
            )
        if self.required_sequences * 3 > self.cards_in_hand:
            raise ValueError(
                "required_sequences %d needs at least %d cards, hand is %d"
                % (
                    self.required_sequences,
                    self.required_sequences * 3,
                    self.cards_in_hand,
                )
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "GameConfig":
        if not isinstance(data, dict):
            raise TypeError("game_config must be a dict")
        fields = set(GameConfig.__dataclass_fields__)
        if set(data) == fields:
            return GameConfig(**data)
        raise ValueError("game_config must contain all five game settings")


@dataclass(frozen=True)
class Observation:
    """The observer's own cards, public table, and complete visible history."""

    hand: Tuple[int, ...]
    discard_pile: Tuple[int, ...]
    joker: int
    phase: Phase
    turns_remaining: int
    stock_remaining: int
    config: GameConfig
    players: Tuple[PlayerView, ...]
    history: Tuple[InformationEvent, ...]
    last_reward: float = 0.0
    won: bool = False
    warm_start: bool = False

    @property
    def discard_top(self) -> Optional[int]:
        if not self.discard_pile:
            return None
        return self.discard_pile[-1]

    @property
    def done(self) -> bool:
        return self.phase is Phase.TERMINAL

    def to_dict(self) -> dict:
        return {
            **self.to_snapshot(),
            "config": self.config.to_dict(),
            "joker": self.joker,
            "history": tuple(event.to_dict() for event in self.history),
        }

    def to_snapshot(self) -> dict:
        return {
            "hand": self.hand,
            "discard_pile": self.discard_pile,
            "phase": self.phase.value,
            "turns_remaining": self.turns_remaining,
            "stock_remaining": self.stock_remaining,
            "players": tuple(asdict(player) for player in self.players),
            "last_reward": self.last_reward,
            "won": self.won,
            "warm_start": self.warm_start,
        }

    @staticmethod
    def from_dict(data: dict) -> "Observation":
        fields = set(Observation.__dataclass_fields__)
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("observation must use the current information schema")
        config = GameConfig.from_dict(data["config"])
        if not isinstance(data["players"], tuple):
            raise ValueError("observation needs 1..6 player views")
        history = InformationHistory.from_dict(
            data["history"], config, len(data["players"]),
        )
        snapshot = {
            key: value for key, value in data.items()
            if key not in {"config", "joker", "history"}
        }
        return Observation.from_snapshot(
            snapshot, history=history, prefix=len(history.events), joker=data["joker"],
        )

    @staticmethod
    def from_snapshot(
        data: dict, *, history: InformationHistory, prefix: int, joker: int,
    ) -> "Observation":
        fields = set(Observation.__dataclass_fields__) - {"config", "joker", "history"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("observation snapshot must use the current information schema")
        if type(prefix) is not int or not 1 <= prefix <= len(history.events):
            raise ValueError("observation history prefix must be in 1..%d" % len(history.events))
        config = history.config
        expected = history.states[prefix - 1]

        def integer(value: object, lower: int, upper: int, name: str) -> int:
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("observation %s is invalid" % name)
            return value

        def hand_counts(value: object) -> Tuple[int, ...]:
            if not isinstance(value, tuple) or len(value) != NUM_FACES:
                raise ValueError("observation hand must contain 52 counts")
            return tuple(integer(v, 0, config.num_decks, "card count") for v in value)

        hand = hand_counts(data["hand"])
        discard = data["discard_pile"]
        if not isinstance(discard, tuple):
            raise ValueError("observation discard_pile must be a tuple")
        discard = tuple(integer(v, 0, NUM_FACES - 1, "discard") for v in discard)
        raw_players = data["players"]
        if not isinstance(raw_players, tuple) or len(raw_players) != history.player_count:
            raise ValueError("observation snapshot changes journal player count")
        count = len(raw_players)
        players = []
        for raw in raw_players:
            if not isinstance(raw, dict) or set(raw) != set(PlayerView.__dataclass_fields__):
                raise ValueError("observation player fields are invalid")
            if type(raw["active"]) is not bool:
                raise ValueError("observation player active must be a bool")
            players.append(PlayerView(
                integer(raw["hand_size"], 0, config.cards_in_hand + 1, "hand_size"),
                integer(raw["turns_remaining"], 0, config.max_turns, "player turns"),
                integer(raw["next_player"], 0, count - 1, "next_player"),
                integer(raw["previous_player"], 0, count - 1, "previous_player"),
                raw["active"],
            ))
        if players[0].hand_size != sum(hand):
            raise ValueError("observation own player hand_size does not match hand")
        if expected.hand != hand:
            raise ValueError("observation own card history does not match current hand")
        if players[0].turns_remaining != data["turns_remaining"]:
            raise ValueError("observation own player turns do not match current turns")
        if players[0].turns_remaining != expected.players[0].turns_remaining:
            raise ValueError("observation own player turns do not match history")
        if tuple(players) != expected.players:
            raise ValueError("observation player views do not match history prefix")
        if discard != expected.discard_pile or data["stock_remaining"] != expected.stock_remaining:
            raise ValueError("observation public table does not match history prefix")
        event = history.events[prefix - 1]
        expected_phase = {
            EventKind.INITIAL: (
                Phase.DISCARD if sum(history.events[0].initial_hand) > config.cards_in_hand
                else Phase.DRAW
            ),
            EventKind.DRAW_STOCK: Phase.DISCARD,
            EventKind.TAKE_DISCARD: Phase.DISCARD,
            EventKind.DISCARD: Phase.DRAW,
            EventKind.REFILL: Phase.DRAW,
            EventKind.TERMINAL: Phase.TERMINAL,
        }[event.kind]
        phase = Phase(integer(data["phase"], 0, 2, "phase"))
        if phase is not expected_phase:
            raise ValueError("observation phase does not match history prefix")
        for name in ("won", "warm_start"):
            if type(data[name]) is not bool:
                raise ValueError("observation %s must be a bool" % name)
        reward = data["last_reward"]
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError("observation last_reward must be finite")
        warm_start = (
            history.player_count == 1
            and sum(history.events[0].initial_hand) == config.cards_in_hand + 1
        )
        if data["warm_start"] != warm_start:
            raise ValueError("observation warm_start does not match initial history")
        expected_reward = (
            hand_reward(
                hand, joker, required_sequences=config.required_sequences,
                cards_in_hand=config.cards_in_hand,
            )
            if phase is Phase.TERMINAL and event.actor == 0 else 0.0
        )
        if reward != expected_reward or data["won"] != bool(expected_reward):
            raise ValueError("observation reward and win state do not match history")
        return Observation(
            hand, discard,
            integer(joker, 0, NUM_FACES - 1, "joker"), phase,
            integer(data["turns_remaining"], 0, config.max_turns, "turns_remaining"),
            integer(data["stock_remaining"], 0, config.num_decks * NUM_FACES, "stock_remaining"),
            config, tuple(players), history.events[:prefix], float(reward),
            data["won"], data["warm_start"],
        )


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an int, got %r" % (name, type(value).__name__))
    if value <= 0:
        raise ValueError("%s must be positive, got %d" % (name, value))
    return value


def discard_action(face: int) -> int:
    if face < 0 or face >= NUM_FACES:
        raise ValueError("face must be in 0..%d, got %d" % (NUM_FACES - 1, face))
    return 2 + face


def action_to_discard_face(action: int) -> int:
    if action < 2 or action >= NUM_ACTIONS:
        raise ValueError("not a discard action: %d" % action)
    return action - 2


def legal_action_mask(obs: Observation) -> Tuple[bool, ...]:
    mask = [False] * NUM_ACTIONS
    if obs.phase is Phase.TERMINAL:
        return tuple(mask)
    if obs.phase is Phase.DRAW:
        mask[ACTION_DRAW_STOCK] = bool(obs.stock_remaining)
        if obs.discard_pile:
            mask[ACTION_TAKE_DISCARD] = True
        return tuple(mask)
    for face, count in enumerate(obs.hand):
        if count > 0:
            mask[discard_action(face)] = True
    return tuple(mask)


class PappluEnv:
    """Mutable solo game. reset/step return immutable Observation snapshots."""

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self.config = config if config is not None else GameConfig()
        self._rng = random.Random()
        self._stock: List[int] = []
        self._hand: List[int] = [0] * NUM_FACES
        self._discard: List[int] = []
        self._joker = 0
        self._phase = Phase.TERMINAL
        self._turns_remaining = 0
        self._last_reward = 0.0
        self._won = False
        self._warm_start = False
        self._reward_cache = RewardCache()
        self._journal = InformationJournal(1)

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        cfg = self.config
        deck: List[int] = []
        for face in range(NUM_FACES):
            deck.extend([face] * cfg.num_decks)
        self._rng.shuffle(deck)

        self._hand = [0] * NUM_FACES
        for _ in range(cfg.cards_in_hand):
            card = deck.pop()
            self._hand[card] += 1

        self._joker = deck.pop()

        self._discard = [deck.pop()]
        self._stock = deck
        self._phase = Phase.DRAW
        self._turns_remaining = cfg.max_turns
        self._last_reward = 0.0
        self._won = False
        self._warm_start = False
        self._start_history()
        return self.observe()

    def reset_warm_start(self, seed: Optional[int] = None) -> Observation:
        """Discard-phase start: winning hand plus one extra card.

        The agent only chooses the discard. Reward is still hand_reward on the
        resulting cards_in_hand-sized hand. Not a held-out full-game win.
        """
        if seed is not None:
            self._rng.seed(seed)
        cfg = self.config
        hand, joker, stock = build_warm_start_deal(cfg, self._rng)
        self._hand = list(hand)
        self._joker = joker
        self._stock = list(stock)
        self._discard = []
        self._phase = Phase.DISCARD
        self._turns_remaining = cfg.max_turns
        self._last_reward = 0.0
        self._won = False
        self._warm_start = True
        self._start_history()
        return self.observe()

    def reset_curriculum(
        self, distance: int, seed: Optional[int] = None
    ) -> Observation:
        if isinstance(distance, bool) or not isinstance(distance, int):
            raise TypeError("distance must be an int")
        if distance < 1 or distance > self.config.cards_in_hand:
            raise ValueError(
                "distance must be in 1..%d, got %d"
                % (self.config.cards_in_hand, distance)
            )

        rng = random.Random()
        if seed is None:
            rng.setstate(self._rng.getstate())
        else:
            rng.seed(seed)

        cfg = self.config
        for _ in range(200):
            winning_hand, joker, remaining = _build_natural_winning_deal(cfg, rng)
            physical_hand = [
                face
                for face, count in enumerate(winning_hand)
                for _ in range(count)
            ]
            removed = [
                physical_hand[index]
                for index in rng.sample(range(len(physical_hand)), distance)
            ]
            removed_faces = set(removed)
            candidates = [
                index
                for index, face in enumerate(remaining)
                if face not in removed_faces
            ]
            if len(candidates) < distance:
                continue
            replacement_indices = set(rng.sample(candidates, distance))
            replacements = [
                face
                for index, face in enumerate(remaining)
                if index in replacement_indices
            ]

            hand = winning_hand.copy()
            for face in removed:
                hand[face] -= 1
            for face in replacements:
                hand[face] += 1
            if is_valid_hand(
                hand,
                joker,
                required_sequences=cfg.required_sequences,
                cards_in_hand=cfg.cards_in_hand,
            ):
                continue

            deck = [
                face
                for index, face in enumerate(remaining)
                if index not in replacement_indices
            ]
            deck.extend(removed)
            rng.shuffle(deck)
            discard = [deck.pop()]

            self._rng.setstate(rng.getstate())
            self._hand = hand
            self._joker = joker
            self._stock = deck
            self._discard = discard
            self._phase = Phase.DRAW
            self._turns_remaining = cfg.max_turns
            self._last_reward = 0.0
            self._won = False
            self._warm_start = False
            self._start_history()
            return self.observe()

        raise ValueError(
            "could not build curriculum deal at distance %d for decks=%d "
            "hand=%d seq=%d after 200 attempts"
            % (
                distance,
                cfg.num_decks,
                cfg.cards_in_hand,
                cfg.required_sequences,
            )
        )

    def observe(self) -> Observation:
        return Observation(
            hand=tuple(self._hand),
            discard_pile=tuple(self._discard),
            joker=self._joker,
            phase=self._phase,
            turns_remaining=self._turns_remaining,
            stock_remaining=len(self._stock),
            config=self.config,
            players=player_views([self._hand], [self._turns_remaining], 0),
            history=self._journal.observe(0),
            last_reward=self._last_reward,
            won=self._won,
            warm_start=self._warm_start,
        )

    def step(self, action: int) -> Observation:
        if self._phase is Phase.TERMINAL:
            raise ValueError("episode is terminal; call reset")
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action must be an int, got %r" % type(action).__name__)
        if action < 0 or action >= NUM_ACTIONS:
            raise ValueError(
                "action must be in 0..%d, got %d" % (NUM_ACTIONS - 1, action)
            )

        mask = legal_action_mask(self.observe())
        if not mask[action]:
            raise ValueError(
                "illegal action %d in phase %s" % (action, self._phase.name)
            )

        if self._phase is Phase.DRAW:
            offered_top = self._discard[-1] if self._discard else None
            if action == ACTION_DRAW_STOCK:
                card = self._stock.pop()
                kind = EventKind.DRAW_STOCK
            else:
                card = self._discard.pop()
                kind = EventKind.TAKE_DISCARD
            self._hand[card] += 1
            self._phase = Phase.DISCARD
            self._last_reward = 0.0
            self._record(kind, face=card, offered_top=offered_top)
            return self.observe()

        face = action_to_discard_face(action)
        next_hand = self._hand.copy()
        next_hand[face] -= 1

        reward = self._reward_cache.score(
            next_hand,
            self._joker,
            required_sequences=self.config.required_sequences,
            cards_in_hand=self.config.cards_in_hand,
            evaluator=hand_reward,
        )
        self._hand = next_hand
        self._discard.append(face)
        self._turns_remaining -= 1
        self._last_reward = reward
        self._won = reward == 1.0
        self._record(EventKind.DISCARD, face=face)
        done = self._won or self._warm_start or self._turns_remaining <= 0
        if done:
            self._phase = Phase.TERMINAL
            self._record(EventKind.TERMINAL)
        else:
            self._refill_stock()
            self._phase = Phase.DRAW
        return self.observe()

    def _start_history(self) -> None:
        self._journal.start(
            [self._hand],
            [self._turns_remaining],
            len(self._stock),
            self._discard[-1] if self._discard else None,
        )

    def _record(
        self,
        kind: EventKind,
        face: Optional[int] = None,
        offered_top: Optional[int] = None,
    ) -> None:
        self._journal.record(
            kind, 0, sum(self._hand), self._turns_remaining,
            len(self._stock), 0, face, offered_top,
        )

    def _refill_stock(self) -> None:
        if refill_stock(
            self._stock, self._discard, self._rng, self.config.recycle_discard,
        ):
            self._record(EventKind.REFILL)

    def total_cards_in_play(self) -> int:
        """Physical cards still tracked (excludes removed joker indicator)."""
        return sum(self._hand) + len(self._stock) + len(self._discard)


def build_warm_start_deal(
    config: GameConfig, rng: random.Random
) -> Tuple[List[int], int, List[int]]:
    """Return (hand_counts with +1 extra, joker, remaining stock faces).

    Builds natural melds under deck multiplicity, reserves the indicator, and
    adds one extra card from leftover supply. Fails with ValueError rather than
    hanging when the config cannot support a constructed win.
    """
    hand, joker, stock = _build_natural_winning_deal(config, rng)
    hand[stock.pop()] += 1
    return hand, joker, stock


def _build_natural_winning_deal(
    config: GameConfig, rng: random.Random
) -> Tuple[List[int], int, List[int]]:
    if config.cards_in_hand < 3:
        raise ValueError("constructed starts require at least three cards")
    lengths = [3] * (config.cards_in_hand // 3)
    lengths[0] += config.cards_in_hand % 3
    for _ in range(200):
        hand = [0] * NUM_FACES
        supply = [config.num_decks] * NUM_FACES
        joker = rng.randrange(NUM_FACES)
        supply[joker] -= 1
        for group_index, size in enumerate(lengths):
            candidates = [
                tuple(suit * NUM_RANKS + rank for rank in pattern)
                for suit in range(NUM_SUITS)
                for pattern in RANK_PATTERNS
                if len(pattern) == size
            ]
            if group_index >= config.required_sequences and size <= NUM_SUITS:
                candidates.extend(
                    tuple(suit * NUM_RANKS + rank for suit in suits)
                    for rank in range(NUM_RANKS)
                    for suits in combinations(range(NUM_SUITS), size)
                )
            available = [
                meld for meld in candidates if all(supply[face] > 0 for face in meld)
            ]
            if not available:
                break
            for face in rng.choice(available):
                hand[face] += 1
                supply[face] -= 1
        if sum(hand) != config.cards_in_hand:
            continue
        if not is_valid_hand(
            hand,
            joker,
            required_sequences=config.required_sequences,
            cards_in_hand=config.cards_in_hand,
        ):
            raise RuntimeError("constructed natural melds failed the evaluator")

        stock: List[int] = []
        for face, left in enumerate(supply):
            stock.extend([face] * left)
        rng.shuffle(stock)
        return hand, joker, stock

    raise ValueError(
        "could not build natural winning deal for decks=%d hand=%d seq=%d "
        "after 200 attempts"
        % (
            config.num_decks,
            config.cards_in_hand,
            config.required_sequences,
        )
    )
