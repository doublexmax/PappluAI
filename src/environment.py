"""Solo finite-deck Papplu environment for Monte Carlo Q training.

Phases are DRAW -> DISCARD -> (DRAW | TERMINAL). Actions are a fixed 54-slot
mask: 0 draw stock, 1 take discard top, 2..53 discard face 0..51. Terminal
reward is the evaluator binary hand_reward on the post-discard hand of size
cards_in_hand. Illegal actions and bad config raise without mutating state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
import random
from typing import List, Optional, Tuple

from src.evaluate import NUM_FACES, NUM_RANKS, NUM_SUITS, RANK_PATTERNS, hand_reward, is_valid_hand

ACTION_DRAW_STOCK = 0
ACTION_TAKE_DISCARD = 1
NUM_DISCARD_ACTIONS = NUM_FACES
NUM_ACTIONS = 2 + NUM_DISCARD_ACTIONS

# State layout (float32):
#   [0:52]   hand counts / cards_in_hand
#   [52:104] discard-top one-hot (all zero if empty)
#   [104:156] joker face one-hot
#   [156:159] phase one-hot DRAW, DISCARD, TERMINAL
#   [159]    turns_remaining / max_turns
#   [160]    stock_remaining / (num_decks * 52)
#   [161]    num_decks / 6
#   [162]    cards_in_hand / 30
#   [163]    required_sequences / 10
ENCODING_VERSION = 1
STATE_DIM = NUM_FACES * 3 + 3 + 2 + 3

_MAX_DECKS_NORM = 6.0
_MAX_CARDS_NORM = 30.0
_MAX_SEQ_NORM = 10.0


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
        if set(data) != set(GameConfig.__dataclass_fields__):
            raise ValueError("game_config must contain all four game settings")
        return GameConfig(**data)


@dataclass(frozen=True)
class Observation:
    """Immutable public view of a solo game. Stock order is not exposed."""

    hand: Tuple[int, ...]
    discard_pile: Tuple[int, ...]
    joker: int
    phase: Phase
    turns_remaining: int
    stock_remaining: int
    config: GameConfig
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
        if obs.stock_remaining > 0:
            mask[ACTION_DRAW_STOCK] = True
        if obs.discard_pile:
            mask[ACTION_TAKE_DISCARD] = True
        return tuple(mask)
    for face, count in enumerate(obs.hand):
        if count > 0:
            mask[discard_action(face)] = True
    return tuple(mask)


def encode_observation(obs: Observation) -> Tuple[float, ...]:
    """Fixed-length network input. No hidden stock order."""
    cfg = obs.config
    hand_scale = float(cfg.cards_in_hand)
    deck_scale = float(cfg.num_decks * NUM_FACES)
    turn_scale = float(cfg.max_turns)

    vec = [0.0] * STATE_DIM
    for face, count in enumerate(obs.hand):
        vec[face] = count / hand_scale

    top = obs.discard_top
    if top is not None:
        vec[NUM_FACES + top] = 1.0

    vec[NUM_FACES * 2 + obs.joker] = 1.0

    phase_base = NUM_FACES * 3
    vec[phase_base + int(obs.phase.value)] = 1.0
    vec[phase_base + 3] = obs.turns_remaining / turn_scale
    vec[phase_base + 4] = obs.stock_remaining / deck_scale
    vec[phase_base + 5] = cfg.num_decks / _MAX_DECKS_NORM
    vec[phase_base + 6] = cfg.cards_in_hand / _MAX_CARDS_NORM
    vec[phase_base + 7] = cfg.required_sequences / _MAX_SEQ_NORM
    return tuple(vec)


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
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            hand=tuple(self._hand),
            discard_pile=tuple(self._discard),
            joker=self._joker,
            phase=self._phase,
            turns_remaining=self._turns_remaining,
            stock_remaining=len(self._stock),
            config=self.config,
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
            if action == ACTION_DRAW_STOCK:
                card = self._stock.pop()
            else:
                card = self._discard.pop()
            self._hand[card] += 1
            self._phase = Phase.DISCARD
            self._last_reward = 0.0
            return self.observe()

        face = action_to_discard_face(action)
        next_hand = self._hand.copy()
        next_hand[face] -= 1

        reward = hand_reward(
            next_hand,
            self._joker,
            required_sequences=self.config.required_sequences,
            cards_in_hand=self.config.cards_in_hand,
        )
        self._hand = next_hand
        self._discard.append(face)
        self._turns_remaining -= 1
        self._last_reward = reward
        self._won = reward == 1.0
        done = self._won or self._warm_start or self._turns_remaining <= 0
        self._phase = Phase.TERMINAL if done else Phase.DRAW
        return self.observe()

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
    if config.cards_in_hand < 3:
        raise ValueError("warm starts require at least three cards")
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
        hand[stock.pop()] += 1
        return hand, joker, stock

    raise ValueError(
        "could not build warm-start deal for decks=%d hand=%d seq=%d after 200 attempts"
        % (
            config.num_decks,
            config.cards_in_hand,
            config.required_sequences,
        )
    )
