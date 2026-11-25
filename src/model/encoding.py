from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from src.evaluate import NUM_RANKS, NUM_SUITS
from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, ENCODING_VERSION, NUM_ACTIONS, NUM_DISCARD_ACTIONS,
    GameConfig, Observation, Phase, discard_action,
)
from src.game.information import EventKind, InformationEvent
from src.model.spec import BYTE_VALUES, TokenSchema


class Kind(IntEnum):
    PAD = 0
    RULE_DECKS = 1
    RULE_HAND = 2
    RULE_SEQUENCES = 3
    RULE_TURNS = 4
    RULE_RECYCLE = 5
    RULE_PLAYERS = 6
    JOKER = 7
    EVENT_INITIAL = 8
    ACT = 14
    TABLE_STATE = 15
    HAND_INIT = 16
    SEAT = 17
    HAND = 18


FIELDS = 8
RULE_TOKENS = 7
INITIAL_HAND_TOKENS = 22
HEADER_TOKENS = RULE_TOKENS + INITIAL_HAND_TOKENS
SEAT_TOKENS = 6
TABLE_TOKENS = 2
RULE_OFFSET = 20
MAX_TURNS = BYTE_VALUES - 1 - RULE_OFFSET
MAX_DECKS = 4
NONE_FACE = NUM_DISCARD_ACTIONS + 1
PAD = bytes(FIELDS)
SCHEMA = TokenSchema("papplu-events", ENCODING_VERSION, FIELDS, 30, NUM_ACTIONS, pointer_field=7)


@dataclass(frozen=True)
class Encoded:
    prefix: bytes
    events: int
    suffix: bytes
    legal: int

    @property
    def cut(self) -> int:
        return HEADER_TOKENS + self.events


def validate_game(config: GameConfig) -> None:
    if config.cards_in_hand >= INITIAL_HAND_TOKENS:
        raise ValueError(f"token schema supports at most {INITIAL_HAND_TOKENS - 1} cards before drawing")
    if config.max_turns > MAX_TURNS:
        raise ValueError(f"token schema supports at most {MAX_TURNS} turns")
    if config.num_decks > MAX_DECKS:
        raise ValueError(f"token schema supports at most {MAX_DECKS} decks")


def legal_mask(obs: Observation) -> int:
    if obs.done or not obs.players[0].active:
        return 0
    last = obs.history[-1]
    if obs.phase is Phase.DRAW:
        if last.kind is not EventKind.INITIAL and last.next_player != 0:
            return 0
        take = bool(obs.discard_pile) and not (
            obs.stock_remaining and len(obs.players) == 1
            and last.kind in (EventKind.DISCARD, EventKind.REFILL) and last.actor == 0
        )
        return (int(bool(obs.stock_remaining)) << ACTION_DRAW_STOCK) | (int(take) << ACTION_TAKE_DISCARD)
    if last.actor != 0:
        return 0
    bits = sum(1 << discard_action(face) for face, count in enumerate(obs.hand) if count)
    if last.kind is EventKind.TAKE_DISCARD and last.actor == 0 and last.face is not None:
        taken = 1 << discard_action(last.face)
        if bits & ~taken:
            bits &= ~taken
    return bits


def _face_byte(face: int | None) -> int:
    return NONE_FACE if face is None else face + 1


def _card(face: int | None) -> tuple[int, int, int]:
    if face is None:
        return NONE_FACE, NUM_SUITS + 1, NUM_RANKS + 1
    suit, rank = divmod(face, NUM_RANKS)
    return face + 1, suit + 1, rank + 1


def _token(
    kind: int, *, seat_or_value: int = 0, card: tuple[int, int, int] = (0, 0, 0),
    offered_or_count: int = 0, clock_or_stock: int = 0, pointer: int = 0,
) -> bytes:
    return bytes((kind, seat_or_value, *card, offered_or_count, clock_or_stock, pointer))


def _clock(turns: int) -> int:
    return 1 + min(turns, 120)


def _event(event: InformationEvent) -> bytes:
    return _token(
        Kind.EVENT_INITIAL + event.kind, seat_or_value=1 + event.actor, card=_card(event.face),
        offered_or_count=_face_byte(event.offered_top), clock_or_stock=_clock(event.turns_remaining),
    )


def _hand(kind: Kind, counts: tuple[int, ...], limit: int) -> bytes:
    cards = b"".join(
        _token(kind, seat_or_value=10 + copies, card=_card(face),
               pointer=1 + discard_action(face) if kind == Kind.HAND else 0)
        for face, copies in enumerate(counts) if copies
    )
    if len(cards) > limit * FIELDS:
        raise ValueError("too many distinct faces for the token schema")
    return cards.ljust(limit * FIELDS, b"\x00")


def encode(obs: Observation, since: int = 0) -> Encoded:
    events = len(obs.history)
    if type(since) is not int or not 0 <= since <= events or not events:
        raise ValueError("invalid event cursor or missing initial event")
    legal = legal_mask(obs)
    if not legal:
        raise ValueError("cannot encode an observation with no legal actions")
    prefix = b""
    if since == 0:
        validate_game(obs.config)
        config = obs.config
        prefix = b"".join((
            _token(Kind.RULE_DECKS, seat_or_value=RULE_OFFSET + config.num_decks),
            _token(Kind.RULE_HAND, seat_or_value=RULE_OFFSET + config.cards_in_hand),
            _token(Kind.RULE_SEQUENCES, seat_or_value=RULE_OFFSET + config.required_sequences),
            _token(Kind.RULE_TURNS, seat_or_value=RULE_OFFSET + config.max_turns),
            _token(Kind.RULE_RECYCLE, seat_or_value=RULE_OFFSET + int(config.recycle_discard)),
            _token(Kind.RULE_PLAYERS, seat_or_value=RULE_OFFSET + len(obs.players)),
            _token(Kind.JOKER, card=_card(obs.joker)),
            _hand(Kind.HAND_INIT, obs.history[0].initial_hand, INITIAL_HAND_TOKENS),
        ))
    prefix += b"".join(_event(event) for event in obs.history[since:])
    suffix = _token(
        Kind.ACT, seat_or_value=1, card=_card(obs.discard_top), offered_or_count=100 + min(len(obs.discard_pile), 150),
        clock_or_stock=_clock(obs.turns_remaining),
    )
    own = obs.players[0]
    suffix += _token(
        Kind.TABLE_STATE, seat_or_value=7 + obs.phase.value,
        offered_or_count=83 + 2 * own.next_player + int(own.active),
        clock_or_stock=130 + min(obs.stock_remaining // 2, 120),
    )
    suffix += b"".join(
        _token(Kind.SEAT, seat_or_value=1 + seat, offered_or_count=60 + view.hand_size,
               clock_or_stock=_clock(view.turns_remaining))
        for seat, view in enumerate(obs.players)
    )
    suffix += b"".join(_token(Kind.SEAT, seat_or_value=1 + seat) for seat in range(len(obs.players), SEAT_TOKENS))
    suffix += _hand(Kind.HAND, obs.hand, SCHEMA.suffix_tokens - TABLE_TOKENS - SEAT_TOKENS)
    return Encoded(prefix, events, suffix, legal)
