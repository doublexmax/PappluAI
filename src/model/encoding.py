from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, ENCODING_VERSION, NUM_ACTIONS, NUM_DISCARD_ACTIONS,
    GameConfig, Observation, Phase, discard_action,
)
from src.game.information import EventKind, InformationEvent
from src.model.spec import TokenSchema


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
    ACT_DRAW = 14
    ACT_DISCARD = 15
    HAND_INIT = 16
    SEAT = 17
    HAND = 18


FIELDS = 8
RULE_TOKENS = 7
INITIAL_HAND_TOKENS = 22
HEADER_TOKENS = RULE_TOKENS + INITIAL_HAND_TOKENS
SEAT_TOKENS = 6
MAX_FIELD_VALUE = 254
PAD = bytes(FIELDS)
SCHEMA = TokenSchema("papplu-events", ENCODING_VERSION, FIELDS, 30, NUM_ACTIONS)


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
    if config.cards_in_hand > INITIAL_HAND_TOKENS:
        raise ValueError(f"token schema supports at most {INITIAL_HAND_TOKENS} cards per hand")
    if config.max_turns > MAX_FIELD_VALUE:
        raise ValueError(f"token schema supports at most {MAX_FIELD_VALUE} turns")
    free_cards = config.num_decks * NUM_DISCARD_ACTIONS - config.cards_in_hand - 1
    if free_cards > MAX_FIELD_VALUE:
        raise ValueError(f"token schema supports at most {MAX_FIELD_VALUE} cards outside hands and the indicator")


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


def _face(face: int | None) -> int:
    return NUM_DISCARD_ACTIONS if face is None else face


def _token(
    kind: int, *, seat: int = -1, face: int = -1, offered_face: int = -1, count: int = -1,
    clock: int = -1, stock: int = -1, meta: int = -1,
) -> bytes:
    return bytes((kind, seat + 1, face + 1, offered_face + 1, count + 1, clock + 1, stock + 1, meta + 1))


def _event(event: InformationEvent) -> bytes:
    return _token(
        Kind.EVENT_INITIAL + event.kind, seat=event.actor, face=_face(event.face),
        offered_face=_face(event.offered_top), count=event.hand_size,
        clock=min(event.turns_remaining, MAX_FIELD_VALUE), stock=min(event.stock_remaining, MAX_FIELD_VALUE),
        meta=event.next_player,
    )


def _hand(kind: Kind, counts: tuple[int, ...], limit: int) -> bytes:
    cards = b"".join(_token(kind, face=face, count=copies) for face, copies in enumerate(counts) if copies)
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
            _token(Kind.RULE_DECKS, count=config.num_decks),
            _token(Kind.RULE_HAND, count=config.cards_in_hand),
            _token(Kind.RULE_SEQUENCES, count=config.required_sequences),
            _token(Kind.RULE_TURNS, clock=config.max_turns),
            _token(Kind.RULE_RECYCLE, count=int(config.recycle_discard)),
            _token(Kind.RULE_PLAYERS, count=len(obs.players)),
            _token(Kind.JOKER, face=obs.joker),
            _hand(Kind.HAND_INIT, obs.history[0].initial_hand, INITIAL_HAND_TOKENS),
        ))
    prefix += b"".join(_event(event) for event in obs.history[since:])
    suffix = _token(
        Kind.ACT_DRAW if obs.phase is Phase.DRAW else Kind.ACT_DISCARD, seat=0, face=_face(obs.discard_top),
        count=min(len(obs.discard_pile), MAX_FIELD_VALUE), clock=min(obs.turns_remaining, MAX_FIELD_VALUE),
        stock=min(obs.stock_remaining, MAX_FIELD_VALUE), meta=obs.players[0].next_player,
    )
    suffix += b"".join(
        _token(Kind.SEAT, seat=seat, count=view.hand_size, clock=min(view.turns_remaining, MAX_FIELD_VALUE),
               meta=16 + (int(view.active) << 3) + view.next_player)
        for seat, view in enumerate(obs.players)
    )
    suffix += b"".join(_token(Kind.SEAT, seat=seat, meta=0) for seat in range(len(obs.players), SEAT_TOKENS))
    suffix += _hand(Kind.HAND, obs.hand, SCHEMA.suffix_tokens - 1 - SEAT_TOKENS)
    return Encoded(prefix, events, suffix, legal)
