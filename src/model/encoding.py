from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, ENCODING_VERSION, NUM_ACTIONS,
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
NONE = NUM_ACTIONS - 2
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
    if config.max_turns > 255 or config.num_decks * NONE > 4095:
        raise ValueError("token schema supports at most 255 turns and 4095 cards")


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
    return NONE if face is None else face


def _token(*fields: int) -> bytes:
    return bytes(fields).ljust(FIELDS, b"\x00")


def _stock_and_seat(stock: int, seat: int) -> int:
    return ((stock >> 8) << 4) | seat


def _event(event: InformationEvent) -> bytes:
    return _token(
        Kind.EVENT_INITIAL + event.kind, event.actor, _face(event.face), _face(event.offered_top),
        event.hand_size, event.turns_remaining, event.stock_remaining & 255,
        _stock_and_seat(event.stock_remaining, event.next_player),
    )


def _hand(kind: Kind, counts: tuple[int, ...], limit: int) -> bytes:
    cards = b"".join(_token(kind, face, copies) for face, copies in enumerate(counts) if copies)
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
            _token(Kind.RULE_DECKS, config.num_decks),
            _token(Kind.RULE_HAND, config.cards_in_hand),
            _token(Kind.RULE_SEQUENCES, config.required_sequences),
            _token(Kind.RULE_TURNS, config.max_turns),
            _token(Kind.RULE_RECYCLE, int(config.recycle_discard)),
            _token(Kind.RULE_PLAYERS, len(obs.players)),
            _token(Kind.JOKER, obs.joker),
            _hand(Kind.HAND_INIT, obs.history[0].initial_hand, INITIAL_HAND_TOKENS),
        ))
    prefix += b"".join(_event(event) for event in obs.history[since:])
    pile = len(obs.discard_pile)
    suffix = _token(
        Kind.ACT_DRAW if obs.phase is Phase.DRAW else Kind.ACT_DISCARD, obs.phase.value,
        _face(obs.discard_top), pile & 255, obs.turns_remaining, obs.stock_remaining & 255,
        _stock_and_seat(obs.stock_remaining, obs.players[0].next_player), pile >> 8,
    )
    suffix += b"".join(
        _token(Kind.SEAT, 1, seat, int(view.active), view.hand_size, view.turns_remaining,
               view.next_player, view.previous_player)
        for seat, view in enumerate(obs.players)
    )
    suffix += b"".join(_token(Kind.SEAT, 0, seat) for seat in range(len(obs.players), SEAT_TOKENS))
    suffix += _hand(Kind.HAND, obs.hand, SCHEMA.suffix_tokens - 1 - SEAT_TOKENS)
    return Encoded(prefix, events, suffix, legal)
