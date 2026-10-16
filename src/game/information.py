from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

from src.evaluate import NUM_FACES

if TYPE_CHECKING:
    from src.game.environment import GameConfig


class EventKind(IntEnum):
    INITIAL = 0
    DRAW_STOCK = 1
    TAKE_DISCARD = 2
    DISCARD = 3
    REFILL = 4
    TERMINAL = 5


@dataclass(frozen=True)
class InformationEvent:
    kind: EventKind
    actor: int
    hand_size: int
    turns_remaining: int
    stock_remaining: int
    next_player: int
    face: Optional[int] = None
    offered_top: Optional[int] = None
    initial_hand: Tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": int(self.kind),
            "actor": self.actor,
            "hand_size": self.hand_size,
            "turns_remaining": self.turns_remaining,
            "stock_remaining": self.stock_remaining,
            "next_player": self.next_player,
            "face": self.face,
            "offered_top": self.offered_top,
            "initial_hand": self.initial_hand,
        }


@dataclass(frozen=True)
class PlayerView:
    hand_size: int
    turns_remaining: int
    next_player: int
    previous_player: int
    active: bool


def player_views(
    hands: Sequence[Sequence[int]],
    turns: Sequence[int],
    observer: int,
) -> Tuple[PlayerView, ...]:
    return _player_views([sum(hand) for hand in hands], turns, observer)


def _player_views(
    hand_sizes: Sequence[int],
    turns: Sequence[int],
    observer: int,
) -> Tuple[PlayerView, ...]:
    count = len(hand_sizes)
    views = []
    for relative in range(count):
        seat = (observer + relative) % count
        next_seat = next(
            ((seat + offset) % count for offset in range(1, count + 1)
             if turns[(seat + offset) % count] > 0),
            seat,
        )
        previous_seat = next(
            ((seat - offset) % count for offset in range(1, count + 1)
             if turns[(seat - offset) % count] > 0),
            seat,
        )
        views.append(PlayerView(
            hand_sizes[seat],
            turns[seat],
            (next_seat - observer) % count,
            (previous_seat - observer) % count,
            turns[seat] > 0,
        ))
    return tuple(views)


@dataclass(frozen=True)
class InformationState:
    hand: Tuple[int, ...]
    players: Tuple[PlayerView, ...]
    discard_pile: Tuple[int, ...]
    stock_remaining: int


@dataclass(frozen=True)
class InformationHistory:
    config: "GameConfig"
    player_count: int
    events: Tuple[InformationEvent, ...]
    states: Tuple[InformationState, ...]

    @staticmethod
    def from_dict(
        data: object, config: "GameConfig", player_count: int,
    ) -> "InformationHistory":
        if type(player_count) is not int or not 1 <= player_count <= 6:
            raise ValueError("observation needs 1..6 player views")
        if not isinstance(data, tuple) or not data:
            raise ValueError("observation needs its complete initial history")

        def integer(value: object, lower: int, upper: int, name: str) -> int:
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("observation %s is invalid" % name)
            return value

        def face(value: object) -> Optional[int]:
            return None if value is None else integer(value, 0, NUM_FACES - 1, "face")

        events = []
        states = []
        hand_sizes = [config.cards_in_hand] * player_count
        turns = [config.max_turns] * player_count
        known_hand = []
        hand = ()
        discard = ()
        stock = 0
        pending_actor = None
        next_actor = None
        fields = set(InformationEvent.__dataclass_fields__)
        for index, raw in enumerate(data):
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("observation event fields are invalid")
            kind = EventKind(integer(raw["kind"], 0, len(EventKind) - 1, "event kind"))
            actor = integer(raw["actor"], 0, player_count - 1, "actor")
            visible_face = face(raw["face"])
            initial = raw["initial_hand"]
            if index == 0:
                if kind is not EventKind.INITIAL or actor != 0 or visible_face is not None:
                    raise ValueError("observation history must begin with the own initial hand")
                if not isinstance(initial, tuple) or len(initial) != NUM_FACES:
                    raise ValueError("observation hand must contain 52 counts")
                initial = tuple(
                    integer(v, 0, config.num_decks, "card count") for v in initial
                )
            elif kind is EventKind.INITIAL or initial != ():
                raise ValueError("observation has an invalid later initial event")
            if kind is EventKind.DRAW_STOCK and actor != 0 and visible_face is not None:
                raise ValueError("observation exposes an opponent stock card")
            if kind is EventKind.DRAW_STOCK and actor == 0 and visible_face is None:
                raise ValueError("observation is missing an own stock draw face")
            if kind in (EventKind.REFILL, EventKind.TERMINAL) and visible_face is not None:
                raise ValueError("observation exposes a face on a non-card event")
            if kind in (EventKind.TAKE_DISCARD, EventKind.DISCARD) and visible_face is None:
                raise ValueError("observation public card event needs a face")
            event = InformationEvent(
                kind, actor,
                integer(raw["hand_size"], 0, config.cards_in_hand + 1, "event hand_size"),
                integer(raw["turns_remaining"], 0, config.max_turns, "event turns"),
                integer(raw["stock_remaining"], 0, config.num_decks * NUM_FACES, "event stock"),
                integer(raw["next_player"], 0, player_count - 1, "event next_player"),
                visible_face, face(raw["offered_top"]), initial,
            )
            if index == 0:
                known_hand = list(initial)
                hand = initial
                hand_sizes[0] = sum(initial)
                if hand_sizes[0] not in (config.cards_in_hand, config.cards_in_hand + 1):
                    raise ValueError("observation own card history has an invalid initial hand")
                if event.turns_remaining != config.max_turns:
                    raise ValueError("observation initial turns must equal the configured budget")
                warm_start = hand_sizes[0] == config.cards_in_hand + 1
                if warm_start:
                    if player_count != 1 or event.offered_top is not None:
                        raise ValueError("observation initial extra card requires a solo warm start")
                    expected_stock = config.num_decks * NUM_FACES - hand_sizes[0] - 1
                else:
                    if event.offered_top is None:
                        raise ValueError("observation initial ordinary deal requires a discard")
                    expected_stock = config.num_decks * NUM_FACES - player_count * config.cards_in_hand - 2
                if event.stock_remaining != expected_stock:
                    raise ValueError("observation initial stock does not match the deal")
                pending_actor = 0 if warm_start else None
                stock = event.stock_remaining
                discard = () if event.offered_top is None else (event.offered_top,)
            else:
                previous = events[-1]
                if previous.kind is EventKind.TERMINAL:
                    raise ValueError("observation event chronology continues after terminal")
                if kind in (EventKind.DRAW_STOCK, EventKind.TAKE_DISCARD):
                    if pending_actor is not None or next_actor not in (None, actor):
                        raise ValueError("observation event chronology has an unexpected draw")
                    if turns[actor] == 0:
                        raise ValueError("observation event chronology uses an inactive player")
                    if event.offered_top != (discard[-1] if discard else None):
                        raise ValueError("observation event chronology has the wrong offered top")
                    if kind is EventKind.DRAW_STOCK:
                        if stock == 0:
                            raise ValueError("observation event chronology draws empty stock")
                        stock -= 1
                    else:
                        if not discard or visible_face != discard[-1]:
                            raise ValueError("observation event chronology takes the wrong discard")
                        discard = discard[:-1]
                    hand_sizes[actor] += 1
                    pending_actor = actor
                    if actor == 0:
                        known_hand[visible_face] += 1
                        hand = tuple(known_hand)
                elif kind is EventKind.DISCARD:
                    if pending_actor != actor:
                        raise ValueError("observation event chronology discards without a draw")
                    hand_sizes[actor] -= 1
                    turns[actor] -= 1
                    discard += (visible_face,)
                    pending_actor = None
                    next_actor = event.next_player
                    if actor == 0:
                        known_hand[visible_face] -= 1
                        hand = tuple(known_hand)
                elif kind is EventKind.REFILL:
                    if stock != 0 or len(discard) <= 1 or not config.recycle_discard:
                        raise ValueError("observation event chronology has an invalid refill")
                    if pending_actor is not None:
                        raise ValueError("observation event chronology refills after a draw")
                    if previous.kind is EventKind.DISCARD and actor != previous.actor:
                        raise ValueError("observation event chronology refills for the wrong player")
                    stock = len(discard) - 1
                    discard = discard[-1:]
                elif kind is EventKind.TERMINAL:
                    if previous.kind is not EventKind.DISCARD or actor != previous.actor:
                        raise ValueError("observation event chronology has an invalid terminal")
                if kind not in (EventKind.DRAW_STOCK, EventKind.TAKE_DISCARD):
                    if event.offered_top is not None:
                        raise ValueError("observation non-draw event has an offered top")

            if actor == 0 and (
                any(not 0 <= value <= config.num_decks for value in known_hand)
                or event.hand_size != sum(known_hand)
            ):
                raise ValueError("observation own card history is inconsistent")
            if event.hand_size != hand_sizes[actor] or event.turns_remaining != turns[actor]:
                raise ValueError("observation event chronology has inconsistent player state")
            if event.stock_remaining != stock:
                raise ValueError("observation event chronology has inconsistent stock")
            players = _player_views(hand_sizes, turns, 0)
            if event.next_player != players[actor].next_player:
                raise ValueError("observation event chronology has an inconsistent next player")
            events.append(event)
            states.append(InformationState(hand, players, discard, stock))
        return InformationHistory(config, player_count, tuple(events), tuple(states))


class InformationJournal:
    def __init__(self, players: int) -> None:
        self._histories: list[list[InformationEvent]] = [
            [] for _ in range(players)
        ]

    def start(
        self,
        hands: Sequence[Sequence[int]],
        turns: Sequence[int],
        stock_remaining: int,
        discard_top: Optional[int],
    ) -> None:
        self._histories = [
            [InformationEvent(
                EventKind.INITIAL,
                0,
                sum(hand),
                turns[seat],
                stock_remaining,
                player_views(hands, turns, seat)[0].next_player,
                offered_top=discard_top,
                initial_hand=tuple(hand),
            )]
            for seat, hand in enumerate(hands)
        ]

    def record(
        self,
        kind: EventKind,
        actor: int,
        hand_size: int,
        turns_remaining: int,
        stock_remaining: int,
        next_player: int,
        face: Optional[int] = None,
        offered_top: Optional[int] = None,
    ) -> None:
        count = len(self._histories)
        for observer, history in enumerate(self._histories):
            visible_face = face
            if kind is EventKind.DRAW_STOCK and actor != observer:
                visible_face = None
            history.append(InformationEvent(
                kind,
                (actor - observer) % count,
                hand_size,
                turns_remaining,
                stock_remaining,
                (next_player - observer) % count,
                visible_face,
                offered_top,
            ))

    def observe(self, observer: int) -> Tuple[InformationEvent, ...]:
        return tuple(self._histories[observer])
