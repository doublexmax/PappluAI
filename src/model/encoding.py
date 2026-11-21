from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence

from src.game.environment import ENCODING_VERSION, NUM_ACTIONS, NUM_FACES, Observation, Phase, legal_action_mask
from src.game.information import EventKind


NUM_SEAT_SLOTS = 6
NO_TOP = NUM_FACES
CARD_CHANNELS = (
    "own", "own_eq1", "own_eq2", "own_ge3", "pile", "top", "joker", "wild",
    "points", "unseen", "opp_discarded", "next_held", "other_held", "next_passed",
)
SEAT_CHANNELS = (
    "present", "observer", "active", "hand_size", "turns", "next", "takes",
    "passes", "held_total",
)
CONTEXT_CHANNELS = (
    "draw", "discard", "terminal", "to_act", "turns", "stock", "pile", "refills",
    "decks", "hand", "sequences", "recycle", "players", "max_turns",
    "jokers_held", "wilds_held", "raw_points",
)
CARD_OFFSET = 0
SEAT_OFFSET = CARD_OFFSET + NUM_FACES * len(CARD_CHANNELS)
CONTEXT_OFFSET = SEAT_OFFSET + NUM_SEAT_SLOTS * len(SEAT_CHANNELS)
LEGAL_OFFSET = CONTEXT_OFFSET + len(CONTEXT_CHANNELS)
TOP_OFFSET = LEGAL_OFFSET + NUM_ACTIONS
FEATURE_SIZE = TOP_OFFSET + 1


@dataclass(frozen=True)
class FeatureField:
    name: str
    shape: tuple[int, ...]
    offset: int

    @property
    def size(self) -> int:
        return math.prod(self.shape)


def _layout(fields: Sequence[tuple[str, tuple[int, ...]]]) -> tuple[FeatureField, ...]:
    offset, result = 0, []
    for name, shape in fields:
        result.append(FeatureField(name, shape, offset))
        offset += math.prod(shape)
    return tuple(result)


LAYOUT = _layout((
    ("card", (NUM_FACES, len(CARD_CHANNELS))),
    ("seat", (NUM_SEAT_SLOTS, len(SEAT_CHANNELS))),
    ("context", (len(CONTEXT_CHANNELS),)),
    ("legal", (NUM_ACTIONS,)),
    ("top", (1,)),
))
_RECORD = struct.Struct("<%df" % FEATURE_SIZE)


@dataclass(frozen=True)
class Features:
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or len(self.data) != _RECORD.size:
            raise ValueError("features must contain %d bytes of little-endian float32 data" % _RECORD.size)


def encode(obs: Observation) -> Features:
    cfg, hand, players = obs.config, obs.hand, obs.players
    count, decks = len(players), cfg.num_decks
    held = [[0] * NUM_FACES for _ in players]
    takes, passes = [0] * count, [0] * count
    discarded, next_passed, pile = [0] * NUM_FACES, [0] * NUM_FACES, [0] * NUM_FACES
    next_opp = players[0].next_player if count > 1 else -1
    refills = 0
    for face in obs.discard_pile:
        pile[face] += 1
    for event in obs.history:
        actor, face = event.actor, event.face
        if event.kind is EventKind.TAKE_DISCARD:
            held[actor][face] += 1
            takes[actor] += 1
        elif event.kind is EventKind.DISCARD:
            held[actor][face] = max(0, held[actor][face] - 1)
            if actor > 0:
                discarded[face] += 1
        elif event.kind is EventKind.DRAW_STOCK and event.offered_top is not None:
            passes[actor] += 1
            if actor == next_opp:
                next_passed[event.offered_top] += 1
        elif event.kind is EventKind.REFILL:
            refills += 1
    values, raw_points, wild_count = [], 0, 0
    for face, copies in enumerate(hand):
        rank = face % 13
        wild = rank == obs.joker % 13
        points = 0 if wild else 10 if rank == 0 or rank >= 10 else rank + 1
        raw_points += copies * points
        wild_count += copies if wild else 0
        opp_held = sum(held[seat][face] for seat in range(1, count))
        next_held = held[next_opp][face] if next_opp > 0 else 0
        unseen = max(0, decks - copies - pile[face] - int(face == obs.joker) - sum(h[face] for h in held))
        values.extend((
            copies / decks, float(copies == 1), float(copies == 2), float(copies >= 3),
            pile[face] / decks, float(face == obs.discard_top), float(face == obs.joker),
            float(wild), points / 10, unseen / decks, discarded[face] / decks,
            next_held / decks, (opp_held - next_held) / decks, next_passed[face] / decks,
        ))
    for seat in range(NUM_SEAT_SLOTS):
        if seat >= count:
            values.extend((0.0,) * len(SEAT_CHANNELS))
            continue
        player = players[seat]
        values.extend((
            1.0, float(seat == 0), float(player.active), player.hand_size / cfg.cards_in_hand,
            player.turns_remaining / cfg.max_turns, float(seat > 0 and seat == next_opp),
            takes[seat] / cfg.max_turns, passes[seat] / cfg.max_turns,
            sum(held[seat]) / cfg.cards_in_hand,
        ))
    last = obs.history[-1]
    to_act = not obs.done and players[0].active and (
        last.kind is EventKind.INITIAL
        or obs.phase is Phase.DISCARD and last.actor == 0
        and last.kind in (EventKind.DRAW_STOCK, EventKind.TAKE_DISCARD)
        or obs.phase is Phase.DRAW and last.kind in (EventKind.DISCARD, EventKind.REFILL)
        and last.next_player == 0
    )
    supply = decks * NUM_FACES
    values.extend((
        float(obs.phase is Phase.DRAW), float(obs.phase is Phase.DISCARD), float(obs.done), float(to_act),
        obs.turns_remaining / cfg.max_turns, obs.stock_remaining / supply, len(obs.discard_pile) / supply,
        min(refills, 3) / 3, decks / 6, cfg.cards_in_hand / 30, cfg.required_sequences / 10,
        float(cfg.recycle_discard), count / 6, cfg.max_turns / 60, hand[obs.joker] / decks,
        wild_count / (4 * decks), raw_points / 100,
    ))
    legal = list(legal_action_mask(obs))
    if (
        obs.phase is Phase.DISCARD and last.kind is EventKind.TAKE_DISCARD
        and last.actor == 0 and sum(c > 0 for c in hand) > 1
    ):
        legal[2 + last.face] = False
    values.extend(float(allowed and to_act) for allowed in legal)
    values.append(float(NO_TOP if obs.discard_top is None else obs.discard_top))
    return Features(_RECORD.pack(*values))


def legal_actions(features: Features) -> tuple[int, ...]:
    return tuple(action for action, allowed in enumerate(
        struct.unpack_from("<%df" % NUM_ACTIONS, features.data, LEGAL_OFFSET * 4),
    ) if allowed)


def feature_values(features: Features) -> tuple[float, ...]:
    return _RECORD.unpack(features.data)


def unpack(features: Features) -> dict[str, tuple[float, ...]]:
    values = feature_values(features)
    return {field.name: values[field.offset:field.offset + field.size] for field in LAYOUT}


def encoding_layout() -> tuple[FeatureField, ...]:
    return LAYOUT


def encoding_metadata() -> dict:
    return {
        "encoding_version": ENCODING_VERSION,
        "feature_size": FEATURE_SIZE,
        "feature_bytes": _RECORD.size,
        "layout": [
            {"name": field.name, "shape": list(field.shape), "offset": field.offset}
            for field in LAYOUT
        ],
        "channels": {"card": CARD_CHANNELS, "seat": SEAT_CHANNELS, "context": CONTEXT_CHANNELS},
    }
