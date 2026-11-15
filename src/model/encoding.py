from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import struct
from typing import Sequence

from src.evaluate import NUM_FACES
from src.game.environment import ENCODING_VERSION, NUM_ACTIONS, Observation, Phase, legal_action_mask
from src.game.information import EventKind
from src.game.lookahead import LEGAL_RUNS


NUM_SEAT_SLOTS = 6
NO_TOP = NUM_FACES
CARD_CHANNELS = (
    "own", "own_ge1", "own_ge2", "own_ge3", "pile", "top", "indicator", "wild",
    "points", "opp_discarded", "next_held", "other_held", "next_passed",
)
SEAT_CHANNELS = (
    "present", "observer", "active", "hand_size", "turns", "next", "takes",
    "passes", "held_total",
)
CONTEXT_CHANNELS = (
    "draw", "discard", "terminal", "to_act", "turns", "stock", "pile", "refills",
    "decks", "hand", "sequences", "recycle", "players", "max_turns", "warm_start",
    "wilds", "deadwood",
)
CARD_OFFSET = 0
SEAT_OFFSET = CARD_OFFSET + NUM_FACES * len(CARD_CHANNELS)
CONTEXT_OFFSET = SEAT_OFFSET + NUM_SEAT_SLOTS * len(SEAT_CHANNELS)
LEGAL_OFFSET = CONTEXT_OFFSET + len(CONTEXT_CHANNELS)
TOP_OFFSET = LEGAL_OFFSET + NUM_ACTIONS
FEATURE_SIZE = TOP_OFFSET + 1
_RUN_FACES = tuple(tuple(suit * 13 + rank for rank in run)
                   for suit in range(4) for run in LEGAL_RUNS)
_SET_FACES = tuple(tuple(suit * 13 + rank for suit in suits)
                   for rank in range(13) for suits in combinations(range(4), 3))


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


def _meld_membership(hand: Sequence[int], joker: int) -> tuple[list[int], list[int], list[int]]:
    natural, pure, any_meld = [[0] * NUM_FACES for _ in range(3)]
    wilds = tuple(suit * 13 + joker % 13 for suit in range(4))

    def mark(represented: tuple[int, ...], flexible: tuple[int, ...], result: list[int]) -> None:
        missing = sum(not hand[face] or face in flexible for face in represented)
        available = sum(hand[face] for face in flexible)
        if missing > available:
            return
        for face in represented:
            if hand[face] and face not in flexible:
                result[face] = 1
        if available:
            for face in flexible:
                if hand[face]:
                    result[face] = 1

    for represented in _RUN_FACES:
        if all(hand[face] for face in represented):
            for face in represented:
                natural[face] = 1
        mark(represented, (joker,), pure)
        mark(represented, wilds, any_meld)
    for represented in _SET_FACES:
        mark(represented, wilds, any_meld)
    for face, copies in enumerate(hand):
        if copies >= 3:
            pure[face] = any_meld[face] = 1
    return natural, pure, any_meld


def encode(obs: Observation) -> Features:
    cfg, hand, count = obs.config, obs.hand, len(obs.players)
    decks, supply = cfg.num_decks, cfg.num_decks * NUM_FACES
    pile = [0] * NUM_FACES
    for face in obs.discard_pile:
        pile[face] += 1
    seat_held = [[0] * NUM_FACES for _ in range(count)]
    seat_held_total, seat_takes, seat_passes = [0] * count, [0] * count, [0] * count
    opp_disc, next_passed = [0] * NUM_FACES, [0] * NUM_FACES
    next_opp = obs.players[0].next_player if count > 1 and obs.players[0].next_player > 0 else -1
    refills = 0
    for event in obs.history:
        actor, face = event.actor, event.face
        if event.kind is EventKind.TAKE_DISCARD:
            seat_held[actor][face] += 1
            seat_held_total[actor] += 1
            seat_takes[actor] += 1
        elif event.kind is EventKind.DISCARD:
            if seat_held[actor][face] > 0:
                seat_held[actor][face] -= 1
                seat_held_total[actor] -= 1
            if actor > 0:
                opp_disc[face] += 1
        elif event.kind is EventKind.DRAW_STOCK:
            if event.offered_top is not None:
                seat_passes[actor] += 1
                if actor == next_opp:
                    next_passed[event.offered_top] += 1
        elif event.kind is EventKind.REFILL:
            refills += 1
    top_face, wild_rank = obs.discard_top, obs.joker % 13
    raw_pts_sum, wild_cnt = 0, 0
    values = []
    for face, copies in enumerate(hand):
        rank = face % 13
        is_wild = rank == wild_rank
        pts = 0 if is_wild else (10 if rank == 0 or rank >= 10 else rank + 1)
        raw_pts_sum += copies * pts
        wild_cnt += copies if is_wild else 0
        opp_held_face = sum(seat_held[seat][face] for seat in range(1, count))
        next_held_face = seat_held[next_opp][face] if next_opp > 0 else 0
        values.extend((
            copies / decks, float(copies >= 1), float(copies >= 2), float(copies >= 3),
            pile[face] / decks, float(face == top_face), float(face == obs.joker),
            float(is_wild), pts / 10.0, opp_disc[face] / decks, next_held_face / decks,
            (opp_held_face - next_held_face) / decks, next_passed[face] / decks,
        ))
    for seat in range(NUM_SEAT_SLOTS):
        if seat >= count:
            values.extend((0.0,) * len(SEAT_CHANNELS))
            continue
        player = obs.players[seat]
        values.extend((
            1.0, float(seat == 0), float(player.active), player.hand_size / cfg.cards_in_hand,
            player.turns_remaining / cfg.max_turns,
            float(seat > 0 and seat == obs.players[0].next_player),
            seat_takes[seat] / cfg.max_turns, seat_passes[seat] / cfg.max_turns,
            seat_held_total[seat] / cfg.cards_in_hand,
        ))
    last = obs.history[-1]
    to_act = not obs.done and obs.players[0].active and (
        last.kind is EventKind.INITIAL
        or obs.phase is Phase.DISCARD and last.kind in (EventKind.DRAW_STOCK, EventKind.TAKE_DISCARD)
        and last.actor == 0
        or obs.phase is Phase.DRAW and last.kind in (EventKind.DISCARD, EventKind.REFILL)
        and last.next_player == 0
    )
    values.extend((
        float(obs.phase is Phase.DRAW), float(obs.phase is Phase.DISCARD), float(obs.done), float(to_act),
        obs.turns_remaining / cfg.max_turns, obs.stock_remaining / supply, len(obs.discard_pile) / supply,
        min(refills, 3) / 3.0, decks / 6.0, cfg.cards_in_hand / 30.0,
        cfg.required_sequences / 10.0, float(cfg.recycle_discard), count / 6.0, cfg.max_turns / 60.0,
        float(count == 1 and sum(obs.history[0].initial_hand) == cfg.cards_in_hand + 1),
        wild_cnt / (4 * decks), raw_pts_sum / 100.0,
    ))
    values.extend(float(allowed and to_act) for allowed in legal_action_mask(obs))
    values.append(float(NO_TOP if top_face is None else top_face))
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
