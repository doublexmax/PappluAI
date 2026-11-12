from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import struct
from typing import Sequence

from src.evaluate import NUM_FACES, card_points, pure_sequence_upper_bound
from src.game.belief import BeliefState
from src.game.environment import NUM_ACTIONS, Observation, Phase, legal_action_mask
from src.game.information import EventKind
from src.game.lookahead import LEGAL_RUNS


NUM_SEAT_SLOTS = 6
NO_TOP = NUM_FACES
CARD_CHANNELS = (
    "own", "own_ge1", "own_ge2", "own_ge3", "initial", "pile", "unseen", "top",
    "stock", "indicator", "wild", "points", "adjacent", "set_mates",
    "natural_run_member", "pure_meld_member", "any_meld_member",
)
HISTORY_CHANNELS = (
    "took", "held", "discarded", "passed", "last_take", "last_discard", "last_pass",
)
SEAT_CHANNELS = (
    "present", "observer", "active", "hand_size", "turns", "next", "previous",
    "take_rate", "takes", "passes", "held_total",
)
CONTEXT_CHANNELS = (
    "draw", "discard", "terminal", "to_act", "turns", "stock", "pile", "refills",
    "decks", "hand", "sequences", "recycle", "players", "max_turns", "warm_start",
    "wilds", "deadwood", "pure_quota_deficit", "unavoidable_deadwood",
)
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
    ("history", (NUM_FACES, NUM_SEAT_SLOTS, len(HISTORY_CHANNELS))),
    ("seat", (NUM_SEAT_SLOTS, len(SEAT_CHANNELS))),
    ("context", (len(CONTEXT_CHANNELS),)),
    ("legal", (NUM_ACTIONS,)),
    ("top", (1,)),
))
FEATURE_SIZE = LAYOUT[-1].offset + LAYOUT[-1].size
_RECORD = struct.Struct("<%df" % FEATURE_SIZE)
_LEGAL_OFFSET = next(field.offset for field in LAYOUT if field.name == "legal") * 4


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
    belief = BeliefState.from_observation(obs)
    ledger = belief.ledger
    decks, supply = cfg.num_decks, cfg.num_decks * NUM_FACES
    stock_total, history_length = sum(belief.stock_weights), len(obs.history)
    pile = [0] * NUM_FACES
    for face in obs.discard_pile:
        pile[face] += 1
    natural, pure, any_meld = _meld_membership(hand, obs.joker)
    points = [card_points(face, obs.joker) for face in range(NUM_FACES)]
    values = []
    for face, copies in enumerate(hand):
        suit, rank = divmod(face, 13)
        neighbors = (1, 12) if rank == 0 else (11, 0) if rank == 12 else (rank - 1, rank + 1)
        values.extend((
            copies / decks, float(copies >= 1), float(copies >= 2), float(copies >= 3),
            obs.history[0].initial_hand[face] / decks, pile[face] / decks, belief.unseen[face] / decks,
            float(face == obs.discard_top),
            belief.stock_weights[face] / stock_total if stock_total else 0.0,
            float(face == obs.joker), float(rank == obs.joker % 13), points[face] / 10.0,
            sum(hand[suit * 13 + neighbor] > 0 for neighbor in neighbors) / 2.0,
            sum(hand[other * 13 + rank] > 0 for other in range(4) if other != suit) / 3.0,
            natural[face], pure[face], any_meld[face],
        ))
    for face in range(NUM_FACES):
        for seat in range(NUM_SEAT_SLOTS):
            if seat >= count:
                values.extend((0.0,) * len(HISTORY_CHANNELS))
                continue
            row = ledger.seats[seat]
            values.extend((
                row.took[face] / decks, row.held[face] / decks,
                row.discarded[face] / decks, row.passed[face] / decks,
                row.last_take[face] / history_length, row.last_discard[face] / history_length,
                row.last_pass[face] / history_length,
            ))
    for seat in range(NUM_SEAT_SLOTS):
        if seat >= count:
            values.extend((0.0,) * len(SEAT_CHANNELS))
            continue
        player, row = obs.players[seat], ledger.seats[seat]
        values.extend((
            1.0, float(seat == 0), float(player.active), player.hand_size / cfg.cards_in_hand,
            player.turns_remaining / cfg.max_turns,
            float(seat > 0 and seat == obs.players[0].next_player),
            float(seat > 0 and seat == obs.players[0].previous_player),
            belief.take_rate(seat), row.takes / cfg.max_turns, row.passes / cfg.max_turns,
            sum(row.held) / cfg.cards_in_hand,
        ))
    last = obs.history[-1]
    to_act = not obs.done and obs.players[0].active and (
        last.kind is EventKind.INITIAL
        or obs.phase is Phase.DISCARD and last.kind in (EventKind.DRAW_STOCK, EventKind.TAKE_DISCARD)
        and last.actor == 0
        or obs.phase is Phase.DRAW and last.kind in (EventKind.DISCARD, EventKind.REFILL)
        and last.next_player == 0
    )
    pure_upper = pure_sequence_upper_bound(hand, obs.joker)
    values.extend((
        float(obs.phase is Phase.DRAW), float(obs.phase is Phase.DISCARD), float(obs.done), float(to_act),
        obs.turns_remaining / cfg.max_turns, obs.stock_remaining / supply, len(obs.discard_pile) / supply,
        min(ledger.refills, 3) / 3.0, decks / 6.0, cfg.cards_in_hand / 30.0,
        cfg.required_sequences / 10.0, float(cfg.recycle_discard), count / 6.0, cfg.max_turns / 60.0,
        float(count == 1 and sum(obs.history[0].initial_hand) == cfg.cards_in_hand + 1),
        sum(hand[suit * 13 + obs.joker % 13] for suit in range(4)) / (4 * decks),
        sum(copies * points[face] for face, copies in enumerate(hand)) / 100.0,
        max(0, cfg.required_sequences - pure_upper) / max(1, cfg.required_sequences),
        sum(copies * points[face] for face, copies in enumerate(hand) if not any_meld[face])
        / (10 * cfg.cards_in_hand),
    ))
    values.extend(float(allowed and to_act) for allowed in legal_action_mask(obs))
    values.append(float(NO_TOP if obs.discard_top is None else obs.discard_top))
    return Features(_RECORD.pack(*values))


def legal_actions(features: Features) -> tuple[int, ...]:
    return tuple(action for action, allowed in enumerate(
        struct.unpack_from("<%df" % NUM_ACTIONS, features.data, _LEGAL_OFFSET),
    ) if allowed)


def feature_values(features: Features) -> tuple[float, ...]:
    return _RECORD.unpack(features.data)
