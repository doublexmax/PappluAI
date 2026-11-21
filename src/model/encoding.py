from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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
    "adjacent", "set_mates", "natural", "pure", "any_meld",
    "spare_pure", "spare_meld", "disc_pseq_delta", "disc_gated_saved", "disc_pairs_delta",
)
SEAT_CHANNELS = (
    "present", "observer", "active", "hand_size", "turns", "next", "takes",
    "passes", "held_total",
)
CONTEXT_CHANNELS = (
    "draw", "discard", "terminal", "to_act", "turns", "stock", "pile", "refills",
    "decks", "hand", "sequences", "recycle", "players", "max_turns",
    "jokers_held", "wilds_held", "raw_points",
    "unavoidable_dw", "disjoint_pseq_frac", "disjoint_gated_pts",
    "top_gain_pseq", "top_gain_gated", "top_gain_pairs",
)
CARD_OFFSET = 0
SEAT_OFFSET = CARD_OFFSET + NUM_FACES * len(CARD_CHANNELS)
CONTEXT_OFFSET = SEAT_OFFSET + NUM_SEAT_SLOTS * len(SEAT_CHANNELS)
LEGAL_OFFSET = CONTEXT_OFFSET + len(CONTEXT_CHANNELS)
TOP_OFFSET = LEGAL_OFFSET + NUM_ACTIONS
FEATURE_SIZE = TOP_OFFSET + 1
LEGAL_RUNS = tuple((r, r + 1, r + 2) for r in range(11)) + ((11, 12, 0),)
_RUN_FACES = tuple(tuple(suit * 13 + rank for rank in run)
                   for suit in range(4) for run in LEGAL_RUNS)
_SET_FACES = tuple(tuple(suit * 13 + rank for suit in suits)
                   for rank in range(13) for suits in combinations(range(4), 3))
_RANK_POINTS = (10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)
_ORDERED_RUNS = tuple(sorted(_RUN_FACES, key=lambda run: (
    run[0] // 13, -sum(_RANK_POINTS[f % 13] for f in run), run[0] % 13,
)))
_RUN_PAIRS = tuple(sorted(
    {tuple(sorted(pair)) for run in _RUN_FACES for pair in combinations(run, 2)},
    key=lambda pair: (-sum(_RANK_POINTS[f % 13] for f in pair), pair),
))
_PURE_PAIRS = tuple(tuple(pair for pair in _RUN_PAIRS if joker not in pair) for joker in range(NUM_FACES))
_WILD_PAIRS = tuple(tuple(sorted(
    {pair for pair in _RUN_PAIRS if all(f % 13 != rank for f in pair)}
    | {(a * 13 + r, b * 13 + r) for r in range(13) if r != rank
       for a, b in combinations(range(4), 2)},
    key=lambda pair: (-sum(_RANK_POINTS[f % 13] for f in pair), pair),
)) for rank in range(13))
_SET_RANKS = tuple(sorted(range(13), key=lambda rank: (-_RANK_POINTS[rank], rank)))
_POINT_ORDER = tuple(sorted(range(NUM_FACES), key=lambda face: (-_RANK_POINTS[face % 13], face)))
_RUN_STARTS = {run[0]: run for run in _RUN_FACES}


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


def disjoint_pack(
    hand: Sequence[int], joker: int, required_sequences: int,
) -> tuple[int, int, int, int, list[int], list[int]]:
    rem = list(hand)
    pure_seqs = 0
    for a, b, c in _ORDERED_RUNS:
        if joker in (a, b, c):
            continue
        while rem[a] and rem[b] and rem[c]:
            rem[a] -= 1
            rem[b] -= 1
            rem[c] -= 1
            pure_seqs += 1
            if c % 13 < 12 and c == b + 1:
                nxt = c + 1
                run = _RUN_STARTS.get(nxt)
                starts = run is not None and rem[run[0]] and rem[run[1]] and rem[run[2]]
                if nxt != joker and rem[nxt] and not starts:
                    rem[nxt] -= 1
    for face in range(NUM_FACES):
        if face != joker:
            while rem[face] >= 3:
                rem[face] -= 3
                pure_seqs += 1
    for a, b, c in _ORDERED_RUNS:
        if joker not in (a, b, c):
            continue
        while rem[a] and rem[b] and rem[c]:
            rem[a] -= 1
            rem[b] -= 1
            rem[c] -= 1
            pure_seqs += 1
    while rem[joker]:
        pair = next((pair for pair in _PURE_PAIRS[joker] if rem[pair[0]] and rem[pair[1]]), None)
        if pair is not None:
            rem[pair[0]] -= 1
            rem[pair[1]] -= 1
            rem[joker] -= 1
            pure_seqs += 1
        elif rem[joker] >= 2:
            face = next((face for face in _POINT_ORDER if face != joker and rem[face]), None)
            if face is None:
                break
            rem[face] -= 1
            rem[joker] -= 2
            pure_seqs += 1
        else:
            break
    after_pure = list(rem)
    pairs = sum(
        ((a != joker and rem[a] > 0) + (b != joker and rem[b] > 0) + (c != joker and rem[c] > 0)) >= 2
        for a, b, c in _RUN_FACES
    )
    rank, wilds = joker % 13, 0
    for suit in range(4):
        face = suit * 13 + rank
        wilds += rem[face]
        rem[face] = 0
    for r in _SET_RANKS:
        if r == rank:
            continue
        while True:
            faces = [suit * 13 + r for suit in range(4) if rem[suit * 13 + r]]
            if len(faces) < 3:
                break
            for face in faces:
                rem[face] -= 1
    while wilds:
        pair = next((pair for pair in _WILD_PAIRS[rank] if rem[pair[0]] and rem[pair[1]]), None)
        if pair is None:
            break
        rem[pair[0]] -= 1
        rem[pair[1]] -= 1
        wilds -= 1
    points = sum(n * _RANK_POINTS[f % 13] for f, n in enumerate(rem) if n)
    gated = points if pure_seqs >= required_sequences else sum(
        n * _RANK_POINTS[f % 13] for f, n in enumerate(after_pure) if n and f % 13 != rank
    )
    return pure_seqs, gated, points, pairs, rem, after_pure


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
    required = cfg.required_sequences
    base = disjoint_pack(hand, obs.joker, required)
    natural, pure, any_meld = _meld_membership(hand, obs.joker)
    values, raw_points, wild_count, unavoidable = [], 0, 0, 0
    for face, copies in enumerate(hand):
        suit, rank = divmod(face, 13)
        wild = rank == obs.joker % 13
        points = 0 if wild else 10 if rank == 0 or rank >= 10 else rank + 1
        raw_points += copies * points
        wild_count += copies if wild else 0
        unavoidable += copies * points * (1 - any_meld[face])
        opp_held = sum(held[seat][face] for seat in range(1, count))
        next_held = held[next_opp][face] if next_opp > 0 else 0
        unseen = max(0, decks - copies - pile[face] - int(face == obs.joker) - sum(h[face] for h in held))
        values.extend((
            copies / decks, float(copies == 1), float(copies == 2), float(copies >= 3),
            pile[face] / decks, float(face == obs.discard_top), float(face == obs.joker),
            float(wild), points / 10, unseen / decks, discarded[face] / decks,
            next_held / decks, (opp_held - next_held) / decks, next_passed[face] / decks,
            (int(hand[suit * 13 + (rank - 1) % 13] > 0)
             + int(hand[suit * 13 + (rank + 1) % 13] > 0)) / 2,
            sum(hand[other * 13 + rank] > 0 for other in range(4) if other != suit) / 3,
            natural[face], pure[face], any_meld[face],
            base[5][face] / copies if copies else 0.0,
            base[4][face] / copies if copies else 0.0, 0.0, 0.0, 0.0,
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
        unavoidable / (10 * cfg.cards_in_hand), min(base[0], required) / max(1, required),
        base[1] / 100, 0.0, 0.0, 0.0,
    ))
    legal = list(legal_action_mask(obs))
    if (
        obs.phase is Phase.DISCARD and last.kind is EventKind.TAKE_DISCARD
        and last.actor == 0 and sum(c > 0 for c in hand) > 1
    ):
        legal[2 + last.face] = False
    if to_act and obs.phase is Phase.DISCARD:
        for face, copies in enumerate(hand):
            if not copies or not legal[2 + face]:
                continue
            after = list(hand)
            after[face] -= 1
            packed = disjoint_pack(after, obs.joker, required)
            offset = face * len(CARD_CHANNELS) + CARD_CHANNELS.index("disc_pseq_delta")
            values[offset:offset + 3] = (
                min(packed[0], required) - min(base[0], required),
                (base[1] - packed[1]) / 10, (packed[3] - base[3]) / 3,
            )
    if to_act and obs.phase is Phase.DRAW and legal[1]:
        plus = list(hand)
        plus[obs.discard_top] += 1
        best, best_key = None, None
        for face, copies in enumerate(plus):
            if not copies or face == obs.discard_top:
                continue
            plus[face] -= 1
            packed = disjoint_pack(plus, obs.joker, required)
            plus[face] += 1
            key = (min(packed[0], required), -packed[1], packed[3], -packed[2])
            if best_key is None or key > best_key:
                best, best_key = packed, key
        if best is not None:
            offset = CONTEXT_OFFSET + CONTEXT_CHANNELS.index("top_gain_pseq")
            values[offset:offset + 3] = (
                min(best[0], required) - min(base[0], required),
                (base[1] - best[1]) / 20, (best[3] - base[3]) / 3,
            )
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
