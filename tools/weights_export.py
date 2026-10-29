"""Export dynamic_table Q-network weights for the browser engine in web/qnet.mjs.

    python tools/weights_export.py                         seeded untrained weights, no torch
    python tools/weights_export.py --checkpoint model.pt   needs torch
    python tools/weights_export.py --registry DIR          exports the champion, needs torch

Writes web/papplu-model.json and tests/fixtures/qnet-golden.json. Golden observations come
from real src.game tables. Bootstrap Q values come from the pure-Python port of
src/model/encoding.py and src/model/network.py below. A torch export stores torch's Q values
and fails unless that port agrees with torch on every golden observation.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import operator
from pathlib import Path
import random
import struct
import sys
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, ENCODING_VERSION, NUM_ACTIONS, NUM_FACES,
    GameConfig, PappluEnv, Phase, legal_action_mask,
)
from src.game.information import EventKind
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.model.spec import ARCHITECTURE, CHECKPOINT_VERSION, ModelSpec

FORMAT = "papplu-qnet"
GOLDEN_FORMAT = "papplu-qnet-golden"
FORMAT_VERSION = 1
BOOTSTRAP_SEED = 1729
BOOTSTRAP_SENSITIVITY_GAINS = {"card_attention.in_proj_weight": 8.0, "event_summary.weight": 16.0}
PRODUCTION = GameConfig(max_turns=60)
SPEC = ModelSpec()
NO_FACE = NUM_FACES
TOLERANCE = 1e-4
INPUT_TOLERANCE = 1e-6
STEP_LIMIT = 1000
SOURCES = ("src/model/encoding.py", "src/model/network.py", "tools/weights_export.py")
TENSOR_SHAPES = (
    ("suit_conv1.weight", (64, 5, 3)), ("suit_conv1.bias", (64,)),
    ("suit_conv2.weight", (64, 64, 3)), ("suit_conv2.bias", (64,)),
    ("card_attention.in_proj_weight", (192, 64)), ("card_attention.in_proj_bias", (192,)),
    ("card_attention.out_proj.weight", (64, 64)), ("card_attention.out_proj.bias", (64,)),
    ("card_norm.weight", (64,)), ("card_norm.bias", (64,)),
    ("history_proj.weight", (1, 5)), ("history_proj.bias", (1,)),
    ("context_features.weight", (64, 11)), ("context_features.bias", (64,)),
    ("player_features.weight", (64, 6)), ("player_features.bias", (64,)),
    ("event_summary.weight", (64, 8)), ("event_summary.bias", (64,)),
    ("turn_relations.weight", (64, 192)), ("turn_relations.bias", (64,)),
    ("context_gate.weight", (2, 64)), ("context_gate.bias", (2,)),
    ("draw_head.0.weight", (64, 197)), ("draw_head.0.bias", (64,)),
    ("draw_head.2.weight", (2, 64)), ("draw_head.2.bias", (2,)),
    ("score.0.weight", (64, 210)), ("score.0.bias", (64,)),
    ("score.2.weight", (1, 64)), ("score.2.bias", (1,)),
)


def packed(values):
    return struct.pack("<%df" % len(values), *values)


def float32(values):
    return list(struct.unpack("<%df" % len(values), packed(values)))


def bootstrap_tensors(seed):
    shapes = dict(TENSOR_SHAPES)
    rng = random.Random(seed)
    tensors = {}
    for name, shape in TENSOR_SHAPES:
        bound = BOOTSTRAP_SENSITIVITY_GAINS.get(name, 1.0) * bootstrap_weight_bound(name, shapes)
        center = 1.0 if name == "card_norm.weight" else 0.0
        tensors[name] = float32([center + rng.uniform(-bound, bound) for _ in range(math.prod(shape))])
    return tensors


def bootstrap_weight_bound(name, shapes):
    if name == "card_attention.in_proj_weight":
        return math.sqrt(6.0 / sum(shapes[name]))
    if name.startswith("card_norm.") or name.endswith(("in_proj_bias", "out_proj.bias")):
        return 0.1
    weight = shapes[name.rsplit(".", 1)[0] + ".weight"]
    return 1.0 / math.sqrt(math.prod(weight[1:]))


def model_source(kind, tensors, sha256=None, seed=None, registry_id=None):
    digest = hashlib.sha256(b"".join(packed(tensors[name]) for name, _ in TENSOR_SHAPES)).hexdigest()
    return {
        "kind": kind,
        "trained": kind != "bootstrap",
        "champion": kind == "registry",
        "seed": seed,
        "registryId": registry_id,
        "sha256": digest if sha256 is None else sha256,
        "tensorsSha256": digest,
    }


def game_config_json(config):
    return {
        "numDecks": config.num_decks,
        "cardsInHand": config.cards_in_hand,
        "requiredSequences": config.required_sequences,
        "maxTurns": config.max_turns,
        "recycleDiscard": config.recycle_discard,
    }


def model_document(tensors, config, source):
    return {
        "format": FORMAT,
        "formatVersion": FORMAT_VERSION,
        "checkpointVersion": CHECKPOINT_VERSION,
        "encodingVersion": ENCODING_VERSION,
        "architecture": ARCHITECTURE,
        "modelSpec": SPEC.to_dict(),
        "numActions": NUM_ACTIONS,
        "gameConfig": game_config_json(config),
        "source": source,
        "tensors": {
            name: {"shape": list(shape), "data": base64.b64encode(packed(tensors[name])).decode("ascii")}
            for name, shape in TENSOR_SHAPES
        },
    }


def observation_json(obs):
    return {
        "hand": list(obs.hand),
        "discardPile": list(obs.discard_pile),
        "joker": obs.joker,
        "phase": obs.phase.value,
        "turnsRemaining": obs.turns_remaining,
        "stockRemaining": obs.stock_remaining,
        "config": game_config_json(obs.config),
        "players": [
            {
                "handSize": view.hand_size, "turnsRemaining": view.turns_remaining,
                "nextPlayer": view.next_player, "previousPlayer": view.previous_player,
                "active": view.active,
            }
            for view in obs.players
        ],
        "history": [
            {
                "kind": int(event.kind), "actor": event.actor, "handSize": event.hand_size,
                "turnsRemaining": event.turns_remaining, "stockRemaining": event.stock_remaining,
                "nextPlayer": event.next_player, "face": event.face,
                "offeredTop": event.offered_top, "initialHand": list(event.initial_hand),
            }
            for event in obs.history
        ],
    }


def encode(obs):
    config = obs["config"]
    decks = config["numDecks"]
    supply = decks * NUM_FACES
    history = obs["history"]
    count = len(obs["players"])
    if not history or not count:
        raise ValueError("model input needs complete history and player views")
    sins = [math.sin(2.0 * math.pi * i / count) for i in range(count)]
    coss = [math.cos(2.0 * math.pi * i / count) for i in range(count)]
    cards = [[n / decks, 0.0, 0.0, 0.0, 0.0] for n in obs["hand"]]
    for face in obs["discardPile"]:
        cards[face][1] += 1.0 / decks
    top = obs["discardPile"][-1] if obs["discardPile"] else NO_FACE
    if top != NO_FACE:
        cards[top][2] = 1.0
    cards[obs["joker"]][3] = 1.0
    for suit in range(4):
        cards[suit * 13 + obs["joker"] % 13][4] = 1.0
    players = []
    for relative, player in enumerate(obs["players"]):
        players.extend([
            float(relative == 0), player["handSize"] / config["cardsInHand"],
            player["turnsRemaining"] / config["maxTurns"], float(player["active"]),
            sins[relative], coss[relative],
        ])
    events = []
    for event in history:
        events.extend([
            float(event["actor"] == 0), sins[event["actor"]], coss[event["actor"]],
            event["handSize"] / config["cardsInHand"], event["turnsRemaining"] / config["maxTurns"],
            event["stockRemaining"] / supply, sins[event["nextPlayer"]], coss[event["nextPlayer"]],
        ])
    return {
        "cards": [value for row in cards for value in row],
        "initialHand": [n / decks for n in history[0]["initialHand"]],
        "top": top,
        "context": [float(obs["phase"] == phase) for phase in range(3)] + [
            obs["turnsRemaining"] / config["maxTurns"], obs["stockRemaining"] / supply,
            decks / 6.0, config["cardsInHand"] / 30.0, config["requiredSequences"] / 10.0,
            float(config["recycleDiscard"]), count / 6.0, config["maxTurns"] / 60.0,
        ],
        "players": players,
        "nextPlayer": [player["nextPlayer"] for player in obs["players"]],
        "previousPlayer": [player["previousPlayer"] for player in obs["players"]],
        "eventKind": [event["kind"] for event in history],
        "eventFace": [NO_FACE if event["face"] is None else event["face"] for event in history],
        "offeredTop": [
            NO_FACE if event["offeredTop"] is None else event["offeredTop"] for event in history
        ],
        "eventActor": [event["actor"] for event in history],
        "events": events,
    }


def history_channels(obs):
    decks = obs["config"]["numDecks"]
    initial = obs["history"][0]["initialHand"]
    channels = [
        [initial[face] / decks - obs["hand"][face] / decks, 0.0, 0.0, 0.0, 0.0]
        for face in range(NUM_FACES)
    ]
    channels.append([0.0] * 5)
    length = len(obs["history"])
    for position, event in enumerate(obs["history"], start=1):
        recency = (0.5 + 0.5 * (position / length)) / length
        face = NO_FACE if event["face"] is None else event["face"]
        offered = NO_FACE if event["offeredTop"] is None else event["offeredTop"]
        if event["actor"] == 0:
            channels[face][0] += recency
            continue
        base = 1 if event["actor"] == 1 else 3
        if event["kind"] == EventKind.TAKE_DISCARD:
            channels[face][base] += recency
        elif event["kind"] == EventKind.DISCARD:
            channels[face][base + 1] += recency
        elif event["kind"] == EventKind.DRAW_STOCK:
            channels[offered][base + 1] += recency
    return [value for row in channels[:NUM_FACES] for value in row]


def rows(values, width):
    return [values[start:start + width] for start in range(0, len(values), width)]


def dot(left, right):
    return math.fsum(map(operator.mul, left, right))


def linear(layer, x):
    weight, bias = layer
    return [b + dot(row, x) for row, b in zip(weight, bias)]


def relu(values):
    return [value if value > 0.0 else 0.0 for value in values]


def conv_relu(layer, steps):
    zero = [0.0] * len(steps[0])
    padded = [zero] + steps + [zero]
    channels = range(len(zero))
    return [
        relu(linear(layer, [padded[t + k][c] for c in channels for k in range(3)]))
        for t in range(len(steps))
    ]


def layer_norm(x, gamma, beta):
    mean = math.fsum(x) / len(x)
    variance = math.fsum((value - mean) ** 2 for value in x) / len(x)
    scale = 1.0 / math.sqrt(variance + 1e-5)
    return [(value - mean) * scale * g + b for value, g, b in zip(x, gamma, beta)]


def attention(layers, tokens):
    width = SPEC.width
    size = width // SPEC.heads
    projected = [linear(layers["card_attention.in_proj"], token) for token in tokens]
    mixed = [[0.0] * width for _ in tokens]
    for start in range(0, width, size):
        keys = [p[width + start:width + start + size] for p in projected]
        values = [p[2 * width + start:2 * width + start + size] for p in projected]
        columns = [list(column) for column in zip(*values)]
        for row, p in zip(mixed, projected):
            query = p[start:start + size]
            scores = [dot(query, key) / math.sqrt(size) for key in keys]
            peak = max(scores)
            weights = [math.exp(score - peak) for score in scores]
            total = math.fsum(weights)
            for offset, column in enumerate(columns):
                row[start + offset] = dot(weights, column) / total
    return [linear(layers["card_attention.out_proj"], row) for row in mixed]


def reference_layers(tensors):
    shapes = dict(TENSOR_SHAPES)

    def layer(weight, bias):
        return rows(tensors[weight], math.prod(shapes[weight][1:])), tensors[bias]

    names = (
        "suit_conv1", "suit_conv2", "card_attention.out_proj", "history_proj", "context_features",
        "player_features", "event_summary", "turn_relations", "context_gate",
        "draw_head.0", "draw_head.2", "score.0", "score.2",
    )
    layers = {name: layer(name + ".weight", name + ".bias") for name in names}
    layers["card_attention.in_proj"] = layer(
        "card_attention.in_proj_weight", "card_attention.in_proj_bias",
    )
    layers["card_norm"] = (tensors["card_norm.weight"], tensors["card_norm.bias"])
    return layers


def reference_q(layers, encoded, channels):
    cards = rows(encoded["cards"], 5)
    tokens = [None] * NUM_FACES
    for suit in range(4):
        steps = [cards[suit * 13 + t % 13] for t in range(14)]
        local = conv_relu(layers["suit_conv2"], conv_relu(layers["suit_conv1"], steps))
        tokens[suit * 13] = [0.5 * (a + b) for a, b in zip(local[0], local[13])]
        tokens[suit * 13 + 1:suit * 13 + 13] = local[1:13]
    mixed = attention(layers, tokens)
    local = [
        layer_norm([a + b for a, b in zip(token, extra)], *layers["card_norm"])
        for token, extra in zip(tokens, mixed)
    ]
    sets = [
        [math.fsum(column) / 4.0 for column in zip(*local[rank::13])]
        for rank in range(13)
    ]
    global_cards = [math.fsum(column) / NUM_FACES for column in zip(*local)]
    bonus = [0.02 * math.tanh(linear(layers["history_proj"], row)[0]) for row in rows(channels, 5)]

    players = rows(encoded["players"], 6)
    own, after, before = (
        linear(layers["player_features"], players[index])
        for index in (0, encoded["nextPlayer"][0], encoded["previousPlayer"][0])
    )
    relation = relu(linear(layers["turn_relations"], own + after + before))
    events = rows(encoded["events"], 8)
    weights = [position / len(events) for position in range(1, len(events) + 1)]
    total = max(math.fsum(weights), 1.0)
    summary = linear(layers["event_summary"], [
        math.fsum(w * event[k] for w, event in zip(weights, events)) / total for k in range(8)
    ])
    table = [
        a + b + 0.1 * c
        for a, b, c in zip(linear(layers["context_features"], encoded["context"]), relation, summary)
    ]
    gate = [0.02 * math.tanh(value) for value in linear(layers["context_gate"], table)]

    top = encoded["top"]
    if top < NUM_FACES:
        top_features = local[top] + sets[top % 13] + cards[top]
    else:
        top_features = [0.0] * (2 * SPEC.width + 5)
    raw = linear(layers["draw_head.2"], relu(linear(layers["draw_head.0"], global_cards + top_features)))
    stock = raw[0] + gate[0]
    q = [stock, stock + 0.25 * math.tanh(raw[1] + gate[1])]
    for face in range(NUM_FACES):
        rank = face % 13
        features = local[face] + sets[rank] + global_cards + cards[face] + [
            float(rank == other) for other in range(13)
        ]
        score = linear(layers["score.2"], relu(linear(layers["score.0"], features)))[0]
        q.append(score + bonus[face] + gate[0])
    return q


class GoldenCase(NamedTuple):
    name: str
    seats: int
    config: GameConfig
    seed: int
    until: Callable
    stock_bias: float = 0.5
    warm_start: bool = False


def dealt(obs):
    return True


def refilled(obs):
    return any(event.kind == EventKind.REFILL for event in obs.history)


def refilled_empty_pile(obs):
    return refilled(obs) and not obs.discard_pile


def stock_exhausted(obs):
    return obs.phase is Phase.DRAW and obs.stock_remaining == 0


def seat_inactive(obs):
    return not all(player.active for player in obs.players)


def terminal(obs):
    return obs.phase is Phase.TERMINAL


def reached(events, phase):
    def predicate(obs):
        return len(obs.history) >= events and obs.phase is phase
    return predicate


GOLDEN_CASES = (
    GoldenCase("solo-deal", 1, PRODUCTION, 101, dealt),
    GoldenCase("solo-mid-discard", 1, PRODUCTION, 102, reached(10, Phase.DISCARD)),
    GoldenCase("solo-warm-start", 1, PRODUCTION, 103, dealt, warm_start=True),
    GoldenCase("solo-refill", 1, GameConfig(1, 9, 3, 60), 104, refilled, stock_bias=0.9),
    GoldenCase("duo-deal", 2, PRODUCTION, 201, dealt),
    GoldenCase("duo-mid", 2, PRODUCTION, 202, reached(24, Phase.DISCARD)),
    GoldenCase("duo-refill-empty-pile", 2, GameConfig(1, 7, 2, 60), 203, refilled_empty_pile, 0.9),
    GoldenCase("trio-discard", 3, GameConfig(2, 13, 4, 30), 301, reached(40, Phase.DISCARD)),
    GoldenCase("trio-no-recycle", 3, GameConfig(1, 10, 3, 40, False), 302, stock_exhausted, 1.0),
    GoldenCase("quad-mid", 4, PRODUCTION, 401, reached(30, Phase.DRAW)),
    GoldenCase("quad-inactive", 4, GameConfig(max_turns=3), 402, seat_inactive),
    GoldenCase("quint-four-decks", 5, GameConfig(4, 21, 5, 30), 501, reached(37, Phase.DISCARD)),
    GoldenCase("hexa-deal", 6, PRODUCTION, 601, dealt),
    GoldenCase("hexa-late", 6, PRODUCTION, 602, reached(150, Phase.DRAW)),
    GoldenCase("hexa-terminal", 6, GameConfig(max_turns=2), 603, terminal),
)


def choose(obs, rng, stock_bias):
    legal = [action for action, ok in enumerate(legal_action_mask(obs)) if ok]
    if legal == [ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD]:
        if len(obs.discard_pile) == 1 and refilled(obs):
            return ACTION_TAKE_DISCARD
        return ACTION_DRAW_STOCK if rng.random() < stock_bias else ACTION_TAKE_DISCARD
    return rng.choice(legal)


def golden_observation(case):
    if case.seats == 1:
        env = PappluEnv(case.config)
        if case.warm_start:
            env.reset_warm_start(case.seed)
        else:
            env.reset(case.seed)
    else:
        env = MultiplayerEnv(MatchConfig(game=case.config, players=case.seats))
        env.reset(case.seed)
    rng = random.Random(case.seed)
    obs = env.observe()
    for _ in range(STEP_LIMIT):
        if case.until(obs):
            return obs
        if obs.phase is Phase.TERMINAL:
            break
        env.step(choose(obs, rng, case.stock_bias))
        obs = env.observe()
    raise RuntimeError("golden case %s never reached its position; pick another seed" % case.name)


def golden_entry(case, obs):
    observation = observation_json(obs)
    return {
        "name": case.name,
        "seats": case.seats,
        "observation": observation,
        "encoded": encode(observation),
        "historyChannels": history_channels(observation),
    }


def add_reference_q(tensors, golden):
    layers = reference_layers(tensors)
    for entry in golden:
        entry["q"] = reference_q(layers, entry["encoded"], entry["historyChannels"])


def require_close(label, actual, expected, tolerance):
    if len(actual) != len(expected) or any(abs(a - b) > tolerance for a, b in zip(actual, expected)):
        raise SystemExit("%s: the pure-Python port disagrees with torch" % label)


def export_checkpoint(args, observations, golden):
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "--checkpoint and --registry need torch (pip install -r requirements-training.txt); "
            "--bootstrap runs without it"
        ) from None
    from src.checkpoints.io import file_sha256
    from src.checkpoints.registry import ModelRegistry
    from src.model.encoding import collate_observations
    from src.model.network import load_checkpoint

    if args.checkpoint is not None:
        path, kind, registry_id = args.checkpoint, "checkpoint", None
        if not path.is_file():
            raise SystemExit("%s is not a checkpoint file" % path)
        sha256 = file_sha256(path)
    else:
        if not (args.registry / "champion.json").is_file():
            raise SystemExit("%s has no champion.json; pass --checkpoint instead" % args.registry)
        registry = ModelRegistry(args.registry)
        record = registry.champion()
        path, kind, sha256, registry_id = registry.model_path(record.id), "registry", record.sha256, record.id
    network, config, _ = load_checkpoint(str(path), device=torch.device("cpu"))
    state = network.state_dict()
    if network.spec != SPEC or [(name, tuple(value.shape)) for name, value in state.items()] != list(TENSOR_SHAPES):
        raise SystemExit("web/qnet.mjs runs the width 64, 4 head dynamic_table network; %s differs" % path)
    tensors = {name: value.detach().cpu().float().reshape(-1).tolist() for name, value in state.items()}
    add_reference_q(tensors, golden)
    with torch.no_grad():
        for obs, entry in zip(observations, golden):
            batch = collate_observations([obs], torch.device("cpu"))
            label = "golden case " + entry["name"]
            encoded = entry["encoded"]
            for key, tensor in (
                ("cards", batch.cards), ("initialHand", batch.initial_hand), ("context", batch.context),
                ("players", batch.players), ("events", batch.event_features),
            ):
                require_close(label + " " + key, encoded[key], tensor[0].reshape(-1).tolist(), INPUT_TOLERANCE)
            for key, tensor in (
                ("top", batch.top), ("nextPlayer", batch.next_player),
                ("previousPlayer", batch.previous_player), ("eventKind", batch.event_kind),
                ("eventFace", batch.event_face), ("offeredTop", batch.offered_top),
                ("eventActor", batch.event_actor),
            ):
                if encoded[key] != tensor[0].tolist():
                    raise SystemExit("%s %s: the pure-Python port disagrees with torch" % (label, key))
            channels = network._history_channels(batch)[0].reshape(-1).tolist()
            require_close(label + " historyChannels", entry["historyChannels"], channels, INPUT_TOLERANCE)
            q = network(batch)[0].tolist()
            require_close(label + " q", entry["q"], q, TOLERANCE)
            entry["q"] = q
    return tensors, config, model_source(kind, tensors, sha256=sha256, registry_id=registry_id)


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    path.write_bytes((text + "\n").encode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export dynamic_table weights for web/qnet.mjs.")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path, help="export this .pt checkpoint (needs torch)")
    weights.add_argument("--registry", type=Path, help="export this registry's champion (needs torch)")
    weights.add_argument(
        "--bootstrap", action="store_true",
        help="seeded untrained weights without torch; the default when no checkpoint is given",
    )
    parser.add_argument("--seed", type=int, help="bootstrap seed (default %d)" % BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "web" / "papplu-model.json")
    parser.add_argument("--golden", type=Path, default=ROOT / "tests" / "fixtures" / "qnet-golden.json")
    args = parser.parse_args(argv)
    from_torch = args.checkpoint is not None or args.registry is not None
    if from_torch and args.seed is not None:
        parser.error("--seed only applies to bootstrap weights")

    observations = [golden_observation(case) for case in GOLDEN_CASES]
    golden = [golden_entry(case, obs) for case, obs in zip(GOLDEN_CASES, observations)]
    if from_torch:
        tensors, config, source = export_checkpoint(args, observations, golden)
    else:
        seed = BOOTSTRAP_SEED if args.seed is None else args.seed
        tensors, config = bootstrap_tensors(seed), PRODUCTION
        source = model_source("bootstrap", tensors, seed=seed)
        add_reference_q(tensors, golden)
    write_json(args.output, model_document(tensors, config, source))
    write_json(args.golden, {
        "format": GOLDEN_FORMAT,
        "formatVersion": FORMAT_VERSION,
        "modelSha256": source["sha256"],
        "tensorsSha256": source["tensorsSha256"],
        "oracle": "torch" if from_torch else "python-reference",
        "sources": {
            path: hashlib.sha256((ROOT / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for path in SOURCES
        },
        "tolerance": TOLERANCE,
        "cases": golden,
    })
    label = {
        "bootstrap": "UNTRAINED bootstrap weights, seed %s, not a champion" % source["seed"],
        "checkpoint": "checkpoint weights",
        "registry": "registry champion %s" % source["registryId"],
    }[source["kind"]]
    print("wrote %s: %s, sha256 %s" % (args.output, label, source["sha256"]))
    print("wrote %s: %d golden observations" % (args.golden, len(golden)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
