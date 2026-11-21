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
from src.model.encoding import (
    CARD_CHANNELS, CONTEXT_CHANNELS, LAYOUT, NUM_SEAT_SLOTS, SEAT_CHANNELS,
    Features, encode, feature_values,
)
from src.model.spec import ARCHITECTURE, CHECKPOINT_VERSION, ModelSpec


FORMAT = "papplu-qnet"
GOLDEN_FORMAT = "papplu-qnet-golden"
FORMAT_VERSION = 2
BOOTSTRAP_SEED = 1729
BOOTSTRAP_SENSITIVITY_GAINS = {
    "card_embed.weight": 2.0, "table_embed.weight": 2.0,
    "draw_top.weight": 2.0, "draw_set.weight": 2.0,
    "face_out.weight": 2.0, "draw_out.weight": 2.0,
}
PRODUCTION = GameConfig(max_turns=60)
TOLERANCE = 1e-5
INPUT_TOLERANCE = 1e-6
STEP_LIMIT = 1000
SOURCES = (
    "src/evaluate.py", "src/game/environment.py", "src/game/information.py",
    "src/model/encoding.py", "src/model/network.py", "src/model/spec.py", "tools/weights_export.py",
)


def tensor_shapes(spec=ModelSpec()):
    width, shapes = spec.width, []

    def layer(name, output, inputs=None, bias=True):
        shape = (output,) if inputs is None else (output, inputs)
        shapes.append(("policy." + name + ".weight", shape))
        if bias:
            shapes.append(("policy." + name + ".bias", (output,)))

    layer("card_embed", width, len(CARD_CHANNELS))
    layer("table_embed", width, NUM_SEAT_SLOTS * len(SEAT_CHANNELS) + len(CONTEXT_CHANNELS))
    for block in range(spec.blocks):
        prefix = "blocks.%d." % block
        layer(prefix + "card_norm", width)
        layer(prefix + "table_norm", width)
        layer(prefix + "w_seq", width * 5, width, bias=False)
        for name in ("w_rank_set", "w_suit_run", "w_table"):
            layer(prefix + name, width, width, bias=False)
        layer(prefix + "card_ffn", width * 2, width)
        layer(prefix + "card_out", width, width)
        layer(prefix + "table_fc1", width, width * 3)
        layer(prefix + "table_fc2", width, width)
    layer("card_final", width)
    layer("table_final", width)
    layer("face_local", width, width)
    layer("face_set", width, width, bias=False)
    layer("face_table", width, width, bias=False)
    layer("face_out", 1, width)
    layer("draw_table", width, width)
    layer("draw_top", width, width, bias=False)
    layer("draw_set", width, width, bias=False)
    layer("draw_out", 2, width)
    layer("value_hidden", width, width)
    layer("value_out", 1, width)
    return tuple(shapes)


def packed(values):
    return struct.pack("<%df" % len(values), *values)


def float32(values):
    return list(struct.unpack("<%df" % len(values), packed(values)))


def bootstrap_tensors(seed, spec=ModelSpec()):
    shapes, rng, tensors = dict(tensor_shapes(spec)), random.Random(seed), {}
    for name, shape in shapes.items():
        local = name.removeprefix("policy.")
        is_norm = local.endswith(("_norm.weight", "_norm.bias", "_final.weight", "_final.bias"))
        weight_shape = shapes[name.rsplit(".", 1)[0] + ".weight"]
        bound = 0.1 if is_norm else 1.0 / math.sqrt(math.prod(weight_shape[1:]))
        bound *= BOOTSTRAP_SENSITIVITY_GAINS.get(local, 1.0)
        center = 1.0 if is_norm and name.endswith(".weight") else 0.0
        tensors[name] = float32([center + rng.uniform(-bound, bound) for _ in range(math.prod(shape))])
    return tensors


def model_source(kind, tensors, sha256=None, seed=None):
    digest = hashlib.sha256(b"".join(packed(values) for values in tensors.values())).hexdigest()
    return {
        "kind": kind, "trained": kind != "bootstrap",
        "seed": seed, "sha256": digest if sha256 is None else sha256,
        "tensorsSha256": digest,
    }


def game_config_json(config):
    return {
        "numDecks": config.num_decks, "cardsInHand": config.cards_in_hand,
        "requiredSequences": config.required_sequences, "maxTurns": config.max_turns,
        "recycleDiscard": config.recycle_discard,
    }


def model_document(tensors, config, source, spec=ModelSpec()):
    return {
        "format": FORMAT, "formatVersion": FORMAT_VERSION, "checkpointVersion": CHECKPOINT_VERSION,
        "encodingVersion": ENCODING_VERSION, "architecture": ARCHITECTURE,
        "modelSpec": spec.to_dict(), "numActions": NUM_ACTIONS,
        "gameConfig": game_config_json(config), "source": source,
        "tensors": {
            name: {"shape": list(shape), "data": base64.b64encode(packed(tensors[name])).decode("ascii")}
            for name, shape in tensor_shapes(spec)
        },
    }


def observation_json(obs):
    return {
        "hand": list(obs.hand), "discardPile": list(obs.discard_pile), "joker": obs.joker,
        "phase": obs.phase.value, "turnsRemaining": obs.turns_remaining,
        "stockRemaining": obs.stock_remaining, "config": game_config_json(obs.config),
        "players": [
            {"handSize": view.hand_size, "turnsRemaining": view.turns_remaining,
             "nextPlayer": view.next_player, "previousPlayer": view.previous_player, "active": view.active}
            for view in obs.players
        ],
        "history": [
            {"kind": int(event.kind), "actor": event.actor, "handSize": event.hand_size,
             "turnsRemaining": event.turns_remaining, "stockRemaining": event.stock_remaining,
             "nextPlayer": event.next_player, "face": event.face, "offeredTop": event.offered_top,
             "initialHand": list(event.initial_hand)}
            for event in obs.history
        ],
    }


def rows(values, width):
    return [values[start:start + width] for start in range(0, len(values), width)]


def dot(left, right):
    return math.fsum(map(operator.mul, left, right))


def linear(layers, name, x):
    weight = layers[name + ".weight"]
    bias = layers.get(name + ".bias", (0.0,) * len(weight))
    return [b + dot(row, x) for row, b in zip(weight, bias)]


def silu(values):
    return [value / (1.0 + math.exp(-value)) if value >= 0.0
            else value * math.exp(value) / (1.0 + math.exp(value)) for value in values]


def layer_norm(layers, name, x):
    mean = math.fsum(x) / len(x)
    scale = 1.0 / math.sqrt(math.fsum((value - mean) ** 2 for value in x) / len(x) + 1e-5)
    return [(value - mean) * scale * g + b
            for value, g, b in zip(x, layers[name + ".weight"], layers[name + ".bias"])]


def add(*vectors):
    return [math.fsum(column) for column in zip(*vectors)]


def suit_means(cards):
    return [[math.fsum(column) / 4 for column in zip(*cards[rank::13])] for rank in range(13)]


def reference_layers(tensors, spec=ModelSpec()):
    return {
        name.removeprefix("policy."): rows(tensors[name], math.prod(shape[1:])) if len(shape) > 1 else tensors[name]
        for name, shape in tensor_shapes(spec)
    }


def reference_q(layers, features, spec=ModelSpec()):
    values = feature_values(features) if isinstance(features, Features) else features
    fields = {field.name: list(values[field.offset:field.offset + field.size]) for field in LAYOUT}
    width = spec.width
    card_rows = rows(fields["card"], len(CARD_CHANNELS))
    cards = [linear(layers, "card_embed", card) for card in card_rows]
    table = linear(layers, "table_embed", fields["seat"] + fields["context"])
    zeros = [0.0] * width
    for block in range(spec.blocks):
        prefix = "blocks.%d." % block
        normalized = [layer_norm(layers, prefix + "card_norm", card) for card in cards]
        normalized_table = layer_norm(layers, prefix + "table_norm", table)
        sequence = []
        suit_context = []
        for suit in range(4):
            track = normalized[suit * 13:(suit + 1) * 13]
            projected = [linear(layers, prefix + "w_seq", row) for row in track + track[:1]]
            seq14 = [add(
                row[width * 2:width * 3],
                projected[position - 1][width:width * 2] if position > 0 else zeros,
                projected[position + 1][width * 3:width * 4] if position < 13 else zeros,
                projected[position - 2][:width] if position > 1 else zeros,
                projected[position + 2][width * 4:] if position < 12 else zeros,
            ) for position, row in enumerate(projected)]
            sequence.extend([[0.5 * (a + b) for a, b in zip(seq14[0], seq14[13])], *seq14[1:13]])
            suit_context.append(linear(layers, prefix + "w_suit_run",
                                       [math.fsum(column) / 13.0 for column in zip(*track)]))
        rank_context = [linear(layers, prefix + "w_rank_set", row) for row in suit_means(normalized)]
        table_context = linear(layers, prefix + "w_table", normalized_table)
        updated = []
        for face, card in enumerate(cards):
            gated = linear(layers, prefix + "card_ffn", add(
                sequence[face], rank_context[face % 13], suit_context[face // 13], table_context,
            ))
            activated = [a * b for a, b in zip(silu(gated[:width]), gated[width:])]
            updated.append(add(card, linear(layers, prefix + "card_out", activated)))
        cards = updated
        pooled = list(zip(*[layer_norm(layers, prefix + "card_norm", card) for card in cards]))
        table_input = normalized_table + [math.fsum(column) / NUM_FACES for column in pooled] + [
            max(column) for column in pooled
        ]
        table = add(table, linear(layers, prefix + "table_fc2",
                                 silu(linear(layers, prefix + "table_fc1", table_input))))
    cards = [layer_norm(layers, "card_final", card) for card in cards]
    table = layer_norm(layers, "table_final", table)
    sets = suit_means(cards)
    set_hidden = [linear(layers, "face_set", row) for row in sets]
    table_hidden = linear(layers, "face_table", table)
    faces = [silu(add(linear(layers, "face_local", card), set_hidden[face % 13], table_hidden))
             for face, card in enumerate(cards)]
    top = int(fields["top"][0])
    top_card, top_set = (cards[top], sets[top % 13]) if 0 <= top < NUM_FACES else (zeros, zeros)
    draw = silu(add(linear(layers, "draw_table", table), linear(layers, "draw_top", top_card), linear(layers, "draw_set", top_set)))
    advantage = linear(layers, "draw_out", draw) + [linear(layers, "face_out", face)[0] for face in faces]
    center = dot(advantage, fields["legal"]) / max(math.fsum(fields["legal"]), 1.0)
    value = linear(layers, "value_out", silu(linear(layers, "value_hidden", table)))[0]
    return [value + action - center for action in advantage]


def require_sensitive(tensors, features, golden, spec=ModelSpec()):
    layers = reference_layers(tensors, spec)
    for field in LAYOUT:
        for feature, entry in zip(features, golden):
            values = list(feature_values(feature))
            if not any(values[field.offset:field.offset + field.size]):
                continue
            values[field.offset:field.offset + field.size] = [0.0] * field.size
            changed = reference_q(layers, values, spec)
            if max(abs(a - b) for a, b in zip(changed, entry["q"])) > 100 * TOLERANCE:
                break
        else:
            raise SystemExit("bootstrap Q values are not sensitive to the %s feature field" % field.name)


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
    rng, obs = random.Random(case.seed), env.observe()
    for _ in range(STEP_LIMIT):
        if case.until(obs):
            return obs
        if obs.phase is Phase.TERMINAL:
            break
        env.step(choose(obs, rng, case.stock_bias))
        obs = env.observe()
    raise RuntimeError("golden case %s never reached its position; pick another seed" % case.name)


def require_close(label, actual, expected, tolerance):
    if len(actual) != len(expected) or any(
        not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > tolerance
        for a, b in zip(actual, expected)
    ):
        raise SystemExit("%s: the pure-Python reference disagrees with torch" % label)


def export_checkpoint(args, observations, features, golden):
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "--checkpoint needs torch (pip install -r requirements-training.txt); "
            "--bootstrap runs without it"
        ) from None
    from src.checkpoints.io import file_sha256
    from src.model.network import collate, load_checkpoint

    torch.set_num_threads(1)
    path = args.checkpoint
    if not path.is_file():
        raise SystemExit("%s is not a checkpoint file" % path)
    sha256 = file_sha256(path)
    network, config, _ = load_checkpoint(str(path), device=torch.device("cpu"))
    spec = network.spec
    tensors = {"policy." + name: value.detach().cpu().float().reshape(-1).tolist()
               for name, value in network.policy.state_dict().items()}
    if [("policy." + name, tuple(value.shape)) for name, value in network.policy.state_dict().items()] != list(tensor_shapes(spec)):
        raise SystemExit("policy tensors do not match the card_mlp export schema")
    layers = reference_layers(tensors, spec)
    with torch.inference_mode():
        batch = collate(features, torch.device("cpu"))
        q = network(batch).tolist()
        for row, (obs, feature, entry) in enumerate(zip(observations, features, golden)):
            label = "golden case " + entry["name"]
            require_close(label + " encode", feature_values(encode(obs)), entry["features"], INPUT_TOLERANCE)
            for field in LAYOUT:
                actual = getattr(batch, field.name)[row].reshape(-1).tolist()
                require_close(label + " " + field.name, actual, entry["features"][field.offset:field.offset + field.size], INPUT_TOLERANCE)
            require_close(label + " q", reference_q(layers, feature, spec), q[row], TOLERANCE)
            entry["q"] = q[row]
    return tensors, config, model_source("checkpoint", tensors, sha256=sha256), spec


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export card_mlp policy weights with v7 encoding for web/qnet.mjs.")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path, help="export this .pt checkpoint (needs torch)")
    weights.add_argument("--bootstrap", action="store_true", help="seeded untrained weights without torch, the default")
    parser.add_argument("--seed", type=int, help="bootstrap seed (default %d)" % BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "web" / "papplu-model.json")
    parser.add_argument("--golden", type=Path, default=ROOT / "tests" / "fixtures" / "qnet-golden.json")
    args = parser.parse_args(argv)
    from_torch = args.checkpoint is not None
    if from_torch and args.seed is not None:
        parser.error("--seed only applies to bootstrap weights")
    observations = [golden_observation(case) for case in GOLDEN_CASES]
    features = [encode(obs) for obs in observations]
    golden = [
        {"name": case.name, "seats": case.seats, "observation": observation_json(obs),
         "features": list(feature_values(feature))}
        for case, obs, feature in zip(GOLDEN_CASES, observations, features)
    ]
    if from_torch:
        tensors, config, source, spec = export_checkpoint(args, observations, features, golden)
    else:
        seed, spec = BOOTSTRAP_SEED if args.seed is None else args.seed, ModelSpec()
        tensors, config = bootstrap_tensors(seed, spec), PRODUCTION
        source = model_source("bootstrap", tensors, seed=seed)
        layers = reference_layers(tensors, spec)
        for entry, feature in zip(golden, features):
            entry["q"] = reference_q(layers, feature, spec)
        require_sensitive(tensors, features, golden, spec)
    write_json(args.output, model_document(tensors, config, source, spec))
    write_json(args.golden, {
        "format": GOLDEN_FORMAT, "formatVersion": FORMAT_VERSION, "modelSha256": source["sha256"],
        "tensorsSha256": source["tensorsSha256"], "oracle": "torch" if from_torch else "python-reference",
        "sources": {path: hashlib.sha256((ROOT / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                    for path in SOURCES},
        "tolerance": TOLERANCE, "featureLayout": [[field.name, list(field.shape)] for field in LAYOUT],
        "channels": {"card": CARD_CHANNELS, "seat": SEAT_CHANNELS, "context": CONTEXT_CHANNELS},
        "cases": golden,
    })
    label = {
        "bootstrap": "UNTRAINED bootstrap weights, seed %s" % source["seed"],
        "checkpoint": "checkpoint weights",
    }[source["kind"]]
    print("wrote %s: %s, sha256 %s" % (args.output, label, source["sha256"]))
    print("wrote %s: %d golden observations" % (args.golden, len(golden)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
