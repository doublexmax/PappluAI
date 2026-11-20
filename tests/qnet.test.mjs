import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import {
  CHANNELS, FEATURE_LAYOUT, FEATURE_SIZE, encodeObservation, parseQModel, qValues, tensorShapes,
} from "../web/qnet.mjs";

const ROOT = new URL("../", import.meta.url);
const MODEL = readJson("web/papplu-model.json");
const GOLDEN = readJson("tests/fixtures/qnet-golden.json");
const OFFSETS = {};
let offset = 0;
for (const [name, shape] of FEATURE_LAYOUT) {
  const size = shape.reduce((a, b) => a * b, 1);
  OFFSETS[name] = offset;
  offset += size;
}

function readJson(path) {
  return JSON.parse(readFileSync(new URL(path, ROOT), "utf8"));
}

function assertClose(actual, expected, tolerance, label) {
  assert.equal(actual.length, expected.length, `${label} length`);
  actual.forEach((value, i) => {
    if (!(Math.abs(value - expected[i]) <= tolerance)) assert.fail(`${label}[${i}]: got ${value}, expected ${expected[i]}`);
  });
}

function counts(cards) {
  const hand = new Array(52).fill(0);
  cards.forEach(face => { hand[face] += 1; });
  return hand;
}

function observation(cards = [0, 1, 30], joker = 7) {
  const hand = counts(cards);
  return {
    hand, joker, discardPile: [2], phase: 0, turnsRemaining: 40, stockRemaining: 151,
    config: { numDecks: 3, cardsInHand: 3, requiredSequences: 1, maxTurns: 40, recycleDiscard: true },
    players: [{ handSize: cards.length, turnsRemaining: 40, nextPlayer: 0, previousPlayer: 0, active: true }],
    history: [{
      kind: 0, actor: 0, handSize: cards.length, turnsRemaining: 40, stockRemaining: 151, nextPlayer: 0,
      face: null, offeredTop: 2, initialHand: hand.slice(),
    }],
  };
}

function cardChannel(features, face, name) {
  return features[OFFSETS.card + face * 18 + CHANNELS.card.indexOf(name)];
}

function context(features, name) {
  return features[OFFSETS.context + CHANNELS.context.indexOf(name)];
}

function zeroModel(spec, biases = {}) {
  const json = structuredClone(MODEL);
  json.modelSpec = spec;
  json.tensors = Object.fromEntries(tensorShapes(spec).map(([name, shape]) => {
    const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
    if (Object.hasOwn(biases, name)) data.set(biases[name]);
    return [name, { shape, data: Buffer.from(data.buffer).toString("base64") }];
  }));
  return json;
}

describe("qnet golden parity", () => {
  it("encodes real solo through hexa tables with every public event kind", () => {
    const kinds = new Set();
    const seats = new Set();
    for (const c of GOLDEN.cases) {
      seats.add(c.seats);
      assert.equal(c.observation.players.length, c.seats, c.name);
      c.observation.history.forEach(e => kinds.add(e.kind));
      const actual = encodeObservation(c.observation);
      assert.ok(actual instanceof Float32Array);
      assert.equal(actual.length, 1063);
      assert.deepEqual(Array.from(actual), c.features, c.name);
    }
    assert.deepEqual([...seats].sort(), [1, 2, 3, 4, 5, 6]);
    assert.deepEqual([...kinds].sort(), [0, 1, 2, 3, 4, 5]);
    assert.deepEqual(FEATURE_LAYOUT, GOLDEN.featureLayout);
    assert.deepEqual(CHANNELS, GOLDEN.channels);
    assert.equal(FEATURE_SIZE, offset);
  });

  it("reproduces all 54 reference Q values within 1e-5", () => {
    const results = qValues(parseQModel(MODEL), GOLDEN.cases.map(c => c.observation));
    assert.equal(results.length, GOLDEN.cases.length);
    results.forEach((q, i) => {
      assert.ok(q instanceof Float32Array);
      assertClose(Array.from(q), GOLDEN.cases[i].q, 1e-5, GOLDEN.cases[i].name);
    });
  });

  it("binds the golden oracle to the model bytes and current Python sources", () => {
    assert.equal(GOLDEN.modelSha256, MODEL.source.sha256);
    assert.equal(GOLDEN.tensorsSha256, MODEL.source.tensorsSha256);
    assert.deepEqual(Object.keys(GOLDEN.sources).sort(), [
      "src/evaluate.py", "src/game/environment.py", "src/game/information.py",
      "src/model/encoding.py", "src/model/network.py", "src/model/spec.py", "tools/weights_export.py",
    ]);
    for (const [path, digest] of Object.entries(GOLDEN.sources)) {
      const text = readFileSync(new URL(path, ROOT), "utf8").replace(/\r\n?/g, "\n");
      assert.equal(createHash("sha256").update(text).digest("hex"), digest, `${path} changed; rerun tools/weights_export.py`);
    }
    const shapes = tensorShapes(MODEL.modelSpec);
    assert.deepEqual(Object.keys(MODEL.tensors), shapes.map(([name]) => name));
    const hash = createHash("sha256");
    for (const [name, shape] of shapes) {
      assert.ok(name.startsWith("policy."));
      const bytes = Buffer.from(MODEL.tensors[name].data, "base64");
      assert.deepEqual(MODEL.tensors[name].shape, shape, name);
      assert.equal(bytes.length, 4 * shape.reduce((a, b) => a * b, 1), name);
      hash.update(bytes);
    }
    assert.equal(hash.digest("hex"), MODEL.source.tensorsSha256);
  });

  it("evaluates dynamically shaped nonzero width-128, three-block weights", () => {
    const spec = { width: 128, blocks: 3 };
    let state = 1729;
    const next = () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return (state / 0x100000000) * 0.4 - 0.2;
    };
    const json = structuredClone(MODEL);
    json.modelSpec = spec;
    json.tensors = Object.fromEntries(tensorShapes(spec).map(([name, shape]) => {
      const len = shape.reduce((a, b) => a * b, 1);
      const data = new Float32Array(len);
      for (let i = 0; i < len; i += 1) {
        data[i] = name.endsWith(".norm.weight") || name.includes("_norm.weight") ? 1 + next() : next();
      }
      return [name, { shape, data: Buffer.from(data.buffer).toString("base64") }];
    }));
    assert.deepEqual(json.tensors["policy.card_embed.weight"].shape, [128, 18]);
    assert.deepEqual(json.tensors["policy.table_embed.weight"].shape, [128, 72]);
    assert.equal(Object.values(json.tensors).reduce((sum, tensor) =>
      sum + tensor.shape.reduce((a, b) => a * b, 1), 0), 672388);
    const inputs = [observation(), GOLDEN.cases.at(-1).observation];
    const full = qValues(parseQModel(json), inputs);
    const ablatedJson = structuredClone(json);
    const lastWeight = ablatedJson.tensors["policy.blocks.2.card_out.weight"];
    ablatedJson.tensors["policy.blocks.2.card_out.weight"] = {
      shape: lastWeight.shape,
      data: Buffer.from(new Float32Array(lastWeight.shape[0] * lastWeight.shape[1]).buffer).toString("base64"),
    };
    const ablated = qValues(parseQModel(ablatedJson), inputs);
    full.forEach((q, idx) => {
      assert.equal(q.length, 54);
      assert.ok(q.every(Number.isFinite));
      assert.ok(Math.max(...q) - Math.min(...q) > 1e-2);
      const blockDelta = Math.max(...Array.from(q, (v, a) => Math.abs(v - ablated[idx][a])));
      assert.ok(blockDelta > 1e-3, `block 2 did not affect output ${idx}: ${blockDelta}`);
    });
  });

  it("centers dueling advantages on legal actions and leaves terminal values uncentered", () => {
    const model = parseQModel(zeroModel({ width: 64, blocks: 2 }, {
      "policy.value_out.bias": [0.25], "policy.draw_out.bias": [1, 3], "policy.face_out.bias": [7],
    }));
    const draw = observation();
    const discard = observation();
    discard.phase = 1;
    const terminal = observation();
    terminal.phase = 2;
    const results = qValues(model, [draw, discard, terminal]);
    assert.deepEqual(Array.from(results[0]), [-0.75, 1.25, ...new Array(52).fill(5.25)]);
    assert.deepEqual(Array.from(results[1]), [-5.75, -3.75, ...new Array(52).fill(0.25)]);
    assert.deepEqual(Array.from(results[2]), [1.25, 3.25, ...new Array(52).fill(7.25)]);
  });
});

describe("parseQModel", () => {
  it("returns a frozen model handle with private policy weights", () => {
    const model = parseQModel(MODEL);
    assert.deepEqual(Object.keys(model), ["kind", "spec", "gameConfig", "source"]);
    assert.equal(model.kind, "papplu-qnet");
    assert.ok(Object.isFrozen(model) && Object.isFrozen(model.spec) && Object.isFrozen(model.gameConfig) && Object.isFrozen(model.source));
    assert.throws(() => qValues({ ...model }, []), /must come from parseQModel/);
    assert.deepEqual(qValues(model, []), []);
    assert.deepEqual(MODEL.tensors["policy.card_embed.weight"].shape, [64, 18]);
    assert.deepEqual(MODEL.tensors["policy.table_embed.weight"].shape, [64, 72]);
    assert.equal(Object.values(MODEL.tensors).reduce((sum, tensor) =>
      sum + tensor.shape.reduce((a, b) => a * b, 1), 0), 126660);
  });

  it("accepts bootstrap and checkpoint source identities without registry metadata", () => {
    for (const source of [
      { kind: "bootstrap", trained: false, seed: 1729 },
      { kind: "checkpoint", trained: true, seed: null },
    ]) {
      const json = structuredClone(MODEL);
      Object.assign(json.source, source);
      assert.deepEqual(parseQModel(json).source, json.source);
      assert.deepEqual(Object.keys(parseQModel(json).source), ["kind", "trained", "seed", "sha256", "tensorsSha256"]);
    }
  });

  it("accepts widths that do not need a head partition", () => {
    const json = zeroModel({ width: 9, blocks: 1 }, { "policy.value_out.bias": [1.5] });
    const model = parseQModel(json);
    assert.deepEqual(model.spec, { width: 9, blocks: 1 });
    assert.deepEqual(Array.from(qValues(model, [observation()])[0]), new Array(54).fill(1.5));
  });

  it("rejects values that are not model objects", () => {
    for (const value of [null, [], "model", 3]) assert.throws(() => parseQModel(value), /^TypeError: model: expected an object$/);
  });

  const MALFORMED = [
    ["a missing source", m => { delete m.source; }, /model\.source: missing/],
    ["an extra field", m => { m.comment = "hi"; }, /model\.comment: unexpected/],
    ["another format", m => { m.format = "onnx"; }, /model\.format:/],
    ["a string format version", m => { m.formatVersion = "2"; }, /model\.formatVersion:/],
    ["a v4 checkpoint", m => { m.checkpointVersion = 4; }, /model\.checkpointVersion:/],
    ["an old encoding", m => { m.encodingVersion = 3; }, /model\.encodingVersion:/],
    ["another architecture", m => { m.architecture = "axial_table"; }, /model\.architecture:/],
    ["53 actions", m => { m.numActions = 53; }, /model\.numActions:/],
    ["a missing model spec", m => { delete m.modelSpec; }, /model\.modelSpec: missing/],
    ["a null model spec", m => { m.modelSpec = null; }, /model\.modelSpec: expected an object/],
    ["width four", m => { m.modelSpec.width = 4; }, /model\.modelSpec\.width:/],
    ["legacy heads", m => { m.modelSpec.heads = 4; }, /model\.modelSpec\.heads: unexpected/],
    ["a missing width setting", m => { delete m.modelSpec.width; }, /model\.modelSpec\.width: missing/],
    ["zero blocks", m => { m.modelSpec.blocks = 0; }, /model\.modelSpec\.blocks:/],
    ["a missing blocks setting", m => { delete m.modelSpec.blocks; }, /model\.modelSpec\.blocks: missing/],
    ["an extra spec field", m => { m.modelSpec.dropout = 0; }, /model\.modelSpec\.dropout: unexpected/],
    ["zero decks", m => { m.gameConfig.numDecks = 0; }, /model\.gameConfig\.numDecks:/],
    ["fractional turns", m => { m.gameConfig.maxTurns = 1.5; }, /model\.gameConfig\.maxTurns:/],
    ["a string recycle flag", m => { m.gameConfig.recycleDiscard = "yes"; }, /model\.gameConfig\.recycleDiscard:/],
    ["an impossible quota", m => { m.gameConfig.requiredSequences = 8; }, /model\.gameConfig\.requiredSequences:/],
    ["too few cards", m => { Object.assign(m.gameConfig, { numDecks: 1, cardsInHand: 51 }); }, /model\.gameConfig\.numDecks:/],
    ["an unknown source", m => { m.source.kind = "trained"; }, /model\.source\.kind:/],
    ["a legacy registry source", m => { m.source.kind = "registry"; }, /model\.source\.kind:/],
    ["bootstrap training claims", m => { m.source.trained = true; }, /model\.source\.trained:/],
    ["bootstrap champion claims", m => { m.source.champion = true; }, /model\.source\.champion:/],
    ["a legacy registry id", m => { m.source.registryId = null; }, /model\.source\.registryId: unexpected field/],
    ["a missing bootstrap seed", m => { m.source.seed = null; }, /model\.source\.seed:/],
    ["an untrained checkpoint", m => { Object.assign(m.source, { kind: "checkpoint", seed: null }); }, /model\.source\.trained:/],
    ["a checkpoint seed", m => { Object.assign(m.source, { kind: "checkpoint", trained: true }); }, /model\.source\.seed:/],
    ["an uppercase digest", m => { m.source.sha256 = m.source.sha256.toUpperCase(); }, /model\.source\.sha256:/],
    ["a short digest", m => { m.source.tensorsSha256 = "abc"; }, /model\.source\.tensorsSha256:/],
    ["a missing tensor", m => { delete m.tensors["policy.face_out.bias"]; }, /model\.tensors\.policy\.face_out\.bias: missing/],
    ["an auxiliary tensor", m => { m.tensors["aux.face_win.bias"] = m.tensors["policy.face_out.bias"]; }, /model\.tensors\.aux\.face_win\.bias: unexpected/],
    ["an extra tensor field", m => { m.tensors["policy.face_out.bias"].dtype = "float32"; }, /model\.tensors\["policy.face_out.bias"\]\.dtype:/],
    ["a transposed shape", m => { m.tensors["policy.face_out.weight"].shape = [64, 1]; }, /model\.tensors\["policy.face_out.weight"\]\.shape:/],
    ["a string shape", m => { m.tensors["policy.face_out.bias"].shape = "1"; }, /model\.tensors\["policy.face_out.bias"\]\.shape:/],
    ["truncated data", m => { m.tensors["policy.face_local.bias"].data = m.tensors["policy.face_local.bias"].data.slice(0, -8); }, /expected 256 bytes, got 252/],
    ["invalid base64", m => { m.tensors["policy.face_out.bias"].data = "not base64"; }, /expected base64 text/],
    ["unpadded base64", m => { m.tensors["policy.face_out.bias"].data = "AAAAAA"; }, /expected base64 text/],
    ["numeric tensor data", m => { m.tensors["policy.face_out.bias"].data = 0; }, /expected base64 text/],
    ["a NaN weight", m => {
      m.tensors["policy.face_out.bias"].data = Buffer.from(new Float32Array([NaN]).buffer).toString("base64");
    }, /value 0 is not finite/],
  ];
  for (const [label, mutate, message] of MALFORMED) {
    it(`rejects ${label}`, () => {
      const json = structuredClone(MODEL);
      mutate(json);
      assert.throws(() => parseQModel(json), { name: "TypeError", message });
    });
  }

  for (const field of ["width", "blocks"]) {
    it(`rejects noninteger ${field} settings`, () => {
      for (const value of [1.5, true, "64", null, undefined, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
        const json = structuredClone(MODEL);
        json.modelSpec[field] = value;
        assert.throws(() => parseQModel(json), new RegExp(`model\\.modelSpec\\.${field}:`));
      }
    });
  }
});

describe("encoded meld channels", () => {
  it("marks Q-K-A and A-2-3 as natural and excludes K-A-2", () => {
    for (const cards of [[0, 1, 2], [11, 12, 0]]) {
      const features = encodeObservation(observation(cards));
      for (const channel of ["natural", "pure", "any_meld"]) {
        assert.deepEqual(Array.from({ length: 52 }, (_, face) => face)
          .filter(face => cardChannel(features, face, channel)), cards.toSorted((a, b) => a - b));
      }
      assert.equal(context(features, "unavoidable_dw"), 0);
    }
    const bad = encodeObservation(observation([12, 0, 1]));
    for (let face = 0; face < 52; face += 1) {
      assert.deepEqual(["natural", "pure", "any_meld"].map(name => cardChannel(bad, face, name)), [0, 0, 0]);
    }
    assert.equal(cardChannel(bad, 0, "adjacent"), 1);
    assert.equal(cardChannel(bad, 1, "adjacent"), 0.5);
    assert.equal(cardChannel(bad, 12, "adjacent"), 0.5);
    assertClose([context(bad, "unavoidable_dw")], [22 / 30], 1e-7, "deadwood");
  });

  it("distinguishes exact-joker purity from same-rank wild substitution", () => {
    const pure = encodeObservation(observation([16, 17, 7]));
    const impure = encodeObservation(observation([16, 17, 33]));
    for (const face of [16, 17, 7]) assert.equal(cardChannel(pure, face, "pure"), 1);
    for (const face of [16, 17, 33]) {
      assert.equal(cardChannel(impure, face, "pure"), 0);
      assert.equal(cardChannel(impure, face, "any_meld"), 1);
    }
    assert.equal(cardChannel(pure, 16, "natural"), 0);
    assert.equal(context(pure, "unavoidable_dw"), 0);
    assert.equal(context(impure, "unavoidable_dw"), 0);
    const naturalWild = encodeObservation(observation([32, 33, 34]));
    for (const face of [32, 33, 34]) {
      assert.deepEqual(["natural", "pure", "any_meld"].map(name => cardChannel(naturalWild, face, name)), [1, 1, 1]);
    }
  });

  it("counts physical triplets and distinct-suit set mates without reusing duplicate faces", () => {
    const triplet = encodeObservation(observation([5, 5, 5]));
    assert.deepEqual(["adjacent", "set_mates", "natural", "pure", "any_meld"]
      .map(name => cardChannel(triplet, 5, name)), [0, 0, 0, 1, 1]);
    const set = encodeObservation(observation([5, 18, 31]));
    for (const face of [5, 18, 31]) {
      assert.deepEqual(["natural", "pure", "any_meld"].map(name => cardChannel(set, face, name)), [0, 0, 1]);
    }
    assertClose([cardChannel(set, 5, "set_mates"), cardChannel(set, 44, "set_mates")],
      [2 / 3, 1], 1e-7, "set mates");
    assert.equal(cardChannel(set, 4, "adjacent"), 0.5);
    const duplicate = encodeObservation(observation([5, 5, 7]));
    for (let face = 0; face < 52; face += 1) {
      assert.deepEqual(["natural", "pure", "any_meld"].map(name => cardChannel(duplicate, face, name)), [0, 0, 0]);
    }
    assertClose([context(duplicate, "unavoidable_dw")], [12 / 30], 1e-7, "duplicate deadwood");
    const wildSet = encodeObservation(observation([7, 20, 33]));
    for (const face of [7, 20, 33]) assert.equal(cardChannel(wildSet, face, "any_meld"), 1);
  });
});

describe("encodeObservation", () => {
  it("separates next-opponent and other held cards, passes, discards, and seat totals", () => {
    const obs = observation();
    obs.players = Array.from({ length: 4 }, (_, seat) => ({
      ...obs.players[0], nextPlayer: (seat + 1) % 4, previousPlayer: (seat + 3) % 4,
    }));
    obs.players[0].nextPlayer = 3;
    obs.players[2].active = false;
    obs.history[0].offeredTop = 5;
    const event = (kind, actor, face, offeredTop = null) => ({
      ...obs.history[0], kind, actor, face, offeredTop, initialHand: [],
    });
    obs.history.push(
      event(2, 3, 5, 5), event(2, 3, 5, 5), event(2, 1, 5, 5), event(2, 2, 16, 16),
      event(3, 2, 16), event(3, 2, 16), event(3, 3, 5), event(3, 1, 30),
      event(2, 0, 5, 5), event(3, 0, 5), event(1, 3, null, 16), event(1, 1, null, 16),
      event(1, 2, null), event(1, 3, null, 16),
      ...Array.from({ length: 4 }, () => event(4, 2, null)), event(1, 0, 14, 28),
    );
    obs.hand[14] += 1;
    obs.players[0].handSize = 4;
    obs.discardPile = [28];
    obs.phase = 1;
    const features = encodeObservation(obs);
    const seat = (s, channel) => features[OFFSETS.seat + s * 9 + CHANNELS.seat.indexOf(channel)];
    assertClose([
      cardChannel(features, 5, "next_held"), cardChannel(features, 5, "other_held"),
      cardChannel(features, 5, "opp_discarded"), cardChannel(features, 16, "opp_discarded"),
      cardChannel(features, 30, "opp_discarded"), cardChannel(features, 16, "next_passed"),
    ], [1 / 3, 1 / 3, 1 / 3, 2 / 3, 1 / 3, 2 / 3], 1e-7, "opponent cards");
    assertClose([seat(3, "takes"), seat(3, "passes"), seat(3, "held_total")],
      [2 / 40, 2 / 40, 1 / 3], 1e-7, "next opponent");
    assertClose([seat(1, "takes"), seat(1, "passes"), seat(1, "held_total")],
      [1 / 40, 1 / 40, 1 / 3], 1e-7, "other opponent");
    assertClose([seat(2, "takes"), seat(2, "passes"), seat(2, "held_total")],
      [1 / 40, 0, 0], 1e-7, "discarded known card");
    assertClose([seat(0, "takes"), seat(0, "passes"), seat(0, "held_total")],
      [1 / 40, 1 / 40, 0], 1e-7, "observer");
    assert.equal(seat(3, "next"), 1);
    assert.equal(seat(1, "next"), 0);
    assert.equal(cardChannel(features, 14, "next_held"), 0);
    assert.equal(context(features, "refills"), 1);
    assert.deepEqual(Array.from(features.slice(OFFSETS.seat + 4 * 9, OFFSETS.context)), new Array(18).fill(0));
    obs.players[0].nextPlayer = 0;
    const noNext = encodeObservation(obs);
    assert.equal(cardChannel(noNext, 5, "next_held"), 0);
    assert.equal(cardChannel(noNext, 16, "next_passed"), 0);
    assertClose([cardChannel(noNext, 5, "other_held")], [2 / 3], 1e-7, "all opponents");
  });

  it("encodes raw point totals and wild counts alongside meld membership", () => {
    const obs = observation([0, 1, 7, 12, 20]);
    obs.config.cardsInHand = 5;
    const features = encodeObservation(obs);
    assertClose([context(features, "deadwood"), context(features, "wilds")], [0.22, 2 / 12], 1e-7, "hand totals");
    assert.equal(cardChannel(features, 7, "indicator"), 1);
    assert.equal(cardChannel(features, 20, "wild"), 1);
    assert.equal(cardChannel(features, 20, "points"), 0);
    assert.equal(cardChannel(features, 0, "points"), 1);
    assert.equal(features.length, 1063);
    assert.equal(context(features, "unavoidable_dw"), 0);
  });

  it("writes the exact 18-channel card and context layouts", () => {
    const features = encodeObservation(observation([5, 18, 31]));
    assert.deepEqual(Array.from(features.slice(5 * 18, 6 * 18)), [
      1 / 3, 1, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 2 / 3, 0, 0, 1,
    ].map(Math.fround));
    assert.deepEqual(Array.from(features.slice(OFFSETS.context, OFFSETS.legal)), [
      1, 0, 0, 1, 1, 151 / 156, 1 / 156, 0, 0.5, 0.1, 0.1, 1, 1 / 6, 2 / 3, 0, 0, 0.18, 0,
    ].map(Math.fround));
  });

  it("masks a just-taken face even when another copy is held and keeps a single-face fallback", () => {
    const obs = observation([0, 5, 30]);
    obs.hand[30] += 1;
    obs.phase = 1;
    obs.players[0].handSize = 4;
    obs.history.push({
      ...obs.history[0], kind: 2, face: 30, offeredTop: 30, initialHand: [], handSize: 4,
    });
    const legal = input => Array.from(encodeObservation(input).slice(OFFSETS.legal, OFFSETS.top))
      .flatMap((allowed, action) => allowed ? [action] : []);
    assert.deepEqual(legal(obs), [2, 7]);
    obs.history.at(-1).kind = 1;
    assert.deepEqual(legal(obs), [2, 7, 32]);
    obs.history.at(-1).kind = 2;
    obs.hand = counts([30, 30, 30, 30]);
    obs.config.numDecks = 4;
    assert.deepEqual(legal(obs), [32]);
    obs.phase = 2;
    assert.deepEqual(legal(obs), []);
  });

  it("zeros the legal mask for terminal states and afterstates awaiting another seat", () => {
    const obs = observation();
    assert.deepEqual(Array.from(encodeObservation(obs).slice(OFFSETS.legal, OFFSETS.top)), [1, 1, ...new Array(52).fill(0)]);
    obs.phase = 2;
    assert.equal(context(encodeObservation(obs), "to_act"), 0);
    assert.deepEqual(Array.from(encodeObservation(obs).slice(OFFSETS.legal, OFFSETS.top)), new Array(54).fill(0));
    obs.phase = 0;
    obs.players.push({ ...obs.players[0] });
    obs.history.push({ ...obs.history[0], kind: 3, face: 30, offeredTop: null, initialHand: [], nextPlayer: 1 });
    const after = encodeObservation(obs);
    assert.equal(context(after, "to_act"), 0);
    assert.deepEqual(Array.from(after.slice(OFFSETS.legal, OFFSETS.top)), new Array(54).fill(0));
    obs.phase = 1;
    Object.assign(obs.history.at(-1), { kind: 2, actor: 1 });
    const otherDiscard = encodeObservation(obs);
    assert.equal(context(otherDiscard, "to_act"), 0);
    assert.deepEqual(Array.from(otherDiscard.slice(OFFSETS.legal, OFFSETS.top)), new Array(54).fill(0));
  });

  it("uses the empty-top sentinel and derives warm starts from the initial hand", () => {
    const obs = observation([0, 1, 2, 30]);
    obs.discardPile = [];
    obs.history[0].offeredTop = null;
    obs.phase = 1;
    const features = encodeObservation(obs);
    assert.equal(features[OFFSETS.top], 52);
    assert.equal(context(features, "warm_start"), 1);
    assert.equal(context(features, "to_act"), 1);
  });

  const MALFORMED = [
    ["a short hand", o => { o.hand.pop(); }, /observation\.hand:/],
    ["no config", o => { delete o.config; }, /observation\.config:/],
    ["no pile", o => { delete o.discardPile; }, /observation\.discardPile:/],
    ["a bad pile face", o => { o.discardPile.push(52); }, /observation\.discardPile\[1\]:/],
    ["a fractional joker", o => { o.joker = 1.5; }, /observation\.joker:/],
    ["phase three", o => { o.phase = 3; }, /observation\.phase:/],
    ["no players", o => { o.players = []; }, /observation\.players:/],
    ["no history", o => { o.history = []; }, /observation\.history:/],
    ["no initial hand", o => { delete o.history[0].initialHand; }, /observation\.history\[0\]\.initialHand:/],
    ["an actor off the table", o => { o.history[0].actor = 1; }, /observation\.history\[0\]\.actor:/],
    ["an unknown event kind", o => { o.history[0].kind = 6; }, /observation\.history\[0\]\.kind:/],
    ["a missing face", o => { delete o.history[0].face; }, /observation\.history\[0\]\.face:/],
    ["missing turns", o => { delete o.turnsRemaining; }, /observation\.turnsRemaining:/],
    ["a hidden discard take", o => { o.history[0].kind = 2; }, /observation\.history\[0\]\.face:/],
    ["a hidden discard", o => { o.history[0].kind = 3; }, /observation\.history\[0\]\.face:/],
    ["zero decks", o => { o.config.numDecks = 0; }, /observation\.config\.numDecks:/],
  ];
  for (const [label, mutate, message] of MALFORMED) {
    it(`rejects ${label}`, () => {
      const obs = observation();
      mutate(obs);
      assert.throws(() => encodeObservation(obs), { name: "TypeError", message });
    });
  }

  for (const [label, select, field] of [
    ["observer turns", o => o, "turnsRemaining"],
    ["stock", o => o, "stockRemaining"],
    ["hand size", o => o.players[0], "handSize"],
    ["player turns", o => o.players[0], "turnsRemaining"],
  ]) {
    it(`rejects invalid ${label} counts at the observation boundary`, () => {
      for (const value of [-1, 1.5, Number.MAX_SAFE_INTEGER + 1, Infinity, NaN, true, "3", null, undefined]) {
        const obs = observation();
        select(obs)[field] = value;
        assert.throws(() => encodeObservation(obs), /expected a nonnegative safe integer/);
      }
    });
  }

  it("accepts zero remaining counts in a terminal observation", () => {
    const obs = observation();
    Object.assign(obs, { phase: 2, turnsRemaining: 0, stockRemaining: 0 });
    Object.assign(obs.players[0], { handSize: 0, turnsRemaining: 0, active: false });
    const features = encodeObservation(obs);
    assert.equal(context(features, "terminal"), 1);
    assert.equal(context(features, "turns"), 0);
    assert.equal(context(features, "stock"), 0);
    assert.deepEqual(Array.from(features.slice(OFFSETS.seat, OFFSETS.seat + 5)), [1, 1, 0, 0, 0]);
  });

  it("names the bad observation in a qValues batch", () => {
    const model = parseQModel(MODEL);
    const bad = observation();
    bad.hand.pop();
    assert.throws(() => qValues(model, [observation(), bad]), /observations\[1\]\.hand:/);
    assert.throws(() => qValues(model, observation()), /observations must be an array/);
    assert.throws(() => encodeObservation(null), /observation: expected an observation/);
  });
});
