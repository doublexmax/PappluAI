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
  return features[OFFSETS.card + face * 17 + CHANNELS.card.indexOf(name)];
}

function context(features, name) {
  return features[OFFSETS.context + CHANNELS.context.indexOf(name)];
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
      assert.equal(actual.length, 3208);
      assert.deepEqual(Array.from(actual), c.features, c.name);
    }
    assert.deepEqual([...seats].sort(), [1, 2, 3, 4, 5, 6]);
    assert.deepEqual([...kinds].sort(), [0, 1, 2, 3, 4, 5]);
    assert.deepEqual(FEATURE_LAYOUT, GOLDEN.featureLayout);
    assert.deepEqual(CHANNELS, GOLDEN.channels);
    assert.equal(FEATURE_SIZE, offset);
  });

  it("reproduces all 54 reference Q values on all 15 cases within 1e-4", () => {
    const results = qValues(parseQModel(MODEL), GOLDEN.cases.map(c => c.observation));
    assert.equal(results.length, 15);
    results.forEach((q, i) => {
      assert.ok(q instanceof Float32Array);
      assertClose(Array.from(q), GOLDEN.cases[i].q, 1e-4, GOLDEN.cases[i].name);
    });
  });

  it("binds the golden oracle to the model bytes and current Python sources", () => {
    assert.equal(GOLDEN.modelSha256, MODEL.source.sha256);
    assert.equal(GOLDEN.tensorsSha256, MODEL.source.tensorsSha256);
    assert.deepEqual(Object.keys(GOLDEN.sources).sort(), [
      "src/game/belief.py", "src/model/encoding.py", "src/model/network.py", "tools/weights_export.py",
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

  it("evaluates dynamically shaped width-128, eight-head, three-block weights", () => {
    const json = structuredClone(MODEL);
    json.modelSpec = { width: 128, heads: 8, blocks: 3 };
    json.tensors = Object.fromEntries(tensorShapes(json.modelSpec).map(([name, shape]) => {
      const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
      if (name === "policy.value_out.bias") data[0] = 3;
      return [name, { shape, data: Buffer.from(data.buffer).toString("base64") }];
    }));
    const model = parseQModel(json);
    assert.deepEqual(model.spec, { width: 128, heads: 8, blocks: 3 });
    for (const q of qValues(model, [observation(), GOLDEN.cases.at(-1).observation])) {
      assert.deepEqual(Array.from(q), new Array(54).fill(3));
    }
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
  });

  it("accepts checkpoint and registry source identities", () => {
    for (const source of [
      { kind: "checkpoint", trained: true, champion: false, seed: null, registryId: null },
      { kind: "registry", trained: true, champion: true, seed: null, registryId: "gen-12" },
    ]) {
      const json = structuredClone(MODEL);
      Object.assign(json.source, source);
      assert.deepEqual(parseQModel(json).source, json.source);
    }
  });

  it("accepts a different valid head partition without changing tensor dimensions", () => {
    const json = structuredClone(MODEL);
    json.modelSpec.heads = 8;
    const model = parseQModel(json);
    assert.equal(model.spec.heads, 8);
    assert.equal(qValues(model, [observation()])[0].length, 54);
  });

  it("rejects values that are not model objects", () => {
    for (const value of [null, [], "model", 3]) assert.throws(() => parseQModel(value), /^TypeError: model: expected an object$/);
  });

  const MALFORMED = [
    ["a missing source", m => { delete m.source; }, /model\.source: missing/],
    ["an extra field", m => { m.comment = "hi"; }, /model\.comment: unexpected/],
    ["another format", m => { m.format = "onnx"; }, /model\.format:/],
    ["a string format version", m => { m.formatVersion = "2"; }, /model\.formatVersion:/],
    ["a v3 checkpoint", m => { m.checkpointVersion = 3; }, /model\.checkpointVersion:/],
    ["an old encoding", m => { m.encodingVersion = 1; }, /model\.encodingVersion:/],
    ["another architecture", m => { m.architecture = "dynamic_table"; }, /model\.architecture:/],
    ["53 actions", m => { m.numActions = 53; }, /model\.numActions:/],
    ["width four", m => { m.modelSpec.width = 4; }, /model\.modelSpec\.width:/],
    ["an indivisible head count", m => { m.modelSpec.heads = 3; }, /model\.modelSpec\.heads:/],
    ["zero heads", m => { m.modelSpec.heads = 0; }, /model\.modelSpec\.heads:/],
    ["zero blocks", m => { m.modelSpec.blocks = 0; }, /model\.modelSpec\.blocks:/],
    ["a missing blocks setting", m => { delete m.modelSpec.blocks; }, /model\.modelSpec\.blocks: missing/],
    ["an extra spec field", m => { m.modelSpec.dropout = 0; }, /model\.modelSpec\.dropout: unexpected/],
    ["zero decks", m => { m.gameConfig.numDecks = 0; }, /model\.gameConfig\.numDecks:/],
    ["fractional turns", m => { m.gameConfig.maxTurns = 1.5; }, /model\.gameConfig\.maxTurns:/],
    ["a string recycle flag", m => { m.gameConfig.recycleDiscard = "yes"; }, /model\.gameConfig\.recycleDiscard:/],
    ["an impossible quota", m => { m.gameConfig.requiredSequences = 8; }, /model\.gameConfig\.requiredSequences:/],
    ["too few cards", m => { Object.assign(m.gameConfig, { numDecks: 1, cardsInHand: 51 }); }, /model\.gameConfig\.numDecks:/],
    ["an unknown source", m => { m.source.kind = "trained"; }, /model\.source\.kind:/],
    ["bootstrap training claims", m => { m.source.trained = true; }, /model\.source\.trained:/],
    ["bootstrap champion claims", m => { m.source.champion = true; }, /model\.source\.champion:/],
    ["a missing bootstrap seed", m => { m.source.seed = null; }, /model\.source\.seed:/],
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
});

describe("encodeObservation", () => {
  it("marks Q-K-A and A-2-3 as natural and excludes K-A-2", () => {
    for (const cards of [[0, 1, 2], [11, 12, 0]]) {
      const features = encodeObservation(observation(cards));
      for (const face of cards) {
        assert.equal(cardChannel(features, face, "natural_run_member"), 1);
        assert.equal(cardChannel(features, face, "pure_meld_member"), 1);
      }
    }
    const bad = encodeObservation(observation([12, 0, 1]));
    for (const face of [12, 0, 1]) assert.equal(cardChannel(bad, face, "any_meld_member"), 0);
    assert.equal(context(bad, "pure_quota_deficit"), 1);
    assertClose([context(bad, "unavoidable_deadwood")], [22 / 30], 1e-7, "deadwood");
  });

  it("distinguishes exact-joker purity from same-rank wild substitution", () => {
    const pure = encodeObservation(observation([16, 17, 7]));
    const impure = encodeObservation(observation([16, 17, 33]));
    for (const face of [16, 17, 7]) assert.equal(cardChannel(pure, face, "pure_meld_member"), 1);
    for (const face of [16, 17, 33]) {
      assert.equal(cardChannel(impure, face, "pure_meld_member"), 0);
      assert.equal(cardChannel(impure, face, "any_meld_member"), 1);
    }
    assert.equal(cardChannel(pure, 16, "natural_run_member"), 0);
    const triplet = encodeObservation(observation([5, 5, 5]));
    assert.equal(cardChannel(triplet, 5, "pure_meld_member"), 1);
    assert.equal(cardChannel(triplet, 5, "natural_run_member"), 0);
  });

  it("records per-kind recency for each relative seat without collapsing opposite seats", () => {
    const obs = observation();
    obs.players.push({ ...obs.players[0], nextPlayer: 0, previousPlayer: 0 });
    obs.players[0].nextPlayer = obs.players[0].previousPlayer = 1;
    obs.history[0].offeredTop = 5;
    const event = (kind, face, offeredTop = null) => ({
      ...obs.history[0], kind, actor: 1, face, offeredTop, initialHand: [],
    });
    obs.history.push(event(2, 5, 5), event(3, 16), event(1, null, 16), event(3, 28));
    obs.history.push({ ...event(1, 14, 28), actor: 0 });
    obs.hand[14] += 1;
    obs.discardPile = [16, 28];
    obs.phase = 1;
    const features = encodeObservation(obs);
    const history = (face, seat, channel) =>
      features[OFFSETS.history + (face * 6 + seat) * 7 + CHANNELS.history.indexOf(channel)];
    assertClose([history(5, 1, "took"), history(5, 1, "held")], [1 / 3, 1 / 3], 1e-7, "known take");
    assertClose([history(16, 1, "discarded"), history(16, 1, "passed")], [1 / 3, 1 / 3], 1e-7, "safe card");
    assertClose([history(5, 1, "last_take"), history(16, 1, "last_discard"), history(16, 1, "last_pass")],
      [2 / 6, 3 / 6, 4 / 6], 1e-7, "recencies");
    assert.equal(history(14, 1, "took"), 0);
    assert.deepEqual(Array.from(features.slice(OFFSETS.seat + 2 * 11, OFFSETS.context)), new Array(44).fill(0));
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
    ["missing turns", o => { delete o.turnsRemaining; }, /features are not finite/],
    ["zero decks", o => { o.config.numDecks = 0; }, /observation\.config\.numDecks:/],
  ];
  for (const [label, mutate, message] of MALFORMED) {
    it(`rejects ${label}`, () => {
      const obs = observation();
      mutate(obs);
      assert.throws(() => encodeObservation(obs), { name: "TypeError", message });
    });
  }

  it("names the bad observation in a qValues batch", () => {
    const model = parseQModel(MODEL);
    const bad = observation();
    bad.hand.pop();
    assert.throws(() => qValues(model, [observation(), bad]), /observations\[1\]\.hand:/);
    assert.throws(() => qValues(model, observation()), /observations must be an array/);
    assert.throws(() => encodeObservation(null), /observation: expected an observation/);
  });
});
