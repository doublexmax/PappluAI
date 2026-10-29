import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import { encodeObservation, historyChannels, parseQModel, qValues } from "../web/qnet.mjs";

const ROOT = new URL("../", import.meta.url);
const MODEL = readJson("web/papplu-model.json");
const GOLDEN = readJson("tests/fixtures/qnet-golden.json");
const [INITIAL, DRAW_STOCK, TAKE_DISCARD, DISCARD, REFILL, TERMINAL] = [0, 1, 2, 3, 4, 5];
const STATE_DICT = [
  ["suit_conv1.weight", [64, 5, 3]], ["suit_conv1.bias", [64]],
  ["suit_conv2.weight", [64, 64, 3]], ["suit_conv2.bias", [64]],
  ["card_attention.in_proj_weight", [192, 64]], ["card_attention.in_proj_bias", [192]],
  ["card_attention.out_proj.weight", [64, 64]], ["card_attention.out_proj.bias", [64]],
  ["card_norm.weight", [64]], ["card_norm.bias", [64]],
  ["history_proj.weight", [1, 5]], ["history_proj.bias", [1]],
  ["context_features.weight", [64, 11]], ["context_features.bias", [64]],
  ["player_features.weight", [64, 6]], ["player_features.bias", [64]],
  ["event_summary.weight", [64, 8]], ["event_summary.bias", [64]],
  ["turn_relations.weight", [64, 192]], ["turn_relations.bias", [64]],
  ["context_gate.weight", [2, 64]], ["context_gate.bias", [2]],
  ["draw_head.0.weight", [64, 197]], ["draw_head.0.bias", [64]],
  ["draw_head.2.weight", [2, 64]], ["draw_head.2.bias", [2]],
  ["score.0.weight", [64, 210]], ["score.0.bias", [64]],
  ["score.2.weight", [1, 64]], ["score.2.bias", [1]],
];

function readJson(path) {
  return JSON.parse(readFileSync(new URL(path, ROOT), "utf8"));
}

function assertClose(actual, expected, tolerance, label) {
  assert.equal(actual.length, expected.length, `${label} length`);
  actual.forEach((value, i) => {
    if (!(Math.abs(value - expected[i]) <= tolerance)) assert.fail(`${label}[${i}]: got ${value}, expected ${expected[i]}`);
  });
}

function faces(counts) {
  const result = new Array(52).fill(0);
  for (const [face, n] of Object.entries(counts)) result[face] = n;
  return result;
}

function event(kind, actor, fields = {}) {
  return {
    kind, actor, handSize: 13, turnsRemaining: 10, stockRemaining: 52, nextPlayer: (actor + 1) % 3,
    face: null, offeredTop: null, initialHand: faces({}), ...fields,
  };
}

function threeSeatObservation() {
  return {
    hand: faces({ 0: 2, 14: 1 }),
    discardPile: [5, 7, 5],
    joker: 16,
    phase: 1,
    turnsRemaining: 20,
    stockRemaining: 60,
    config: { numDecks: 2, cardsInHand: 13, requiredSequences: 4, maxTurns: 30, recycleDiscard: true },
    players: [0, 1, 2].map((seat) => ({
      handSize: 13 - seat, turnsRemaining: 20, nextPlayer: (seat + 1) % 3, previousPlayer: (seat + 2) % 3, active: seat !== 2,
    })),
    history: [
      event(INITIAL, 0, { initialHand: faces({ 0: 1, 3: 1 }) }),
      event(DISCARD, 0, { face: 3 }),
      event(DRAW_STOCK, 1, { offeredTop: 3 }),
      event(TAKE_DISCARD, 2, { face: 3 }),
      event(DISCARD, 2, { face: 9 }),
      event(TAKE_DISCARD, 1, { face: 9 }),
      event(DISCARD, 1, { face: 11 }),
      event(DRAW_STOCK, 2, { offeredTop: 11 }),
      event(REFILL, 1, { face: 20 }),
      event(TERMINAL, 2, { face: 21 }),
      event(DRAW_STOCK, 0, { face: 14, offeredTop: 11 }),
    ],
  };
}

function turn(seat, seats) {
  return [Math.sin(2 * Math.PI * seat / seats), Math.cos(2 * Math.PI * seat / seats)];
}

describe("qnet golden parity", () => {
  const cases = GOLDEN.cases;

  it("pins real tables from one to six seats with every event kind", () => {
    assert.deepEqual([...new Set(cases.map((c) => c.seats))].sort(), [1, 2, 3, 4, 5, 6]);
    for (const c of cases) assert.equal(c.observation.players.length, c.seats, c.name);
    for (let seats = 1; seats <= 6; seats += 1) {
      const longest = Math.max(...cases.filter((c) => c.seats === seats).map((c) => c.observation.history.length));
      assert.ok(longest >= 10, `${seats} seats need a case with real history`);
    }
    const kinds = new Set(cases.flatMap((c) => c.observation.history.map((e) => e.kind)));
    assert.deepEqual([...kinds].sort(), [0, 1, 2, 3, 4, 5]);
  });

  it("was written for the committed model and the current Python sources", () => {
    assert.equal(GOLDEN.modelSha256, MODEL.source.sha256);
    assert.equal(GOLDEN.tensorsSha256, MODEL.source.tensorsSha256);
    assert.deepEqual(Object.keys(GOLDEN.sources).sort(), [
      "src/model/encoding.py", "src/model/network.py", "tools/weights_export.py",
    ]);
    for (const [path, digest] of Object.entries(GOLDEN.sources)) {
      // Python hashed read_text(), which folds CRLF, so a CRLF checkout must hash the same.
      const text = readFileSync(new URL(path, ROOT), "utf8").replace(/\r\n?/g, "\n");
      assert.equal(createHash("sha256").update(text).digest("hex"), digest, `${path} changed; rerun tools/weights_export.py`);
    }
  });

  it("encodes every observation as encoding.py does", () => {
    for (const c of cases) {
      const encoded = encodeObservation(c.observation);
      assert.deepEqual(Object.keys(encoded).sort(), Object.keys(c.encoded).sort());
      for (const [key, expected] of Object.entries(c.encoded)) {
        if (typeof expected === "number") assert.equal(encoded[key], expected, `${c.name} ${key}`);
        else assertClose(Array.from(encoded[key]), expected, 1e-12, `${c.name} ${key}`);
      }
    }
  });

  it("weighs history as QNetwork._history_channels does", () => {
    for (const c of cases) assertClose(Array.from(historyChannels(c.observation)), c.historyChannels, 1e-12, c.name);
  });

  it("reproduces all 54 reference Q values within 1e-4", () => {
    assert.equal(GOLDEN.tolerance, 1e-4);
    const results = qValues(parseQModel(MODEL), cases.map((c) => c.observation));
    assert.equal(results.length, cases.length);
    results.forEach((q, i) => {
      assert.ok(q instanceof Float32Array);
      assertClose(Array.from(q), cases[i].q, GOLDEN.tolerance, cases[i].name);
    });
  });
});

describe("papplu-model.json", () => {
  it("stores 70,795 float32 parameters in state_dict order under its tensor SHA-256", () => {
    assert.deepEqual(Object.keys(MODEL.tensors).sort(), STATE_DICT.map(([name]) => name).sort());
    const hash = createHash("sha256");
    let parameters = 0;
    for (const [name, shape] of STATE_DICT) {
      const size = shape.reduce((a, b) => a * b, 1);
      const bytes = Buffer.from(MODEL.tensors[name].data, "base64");
      assert.deepEqual(MODEL.tensors[name].shape, shape, name);
      assert.equal(bytes.length, 4 * size, name);
      parameters += size;
      hash.update(bytes);
    }
    assert.equal(parameters, 70795);
    assert.equal(hash.digest("hex"), MODEL.source.tensorsSha256);
  });
});

describe("parseQModel", () => {
  it("returns a frozen handle that keeps the tensors private", () => {
    const model = parseQModel(MODEL);
    assert.deepEqual(Object.keys(model), ["kind", "gameConfig", "source"]);
    assert.equal(model.kind, "papplu-qnet");
    assert.ok(Object.isFrozen(model) && Object.isFrozen(model.gameConfig) && Object.isFrozen(model.source));
    assert.throws(() => qValues({ ...model }, []), { name: "TypeError", message: /must come from parseQModel/ });
  });

  it("accepts checkpoint and registry labels", () => {
    for (const source of [
      { kind: "checkpoint", trained: true, champion: false, seed: null, registryId: null },
      { kind: "registry", trained: true, champion: true, seed: null, registryId: "gen-12" },
    ]) {
      const json = structuredClone(MODEL);
      Object.assign(json.source, source);
      assert.deepEqual(parseQModel(json).source, json.source);
    }
  });

  it("rejects anything but an object", () => {
    for (const value of [null, [], "model", 3]) {
      assert.throws(() => parseQModel(value), { name: "TypeError", message: /^model: expected an object$/ });
    }
  });

  const MALFORMED = [
    ["a missing field", (m) => { delete m.source; }, /^model\.source: missing$/],
    ["an unexpected field", (m) => { m.comment = "hi"; }, /^model\.comment: unexpected field$/],
    ["another format", (m) => { m.format = "onnx"; }, /^model\.format: /],
    ["a string format version", (m) => { m.formatVersion = "1"; }, /^model\.formatVersion: /],
    ["checkpoint version 2", (m) => { m.checkpointVersion = 2; }, /^model\.checkpointVersion: /],
    ["encoding version 1", (m) => { m.encodingVersion = 1; }, /^model\.encodingVersion: /],
    ["another architecture", (m) => { m.architecture = "conv"; }, /^model\.architecture: /],
    ["53 actions", (m) => { m.numActions = 53; }, /^model\.numActions: /],
    ["width 128", (m) => { m.modelSpec.width = 128; }, /^model\.modelSpec: expected width 64 and heads 4$/],
    ["8 heads", (m) => { m.modelSpec.heads = 8; }, /^model\.modelSpec: expected width 64 and heads 4$/],
    ["an extra spec field", (m) => { m.modelSpec.dropout = 0; }, /^model\.modelSpec\.dropout: unexpected field$/],
    ["zero decks", (m) => { m.gameConfig.numDecks = 0; }, /^model\.gameConfig\.numDecks: expected a positive integer$/],
    ["zero turns", (m) => { m.gameConfig.maxTurns = 0; }, /^model\.gameConfig\.maxTurns: expected a positive integer$/],
    ["fractional turns", (m) => { m.gameConfig.maxTurns = 1.5; }, /^model\.gameConfig\.maxTurns: /],
    ["a string recycle flag", (m) => { m.gameConfig.recycleDiscard = "yes"; }, /^model\.gameConfig\.recycleDiscard: /],
    ["more sequences than a hand holds", (m) => { m.gameConfig.requiredSequences = 8; }, /^model\.gameConfig\.requiredSequences: /],
    ["too few cards to deal", (m) => { Object.assign(m.gameConfig, { numDecks: 1, cardsInHand: 51 }); }, /^model\.gameConfig\.numDecks: /],
    ["an unknown source kind", (m) => { m.source.kind = "trained"; }, /^model\.source\.kind: /],
    ["bootstrap weights claiming training", (m) => { m.source.trained = true; }, /^model\.source\.trained: must be false for bootstrap weights$/],
    ["bootstrap weights claiming the title", (m) => { m.source.champion = true; }, /^model\.source\.champion: must be false for bootstrap weights$/],
    ["bootstrap weights without a seed", (m) => { m.source.seed = null; }, /^model\.source\.seed: /],
    [
      "a checkpoint claiming the title",
      (m) => { Object.assign(m.source, { kind: "checkpoint", trained: true, champion: true, seed: null }); },
      /^model\.source\.champion: must be false for checkpoint weights$/,
    ],
    [
      "a registry export without its id",
      (m) => { Object.assign(m.source, { kind: "registry", trained: true, champion: true, seed: null }); },
      /^model\.source\.registryId: /,
    ],
    [
      "a registry export with an empty id",
      (m) => { Object.assign(m.source, { kind: "registry", trained: true, champion: true, seed: null, registryId: "" }); },
      /^model\.source\.registryId: expected the champion's registry id$/,
    ],
    ["an uppercase digest", (m) => { m.source.sha256 = m.source.sha256.toUpperCase(); }, /^model\.source\.sha256: /],
    ["a short digest", (m) => { m.source.tensorsSha256 = "abc"; }, /^model\.source\.tensorsSha256: expected 64 lowercase hex digits$/],
    ["a missing tensor", (m) => { delete m.tensors["score.2.bias"]; }, /^model\.tensors\.score\.2\.bias: missing$/],
    ["an extra tensor", (m) => { m.tensors["score.4.bias"] = m.tensors["score.2.bias"]; }, /^model\.tensors\.score\.4\.bias: unexpected field$/],
    ["an extra tensor field", (m) => { m.tensors["score.2.bias"].dtype = "float32"; }, /^model\.tensors\["score\.2\.bias"\]\.dtype: unexpected field$/],
    ["a transposed shape", (m) => { m.tensors["score.2.weight"].shape = [64, 1]; }, /^model\.tensors\["score\.2\.weight"\]\.shape: expected \[1, 64\]$/],
    ["a string shape", (m) => { m.tensors["score.2.bias"].shape = "1"; }, /^model\.tensors\["score\.2\.bias"\]\.shape: /],
    [
      "truncated data",
      (m) => { m.tensors["score.0.bias"].data = m.tensors["score.0.bias"].data.slice(0, -8); },
      /^model\.tensors\["score\.0\.bias"\]\.data: expected 256 bytes, got 252$/,
    ],
    ["text that is not base64", (m) => { m.tensors["score.2.bias"].data = "not base64"; }, /^model\.tensors\["score\.2\.bias"\]\.data: expected base64 text$/],
    ["unpadded base64", (m) => { m.tensors["score.2.bias"].data = "AAAAAA"; }, /^model\.tensors\["score\.2\.bias"\]\.data: expected base64 text$/],
    ["numeric data", (m) => { m.tensors["score.2.bias"].data = 0; }, /^model\.tensors\["score\.2\.bias"\]\.data: expected base64 text$/],
    [
      "a NaN weight",
      (m) => { m.tensors["score.2.bias"].data = Buffer.from(new Float32Array([NaN]).buffer).toString("base64"); },
      /^model\.tensors\["score\.2\.bias"\]\.data: value 0 is not finite$/,
    ],
  ];

  for (const [label, mutate, message] of MALFORMED) {
    it(`rejects ${label}`, () => {
      const json = structuredClone(MODEL);
      mutate(json);
      assert.throws(() => parseQModel(json), { name: "TypeError", message });
    });
  }
});

describe("encodeObservation and historyChannels", () => {
  it("lays out cards, context, seats, and events as encoding.py does", () => {
    const encoded = encodeObservation(threeSeatObservation());
    const card = (face) => Array.from(encoded.cards.subarray(face * 5, face * 5 + 5));
    assert.deepEqual(card(0), [1, 0, 0, 0, 0]);
    assert.deepEqual(card(5), [0, 1, 1, 0, 0]);
    assert.deepEqual(card(7), [0, 0.5, 0, 0, 0]);
    assert.deepEqual(card(16), [0, 0, 0, 1, 1]);
    assert.deepEqual([3, 29, 42].map((face) => card(face)[4]), [1, 1, 1]);
    assert.equal(encoded.top, 5);
    assertClose(Array.from(encoded.initialHand), faces({ 0: 0.5, 3: 0.5 }), 0, "initialHand");
    assertClose(Array.from(encoded.context), [0, 1, 0, 20 / 30, 60 / 104, 2 / 6, 13 / 30, 0.4, 1, 0.5, 0.5], 1e-15, "context");
    assertClose(Array.from(encoded.players.subarray(6, 12)), [0, 12 / 13, 20 / 30, 1, ...turn(1, 3)], 1e-15, "seat 1");
    assert.equal(encoded.players[2 * 6 + 3], 0);
    assert.deepEqual(Array.from(encoded.nextPlayer), [1, 2, 0]);
    assert.deepEqual(Array.from(encoded.previousPlayer), [2, 0, 1]);
    assert.deepEqual(Array.from(encoded.eventKind.subarray(0, 3)), [INITIAL, DISCARD, DRAW_STOCK]);
    assert.deepEqual(Array.from(encoded.eventFace.subarray(0, 3)), [52, 3, 52]);
    assert.deepEqual(Array.from(encoded.offeredTop.subarray(0, 3)), [52, 52, 3]);
    assertClose(Array.from(encoded.events.subarray(16, 24)), [0, ...turn(1, 3), 1, 10 / 30, 0.5, ...turn(2, 3)], 1e-15, "event 2");
  });

  it("routes each event to its face and channel with recency weights", () => {
    const length = 11;
    const recency = (position) => (0.5 + 0.5 * (position / length)) / length;
    const expected = new Array(52 * 5).fill(0);
    const add = (face, channel, value) => { expected[face * 5 + channel] += value; };
    add(0, 0, 0.5 - 1);
    add(3, 0, 0.5);
    add(14, 0, -0.5);
    add(3, 0, recency(2));
    add(3, 2, recency(3));
    add(3, 3, recency(4));
    add(9, 4, recency(5));
    add(9, 1, recency(6));
    add(11, 2, recency(7));
    add(11, 4, recency(8));
    add(14, 0, recency(11));
    assertClose(Array.from(historyChannels(threeSeatObservation())), expected, 1e-15, "channels");
  });

  it("rejects anything but an object", () => {
    assert.throws(() => encodeObservation(null), { name: "TypeError", message: /^observation: expected an observation$/ });
  });

  const MALFORMED = [
    ["a short hand", (o) => { o.hand.pop(); }, /^observation\.hand: expected 52 face counts$/],
    ["no config", (o) => { delete o.config; }, /^observation\.config: expected a game config$/],
    ["no discard pile", (o) => { delete o.discardPile; }, /^observation\.discardPile: expected an array of faces$/],
    ["a pile face out of range", (o) => { o.discardPile.push(52); }, /^observation\.discardPile\[3\]: expected an integer from 0 to 51$/],
    ["a fractional joker", (o) => { o.joker = 1.5; }, /^observation\.joker: /],
    ["phase 3", (o) => { o.phase = 3; }, /^observation\.phase: expected an integer from 0 to 2$/],
    ["no players", (o) => { o.players = []; }, /^observation\.players: expected at least one player view$/],
    ["no history", (o) => { o.history = []; }, /^observation\.history: expected at least one event$/],
    ["no initial hand", (o) => { delete o.history[0].initialHand; }, /^observation\.history\[0\]\.initialHand: expected 52 face counts$/],
    ["a next seat off the table", (o) => { o.players[1].nextPlayer = 3; }, /^observation\.players\[1\]\.nextPlayer: expected an integer from 0 to 2$/],
    ["an actor off the table", (o) => { o.history[4].actor = 3; }, /^observation\.history\[4\]\.actor: /],
    ["an unknown event kind", (o) => { o.history[4].kind = 6; }, /^observation\.history\[4\]\.kind: expected an integer from 0 to 5$/],
    ["a missing event face", (o) => { delete o.history[4].face; }, /^observation\.history\[4\]\.face: /],
    ["a missing turn count", (o) => { delete o.turnsRemaining; }, /^observation: context features are not finite/],
    ["zero decks", (o) => { o.config.numDecks = 0; }, /^observation: cards features are not finite/],
  ];

  for (const [label, mutate, message] of MALFORMED) {
    it(`rejects ${label}`, () => {
      const obs = threeSeatObservation();
      mutate(obs);
      assert.throws(() => encodeObservation(obs), { name: "TypeError", message });
      assert.throws(() => historyChannels(obs), { name: "TypeError", message });
    });
  }

  it("names the bad observation in a qValues batch", () => {
    const model = parseQModel(MODEL);
    const bad = threeSeatObservation();
    bad.hand.pop();
    assert.throws(() => qValues(model, [threeSeatObservation(), bad]), { name: "TypeError", message: /^observations\[1\]\.hand: / });
    assert.throws(() => qValues(model, threeSeatObservation()), { name: "TypeError", message: /^qValues: observations must be an array$/ });
    assert.equal(qValues(model, [threeSeatObservation()])[0].length, 54);
  });
});
