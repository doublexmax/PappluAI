const NUM_FACES = 52;
const NUM_ACTIONS = 54;
const SEATS = 6;
const [INITIAL, DRAW_STOCK, TAKE_DISCARD, DISCARD, REFILL] = [0, 1, 2, 3, 4];
const RANK_POINTS = [10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];
const LEGAL_RUNS = [...Array.from({ length: 11 }, (_, r) => [r, r + 1, r + 2]), [11, 12, 0]];
const RUN_FACES = Array.from({ length: 4 }, (_, s) => LEGAL_RUNS.map(run => run.map(r => s * 13 + r))).flat();
const SET_FACES = Array.from({ length: 13 }, (_, r) =>
  [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]].map(suits => suits.map(s => s * 13 + r))).flat();

export const CHANNELS = Object.freeze({
  card: Object.freeze([
    "own", "own_ge1", "own_ge2", "own_ge3", "pile", "top", "indicator", "wild",
    "points", "opp_discarded", "next_held", "other_held", "next_passed",
  ]),
  seat: Object.freeze(["present", "observer", "active", "hand_size", "turns", "next", "takes", "passes", "held_total"]),
  context: Object.freeze([
    "draw", "discard", "terminal", "to_act", "turns", "stock", "pile", "refills", "decks", "hand",
    "sequences", "recycle", "players", "max_turns", "warm_start", "wilds", "deadwood",
  ]),
});
export const FEATURE_LAYOUT = Object.freeze([
  ["card", [52, 13]], ["seat", [6, 9]], ["context", [17]], ["legal", [54]], ["top", [1]],
].map(([name, shape]) => Object.freeze([name, Object.freeze(shape)])));
export const FEATURE_SIZE = 802;
const FIELDS = {};
let offset = 0;
for (const [name, shape] of FEATURE_LAYOUT) {
  const size = shape.reduce((a, b) => a * b, 1);
  FIELDS[name] = { offset, size };
  offset += size;
}

const HEADER = Object.freeze({
  format: "papplu-qnet", formatVersion: 2, checkpointVersion: 5,
  encodingVersion: 3, architecture: "card_mlp", numActions: NUM_ACTIONS,
});
const MODEL_KEYS = [...Object.keys(HEADER), "modelSpec", "gameConfig", "source", "tensors"];
const GAME_CONFIG_KEYS = ["numDecks", "cardsInHand", "requiredSequences", "maxTurns", "recycleDiscard"];
const SOURCE_KEYS = ["kind", "trained", "champion", "seed", "registryId", "sha256", "tensorsSha256"];
const SOURCE_KINDS = ["bootstrap", "checkpoint", "registry"];
const SHA256 = /^[0-9a-f]{64}$/;
const BASE64 = /^[A-Za-z0-9+/]*={0,2}$/;
const WEIGHTS = new WeakMap();

export function tensorShapes(spec) {
  const { width, blocks } = parseModelSpec(spec, "modelSpec");
  const shapes = [];
  function layer(name, output, inputs = null, bias = true) {
    shapes.push([`policy.${name}.weight`, inputs === null ? [output] : [output, inputs]]);
    if (bias) shapes.push([`policy.${name}.bias`, [output]]);
  }
  layer("card_embed", width, CHANNELS.card.length + 5);
  layer("table_embed", width, SEATS * CHANNELS.seat.length + CHANNELS.context.length + 1);
  for (let b = 0; b < blocks; b += 1) {
    const prefix = `blocks.${b}.`;
    layer(`${prefix}card_norm`, width);
    layer(`${prefix}table_norm`, width);
    layer(`${prefix}w_seq`, width * 3, width);
    for (const name of ["w_rank_set", "w_suit_run", "w_table"]) layer(prefix + name, width, width, false);
    layer(`${prefix}card_out`, width, width);
    layer(`${prefix}table_fc1`, width, width * 3);
    layer(`${prefix}table_fc2`, width, width);
  }
  layer("card_final", width);
  layer("table_final", width);
  layer("face_local", width, width);
  layer("face_set", width, width, false);
  layer("face_table", width, width, false);
  layer("face_out", 1, width);
  layer("draw_table", width, width);
  layer("draw_top", width, width, false);
  layer("draw_set", width, width, false);
  layer("draw_out", 2, width);
  layer("value_hidden", width, width);
  layer("value_out", 1, width);
  return shapes;
}

export function parseQModel(json) {
  requireKeys(json, MODEL_KEYS, "model");
  for (const [key, expected] of Object.entries(HEADER)) {
    if (json[key] !== expected) fail(`model.${key}`, `expected ${JSON.stringify(expected)}`);
  }
  const spec = parseModelSpec(json.modelSpec, "model.modelSpec");
  const gameConfig = parseGameConfig(json.gameConfig, "model.gameConfig");
  const source = parseSource(json.source, "model.source");
  const weights = parseTensors(json.tensors, spec, "model.tensors");
  const model = Object.freeze({ kind: "papplu-qnet", spec, gameConfig, source });
  WEIGHTS.set(model, weights);
  return model;
}

function parseModelSpec(spec, path) {
  requireKeys(spec, ["width", "blocks"], path);
  if (!Number.isSafeInteger(spec.width) || spec.width < 8) fail(`${path}.width`, "expected an integer of at least 8");
  if (!Number.isSafeInteger(spec.blocks) || spec.blocks < 1) fail(`${path}.blocks`, "expected a positive integer");
  return Object.freeze({ width: spec.width, blocks: spec.blocks });
}

function parseGameConfig(config, path) {
  requireKeys(config, GAME_CONFIG_KEYS, path);
  for (const key of ["numDecks", "cardsInHand", "maxTurns"]) {
    if (!Number.isSafeInteger(config[key]) || config[key] < 1) fail(`${path}.${key}`, "expected a positive integer");
  }
  const { numDecks, cardsInHand, requiredSequences, maxTurns, recycleDiscard } = config;
  if (!Number.isSafeInteger(requiredSequences) || requiredSequences < 0) fail(`${path}.requiredSequences`, "expected a nonnegative integer");
  if (typeof recycleDiscard !== "boolean") fail(`${path}.recycleDiscard`, "expected a boolean");
  if (numDecks * NUM_FACES < cardsInHand + 2) fail(`${path}.numDecks`, "too few cards to deal a hand plus the joker and first discard");
  if (requiredSequences * 3 > cardsInHand) fail(`${path}.requiredSequences`, "needs more cards than a hand holds");
  return Object.freeze({ numDecks, cardsInHand, requiredSequences, maxTurns, recycleDiscard });
}

function parseSource(source, path) {
  requireKeys(source, SOURCE_KEYS, path);
  const { kind, trained, champion, seed, registryId, sha256, tensorsSha256 } = source;
  if (!SOURCE_KINDS.includes(kind)) fail(`${path}.kind`, `expected one of ${SOURCE_KINDS.join(", ")}`);
  const bootstrap = kind === "bootstrap";
  const registry = kind === "registry";
  if (trained !== !bootstrap) fail(`${path}.trained`, `must be ${!bootstrap} for ${kind} weights`);
  if (champion !== registry) fail(`${path}.champion`, `must be ${registry} for ${kind} weights`);
  if (bootstrap ? !Number.isSafeInteger(seed) : seed !== null) {
    fail(`${path}.seed`, bootstrap ? "expected the integer bootstrap seed" : "expected null");
  }
  if (registry ? typeof registryId !== "string" || registryId === "" : registryId !== null) {
    fail(`${path}.registryId`, registry ? "expected the champion's registry id" : "expected null");
  }
  for (const key of ["sha256", "tensorsSha256"]) {
    if (typeof source[key] !== "string" || !SHA256.test(source[key])) fail(`${path}.${key}`, "expected 64 lowercase hex digits");
  }
  return Object.freeze({ kind, trained, champion, seed, registryId, sha256, tensorsSha256 });
}

function parseTensors(tensors, spec, path) {
  const shapes = tensorShapes(spec);
  requireKeys(tensors, shapes.map(([name]) => name), path);
  const weights = {};
  for (const [name, shape] of shapes) {
    const at = `${path}["${name}"]`;
    const entry = tensors[name];
    requireKeys(entry, ["shape", "data"], at);
    if (!Array.isArray(entry.shape) || entry.shape.length !== shape.length || entry.shape.some((n, i) => n !== shape[i])) {
      fail(`${at}.shape`, `expected [${shape.join(", ")}]`);
    }
    weights[name.slice("policy.".length)] = decodeFloat32(entry.data, shape.reduce((a, b) => a * b, 1), `${at}.data`);
  }
  return weights;
}

function decodeFloat32(data, length, path) {
  if (typeof data !== "string" || data.length % 4 !== 0 || !BASE64.test(data)) fail(path, "expected base64 text");
  const bytes = atob(data);
  if (bytes.length !== length * 4) fail(path, `expected ${length * 4} bytes, got ${bytes.length}`);
  const view = new DataView(new ArrayBuffer(bytes.length));
  for (let i = 0; i < bytes.length; i += 1) view.setUint8(i, bytes.charCodeAt(i));
  const values = new Float64Array(length);
  for (let i = 0; i < length; i += 1) {
    values[i] = view.getFloat32(i * 4, true);
    if (!Number.isFinite(values[i])) fail(path, `value ${i} is not finite`);
  }
  return values;
}

function requireKeys(value, keys, path) {
  if (!isRecord(value)) fail(path, "expected an object");
  const present = Object.keys(value);
  for (const key of keys) if (!present.includes(key)) fail(`${path}.${key}`, "missing");
  for (const key of present) if (!keys.includes(key)) fail(`${path}.${key}`, "unexpected field");
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function fail(path, message) {
  throw new TypeError(`${path}: ${message}`);
}

function index(value, limit, path) {
  if (!Number.isInteger(value) || value < 0 || value >= limit) fail(path, `expected an integer from 0 to ${limit - 1}`);
  return value;
}

function requireFaceCounts(value, path) {
  if (!Array.isArray(value) || value.length !== NUM_FACES) fail(path, "expected 52 face counts");
  if (value.some(n => !Number.isSafeInteger(n) || n < 0)) fail(path, "expected nonnegative integer face counts");
}

function nonnegativeInteger(value, path) {
  if (!Number.isSafeInteger(value) || value < 0) fail(path, "expected a nonnegative safe integer");
}

function validateObservation(obs, path) {
  if (!isRecord(obs)) fail(path, "expected an observation");
  if (!isRecord(obs.config)) fail(`${path}.config`, "expected a game config");
  parseGameConfig(obs.config, `${path}.config`);
  requireFaceCounts(obs.hand, `${path}.hand`);
  index(obs.joker, NUM_FACES, `${path}.joker`);
  index(obs.phase, 3, `${path}.phase`);
  nonnegativeInteger(obs.turnsRemaining, `${path}.turnsRemaining`);
  nonnegativeInteger(obs.stockRemaining, `${path}.stockRemaining`);
  if (!Array.isArray(obs.discardPile)) fail(`${path}.discardPile`, "expected an array of faces");
  obs.discardPile.forEach((face, i) => index(face, NUM_FACES, `${path}.discardPile[${i}]`));
  if (!Array.isArray(obs.players) || obs.players.length < 1 || obs.players.length > SEATS) {
    fail(`${path}.players`, "expected 1..6 player views");
  }
  obs.players.forEach((player, seat) => {
    const at = `${path}.players[${seat}]`;
    if (!isRecord(player)) fail(at, "expected a player view");
    nonnegativeInteger(player.handSize, `${at}.handSize`);
    nonnegativeInteger(player.turnsRemaining, `${at}.turnsRemaining`);
    index(player.nextPlayer, obs.players.length, `${at}.nextPlayer`);
    index(player.previousPlayer, obs.players.length, `${at}.previousPlayer`);
    if (typeof player.active !== "boolean") fail(`${at}.active`, "expected a boolean");
  });
  if (!Array.isArray(obs.history) || obs.history.length === 0) fail(`${path}.history`, "expected at least one event");
}

function countsOf(faces) {
  const counts = new Int32Array(NUM_FACES);
  for (const face of faces) counts[face] += 1;
  return counts;
}

function meldMembership(hand, joker) {
  const natural = new Int32Array(52);
  const pure = new Int32Array(52);
  const anyMeld = new Int32Array(52);
  const wilds = [0, 1, 2, 3].map(s => s * 13 + joker % 13);
  function mark(represented, flexible, result) {
    const missing = represented.reduce((sum, face) => sum + Number(!hand[face] || flexible.includes(face)), 0);
    const available = flexible.reduce((sum, face) => sum + hand[face], 0);
    if (missing > available) return;
    for (const face of represented) if (hand[face] && !flexible.includes(face)) result[face] = 1;
    if (available) for (const face of flexible) if (hand[face]) result[face] = 1;
  }
  for (const represented of RUN_FACES) {
    if (represented.every(face => hand[face] > 0)) for (const face of represented) natural[face] = 1;
    mark(represented, [joker], pure);
    mark(represented, wilds, anyMeld);
  }
  for (const represented of SET_FACES) mark(represented, wilds, anyMeld);
  hand.forEach((copies, face) => {
    if (copies >= 3) pure[face] = anyMeld[face] = 1;
  });
  return { natural, pure, anyMeld };
}

export function meldChannels(card, context) {
  const decks = Math.max(roundEven(context[8] * 6), 1);
  const handSize = Math.max(roundEven(context[9] * 30), 1);
  const hand = Array.from({ length: NUM_FACES }, (_, face) => roundEven(card[face * CHANNELS.card.length] * decks));
  let joker = 0;
  for (let face = 0; face < NUM_FACES; face += 1) {
    if (card[face * CHANNELS.card.length + 6]) {
      joker = face;
      break;
    }
  }
  const { natural, pure, anyMeld } = meldMembership(hand, joker);
  let unavoidable = 0;
  const cards = Array.from({ length: NUM_FACES }, (_, face) => {
    const suit = Math.floor(face / 13);
    const rank = face % 13;
    const adjacent = Number(hand[suit * 13 + (rank + 12) % 13] > 0) + Number(hand[suit * 13 + (rank + 1) % 13] > 0);
    const mates = [0, 1, 2, 3].reduce((sum, other) => sum + Number(other !== suit && hand[other * 13 + rank] > 0), 0);
    unavoidable += hand[face] * card[face * CHANNELS.card.length + 8] * 10 * (1 - anyMeld[face]);
    return Float32Array.of(adjacent / 2, mates / 3, natural[face], pure[face], anyMeld[face]);
  });
  return { cards, unavoidableDw: Math.fround(unavoidable / (10 * handSize)) };
}

function roundEven(value) {
  const lower = Math.floor(value);
  return value - lower === 0.5 ? lower + Number(lower % 2 !== 0) : Math.round(value);
}

export function encodeObservation(obs) {
  return encode(obs, "observation");
}

function encode(obs, path) {
  validateObservation(obs, path);
  const { hand, config, players, history, joker, phase, discardPile } = obs;
  const { numDecks: decks, cardsInHand: cards, maxTurns: turns } = config;
  const held = Array.from({ length: players.length }, () => new Int32Array(NUM_FACES));
  const heldTotal = new Int32Array(players.length);
  const takes = new Int32Array(players.length);
  const passes = new Int32Array(players.length);
  const oppDiscarded = new Int32Array(NUM_FACES);
  const nextPassed = new Int32Array(NUM_FACES);
  const nextOpponent = players.length > 1 && players[0].nextPlayer > 0 ? players[0].nextPlayer : -1;
  let refills = 0;
  for (let i = 0; i < history.length; i += 1) {
    const event = history[i];
    const at = `${path}.history[${i}]`;
    if (!isRecord(event)) fail(at, "expected an event");
    index(event.kind, 6, `${at}.kind`);
    index(event.actor, players.length, `${at}.actor`);
    index(event.nextPlayer, players.length, `${at}.nextPlayer`);
    if (event.face !== null || event.kind === TAKE_DISCARD || event.kind === DISCARD) {
      index(event.face, NUM_FACES, `${at}.face`);
    }
    if (event.offeredTop !== null) index(event.offeredTop, NUM_FACES, `${at}.offeredTop`);
    if (i === 0) requireFaceCounts(event.initialHand, `${at}.initialHand`);
    const { actor, face, kind } = event;
    if (kind === TAKE_DISCARD) {
      held[actor][face] += 1;
      heldTotal[actor] += 1;
      takes[actor] += 1;
    } else if (kind === DISCARD) {
      if (held[actor][face] > 0) {
        held[actor][face] -= 1;
        heldTotal[actor] -= 1;
      }
      if (actor > 0) oppDiscarded[face] += 1;
    } else if (kind === DRAW_STOCK && event.offeredTop !== null) {
      passes[actor] += 1;
      if (actor === nextOpponent) nextPassed[event.offeredTop] += 1;
    } else if (kind === REFILL) refills += 1;
  }
  const pileCounts = countsOf(discardPile);
  const top = discardPile.length ? discardPile[discardPile.length - 1] : NUM_FACES;
  const supply = decks * NUM_FACES;
  const values = new Float32Array(FEATURE_SIZE);
  let rawPoints = 0;
  let wildCount = 0;
  hand.forEach((copies, face) => {
    const rank = face % 13;
    const wild = rank === joker % 13;
    const points = wild ? 0 : RANK_POINTS[rank];
    rawPoints += copies * points;
    if (wild) wildCount += copies;
    let opponentHeld = 0;
    for (let seat = 1; seat < players.length; seat += 1) opponentHeld += held[seat][face];
    const nextHeld = nextOpponent > 0 ? held[nextOpponent][face] : 0;
    values.set([
      copies / decks, Number(copies >= 1), Number(copies >= 2), Number(copies >= 3), pileCounts[face] / decks,
      Number(face === top), Number(face === joker), Number(wild), points / 10, oppDiscarded[face] / decks,
      nextHeld / decks, (opponentHeld - nextHeld) / decks, nextPassed[face] / decks,
    ], FIELDS.card.offset + face * CHANNELS.card.length);
  });
  players.forEach((player, s) => {
    values.set([
      1, Number(s === 0), Number(player.active), player.handSize / cards, player.turnsRemaining / turns,
      Number(s > 0 && s === players[0].nextPlayer), takes[s] / turns, passes[s] / turns, heldTotal[s] / cards,
    ], FIELDS.seat.offset + s * CHANNELS.seat.length);
  });
  const last = history[history.length - 1];
  const toAct = phase !== 2 && players[0].active && (
    last.kind === INITIAL ||
    (phase === 1 && [DRAW_STOCK, TAKE_DISCARD].includes(last.kind) && last.actor === 0) ||
    (phase === 0 && [DISCARD, REFILL].includes(last.kind) && last.nextPlayer === 0)
  );
  values.set([
    Number(phase === 0), Number(phase === 1), Number(phase === 2), Number(toAct), obs.turnsRemaining / turns,
    obs.stockRemaining / supply, discardPile.length / supply, Math.min(refills, 3) / 3, decks / 6, cards / 30,
    config.requiredSequences / 10, Number(config.recycleDiscard), players.length / 6, turns / 60,
    Number(players.length === 1 && history[0].initialHand.reduce((a, b) => a + b, 0) === cards + 1),
    wildCount / (4 * decks), rawPoints / 100,
  ], FIELDS.context.offset);
  if (toAct) {
    if (phase === 0) {
      values[FIELDS.legal.offset] = Number(obs.stockRemaining > 0);
      values[FIELDS.legal.offset + 1] = Number(discardPile.length > 0);
    } else {
      hand.forEach((n, face) => { values[FIELDS.legal.offset + 2 + face] = Number(n > 0); });
    }
  }
  values[FIELDS.top.offset] = top;
  if (!values.every(Number.isFinite)) fail(path, "features are not finite; check the counts and config");
  return values;
}

export function qValues(model, observations) {
  const weights = WEIGHTS.get(model);
  if (weights === undefined) throw new TypeError("qValues: model must come from parseQModel");
  if (!Array.isArray(observations)) throw new TypeError("qValues: observations must be an array");
  return observations.map((obs, i) => forward(weights, model.spec, encode(obs, `observations[${i}]`)));
}

function forward(w, spec, features) {
  const { width, blocks } = spec;
  const field = name => features.subarray(FIELDS[name].offset, FIELDS[name].offset + FIELDS[name].size);
  const card = field("card");
  const extra = meldChannels(card, field("context"));
  let x = Array.from({ length: NUM_FACES }, (_, face) =>
    linear(w, "card_embed", [...row(card, face, CHANNELS.card.length), ...extra.cards[face]]));
  let table = linear(w, "table_embed", [...field("seat"), ...field("context"), extra.unavoidableDw]);
  for (let block = 0; block < blocks; block += 1) {
    const prefix = `blocks.${block}.`;
    const normalized = x.map(card => layerNorm(w, `${prefix}card_norm`, card));
    const normalizedTable = layerNorm(w, `${prefix}table_norm`, table);
    const sequence = [];
    const suitContext = [];
    for (let suit = 0; suit < 4; suit += 1) {
      const track = normalized.slice(suit * 13, (suit + 1) * 13);
      const projected = [...track, track[0]].map(row => linear(w, `${prefix}w_seq`, row));
      const seq14 = projected.map((row, position) => Float64Array.from({ length: width }, (_, d) =>
        row[d] + (position > 0 ? projected[position - 1][width + d] : 0) +
        (position < 13 ? projected[position + 1][width * 2 + d] : 0)));
      sequence.push(Float64Array.from(seq14[0], (v, d) => (v + seq14[13][d]) * 0.5), ...seq14.slice(1, 13));
      suitContext.push(linear(w, `${prefix}w_suit_run`, Float64Array.from({ length: width }, (_, d) =>
        track.reduce((sum, row) => sum + row[d], 0) / 13)));
    }
    const rankContext = suitMeans(normalized, width).map(row => linear(w, `${prefix}w_rank_set`, row));
    const tableContext = linear(w, `${prefix}w_table`, normalizedTable);
    x = x.map((card, face) => add(card, linear(w, `${prefix}card_out`, silu(add(
      sequence[face], rankContext[face % 13], suitContext[Math.floor(face / 13)], tableContext,
    )))));
    const pooled = x.map(card => layerNorm(w, `${prefix}card_norm`, card));
    const mean = Float64Array.from({ length: width }, (_, d) => pooled.reduce((sum, row) => sum + row[d], 0) / NUM_FACES);
    const maximum = Float64Array.from({ length: width }, (_, d) => Math.max(...pooled.map(row => row[d])));
    table = add(table, linear(w, `${prefix}table_fc2`,
      silu(linear(w, `${prefix}table_fc1`, [...normalizedTable, ...mean, ...maximum]))));
  }
  x = x.map(card => layerNorm(w, "card_final", card));
  table = layerNorm(w, "table_final", table);
  const sets = suitMeans(x, width);
  const setHidden = sets.map(rank => linear(w, "face_set", rank));
  const tableHidden = linear(w, "face_table", table);
  const faceValues = x.map((card, face) =>
    linear(w, "face_out", silu(add(linear(w, "face_local", card), setHidden[face % 13], tableHidden)))[0]);
  const top = field("top")[0];
  const zero = new Float64Array(width);
  const hasTop = top >= 0 && top < NUM_FACES;
  const draw = silu(add(
    linear(w, "draw_table", table), linear(w, "draw_top", hasTop ? x[top] : zero),
    linear(w, "draw_set", hasTop ? sets[top % 13] : zero),
  ));
  const advantage = [...linear(w, "draw_out", draw), ...faceValues];
  const legal = field("legal");
  let mean = 0;
  let count = 0;
  for (let action = 0; action < NUM_ACTIONS; action += 1) {
    mean += advantage[action] * legal[action];
    count += legal[action];
  }
  mean /= Math.max(count, 1);
  const value = linear(w, "value_out", silu(linear(w, "value_hidden", table)))[0];
  return Float32Array.from(advantage, a => value + a - mean);
}

function linear(w, name, input) {
  const weight = w[`${name}.weight`];
  const bias = w[`${name}.bias`];
  const cols = input.length;
  const out = new Float64Array(weight.length / cols);
  for (let r = 0; r < out.length; r += 1) {
    let sum = bias === undefined ? 0 : bias[r];
    for (let c = 0; c < cols; c += 1) sum += weight[r * cols + c] * input[c];
    out[r] = sum;
  }
  return out;
}

function layerNorm(w, name, input) {
  let mean = 0;
  for (const value of input) mean += value;
  mean /= input.length;
  let variance = 0;
  for (const value of input) variance += (value - mean) ** 2;
  const scale = 1 / Math.sqrt(variance / input.length + 1e-5);
  return Float64Array.from(input, (v, i) => (v - mean) * scale * w[`${name}.weight`][i] + w[`${name}.bias`][i]);
}

function suitMeans(cards, width) {
  return Array.from({ length: 13 }, (_, rank) => Float64Array.from({ length: width }, (_, d) =>
    (cards[rank][d] + cards[13 + rank][d] + cards[26 + rank][d] + cards[39 + rank][d]) / 4));
}

function add(...vectors) {
  return Float64Array.from(vectors[0], (_, i) => vectors.reduce((sum, v) => sum + v[i], 0));
}

function silu(values) {
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    values[i] = value >= 0 ? value / (1 + Math.exp(-value)) : value * Math.exp(value) / (1 + Math.exp(value));
  }
  return values;
}

function row(values, n, width) {
  return values.subarray(n * width, (n + 1) * width);
}
