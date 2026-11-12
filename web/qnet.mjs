import { pureSequenceUpperBound } from "./evaluate.mjs";

/**
 * @typedef {Readonly<{width: number, heads: number, blocks: number}>} ModelSpec
 * @typedef {Readonly<{
 *   numDecks: number, cardsInHand: number, requiredSequences: number,
 *   maxTurns: number, recycleDiscard: boolean
 * }>} ModelGameConfig
 * @typedef {Readonly<{
 *   kind: "bootstrap"|"checkpoint"|"registry", trained: boolean, champion: boolean,
 *   seed: number|null, registryId: string|null, sha256: string, tensorsSha256: string
 * }>} ModelSource
 * @typedef {Readonly<{
 *   kind: "papplu-qnet", spec: ModelSpec, gameConfig: ModelGameConfig, source: ModelSource
 * }>} QModel
 * @typedef {object} Observation
 * @property {number[]} hand
 * @property {number[]} discardPile
 * @property {number} joker
 * @property {0|1|2} phase
 * @property {number} turnsRemaining
 * @property {number} stockRemaining
 * @property {ModelGameConfig} config
 * @property {{handSize: number, turnsRemaining: number, nextPlayer: number, previousPlayer: number, active: boolean}[]} players
 * @property {{kind: number, actor: number, handSize: number, turnsRemaining: number, stockRemaining: number, nextPlayer: number, face: number|null, offeredTop: number|null, initialHand: number[]}[]} history
 */

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
    "own", "own_ge1", "own_ge2", "own_ge3", "initial", "pile", "unseen", "top", "stock",
    "indicator", "wild", "points", "adjacent", "set_mates", "natural_run_member", "pure_meld_member", "any_meld_member",
  ]),
  history: Object.freeze(["took", "held", "discarded", "passed", "last_take", "last_discard", "last_pass"]),
  seat: Object.freeze(["present", "observer", "active", "hand_size", "turns", "next", "previous", "take_rate", "takes", "passes", "held_total"]),
  context: Object.freeze([
    "draw", "discard", "terminal", "to_act", "turns", "stock", "pile", "refills", "decks", "hand",
    "sequences", "recycle", "players", "max_turns", "warm_start", "wilds", "deadwood", "pure_quota_deficit", "unavoidable_deadwood",
  ]),
});
export const FEATURE_LAYOUT = Object.freeze([
  ["card", [52, 17]], ["history", [52, 6, 7]], ["seat", [6, 11]],
  ["context", [19]], ["legal", [54]], ["top", [1]],
].map(([name, shape]) => Object.freeze([name, Object.freeze(shape)])));
export const FEATURE_SIZE = 3208;
const FIELDS = {};
let offset = 0;
for (const [name, shape] of FEATURE_LAYOUT) {
  const size = shape.reduce((a, b) => a * b, 1);
  FIELDS[name] = { offset, size };
  offset += size;
}

const HEADER = Object.freeze({
  format: "papplu-qnet", formatVersion: 2, checkpointVersion: 4,
  encodingVersion: 2, architecture: "axial_table", numActions: NUM_ACTIONS,
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
  layer("card_embed", width, 59);
  layer("table_embed", width, 85);
  for (let b = 0; b < blocks; b += 1) {
    const prefix = `blocks.${b}.`;
    layer(`${prefix}rank_norm`, width);
    shapes.push([`policy.${prefix}rank_conv.weight`, [width, 1, 3]], [`policy.${prefix}rank_conv.bias`, [width]]);
    for (const name of ["rank_in", "rank_out"]) layer(prefix + name, width, width);
    layer(`${prefix}suit_norm`, width);
    for (const name of ["suit_in", "suit_out"]) layer(prefix + name, width, width);
    layer(`${prefix}card_norm`, width);
    layer(`${prefix}table_norm`, width);
    layer(`${prefix}table_query`, width, width);
    layer(`${prefix}table_in`, width, width * 2);
    layer(`${prefix}table_out`, width, width);
    layer(`${prefix}film`, width * 2, width);
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

/**
 * @param {unknown} json A v4 checkpoint export in the papplu-qnet format.
 * @returns {QModel} A frozen handle with private policy weights.
 */
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
  requireKeys(spec, ["width", "heads", "blocks"], path);
  if (!Number.isSafeInteger(spec.width) || spec.width < 8) fail(`${path}.width`, "expected an integer of at least 8");
  if (!Number.isSafeInteger(spec.heads) || spec.heads < 1 || spec.width % spec.heads !== 0) {
    fail(`${path}.heads`, "expected a positive integer dividing width");
  }
  if (!Number.isSafeInteger(spec.blocks) || spec.blocks < 1) fail(`${path}.blocks`, "expected a positive integer");
  return Object.freeze({ width: spec.width, heads: spec.heads, blocks: spec.blocks });
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
  obs.history.forEach((event, i) => {
    const at = `${path}.history[${i}]`;
    if (!isRecord(event)) fail(at, "expected an event");
    index(event.kind, 6, `${at}.kind`);
    index(event.actor, obs.players.length, `${at}.actor`);
    index(event.nextPlayer, obs.players.length, `${at}.nextPlayer`);
    if (event.face !== null || event.kind === TAKE_DISCARD || event.kind === DISCARD) {
      index(event.face, NUM_FACES, `${at}.face`);
    }
    if (event.offeredTop !== null) index(event.offeredTop, NUM_FACES, `${at}.offeredTop`);
  });
  requireFaceCounts(obs.history[0].initialHand, `${path}.history[0].initialHand`);
}

function countsOf(faces) {
  const counts = new Int32Array(NUM_FACES);
  for (const face of faces) counts[face] += 1;
  return counts;
}

function tableBelief(obs) {
  const seats = Array.from({ length: obs.players.length }, () => ({
    took: new Int32Array(52), held: new Int32Array(52), discarded: new Int32Array(52), passed: new Int32Array(52),
    last_take: new Int32Array(52), last_discard: new Int32Array(52), last_pass: new Int32Array(52), takes: 0, passes: 0,
  }));
  let pile = [];
  let refillPool = null;
  let refills = 0;
  obs.history.forEach((event, i) => {
    const seat = seats[event.actor];
    if (event.kind === INITIAL) pile = event.offeredTop === null ? [] : [event.offeredTop];
    else if (event.kind === TAKE_DISCARD) {
      pile.pop();
      seat.took[event.face] += 1;
      seat.held[event.face] += 1;
      seat.last_take[event.face] = i + 1;
      seat.takes += 1;
    } else if (event.kind === DISCARD) {
      pile.push(event.face);
      seat.held[event.face] = Math.max(0, seat.held[event.face] - 1);
      seat.discarded[event.face] += 1;
      seat.last_discard[event.face] = i + 1;
    } else if (event.kind === DRAW_STOCK) {
      if (refillPool !== null && event.face !== null) refillPool[event.face] = Math.max(0, refillPool[event.face] - 1);
      if (event.offeredTop !== null) {
        seat.passed[event.offeredTop] += 1;
        seat.last_pass[event.offeredTop] = i + 1;
        seat.passes += 1;
      }
    } else if (event.kind === REFILL) {
      refillPool = countsOf(pile.slice(0, -1));
      pile = pile.slice(-1);
      refills += 1;
    }
  });
  const pileCounts = countsOf(obs.discardPile);
  const unseen = Int32Array.from(obs.hand, (copies, face) => Math.max(
    0, obs.config.numDecks - copies - pileCounts[face] - Number(face === obs.joker) -
    seats.slice(1).reduce((sum, seat) => sum + seat.held[face], 0),
  ));
  let stockWeights = refillPool === null ? unseen : Int32Array.from(unseen, (n, face) => Math.min(n, refillPool[face]));
  if (stockWeights.every(n => n === 0)) stockWeights = unseen;
  return { seats, refills, pileCounts, unseen, stockWeights };
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

/**
 * @param {Observation} obs An observer-relative table and its public event history.
 * @returns {Float32Array} The 3,208 v2 input features.
 */
export function encodeObservation(obs) {
  return encode(obs, "observation");
}

function encode(obs, path) {
  validateObservation(obs, path);
  const { hand, config, players, history, joker, phase, discardPile } = obs;
  const { numDecks: decks, cardsInHand: cards, maxTurns: turns } = config;
  const belief = tableBelief(obs);
  const { natural, pure, anyMeld } = meldMembership(hand, joker);
  const top = discardPile.length ? discardPile[discardPile.length - 1] : NUM_FACES;
  const supply = decks * NUM_FACES;
  const stockTotal = belief.stockWeights.reduce((a, b) => a + b, 0);
  const points = hand.map((_, face) => face % 13 === joker % 13 ? 0 : RANK_POINTS[face % 13]);
  const values = new Float32Array(FEATURE_SIZE);
  hand.forEach((copies, face) => {
    const suit = Math.floor(face / 13);
    const rank = face % 13;
    const neighbors = rank === 0 ? [1, 12] : rank === 12 ? [11, 0] : [rank - 1, rank + 1];
    const adjacent = neighbors.reduce((sum, r) => sum + Number(hand[suit * 13 + r] > 0), 0);
    const mates = [0, 1, 2, 3].reduce((sum, s) => sum + Number(s !== suit && hand[s * 13 + rank] > 0), 0);
    values.set([
      copies / decks, Number(copies >= 1), Number(copies >= 2), Number(copies >= 3), history[0].initialHand[face] / decks,
      belief.pileCounts[face] / decks, belief.unseen[face] / decks, Number(face === top),
      stockTotal ? belief.stockWeights[face] / stockTotal : 0,
      Number(face === joker), Number(rank === joker % 13), points[face] / 10, adjacent / 2, mates / 3,
      natural[face], pure[face], anyMeld[face],
    ], FIELDS.card.offset + face * 17);
    belief.seats.forEach((seat, s) => {
      values.set([
        seat.took[face] / decks, seat.held[face] / decks, seat.discarded[face] / decks, seat.passed[face] / decks,
        seat.last_take[face] / history.length, seat.last_discard[face] / history.length, seat.last_pass[face] / history.length,
      ], FIELDS.history.offset + (face * SEATS + s) * 7);
    });
  });
  players.forEach((player, s) => {
    const seat = belief.seats[s];
    values.set([
      1, Number(s === 0), Number(player.active), player.handSize / cards, player.turnsRemaining / turns,
      Number(s > 0 && s === players[0].nextPlayer), Number(s > 0 && s === players[0].previousPlayer),
      (1 + seat.takes) / (5 + seat.takes + seat.passes), seat.takes / turns, seat.passes / turns,
      seat.held.reduce((a, b) => a + b, 0) / cards,
    ], FIELDS.seat.offset + s * 11);
  });
  const last = history[history.length - 1];
  const toAct = phase !== 2 && players[0].active && (
    last.kind === INITIAL ||
    (phase === 1 && [DRAW_STOCK, TAKE_DISCARD].includes(last.kind) && last.actor === 0) ||
    (phase === 0 && [DISCARD, REFILL].includes(last.kind) && last.nextPlayer === 0)
  );
  const pureUpper = pureSequenceUpperBound(hand, joker);
  values.set([
    Number(phase === 0), Number(phase === 1), Number(phase === 2), Number(toAct), obs.turnsRemaining / turns,
    obs.stockRemaining / supply, discardPile.length / supply, Math.min(belief.refills, 3) / 3, decks / 6, cards / 30,
    config.requiredSequences / 10, Number(config.recycleDiscard), players.length / 6, turns / 60,
    Number(players.length === 1 && history[0].initialHand.reduce((a, b) => a + b, 0) === cards + 1),
    [0, 1, 2, 3].reduce((sum, suit) => sum + hand[suit * 13 + joker % 13], 0) / (4 * decks),
    hand.reduce((sum, n, face) => sum + n * points[face], 0) / 100,
    Math.max(0, config.requiredSequences - pureUpper) / Math.max(1, config.requiredSequences),
    hand.reduce((sum, n, face) => sum + (anyMeld[face] ? 0 : n * points[face]), 0) / (10 * cards),
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

/**
 * @param {QModel} model The handle returned by parseQModel.
 * @param {Observation[]} observations The observation batch.
 * @returns {Float32Array[]} One row of 54 unmasked Q values per observation.
 */
export function qValues(model, observations) {
  const weights = WEIGHTS.get(model);
  if (weights === undefined) throw new TypeError("qValues: model must come from parseQModel");
  if (!Array.isArray(observations)) throw new TypeError("qValues: observations must be an array");
  return observations.map((obs, i) => forward(weights, model.spec, encode(obs, `observations[${i}]`)));
}

function forward(w, spec, features) {
  const { width, heads, blocks } = spec;
  const field = name => features.subarray(FIELDS[name].offset, FIELDS[name].offset + FIELDS[name].size);
  const card = field("card");
  const history = field("history");
  let x = Array.from({ length: NUM_FACES }, (_, face) =>
    linear(w, "card_embed", [...row(card, face, 17), ...row(history, face, 42)]));
  let table = linear(w, "table_embed", [...field("context"), ...field("seat")]);
  for (let block = 0; block < blocks; block += 1) {
    const prefix = `blocks.${block}.`;
    const normalized = x.map(card => layerNorm(w, `${prefix}rank_norm`, card));
    const rankRows = [];
    for (let suit = 0; suit < 4; suit += 1) {
      const track = normalized.slice(suit * 13, (suit + 1) * 13);
      track.push(track[0]);
      const convolved = track.map((_, position) => {
        const out = new Float64Array(width);
        for (let c = 0; c < width; c += 1) {
          let sum = w[`${prefix}rank_conv.bias`][c];
          for (let k = 0; k < 3; k += 1) {
            const at = position + k - 1;
            if (at >= 0 && at < 14) sum += w[`${prefix}rank_conv.weight`][c * 3 + k] * track[at][c];
          }
          out[c] = sum;
        }
        return out;
      });
      rankRows.push(Float64Array.from(convolved[0], (v, i) => (v + convolved[13][i]) * 0.5), ...convolved.slice(1, 13));
    }
    x = x.map((card, face) => add(card, linear(w, `${prefix}rank_out`, relu(linear(w, `${prefix}rank_in`, rankRows[face])))));
    const means = suitMeans(x.map(card => layerNorm(w, `${prefix}suit_norm`, card)), width);
    const sets = means.map(rank => linear(w, `${prefix}suit_out`, relu(linear(w, `${prefix}suit_in`, rank))));
    x = x.map((card, face) => add(card, sets[face % 13]));
    const keys = x.map(card => layerNorm(w, `${prefix}card_norm`, card));
    const normalizedTable = layerNorm(w, `${prefix}table_norm`, table);
    const query = linear(w, `${prefix}table_query`, normalizedTable);
    const pooled = new Float64Array(width);
    const headSize = width / heads;
    for (let start = 0; start < width; start += headSize) {
      const scores = Float64Array.from(keys, key => {
        let score = 0;
        for (let d = start; d < start + headSize; d += 1) score += query[d] * key[d];
        return score / Math.sqrt(headSize);
      });
      const peak = Math.max(...scores);
      let total = 0;
      for (let face = 0; face < NUM_FACES; face += 1) {
        scores[face] = Math.exp(scores[face] - peak);
        total += scores[face];
      }
      for (let d = start; d < start + headSize; d += 1) {
        for (let face = 0; face < NUM_FACES; face += 1) pooled[d] += scores[face] * keys[face][d];
        pooled[d] /= total;
      }
    }
    table = add(table, linear(w, `${prefix}table_out`, relu(linear(w, `${prefix}table_in`, [...normalizedTable, ...pooled]))));
    const film = linear(w, `${prefix}film`, table);
    x = x.map((card, face) => Float64Array.from(card, (v, d) => v + keys[face][d] * film[d] + film[width + d]));
  }
  x = x.map(card => layerNorm(w, "card_final", card));
  table = layerNorm(w, "table_final", table);
  const sets = suitMeans(x, width);
  const setHidden = sets.map(rank => linear(w, "face_set", rank));
  const tableHidden = linear(w, "face_table", table);
  const faceValues = x.map((card, face) =>
    linear(w, "face_out", relu(add(linear(w, "face_local", card), setHidden[face % 13], tableHidden)))[0]);
  const top = field("top")[0];
  const zero = new Float64Array(width);
  const draw = relu(add(
    linear(w, "draw_table", table), linear(w, "draw_top", top < NUM_FACES ? x[top] : zero),
    linear(w, "draw_set", top < NUM_FACES ? sets[top % 13] : zero),
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
  const value = linear(w, "value_out", relu(linear(w, "value_hidden", table)))[0];
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

function relu(values) {
  for (let i = 0; i < values.length; i += 1) if (values[i] < 0) values[i] = 0;
  return values;
}

function row(values, n, width) {
  return values.subarray(n * width, (n + 1) * width);
}
