const NUM_FACES = 52;
const NUM_ACTIONS = 54;
const NO_FACE = NUM_FACES;
const WIDTH = 64;
const HEADS = 4;
const HEAD_SIZE = WIDTH / HEADS;
const STEPS = 14;
const CARD_FEATURES = 5;
const CONTEXT_FEATURES = 11;
const PLAYER_FEATURES = 6;
const EVENT_FEATURES = 8;
const EVENT_KINDS = 6;
const DRAW_STOCK = 1;
const TAKE_DISCARD = 2;
const DISCARD = 3;

const HEADER = Object.freeze({
  format: "papplu-qnet",
  formatVersion: 1,
  checkpointVersion: 3,
  encodingVersion: 2,
  architecture: "dynamic_table",
  numActions: NUM_ACTIONS,
});
const MODEL_KEYS = [...Object.keys(HEADER), "modelSpec", "gameConfig", "source", "tensors"];
const GAME_CONFIG_KEYS = ["numDecks", "cardsInHand", "requiredSequences", "maxTurns", "recycleDiscard"];
const SOURCE_KEYS = ["kind", "trained", "champion", "seed", "registryId", "sha256", "tensorsSha256"];
const SOURCE_KINDS = ["bootstrap", "checkpoint", "registry"];
const SHA256 = /^[0-9a-f]{64}$/;
const BASE64 = /^[A-Za-z0-9+/]*={0,2}$/;

const TENSOR_SHAPES = Object.freeze({
  "suit_conv1.weight": [64, 5, 3], "suit_conv1.bias": [64],
  "suit_conv2.weight": [64, 64, 3], "suit_conv2.bias": [64],
  "card_attention.in_proj_weight": [192, 64], "card_attention.in_proj_bias": [192],
  "card_attention.out_proj.weight": [64, 64], "card_attention.out_proj.bias": [64],
  "card_norm.weight": [64], "card_norm.bias": [64],
  "history_proj.weight": [1, 5], "history_proj.bias": [1],
  "context_features.weight": [64, 11], "context_features.bias": [64],
  "player_features.weight": [64, 6], "player_features.bias": [64],
  "event_summary.weight": [64, 8], "event_summary.bias": [64],
  "turn_relations.weight": [64, 192], "turn_relations.bias": [64],
  "context_gate.weight": [2, 64], "context_gate.bias": [2],
  "draw_head.0.weight": [64, 197], "draw_head.0.bias": [64],
  "draw_head.2.weight": [2, 64], "draw_head.2.bias": [2],
  "score.0.weight": [64, 210], "score.0.bias": [64],
  "score.2.weight": [1, 64], "score.2.bias": [1],
});

const WEIGHTS = new WeakMap();

/**
 * @typedef {{ numDecks: number, cardsInHand: number, requiredSequences: number, maxTurns: number, recycleDiscard: boolean }} ModelGameConfig
 *
 * Bootstrap weights are seeded and untrained. A registry export is the champion it names.
 * @typedef {{
 *   kind: "bootstrap" | "checkpoint" | "registry", trained: boolean, champion: boolean,
 *   seed: number | null, registryId: string | null, sha256: string, tensorsSha256: string,
 * }} ModelSource
 *
 * @typedef {{ readonly kind: "papplu-qnet", readonly gameConfig: ModelGameConfig, readonly source: ModelSource }} QModel
 *
 * The Python Observation with camelCase fields. Seats are relative to the observer.
 * @typedef {{
 *   hand: number[], discardPile: number[], joker: number, phase: 0 | 1 | 2,
 *   turnsRemaining: number, stockRemaining: number, config: ModelGameConfig,
 *   players: Array<{ handSize: number, turnsRemaining: number, nextPlayer: number, previousPlayer: number, active: boolean }>,
 *   history: Array<{ kind: 0 | 1 | 2 | 3 | 4 | 5, actor: number, handSize: number, turnsRemaining: number,
 *                    stockRemaining: number, nextPlayer: number, face: number | null,
 *                    offeredTop: number | null, initialHand: number[] }>,
 * }} Observation
 */

/**
 * Validates a parsed web/papplu-model.json and throws a TypeError naming the first bad field.
 * @param {unknown} json
 * @returns {QModel}
 */
export function parseQModel(json) {
  requireKeys(json, MODEL_KEYS, "model");
  for (const [key, expected] of Object.entries(HEADER)) {
    if (json[key] !== expected) fail(`model.${key}`, `expected ${JSON.stringify(expected)}`);
  }
  requireKeys(json.modelSpec, ["width", "heads"], "model.modelSpec");
  if (json.modelSpec.width !== WIDTH || json.modelSpec.heads !== HEADS) {
    fail("model.modelSpec", `expected width ${WIDTH} and heads ${HEADS}`);
  }
  const gameConfig = parseGameConfig(json.gameConfig, "model.gameConfig");
  const source = parseSource(json.source, "model.source");
  const weights = parseTensors(json.tensors, "model.tensors");
  const model = Object.freeze({ kind: "papplu-qnet", gameConfig, source });
  WEIGHTS.set(model, weights);
  return model;
}

function parseGameConfig(config, path) {
  requireKeys(config, GAME_CONFIG_KEYS, path);
  for (const key of ["numDecks", "cardsInHand", "maxTurns"]) {
    if (!Number.isSafeInteger(config[key]) || config[key] < 1) fail(`${path}.${key}`, "expected a positive integer");
  }
  const { numDecks, cardsInHand, requiredSequences, maxTurns, recycleDiscard } = config;
  if (!Number.isSafeInteger(requiredSequences) || requiredSequences < 0) {
    fail(`${path}.requiredSequences`, "expected a nonnegative integer");
  }
  if (typeof recycleDiscard !== "boolean") fail(`${path}.recycleDiscard`, "expected a boolean");
  if (numDecks * NUM_FACES < cardsInHand + 2) {
    fail(`${path}.numDecks`, "too few cards to deal a hand plus the joker and first discard");
  }
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

function parseTensors(tensors, path) {
  requireKeys(tensors, Object.keys(TENSOR_SHAPES), path);
  const weights = {};
  for (const [name, shape] of Object.entries(TENSOR_SHAPES)) {
    const at = `${path}["${name}"]`;
    const entry = tensors[name];
    requireKeys(entry, ["shape", "data"], at);
    if (!Array.isArray(entry.shape) || entry.shape.length !== shape.length || entry.shape.some((n, i) => n !== shape[i])) {
      fail(`${at}.shape`, `expected [${shape.join(", ")}]`);
    }
    weights[name] = decodeFloat32(entry.data, shape.reduce((a, b) => a * b, 1), `${at}.data`);
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

/**
 * src/model/encoding.py for one observation, as flat row-major arrays: cards [52, 5],
 * initialHand [52], context [11], players [n, 6], events [m, 8]. A missing face encodes as 52.
 * @param {Observation} obs
 */
export function encodeObservation(obs) {
  return encode(obs, "observation");
}

/**
 * QNetwork._history_channels as a [52, 5] row-major array. The channels are own events, next
 * seat takes, next seat discards, other seat takes, and other seat discards. An opponent's stock
 * draw counts as passing on the card it left on the pile. Event i of m weighs (0.5 + 0.5 i / m) / m.
 * @param {Observation} obs
 */
export function historyChannels(obs) {
  return channelsOf(encode(obs, "observation"));
}

function encode(obs, path) {
  if (!isRecord(obs)) fail(path, "expected an observation");
  const { hand, discardPile, joker, phase, config, players, history } = obs;
  if (!isRecord(config)) fail(`${path}.config`, "expected a game config");
  requireFaceCounts(hand, `${path}.hand`);
  if (!Array.isArray(discardPile)) fail(`${path}.discardPile`, "expected an array of faces");
  if (!Array.isArray(players) || players.length === 0) fail(`${path}.players`, "expected at least one player view");
  if (!Array.isArray(history) || history.length === 0) fail(`${path}.history`, "expected at least one event");
  if (!isRecord(history[0])) fail(`${path}.history[0]`, "expected an event");
  requireFaceCounts(history[0].initialHand, `${path}.history[0].initialHand`);

  const decks = config.numDecks;
  const supply = decks * NUM_FACES;
  const count = players.length;
  const sin = new Float64Array(count);
  const cos = new Float64Array(count);
  for (let seat = 0; seat < count; seat += 1) {
    sin[seat] = Math.sin(2 * Math.PI * seat / count);
    cos[seat] = Math.cos(2 * Math.PI * seat / count);
  }

  const cards = new Float64Array(NUM_FACES * CARD_FEATURES);
  for (let face = 0; face < NUM_FACES; face += 1) cards[face * CARD_FEATURES] = hand[face] / decks;
  discardPile.forEach((face, i) => {
    cards[faceIndex(face, `${path}.discardPile[${i}]`) * CARD_FEATURES + 1] += 1 / decks;
  });
  const top = discardPile.length > 0 ? discardPile[discardPile.length - 1] : NO_FACE;
  if (top !== NO_FACE) cards[top * CARD_FEATURES + 2] = 1;
  const wild = faceIndex(joker, `${path}.joker`);
  cards[wild * CARD_FEATURES + 3] = 1;
  for (let suit = 0; suit < 4; suit += 1) cards[(suit * 13 + (wild % 13)) * CARD_FEATURES + 4] = 1;

  const context = new Float64Array(CONTEXT_FEATURES);
  context[index(phase, 3, `${path}.phase`)] = 1;
  context.set([
    obs.turnsRemaining / config.maxTurns, obs.stockRemaining / supply, decks / 6, config.cardsInHand / 30,
    config.requiredSequences / 10, Number(config.recycleDiscard), count / 6, config.maxTurns / 60,
  ], 3);

  const playerRows = new Float64Array(count * PLAYER_FEATURES);
  const nextPlayer = new Int32Array(count);
  const previousPlayer = new Int32Array(count);
  players.forEach((player, seat) => {
    const at = `${path}.players[${seat}]`;
    if (!isRecord(player)) fail(at, "expected a player view");
    nextPlayer[seat] = index(player.nextPlayer, count, `${at}.nextPlayer`);
    previousPlayer[seat] = index(player.previousPlayer, count, `${at}.previousPlayer`);
    playerRows.set([
      Number(seat === 0), player.handSize / config.cardsInHand, player.turnsRemaining / config.maxTurns,
      Number(player.active), sin[seat], cos[seat],
    ], seat * PLAYER_FEATURES);
  });

  const length = history.length;
  const eventKind = new Int32Array(length);
  const eventFace = new Int32Array(length);
  const offeredTop = new Int32Array(length);
  const eventActor = new Int32Array(length);
  const events = new Float64Array(length * EVENT_FEATURES);
  history.forEach((event, i) => {
    const at = `${path}.history[${i}]`;
    if (!isRecord(event)) fail(at, "expected an event");
    const actor = index(event.actor, count, `${at}.actor`);
    const next = index(event.nextPlayer, count, `${at}.nextPlayer`);
    eventKind[i] = index(event.kind, EVENT_KINDS, `${at}.kind`);
    eventFace[i] = event.face === null ? NO_FACE : faceIndex(event.face, `${at}.face`);
    offeredTop[i] = event.offeredTop === null ? NO_FACE : faceIndex(event.offeredTop, `${at}.offeredTop`);
    eventActor[i] = actor;
    events.set([
      Number(actor === 0), sin[actor], cos[actor], event.handSize / config.cardsInHand,
      event.turnsRemaining / config.maxTurns, event.stockRemaining / supply, sin[next], cos[next],
    ], i * EVENT_FEATURES);
  });

  const encoded = {
    cards,
    initialHand: Float64Array.from(history[0].initialHand, (n) => n / decks),
    top,
    context,
    players: playerRows,
    nextPlayer,
    previousPlayer,
    eventKind,
    eventFace,
    offeredTop,
    eventActor,
    events,
  };
  for (const key of ["cards", "initialHand", "context", "players", "events"]) {
    if (!encoded[key].every(Number.isFinite)) fail(path, `${key} features are not finite; check the counts and config`);
  }
  return encoded;
}

function channelsOf(encoded) {
  const channels = new Float64Array((NUM_FACES + 1) * CARD_FEATURES);
  for (let face = 0; face < NUM_FACES; face += 1) {
    channels[face * CARD_FEATURES] = encoded.initialHand[face] - encoded.cards[face * CARD_FEATURES];
  }
  const length = encoded.eventKind.length;
  for (let i = 0; i < length; i += 1) {
    const recency = (0.5 + 0.5 * ((i + 1) / length)) / length;
    const actor = encoded.eventActor[i];
    const kind = encoded.eventKind[i];
    if (actor === 0) {
      channels[encoded.eventFace[i] * CARD_FEATURES] += recency;
      continue;
    }
    const take = actor === 1 ? 1 : 3;
    if (kind === TAKE_DISCARD) channels[encoded.eventFace[i] * CARD_FEATURES + take] += recency;
    else if (kind === DISCARD) channels[encoded.eventFace[i] * CARD_FEATURES + take + 1] += recency;
    else if (kind === DRAW_STOCK) channels[encoded.offeredTop[i] * CARD_FEATURES + take + 1] += recency;
  }
  return channels.slice(0, NUM_FACES * CARD_FEATURES);
}

function requireFaceCounts(value, path) {
  if (!Array.isArray(value) || value.length !== NUM_FACES) fail(path, `expected ${NUM_FACES} face counts`);
}

function faceIndex(value, path) {
  return index(value, NUM_FACES, path);
}

function index(value, limit, path) {
  if (!Number.isInteger(value) || value < 0 || value >= limit) fail(path, `expected an integer from 0 to ${limit - 1}`);
  return value;
}

/**
 * QNetwork.forward over a batch. Each result holds 54 Q values: 0 draws from the stock,
 * 1 takes the discard, and 2 + face discards that face.
 * @param {QModel} model
 * @param {Observation[]} observations
 * @returns {Float32Array[]}
 */
export function qValues(model, observations) {
  const weights = WEIGHTS.get(model);
  if (weights === undefined) throw new TypeError("qValues: model must come from parseQModel");
  if (!Array.isArray(observations)) throw new TypeError("qValues: observations must be an array");
  return observations.map((obs, i) => {
    const encoded = encode(obs, `observations[${i}]`);
    return forward(weights, encoded, channelsOf(encoded));
  });
}

function forward(w, encoded, channels) {
  const { cards } = encoded;
  const tokens = new Float64Array(NUM_FACES * WIDTH);
  const suitInput = new Float64Array(CARD_FEATURES * STEPS);
  const conv1 = new Float64Array(WIDTH * STEPS);
  const conv2 = new Float64Array(WIDTH * STEPS);
  for (let suit = 0; suit < 4; suit += 1) {
    for (let t = 0; t < STEPS; t += 1) {
      for (let c = 0; c < CARD_FEATURES; c += 1) {
        suitInput[c * STEPS + t] = cards[(suit * 13 + (t % 13)) * CARD_FEATURES + c];
      }
    }
    convRelu(conv1, w["suit_conv1.weight"], w["suit_conv1.bias"], suitInput);
    convRelu(conv2, w["suit_conv2.weight"], w["suit_conv2.bias"], conv1);
    for (let rank = 0; rank < 13; rank += 1) {
      const token = (suit * 13 + rank) * WIDTH;
      for (let o = 0; o < WIDTH; o += 1) {
        tokens[token + o] = rank === 0
          ? 0.5 * (conv2[o * STEPS] + conv2[o * STEPS + 13])
          : conv2[o * STEPS + rank];
      }
    }
  }

  const mixed = attention(w, tokens);
  const local = new Float64Array(NUM_FACES * WIDTH);
  for (let i = 0; i < mixed.length; i += 1) mixed[i] += tokens[i];
  for (let face = 0; face < NUM_FACES; face += 1) {
    layerNorm(row(local, face, WIDTH), row(mixed, face, WIDTH), w["card_norm.weight"], w["card_norm.bias"]);
  }
  const sets = new Float64Array(13 * WIDTH);
  const global = new Float64Array(WIDTH);
  for (let face = 0; face < NUM_FACES; face += 1) {
    for (let o = 0; o < WIDTH; o += 1) {
      sets[(face % 13) * WIDTH + o] += local[face * WIDTH + o];
      global[o] += local[face * WIDTH + o];
    }
  }
  for (let i = 0; i < sets.length; i += 1) sets[i] /= 4;
  for (let o = 0; o < WIDTH; o += 1) global[o] /= NUM_FACES;

  const relationInput = new Float64Array(3 * WIDTH);
  [0, encoded.nextPlayer[0], encoded.previousPlayer[0]].forEach((seat, slot) => {
    linear(
      row(relationInput, slot, WIDTH), w["player_features.weight"], w["player_features.bias"],
      row(encoded.players, seat, PLAYER_FEATURES),
    );
  });
  const relation = relu(linear(new Float64Array(WIDTH), w["turn_relations.weight"], w["turn_relations.bias"], relationInput));
  const length = encoded.eventKind.length;
  const eventMean = new Float64Array(EVENT_FEATURES);
  let weightTotal = 0;
  for (let i = 0; i < length; i += 1) {
    const weight = (i + 1) / length;
    weightTotal += weight;
    for (let k = 0; k < EVENT_FEATURES; k += 1) eventMean[k] += weight * encoded.events[i * EVENT_FEATURES + k];
  }
  for (let k = 0; k < EVENT_FEATURES; k += 1) eventMean[k] /= Math.max(weightTotal, 1);
  const summary = linear(new Float64Array(WIDTH), w["event_summary.weight"], w["event_summary.bias"], eventMean);
  const table = linear(new Float64Array(WIDTH), w["context_features.weight"], w["context_features.bias"], encoded.context);
  for (let o = 0; o < WIDTH; o += 1) table[o] += relation[o] + 0.1 * summary[o];
  const gate = linear(new Float64Array(2), w["context_gate.weight"], w["context_gate.bias"], table).map((v) => 0.02 * Math.tanh(v));

  const q = new Float32Array(NUM_ACTIONS);
  const hidden = new Float64Array(WIDTH);
  const drawInput = new Float64Array(3 * WIDTH + CARD_FEATURES);
  drawInput.set(global);
  const { top } = encoded;
  if (top !== NO_FACE) {
    drawInput.set(row(local, top, WIDTH), WIDTH);
    drawInput.set(row(sets, top % 13, WIDTH), 2 * WIDTH);
    drawInput.set(row(cards, top, CARD_FEATURES), 3 * WIDTH);
  }
  relu(linear(hidden, w["draw_head.0.weight"], w["draw_head.0.bias"], drawInput));
  const raw = linear(new Float64Array(2), w["draw_head.2.weight"], w["draw_head.2.bias"], hidden);
  q[0] = raw[0] + gate[0];
  q[1] = raw[0] + gate[0] + 0.25 * Math.tanh(raw[1] + gate[1]);

  const faceInput = new Float64Array(3 * WIDTH + CARD_FEATURES + 13);
  const score = new Float64Array(1);
  const bonus = new Float64Array(1);
  faceInput.set(global, 2 * WIDTH);
  for (let face = 0; face < NUM_FACES; face += 1) {
    const rank = face % 13;
    faceInput.set(row(local, face, WIDTH));
    faceInput.set(row(sets, rank, WIDTH), WIDTH);
    faceInput.set(row(cards, face, CARD_FEATURES), 3 * WIDTH);
    faceInput.fill(0, 3 * WIDTH + CARD_FEATURES);
    faceInput[3 * WIDTH + CARD_FEATURES + rank] = 1;
    relu(linear(hidden, w["score.0.weight"], w["score.0.bias"], faceInput));
    linear(score, w["score.2.weight"], w["score.2.bias"], hidden);
    linear(bonus, w["history_proj.weight"], w["history_proj.bias"], row(channels, face, CARD_FEATURES));
    q[2 + face] = score[0] + 0.02 * Math.tanh(bonus[0]) + gate[0];
  }
  return q;
}

function attention(w, tokens) {
  const qkv = new Float64Array(NUM_FACES * 3 * WIDTH);
  for (let face = 0; face < NUM_FACES; face += 1) {
    linear(
      row(qkv, face, 3 * WIDTH), w["card_attention.in_proj_weight"], w["card_attention.in_proj_bias"],
      row(tokens, face, WIDTH),
    );
  }
  const heads = new Float64Array(NUM_FACES * WIDTH);
  const scores = new Float64Array(NUM_FACES);
  const scale = 1 / Math.sqrt(HEAD_SIZE);
  for (let head = 0; head < WIDTH; head += HEAD_SIZE) {
    for (let query = 0; query < NUM_FACES; query += 1) {
      let peak = -Infinity;
      for (let key = 0; key < NUM_FACES; key += 1) {
        let dot = 0;
        for (let d = 0; d < HEAD_SIZE; d += 1) {
          dot += qkv[query * 3 * WIDTH + head + d] * qkv[key * 3 * WIDTH + WIDTH + head + d];
        }
        scores[key] = dot * scale;
        peak = Math.max(peak, scores[key]);
      }
      let total = 0;
      for (let key = 0; key < NUM_FACES; key += 1) {
        scores[key] = Math.exp(scores[key] - peak);
        total += scores[key];
      }
      for (let d = 0; d < HEAD_SIZE; d += 1) {
        let sum = 0;
        for (let key = 0; key < NUM_FACES; key += 1) sum += scores[key] * qkv[key * 3 * WIDTH + 2 * WIDTH + head + d];
        heads[query * WIDTH + head + d] = sum / total;
      }
    }
  }
  const mixed = new Float64Array(NUM_FACES * WIDTH);
  for (let face = 0; face < NUM_FACES; face += 1) {
    linear(
      row(mixed, face, WIDTH), w["card_attention.out_proj.weight"], w["card_attention.out_proj.bias"],
      row(heads, face, WIDTH),
    );
  }
  return mixed;
}

function convRelu(out, weight, bias, input) {
  const channels = input.length / STEPS;
  for (let o = 0; o < WIDTH; o += 1) {
    for (let t = 0; t < STEPS; t += 1) {
      let sum = bias[o];
      for (let c = 0; c < channels; c += 1) {
        const k = (o * channels + c) * 3;
        const x = c * STEPS + t;
        if (t > 0) sum += weight[k] * input[x - 1];
        sum += weight[k + 1] * input[x];
        if (t < STEPS - 1) sum += weight[k + 2] * input[x + 1];
      }
      out[o * STEPS + t] = sum > 0 ? sum : 0;
    }
  }
}

function layerNorm(out, input, gamma, beta) {
  let mean = 0;
  for (let i = 0; i < input.length; i += 1) mean += input[i];
  mean /= input.length;
  let variance = 0;
  for (let i = 0; i < input.length; i += 1) variance += (input[i] - mean) ** 2;
  const scale = 1 / Math.sqrt(variance / input.length + 1e-5);
  for (let i = 0; i < input.length; i += 1) out[i] = (input[i] - mean) * scale * gamma[i] + beta[i];
}

function linear(out, weight, bias, input) {
  const cols = input.length;
  for (let r = 0; r < out.length; r += 1) {
    let sum = bias[r];
    for (let c = 0; c < cols; c += 1) sum += weight[r * cols + c] * input[c];
    out[r] = sum;
  }
  return out;
}

function relu(values) {
  for (let i = 0; i < values.length; i += 1) if (values[i] < 0) values[i] = 0;
  return values;
}

function row(values, n, width) {
  return values.subarray(n * width, (n + 1) * width);
}
