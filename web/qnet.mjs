const NUM_FACES = 52;
const NUM_ACTIONS = 54;
const SEATS = 6;
const [INITIAL, DRAW_STOCK, TAKE_DISCARD, DISCARD, REFILL] = [0, 1, 2, 3, 4];
const RANK_POINTS = [10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];
const LEGAL_RUNS = [...Array.from({ length: 11 }, (_, r) => [r, r + 1, r + 2]), [11, 12, 0]];
const RUN_FACES = Array.from({ length: 4 }, (_, s) => LEGAL_RUNS.map(run => run.map(r => s * 13 + r))).flat();
const SET_FACES = Array.from({ length: 13 }, (_, r) =>
  [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]].map(suits => suits.map(s => s * 13 + r))).flat();
const meldPoints = faces => faces.reduce((sum, face) => sum + RANK_POINTS[face % 13], 0);
const pairOrder = (a, b) => meldPoints(b) - meldPoints(a) || a[0] - b[0] || a[1] - b[1];
const ORDERED_RUNS = [...RUN_FACES].sort((a, b) =>
  Math.floor(a[0] / 13) - Math.floor(b[0] / 13) || meldPoints(b) - meldPoints(a) || a[0] % 13 - b[0] % 13);
const pairMap = new Map();
for (const run of RUN_FACES) for (let a = 0; a < 3; a += 1) for (let b = a + 1; b < 3; b += 1) {
  const pair = [run[a], run[b]].sort((a, b) => a - b);
  pairMap.set(pair.join(","), pair);
}
const RUN_PAIRS = [...pairMap.values()].sort(pairOrder);
const PURE_PAIRS = Array.from({ length: NUM_FACES }, (_, joker) => RUN_PAIRS.filter(pair => !pair.includes(joker)));
const WILD_PAIRS = Array.from({ length: 13 }, (_, rank) => {
  const pairs = new Map(RUN_PAIRS.filter(pair => pair.every(face => face % 13 !== rank))
    .map(pair => [pair.join(","), pair]));
  for (let r = 0; r < 13; r += 1) if (r !== rank) {
    for (let a = 0; a < 4; a += 1) for (let b = a + 1; b < 4; b += 1) {
      const pair = [a * 13 + r, b * 13 + r];
      pairs.set(pair.join(","), pair);
    }
  }
  return [...pairs.values()].sort(pairOrder);
});
const SET_RANKS = Array.from({ length: 13 }, (_, rank) => rank)
  .sort((a, b) => RANK_POINTS[b] - RANK_POINTS[a] || a - b);
const POINT_ORDER = Array.from({ length: NUM_FACES }, (_, face) => face)
  .sort((a, b) => RANK_POINTS[b % 13] - RANK_POINTS[a % 13] || a - b);
const RUN_STARTS = new Map(RUN_FACES.map(run => [run[0], run]));

export const CHANNELS = Object.freeze({
  card: Object.freeze([
    "own", "own_eq1", "own_eq2", "own_ge3", "pile", "top", "joker", "wild",
    "points", "unseen", "opp_discarded", "next_held", "other_held", "next_passed",
    "adjacent", "set_mates", "natural", "pure", "any_meld",
    "spare_pure", "spare_meld", "disc_pseq_delta", "disc_gated_saved", "disc_pairs_delta",
  ]),
  seat: Object.freeze(["present", "observer", "active", "hand_size", "turns", "next", "takes", "passes", "held_total"]),
  context: Object.freeze([
    "draw", "discard", "terminal", "to_act", "turns", "stock", "pile", "refills", "decks", "hand",
    "sequences", "recycle", "players", "max_turns", "jokers_held", "wilds_held", "raw_points",
    "unavoidable_dw", "disjoint_pseq_frac", "disjoint_gated_pts",
    "top_gain_pseq", "top_gain_gated", "top_gain_pairs",
  ]),
});
export const FEATURE_LAYOUT = Object.freeze([
  ["card", [52, 24]], ["seat", [6, 9]], ["context", [23]], ["legal", [54]], ["top", [1]],
].map(([name, shape]) => Object.freeze([name, Object.freeze(shape)])));
export const FEATURE_SIZE = 1380;
const FIELDS = {};
let offset = 0;
for (const [name, shape] of FEATURE_LAYOUT) {
  const size = shape.reduce((a, b) => a * b, 1);
  FIELDS[name] = { offset, size };
  offset += size;
}

const HEADER = Object.freeze({
  format: "papplu-qnet", formatVersion: 2, checkpointVersion: 6,
  encodingVersion: 7, architecture: "card_mlp", numActions: NUM_ACTIONS,
});
const MODEL_KEYS = [...Object.keys(HEADER), "modelSpec", "gameConfig", "source", "tensors"];
const GAME_CONFIG_KEYS = ["numDecks", "cardsInHand", "requiredSequences", "maxTurns", "recycleDiscard"];
const SOURCE_KEYS = ["kind", "trained", "seed", "sha256", "tensorsSha256"];
const SOURCE_KINDS = ["bootstrap", "checkpoint"];
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
  layer("card_embed", width, CHANNELS.card.length);
  layer("table_embed", width, SEATS * CHANNELS.seat.length + CHANNELS.context.length);
  for (let b = 0; b < blocks; b += 1) {
    const prefix = `blocks.${b}.`;
    layer(`${prefix}card_norm`, width);
    layer(`${prefix}table_norm`, width);
    layer(`${prefix}w_seq`, width * 5, width, false);
    for (const name of ["w_rank_set", "w_suit_run", "w_table"]) layer(prefix + name, width, width, false);
    layer(`${prefix}card_ffn`, width * 2, width);
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
  WEIGHTS.set(model, { weights, buffers: forwardBuffers(spec.width) });
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
  const { kind, trained, seed, sha256, tensorsSha256 } = source;
  if (!SOURCE_KINDS.includes(kind)) fail(`${path}.kind`, `expected one of ${SOURCE_KINDS.join(", ")}`);
  const bootstrap = kind === "bootstrap";
  if (trained !== !bootstrap) fail(`${path}.trained`, `must be ${!bootstrap} for ${kind} weights`);
  if (bootstrap ? !Number.isSafeInteger(seed) : seed !== null) {
    fail(`${path}.seed`, bootstrap ? "expected the integer bootstrap seed" : "expected null");
  }
  for (const key of ["sha256", "tensorsSha256"]) {
    if (typeof source[key] !== "string" || !SHA256.test(source[key])) fail(`${path}.${key}`, "expected 64 lowercase hex digits");
  }
  return Object.freeze({ kind, trained, seed, sha256, tensorsSha256 });
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
  const values = new Float32Array(length);
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

export function disjointPack(hand, joker, requiredSequences) {
  const rem = [...hand];
  let pure = 0;
  for (const [a, b, c] of ORDERED_RUNS) {
    if (joker === a || joker === b || joker === c) continue;
    while (rem[a] && rem[b] && rem[c]) {
      rem[a] -= 1;
      rem[b] -= 1;
      rem[c] -= 1;
      pure += 1;
      if (c % 13 < 12 && c === b + 1) {
        const next = c + 1;
        const run = RUN_STARTS.get(next);
        const starts = run !== undefined && run.every(face => rem[face]);
        if (next !== joker && rem[next] && !starts) rem[next] -= 1;
      }
    }
  }
  for (let face = 0; face < NUM_FACES; face += 1) if (face !== joker) {
    while (rem[face] >= 3) {
      rem[face] -= 3;
      pure += 1;
    }
  }
  for (const [a, b, c] of ORDERED_RUNS) {
    if (joker !== a && joker !== b && joker !== c) continue;
    while (rem[a] && rem[b] && rem[c]) {
      rem[a] -= 1;
      rem[b] -= 1;
      rem[c] -= 1;
      pure += 1;
    }
  }
  while (rem[joker]) {
    const pair = PURE_PAIRS[joker].find(([a, b]) => rem[a] && rem[b]);
    if (pair) {
      rem[pair[0]] -= 1;
      rem[pair[1]] -= 1;
      rem[joker] -= 1;
      pure += 1;
    } else if (rem[joker] >= 2) {
      const face = POINT_ORDER.find(face => face !== joker && rem[face]);
      if (face === undefined) break;
      rem[face] -= 1;
      rem[joker] -= 2;
      pure += 1;
    } else break;
  }
  const afterPure = [...rem];
  const pairs = RUN_FACES.filter(run => run.filter(face => face !== joker && rem[face] > 0).length >= 2).length;
  const rank = joker % 13;
  let wilds = 0;
  for (let suit = 0; suit < 4; suit += 1) {
    const face = suit * 13 + rank;
    wilds += rem[face];
    rem[face] = 0;
  }
  for (const r of SET_RANKS) {
    if (r === rank) continue;
    while (true) {
      const faces = [0, 1, 2, 3].map(suit => suit * 13 + r).filter(face => rem[face]);
      if (faces.length < 3) break;
      for (const face of faces) rem[face] -= 1;
    }
  }
  while (wilds) {
    const pair = WILD_PAIRS[rank].find(([a, b]) => rem[a] && rem[b]);
    if (!pair) break;
    rem[pair[0]] -= 1;
    rem[pair[1]] -= 1;
    wilds -= 1;
  }
  const score = counts => counts.reduce((sum, n, face) => sum + (face % 13 === rank ? 0 : n * RANK_POINTS[face % 13]), 0);
  const points = score(rem);
  return [pure, pure >= requiredSequences ? points : score(afterPure), points, pairs, rem, afterPure];
}

function meldMembership(hand, joker) {
  const natural = new Int32Array(NUM_FACES);
  const pure = new Int32Array(NUM_FACES);
  const anyMeld = new Int32Array(NUM_FACES);
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
  const required = config.requiredSequences;
  const base = disjointPack(hand, joker, required);
  const { natural, pure, anyMeld } = meldMembership(hand, joker);
  let rawPoints = 0;
  let wildCount = 0;
  let unavoidable = 0;
  hand.forEach((copies, face) => {
    const suit = Math.floor(face / 13);
    const rank = face % 13;
    const wild = rank === joker % 13;
    const points = wild ? 0 : RANK_POINTS[rank];
    rawPoints += copies * points;
    if (wild) wildCount += copies;
    unavoidable += copies * points * (1 - anyMeld[face]);
    let setMates = 0;
    for (let other = 0; other < 4; other += 1) {
      if (other !== suit && hand[other * 13 + rank]) setMates += 1;
    }
    let opponentHeld = 0;
    for (let seat = 1; seat < players.length; seat += 1) opponentHeld += held[seat][face];
    const nextHeld = nextOpponent > 0 ? held[nextOpponent][face] : 0;
    const unseen = Math.max(0, decks - copies - pileCounts[face] - Number(face === joker) -
      held[0][face] - opponentHeld);
    values.set([
      copies / decks, Number(copies === 1), Number(copies === 2), Number(copies >= 3), pileCounts[face] / decks,
      Number(face === top), Number(face === joker), Number(wild), points / 10, unseen / decks,
      oppDiscarded[face] / decks, nextHeld / decks, (opponentHeld - nextHeld) / decks, nextPassed[face] / decks,
      (Number(hand[suit * 13 + (rank + 12) % 13] > 0) + Number(hand[suit * 13 + (rank + 1) % 13] > 0)) / 2,
      setMates / 3, natural[face], pure[face], anyMeld[face],
      copies ? base[5][face] / copies : 0, copies ? base[4][face] / copies : 0, 0, 0, 0,
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
    hand[joker] / decks, wildCount / (4 * decks), rawPoints / 100,
    unavoidable / (10 * cards), Math.min(base[0], required) / Math.max(1, required), base[1] / 100, 0, 0, 0,
  ], FIELDS.context.offset);
  if (toAct) {
    if (phase === 0) {
      values[FIELDS.legal.offset] = Number(obs.stockRemaining > 0);
      values[FIELDS.legal.offset + 1] = Number(discardPile.length > 0);
      if (last.kind === DISCARD && last.actor === 0 && values[FIELDS.legal.offset]) {
        values[FIELDS.legal.offset + 1] = 0;
      }
    } else {
      hand.forEach((n, face) => { values[FIELDS.legal.offset + 2 + face] = Number(n > 0); });
      if (last.kind === TAKE_DISCARD && last.actor === 0 && last.face !== null &&
          hand.filter(n => n > 0).length > 1) {
        values[FIELDS.legal.offset + 2 + last.face] = 0;
      }
    }
  }
  if (toAct && phase === 1) {
    hand.forEach((copies, face) => {
      if (!copies || !values[FIELDS.legal.offset + 2 + face]) return;
      const after = [...hand];
      after[face] -= 1;
      const packed = disjointPack(after, joker, required);
      values.set([
        Math.min(packed[0], required) - Math.min(base[0], required),
        (base[1] - packed[1]) / 10, (packed[3] - base[3]) / 3,
      ], FIELDS.card.offset + face * CHANNELS.card.length + CHANNELS.card.indexOf("disc_pseq_delta"));
    });
  }
  if (toAct && phase === 0 && values[FIELDS.legal.offset + 1]) {
    const plus = [...hand];
    plus[top] += 1;
    let best = null;
    let bestKey = null;
    plus.forEach((copies, face) => {
      if (!copies || face === top) return;
      plus[face] -= 1;
      const packed = disjointPack(plus, joker, required);
      plus[face] += 1;
      const key = [Math.min(packed[0], required), -packed[1], packed[3], -packed[2]];
      const difference = bestKey === null ? -1 : key.findIndex((value, i) => value !== bestKey[i]);
      if (bestKey === null || difference >= 0 && key[difference] > bestKey[difference]) {
        best = packed;
        bestKey = key;
      }
    });
    if (best !== null) {
      values.set([
        Math.min(best[0], required) - Math.min(base[0], required),
        (base[1] - best[1]) / 20, (best[3] - base[3]) / 3,
      ], FIELDS.context.offset + CHANNELS.context.indexOf("top_gain_pseq"));
    }
  }
  values[FIELDS.top.offset] = top;
  if (!values.every(Number.isFinite)) fail(path, "features are not finite; check the counts and config");
  return values;
}

export function qValues(model, observations) {
  const state = WEIGHTS.get(model);
  if (state === undefined) throw new TypeError("qValues: model must come from parseQModel");
  if (!Array.isArray(observations)) throw new TypeError("qValues: observations must be an array");
  const results = [];
  for (let i = 0; i < observations.length; i += 1) {
    results.push(forward(state.weights, model.spec, encode(observations[i], `observations[${i}]`), state.buffers));
  }
  return results;
}

function forwardBuffers(width) {
  return {
    cards: new Float32Array(NUM_FACES * width),
    normalized: new Float32Array(NUM_FACES * width),
    projected: new Float32Array(NUM_FACES * width * 5),
    sequence: new Float32Array(4 * 14 * width),
    gated: new Float32Array(NUM_FACES * width * 2),
    ranks: new Float32Array(13 * width),
    rankContext: new Float32Array(13 * width),
    suits: new Float32Array(4 * width),
    suitContext: new Float32Array(4 * width),
    table: new Float32Array(width),
    normalizedTable: new Float32Array(width),
    tableContext: new Float32Array(width),
    tableInput: new Float32Array(width * 3),
    tableHidden: new Float32Array(width),
    zero: new Float32Array(width),
    advantage: new Float32Array(NUM_ACTIONS),
  };
}

function forward(w, spec, features, buffers) {
  const { width, blocks } = spec;
  const {
    cards, normalized, projected, sequence, gated, ranks, rankContext, suits, suitContext,
    table, normalizedTable, tableContext, tableInput, tableHidden, zero, advantage,
  } = buffers;
  linear(w, "card_embed", features, CHANNELS.card.length, cards, width, NUM_FACES);
  linear(w, "table_embed", features, FIELDS.seat.size + FIELDS.context.size, table, width, 1, FIELDS.seat.offset);
  for (let block = 0; block < blocks; block += 1) {
    const prefix = `blocks.${block}.`;
    layerNorm(w, `${prefix}card_norm`, cards, normalized, width, NUM_FACES);
    layerNorm(w, `${prefix}table_norm`, table, normalizedTable, width);
    linear(w, `${prefix}w_seq`, normalized, width, projected, width * 5, NUM_FACES);
    suitMeans(normalized, ranks, width);
    linear(w, `${prefix}w_rank_set`, ranks, width, rankContext, width, 13);
    for (let suit = 0; suit < 4; suit += 1) {
      for (let d = 0; d < width; d += 1) {
        let sum = 0;
        for (let rank = 0; rank < 13; rank += 1) sum += normalized[(suit * 13 + rank) * width + d];
        suits[suit * width + d] = sum / 13;
      }
    }
    linear(w, `${prefix}w_suit_run`, suits, width, suitContext, width, 4);
    linear(w, `${prefix}w_table`, normalizedTable, width, tableContext, width);
    for (let suit = 0; suit < 4; suit += 1) {
      for (let position = 0; position < 14; position += 1) {
        const face = suit * 13 + position % 13;
        for (let d = 0; d < width; d += 1) {
          let value = projected[face * width * 5 + width * 2 + d];
          if (position > 0) value = Math.fround(value +
            projected[(suit * 13 + position - 1) * width * 5 + width + d]);
          if (position < 13) value = Math.fround(value +
            projected[(suit * 13 + (position + 1) % 13) * width * 5 + width * 3 + d]);
          if (position > 1) value = Math.fround(value +
            projected[(suit * 13 + position - 2) * width * 5 + d]);
          if (position < 12) value = Math.fround(value +
            projected[(suit * 13 + (position + 2) % 13) * width * 5 + width * 4 + d]);
          sequence[(suit * 14 + position) * width + d] = value;
        }
      }
    }
    for (let face = 0; face < NUM_FACES; face += 1) {
      const suit = Math.floor(face / 13);
      const rank = face % 13;
      for (let d = 0; d < width; d += 1) {
        const seq = rank === 0 ? Math.fround(
          sequence[suit * 14 * width + d] + sequence[(suit * 14 + 13) * width + d],
        ) * 0.5 : sequence[(suit * 14 + rank) * width + d];
        normalized[face * width + d] = Math.fround(Math.fround(
          seq + rankContext[rank * width + d],
        ) + suitContext[suit * width + d]) + tableContext[d];
      }
    }
    linear(w, `${prefix}card_ffn`, normalized, width, gated, width * 2, NUM_FACES);
    for (let face = 0; face < NUM_FACES; face += 1) {
      for (let d = 0; d < width; d += 1) {
        const at = face * width * 2 + d;
        const a = gated[at];
        normalized[face * width + d] = Math.fround(a / (1 + Math.exp(-a))) * gated[at + width];
      }
    }
    linear(w, `${prefix}card_out`, normalized, width, cards, width, NUM_FACES, 0, 0, true);
    layerNorm(w, `${prefix}card_norm`, cards, normalized, width, NUM_FACES);
    tableInput.set(normalizedTable);
    for (let d = 0; d < width; d += 1) {
      let sum = 0;
      let maximum = -Infinity;
      for (let face = 0; face < NUM_FACES; face += 1) {
        const value = normalized[face * width + d];
        sum += value;
        maximum = Math.max(maximum, value);
      }
      tableInput[width + d] = sum / NUM_FACES;
      tableInput[width * 2 + d] = maximum;
    }
    linear(w, `${prefix}table_fc1`, tableInput, width * 3, tableHidden, width);
    silu(tableHidden);
    linear(w, `${prefix}table_fc2`, tableHidden, width, table, width, 1, 0, 0, true);
  }
  layerNorm(w, "card_final", cards, normalized, width, NUM_FACES);
  layerNorm(w, "table_final", table, normalizedTable, width);
  suitMeans(normalized, ranks, width);
  linear(w, "face_set", ranks, width, rankContext, width, 13);
  linear(w, "face_table", normalizedTable, width, tableContext, width);
  linear(w, "face_local", normalized, width, cards, width, NUM_FACES);
  for (let face = 0; face < NUM_FACES; face += 1) {
    for (let d = 0; d < width; d += 1) {
      const at = face * width + d;
      cards[at] = Math.fround(cards[at] + rankContext[(face % 13) * width + d]) + tableContext[d];
    }
  }
  silu(cards);
  linear(w, "face_out", cards, width, advantage, 1, NUM_FACES, 0, 2);
  const top = features[FIELDS.top.offset];
  const hasTop = top >= 0 && top < NUM_FACES;
  linear(w, "draw_table", normalizedTable, width, tableHidden, width);
  linear(w, "draw_top", hasTop ? normalized : zero, width, tableContext, width, 1, hasTop ? top * width : 0);
  linear(w, "draw_set", hasTop ? ranks : zero, width, tableInput, width, 1, hasTop ? (top % 13) * width : 0);
  for (let d = 0; d < width; d += 1) {
    tableHidden[d] = Math.fround(tableHidden[d] + tableContext[d]) + tableInput[d];
  }
  silu(tableHidden);
  linear(w, "draw_out", tableHidden, width, advantage, 2);
  let mean = 0;
  let count = 0;
  for (let action = 0; action < NUM_ACTIONS; action += 1) {
    const legal = features[FIELDS.legal.offset + action];
    mean += advantage[action] * legal;
    count += legal;
  }
  mean = Math.fround(Math.fround(mean) / Math.max(count, 1));
  linear(w, "value_hidden", normalizedTable, width, tableHidden, width);
  silu(tableHidden);
  linear(w, "value_out", tableHidden, width, tableInput, 1);
  const result = new Float32Array(NUM_ACTIONS);
  for (let action = 0; action < NUM_ACTIONS; action += 1) {
    result[action] = Math.fround(tableInput[0] + advantage[action]) - mean;
  }
  return result;
}

function linear(w, name, input, cols, output, outputs, rows = 1, inputOffset = 0, outputOffset = 0, residual = false) {
  const weight = w[`${name}.weight`];
  const bias = w[`${name}.bias`];
  for (let row = 0; row < rows; row += 1) {
    const inputAt = inputOffset + row * cols;
    let r = 0;
    for (; r + 3 < outputs; r += 4) {
      const wa = r * cols, wb = wa + cols, wc = wb + cols, wd = wc + cols;
      let a = 0, b = 0, c = 0, d = 0;
      for (let column = 0; column < cols; column += 1) {
        const value = input[inputAt + column];
        a += weight[wa + column] * value;
        b += weight[wb + column] * value;
        c += weight[wc + column] * value;
        d += weight[wd + column] * value;
      }
      const at = outputOffset + row * outputs + r;
      a = Math.fround(a + (bias === undefined ? 0 : bias[r]));
      b = Math.fround(b + (bias === undefined ? 0 : bias[r + 1]));
      c = Math.fround(c + (bias === undefined ? 0 : bias[r + 2]));
      d = Math.fround(d + (bias === undefined ? 0 : bias[r + 3]));
      output[at] = residual ? output[at] + a : a;
      output[at + 1] = residual ? output[at + 1] + b : b;
      output[at + 2] = residual ? output[at + 2] + c : c;
      output[at + 3] = residual ? output[at + 3] + d : d;
    }
    for (; r < outputs; r += 1) {
      const weightAt = r * cols;
      let sum = 0;
      for (let column = 0; column < cols; column += 1) sum += weight[weightAt + column] * input[inputAt + column];
      const value = Math.fround(sum + (bias === undefined ? 0 : bias[r]));
      const at = outputOffset + row * outputs + r;
      output[at] = residual ? output[at] + value : value;
    }
  }
}

function layerNorm(w, name, input, output, width, rows = 1) {
  const weight = w[`${name}.weight`];
  const bias = w[`${name}.bias`];
  for (let row = 0; row < rows; row += 1) {
    const at = row * width;
    let mean = 0;
    for (let d = 0; d < width; d += 1) mean += input[at + d];
    mean /= width;
    let variance = 0;
    for (let d = 0; d < width; d += 1) variance += (input[at + d] - mean) ** 2;
    const scale = 1 / Math.sqrt(variance / width + 1e-5);
    for (let d = 0; d < width; d += 1) {
      output[at + d] = (input[at + d] - mean) * scale * weight[d] + bias[d];
    }
  }
}

function silu(values) {
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    values[i] = value >= 0 ? value / (1 + Math.exp(-value)) : value * Math.exp(value) / (1 + Math.exp(value));
  }
}

function suitMeans(cards, output, width) {
  for (let rank = 0; rank < 13; rank += 1) {
    for (let d = 0; d < width; d += 1) {
      const at = rank * width + d;
      output[at] = (cards[at] + cards[13 * width + at] + cards[26 * width + at] + cards[39 * width + at]) / 4;
    }
  }
}
