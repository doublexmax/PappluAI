export const NUM_FACES = 52;
export const SCORING_VERSION = 1;

const NUM_RANKS = 13;
const NUM_SUITS = 4;
const MIN_MELD = 3;
const RANK_POINTS = [10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];
const SUIT_COMBOS = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]];
const RANK_PATTERNS = rankPatterns();
const IRREDUCIBLE_SEQUENCES = Array.from({ length: NUM_SUITS }, (_, suit) =>
  RANK_PATTERNS.filter((pattern) => pattern.length < 2 * MIN_MELD)
    .map((pattern) => pattern.map((rank) => suit * NUM_RANKS + rank))).flat();

export function evaluateHand(counts, joker, { requiredSequences = 5, cardsInHand = 21 } = {}) {
  const hand = parseCounts(counts);
  const jokerFace = parseJoker(joker);
  const needed = requireNonnegative(requiredSequences, "required_sequences");
  const target = requirePositive(cardsInHand, "cards_in_hand");
  if (sum(hand) !== target) return { is_valid: false, melds: [] };
  const melds = searchMelds(hand, needed, jokerFace);
  return melds === null ? { is_valid: false, melds: [] } : { is_valid: true, melds };
}

export function isValidHand(counts, joker, options = {}) {
  return evaluateHand(counts, joker, options).is_valid;
}

export function handReward(counts, joker, options = {}) {
  return isValidHand(counts, joker, options) ? 1 : 0;
}

export function minimumPenalty(counts, joker, { requiredSequences = 5 } = {}) {
  const hand = parseCounts(counts);
  const jokerFace = parseJoker(joker);
  const quota = requireNonnegative(requiredSequences, "required_sequences");
  const { faces, options } = compileMeldOptions(hand, jokerFace);
  const start = faces.map((face) => hand[face]);
  const points = faces.map((face) => cardPoints(face, jokerFace));
  const quotaSearch = new PenaltySearch(points, options);
  const quotaScore = quotaSearch.cost(start, quota);
  let selected = quotaSearch;
  let need = quota;
  let best = quotaScore;
  if (quotaScore !== 0) {
    const pureSearch = new PenaltySearch(
      points, options.map((anchored) => anchored.filter((option) => option.meld.is_pure)),
    );
    const pureScore = pureSearch.cost(start, 0);
    if (pureScore < quotaScore) {
      selected = pureSearch;
      need = 0;
      best = pureScore;
    }
  }
  const melds = selected.witness(start, need);
  let counted = hand;
  for (const meld of melds) counted = subtractFaces(counted, meld.cards);
  const total = counted.reduce((value, copies, face) =>
    value + cardPoints(face, jokerFace) * copies, 0);
  if (total !== best) throw new Error("internal penalty witness mismatch");
  return {
    points: total,
    exempt_melds: melds,
    counted_cards: counted,
    qualifying_sequences: melds.filter((meld) => meld.kind === "sequence" && meld.is_pure).length,
  };
}

export function qualifyingSequences(counts, joker, options = {}) {
  return minimumPenalty(counts, joker, options).qualifying_sequences;
}

export function scoreHands(hands, joker, options = {}) {
  return hands.map((counts) => minimumPenalty(counts, joker, options));
}

function requireIntegral(value, name) {
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new TypeError(`${name} must be an integral number, got '${typeof value}'`);
  }
  return value;
}

function requireNonnegative(value, name) {
  const number = requireIntegral(value, name);
  if (number < 0) throw new RangeError(`${name} must be nonnegative, got ${number}`);
  return number;
}

function requirePositive(value, name) {
  const number = requireIntegral(value, name);
  if (number <= 0) throw new RangeError(`${name} must be positive, got ${number}`);
  return number;
}

function parseCounts(counts) {
  if (counts !== null && typeof counts === "object"
      && !Array.isArray(counts) && !ArrayBuffer.isView(counts)) {
    counts = Object.keys(counts);
  }
  if (counts == null || typeof counts.length !== "number") {
    throw new TypeError(`hand must be a sequence of length ${NUM_FACES}`);
  }
  if (counts.length !== NUM_FACES) {
    throw new RangeError(`hand must have length ${NUM_FACES}, got ${counts.length}`);
  }
  return Array.from(counts, (value, index) => requireNonnegative(value, `hand[${index}]`));
}

function parseJoker(joker) {
  const number = requireIntegral(joker, "joker");
  if (number < 0 || number >= NUM_FACES) {
    throw new RangeError(`joker must be in 0..${NUM_FACES - 1}, got ${number}`);
  }
  return number;
}

function cardPoints(face, joker) {
  const rank = face % NUM_RANKS;
  return rank === joker % NUM_RANKS ? 0 : RANK_POINTS[rank];
}

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

function rankPatterns() {
  const patterns = [];
  for (let start = 0; start < NUM_RANKS; start += 1) {
    for (let end = start + MIN_MELD - 1; end < NUM_RANKS; end += 1) {
      patterns.push(Array.from({ length: end - start + 1 }, (_, index) => start + index));
    }
  }
  for (let start = 1; start < NUM_RANKS - 1; start += 1) {
    patterns.push([...Array.from({ length: NUM_RANKS - start }, (_, index) => start + index), 0]);
  }
  return patterns;
}

function subtractFaces(counts, used) {
  const remaining = counts.slice();
  for (const face of used) {
    remaining[face] -= 1;
    if (remaining[face] < 0) throw new Error("internal count underflow");
  }
  return remaining;
}

function wildFaces(joker) {
  return Array.from({ length: NUM_SUITS }, (_, suit) => suit * NUM_RANKS + joker % NUM_RANKS);
}

function* iterAllocations(counts, represented, joker) {
  const wilds = wildFaces(joker);
  const actuals = Array(represented.length).fill(0);
  const pool = counts.slice();
  function* place(slot) {
    if (slot === represented.length) {
      yield actuals.slice();
      return;
    }
    const target = represented[slot];
    const candidates = pool[target] > 0 ? [target] : [];
    for (const wild of wilds) {
      if (wild !== target && pool[wild] > 0) candidates.push(wild);
    }
    for (const card of candidates) {
      pool[card] -= 1;
      actuals[slot] = card;
      yield* place(slot + 1);
      pool[card] += 1;
    }
  }
  yield* place(0);
}

function meldFrom(kind, actuals, represented, joker) {
  return {
    kind,
    cards: actuals,
    represented_cards: represented.slice(),
    is_pure: kind === "sequence" && actuals.every((card, index) =>
      card === represented[index] || card === joker),
  };
}

function* iterSequences(counts, joker) {
  const total = sum(counts);
  const wilds = wildFaces(joker);
  const wildCount = sum(wilds.map((face) => counts[face]));
  const natural = counts.map((count, face) => count > 0 && !wilds.includes(face));
  for (let face = 0; face < NUM_FACES; face += 1) {
    if (counts[face] >= MIN_MELD) {
      const triplet = [face, face, face];
      yield { kind: "sequence", cards: triplet, represented_cards: triplet, is_pure: true };
    }
  }
  for (const represented of IRREDUCIBLE_SEQUENCES) {
    if (represented.length > total) continue;
    if (represented.length - sum(represented.map((face) => Number(natural[face]))) > wildCount) {
      continue;
    }
    for (const actuals of iterAllocations(counts, represented, joker)) {
      yield meldFrom("sequence", actuals, represented, joker);
    }
  }
}

function* iterSets(counts, joker) {
  for (let rank = 0; rank < NUM_RANKS; rank += 1) {
    for (const suits of SUIT_COMBOS) {
      const represented = suits.map((suit) => suit * NUM_RANKS + rank);
      for (const actuals of iterAllocations(counts, represented, joker)) {
        yield meldFrom("set", actuals, represented, joker);
      }
    }
  }
}

function compareTuple(left, right) {
  if (Array.isArray(left)) {
    for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
      const order = compareTuple(left[index], right[index]);
      if (order !== 0) return order;
    }
    return Math.sign(left.length - right.length);
  }
  return left === right ? 0 : left < right ? -1 : 1;
}

function compileMeldOptions(counts, joker) {
  const wilds = wildFaces(joker);
  const faces = counts.flatMap((count, face) => count ? [face] : [])
    .sort((left, right) => Number(wilds.includes(left)) - Number(wilds.includes(right)) || left - right);
  const positions = new Map(faces.map((face, index) => [face, index]));
  const best = new Map();
  for (const melds of [iterSequences(counts, joker), iterSets(counts, joker)]) {
    for (const meld of melds) {
      const used = new Map();
      for (const face of meld.cards) {
        const position = positions.get(face);
        used.set(position, (used.get(position) || 0) + 1);
      }
      const take = [...used].sort(([left], [right]) => left - right);
      const key = JSON.stringify(take);
      const previous = best.get(key);
      if (previous === undefined || meld.is_pure && !previous.meld.is_pure) {
        best.set(key, { meld, take });
      }
    }
  }
  const options = faces.map(() => []);
  for (const option of best.values()) options[option.take[0][0]].push(option);
  const orderKey = ({ meld, take }) =>
    [!meld.is_pure, meld.cards.length, meld.kind, meld.represented_cards, take];
  for (const anchored of options) {
    anchored.sort((left, right) => compareTuple(orderKey(left), orderKey(right)));
  }
  return { faces, options };
}

function stateKey(state, need) {
  return `${state.join(",")}|${need}`;
}

function takeOption(state, take) {
  const remaining = state.slice();
  for (const [position, copies] of take) remaining[position] -= copies;
  return remaining;
}

function fits(state, take) {
  return take.every(([position, copies]) => state[position] >= copies);
}

function searchMelds(counts, pureNeeded, joker) {
  if (sum(counts) < pureNeeded * MIN_MELD) return null;
  const { faces, options } = compileMeldOptions(counts, joker);
  const initial = faces.map((face) => counts[face]);
  const memo = new Map();
  function visit(state, needed) {
    const key = stateKey(state, needed);
    if (memo.has(key)) return memo.get(key);
    const total = sum(state);
    let result = null;
    if (total === 0) {
      result = needed === 0 ? [] : null;
    } else if (total >= MIN_MELD && total >= needed * MIN_MELD) {
      const anchor = state.findIndex((count) => count > 0);
      for (const option of options[anchor]) {
        if (!fits(state, option.take)) continue;
        const suffix = visit(
          takeOption(state, option.take), Math.max(0, needed - Number(option.meld.is_pure)),
        );
        if (suffix !== null) {
          result = [option.meld, ...suffix];
          break;
        }
      }
    }
    memo.set(key, result);
    return result;
  }
  return visit(initial, pureNeeded);
}

class PenaltySearch {
  constructor(points, options) {
    this.points = points;
    this.scoring = points.flatMap((value, position) => value ? [position] : []);
    this.options = options;
    this.memo = new Map();
  }

  cost(state, need) {
    const key = stateKey(state, need);
    if (!this.memo.has(key)) this.memo.set(key, this.solve(state, need));
    return this.memo.get(key);
  }

  witness(state, need) {
    const melds = [];
    while (!this.settled(state, need)) {
      const target = this.cost(state, need);
      let selected;
      for (const branch of this.branches(state, need)) {
        const [, rest, restNeed, points] = branch;
        if (points + this.cost(rest, restNeed) === target) {
          selected = branch;
          break;
        }
      }
      if (selected === undefined) throw new Error("internal penalty witness mismatch");
      const [meld, rest, restNeed] = selected;
      if (meld !== null) melds.push(meld);
      state = rest;
      need = restNeed;
    }
    return melds;
  }

  settled(state, need) {
    return need === 0 && !this.scoring.some((position) => state[position] !== 0);
  }

  solve(state, need) {
    if (this.settled(state, need)) return 0;
    if (sum(state) < need * MIN_MELD) return Infinity;
    let best = Infinity;
    for (const [, rest, restNeed, points] of this.branches(state, need)) {
      best = Math.min(best, points + this.cost(rest, restNeed));
      if (best === 0) break;
    }
    return best;
  }

  *branches(state, need) {
    const anchor = state.findIndex((count) => count > 0);
    for (const option of this.options[anchor]) {
      if (fits(state, option.take)) {
        yield [
          option.meld, takeOption(state, option.take),
          Math.max(0, need - Number(option.meld.is_pure)), 0,
        ];
      }
    }
    const remaining = state.slice();
    remaining[anchor] -= 1;
    yield [null, remaining, need, this.points[anchor]];
  }
}
