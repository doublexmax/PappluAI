import { evaluateHand, minimumPenalty } from "./evaluate.mjs";
import { qValues } from "./qnet.mjs";

const INITIAL = 0;
const DRAW_STOCK = 1;
const TAKE_DISCARD = 2;
const DISCARD = 3;
const REFILL = 4;
const DEFAULT_MAX_TURNS = 60;
const BEAM = 4;

function countsOf(faces) {
  const counts = Array(52).fill(0);
  for (const face of faces) counts[face] += 1;
  return counts;
}

export function observationFromView(view, maxTurns = DEFAULT_MAX_TURNS) {
  const count = view.players.length;
  const relative = (seat) => (seat - view.seat + count) % count;
  const sizes = Array(count).fill(view.rules.cardsInHand);
  const discards = Array(count).fill(0);
  const active = Array(count).fill(true);
  const turns = (seat) => active[seat] ? Math.max(1, maxTurns - discards[seat]) : 0;
  const neighbour = (seat, direction) => {
    for (let offset = 1; offset <= count; offset += 1) {
      const candidate = (seat + direction * offset + count) % count;
      if (active[candidate]) return candidate;
    }
    return seat;
  };
  const history = [];
  for (const event of view.events) {
    if (event.kind === "declare") {
      sizes[event.seat] -= 1;
      continue;
    }
    if (event.kind === "verdict") {
      if (!event.valid) active[event.seat] = false;
      continue;
    }
    if (event.kind === "finish") continue;
    let kind;
    const seat = event.kind === "deal" ? view.seat : event.seat;
    if (event.kind === "deal") kind = INITIAL;
    else if (event.kind === "draw") {
      kind = event.source === "stock" ? DRAW_STOCK : TAKE_DISCARD;
      sizes[seat] += 1;
    } else if (event.kind === "discard") {
      kind = DISCARD;
      sizes[seat] -= 1;
      discards[seat] += 1;
    } else if (event.kind === "refill") kind = REFILL;
    else continue;
    history.push({
      kind, actor: relative(seat), handSize: sizes[seat], turnsRemaining: turns(seat),
      stockRemaining: event.stock, nextPlayer: relative(neighbour(seat, 1)),
      face: event.kind === "draw" || event.kind === "discard" ? event.face : null,
      offeredTop: event.kind === "deal" ? event.top : event.kind === "draw" ? event.offeredTop : null,
      initialHand: event.kind === "deal" ? countsOf(event.hand) : [],
    });
  }
  const players = Array.from({ length: count }, (_, index) => {
    const seat = (view.seat + index) % count;
    active[seat] = view.players[seat].active;
    return seat;
  }).map((seat) => ({
    handSize: view.players[seat].handSize,
    turnsRemaining: turns(seat),
    nextPlayer: relative(neighbour(seat, 1)),
    previousPlayer: relative(neighbour(seat, -1)),
    active: active[seat],
  }));
  return {
    hand: countsOf(view.hand.map((card) => card.face)),
    discardPile: [...view.pile], joker: view.joker,
    phase: view.phase === "draw" ? 0 : view.phase === "discard" ? 1 : 2,
    turnsRemaining: players[0].turnsRemaining,
    stockRemaining: view.stockCount,
    config: {
      numDecks: view.rules.decks, cardsInHand: view.rules.cardsInHand,
      requiredSequences: view.rules.requiredSequences, maxTurns,
      recycleDiscard: view.rules.recycleDiscard,
    },
    players, history,
  };
}

export function canDeclareUpperBound(hand, joker, requiredSequences) {
  let upper = hand[joker];
  if (upper >= requiredSequences) return true;
  for (let suit = 0; suit < 4; suit += 1) {
    const remaining = hand.slice(suit * 13, (suit + 1) * 13);
    if (Math.floor(joker / 13) === suit) remaining[joker % 13] = 0;
    let triples = remaining.reduce((total, copies) => total + Math.floor(copies / 3), 0);
    const extended = [...remaining, remaining[0]];
    for (let start = 0; start < 12; start += 1) {
      triples += Math.min(...extended.slice(start, start + 3));
    }
    upper += Math.min(Math.floor(remaining.reduce((a, b) => a + b, 0) / 3), triples);
  }
  return upper >= requiredSequences;
}

export function winningDiscard(hand, joker, { requiredSequences = 5, cardsInHand = 21 } = {}) {
  if (hand.reduce((a, b) => a + b, 0) !== cardsInHand + 1 ||
      !canDeclareUpperBound(hand, joker, requiredSequences)) return null;
  const remaining = [...hand];
  for (let face = 0; face < 52; face += 1) {
    if (!hand[face]) continue;
    remaining[face] -= 1;
    let evaluation = null;
    if (canDeclareUpperBound(remaining, joker, requiredSequences)) {
      evaluation = evaluateHand(remaining, joker, { requiredSequences, cardsInHand });
    }
    remaining[face] += 1;
    if (evaluation?.is_valid) return { face, evaluation };
  }
  return null;
}

export class BeliefState {
  constructor(obs) {
    const count = obs.players.length;
    this.knownHeld = Array.from({ length: count }, () => Array(52).fill(0));
    this.oppSafe = Array.from({ length: count }, () => Array(52).fill(0));
    const takes = Array(count).fill(0);
    const passes = Array(count).fill(0);
    let pile = [];
    let refillPool = null;
    for (const event of obs.history) {
      if (event.kind === INITIAL) pile = event.offeredTop === null ? [] : [event.offeredTop];
      else if (event.kind === TAKE_DISCARD) {
        pile.pop();
        if (event.actor !== 0) {
          this.knownHeld[event.actor][event.face] += 1;
          takes[event.actor] += 1;
        }
      } else if (event.kind === DISCARD) {
        pile.push(event.face);
        if (event.actor !== 0) {
          const held = this.knownHeld[event.actor];
          held[event.face] = Math.max(0, held[event.face] - 1);
          this.oppSafe[event.actor][event.face] += 1;
        }
      } else if (event.kind === DRAW_STOCK) {
        if (refillPool && event.face !== null) {
          refillPool[event.face] = Math.max(0, refillPool[event.face] - 1);
        }
        if (event.actor !== 0 && event.offeredTop !== null) {
          this.oppSafe[event.actor][event.offeredTop] += 1;
          passes[event.actor] += 1;
        }
      } else if (event.kind === REFILL) {
        refillPool = countsOf(pile.slice(0, -1));
        pile = pile.slice(-1);
      }
    }
    const pileCounts = countsOf(obs.discardPile);
    this.unseenCounts = obs.hand.map((copies, face) => Math.max(
      0, obs.config.numDecks - copies - pileCounts[face] - Number(face === obs.joker) -
        this.knownHeld.reduce((total, held) => total + held[face], 0),
    ));
    this.stockWeights = refillPool ?
      refillPool.map((copies, face) => Math.min(copies, this.unseenCounts[face])) : this.unseenCounts;
    if (!this.stockWeights.some(Boolean)) this.stockWeights = this.unseenCounts;
    this.oppWanted = this.knownHeld.map((held) => Array.from({ length: 52 }, (_, face) => {
      let wanted = 0;
      for (let known = 0; known < 52; known += 1) {
        if (!held[known]) continue;
        const rank = face % 13;
        const knownRank = known % 13;
        let weight;
        if (face === known || rank === obs.joker % 13 || knownRank === obs.joker % 13) weight = 1;
        else if (Math.floor(face / 13) === Math.floor(known / 13)) {
          let distance = Math.abs(rank - knownRank);
          if (rank === 0 || knownRank === 0) distance = Math.min(distance, 13 - distance);
          weight = distance === 1 ? 0.6 : distance === 2 ? 0.3 : 0;
        } else weight = rank === knownRank ? 0.5 : 0;
        wanted = Math.max(wanted, weight);
      }
      return wanted;
    }));
    this.takeRate = takes.map((take, seat) => (1 + take) / (5 + take + passes[seat]));
    this.joker = obs.joker;
  }

  stockProbabilities() {
    const total = this.stockWeights.reduce((a, b) => a + b, 0);
    return this.stockWeights.flatMap((weight, face) => weight ? [[face, weight / total]] : []);
  }

  feedRisk(face, seat = 1) {
    if (seat >= this.oppSafe.length || this.oppSafe[seat][face]) return 0;
    return this.takeRate[seat] * this.oppWanted[seat][face];
  }
}

export function outs1ply(hand, belief, rules) {
  if (!canDeclareUpperBound(hand, belief.joker, rules.requiredSequences - 1)) return 0;
  let probability = 0;
  const drawn = [...hand];
  for (const [face, chance] of belief.stockProbabilities()) {
    drawn[face] += 1;
    if (winningDiscard(drawn, belief.joker, rules)) probability += chance;
    drawn[face] -= 1;
  }
  return probability;
}

export function chooseMove(view, tier, { model = null, random = Math.random } = {}) {
  if (!["neural", "tactical", "casual"].includes(tier)) throw new TypeError("unknown AI tier");
  if (view.seat !== view.currentPlayer || !["draw", "discard"].includes(view.phase) ||
      !view.players[view.seat].active) throw new Error("seat cannot act");
  let effective = tier;
  let downgrade = null;
  if (tier === "neural") {
    if (!model) downgrade = "no-model";
    else if (model.gameConfig.numDecks !== view.rules.decks ||
             model.gameConfig.cardsInHand !== view.rules.cardsInHand ||
             model.gameConfig.requiredSequences !== view.rules.requiredSequences ||
             model.gameConfig.recycleDiscard !== view.rules.recycleDiscard) downgrade = "rules-mismatch";
    else if (view.players.length < 2 || view.players.length > 6) downgrade = "seat-count";
    if (downgrade) effective = "tactical";
  }
  const obs = observationFromView(view, model?.gameConfig.maxTurns ?? DEFAULT_MAX_TURNS);
  if (obs.phase === 1) {
    const win = winningDiscard(obs.hand, obs.joker, view.rules);
    if (win) return {
      move: { type: "declare", cardId: cardIdForFace(view, win.face), evaluation: win.evaluation },
      tier: effective, downgrade, reason: "exact-win",
    };
  }
  const legal = legalActions(obs);
  if (!legal.length) throw new Error("no legal move");
  const result = legal.length === 1 ? { action: legal[0], reason: "forced" } :
    effective === "neural" ? neuralPolicy(obs, model) :
    effective === "tactical" ? tacticalPolicy(obs) : casualPolicy(obs, random);
  const move = result.action < 2 ? { type: "draw", source: result.action === 0 ? "stock" : "discard" } :
    { type: "discard", cardId: cardIdForFace(view, result.action - 2) };
  return { move, tier: effective, downgrade, reason: result.reason };
}

function cardIdForFace(view, face) {
  return view.hand.find((card) => card.face === face).id;
}

function legalActions(obs) {
  return obs.phase === 0 ?
    [...(obs.stockRemaining > 0 ? [0] : []), ...(obs.discardPile.length ? [1] : [])] :
    obs.hand.flatMap((copies, face) => copies > 0 ? [face + 2] : []);
}

function candidateFaces(obs, excludeTaken = false) {
  let faces = legalActions(obs).map((action) => action - 2);
  const last = obs.history.at(-1);
  if (excludeTaken && faces.length > 1 && last?.actor === 0 && last.kind === TAKE_DISCARD) {
    faces = faces.filter((face) => face !== last.face);
  }
  return faces;
}

function bestFace(faces, values) {
  return faces.reduce((best, face) => values[face] > values[best] ? face : best, faces[0]);
}

function tacticalPolicy(obs) {
  const belief = new BeliefState(obs);
  const cache = new Map();
  const penalty = (hand) => {
    const key = hand.join(",");
    if (!cache.has(key)) cache.set(key, minimumPenalty(hand, obs.joker, obs.config).points);
    return cache.get(key);
  };
  if (obs.phase === 0) {
    const hand = [...obs.hand];
    const top = obs.discardPile.at(-1);
    hand[top] += 1;
    if (winningDiscard(hand, obs.joker, obs.config)) return { action: 1, reason: "exact-win" };
    let best = Infinity;
    for (let face = 0; face < 52; face += 1) {
      if (!hand[face]) continue;
      hand[face] -= 1;
      best = Math.min(best, penalty(hand));
      hand[face] += 1;
    }
    return { action: best < penalty(obs.hand) ? 1 : 0, reason: "min-penalty" };
  }
  const faces = candidateFaces(obs);
  const values = Array(52).fill(-Infinity);
  const hand = [...obs.hand];
  for (const face of faces) {
    hand[face] -= 1;
    values[face] = -penalty(hand) - 10 * belief.feedRisk(face);
    hand[face] += 1;
  }
  const candidates = [...faces].sort((a, b) => values[b] - values[a] || a - b).slice(0, BEAM);
  for (const face of candidates) {
    hand[face] -= 1;
    values[face] += 40 * outs1ply(hand, belief, obs.config);
    hand[face] += 1;
  }
  return { action: bestFace(faces, values) + 2, reason: "min-penalty" };
}

function casualPolicy(obs, random) {
  const partners = (face) => {
    let score = 0;
    for (let other = 0; other < 52; other += 1) {
      const copies = obs.hand[other] - Number(other === face && obs.phase === 1);
      if (copies <= 0) continue;
      const rank = face % 13;
      const otherRank = other % 13;
      if (rank === otherRank) score += copies;
      else if (Math.floor(face / 13) === Math.floor(other / 13)) {
        let distance = Math.abs(rank - otherRank);
        if (rank === 0 || otherRank === 0) distance = Math.min(distance, 13 - distance);
        if (distance <= 2) score += copies;
      }
    }
    return score;
  };
  if (obs.phase === 0) {
    return { action: partners(obs.discardPile.at(-1)) >= 2 ? 1 : 0, reason: "keep-score" };
  }
  let faces = candidateFaces(obs);
  const natural = faces.filter((face) => face % 13 !== obs.joker % 13);
  if (natural.length) faces = natural;
  const lowest = Math.min(...faces.map(partners));
  const tied = faces.filter((face) => partners(face) === lowest);
  return { action: tied[Math.floor(random() * tied.length)] + 2, reason: "greedy" };
}

function afterDraw(obs, face, kind) {
  const hand = [...obs.hand];
  hand[face] += 1;
  const take = kind === TAKE_DISCARD;
  const stockRemaining = obs.stockRemaining - Number(!take);
  const player = { ...obs.players[0], handSize: obs.players[0].handSize + 1 };
  return {
    ...obs, hand, phase: 1, stockRemaining,
    discardPile: take ? obs.discardPile.slice(0, -1) : obs.discardPile,
    players: [player, ...obs.players.slice(1)],
    history: [...obs.history, {
      kind, actor: 0, handSize: player.handSize, turnsRemaining: obs.turnsRemaining,
      stockRemaining, nextPlayer: player.nextPlayer, face,
      offeredTop: obs.discardPile.at(-1) ?? null, initialHand: [],
    }],
  };
}

function neuralPolicy(obs, model) {
  const belief = new BeliefState(obs);
  if (obs.phase === 1) {
    const values = Array.from(qValues(model, [obs])[0].slice(2));
    const faces = candidateFaces(obs, true);
    for (const face of faces) values[face] -= 0.03 * belief.feedRisk(face);
    const candidates = [...faces].sort((a, b) => values[b] - values[a] || a - b).slice(0, BEAM);
    const hand = [...obs.hand];
    for (const face of candidates) {
      hand[face] -= 1;
      values[face] += 0.15 * outs1ply(hand, belief, obs.config);
      hand[face] += 1;
    }
    return { action: bestFace(faces, values) + 2, reason: "discard-search" };
  }
  const top = obs.discardPile.at(-1);
  const hand = [...obs.hand];
  hand[top] += 1;
  if (winningDiscard(hand, obs.joker, obs.config)) return { action: 1, reason: "exact-win" };
  const probabilities = belief.stockProbabilities();
  if (probabilities.length) {
    const children = [afterDraw(obs, top, TAKE_DISCARD),
      ...probabilities.map(([face]) => afterDraw(obs, face, DRAW_STOCK))];
    const rows = qValues(model, children);
    const best = rows.map((row, index) => {
      const faces = candidateFaces(children[index], index === 0);
      return Math.max(...faces.map((face) => row[face + 2] - 0.03 * belief.feedRisk(face)));
    });
    const stock = probabilities.reduce((total, [, chance], index) => total + chance * best[index + 1], 0);
    return { action: best[0] > stock ? 1 : 0, reason: "draw-lookahead" };
  }
  const values = qValues(model, [obs])[0];
  return { action: values[1] > values[0] ? 1 : 0, reason: "greedy" };
}
