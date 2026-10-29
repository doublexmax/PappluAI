/** @typedef {{ id: number, face: number }} Card */
/**
 * @typedef {{
 *   name: string,
 *   hand: Card[],
 *   active: boolean,
 *   roundPenalty: number,
 *   penaltyPoints: number
 * }} Player
 */
/**
 * @typedef {{
 *   card: Card,
 *   ownerIndex: number,
 *   verdict: 'pending' | 'valid' | 'invalid'
 * }} Declaration
 */
/**
 * @typedef {{
 *   players: Player[],
 *   stock: Card[],
 *   discard: Card[],
 *   declarations: Declaration[],
 *   joker: Card,
 *   currentPlayer: number,
 *   phase: 'draw' | 'discard' | 'declaring' | 'finished',
 *   drawnCardId: number | null,
 *   turn: number,
 *   winnerIndex: number | null,
 *   outcomeReason: 'valid-declaration' | 'last-remaining' | 'solo-invalid' | null,
 *   rules: { cardsInHand: number, requiredSequences: number, decks: number, recycleDiscard: boolean },
 *   events: object[]
 * }} GameState
 */

const FACES = 52;
export const DECLARATION_PENALTY = 80;

/**
 * @param {Card[]} cards
 * @returns {number[]}
 */
export function toCounts(cards) {
  const counts = Array(FACES).fill(0);
  for (const card of cards) {
    if (card.face < 0 || card.face >= FACES) {
      throw new Error(`face out of range: ${card.face}`);
    }
    counts[card.face] += 1;
  }
  return counts;
}

/**
 * @param {Card[]} cards
 * @returns {Card[]}
 */
export function sortCards(cards) {
  return cards.slice().sort((a, b) => a.face - b.face || a.id - b.id);
}

/**
 * @param {Card[]} cards
 * @param {number} physicalId
 * @param {number} targetIndex
 * @returns {Card[]}
 */
export function moveCard(cards, physicalId, targetIndex) {
  if (!Number.isInteger(physicalId)) {
    throw new Error("physicalId must be an integer");
  }
  if (!Number.isInteger(targetIndex) || targetIndex < 0 || targetIndex >= cards.length) {
    throw new Error("targetIndex is out of range");
  }
  const sourceIndex = cards.findIndex((card) => card.id === physicalId);
  if (sourceIndex < 0) {
    throw new Error("card is not in the hand");
  }
  const next = cards.slice();
  if (sourceIndex === targetIndex) return next;
  const [card] = next.splice(sourceIndex, 1);
  next.splice(targetIndex, 0, card);
  return next;
}

/**
 * @param {Card[]} cards
 * @param {() => number} random
 */
function shuffle(cards, random) {
  const out = cards.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

/**
 * @param {GameState} state
 * @returns {GameState}
 */
function cloneState(state) {
  return {
    players: state.players.map((player) => ({
      name: player.name,
      hand: player.hand.map((card) => ({ ...card })),
      active: player.active,
      roundPenalty: player.roundPenalty,
      penaltyPoints: player.penaltyPoints,
    })),
    stock: state.stock.map((card) => ({ ...card })),
    discard: state.discard.map((card) => ({ ...card })),
    declarations: state.declarations.map((declaration) => ({
      card: { ...declaration.card },
      ownerIndex: declaration.ownerIndex,
      verdict: declaration.verdict,
    })),
    joker: { ...state.joker },
    currentPlayer: state.currentPlayer,
    phase: state.phase,
    drawnCardId: state.drawnCardId,
    turn: state.turn,
    winnerIndex: state.winnerIndex,
    outcomeReason: state.outcomeReason,
    rules: { ...state.rules },
    events: state.events.slice(),
  };
}

/**
 * @param {Player[]} players
 * @param {number} current
 */
function nextActivePlayer(players, current) {
  for (let step = 1; step <= players.length; step += 1) {
    const candidate = (current + step) % players.length;
    if (players[candidate].active) return candidate;
  }
  throw new Error("no active player remains");
}

/**
 * @param {{
 *   players?: number,
 *   decks?: number,
 *   cardsInHand?: number,
 *   requiredSequences?: number,
 *   penaltyTotals?: number[]
 * }} [options]
 * @param {() => number} [random]
 * @returns {GameState}
 */
export function createGame(
  {
    players = 2,
    decks = 3,
    cardsInHand = 21,
    requiredSequences = 5,
    penaltyTotals,
    recycleDiscard = false,
    names,
  } = {},
  random = Math.random,
) {
  if (!Number.isInteger(players) || players < 1 || players > 6) {
    throw new Error("players must be an integer from 1 to 6");
  }
  if (!Number.isInteger(decks) || decks < 1 || decks > 6) {
    throw new Error("decks must be an integer from 1 to 6");
  }
  if (!Number.isInteger(cardsInHand) || cardsInHand < 3 || cardsInHand > 30) {
    throw new Error("cardsInHand must be an integer from 3 to 30");
  }
  if (
    !Number.isInteger(requiredSequences) ||
    requiredSequences < 0 ||
    requiredSequences > 10
  ) {
    throw new Error("requiredSequences must be an integer from 0 to 10");
  }
  if (
    penaltyTotals !== undefined &&
    (!Array.isArray(penaltyTotals) ||
      penaltyTotals.length !== players ||
      penaltyTotals.some((points) => !Number.isInteger(points) || points < 0))
  ) {
    throw new Error("penaltyTotals must contain one non-negative integer per player");
  }
  if (typeof recycleDiscard !== "boolean") {
    throw new TypeError("recycleDiscard must be a boolean");
  }
  if (names !== undefined && (
    !Array.isArray(names) || names.length !== players ||
    names.some((name) => typeof name !== "string" || !name.trim())
  )) {
    throw new TypeError("names must contain one nonempty string per player");
  }
  const initialPenalties = penaltyTotals || Array(players).fill(0);

  const needed = players * cardsInHand + 2;
  const supply = decks * FACES;
  if (needed > supply) {
    throw new Error(
      `not enough cards: need ${needed} for ${players}×${cardsInHand}+2, have ${supply}`,
    );
  }

  /** @type {Card[]} */
  const built = [];
  let nextId = 0;
  for (let deck = 0; deck < decks; deck += 1) {
    for (let face = 0; face < FACES; face += 1) {
      built.push({ id: nextId, face });
      nextId += 1;
    }
  }

  const deck = shuffle(built, random);
  /** @type {Player[]} */
  const table = [];
  for (let player = 0; player < players; player += 1) {
    table.push({
      name: names ? names[player].trim() : players === 1 ? "You" : `Player ${player + 1}`,
      hand: [],
      active: true,
      roundPenalty: 0,
      penaltyPoints: initialPenalties[player],
    });
  }

  for (let cardIndex = 0; cardIndex < cardsInHand; cardIndex += 1) {
    for (let player = 0; player < players; player += 1) {
      const card = deck.pop();
      if (!card) throw new Error("deck exhausted while dealing");
      table[player].hand.push(card);
    }
  }
  for (const player of table) player.hand = sortCards(player.hand);

  const joker = deck.pop();
  if (!joker) throw new Error("deck exhausted before joker");
  const firstDiscard = deck.pop();
  if (!firstDiscard) throw new Error("deck exhausted before discard");

  return {
    players: table,
    stock: deck,
    discard: [firstDiscard],
    declarations: [],
    joker,
    currentPlayer: 0,
    phase: "draw",
    drawnCardId: null,
    turn: 1,
    winnerIndex: null,
    outcomeReason: null,
    rules: { cardsInHand, requiredSequences, decks, recycleDiscard },
    events: [Object.freeze({
      kind: "deal",
      hands: Object.freeze(table.map((player) => Object.freeze(player.hand.map((card) => card.face)))),
      stock: deck.length,
      top: firstDiscard.face,
    })],
  };
}

/**
 * @param {GameState} state
 * @param {'stock' | 'discard'} source
 * @returns {GameState}
 */
export function drawCard(state, source) {
  if (state.phase !== "draw") {
    throw new Error("cannot draw unless phase is draw");
  }
  if (source !== "stock" && source !== "discard") {
    throw new Error("source must be stock or discard");
  }

  const next = cloneState(state);
  const offeredTop = state.discard.length ? state.discard[state.discard.length - 1].face : null;
  /** @type {Card | undefined} */
  let card;
  if (source === "stock") {
    if (next.stock.length === 0) {
      throw new Error("stock is empty");
    }
    card = next.stock.pop();
  } else {
    if (next.discard.length === 0) {
      throw new Error("discard is empty");
    }
    card = next.discard.pop();
  }
  if (!card) throw new Error("draw failed");

  const player = next.players[next.currentPlayer];
  player.hand.push(card);
  next.phase = "discard";
  next.drawnCardId = card.id;
  appendEvent(next, {
    kind: "draw", seat: next.currentPlayer, source, face: card.face,
    offeredTop, stock: next.stock.length,
  });
  return next;
}

/**
 * @param {GameState} state
 * @param {number} physicalId
 * @returns {GameState}
 */
export function discardCard(state, physicalId, random = Math.random) {
  if (state.phase !== "discard") {
    throw new Error("cannot discard unless phase is discard");
  }
  if (!Number.isInteger(physicalId)) {
    throw new Error("physicalId must be an integer");
  }

  const next = cloneState(state);
  const player = next.players[next.currentPlayer];
  const index = player.hand.findIndex((card) => card.id === physicalId);
  if (index < 0) {
    throw new Error("card is not in the active hand");
  }

  const [card] = player.hand.splice(index, 1);
  next.discard.push(card);

  if (player.hand.length !== next.rules.cardsInHand) {
    throw new Error(
      `hand size ${player.hand.length} after discard, expected ${next.rules.cardsInHand}`,
    );
  }

  next.phase = "draw";
  next.drawnCardId = null;
  next.turn += 1;
  next.currentPlayer = nextActivePlayer(next.players, next.currentPlayer);
  appendEvent(next, { kind: "discard", seat: state.currentPlayer, face: card.face, stock: next.stock.length });
  refillIfNeeded(next, state.currentPlayer, random);
  return next;
}

/**
 * @param {GameState} state
 * @param {number} physicalId
 * @returns {GameState}
 */
export function beginDeclaration(state, physicalId) {
  if (state.phase !== "discard") {
    throw new Error("cannot declare unless phase is discard");
  }
  if (!Number.isInteger(physicalId)) {
    throw new Error("physicalId must be an integer");
  }

  const next = cloneState(state);
  const player = next.players[next.currentPlayer];
  const index = player.hand.findIndex((card) => card.id === physicalId);
  if (index < 0) {
    throw new Error("card is not in the active hand");
  }
  const [card] = player.hand.splice(index, 1);
  if (player.hand.length !== next.rules.cardsInHand) {
    throw new Error(
      `hand size ${player.hand.length} after declaration, expected ${next.rules.cardsInHand}`,
    );
  }

  next.declarations.push({
    card,
    ownerIndex: next.currentPlayer,
    verdict: "pending",
  });
  next.phase = "declaring";
  next.drawnCardId = null;
  appendEvent(next, { kind: "declare", seat: next.currentPlayer });
  return next;
}

/**
 * @param {GameState} state
 * @param {boolean} isValid
 * @returns {GameState}
 */
export function resolveDeclaration(state, isValid, random = Math.random) {
  if (state.phase !== "declaring") {
    throw new Error("no declaration is pending");
  }
  if (typeof isValid !== "boolean") {
    throw new Error("isValid must be a boolean");
  }

  const next = cloneState(state);
  const declaration = next.declarations[next.declarations.length - 1];
  if (!declaration || declaration.verdict !== "pending") {
    throw new Error("no declaration is pending");
  }
  appendEvent(next, { kind: "verdict", seat: declaration.ownerIndex, valid: isValid });

  if (isValid) {
    declaration.verdict = "valid";
    next.phase = "finished";
    next.winnerIndex = declaration.ownerIndex;
    next.outcomeReason = "valid-declaration";
    appendEvent(next, { kind: "finish", winner: next.winnerIndex, reason: next.outcomeReason });
    return next;
  }

  declaration.verdict = "invalid";
  const player = next.players[declaration.ownerIndex];
  player.active = false;
  player.roundPenalty += DECLARATION_PENALTY;
  player.penaltyPoints += DECLARATION_PENALTY;

  const active = next.players
    .map((candidate, index) => ({ candidate, index }))
    .filter(({ candidate }) => candidate.active);
  if (active.length <= 1) {
    next.phase = "finished";
    next.winnerIndex = active.length === 1 ? active[0].index : null;
    next.outcomeReason = active.length === 1 ? "last-remaining" : "solo-invalid";
    appendEvent(next, { kind: "finish", winner: next.winnerIndex, reason: next.outcomeReason });
    return next;
  }

  next.phase = "draw";
  next.turn += 1;
  next.currentPlayer = nextActivePlayer(next.players, declaration.ownerIndex);
  refillIfNeeded(next, declaration.ownerIndex, random);
  return next;
}

function appendEvent(next, event) {
  next.events.push(Object.freeze(event));
}

function refillIfNeeded(next, seat, random) {
  if (!next.rules.recycleDiscard || next.stock.length || next.discard.length <= 1) return;
  next.stock = shuffle(next.discard.slice(0, -1), random);
  next.discard = next.discard.slice(-1);
  appendEvent(next, { kind: "refill", seat, stock: next.stock.length });
}

export function observe(state, seat) {
  if (!Number.isInteger(seat) || seat < 0 || seat >= state.players.length) {
    throw new RangeError("seat must be a player index");
  }
  return {
    seat,
    hand: state.players[seat].hand.map((card) => ({ ...card })),
    drawnCardId: seat === state.currentPlayer ? state.drawnCardId : null,
    pile: state.discard.map((card) => card.face),
    joker: state.joker.face,
    stockCount: state.stock.length,
    phase: state.phase,
    currentPlayer: state.currentPlayer,
    turn: state.turn,
    rules: { ...state.rules },
    players: state.players.map((player, index) => ({
      name: player.name,
      handSize: player.hand.length,
      active: player.active,
      discards: state.events.filter((event) => event.kind === "discard" && event.seat === index).length,
      roundPenalty: player.roundPenalty,
      penaltyPoints: player.penaltyPoints,
    })),
    declarations: state.declarations.map(({ ownerIndex, verdict }) => ({ ownerIndex, verdict })),
    events: state.events.map((event) => {
      if (event.kind === "deal") {
        return { kind: "deal", hand: [...event.hands[seat]], stock: event.stock, top: event.top };
      }
      if (event.kind === "draw" && event.source === "stock" && event.seat !== seat) {
        return { ...event, face: null };
      }
      return { ...event };
    }),
  };
}

export function revealHands(state) {
  if (state.phase !== "finished") {
    throw new Error("hands are revealed only after the round is finished");
  }
  return {
    joker: state.joker.face,
    requiredSequences: state.rules.requiredSequences,
    hands: state.players.map((player, seat) => ({
      seat, name: player.name, faces: player.hand.map((card) => card.face),
      faceDown: state.declarations.find((declaration) => declaration.ownerIndex === seat)?.card.face ?? null,
    })),
  };
}

/**
 * @param {GameState} state
 * @param {number} physicalId
 * @param {number} targetIndex
 * @returns {GameState}
 */
export function reorderHand(state, physicalId, targetIndex) {
  if (state.phase !== "draw" && state.phase !== "discard") {
    throw new Error("cannot reorder during declaration or after the round");
  }
  const next = cloneState(state);
  const player = next.players[next.currentPlayer];
  player.hand = moveCard(player.hand, physicalId, targetIndex);
  return next;
}

/**
 * @param {GameState} state
 * @returns {GameState}
 */
export function sortHand(state) {
  if (state.phase !== "draw" && state.phase !== "discard") {
    throw new Error("cannot sort during declaration or after the round");
  }
  const next = cloneState(state);
  next.players[next.currentPlayer].hand = sortCards(
    next.players[next.currentPlayer].hand,
  );
  return next;
}
