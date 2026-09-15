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
 *   rules: { cardsInHand: number, requiredSequences: number, decks: number }
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
      name: players === 1 ? "You" : `Player ${player + 1}`,
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
    rules: { cardsInHand, requiredSequences, decks },
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
  return next;
}

/**
 * @param {GameState} state
 * @param {number} physicalId
 * @returns {GameState}
 */
export function discardCard(state, physicalId) {
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
  return next;
}

/**
 * @param {GameState} state
 * @param {boolean} isValid
 * @returns {GameState}
 */
export function resolveDeclaration(state, isValid) {
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

  if (isValid) {
    declaration.verdict = "valid";
    next.phase = "finished";
    next.winnerIndex = declaration.ownerIndex;
    next.outcomeReason = "valid-declaration";
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
    return next;
  }

  next.phase = "draw";
  next.turn += 1;
  next.currentPlayer = nextActivePlayer(next.players, declaration.ownerIndex);
  return next;
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
