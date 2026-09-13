/** @typedef {{ id: number, face: number }} Card */
/** @typedef {{ name: string, hand: Card[] }} Player */
/**
 * @typedef {{
 *   players: Player[],
 *   stock: Card[],
 *   discard: Card[],
 *   joker: Card,
 *   currentPlayer: number,
 *   phase: 'draw' | 'discard',
 *   drawnCardId: number | null,
 *   turn: number,
 *   rules: { cardsInHand: number, requiredSequences: number, decks: number }
 * }} GameState
 */

const FACES = 52;

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
    players: state.players.map((p) => ({
      name: p.name,
      hand: p.hand.map((c) => ({ id: c.id, face: c.face })),
    })),
    stock: state.stock.map((c) => ({ id: c.id, face: c.face })),
    discard: state.discard.map((c) => ({ id: c.id, face: c.face })),
    joker: { id: state.joker.id, face: state.joker.face },
    currentPlayer: state.currentPlayer,
    phase: state.phase,
    drawnCardId: state.drawnCardId,
    turn: state.turn,
    rules: { ...state.rules },
  };
}

/**
 * @param {{
 *   players?: number,
 *   decks?: number,
 *   cardsInHand?: number,
 *   requiredSequences?: number,
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
  for (let d = 0; d < decks; d += 1) {
    for (let face = 0; face < FACES; face += 1) {
      built.push({ id: nextId, face });
      nextId += 1;
    }
  }

  const deck = shuffle(built, random);
  /** @type {Player[]} */
  const table = [];
  for (let p = 0; p < players; p += 1) {
    table.push({ name: players === 1 ? "You" : `Player ${p + 1}`, hand: [] });
  }

  for (let i = 0; i < cardsInHand; i += 1) {
    for (let p = 0; p < players; p += 1) {
      const card = deck.pop();
      if (!card) throw new Error("deck exhausted while dealing");
      table[p].hand.push(card);
    }
  }

  const joker = deck.pop();
  if (!joker) throw new Error("deck exhausted before joker");
  const firstDiscard = deck.pop();
  if (!firstDiscard) throw new Error("deck exhausted before discard");

  return {
    players: table,
    stock: deck,
    discard: [firstDiscard],
    joker,
    currentPlayer: 0,
    phase: "draw",
    drawnCardId: null,
    turn: 1,
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
  const index = player.hand.findIndex((c) => c.id === physicalId);
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
  next.currentPlayer = (next.currentPlayer + 1) % next.players.length;
  return next;
}
