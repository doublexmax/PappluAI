import assert from "node:assert/strict";
import test from "node:test";
import {
  beginDeclaration, createGame, discardCard, drawCard, observe, resolveDeclaration, revealHands,
} from "../web/game.mjs";

function cards(state) {
  return [
    ...state.players.flatMap((player) => player.hand), ...state.stock, ...state.discard,
    state.joker, ...state.declarations.map((entry) => entry.card),
  ].map((card) => card.id).sort((a, b) => a - b);
}

function exhaust(options) {
  let state = createGame({ players: 3, decks: 1, cardsInHand: 3, ...options }, () => 0.6);
  while (state.stock.length) {
    state = drawCard(state, "stock");
    if (!state.stock.length) return state;
    state = discardCard(state, state.drawnCardId);
  }
}

test("events are append-only and seat views mask every private zone", () => {
  const initial = createGame({ names: ["Alice", "Bob"] }, () => 0.5);
  const before = structuredClone(initial);
  const drawn = drawCard(initial, "stock");
  const discarded = discardCard(drawn, drawn.drawnCardId);
  assert.deepEqual(initial, before);
  assert.deepEqual(discarded.events.map((event) => event.kind), ["deal", "draw", "discard"]);
  assert.equal(Object.isFrozen(drawn.events[0].hands[0]), true);
  const view = observe(drawn, 1);
  assert.equal("hands" in view.events[0], false);
  assert.deepEqual(view.events[0].hand, initial.players[1].hand.map((card) => card.face));
  assert.equal(view.events[1].face, null);
  assert.equal(observe(drawn, 0).events[1].face, drawn.players[0].hand.at(-1).face);
  assert.equal("stock" in view, false);
  assert.equal("hand" in view.players[0], false);
  const pending = beginDeclaration(drawn, drawn.drawnCardId);
  assert.deepEqual(observe(pending, 1).declarations, [{ ownerIndex: 0, verdict: "pending" }]);
  assert.deepEqual(pending.events.at(-1), { kind: "declare", seat: 0 });
  view.hand[0].face = 99;
  view.events[0].hand[0] = 99;
  assert.deepEqual(initial, before);
});

test("refill follows a continuing discard, preserves the top, and conserves all cards", () => {
  const drawn = exhaust({ recycleDiscard: true });
  const before = structuredClone(drawn);
  const discarded = drawn.players[drawn.currentPlayer].hand.at(-1);
  const next = discardCard(drawn, discarded.id, () => 0.3);
  assert.equal(next.discard.length, 1);
  assert.equal(next.discard[0].id, discarded.id);
  assert.ok(next.stock.length > 1);
  assert.deepEqual(next.events.slice(-2).map((event) => event.kind), ["discard", "refill"]);
  assert.deepEqual(cards(next), cards(drawn));
  assert.deepEqual(drawn, before);
  const disabled = exhaust({});
  assert.equal(discardCard(disabled, disabled.drawnCardId).stock.length, 0);
  assert.equal(discardCard(disabled, disabled.drawnCardId).events.at(-1).kind, "discard");
});

test("only a continuing invalid declaration refills the stock", () => {
  const drawn = exhaust({ recycleDiscard: true });
  const pending = beginDeclaration(drawn, drawn.drawnCardId);
  const continuing = resolveDeclaration(pending, false, () => 0.2);
  assert.equal(continuing.phase, "draw");
  assert.ok(continuing.stock.length > 0);
  assert.equal(continuing.events.at(-1).kind, "refill");
  assert.deepEqual(cards(continuing), cards(drawn));
  const finished = resolveDeclaration(pending, true);
  assert.equal(finished.stock.length, 0);
  assert.deepEqual(finished.events.slice(-2).map((event) => event.kind), ["verdict", "finish"]);
  assert.throws(() => revealHands(pending), /only after/);
  const reveal = revealHands(finished);
  assert.equal(reveal.hands[0].faces.length, 3);
  assert.equal(reveal.hands[pending.currentPlayer].faceDown, pending.declarations[0].card.face);
});

test("one-card piles cannot refill and validation rejects malformed settings or seats", () => {
  const state = createGame();
  assert.equal(state.rules.recycleDiscard, false);
  assert.throws(() => createGame({ recycleDiscard: 1 }), TypeError);
  assert.throws(() => createGame({ names: ["Only one"] }), TypeError);
  for (const seat of [-1, 2, 0.5]) assert.throws(() => observe(state, seat), RangeError);
  let empty = createGame({ players: 1, decks: 1, cardsInHand: 3, recycleDiscard: true });
  while (empty.stock.length) {
    empty = drawCard(empty, "stock");
    if (empty.stock.length) {
      empty = discardCard(empty, empty.drawnCardId);
      empty = drawCard(empty, "discard");
      empty = discardCard(empty, empty.drawnCardId);
    }
  }
  empty.discard = [];
  const next = discardCard(empty, empty.drawnCardId);
  assert.equal(next.stock.length, 0);
  assert.equal(next.discard.length, 1);
});
