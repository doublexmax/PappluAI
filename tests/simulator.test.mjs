import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  DECLARATION_PENALTY,
  beginDeclaration,
  createGame,
  discardCard,
  drawCard,
  moveCard,
  reorderHand,
  resolveDeclaration,
  sortCards,
  sortHand,
  toCounts,
} from "../web/game.mjs";

function seqRandom(values) {
  let i = 0;
  return () => {
    if (i >= values.length) return 0;
    const v = values[i];
    i += 1;
    return v;
  };
}

function allCards(state) {
  const cards = [];
  for (const p of state.players) cards.push(...p.hand);
  cards.push(...state.stock, ...state.discard, state.joker);
  for (const declaration of state.declarations) cards.push(declaration.card);
  return cards;
}

function freeze(state) {
  return JSON.parse(JSON.stringify(state));
}

describe("createGame", () => {
  it("deals round-robin with conservation and unique ids", () => {
    const state = createGame(
      { players: 2, decks: 1, cardsInHand: 5, requiredSequences: 1 },
      () => 0,
    );
    assert.equal(state.players.length, 2);
    assert.equal(state.players[0].hand.length, 5);
    assert.equal(state.players[1].hand.length, 5);
    assert.equal(state.phase, "draw");
    assert.equal(state.drawnCardId, null);
    assert.equal(state.turn, 1);
    assert.equal(state.currentPlayer, 0);
    assert.ok(state.joker);
    assert.equal(state.discard.length, 1);
    assert.deepEqual(state.declarations, []);
    assert.equal(state.winnerIndex, null);
    assert.equal(state.outcomeReason, null);
    assert.equal(state.rules.cardsInHand, 5);
    assert.equal(state.rules.decks, 1);
    assert.ok(state.players.every((player) => player.active));
    assert.ok(state.players.every((player) => player.roundPenalty === 0));
    assert.ok(state.players.every((player) => player.penaltyPoints === 0));
    assert.ok(
      state.players.every((player) =>
        player.hand.every((card, index, hand) =>
          index === 0 ||
          hand[index - 1].face < card.face ||
          (hand[index - 1].face === card.face && hand[index - 1].id < card.id))),
    );

    const cards = allCards(state);
    assert.equal(cards.length, 52);
    const ids = new Set(cards.map((c) => c.id));
    assert.equal(ids.size, 52);
    const faces = cards.map((c) => c.face).sort((a, b) => a - b);
    assert.deepEqual(
      faces,
      Array.from({ length: 52 }, (_, i) => i),
    );
  });

  it("rejects invalid configs", () => {
    assert.throws(() => createGame({ players: 0 }), /players/);
    assert.throws(() => createGame({ players: 7 }), /players/);
    assert.throws(() => createGame({ decks: 0 }), /decks/);
    assert.throws(() => createGame({ decks: 7 }), /decks/);
    assert.throws(() => createGame({ cardsInHand: 2 }), /cardsInHand/);
    assert.throws(() => createGame({ players: 1.5 }), /players/);
    assert.throws(() => createGame({ cardsInHand: 31 }), /cardsInHand/);
    assert.throws(() => createGame({ requiredSequences: -1 }), /requiredSequences/);
    assert.throws(() => createGame({ requiredSequences: 11 }), /requiredSequences/);
    assert.throws(() => createGame({ requiredSequences: NaN }), /requiredSequences/);
    assert.throws(() => createGame({ players: 2, penaltyTotals: [0] }), /penaltyTotals/);
    assert.throws(() => createGame({ players: 1, penaltyTotals: [-1] }), /penaltyTotals/);
    assert.throws(
      () => createGame({ players: 6, decks: 1, cardsInHand: 21 }),
      /not enough cards/,
    );
  });

  it("supports solo", () => {
    const state = createGame(
      { players: 1, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    assert.equal(state.players.length, 1);
    assert.equal(state.players[0].hand.length, 3);
    assert.equal(allCards(state).length, 52);
  });

  it("carries penalty totals into a same-seat-count deal", () => {
    const state = createGame(
      {
        players: 2,
        decks: 1,
        cardsInHand: 3,
        requiredSequences: 1,
        penaltyTotals: [80, 160],
      },
      () => 0,
    );
    assert.deepEqual(
      state.players.map((player) => player.penaltyPoints),
      [80, 160],
    );
    assert.deepEqual(
      state.players.map((player) => player.roundPenalty),
      [0, 0],
    );
  });
});

describe("immutability", () => {
  it("draw and discard return new state without mutating input", () => {
    const state = createGame(
      { players: 2, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const before = freeze(state);
    const drawn = drawCard(state, "stock");
    assert.deepEqual(freeze(state), before);
    assert.notEqual(drawn, state);
    assert.notEqual(drawn.players[0].hand, state.players[0].hand);

    const beforeDraw = freeze(drawn);
    const discarded = discardCard(drawn, drawn.drawnCardId);
    assert.deepEqual(freeze(drawn), beforeDraw);
    assert.notEqual(discarded, drawn);
  });
});

describe("draw and discard", () => {
  it("draws from stock then can return the drawn card", () => {
    const state = createGame(
      { players: 2, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const stockTop = state.stock[state.stock.length - 1];
    const stockLen = state.stock.length;
    const drawn = drawCard(state, "stock");
    assert.equal(drawn.phase, "discard");
    assert.equal(drawn.drawnCardId, stockTop.id);
    assert.equal(drawn.stock.length, stockLen - 1);
    assert.equal(drawn.players[0].hand.length, 4);
    assert.ok(drawn.players[0].hand.some((c) => c.id === stockTop.id));

    const back = discardCard(drawn, stockTop.id);
    assert.equal(back.phase, "draw");
    assert.equal(back.drawnCardId, null);
    assert.equal(back.players[0].hand.length, 3);
    assert.equal(back.discard[back.discard.length - 1].id, stockTop.id);
    assert.equal(back.currentPlayer, 1);
    assert.equal(back.turn, 2);
  });

  it("draws from discard top", () => {
    const state = createGame(
      { players: 2, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const top = state.discard[state.discard.length - 1];
    const drawn = drawCard(state, "discard");
    assert.equal(drawn.drawnCardId, top.id);
    assert.equal(drawn.discard.length, 0);
    assert.ok(drawn.players[0].hand.some((c) => c.id === top.id));
  });

  it("keeps drawn card and discards another", () => {
    const state = createGame(
      { players: 1, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const keepId = state.stock[state.stock.length - 1].id;
    const dropId = state.players[0].hand[0].id;
    const drawn = drawCard(state, "stock");
    const next = discardCard(drawn, dropId);
    assert.ok(next.players[0].hand.some((c) => c.id === keepId));
    assert.ok(!next.players[0].hand.some((c) => c.id === dropId));
    assert.equal(next.discard[next.discard.length - 1].id, dropId);
    assert.equal(next.players[0].hand.length, 3);
    assert.equal(next.currentPlayer, 0);
  });

  it("selects duplicate faces by physical id", () => {
    const state = createGame(
      { players: 1, decks: 2, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const hand = state.players[0].hand;
    const twinIndex = state.stock.findIndex((card) => card.face === hand[0].face);
    assert.notEqual(twinIndex, -1);
    [hand[1], state.stock[twinIndex]] = [state.stock[twinIndex], hand[1]];
    const drawn = drawCard(state, "stock");
    const face = drawn.players[0].hand[0].face;
    const twins = drawn.players[0].hand.filter((c) => c.face === face);
    assert.ok(twins.length >= 2);
    const drop = twins[0];
    const keep = twins[1];
    const after = discardCard(drawn, drop.id);
    assert.ok(after.players[0].hand.some((c) => c.id === keep.id));
    assert.ok(!after.players[0].hand.some((c) => c.id === drop.id));
  });

  it("rotates players including multiplayer", () => {
    let state = createGame(
      { players: 3, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    for (let i = 0; i < 3; i += 1) {
      assert.equal(state.currentPlayer, i);
      state = drawCard(state, "stock");
      state = discardCard(state, state.drawnCardId);
    }
    assert.equal(state.currentPlayer, 0);
    assert.equal(state.turn, 4);
  });
});

describe("illegal actions and exhaustion", () => {
  it("forbids double draw, discard without draw, empty piles, other hand", () => {
    const state = createGame(
      { players: 2, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    assert.throws(() => discardCard(state, state.players[0].hand[0].id), /discard/);
    const drawn = drawCard(state, "stock");
    assert.throws(() => drawCard(drawn, "stock"), /draw/);
    const otherId = state.players[1].hand[0].id;
    assert.throws(() => discardCard(drawn, otherId), /active hand/);

    let emptyStock = createGame(
      { players: 1, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    while (emptyStock.stock.length > 0) {
      emptyStock = drawCard(emptyStock, "stock");
      emptyStock = discardCard(emptyStock, emptyStock.drawnCardId);
    }
    assert.equal(emptyStock.stock.length, 0);
    assert.throws(() => drawCard(emptyStock, "stock"), /stock is empty/);

    const tookDiscard = drawCard(
      createGame(
        { players: 1, decks: 1, cardsInHand: 3, requiredSequences: 1 },
        () => 0,
      ),
      "discard",
    );
    assert.equal(tookDiscard.discard.length, 0);
    assert.throws(() => drawCard(tookDiscard, "discard"), /draw/);
    const emptyDiscardDraw = freeze(
      createGame(
        { players: 1, decks: 1, cardsInHand: 3, requiredSequences: 1 },
        () => 0,
      ),
    );
    emptyDiscardDraw.discard = [];
    assert.throws(
      () => drawCard(emptyDiscardDraw, "discard"),
      /discard is empty/,
    );
  });
});

describe("whole rounds", () => {
  it("conserves every physical card through repeated stock and pile draws", () => {
    for (const players of [1, 2, 6]) {
      let state = createGame({ players });
      const original = allCards(state).sort((a, b) => a.id - b.id);
      for (let turn = 0; turn < 100; turn += 1) {
        const source = state.stock.length && turn % 3 ? "stock" : "discard";
        state = drawCard(state, source);
        const hand = state.players[state.currentPlayer].hand;
        state = discardCard(state, turn % 2 ? hand[0].id : state.drawnCardId);
        assert.deepEqual(allCards(state).sort((a, b) => a.id - b.id), original);
        assert.equal(new Set(allCards(state).map((card) => card.id)).size, 156);
        assert.ok(state.players.every((player) => player.hand.length === 21));
      }
    }
  });
});

describe("declarations", () => {
  it("places an existing card face down and finishes a valid declaration", () => {
    const initial = createGame(
      { players: 2, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const original = allCards(initial).sort((a, b) => a.id - b.id);
    const drawn = drawCard(initial, "stock");
    const declaredId = drawn.players[0].hand[0].id;
    const pending = beginDeclaration(drawn, declaredId);

    assert.equal(pending.phase, "declaring");
    assert.equal(pending.drawnCardId, null);
    assert.equal(pending.players[0].hand.length, 3);
    assert.equal(pending.declarations.length, 1);
    assert.equal(pending.declarations[0].card.id, declaredId);
    assert.equal(pending.declarations[0].ownerIndex, 0);
    assert.equal(pending.declarations[0].verdict, "pending");
    assert.deepEqual(allCards(pending).sort((a, b) => a.id - b.id), original);

    const finished = resolveDeclaration(pending, true);
    assert.equal(finished.phase, "finished");
    assert.equal(finished.winnerIndex, 0);
    assert.equal(finished.currentPlayer, 0);
    assert.equal(finished.outcomeReason, "valid-declaration");
    assert.equal(finished.declarations[0].verdict, "valid");
    assert.deepEqual(allCards(finished).sort((a, b) => a.id - b.id), original);
  });

  it("eliminates an invalid declarer, adds 80 points, and continues", () => {
    const initial = createGame(
      { players: 3, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const drawn = drawCard(initial, "stock");
    const pending = beginDeclaration(drawn, drawn.drawnCardId);
    const continued = resolveDeclaration(pending, false);

    assert.equal(DECLARATION_PENALTY, 80);
    assert.equal(continued.phase, "draw");
    assert.equal(continued.currentPlayer, 1);
    assert.equal(continued.turn, 2);
    assert.equal(continued.players[0].active, false);
    assert.equal(continued.players[0].roundPenalty, 80);
    assert.equal(continued.players[0].penaltyPoints, 80);
    assert.equal(continued.declarations[0].verdict, "invalid");
    assert.equal(continued.winnerIndex, null);
    assert.equal(continued.outcomeReason, null);
  });

  it("awards the last active player after an invalid declaration", () => {
    const initial = createGame(
      { players: 2, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const drawn = drawCard(initial, "stock");
    const finished = resolveDeclaration(
      beginDeclaration(drawn, drawn.drawnCardId),
      false,
    );

    assert.equal(finished.phase, "finished");
    assert.equal(finished.currentPlayer, 0);
    assert.equal(finished.winnerIndex, 1);
    assert.equal(finished.outcomeReason, "last-remaining");
  });

  it("finishes a solo invalid declaration without a winner", () => {
    const initial = createGame(
      { players: 1, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const drawn = drawCard(initial, "stock");
    const finished = resolveDeclaration(
      beginDeclaration(drawn, drawn.drawnCardId),
      false,
    );

    assert.equal(finished.phase, "finished");
    assert.equal(finished.winnerIndex, null);
    assert.equal(finished.outcomeReason, "solo-invalid");
    assert.equal(finished.players[0].penaltyPoints, 80);
  });

  it("does not award a declaration penalty twice", () => {
    const initial = createGame(
      { players: 3, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const drawn = drawCard(initial, "stock");
    const resolved = resolveDeclaration(
      beginDeclaration(drawn, drawn.drawnCardId),
      false,
    );
    const before = freeze(resolved);

    assert.throws(() => resolveDeclaration(resolved, false), /pending/);
    assert.deepEqual(freeze(resolved), before);
    assert.equal(resolved.players[0].penaltyPoints, 80);
  });

  it("rejects wrong cards and non-boolean verdicts without mutation", () => {
    const initial = createGame(
      { players: 2, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const drawn = drawCard(initial, "stock");
    const beforeDrawn = freeze(drawn);
    assert.throws(
      () => beginDeclaration(drawn, drawn.players[1].hand[0].id),
      /active hand/,
    );
    assert.deepEqual(freeze(drawn), beforeDrawn);

    const pending = beginDeclaration(drawn, drawn.drawnCardId);
    const beforePending = freeze(pending);
    assert.throws(() => resolveDeclaration(pending, 1), /boolean/);
    assert.deepEqual(freeze(pending), beforePending);
  });

  it("blocks hand and turn actions while pending and after finishing", () => {
    const initial = createGame(
      { players: 2, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const drawn = drawCard(initial, "stock");
    const pending = beginDeclaration(drawn, drawn.drawnCardId);
    const pendingCardId = pending.players[0].hand[0].id;
    assert.throws(() => drawCard(pending, "stock"), /draw/);
    assert.throws(() => discardCard(pending, pendingCardId), /discard/);
    assert.throws(() => beginDeclaration(pending, pendingCardId), /declare/);
    assert.throws(() => reorderHand(pending, pendingCardId, 0), /reorder/);
    assert.throws(() => sortHand(pending), /sort/);

    const finished = resolveDeclaration(pending, true);
    assert.throws(() => drawCard(finished, "stock"), /draw/);
    assert.throws(() => discardCard(finished, pendingCardId), /discard/);
    assert.throws(() => beginDeclaration(finished, pendingCardId), /declare/);
    assert.throws(() => reorderHand(finished, pendingCardId, 0), /reorder/);
    assert.throws(() => sortHand(finished), /sort/);
  });

  it("skips eliminated players through several ordinary turns", () => {
    let state = createGame(
      { players: 4, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    state = drawCard(state, "stock");
    state = resolveDeclaration(
      beginDeclaration(state, state.drawnCardId),
      false,
    );
    assert.equal(state.currentPlayer, 1);

    for (const expected of [2, 3, 1]) {
      state = drawCard(state, "stock");
      state = discardCard(state, state.drawnCardId);
      assert.equal(state.currentPlayer, expected);
    }
    assert.equal(state.players[0].active, false);
  });

  it("keeps multiple failed declarations in their own zone", () => {
    let state = createGame(
      { players: 4, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    for (const expectedPlayer of [1, 2]) {
      state = drawCard(state, "stock");
      state = beginDeclaration(state, state.drawnCardId);
      state = resolveDeclaration(state, false);
      assert.equal(state.currentPlayer, expectedPlayer);
      assert.equal(state.phase, "draw");
    }
    assert.equal(state.declarations.length, 2);
    assert.ok(
      state.declarations.every((declaration) => declaration.verdict === "invalid"),
    );
    assert.deepEqual(
      state.players.map((player) => player.penaltyPoints),
      [80, 80, 0, 0],
    );
    assert.equal(allCards(state).length, 52);
    assert.equal(new Set(allCards(state).map((card) => card.id)).size, 52);
  });

  it("keeps inputs immutable across declaration transitions", () => {
    const initial = createGame(
      { players: 3, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const drawn = drawCard(initial, "stock");
    const beforeDrawn = freeze(drawn);
    const pending = beginDeclaration(drawn, drawn.drawnCardId);
    assert.deepEqual(freeze(drawn), beforeDrawn);
    assert.notEqual(pending, drawn);
    assert.notEqual(pending.declarations, drawn.declarations);

    const beforePending = freeze(pending);
    const resolved = resolveDeclaration(pending, false);
    assert.deepEqual(freeze(pending), beforePending);
    assert.notEqual(resolved.players[0], pending.players[0]);
  });
});

describe("card ordering", () => {
  it("moves in both directions without mutating the source", () => {
    const cards = [
      { id: 1, face: 10 },
      { id: 2, face: 20 },
      { id: 3, face: 30 },
    ];
    const before = freeze(cards);
    assert.deepEqual(moveCard(cards, 1, 2).map((card) => card.id), [2, 3, 1]);
    assert.deepEqual(moveCard(cards, 3, 0).map((card) => card.id), [3, 1, 2]);
    assert.deepEqual(moveCard(cards, 1, 0), cards);
    assert.deepEqual(moveCard(cards, 3, 2), cards);
    assert.deepEqual(moveCard(cards, 2, 1), cards);
    assert.deepEqual(freeze(cards), before);
    assert.throws(() => moveCard(cards, 99, 0), /not in the hand/);
    assert.throws(() => moveCard(cards, 1, -1), /out of range/);
    assert.deepEqual(freeze(cards), before);
  });

  it("sorts by suit and rank, then physical id", () => {
    const cards = [
      { id: 5, face: 13 },
      { id: 7, face: 0 },
      { id: 3, face: 13 },
    ];
    assert.deepEqual(sortCards(cards).map((card) => card.id), [7, 3, 5]);
    assert.deepEqual(cards.map((card) => card.id), [5, 7, 3]);
  });

  it("persists manual order through draw, discard, and turn switches", () => {
    let state = createGame(
      { players: 2, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const movedId = state.players[0].hand[0].id;
    state = reorderHand(state, movedId, 2);
    const playerZeroOrder = state.players[0].hand.map((card) => card.id);
    assert.equal(playerZeroOrder[2], movedId);

    state = drawCard(state, "stock");
    assert.deepEqual(
      state.players[0].hand.slice(0, 3).map((card) => card.id),
      playerZeroOrder,
    );
    state = discardCard(state, state.drawnCardId);
    assert.deepEqual(state.players[0].hand.map((card) => card.id), playerZeroOrder);

    state = drawCard(state, "stock");
    state = discardCard(state, state.drawnCardId);
    assert.equal(state.currentPlayer, 0);
    assert.deepEqual(state.players[0].hand.map((card) => card.id), playerZeroOrder);
  });

  it("rejects a missing physical id without mutating game state", () => {
    const state = createGame(
      { players: 1, decks: 1, cardsInHand: 3, requiredSequences: 1 },
      () => 0,
    );
    const before = freeze(state);
    assert.throws(() => reorderHand(state, 999, 0), /not in the hand/);
    assert.deepEqual(freeze(state), before);
  });

  it("restores default order without disturbing identical faces", () => {
    let state = createGame(
      { players: 1, decks: 2, cardsInHand: 4, requiredSequences: 1 },
      () => 0,
    );
    const face = state.players[0].hand[0].face;
    const twinIndex = state.stock.findIndex((card) => card.face === face);
    assert.notEqual(twinIndex, -1);
    [state.players[0].hand[1], state.stock[twinIndex]] = [
      state.stock[twinIndex],
      state.players[0].hand[1],
    ];
    state.players[0].hand = sortCards(state.players[0].hand);
    const twins = state.players[0].hand.filter((card) => card.face === face);
    assert.equal(twins.length, 2);

    const before = freeze(state);
    const sourceIndex = state.players[0].hand.findIndex(
      (card) => card.id === twins[0].id,
    );
    const targetIndex = sourceIndex === 0 ? state.players[0].hand.length - 1 : 0;
    const moved = reorderHand(state, twins[0].id, targetIndex);
    assert.deepEqual(freeze(state), before);
    assert.notDeepEqual(
      moved.players[0].hand.map((card) => card.id),
      state.players[0].hand.map((card) => card.id),
    );
    const sorted = sortHand(moved);
    assert.deepEqual(
      sorted.players[0].hand.map((card) => [card.face, card.id]),
      sortCards(moved.players[0].hand).map((card) => [card.face, card.id]),
    );
  });
});

describe("toCounts", () => {
  it("maps faces to 52-length counts", () => {
    const counts = toCounts([
      { id: 1, face: 0 },
      { id: 2, face: 0 },
      { id: 3, face: 51 },
    ]);
    assert.equal(counts.length, 52);
    assert.equal(counts[0], 2);
    assert.equal(counts[51], 1);
    assert.equal(counts.reduce((a, b) => a + b, 0), 3);
  });
});

describe("deal order uses injected random", () => {
  it("is deterministic for a fixed random sequence", () => {
    const r1 = seqRandom(Array(200).fill(0));
    const r2 = seqRandom(Array(200).fill(0));
    const a = createGame(
      { players: 2, decks: 1, cardsInHand: 4, requiredSequences: 1 },
      r1,
    );
    const b = createGame(
      { players: 2, decks: 1, cardsInHand: 4, requiredSequences: 1 },
      r2,
    );
    assert.deepEqual(freeze(a), freeze(b));
  });
});
