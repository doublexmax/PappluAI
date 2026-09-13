import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  createGame,
  discardCard,
  drawCard,
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
    assert.equal(state.rules.cardsInHand, 5);
    assert.equal(state.rules.decks, 1);

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
