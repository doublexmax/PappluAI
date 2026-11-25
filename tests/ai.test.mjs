import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  BeliefState, canDeclareUpperBound, chooseMove, observationFromView, winningDiscard,
} from "../web/ai.mjs";
import {
  beginDeclaration, createGame, discardCard, drawCard, observe, resolveDeclaration, toCounts,
} from "../web/game.mjs";
import { evaluateHand } from "../web/evaluate.mjs";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/evaluate-parity.json", import.meta.url)));
const tiers = ["tactical", "casual"];
function seeded(seed) {
  return () => {
    seed = Math.imul(seed, 1664525) + 1013904223 | 0;
    return (seed >>> 0) / 4294967296;
  };
}

function policyView(cards, phase = "discard") {
  const drawing = phase === "draw";
  const initial = drawing ? cards : cards.slice(0, -1);
  return {
    seat: 0, currentPlayer: 0, phase, joker: 7, stockCount: 100, pile: [2],
    hand: cards.map((face, id) => ({ face, id })),
    rules: { decks: 3, cardsInHand: initial.length, requiredSequences: 1, recycleDiscard: true },
    players: [{ handSize: cards.length, active: true }, { handSize: initial.length, active: true }],
    events: [
      { kind: "deal", hand: initial, stock: 101, top: 2 },
      ...(drawing ? [] : [{
        kind: "draw", source: "stock", seat: 0, face: cards.at(-1), offeredTop: 2, stock: 100,
      }]),
    ],
  };
}

test("winning discard and sequence upper bounds match the Python search fixtures", () => {
  for (const entry of fixture.cases) {
    const rules = { requiredSequences: entry.required_sequences, cardsInHand: entry.cards_in_hand };
    if (Object.hasOwn(entry, "can_declare_ub")) {
      assert.equal(canDeclareUpperBound(entry.hand, entry.joker, rules.requiredSequences),
        entry.can_declare_ub, entry.name);
    }
    if (Object.hasOwn(entry, "winning_discard")) {
      assert.equal(winningDiscard(entry.hand, entry.joker, rules)?.face ?? null,
        entry.winning_discard, entry.name);
    }
  }
});

test("every tier declares an exact win with a valid snake_case witness", () => {
  let state = createGame({ players: 2, cardsInHand: 3, requiredSequences: 1 }, seeded(9));
  state = drawCard(state, "stock");
  const view = observe(state, 0);
  view.hand = [0, 1, 2, 9].map((face, id) => ({ face, id }));
  view.joker = 7;
  for (const tier of tiers) {
    const choice = chooseMove(view, tier, { random: seeded(3) });
    assert.equal(choice.move.type, "declare", tier);
    assert.equal(choice.move.cardId, 3);
    assert.equal(choice.reason, "exact-win");
    assert.equal(choice.tier, tier);
    const remaining = view.hand.filter((card) => card.id !== choice.move.cardId);
    assert.deepEqual(choice.move.evaluation, evaluateHand(toCounts(remaining), 7, view.rules));
    assert.equal(choice.move.evaluation.is_valid, true);
  }
});

test("tactical takes the top discard when it permits an exact win", () => {
  const view = policyView([0, 1, 9], "draw");
  assert.deepEqual(chooseMove(view, "tactical"), {
    move: { type: "draw", source: "discard" }, tier: "tactical", reason: "exact-win",
  });
});

test("casual takes the top discard only when enough cards match it", () => {
  assert.deepEqual(chooseMove(policyView([0, 1, 9], "draw"), "casual"), {
    move: { type: "draw", source: "discard" }, tier: "casual", reason: "keep-score",
  });
  assert.deepEqual(chooseMove(policyView([0, 5, 30], "draw"), "casual"), {
    move: { type: "draw", source: "stock" }, tier: "casual", reason: "keep-score",
  });
});

test("casual uses the supplied randomness between equally weak discards", () => {
  const view = policyView([0, 5, 30, 18]);
  assert.deepEqual(chooseMove(view, "casual", { random: () => 0 }), {
    move: { type: "discard", cardId: 0 }, tier: "casual", reason: "greedy",
  });
  assert.deepEqual(chooseMove(view, "casual", { random: () => 0.75 }), {
    move: { type: "discard", cardId: 2 }, tier: "casual", reason: "greedy",
  });
});

test("all tiers avoid solo self-discard pickup without blocking an empty-stock draw", () => {
  const view = policyView([0, 1, 30], "draw");
  view.players = [{ handSize: 3, active: true }];
  view.events.push(
    { kind: "draw", source: "stock", seat: 0, face: 2, offeredTop: 2, stock: 100 },
    { kind: "discard", seat: 0, face: 2, stock: 100 },
  );
  for (const tier of tiers) {
    const choice = chooseMove(view, tier);
    assert.deepEqual(choice.move, { type: "draw", source: "stock" }, tier);
    assert.equal(choice.reason, "forced", tier);
  }
  view.stockCount = 0;
  for (const tier of tiers) {
    const choice = chooseMove(view, tier);
    assert.deepEqual(choice.move, { type: "draw", source: "discard" }, tier);
    assert.equal(choice.reason, "forced", tier);
  }
});

test("tactical discards the minimum-penalty face", () => {
  assert.deepEqual(chooseMove(policyView([0, 5, 30, 18]), "tactical"), {
    move: { type: "discard", cardId: 0 }, tier: "tactical", reason: "min-penalty",
  });
});

test("hidden opponent hands and stock order cannot change a tier's decision", () => {
  for (let seed = 1; seed <= 8; seed += 1) {
    let state = createGame({ players: 3, recycleDiscard: true }, seeded(seed));
    if (seed % 2) state = drawCard(state, "stock");
    const swapped = structuredClone(state);
    for (let index = 0; index < swapped.players[1].hand.length; index += 1) {
      [swapped.players[1].hand[index], swapped.players[2].hand[index]] =
        [swapped.players[2].hand[index], swapped.players[1].hand[index]];
    }
    swapped.stock.reverse();
    const a = observe(state, 0);
    const b = observe(swapped, 0);
    assert.deepEqual(a, b);
    for (const tier of tiers) {
      const guarded = structuredClone(a);
      for (const player of guarded.players) {
        Object.defineProperty(player, "hand", { get() { throw new Error("opponent hand accessed"); } });
      }
      assert.deepEqual(chooseMove(a, tier, { random: seeded(21) }),
        chooseMove(b, tier, { random: seeded(21) }), `${tier} seed ${seed}`);
      assert.deepEqual(chooseMove(guarded, tier, { random: seeded(21) }),
        chooseMove(a, tier, { random: seeded(21) }));
    }
  }
});

test("observation replay uses relative seats and masks opponent stock faces", () => {
  let state = createGame({ players: 3, cardsInHand: 3, requiredSequences: 1 }, seeded(3));
  state = drawCard(state, "stock");
  state = discardCard(state, state.drawnCardId);
  state = drawCard(state, "discard");
  const obs = observationFromView(observe(state, 1));
  assert.equal(obs.history[0].actor, 0);
  assert.equal(obs.history[1].actor, 2);
  assert.equal(obs.history[1].face, null);
  assert.equal(obs.history[2].kind, 3);
  assert.equal(obs.history.at(-1).actor, 0);
  assert.deepEqual(obs.hand, toCounts(state.players[1].hand));
  const belief = new BeliefState(obs);
  assert.equal(belief.stockProbabilities().reduce((total, [, p]) => total + p, 0) > 0.999, true);
  assert.equal(belief.feedRisk(0, 9), 0);
});

test("AI rejects unsupported tiers and out-of-turn calls", () => {
  const state = createGame({ recycleDiscard: true }, seeded(10));
  assert.throws(() => chooseMove(observe(state, 0), "neural"), { name: "TypeError", message: "unknown AI tier" });
  assert.throws(() => chooseMove(observe(state, 1), "tactical"), /seat cannot act/);
});

test("all tiers complete legal rounds without false declarations", () => {
  for (const tier of tiers) {
    for (const seed of [1, 4, 12]) {
      let state = createGame({
        players: 2, decks: 2, cardsInHand: 3, requiredSequences: 0, recycleDiscard: true,
      }, seeded(seed));
      const rng = seeded(seed + 1);
      let moves = 0;
      while (state.phase !== "finished" && moves < 500) {
        const choice = chooseMove(observe(state, state.currentPlayer), tier, { random: rng });
        assert.equal(choice.tier, tier);
        const move = choice.move;
        if (move.type === "draw") state = drawCard(state, move.source);
        else if (move.type === "discard") state = discardCard(state, move.cardId, rng);
        else {
          assert.equal(move.evaluation.is_valid, true);
          state = resolveDeclaration(beginDeclaration(state, move.cardId), true);
        }
        moves += 1;
      }
      assert.equal(state.phase, "finished", `${tier} seed ${seed}, ${moves} moves`);
      assert.equal(state.outcomeReason, "valid-declaration");
      assert.equal(state.players.reduce((sum, player) => sum + player.roundPenalty, 0), 0);
    }
  }
});
