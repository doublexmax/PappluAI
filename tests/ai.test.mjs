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
import { parseQModel } from "../web/qnet.mjs";

const modelFile = JSON.parse(readFileSync(new URL("../web/papplu-model.json", import.meta.url)));
const model = parseQModel(modelFile);
const fixture = JSON.parse(readFileSync(new URL("./fixtures/evaluate-parity.json", import.meta.url)));
const tiers = ["neural", "tactical", "casual"];
function seeded(seed) {
  return () => {
    seed = Math.imul(seed, 1664525) + 1013904223 | 0;
    return (seed >>> 0) / 4294967296;
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
    const choice = chooseMove(view, tier, { model, random: seeded(3) });
    assert.equal(choice.move.type, "declare", tier);
    assert.equal(choice.reason, "exact-win");
    const remaining = view.hand.filter((card) => card.id !== choice.move.cardId);
    assert.deepEqual(choice.move.evaluation, evaluateHand(toCounts(remaining), 7, view.rules));
    assert.equal(choice.move.evaluation.is_valid, true);
  }
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
      assert.deepEqual(chooseMove(a, tier, { model, random: seeded(21) }),
        chooseMove(b, tier, { model, random: seeded(21) }), `${tier} seed ${seed}`);
      assert.deepEqual(chooseMove(guarded, tier, { model, random: seeded(21) }),
        chooseMove(a, tier, { model, random: seeded(21) }));
    }
  }
});

test("observation replay uses relative seats and masks opponent stock faces", () => {
  let state = createGame({ players: 3, cardsInHand: 3, requiredSequences: 1 }, seeded(3));
  state = drawCard(state, "stock");
  state = discardCard(state, state.drawnCardId);
  state = drawCard(state, "discard");
  const obs = observationFromView(observe(state, 1), 60);
  assert.equal(obs.history[0].actor, 0);
  assert.equal(obs.history[1].actor, 2);
  assert.equal(obs.history[1].face, null);
  assert.equal(obs.history[2].kind, 3);
  assert.equal(obs.history[2].handSize, 3);
  assert.equal(obs.history[2].turnsRemaining, 59);
  assert.equal(obs.history.at(-1).actor, 0);
  assert.deepEqual(obs.players.map((player) => player.handSize), [4, 3, 3]);
  assert.deepEqual(obs.players.map((player) => player.nextPlayer), [1, 2, 0]);
  assert.deepEqual(obs.hand, toCounts(state.players[1].hand));
  const belief = new BeliefState(obs);
  assert.equal(belief.stockProbabilities().reduce((total, [, p]) => total + p, 0) > 0.999, true);
  assert.equal(belief.feedRisk(0, 9), 0);
});

test("observation replay matches Python multiplayer history field for field", () => {
  const golden = JSON.parse(readFileSync(new URL("./fixtures/qnet-golden.json", import.meta.url)));
  let checked = 0;
  const expand = (hand) => hand.flatMap((copies, face) => Array(copies).fill(face));
  for (const entry of golden.cases) {
    const obs = entry.observation;
    if (obs.phase === 2 || obs.players.some((player) => !player.active) ||
        obs.history[0].initialHand.reduce((a, b) => a + b, 0) !== obs.config.cardsInHand) continue;
    const view = {
      seat: 0, currentPlayer: 0, hand: expand(obs.hand).map((face, id) => ({ face, id })),
      pile: obs.discardPile, joker: obs.joker, stockCount: obs.stockRemaining,
      phase: obs.phase === 0 ? "draw" : "discard", turn: 1,
      rules: { decks: obs.config.numDecks, ...obs.config },
      players: obs.players.map((player) => ({ handSize: player.handSize, active: player.active })),
      events: obs.history.map((event) => {
        if (event.kind === 0) return {
          kind: "deal", hand: expand(event.initialHand), stock: event.stockRemaining, top: event.offeredTop,
        };
        if (event.kind === 1 || event.kind === 2) return {
          kind: "draw", source: event.kind === 1 ? "stock" : "discard", seat: event.actor,
          face: event.face, offeredTop: event.offeredTop, stock: event.stockRemaining,
        };
        return {
          kind: event.kind === 3 ? "discard" : "refill", seat: event.actor,
          face: event.face, stock: event.stockRemaining,
        };
      }),
    };
    assert.deepEqual(observationFromView(view, obs.config.maxTurns), obs, entry.name);
    checked += 1;
  }
  assert.equal(checked, 12);
});

test("neural reports explicit downgrades and rejects out-of-turn calls", () => {
  const state = createGame({ recycleDiscard: true }, seeded(10));
  const view = observe(state, 0);
  assert.equal(chooseMove(view, "neural").downgrade, "no-model");
  const mismatch = structuredClone(view);
  mismatch.rules.recycleDiscard = false;
  assert.equal(chooseMove(mismatch, "neural", { model }).downgrade, "rules-mismatch");
  assert.throws(() => chooseMove(observe(state, 1), "tactical"), /seat cannot act/);
  assert.throws(() => chooseMove(view, "expert"), /unknown AI tier/);
});

test("all tiers complete legal rounds without false declarations", () => {
  const smallModel = parseQModel({
    ...modelFile, gameConfig: {
      numDecks: 2, cardsInHand: 3, requiredSequences: 0, recycleDiscard: true, maxTurns: 60,
    },
  });
  for (const tier of tiers) {
    for (const seed of [1, 4, 12]) {
      let state = createGame({
        players: 2, decks: 2, cardsInHand: 3, requiredSequences: 0, recycleDiscard: true,
      }, seeded(seed));
      const rng = seeded(seed + 1);
      let moves = 0;
      while (state.phase !== "finished" && moves < 500) {
        const choice = chooseMove(observe(state, state.currentPlayer), tier, { model: smallModel, random: rng });
        assert.equal(choice.tier, tier);
        assert.equal(choice.downgrade, null);
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
