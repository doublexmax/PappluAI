import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { cpus } from "node:os";
import { performance } from "node:perf_hooks";
import { observationFromView } from "../web/ai.mjs";
import { createGame, discardCard, drawCard, observe } from "../web/game.mjs";
import { parseQModel, qValues } from "../web/qnet.mjs";

const bytes = readFileSync(process.argv[2] ?? new URL("../web/papplu-model.json", import.meta.url));
const model = parseQModel(JSON.parse(bytes));
const runs = Number(process.argv[3] ?? 5);
assert.ok(Number.isInteger(runs) && runs > 0, "runs must be a positive integer");
const { numDecks, cardsInHand, requiredSequences, maxTurns, recycleDiscard } = model.gameConfig;
const maxPlayers = Math.min(6, Math.floor((52 * numDecks - 2) / cardsInHand));
const observations = [];
for (let deal = 0; observations.length < 1000; deal += 1) {
  let seed = 1729 + deal;
  const random = () => {
    seed = Math.imul(seed, 1664525) + 1013904223 | 0;
    return (seed >>> 0) / 4294967296;
  };
  const players = 1 + deal % maxPlayers;
  let state = createGame({ players, decks: numDecks, cardsInHand, requiredSequences, recycleDiscard }, random);
  for (let turn = 0; turn < Math.min(30, maxTurns * players) && observations.length < 1000; turn += 1) {
    observations.push(observationFromView(observe(state, state.currentPlayer), maxTurns));
    const source = state.discard.length && turn % 3 === 0 || !state.stock.length ? "discard" : "stock";
    const taken = source === "discard" ? state.discard.at(-1).face : null;
    state = drawCard(state, source);
    if (observations.length < 1000) observations.push(observationFromView(observe(state, state.currentPlayer), maxTurns));
    const hand = state.players[state.currentPlayer].hand;
    const others = hand.filter(card => card.face !== taken);
    const choices = others.length ? others : hand;
    state = discardCard(state, choices[Math.floor(random() * choices.length)].id);
  }
}
assert.equal(new Set(observations.map(obs => JSON.stringify(obs))).size, 1000);
for (let i = 0; i < 200; i += 1) qValues(model, [observations[i]]);

const results = [];
let checksum = 0;
for (let run = 0; run < runs; run += 1) {
  const milliseconds = [];
  for (let i = 0; i < observations.length; i += 1) {
    const started = performance.now();
    const q = qValues(model, [observations[i]])[0];
    milliseconds.push(performance.now() - started);
    assert.equal(q.length, 54);
    assert.ok(q.every(Number.isFinite));
    checksum += q[i % 54];
  }
  milliseconds.sort((a, b) => a - b);
  results.push({
    median_ms: (milliseconds[499] + milliseconds[500]) / 2,
    p95_ms: milliseconds[949],
    minimum_ms: milliseconds[0],
    maximum_ms: milliseconds[999],
  });
}
assert.ok(Number.isFinite(checksum));
const medians = results.map(row => row.median_ms).sort((a, b) => a - b);
const p95s = results.map(row => row.p95_ms).sort((a, b) => a - b);
const middle = values => (values[Math.floor((values.length - 1) / 2)] + values[Math.floor(values.length / 2)]) / 2;
console.log(JSON.stringify({
  node: process.version, platform: process.platform, arch: process.arch, cpu: cpus()[0].model,
  model_spec: model.spec, model_file_sha256: createHash("sha256").update(bytes).digest("hex"),
  observations_sha256: createHash("sha256").update(JSON.stringify(observations)).digest("hex"),
  samples_per_run: observations.length, runs, warmup_calls: 200, failures: 0, checksum,
  draw_observations: observations.filter(obs => obs.phase === 0).length,
  take_discard_observations: observations.filter(obs => obs.history.at(-1).kind === 2).length,
  timed_work: "qValues with validation, encoding, and forward computation; excludes model load and observationFromView",
  median_ms: middle(medians), p95_ms: middle(p95s),
  median_range_ms: [medians[0], medians.at(-1)], p95_range_ms: [p95s[0], p95s.at(-1)],
  results,
}, null, 2));
