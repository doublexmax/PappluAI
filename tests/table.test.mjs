import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  createEngine, createLocalEngine, ENGINE_WORKER_SOURCE, EngineDisposed, EngineTimeout, serveEngine,
} from "../web/engine.mjs";
import { evaluateHand, minimumPenalty } from "../web/evaluate.mjs";
import { createTable } from "../web/table.mjs";

const tick = () => new Promise((resolve) => setImmediate(resolve));
function harness(engineFactory = () => createLocalEngine(), random = () => 0.9999) {
  const jobs = [];
  const stored = new Map();
  const table = createTable({
    engineFactory, random,
    storage: { getItem: (key) => stored.get(key), setItem: (key, value) => stored.set(key, value) },
    schedule(fn, ms) {
      const job = { fn, ms, cancelled: false };
      jobs.push(job);
      return job;
    },
    cancel(job) { job.cancelled = true; },
  });
  table.start();
  return {
    table, jobs, stored,
    async advance() {
      while (jobs.length) {
        const job = jobs.shift();
        if (job.cancelled) continue;
        job.fn();
        await tick();
        await tick();
        return true;
      }
      return false;
    },
  };
}

function act(table, type, fields = {}) {
  table.dispatch({ type, stamp: table.getSnapshot().round.stamp, ...fields });
}

function humanDeal(table, rules = {}) {
  table.dispatch({ type: "preset", name: "solo" });
  table.dispatch({ type: "configure", patch: { decks: 1, cardsInHand: 3, requiredSequences: 1, ...rules } });
  table.dispatch({ type: "deal" });
}

test("presets, per-seat settings, storage, and frozen snapshots belong to the table", () => {
  const { table, stored } = harness();
  assert.deepEqual(table.getSnapshot().settings.seats.map((seat) => seat.kind), ["human", "ai"]);
  assert.equal(table.getSnapshot().settings.recycleDiscard, true);
  for (const [name, count] of [["solo", 1], ["pass-and-play", 2], ["ai-watch", 3], ["vs-ai", 2]]) {
    table.dispatch({ type: "preset", name });
    assert.equal(table.getSnapshot().settings.seats.length, count);
  }
  table.dispatch({ type: "configure", patch: { seats: [
    { kind: "human", name: " Alice " }, { kind: "ai", name: "Bot", tier: "casual" },
  ], recycleDiscard: false } });
  table.dispatch({ type: "deal" });
  const snapshot = table.getSnapshot();
  assert.equal(snapshot.settings.preset, "custom");
  assert.equal(snapshot.round.seats[0].name, "Alice");
  assert.equal(snapshot.round.seats[1].tier, "casual");
  assert.equal(snapshot.round.seats[1].hand, null);
  assert.equal(snapshot.round.recycleDiscard, false);
  assert.equal(Object.isFrozen(snapshot.round.seats[0].hand[0]), true);
  assert.equal(JSON.parse(stored.get("papplu:settings:v1")).seats[0].name, "Alice");
  table.dispatch({ type: "configure", patch: { seats: [{ kind: "ai", name: "Next game", tier: "tactical" }] } });
  assert.equal(table.getSnapshot().round.seats[0].kind, "human");
  assert.equal(table.getSnapshot().round.seats.length, 2);
  table.stop();
});

test("pass-and-play covers hands between turns but not between draw and discard", () => {
  const { table } = harness();
  table.dispatch({ type: "preset", name: "pass-and-play" });
  table.dispatch({ type: "deal" });
  assert.equal(table.getSnapshot().round.viewer.mode, "cover");
  assert.ok(table.getSnapshot().round.seats.every((seat) => seat.hand === null));
  const stamp = table.getSnapshot().round.stamp;
  act(table, "draw", { source: "stock" });
  assert.equal(table.getSnapshot().round.stamp, stamp);
  act(table, "reveal");
  assert.equal(table.getSnapshot().round.viewer.mode, "hand");
  act(table, "draw", { source: "stock" });
  const round = table.getSnapshot().round;
  assert.equal(round.viewer.mode, "hand");
  assert.equal(round.seats[0].hand.length, 22);
  table.dispatch({ type: "draw", source: "stock", stamp });
  assert.equal(table.getSnapshot().round.seats[0].hand.length, 22);
  act(table, "discard", { cardId: round.seats[0].drawnCardId });
  assert.equal(table.getSnapshot().round.viewer.mode, "cover");
  assert.equal(table.getSnapshot().round.currentSeat, 1);
  assert.ok(table.getSnapshot().round.seats.every((seat) => seat.hand === null));
  table.stop();
});

test("human declarations resolve asynchronously and score every final hand", async () => {
  const { table } = harness();
  humanDeal(table);
  act(table, "draw", { source: "stock" });
  const cardId = table.getSnapshot().round.seats[0].drawnCardId;
  act(table, "declare", { cardId });
  assert.equal(table.getSnapshot().round.phase, "declaring");
  assert.equal(table.getSnapshot().round.declaration.status, "checking");
  await tick();
  const round = table.getSnapshot().round;
  assert.equal(round.phase, "finished");
  assert.equal(round.declaration.valid, true);
  assert.equal(round.summary.status, "ready");
  const row = round.summary.rows[0];
  const counts = Array(52).fill(0);
  for (const face of row.faces) counts[face] += 1;
  assert.deepEqual(row.penalty, minimumPenalty(counts, round.joker, { requiredSequences: 1 }));
  assert.equal(row.faceDown, 46);
  assert.equal(row.roundPenalty, 0);
  table.stop();
});

test("an evaluation error or timeout leaves a committed declaration retryable without a penalty", async () => {
  let failures = 2;
  const local = createLocalEngine();
  const { table } = harness(() => ({
    ...local,
    evaluate(...args) {
      if (failures-- > 0) return Promise.reject(failures === 1 ? new EngineTimeout("evaluate", 10000) : new Error("offline"));
      return local.evaluate(...args);
    },
  }));
  humanDeal(table, { requiredSequences: 2 });
  act(table, "draw", { source: "stock" });
  act(table, "declare", { cardId: table.getSnapshot().round.seats[0].drawnCardId });
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await tick();
    const round = table.getSnapshot().round;
    assert.equal(round.phase, "declaring");
    assert.equal(round.declaration.status, "error");
    assert.match(round.declaration.message, /No penalty was applied/);
    assert.equal(round.seats[0].handSize, 3);
    assert.equal(round.seats[0].penaltyPoints, 0);
    table.dispatch({ type: "retry", what: "declaration" });
  }
  await tick();
  assert.equal(table.getSnapshot().round.phase, "finished");
  assert.equal(table.getSnapshot().round.seats[0].penaltyPoints, 80);
  table.dispatch({ type: "deal" });
  assert.equal(table.getSnapshot().round.seats[0].penaltyPoints, 80);
  table.stop();
});

test("verified AI declarations resolve synchronously without an evaluation round trip", async () => {
  let checks = 0;
  const local = createLocalEngine();
  const h = harness(() => ({
    ...local,
    evaluate() { checks += 1; throw new Error("AI must not evaluate again"); },
  }));
  h.table.dispatch({ type: "configure", patch: {
    seats: [{ kind: "ai", name: "Bot", tier: "casual" }],
    decks: 1, cardsInHand: 3, requiredSequences: 1, aiSpeed: "instant",
  } });
  h.table.dispatch({ type: "deal" });
  assert.equal(h.table.getSnapshot().round.viewer.mode, "spectator");
  assert.equal(await h.advance(), true);
  assert.equal(await h.advance(), true);
  const round = h.table.getSnapshot().round;
  assert.equal(round.phase, "finished");
  assert.equal(round.declaration.evaluation.is_valid, true);
  assert.equal(round.summary.status, "ready");
  assert.equal(checks, 0);
  h.table.stop();
});

test("AI pause, stale async replies, and stop-start preserve a single turn writer", async () => {
  let resolveMove;
  let calls = 0;
  const h = harness(() => ({
    ...createLocalEngine(),
    chooseMove() {
      calls += 1;
      return new Promise((resolve) => { resolveMove = resolve; });
    },
  }));
  h.table.dispatch({ type: "preset", name: "ai-watch" });
  h.table.dispatch({ type: "deal" });
  await h.advance();
  assert.equal(calls, 1);
  h.table.dispatch({ type: "pause" });
  resolveMove({ move: { type: "draw", source: "stock" }, tier: "casual", downgrade: null, reason: "forced" });
  await tick();
  assert.equal(h.table.getSnapshot().round.phase, "draw");
  assert.equal(h.table.getSnapshot().paused, true);
  assert.equal(await h.advance(), false);
  h.table.dispatch({ type: "resume" });
  await h.advance();
  assert.equal(calls, 2);
  h.table.stop();
  h.table.stop();
  h.table.start();
  h.table.start();
  await h.advance();
  assert.equal(calls, 3);
  h.table.stop();
});

test("AI Watch pauses after 200 completed turns and resumes on demand", async () => {
  const h = harness(() => ({
    ...createLocalEngine(),
    chooseMove(view, tier) {
      return Promise.resolve({
        move: view.phase === "draw" ? { type: "draw", source: "discard" } :
          { type: "discard", cardId: view.drawnCardId },
        tier, downgrade: null, reason: "forced",
      });
    },
  }));
  h.table.dispatch({ type: "preset", name: "ai-watch" });
  h.table.dispatch({ type: "configure", patch: { aiSpeed: "instant" } });
  h.table.dispatch({ type: "deal" });
  for (let step = 0; step < 400; step += 1) assert.equal(await h.advance(), true, `step ${step}`);
  const snapshot = h.table.getSnapshot();
  assert.equal(snapshot.round.turn, 201);
  assert.equal(snapshot.paused, true);
  assert.equal(snapshot.notice, "Paused after 200 turns. Resume to keep watching.");
  h.table.dispatch({ type: "resume" });
  assert.equal(h.table.getSnapshot().paused, false);
  assert.equal(await h.advance(), true);
  h.table.stop();
});

test("a continuing invalid declaration with no stock or pile stalls without an AI job", async () => {
  let choices = 0;
  const h = harness(() => ({
    ...createLocalEngine(),
    chooseMove() { choices += 1; throw new Error("a stalled table cannot ask for a move"); },
  }));
  h.table.dispatch({ type: "configure", patch: {
    seats: [{ kind: "human", name: "You" }, ...Array.from({ length: 4 }, (_, index) => ({
      kind: "ai", name: `Bot ${index}`, tier: "casual",
    }))],
    decks: 1, cardsInHand: 10, requiredSequences: 10,
  } });
  h.table.dispatch({ type: "deal" });
  act(h.table, "draw", { source: "discard" });
  act(h.table, "declare", { cardId: h.table.getSnapshot().round.seats[0].drawnCardId });
  await tick();
  const snapshot = h.table.getSnapshot();
  assert.equal(snapshot.round.phase, "draw");
  assert.equal(snapshot.round.stockCount, 0);
  assert.equal(snapshot.round.pileCount, 0);
  assert.match(snapshot.notice, /No legal move/);
  assert.equal(await h.advance(), false);
  assert.equal(choices, 0);
  h.table.stop();
});

test("hand checks drop stale responses and report timeouts as errors rather than losing verdicts", async () => {
  const waiting = [];
  const h = harness(() => ({
    ...createLocalEngine(),
    evaluate: () => new Promise((resolve, reject) => waiting.push({ resolve, reject })),
  }));
  humanDeal(h.table);
  h.table.dispatch({ type: "check-hand", on: true });
  assert.equal(h.table.getSnapshot().round.handCheck.status, "checking");
  act(h.table, "draw", { source: "stock" });
  waiting[0].resolve({ is_valid: true, melds: [] });
  await tick();
  assert.equal(h.table.getSnapshot().round.handCheck.status, "unavailable");
  act(h.table, "discard", { cardId: h.table.getSnapshot().round.seats[0].drawnCardId });
  waiting[1].reject(new EngineTimeout("evaluate", 10000));
  await tick();
  assert.equal(h.table.getSnapshot().round.handCheck.status, "error");
  assert.match(h.table.getSnapshot().round.handCheck.message, /not a losing-hand verdict/);
  h.table.stop();
});

test("local engine and worker protocol return the same evaluator values and mapped errors", async () => {
  const engine = createLocalEngine();
  const counts = Array(52).fill(0);
  counts[0] = counts[1] = counts[2] = 1;
  const rules = { joker: 7, requiredSequences: 1, cardsInHand: 3 };
  const expected = evaluateHand(counts, 7, rules);
  assert.deepEqual(await engine.evaluate(counts, rules), expected);
  await assert.rejects(engine.evaluate([], rules), RangeError);
  const replies = [];
  let receive;
  serveEngine({ addEventListener(_, listener) { receive = listener; }, postMessage(data) { replies.push(data); } });
  receive({ data: { id: 7, op: "evaluate", counts, ...rules } });
  assert.deepEqual(replies[0], { id: 7, ok: true, value: expected });
  receive({ data: { id: 8, op: "evaluate", counts: [], ...rules } });
  assert.equal(replies[1].error.name, "RangeError");
  receive({ data: { id: 9, op: "unknown" } });
  assert.equal(replies[2].error.name, "TypeError");
  engine.dispose();
  engine.dispose();
  await assert.rejects(engine.evaluate(counts, rules), EngineDisposed);
  assert.equal(readFileSync(new URL("../web/engine-worker.mjs", import.meta.url), "utf8").replace(/\r\n/g, "\n"),
    ENGINE_WORKER_SOURCE);
});

test("worker engine times out pending calls, restarts, and rejects calls after dispose", async () => {
  const original = globalThis.Worker;
  const workers = [];
  class FakeWorker {
    constructor(url, options) {
      assert.equal(url.pathname.endsWith("/engine-worker.mjs"), true);
      assert.deepEqual(options, { type: "module" });
      this.stopped = false;
      workers.push(this);
      serveEngine({
        addEventListener: (_, receive) => { this.receive = receive; },
        postMessage: (data) => this.onmessage?.({ data }),
      });
    }
    postMessage(data) {
      if (data.op === "init" || this.hang) return;
      queueMicrotask(() => { if (!this.stopped) this.receive({ data }); });
    }
    terminate() { this.stopped = true; }
  }
  globalThis.Worker = FakeWorker;
  try {
    const engine = createEngine({ timeoutMs: 25 });
    workers[0].hang = true;
    await assert.rejects(engine.evaluate(Array(52).fill(0), {
      joker: 7, requiredSequences: 1, cardsInHand: 3,
    }), EngineTimeout);
    assert.equal(workers[0].stopped, true);
    assert.deepEqual(await engine.evaluate(Array(52).fill(0), {
      joker: 7, requiredSequences: 1, cardsInHand: 3,
    }), { is_valid: false, melds: [] });
    assert.equal(workers.length, 2);
    await assert.rejects(engine.evaluate([], { joker: 7, requiredSequences: 1, cardsInHand: 3 }), RangeError);
    engine.dispose();
    await assert.rejects(engine.evaluate([], {}), EngineDisposed);
  } finally { globalThis.Worker = original; }
});
