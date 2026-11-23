import { evaluateHand, minimumPenalty } from "./evaluate.mjs";
import { chooseMove } from "./ai.mjs";

export class EngineTimeout extends Error {
  constructor(op, timeoutMs) {
    super(`${op} exceeded ${timeoutMs / 1000} seconds`);
    this.name = "EngineTimeout";
    this.op = op;
  }
}

export class EngineDisposed extends Error {
  constructor() {
    super("engine disposed");
    this.name = "EngineDisposed";
  }
}

export function createEngine({ timeoutMs = 10_000 } = {}) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new RangeError("timeoutMs must be positive");
  let worker;
  let disposed = false;
  let lastId = 0;
  const pending = new Map();
  function rejectPending(errorFor) {
    for (const entry of pending.values()) {
      clearTimeout(entry.timer);
      entry.reject(errorFor(entry));
    }
    pending.clear();
  }
  function spawn() {
    worker = new Worker(new URL("./engine-worker.mjs", import.meta.url), { type: "module" });
    worker.onmessage = ({ data }) => {
      const entry = pending.get(data.id);
      if (!entry) return;
      pending.delete(data.id);
      clearTimeout(entry.timer);
      if (data.ok) entry.resolve(data.value);
      else {
        const Class = data.error.name === "TypeError" ? TypeError :
          data.error.name === "RangeError" ? RangeError : Error;
        const error = new Class(data.error.message);
        error.name = data.error.name;
        entry.reject(error);
      }
    };
    worker.onerror = () => {
      rejectPending(() => new Error("engine failed to start"));
      worker.terminate();
      worker = null;
    };
  }
  function call(op, args) {
    if (disposed) return Promise.reject(new EngineDisposed());
    if (!worker) {
      try { spawn(); } catch (error) { return Promise.reject(error); }
    }
    return new Promise((resolve, reject) => {
      const id = ++lastId;
      const timer = setTimeout(() => {
        worker.terminate();
        worker = null;
        rejectPending((entry) => new EngineTimeout(entry.op, timeoutMs));
      }, timeoutMs);
      pending.set(id, { op, timer, resolve, reject });
      try { worker.postMessage({ id, op, ...args }); }
      catch (error) {
        clearTimeout(timer);
        pending.delete(id);
        reject(error);
      }
    });
  }
  spawn();
  return {
    evaluate: (counts, rules) => call("evaluate", { counts, ...rules }),
    penalties: (reveal) => call("penalties", { reveal }),
    chooseMove: (view, tier, seed) => call("chooseMove", { view, tier, seed }),
    dispose() {
      if (disposed) return;
      disposed = true;
      worker?.terminate();
      worker = null;
      rejectPending(() => new EngineDisposed());
    },
  };
}

export function createLocalEngine() {
  let disposed = false;
  const call = (request) => Promise.resolve().then(() => {
    if (disposed) throw new EngineDisposed();
    validateRequest(request);
    return handleRequest(request);
  });
  return {
    evaluate: (counts, rules) => call({ id: 1, op: "evaluate", counts, ...rules }),
    penalties: (reveal) => call({ id: 1, op: "penalties", reveal }),
    chooseMove: (view, tier, seed = 0) => call({ id: 1, op: "chooseMove", view, tier, seed }),
    dispose() { disposed = true; },
  };
}

export function serveEngine(scope) {
  scope.addEventListener("message", ({ data }) => {
    try {
      validateRequest(data);
      scope.postMessage({ id: data.id, ok: true, value: handleRequest(data) });
    } catch (error) {
      scope.postMessage({
        id: Number.isInteger(data?.id) ? data.id : -1,
        ok: false, error: { name: error.name, message: error.message },
      });
    }
  });
}

export const ENGINE_WORKER_SOURCE = 'import { serveEngine } from "./engine.mjs";\nserveEngine(self);\n';

function validateRequest(data) {
  if (!data || typeof data !== "object" || !Number.isInteger(data.id)) {
    throw new TypeError("engine request needs an integer id");
  }
  const fields = {
    evaluate: ["counts", "joker", "requiredSequences", "cardsInHand"],
    penalties: ["reveal"], chooseMove: ["view", "tier", "seed"],
  }[data.op];
  if (!fields) throw new TypeError("unknown engine operation");
  if (fields.some((field) => !(field in data))) throw new TypeError("engine request has missing fields");
  if (data.op === "chooseMove" && (
    !Number.isInteger(data.seed) || data.seed < 0 || data.seed > 0xffffffff
  )) throw new TypeError("chooseMove needs a uint32 seed");
}

function handleRequest(request) {
  if (request.op === "evaluate") {
    return evaluateHand(request.counts, request.joker, request);
  }
  if (request.op === "penalties") {
    const { reveal } = request;
    return reveal.hands.map((hand) => {
      const counts = Array(52).fill(0);
      for (const face of hand.faces) counts[face] += 1;
      return minimumPenalty(counts, reveal.joker, reveal);
    });
  }
  return chooseMove(request.view, request.tier, { random: mulberry32(request.seed) });
}

function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = seed + 0x6d2b79f5 | 0;
    let value = Math.imul(seed ^ seed >>> 15, 1 | seed);
    value = value + Math.imul(value ^ value >>> 7, 61 | value) ^ value;
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}
