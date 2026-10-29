import {
  beginDeclaration, createGame, discardCard, drawCard, observe, reorderHand,
  resolveDeclaration, revealHands, sortHand, toCounts,
} from "./game.mjs";

const SETTINGS_KEY = "papplu:settings:v1";
const AI_SPEED_MS = { instant: 0, normal: 600, slow: 1200 };
const TIERS = ["neural", "tactical", "casual"];
const STALL_NOTICE = "No legal move. The stock and the pile are both empty. Deal again to continue.";
const PRESETS = {
  "vs-ai": [{ kind: "human", name: "You" }, { kind: "ai", name: "Neural AI", tier: "neural" }],
  solo: [{ kind: "human", name: "You" }],
  "pass-and-play": [{ kind: "human", name: "Player 1" }, { kind: "human", name: "Player 2" }],
  "ai-watch": [
    { kind: "ai", name: "Neural", tier: "neural" },
    { kind: "ai", name: "Tactical", tier: "tactical" },
    { kind: "ai", name: "Casual", tier: "casual" },
  ],
};

function defaultSettings() {
  return {
    preset: "vs-ai", seats: PRESETS["vs-ai"].map((seat) => ({ ...seat })),
    decks: 3, cardsInHand: 21, requiredSequences: 5, recycleDiscard: true, aiSpeed: "normal",
  };
}

function parseSettings(raw) {
  const defaults = defaultSettings();
  if (typeof raw === "string") {
    try { raw = JSON.parse(raw); } catch { return defaults; }
  }
  if (!raw || typeof raw !== "object") return defaults;
  const validSeats = Array.isArray(raw.seats) && raw.seats.length >= 1 && raw.seats.length <= 6 &&
    raw.seats.every((seat) => seat && ["human", "ai"].includes(seat.kind) &&
      typeof seat.name === "string" && seat.name.trim() &&
      (seat.kind === "human" || TIERS.includes(seat.tier)));
  const settings = {
    ...defaults,
    seats: validSeats ? raw.seats.map((seat) => ({
      kind: seat.kind, name: seat.name.trim().slice(0, 24),
      ...(seat.kind === "ai" ? { tier: seat.tier } : {}),
    })) : defaults.seats,
  };
  for (const [key, min, max] of [["decks", 1, 6], ["cardsInHand", 3, 30], ["requiredSequences", 0, 10]]) {
    if (Number.isInteger(raw[key]) && raw[key] >= min && raw[key] <= max) settings[key] = raw[key];
  }
  if (typeof raw.recycleDiscard === "boolean") settings.recycleDiscard = raw.recycleDiscard;
  if (Object.hasOwn(AI_SPEED_MS, raw.aiSpeed)) settings.aiSpeed = raw.aiSpeed;
  if (raw.preset === "custom" || Object.hasOwn(PRESETS, raw.preset)) settings.preset = raw.preset;
  if (settings.seats.length * settings.cardsInHand + 2 > settings.decks * 52) {
    settings.decks = defaults.decks;
    settings.cardsInHand = defaults.cardsInHand;
    settings.requiredSequences = defaults.requiredSequences;
  }
  return settings;
}

function freeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}

export function createTable({
  engineFactory, storage = null, random = Math.random, schedule = setTimeout, cancel = clearTimeout,
}) {
  let stored = null;
  try { stored = storage?.getItem(SETTINGS_KEY); } catch {}
  let settings = parseSettings(stored);
  let game = null;
  let roundSeats = [];
  let dealId = 0;
  let epoch = 0;
  let aiJob = null;
  let engine = null;
  let started = false;
  let paused = false;
  let notice = null;
  let revealedTurn = null;
  let checkOn = false;
  let checkKey = null;
  let checkRevision = 0;
  let watchTurns = 0;
  let lastChoices = [];
  let declaration = { status: "none" };
  let handCheck = { status: "off" };
  let summary = { status: "none" };
  let snapshot;
  const listeners = new Set();
  const stamp = () => game ? `d${dealId}.t${game.turn}.${game.phase}` : null;
  const turnKey = () => `d${dealId}.t${game.turn}`;
  const stalled = () => game?.phase === "draw" && !game.stock.length && !game.discard.length;
  const persist = () => {
    try { storage?.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch {}
  };

  function viewer() {
    const humans = roundSeats.flatMap((seat, index) => seat.kind === "human" ? [index] : []);
    if (!humans.length) return { mode: "spectator" };
    if (humans.length === 1) return { mode: "seat", seat: humans[0] };
    if (roundSeats[game.currentPlayer].kind === "ai") return { mode: "table" };
    return { mode: revealedTurn === turnKey() ? "hand" : "cover", seat: game.currentPlayer };
  }

  function visibleCheck() {
    if (!checkOn) return { status: "off" };
    if (!game) return { status: "unavailable", message: "Deal a hand to evaluate it." };
    const eyes = viewer();
    if (!["seat", "hand"].includes(eyes.mode)) {
      return { status: "unavailable", message: "Hand hidden. Reveal to evaluate the active player." };
    }
    if (game.players[eyes.seat].hand.length !== game.rules.cardsInHand) {
      return { status: "unavailable", message: `Discard one card to evaluate your final ${game.rules.cardsInHand}-card hand.` };
    }
    return handCheck;
  }

  function toSnapshot() {
    let round = null;
    if (game) {
      const eyes = viewer();
      const human = roundSeats[game.currentPlayer].kind === "human";
      const canAct = started && !paused && !stalled() && human && ["draw", "discard"].includes(game.phase) &&
        ["seat", "hand"].includes(eyes.mode) && eyes.seat === game.currentPlayer;
      round = {
        stamp: stamp(), phase: game.phase, turn: game.turn, currentSeat: game.currentPlayer,
        viewer: eyes, canAct, joker: game.joker.face,
        pileTop: game.discard.at(-1)?.face ?? null, pileCount: game.discard.length,
        stockCount: game.stock.length, recycleDiscard: game.rules.recycleDiscard,
        seats: game.players.map((player, index) => {
          const visible = eyes.mode === "spectator" ||
            (["seat", "hand"].includes(eyes.mode) && eyes.seat === index);
          const choice = lastChoices[index];
          return {
            ...roundSeats[index], tier: roundSeats[index].tier ?? null,
            effectiveTier: choice?.tier ?? null, downgrade: choice?.downgrade ?? null,
            handSize: player.hand.length, active: player.active,
            roundPenalty: player.roundPenalty, penaltyPoints: player.penaltyPoints,
            thinking: aiJob?.seat === index && aiJob.running,
            lastReason: choice?.reason ?? null,
            hand: visible ? player.hand.map((card) => ({ ...card })) : null,
            drawnCardId: visible && index === game.currentPlayer ? game.drawnCardId : null,
          };
        }),
        declaration, handCheck: visibleCheck(), summary,
        outcome: game.phase === "finished" ? { reason: game.outcomeReason, winner: game.winnerIndex } : null,
      };
    }
    return freeze({ settings, paused, notice: notice ?? (stalled() ? STALL_NOTICE : null), round });
  }

  function commit() {
    snapshot = toSnapshot();
    for (const listener of listeners) listener();
    pump();
  }

  function cancelAi() {
    if (aiJob && !aiJob.running) cancel(aiJob.handle);
    aiJob = null;
  }

  function pump() {
    if (!started || !game) return;
    if (game.phase === "declaring" && declaration.status === "none") void runDeclaration();
    if (game.phase === "finished" && summary.status === "none") void runSummary();
    if (checkOn) void runHandCheck();
    if (paused || aiJob || stalled() || !["draw", "discard"].includes(game.phase) ||
        roundSeats[game.currentPlayer].kind !== "ai") return;
    const job = { seat: game.currentPlayer, stamp: stamp(), epoch, running: false, handle: null };
    aiJob = job;
    job.handle = schedule(() => void runAi(job), AI_SPEED_MS[settings.aiSpeed]);
  }

  async function runAi(job) {
    if (aiJob !== job || !started || paused) return;
    job.running = true;
    commit();
    let choice = null;
    let move;
    let failure = null;
    try {
      choice = await engine.chooseMove(observe(game, job.seat), roundSeats[job.seat].tier,
        Math.floor(random() * 4294967296));
      move = choice.move;
    } catch (error) {
      if (error.name === "EngineDisposed") return;
      failure = error;
    }
    if (aiJob !== job || epoch !== job.epoch || stamp() !== job.stamp || !started || paused) return;
    aiJob = null;
    if (failure) {
      const view = observe(game, job.seat);
      move = view.phase === "draw" ? { type: "draw", source: view.stockCount ? "stock" : "discard" } :
        { type: "discard", cardId: view.drawnCardId ?? view.hand.at(-1).id };
      notice = `${roundSeats[job.seat].name} ${failure.name === "EngineTimeout" ? "took too long" : "could not choose a move"}, so it played a safe move.`;
    }
    lastChoices[job.seat] = choice;
    applyMove(move, true);
    if (move.type !== "draw" && roundSeats.every((seat) => seat.kind === "ai")) {
      watchTurns += 1;
      if (watchTurns >= 200 && game.phase !== "finished") {
        paused = true;
        notice = "Paused after 200 turns. Resume to keep watching.";
      }
    }
    commit();
  }

  function applyMove(move, ai = false) {
    try {
      if (move.type === "draw") game = drawCard(game, move.source);
      else if (move.type === "discard") game = discardCard(game, move.cardId, random);
      else if (move.type === "declare") {
        if (ai && move.evaluation?.is_valid !== true) throw new Error("AI declaration needs a verified winning evaluation");
        const pending = beginDeclaration(game, move.cardId);
        if (ai) {
          const seat = game.currentPlayer;
          const faceDown = pending.declarations.at(-1).card.face;
          game = resolveDeclaration(pending, true, random);
          declaration = { status: "resolved", seat, valid: true, evaluation: move.evaluation, faceDown };
        } else {
          game = pending;
          declaration = { status: "none" };
        }
      } else throw new Error("unknown move");
      checkKey = null;
      handCheck = { status: checkOn ? "checking" : "off" };
      checkRevision += 1;
    } catch (error) {
      notice = error.message;
      if (ai) paused = true;
    }
  }

  async function runDeclaration() {
    if (!started || game.phase !== "declaring" || declaration.status === "checking") return;
    const pending = game.declarations.at(-1);
    const seat = pending.ownerIndex;
    const expected = stamp();
    const revision = epoch;
    declaration = { status: "checking", seat };
    commit();
    try {
      const evaluation = await engine.evaluate(toCounts(game.players[seat].hand), {
        joker: game.joker.face, ...game.rules,
      });
      if (revision !== epoch || expected !== stamp() || !started) return;
      game = resolveDeclaration(game, evaluation.is_valid, random);
      declaration = {
        status: "resolved", seat, valid: evaluation.is_valid, evaluation, faceDown: pending.card.face,
      };
    } catch (error) {
      if (revision !== epoch || expected !== stamp() || !started) return;
      declaration = {
        status: "error", seat,
        message: `Declaration check failed. ${error.message}. No penalty was applied. Retry uses the same committed hand.`,
      };
    }
    checkKey = null;
    checkRevision += 1;
    commit();
  }

  async function runSummary() {
    if (!started || game.phase !== "finished" || summary.status === "scoring") return;
    const reveal = revealHands(game);
    const revision = epoch;
    summary = { status: "scoring" };
    commit();
    try {
      const penalties = await engine.penalties(reveal);
      if (revision !== epoch || !started) return;
      summary = {
        status: "ready",
        rows: reveal.hands.map((hand, index) => ({
          ...hand, penalty: penalties[index],
          roundPenalty: game.players[hand.seat].roundPenalty,
          penaltyPoints: game.players[hand.seat].penaltyPoints,
        })),
      };
    } catch (error) {
      if (revision !== epoch || !started) return;
      summary = { status: "error", message: `Round summary failed. ${error.message}` };
    }
    commit();
  }

  async function runHandCheck() {
    const visible = visibleCheck();
    if (visible.status === "off" || visible.status === "unavailable") return;
    const seat = viewer().seat;
    const hand = game.players[seat].hand;
    const key = `${stamp()}.${seat}.${hand.map((card) => card.id).join(",")}`;
    if (key === checkKey) return;
    checkKey = key;
    const revision = ++checkRevision;
    const generation = epoch;
    handCheck = { status: "checking" };
    commit();
    try {
      const evaluation = await engine.evaluate(toCounts(hand), { joker: game.joker.face, ...game.rules });
      if (revision !== checkRevision || generation !== epoch || !started || !checkOn) return;
      handCheck = { status: "ready", evaluation };
    } catch (error) {
      if (revision !== checkRevision || generation !== epoch || !started || !checkOn) return;
      handCheck = {
        status: "error", message: error.name === "EngineTimeout" ?
          "Evaluation exceeded 10 seconds. Try fewer wildcards or a smaller hand. This is not a losing-hand verdict." :
          error.message,
      };
    }
    commit();
  }

  function dispatch(intent) {
    if (!intent || typeof intent !== "object") return;
    if (intent.type === "preset") {
      if (!Object.hasOwn(PRESETS, intent.name)) return;
      settings = parseSettings({
        ...settings, preset: intent.name, seats: PRESETS[intent.name],
        ...(["vs-ai", "ai-watch"].includes(intent.name) ? { decks: 3, cardsInHand: 21, requiredSequences: 5 } : {}),
      });
      persist();
    } else if (intent.type === "configure") {
      settings = parseSettings({ ...settings, ...intent.patch, preset: "custom" });
      persist();
    } else if (intent.type === "deal") {
      cancelAi();
      epoch += 1;
      const sameCount = game?.players.length === settings.seats.length;
      notice = game && !sameCount ? "Penalty points reset because the player count changed." : null;
      game = createGame({
        ...settings, players: settings.seats.length, names: settings.seats.map((seat) => seat.name),
        penaltyTotals: sameCount ? game.players.map((player) => player.penaltyPoints) : undefined,
      }, random);
      roundSeats = settings.seats.map((seat) => ({ ...seat }));
      dealId += 1;
      lastChoices = [];
      paused = false;
      revealedTurn = null;
      declaration = { status: "none" };
      summary = { status: "none" };
      handCheck = { status: checkOn ? "checking" : "off" };
      checkKey = null;
      checkRevision += 1;
      watchTurns = 0;
    } else if (intent.type === "pause") {
      paused = true;
      cancelAi();
    } else if (intent.type === "resume") {
      paused = false;
      notice = null;
      watchTurns = 0;
    } else if (intent.type === "check-hand") {
      checkOn = Boolean(intent.on);
      checkKey = null;
      checkRevision += 1;
      handCheck = { status: checkOn ? "checking" : "off" };
    } else if (intent.type === "retry") {
      if (!started || !game) return;
      if (intent.what === "declaration" && declaration.status === "error") void runDeclaration();
      else if (intent.what === "summary" && summary.status === "error") void runSummary();
      else if (intent.what === "hand-check" && handCheck.status === "error") {
        checkKey = null;
        void runHandCheck();
      }
    } else {
      if (!game || intent.stamp !== stamp()) return;
      const eyes = viewer();
      if (intent.type === "reveal") {
        if (eyes.mode !== "cover" || paused) return;
        revealedTurn = turnKey();
      } else {
        if (!snapshot.round.canAct) return;
        notice = null;
        if (["draw", "discard", "declare"].includes(intent.type)) applyMove(intent);
        else {
          try {
            if (intent.type === "arrange") game = reorderHand(game, intent.cardId, intent.index);
            else if (intent.type === "sort-hand") game = sortHand(game);
            else return;
          } catch (error) { notice = error.message; }
        }
      }
    }
    commit();
  }

  snapshot = toSnapshot();
  return {
    getSnapshot: () => snapshot,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    dispatch,
    start() {
      if (started) return;
      engine = engineFactory();
      started = true;
      commit();
    },
    stop() {
      if (!started) return;
      cancelAi();
      epoch += 1;
      checkRevision += 1;
      checkKey = null;
      if (declaration.status === "checking") declaration = { status: "none" };
      if (summary.status === "scoring") summary = { status: "none" };
      engine.dispose();
      engine = null;
      started = false;
      snapshot = toSnapshot();
    },
  };
}
