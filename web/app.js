import {
  createGame,
  discardCard,
  drawCard,
  toCounts,
} from "/game.mjs";

const SUITS = ["♠", "♥", "♦", "♣"];
const SUIT_NAMES = ["Spades", "Hearts", "Diamonds", "Clubs"];
const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
const RED_SUITS = new Set([1, 2]);
const DEBOUNCE_MS = 180;

/** @typedef {{ id: number, face: number }} Card */

const el = {
  status: document.getElementById("status"),
  tabBuilder: document.getElementById("tab-builder"),
  tabTable: document.getElementById("tab-table"),
  panelBuilder: document.getElementById("panel-builder"),
  panelTable: document.getElementById("panel-table"),
  builderHandSize: document.getElementById("builder-hand-size"),
  builderRequired: document.getElementById("builder-required"),
  builderDecks: document.getElementById("builder-decks"),
  builderJokerSuit: document.getElementById("builder-joker-suit"),
  builderJokerRank: document.getElementById("builder-joker-rank"),
  builderSample: document.getElementById("builder-sample"),
  builderLoadSample: document.getElementById("builder-load-sample"),
  builderClear: document.getElementById("builder-clear"),
  builderGallery: document.getElementById("builder-gallery"),
  builderGalleryHint: document.getElementById("builder-gallery-hint"),
  builderHand: document.getElementById("builder-hand"),
  builderHandCount: document.getElementById("builder-hand-count"),
  builderEval: document.getElementById("builder-eval"),
  builderRetry: document.getElementById("builder-retry"),
  tablePlayers: document.getElementById("table-players"),
  tableDecks: document.getElementById("table-decks"),
  tableHandSize: document.getElementById("table-hand-size"),
  tableRequired: document.getElementById("table-required"),
  tableDeal: document.getElementById("table-deal"),
  tableBoard: document.getElementById("table-board"),
  tableEval: document.getElementById("table-eval"),
  tableRetry: document.getElementById("table-retry"),
};

const builder = {
  /** @type {Card[]} */
  hand: [],
  nextId: 1,
  selectedSample: "classic21",
  lastConfig: null,
};

const table = {
  /** @type {import('./game.mjs').GameState | null} */
  state: null,
  selectedId: /** @type {number | null} */ (null),
  revealed: true,
  started: false,
};

const evalCtl = {
  timer: /** @type {ReturnType<typeof setTimeout> | null} */ (null),
  inflight: false,
  /** @type {null | { mode: string, hand: number[], joker: number, required: number, cards: number, rev: number }} */
  queued: null,
  rev: 0,
  /** @type {string} */
  mode: "builder",
  lastError: /** @type {string | null} */ (null),
};

function faceOf(suit, rank) {
  return suit * 13 + rank;
}

function suitOf(face) {
  return Math.floor(face / 13);
}

function rankOf(face) {
  return face % 13;
}

function labelFace(face) {
  return `${RANKS[rankOf(face)]}${SUITS[suitOf(face)]}`;
}

function setStatus(msg) {
  el.status.textContent = msg || "";
}

function readInt(input) {
  const n = Number(input.value);
  if (!input.value || !input.checkValidity() || !Number.isInteger(n)) {
    const label = input.labels[0].firstChild.textContent.trim();
    throw new Error(`${label} must be an integer from ${input.min} to ${input.max}.`);
  }
  return n;
}

function builderConfig() {
  return {
    cardsInHand: readInt(el.builderHandSize),
    requiredSequences: readInt(el.builderRequired),
    decks: readInt(el.builderDecks),
    joker: faceOf(
      readInt(el.builderJokerSuit),
      readInt(el.builderJokerRank),
    ),
  };
}

function writeBuilderConfig(cfg) {
  el.builderHandSize.value = String(cfg.cardsInHand);
  el.builderRequired.value = String(cfg.requiredSequences);
  el.builderDecks.value = String(cfg.decks);
  el.builderJokerSuit.value = String(suitOf(cfg.joker));
  el.builderJokerRank.value = String(rankOf(cfg.joker));
}

function badgeFor(face, jokerFace) {
  if (face === jokerFace) return { text: "exact", cls: "exact" };
  if (rankOf(face) === rankOf(jokerFace)) return { text: "wild", cls: "wild" };
  return null;
}

/**
 * @param {number} face
 * @param {{ jokerFace?: number, physicalId?: number, selected?: boolean, drawn?: boolean, compact?: boolean, title?: string }} [opts]
 */
function cardNode(face, opts = {}) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "playing-card";
  if (RED_SUITS.has(suitOf(face))) btn.classList.add("red");
  if (opts.selected) btn.classList.add("selected");
  if (opts.drawn) btn.classList.add("drawn");
  if (opts.compact) btn.classList.add("compact");
  btn.dataset.face = String(face);
  if (opts.physicalId != null) btn.dataset.id = String(opts.physicalId);
  const rank = document.createElement("span");
  rank.className = "rank";
  rank.textContent = RANKS[rankOf(face)];
  const suit = document.createElement("span");
  suit.className = "suit";
  suit.textContent = SUITS[suitOf(face)];
  btn.append(rank, suit);
  if (opts.jokerFace != null) {
    const badge = badgeFor(face, opts.jokerFace);
    if (badge) {
      const b = document.createElement("span");
      b.className = `badge ${badge.cls}`;
      b.textContent = badge.text;
      btn.append(b);
    }
  }
  btn.title = opts.title || labelFace(face);
  btn.setAttribute(
    "aria-label",
    opts.title ||
      `${RANKS[rankOf(face)]} of ${SUIT_NAMES[suitOf(face)]}${
        opts.physicalId != null ? `, id ${opts.physicalId}` : ""
      }`,
  );
  return btn;
}

function faceButton(face, used, maxCopies, jokerFace) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "face-btn";
  if (RED_SUITS.has(suitOf(face))) btn.classList.add("red");
  const rank = document.createElement("span");
  rank.className = "rank";
  rank.textContent = RANKS[rankOf(face)];
  const suit = document.createElement("span");
  suit.className = "suit";
  suit.textContent = SUITS[suitOf(face)];
  const count = document.createElement("span");
  count.className = "copy-count";
  count.textContent = `${used}/${maxCopies}`;
  btn.append(rank, suit, count);
  const badge = badgeFor(face, jokerFace);
  if (badge) {
    const b = document.createElement("span");
    b.className = `badge ${badge.cls}`;
    b.textContent = badge.text;
    btn.append(b);
  }
  btn.disabled = used >= maxCopies;
  btn.dataset.face = String(face);
  btn.setAttribute(
    "aria-label",
    `Add ${RANKS[rankOf(face)]} of ${SUIT_NAMES[suitOf(face)]}, ${used} of ${maxCopies} used`,
  );
  return btn;
}

function classicValid21Faces() {
  /** @type {number[]} */
  const cards = [];
  for (let suit = 0; suit < 3; suit += 1) {
    cards.push(faceOf(suit, 0), faceOf(suit, 1), faceOf(suit, 2));
  }
  cards.push(faceOf(0, 3), faceOf(0, 4), faceOf(0, 5));
  cards.push(faceOf(1, 6), faceOf(1, 7), faceOf(1, 8));
  for (let suit = 0; suit < 3; suit += 1) cards.push(faceOf(suit, 9));
  for (let suit = 0; suit < 3; suit += 1) cards.push(faceOf(suit, 10));
  return cards;
}

function sampleFaces(name) {
  if (name === "classic21") return classicValid21Faces();
  if (name === "jokerRun") {
    return [faceOf(0, 7), faceOf(0, 7), faceOf(0, 7)];
  }
  if (name === "naturalSet") {
    return [faceOf(0, 9), faceOf(1, 9), faceOf(2, 9)];
  }
  if (name === "invalidNoise") {
    const faces = classicValid21Faces();
    faces[faces.indexOf(faceOf(0, 5))] = faceOf(3, 12);
    return faces;
  }
  const base = [0, 1, 2, 26, 27, 28, 39, 40, 41, 10, 11, 12, 8, 34, 47, 25, 38, 51];
  if (name === "exactJoker21") return [...base, 16, 17, 7];
  if (name === "offSuitJoker21") return [...base, 16, 17, 33];
  if (name === "naturalJoker21") return [...base, 19, 20, 21];
  if (name === "duplicateSet21") return [...base.slice(0, 15), 16, 17, 18, 25, 25, 33];
  throw new Error(`Unknown sample: ${name}`);
}

function loadSample(name) {
  if (!name) {
    setStatus("Choose a sample first.");
    return;
  }
  writeBuilderConfig({
    cardsInHand: ["jokerRun", "naturalSet"].includes(name) ? 3 : 21,
    requiredSequences: name === "jokerRun" ? 1 : name === "naturalSet" ? 0 : 5,
    decks: 3,
    joker: 7,
  });
  const faces = sampleFaces(name);
  builder.hand = faces.map((face) => {
    const card = { id: builder.nextId, face };
    builder.nextId += 1;
    return card;
  });
  builder.selectedSample = name;
  setStatus(`Loaded sample: ${el.builderSample.selectedOptions[0]?.text || name}`);
  renderBuilder();
  scheduleEval("builder");
}

function renderBuilder() {
  const cfg = builderConfig();
  builder.lastConfig = cfg;

  const used = toCounts(builder.hand);
  el.builderGallery.replaceChildren();
  for (let suit = 0; suit < 4; suit += 1) {
    const row = document.createElement("div");
    row.className = "suit-row";
    row.setAttribute("aria-label", SUIT_NAMES[suit]);
    for (let rank = 0; rank < 13; rank += 1) {
      const face = faceOf(suit, rank);
      const btn = faceButton(face, used[face], cfg.decks, cfg.joker);
      btn.addEventListener("click", () => addBuilderFace(face));
      row.append(btn);
    }
    el.builderGallery.append(row);
  }
  el.builderGalleryHint.textContent = `Up to ${cfg.decks} copies per face. Hand target ${cfg.cardsInHand}. Joker ${labelFace(cfg.joker)}.`;

  el.builderHand.replaceChildren();
  const sorted = builder.hand
    .slice()
    .sort((a, b) => a.face - b.face || a.id - b.id);
  for (const card of sorted) {
    const node = cardNode(card.face, {
      jokerFace: cfg.joker,
      physicalId: card.id,
      title: `${labelFace(card.face)} (remove)`,
    });
    node.addEventListener("click", () => removeBuilderCard(card.id));
    el.builderHand.append(node);
  }
  el.builderHandCount.textContent = `${builder.hand.length} / ${cfg.cardsInHand}`;
}

function addBuilderFace(face) {
  const cfg = builderConfig();
  if (builder.hand.length >= cfg.cardsInHand) {
    setStatus(`Hand is full (${cfg.cardsInHand}). Remove a card first.`);
    return;
  }
  const used = builder.hand.filter((c) => c.face === face).length;
  if (used >= cfg.decks) {
    setStatus(`No more copies of ${labelFace(face)} (decks=${cfg.decks}).`);
    return;
  }
  builder.hand.push({ id: builder.nextId, face });
  builder.nextId += 1;
  setStatus(`Added ${labelFace(face)}.`);
  renderBuilder();
  scheduleEval("builder");
}

function removeBuilderCard(id) {
  const idx = builder.hand.findIndex((c) => c.id === id);
  if (idx < 0) return;
  const [removed] = builder.hand.splice(idx, 1);
  setStatus(`Removed ${labelFace(removed.face)}.`);
  renderBuilder();
  scheduleEval("builder");
}

function clearBuilderHand() {
  builder.hand = [];
  setStatus("Hand cleared.");
  renderBuilder();
  scheduleEval("builder");
}

function setMode(mode) {
  const builderOn = mode === "builder";
  el.tabBuilder.setAttribute("aria-selected", builderOn ? "true" : "false");
  el.tabTable.setAttribute("aria-selected", builderOn ? "false" : "true");
  el.panelBuilder.classList.toggle("hidden", !builderOn);
  el.panelBuilder.hidden = !builderOn;
  el.panelTable.classList.toggle("hidden", builderOn);
  el.panelTable.hidden = builderOn;
  evalCtl.mode = mode;
  if (builderOn) {
    renderBuilder();
    scheduleEval("builder");
  } else {
    renderTable();
    scheduleEval("table");
  }
}

/**
 * @param {'builder' | 'table'} mode
 * @param {HTMLElement} target
 * @param {{ hand: Card[], joker: number, required: number, cardsInHand: number, hidden?: boolean, problem?: string | null, needsDiscard?: boolean }} snap
 */
function paintEvalState(mode, target, snap) {
  if (snap.hidden) {
    target.innerHTML =
      '<p class="muted">Hand hidden. Reveal to evaluate the active player.</p>';
    return;
  }
  if (snap.problem) {
    target.innerHTML = `<p class="eval-wait">${escapeHtml(snap.problem)}</p>`;
    return;
  }
  if (snap.needsDiscard) {
    target.innerHTML = `<p class="eval-wait">Discard one card to evaluate your final ${snap.cardsInHand}-card hand.</p>`;
    return;
  }
  if (snap.hand.length === 0) {
    target.innerHTML = `<p class="muted">${mode === "table" ? "Deal a game to start." : "Add cards to evaluate."}</p>`;
    return;
  }
  if (snap.hand.length !== snap.cardsInHand) {
    const missing = snap.cardsInHand - snap.hand.length;
    target.innerHTML = `<p class="eval-wait">${snap.hand.length}/${snap.cardsInHand} cards. ${missing > 0 ? `Add ${missing}` : `Remove ${-missing}`} to evaluate.</p>`;
    return;
  }
  target.innerHTML = '<p class="eval-wait">Checking…</p>';
}

function scheduleEval(mode) {
  if (evalCtl.timer) {
    clearTimeout(evalCtl.timer);
    evalCtl.timer = null;
  }
  const snap = snapshotFor(mode);
  const target = mode === "builder" ? el.builderEval : el.tableEval;
  const retry = mode === "builder" ? el.builderRetry : el.tableRetry;
  retry.classList.add("hidden");
  evalCtl.lastError = null;
  paintEvalState(mode, target, snap);

  if (snap.hidden || snap.problem || snap.needsDiscard || snap.hand.length === 0 || snap.hand.length !== snap.cardsInHand) {
    evalCtl.rev += 1;
    evalCtl.queued = null;
    return;
  }

  evalCtl.rev += 1;
  const rev = evalCtl.rev;
  const payload = {
    mode,
    hand: toCounts(snap.hand),
    joker: snap.joker,
    required: snap.required,
    cards: snap.cardsInHand,
    rev,
  };

  evalCtl.timer = setTimeout(() => {
    evalCtl.timer = null;
    enqueueEval(payload);
  }, DEBOUNCE_MS);
}

function snapshotFor(mode) {
  if (mode === "builder") {
    const cfg = builderConfig();
    return {
      hand: builder.hand.slice(),
      joker: cfg.joker,
      required: cfg.requiredSequences,
      cardsInHand: cfg.cardsInHand,
      hidden: false,
      problem: toCounts(builder.hand).some((count) => count > cfg.decks)
        ? `This hand exceeds the ${cfg.decks}-copy limit. Remove extra copies or increase the deck count.`
        : null,
    };
  }
  if (!table.state) {
    return {
      hand: [],
      joker: 0,
      required: 5,
      cardsInHand: 21,
      hidden: false,
    };
  }
  const st = table.state;
  const hidden = st.players.length > 1 && !table.revealed;
  const hand = st.players[st.currentPlayer].hand.slice();
  return {
    hand,
    joker: st.joker.face,
    required: st.rules.requiredSequences,
    cardsInHand: st.rules.cardsInHand,
    hidden,
    needsDiscard: st.phase === "discard",
  };
}

function enqueueEval(payload) {
  if (payload.rev !== evalCtl.rev || payload.mode !== evalCtl.mode) return;
  if (evalCtl.inflight) {
    evalCtl.queued = payload;
    return;
  }
  void runEval(payload);
}

async function runEval(payload) {
  evalCtl.inflight = true;
  const target = payload.mode === "builder" ? el.builderEval : el.tableEval;
  const retry = payload.mode === "builder" ? el.builderRetry : el.tableRetry;
  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hand: payload.hand,
        joker: payload.joker,
        required_sequences: payload.required,
        cards_in_hand: payload.cards,
      }),
    });
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error("The server returned unreadable JSON.");
    }

    if (payload.rev !== evalCtl.rev || payload.mode !== evalCtl.mode) {
      return;
    }

    if (res.status === 503) {
      const msg = data.error || "Evaluator unavailable";
      evalCtl.lastError = msg;
      const kind = /exceeded|timeout/i.test(msg) ? "Timed out" : "Unavailable";
      target.innerHTML = `<p class="eval-bad">${kind}: ${escapeHtml(msg)}</p>
        <p class="muted">Change the hand or press Retry to evaluate again. This is not scored as invalid.</p>`;
      retry.classList.remove("hidden");
      return;
    }
    if (!res.ok) {
      evalCtl.lastError = data.error || `HTTP ${res.status}`;
      target.innerHTML = `<p class="eval-bad">Error: ${escapeHtml(evalCtl.lastError)}</p>`;
      retry.classList.remove("hidden");
      return;
    }

    const isFace = (face) => Number.isInteger(face) && face >= 0 && face < 52;
    if (
      typeof data?.is_valid !== "boolean" ||
      data.reward !== Number(data.is_valid) ||
      !Array.isArray(data.melds) ||
      (data.is_valid ? data.melds.length === 0 : data.melds.length !== 0) ||
      !data.melds.every((meld) =>
        ["sequence", "set"].includes(meld.kind) &&
        typeof meld.is_pure === "boolean" &&
        Array.isArray(meld.cards) && Array.isArray(meld.represented_cards) &&
        meld.cards.length >= 3 &&
        meld.cards.length === meld.represented_cards.length &&
        meld.cards.every(isFace) && meld.represented_cards.every(isFace))
    ) {
      throw new Error("The server returned an invalid evaluation response.");
    }
    renderEvalResult(target, data, payload.joker);
  } catch (err) {
    if (payload.rev !== evalCtl.rev || payload.mode !== evalCtl.mode) return;
    evalCtl.lastError = err instanceof Error ? err.message : String(err);
    target.innerHTML = `<p class="eval-bad">Evaluation error: ${escapeHtml(evalCtl.lastError)}</p>`;
    retry.classList.remove("hidden");
  } finally {
    evalCtl.inflight = false;
    if (evalCtl.queued) {
      const next = evalCtl.queued;
      evalCtl.queued = null;
      if (next.rev === evalCtl.rev) void runEval(next);
    }
  }
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderEvalResult(target, data, jokerFace) {
  const ok = data.is_valid;
  const reward = data.reward;
  const head = document.createElement("div");
  if (ok) {
    head.innerHTML = `<p class="eval-ok">Valid hand · reward ${escapeHtml(String(reward))}</p>`;
  } else {
    head.innerHTML = `<p class="eval-bad">Invalid · reward ${reward} · No complete grouping meets the required sequences</p>`;
  }
  target.replaceChildren(head);

  if (!ok) return;

  const list = document.createElement("div");
  list.className = "meld-list";
  let pureCount = 0;
  let seqCount = 0;
  for (const meld of data.melds) {
    if (meld.kind === "sequence") seqCount += 1;
    if (meld.is_pure) pureCount += 1;
    const block = document.createElement("div");
    block.className = "meld";
    const mh = document.createElement("div");
    mh.className = "meld-head";
    mh.innerHTML = `<strong>${escapeHtml(meld.kind)}</strong><span>${meld.kind === "set" ? "distinct suits" : meld.is_pure ? "qualifying pure" : "impure"}</span>`;
    const cards = document.createElement("div");
    cards.className = "meld-cards";
    const actual = meld.cards;
    const represented = meld.represented_cards;
    for (let i = 0; i < actual.length; i += 1) {
      const a = actual[i];
      const r = represented[i];
      const node = cardNode(a, { jokerFace, compact: true });
      node.disabled = true;
      cards.append(node);
      if (r !== a) {
        const sub = document.createElement("span");
        sub.className = "sub-label";
        sub.textContent = `${labelFace(a)} → ${labelFace(r)}`;
        cards.append(sub);
      }
    }
    block.append(mh, cards);
    list.append(block);
  }
  const summary = document.createElement("p");
  summary.className = "muted";
  summary.textContent = `${seqCount} sequences (${pureCount} pure), ${data.melds.length - seqCount} sets`;
  target.append(summary, list);
}

function dealNewGame() {
  if (table.started && table.state) {
    const ok = window.confirm(
      "Deal a new game? The current round will be discarded.",
    );
    if (!ok) return;
  }
  try {
    const players = readInt(el.tablePlayers);
    const decks = readInt(el.tableDecks);
    const cardsInHand = readInt(el.tableHandSize);
    const requiredSequences = readInt(el.tableRequired);
    table.state = createGame({
      players,
      decks,
      cardsInHand,
      requiredSequences,
    });
    table.selectedId = null;
    table.revealed = true;
    table.started = true;
    setStatus(
      `Dealt ${players === 1 ? "solo" : players + " players"} · ${cardsInHand} cards · joker ${labelFace(table.state.joker.face)}`,
    );
    renderTable();
    scheduleEval("table");
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  }
}

function renderTable() {
  const board = el.tableBoard;
  if (!table.state) {
    board.innerHTML = '<p class="muted">Deal a game to start the table.</p>';
    el.tableEval.innerHTML = '<p class="muted">No active hand.</p>';
    return;
  }
  const st = table.state;
  const multi = st.players.length > 1;
  const covered = multi && !table.revealed;
  const player = st.players[st.currentPlayer];
  const handSize = player.hand.length;
  const target = st.rules.cardsInHand;
  const discardTop = st.discard[st.discard.length - 1];

  board.replaceChildren();

  const meta = document.createElement("div");
  meta.className = "meta-row";
  meta.innerHTML = `
    <span class="chip">Turn ${st.turn}</span>
    <span class="chip">Phase: ${st.phase}</span>
    <span class="chip">Active: ${escapeHtml(player.name)}</span>
    <span class="chip">Hand ${handSize}/${target}</span>
  `;
  board.append(meta);

  const jokerBox = document.createElement("div");
  jokerBox.className = "joker-spotlight";
  const jLabel = document.createElement("div");
  jLabel.innerHTML = `<strong>Joker indicator</strong><div class="muted">Outside play for the round. Rank ${escapeHtml(RANKS[rankOf(st.joker.face)])} is wild.</div>`;
  jokerBox.append(cardNode(st.joker.face, { jokerFace: st.joker.face }), jLabel);
  board.append(jokerBox);

  const piles = document.createElement("div");
  piles.className = "piles";

  const stockPile = document.createElement("div");
  stockPile.className = "pile";
  stockPile.innerHTML = `<h3>Stock · ${st.stock.length} left</h3>`;
  if (st.stock.length === 0) {
    const warn = document.createElement("p");
    warn.className = "warn";
    warn.textContent = "Stock exhausted. No automatic reshuffle. Discard or deal a new game.";
    stockPile.append(warn);
  } else {
    const back = document.createElement("div");
    back.className = "playing-card";
    back.textContent = "🂠";
    back.setAttribute("aria-hidden", "true");
    stockPile.append(back);
  }
  const drawStock = document.createElement("button");
  drawStock.type = "button";
  drawStock.className = "secondary";
  drawStock.textContent = "Draw from stock";
  drawStock.disabled = st.phase !== "draw" || st.stock.length === 0 || covered;
  drawStock.addEventListener("click", () => doDraw("stock"));
  stockPile.append(drawStock);
  piles.append(stockPile);

  const discPile = document.createElement("div");
  discPile.className = "pile";
  discPile.innerHTML = `<h3>Discard · ${st.discard.length}</h3>`;
  if (discardTop) {
    discPile.append(
      cardNode(discardTop.face, {
        jokerFace: st.joker.face,
        title: `Top discard ${labelFace(discardTop.face)}`,
      }),
    );
  } else {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Empty";
    discPile.append(empty);
  }
  const drawDisc = document.createElement("button");
  drawDisc.type = "button";
  drawDisc.className = "secondary";
  drawDisc.textContent = "Draw from discard";
  drawDisc.disabled = st.phase !== "draw" || st.discard.length === 0 || covered;
  drawDisc.addEventListener("click", () => doDraw("discard"));
  discPile.append(drawDisc);
  piles.append(discPile);
  board.append(piles);

  const others = document.createElement("div");
  others.className = "players-list";
  st.players.forEach((p, i) => {
    const chip = document.createElement("span");
    chip.className = "player-chip" + (i === st.currentPlayer ? " active" : "");
    chip.textContent =
      i === st.currentPlayer
        ? `${p.name} · ${p.hand.length} cards (active)`
        : `${p.name} · ${p.hand.length} cards`;
    others.append(chip);
  });
  board.append(others);

  const handSection = document.createElement("div");
  handSection.style.marginTop = "1rem";
  const head = document.createElement("div");
  head.className = "section-head";
  head.innerHTML = `<h2>${escapeHtml(player.name)} hand</h2><p class="count-pill">${handSize} / ${target}${st.phase === "discard" ? " · discard one" : " · draw first"}</p>`;
  handSection.append(head);

  if (covered) {
    const cover = document.createElement("div");
    cover.className = "cover";
    cover.innerHTML = `<p>Hand hidden for pass-and-play. Pass the device to ${escapeHtml(player.name)}, then reveal.</p>`;
    const reveal = document.createElement("button");
    reveal.type = "button";
    reveal.className = "primary";
    reveal.style.width = "auto";
    reveal.textContent = "Reveal hand";
    reveal.addEventListener("click", () => {
      table.revealed = true;
      table.selectedId = null;
      setStatus(`${player.name} hand revealed.`);
      renderTable();
      scheduleEval("table");
    });
    cover.append(reveal);
    handSection.append(cover);
  } else {
    const handEl = document.createElement("div");
    handEl.className = "hand";
    const sorted = player.hand
      .slice()
      .sort((a, b) => a.face - b.face || a.id - b.id);
    for (const card of sorted) {
      const node = cardNode(card.face, {
        jokerFace: st.joker.face,
        physicalId: card.id,
        selected: table.selectedId === card.id,
        drawn: st.drawnCardId === card.id,
      });
      node.addEventListener("click", () => {
        if (st.phase !== "discard") {
          setStatus("Draw a card before selecting a discard.");
          return;
        }
        table.selectedId = card.id;
        renderTable();
      });
      handEl.append(node);
    }
    handSection.append(handEl);

    const actions = document.createElement("div");
    actions.className = "actions";
    const discardSelected = document.createElement("button");
    discardSelected.type = "button";
    discardSelected.className = "primary";
    discardSelected.style.width = "auto";
    discardSelected.textContent = "Discard selected";
    discardSelected.disabled =
      st.phase !== "discard" || table.selectedId == null;
    discardSelected.addEventListener("click", () => {
      if (table.selectedId == null) return;
      doDiscard(table.selectedId);
    });
    const discardDrawn = document.createElement("button");
    discardDrawn.type = "button";
    discardDrawn.className = "secondary";
    discardDrawn.textContent = "Discard drawn card";
    discardDrawn.disabled = st.phase !== "discard" || st.drawnCardId == null;
    discardDrawn.addEventListener("click", () => {
      if (st.drawnCardId == null) return;
      doDiscard(st.drawnCardId);
    });
    actions.append(discardSelected, discardDrawn);
    handSection.append(actions);
  }
  board.append(handSection);
}

function doDraw(source) {
  if (!table.state) return;
  try {
    table.state = drawCard(table.state, source);
    table.selectedId = table.state.drawnCardId;
    setStatus(
      `Drew from ${source}: ${labelFace(
        table.state.players[table.state.currentPlayer].hand.find(
          (c) => c.id === table.state.drawnCardId,
        ).face,
      )}`,
    );
    renderTable();
    scheduleEval("table");
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  }
}

function doDiscard(physicalId) {
  if (!table.state) return;
  const multi = table.state.players.length > 1;
  try {
    table.state = discardCard(table.state, physicalId);
    table.selectedId = null;
    if (multi) {
      table.revealed = false;
      setStatus(
        `Discarded. Pass to ${table.state.players[table.state.currentPlayer].name}.`,
      );
    } else {
      table.revealed = true;
      setStatus("Discarded. Your turn again.");
    }
    renderTable();
    scheduleEval("table");
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  }
}

function wire() {
  el.tabBuilder.addEventListener("click", () => setMode("builder"));
  el.tabTable.addEventListener("click", () => setMode("table"));
  el.builderLoadSample.addEventListener("click", () => {
    loadSample(el.builderSample.value || builder.selectedSample);
  });
  el.builderClear.addEventListener("click", clearBuilderHand);
  for (const input of [
    el.builderHandSize,
    el.builderRequired,
    el.builderDecks,
    el.builderJokerSuit,
    el.builderJokerRank,
  ]) {
    input.addEventListener("change", () => {
      try {
        renderBuilder();
        scheduleEval("builder");
        setStatus("Settings updated. Your cards have been kept.");
      } catch (error) {
        writeBuilderConfig(builder.lastConfig);
        setStatus(`Settings unchanged. ${error.message}`);
      }
    });
  }
  el.builderRetry.addEventListener("click", () => scheduleEval("builder"));
  el.tableRetry.addEventListener("click", () => scheduleEval("table"));
  el.tableDeal.addEventListener("click", dealNewGame);
}

wire();
loadSample("classic21");
el.builderSample.value = "classic21";
setMode("builder");
