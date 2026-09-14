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
} from "/game.mjs";

const SUITS = ["♠", "♥", "♦", "♣"];
const SUIT_NAMES = ["Spades", "Hearts", "Diamonds", "Clubs"];
const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
const RED_SUITS = new Set([1, 2]);
const DEBOUNCE_MS = 180;

/** @typedef {{ id: number, face: number }} Card */
/**
 * @typedef {{
 *   kind: 'sequence' | 'set',
 *   is_pure: boolean,
 *   cards: number[],
 *   represented_cards: number[]
 * }} Meld
 */
/** @typedef {{ is_valid: boolean, reward: 0 | 1, melds: Meld[] }} Evaluation */

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
  builderRemove: document.getElementById("builder-remove"),
  builderMoveLeft: document.getElementById("builder-move-left"),
  builderMoveRight: document.getElementById("builder-move-right"),
  builderSort: document.getElementById("builder-sort"),
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
  tableDeclaration: document.getElementById("table-declaration"),
  tableDeclarationRetry: document.getElementById("table-declaration-retry"),
  tableEval: document.getElementById("table-eval"),
  tableRetry: document.getElementById("table-retry"),
};

const builder = {
  /** @type {Card[]} */
  hand: [],
  nextId: 1,
  selectedSample: "classic21",
  lastConfig: null,
  selectedId: /** @type {number | null} */ (null),
  draggedId: /** @type {number | null} */ (null),
};

const table = {
  /** @type {import('./game.mjs').GameState | null} */
  state: null,
  selectedId: /** @type {number | null} */ (null),
  draggedId: /** @type {number | null} */ (null),
  revealed: true,
  started: false,
  checkHand: /** @type {boolean[]} */ ([]),
  generation: 0,
  /** @type {null | { data: Evaluation, ownerIndex: number }} */
  declarationResult: null,
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

const declarationCtl = {
  inflight: false,
  error: /** @type {string | null} */ (null),
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
  builder.hand = sortCards(
    faces.map((face) => {
      const card = { id: builder.nextId, face };
      builder.nextId += 1;
      return card;
    }),
  );
  builder.selectedId = null;
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
  for (const [index, card] of builder.hand.entries()) {
    const node = cardNode(card.face, {
      jokerFace: cfg.joker,
      physicalId: card.id,
      selected: builder.selectedId === card.id,
      title: `${labelFace(card.face)} (select)`,
    });
    node.draggable = true;
    node.addEventListener("click", () => {
      builder.selectedId = builder.selectedId === card.id ? null : card.id;
      renderBuilder();
      focusSelected(el.builderHand, builder.selectedId);
    });
    node.addEventListener("dragstart", () => {
      builder.draggedId = card.id;
    });
    node.addEventListener("dragend", () => {
      builder.draggedId = null;
    });
    node.addEventListener("dragover", (event) => event.preventDefault());
    node.addEventListener("drop", (event) => {
      event.preventDefault();
      if (builder.draggedId == null) return;
      builder.selectedId = builder.draggedId;
      moveBuilderCard(index);
      builder.draggedId = null;
    });
    el.builderHand.append(node);
  }
  el.builderHandCount.textContent = `${builder.hand.length} / ${cfg.cardsInHand}`;
  const selectedIndex = builder.hand.findIndex(
    (card) => card.id === builder.selectedId,
  );
  el.builderRemove.disabled = selectedIndex < 0;
  el.builderMoveLeft.disabled = selectedIndex <= 0;
  el.builderMoveRight.disabled =
    selectedIndex < 0 || selectedIndex === builder.hand.length - 1;
  el.builderSort.disabled = builder.hand.length < 2;
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
  builder.selectedId = builder.nextId;
  builder.nextId += 1;
  setStatus(`Added ${labelFace(face)}.`);
  renderBuilder();
  scheduleEval("builder");
}

function removeBuilderCard() {
  const idx = builder.hand.findIndex((card) => card.id === builder.selectedId);
  if (idx < 0) return;
  const [removed] = builder.hand.splice(idx, 1);
  builder.selectedId =
    builder.hand[Math.min(idx, builder.hand.length - 1)]?.id ?? null;
  setStatus(`Removed ${labelFace(removed.face)}.`);
  renderBuilder();
  focusSelected(el.builderHand, builder.selectedId);
  scheduleEval("builder");
}

function clearBuilderHand() {
  builder.hand = [];
  builder.selectedId = null;
  setStatus("Hand cleared.");
  renderBuilder();
  scheduleEval("builder");
}

function focusSelected(container, physicalId) {
  if (physicalId == null) return;
  container.querySelector(`[data-id="${physicalId}"]`)?.focus();
}

function moveBuilderCard(targetIndex) {
  if (builder.selectedId == null) return;
  builder.hand = moveCard(builder.hand, builder.selectedId, targetIndex);
  renderBuilder();
  focusSelected(el.builderHand, builder.selectedId);
}

function moveBuilderBy(offset) {
  const index = builder.hand.findIndex(
    (card) => card.id === builder.selectedId,
  );
  if (index < 0) return;
  moveBuilderCard(Math.max(0, Math.min(builder.hand.length - 1, index + offset)));
}

function sortBuilderHand() {
  builder.hand = sortCards(builder.hand);
  renderBuilder();
  focusSelected(el.builderHand, builder.selectedId);
  setStatus("Hand sorted by suit and rank.");
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
 * @param {{ hand: Card[], joker: number, required: number, cardsInHand: number, hidden?: boolean, unchecked?: boolean, problem?: string | null, needsDiscard?: boolean }} snap
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
  if (snap.unchecked) {
    target.innerHTML =
      '<p class="muted">Check hand is off. Turn it on to evaluate this player.</p>';
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

  if (
    snap.hidden ||
    snap.unchecked ||
    snap.problem ||
    snap.needsDiscard ||
    snap.hand.length === 0 ||
    snap.hand.length !== snap.cardsInHand
  ) {
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
  const unavailable =
    st.phase === "declaring"
      ? "A declaration is pending."
      : st.phase === "finished"
        ? "The round is finished."
        : null;
  return {
    hand,
    joker: st.joker.face,
    required: st.rules.requiredSequences,
    cardsInHand: st.rules.cardsInHand,
    hidden,
    unchecked: !table.checkHand[st.currentPlayer],
    problem: unavailable,
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
    const data = await requestEvaluation(payload);

    if (payload.rev !== evalCtl.rev || payload.mode !== evalCtl.mode) {
      return;
    }
    renderEvalResult(target, data, payload.joker);
  } catch (err) {
    if (payload.rev !== evalCtl.rev || payload.mode !== evalCtl.mode) return;
    evalCtl.lastError = err instanceof Error ? err.message : String(err);
    const status = err instanceof Error ? err.status : undefined;
    const kind =
      status === 503
        ? /exceeded|timeout/i.test(evalCtl.lastError)
          ? "Timed out"
          : "Unavailable"
        : "Evaluation error";
    target.innerHTML = `<p class="eval-bad">${kind}: ${escapeHtml(evalCtl.lastError)}</p>
      <p class="muted">Change the hand or press Retry to evaluate again. This is not scored as invalid.</p>`;
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

/**
 * @param {unknown} data
 * @returns {Evaluation}
 */
function validateEvaluationResponse(data) {
  const isFace = (face) => Number.isInteger(face) && face >= 0 && face < 52;
  if (
    data === null ||
    typeof data !== "object" ||
    !("is_valid" in data) ||
    !("reward" in data) ||
    !("melds" in data) ||
    typeof data.is_valid !== "boolean" ||
    data.reward !== Number(data.is_valid) ||
    !Array.isArray(data.melds) ||
    (data.is_valid ? data.melds.length === 0 : data.melds.length !== 0) ||
    !data.melds.every((meld) =>
      meld !== null &&
      typeof meld === "object" &&
      ["sequence", "set"].includes(meld.kind) &&
      typeof meld.is_pure === "boolean" &&
      Array.isArray(meld.cards) &&
      Array.isArray(meld.represented_cards) &&
      meld.cards.length >= 3 &&
      meld.cards.length === meld.represented_cards.length &&
      meld.cards.every(isFace) &&
      meld.represented_cards.every(isFace))
  ) {
    throw new Error("The server returned an invalid evaluation response.");
  }
  return {
    is_valid: data.is_valid,
    reward: data.is_valid ? 1 : 0,
    melds: data.melds,
  };
}

/**
 * @param {{ hand: number[], joker: number, required: number, cards: number }} payload
 * @returns {Promise<Evaluation>}
 */
async function requestEvaluation(payload) {
  const response = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      hand: payload.hand,
      joker: payload.joker,
      required_sequences: payload.required,
      cards_in_hand: payload.cards,
    }),
  });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("The server returned unreadable JSON.");
  }
  if (!response.ok) {
    const error = new Error(data?.error || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return validateEvaluationResponse(data);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

/**
 * @param {HTMLElement} target
 * @param {Evaluation} data
 * @param {number} jokerFace
 */
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
    const hadGame = table.state !== null;
    const sameSeats = table.state?.players.length === players;
    const penaltyTotals = sameSeats
      ? table.state.players.map((player) => player.penaltyPoints)
      : Array(players).fill(0);
    const nextState = createGame({
      players,
      decks,
      cardsInHand,
      requiredSequences,
      penaltyTotals,
    });
    table.generation += 1;
    table.state = nextState;
    table.selectedId = null;
    table.draggedId = null;
    table.revealed = true;
    table.started = true;
    table.checkHand = Array(players).fill(false);
    table.declarationResult = null;
    declarationCtl.inflight = false;
    declarationCtl.error = null;
    setStatus(
      `Dealt ${players === 1 ? "solo" : players + " players"} · ${cardsInHand} cards · joker ${labelFace(table.state.joker.face)}${
        hadGame && !sameSeats
          ? " · Penalty points reset because the player count changed."
          : ""
      }`,
    );
    renderTable();
    scheduleEval("table");
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  }
}

function renderDeclarationResult(st) {
  el.tableDeclarationRetry.classList.add("hidden");
  if (st.phase === "declaring") {
    if (declarationCtl.error) {
      el.tableDeclaration.innerHTML = `<p class="eval-bad">Declaration check failed: ${escapeHtml(declarationCtl.error)}</p>
        <p class="muted">No penalty was applied. Retry uses the same committed hand.</p>`;
      el.tableDeclarationRetry.classList.remove("hidden");
    } else {
      el.tableDeclaration.innerHTML =
        '<p class="eval-wait">Checking declaration...</p>';
    }
    return;
  }
  if (!table.declarationResult) {
    el.tableDeclaration.innerHTML =
      '<p class="muted">No declaration has been resolved this round.</p>';
    return;
  }
  const { data, ownerIndex } = table.declarationResult;
  renderEvalResult(el.tableDeclaration, data, st.joker.face);
  const heading = document.createElement("p");
  heading.className = data.is_valid ? "eval-ok" : "eval-bad";
  heading.textContent = `${st.players[ownerIndex].name} declared ${data.is_valid ? "a valid hand." : "an invalid hand."}`;
  el.tableDeclaration.prepend(heading);
}

function outcomeText(st) {
  if (st.outcomeReason === "valid-declaration") {
    return `${st.players[st.winnerIndex].name} wins with a valid declaration.`;
  }
  if (st.outcomeReason === "last-remaining") {
    return `${st.players[st.winnerIndex].name} wins as the last active player.`;
  }
  return "Solo round lost after an invalid declaration.";
}

function renderTable() {
  const board = el.tableBoard;
  if (!table.state) {
    board.innerHTML = '<p class="muted">Deal a game to start the table.</p>';
    el.tableDeclaration.innerHTML =
      '<p class="muted">No declaration this round.</p>';
    el.tableEval.innerHTML = '<p class="muted">No active hand.</p>';
    el.tableDeclarationRetry.classList.add("hidden");
    return;
  }
  const st = table.state;
  const multi = st.players.length > 1;
  const covered = multi && !table.revealed && st.phase !== "finished";
  const player = st.players[st.currentPlayer];
  const handSize = player.hand.length;
  const target = st.rules.cardsInHand;
  const discardTop = st.discard[st.discard.length - 1];
  const canArrange = st.phase === "draw" || st.phase === "discard";

  renderDeclarationResult(st);
  board.replaceChildren();

  if (st.phase === "finished") {
    const outcome = document.createElement("div");
    outcome.className = "outcome-banner";
    outcome.dataset.reason = st.outcomeReason;
    outcome.textContent = outcomeText(st);
    board.append(outcome);
  }

  const meta = document.createElement("div");
  meta.className = "meta-row";
  meta.innerHTML = `
    <span class="chip">Turn ${st.turn}</span>
    <span class="chip">Phase: ${st.phase}</span>
    <span class="chip">Current: ${escapeHtml(player.name)}</span>
    <span class="chip">Hand ${handSize}/${target}</span>
  `;
  board.append(meta);

  const jokerBox = document.createElement("div");
  jokerBox.className = "joker-spotlight";
  const jokerLabel = document.createElement("div");
  jokerLabel.innerHTML = `<strong>Joker indicator</strong><div class="muted">Outside play for the round. Rank ${escapeHtml(RANKS[rankOf(st.joker.face)])} is wild.</div>`;
  jokerBox.append(
    cardNode(st.joker.face, { jokerFace: st.joker.face }),
    jokerLabel,
  );
  board.append(jokerBox);

  const piles = document.createElement("div");
  piles.className = "piles";

  const stockPile = document.createElement("div");
  stockPile.className = "pile";
  stockPile.innerHTML = `<h3>Stock · ${st.stock.length} left</h3>`;
  if (st.stock.length === 0) {
    const warn = document.createElement("p");
    warn.className = "warn";
    warn.textContent =
      "Stock exhausted. No automatic reshuffle. Discard or deal a new game.";
    stockPile.append(warn);
  } else {
    const back = document.createElement("div");
    back.className = "playing-card card-back";
    back.setAttribute("aria-hidden", "true");
    stockPile.append(back);
  }
  const drawStock = document.createElement("button");
  drawStock.type = "button";
  drawStock.className = "secondary";
  drawStock.textContent = "Draw from stock";
  drawStock.disabled =
    st.phase !== "draw" || st.stock.length === 0 || covered;
  drawStock.addEventListener("click", () => doDraw("stock"));
  stockPile.append(drawStock);
  piles.append(stockPile);

  const discardPile = document.createElement("div");
  discardPile.className = "pile";
  discardPile.innerHTML = `<h3>Discard · ${st.discard.length}</h3>`;
  if (discardTop) {
    discardPile.append(
      cardNode(discardTop.face, {
        jokerFace: st.joker.face,
        title: `Top discard ${labelFace(discardTop.face)}`,
      }),
    );
  } else {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Empty";
    discardPile.append(empty);
  }
  const drawDiscard = document.createElement("button");
  drawDiscard.type = "button";
  drawDiscard.className = "secondary";
  drawDiscard.textContent = "Draw from discard";
  drawDiscard.disabled =
    st.phase !== "draw" || st.discard.length === 0 || covered;
  drawDiscard.addEventListener("click", () => doDraw("discard"));
  discardPile.append(drawDiscard);
  piles.append(discardPile);

  const declarationPile = document.createElement("div");
  declarationPile.className = "pile";
  declarationPile.innerHTML = `<h3>Declarations · ${st.declarations.length}</h3>`;
  if (st.declarations.length === 0) {
    declarationPile.insertAdjacentHTML(
      "beforeend",
      '<p class="muted">Declared cards stay face down here.</p>',
    );
  } else {
    const declarationCards = document.createElement("div");
    declarationCards.className = "declaration-zone";
    for (const declaration of st.declarations) {
      const item = document.createElement("div");
      item.className = "declaration-item";
      item.dataset.owner = String(declaration.ownerIndex);
      const back = document.createElement("div");
      back.className = "playing-card card-back compact";
      back.setAttribute("role", "img");
      back.setAttribute("aria-label", "Face-down declared card");
      const label = document.createElement("span");
      label.textContent = `${st.players[declaration.ownerIndex].name} · ${
        declaration.verdict === "pending"
          ? "checking"
          : declaration.verdict === "valid"
            ? "valid"
            : `invalid (+${st.players[declaration.ownerIndex].roundPenalty})`
      }`;
      item.append(back, label);
      declarationCards.append(item);
    }
    declarationPile.append(declarationCards);
  }
  piles.append(declarationPile);
  board.append(piles);

  const players = document.createElement("div");
  players.className = "players-list";
  st.players.forEach((candidate, index) => {
    const finishedStatus =
      st.phase === "finished" && candidate.active
        ? index === st.winnerIndex
          ? "winner"
          : "lost"
        : null;
    const status = candidate.active
      ? finishedStatus || "active"
      : "eliminated";
    const chip = document.createElement("span");
    chip.className = [
      "player-chip",
      index === st.currentPlayer ? "current" : "",
      status,
    ].filter(Boolean).join(" ");
    chip.dataset.playerIndex = String(index);
    chip.dataset.penaltyTotal = String(candidate.penaltyPoints);
    chip.textContent = `${candidate.name} · ${candidate.hand.length} cards · Penalty points ${candidate.penaltyPoints} (lower is better) · ${status}${
      candidate.roundPenalty ? ` · +${candidate.roundPenalty} this round` : ""
    }`;
    players.append(chip);
  });
  board.append(players);

  const handSection = document.createElement("div");
  handSection.className = "hand-section";
  const head = document.createElement("div");
  head.className = "section-head";
  const prompt =
    st.phase === "discard"
      ? "discard or declare"
      : st.phase === "draw"
        ? "draw first"
        : st.phase === "declaring"
          ? "declaration committed"
          : "round finished";
  head.innerHTML = `<h2>${escapeHtml(player.name)} hand</h2><p class="count-pill">${handSize} / ${target} · ${prompt}</p>`;
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
    const checkLabel = document.createElement("label");
    checkLabel.className = "check-control";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = Boolean(table.checkHand[st.currentPlayer]);
    check.disabled = !canArrange;
    check.dataset.action = "check-hand";
    check.addEventListener("change", () => {
      table.checkHand[st.currentPlayer] = check.checked;
      renderTable();
      scheduleEval("table");
    });
    checkLabel.append(check, document.createTextNode("Check hand"));
    handSection.append(checkLabel);

    const hand = document.createElement("div");
    hand.className = "hand";
    hand.dataset.playerHand = String(st.currentPlayer);
    for (const [index, card] of player.hand.entries()) {
      const node = cardNode(card.face, {
        jokerFace: st.joker.face,
        physicalId: card.id,
        selected: table.selectedId === card.id,
        drawn: st.drawnCardId === card.id,
      });
      node.draggable = canArrange;
      if (canArrange) {
        node.addEventListener("click", () => {
          table.selectedId = table.selectedId === card.id ? null : card.id;
          renderTable();
          focusSelected(el.tableBoard, table.selectedId);
        });
        node.addEventListener("dragstart", () => {
          table.draggedId = card.id;
        });
        node.addEventListener("dragend", () => {
          table.draggedId = null;
        });
        node.addEventListener("dragover", (event) => event.preventDefault());
        node.addEventListener("drop", (event) => {
          event.preventDefault();
          if (table.draggedId == null) return;
          table.selectedId = table.draggedId;
          reorderTableHand(index);
          table.draggedId = null;
        });
      }
      hand.append(node);
    }
    handSection.append(hand);

    const selectedIndex = player.hand.findIndex(
      (card) => card.id === table.selectedId,
    );
    const actions = document.createElement("div");
    actions.className = "actions";

    const moveLeft = document.createElement("button");
    moveLeft.type = "button";
    moveLeft.className = "secondary";
    moveLeft.textContent = "Move selected left";
    moveLeft.disabled = !canArrange || selectedIndex <= 0;
    moveLeft.addEventListener("click", () => reorderTableHand(selectedIndex - 1));

    const moveRight = document.createElement("button");
    moveRight.type = "button";
    moveRight.className = "secondary";
    moveRight.textContent = "Move selected right";
    moveRight.disabled =
      !canArrange ||
      selectedIndex < 0 ||
      selectedIndex === player.hand.length - 1;
    moveRight.addEventListener("click", () => reorderTableHand(selectedIndex + 1));

    const sort = document.createElement("button");
    sort.type = "button";
    sort.className = "secondary";
    sort.textContent = "Sort hand";
    sort.disabled = !canArrange || player.hand.length < 2;
    sort.addEventListener("click", sortTableHand);

    const discardSelected = document.createElement("button");
    discardSelected.type = "button";
    discardSelected.className = "primary";
    discardSelected.style.width = "auto";
    discardSelected.textContent = "Discard selected";
    discardSelected.disabled =
      st.phase !== "discard" || table.selectedId == null;
    discardSelected.addEventListener("click", () => {
      if (table.selectedId != null) doDiscard(table.selectedId);
    });

    const discardDrawn = document.createElement("button");
    discardDrawn.type = "button";
    discardDrawn.className = "secondary";
    discardDrawn.textContent = "Discard drawn card";
    discardDrawn.disabled = st.phase !== "discard" || st.drawnCardId == null;
    discardDrawn.addEventListener("click", () => {
      if (st.drawnCardId != null) doDiscard(st.drawnCardId);
    });

    const declare = document.createElement("button");
    declare.type = "button";
    declare.className = "declare";
    declare.textContent = "Declare win";
    declare.disabled = st.phase !== "discard" || table.selectedId == null;
    declare.addEventListener("click", () => {
      if (table.selectedId != null) doDeclare(table.selectedId);
    });

    actions.append(
      moveLeft,
      moveRight,
      sort,
      discardSelected,
      discardDrawn,
      declare,
    );
    handSection.append(actions);
  }
  board.append(handSection);
}

function reorderTableHand(targetIndex) {
  if (!table.state || table.selectedId == null) return;
  try {
    table.state = reorderHand(table.state, table.selectedId, targetIndex);
    renderTable();
    focusSelected(el.tableBoard, table.selectedId);
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  }
}

function sortTableHand() {
  if (!table.state) return;
  try {
    table.state = sortHand(table.state);
    renderTable();
    focusSelected(el.tableBoard, table.selectedId);
    setStatus("Hand sorted by suit and rank.");
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  }
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

function doDeclare(physicalId) {
  if (!table.state || declarationCtl.inflight) return;
  try {
    table.state = beginDeclaration(table.state, physicalId);
    table.selectedId = null;
    table.declarationResult = null;
    declarationCtl.error = null;
    scheduleEval("table");
    renderTable();
    setStatus("Declaration committed. Checking the remaining hand.");
    void submitDeclaration(table.generation);
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  }
}

async function submitDeclaration(generation) {
  if (
    !table.state ||
    table.state.phase !== "declaring" ||
    declarationCtl.inflight
  ) {
    return;
  }
  const state = table.state;
  const declaration = state.declarations[state.declarations.length - 1];
  const declarationKey = `${declaration.ownerIndex}:${declaration.card.id}:${state.turn}`;
  const hand = state.players[declaration.ownerIndex].hand;
  declarationCtl.inflight = true;
  declarationCtl.error = null;
  renderTable();
  try {
    const data = await requestEvaluation({
      hand: toCounts(hand),
      joker: state.joker.face,
      required: state.rules.requiredSequences,
      cards: state.rules.cardsInHand,
    });
    const live = table.state;
    const liveDeclaration = live?.declarations[live.declarations.length - 1];
    if (
      generation !== table.generation ||
      !live ||
      live.phase !== "declaring" ||
      `${liveDeclaration.ownerIndex}:${liveDeclaration.card.id}:${live.turn}` !==
        declarationKey
    ) {
      return;
    }
    table.state = resolveDeclaration(live, data.is_valid);
    table.declarationResult = {
      data,
      ownerIndex: declaration.ownerIndex,
    };
    declarationCtl.error = null;
    if (table.state.phase === "draw" && table.state.players.length > 1) {
      table.revealed = false;
    }
    setStatus(
      data.is_valid
        ? `${state.players[declaration.ownerIndex].name} made a valid declaration.`
        : `${state.players[declaration.ownerIndex].name} made an invalid declaration and received ${DECLARATION_PENALTY} penalty points.`,
    );
    renderTable();
    if (evalCtl.mode === "table") scheduleEval("table");
  } catch (err) {
    if (generation !== table.generation || table.state?.phase !== "declaring") {
      return;
    }
    declarationCtl.error = err instanceof Error ? err.message : String(err);
    setStatus("Declaration could not be checked. No penalty was applied.");
    renderTable();
  } finally {
    if (generation === table.generation) declarationCtl.inflight = false;
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
  el.builderRemove.addEventListener("click", removeBuilderCard);
  el.builderMoveLeft.addEventListener("click", () => moveBuilderBy(-1));
  el.builderMoveRight.addEventListener("click", () => moveBuilderBy(1));
  el.builderSort.addEventListener("click", sortBuilderHand);
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
  el.tableDeclarationRetry.addEventListener("click", () => {
    void submitDeclaration(table.generation);
  });
  el.tableDeal.addEventListener("click", dealNewGame);
}

wire();
loadSample("classic21");
el.builderSample.value = "classic21";
setMode("builder");
