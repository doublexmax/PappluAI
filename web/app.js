import { moveCard, sortCards, toCounts } from "./game.mjs";
import { cardBackSvg, cardFaceSvg } from "./cards.mjs";
import { createEngine, createLocalEngine } from "./engine.mjs";
import { parseQModel } from "./qnet.mjs";
import { createTable } from "./table.mjs";

const SUITS = ["♠", "♥", "♦", "♣"];
const SUIT_NAMES = ["Spades", "Hearts", "Diamonds", "Clubs"];
const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
const TIER_NAMES = { neural: "Neural + Search", tactical: "Tactical", casual: "Casual" };
const $ = (id) => document.getElementById(id);
const builder = { hand: [], nextId: 1, selectedId: null, draggedId: null, revision: 0, timer: null };
const pointer = { selectedId: null, draggedId: null, stamp: null };
const modelUrl = new URL("./papplu-model.json", import.meta.url);

function uiEngine(url = null) {
  if (typeof Worker !== "undefined") {
    try { return createEngine({ modelUrl: url }); } catch {}
  }
  let local = createLocalEngine();
  let disposed = false;
  if (url) {
    fetch(url).then((response) => {
      if (!response.ok) throw new Error("model download failed");
      return response.json();
    }).then((json) => {
      if (disposed) return;
      const model = parseQModel(json);
      local.dispose();
      local = createLocalEngine({ model });
    }).catch(() => {});
  }
  return {
    evaluate: (...args) => local.evaluate(...args),
    penalties: (...args) => local.penalties(...args),
    chooseMove: (...args) => local.chooseMove(...args),
    dispose() { disposed = true; local.dispose(); },
  };
}

const builderEngine = uiEngine();
let storage = null;
try { storage = window.localStorage; } catch {}
const table = createTable({ engineFactory: () => uiEngine(modelUrl), storage });

function node(tag, className = "", text = "") {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text) value.textContent = text;
  return value;
}

function button(text, onClick, { id, disabled = false, className = "secondary" } = {}) {
  const value = node("button", className, text);
  value.type = "button";
  value.disabled = disabled;
  if (id) value.id = id;
  value.addEventListener("click", onClick);
  return value;
}

function labelFace(face) {
  return `${RANKS[face % 13]}${SUITS[Math.floor(face / 13)]}`;
}

function cardNode(face, { id, selected = false, drawn = false, joker, compact = false } = {}) {
  const card = node("button", `playing-card${compact ? " compact" : ""}${selected ? " selected" : ""}${drawn ? " drawn" : ""}`);
  card.type = "button";
  card.innerHTML = cardFaceSvg(face);
  card.dataset.face = String(face);
  if (id !== undefined) {
    card.dataset.id = String(id);
    card.dataset.cardId = String(id);
  }
  card.title = labelFace(face);
  card.setAttribute("aria-label", `${RANKS[face % 13]} of ${SUIT_NAMES[Math.floor(face / 13)]}${drawn ? ", drawn card" : ""}`);
  card.setAttribute("aria-pressed", String(selected));
  if (joker !== undefined && face % 13 === joker % 13) {
    card.append(node("span", `badge ${face === joker ? "exact" : "wild"}`, face === joker ? "exact" : "wild"));
  }
  return card;
}

function faceList(faces, joker) {
  const list = node("div", "meld-cards");
  for (const face of faces) {
    const card = cardNode(face, { joker, compact: true });
    card.disabled = true;
    list.append(card);
  }
  return list;
}

function meldList(melds, joker) {
  const list = node("div", "meld-list");
  for (const meld of melds) {
    const block = node("div", "meld");
    block.append(node("p", "meld-head",
      `${meld.kind === "set" ? "Set" : "Sequence"} · ${meld.is_pure ? "qualifying pure" : "non-pure"}`));
    const cards = faceList(meld.cards, joker);
    for (let index = 0; index < meld.cards.length; index += 1) {
      if (meld.cards[index] !== meld.represented_cards[index]) {
        cards.append(node("span", "sub-label",
          `${labelFace(meld.cards[index])} → ${labelFace(meld.represented_cards[index])}`));
      }
    }
    block.append(cards);
    list.append(block);
  }
  return list;
}

function renderEvaluation(target, evaluation, joker) {
  target.replaceChildren(node("p", evaluation.is_valid ? "eval-ok" : "eval-bad",
    evaluation.is_valid ? "Valid hand" : "Not valid. No complete grouping meets the required sequences."));
  if (evaluation.is_valid) target.append(meldList(evaluation.melds, joker));
}

function renderPenalty(target, penalty, joker) {
  const section = node("section", "penalty-result");
  section.append(
    node("h3", "", `Minimum deadwood · ${penalty.points} points`),
    node("p", "muted", `${penalty.qualifying_sequences} qualifying pure sequences. False-declaration penalties are separate.`),
    node("h4", "", "Exempt melds"),
    penalty.exempt_melds.length ? meldList(penalty.exempt_melds, joker) : node("p", "muted", "No exempt melds."),
    node("h4", "", "Counted deadwood cards"),
  );
  const counted = penalty.counted_cards.flatMap((count, face) => Array(count).fill(face));
  section.append(counted.length ? faceList(counted, joker) : node("p", "muted", "No counted cards."));
  target.append(section);
}

function readInt(id) {
  const input = $(id);
  const value = Number(input.value);
  if (!input.value || !Number.isInteger(value) || !input.checkValidity()) {
    throw new Error(`${input.labels[0].firstChild.textContent.trim()} must be an integer from ${input.min} to ${input.max}.`);
  }
  return value;
}

function builderConfig() {
  return {
    cardsInHand: readInt("builder-hand-size"), requiredSequences: readInt("builder-required"),
    decks: readInt("builder-decks"),
    joker: Number($("builder-joker-suit").value) * 13 + Number($("builder-joker-rank").value),
  };
}

function scheduleBuilderEvaluation() {
  clearTimeout(builder.timer);
  const revision = ++builder.revision;
  $("builder-retry").classList.add("hidden");
  $("builder-eval").replaceChildren(node("p", "muted", "Checking hand and minimum deadwood..."));
  builder.timer = setTimeout(async () => {
    try {
      const config = builderConfig();
      const counts = toCounts(builder.hand);
      const [evaluation, penalties] = await Promise.all([
        builderEngine.evaluate(counts, config),
        builderEngine.penalties({
          joker: config.joker, requiredSequences: config.requiredSequences,
          hands: [{ faces: builder.hand.map((card) => card.face) }],
        }),
      ]);
      if (revision !== builder.revision) return;
      renderEvaluation($("builder-eval"), evaluation, config.joker);
      if (builder.hand.length !== config.cardsInHand) {
        $("builder-eval").prepend(node("p", "muted",
          `${builder.hand.length} of ${config.cardsInHand} cards. Partial hands still have a minimum deadwood score.`));
      }
      renderPenalty($("builder-eval"), penalties[0], config.joker);
    } catch (error) {
      if (revision !== builder.revision) return;
      $("builder-eval").replaceChildren(node("p", "eval-bad", error.name === "EngineTimeout" ?
        "Evaluation exceeded 10 seconds. Try fewer wildcards or a smaller hand. This is not a losing-hand verdict." :
        error.message));
      $("builder-retry").classList.remove("hidden");
    }
  }, 180);
}

function renderBuilder() {
  try {
    const config = builderConfig();
    $("builder-settings-summary").textContent =
      `${config.cardsInHand} cards · ${config.requiredSequences} pure · Joker ${labelFace(config.joker)}`;
    $("builder-hand-count").textContent = `${builder.hand.length} / ${config.cardsInHand}`;
    const counts = toCounts(builder.hand);
    $("builder-gallery").replaceChildren();
    for (let suit = 0; suit < 4; suit += 1) {
      const row = node("div", "suit-row");
      for (let rank = 0; rank < 13; rank += 1) {
        const face = suit * 13 + rank;
        const card = cardNode(face, { joker: config.joker });
        card.className = "face-btn";
        card.append(node("span", `copy-count${counts[face] ? "" : " zero"}`, `${counts[face]}/${config.decks}`));
        card.disabled = counts[face] >= config.decks || builder.hand.length >= config.cardsInHand;
        card.setAttribute("aria-label", `Add ${RANKS[rank]} of ${SUIT_NAMES[suit]}`);
        card.addEventListener("click", () => {
          builder.hand.push({ id: builder.nextId++, face });
          builder.selectedId = builder.hand.at(-1).id;
          builderChanged();
        });
        row.append(card);
      }
      $("builder-gallery").append(row);
    }
    $("builder-gallery-hint").textContent = `Up to ${config.decks} copies per face. Joker ${labelFace(config.joker)}.`;
    $("builder-hand").replaceChildren();
    for (const [index, card] of builder.hand.entries()) {
      const value = cardNode(card.face, { id: card.id, joker: config.joker, selected: card.id === builder.selectedId });
      value.addEventListener("click", () => {
        builder.selectedId = builder.selectedId === card.id ? null : card.id;
        renderBuilder();
      });
      value.draggable = true;
      value.addEventListener("dragstart", () => { builder.draggedId = card.id; });
      value.addEventListener("dragend", () => { builder.draggedId = null; });
      value.addEventListener("dragover", (event) => event.preventDefault());
      value.addEventListener("drop", (event) => {
        event.preventDefault();
        if (builder.draggedId === null) return;
        builder.hand = moveCard(builder.hand, builder.draggedId, index);
        builder.draggedId = null;
        renderBuilder();
      });
      $("builder-hand").append(value);
    }
    const selected = builder.hand.findIndex((card) => card.id === builder.selectedId);
    $("builder-remove").disabled = selected < 0;
    $("builder-move-left").disabled = selected <= 0;
    $("builder-move-right").disabled = selected < 0 || selected === builder.hand.length - 1;
    $("builder-sort").disabled = builder.hand.length < 2;
    $("status").textContent = "";
  } catch (error) { $("status").textContent = error.message; }
}

function builderChanged() {
  renderBuilder();
  scheduleBuilderEvaluation();
}

function sampleFaces(name) {
  const classic = [0, 1, 2, 13, 14, 15, 26, 27, 28, 3, 4, 5, 19, 20, 21, 9, 22, 35, 11, 24, 37];
  const base = [0, 1, 2, 26, 27, 28, 39, 40, 41, 10, 11, 12, 8, 34, 47, 25, 38, 51];
  return {
    classic21: classic,
    exactJoker21: [...base, 16, 17, 7],
    offSuitJoker21: [...base, 16, 17, 33],
    naturalJoker21: [...base, 19, 20, 21],
    duplicateSet21: [...base.slice(0, 15), 16, 17, 18, 25, 25, 33],
    jokerRun: [7, 7, 7],
    naturalSet: [9, 22, 35],
    invalidNoise: classic.map((face) => face === 5 ? 51 : face),
  }[name];
}

function loadSample() {
  const name = $("builder-sample").value;
  const faces = sampleFaces(name);
  if (!faces) {
    $("status").textContent = "Choose a sample first.";
    return;
  }
  $("builder-hand-size").value = String(faces.length);
  $("builder-required").value = name === "jokerRun" ? "1" : name === "naturalSet" ? "0" : "5";
  $("builder-decks").value = "3";
  $("builder-joker-suit").value = "0";
  $("builder-joker-rank").value = "7";
  builder.hand = sortCards(faces.map((face) => ({ id: builder.nextId++, face })));
  builder.selectedId = null;
  builderChanged();
}

let renderedSeats = "";
function renderSeatEditor(settings) {
  const signature = JSON.stringify(settings.seats);
  if (signature === renderedSeats) return;
  renderedSeats = signature;
  $("table-seats").replaceChildren();
  for (const [index, spec] of settings.seats.entries()) {
    const row = node("div", "seat-editor-row");
    const nameLabel = node("label", "", `Seat ${index + 1}`);
    const name = node("input");
    name.value = spec.name;
    name.maxLength = 24;
    name.setAttribute("aria-label", `Seat ${index + 1} name`);
    nameLabel.append(name);
    const kindLabel = node("label", "", "Player type");
    const kind = node("select");
    kind.setAttribute("aria-label", `Seat ${index + 1} type`);
    for (const value of ["human", "ai"]) kind.append(new Option(value === "human" ? "Human" : "AI", value));
    kind.value = spec.kind;
    kindLabel.append(kind);
    const tierLabel = node("label", "", "AI tier");
    const tier = node("select");
    tier.setAttribute("aria-label", `Seat ${index + 1} AI tier`);
    for (const [value, label] of Object.entries(TIER_NAMES)) tier.append(new Option(label, value));
    tier.value = spec.tier ?? "neural";
    tierLabel.hidden = spec.kind !== "ai";
    tierLabel.append(tier);
    const change = () => {
      const seats = table.getSnapshot().settings.seats.map((seat) => ({ ...seat }));
      seats[index] = { name: name.value || `Player ${index + 1}`, kind: kind.value,
        ...(kind.value === "ai" ? { tier: tier.value } : {}) };
      table.dispatch({ type: "configure", patch: { seats } });
    };
    for (const field of [name, kind, tier]) field.addEventListener("change", change);
    const remove = button("Remove", () => {
      table.dispatch({ type: "configure", patch: {
        seats: table.getSnapshot().settings.seats.filter((_, seat) => seat !== index),
      } });
    }, { disabled: settings.seats.length === 1, className: "ghost" });
    remove.setAttribute("aria-label", `Remove seat ${index + 1}`);
    row.append(nameLabel, kindLabel, tierLabel, remove);
    $("table-seats").append(row);
  }
  $("table-add-seat").disabled = settings.seats.length >= 6;
}

function renderSummary(round) {
  const target = $("table-summary");
  const summary = round?.summary;
  target.hidden = !summary || summary.status === "none";
  target.replaceChildren();
  if (target.hidden) return;
  target.append(node("h2", "", "Round summary"));
  if (summary.status === "scoring") target.append(node("p", "muted", "Scoring hands..."));
  else if (summary.status === "error") {
    target.append(node("p", "eval-bad", summary.message),
      button("Retry summary", () => table.dispatch({ type: "retry", what: "summary" })));
  } else {
    target.append(node("p", "muted", "Deadwood scores describe this round. The scoreboard separately tracks false-declaration penalties."));
    for (const row of summary.rows) {
      const section = node("section", "summary-seat");
      section.dataset.summarySeat = String(row.seat);
      section.append(
        node("h3", "", row.name),
        node("p", "", `False declarations · ${row.roundPenalty} this round · ${row.penaltyPoints} total penalty points`),
        node("h4", "", "Final hand"),
        faceList(row.faces, round.joker),
        node("h4", "", "Declared card"),
        row.faceDown === null ? node("p", "muted", "No declared card.") : faceList([row.faceDown], round.joker),
      );
      renderPenalty(section, row.penalty, round.joker);
      target.append(section);
    }
  }
}

function renderTable() {
  const snapshot = table.getSnapshot();
  const { settings, round } = snapshot;
  renderSeatEditor(settings);
  for (const [id, value] of [
    ["table-preset", settings.preset], ["table-decks", settings.decks],
    ["table-hand-size", settings.cardsInHand], ["table-required", settings.requiredSequences],
    ["table-ai-speed", settings.aiSpeed],
  ]) $(id).value = String(value);
  $("table-recycle").checked = settings.recycleDiscard;
  $("table-notice").textContent = snapshot.notice ?? "";
  $("table-pause").hidden = !(round?.seats ?? settings.seats).some((seat) => seat.kind === "ai");
  $("table-pause").disabled = !round || round.phase === "finished";
  $("table-pause").textContent = snapshot.paused ? "Resume" : "Pause";
  const board = $("table-board");
  board.replaceChildren();
  if (!round) {
    board.append(node("p", "empty-state", "Choose a mode, edit the seats, and deal a game."));
    return;
  }
  const send = (type, fields = {}) => table.dispatch({ type, stamp: round.stamp, ...fields });
  const active = round.seats[round.currentSeat];
  if (pointer.stamp !== round.stamp) {
    pointer.stamp = round.stamp;
    pointer.selectedId = active.drawnCardId;
  }
  const outcome = round.outcome;
  const heading = outcome ? outcome.winner === null ? "Round over. No winner." :
    `${round.seats[outcome.winner].name} wins${outcome.reason === "last-remaining" ? " as the last active player" : " with a valid declaration"}.` :
    `${active.name} · Turn ${round.turn} · ${snapshot.paused ? "Paused" : active.thinking ? "Thinking..." : round.phase}`;
  board.append(node("h2", "", heading));
  const scoreboard = node("div", "seat-list");
  scoreboard.setAttribute("aria-label", "False-declaration scoreboard");
  for (const [index, seat] of round.seats.entries()) {
    const item = node("div", `seat${index === round.currentSeat ? " active" : ""}${seat.active ? "" : " eliminated"}`);
    item.dataset.penaltyTotal = String(seat.penaltyPoints);
    item.append(
      node("strong", "", seat.name),
      node("span", "muted", `${seat.handSize} cards · ${seat.active ? "Active" : "Eliminated"}`),
      node("span", "", `${seat.roundPenalty} round · ${seat.penaltyPoints} total penalty points`),
    );
    if (seat.kind === "ai") {
      const tier = TIER_NAMES[seat.effectiveTier ?? seat.tier];
      item.append(node("span", "ai-tier", `${tier}${seat.downgrade ? ` · ${seat.downgrade}` : ""}`));
      if (seat.lastReason) item.append(node("span", "muted", seat.lastReason.replaceAll("-", " ")));
    }
    scoreboard.append(item);
  }
  board.append(scoreboard);
  const piles = node("div", "table-piles");
  const stock = node("div", "pile");
  const back = node("div", "pile-card");
  back.innerHTML = cardBackSvg();
  stock.append(node("h3", "", `Stock · ${round.stockCount}`), back,
    button("Draw from stock", () => send("draw", { source: "stock" }), {
      id: "table-draw-stock", disabled: !round.canAct || round.phase !== "draw" || !round.stockCount,
    }));
  const pile = node("div", "pile");
  pile.append(node("h3", "", `Discard · ${round.pileCount}`));
  if (round.pileTop !== null) {
    const top = cardNode(round.pileTop, { joker: round.joker });
    top.disabled = true;
    pile.append(top);
  } else pile.append(node("p", "muted", "Empty pile"));
  pile.append(button("Take discard", () => send("draw", { source: "discard" }), {
    id: "table-draw-discard", disabled: !round.canAct || round.phase !== "draw" || !round.pileCount,
  }));
  const joker = node("div", "pile");
  const jokerCard = cardNode(round.joker);
  jokerCard.disabled = true;
  joker.append(node("h3", "", "Joker"), jokerCard,
    node("p", "muted", round.recycleDiscard ? "Discard recycling on" : "Discard recycling off"));
  piles.append(stock, pile, joker);
  board.append(piles);

  if (round.viewer.mode === "cover") {
    board.append(node("p", "empty-state",
      `Hand hidden for pass-and-play. Pass the device to ${active.name}, then reveal.`),
    button("Reveal hand", () => send("reveal"), { id: "table-reveal", disabled: snapshot.paused }));
  } else {
    for (const [index, seat] of round.seats.entries()) {
      if (!seat.hand) continue;
      const section = node("section", "table-hand-section");
      section.append(node("h3", "", `${seat.name}'s hand`));
      const hand = node("div", "hand");
      hand.dataset.seat = String(index);
      const actionable = round.canAct && index === round.currentSeat;
      for (const [position, card] of seat.hand.entries()) {
        const value = cardNode(card.face, {
          id: card.id, joker: round.joker, selected: actionable && card.id === pointer.selectedId,
          drawn: card.id === seat.drawnCardId,
        });
        value.disabled = !actionable;
        value.draggable = actionable;
        value.addEventListener("click", () => {
          pointer.selectedId = pointer.selectedId === card.id ? null : card.id;
          renderTable();
        });
        value.addEventListener("dragstart", () => { pointer.draggedId = card.id; });
        value.addEventListener("dragend", () => { pointer.draggedId = null; });
        value.addEventListener("dragover", (event) => event.preventDefault());
        value.addEventListener("drop", (event) => {
          event.preventDefault();
          if (pointer.draggedId !== null) send("arrange", { cardId: pointer.draggedId, index: position });
          pointer.draggedId = null;
        });
        hand.append(value);
      }
      section.append(hand);
      if (actionable) {
        const selected = seat.hand.findIndex((card) => card.id === pointer.selectedId);
        const actions = node("div", "hand-actions");
        actions.append(
          button("Move left", () => send("arrange", { cardId: pointer.selectedId, index: selected - 1 }),
            { disabled: selected <= 0 }),
          button("Move right", () => send("arrange", { cardId: pointer.selectedId, index: selected + 1 }),
            { disabled: selected < 0 || selected === seat.hand.length - 1 }),
          button("Sort hand", () => send("sort-hand")),
          button("Discard selected", () => send("discard", { cardId: pointer.selectedId }), {
            id: "table-discard-selected", disabled: round.phase !== "discard" || selected < 0,
          }),
          button("Discard drawn", () => send("discard", { cardId: seat.drawnCardId }), {
            id: "table-discard-drawn", disabled: round.phase !== "discard" || seat.drawnCardId === null,
          }),
          button("Declare win", () => send("declare", { cardId: pointer.selectedId }), {
            id: "table-declare", className: "primary", disabled: round.phase !== "discard" || selected < 0,
          }),
        );
        section.append(actions, node("p", "hand-hint",
          "Select one card to discard or place face-down for a declaration. An invalid declaration adds 80 penalty points."));
      }
      board.append(section);
    }
  }
  if (round.viewer.mode === "seat" || round.viewer.mode === "hand") {
    const label = node("label", "toggle-field");
    const check = node("input");
    check.type = "checkbox";
    check.id = "table-check-hand";
    check.checked = round.handCheck.status !== "off";
    check.addEventListener("change", () => table.dispatch({ type: "check-hand", on: check.checked }));
    label.append(check, document.createTextNode("Check hand"));
    board.append(label);
  }
  const declaration = round.declaration;
  $("table-declaration-panel").hidden = declaration.status === "none";
  $("table-declaration-retry").classList.toggle("hidden", declaration.status !== "error");
  if (declaration.status === "resolved") {
    renderEvaluation($("table-declaration"), declaration.evaluation, round.joker);
    $("table-declaration").prepend(node("p", "", `${round.seats[declaration.seat].name}'s declaration`));
  } else $("table-declaration").textContent = declaration.status === "checking" ?
    "Checking the committed declaration..." : declaration.message ?? "";
  const check = round.handCheck;
  $("table-eval-panel").hidden = check.status === "off";
  $("table-retry").classList.toggle("hidden", check.status !== "error");
  if (check.status === "ready") renderEvaluation($("table-eval"), check.evaluation, round.joker);
  else $("table-eval").textContent = check.status === "checking" ? "Checking hand..." : check.message ?? "";
  renderSummary(round);
}

function setMode(mode) {
  for (const name of ["builder", "table"]) {
    const active = mode === name;
    $(`tab-${name}`).setAttribute("aria-selected", String(active));
    $(`panel-${name}`).hidden = !active;
    $(`panel-${name}`).classList.toggle("hidden", !active);
  }
}

for (const mode of ["builder", "table"]) $(`tab-${mode}`).addEventListener("click", () => setMode(mode));
$("builder-load-sample").addEventListener("click", loadSample);
$("builder-clear").addEventListener("click", () => {
  builder.hand = [];
  builder.selectedId = null;
  builderChanged();
});
$("builder-remove").addEventListener("click", () => {
  builder.hand = builder.hand.filter((card) => card.id !== builder.selectedId);
  builder.selectedId = null;
  builderChanged();
});
for (const [id, offset] of [["builder-move-left", -1], ["builder-move-right", 1]]) {
  $(id).addEventListener("click", () => {
    const index = builder.hand.findIndex((card) => card.id === builder.selectedId);
    if (index < 0) return;
    builder.hand = moveCard(builder.hand, builder.selectedId, index + offset);
    renderBuilder();
  });
}
$("builder-sort").addEventListener("click", () => { builder.hand = sortCards(builder.hand); renderBuilder(); });
for (const id of ["builder-hand-size", "builder-required", "builder-decks", "builder-joker-suit", "builder-joker-rank"]) {
  $(id).addEventListener("change", builderChanged);
}
$("builder-retry").addEventListener("click", scheduleBuilderEvaluation);
$("table-preset").addEventListener("change", (event) => {
  if (event.target.value !== "custom") table.dispatch({ type: "preset", name: event.target.value });
});
for (const [id, key] of [
  ["table-decks", "decks"], ["table-hand-size", "cardsInHand"], ["table-required", "requiredSequences"],
  ["table-ai-speed", "aiSpeed"], ["table-recycle", "recycleDiscard"],
]) {
  $(id).addEventListener("change", () => {
    try {
      const value = key === "aiSpeed" ? $(id).value : key === "recycleDiscard" ? $(id).checked : readInt(id);
      table.dispatch({ type: "configure", patch: { [key]: value } });
    } catch (error) { $("table-notice").textContent = error.message; }
  });
}
$("table-add-seat").addEventListener("click", () => {
  const seats = table.getSnapshot().settings.seats;
  if (seats.length < 6) table.dispatch({ type: "configure", patch: {
    seats: [...seats, { kind: "human", name: `Player ${seats.length + 1}` }],
  } });
});
$("table-deal").addEventListener("click", () => table.dispatch({ type: "deal" }));
$("table-pause").addEventListener("click", () => table.dispatch({ type: table.getSnapshot().paused ? "resume" : "pause" }));
$("table-declaration-retry").addEventListener("click", () => table.dispatch({ type: "retry", what: "declaration" }));
$("table-retry").addEventListener("click", () => table.dispatch({ type: "retry", what: "hand-check" }));
table.subscribe(renderTable);
table.start();
$("builder-sample").value = "classic21";
loadSample();
window.addEventListener("pagehide", () => {
  clearTimeout(builder.timer);
  builderEngine.dispose();
  table.stop();
});
