import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const driver = process.env.PAPPLU_PLAYWRIGHT;
const { chromium } = await import(driver ? pathToFileURL(driver).href : "playwright");
const browser = await chromium.launch({ channel: process.env.PAPPLU_BROWSER_CHANNEL ?? "chrome", headless: true });
const url = process.argv[2] ?? "http://127.0.0.1:8765";
const errors = [];
const apiCalls = [];

async function setup({ fallback = false, mobile = false } = {}) {
  const context = await browser.newContext({ viewport: mobile ? { width: 390, height: 844 } : { width: 1280, height: 900 } });
  await context.addInitScript((withoutWorkers) => {
    Math.random = () => 0.9999;
    if (withoutWorkers) Object.defineProperty(window, "Worker", { value: undefined });
  }, fallback);
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.startsWith("/api/")) apiCalls.push(request.url());
  });
  await page.goto(url);
  await page.getByText("Minimum deadwood · 0 points", { exact: true }).waitFor();
  return { context, page };
}

async function setNumber(page, id, value) {
  await page.locator(`#${id}`).fill(String(value));
  await page.locator(`#${id}`).press("Tab");
}

try {
  const { context, page } = await setup();
  assert.equal(await page.locator("#builder-hand [data-card-id]").count(), 21);
  await page.locator("#builder-clear").click();
  await page.getByText("0 of 21 cards.", { exact: false }).waitFor();
  await page.getByRole("button", { name: "Add A of Spades", exact: true }).click();
  await page.getByText("Minimum deadwood · 10 points", { exact: true }).waitFor();
  await page.locator("#builder-sample").selectOption("exactJoker21");
  await page.locator("#builder-load-sample").click();
  await page.locator("#builder-eval .eval-ok").waitFor();
  assert.equal(await page.locator("#builder-hand [data-card-id]").count(), 21);
  console.log("PASS builder partial scoring and exact-joker grouping.");

  await page.locator("#tab-table").click();
  assert.equal(await page.locator(".seat-editor-row").count(), 2);
  await page.getByRole("textbox", { name: "Seat 1 name", exact: true }).fill("Alice");
  await page.getByRole("textbox", { name: "Seat 1 name", exact: true }).press("Tab");
  await page.locator("#table-add-seat").click();
  assert.equal(await page.locator(".seat-editor-row").count(), 3);
  await page.getByRole("button", { name: "Remove seat 3", exact: true }).click();
  assert.equal(await page.locator(".seat-editor-row").count(), 2);
  await page.locator("#table-preset").selectOption("solo");
  await setNumber(page, "table-hand-size", 3);
  await setNumber(page, "table-required", 1);
  await setNumber(page, "table-decks", 1);
  await page.locator("#table-deal").click();
  assert.equal(await page.locator("#table-board .hand [data-card-id]").count(), 3);
  await page.locator("#table-draw-stock").click();
  assert.equal(await page.locator("#table-board .hand [data-card-id]").count(), 4);
  await page.locator("#table-declare").click();
  await page.locator("#table-summary [data-summary-seat]").waitFor();
  assert.equal(await page.locator("#table-summary [data-summary-seat]").count(), 1);
  assert.match(await page.locator("#table-summary").innerText(), /Minimum deadwood · 0 points/);
  assert.match(await page.locator("#table-summary").innerText(), /Declared card/);
  assert.match(await page.locator("#table-board h2").innerText(), /valid declaration/);
  console.log("PASS custom seats, worker declaration, and round summary.");

  await page.locator("#table-preset").selectOption("pass-and-play");
  await page.locator("#table-deal").click();
  assert.equal(await page.locator("#table-board .hand [data-card-id]").count(), 0);
  await page.locator("#table-reveal").click();
  assert.equal(await page.locator("#table-board .hand [data-card-id]").count(), 3);
  await page.locator("#table-draw-stock").click();
  assert.equal(await page.locator("#table-board .hand [data-card-id]").count(), 4);
  await page.locator("#table-discard-drawn").click();
  await page.locator("#table-reveal").waitFor();
  assert.equal(await page.locator("#table-board .hand [data-card-id]").count(), 0);
  console.log("PASS pass-and-play privacy between turns.");

  const workers = page.workers().filter((worker) => new URL(worker.url()).pathname === "/engine-worker.mjs");
  assert.ok(workers.length > 0);
  await Promise.all(workers.map((worker) => worker.evaluate(() => {
    self.tacticalPhases = [];
    self.addEventListener("message", ({ data }) => {
      if (data.op === "chooseMove" && data.tier === "tactical") self.tacticalPhases.push(data.view.phase);
    });
  })));
  await page.locator("#table-preset").selectOption("vs-ai");
  await page.locator("#table-ai-speed").selectOption("instant");
  await page.locator("#table-deal").click();
  await page.locator("#table-draw-stock").click();
  await page.locator("#table-discard-drawn").click();
  await page.getByRole("heading", { name: "You · Turn 3 · draw", exact: true }).waitFor({ timeout: 30_000 });
  assert.equal(await page.locator("#table-board .hand [data-card-id]").count(), 21);
  assert.equal(await page.locator(".seat-list .ai-tier").innerText(), "Tactical");
  assert.equal(await page.getByText("min penalty", { exact: true }).count(), 1);
  const phases = await Promise.all(workers.map((worker) => worker.evaluate(() => self.tacticalPhases)));
  assert.deepEqual(phases.flat(), ["draw", "discard"]);
  console.log("PASS a Tactical AI draw and discard through the real worker.");

  await page.locator("#table-preset").selectOption("ai-watch");
  await page.locator("#table-ai-speed").selectOption("slow");
  await page.locator("#table-deal").click();
  await page.locator("#table-pause").click();
  assert.equal(await page.locator("#table-pause").innerText(), "Resume");
  const paused = await page.locator("#table-board h2").innerText();
  await page.waitForTimeout(400);
  assert.equal(await page.locator("#table-board h2").innerText(), paused);
  assert.ok(await page.locator("#table-board .hand [data-card-id]").count() >= 63);
  assert.deepEqual(await page.locator(".seat-list .ai-tier").allTextContents(), ["Tactical", "Tactical", "Casual"]);
  await page.locator("#table-pause").click();
  assert.equal(await page.locator("#table-pause").innerText(), "Pause");
  await page.locator("#table-pause").click();
  console.log("PASS AI Watch spectator hands and pause-resume.");
  await context.close();

  const fallback = await setup({ fallback: true, mobile: true });
  await fallback.page.locator("#tab-table").click();
  await fallback.page.locator("#table-preset").selectOption("solo");
  await setNumber(fallback.page, "table-hand-size", 3);
  await setNumber(fallback.page, "table-required", 1);
  await setNumber(fallback.page, "table-decks", 1);
  await fallback.page.locator("#table-deal").click();
  await fallback.page.locator("#table-draw-stock").click();
  await fallback.page.locator("#table-declare").click();
  await fallback.page.locator("#table-summary [data-summary-seat]").waitFor();
  assert.match(await fallback.page.locator("#table-summary").innerText(), /Minimum deadwood · 0 points/);
  assert.equal(await fallback.page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  const screenshot = process.env.PAPPLU_SCREENSHOT;
  if (screenshot) await fallback.page.screenshot({ path: screenshot, fullPage: true });
  await fallback.context.close();
  assert.deepEqual(apiCalls, []);
  assert.deepEqual(errors, []);
  console.log("PASS mobile layout, main-thread fallback, and zero evaluator API requests.");
} finally {
  await browser.close();
}
