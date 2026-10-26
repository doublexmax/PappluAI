import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  evaluateHand, handReward, isValidHand, minimumPenalty, qualifyingSequences, scoreHands,
} from "../web/evaluate.mjs";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/evaluate-parity.json", import.meta.url), "utf8"));
for (const entry of fixture.cases) {
  if (Array.isArray(entry.hand)) Object.freeze(entry.hand);
}

function optionsFor(entry) {
  return { requiredSequences: entry.required_sequences, cardsInHand: entry.cards_in_hand };
}

function assertPythonError(call, expected, name) {
  const type = expected.type === "ValueError" ? RangeError : TypeError;
  assert.throws(call, (error) => {
    assert.equal(error.constructor, type, name);
    if (type === RangeError) assert.equal(error.message, expected.message, name);
    return true;
  }, name);
}

test("fixtures identify their current Python sources and exporter", () => {
  for (const [path, expected] of Object.entries(fixture.sources)) {
    const source = readFileSync(new URL(`../${path}`, import.meta.url), "utf8").replace(/\r\n?/g, "\n");
    assert.equal(createHash("sha256").update(source).digest("hex"), expected, path);
  }
});

test("evaluateHand matches every Python witness and validation error", () => {
  for (const entry of fixture.cases) {
    const call = () => evaluateHand(entry.hand, entry.joker, optionsFor(entry));
    if (entry.errors?.evaluation) {
      assertPythonError(call, entry.errors.evaluation, entry.name);
    } else {
      assert.deepEqual(call(), entry.evaluation, entry.name);
    }
  }
});

test("isValidHand and handReward match every Python result and error", () => {
  for (const entry of fixture.cases) {
    for (const [function_, field] of [[isValidHand, "is_valid_hand"], [handReward, "hand_reward"]]) {
      const call = () => function_(entry.hand, entry.joker, optionsFor(entry));
      if (entry.errors?.evaluation) {
        assertPythonError(call, entry.errors.evaluation, `${field} ${entry.name}`);
      } else {
        assert.equal(call(), entry[field], `${field} ${entry.name}`);
      }
    }
  }
});

test("minimumPenalty matches every Python optimum and exact ordered witness", () => {
  for (const entry of fixture.cases) {
    const call = () => minimumPenalty(entry.hand, entry.joker, optionsFor(entry));
    if (entry.errors?.penalty) {
      assertPythonError(call, entry.errors.penalty, entry.name);
    } else {
      assert.deepEqual(call(), entry.penalty, entry.name);
    }
  }
});

test("qualifyingSequences reports the selected penalty grouping for every case", () => {
  for (const entry of fixture.cases) {
    const call = () => qualifyingSequences(entry.hand, entry.joker, optionsFor(entry));
    if (entry.errors?.penalty) {
      assertPythonError(call, entry.errors.penalty, entry.name);
    } else {
      assert.equal(call(), entry.penalty.qualifying_sequences, entry.name);
    }
  }
});

test("scoreHands scores count vectors in order and propagates every scoring error", () => {
  const batches = new Map();
  for (const entry of fixture.cases) {
    if (entry.errors?.penalty) {
      assertPythonError(
        () => scoreHands([entry.hand], entry.joker, optionsFor(entry)),
        entry.errors.penalty, entry.name,
      );
      continue;
    }
    const key = JSON.stringify([entry.joker, entry.required_sequences, entry.cards_in_hand]);
    if (!batches.has(key)) batches.set(key, []);
    batches.get(key).push(entry);
  }
  for (const entries of batches.values()) {
    const first = entries[0];
    const hands = Object.freeze(entries.map((entry) => entry.hand));
    assert.deepEqual(
      scoreHands(hands, first.joker, optionsFor(first)),
      entries.map((entry) => entry.penalty), entries.map((entry) => entry.name).join(", "),
    );
  }
  assert.deepEqual(scoreHands([], 7), []);
});

test("the default options match the Python 21-card five-sequence declaration", () => {
  const entry = fixture.cases.find((candidate) => candidate.name === "classic_21");
  assert.deepEqual(evaluateHand(entry.hand, entry.joker), entry.evaluation);
  assert.equal(isValidHand(entry.hand, entry.joker), true);
  assert.equal(handReward(entry.hand, entry.joker), 1);
  assert.deepEqual(minimumPenalty(entry.hand, entry.joker), entry.penalty);
  assert.equal(qualifyingSequences(entry.hand, entry.joker), entry.penalty.qualifying_sequences);
  assert.deepEqual(scoreHands([entry.hand], entry.joker), [entry.penalty]);
});

test("penalty ties select a quota-meeting grouping even when unmatched cards cost zero", () => {
  for (const [name, points, qualifying] of [
    ["zero_point_quota_tie", 0, 2], ["positive_point_quota_tie", 8, 3],
  ]) {
    const entry = fixture.cases.find((candidate) => candidate.name === name);
    const result = minimumPenalty(entry.hand, entry.joker, optionsFor(entry));
    assert.equal(result.points, points, name);
    assert.equal(result.qualifying_sequences, qualifying, name);
    assert.deepEqual(result, entry.penalty, name);
  }
});

test("Python winning_discard fixtures select the first face whose remainder declares", () => {
  let checked = 0;
  for (const entry of fixture.cases) {
    if (!Object.hasOwn(entry, "winning_discard")) continue;
    const total = entry.hand.reduce((value, copies) => value + copies, 0);
    let winner = null;
    if (total === entry.cards_in_hand + 1) {
      checked += 1;
      const remaining = entry.hand.slice();
      for (let face = 0; face < remaining.length; face += 1) {
        if (remaining[face] === 0) continue;
        remaining[face] -= 1;
        const valid = isValidHand(remaining, entry.joker, optionsFor(entry));
        remaining[face] += 1;
        if (valid) {
          winner = face;
          break;
        }
      }
    }
    assert.equal(winner, entry.winning_discard, entry.name);
    if (isValidHand(entry.hand, entry.joker, optionsFor(entry))) {
      assert.equal(entry.can_declare_ub, true, entry.name);
    }
  }
  assert.equal(checked >= 200, true);
});

test("typed count vectors are accepted and returned witnesses do not leak mutable state", () => {
  const entry = fixture.cases.find((candidate) => candidate.name === "classic_21");
  const hand = Uint8Array.from(entry.hand);
  assert.deepEqual(evaluateHand(hand, entry.joker), entry.evaluation);
  assert.deepEqual(minimumPenalty(hand, entry.joker), entry.penalty);
  assert.deepEqual(Array.from(hand), entry.hand);
  const evaluation = evaluateHand(entry.hand, entry.joker);
  evaluation.melds[0].cards[0] = 51;
  evaluation.melds[0].represented_cards[0] = 51;
  const penalty = minimumPenalty(entry.hand, entry.joker);
  penalty.counted_cards[0] += 1;
  penalty.exempt_melds[0].cards[0] = 51;
  assert.deepEqual(evaluateHand(entry.hand, entry.joker), entry.evaluation);
  assert.deepEqual(minimumPenalty(entry.hand, entry.joker), entry.penalty);
});
