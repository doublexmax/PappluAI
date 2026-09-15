import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { cardFaceSvg, cardBackSvg } from "../web/cards.mjs";

function classCount(svg, name) {
  return [...svg.matchAll(/\bclass=["']([^"']*)["']/g)]
    .filter((match) => match[1].split(/\s+/).includes(name)).length;
}

function assertSelfContained(svg) {
  assert.match(svg, /^\s*<svg\b/);
  assert.match(svg, /\bviewBox=["'][^"']+["']/);
  assert.match(svg, /\baria-hidden=["']true["']/);
  assert.doesNotMatch(svg, /<script|<image|<foreignObject|\bon\w+=|\bid=|url\(|href=/i);
  assert.doesNotMatch(svg, /[\u2660\u2665\u2666\u2663\u{1F0A0}]/u);
}

describe("card artwork", () => {
  it("renders all 52 faces with two corners and the right pip count", () => {
    const faces = new Set();
    for (let face = 0; face < 52; face += 1) {
      const svg = cardFaceSvg(face);
      const rank = face % 13;
      assertSelfContained(svg);
      assert.equal(classCount(svg, "card-corner"), 2, `face ${face}`);
      assert.equal(classCount(svg, "card-pip"), rank < 10 ? rank + 1 : 0, `face ${face}`);
      assert.equal(classCount(svg, "card-court"), rank >= 10 ? 1 : 0, `face ${face}`);
      assert.doesNotMatch(svg, /\bdata-face=/);
      faces.add(svg);
    }
    assert.equal(faces.size, 52);
  });

  it("keeps the renderer deterministic without altering game data", () => {
    const cards = Object.freeze([
      Object.freeze({ id: 1, face: 7 }),
      Object.freeze({ id: 53, face: 7 }),
    ]);
    assert.equal(cardFaceSvg(cards[0].face), cardFaceSvg(cards[1].face));
    assert.deepEqual(cards, [{ id: 1, face: 7 }, { id: 53, face: 7 }]);
  });

  it("balances pip positions on centrally symmetric number cards", () => {
    for (const rank of [0, 1, 2, 3, 4, 5, 7, 8, 9]) {
      const positions = [...cardFaceSvg(rank).matchAll(
        /class=["']card-pip["']\s+transform=["']translate\(([\d.]+)\s+([\d.]+)/g,
      )].map((match) => [Number(match[1]), Number(match[2])]);
      assert.equal(positions.length, rank + 1);
      assert.deepEqual(
        positions.map(([x, y]) => `${x},${y}`).sort(),
        positions.map(([x, y]) => `${120 - x},${168 - y}`).sort(),
        `rank offset ${rank}`,
      );
    }
  });

  it("rejects invalid card indices explicitly", () => {
    for (const value of [-1, 52, 1.5, NaN, Infinity, "7", null, undefined, true]) {
      assert.throws(
        () => cardFaceSvg(value),
        (error) => error instanceof TypeError || error instanceof RangeError,
        String(value),
      );
    }
  });

  it("uses one generic back with no face or corner information", () => {
    const svg = cardBackSvg();
    assertSelfContained(svg);
    assert.equal(svg, cardBackSvg());
    assert.equal(classCount(svg, "card-corner"), 0);
    assert.equal(classCount(svg, "card-pip"), 0);
    assert.equal(classCount(svg, "card-court"), 0);
    assert.doesNotMatch(svg, /data-(?:card-)?face|data-rank|data-suit/);
  });
});
