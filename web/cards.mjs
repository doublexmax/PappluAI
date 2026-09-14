const RANKS = Object.freeze([
  "A",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "10",
  "J",
  "Q",
  "K",
]);

const SUITS = Object.freeze([
  {
    color: "#111",
    path: "M0-8C-2-4-7-1-7 3c0 4 5 6 7 2 0 3-1 5-3 6h6c-2-1-3-3-3-6 2 4 7 2 7-2 0-4-5-7-7-11Z",
  },
  {
    color: "#b81828",
    path: "M0 8C-2 5-8 1-8-5c0-5 6-7 8-2 2-5 8-3 8 2 0 6-6 10-8 13Z",
  },
  {
    color: "#b81828",
    path: "M0-9 7 0 0 9-7 0Z",
  },
  {
    color: "#111",
    path: "M-2-1C-5-4-4-9 0-9c4 0 5 5 2 8 4-3 7 0 7 4 0 4-5 6-8 2 0 2 1 4 3 5h-8c2-1 3-3 3-5-3 4-8 2-8-2 0-4 3-7 7-4Z",
  },
]);

const PIP_LAYOUTS = Object.freeze([
  [[60, 84, 0, 2.15]],
  [
    [60, 36, 0],
    [60, 132, 180],
  ],
  [
    [60, 34, 0],
    [60, 84, 0],
    [60, 134, 180],
  ],
  [
    [39, 38, 0],
    [81, 38, 0],
    [39, 130, 180],
    [81, 130, 180],
  ],
  [
    [39, 36, 0],
    [81, 36, 0],
    [60, 84, 0],
    [39, 132, 180],
    [81, 132, 180],
  ],
  [
    [39, 34, 0],
    [81, 34, 0],
    [39, 84, 0],
    [81, 84, 0],
    [39, 134, 180],
    [81, 134, 180],
  ],
  [
    [39, 32, 0],
    [81, 32, 0],
    [60, 58, 0],
    [39, 84, 0],
    [81, 84, 0],
    [39, 136, 180],
    [81, 136, 180],
  ],
  [
    [39, 30, 0],
    [81, 30, 0],
    [60, 56, 0],
    [39, 84, 0],
    [81, 84, 0],
    [60, 112, 180],
    [39, 138, 180],
    [81, 138, 180],
  ],
  [
    [39, 29, 0],
    [81, 29, 0],
    [39, 65, 0],
    [81, 65, 0],
    [60, 84, 0],
    [39, 103, 180],
    [81, 103, 180],
    [39, 139, 180],
    [81, 139, 180],
  ],
  [
    [39, 28, 0],
    [81, 28, 0],
    [60, 51, 0],
    [39, 65, 0],
    [81, 65, 0],
    [39, 103, 180],
    [81, 103, 180],
    [60, 117, 180],
    [39, 140, 180],
    [81, 140, 180],
  ],
]);

function validateFace(face) {
  if (typeof face !== "number") {
    throw new TypeError("Card face must be a number.");
  }
  if (!Number.isInteger(face)) {
    throw new TypeError("Card face must be an integer.");
  }
  if (face < 0 || face > 51) {
    throw new RangeError("Card face must be between 0 and 51.");
  }
}

function suitPath(suit) {
  return `<path d='${suit.path}'/>`;
}

function corner(rank, suit, transform = "") {
  const groupTransform = transform ? ` transform='${transform}'` : "";
  return `<g class='card-corner'${groupTransform} fill='${suit.color}'>
    <text x='16' y='28' text-anchor='middle' font-family='Times New Roman, Times, serif' font-size='${rank === "10" ? 24 : 28}' font-weight='700'>${rank}</text>
    <g transform='translate(16 40) scale(.75)'>${suitPath(suit)}</g>
  </g>`;
}

function pip(suit, [x, y, rotation, scale = 1.08]) {
  return `<g class='card-pip' transform='translate(${x} ${y}) rotate(${rotation}) scale(${scale})' fill='${suit.color}'>${suitPath(suit)}</g>`;
}

function kingHalf() {
  return `<g>
    <path d='M35 80l4-18 13-8h16l13 8 4 18Z' fill='#245a9b' stroke='#111' stroke-width='1.3'/>
    <path d='m39 63 13 17h8L48 57Zm42 0L68 80h-8l12-23Z' fill='#b81828' stroke='#111' stroke-width='1.1'/>
    <path d='M53 54h14l-1 8H54Z' fill='#d2a53a' stroke='#111' stroke-width='1'/>
    <path d='M48 44c0-9 24-9 24 0v7c0 10-5 15-12 15s-12-5-12-15Z' fill='#f0c59d' stroke='#111' stroke-width='1.2'/>
    <path d='M47 44c-5 5-4 15 2 19l5-2-3-9 2-10Zm26 0c5 5 4 15-2 19l-5-2 3-9-2-10Z' fill='#5b3528' stroke='#111' stroke-width='1'/>
    <path d='M49 42 47 31l7 6 6-9 6 9 7-6-2 11Z' fill='#d2a53a' stroke='#111' stroke-width='1.2'/>
    <circle cx='60' cy='35' r='2.2' fill='#b81828' stroke='#111' stroke-width='.8'/>
    <path d='M55 48h3m4 0h3m-8 9c2 2 4 2 6 0' fill='none' stroke='#111' stroke-width='1.1' stroke-linecap='round'/>
  </g>`;
}

function queenHalf() {
  return `<g>
    <path d='M34 80l5-18 14-8h14l14 8 5 18Z' fill='#b81828' stroke='#111' stroke-width='1.3'/>
    <path d='m40 62 13 18h7L48 57Zm40 0L67 80h-7l12-23Z' fill='#245a9b' stroke='#111' stroke-width='1.1'/>
    <path d='M51 56h18l-3 10H54Z' fill='#d2a53a' stroke='#111' stroke-width='1'/>
    <path d='M49 43c0-9 22-9 22 0v9c0 9-4 14-11 14s-11-5-11-14Z' fill='#f0c59d' stroke='#111' stroke-width='1.2'/>
    <path d='M47 43c-3 5-4 17 2 23l6-3-3-11 2-12Zm26 0c3 5 4 17-2 23l-6-3 3-11-2-12Z' fill='#9b632d' stroke='#111' stroke-width='1'/>
    <path d='M50 42 48 34l7 4 5-9 5 9 7-4-2 8Z' fill='#d2a53a' stroke='#111' stroke-width='1.2'/>
    <circle cx='55' cy='38' r='1.6' fill='#245a9b'/>
    <circle cx='65' cy='38' r='1.6' fill='#b81828'/>
    <path d='M55 48h3m4 0h3m-7 9c1 1 3 1 4 0' fill='none' stroke='#111' stroke-width='1.1' stroke-linecap='round'/>
  </g>`;
}

function jackHalf() {
  return `<g>
    <path d='M34 80l6-18 13-7h14l13 7 6 18Z' fill='#245a9b' stroke='#111' stroke-width='1.3'/>
    <path d='m40 62 11 18h9L49 58Zm40 0L69 80h-9l11-22Z' fill='#b81828' stroke='#111' stroke-width='1.1'/>
    <circle cx='55' cy='68' r='2' fill='#d2a53a' stroke='#111' stroke-width='.8'/>
    <circle cx='65' cy='68' r='2' fill='#d2a53a' stroke='#111' stroke-width='.8'/>
    <path d='M49 44c0-8 22-8 22 0v8c0 9-4 14-11 14s-11-5-11-14Z' fill='#f0c59d' stroke='#111' stroke-width='1.2'/>
    <path d='M47 44c-2 6 0 14 4 18l4-3-3-9 2-9Zm26 0c2 6 0 14-4 18l-4-3 3-9-2-9Z' fill='#6a412a' stroke='#111' stroke-width='1'/>
    <path d='M47 43c5-10 21-13 28-3l-5 6c-7-4-14-4-21 0Z' fill='#b81828' stroke='#111' stroke-width='1.2'/>
    <path d='M70 37c4-8 9-10 12-8-4 2-5 6-6 11Z' fill='#d2a53a' stroke='#111' stroke-width='1'/>
    <path d='M55 48h3m4 0h3m-7 9c1 1 3 1 4 0' fill='none' stroke='#111' stroke-width='1.1' stroke-linecap='round'/>
  </g>`;
}

function court(rankOffset, suit) {
  const portrait =
    rankOffset === 10
      ? jackHalf()
      : rankOffset === 11
        ? queenHalf()
        : kingHalf();
  return `<g class='card-court'>
    <rect x='31' y='28' width='58' height='112' fill='#f8e8bd' stroke='#111' stroke-width='1.5'/>
    ${portrait}
    <g transform='rotate(180 60 84)'>${portrait}</g>
    <path d='M31 78h58v12H31Z' fill='#d2a53a' stroke='#111' stroke-width='1.2'/>
    <path d='M38 84h44' fill='none' stroke='#b81828' stroke-width='4'/>
    <g transform='translate(60 84) scale(.7)' fill='${suit.color}' stroke='#f8e8bd' stroke-width='1.2'>${suitPath(suit)}</g>
  </g>`;
}

function faceBody(rankOffset, suit) {
  if (rankOffset <= 9) {
    return PIP_LAYOUTS[rankOffset].map((position) => pip(suit, position)).join("");
  }
  return court(rankOffset, suit);
}

export function cardFaceSvg(face) {
  validateFace(face);
  const suit = SUITS[Math.floor(face / 13)];
  const rankOffset = face % 13;
  const rank = RANKS[rankOffset];
  return `<svg class='card-art card-art--face' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 168' aria-hidden='true' focusable='false' data-card-face='${face}'>
    <rect x='1' y='1' width='118' height='166' rx='3' fill='#fffdf7' stroke='#292724' stroke-width='2'/>
    ${corner(rank, suit)}
    ${faceBody(rankOffset, suit)}
    ${corner(rank, suit, "rotate(180 60 84)")}
  </svg>`;
}

export function cardBackSvg() {
  return `<svg class='card-art card-art--back' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 168' aria-hidden='true' focusable='false'>
    <rect x='1' y='1' width='118' height='166' rx='3' fill='#fffdf7' stroke='#292724' stroke-width='2'/>
    <rect x='7' y='7' width='106' height='154' rx='2' fill='#173f78' stroke='#111' stroke-width='1'/>
    <rect x='12' y='12' width='96' height='144' fill='none' stroke='#f4e8c7' stroke-width='2'/>
    <path d='M14 28 60 13l46 15-46 15Zm0 28 46-15 46 15-46 15Zm0 28 46-15 46 15-46 15Zm0 28 46-15 46 15-46 15Zm0 28 46-15 46 15-46 15Z' fill='none' stroke='#d7a83d' stroke-width='2'/>
    <path d='M14 14 106 54M14 42l92 40M14 70l92 40M14 98l92 40M14 126l66 29M106 14 14 54m92-12L14 82m92-12-92 40m92-12-92 40m92-12-66 29' fill='none' stroke='#f4e8c7' stroke-width='1.4'/>
    <path d='M60 48 86 84 60 120 34 84Zm0 13 16 23-16 23-16-23Z' fill='#b81828' stroke='#f4e8c7' stroke-width='2'/>
    <path d='M60 69 71 84 60 99 49 84Z' fill='#d7a83d' stroke='#111' stroke-width='1'/>
    <circle cx='60' cy='84' r='4' fill='#173f78' stroke='#f4e8c7' stroke-width='1.5'/>
  </svg>`;
}
