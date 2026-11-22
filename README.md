# PappluAI

Papplu hand evaluation, shared-deck play, and a Q-network trained by self-play.
The evaluator implements this project's house rules, not a universal rummy ruleset.
The browser runs without a build or a remote API. Training needs PyTorch.

## Run the simulator

Start the local server from the repository root with Python 3.9 or later.

```powershell
python -m src.simulator
```

Open `http://127.0.0.1:8765`. Use `--port 8766` to change the port.
Stop the server with Ctrl+C. The server accepts local requests only.

### Hand builder

The settings panel controls the hand size, sequence requirement, and selected joker.
Add cards with the picker, remove a selected copy, or load a sample.
The evaluator reports complete meld witnesses for winning hands.
It also reports minimum deadwood points for partial and complete hands.
A losing declaration has binary reward zero, not zero useful sequences.

Card faces use original SVG artwork. No external images or fonts are downloaded.
Face-down cards use a generic back that reveals no face information.

### Game table

Tables support one through six Human or AI seats.
The presets are vs AI, Solo Practice, Pass & Play, and AI Watch.
Seat names, AI tiers, and rules apply at the next deal.
The current round keeps its original settings.

Draw from the stock or the top discard, then discard from the enlarged hand.
Humans can return either drawn card immediately.
Neural policies keep a discard-pile pickup when another face can be discarded.
That policy restriction prevents repeated pickup-and-return cycles.
The underlying game still allows the move.

Pass & Play covers hands between turns.
With one human, that hand stays visible during AI turns.
AI Watch reveals hands to spectators, but each AI receives only its own hand and public events.
Opponent stock faces and face-down declarations stay hidden.

The speed settings are Instant, Normal, and Slow.
Pause and Resume control AI play. AI Watch pauses after 200 completed turns.
The optional Check hand panel does not declare a win or end a round.

When stock empties, the default rule shuffles all but the top discard into stock.
Turn off recycling to leave stock empty. Deal again if both piles are empty.
The selected joker indicator stays outside the playable piles.

### AI tiers

- **Neural** chooses the legal action with the highest Q-value. Its reported reason is `greedy`.
- **Tactical** minimizes exact deadwood penalty and uses public-card beliefs, feed risk, and one-ply outs.
- **Casual** keeps wild cards and favors nearby ranks and matching cards.

Every tier first declares an exact winning discard when one exists.
Neural has no draw lookahead, belief search, or feed-risk adjustment.
It falls back to Tactical for unavailable weights, incompatible rules, or solo browser play.
The scoreboard reports the effective tier and any downgrade.

`web/papplu-model.json` contains the trained Azure `card_mlp` v5 checkpoint.
It uses `ModelSpec(128, 3)`, has 673,412 parameters, and reads the 1,275-float version 5 encoding.
The export records `source.kind` as `checkpoint` and `source.trained` as `true`.

### Held-out results

The evaluation locked the checkpoint before the held-out run. The deal seeds did not enter its training or selection.
Earlier project models used some of these seed ranges, so the deals are held out for this checkpoint, not for the project as a whole.
Node ran the browser engine with three decks, 21 cards, five required pure sequences, and a 60-turn cap per seat.

Cold solo won 488 of 500 deals starting at seed `850000`, or 97.60%. The 95% Wilson interval is 95.85% to 98.62%.
Mean terminal deadwood was 1.148 points. This test calls the browser neural policy directly. The solo UI still downgrades Neural to Tactical.

For multiplayer, the candidate played once in every seat on each deal, with the same opponent in all other seats.
A1 is the 64-by-2 model exported in `a4aa60c`.
First-place credit goes to the declaration winner, or to the lowest deadwood penalty at the cap. Tied first seats split the credit.
Pairwise credit gives 1 for placing ahead of an opponent and 0.5 for a tie. Neutral pairwise credit is 50%.
The first-place intervals use 10,000 resamples of complete deals, preserving their seat rotations.

| Opponent | Players | First seed | Deals | Declaration wins | First-place credit, 95% deal interval | Pairwise credit | Mean penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 2 | 860000 | 120 | 129/240 | 53.75% (48.75% to 58.75%) | 53.75% | 17.51 |
| A1 | 3 | 860000 | 80 | 76/240 | 31.67% (26.67% to 36.67%) | 52.08% | 28.18 |
| A1 | 4 | 860000 | 60 | 64/240 | 26.67% (22.08% to 31.25%) | 54.58% | 32.67 |
| Casual | 2 | 870000 | 100 | 175/200 | 87.50% (83.00% to 92.00%) | 87.50% | 4.30 |
| Casual | 3 | 870000 | 60 | 124/180 | 68.89% (62.22% to 75.00%) | 82.50% | 10.25 |
| Casual | 4 | 870000 | 50 | 118/200 | 59.00% (52.50% to 65.50%) | 81.67% | 15.37 |
| Tactical | 2 | 880000 | 60 | 68/120 | 56.67% (49.17% to 64.17%) | 56.67% | 17.23 |
| Tactical | 3 | 880000 | 40 | 41/120 | 34.17% (27.50% to 40.83%) | 49.79% | 27.77 |
| Tactical | 4 | 880000 | 30 | 37/120 | 30.83% (24.17% to 37.50%) | 55.69% | 28.29 |

All multiplayer games ended in a declaration, so the declaration counts are also the outright first-place counts.
The model beats equal-seat first-place parity against Casual at each table size. The intervals against A1 and Tactical include parity.
See the [training guide](TRAINING.md#shipped-checkpoint) for the separate Python arena results and inference timing.

### Declarations and scoring

After drawing, select a card and choose Declare win.
The card goes face down into a separate declaration area.
The evaluator checks the remaining hand.

| Result | Outcome |
| --- | --- |
| Valid declaration | The declaring player wins. |
| Invalid declaration | The player receives 80 penalty points and leaves the round. |
| Evaluation error or timeout | The declaration stays pending and retryable. No penalty is assigned. |

Other active players continue after an invalid declaration.
The last remaining player wins. An invalid solo declaration loses.
Cards cannot change while the verdict is pending.
Penalty totals carry across deals with the same player count.
The final summary reveals hands, declaration cards, deadwood, and exempt melds.
Deadwood points remain separate from false-declaration penalties and binary reward.

The declaration penalty follows [RummyCircle's points-rummy rules](https://www.rummycircle.com/rummy-variations/points-rummy.html).
That source describes 13-card play. This project adopts the penalty as a house rule for 21 cards.
[Pagat describes differences between 21-card variants](https://www.pagat.com/rummy/indian.html).

### Card order and persistence

Drag cards or use the move controls to arrange a hand.
Sort restores suit, rank, and physical-copy order.
Manual order survives draws, discards, and checks within the round.
Drawing appends the new card.

Builder and table hands are separate.
Reloading resets hands and scores.
Seat and rule preferences use `papplu:settings:v1` in local storage.
There are no network players or saved rounds.

Module workers run evaluation and AI decisions.
Browsers without workers use the same local engine on the main thread.
That fallback can block during an exact evaluation and cannot interrupt it.
A worker timeout reports an error, never a losing-hand verdict.
Any static HTTP server can serve `web/`.
The Python server also supports `POST /api/evaluate`.

## House rules

The default winning hand has 21 cards and at least five pure sequences.
Every card belongs to exactly one legal group.

| Group | Rule |
| --- | --- |
| Sequence | At least three consecutive ranks in one suit, or three identical natural copies of one face. |
| Required pure sequence | No substitution except by the exact selected joker face. Identical natural triplets also qualify. |
| Set | Three or four cards of one rank with distinct represented suits. Sets do not meet the pure-sequence requirement. |

Ace is low in A-2-3 or high in Q-K-A. K-A-2 is invalid.
A long sequence counts once unless it splits into disjoint legal sequences.
Additional sequences and sets can use joker-rank wild cards.
Multiple physical copies can belong to separate groups.

When `8s` is selected, all eights are wild.
The exact `8s` can substitute in a required pure sequence such as `4h 5h 8s`.
An `8d` can appear naturally in `7d 8d 9d`.
It cannot substitute for `6h` in a required pure sequence.
Three copies of the exact selected joker form a qualifying triplet.
`7s 7s 8d` is not a set, but `7s 7h 8d` can be.

The browser supports one through six decks, hand sizes of 3 through 30, and sequence requirements of 0 through 10.
The deal must leave enough cards for every hand, the indicator, and the first discard.

## Python evaluator

`src.evaluate` has no third-party dependencies.
Card indices are `suit * 13 + rank`.
Suits are spades, hearts, diamonds, and clubs.
Ranks are Ace through King, with Ace at zero.
Hands are 52 nonnegative integer counts, not physical card IDs.

```python
from src.evaluate import evaluate_hand, hand_reward, minimum_penalty

hand = [0] * 52
hand[0] = hand[1] = hand[2] = 1
result = evaluate_hand(hand, joker=7, required_sequences=1, cards_in_hand=3)
assert result.is_valid
assert hand_reward(hand, joker=7, required_sequences=1, cards_in_hand=3) == 1.0
assert minimum_penalty(hand, joker=7, required_sequences=1).points == 0
```

`evaluate_hand` returns an immutable `HandEvaluation` with exact meld witnesses.
`is_valid_hand` returns the declaration verdict.
`hand_reward` returns 1.0 for a valid complete declaration and 0.0 otherwise.
`minimum_penalty` accepts any hand size and returns an exact `HandPenalty`.
A zero penalty alone does not establish a valid declaration.

Numbered cards score face value. A/J/Q/K score 10. Joker-rank cards score zero.
Qualifying pure sequences are exempt.
Other melds become exempt only when that grouping meets the sequence requirement.
The search considers alternative groupings and can be costly on joker-heavy hands.
Malformed counts and settings raise `TypeError` or `ValueError`.
The evaluator does not mutate its inputs.

## Model and training contract

Python training uses `GameTransformer` v6. The shipped browser model above is a separate v5 artifact. Python training does not export or overwrite it.
See the [training guide](TRAINING.md) to start, resume, or evaluate a run on Azure.

| Package | Responsibility |
| --- | --- |
| `src.game` | Environments, public observations, terminal outcomes, and stock handling. |
| `src.model` | Byte encoding, the game-independent Transformer, and versioned model checkpoints. |
| `src.training.train` | Self-play collection, rollout loading, lag-bounded replay, and V-trace updates. |
| `src.evaluation` | One batched arena for collection, solo evaluation, and multiplayer evaluation. |
| `src.checkpoints` | Atomic files and tensor validation. |

The encoder stores eight bytes per token. Its append-only prefix contains the rules, indicator, observer's initial hand, and visible events. Each decision adds a 30-token snapshot of the current table and the observer's hand. No token contains meld estimates, unseen cards, or point features.

The Transformer reads the prefix causally. Each decision reads its own snapshot and the prefix available at that point. Actors keep prefix keys and values per seat. The learner computes a shared prefix once per sampled window. The default model has six layers, width 256, eight heads, and a SwiGLU width of 704.

The raw terminal score is `+1` for the declarer and `-penalty / (10 * cards_in_hand)` for other seats. Solo keeps the raw score. Multiplayer gives each seat half the difference between its score and the other seats' mean, so utilities sum to zero. No deadwood points are clipped.

## Verification

Run model verification from the repository root on the Azure compute host.

```bash
python tools/check_training_architecture.py
python -m unittest discover -s tests -v
node --test tests/*.test.mjs
```

The architecture checker rejects import cycles and function-local imports.
It checks portable imports with Torch and NumPy blocked.
CI also exercises training, resume, standalone collection, and both evaluation commands.

`tools/verify_browser.mjs` exercises the real browser through Playwright.
It checks worker decisions, the Neural label and greedy reason, declarations, privacy, and the main-thread fallback.
