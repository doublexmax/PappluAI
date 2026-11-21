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

`web/papplu-model.json` contains trained `card_mlp` v5 weights from Azure checkpoint A1, `trackA_p1/best-model.pt`.
Its `source.kind` is `checkpoint` and `source.trained` is `true`.
The network has 127,172 parameters, width 64, and two residual blocks.
It reads the 1,275-float version 5 encoding.
The checkpoint SHA256 is `094bbe8946fc890e93de7a2ce77f0ad45ad8f8d24a38f2249ffb31fdef39882b`.

### A1 held-out results

A1 was locked before this evaluation.
These seeds are disjoint from its production training and selection.
Earlier project models used the same seed ranges, so they are not project-wide unseen deals.
Multiplayer matches use three decks, 21 cards, five required pure sequences, and 60 turns per seat.

Python cold solo won 243 of 256 deals at seed 850000, or 94.9%.
Its 95% Wilson interval is 91.5% to 97.0%, with mean terminal penalty 1.50 points.
Random won 0 of 256.
The separate warm discard puzzles at seed 860000 yielded 208 of 256 wins, or 81.25%, against Random's 64 of 256.
Those one-discard puzzles are not full cold games.

The table reports valid declaration wins.

| Players | Random, Python | Heuristic, Python | Casual, browser | Tactical, browser |
| --- | --- | --- | --- | --- |
| 2 | 119/120, 99.2% | 54/120, 45.0% | 100/120, 83.3% | 41/96, 42.7% |
| 3 | 119/120, 99.2% | 38/120, 31.7% | 86/120, 71.7% | 19/60, 31.7% |
| 4 | 117/120, 97.5% | 20/120, 16.7% | 69/120, 57.5% | 15/60, 25.0% |

Python arena uses seed 870000 and cycles Neural equally through the seats across 120 distinct deals per cell.
The browser loads the shipped JSON and uses seed 880000 with every seat on every deal.
Casual uses 60, 40, and 30 deal seeds for two, three, and four players.
Tactical uses 48, 20, and 15 deal seeds.
All 576 browser games finished without errors or tier downgrades.

First-place credit assigns the declaration winner first, or ranks terminal penalties at the turn cap.
It splits one unit across tied first seats.
Against 2p Casual, Neural won 100/120 by declaration, or 83.3%.
It finished first in 101/120, or 84.2%, including one turn-cap penalty win.

| Players | Neural credit against Tactical | Equal-seat parity | 95% deal-cluster interval |
| --- | --- | --- | --- |
| 2 | 42.7% | 50.0% | 35.4% to 49.0% |
| 3 | 31.7% | 33.3% | 25.0% to 38.3% |
| 4 | 25.0% | 25.0% | 18.3% to 31.7% |

The intervals use 2,000 resamples of deal seeds and preserve all seat rotations.
Neural beats Random and Casual in these cells and has a high cold-solo win rate.
Its three-player and four-player first-place credit is near Tactical parity, but its two-player credit is below parity.
The four-player Python Heuristic result is also below parity.
These results do not establish a rating or equal strength on every scoring measure.

The idle Node v24.19.0 bundle benchmark measured 4.44 ms median and 4.84 ms p95 for `qValues`.
It used five passes over 1,000 real observations, with per-pass medians from 4.42 to 4.45 ms and no failures.
The timed call includes validation, encoding, and forward computation, but excludes model loading and `observationFromView`.
It is not a complete draw-and-discard turn.
During the 18-worker Tactical matches, mean observation and move-selection time was 30.59 to 43.04 ms for Neural and 4.76 to 4.98 seconds for Tactical.
Tactical runs exact `minimumPenalty` and `outs1ply` searches.
Those loaded worker timings are not directly comparable to the idle `qValues` benchmark.
The browser evaluation took 52.3 minutes, longer than the initial estimate.

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

The only architecture is `card_mlp` v5.
The [training guide](TRAINING.md) covers setup, self-play, resume, evaluation, and browser export.

| Package | Responsibility |
| --- | --- |
| `src.game` | Environments, public observations, terminal outcomes, and stock handling. |
| `src.model` | Pure observation encoding, the Q-network, and model checkpoint loading. |
| `src.training.train` | Tensor replay, DMC updates, self-play actors, snapshots, and fixed validation. |
| `src.evaluation` | Solo evaluation and shared-deck arena matches. |
| `src.checkpoints` | Atomic files and tensor validation. |

### Observation encoding

`encode` returns 1,275 little-endian float32 values, or 5,100 bytes.
Encoding version 5 uses the following layout.

| Field | Shape | Float offset |
| --- | --- | --- |
| `card` | 52 by 22 | 0 |
| `seat` | 6 by 9 | 1144 |
| `context` | 22 | 1198 |
| `legal` | 54 | 1220 |
| `top` | 1 | 1274 |

The card channels contain own copies, pile copies, top and indicator flags, wild flags, points, and visible opponent activity.
Five further channels describe `adjacent`, `set_mates`, `natural`, `pure`, and `any_meld`.
Four disjoint-packing channels add `spare_frac`, `disc_pseq_delta`, `disc_pts_saved`, and `disc_pairs_delta`.
The context adds `disjoint_pseq_frac`, `disjoint_gated_pts`, `top_gain_pseq`, and `top_gain_pts` alongside `unavoidable_dw`.
The deterministic pack spends each card and joker at most once.
It preserves the 2 after Q-K-A and splits six-card runs when possible.
Packing is a greedy feature estimate, not the exact declaration or penalty search.
Legal discard and top-card exchange candidates use the same pack.
Seats are relative to the observer. Unused seat slots are zero.
No feature contains hidden opponent hands or stock order.

The legal mask excludes returning a just-taken discard when another face is held.
It retains the action for a single-face hand.
Terminal observations and afterstates awaiting another seat have no legal actions.
Training, solo evaluation, arena policies, and browser inference use this encoding.

### Neural network

`PolicyNet` embeds the 22 card channels and 76 table inputs.
Residual `CardBlock` modules mix neighboring ranks, matching ranks, suits, and table context.
The rank strip is `[A, 2, ..., K, A]`.
Shared normalization and mean and max pooling update the table representation.
The draw and face heads produce 54 advantages.
A value head and legal-action centering produce Q-values.
There are no auxiliary heads, target network, or meld computations inside the network.

`ModelSpec(64, 2)` has 127,172 parameters.
`ModelSpec(128, 3)` has 673,412 parameters.
Python and Node tests compare nonzero exported weights at both sizes within `1e-5` for all 54 actions.
Browser forward computation uses flat Float32 buffers.

### DMC learning

The trainer cycles cold deals across one, two, three, and four players by default.
On 50% of multiplayer deals, every seat uses the current epsilon-greedy network.
The remaining deals place one learner seat against a feature-only `HeuristicPolicy`, uniform random, and frozen snapshot opponents.
The heuristic receives at least half the opponent probability.
Every seat contributes its own outcome and trajectory, including opponent wins and losses.
Solo deals always use the learner.
The active snapshot pool holds at most eight checkpoints.

Replay stores contiguous float32 features, int64 actions, and float32 discounted returns.
Replay retains the most recent 40,000 steps.
For terminal utility `u`, step `t` of a length-`T` trajectory targets `u * 0.98 ** (T - 1 - t)`.
AdamW fits selected-action Q-values with Smooth-L1 loss and gradient clipping.
Winning trajectories also rank their taken action above other legal actions with a DQfD margin of 0.10 and default weight 0.50.
`--margin-weight 0` disables the margin term.
There is no warm-start curriculum, target network, or registry.

Solo and multiplayer use the same declaration-first utility.
A valid declaration earns +1.
Losing to a declaration earns `-0.5 - 0.5 * min(p, 100) / 100`, where `p` is the seat's penalty.
Without a declaration, utility is `-0.5 + 0.4 * seq_frac - 0.3 * min(p, 100) / 100`.
`seq_frac` caps qualifying sequences at the required quota and equals 1 when no sequences are required.
Exhaustion always earns negative utility, even for zero penalty.
Validation compares against the fixed heuristic and reigning best checkpoint.

Model checkpoints use version 5 and encoding version 5.
Trainer state uses version 8.
Older encodings and trainer states require a fresh run.
Bootstrap weights and short smoke runs do not establish competitive strength.

## Verification

Run the complete checks from the repository root.

```powershell
.\.venv\Scripts\python tools\check_training_architecture.py
.\.venv\Scripts\python -m unittest discover -s tests -v
node --test tests/*.test.mjs
```

The architecture checker rejects import cycles and function-local imports.
It also checks optional-dependency boundaries with Torch and NumPy blocked.
The tests cover exact evaluator witnesses, game conservation, masks, DMC replay, spawned actors, deterministic resume, and browser parity.
CI also runs a bounded train, resume, export, solo evaluation, and arena lifecycle.

Regenerate the portable evaluator fixtures with this command.

```powershell
python tools\export_parity_fixtures.py
```

Export a trained checkpoint to update the browser model and its matching Q-value fixture together.

```powershell
.\.venv\Scripts\python tools\weights_export.py --checkpoint runs\full21\best-model.pt
```

`--bootstrap --seed 1729` replaces the trained bundle with untrained compatibility weights.
Do not use bootstrap results as evidence of playing strength.

`tools/verify_browser.mjs` exercises the real browser through Playwright.
It checks worker decisions, the Neural label and greedy reason, declarations, privacy, and the main-thread fallback.
`node tools/benchmark_qnet.mjs` times five passes over 1,000 real draw and discard observations.
It reports median and p95 latency for validation, encoding, and one forward pass.
