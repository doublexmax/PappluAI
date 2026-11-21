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

`web/papplu-model.json` contains untrained `card_mlp` v6 bootstrap weights with seed 1729.
Its `source.kind` is `bootstrap` and `source.trained` is `false`.
The default model has width 80, three five-tap SwiGLU blocks, and 342,804 parameters.
It reads 854 observable float32 inputs with encoding version 6 and uses checkpoint version 6.
These weights verify compatibility, not playing strength.
Export a trained v6 checkpoint with the [training guide](TRAINING.md) to replace them.

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

The only architecture is `card_mlp` v6.
The [training guide](TRAINING.md) covers setup, self-play, resume, evaluation, and browser export.

| Package | Responsibility |
| --- | --- |
| `src.game` | Environments, public observations, terminal outcomes, and stock handling. |
| `src.model` | Pure observation encoding, the Q-network, and model checkpoint loading. |
| `src.training.train` | Tensor replay, DMC updates, self-play actors, snapshots, and fixed validation. |
| `src.evaluation` | Solo evaluation and shared-deck arena matches. |
| `src.checkpoints` | Atomic files and tensor validation. |

### Observation encoding

`encode` returns 854 little-endian float32 values, or 3,416 bytes.
Encoding version 6 uses the following layout.

| Field | Shape | Float offset |
| --- | --- | --- |
| `card` | 52 by 14 | 0 |
| `seat` | 6 by 9 | 728 |
| `context` | 17 | 782 |
| `legal` | 54 | 799 |
| `top` | 1 | 853 |

The card channels are `own`, `own_eq1`, `own_eq2`, `own_ge3`, `pile`, `top`, `joker`, `wild`,
`points`, `unseen`, `opp_discarded`, `next_held`, `other_held`, and `next_passed`.
The copy flags distinguish exactly one, exactly two, and at least three copies.
`unseen` subtracts own copies, pile copies, the indicator, and all publicly held copies from the deck count, with a floor of zero.
Seat channels record presence, observer identity, activity, hand size, remaining turns, the next player, takes, passes, and held totals.
Context channels record phases, turn ownership, supply, rules, table size, exact jokers held, wild cards held, and raw points.
The encoder performs no meld search, greedy packing, or action lookahead.
Seats are relative to the observer. Unused seat slots are zero.
No feature contains hidden opponent hands or stock order.

The legal mask excludes returning a just-taken discard when another face is held.
It retains the action for a single-face hand.
Terminal observations and afterstates awaiting another seat have no legal actions.
Training, solo evaluation, arena policies, and browser inference use this encoding.

### Neural network

`PolicyNet` embeds the 14 card channels and 71 table inputs.
Residual `CardBlock` modules mix five rank taps, matching ranks, suits, and table context.
The rank strip is `[A_low, 2, ..., K, A_high]`, with zero padding at both ends.
The network averages the two Ace representations and applies a SwiGLU card feed-forward layer.
Shared normalization and mean and max pooling update the table representation.
The draw and face heads produce 54 advantages.
A value head and legal-action centering produce Q-values.
There are no auxiliary heads, target network, or meld computations inside the network.

The default `ModelSpec(80, 3)` has 342,804 parameters.
`ModelSpec(128, 3)` has 867,972 parameters.
Python and Node tests compare nonzero exported weights at both sizes within `1e-5` for all 54 actions.
Browser forward computation uses flat Float32 buffers.

### DMC learning

The trainer cycles cold deals across one, two, three, and four players by default.
On 75% of multiplayer deals, every seat uses the current epsilon-greedy network and contributes its own trajectory.
Other deals record only one randomly selected learner seat.
Each opponent is uniform random with probability 15%, or a uniformly selected greedy snapshot with probability 85%.
Solo deals always use the learner.
The active snapshot pool holds at most eight checkpoints.
Each learner seat draws one exploration rate per deal from `(0.01, 0.01, 0.03, 0.08)` using its own RNG.
Exploration does not depend on seat index or training progress.
All active learner rows, including solo deals, share one neural forward batch.

Replay stores contiguous float32 features, int64 actions, and float32 discounted returns.
Replay retains the most recent 40,000 steps.
For terminal utility `u`, step `t` of a length-`T` trajectory targets `u * 0.99 ** (T - 1 - t)`.
AdamW fits selected-action Q-values with Smooth-L1 loss and gradient clipping.
The default round holds 32 deals and schedules two updates per completed deal, with up to 256 replay samples per update.
The learning rate starts at `5e-4` and decays with `lr / sqrt(1 + updates / 8000)`, with a floor of `7.5e-5`.
There is no margin loss, scripted opponent, epsilon schedule, warm-start curriculum, target network, or registry.

Solo and multiplayer use the same declaration-first utility.
A valid declaration earns +1.
Losing to a declaration earns `-0.5 - 0.5 * min(p, 100) / 100`, where `p` is the seat's penalty.
Without a declaration, utility is `-0.5 + 0.4 * seq_frac - 0.3 * min(p, 100) / 100`.
`seq_frac` caps qualifying sequences at the required quota and equals 1 when no sequences are required.
Exhaustion always earns negative utility, even for zero penalty.
Validation compares against random play, the frozen initial model, and the reigning best checkpoint.
The fixed-anchor score averages solo, random-opponent, and initial-opponent results.

Model checkpoints use version 6 and encoding version 6.
Trainer state uses version 9.
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
.\.venv\Scripts\python tools\export_parity_fixtures.py
```

Export a trained checkpoint to update the browser model and its matching Q-value fixture together.

```powershell
.\.venv\Scripts\python tools\weights_export.py --checkpoint runs\full21\best-model.pt
```

`--bootstrap --seed 1729` generates the untrained compatibility bundle.
Do not use bootstrap results as evidence of playing strength.

`tools/verify_browser.mjs` exercises the real browser through Playwright.
It checks worker decisions, the Neural label and greedy reason, declarations, privacy, and the main-thread fallback.
`node tools/benchmark_qnet.mjs` times five passes over 1,000 real draw and discard observations.
It reports median and p95 latency for validation, encoding, and one forward pass.
