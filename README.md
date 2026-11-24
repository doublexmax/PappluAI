# PappluAI

Papplu hand evaluation, shared-deck play, and a GameTransformer trained by self-play.
The evaluator implements this project's house rules, not a universal rummy ruleset.
The browser simulator has no third-party dependencies and runs without a build or a remote API.
Training needs PyTorch.

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

Pass & Play covers hands between turns.
With one human, that hand stays visible during AI turns.
AI Watch reveals hands to spectators, but each AI receives only its own hand and public events.
Opponent stock faces and face-down declarations stay hidden.
The vs AI preset pairs you with a Tactical AI. AI Watch has two Tactical seats and one Casual seat.

The speed settings are Instant, Normal, and Slow.
Pause and Resume control AI play. AI Watch pauses after 200 completed turns.
The optional Check hand panel does not declare a win or end a round.

When stock empties, the default rule shuffles all but the top discard into stock.
Turn off recycling to leave stock empty. Deal again if both piles are empty.
The selected joker indicator stays outside the playable piles.

### AI tiers

- **Tactical** minimizes exact deadwood penalty and uses public-card beliefs, feed risk, and one-ply outs.
- **Casual** keeps wild cards and favors nearby ranks and matching cards.

Both tiers declare an exact win when a discard leaves a valid hand.
The scoreboard shows the seat's tier and the reason for its last decision.

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

`GameTransformer` in [`src/model/`](src/model/) is the repository's only neural model. It uses checkpoint format 6 and token schema 8. The default model has 2,175,224 parameters. Self-play uses 1-ply search to find wins on the current turn. The learner computes targets from shaped rewards across the full trajectory and trains on sampled decisions. The hand potential stays out of the model tokens. Set `--auto-curriculum` to advance training starts when the 0-ply policy meets the current win-rate threshold. See the [training guide](TRAINING.md) for run commands, defaults, and the token layout.

The run selected and locked Final at revision 1,184 using seed `600_000`. On held-out seed `930_000`, the locked 0-ply model declared on the first discard in 44 of 128 distance-zero solo deals. The untrained 0-ply model declared on the first discard in 30 of the same 128 deals. On 256 cold solo deals, the locked model declared zero times at 0-ply and once at 1-ply. See the [Azure results](TRAINING.md#azure-results) for the lock and full solo and multiplayer tables.

The Azure run selected `phase4-c3` on seed `600_000` before evaluating seed `940_000`. At 0-ply the selected checkpoint declared on 37 of 128 first discards, against 40 for the previous winner on the same deals. The sampled learner processed 13.03x as many updates per second on the fixed learner benchmark. See the [Azure mastery run and paired holdout](TRAINING.md#azure-mastery-run-and-paired-holdout) for the full results.

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
It checks a full Tactical turn in the module worker, scoring, declarations, pass-and-play privacy, AI Watch, and the main-thread fallback.
