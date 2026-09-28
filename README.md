# PappluAI

Papplu hand correctness and neural-network training from simulated games.
The evaluator uses this project's house rules, not a universal rummy ruleset.
The local simulator provides a hand builder and a pass-and-play card table.
The Python trainer uses the evaluator's binary reward in solo Monte Carlo
episodes. The browser simulator remains separate from model training.

The Social club layout uses a warm background, player seats, a green hand mat,
and a separate draw tray. Cards use clear sans-serif indices and a softer
palette while keeping recognizable playing-card artwork.

Card faces use original SVG artwork with red hearts and diamonds, black spades
and clubs, mirrored corner indices, number-card pips, and colored court cards.
Face-down cards use a generic patterned back. No external images or fonts are
downloaded.

## Run the simulator

From the repository root, start the local server with Python 3.9 or later.
No package installation or JavaScript build is needed.

```powershell
python -m src.simulator
```

Open `http://127.0.0.1:8765` in your browser. To use another port, run
`python -m src.simulator --port 8766`. Stop the server with Ctrl+C.
The server accepts only local requests and is not a production web server.

### Hand builder

Expand the settings panel to change the hand rules or selected joker.
Samples and the card picker remain available alongside the hand.
Click a card under **Add cards** to add a copy. Select a card in your hand and
choose **Remove** to remove only that copy.
Load a sample to inspect a complete hand without entering all 21 cards.

The result updates after each edit. A winning hand shows its complete grouping
and what each substituted joker represents. A losing hand has reward zero, but
the display does not claim that it contains zero useful sequences.
Incomplete hands show the number of cards still needed.

### Game table

Choose one through six players for solo or local pass-and-play. The default
uses three decks, 21 cards per player, and five required pure sequences.
Deal a new game to shuffle and deal each player's hand.
The game then removes one card as the joker indicator and places another on
the discard pile. The indicator cannot be drawn during the round.
Settings collapse after a successful deal to leave room for play. Expand them
to configure the next round; the current round keeps its original rules.
On wide screens, the hand and draw tray sit side by side. On narrow screens,
draw controls appear above the hand and turn-completion controls remain below it.

Draw from the stock or the top of the discard pile. Choose a card to discard
from your enlarged hand, or discard the drawn card immediately.
Both choices finish the turn with the original hand size.
Returning a card drawn from the discard pile is allowed in this simulator.
The game does not decide whether that choice improved your hand.

On multiplayer turns, pass the screen and reveal the next player's hand.
Other players' cards are not shown. Solo play continues without the reveal step.
**Check hand** is off by default for every player. Enable it to see live
validity and grouping for your own hand. The next player does not
inherit your setting, and a new round starts with checking off again.
Having a valid hand does not automatically end the round.
The hand-check panel appears only when requested. Declaration results appear
when a declaration is made, rather than occupying an empty panel during play.
An empty stock is not reshuffled automatically. Draw from the discard pile or
deal a new game.

### Declare a win

After drawing, select the card you want to put down and choose **Declare win**.
That card goes face down into a separate declaration area, not the playable
discard pile. The Python evaluator checks the remaining hand against the
round's rules, even when **Check hand** is off.

| Result | Outcome |
| --- | --- |
| Valid hand | The declaring player wins and the round ends. |
| Invalid hand | The player loses, is eliminated from the round, and receives 80 penalty points. |
| Evaluation error or timeout | No win, loss, or penalty is assigned. Retry the same pending declaration. |

Other active players continue after an invalid declaration. Eliminated players
are skipped, and their cards remain out of play. When only one player remains,
that player wins automatically. An invalid solo declaration ends in a loss.
Cards and turns cannot be changed while a declaration is awaiting a verdict.

Penalty totals accumulate across deals with the same player count.
Changing the player count resets those totals. These are false-declaration
penalties only, not a full deadwood-scoring system. Lower totals are better.
The evaluator's binary reward remains separate from these game penalties.

The 80-point penalty and continued multiplayer play follow
[RummyCircle's points-rummy declaration rules](https://www.rummycircle.com/rummy-variations/points-rummy.html).
Those rules describe a 13-card game. This simulator adopts the penalty as a
house rule for its 21-card game, not as a universal Papplu rule.
[Pagat notes that 21-card variants have differing penalties](https://www.pagat.com/rummy/indian.html).

### Arrange your cards

Hands start sorted by suit and rank. Drag cards into your preferred order, or
select a card and use the move controls. **Sort** restores the default
order. Individual copies move separately, including identical-looking cards.
Your order persists across draws, discards, checks, and turns.
Drawing appends the new card instead of rearranging your hand.

Builder and table hands are separate. Table settings apply when a new game is
dealt, not halfway through a round. Reloading the page resets both modes.
There are no automated opponents, network players, or saved games.

### Simulator limits

The UI supports one through six decks, hand sizes from 3 through 30, and
required sequence counts from 0 through 10.
The deal must leave enough cards for all players, the indicator, and the first
discard. Individual copies retain their identities during draw and discard.

The browser calls `src\evaluate.py` through the local server. There is no second
JavaScript version of the hand-validation rules.
Requests are coalesced so an older result cannot overwrite an edited hand.
An evaluation that exceeds ten seconds stops and reports an error instead of
returning a false losing-hand result. Preview errors leave the hand editable.
A pending declaration remains committed until a verdict or a new round.

## House rules

The default declaration contains **21 cards and at least 5 pure sequences**.
Both values are configurable. Every card must belong to exactly one legal group.

| Group | Rule |
| --- | --- |
| Sequence | At least three consecutive ranks of one suit. Ace can be low in A-2-3 or high in Q-K-A, but K-A-2 is invalid. |
| Required pure sequence | A sequence with no substitutions except by the exact selected joker card. This exception is specific to these house rules. |
| Set | Three or four cards of one rank with distinct represented suits. Sets do not count toward the pure-sequence requirement. |

A long sequence counts as one group. It can count as multiple sequences only if
it can be split into disjoint sequences of at least three cards each.
Additional sequences and sets may use wildcards once the required sequences
are accounted for. The input order does not matter.

### Jokers

The card drawn at the start identifies the selected joker, such as `8s`.
Every eight is then wild, with different permissions for required sequences.

| Cards when `8s` is selected | Counts as a required pure sequence? |
| --- | --- |
| `4h 5h 8s`, with `8s` representing `6h` | Yes. The exact selected card can substitute anywhere. |
| `7d 8d 9d` | Yes. The off-suit eight acts as its natural card. |
| `4h 5h 8d`, with `8d` representing `6h` | No. It is an additional impure sequence only. |

Jokers may fill every position in a group. An all-joker sequence counts toward
the required minimum only if it satisfies the same substitution rule.
For example, three copies of `8s` qualify when `8s` is selected.

Jokers do not remove duplicate natural cards. `7s 7s 8d` is not a legal set
when eights are wild, but `7s 7h 8d` is legal because `8d` can represent `7c`.
Multiple copies of a face may belong to separate groups.

## Python API

`src\evaluate.py` requires Python 3.9 or later and has no third-party dependencies.

```python
from src.evaluate import evaluate_hand, hand_reward, is_valid_hand

ranks = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
suits = ("s", "h", "d", "c")

def card_index(card):
	return suits.index(card[-1]) * 13 + ranks.index(card[:-1])

cards = (
	"3s 4s 5s "
	"6h 7h 8h "
	"9d 10d Jd "
	"Ac 2c 3c "
	"Js Qs Ks "
	"2s 2h 2d "
	"5h 5d 5c"
).split()
hand = [0] * 52
for card in cards:
	hand[card_index(card)] += 1

joker = card_index("8s")
result = evaluate_hand(hand, joker)
assert result.is_valid
assert is_valid_hand(hand, joker)
assert hand_reward(hand, joker) == 1.0

for meld in result.melds:
	print(meld.kind, meld.cards, meld.represented_cards, meld.is_pure)
```

All three functions accept the same arguments.

| Argument | Meaning |
| --- | --- |
| `hand` | A sequence of 52 nonnegative integer counts, not a list of card IDs or a one-hot encoding. |
| `joker` | The exact selected card's index from 0 through 51. It need not be present in the hand. |
| `required_sequences=5` | Minimum number of disjoint qualifying sequences. May be zero. |
| `cards_in_hand=21` | Exact number of cards in a winning declaration. Must be positive. |

Card indices are `suit_index * 13 + rank_index`. Suits are spades, hearts,
diamonds, and clubs. Ranks are Ace through King, with Ace at index zero.
Counts support multiple decks. The evaluator does not validate the deck supply
or limit the number of copies of a face. Printed jokers are not represented.

`evaluate_hand` returns an immutable `HandEvaluation`. For a valid hand,
`melds` contains a complete grouping. Each `Meld` records its `kind`, consumed
`cards`, aligned `represented_cards`, and whether it `is_pure`.
For an invalid declaration, `is_valid` is false and `melds` is empty.

`is_valid_hand` returns a boolean. `hand_reward` returns **1.0 for a valid
declaration and 0.0 otherwise**, with no partial credit or deadwood score.
Use the reward on the final hand after discarding, not on the extra-card draw
state. A hand whose total differs from `cards_in_hand` is invalid.

Malformed inputs, including negative counts, nonintegral counts, invalid joker
indices, and invalid configuration values, raise `TypeError` or `ValueError`.
Booleans are not accepted as integers. The evaluator does not mutate its inputs.

The old `matched_rate` prototype and its matching helpers have been removed.
Use `hand_reward` for training rewards and `evaluate_hand` for a grouping.
The new reward is not the old prototype's unmatched-card count.

The evaluator searches alternative groupings instead of greedily removing the
first sequence it finds. It caches remaining card counts and the unmet sequence
quota within each call. The search is exact, but its worst-case cost grows
combinatorially. Joker-heavy hands can be slower than ordinary deals.
This version provides correctness, not an optimized batch-training engine.

## Train the model

Training uses PyTorch. The evaluator and browser simulator still run without
third-party packages. From the repository root, create a training environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-training.txt
```

Start with shorter, three-card games to exercise the training loop.

```powershell
.\.venv\Scripts\python -m src.train --episodes 200 --cards-in-hand 3 --required-sequences 1 --max-turns 10 --warm-start-fraction 0.5 --seed 7 --eval-episodes 20 --eval-seed 10007 --checkpoint checkpoints\small-hand.pt
```

For the default house rules, omit the hand-size and sequence arguments.

```powershell
.\.venv\Scripts\python -m src.train --episodes 1000 --max-turns 30 --warm-start-fraction 0.5 --seed 7 --eval-episodes 50 --eval-seed 10007 --checkpoint checkpoints\papplu.pt
```

Use `python -m src.train --help` for the available training parameters.
Choose different training and evaluation seeds. A successful short run proves
that the training pipeline runs, not that the model plays a strong 21-card game.

### Monte Carlo learning

The trainer keeps the notebook's small neural network with 128-unit and 64-unit
hidden layers. It also keeps epsilon-greedy exploration and Adam optimization.
The unfinished tree search is replaced by complete simulated episodes.
Each recorded action learns from the discounted rewards that actually follow
it in that episode. Replay minibatches reuse those observations.

The default `--replay-sampling transition` samples stored transitions uniformly.
Long episodes therefore contribute more training examples than short puzzles.
The optional `--replay-sampling episode` first samples a retained episode
uniformly, then a transition within it. It samples with replacement.
Both modes limit the buffer by the number of retained transitions; an older
episode can be trimmed at that boundary.

These are Monte Carlo return targets, not DQN's bootstrapped next-state
estimates. There is no MCTS tree, target network, or second implementation of
the hand evaluator.

The environment alternates between drawing and discarding. The model chooses
between stock and the top discard, then chooses which card face to discard.
Illegal actions are masked during both exploration and greedy play.
The observation includes the hand's card counts, the top discard, the exact
joker face, and the current phase and remaining resources.
The model cannot see the hidden stock order.

After each discard, `hand_reward` checks the original-size hand.
A valid hand earns 1 and ends the episode. All other rewards are 0.
The turn limit also ends an episode, so repeatedly taking and returning a
discard cannot create an infinite game. The indicator stays out of play,
and the stock is not reshuffled.

This is solo hand-completion training. It automatically recognizes a valid
post-discard hand, unlike the browser's explicit declaration action.
It does not train bluffing, declaration penalties, multiplayer opponents,
or a policy for the browser UI.

### Sparse rewards and warm starts

A random 21-card hand almost never satisfies all the rules.
Learning only from random deals can produce long runs with no positive reward.
The evaluator's reward remains binary. The trainer does not replace it with a
heuristic score for incomplete groups.

`--warm-start-fraction` mixes in single-decision discard episodes.
These starts contain an evaluator-confirmed winning hand plus an extra card.
The policy must choose what to discard using the ordinary observation.
These episodes end after that discard, whether it wins or loses.
The remaining episodes start from shuffled random deals.
The default fraction is zero. The examples above use 0.5 to supply early
positive examples.

Warm-start wins are training results, not evidence of full-game skill.
Held-out evaluation uses only random deals and compares greedy model play
with a policy that samples legal actions uniformly.
Evaluation does not update the model.

The exact evaluator can be slow on difficult joker-heavy hands.
Turn limits bound the number of decisions, not the time taken by an individual
evaluation. Use small runs before committing to a larger training budget.
Evaluator errors propagate rather than becoming zero-reward examples.

### Model files

`src\environment.py` owns the solo game and legal actions.
`src\model.py` owns the neural network, masked action selection, and checkpoints.
`src\train.py` owns Monte Carlo updates, training, and held-out evaluation.
`src\Model.ipynb` imports these modules instead of maintaining separate rules
or model implementations.

Checkpoints store model weights and their configuration.
Loading a checkpoint restores its game settings and starts a fresh replay
buffer and optimizer. Explicit game settings must match the saved settings.
It is fine-tuning, not an exact continuation of the previous random stream.
Models trained with different hand rules are separate experiments.
Training outputs under `checkpoints` are not tracked by Git.

```powershell
.\.venv\Scripts\python -m src.train --load checkpoints\small-hand.pt --episodes 100 --warm-start-fraction 0.5 --seed 8 --checkpoint checkpoints\small-hand-tuned.pt
```

The CPU trainer uses one PyTorch thread by default because the network is small.
Use `--torch-threads` to change it or `--device cuda` with a CUDA-capable PyTorch
installation.

## Verification

### Architecture comparison

`mlp` retains the original 128/64 network and remains the default.
`wide_mlp` tests a larger 256/128/64 network.
`suit_conv` shares rank-local convolutions and discard scoring across suits.
It tests whether card structure generalizes better than unrelated outputs
for each face. All three use the same observations, legal-action masks,
Monte Carlo returns, and binary evaluator reward.

Use `--architecture` when starting a new training run. A checkpoint records its
architecture, and `--load` restores it. Existing version-1 checkpoints load as
the original MLP.

The repeatable comparison command is:

```powershell
python -m src.compare --plan experiments\networks.json --output-dir checkpoints\network-comparison
```

The plan starts with three-card games, where random play sometimes wins.
It runs three training seeds for each structure with equal episode and update
budgets. It checks that an evaluator-backed discard oracle solves the warm-start
puzzles while random play does not. That oracle is only a benchmark control,
not a policy available to the trained agent.

Architecture selection uses validation deals. Only the selected structure and
the original MLP then see the separate final test deals. Reports contain per-game
outcomes, parameter counts, draw-source counts, and paired bootstrap intervals
over both training seeds and shared test deals.
With three training seeds, these intervals are exploratory rather than
strong statistical confirmation.
A candidate passes the promotion gate only if it gains at least five percentage
points over both controls and both confidence intervals exclude zero.
This gate does not change the default model automatically.
The screen is not evidence of strong 21-card play; that requires a separate
full-game comparison.

The 2026-09-28 screen used 6,000 episodes per model and training seeds 11, 22,
and 33. The same 300 validation deals selected `suit_conv`. Final evaluation
used 1,000 separate deals shared across the three trained seeds.

| Architecture | Parameters | Mean validation wins | Mean final-test wins |
| --- | ---: | ---: | ---: |
| Original MLP | 32,886 | 1.78% | 1.10% |
| Wider MLP | 86,902 | 2.56% | Not selected |
| Shared suit/rank CNN | 15,939 | 22.89% | 19.90% |
| Random legal policy | N/A | 2.33% | 2.40% |

The CNN improved the final-test win rate by 18.8 percentage points over the
original MLP. The exploratory paired 95% interval was 14.7 to 24.1 points.
Its gain over random play was 17.5 points, with an interval of 13.5 to 22.5.
All three CNN seeds beat both controls. These results are for three-card,
one-sequence games with a ten-turn limit, not default Papplu.

The paired 21-card followup used 2,000 warm-start episodes and 1,000 mixed
episodes per architecture, with seeds 17 and 18. Both the MLP and CNN won
**0 of 100** fresh full games, as did random play. On 200 separate discard
puzzles, the final MLP scored 26%, the CNN 20%, and random play 26%.
The reduced-game improvement did not transfer under this training recipe.
The original default is therefore unchanged.

A separate replay-sampling experiment continued from the same warm-trained CNN,
using the same 1,000-episode mixed-training budget and a new evaluation set.

| CNN checkpoint | Full games won / 100 | Winning discards / 200 puzzles |
| --- | ---: | ---: |
| Warm training only | 1 | 138 (69%) |
| After transition-uniform replay | 0 | 59 (29.5%) |
| After episode-uniform replay | 2 | 101 (50.5%) |
| Random legal policy | 0 | 56 (28%) |

Episode sampling retained more discard skill than transition sampling, but less
than the warm-only checkpoint. Its two full-game wins are exploratory evidence,
not a reliable advantage. This was one training seed. Sampling with replacement
also changes random-number consumption, so the treatments did not follow
identical training trajectories. The reward and architecture were unchanged.
Both architecture and replay defaults remain unchanged.

An experimental CNN run with episode sampling can be started with:

```powershell
python -m src.train --architecture suit_conv --replay-sampling episode --warm-start-fraction 0.5 --checkpoint checkpoints\cnn-episode.pt
```

To evaluate an individual checkpoint:

```powershell
python -m src.benchmark --checkpoint checkpoints\papplu.pt --games 100 --seed 900000 --warm-games 100 --warm-seed 600000 --output checkpoints\evaluation.json
```

### Tests

From the repository root, the standard-library test command is:

```powershell
python -m unittest discover -s tests -v
node --test tests\simulator.test.mjs tests\card_art.test.mjs
```

The tests cover the house rules and compare the evaluator with an independent
brute-force partition oracle on small hands. Valid results are also checked
for card conservation and legal joker assignments.
API tests cover request validation and bounded evaluation. Game-state tests
cover card conservation, physical copies, card order, draw/discard transitions,
declaration outcomes, and elimination penalties.
Card-art tests cover all 52 faces, pip counts, and generic face-down backs.
Node.js is needed only for the JavaScript tests, not to run the simulator.

Environment tests use only the standard library. Model and training tests are
skipped when PyTorch is absent. To include the learning, checkpoint, and CLI
checks, run the Python suite with the training environment.

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```