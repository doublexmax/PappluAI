# PappluAI

Papplu hand correctness and a binary reward function for future model training.
The evaluator uses this project's house rules, not a universal rummy ruleset.
The local simulator provides a hand builder and a pass-and-play card table.
The environment, training code, and notebook remain unfinished experiments.

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

## Verification

From the repository root, the standard-library test command is:

```powershell
python -m unittest discover -s tests -v
node --test tests\simulator.test.mjs
```

The tests cover the house rules and compare the evaluator with an independent
brute-force partition oracle on small hands. Valid results are also checked
for card conservation and legal joker assignments.
API tests cover request validation and bounded evaluation. Game-state tests
cover card conservation, physical copies, card order, draw/discard transitions,
declaration outcomes, and elimination penalties.
Node.js is needed only for the JavaScript tests, not to run the simulator.