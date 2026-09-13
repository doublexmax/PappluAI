# PappluAI

Papplu hand correctness and a binary reward function for future model training.
The evaluator uses this project's house rules, not a universal rummy ruleset.
The environment, training code, and notebook remain unfinished experiments.

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
```

The tests cover the house rules and compare the evaluator with an independent
brute-force partition oracle on small hands. Valid results are also checked
for card conservation and legal joker assignments.