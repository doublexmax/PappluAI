# Multiplayer model ratings

The rating system compares immutable checkpoints in actual shared-deck games.
It does not interpret solo hand-completion rates as competitive win rates.
Each player count and game configuration has its own evidence stream.

Install `requirements-training.txt` using Python 3.10 or newer. OpenSkill
6.2.0 is pinned so the same evidence can reproduce the same rating calculation.

## Points determine the losers' placements

The declaring winner places first. Losers with fewer unmatched-card points
place better, and equal totals receive tied placements.

Numbered cards score their face value. A, J, Q, and K each score 10. Every card
of the selected joker's rank scores zero.

Qualifying sequences always protect their cards from points. Sets and other
melds protect their ordinary cards only when the required qualifying sequences
exist in the same non-overlapping grouping. Two ordinary sevens plus a wildcard
therefore score 14 before the quota is met and zero afterward.

`src.evaluate.minimum_penalty` finds the global minimum and returns the grouping
that establishes it. The binary declaration evaluator remains separate.
Terminal hand snapshots are available to the referee and scorer, not to the
policy observations during a game.

For a winner followed by losers scoring 14, 14, and 38, the ranks are
`[1, 2, 2, 4]`. OpenSkill receives the whole multiplayer result in one update.
Raw point gaps do not multiply the rating change.

## Draw scores and evidence weights are different

A standard chess draw scores 0.5, but it is still an ordinary rated game.
The [FIDE calculation](https://handbook.fide.com/chapter/B022024) uses
`score - expected_score` for each game, with scores 1, 0.5, or 0.
That does not make a draw a half-confidence observation.

Engine testing can adjudicate a result using established criteria rather than
playing every remaining move. For example, the
[Leela Chess Zero testing guide](https://lczero.org/dev/wiki/testing-guide/)
uses stable evaluation conditions for draw adjudication and reverses colors
on the same opening to control bias.

An arbitrary evaluation cutoff is not automatically the same as a completed
chess draw. Capped Papplu hands provide point-based positions but no declaration.
The implementation therefore keeps those facts and their interpretation
separate.

| Cap policy | Treatment |
|---|---|
| `undecided` | Keep capped facts, but update only from declared games. The report is explicitly declaration-conditional. |
| `exclude` | The same conditional calculation, explicitly selected by the operator. |
| `points` | Rank all capped hands by points with full weight. No declaration win is invented. |
| `tie` | Treat the capped game as an all-player tie. |
| `weighted-points` | Use capped point placements as discounted, experimental evidence. |

The initial `weighted-points` view uses a configurable weight of 0.25.
This is a provisional hypothesis, not a calibrated constant or a chess rule.
Its report always says `cap_weight_validated: false`.

The discount interpolates the approximate Gaussian update in information
parameters, affecting both the skill estimate and its internal uncertainty.
It does not misuse OpenSkill's player-contribution `weights` parameter.
A weight of zero leaves the prior unchanged; a weight of one reproduces the
full update. Declared games retain full weight.

Keep the completed-game-only view beside the discounted view. Validate any
chosen discount against fresh declared-game predictions before interpreting it
as established competitive evidence.

## Calibrate a cap before using it

The calibration command and its evidence rules are described in
[TRAINING.md](TRAINING.md#calibrate-multiplayer-draw-limits).

A frozen-pool study selected 60 turns per player for two- and three-player
games and 30 for four-player games. Those choices were checked against a
180-turn guard on independent confirmation deals. They are specific to those
bot snapshots and house rules, not universal human-game limits.

Caps below are explicit arguments. Changing a cap creates a separate protocol
instead of merging incompatible evidence.

## Run continuing tournaments

Snapshot sources contain immutable `model-<sha256>.pt` files, such as the model
files in an improvement run's registry. Mutable files such as `latest-model.pt`
are not admitted as competitors.

Use a checkpoint SHA-256, or a unique prefix, for the reference. Its displayed
rating is anchored at 1500 by convention, not by comparison with chess players.

```powershell
$reference = "replace-with-checkpoint-sha-or-unique-prefix"
.\.venv\Scripts\python -m src.cli.tournament --root runs\ratings --snapshots runs\improve --cap 2:60 --cap 3:60 --cap 4:30 --reference $reference --continuous --workers 4 --cap-policy weighted-points --cap-weight 0.25
```

To run a bounded campaign instead, replace `--continuous` with `--blocks 128`.
That is a cumulative target per protocol, so repeating the same command
resumes rather than adding another 128 blocks. `--max-seconds` sets an
operational runtime budget.

The coordinator discovers new frozen hashes between batches. It retains its
own verified copies, so deleting a source does not change an already admitted
competitor. A seeded random policy provides a fixed baseline.

Lineups favor less-tested models and less-covered opponent pairs, not current
rating estimates. Every deal uses all seat permutations. The raw match collection
therefore balances seat position and neighboring opponents.

Dropping or discounting capped games can break that balance within a mixed
deal block. Reports expose `mixed_cap_blocks` and `effective_seatings_balanced`.
The completed-only control is conditional evidence, not a claim of perfect
effective seat balance.

One coordinator owns the SQLite database. Workers play or score jobs and never
write ratings. A hard watchdog terminates owned workers that exceed their
task budget. Timeouts and worker exits are failures, not draws.

Raw terminal games are persisted before scoring. A scorer failure can therefore
be retried without replaying the game. Identical repeated results are
idempotent; conflicting results fail explicitly.

Rating calculations follow the persisted block and seating order, not worker
completion order. Only complete, scored blocks in the settled prefix are used.
Failed blocks remain visible. The live writer retains a disposable projection
and applies only newly settled blocks; a read-only report can replay from the
canonical database. Standings publication is rate-limited.

## Read the results

The coordinator publishes:

- `status.json` with activity and completed, pending, and failed job counts.
- `standings.json` with the selected rating view.
- `standings-completed-only.json` as the declared-game control.
- `ratings.sqlite3` with immutable schedules, terminal hands, scores, and attempt history.
- `executions` with source fingerprints, runtime versions, and run settings.

Generate a separate uncertainty report while the coordinator runs:

```powershell
$reference = (Get-Content runs\ratings\status.json -Raw | ConvertFrom-Json).reference
.\.venv\Scripts\python -m src.cli.ratings --database runs\ratings\ratings.sqlite3 --output runs\ratings\report.json --reference $reference --cap-policy weighted-points --cap-weight 0.25 --samples 200
```

Use a separate JSON path rather than a coordinator-owned state filename.
The reporter is read-only with respect to the database.

Reports include declaration wins, valid games, rated games, weighted game
counts, penalties, opponent coverage, independent deal counts, and the
reference-relative estimate. OpenSkill's `sigma` is an internal model spread,
not a deal-cluster confidence interval.

Bootstrap reports resample whole deal blocks, including every seating, and
recenter each replicate on its reference. They preserve the evidence revision
and sampling settings. Small samples, disconnected comparison groups,
one-sided results, and experimental cap discounts remain provisional.

This service does not modify model weights or promote a champion. Training
and protected promotion remain separate from the rating ladder.
