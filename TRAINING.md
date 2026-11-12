# Training walkthrough

All commands in this guide run from a fresh repository clone on Windows
PowerShell with Python 3.10 or newer. Training is CPU-compatible but requires
the optional training dependencies.

## Set up a fresh clone

```powershell
git clone https://github.com/doublexmax/PappluAI.git
Set-Location PappluAI
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-training.txt
```

Command-line entry points live in `src.cli`. They parse `--help` before loading
PyTorch. Game rules are in `src.game`, the current network is in `src.model`, training
implementations in `src.training`, evaluation in `src.evaluation`, and
persistent state operations in `src.checkpoints`.

## Run a short smoke test

This uses a three-card warm-start puzzle. It verifies training, model
checkpoint writing, loading, and evaluation without representing full Papplu
performance.

```powershell
.\.venv\Scripts\python -m src.cli.solo --episodes 20 --num-decks 2 --cards-in-hand 3 --required-sequences 1 --max-turns 5 --warm-start-fraction 1 --eval-episodes 10 --eval-seed 1000 --log-every 0 --checkpoint checkpoints\smoke.pt
.\.venv\Scripts\python -m src.cli.benchmark --checkpoint checkpoints\smoke.pt --output runs\smoke-evaluation.json --games 20 --warm-games 20
```

The first command writes `checkpoints\smoke.pt`. The second writes paired
evaluation details to `runs\smoke-evaluation.json`.

To fine-tune those weights, start with fresh optimizer and replay state.

```powershell
.\.venv\Scripts\python -m src.cli.solo --load checkpoints\smoke.pt --episodes 5 --checkpoint checkpoints\smoke-finetuned.pt
```

## Train and resume full 21-card play

The curriculum trainer defaults to three decks, 21 cards, five required pure
sequences, discard recycling, a 60-turn limit, and the `axial_table` network.
This is a one-player curriculum for the same model that plays multiplayer.

```powershell
.\.venv\Scripts\python -m src.cli.curriculum --output-dir runs\full21
```

For a larger model, set `--width`, `--heads`, and `--blocks` on the first run.
Both `src.cli.solo` and `src.cli.curriculum` accept these flags.
The default spec is `ModelSpec(width=64, heads=4, blocks=2)`.

```powershell
.\.venv\Scripts\python -m src.cli.curriculum --output-dir runs\full21-wide --width 128 --heads 8 --blocks 3
```

The output directory contains `latest-state.pt`, `latest-model.pt`,
`best-model.pt`, `baseline-model.pt`, and `metrics.jsonl`. The full state
checkpoint includes model and optimizer tensors, replay, curriculum and
learning counters, plus Python and PyTorch random-number-generator states.

Resume the saved run, then benchmark its latest model.

```powershell
.\.venv\Scripts\python -m src.cli.curriculum --resume runs\full21\latest-state.pt --output-dir runs\full21
.\.venv\Scripts\python -m src.cli.benchmark --checkpoint runs\full21\latest-model.pt --output runs\full21-evaluation.json --games 300 --warm-games 0
```

Model checkpoint version 4 and information encoding version 2 remain unchanged.
Existing `axial_table` v4 weights can warm-start a new run.
`TrajectoryReplay` v4 stores shaped terminal utilities.
Resume rejects replay v3 with an instruction to warm-start through `--initial-model`.
Old model architectures and old registry records remain unsupported.

To finetune existing weights at curriculum stage 2, start fresh optimizer and replay state.

```powershell
.\.venv\Scripts\python -m src.cli.curriculum --initial-model runs\full21\best-model.pt --initial-stage 2 --output-dir runs\full21-finetune --learning-rate 0.0001 --epsilon-start 0.08 --epsilon-end 0.02 --epsilon-decay-episodes 10000
```

Curriculum defaults remain learning rate 0.0003 and epsilon 0.3 to 0.1 over 10000 stage episodes.
All four settings persist in `TrainingConfig`.
Resume rejects changed values in both spaced and `=` argument forms.

`Learner` uses 5-step Double DQN with `gamma = 1.0`.
The online network selects a legal bootstrap action, and the target network values that action.
Solo declaration earns +1.
A solo loss with `p` deadwood points earns `-1 + 20 / (40 + p)`, inside `(-1, -0.5]`.
Multiplayer utility averages pairwise margins weighted by 1 when either seat declared, and by 0.5 otherwise.
The declarer earns +1. The best exhausted hand earns at most +0.5.
An opponent's declaration never improves another seat's utility.
Multiplayer utilities remain zero-sum.
`DealOutcome.margin` stays unshaped for PFSP, ratings, and promotion.
Solo, curriculum, and terminal lookahead targets all use `solo_outcome`.
Curriculum episode metrics expose `utility` and `penalty`.
Greedy validation and best-model selection still use binary declaration win rates.
AdamW, a Polyak target network, and gradient clipping update the challenger.
Win, penalty, opponent-card, afterstate, and budgeted lookahead losses provide additional training targets.

`encode` converts each visible observation into a pure `Features` record of 3,208 float32 values.
The encoder includes meld membership, public-card beliefs, per-kind history recencies, and six relative seat slots.
`PolicyNet` combines rank convolutions, suit pooling, table attention, and FiLM on a four-suit, 13-rank grid.
`collate` batches cached records without rescanning history.
Hidden hands provide auxiliary training targets only.
Browser exports omit `aux.*` weights and retain only `policy.*`.
See the [current model contract](README.md#current-model-contract) for the model layout.

`TrajectoryReplay` v4 samples decisions uniformly within each player count.
Checkpoints store one visible-history journal per trajectory and a prefix for each decision.
Resume validates each snapshot against its prefix and restores the replay sampling cursor.

## Run shared-deck arena matches

Repeat `--model` once per seat, using either a checkpoint or `random`.

```powershell
.\.venv\Scripts\python -m src.cli.arena --model runs\full21\latest-model.pt --model random --games 20 --seed 1000 --max-turns 60 --output runs\arena.json
```

`runs\arena.json` records seating, winners, terminal reasons, action counts,
turn counts, and remaining stock. Its `telemetry` field records stock draws
and discard draws per seat, plus the completed-turn number of each stock
refill. A refill does not reset the turn budget.
`run_matches` encodes each active state once and batches inference by policy.
`evaluate_games` advances up to 32 solo deals in lockstep while preserving `seed + index` order.

## Calibrate multiplayer draw limits

This study measures extra completion probability only. It does not validate
winners, loser placements, rating rankings, or the provisional capped-game
weight of 0.25. A supported cap is not evidence that those outcomes agree
with the guard.

The cap and remaining turn budgets are part of policy observations before
the cutoff. Matching seeds therefore do not guarantee identical action
prefixes. The study compares cap-conditioned play, not one fixed trajectory
truncated at different turns.

Use a directory containing immutable `model-<sha256>.pt` snapshots, such as
an improvement run's registry. The calibration adds a seeded random baseline.
The pool must contain enough distinct competitors for the largest table.

```powershell
.\.venv\Scripts\python -m src.cli.calibration --models-dir runs\improve --output-dir runs\calibration --caps 30 60 90 120 180 --players 2 3 4 --blocks 512 --workers 4
```

Each independent deal samples a lineup and seating independently of the deck,
then plays that same assignment under every candidate cap. This command runs
7,680 games. It uses independent deals rather than counting correlated seat
permutations as additional trials.

`protocol.json` fixes the models, source, caps, seeds, and sample budget.
Completed cases are persisted individually. Repeating the same command resumes
unfinished work. Use a new output directory and fresh seeds for a new study.
`status.json` reports progress. `summary.json` contains outcomes, refill counts,
and the cap comparison.

The default selection bound allows at most a 5% probability that the larger
guard obtains a declaration where the candidate cap does not. One-sided exact
binomial bounds account for the compared caps and table sizes. An incomplete
or underpowered study does not select a cap, and the largest guard is never
chosen as a fallback. This is evidence about the frozen model pool, not a
universal human-game limit.

The default budget is 128 independent blocks per table size. The default
caps and table sizes make six comparisons. Even with zero extra completions,
at least 94 complete blocks are needed to bring the upper bound below 5%.
Explicit smaller budgets remain available for smoke tests and can return
`insufficient_evidence`.

Workers have coordinator-enforced deadlines. A timeout is an execution failure,
not a game draw. The command stops on a worker failure and retains completed
cases for recovery.

## Train against a frozen league

Only the challenger is updated. The initial model, checkpoint opponents, and
random policy remain frozen. Copy the generated full-hand model to make the
opponent snapshot explicit.
Training defaults to two, three, and four players.

```powershell
New-Item -ItemType Directory -Force checkpoints
Copy-Item runs\full21\latest-model.pt checkpoints\frozen-full21.pt
.\.venv\Scripts\python -m src.cli.league --initial-model runs\full21\latest-model.pt --opponent checkpoints\frozen-full21.pt --output-dir runs\league --players 2 3 4 --matches 128 --matches-per-round 16 --max-seconds 3600 --checkpoint-every 16
.\.venv\Scripts\python -m src.cli.league --initial-model runs\full21\latest-model.pt --opponent checkpoints\frozen-full21.pt --resume runs\league\latest-state.pt --output-dir runs\league --matches 128 --max-seconds 3600 --checkpoint-every 16
```

League options have these defaults.

- `--max-turns 60` limits each seat's completed draw-discard turns.
- `--batch-size 128` caps the replay samples in each optimizer update.
- `--updates-per-match 4` sets the updates after a match with challenger actions.
- `--matches-per-round 16` batches deals before updates.
- `--learning-rate 0.0003` sets the optimizer rate.
- `--epsilon-start 0.2` and `--epsilon-end 0.05` set exploration.
- `--epsilon-decay-matches 10000` sets the exploration decay budget.

League checkpoints are published only at match boundaries and preserve the
opponent-pool identity, replay, optimizer, counters, and random streams.
Count and challenger-seat schedules are independent and deterministic.
Replay gives each count equal sampling shares and samples its decisions uniformly.
Resume restores the saved count selection and schedule cursors.
`OpponentScheduler` uses PFSP weights `(1 - p)^2` from smoothed pairwise placement scores.
The default schedule reserves 35% for the champion and 15% for available baseline and random opponents.
Other frozen opponents share the remaining probability according to those weights.

On resume, omit learning and exploration options to retain their saved values, or supply matching values.
The same rule applies to `--max-turns`, `--batch-size`, `--updates-per-match`, and `--matches-per-round`.
Changes to those options, `--players`, or `--seed` fail before checkpoint publication.
Both spaced and `=` argument forms enforce this rule.
Player counts must be sorted and distinct.
Starting without `--resume` rejects a directory that already contains a league
model or state checkpoint. Use a new output directory for a fresh run.

## Run finite improvement cycles

`ImprovementController` v4 trains one league challenger, selects it, and confirms it on fresh seeds.
It registers one candidate per cycle and promotes only through `evaluate_gate` evidence.

```powershell
.\.venv\Scripts\python -m src.cli.improve --initial-model runs\full21\latest-model.pt --output-dir runs\improve --max-seconds 3600 --max-cycles 1 --matches-per-round 16 --workers 4
.\.venv\Scripts\python -m src.cli.improve --resume runs\improve\latest-state.pt --output-dir runs\improve --max-seconds 3600 --workers 2
```

`champion.json` is the sole champion pointer. Registered model snapshots are
immutable. `status.json` describes the latest stable published state.

The challenger follows the champion at each cycle boundary.
Self-promotion keeps its weights, AdamW state, and replay.
An external champion change reloads weights and resets the optimizer.
`--rebase-after-rejections 8` applies the same rebase after eight rejected cycles.
Replay and random streams survive both forms of rebase.

`FinetuneSchedule` defaults to learning rate 0.0001 and epsilon 0.08 to 0.02.
Verified ordinary full21 win rate of at least 0.95 selects learning rate 0.00002 and epsilon 0.04 to 0.01.
The controller reads `final-best.json` beside the initial model.
That file must bind ordinary outcomes to the model's SHA-256 and matching recycling rules.
`--learning-rate`, `--epsilon-start`, and `--epsilon-end` override individual defaults.
`--epsilon-decay-matches` defaults to 10000.
`LeagueConfig` stores the resolved values and remains the source of truth on resume.
An explicit changed finetune option fails even when written as `--epsilon-start=value`.

Keep the default `--selection-blocks 32`, `--confirmation-blocks 128`, and `--solo-games 256` for promotion.
The gate uses 4x, 2x, and 1x the base block count for two, three, and four players.
Selection therefore plays 256, 384, and 768 games.
Confirmation plays 1,024, 1,536, and 3,072 games.
Each block batches all 2, 6, or 24 seat orders.
The registry requires disjoint selection and confirmation seeds.

`gate_schema_version == 4` records placement margins, declaration margins, `phase`, and `gate_failures`.
Better loser placements count even when neither contender declares.
Within base block `b`, each count's replica `r` uses deal seed `seed + 4 * b + r`.
The gate averages replicas within each count, then averages counts equally.
The 95% bootstrap resamples whole base blocks and preserves cross-count seed covariance.
Evidence includes `multiplayer_ci95`, `multiplayer_ci95_by_count`, `declaration_ci95`, and paired binary `solo_ci95`.
Selection requires positive aggregate mean placement gain and screens count, declaration, and solo means.
Confirmation requires aggregate placement superiority and bounded per-count, declaration, and solo noninferiority.
Both phases retain the 256-game paired solo minimum.
See [promotion evidence](RATINGS.md#promotion-evidence-stays-separate) for the complete gate contract.

Use `--workers` to set the gate process count. The default is 1.
Each worker verifies the source fingerprint and loads frozen policies once.
Workers use one PyTorch thread each.
Only the coordinator appends and flushes completed jobs to `gates/<gate_id>.jsonl`.
Interrupted gates retain completed blocks and paired solo slices.
Resume with the same model hashes, rules, seeds, and budgets to skip that work.
Result assembly follows block index and solo-slice start, not worker completion order.

On `--resume`, change `--workers`, `--checkpoint-every`, or `--max-cycles` as needed.
Omit those options to retain their saved values.
Learning settings, player counts, and the initial opponent pool cannot change on resume.

Keep full 21-card, 60-turn recycling rules for promotion.
Smaller `--selection-blocks`, `--confirmation-blocks`, and `--solo-games`
values exercise the workflow but cannot authorize promotion.
An explicit reduced `--max-turns` cap also cannot promote.
Fresh random weights or a few updates do not establish trained strength.

## Verify the complete current lifecycle

Use a temporary output directory for a bounded local check.

```powershell
.\.venv\Scripts\python tools\verify_dynamic_model.py --output-dir "$env:TEMP\papplu-dynamic-verification"
```

The script verifies `axial_table` v4 checkpoints and runs full 60-turn games at each supported player count.
Separate one-turn smoke games exercise mixed-count updates, exact full-state
resume, a rejected improvement cycle, and persisted tournaments and reports.
The resume check compares an uninterrupted learner's actual matches and updates
with the checkpoint restored by the CLI.
Gate v4 checks compare in-process and two-worker results byte for byte and reuse a completed JSONL journal.
The script checks persisted finetune flags in curriculum, league, and improvement states.
Unit tests also interrupt a gate between jobs and verify that resume skips completed blocks and solo slices.
Each invocation creates a new timestamped directory, so repeated checks
preserve earlier artifacts.
Read `evidence.json` for the commands, exit codes, checkpoint identities, and
actual output paths. Inspect the provisional rating report without treating
one deal block as strength evidence.

## What training CI checks

`.github\workflows\training-ci.yml` is a CPU regression gate, not deployment
automation or a long-running training job. It keeps evaluator and browser
simulator checks free of PyTorch, verifies CLI help at the optional-dependency
boundary, and runs `python -m unittest discover -s tests -v` across the complete
Python suite.
The training job then runs `tools/verify_dynamic_model.py` for play, checkpoint
publication, exact resume, improvement, and persisted ratings verification.

Every pull request runs the `portable` and `training` jobs before merge. Push
path filters do not affect pull requests and apply only to direct pushes.
`workflow_dispatch` remains available for a manual run.
CI uses the current-only verification script instead of old-format
compatibility comparisons.
Artifact upload runs even when verification fails.
It includes `evidence.json`, `rating-report.json`, and `benchmark.json`.
