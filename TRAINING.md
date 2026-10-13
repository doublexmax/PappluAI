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

To fine-tune those weights with fresh optimizer and replay state:

```powershell
.\.venv\Scripts\python -m src.cli.solo --load checkpoints\smoke.pt --episodes 5 --checkpoint checkpoints\smoke-finetuned.pt
```

## Train and resume full 21-card play

The curriculum trainer defaults to three decks, 21 cards, five required pure
sequences, discard recycling, a 60-turn limit, and the `dynamic_table` network.
This is a one-player curriculum for the same model that plays multiplayer.

```powershell
.\.venv\Scripts\python -m src.cli.curriculum --output-dir runs\full21
```

The output directory contains `latest-state.pt`, `latest-model.pt`,
`best-model.pt`, `baseline-model.pt`, and `metrics.jsonl`. The full state
checkpoint includes model and optimizer tensors, replay, curriculum and
learning counters, plus Python and PyTorch random-number-generator states.

Resume the exact run and then benchmark its latest model:

```powershell
.\.venv\Scripts\python -m src.cli.curriculum --resume runs\full21\latest-state.pt --output-dir runs\full21
.\.venv\Scripts\python -m src.cli.benchmark --checkpoint runs\full21\latest-model.pt --output runs\full21-evaluation.json --games 300 --warm-games 0
```

Start this architecture with fresh weights in a new output directory.
Checkpoint version 3, information encoding version 2, and replay version 2
are the only supported formats. Historical checkpoints, old replay, and old
registry records fail explicitly. No conversion path remains.

Decision observations retain their complete visible histories.
The current and initial own hand, public draws and discards, and own stock
draw faces feed the shared history encoder. Peers' stock faces and hidden
stock order never enter it. Inference reconstructs history without a cached
private recurrent state. Replay uses the same collation.
Checkpoints store one journal per episode and a history-prefix length for each
decision. Resume validates that each snapshot matches its history prefix and
restores shared event objects.

## Run shared-deck arena matches

Repeat `--model` once per seat, using either a checkpoint or `random`.

```powershell
.\.venv\Scripts\python -m src.cli.arena --model runs\full21\latest-model.pt --model random --games 20 --seed 1000 --max-turns 60 --output runs\arena.json
```

`runs\arena.json` records seating, winners, terminal reasons, action counts,
turn counts, and remaining stock. Its `telemetry` field records stock draws
and discard draws per seat, plus the completed-turn number of each stock
refill. A refill does not reset the turn budget.

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
`status.json` reports progress; `summary.json` contains outcomes, refill counts,
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
.\.venv\Scripts\python -m src.cli.league --initial-model runs\full21\latest-model.pt --opponent checkpoints\frozen-full21.pt --output-dir runs\league --players 2 3 4 --matches 128 --max-seconds 3600 --checkpoint-every 16
.\.venv\Scripts\python -m src.cli.league --initial-model runs\full21\latest-model.pt --opponent checkpoints\frozen-full21.pt --resume runs\league\latest-state.pt --output-dir runs\league --matches 128 --max-seconds 3600 --checkpoint-every 16
```

League checkpoints are published only at match boundaries and preserve the
opponent-pool identity, replay, optimizer, counters, and random streams.
Count and challenger-seat schedules are independent and deterministic.
Replay samples counts equally before episodes and decisions.
Resume restores the saved count selection and schedule cursors.
An explicit count selection must match the saved configuration.
Starting without `--resume` rejects a directory that already contains a league
model or state checkpoint. Use a new output directory for a fresh run.

## Run finite improvement cycles

Improvement alternates league and independent research training, evaluates
both candidates, and promotes only through the guarded evidence gate.

```powershell
.\.venv\Scripts\python -m src.cli.improve --initial-model runs\full21\latest-model.pt --output-dir runs\improve --max-seconds 3600 --max-cycles 1
.\.venv\Scripts\python -m src.cli.improve --resume runs\improve\latest-state.pt --output-dir runs\improve --max-seconds 3600
```

`champion.json` is the sole champion pointer. Registered model snapshots are
immutable. `status.json` describes the latest stable published state.

Selection needs at least 32 independent seed blocks for each player count.
Confirmation needs at least 128 blocks for each count on disjoint seeds.
Four-player blocks contain all 24 seat permutations.
Each count needs a positive lower 95% confidence bound.
The aggregate gain threshold remains 0.05.
The paired solo retention check still needs 256 games.

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

The script runs full 60-turn legal games at each supported player count.
Separate one-turn smoke games exercise mixed-count updates, exact full-state
resume, a rejected improvement cycle, and persisted tournaments and reports.
The resume check compares an uninterrupted learner's actual matches and updates
with the checkpoint restored by the CLI.
Each invocation creates a new timestamped directory, so repeated checks
preserve earlier artifacts.
Read `evidence.json` for the commands, exit codes, checkpoint identities, and
actual output paths. Inspect the provisional rating report without treating
one deal block as strength evidence.

## What Training CI checks

`.github\workflows\training-ci.yml` is a CPU regression gate, not deployment
automation or a long-running training job. It keeps evaluator and browser
simulator checks free of PyTorch, verifies CLI help at the optional-dependency
boundary, runs the focused training tests, and exercises short checkpoint and
resume workflows.

Every pull request runs the `portable` and `training` jobs before merge. Push
path filters do not affect pull requests and apply only to direct pushes.
`workflow_dispatch` remains available for a manual run.
CI uses the current-only verification script instead of old-format
compatibility comparisons. It uploads lifecycle evidence and reports.
