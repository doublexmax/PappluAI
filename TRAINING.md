# Training walkthrough

All commands in this guide run from a fresh repository clone on Windows
PowerShell. Training is CPU-compatible but requires the optional PyTorch
dependency.

## Set up a fresh clone

```powershell
git clone https://github.com/doublexmax/PappluAI.git
Set-Location PappluAI
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-training.txt
```

Command-line entry points live in `src.cli`. They parse `--help` before loading
PyTorch. Game rules are in `src.game`, networks in `src.model`, training
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
sequences, discard recycling, a 60-turn limit, and the `suit_conv` network.

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

`src.model.network.load_checkpoint` accepts version 1 and version 2 model
checkpoints. Historical version 1 checkpoints retain the nonrecycling rule
when that field is absent.

## Run shared-deck arena matches

Repeat `--model` once per seat, using either a checkpoint or `random`.

```powershell
.\.venv\Scripts\python -m src.cli.arena --model runs\full21\latest-model.pt --model random --games 20 --seed 1000 --max-turns 60 --output runs\arena.json
```

`runs\arena.json` records seating, winners, terminal reasons, action counts,
turn counts, and remaining stock. Its `telemetry` field records stock draws
and discard draws per seat, plus the completed-turn number of each stock
refill. A refill does not reset the turn budget.

## Train against a frozen league

Only the challenger is updated. The initial model, checkpoint opponents, and
random policy remain frozen. Copy the generated full-hand model to make the
opponent snapshot explicit.

```powershell
New-Item -ItemType Directory -Force checkpoints
Copy-Item runs\full21\latest-model.pt checkpoints\frozen-full21.pt
.\.venv\Scripts\python -m src.cli.league --initial-model runs\full21\latest-model.pt --opponent checkpoints\frozen-full21.pt --output-dir runs\league --matches 128 --max-seconds 3600 --checkpoint-every 16
.\.venv\Scripts\python -m src.cli.league --initial-model runs\full21\latest-model.pt --opponent checkpoints\frozen-full21.pt --resume runs\league\latest-state.pt --output-dir runs\league --matches 128 --max-seconds 3600 --checkpoint-every 16
```

League checkpoints are published only at match boundaries and preserve the
opponent-pool identity, replay, optimizer, counters, and random streams.

## Run finite improvement cycles

Improvement alternates league and independent research training, evaluates
both candidates, and promotes only through the guarded evidence gate.

```powershell
.\.venv\Scripts\python -m src.cli.improve --initial-model runs\full21\latest-model.pt --output-dir runs\improve --max-seconds 3600 --max-cycles 1
.\.venv\Scripts\python -m src.cli.improve --resume runs\improve\latest-state.pt --output-dir runs\improve --max-seconds 3600
```

`champion.json` is the sole champion pointer. Registered model snapshots are
immutable. `status.json` describes the latest stable published state.

## Migrate to discard recycling

The fresh run above already uses discard recycling, so it does not need
migration. If you own a historical nonrecycling improvement directory, set its
path explicitly and migrate it into a new directory:

```powershell
$legacyRun = "C:\path\to\historical-nonrecycling-run"
if (-not (Test-Path "$legacyRun\latest-state.pt")) { throw "latest-state.pt not found" }
.\.venv\Scripts\python -m src.cli.rule_migration --source $legacyRun --output runs\recycling
```

The migration preserves model weights, counters, random streams, and champion
identity. It clears old-rule replay, optimizer moments, candidates, and
evaluation evidence. Repeating the command against a completed destination is
idempotent.

## What Training CI checks

`.github\workflows\training-ci.yml` is a CPU regression gate, not deployment
automation or a long-running training job. It keeps evaluator and browser
simulator checks free of PyTorch, verifies CLI help at the optional-dependency
boundary, runs the focused training tests, and exercises short checkpoint and
resume workflows.

Every pull request runs the `portable` and `training` jobs before merge. Push
path filters do not affect pull requests and apply only to direct pushes.
`workflow_dispatch`
remains available when the complete gate needs to be started manually. Set its
optional `baseline_ref` input only for a one-off package or serialization
migration. That separate job compares deterministic model, replay, checkpoint,
and full-state behavior with the selected baseline and uploads both reports.
