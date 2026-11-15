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
PyTorch. Game rules are in `src.game`, the current network is in `src.model`, the unified training
runtime lives in `src.training.train`, evaluation is in `src.evaluation`, and
persistent state operations live in `src.checkpoints`.

## Run a short smoke test

This uses a three-card warm-start puzzle and exports a standalone model checkpoint. It verifies training, model
checkpoint writing, loading, and evaluation without representing full Papplu
performance.

```powershell
.\.venv\Scripts\python -m src.cli.train --checkpoint checkpoints\smoke.pt --max-episodes 20 --num-decks 2 --cards-in-hand 3 --required-sequences 1 --max-turns 5 --warm-start-fraction 1 --validation-games 0
.\.venv\Scripts\python -m src.cli.benchmark --checkpoint checkpoints\smoke.pt --output runs\smoke-evaluation.json --games 20 --warm-games 20
```

The first command writes `checkpoints\smoke.pt`. The second writes paired
evaluation details to `runs\smoke-evaluation.json`.

To fine-tune those weights with fresh optimizer and replay state, pass `--initial-model`.

```powershell
.\.venv\Scripts\python -m src.cli.train --initial-model checkpoints\smoke.pt --checkpoint checkpoints\smoke-finetuned.pt --max-episodes 5
```

## Train and resume full 21-card play

`src.cli.train` defaults to one player, three decks, 21 cards, five required pure
sequences, discard recycling, a 60-turn limit, and the `card_mlp` network.
Passing `--curriculum-distances` enables staged solo curriculum deals for the same model that plays multiplayer.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\full21 --curriculum-distances 2 4 6 10
```

For a larger model, set `--width` and `--blocks` when starting a fresh run without `--initial-model`.
The default spec is `ModelSpec(width=64, blocks=2)`.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\full21-wide --curriculum-distances 2 4 6 10 --width 128 --blocks 3
```

The output directory contains `latest-state.pt`, `latest-model.pt`,
`best-model.pt`, `status.json`, and `metrics.jsonl`. The full state
checkpoint includes model and optimizer tensors, compressed replay bytes, curriculum and
learning counters, plus Python and PyTorch random-number-generator states.

Repeat the same `--output-dir` to resume `latest-state.pt`, then benchmark `latest-model.pt`.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\full21 --max-episodes 5000
.\.venv\Scripts\python -m src.cli.benchmark --checkpoint runs\full21\latest-model.pt --output runs\full21-evaluation.json --games 300 --warm-games 0
```

Model checkpoints use version 5 and information encoding version 3.
Current `card_mlp` v5 model weights can warm-start a new run through `--initial-model`.
Trainer state uses version 6 and `Replay` v6.
Incompatible older checkpoints and state files fail cleanly.
Old model architectures and old registry records remain unsupported.

To finetune existing weights at curriculum stage 2, start fresh optimizer and replay state in a new directory.

```powershell
.\.venv\Scripts\python -m src.cli.train --initial-model runs\full21\best-model.pt --output-dir runs\full21-finetune --curriculum-distances 2 4 6 10 --initial-stage 2 --learning-rate 0.0001 --epsilon-start 0.08 --epsilon-end 0.02 --epsilon-decay-episodes 10000
```

Default hyperparameters use learning rate 0.0003 and epsilon 0.30 to 0.05 over 10000 episodes.
All settings persist in `TrainConfig`, and resume rejects hyperparameter changes while allowing `--checkpoint-every`, `--gate-workers`, `--torch-threads`, and `--device` overrides.

`Learner` uses Q-only Deep Monte-Carlo with `gamma = 0.99`.
Each trajectory step fits `Q(s, a)` via smooth L1 loss to discounted terminal utility `u * gamma ** (T - 1 - t)`, with no target network or auxiliary heads.
Solo declaration earns +1.
A solo loss with `p` deadwood points earns `-1 + 20 / (40 + p)`, bounded between -1 and -0.5.
Multiplayer utility averages pairwise margins weighted by 1 when either seat declared, and by 0.5 otherwise.
The declarer earns +1. The best exhausted hand earns at most +0.5.
An opponent's declaration never improves another seat's utility.
Multiplayer utilities remain zero-sum.
`DealOutcome.margin` stays unshaped for PFSP, ratings, and promotion.
Solo and curriculum deals score terminal outcomes through `solo_outcome`.
Greedy validation and `best-model.pt` selection use binary declaration win rates.
AdamW and gradient clipping update the online network on Q-value returns.

`encode` converts each visible observation into a pure `Features` record of 802 float32 values, or 3,208 bytes.
A single pass over visible history populates 13 per-card channels, including opponent discards, known copies held by the next opponent and other opponents, and cards passed by the next opponent. It also writes nine channels across six relative seat slots and 17 context features.
`PolicyNet` computes vectorized meld-participation channels and runs `CardBlock` blocks over a 14-slot `[A, 2, ..., K, A]` rank strip with shared `card_norm` table pooling on the four-suit, 13-rank grid.
`collate` batches cached records without rescanning history.
Browser exports retain `policy.*` tensors and match Python Q-values.
See the [current model contract](README.md#current-model-contract) for the model layout.

`Replay` v6 enforces strict step-level capacity, samples steps uniformly within each populated player count, and gives populated player counts equal shares.
State checkpoints compress concatenated `Features` bytes with `zlib` so `torch.load(..., weights_only=True)` restores replay directly without retaining `Observation` instances.

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
`runs\improve\registry`. The calibration adds a seeded random baseline.
The pool must contain enough distinct competitors for the largest table.

```powershell
.\.venv\Scripts\python -m src.cli.calibration --models-dir runs\improve\registry --output-dir runs\calibration --caps 30 60 90 120 180 --players 2 3 4 --blocks 512 --workers 4
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

## Train across mixed tables and PFSP opponents

Only the online learner is updated. The initial model, `--opponent` checkpoints, and
random policy remain frozen in the opponent pool. Copy the generated full-hand model to make an additional
opponent snapshot explicit, and list every table size under `--players`.

```powershell
New-Item -ItemType Directory -Force checkpoints
Copy-Item runs\full21\latest-model.pt checkpoints\frozen-full21.pt
.\.venv\Scripts\python -m src.cli.train --initial-model runs\full21\latest-model.pt --opponent checkpoints\frozen-full21.pt --output-dir runs\league --players 1 2 3 4 --max-episodes 128 --round-episodes 16 --max-seconds 3600 --checkpoint-every 16
.\.venv\Scripts\python -m src.cli.train --output-dir runs\league --max-episodes 256 --max-seconds 3600 --checkpoint-every 16
```

Key training options have these defaults.

- `--max-turns 60` limits each seat's completed draw-discard turns.
- `--batch-size 128` caps the replay samples in each optimizer update.
- `--updates-per-episode 4` sets gradient updates per completed trajectory.
- `--round-episodes 16` batches episodes before updates.
- `--learning-rate 0.0003` sets the AdamW learning rate.
- `--epsilon-start 0.30` and `--epsilon-end 0.05` set exploration bounds.
- `--epsilon-decay-episodes 10000` sets the exploration decay budget.

Checkpoints are published at round and gate boundaries and preserve the
opponent-pool order, compressed replay, optimizer, counters, and random streams.
Player counts are sorted and distinct and cycle deterministically across episodes.
Replay gives each populated count equal sampling shares and samples steps uniformly within each bucket.
`pfsp_probabilities` weights non-anchor opponents by `(1 - p)^2` from Bayesian-smoothed pairwise placement scores.
The default schedule reserves 35% for the champion and 15% for baseline and random anchors.
Other frozen opponents share the remaining probability according to those weights.

On resume, omit training and exploration options to retain their saved values, or supply matching values.
Changes to rules, learning hyperparameters, `--players`, or `--seed` fail before checkpoint publication.
Only `--checkpoint-every`, `--gate-workers`, `--torch-threads`, and `--device` are tunable on resume.

## Run finite improvement cycles

Passing `--promote` enables two-stage selection and confirmation gates inside `Trainer`.
Each cycle trains for `--cycle-episodes` episodes, registers one candidate snapshot in `registry/`, and promotes only through `evaluate_gate` evidence.

```powershell
.\.venv\Scripts\python -m src.cli.train --initial-model runs\full21\latest-model.pt --output-dir runs\improve --players 2 3 4 --promote --max-seconds 3600 --max-cycles 1 --round-episodes 16 --gate-workers 4
.\.venv\Scripts\python -m src.cli.train --output-dir runs\improve --max-seconds 3600 --gate-workers 2
```

`registry/champion.json` is the sole champion pointer, mirrored to `champion-model.pt`. Registered model snapshots are
immutable. `status.json` describes the latest stable published state.

Self-promotion keeps learner weights, AdamW state, and replay intact, and adds the promoted champion snapshot to the PFSP pool.
`--rebase-after-rejections 8` reloads champion weights and resets AdamW moments after eight consecutive rejected cycles while preserving replay and random streams.
Supply explicit `--learning-rate`, `--epsilon-start`, `--epsilon-end`, and `--epsilon-decay-episodes` flags when fine-tuning a strong warm-start checkpoint.
`TrainConfig` stores the resolved values and remains the source of truth on resume.

Keep the default `--selection-blocks 32`, `--confirmation-blocks 128`, and `--solo-games 256` for promotion.
The gate uses 4x, 2x, and 1x the base block count for two, three, and four players.
Selection therefore plays 256, 384, and 768 games.
Confirmation plays 1,024, 1,536, and 3,072 games.
Each block batches all 2, 6, or 24 seat orders.
The registry requires disjoint selection and confirmation seeds.

`gate_schema_version == 5` records placement margins, declaration margins, `phase`, and `gate_failures`.
Better loser placements count even when neither contender declares.
Within base block `b`, each count's replica `r` uses deal seed `seed + 4 * b + r`.
The gate averages replicas within each count, then averages counts equally.
The 95% bootstrap resamples whole base blocks and preserves cross-count seed covariance.
Evidence includes `multiplayer_ci95`, `multiplayer_ci95_by_count`, `declaration_ci95`, and paired binary `solo_ci95`.
Selection requires positive aggregate mean placement gain and screens count, declaration, and solo means.
Confirmation requires aggregate placement superiority and bounded per-count, declaration, and solo noninferiority.
Both phases retain the 256-game paired solo minimum.
See [promotion evidence](RATINGS.md#promotion-evidence-stays-separate) for the complete gate contract.

Use `--gate-workers` to set the gate process count. The default is 1.
Each worker verifies the source fingerprint and loads frozen policies once.
Workers use one PyTorch thread each.
Only the coordinator appends and flushes completed jobs to `gates/<gate_id>.jsonl`.
Interrupted gates persist explicit `Selecting` or `Confirming` states and retain completed blocks and paired solo slices.
Resume with the same model hashes, rules, seeds, and budgets to skip that work.
Result assembly follows block index and solo-slice start, not worker completion order.

On resume, change `--gate-workers`, `--checkpoint-every`, `--torch-threads`, `--device`, or `--max-cycles` as needed.
Omit those options to retain their saved values.
Learning settings, player counts, and the seed cannot change on resume.

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

The script verifies `card_mlp` v5 checkpoints and runs full 60-turn games at each supported player count.
Separate one-turn smoke games exercise solo curriculum resume, mixed-count updates, exact full-state
resume, a rejected promotion gate cycle, and persisted tournaments and reports.
The resume check compares an uninterrupted `Trainer` run's actual matches and updates
with the checkpoint restored by `src.cli.train`.
Gate v5 checks compare in-process and two-worker results byte for byte and reuse a completed JSONL journal.
The script checks persisted `TrainConfig` hyperparameters and tunable `--gate-workers` resume updates.
Unit tests also interrupt a gate between phases and verify deterministic recovery.
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
