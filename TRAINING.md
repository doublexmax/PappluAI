# Train a Papplu model

Use this guide to create a resumable self-play run and export its selected checkpoint to the browser.
The [model contract](README.md#model-and-training-contract) describes the network, encoding, and terminal utility.
No command here provisions Azure resources or spends cloud credit.

## Browser model

`web/papplu-model.json` contains untrained v6 bootstrap weights with seed 1729.
Its `source.kind` is `bootstrap` and `source.trained` is `false`.
The default `ModelSpec(80, 3)` has 342,804 parameters and five-tap SwiGLU card blocks.
Encoding version 6 contains 854 observable float32 inputs. Model checkpoints use version 6.
The bundle is for compatibility checks. It has no measured playing-strength claim.
Replace it with a trained v6 checkpoint after a separate held-out evaluation.

## Install PyTorch

Use Python 3.11, as CI does.
Create the environment from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-training.txt
```

The only direct training dependency is `torch>=2.6`.
The trainer and exporter do not use NumPy.
Torch can emit its own NumPy initialization warning when NumPy is absent.
The evaluator, encoders, bootstrap exporter, and command help need neither package.

## Check the lifecycle before a long run

Start four cold deals with a two-turn cap.
This tests software behavior, not playing strength.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\smoke --players 1 2 3 4 --max-turns 2 --round-episodes 4 --batch-size 8 --replay-capacity 128 --validate-every 4 --validation-games 1 --max-episodes 4
```

The JSON response reports `episodes` as 4 and `status` as `max_episodes`.
The directory contains these outputs.

| File | Contents |
| --- | --- |
| `latest-model.pt` | Current network weights and game settings. |
| `best-model.pt` | Selected validation checkpoint, or the initial model before the first validation. |
| `latest-state.pt` | Network, AdamW state, compressed replay, sampling RNG, counters, configuration, and snapshot paths. |
| `initial-model.pt` | Frozen starting weights in the initial opponent pool. |
| `opponents/` | Imported opponents, selected best snapshots, and periodic strong-policy snapshots. |
| `status.json` | Saved counters, validation results, active snapshots, and stop reason. |
| `metrics.jsonl` | One record per completed training round. |

Resume the same run to eight total deals.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\smoke --max-episodes 8
```

`--max-episodes` is an absolute deal count, not an additional count.
Self-play can produce several seat trajectories per deal.
The resumed response reports eight episodes.
Omit the original recipe flags to restore their saved values.

Evaluate and export the selected checkpoint.

```powershell
.\.venv\Scripts\python -m src.cli.evaluate --checkpoint runs\smoke\best-model.pt --games 4 --warm-games 0 --output runs\smoke\evaluation.json
.\.venv\Scripts\python -m src.cli.arena --model runs\smoke\best-model.pt --model random --games 4 --max-turns 2 --output runs\smoke\arena.json
.\.venv\Scripts\python tools\weights_export.py --checkpoint runs\smoke\best-model.pt --output runs\smoke\model.json --golden runs\smoke\golden.json
```

The solo report separates greedy and random results.
The arena report records seat order, winner, terminal reason, draw sources, and refill turns.
Neither report assigns a rating.

## Start a full 21-card run

Use a new directory so the smoke cap does not carry into training.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\full21 --max-episodes 10000 --max-seconds 3600
```

The defaults use three decks, 21 cards, five required pure sequences, and 60 turns per seat.
The trainer cycles `--players 1 2 3 4`.
It uses the single default `ModelSpec(80, 3)`, `gamma=0.99`, a 40,000-step replay, and batches of at most 256 steps.
Each 32-deal round schedules two learner updates per completed deal. Use `--updates-per-episode` to change that count.
Smooth-L1 fits the chosen action to its discounted terminal utility. There is no margin loss.
Each learner seat draws a fixed exploration rate for its deal from `(0.01, 0.01, 0.03, 0.08)` using its own RNG.
Seat index and training progress do not determine exploration.
AdamW starts at `learning_rate=5e-4`.
Before each update, the rate is `max(learning_rate * 0.15, learning_rate / sqrt(1 + updates / lr_decay_updates))`.
`--lr-decay-updates` defaults to 8,000. A nonpositive value disables learning-rate decay.

Every deal starts cold.
There are no solved-hand warm starts or curriculum stages.
On each multiplayer deal, self-play has probability 0.75.
Every self-play seat contributes its own terminal utility and trajectory.
Other deals record only one randomly selected learner seat.
Each opponent seat uses `RandomPolicy` with probability 0.15, or a greedy checkpoint from the active snapshot pool with probability 0.85.
Checkpoint selection is uniform over the latest `max_pool` entries.
Solo deals always use the learner.
The rollout batches all active learner seats together, including solo and multiplayer deals.
No scripted bot or meld heuristic enters the encoder, rollout, or loss.

Declaring earns +1.
Losing to another declaration earns between -1 and -0.5 according to capped penalty points.
Exhaustion earns between -0.8 and -0.1 according to qualifying sequences and capped penalty points.
The same objective applies to solo and multiplayer. See the [utility formulas](README.md#dmc-learning).

Continue the run with a larger absolute episode limit.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\full21 --max-episodes 20000 --max-seconds 3600
```

Without either limit, the command runs one more training round.
Ctrl+C requests a stop.
Time limits and stop requests are checked between completed rounds.
A rollout round or validation can overrun the requested time limit.
Use a smaller `--round-episodes` when starting a run that needs shorter stop intervals.
The trainer saves completed work and closes actor processes before returning.

## Use more CPU actors or a GPU learner

Set the actor count when starting a run, or change it on resume.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\parallel --actors 4 --round-episodes 32 --validation-games 16 --max-episodes 10000
```

One actor runs inline.
More than one actor uses a persistent process pool.
Each worker uses one Torch thread and simulates its chunk with the current frozen round weights.
Validation uses the same pool for solo games and multiplayer seat rotations.
Its chunk boundaries and seeds do not depend on the actor count.
Set `--torch-threads` to change the learner's Torch thread count. The default is one.
Only the learner updates weights, appends replay, and writes checkpoints.
Rollout workers return features, actions, and terminal utility, not private observations.
Validation workers return metric totals.
The pool stays alive across rounds and closes when `run` returns.

If the host has a CUDA-capable Torch installation, select the learner device.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\gpu --device cuda --actors 4 --round-episodes 32 --batch-size 256 --validation-games 16 --max-seconds 3600
```

Rollout workers remain on CPU. Replay stays on CPU.
The default replay allocates about 131 MiB before Python and optimizer overhead.
Checkpoint compression reduces saved bytes, not the live replay allocation.
Use `--replay-capacity` to set the memory budget when starting a run.

More actors or a larger network do not guarantee better throughput.
Compare completed deals and updates over the same time budget before renting more compute.
Keep validation settings fixed when comparing runs.
The fixed validation workload can cost more than a short training round.

## Use an initial checkpoint

The default 80-by-3 model has 342,804 parameters.
The same architecture supports other widths for compatibility tests and resource constraints.
The 128-by-3 model has 867,972 parameters.
Both sizes use the same encoding and browser inference implementation.
If you omit either `--width` or `--blocks`, the CLI uses 80 or 3 for that dimension.

To keep weights but start new replay and AdamW state, supply an initial checkpoint.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\continued --initial-model runs\full21\best-model.pt --opponent runs\full21\latest-model.pt --max-episodes 10000
```

The initial checkpoint determines the network spec and game settings.
Do not combine `--initial-model` with `--width` or `--blocks`.
Optional `--opponent` checkpoints must match those game settings.
The trainer copies them into the new run.
Initial models and imported opponents apply only to fresh runs.

Resume restores the recipe rather than changing it.
The CLI allows `--actors`, `--device`, and `--torch-threads` changes on resume.
The Python API also allows `checkpoint_every` changes.
Other changed settings fail before training.
Encoding and checkpoint versions before 6 and trainer state versions before 9 require a fresh run.

## Inspect validation

By default, validation runs after each 1,000 completed deals.
`--validation-games 64` controls held-out solo games and multiplayer seed blocks.
Set it to zero only to disable validation for a bounded diagnostic run.

Solo validation measures cold greedy declarations when solo is in `--players`.
For every configured multiplayer count, validation rotates the learner through every seat on every seed.
It compares against `RandomPolicy`, the frozen `initial-model.pt`, and the reigning `best-model.pt`.
A pairwise placement win earns 1, a tie earns 0.5, and a loss earns 0.
Other losing seats still compare their final deadwood penalties after a declaration.

The validation `score` is the unweighted mean of solo, `Xp_random`, and `Xp_initial` rates for the configured counts.
The default counts produce seven anchor rates, solo and random and initial opponents at two, three, and four players.
The score lies in `[0, 1]`.
Training seeds are even and validation seeds are odd, so those deal seeds do not overlap.
Validation reuses fixed seeds and the same frozen initial model across the run.

Each `Xp_best` rate measures head-to-head placement credit against the reigning best for that player count.
`vs_best` is their unweighted mean and does not enter the fixed-anchor `score`.
With multiplayer enabled, the first validation selects the current network.
Later validations replace `best-model.pt` when `vs_best > 0.51` and `score >= best_score - 0.02`.
An anchor gain alone does not promote a multiplayer model.
`best_score` retains the highest fixed-anchor score among promoted models.
Solo-only runs promote when `score >= best_score`, including equal scores.
Every validation records `promoted` as a boolean.

On promotion, the trainer copies the best checkpoint to `opponents/best-<episode>.pt`.
Without promotion, `vs_best >= 0.46` saves the current network as `opponents/snap-<episode>.pt`.
Snapshot names pad the episode count to 12 digits.
The active pool retains the latest eight snapshots by default.
Older snapshot files remain on disk but no longer enter the active pool.
The current network can regress between validations.
The selected validation score is not a rating or proof of strength against unseen opponents.

Read `status.json` for the latest saved selection.
Read `metrics.jsonl` for each round's episode count, exploration spectrum, trajectory count, Monte Carlo loss, updates, samples, and optional validation.
`latest-state.pt` compresses tensor bytes with zlib and restores them without NumPy.
With one actor, splitting at the same round boundary preserves model, optimizer, replay, and sampling exactly.
Changing the actor count can change batch composition and learning order.

## Evaluate on separate deals and publish browser weights

Use additional evaluation seeds rather than selecting against the same validation deals again.

```powershell
.\.venv\Scripts\python -m src.cli.evaluate --checkpoint runs\full21\best-model.pt --games 300 --seed 9000001 --warm-games 0 --output runs\full21\evaluation.json
.\.venv\Scripts\python -m src.cli.arena --model runs\full21\best-model.pt --model random --games 100 --seed 9100001 --max-turns 60 --output runs\full21\arena.json
```

Repeat `--model` for each additional arena seat.
Arena accepts `--model random` or checkpoint paths for frozen neural opponents.
Keep the game rules and turn cap the same across comparisons.
Optional `--warm-games` and `--sensitivity` test discard puzzles only.
Puzzle results do not replace cold-game evidence and never enter training.

Export weights into the browser's default paths.

```powershell
.\.venv\Scripts\python tools\weights_export.py --checkpoint runs\full21\best-model.pt
node --test tests/qnet.test.mjs tests/ai.test.mjs tests/table.test.mjs
```

The exporter writes the model and its matching golden cases together.
It checks the portable forward pass against Torch.
The full Python suite exports nonzero 80-by-3 and 128-by-3 weights and checks all 54 Node Q-values within `1e-5`.
Keep `web/papplu-model.json` and `tests/fixtures/qnet-golden.json` together.
`--bootstrap --seed 1729` regenerates the deterministic untrained compatibility model.
Use that option for compatibility checks, not to publish playing-strength results.

Run the [verification commands](README.md#verification) before accepting a training-code change.

## Training CLI options

Run `python -m src.cli.train --help` for argument syntax.
Fresh runs use the defaults described above. Resume restores saved settings unless an allowed override is supplied.

| Option | Meaning |
| --- | --- |
| `-h`, `--help` | Show help without loading Torch. |
| `--output-dir` | Required run directory. Reuse it to resume. |
| `--initial-model` | Starting checkpoint for a fresh run. |
| `--opponent` | Imported opponent checkpoint. Repeat for multiple opponents. |
| `--players` | Player counts, supplied as one or more integers. |
| `--actors` | Number of rollout and validation actors. |
| `--width` | Network width for a fresh model. |
| `--blocks` | Network block count for a fresh model. |
| `--max-turns` | Turn limit per seat. |
| `--round-episodes` | Deals per training round. |
| `--updates-per-episode` | Learner updates per completed deal. |
| `--batch-size` | Maximum replay samples per update. |
| `--replay-capacity` | Maximum replay steps in memory. |
| `--learning-rate` | Initial AdamW learning rate. |
| `--lr-decay-updates` | Update count for smooth learning-rate decay. Nonpositive disables decay. |
| `--gamma` | Terminal-return discount factor. |
| `--self-play-share` | Probability that every seat uses the learner. |
| `--validate-every` | Completed-deal interval between validations. |
| `--validation-games` | Solo games and multiplayer seed blocks per validation. Zero disables validation. |
| `--seed` | Training seed. |
| `--device` | Learner Torch device. |
| `--torch-threads` | Learner Torch thread count. Workers always use one. |
| `--max-episodes` | Absolute completed-deal limit. |
| `--max-seconds` | Wall-clock budget, checked between rounds. |
