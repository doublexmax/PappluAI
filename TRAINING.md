# Train a Papplu model

Use this guide to create a resumable self-play run and export its selected checkpoint to the browser.
The [model contract](README.md#model-and-training-contract) describes the network, encoding, and terminal utility.
No command here provisions Azure resources or spends cloud credit.

## Shipped checkpoint

The browser loads the locked Azure `seed_22368` checkpoint from `web/papplu-model.json`.
It uses `ModelSpec(128, 3)` with 673,412 parameters and the 1,275-float version 5 encoding.
The checkpoint SHA256 is `a229f90e76ae9da9b8c97d35db10e2d921a3a46780fc54d292ba2bde3df67796`.
Selection used seed `700000` before the [held-out evaluation](README.md#held-out-results).
A1 is the prior merged `ModelSpec(64, 2)` checkpoint with 127,172 parameters.
Its checkpoint SHA256 is `094bbe8946fc890e93de7a2ce77f0ad45ad8f8d24a38f2249ffb31fdef39882b`.

The release check compared all 54 Q-values on 600 real Python observations against the exported model in Node.
The maximum absolute difference was `5.17e-6`, below `1e-5`. The check included 62 solo states after a self-discard with stock available.

The native `src.cli.arena` also compared the checkpoint with A1, starting at seed `860000`.
The candidate rotated through every seat on each deal. Native and browser shuffles differ.

| Players | Deals | Declaration wins | First-place credit, 95% deal interval | Pairwise credit | Mean penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 120 | 135/240 | 56.25% (50.83% to 61.67%) | 56.25% | 15.24 |
| 3 | 80 | 100/240 | 41.67% (36.67% to 46.67%) | 57.81% | 23.14 |
| 4 | 60 | 77/240 | 32.08% (27.08% to 37.50%) | 56.25% | 26.20 |

The Node v24.19.0 benchmark ran on Windows with an Intel Core i7-12700H.
Each pass timed 500 single-state `qValues` calls after 50 warmup calls.
The first pass measured 22.31 ms median, 22.63 ms p95, and 22.37 ms mean.
Five passes had no failures, with medians from 22.29 to 22.31 ms.
The call includes validation, encoding, and inference, but excludes model loading and observation conversion.
A separate CPU profile put 87.44% of samples in the `linear` function in `web/qnet.mjs`.

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
It uses `ModelSpec(128, 3)`, `gamma=0.99`, a 60,000-step replay, and batches of at most 256 steps.
Each 32-deal round schedules two learner updates per deal. Use `--updates-per-episode` to change that count.
Only greedy positive-return steps add a 0.10 legal-action margin to the Smooth-L1 objective.
Exploration and `RandomPolicy` steps do not receive that margin.
`--margin-weight` defaults to 0.35. Set it to zero to train without the margin.
Each learner seat samples one epsilon from `(0.01, 0.02, 0.05)` and keeps it for the whole deal.
Use `--epsilons` to change those stationary exploration rates.
There is no episode-based epsilon decay.
AdamW starts at `4e-4`. Its learning rate is divided by `sqrt(1 + updates / 20000)` with a floor of `6e-5`.
Use `--lr-decay-updates` to change the update scale. Zero disables decay.

Every deal starts cold.
There are no solved-hand warm starts or curriculum stages.
In solo, the policy mask prevents taking its own last discard while stock is available.
If stock is empty and the discard pile is not, the mask still allows taking the discard.
On each multiplayer deal, self-play has probability 0.35.
Other deals place the learner in a random seat against stationary opponents.
Each opponent seat chooses the reigning `best-model.pt` with probability 0.45, `HeuristicPolicy` with 0.20, `RandomPolicy` with 0.05, and a uniform snapshot with 0.30.
The snapshot pool includes imported `--opponent` checkpoints.
`--max-pool` defaults to eight active snapshots and applies when starting a run.
It limits the pool, not checkpoint files on disk.
An empty pool sends its share to the reigning best.
Every seat contributes its own terminal utility and trajectory in both modes.
Solo deals always use the learner.
The heuristic reads cached disjoint-packing features without exact penalty backtracking.
It is a sparring policy, not part of `PolicyNet` or the browser's greedy neural ranking.

Declaring earns +1.
Losing to another declaration earns between -0.85 and -0.35 according to capped penalty points.
Exhaustion earns between -1.00 and -0.60 according to qualifying sequences and capped penalty points.
Only positive returns use temporal discounting. Negative and zero returns keep their full utility at every step.
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
`--checkpoint-every` defaults to 64 completed deals.
Crossing an interval saves the latest model, replay, optimizer, and counters at the end of that round.
Stops always save, regardless of this interval.
You can change `--checkpoint-every` on resume.

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
Rollout workers return features, actions, greedy flags, and terminal utility, not private observations.
Validation workers return metric totals.
The pool stays alive across rounds and closes when `run` returns.

If the host has a CUDA-capable Torch installation, select the learner device.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\gpu --device cuda --actors 4 --round-episodes 32 --batch-size 256 --validation-games 16 --max-seconds 3600
```

Rollout workers remain on CPU. Replay stays on CPU.
The default replay tensors allocate about 293 MiB before Python and optimizer overhead.
Checkpoint compression reduces saved bytes, not the live replay allocation.
Use `--replay-capacity` to set the memory budget when starting a run.

More actors or a larger network do not guarantee better throughput.
Compare completed deals and updates over the same time budget before renting more compute.
Keep validation settings fixed when comparing runs.
The fixed validation workload can cost more than a short training round.

## Feed a learner from other machines

Actor machines can publish rollout batches through a shared filesystem or your own file-transfer tool.
Collect from frozen model files without opening a trainer run.

```powershell
.\.venv\Scripts\python -m src.cli.train --collect-batch runs\parallel\inbox\batch-vm01-000001.pt --initial-model runs\parallel\latest-model.pt --best-model runs\parallel\best-model.pt --opponent runs\parallel\initial-model.pt --players 1 2 3 4 --round-episodes 32 --actors 16 --seed 101 --start-episode 0
```

`--collect-batch` atomically writes compressed feature bytes and trajectory records to the destination.
It prints `episodes` and `trajectories` counts, then exits without creating `latest-state.pt`.
This mode does not require `--output-dir`.
`--initial-model` supplies the online weights and game settings.
`--best-model` supplies the reigning opponent and defaults to `--initial-model`.
Repeated `--opponent` flags supply additional snapshots, bounded by `--max-pool`.
`--round-episodes` sets the batch's deal count. Multiple `--actors` split those deals across processes.
`--start-episode` defaults to zero and sets the first deal index used for seeds and player-count rotation.
Give each actor machine a distinct `--seed` or disjoint episode ranges.
Use a unique destination filename for every batch and only one trainer per run directory.

Before each round, the trainer considers the first two `inbox/batch-*.pt` files in filename order.
If their total reaches `--round-episodes`, the trainer skips local collection and only learns.
Smaller batches reduce the local deal count. An empty inbox uses local rollouts.
Imported deals drive episode counters, learner updates, validation, and checkpoints.
A batch that exceeds the remaining `--max-episodes` budget stays in the inbox.
The trainer fills that round's remaining budget locally rather than splitting or discarding the batch.
To transfer a batch, copy it under a temporary name, then rename it to `batch-*.pt` inside the inbox.
The trainer deletes each batch after loading it, before learning. Keep a copy if you need recovery after a crash.

## Choose a model size or initial checkpoint

Start the smaller v5 network in a fresh directory.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\small --width 64 --blocks 2 --max-episodes 10000
```

The 64-by-2 model has 127,172 parameters.
The default 128-by-3 model has 673,412.
Both use the same encoding and browser inference implementation.

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
The CLI allows `--actors`, `--device`, `--torch-threads`, and `--checkpoint-every` changes on resume.
Other changed settings fail before training.
Encoding versions before 5 and trainer state versions before 12 require a fresh run.
Version 5 model checkpoints remain compatible at both model sizes.

## Inspect validation

By default, validation runs after each 1,000 completed deals.
`--validation-games 64` controls held-out solo games and multiplayer seed blocks.
Set it to zero only to disable validation for a bounded diagnostic run.

Solo validation measures cold greedy declarations when solo is in `--players`.
For every configured multiplayer count, validation rotates the learner through every seat on every seed.
It compares against `HeuristicPolicy` and the reigning `best-model.pt`.
A pairwise placement win earns 1, a tie earns 0.5, and a loss earns 0.
Other losing seats still compare their final deadwood penalties after a declaration.

The validation `score` is the unweighted mean of the solo rate and each fixed multiplayer opponent rate.
For the default counts, those are four rates, solo and `2p_heuristic`, `3p_heuristic`, and `4p_heuristic`.
The score lies in `[0, 1]`.
Training seeds are even and validation seeds are odd, so those deal seeds do not overlap.
Validation reuses fixed seeds and the same heuristic across the run.

Each `Xp_best` rate measures head-to-head placement credit against the reigning best for that player count.
`vs_best` is their unweighted mean and does not enter the fixed-anchor `score`.
With multiplayer enabled, the first validation selects the current network.
Later validations replace `best-model.pt` when `vs_best >= 0.51` and `score >= best_score - 0.02`.
An anchor score above `best_score + 0.02` also promotes when `vs_best >= 0.49`.
`best_score` retains the highest fixed-anchor score among promoted models.
Solo-only runs use the anchor-improvement rule after their first validation.
Every validation records `promoted` as a boolean.

On promotion, the trainer copies the best checkpoint to `opponents/best-<episode>.pt`.
Without promotion, `vs_best >= 0.45` saves the current network as `opponents/snap-<episode>.pt`.
Snapshot names pad the episode count to 12 digits.
The active pool retains the latest eight snapshots by default.
Older snapshot files remain on disk but no longer enter the active pool.
The current network can regress between validations.
The selected validation score is not a rating or proof of strength against unseen opponents.

Read `status.json` for the latest saved selection.
Read `metrics.jsonl` for each round's episode count, `epsilons`, trajectory count, combined learning loss, updates, samples, and optional validation.
`latest-state.pt` uses 64 MiB zlib chunks to save and restore replay tensors without NumPy.
Chunking supports tensors larger than 2 GiB without a full uncompressed copy for the codec.
With one actor, splitting at the same round boundary preserves model, optimizer, replay, and sampling exactly.
Changing the actor count can change batch composition and learning order.

## Evaluate on separate deals and publish browser weights

Use additional evaluation seeds rather than selecting against the same validation deals again.

```powershell
.\.venv\Scripts\python -m src.cli.evaluate --checkpoint runs\full21\best-model.pt --games 300 --seed 9000001 --warm-games 0 --output runs\full21\evaluation.json
.\.venv\Scripts\python -m src.cli.arena --model runs\full21\best-model.pt --model random --games 100 --seed 9100001 --max-turns 60 --output runs\full21\arena.json
```

Repeat `--model` for each additional arena seat.
Use `--model heuristic` for the fixed sparring policy, or supply another checkpoint for a frozen neural opponent.
Keep the game rules and turn cap the same across comparisons.
Optional `--warm-games` and `--sensitivity` test discard puzzles only.
Puzzle results do not replace cold-game evidence and never enter training.

Export weights into the browser's default paths.

```powershell
.\.venv\Scripts\python tools\weights_export.py --checkpoint runs\full21\best-model.pt
node --test tests/qnet.test.mjs tests/ai.test.mjs tests/table.test.mjs
```

The exporter writes the model and its matching golden cases together.
`--checkpoint` marks the export as trained and records the checkpoint and tensor hashes, not the input path.
It checks the portable forward pass against Torch.
The full Python suite also exports nonzero 128-by-3 weights and checks all 54 Node Q-values within `1e-5`.
Keep `web/papplu-model.json` and `tests/fixtures/qnet-golden.json` together.
`--bootstrap --seed 1729` generates deterministic untrained 128-by-3 weights.
Use that option for compatibility checks, not to publish playing-strength results.

Run the [verification commands](README.md#verification) before accepting a training-code change.

## Training CLI options

Run `python -m src.cli.train --help` for argument syntax.
Fresh runs use the defaults described above. Resume restores saved settings unless an allowed override is supplied.

| Option | Meaning |
| --- | --- |
| `-h`, `--help` | Show help without loading Torch. |
| `--output-dir` | Required run directory unless collecting a batch. Reuse it to resume. |
| `--initial-model` | Starting checkpoint for a fresh run. |
| `--collect-batch` | Write one rollout batch without opening a trainer run. |
| `--best-model` | Reigning opponent checkpoint for batch collection. Defaults to `--initial-model`. |
| `--start-episode` | First deal index in batch collection. Defaults to zero. |
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
| `--learning-rate` | AdamW learning rate. |
| `--lr-decay-updates` | Inverse-square-root update scale. Zero disables decay. |
| `--gamma` | Positive terminal-return discount factor. |
| `--margin-weight` | Nonnegative greedy-winning-step margin weight. Zero disables it. |
| `--epsilons` | Stationary exploration rates sampled once per learner seat. |
| `--self-play-share` | Probability that every seat uses the learner. |
| `--validate-every` | Completed-deal interval between validations. |
| `--validation-games` | Solo games and multiplayer seed blocks per validation. Zero disables validation. |
| `--seed` | Training seed. |
| `--device` | Learner Torch device. |
| `--torch-threads` | Learner Torch thread count. Workers always use one. |
| `--max-episodes` | Absolute completed-deal limit. |
| `--max-seconds` | Wall-clock budget, checked between rounds. |
