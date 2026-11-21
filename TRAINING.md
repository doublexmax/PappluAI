# Train a Papplu model

Use this guide to create a resumable self-play run and export its selected checkpoint to the browser.
The [model contract](README.md#model-and-training-contract) describes the network, encoding, and terminal utility.
No command here provisions Azure resources or spends cloud credit.

## Shipped checkpoint

The browser ships trained Azure checkpoint A1, `trackA_p1/best-model.pt`, in `web/papplu-model.json`.
Its `source.kind` is `checkpoint` and `source.trained` is `true`.
The `card_mlp` v5 network has 127,172 parameters at width 64 with two blocks.
Its version 5 encoding contains 1,275 float32 values.
The checkpoint SHA256 is `094bbe8946fc890e93de7a2ce77f0ad45ad8f8d24a38f2249ffb31fdef39882b`.
Training used a 16-vCPU Azure VM and repository commit `5d59de7`.

The final held-out evaluation locked A1 before gameplay.
Cold Python solo won 243 of 256 deals, or 94.9%, with a 95% Wilson interval of 91.5% to 97.0%.
The separate warm discard puzzles yielded 208 of 256 wins, or 81.25%.
For two, three, and four players, declaration rates were 99.2%, 99.2%, and 97.5% against Python Random.
Browser Casual rates were 83.3%, 71.7%, and 57.5%.
Against 2p Casual, Neural won 100/120 by declaration, or 83.3%.
It finished first in 101/120, or 84.2%, including one turn-cap penalty win.
Python Heuristic rates were 45.0%, 31.7%, and 16.7%.
Browser Tactical rates were 42.7%, 31.7%, and 25.0%.
Neural beats Random and Casual in these cells.
It plays near first-place parity against Tactical at three and four players, but trails at two players and against four-player Heuristic.

The fresh idle bundle benchmark measured 4.44 ms median and 4.84 ms p95 per `qValues` call over five 1,000-observation passes.
That call includes validation, encoding, and one forward pass, not a complete turn.
In the 18-worker browser run, Tactical's exact `minimumPenalty` and `outs1ply` searches averaged 4.76 to 4.98 seconds per move.
Neural averaged 30.59 to 43.04 ms under that load.
Do not compare those loaded move timings directly with the idle network benchmark.
See the [A1 held-out results](README.md#a1-held-out-results) for counts, seat balancing, and deal-cluster intervals.
The requested seeds are disjoint from this production run's training and selection, but earlier project models used them.

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
It uses `ModelSpec(64, 2)`, `gamma=0.98`, a 40,000-step replay, and batches of at most 128 steps.
Each completed deal schedules four learner updates by default. Use `--updates-per-episode` to change that count.
Positive-return declaration trajectories add a 0.10 legal-action margin to the Smooth-L1 objective.
`--margin-weight` defaults to 0.50. Set it to zero to train without the margin.
Epsilon decreases from 0.30 to 0.05 over 10,000 deals. Use `--epsilon-decay-episodes` to change the decay length.
The trainer holds epsilon fixed within each 16-deal round.

Every deal starts cold.
There are no solved-hand warm starts or curriculum stages.
On each multiplayer deal, self-play has probability 0.50.
Other deals place the learner in a random seat against `HeuristicPolicy`, `RandomPolicy`, and the active snapshot pool.
Each opponent seat chooses the heuristic with probability 0.50, then otherwise samples uniformly from all opponents, including the heuristic.
Every seat contributes its own terminal utility and trajectory in both modes.
Solo deals always use the learner.
The heuristic reads cached disjoint-packing features without exact penalty backtracking.
It is a sparring policy, not part of `PolicyNet` or the browser's greedy neural ranking.

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
The default replay allocates about 195 MiB before Python and optimizer overhead.
Checkpoint compression reduces saved bytes, not the live replay allocation.
Use `--replay-capacity` to set the memory budget when starting a run.

More actors or a larger network do not guarantee better throughput.
Compare completed deals and updates over the same time budget before renting more compute.
Keep validation settings fixed when comparing runs.
The fixed validation workload can cost more than a short training round.

## Choose a model size or initial checkpoint

Start a larger network in a fresh directory.

```powershell
.\.venv\Scripts\python -m src.cli.train --output-dir runs\wide --width 128 --blocks 3 --max-episodes 10000
```

The default 64-by-2 model has 127,172 parameters.
The 128-by-3 model has 673,412.
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
The CLI allows `--actors`, `--device`, and `--torch-threads` changes on resume.
The Python API also allows `checkpoint_every` changes.
Other changed settings fail before training.
Encoding versions before 5 and trainer state versions before 8 require a fresh run.

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
Later validations replace `best-model.pt` when `vs_best > 0.50` and `score >= best_score - 0.02`.
An anchor score above `best_score + 0.02` also promotes, regardless of `vs_best`.
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
Read `metrics.jsonl` for each round's episode count, epsilon, trajectory count, combined learning loss, updates, samples, and optional validation.
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
It checks the portable forward pass against Torch.
The full Python suite also exports nonzero 128-by-3 weights and checks all 54 Node Q-values within `1e-5`.
Keep `web/papplu-model.json` and `tests/fixtures/qnet-golden.json` together.
`--bootstrap --seed 1729` replaces the shipped trained weights with a deterministic untrained model.
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
| `--learning-rate` | AdamW learning rate. |
| `--gamma` | Terminal-return discount factor. |
| `--margin-weight` | Nonnegative winning-trajectory margin weight. Zero disables it. |
| `--epsilon-decay-episodes` | Deals over which epsilon decreases. Zero selects the final epsilon immediately. |
| `--self-play-share` | Probability that every seat uses the learner. |
| `--validate-every` | Completed-deal interval between validations. |
| `--validation-games` | Solo games and multiplayer seed blocks per validation. Zero disables validation. |
| `--seed` | Training seed. |
| `--device` | Learner Torch device. |
| `--torch-threads` | Learner Torch thread count. Workers always use one. |
| `--max-episodes` | Absolute completed-deal limit. |
| `--max-seconds` | Wall-clock budget, checked between rounds. |
