# Train a Papplu model

`GameTransformer` in [`src/model/`](src/model/) is the repository's only neural model. The [browser simulator](README.md#ai-tiers) uses the Tactical and Casual tiers without model weights or third-party dependencies.

Run the commands in this guide on the Azure compute host. Use Python 3.11 or newer and install the training dependencies in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
```

## Check a new run

Use the small model below to check training, collection, and resume before starting a full run. Two local actor processes collect alongside the learner. All processes run on the Azure host.

```bash
python -m src.cli.train --output-dir runs/smoke --max-turns 2 \
  --width 32 --layers 2 --heads 4 --ffn 64 --tables 4 \
  --batch-size 4 --batch-decisions 16 --min-buffer-decisions 16 \
  --eval-every 1 --eval-deals 1 --max-updates 1 --max-seconds 90
python -m src.cli.train --output-dir runs/smoke --max-updates 2 --max-seconds 90
```

Check `status.json` for `status: max_updates` and `version: 2`. The update limit is an absolute policy revision, not a count of additional updates. Reuse the directory to resume.

On resume you can change actor and table counts, thread counts, device, precision, `--checkpoint-every`, `--learning-rate`, `--eta`, and `--sync-every`. A new `--learning-rate` changes AdamW's base rate without resetting its moments or its progress through warmup. You can also change `--players`, `--curriculum-share`, `--curriculum-max-distance`, and `--champion-share`. The game, model, and other collection settings stay fixed. Restart standalone collectors if you change `--players`.

## Start a full run

```bash
python -m src.cli.train --output-dir runs/full21 --max-seconds 3600
```

The default `GameTransformer` has 2,175,224 parameters. Its four blocks use width 192, six rotary attention heads of width 32, RMSNorm, and a SwiGLU width of 512. It stores weights in checkpoint format 6 and uses eight-byte Schema 8 tokens. Each actor keeps a key and value cache per seat and batches decisions across its live tables.

Training defaults to three decks, 21 cards per hand, five required pure sequences, 60 turns per seat, and discard recycling. Two actor processes per learner rank cycle through one-, two-, three-, and four-seat games, with 64 tables per process by default.

Half of starts use the curriculum by default. Curriculum starts take the smaller of two uniform draws from zero through six in solo and from one through six in multiplayer. Distance zero starts with one extra card and ends after one discard. At positive distances, the dealer replaces that many cards in each seat's hand from a shared shoe. The encoder does not receive the curriculum distance. Standard validation and the arena use cold deals.

Use `--auto-curriculum` to start a fresh run at distance 1. Every `--eval-every` revisions, rank 0 runs 48 fixed solo deals at the current distance using the greedy neural policy. You can change the interval on resume. The table gives the minimum wins needed on those 48 deals. It does not estimate win rates in cold play.

| Distance | Curriculum share | Minimum win rate | Wins needed out of 48 |
| ---: | ---: | ---: | ---: |
| 1 | 88% | 8% | 4 |
| 2 | 82% | 6% | 3 |
| 3 | 75% | 5% | 3 |
| 4 | 70% | 4% | 2 |
| 6 | 60% | 2% | 1 |
| 9 | 50% | 2% | 1 |
| 14 | 40% | 2% | 1 |
| 21 (cold) | 45% | None | None |

The share controls how many collection deals use curriculum starts, with earlier distances still in the mix. At distance 21 the trainer stops advancing the curriculum and switches to cold checkpoint validation. All ranks publish the new plan on promotion. Existing matches keep the plan and weights they started with.

Self-play no longer uses 1-ply lookahead or learns from expert targets. Actors sample the masked `GameTransformer` policy with a 2% uniform mix over legal actions by default. The rollout records the action, its behavior log probability, the behavior value, and the hand potential. The learner applies a clipped importance-weighted policy gradient to sampled decisions. `--entropy-weight` defaults to zero.

`--champion-share` defaults to `0.25`. On that fraction of multiplayer deals the online learner plays against `best-model.pt`. One side gets a uniformly chosen seat and the other side gets the rest. The learner is equally likely to be either side. The champion samples without uniform exploration and contributes no training data. The collector loads a new champion for new tables when the file changes. If the file is absent, a non-recording copy of the current policy role fills its seats. Solo deals always use the learner.

The learner computes targets over the entire seat trajectory from the recorded behavior values. The V-trace target recurrence uses a unit ratio, discount `0.995`, trace cap `0.85`, and zero terminal bootstrap. The sampled policy-gradient terms use the actual clipped online-to-behavior likelihood ratio. The full-trajectory targets do not reevaluate the current policy on omitted steps.

The hand potential counts up to the required number of disjoint qualifying triples. Below quota, disjoint natural pairs count for one quarter of a triple. At quota, disjoint extra three-card melds count for one fifth of a triple. Neither rule uses card point values. The learner adds `0.995 * next_potential - current_potential` to each nonterminal reward and subtracts the last potential from the terminal utility. The terminal state's potential is zero. Discounted return differs from the unshaped return by the starting potential alone. The scoreboard still uses exact deadwood.

Solo pays `+1` for a declaration and `-1 + 0.60 * f` on a timeout. The fraction `f` is the share of the pure sequence quota recorded by `HandPenalty.qualifying_sequences`, capped at 1. A zero quota gives `f = 1`.

In multiplayer, let `n` be the number of seats and `scale = max_points * (n - 1)`. Let `net` be the seat's net deadwood payout. The declarer collects each opponent's deadwood. On a timeout, `net` is the mean table deadwood minus the seat's deadwood.

| Ending | Utility per seat |
| --- | --- |
| Declaration | `win_base + 0.35 * (net / scale)`, where `win_base` is `0.65` for the winner and `-0.65 / (n - 1)` for each other seat. |
| Timeout | `0.30 * (f - mean_f_of_other_seats) + 0.35 * (net / scale)`. |

The trainer uses payoffs that sum to exactly zero in floating point. A declaration win pays at least `0.65`. A timeout pays at most `0.475`. Equal hands expire at zero, which is better than a declaration loss at any supported player count.

The learner drops data more than `--max-policy-lag` revisions old. The default lag limit is 200. Each rank samples `--batch-size` trajectories with replacement from one player-count and length bucket. The default batch size is eight. A bucket's weight is its trajectory count. Each trajectory has equal weight in the policy, value, and anchor KL losses. For each trajectory the learner unrolls up to `--unroll-steps` decisions, 12 by default. For longer trajectories it keeps the terminal decision and samples the other positions without replacement. Inverse inclusion weights preserve the expected full-trajectory mean. `trained_decisions` counts the positions that reach the model, not all positions used to compute targets.

The default learning rate is `5e-4`. AdamW uses `learning_rate * min(1, (version + 1) / warmup_steps)` before each update when `--warmup-steps` is positive. The default is 50 updates. Use `--warmup-steps 0` for the full rate from the first update. A run started with `--initial-model` starts at revision zero. A resume keeps the saved revision.

The default anchor weight (`--eta`) is `0.01`. The frozen anchor evaluates the same sampled positions. The default refresh interval (`--anchor-every`) is 2,000 revisions. Training defaults to CPU with BF16 autocast, FP32 master weights, and FP32 loss math. Use `--precision fp32` to disable autocast.

| File | Contents |
| --- | --- |
| `latest-model.pt` | Published weights, game, policy revision, and collection plan. |
| `best-model.pt` | Cold validation selection or the last model to clear a mastery threshold. Before a promotion, it holds the initial model. |
| `latest-state.pt` | Learner and anchor weights, optimizer, progress, and sampler state. |
| `inbox/batch-*.pt` | Complete byte-token rollout batches awaiting ingestion. |
| `metrics.jsonl` and `status.json` | Update metrics and run status, including collected and trained decisions, throughput, importance ratio, KL, buffer depth, and inbox depth. |

The learner does not save its in-memory rollout buffer. On resume, it refills the buffer from actors and any compatible files still in the inbox. Rollout format 6 drops the expert targets and forced-move flags. Trainer state version 19 stores the collection plan and warmup length. Use `--initial-model` in a new run directory to reuse older Schema 8 weights. Older trainer states and rollout batches from earlier runs are incompatible.

### Schema 8 tokens

The append-only prefix contains six rule tokens, the joker indicator, 22 slots for the observer's initial hand, and visible events. It does not encode unseen cards, meld estimates, or point scores.

The 30-token snapshot has one `ACT`, one `TABLE_STATE`, six `SEAT` slots, and 22 `HAND` slots. Duplicate faces share one hand token with a copy count. The encoder leaves unused hand slots zeroed.

| Byte | Meaning |
| ---: | --- |
| 0 | Token kind. Zero marks padding. |
| 1 | `1 + relative_seat` for events and seat tokens. `ACT` uses `1`. `TABLE_STATE` uses `7 + phase.value`. Hand tokens use `10 + copies`. Rule tokens use `20 + rule_value`. |
| 2, 3, 4 | One-based face, printed suit, and printed rank. An absent face uses `53`, `5`, and `14`. |
| 5 | Events use the one-based offered face or `53` if absent. `SEAT` uses `60 + hand_size` for each present seat. `TABLE_STATE` uses `83 + 2 * next_player + int(active)` for the observer. `ACT` uses `100 + min(discard_pile_size, 150)`. |
| 6 | Events, `ACT`, and present seat tokens use `1 + min(turns_remaining, 120)`. `TABLE_STATE` uses `130 + min(stock_remaining // 2, 120)`. |
| 7 | `1 + discard_action(face)` on `HAND` tokens only. |

Zero means unused in each field. The model encodes the prefix with causal attention. Each decision reads its snapshot and only the prefix available at that decision. The `ACT` token supplies 54 base policy logits and a scalar value. For each nonzero pointer byte, the shared linear pointer head adds that token's score to the named action logit before legal masking. The network does not import the game rules.

Schema 8 supports up to 21 cards before drawing, four decks, and a configured budget of 235 turns per seat. The header stores the full turn budget. Start in a new directory if the saved checkpoint or trainer state uses an incompatible schema.

## Run several CPU learners

On a 20-vCPU host, start two Gloo ranks with eight learner threads and two single-threaded actors each:

```bash
torchrun --standalone --nproc_per_node=2 -m src.cli.train \
  --output-dir runs/multi --torch-threads 8 --actors 2 --actor-threads 1 \
  --batch-size 8 --max-seconds 720
```

For multiple hosts, replace `--standalone` with the same `--nnodes` and private `--rdzv-endpoint` on every host. Use `--rdzv-backend c10d` and the same unique `--rdzv-id`. Use the same model and training flags. Do not expose the Gloo or rendezvous ports to untrusted hosts.

Each rank waits for local replay. The ranks weight buckets by their combined trajectory counts, even when only one rank has the bucket. A rank with no local entries in the chosen bucket samples from its nearest available bucket. With `--sync-every 1`, ranks average gradients before clipping and stepping AdamW. With `--sync-every 8`, each rank takes eight local AdamW steps before ranks average the FP32 model parameters. This Post-Local AdamW mode does not average the optimizer moments. Metrics sum decision counts across ranks and advance the policy revision by eight per full synchronization interval.

Actors use `rank-N/inbox` and read weights from local disk. Rank 0 writes the canonical checkpoints, status, and metrics and runs evaluation. Other ranks publish `rank-N/latest-model.pt` for their actors. For resume, keep `latest-state.pt` and `best-model.pt` on rank 0. Rank 0 shares the saved startup state over Gloo, so no rollout or checkpoint files need to move between hosts during training.

## Feed a learner from other machines

Set `--actors 0` on the learner if all collectors run elsewhere. Copy `latest-model.pt` to each actor host with an atomic rename. Start each collector with a distinct worker ID.

```bash
python -m src.cli.train --collect actor-outbox --model latest-model.pt \
  --worker 100 --tables 64 --max-batches 1
```

Leave off `--max-batches` for continuous collection. Refresh the actor's model file atomically. Live deals keep their admission weights and caches. Copy completed `batch-*.pt` files to temporary names in the learner inbox, then rename each file to its final name. Transfer each batch once and clear it from the outbox after transfer. A continuous collector pauses while eight or more batches for its worker ID remain in its outbox. The learner rejects incompatible files and discards stale batches.

## Evaluate a checkpoint

```bash
python -m src.cli.evaluate --checkpoint runs/full21/best-model.pt \
  --games 300 --warm-games 0 --output runs/full21/evaluation.json
python -m src.cli.arena --model runs/full21/best-model.pt --model random \
  --games 120 --output runs/full21/arena.json
```

`src.cli.evaluate` pairs the model and random policy on the same solo seeds. It reports discard puzzles separately when `--warm-games` is positive. In single-match mode, `src.cli.arena` rotates the lineup between deals and plays each seed once. It reports each deal's winner, utilities, and net deadwood points. Supply one checkpoint or `random` per seat, for up to six seats. Checkpoints must use the same game settings. Both commands pick the highest legal model logit without lookahead.

## Rate checkpoints on cold deals

```bash
python -m src.cli.arena --league \
  --model checkpoints/rev0-model.pt --label rev0 \
  --model runs/full21/best-model.pt --label best \
  --players 2 3 4 --games 48 --seed 970000 --precision bf16 \
  --workers 16 --output runs/full21/league-970000.json
```

List at least two `GameTransformer` checkpoints with matching game settings. The first checkpoint anchors the overall Elo and each player-count Elo at `1000.0` unless you change `--anchor-elo`. The JSON records each checkpoint's SHA-256, the matchup metrics, and the standings. Use `--workers` to run matchups in parallel on a CPU host.

For each checkpoint pair and seed, two-seat matches swap the checkpoints between seats. At larger tables, every seat takes a turn as the only seat running one checkpoint. A mirror runs the other checkpoint in that seat and swaps all remaining seats. Each comparison between seats running different checkpoints counts as one point for a higher finish, half for a tie, and zero for a lower finish. The Bradley-Terry fit uses all of these comparisons, so the overall Elo gives more weight to larger tables.

A declaration pays the winner the sum of everyone else's deadwood. The other seats each pay their own deadwood. On expiry, each seat scores the table mean deadwood minus its own. The payments sum to zero. Standings report net points and declaration rate per seat. Mean winning turn uses declarations only, and losing deadwood uses non-winning seats only.

When auto-curriculum is off or the distance reaches the hand size, `Trainer.evaluate()` compares the learner to `best-model.pt` at the configured multiplayer counts. Solo-only runs compare declarations and then deadwood on identical seeds. The trainer promotes a checkpoint only if the mean placement score is at least `0.5 + --promote-margin`, mean net points are nonnegative, and no player count scores below `0.5 - --promote-margin`. The default margin is `0.02`. These checks do not use a reference bot.

### Five-checkpoint cold league on seed 970,000

The league ran on `vm-papplu-v4-w128`, a 16-vCPU `Standard_D16als_v6` in West US 2. It used BF16, 64 live tables per matchup, and the checkpoint settings of 21 cards and 60 turns per seat. All 10 checkpoint pairs played 48 seeds per player count. Mirrors produced 7,680 matches and 18,240 cross-policy seat comparisons. `rev0` anchors the overall and player-count ratings at 1,000.

| Checkpoint | Overall Elo | 2p Elo | 3p Elo | 4p Elo | Placement score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `balanced` | 1430.6 | 1365.7 | 1424.7 | 1440.2 | 75.42% |
| `mastery` | 1413.1 | 1349.8 | 1419.1 | 1416.4 | 73.20% |
| `phase4` | 1322.4 | 1228.2 | 1331.3 | 1326.8 | 61.09% |
| `v8-final` | 1010.7 | 978.8 | 1014.9 | 1011.5 | 20.74% |
| `rev0` | 1000.0 | 1000.0 | 1000.0 | 1000.0 | 19.55% |

The next table gives net deadwood per seat. Declarations count per seat, and the mean winning turn counts only declared wins. Losing deadwood includes every non-winning seat, even when the deal expires.

| Checkpoint | Overall net | 2p net | 3p net | 4p net | Declaration rate | Mean winning turn | Losing deadwood |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced` | +23.22 | +16.90 | +19.78 | +25.94 | 16.11% | 36.87 | 51.60 |
| `mastery` | +20.00 | +19.32 | +17.69 | +21.38 | 15.20% | 37.12 | 52.32 |
| `phase4` | +4.76 | +3.99 | +5.71 | +4.31 | 7.93% | 42.13 | 56.43 |
| `v8-final` | -21.92 | -20.71 | -19.47 | -23.46 | 0.23% | 22.00 | 100.98 |
| `rev0` | -26.05 | -19.50 | -23.71 | -28.18 | 0.14% | 32.29 | 100.62 |

These ratings describe this seed and checkpoint set. They do not measure results against the browser AI or on curriculum deals.

### Single-VM competitive proof run

The learner ran three 20-minute stages on `vm-papplu-v4-w128`, a 16-vCPU Azure VM. It used 11 single-threaded actors, four learner threads, 32 tables per actor, and BF16.
The stages resumed the same optimizer at base rates `6e-5`, `4e-5`, and `2e-5`. The curriculum share dropped from `0.25` to `0.12` to `0.05`. All three stages used a champion share of `0.25`.

The selection league used seed `600_000` with 24 deals per pair and player count. Its 2,304 matches used two, three, and four seats. The prior checkpoint anchored Elo at 1,000.
Selection ranks by overall Elo, then net deadwood per seat.

| Checkpoint | Overall Elo | Net deadwood per seat | Declarations per seat |
| --- | ---: | ---: | ---: |
| gate-c3 | 1086.6 | +5.71 | 12.04% |
| gate-c2 | 1086.2 | +4.82 | 11.88% |
| gate-c1 | 1056.3 | -0.68 | 11.11% |
| AdamW resume | 1000.0 | -9.86 | 8.02% |

The promoted weights and matching version 19 AdamW state are `locked-gt.pt` and `locked-gt-state.pt`.

After the lock, the seed `980_000` holdout used 48 cold deals per pair and player count. It produced 7,680 mirrored matches at 0-ply. The untrained checkpoint anchors Elo at 1,000.

| Training milestone | Overall Elo | 2p Elo | 3p Elo | 4p Elo | Net deadwood per seat | Declarations per seat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untrained (rev 0) | 1000.0 | 1000.0 | 1000.0 | 1000.0 | -38.02 | 0.15% |
| Potential and subsampling | 1324.1 | 1254.7 | 1348.1 | 1319.4 | -9.72 | 6.81% |
| Cluster cold play | 1422.1 | 1383.7 | 1445.2 | 1415.2 | +10.87 | 14.16% |
| AdamW resume | 1434.0 | 1388.0 | 1466.2 | 1423.3 | +12.13 | 15.01% |
| Pure 0-ply zero-sum | 1512.3 | 1464.3 | 1540.9 | 1503.7 | +24.73 | 20.45% |

The same two locked and prior saves also played 500 paired, cold, 21-card solo deals at 0-ply. The solo deals use seeds `980_000` through `980_499`, separate from the league fixtures.

| Checkpoint | Declarations | Win rate | Mean utility |
| --- | ---: | ---: | ---: |
| Locked checkpoint | 249/500 | 49.80% | +0.2730 |
| Prior checkpoint | 197/500 | 39.40% | +0.1067 |


### Two-learner cluster and final cold holdouts

Two learners and four actors ran on six 16-vCPU Azure VMs. The 96-vCPU cluster resumed at revision 1,022.
The three stages used 0-ply self-play, with no search at inference.

| Revision range | Trained decisions | Trained decisions/s | Dropped batches |
| --- | ---: | ---: | ---: |
| 1,022 to 1,526 | 769,472 | 549.52 | 0 |
| 1,526 to 1,894 | 565,244 | 548.09 | 0 |
| 1,894 to 2,294 | 612,868 | 599.47 | 0 |

The cluster trained on 1,947,584 decisions at 563.88 decisions/s over the summed learner-stage elapsed time.
Across 234 learner samples, CPU busy was 79.98% mean and 98.12% median.
Across 468 actor samples, CPU busy was 98.27% mean and 100.00% median.

#### Selection on seed 700,000

Only `vm-papplu-v4-w128` ran evaluation. It used 16 workers and BF16 at 0-ply.
The built-in league ran 2,304 cold matches with 24 mirrored deals per pair and player count.
It ranked overall Elo first and net deadwood per seat second. An exact tie kept `gate-0ply`.
The gate anchors selection Elo at 1,000.

| Checkpoint | Overall Elo | Net deadwood per seat | Declarations per seat |
| --- | ---: | ---: | ---: |
| c5-s2 | 1028.2 | +0.67 | 12.81% |
| c5-s3 | 1024.1 | +1.86 | 12.71% |
| c5-s1 | 1017.1 | +0.15 | 11.57% |
| gate-0ply | 1000.0 | -2.69 | 9.52% |

Selection locked `c5-s2` at revision 1,894 before either holdout. The best cluster checkpoint replaced the single-VM gate.
The matching AdamW state uses format 19. Neither holdout could change the lock.

#### Six-milestone holdout on seed 1,200,000

The holdout played 8,640 mirrored matches over 36 cold deals per pair and player count.
Its deal seeds did not overlap selection or the recorded earlier evaluations.
Untrained weights anchor each rating at 1,000. Positive net deadwood means the checkpoint gained points.
Declarations per seat counts winning declarations divided by seat appearances.

| Training milestone | Overall Elo | 2p Elo | 3p Elo | 4p Elo | Net deadwood per seat | Declarations per seat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untrained (rev 0) | 1000.0 | 1000.0 | 1000.0 | 1000.0 | -42.11 | 0.16% |
| Potential and subsampling | 1344.3 | 1273.4 | 1343.6 | 1351.3 | -12.37 | 6.98% |
| Cluster cold play | 1439.0 | 1450.8 | 1412.2 | 1452.8 | +4.81 | 12.74% |
| AdamW resume | 1458.9 | 1435.2 | 1433.8 | 1474.7 | +7.35 | 13.79% |
| Single-VM 0-ply gate | 1526.8 | 1522.5 | 1510.4 | 1536.7 | +19.83 | 19.12% |
| Cluster 0-ply zero-sum | 1566.0 | 1526.6 | 1549.0 | 1579.1 | +22.49 | 20.86% |

#### Paired solo holdout on seed 1,500,000

All three checkpoints played the same 500 cold 21-card deals, seeds `1_500_000` through `1_500_499`.
A win is a valid declaration within 60 turns. The 95% intervals resample paired deals, not independent training runs.

| Checkpoint | Declarations | Win rate | Mean utility |
| --- | ---: | ---: | ---: |
| Cluster 0-ply zero-sum | 289/500 | 57.80% | +0.3924 |
| Single-VM 0-ply gate | 255/500 | 51.00% | +0.2941 |
| AdamW resume | 200/500 | 40.00% | +0.1146 |

| Paired comparison | Win rate difference (pp) | 95% interval (pp) |
| --- | ---: | ---: |
| Cluster 0-ply zero-sum minus Single-VM 0-ply gate | +6.80 | +2.20 to +11.20 |
| Cluster 0-ply zero-sum minus AdamW resume | +17.80 | +13.20 to +22.60 |
| Single-VM 0-ply gate minus AdamW resume | +11.00 | +6.00 to +16.00 |

## Azure results

The results below predate the checkpoint league. Historical search-assisted runs used a policy option that has since been removed.

The Schema 8 run used commit `0f00513` on one 16-vCPU `Standard_D16als_v6` in West US 2. Two Gloo ranks ran on that VM with BF16 and `--sync-every 1`. Each rank used seven learner threads and one single-threaded actor with 32 tables. Each update sampled 12 trajectories per rank.

The learner loaded weights from revision 826 and started a new AdamW state at revision 0. Check-In #12 and Final resumed from the preceding phase. The run used 1-ply lookahead, `--eta 0.01`, `--entropy-weight 0.002`, and `--anchor-every 400` throughout.

| Stage | Budget (s) | Players | Curriculum share | Max distance | Learning rate | Revision | Collected decisions/s | Trained decisions/s | Mean CPU busy | Mean / peak sampled RAM (GiB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Check-In #11 | 900 | 1, 2 | 95% | 2 | `0.0006` | 439 | 142.29 | 720.55 | 85.8% | 9.62 / 11.90 |
| Check-In #12 | 1500 | 1, 2, 3, 4 | 70% | 6 | `0.0004` | 842 | 147.95 | 607.34 | 86.7% | 11.57 / 16.02 |
| Final | 1500 | 1, 2, 3, 4 | 35% | 12 | `0.0002` | 1,184 | 217.04 | 594.95 | 86.6% | 11.79 / 15.03 |

Decision rates count both ranks over each phase. The three rows are phases of one run, not independent trials.

### Check-ins and lock at seed 600,000

Each checkpoint used seed `600_000` for 64 deals per solo case and both seats on 48 cold deals against `RandomPolicy`. The distance-zero check-in ends after the first discard. The pipeline averages declaration rates over six cases at each of two search depths. It breaks ties by mean cold declaration rate, then by mean utility. Exact ties favor the earlier checkpoint.

| Stage | Depth | d=0 | d=1 | d=2 | d=4 | Cold solo | Cold 2p vs RandomPolicy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Check-In #11 | 0 | 17/64 | 2/64 | 0/64 | 1/64 | 0/64 | 0/96 |
| Check-In #11 | 1 | 64/64 | 19/64 | 4/64 | 5/64 | 1/64 | 0/96 |
| Check-In #12 | 0 | 17/64 | 3/64 | 1/64 | 1/64 | 0/64 | 0/96 |
| Check-In #12 | 1 | 64/64 | 18/64 | 6/64 | 8/64 | 0/64 | 0/96 |
| Final | 0 | 18/64 | 2/64 | 2/64 | 3/64 | 0/64 | 0/96 |
| Final | 1 | 64/64 | 21/64 | 8/64 | 9/64 | 0/64 | 0/96 |

Final led at 16.54% mean declaration rate. The pipeline locked its revision 1,184 weights before opening held-out seed `930_000`.

The SHA-256 of `locked-gt-v8-final.pt` is `5039c05e013e80c8af4e80ebb344c686421593c022c27d825e70a6a96b20486a`.

### Held-out solo at seed 930,000

Across the solo and multiplayer cases, 16 workers completed 11,800 games and 2,925,096 decisions in 1,126.65 seconds with no evaluation errors. Both the check-ins and the holdout used BF16 inference and a 60-turn cap per seat. `Rev 0` is the fixed untrained Schema 8 baseline, not the weights used to start training. The 0-ply models choose from the masked policy alone. The 1-ply models add terminal-win search.

All five solo policies used the same deals within each start condition. On held-out distance-zero deals, a miss on the first discard leaves the game running until a declaration or the turn limit. The first-discard column counts only declarations on that first move. The other solo cases start in the draw phase.

| Policy | Start | First discard | Declarations / games | Mean qualifying sequences | Mean utility |
| --- | --- | ---: | ---: | ---: | ---: |
| Trained 0-ply | d=0 | 44/128 | 47/128 | 2.9219 | -0.160625 |
| Trained 0-ply | d=1 | n/a | 5/128 | 1.5547 | -0.759687 |
| Trained 0-ply | d=2 | n/a | 6/128 | 1.4844 | -0.757188 |
| Trained 0-ply | d=4 | n/a | 3/128 | 1.4141 | -0.797500 |
| Trained 0-ply | d=6 | n/a | 0/128 | 1.2812 | -0.846250 |
| Trained 0-ply | Cold 21-card | n/a | 0/256 | 1.3633 | -0.836406 |
| Trained 1-ply | d=0 | 128/128 | 128/128 | 5.7344 | 1.000000 |
| Trained 1-ply | d=1 | n/a | 33/128 | 2.4219 | -0.357812 |
| Trained 1-ply | d=2 | n/a | 25/128 | 2.0234 | -0.485625 |
| Trained 1-ply | d=4 | n/a | 8/128 | 1.5938 | -0.722187 |
| Trained 1-ply | d=6 | n/a | 1/128 | 1.3203 | -0.830625 |
| Trained 1-ply | Cold 21-card | n/a | 1/256 | 1.3711 | -0.830000 |
| RandomPolicy | d=0 | 31/128 | 33/128 | 2.7031 | -0.330625 |
| RandomPolicy | d=1 | n/a | 4/128 | 1.9219 | -0.726562 |
| RandomPolicy | d=2 | n/a | 2/128 | 1.6875 | -0.776563 |
| RandomPolicy | d=4 | n/a | 1/128 | 1.8359 | -0.768750 |
| RandomPolicy | d=6 | n/a | 0/128 | 1.5859 | -0.809688 |
| RandomPolicy | Cold 21-card | n/a | 0/256 | 1.6172 | -0.805937 |
| Rev 0, 0-ply | d=0 | 30/128 | 31/128 | 2.4453 | -0.380625 |
| Rev 0, 0-ply | d=1 | n/a | 2/128 | 1.5859 | -0.788750 |
| Rev 0, 0-ply | d=2 | n/a | 2/128 | 1.7031 | -0.774687 |
| Rev 0, 0-ply | d=4 | n/a | 0/128 | 1.5703 | -0.811562 |
| Rev 0, 0-ply | d=6 | n/a | 0/128 | 1.4922 | -0.820937 |
| Rev 0, 0-ply | Cold 21-card | n/a | 0/256 | 1.6133 | -0.806406 |
| Rev 0, 1-ply | d=0 | 128/128 | 128/128 | 5.7344 | 1.000000 |
| Rev 0, 1-ply | d=1 | n/a | 37/128 | 2.5781 | -0.297187 |
| Rev 0, 1-ply | d=2 | n/a | 24/128 | 2.2969 | -0.464688 |
| Rev 0, 1-ply | d=4 | n/a | 10/128 | 1.8594 | -0.668438 |
| Rev 0, 1-ply | d=6 | n/a | 3/128 | 1.5781 | -0.777813 |
| Rev 0, 1-ply | Cold 21-card | n/a | 1/256 | 1.6172 | -0.800469 |

### Held-out 0-ply multiplayer at seed 930,000

In both multiplayer tables, the trained policy plays each seat once per deal. All other seats use the named opponent. The opponent declaration count combines those seats. First-place credit goes to the declarer, or to the lowest deadwood at the turn limit. Tied first places split the credit. Utility does not use point rank.

| Start | Seats | Opponent | Declarations / games | Opponent declarations | First-place credit | Mean utility |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| d=2 | 2 | RandomPolicy | 7/120 | 0 | 68.0000/120 (56.67%) | -0.411000 |
| d=2 | 2 | Rev 0, 0-ply | 7/120 | 4 | 68.5000/120 (57.08%) | -0.426667 |
| d=2 | 2 | Rev 0, 1-ply | 7/120 | 27 | 58.5000/120 (48.75%) | -0.518000 |
| Cold | 2 | RandomPolicy | 0/200 | 0 | 86.0000/200 (43.00%) | -0.525500 |
| Cold | 2 | Rev 0, 0-ply | 0/200 | 0 | 116.0000/200 (58.00%) | -0.487700 |
| Cold | 2 | Rev 0, 1-ply | 0/200 | 0 | 116.0000/200 (58.00%) | -0.487700 |
| d=2 | 3 | RandomPolicy | 6/180 | 4 | 62.5000/180 (34.72%) | -0.460833 |
| d=2 | 3 | Rev 0, 0-ply | 9/180 | 7 | 68.0000/180 (37.78%) | -0.438500 |
| d=2 | 3 | Rev 0, 1-ply | 7/180 | 61 | 56.0000/180 (31.11%) | -0.443500 |
| Cold | 3 | RandomPolicy | 0/240 | 1 | 110.5000/240 (46.04%) | -0.502500 |
| Cold | 3 | Rev 0, 0-ply | 0/240 | 1 | 86.0000/240 (35.83%) | -0.490750 |
| Cold | 3 | Rev 0, 1-ply | 0/240 | 3 | 86.0000/240 (35.83%) | -0.490500 |
| d=2 | 4 | RandomPolicy | 9/240 | 13 | 88.0000/240 (36.67%) | -0.435389 |
| d=2 | 4 | Rev 0, 0-ply | 11/240 | 17 | 56.0000/240 (23.33%) | -0.429444 |
| d=2 | 4 | Rev 0, 1-ply | 10/240 | 117 | 34.0000/240 (14.17%) | -0.359000 |
| Cold | 4 | RandomPolicy | 0/240 | 0 | 73.0000/240 (30.42%) | -0.510250 |
| Cold | 4 | Rev 0, 0-ply | 0/240 | 0 | 58.5000/240 (24.38%) | -0.489917 |
| Cold | 4 | Rev 0, 1-ply | 0/240 | 3 | 58.5000/240 (24.38%) | -0.488417 |

### Held-out 1-ply multiplayer at seed 930,000

| Start | Seats | Opponent | Declarations / games | Opponent declarations | First-place credit | Mean utility |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| d=2 | 2 | RandomPolicy | 24/120 | 0 | 78.0000/120 (65.00%) | -0.193500 |
| d=2 | 2 | Rev 0, 0-ply | 25/120 | 4 | 71.5000/120 (59.58%) | -0.211167 |
| d=2 | 2 | Rev 0, 1-ply | 23/120 | 26 | 62.5000/120 (52.08%) | -0.321833 |
| Cold | 2 | RandomPolicy | 0/200 | 0 | 86.0000/200 (43.00%) | -0.525500 |
| Cold | 2 | Rev 0, 0-ply | 2/200 | 0 | 116.0000/200 (58.00%) | -0.473900 |
| Cold | 2 | Rev 0, 1-ply | 2/200 | 0 | 116.0000/200 (58.00%) | -0.473900 |
| d=2 | 3 | RandomPolicy | 31/180 | 4 | 80.5000/180 (44.72%) | -0.250667 |
| d=2 | 3 | Rev 0, 0-ply | 38/180 | 5 | 85.0000/180 (47.22%) | -0.196500 |
| d=2 | 3 | Rev 0, 1-ply | 30/180 | 52 | 71.0000/180 (39.44%) | -0.254333 |
| Cold | 3 | RandomPolicy | 1/240 | 1 | 110.5000/240 (46.04%) | -0.496500 |
| Cold | 3 | Rev 0, 0-ply | 4/240 | 1 | 86.0000/240 (35.83%) | -0.466625 |
| Cold | 3 | Rev 0, 1-ply | 4/240 | 3 | 86.0000/240 (35.83%) | -0.466375 |
| d=2 | 4 | RandomPolicy | 30/240 | 10 | 102.0000/240 (42.50%) | -0.307139 |
| d=2 | 4 | Rev 0, 0-ply | 37/240 | 16 | 76.0000/240 (31.67%) | -0.267139 |
| d=2 | 4 | Rev 0, 1-ply | 29/240 | 111 | 50.0000/240 (20.83%) | -0.243333 |
| Cold | 4 | RandomPolicy | 1/240 | 0 | 73.0000/240 (30.42%) | -0.503917 |
| Cold | 4 | Rev 0, 0-ply | 0/240 | 0 | 58.5000/240 (24.38%) | -0.489917 |
| Cold | 4 | Rev 0, 1-ply | 0/240 | 3 | 58.5000/240 (24.38%) | -0.488417 |

## Azure mastery run and paired holdout

Two BF16 Gloo ranks on a 16-vCPU `Standard_D16als_v6` started from `locked-gt-v8-final.pt`.
Each rank used six learner threads, two actors, and 32 tables per actor.
Check-ins B and C each resumed the saved AdamW state for 1,200 seconds.
The earlier run won 4 of 48 distance-one mastery deals at revision 800. That meets the calibrated 8% threshold but missed the previous 45% threshold. Check-in B resumed at distance 2 with an 82% curriculum share.
B checked mastery every 300 revisions. C kept that interval and used a learning rate of `2.5e-4`.

The 12-step learner rose from 0.1452 to 1.8922 updates per second, a 13.03x speedup.
The medians use five alternating updates per side on the same 24 archived 120-step trajectories with six threads.
The full learner trained on 2,880 decisions per update. The sampled learner trained on 288.
The profiles put most of the time in the online and anchor unrolls and the backward pass.
The timing excludes rollout collection and gradient synchronization.
`hand_potential` shapes the full-trajectory targets without adding a model token.
Each reward adds `discount * next_potential - current_potential`. The terminal discount is zero.
The learner samples decision indices, computes full-trajectory V-trace targets, and gathers those sampled targets before the online and anchor unrolls.

| `hand_potential` input | Median µs per call | Calls |
| --- | ---: | ---: |
| cold | 17.93 | 10,000 |
| distance 1 | 413.52 | 10,000 |
| distance 0 | 103.16 | 10,000 |

### Select the checkpoint on seed 600,000

Each depth used 64 solo deals per start and both seats of 48 cold two-player deals.
The distance-zero case ends after the first discard.
Selection ranks mean declaration rate first, then cold declaration rate and mean utility.

| Checkpoint | Revision | Active distance | Automatic promotions to date | 0-ply declarations | 1-ply declarations |
| --- | ---: | ---: | --- | ---: | ---: |
| `phase4-c1` | 851 | 1 | None | 34/480 | 125/480 |
| `phase4-c2` | 1640 | 4 | 2 to 3 at rev 900 (4/48), 3 to 4 at rev 1200 (5/48) | 56/480 | 184/480 |
| `phase4-c3` | 2251 | 6 | 2 to 3 at rev 900 (4/48), 3 to 4 at rev 1200 (5/48), 4 to 6 at rev 2100 (4/48) | 73/480 | 229/480 |

The run locked `phase4-c3` as `locked-gt-phase4.pt` before it opened seed `940_000`.
The locked checkpoint SHA-256 is `01ab8d38bc42f103be24adb5b5f1d172c1d76f579aae52649ba97f7588c41bb8`.

### Compare on held-out seed 940,000

Sixteen workers completed 14,872 games.
Policy comparisons use identical deals and focal seats, with a 60-turn cap per seat.
The tables round means. The proof bundle's `holdout.json` includes full precision and per-deal outcomes.

In the solo table, each cell shows the locked model first and the previous winner second.

| Solo start | Depth | Declarations | Mean qualifying sequences | Mean deadwood | Mean utility | Deals per policy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d0_first` | 0 | 37 / 40 | 5.18 / 5.14 | 13.2188 / 12.4531 | -0.008438 / 0.026250 | 128 |
| `d0_first` | 1 | 128 / 128 | 5.74 / 5.74 | 0.0000 / 0.0000 | 1.000000 / 1.000000 | 128 |
| `d0_60turn` | 0 | 53 / 42 | 4.46 / 2.69 | 28.5547 / 70.8594 | 0.092500 / -0.235000 | 128 |
| `d0_60turn` | 1 | 128 / 128 | 5.74 / 5.74 | 0.0000 / 0.0000 | 1.000000 / 1.000000 | 128 |
| `d1` | 0 | 20 / 6 | 3.82 / 1.34 | 43.7812 / 106.4219 | -0.325625 / -0.774062 | 128 |
| `d1` | 1 | 65 / 33 | 4.37 / 2.25 | 28.7656 / 82.1953 | 0.217188 / -0.377500 | 128 |
| `d2` | 0 | 21 / 9 | 3.93 / 1.53 | 41.6406 / 99.4531 | -0.302500 / -0.717812 | 128 |
| `d2` | 1 | 63 / 29 | 4.38 / 2.21 | 26.7188 / 81.9375 | 0.205625 / -0.423125 | 128 |
| `d4` | 0 | 17 / 4 | 3.96 / 1.48 | 42.0938 / 101.5859 | -0.341562 / -0.778125 | 128 |
| `d4` | 1 | 53 / 9 | 4.27 / 1.61 | 31.0859 / 97.8828 | 0.085937 / -0.709375 | 128 |
| `d6` | 0 | 17 / 0 | 3.89 / 1.26 | 45.5625 / 107.8594 | -0.350000 / -0.849063 | 128 |
| `d6` | 1 | 49 / 3 | 4.12 / 1.35 | 35.8047 / 105.4922 | 0.024375 / -0.805000 | 128 |
| `d8` | 0 | 13 / 0 | 3.90 / 1.16 | 41.2812 / 109.8672 | -0.390000 / -0.860313 | 128 |
| `d8` | 1 | 41 / 2 | 4.12 / 1.24 | 33.0078 / 107.9219 | -0.062188 / -0.829063 | 128 |
| `solo_cold` | 0 | 28 / 0 | 3.91 / 1.32 | 43.1250 / 105.4102 | -0.382344 / -0.842031 | 256 |
| `solo_cold` | 1 | 88 / 1 | 4.09 / 1.34 | 35.2383 / 104.9102 | -0.032188 / -0.834219 | 256 |

In the multiplayer table, one focal policy faces random opponents. Each distance-two deal rotates through all seats.
At the turn limit, tied first places split the credit. Each cell shows the locked model first.

| Multiplayer start | Depth | Declarations | First-place credit | Mean deadwood | Games per policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2p_d2` | 0 | 30 / 3 | 111.00 / 59.00 | 33.39 / 99.25 | 120 |
| `2p_d2` | 1 | 77 / 25 | 115.00 / 74.00 | 18.53 / 79.57 | 120 |
| `2p_cold` | 0 | 24 / 1 | 194.00 / 96.00 | 36.57 / 107.87 | 200 |
| `2p_cold` | 1 | 92 / 1 | 195.00 / 96.00 | 25.70 / 107.87 | 200 |
| `3p_d2` | 0 | 62 / 1 | 165.50 / 53.50 | 27.39 / 101.47 | 180 |
| `3p_d2` | 1 | 120 / 23 | 168.50 / 67.00 | 15.96 / 89.54 | 180 |
| `3p_cold` | 0 | 67 / 0 | 227.50 / 83.00 | 28.15 / 102.87 | 240 |
| `3p_cold` | 1 | 149 / 4 | 229.50 / 86.00 | 17.73 / 101.08 | 240 |
| `4p_d2` | 0 | 67 / 9 | 219.00 / 69.00 | 29.27 / 94.46 | 240 |
| `4p_d2` | 1 | 164 / 40 | 222.50 / 87.50 | 15.23 / 82.82 | 240 |
| `4p_cold` | 0 | 56 / 0 | 227.50 / 59.50 | 30.48 / 101.60 | 240 |
| `4p_cold` | 1 | 150 / 4 | 230.50 / 60.50 | 17.06 / 100.57 | 240 |

The direct two-player comparison uses the same depth for both checkpoints.

| Start | Depth | Locked declarations | Previous declarations | No declaration | Locked first-place credit | Games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `d2` | 0 | 17 | 6 | 97 | 100.00 | 120 |
| `d2` | 1 | 45 | 20 | 55 | 90.00 | 120 |
| `cold` | 0 | 5 | 0 | 195 | 175.50 | 200 |
| `cold` | 1 | 34 | 0 | 166 | 175.50 | 200 |

### Controls on the same deals

Each pair lists the untrained Schema 8 model at 0-ply first and `RandomPolicy` second.

| Solo start | Declarations | Mean qualifying sequences | Mean deadwood | Mean utility | Deals per policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d0_first` | 38 / 33 | 5.1016 / 5.0703 | 15.6484 / 15.4062 | 0.000625 / -0.057813 | 128 |
| `d0_60turn` | 40 / 37 | 2.8828 / 2.7422 | 71.7109 / 75.7969 | -0.237187 / -0.282187 | 128 |
| `d1` | 6 / 5 | 1.7500 / 1.8047 | 101.0625 / 100.0156 | -0.726250 / -0.731563 | 128 |
| `d2` | 4 / 3 | 1.6484 / 1.7500 | 102.3594 / 100.4766 | -0.758437 / -0.757188 | 128 |
| `d4` | 1 / 1 | 1.6016 / 1.5938 | 102.8203 / 103.8438 | -0.796875 / -0.797813 | 128 |
| `d6` | 0 / 1 | 1.5234 / 1.6484 | 103.8750 / 105.3359 | -0.817187 / -0.791250 | 128 |
| `d8` | 0 / 0 | 1.4766 / 1.7578 | 104.2578 / 102.5938 | -0.822813 / -0.789062 | 128 |
| `solo_cold` | 2 / 0 | 1.6445 / 1.7773 | 102.2383 / 103.0078 | -0.791719 / -0.786719 | 256 |

| Multiplayer start | Declarations | First-place credit | Mean deadwood | Games per policy |
| --- | ---: | ---: | ---: | ---: |
| `2p_d2` | 2 / 3 | 72.5000 / 60.0000 | 95.0417 / 103.5333 | 120 |
| `2p_cold` | 2 / 0 | 119.5000 / 100.0000 | 103.2300 / 102.2450 | 200 |
| `3p_d2` | 2 / 4 | 57.0000 / 60.0000 | 103.0000 / 97.6333 | 180 |
| `3p_cold` | 2 / 0 | 87.5000 / 80.0000 | 102.2875 / 106.6833 | 240 |
| `4p_d2` | 9 / 3 | 70.5000 / 60.0000 | 95.3083 / 102.1750 | 240 |
| `4p_cold` | 1 / 0 | 74.5000 / 60.0000 | 102.0167 / 104.6625 | 240 |

## Hybrid Azure training and cold holdout

Azure allocated 64 vCPUs, and 64 passed the runtime check. Three on-demand nodes ran six Gloo learner ranks. Spot nodes supplied rollouts when available.
The three training stages used 30-minute learner budgets. The first started from `locked-gt-phase4.pt` with a new optimizer.
The second and third stages reused the optimizer state at lower learning rates.
The automatic curriculum unlocked `d=9`, `d=14`, `d=21`.
The third stage explicitly set `d=21` and a 20% curriculum share. This setting does not establish mastery.

### Training and candidate selection

| Candidate | Revisions | End distance | Updates/s | Trained decisions/s | Cold declaration rate | Overall declaration rate | Mean utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `previous` | n/a | n/a | n/a | n/a | 23.57% | 31.73% | -0.0316 |
| `cluster-c1` | 488 | 14 | 0.271 | 383.15 | 39.32% | 41.59% | 0.1343 |
| `cluster-c2` | 992 | 21 | 0.280 | 450.33 | 48.31% | 48.18% | 0.2331 |
| `cluster-c3` | 1520 | 21 | 0.292 | 483.58 | 56.25% | 54.69% | 0.3387 |

Seed `600_000` selected `cluster-c3` as `locked-gt-mastery.pt` before the run opened seed `950_000`.
Selection ranks cold declaration rate first, then overall declaration rate and mean utility.
The locked SHA-256 is `7f762a66c61adb68eb0ad66a291d5377ddc0721b2b2dca679810bf9208cc84be`.

### Cold holdout on seed 950,000

4 Azure hosts completed 12,024 games and 1,657,920 decisions in 276.81 seconds with 0 incomplete cases.
The 500 solo deals per policy use the same seeds. Multiplayer rotates the tested policy through every seat.
In head-to-head games, the prior checkpoint occupies all other seats. A timeout means no player declared.

| Scenario | Policy | Search ply | Declarations | Opponent declarations | Timeouts | Mean utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| solo_cold | `locked-gt-mastery` | 0 | 153/500 | 0 | 347 | -0.0203 |
| solo_cold | `locked-gt-mastery` | 1 | 342/500 | 0 | 158 | 0.5266 |
| solo_cold | `previous` | 0 | 54/500 | 0 | 446 | -0.3916 |
| solo_cold | `previous` | 1 | 149/500 | 0 | 351 | -0.1045 |
| solo_cold | `rev0` | 0 | 2/500 | 0 | 498 | -0.7952 |
| solo_cold | `random` | 0 | 1/500 | 0 | 499 | -0.8018 |
| 2p_cold | `locked-gt-mastery` | 0 | 111/240 | 0 | 129 | 0.2807 |
| 2p_cold | `locked-gt-mastery` | 1 | 185/240 | 0 | 55 | 0.6913 |
| 3p_cold | `locked-gt-mastery` | 0 | 123/240 | 1 | 116 | 0.3438 |
| 3p_cold | `locked-gt-mastery` | 1 | 192/240 | 0 | 48 | 0.7285 |
| 4p_cold | `locked-gt-mastery` | 0 | 107/240 | 0 | 133 | 0.2587 |
| 4p_cold | `locked-gt-mastery` | 1 | 199/240 | 0 | 41 | 0.7672 |
| 2p_cold_vs_previous | `locked-gt-mastery` | 0 | 29/240 | 6 | 205 | -0.2905 |
| 2p_cold_vs_previous | `locked-gt-mastery` | 1 | 73/240 | 20 | 147 | -0.0592 |
| 3p_cold_vs_previous | `locked-gt-mastery` | 0 | 48/240 | 35 | 157 | -0.1851 |
| 3p_cold_vs_previous | `locked-gt-mastery` | 1 | 99/240 | 74 | 67 | 0.1251 |
| 4p_cold_vs_previous | `locked-gt-mastery` | 0 | 62/240 | 61 | 117 | -0.0600 |
| 4p_cold_vs_previous | `locked-gt-mastery` | 1 | 94/240 | 119 | 27 | 0.1724 |

The run bundle contains per-deal outcomes, the warm-solo cases, checkpoint hashes, and 30-second host samples.

## Balanced Azure resume and cold holdout

Three learners and three rollout hosts used 96 Azure vCPUs. Each learner ran two Gloo ranks. Each rollout host ran 16 workers with 32 tables per worker.
The run resumed AdamW at revision 1520 and trained in two 26-minute stages. The first used a learning rate of `5e-5` and a 15% warm-start share. The second used `2.5e-5` and an 8% share.

### Training and candidate selection

| Candidate | Revisions | End distance | Updates/s | Trained decisions/s | Cold declaration rate | Overall declaration rate | Mean utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous champion | n/a | n/a | n/a | n/a | 56.25% | 54.69% | 0.3387 |
| First resume checkpoint | 1960 | 21 | 0.282 | 476.01 | 58.59% | 61.05% | 0.4362 |
| Second resume checkpoint | 2440 | 21 | 0.306 | 517.92 | 51.43% | 58.67% | 0.4021 |

Seed `600_000` selected the first resume checkpoint as `locked-gt-balanced.pt` before the run opened seed `960_000`.
Selection ranks cold declaration rate first, then overall declaration rate and mean utility.
The locked SHA-256 is `3d0bbd76c606f1877a9b3465b7f8e5c14bebedadd4721fe2bc683c75b3aaa6fc`.

### Cold holdout on seed 960,000

6 Azure hosts completed 10,488 games and 1,472,188 decisions in 183.31 seconds with 0 incomplete cases.
The 500 solo deals per policy use the same seeds. Multiplayer rotates the tested policy through every seat.
In head-to-head games, the prior checkpoint occupies all other seats. A timeout means no player declared.

| Scenario | Policy | Search ply | Declarations | Opponent declarations | Timeouts | Mean utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| solo_cold | `locked-gt-balanced` | 0 | 191/500 | 0 | 309 | 0.0921 |
| solo_cold | `locked-gt-balanced` | 1 | 374/500 | 0 | 126 | 0.6227 |
| solo_cold | `previous` | 0 | 161/500 | 0 | 339 | -0.0049 |
| solo_cold | `previous` | 1 | 333/500 | 0 | 167 | 0.4952 |
| solo_cold | `rev0` | 0 | 0/500 | 0 | 500 | -0.8154 |
| solo_cold | `random` | 0 | 1/500 | 0 | 499 | -0.8078 |
| 2p_cold | `locked-gt-balanced` | 0 | 115/240 | 1 | 124 | 0.3029 |
| 2p_cold | `locked-gt-balanced` | 1 | 186/240 | 0 | 54 | 0.6970 |
| 3p_cold | `locked-gt-balanced` | 0 | 143/240 | 0 | 97 | 0.4607 |
| 3p_cold | `locked-gt-balanced` | 1 | 211/240 | 0 | 29 | 0.8374 |
| 4p_cold | `locked-gt-balanced` | 0 | 129/240 | 1 | 110 | 0.3780 |
| 4p_cold | `locked-gt-balanced` | 1 | 196/240 | 1 | 43 | 0.7507 |
| 2p_cold_vs_previous | `locked-gt-balanced` | 0 | 14/240 | 10 | 216 | -0.4273 |
| 2p_cold_vs_previous | `locked-gt-balanced` | 1 | 29/240 | 26 | 185 | -0.3689 |
| 3p_cold_vs_previous | `locked-gt-balanced` | 0 | 22/240 | 53 | 165 | -0.3580 |
| 3p_cold_vs_previous | `locked-gt-balanced` | 1 | 56/240 | 103 | 81 | -0.1460 |
| 4p_cold_vs_previous | `locked-gt-balanced` | 0 | 28/240 | 100 | 112 | -0.2552 |
| 4p_cold_vs_previous | `locked-gt-balanced` | 1 | 45/240 | 155 | 40 | -0.1108 |

The run bundle contains per-deal outcomes, the warm solo cases, checkpoint hashes, and 30-second host samples.
