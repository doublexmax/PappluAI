# Train a Papplu model

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

On resume you can change actor and table counts, thread counts, device, precision, `--checkpoint-every`, `--learning-rate`, `--eta`, and `--sync-every`. A new `--learning-rate` replaces the rate saved with AdamW without resetting its moments. You can also change `--players`, `--curriculum-share`, and `--curriculum-max-distance`. The game, model, and other collection settings stay fixed. Restart standalone collectors after changing the collection plan.

## Start a full run

```bash
python -m src.cli.train --output-dir runs/full21 --max-seconds 3600
```

The default `GameTransformer` has 2,175,224 parameters. Its four blocks use width 192, six rotary attention heads of width 32, RMSNorm, and a SwiGLU width of 512. It stores weights in checkpoint format 6 and uses eight-byte Schema 8 tokens. Each actor keeps a key and value cache per seat and batches decisions across its live tables.

Training defaults to three decks, 21 cards per hand, five required pure sequences, 60 turns per seat, and discard recycling. Two actor processes per learner rank cycle through one-, two-, three-, and four-seat games, with 64 tables per process by default.

Half of starts use the curriculum by default. Solo curriculum starts take the smaller of two uniform draws from zero through six. Distance zero starts with one extra card and ends after one discard. Multiplayer curriculum starts sample a distance uniformly from one through six and replace that many cards in each seat's hand from a shared shoe. The encoder does not receive the curriculum distance. Validation and the arena use cold deals.

Self-play defaults to `--search-ply 1`. The actor uses a pure sequence upper bound to skip declaration checks when the hand cannot meet the quota. It gives equal probability to every legal winning discard. If taking the top discard permits a legal winning discard that turn, it takes the card with probability one. On other decisions it uses the masked neural policy without scoring candidates. At `--search-ply 0` it skips the declaration checks.

By default, actors mix 2% uniform probability over legal actions into the policy when no move is forced. The rollout records one `uint8` forced flag and the actual behavior log probability for each decision. It quantizes the policy target before exploration to 54 unsigned 16-bit values and stores their bits in an `int16` tensor. The learner validates target mass and legality on load. It applies cross-entropy only to forced decisions and uses a V-trace policy gradient on ordinary decisions. `--entropy-weight` defaults to zero.

The value head uses V-trace with a discount of 0.995 and no terminal bootstrap. A declaration pays the winner `+1`. In multiplayer the other seats split `-1` equally. On a timeout let `f` be the share of the pure sequence quota recorded by `HandPenalty.qualifying_sequences`, capped at 1. A zero quota gives `f = 1`. Solo pays `-1 + 0.60 * f`. Multiplayer pays each seat `-0.50 + 0.30 * (f - mean_of_other_seats)`. The relative term sums to zero, while every seat still gets a negative payoff. The scoreboard still uses exact deadwood.

The learner drops data more than `--max-policy-lag` revisions old. The default lag limit is 200. Each rank samples `--batch-size` trajectories with replacement from one player-count and length bucket. The default batch size is eight. A bucket's weight is its trajectory count. Each trajectory has equal weight in the policy, value, and anchor KL losses.

The default learning rate is `5e-4` and the default anchor weight (`--eta`) is `0.01`. Replay caches anchor log probabilities until the anchor refreshes. The default interval (`--anchor-every`) is 2,000 revisions. Training defaults to CPU with BF16 autocast, FP32 master weights, and FP32 loss math. Use `--precision fp32` to disable autocast.

| File | Contents |
| --- | --- |
| `latest-model.pt` | Published weights, game, policy revision, and collection plan. |
| `best-model.pt` | Cold validation selection, or the initial model until promotion. |
| `latest-state.pt` | Learner and anchor weights, optimizer, progress, and sampler state. |
| `inbox/batch-*.pt` | Complete byte-token rollout batches with quantized policy targets awaiting ingestion. |
| `metrics.jsonl` and `status.json` | Update metrics and run status, including collected and trained decisions, throughput, importance ratio, KL, buffer depth, and inbox depth. |

The learner does not save its in-memory rollout buffer. On resume, it refills the buffer from actors and any compatible files still in the inbox. Rollout format 4 requires forced flags.

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
  --games 300 --warm-games 0 --search-ply 0 --output runs/full21/evaluation.json
python -m src.cli.arena --model runs/full21/best-model.pt --model random \
  --games 120 --search-ply 1 --output runs/full21/arena.json
```

`src.cli.evaluate` pairs the model and random policy on the same solo seeds. It reports discard puzzles separately when `--warm-games` is positive. `src.cli.arena` reports each deal's utilities and winner. It rotates the lineup between deals and plays each seed once. Supply a checkpoint or `random` for each of one to six seats. Checkpoints in the same lineup must use the same game settings. Both commands default to `--search-ply 0`. The model picks the highest legal logit without checking for a winning move first. Use `--search-ply 1` to enable the same search used by self-play.

## Azure results

The Schema 8 run used commit `0f00513` on one 16-vCPU `Standard_D16als_v6` in West US 2. Two Gloo ranks ran on that VM with BF16 and `--sync-every 1`. Each rank used seven learner threads and one single-threaded actor with 32 tables. Each update sampled 12 trajectories per rank.

The learner loaded weights from revision 826 and started a new AdamW state at revision 0. Check-In #12 and Final resumed from the preceding phase. The run used `--search-ply 1`, `--eta 0.01`, `--entropy-weight 0.002`, and `--anchor-every 400` throughout.

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

## Shipped checkpoint

The browser loads the separate v5 `card_mlp` from `web/papplu-model.json`. See the [browser held-out results](README.md#held-out-results) for that checkpoint. The [Azure results](#azure-results) cover the Python GameTransformer and report end-to-end evaluation time, not per-call inference latency.
