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

Check `status.json` for `status: max_updates` and `version: 2`. The update limit is an absolute policy revision, not a count of additional updates. Reuse the directory to resume. You can change the actor count, table count, threads, device, precision, learning rate, and synchronization cadence. A new `--learning-rate` replaces the rate saved with the AdamW state without resetting its moments. You can also change `--curriculum-share` and `--curriculum-max-distance`. The game, model, and other collection settings stay fixed. Restart standalone collectors after a curriculum change.

## Start a full run

```bash
python -m src.cli.train --output-dir runs/full21 --max-seconds 3600
```

The default `GameTransformer` has 2,175,224 parameters. Its four blocks use width 192, six rotary attention heads of width 32, RMSNorm, and a SwiGLU width of 512. It stores weights in checkpoint format 6 and uses the eight-byte Schema 8 tokens. Each actor keeps a key and value cache per seat and batches decisions across its live tables.

Actors play games with one to four seats in rotation, with 64 tables per process by default. Half of starts use the environment curriculum. Solo starts sample a distance from zero through six. Distance zero starts with 22 cards and ends after one discard. Multiplayer starts sample one through six and replace that many cards in each seat's hand from a shared shoe. The encoder does not receive the curriculum distance. Validation and the arena default to cold deals.

Actors sample from the policy with a 2% uniform mixture over legal actions. They write complete seat trajectories with the actual behavior log probability. The learner drops data that exceeds `--max-policy-lag`. By default, each rank draws eight trajectories uniformly with replacement from one player-count and length bucket. A bucket's weight is its trajectory count, not its decision count. The learner averages policy loss, value loss, anchor KL, and entropy within each trajectory before averaging across the batch.

The learner applies V-trace with a discount of 0.995 and no terminal bootstrap. Solo pays `+1` on a declaration and `-1` on a timeout. In declared multiplayer games the winner earns `+1` and the other seats split `-1` equally. A multiplayer timeout pays `-0.5` per seat. Deadwood affects the scoreboard, not training payoffs. The default learning rate is `5e-4`, the anchor weight is `0.05`, and the entropy weight is `0.003`.

The learner regularizes rewards and policy updates against a frozen anchor. It caches the anchor's log probabilities for replayed trajectories until the weights refresh at `--anchor-every`. The default BF16 mode uses FP32 master weights and FP32 loss math. Use `--precision fp32` to disable autocast.

| File | Contents |
| --- | --- |
| `latest-model.pt` | Published weights, game, policy revision, and collection plan. |
| `best-model.pt` | Cold validation selection, or the initial model until promotion. |
| `latest-state.pt` | Learner and anchor weights, optimizer, progress, and sampler state. |
| `inbox/batch-*.pt` | Complete byte-token rollout batches awaiting ingestion. |
| `metrics.jsonl` and `status.json` | Update metrics and run status, including collected and trained decisions, throughput, importance ratio, KL, buffer depth, and inbox depth. |

The learner does not save its in-memory rollout buffer. On resume, it refills the buffer from actors and any compatible files still in the inbox.

### Schema 8 tokens

The 30-token snapshot has one `ACT`, one `TABLE_STATE`, six `SEAT` slots, and 22 `HAND` slots. A 21-card hand can reach at most 22 distinct faces after a draw. Duplicate faces share one token with a copy count.

| Byte | Meaning |
| ---: | --- |
| 0 | Token kind. |
| 1 | Relative seat, phase, copies, or rule value. Each role has a separate byte range. |
| 2, 3, 4 | Face, printed suit, and printed rank. An absent face uses `53`, `5`, and `14`. |
| 5 | Visible offered face, hand size, next seat with active status, or discard pile size. Pile size caps at 150. |
| 6 | Remaining turns up to 120, or the remaining stock in two-card buckets. |
| 7 | `1 + discard_action(face)` on `HAND` tokens. Zero everywhere else. |

Zero means unused in each field. All Transformer layers update all snapshot tokens. The `ACT` representation gives the base policy logits and value. For each nonzero pointer byte, the shared linear pointer head adds that token's score to the named action logit before legal masking. The network has no Papplu imports.

Schema 8 supports up to 21 cards before drawing, four decks, and a configured budget of 235 turns. The header stores the full budget. Start a fresh run in a new directory. Schema 7 weights, trainer state, and rollouts do not use the same byte layout.

## Run several CPU learners

On a 20-vCPU host, start two Gloo ranks with eight learner threads and two single-threaded actors each:

```bash
torchrun --standalone --nproc_per_node=2 -m src.cli.train \
  --output-dir runs/multi --torch-threads 8 --actors 2 --actor-threads 1 \
  --batch-size 8 --max-seconds 720
```

For multiple hosts, replace `--standalone` with the same `--nnodes` and private `--rdzv-endpoint` on every host. Use `--rdzv-backend c10d` and the same unique `--rdzv-id`. Use the same model and training flags. Do not expose the Gloo or rendezvous ports to untrusted hosts.

Each rank waits for local replay. The ranks weight shared buckets by their combined trajectory counts and sample from local replay within the chosen bucket. With `--sync-every 1`, they average gradients before clipping and stepping AdamW. With `--sync-every 8`, each rank takes eight local AdamW steps, then the ranks average the FP32 model parameters. This Post-Local AdamW mode does not average the optimizer moments. Metrics sum decision counts across ranks and advance the policy revision by eight per full synchronization interval.

Actors use `rank-N/inbox` and read weights from local disk. Rank 0 writes the canonical checkpoints, status, and metrics and runs evaluation. Other ranks publish `rank-N/latest-model.pt` for their actors. For resume, keep `latest-state.pt` and `best-model.pt` on rank 0. Rank 0 shares the saved startup state over Gloo, so no rollout or checkpoint files need to move between hosts during training.

## Feed a learner from other machines

Set `--actors 0` on the learner if all collectors run elsewhere. Copy `latest-model.pt` to each actor host with an atomic rename. Start each collector with a distinct worker ID.

```bash
python -m src.cli.train --collect actor-outbox --model latest-model.pt \
  --worker 100 --tables 64 --max-batches 1
```

Leave off `--max-batches` for continuous collection. Refresh the actor's model file atomically. Live deals keep their admission weights and caches. Copy completed `batch-*.pt` files to temporary names in the learner inbox, then rename each file to its final name. Transfer each batch once and clear it from the outbox after transfer. Continuous collectors pause when eight batches from the same worker await transfer or ingestion. The learner rejects incompatible files and discards stale batches.

## Evaluate a checkpoint

```bash
python -m src.cli.evaluate --checkpoint runs/full21/best-model.pt \
  --games 300 --warm-games 0 --output runs/full21/evaluation.json
python -m src.cli.arena --model runs/full21/best-model.pt --model random \
  --games 120 --output runs/full21/arena.json
```

`src.cli.evaluate` pairs the model and random policy on the same solo seeds. It reports discard puzzles separately when `--warm-games` is positive. `src.cli.arena` rotates the lineup over one to six seats and reports each deal's utilities and winner. Supply a checkpoint or `random` for each seat. Checkpoints in the same lineup must use the same game settings.

## Azure checkpoint and held-out results

These are historical Schema 7 results under the old deadwood-based utility. Do not compare these utility values with Schema 8's declaration and timeout payoffs.

Stage 2 used Schema 7 on five `Standard_D16als_v6` VMs with 80 AMD EPYC Genoa vCPUs and 160 GiB of provisioned RAM. Each VM ran two BF16 ranks. Each rank used seven learner threads and one single-threaded actor with 32 tables.

Stage 2a started from Stage 1 at 750 cumulative updates and reached 1,446 with a 22-minute budget. It used batch size 4, learning rate `4e-4`, `eta=0.03`, `anchor_every=480`, and a 50% solo curriculum share up to distance 5. Stage 2b resumed the same run with learning rate `2e-4` and reached 2,158 cumulative updates with a second 22-minute budget. Both stages used `sync_every=8`.

The matched burst used one 60-second budget per cadence. The fleet trained 537.11 decisions per second at `sync_every=1` and 1,098.13 at `sync_every=8`. In the sampled learner window, rank 0 spent 30.96% and 14.35% of time in the FP32 Gloo all-reduce, respectively. These times include rank arrival skew. This single comparison does not measure run-to-run variance. Per-VM CPU use at cadence 8 ranged from 56.40% to 66.85% in the burst and from 78.42% to 84.11% during Stage 2a.

### Candidate lock and Check-In #4

Candidate selection used only seed `600_000`. Check-In #4 evaluated Stage 2a on the same eight cases used for selection: cold solo, distance 4 solo, and multiplayer at two, three, and four seats against both RandomPolicy and Revision 0. Each case had equal weight in the mean utility. Ties use higher mean win rate, lower mean deadwood, then checkpoint name.

| Rank | Checkpoint | Mean utility | Mean win rate | Mean deadwood |
| ---: | --- | ---: | ---: | ---: |
| 1 | Stage 2b | -0.0460 | 0.28% | 82.84 |
| 2 | Stage 2a | -0.0634 | 0.00% | 87.74 |
| 3 | Stage 1 | -0.0916 | 0.20% | 94.26 |
| 4 | Revision 283 | -0.1077 | 0.15% | 98.68 |
| 5 | Revision 0 | -0.1197 | 0.07% | 101.62 |

On the 64 shared cold solo deals, mean deadwood fell from 88.08 for Stage 1 to 71.16 at Check-In #4 and 64.89 for Stage 2b. None of those checkpoints won a cold solo deal at this seed. The run locked Stage 2b before the held-out evaluation. The SHA-256 of `locked-gt-winner.pt` is `78c64274444d94e60b23eaa348effc24bf0e59e7d3dd848adfbb2e4ad25a9cc8`.

### Untouched held-out solo

The held-out run started at seed `900_000` after the lock. It used 16 evaluation processes per VM across all five VMs and took 53 seconds, including process startup. All three solo policies played the same 200 cold deals and the same 128 deals at each curriculum distance. Mean deadwood is terminal deadwood out of 210.

| Start | Policy | Wins / deals | Win rate | Mean utility | Mean deadwood / 210 |
| --- | --- | ---: | ---: | ---: | ---: |
| Cold 21-card | Locked GameTransformer | 2 / 200 | 1.00% | -0.3245 | 70.25 |
| Cold 21-card | RandomPolicy | 1 / 200 | 0.50% | -0.5115 | 108.46 |
| Cold 21-card | Revision 0 | 1 / 200 | 0.50% | -0.4924 | 104.46 |
| Distance 2 | Locked GameTransformer | 1 / 128 | 0.78% | -0.2859 | 61.69 |
| Distance 2 | RandomPolicy | 1 / 128 | 0.78% | -0.4916 | 104.88 |
| Distance 2 | Revision 0 | 1 / 128 | 0.78% | -0.4679 | 99.91 |
| Distance 4 | Locked GameTransformer | 4 / 128 | 3.13% | -0.2872 | 66.88 |
| Distance 4 | RandomPolicy | 3 / 128 | 2.34% | -0.4596 | 101.43 |
| Distance 4 | Revision 0 | 2 / 128 | 1.56% | -0.4426 | 96.23 |
| Distance 6 | Locked GameTransformer | 0 / 128 | 0.00% | -0.3203 | 67.27 |
| Distance 6 | RandomPolicy | 1 / 128 | 0.78% | -0.4893 | 104.39 |
| Distance 6 | Revision 0 | 0 / 128 | 0.00% | -0.4664 | 97.95 |

### Untouched held-out multiplayer

The locked checkpoint played every seat once per deal. All other seats used the named opponent. Opponent win rate and mean deadwood are per seat. The locked checkpoint earned positive mean utility in every matchup but did not win a multiplayer game.

| Seats | Opponent | Deals | Wins / games | Win rate | Opponent win rate | Mean utility | Mean deadwood / 210 | Opponent mean deadwood / 210 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | RandomPolicy | 100 | 0 / 200 | 0.00% | 0.00% | 0.0552 | 84.22 | 107.40 |
| 2 | Revision 0 | 100 | 0 / 200 | 0.00% | 0.00% | 0.0262 | 96.35 | 107.36 |
| 2 | Stage 1 | 100 | 0 / 200 | 0.00% | 0.00% | 0.0191 | 96.92 | 104.93 |
| 3 | RandomPolicy | 80 | 0 / 240 | 0.00% | 0.21% | 0.0604 | 82.80 | 108.60 |
| 3 | Revision 0 | 80 | 0 / 240 | 0.00% | 0.21% | 0.0309 | 93.74 | 107.14 |
| 3 | Stage 1 | 80 | 0 / 240 | 0.00% | 0.21% | 0.0186 | 98.62 | 106.88 |
| 4 | RandomPolicy | 60 | 0 / 240 | 0.00% | 0.14% | 0.0518 | 83.80 | 105.86 |
| 4 | Revision 0 | 60 | 0 / 240 | 0.00% | 0.00% | 0.0290 | 93.90 | 106.08 |
| 4 | Stage 1 | 60 | 0 / 240 | 0.00% | 0.14% | 0.0206 | 95.63 | 104.56 |

## Schema 8 Azure results

Slice 4j trained on one 16-vCPU `Standard_D16als_v6` in West US 2. Two localhost Gloo ranks used BF16, seven learner threads per rank, and one single-threaded actor with 32 tables per rank. Each rank sampled eight trajectories per update. The run used `sync_every=1`, `eta=0.015`, `entropy_weight=0.002`, and `anchor_every=400` throughout.

| Stage | Budget | Curriculum share | Maximum distance | Learning rate | Cumulative revision | Updates | Trained decisions/s | Mean CPU | Mean / peak RAM GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Check-In #7 | 15 min | 85% | 3 | `5e-4` | 283 | 283 | 577.79 | 88.29% | 9.89 / 13.02 |
| Check-In #8 | 25 min | 65% | 7 | `3.5e-4` | 735 | 452 | 563.04 | 89.79% | 9.92 / 13.15 |
| Final Stage | 25 min | 40% | 14 | `2e-4` | 1,190 | 455 | 578.79 | 90.30% | 10.39 / 12.93 |

CPU and RAM are guest-wide measurements sampled every 30 seconds, with a final sample at stage exit. The throughput column counts trained decisions across both ranks over the stage. These are three phases of one run, not repeated benchmarks.

### Candidate lock at seed 600,000

Each checkpoint played 64 paired deals in each solo case and 48 base deals in each two-seat case against RandomPolicy. Each two-seat deal ran once per candidate seat. Selection equally weighted the seven case win rates. Ties use the mean win rate of the two cold cases, then mean utility across all seven cases. Revision 0 is the fixed Schema 8 baseline from the earlier check-ins, not the initial weights of this run.

| Checkpoint | Warm d=0 | d=1 | d=2 | d=4 | Cold solo | 2p d=2 | Cold 2p | Mean win rate | Cold mean | Mean utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Revision 0 | 17/64 | 0/64 | 2/64 | 0/64 | 0/64 | 4/96 | 0/96 | 4.84% | 0.00% | -0.7641 |
| Check-In #7 | 17/64 | 3/64 | 0/64 | 0/64 | 0/64 | 3/96 | 1/96 | 5.06% | 0.52% | -0.7604 |
| Check-In #8 | 17/64 | 1/64 | 0/64 | 0/64 | 0/64 | 2/96 | 0/96 | 4.32% | 0.00% | -0.7746 |
| Final Stage | 15/64 | 2/64 | 0/64 | 0/64 | 0/64 | 4/96 | 0/96 | 4.39% | 0.00% | -0.7738 |
| RandomPolicy | 18/64 | 4/64 | 1/64 | 3/64 | 0/64 | n/a | n/a | n/a | n/a | n/a |

Check-In #7 led selection at 5.06%, 0.22 percentage points ahead of Revision 0. The run copied revision 283 to `locked-gt-winner.pt` before opening seed `910_000`. Its SHA-256 is `804adef0f664593e0b1a5a0a8b49ffc63e4ab7590553c0198810a29b08f28ca7`.

### Held-out solo at seed 910,000

The held-out run completed 6,348 games on 16 local workers with no evaluation errors. All three solo policies played the same seeds within each start condition. Every win below is a valid declaration. The locked checkpoint won two more warm starts than RandomPolicy but two fewer cold solo games.

| Start | Policy | Wins / deals | Win rate | Mean utility | Mean deadwood / 210 |
| --- | --- | ---: | ---: | ---: | ---: |
| Warm start, d=0 | Locked checkpoint | 35 / 128 | 27.34% | -0.4531 | 16.64 |
| Warm start, d=0 | RandomPolicy | 33 / 128 | 25.78% | -0.4844 | 17.59 |
| Warm start, d=0 | Revision 0 | 29 / 128 | 22.66% | -0.5469 | 18.19 |
| Distance 1 | Locked checkpoint | 3 / 128 | 2.34% | -0.9531 | 108.28 |
| Distance 1 | RandomPolicy | 1 / 128 | 0.78% | -0.9844 | 106.34 |
| Distance 1 | Revision 0 | 3 / 128 | 2.34% | -0.9531 | 101.85 |
| Distance 2 | Locked checkpoint | 2 / 128 | 1.56% | -0.9688 | 106.47 |
| Distance 2 | RandomPolicy | 4 / 128 | 3.13% | -0.9375 | 103.32 |
| Distance 2 | Revision 0 | 4 / 128 | 3.13% | -0.9375 | 101.34 |
| Distance 4 | Locked checkpoint | 1 / 128 | 0.78% | -0.9844 | 108.29 |
| Distance 4 | RandomPolicy | 2 / 128 | 1.56% | -0.9688 | 103.35 |
| Distance 4 | Revision 0 | 1 / 128 | 0.78% | -0.9844 | 102.02 |
| Distance 6 | Locked checkpoint | 1 / 128 | 0.78% | -0.9844 | 112.85 |
| Distance 6 | RandomPolicy | 0 / 128 | 0.00% | -1.0000 | 101.23 |
| Distance 6 | Revision 0 | 0 / 128 | 0.00% | -1.0000 | 102.66 |
| Cold 21-card | Locked checkpoint | 1 / 256 | 0.39% | -0.9922 | 109.09 |
| Cold 21-card | RandomPolicy | 3 / 256 | 1.17% | -0.9766 | 102.98 |
| Cold 21-card | Revision 0 | 2 / 256 | 0.78% | -0.9844 | 104.41 |

### Held-out multiplayer at seed 910,000

For each matchup the locked checkpoint played every seat once per base deal. All other seats used the named opponent. Opponent win rate and utility are per opponent seat. Check-In #7 matchups are self-play controls because the lock selected that checkpoint.

| Seats | Start | Opponent | Base deals | Wins / games | Win rate | Mean utility | Opponent win rate | Opponent utility |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | Distance 2 | RandomPolicy | 60 | 4 / 120 | 3.33% | -0.4750 | 5.00% | -0.4417 |
| 2 | Distance 2 | Revision 0 | 60 | 5 / 120 | 4.17% | -0.4417 | 0.83% | -0.5083 |
| 2 | Distance 2 | Check-In #7 | 60 | 8 / 120 | 6.67% | -0.4333 | 6.67% | -0.4333 |
| 3 | Distance 2 | RandomPolicy | 60 | 7 / 180 | 3.89% | -0.4417 | 2.22% | -0.4667 |
| 3 | Distance 2 | Revision 0 | 60 | 6 / 180 | 3.33% | -0.4500 | 2.22% | -0.4667 |
| 3 | Distance 2 | Check-In #7 | 60 | 8 / 180 | 4.44% | -0.4333 | 4.44% | -0.4333 |
| 4 | Distance 2 | RandomPolicy | 60 | 7 / 240 | 2.92% | -0.4514 | 0.97% | -0.4773 |
| 4 | Distance 2 | Revision 0 | 60 | 12 / 240 | 5.00% | -0.4021 | 4.58% | -0.4076 |
| 4 | Distance 2 | Check-In #7 | 60 | 13 / 240 | 5.42% | -0.3917 | 5.42% | -0.3917 |
| 2 | Cold | RandomPolicy | 100 | 0 / 200 | 0.00% | -0.5025 | 0.50% | -0.4925 |
| 2 | Cold | Revision 0 | 100 | 0 / 200 | 0.00% | -0.5000 | 0.00% | -0.5000 |
| 2 | Cold | Check-In #7 | 100 | 0 / 200 | 0.00% | -0.5000 | 0.00% | -0.5000 |
| 3 | Cold | RandomPolicy | 80 | 1 / 240 | 0.42% | -0.4938 | 0.21% | -0.4969 |
| 3 | Cold | Revision 0 | 80 | 0 / 240 | 0.00% | -0.5000 | 0.21% | -0.4969 |
| 3 | Cold | Check-In #7 | 80 | 0 / 240 | 0.00% | -0.5000 | 0.00% | -0.5000 |
| 4 | Cold | RandomPolicy | 60 | 1 / 240 | 0.42% | -0.4924 | 0.28% | -0.4942 |
| 4 | Cold | Revision 0 | 60 | 1 / 240 | 0.42% | -0.4938 | 0.00% | -0.4993 |
| 4 | Cold | Check-In #7 | 60 | 0 / 240 | 0.00% | -0.5000 | 0.00% | -0.5000 |

The locked checkpoint declared three times in 2,040 cold multiplayer games across the nine matchups. The cold results do not establish an advantage over either baseline.

## Shipped checkpoint

The browser still loads the shipped v5 `card_mlp` from `web/papplu-model.json`. Its checkpoint SHA-256 is `a229f90e76ae9da9b8c97d35db10e2d921a3a46780fc54d292ba2bde3df67796`. See the [browser held-out results](README.md#held-out-results). The GameTransformer trainer does not read or export that artifact.
