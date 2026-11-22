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

The default `GameTransformer` has 2,175,224 parameters, four layers, width 192, six attention heads, and a SwiGLU width of 512. Schema 8 uses eight bytes per token. Each actor keeps a key and value cache per seat and batches decisions across its live tables.

Actors play games with one to four seats in rotation, with 64 tables per process by default. Half of starts use the environment curriculum. Solo starts sample a distance from zero through six. Distance zero starts with 22 cards and ends after one discard. Multiplayer starts sample one through six and replace that many cards in each seat's hand from a shared shoe. The encoder does not receive the curriculum distance. Validation and the arena default to cold deals.

Actors sample from the policy with a 2% uniform mixture over legal actions. They write complete seat trajectories with the actual behavior log probability. The learner drops data that exceeds `--max-policy-lag`. Each rank samples six complete seat trajectories of similar length per update by default. The learner applies V-trace with a discount of 0.995 and no terminal bootstrap. The learning rate is `5e-4`, the anchor weight is `0.05`, and the entropy weight is `0.003`.

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
  --batch-size 6 --max-seconds 720
```

For multiple hosts, replace `--standalone` with the same `--nnodes` and private `--rdzv-endpoint` on every host. Use `--rdzv-backend c10d` and the same unique `--rdzv-id`. Use the same model and training flags. Do not expose the Gloo or rendezvous ports to untrusted hosts.

Each rank waits for local replay. With `--sync-every 1`, the Gloo ranks average gradients before clipping and stepping AdamW. With `--sync-every 8`, each rank takes eight local AdamW steps, then the ranks average the FP32 model parameters. This Post-Local AdamW mode does not average the optimizer moments. Metrics sum decision counts across ranks and advance the policy revision by eight per full synchronization interval.

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

## Shipped checkpoint

The browser still loads the shipped v5 `card_mlp` from `web/papplu-model.json`. Its checkpoint SHA-256 is `a229f90e76ae9da9b8c97d35db10e2d921a3a46780fc54d292ba2bde3df67796`. See the [browser held-out results](README.md#held-out-results). The GameTransformer trainer does not read or export that artifact.
