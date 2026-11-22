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

Check `status.json` for `status: max_updates` and `version: 2`. The update limit is an absolute policy revision, not a count of additional updates. Reuse the directory to resume. You can change the runtime settings, including actor count, table count, threads, device, and precision. The game, model, and collection plan stay fixed. Start in a different directory for incompatible checkpoints or trainer state.

## Start a full run

```bash
python -m src.cli.train --output-dir runs/full21 --max-seconds 3600
```

By default, the learner uses a four-layer Transformer with width 192, six heads, and a SwiGLU width of 512. Actors play one to four seats in rotation with 64 tables per process. Half of solo starts use the environment curriculum at a random distance of one to six cards. The encoder has no curriculum features. Validation uses only cold deals.

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

Schema 7 gives every face the same field and reserves byte zero for unused fields. Start a fresh run rather than reusing schema 6 weights, state, or rollouts. The encoder supports up to 254 turns and 254 cards outside the initial solo hand and joker indicator, so clock and card counts fit without truncation.

## Run several CPU learners

On a 20-vCPU host, start two Gloo ranks with eight learner threads and two single-threaded actors each:

```bash
torchrun --standalone --nproc_per_node=2 -m src.cli.train \
  --output-dir runs/multi --torch-threads 8 --actors 2 --actor-threads 1 \
  --batch-size 6 --max-seconds 720
```

For multiple hosts, replace `--standalone` with the same `--nnodes` and private `--rdzv-endpoint` on every host. Use `--rdzv-backend c10d` and the same unique `--rdzv-id`. Use the same model and training flags. Do not expose the Gloo or rendezvous ports to untrusted hosts.

Each rank waits for local replay, then the ranks average gradients before clipping and stepping AdamW. Actors use `rank-N/inbox` and read weights from local disk. Rank 0 writes the canonical checkpoints, status, and metrics and runs evaluation. Other ranks publish only `rank-N/latest-model.pt`. The metrics add decision counts across ranks and report one update per synchronized optimizer step. For resume, keep `latest-state.pt` and `best-model.pt` on rank 0. Rank 0 shares the startup state over Gloo, so no rollout or checkpoint files need to move between hosts during training.

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

## Shipped checkpoint

The browser still loads the shipped v5 `card_mlp` from `web/papplu-model.json`. Its checkpoint SHA256 is `a229f90e76ae9da9b8c97d35db10e2d921a3a46780fc54d292ba2bde3df67796`. See the [browser held-out results](README.md#held-out-results). The GameTransformer trainer does not read or export that artifact.
