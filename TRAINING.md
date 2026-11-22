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
  --batch-size 4 --unroll 4 --batch-decisions 16 --min-buffer-decisions 16 \
  --eval-every 1 --eval-deals 1 --max-updates 1 --max-seconds 90
python -m src.cli.train --output-dir runs/smoke --max-updates 2 --max-seconds 90
```

Check `status.json` for `status: max_updates` and `version: 2`. The update limit is an absolute policy revision, not a count of additional updates. Reuse the directory to resume. You can change the runtime settings, including actor count, table count, threads, device, and precision. The game, model, and collection plan stay fixed. Start in a different directory for incompatible checkpoints or trainer state.

## Start a full run

```bash
python -m src.cli.train --output-dir runs/full21 --max-seconds 3600
```

By default, the learner uses a six-layer Transformer with width 256, eight heads, and a SwiGLU width of 704. Actors play one to four seats in rotation with 64 tables per process. Thirty percent of solo starts use the environment curriculum at a random distance of one to ten cards. The encoder has no curriculum features. Validation uses only cold deals.

Actors sample from the policy with a 2% uniform mixture over legal actions. They write complete seat trajectories with the actual behavior log probability. The learner drops data that exceeds `--max-policy-lag`. It applies V-trace and regularizes rewards and policy updates against a frozen anchor. `--anchor-every` controls when the learner copies the online weights into the anchor. The default BF16 mode uses FP32 master weights and FP32 loss math. Use `--precision fp32` to disable autocast.

| File | Contents |
| --- | --- |
| `latest-model.pt` | Published weights, game, policy revision, and collection plan. |
| `best-model.pt` | Cold validation selection, or the initial model until promotion. |
| `latest-state.pt` | Learner and anchor weights, optimizer, progress, and sampler state. |
| `inbox/batch-*.pt` | Complete byte-token rollout batches awaiting ingestion. |
| `metrics.jsonl` and `status.json` | Update metrics and run status, including throughput, importance ratio, KL, buffer depth, and inbox depth. |

The learner does not save its in-memory rollout buffer. On resume, it refills the buffer from actors and any compatible files still in the inbox.

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

The browser still loads the shipped v5 `card_mlp` from `web/papplu-model.json`. Its checkpoint SHA256 is `a229f90e76ae9da9b8c97d35db10e2d921a3a46780fc54d292ba2bde3df67796`. See the [browser held-out results](README.md#held-out-results). The v6 trainer does not read or export that artifact.
