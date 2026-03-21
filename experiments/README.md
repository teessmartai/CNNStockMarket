# Experiments

Tracks every AutoResearch run. Each entry in `results.json` is one Kaggle GPU experiment.

## Schema

| Field | Description |
|---|---|
| `run_id` | Unique ID — `run_NNN_short_description` |
| `timestamp` | UTC start time |
| `kaggle_kernel_version` | Kernel version pushed |
| `device` | GPU used (P100, T4, etc.) |
| `hypothesis` | What we're testing and why |
| `changes_from_previous` | Exact diff description |
| `config` | Full hyperparameter snapshot |
| `results.best_val_acc` | Best val accuracy across all epochs |
| `results.test_acc` | Test accuracy using best checkpoint |
| `results.best_epoch` | Which epoch achieved best val |
| `results.early_stopped` | Whether patience triggered |
| `verdict` | `kept` or `discarded` |
| `notes` | Observations and next steps |

## Decision Rule

- **keep** if val_acc ↑ AND test_acc ↑ (or at least doesn't drop meaningfully)
- **discard** if test_acc drops (more overfitting) or both metrics drop
- Commit kept changes to `main` before the next run

## Loop Commands

```bash
# Show status and next proposed change
python scripts/autoresearch/loop.py status

# Push current train_experiment.py to Kaggle and run
python scripts/autoresearch/loop.py run

# After kernel completes: collect result, update results.json, commit
python scripts/autoresearch/loop.py collect

# Discard last change (revert train_experiment.py to last kept commit)
python scripts/autoresearch/loop.py revert
```

## Adding to the Queue

```bash
# Add via CLI
python scripts/autoresearch/loop.py add run_010_my_idea \
  --cmd-args "--lr 5e-5 --residual" \
  --hypothesis "Lower LR with residual connections"

# Or edit experiments/queue.json directly and commit
```

## Daemon

```bash
python scripts/autoresearch/loop.py daemon
```

The daemon polls every 5 minutes. It:
1. Collects results from any finished slots
2. Auto-keeps if test_acc improves, auto-discards otherwise
3. Launches the next pending experiment from the queue
4. Commits everything to git after each tick

Add new entries to `queue.json` at any time — picked up on the next tick.
