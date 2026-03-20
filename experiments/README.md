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
