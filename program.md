# AutoResearch Program — CNN Stock Market

## Goal
Maximize **test accuracy** on 5-day binary stock direction prediction (up/down) using the 8-layer Conv1D model.
Secondary goal: improve generalization (close the gap between val_acc and test_acc).

## Model & Data
- Architecture: 8-layer Conv1D, 35M parameters, OHLCV input (5 channels)
- Dataset: `largecap-stable` preset — 54 large-cap defensive stocks, 60,642 windows
- Window: 128 days → predict T+5 day direction
- Split: 70% train / 15% val / 15% test (time-ordered)
- Label balance: 53.4% bullish / 46.6% bearish

## Current Best
- **val_acc: 54.1%** | **test_acc: 51.7%** | test_loss: 0.6923
- Epochs to best: ~2 (out of 17 total; early stopping patience=15)
- Runtime: ~20 min on P100 GPU (Kaggle)

## Baseline Config
```
lr            = 0.001
weight_decay  = 1e-5
dropout       = 0.4
batch_size    = 256
optimizer     = Adam
scheduler     = none
window        = 128
horizon       = 5
years         = 5
```

## Known Issues
- Model peaks at epoch ~2 then plateaus → LR likely too high, overshooting minimum
- val_acc (54.1%) > test_acc (51.7%) → ~2.4% generalization gap, mild overfitting
- No LR scheduler — can't recover once stuck

## Hypotheses to Explore (in priority order)
1. **Lower LR** (1e-4 or 5e-4) — most likely fix for the epoch-2 peak issue
2. **Add LR scheduler** (CosineAnnealingLR or ReduceLROnPlateau) — helps navigate past plateaus
3. **Reduce dropout** (0.3 or 0.2) — may be over-regularizing early
4. **Increase batch size** (512) — smoother gradients, may help convergence
5. **Gradient clipping** — stabilize training if LR reduction alone isn't enough
6. **Warmup + cosine decay** — standard modern training recipe
7. **Weight init tuning** — kaiming vs xavier
8. **Architecture depth** — try 6 or 10 layers instead of 8

## Constraints
- Kaggle free tier: ~30 GPU hrs/week, 9h max session
- Each full run: ~20 min on P100 (17 epochs to early stop)
- Budget: ~90 experiments/week max
- One change per experiment — keep diffs small and reviewable

## Rules for the Agent
- Modify only `train_experiment.py` (and optionally `src/training/trainer.py`)
- One meaningful change per experiment
- Document the hypothesis and expected effect in `results.json` before running
- If val_acc improves AND test_acc improves: keep the change
- If val_acc improves but test_acc drops (more overfitting): discard
- If both drop: discard
- Commit kept changes to `main` before the next experiment
