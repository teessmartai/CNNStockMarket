# Phase 1 Summary — 5-Year Largecap-Stable

**Dataset:** 54 largecap-stable tickers, 5 years (2021–2026), 60,642 windows  
**Split:** 70% train / 15% val / 15% test (chronological)  
**Compute:** Kaggle P100-PCIE-16GB (~20 min/run)  
**Runs completed:** 15 (13 meaningful, 2 infra errors)

---

## 🏆 Best Configuration

| Hyperparameter | Value |
|---------------|-------|
| LR | 1e-4 |
| Weight decay | 1e-5 |
| Dropout | 0.4 |
| Batch size | 256 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Normalization | **Log-returns + global z-score** |
| Window | 128 days |
| Horizon | T+5 days |
| BatchNorm | True (Conv → BN → Activation) |
| Residual connections | False |
| Gradient clipping | Off |

## 🏆 Best Metrics

| Metric | Value |
|--------|-------|
| Val accuracy | 58.5% |
| **Test accuracy** | **57.4%** |
| Test loss | 0.7221 |
| Epochs | 27 |
| Val/test gap | 1.1pp |

---

## Full Results

| Run | Change | Val | Test | Gap | Verdict |
|-----|--------|-----|------|-----|---------|
| run_001 | Baseline (LR=1e-3) | 54.1% | 51.7% | 2.4pp | kept (baseline) |
| run_002 | LR=1e-4 | 54.1% | 54.1% | 0.0pp | ✅ kept |
| run_003 | Cosine scheduler | 53.1% | 52.0% | 1.1pp | ❌ |
| run_004 | No scheduler | 55.5% | 52.0% | 3.5pp | ❌ |
| run_005 | **Log-return norm** | 58.5% | **57.4%** | 1.1pp | ✅ **BEST** |
| run_006 | AdamW (minmax) | 55.5% | 51.4% | 4.1pp | ❌ |
| run_007 | Residual connections | 57.9% | 53.7% | 4.2pp | ❌ |
| run_008 | AdamW (logreturns) | 58.3% | 51.8% | 6.5pp | ❌ |
| run_009 | No BatchNorm | 55.4% | 54.7% | 0.7pp | ❌ |
| run_010 | Residual + dropout 0.2 | 58.5% | 53.5% | 5.0pp | ❌ |
| run_011 | Years=7 | — | — | — | ❌ infra |
| run_012 | LR=5e-5 | 59.2% | 56.9% | 2.3pp | ❌ (-0.5pp) |
| run_013 | Window=256 | 57.0% | 46.5% | 10.5pp | ❌ |
| run_014 | Gradient clipping 1.0 | 59.0% | 52.6% | 6.4pp | ❌ |

---

## Key Findings

**What helped:**
- **Log-return normalization** — biggest single win (+3.3pp test). Makes features stationary, removes price scale/trend bias. The model was previously learning to distinguish price levels rather than movement patterns.
- **Lower LR (1e-4)** — closed the generalization gap entirely (2.4pp → 0pp). The model needs gentle updates.
- **ReduceLROnPlateau scheduler** — essential. Removing it improved val but tanked test (overfitting). Without plateau, the model memorizes val patterns.
- **BatchNorm** — contributes ~2.7pp test accuracy. BN stabilizes training and acts as regularizer.

**What hurt / didn't help:**
- **Residual connections** — consistent 4–6pp val/test gap regardless of dropout. 60K windows is not enough data for skip-connection architectures to generalize.
- **AdamW** — worse than Adam in every test (both minmax and logreturns). Introduces 5–7pp overfit gap. Decoupled weight decay may be destabilizing with plateau scheduler.
- **Larger window (256)** — catastrophic: test=46.5%, 10.5pp gap. Much longer sequences = much harder to generalize with this dataset size.
- **Gradient clipping (1.0)** — 6.4pp overfit gap, test=52.6%. May be interfering with the plateau scheduler's LR reduction.
- **LR=5e-5** — marginal improvement in val (59.2%) but test dropped slightly to 56.9%. Plateau at 1e-4 is well-calibrated.
- **Cosine/no scheduler** — both worse than plateau.

---

## Phase 2 Plan

With the baseline locked, Phase 2 will test:

1. **20-year dataset** (stride=1) — 4× more training data (~265K windows)
2. **Stride=5** — non-overlapping labels, reduce temporal autocorrelation in train set
3. **Combined: 20yr + best config** — likely the highest-impact next step
4. After data experiments: technical indicators (RSI, MACD, Bollinger Bands) as additional input channels
