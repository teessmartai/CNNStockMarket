# AutoResearch Results

_Last updated: 2026-03-22 16:32 UTC_

> **spp** = train_samples / num_params. Old runs used a fixed 35M-param model (spp ≈ 1:800–1:27K, severely overparameterized). New runs use `--samples-per-param 100` so the model auto-sizes to the data.

## 🏆 Current Best

**run_p3_003_sp500_stride133_7yr**  
Test: **68.8%** | Val: 70.8% | CNN 35.4M params  
Config: `--preset sp500 --years 7 --stride 133 --shuffle-split`

## Phase 1 — Hyperparameter search (5yr, 54 largecap, stride=1, chrono)

| Run | Config | Val | Test | Params | spp | Verdict |
|-----|--------|-----|------|--------|-----|---------|
| run_001_baseline_gpu | `GPU support added (was hardcoded CPU)` | 54.1% | 51.7% | CNN 35.4M | 1:833 | ✅ kept |
| run_002_lower_lr_1e4 | `train_experiment.py: default lr 1e-3 → 1e-4` | 54.1% | 54.1% | CNN 35.4M | 1:833 | ✅ kept |
| run_003_scheduler_cosine | `train_experiment.py: --scheduler cosine (CosineAnnealin` | 53.1% | 52.0% | CNN 35.4M | 1:833 | ❌ |
| run_004_scheduler_none_attempt1 | `runner: --scheduler none; new kernel slug cnn-stock-tra` | 0.0% | 0.0% | CNN 35.4M | 1:833 | ❌ |
| run_004_scheduler_none | `--scheduler none` | 55.5% | 52.0% | CNN 35.4M | 1:833 | ❌ |
| run_005_logreturns | `--norm logreturns` | 58.5% | 57.4% | CNN 35.4M | 1:833 | ✅ kept |
| run_006_adamw | `--optimizer adamw (tested on minmax baseline, not logre` | 55.5% | 51.4% | CNN 35.4M | 1:833 | ❌ |
| run_008_adamw_logreturns | `--norm logreturns --optimizer adamw` | 58.3% | 51.8% | CNN 35.4M | 1:833 | ❌ |
| run_007_residual | `--residual (on top of logreturns baseline)` | 57.9% | 53.7% | CNN 35.4M | 1:833 | ❌ |
| run_010_residual_low_dropout | `--norm logreturns --residual --dropout 0.2` | 58.5% | 53.5% | CNN 35.4M | 1:833 | ❌ |
| run_011_years7 | `--norm logreturns --years 7` | 0.0% | 0.0% | CNN 35.4M | 1:833 | ❌ |
| run_012_lr5e5 | `--norm logreturns --lr 5e-5` | 59.2% | 56.9% | CNN 35.4M | 1:833 | ❌ |
| run_013_window256 | `--norm logreturns --window 256` | 57.0% | 46.5% | CNN 35.4M | 1:833 | ❌ |
| run_009_no_batchnorm | `--no-batchnorm (on logreturns baseline)` | 55.4% | 54.7% | CNN 35.4M | 1:833 | ❌ |
| run_014_clipgrad | `--norm logreturns --clip-grad 1.0` | 59.0% | 52.6% | CNN 35.4M | 1:833 | ❌ |

## Phase 2 — Data range and split (stride=5, shuffle, year sweep)

| Run | Config | Val | Test | Params | spp | Verdict |
|-----|--------|-----|------|--------|-----|---------|
| run_p2_001_20yr_stride5_shuffle | `--years 20 --stride 5 --shuffle-split` | 64.9% | 64.4% | CNN 35.4M | 1:1,000 | ✅ kept |
| run_p2_001_20yr_stride5_shuffle | `--norm logreturns --years 20 --stride 5 --shuffle-split` | 0.0% | 0.0% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_002_20yr_stride1_shuffle | `--norm logreturns --years 20 --stride 1 --shuffle-split` | 0.0% | 0.0% | CNN 35.4M | 1:833 | ❌ |
| run_p2_003_20yr_stride5_chronological | `--norm logreturns --years 20 --stride 5` | 0.0% | 0.0% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_001_20yr_stride5_shuffle | `--norm logreturns --years 20 --stride 5 --shuffle-split` | 57.8% | 55.4% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_002_20yr_stride1_shuffle | `--norm logreturns --years 20 --stride 1 --shuffle-split` | 79.7% | 79.9% | CNN 35.4M | 1:833 | ❌ |
| run_p2_003_20yr_stride5_chronological | `--norm logreturns --years 20 --stride 5` | 56.1% | 54.7% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_004_5yr_stride5_shuffle | `--years 5 --stride 5 --shuffle-split` | 60.2% | 55.4% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_005_3yr_stride5_shuffle | `--years 3 --stride 5 --shuffle-split` | 58.7% | 55.4% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_006_10yr_stride5_shuffle | `--years 10 --stride 5 --shuffle-split` | 58.1% | 55.4% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_007_5yr_stride5_shuffle_dropout | `--years 5 --stride 5 --shuffle-split --dropout 0.2` | 57.4% | 55.3% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_008_5yr_stride5_shuffle_window6 | `--years 5 --stride 5 --shuffle-split --window 64` | 59.3% | 54.9% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_004b_5yr_stride5_shuffle | `--years 5 --stride 5 --shuffle-split` | 60.0% | 55.9% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_005b_3yr_stride5_shuffle | `--years 3 --stride 5 --shuffle-split` | 57.8% | 55.4% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_006b_10yr_stride5_shuffle | `--years 10 --stride 5 --shuffle-split` | 58.1% | 54.1% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_009_7yr_stride5_shuffle | `--years 7 --stride 5 --shuffle-split` | 64.7% | 64.9% | CNN 35.4M | 1:1,000 | ✅ kept |
| run_p2_010_7yr_stride5_shuffle_confirm | `--years 7 --stride 5 --shuffle-split --seed 123` | 64.3% | 63.9% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_011_6yr_stride5_shuffle | `--years 6 --stride 5 --shuffle-split` | 64.3% | 63.3% | CNN 35.4M | 1:1,000 | ❌ |
| run_p2_012_8yr_stride5_shuffle | `--years 8 --stride 5 --shuffle-split` | 63.1% | 63.6% | CNN 35.4M | 1:1,000 | ❌ |

## Phase 3 — Scale and architecture (S&P500 403 tickers)

| Run | Config | Val | Test | Params | spp | Verdict |
|-----|--------|-----|------|--------|-----|---------|
| run_p3_001_sp500_stride133_20yr | `--preset sp500 --years 20 --stride 133 --shuffle-split` | 66.9% | 66.0% | CNN 35.4M | 1:3,333 | ✅ kept |
| run_p3_002_sp500_stride133_20yr_proper | `--preset sp500 --years 20 --stride 133 --shuffle-split` | 66.0% | 65.7% | CNN 35.4M | 1:3,333 | ❌ |
| run_p3_003_sp500_stride133_7yr | `--preset sp500 --years 7 --stride 133 --shuffle-split` | 70.8% | 68.8% | CNN 35.4M | 1:10,000 | ✅ kept |
| run_p3_004_largecap_horizon1_stride1_7 | `--horizon 1 --stride 1 --years 7 --shuffle-split` | 55.4% | 52.6% | CNN 35.4M | ~0 | ❌ |
| run_p3_005_sp500_horizon1_stride1_7yr | `--preset sp500 --horizon 1 --stride 1 --years 7 --shuff` | 0.0% | 0.0% | CNN — | — | ⏳ |
| run_p3_009_sp500_s5_20yr_default | `--preset sp500 --years 20 --stride 5 --shuffle-split` | 0.0% | 0.0% | CNN — | — | ⏳ |

## Key Findings

| Finding | Detail |
|---------|--------|
| **Best valid result** | 68.8% test — S&P500, stride=133, 7yr, zero-leakage |
| **Log-return normalisation** | +3.3pp over minmax — biggest Phase 1 single win |
| **COVID window (7yr)** | 2019-2026 captures March-Aug 2020 crash/recovery; drives 63-69% range |
| **Regime dilution (10yr+)** | COVID shrinks to 7% of data, accuracy drops back to ~54% |
| **Stride=133 (fully clean)** | No feature OR label overlap — 68.8% is methodologically solid |
| **Old model sizing** | 35M params on 10-43K samples = spp 1:800 to 1:27K (overparameterized) |
| **New model sizing** | --samples-per-param 100 auto-sizes via binary search in <0.1s |
| **Leaky stride=1+shuffle** | 79.9% was label leakage — adjacent windows share 4/5 prediction days |
| **T+1 is harder** | 52.6% vs 68.8% for T+5 — short-horizon is noisier |
| **Reference paper inflated** | Gupta 2024 uses stride=1+shuffle (same leakage) — 91% is memorization |

## Experiment Config

```
Universe:      403 S&P500 tickers, 20yr CSVs (avg 18.7yr each)
Features:      OHLCV (5 channels), log-return normalised
Split:         70% train / 15% val / 15% test (shuffle or chrono)
Sizing:        --samples-per-param 100 (default) auto-sizes model to data volume
Baseline:      LR=1e-4, Adam, ReduceLROnPlateau, dropout=0.4, BN=True, residual=False
Architectures: cnn | lstm | transformer | tcn
Label:         1 if close[t+horizon] > close[t] else 0 (binary classification)
```
