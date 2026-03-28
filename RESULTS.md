# AutoResearch Results

_Last updated: 2026-03-28 22:38 UTC_

> **spp** = train_samples / num_params. Old runs used a fixed 35M-param model (spp ≈ 1:800–1:27K, severely overparameterized). New runs use `--samples-per-param` so the model auto-sizes to the data.

## 🏆 Current Best

**run_p3_003_sp500_stride133_7yr**  
Test: **68.8%** | Val: 70.8% | CNN 35.4M params  
Config: `--preset sp500 --years 7 --stride 133 --shuffle-split`

## Phase 1 — Hyperparameter search (5yr, 54 largecap, stride=1, chrono)

| Run | Hypothesis | Config | Val | Test | Params | spp | Verdict | Conclusion |
|-----|-----------|--------|-----|------|--------|-----|---------|------------|
| run_001_baseline_gpu | Establish GPU baseline. Can the model train at all? | `GPU support added (was hardcoded CPU)` | 54.1% | 51.7% | CNN 35.4M | 1:833 | ✅ | Training works. 51.7% — barely above coin flip. Starting point. |
| run_002_lower_lr_1e4 | Does LR=1e-4 (vs 1e-3) improve generalization? | `train_experiment.py: default lr 1e-3 → 1e-4` | 54.1% | 54.1% | CNN 35.4M | 1:833 | ✅ | Yes. +2.4pp to 54.1%. Lower LR became permanent default. |
| run_003_scheduler_cosine | CosineAnnealingLR vs baseline plateau — does cosine decay help past 54%? | `train_experiment.py: --scheduler cosine (CosineAnn` | 53.1% | 52.0% | CNN 35.4M | 1:833 | ❌ | — |
| run_004_scheduler_none_attempt1 | No scheduler vs baseline plateau — is plateau even helping? | `runner: --scheduler none; new kernel slug cnn-stoc` | 0.0% | 0.0% | CNN 35.4M | 1:833 | ❌ | — |
| run_004_scheduler_none | No LR scheduler — is ReduceLROnPlateau helping? | `--scheduler none` | 55.5% | 52.0% | CNN 35.4M | 1:833 | ❌ | — |
| run_005_logreturns | Does log-return normalization outperform minmax scaling? | `--norm logreturns` | 58.5% | 57.4% | CNN 35.4M | 1:833 | ✅ | Yes. +3.3pp to 57.4%. Biggest single win in Phase 1. Now permanent default. |
| run_006_adamw | AdamW optimizer — decoupled weight decay vs Adam | `--optimizer adamw (tested on minmax baseline, not ` | 55.5% | 51.4% | CNN 35.4M | 1:833 | ❌ | — |
| run_008_adamw_logreturns | AdamW on logreturns baseline | `--norm logreturns --optimizer adamw` | 58.3% | 51.8% | CNN 35.4M | 1:833 | ❌ | — |
| run_007_residual | Residual skip connections on logreturns baseline | `--residual (on top of logreturns baseline)` | 57.9% | 53.7% | CNN 35.4M | 1:833 | ❌ | — |
| run_010_residual_low_dropout | Residual + dropout 0.2 on logreturns baseline | `--norm logreturns --residual --dropout 0.2` | 58.5% | 53.5% | CNN 35.4M | 1:833 | ❌ | — |
| run_011_years7 | More training data (7yr vs 5yr) — ~85K windows vs 60K | `--norm logreturns --years 7` | 0.0% | 0.0% | CNN 35.4M | 1:833 | ❌ | — |
| run_012_lr5e5 | Lower LR (5e-5) — log returns are smaller magnitude, may need gentler updates | `--norm logreturns --lr 5e-5` | 59.2% | 56.9% | CNN 35.4M | 1:833 | ❌ | — |
| run_013_window256 | Larger window (256 vs 128) — more temporal context per sample | `--norm logreturns --window 256` | 57.0% | 46.5% | CNN 35.4M | 1:833 | ❌ | — |
| run_009_no_batchnorm | Disable BatchNorm — is BN helping on logreturns baseline? | `--no-batchnorm (on logreturns baseline)` | 55.4% | 54.7% | CNN 35.4M | 1:833 | ❌ | — |
| run_014_clipgrad | Gradient clipping (max norm 1.0) — stabilize training | `--norm logreturns --clip-grad 1.0` | 59.0% | 52.6% | CNN 35.4M | 1:833 | ❌ | — |

## Phase 2 — Data range and split (stride=5, shuffle, year sweep)

| Run | Hypothesis | Config | Val | Test | Params | spp | Verdict | Conclusion |
|-----|-----------|--------|-----|------|--------|-----|---------|------------|
| run_p2_001_20yr_stride5_shuffle | Phase 2: 20yr data, stride=5 (non-overlapping labels), random split | `--years 20 --stride 5 --shuffle-split` | 64.9% | 64.4% | CNN 35.4M | 1:1,000 | ✅ | — |
| run_p2_001_20yr_stride5_shuffle | Phase 2 baseline: 20yr data, stride=5 (non-overlapping labels), random split | `--norm logreturns --years 20 --stride 5 --shuffle-` | 0.0% | 0.0% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_002_20yr_stride1_shuffle | Does more data (20yr) with stride=1 shuffle improve results? | `--norm logreturns --years 20 --stride 1 --shuffle-` | 0.0% | 0.0% | CNN 35.4M | 1:833 | ❌ | 79.9% — INVALID. Label leakage: adjacent windows share 4/5 prediction days. Memo |
| run_p2_003_20yr_stride5_chronologic | 20yr data stride=5 chronological — compare vs shuffle to quantify regime bias | `--norm logreturns --years 20 --stride 5` | 0.0% | 0.0% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_001_20yr_stride5_shuffle | Phase 2 baseline: 20yr data, stride=5 (non-overlapping labels), random split | `--norm logreturns --years 20 --stride 5 --shuffle-` | 57.8% | 55.4% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_002_20yr_stride1_shuffle | Does more data (20yr) with stride=1 shuffle improve results? | `--norm logreturns --years 20 --stride 1 --shuffle-` | 79.7% | 79.9% | CNN 35.4M | 1:833 | ❌ | 79.9% — INVALID. Label leakage: adjacent windows share 4/5 prediction days. Memo |
| run_p2_003_20yr_stride5_chronologic | 20yr data stride=5 chronological — compare vs shuffle to quantify regime bias | `--norm logreturns --years 20 --stride 5` | 56.1% | 54.7% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_004_5yr_stride5_shuffle | Reproduce accidental v31: 5yr data, stride=5, shuffle — confirmed best so far at | `--years 5 --stride 5 --shuffle-split` | 60.2% | 55.4% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_005_3yr_stride5_shuffle | Recency hypothesis: 3yr (2023-2026) post-rate-hike regime only. Tighter distribu | `--years 3 --stride 5 --shuffle-split` | 58.7% | 55.4% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_006_10yr_stride5_shuffle | Mid-range data: 10yr. Find the sweet spot between recency and data volume. | `--years 10 --stride 5 --shuffle-split` | 58.1% | 55.4% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_007_5yr_stride5_shuffle_drop | With stride=5 shuffle, fewer windows (~12K). Try lower dropout (0.2) — less regu | `--years 5 --stride 5 --shuffle-split --dropout 0.2` | 57.4% | 55.3% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_008_5yr_stride5_shuffle_wind | Shorter window (64 days) with stride=5 shuffle. May capture shorter-term pattern | `--years 5 --stride 5 --shuffle-split --window 64` | 59.3% | 54.9% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_004b_5yr_stride5_shuffle | RERUN with fixed --years filter: 5yr, stride=5, shuffle | `--years 5 --stride 5 --shuffle-split` | 60.0% | 55.9% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_005b_3yr_stride5_shuffle | RERUN with fixed --years filter: 3yr, stride=5, shuffle | `--years 3 --stride 5 --shuffle-split` | 57.8% | 55.4% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_006b_10yr_stride5_shuffle | RERUN with fixed --years filter: 10yr, stride=5, shuffle | `--years 10 --stride 5 --shuffle-split` | 58.1% | 54.1% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_009_7yr_stride5_shuffle | Does a 7yr window including COVID (2019-2026) outperform other year ranges? | `--years 7 --stride 5 --shuffle-split` | 64.7% | 64.9% | CNN 35.4M | 1:1,000 | ✅ | 64.9%. COVID regime (March-Aug 2020 crash+recovery) drives strong correlated sig |
| run_p2_010_7yr_stride5_shuffle_conf | Confirm 7yr result (64.9%) — rerun with different seed to check variance | `--years 7 --stride 5 --shuffle-split --seed 123` | 64.3% | 63.9% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_011_6yr_stride5_shuffle | Flank 7yr: try 6yr — is it the window or the data volume? | `--years 6 --stride 5 --shuffle-split` | 64.3% | 63.3% | CNN 35.4M | 1:1,000 | ❌ | — |
| run_p2_012_8yr_stride5_shuffle | Flank 7yr: try 8yr — find exact sweet spot | `--years 8 --stride 5 --shuffle-split` | 63.1% | 63.6% | CNN 35.4M | 1:1,000 | ❌ | — |

## Phase 3 — Scale, architecture, and regime (S&P500 403 tickers)

| Run | Hypothesis | Config | Val | Test | Params | spp | Verdict | Conclusion |
|-----|-----------|--------|-----|------|--------|-----|---------|------------|
| run_p3_001_sp500_stride133_20yr | Does expanding to full S&P500 (403 tickers) with stride=133 (zero overlap) impro | `--preset sp500 --years 20 --stride 133 --shuffle-s` | 66.9% | 66.0% | CNN 35.4M | 1:3,333 | ✅ | 66.0%. More tickers help. Stride=133 is methodologically clean. COVID still in d |
| run_p3_002_sp500_stride133_20yr_pro | Confirm p3_001 result with fresh run. | `--preset sp500 --years 20 --stride 133 --shuffle-s` | 66.0% | 65.7% | CNN 35.4M | 1:3,333 | ❌ | 65.7%. Confirms result is real, not a fluke. Slightly lower — run-to-run varianc |
| run_p3_003_sp500_stride133_7yr | S&P500 + zero-leakage stride + COVID window (7yr). Best conditions combined. | `--preset sp500 --years 7 --stride 133 --shuffle-sp` | 70.8% | 68.8% | CNN 35.4M | 1:10,000 | ✅ | 68.8%. Current best. Methodologically clean but COVID-regime dependent. Pre-COVI |
| run_p3_004_largecap_horizon1_stride | Is 1-day prediction (horizon=1) easier or harder than 5-day? | `--horizon 1 --stride 1 --years 7 --shuffle-split` | 55.4% | 52.6% | CNN 35.4M | ~0 | ❌ | 52.6%. Harder. 1-day moves are dominated by noise. 5-day trends are more predict |
| run_p3_009_sp500_s5_20yr_default | S&P 500 full, stride=5, 20yr, shuffle — 259K train with default 35M model. Solid | `--preset sp500 --years 20 --stride 5 --shuffle-spl` | 64.5% | 62.8% | CNN — | — | ❌ | — |
| run_p3_022_sp500_w20_s1_7yr_postcov | Post-COVID baseline: S&P500, window=20 (business month), stride=1, 7yr (2019-202 | `--preset sp500 --window 20 --stride 1 --years 7 --` | 0.0% | 0.0% | CNN — | — | ⏳ | — |
| run_p3_005_sp500_horizon1_stride1_7 | T+1 horizon, S&P 500, 7yr, stride=1, shuffle. ~1.4M windows, fully clean. Upper  | `--preset sp500 --horizon 1 --stride 1 --years 7 --` | 52.1% | 52.0% | CNN — | — | ❌ | — |
| run_p3_026_sp500_w60_s1_7yr_postcov | Post-COVID baseline: window=60 (business quarter), stride=1, 7yr (2019-2026), sp | `--preset sp500 --window 60 --stride 1 --years 7 --` | 67.9% | 68.0% | CNN — | — | ❌ | — |
| run_p3_009_sp500_s5_20yr_spp100 | S&P500 stride=5 20yr with spp=100. ~277K train → ~2,770 params. Properly sized — | `--preset sp500 --years 20 --stride 5 --shuffle-spl` | 54.5% | 55.2% | CNN — | — | ❌ | — |
| run_p3_019_largecap_s5_20yr_precovi | Match run_p2_001 (64.4%) but end 2019-12-31. 54 largecap, stride=5, 20yr (1999-2 | `--preset largecap-stable --years 20 --stride 5 --s` | 0.0% | 0.0% | CNN — | — | ❌ | — |
| run_p3_023_sp500_w20_s1_7yr_precovi | Pre-COVID spp-matched: window=20, stride=1, 7yr end=2019-12-31, spp=10. Is COVID | `--preset sp500 --window 20 --stride 1 --years 7 --` | 0.0% | 0.0% | CNN — | — | ❌ | — |
| run_p3_024_sp500_w20_s1_7yr_precovi | Pre-COVID fixed-params: same data as p3_023 but model pinned to p3_022 param cou | `--preset sp500 --window 20 --stride 1 --years 7 --` | 0.0% | 0.0% | CNN — | — | ❌ | — |
| run_p3_025_largecap_w20_s1_7yr_prec | Largecap pre-COVID: window=20, stride=1, 7yr end=2019-12-31, spp=10. ~65K train  | `--preset largecap-stable --window 20 --stride 1 --` | 0.0% | 0.0% | CNN — | — | ❌ | — |
| run_p3_027_sp500_w60_s1_7yr_precovi | Pre-COVID spp-matched: window=60, stride=1, 7yr end=2019-12-31, spp=10. Pairs wi | `--preset sp500 --window 60 --stride 1 --years 7 --` | 0.0% | 0.0% | CNN — | — | ❌ | — |
| run_p3_019_largecap_s5_20yr_precovi | Match run_p2_001 (64.4%) but end 2019-12-31. 54 largecap, stride=5, 20yr (1999-2 | `--preset largecap-stable --years 20 --stride 5 --s` | 56.0% | 55.9% | CNN — | — | ❌ | — |
| run_p3_023_sp500_w20_s1_7yr_precovi | Pre-COVID spp-matched: window=20, stride=1, 7yr end=2019-12-31, spp=10. Is COVID | `--preset sp500 --window 20 --stride 1 --years 7 --` | 60.0% | 58.7% | CNN — | — | ❌ | — |
| run_p3_025_largecap_w20_s1_7yr_prec | Largecap pre-COVID: window=20, stride=1, 7yr end=2019-12-31, spp=10. ~65K train  | `--preset largecap-stable --window 20 --stride 1 --` | 57.7% | 58.4% | CNN — | — | ❌ | — |
| run_p3_027_sp500_w60_s1_7yr_precovi | Pre-COVID spp-matched: window=60, stride=1, 7yr end=2019-12-31, spp=10. Pairs wi | `--preset sp500 --window 60 --stride 1 --years 7 --` | 0.0% | 0.0% | CNN — | — | ❌ | — |
| run_p3_010_lstm_s5_20yr_spp100 | LSTM medium (2.2M) S&P500 stride=5 20yr. Sequential memory vs CNN local patterns | `--preset sp500 --years 20 --stride 5 --shuffle-spl` | 0.0% | 0.0% | CNN — | — | ❌ | — |
| run_p3_011_transformer_s5_20yr_spp1 | Transformer medium (548K) S&P500 stride=5 20yr. Attention over 128-day window. | `--preset sp500 --years 20 --stride 5 --shuffle-spl` | 0.0% | 0.0% | CNN — | — | ❌ | — |
| run_p3_012_tcn_s5_20yr_spp100 | TCN medium (581K) S&P500 stride=5 20yr. Causal dilated convolutions, full-year r | `--preset sp500 --years 20 --stride 5 --shuffle-spl` | 0.0% | 0.0% | CNN — | — | ❌ | — |
| run_p3_013_cnn_w20_h1_s1_7yr | T+1 window=20 (business month): ~488K train. Does 1-month lookback help 1-day pr | `--preset sp500 --horizon 1 --window 20 --stride 1 ` | 54.4% | 54.2% | CNN — | — | ❌ | — |
| run_p3_014_cnn_w60_h1_s1_7yr | T+1 window=60 (business quarter): ~479K train. 1-quarter lookback vs 1-month vs  | `--preset sp500 --horizon 1 --window 60 --stride 1 ` | 54.1% | 54.1% | CNN — | — | ❌ | — |
| run_p3_015_cnn_w128_h1_s1_7yr | T+1, window=128 (6mo): baseline T+1 with current window. Compare vs w30 and w90. | `--preset sp500 --horizon 1 --window 128 --stride 1` | 52.0% | 51.8% | CNN — | — | ❌ | — |
| run_p3_010_lstm_s5_20yr_spp100 | LSTM medium (2.2M) S&P500 stride=5 20yr. Sequential memory vs CNN local patterns | `--preset sp500 --years 20 --stride 5 --shuffle-spl` | 55.4% | 55.8% | CNN — | — | ❌ | — |
| run_p3_011_transformer_s5_20yr_spp1 | Transformer medium (548K) S&P500 stride=5 20yr. Attention over 128-day window. | `--preset sp500 --years 20 --stride 5 --shuffle-spl` | 55.7% | 56.1% | CNN — | — | ❌ | — |
| run_p3_012_tcn_s5_20yr_spp100 | TCN medium (581K) S&P500 stride=5 20yr. Causal dilated convolutions, full-year r | `--preset sp500 --years 20 --stride 5 --shuffle-spl` | 54.5% | 55.2% | CNN — | — | ❌ | — |

## Key Findings

| Finding | Detail |
|---------|--------|
| **Best valid result** | 68.8% test — S&P500, stride=133, 7yr, zero-leakage |
| **Log-return normalisation** | +3.3pp over minmax — biggest Phase 1 single win |
| **COVID window (7yr)** | 2019-2026 captures March-Aug 2020 crash/recovery; drives 63-69% range |
| **Regime dilution (10yr+)** | COVID shrinks to 7% of data, accuracy drops back to ~54% |
| **Stride=133 (fully clean)** | No feature OR label overlap — 68.8% is methodologically solid |
| **Old model sizing** | 35M params on 10-43K samples = spp 1:800 to 1:27K (overparameterized) |
| **New model sizing** | --samples-per-param auto-sizes via binary search in <0.1s |
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

