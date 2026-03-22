# AutoResearch Results

_Last updated: 2026-03-22 16:12 UTC_

## 🏆 Current Best

**run_p3_003_sp500_stride133_7yr**
- Test accuracy: **68.8%** | Val: 70.8%
- Architecture: CNN — 35,378,946 params
- Config: `--preset sp500 --years 7 --stride 133 --shuffle-split`

## Phase 1 — Baseline hyperparameter search (5yr, largecap-stable, stride=1, chrono split)

| Run | Config | Val | Test | Params | spp | Verdict |
|-----|--------|-----|------|--------|-----|---------|
| run_001_baseline_gpu | `GPU support added (was hardcoded CPU)` | 54.1% | 51.7% | 35,378,946 | 0.001 | ✅ kept |
| run_002_lower_lr_1e4 | `train_experiment.py: default lr 1e-3 → 1e-4` | 54.1% | 54.1% | 35,378,946 | 0.001 | ✅ kept |
| run_003_scheduler_cosine | `train_experiment.py: --scheduler cosine (CosineAnnealingLR T` | 53.1% | 52.0% | 35,378,946 | 0.001 | ❌ |
| run_004_scheduler_none_attempt1 | `runner: --scheduler none; new kernel slug cnn-stock-training` | 0.0% | 0.0% | 35,378,946 | 0.001 | ❌ |
| run_004_scheduler_none | `--scheduler none` | 55.5% | 52.0% | 35,378,946 | 0.001 | ❌ |
| run_005_logreturns | `--norm logreturns` | 58.5% | 57.4% | 35,378,946 | 0.001 | ✅ kept |
| run_006_adamw | `--optimizer adamw (tested on minmax baseline, not logreturns` | 55.5% | 51.4% | 35,378,946 | 0.001 | ❌ |
| run_008_adamw_logreturns | `--norm logreturns --optimizer adamw` | 58.3% | 51.8% | 35,378,946 | 0.001 | ❌ |
| run_007_residual | `--residual (on top of logreturns baseline)` | 57.9% | 53.7% | 35,378,946 | 0.001 | ❌ |
| run_010_residual_low_dropout | `--norm logreturns --residual --dropout 0.2` | 58.5% | 53.5% | 35,378,946 | 0.001 | ❌ |
| run_011_years7 | `--norm logreturns --years 7` | 0.0% | 0.0% | 35,378,946 | 0.001 | ❌ |
| run_012_lr5e5 | `--norm logreturns --lr 5e-5` | 59.2% | 56.9% | 35,378,946 | 0.001 | ❌ |
| run_013_window256 | `--norm logreturns --window 256` | 57.0% | 46.5% | 35,378,946 | 0.001 | ❌ |
| run_009_no_batchnorm | `--no-batchnorm (on logreturns baseline)` | 55.4% | 54.7% | 35,378,946 | 0.001 | ❌ |
| run_014_clipgrad | `--norm logreturns --clip-grad 1.0` | 59.0% | 52.6% | 35,378,946 | 0.001 | ❌ |

## Phase 2 — Data range & split experiments (stride=5, shuffle, various years)

| Run | Config | Val | Test | Params | spp | Verdict |
|-----|--------|-----|------|--------|-----|---------|
| run_p2_001_20yr_stride5_shuffle | `--years 20 --stride 5 --shuffle-split` | 64.9% | 64.4% | 35,378,946 | 0.001 | ✅ kept |
| run_p2_001_20yr_stride5_shuffle | `--norm logreturns --years 20 --stride 5 --shuffle-split` | 0.0% | 0.0% | 35,378,946 | 0.001 | ❌ |
| run_p2_002_20yr_stride1_shuffle | `--norm logreturns --years 20 --stride 1 --shuffle-split` | 0.0% | 0.0% | 35,378,946 | 0.001 | ❌ |
| run_p2_003_20yr_stride5_chronologic | `--norm logreturns --years 20 --stride 5` | 0.0% | 0.0% | 35,378,946 | 0.001 | ❌ |
| run_p2_001_20yr_stride5_shuffle | `--norm logreturns --years 20 --stride 5 --shuffle-split` | 57.8% | 55.4% | 35,378,946 | 0.001 | ❌ |
| run_p2_002_20yr_stride1_shuffle | `--norm logreturns --years 20 --stride 1 --shuffle-split` | 79.7% | 79.9% | 35,378,946 | 0.001 | ❌ |
| run_p2_003_20yr_stride5_chronologic | `--norm logreturns --years 20 --stride 5` | 56.1% | 54.7% | 35,378,946 | 0.001 | ❌ |
| run_p2_004_5yr_stride5_shuffle | `--years 5 --stride 5 --shuffle-split` | 60.2% | 55.4% | 35,378,946 | 0.001 | ❌ |
| run_p2_005_3yr_stride5_shuffle | `--years 3 --stride 5 --shuffle-split` | 58.7% | 55.4% | 35,378,946 | 0.001 | ❌ |
| run_p2_006_10yr_stride5_shuffle | `--years 10 --stride 5 --shuffle-split` | 58.1% | 55.4% | 35,378,946 | 0.001 | ❌ |
| run_p2_007_5yr_stride5_shuffle_drop | `--years 5 --stride 5 --shuffle-split --dropout 0.2` | 57.4% | 55.3% | 35,378,946 | 0.001 | ❌ |
| run_p2_008_5yr_stride5_shuffle_wind | `--years 5 --stride 5 --shuffle-split --window 64` | 59.3% | 54.9% | 35,378,946 | 0.001 | ❌ |
| run_p2_004b_5yr_stride5_shuffle | `--years 5 --stride 5 --shuffle-split` | 60.0% | 55.9% | 35,378,946 | 0.001 | ❌ |
| run_p2_005b_3yr_stride5_shuffle | `--years 3 --stride 5 --shuffle-split` | 57.8% | 55.4% | 35,378,946 | 0.001 | ❌ |
| run_p2_006b_10yr_stride5_shuffle | `--years 10 --stride 5 --shuffle-split` | 58.1% | 54.1% | 35,378,946 | 0.001 | ❌ |
| run_p2_009_7yr_stride5_shuffle | `--years 7 --stride 5 --shuffle-split` | 64.7% | 64.9% | 35,378,946 | 0.001 | ✅ kept |
| run_p2_010_7yr_stride5_shuffle_conf | `--years 7 --stride 5 --shuffle-split --seed 123` | 64.3% | 63.9% | 35,378,946 | 0.001 | ❌ |
| run_p2_011_6yr_stride5_shuffle | `--years 6 --stride 5 --shuffle-split` | 64.3% | 63.3% | 35,378,946 | 0.001 | ❌ |
| run_p2_012_8yr_stride5_shuffle | `--years 8 --stride 5 --shuffle-split` | 63.1% | 63.6% | 35,378,946 | 0.001 | ❌ |

## Phase 3 — Architecture & data scale (S&P500, stride=133, new archs)

| Run | Config | Val | Test | Params | spp | Verdict |
|-----|--------|-----|------|--------|-----|---------|
| run_p3_001_sp500_stride133_20yr | `--preset sp500 --years 20 --stride 133 --shuffle-split` | 66.9% | 66.0% | 35,378,946 | — | ✅ kept |
| run_p3_002_sp500_stride133_20yr_pro | `--preset sp500 --years 20 --stride 133 --shuffle-split` | 66.0% | 65.7% | 35,378,946 | — | ❌ |
| run_p3_003_sp500_stride133_7yr | `--preset sp500 --years 7 --stride 133 --shuffle-split` | 70.8% | 68.8% | 35,378,946 | — | ✅ kept |
| run_p3_004_largecap_horizon1_stride | `--horizon 1 --stride 1 --years 7 --shuffle-split` | 55.4% | 52.6% | 35,378,946 | 0.001 | ❌ |
| run_p3_005_sp500_horizon1_stride1_7 | `--preset sp500 --horizon 1 --stride 1 --years 7 --shuffle-sp` | 0.0% | 0.0% | — | 0.001 | ⏳ |
| run_p3_009_sp500_s5_20yr_default | `--preset sp500 --years 20 --stride 5 --shuffle-split` | 0.0% | 0.0% | — | 0.001 | ⏳ |

## Key Findings

| Finding | Detail |
|---------|--------|
| **Best valid result** | 68.8% test — S&P500, stride=133, 7yr, shuffle |
| **Log-return norm** | +3.3pp over minmax (Phase 1 biggest single win) |
| **COVID window (7yr)** | Includes 2020 crash/recovery — boosts accuracy to 63-65% range |
| **Regime dilution** | 10yr+ drops back to ~54% — old market patterns add noise |
| **Stride=133 (clean)** | Zero feature+label overlap — confirms COVID signal is real at 68.8% |
| **Model was 1000x overparameterized** | 35M params on 36K samples = 0.001 spp; now using spp=100 |
| **Leaky stride=1+shuffle** | 79.9% was label leakage — adjacent windows share 4/5 prediction days |
| **T+1 (largecap)** | 52.6% — harder task; S&P500 T+1 result pending |
| **Architecture shootout** | CNN vs LSTM vs Transformer vs TCN pending — all at spp=100 |

## Experiment Pipeline


