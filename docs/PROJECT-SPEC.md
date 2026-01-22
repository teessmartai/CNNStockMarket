# CNN Stock Movement Prediction - Project Specification

## Overview

**Project:** CNN Stock Movement Predictor
**Status:** Phases 1-5 Complete | Phase 6 Planned
**Last Updated:** 2026-01-22

### Problem Statement

Predicting stock price movement direction (bullish/bearish) is valuable for identifying entry points for trades. Traditional methods rely on technical indicators or fundamental analysis, but deep learning approaches can potentially identify patterns in raw price data that humans miss. This project implements a 1D CNN approach from academic research to predict S&P 500 stock movements.

### Target Users

- Primary: The developer (learning deep learning + personal trading signals)
- Secondary: Anyone interested in reproducing the paper's results

### Success Criteria

1. **Model Training**: ✅ Successfully train CNN on S&P 500 historical data on CPU
2. **Accuracy Target**: ✅ Achieve >60% validation accuracy (paper reports up to 91%, but with GPU and more data)
3. **Prediction Capability**: ✅ Generate BUY/SELL signals for any S&P 500 stock
4. **Learning Outcome**: ✅ Understand CNN architecture, training dynamics, and financial time series preprocessing

**All success criteria met!**

---

## Requirements

### Core Requirements (Must Have)

1. **Data Pipeline** ✅ COMPLETE
   - ✅ Fetch historical OHLCV data for S&P 500 stocks via Yahoo Finance
   - ✅ Preprocess data: normalization, sliding window creation, train/val/test split
   - ✅ Handle missing data and stock delistings gracefully
   - ✅ Acceptance: Can load and preprocess data for any valid ticker

2. **CNN Model Architecture** ✅ COMPLETE
   - ✅ Implement 1D CNN matching paper specifications (8 Conv layers + 2 FC layers)
   - ✅ Adapt for 5 channels (adjusted OHLCV) instead of paper's 10 channels
   - ✅ Binary classification output (bullish/bearish)
   - ✅ Acceptance: Model compiles and produces valid probability outputs

3. **Training Pipeline** ✅ COMPLETE
   - ✅ Train on CPU with configurable batch size and epochs
   - ✅ Support multiple prediction horizons (T+5, T+30)
   - ✅ Implement early stopping to prevent overfitting
   - ✅ Acceptance: Model loss decreases over training epochs

4. **Model Persistence** ✅ COMPLETE
   - ✅ Save trained model weights and hyperparameters
   - ✅ Load previously trained models for inference
   - ✅ Acceptance: Can save, quit, reload, and get identical predictions

5. **Training Visualization** ✅ COMPLETE
   - ✅ Plot loss curves (training and validation)
   - ✅ Plot accuracy curves (training and validation)
   - ✅ Display training progress in real-time or after completion
   - ✅ Acceptance: Generate plots similar to paper's Figure 9/10

6. **Prediction Interface** ✅ COMPLETE
   - ✅ Jupyter notebook for running predictions
   - ✅ Input: ticker symbol, prediction horizon
   - ✅ Output: BUY/SELL signal with confidence score
   - ✅ Acceptance: Can generate prediction for any S&P 500 stock

### Secondary Requirements (Should Have)

1. **Simple Web Dashboard** ✅ COMPLETE
   - ✅ Basic web interface to input ticker and see prediction
   - ✅ Display recent predictions and confidence levels
   - ✅ Show model training history/metrics
   - ✅ Streamlit dashboard with 4 pages (Single, Batch, Top Signals, About)

2. **Batch Predictions** ✅ COMPLETE
   - ✅ Run predictions across all S&P 500 stocks
   - ✅ Rank by confidence score
   - ✅ Export results to CSV

3. **Multiple Model Support** ✅ COMPLETE
   - ✅ Train separate models for different horizons (T+5 vs T+30)
   - ✅ Compare model performances

### Phase 6 Requirements (Planned)

1. **Custom Data Loading** ⬚ PLANNED
   - ⬚ Load OHLCV data from user-provided CSV files
   - ⬚ Flexible column mapping (e.g., "close" → "Close", "vol" → "Volume")
   - ⬚ Auto-detection of date/datetime columns
   - ⬚ Data validation and quality checks
   - ⬚ Acceptance: Can train on any properly formatted CSV file

2. **Arbitrary Timeframe Support** ⬚ PLANNED
   - ⬚ Support 1-minute, 5-minute, 15-minute, hourly, 4-hour, daily, weekly data
   - ⬚ Configurable window sizes (not fixed to 256)
   - ⬚ Configurable prediction horizons
   - ⬚ Acceptance: Can train on 1-minute crypto data with appropriate window size

3. **Multi-Asset Support** ⬚ PLANNED
   - ⬚ Cryptocurrencies (24/7 trading)
   - ⬚ Forex pairs (24/5 trading)
   - ⬚ Futures contracts
   - ⬚ Commodities
   - ⬚ Acceptance: Model works on any OHLCV data regardless of asset class

4. **Asset Class Presets** ⬚ PLANNED
   - ⬚ Pre-configured settings for stocks, crypto, forex, futures
   - ⬚ Sensible default window sizes and horizons per asset type
   - ⬚ Trading day ratio configuration (stocks ~67%, crypto 100%, forex ~71%)
   - ⬚ Acceptance: Can select preset and have appropriate configuration applied

5. **Configurable Window Size** ⬚ PLANNED
   - ⬚ Window size configurable independently of timeframe
   - ⬚ User can specify any window size (e.g., 128, 256, 512, 1024)
   - ⬚ Presets provide sensible defaults but can be overridden
   - ⬚ Model architecture adapts automatically to window size
   - ⬚ Acceptance: Can train with any window size on any timeframe

6. **Training Sample Mode** ⬚ PLANNED
   - ⬚ **Overlapping mode** (default): Sliding window with stride=1, maximizes sample count
   - ⬚ **Non-overlapping mode**: Independent samples with no shared data between windows
   - ⬚ Configurable stride for partial overlap (e.g., stride=10 for 96% overlap vs 99.6%)
   - ⬚ Non-overlapping mode viable when data volume is sufficient (intraday, multi-asset, multi-year)
   - ⬚ Side-by-side comparison: train both modes and compare validation performance
   - ⬚ Automatic sample count estimation before training (warn if too few samples)
   - ⬚ Acceptance: Can toggle between modes and compare model performance

7. **Enhanced Dashboard** ⬚ PLANNED
   - ⬚ CSV file upload widget
   - ⬚ Column mapping interface
   - ⬚ Timeframe and asset type selection
   - ⬚ Window size configuration
   - ⬚ Training mode selection (overlapping vs non-overlapping)
   - ⬚ Model selection for different configurations
   - ⬚ Acceptance: Can upload CSV and get predictions in web interface

### Phase 7 Requirements (Planned)

1. **Backtesting Engine** ⬚ PLANNED
   - ⬚ Run trained model against historical unseen data
   - ⬚ Chronological simulation (no look-ahead bias)
   - ⬚ Walk-forward validation support
   - ⬚ Acceptance: Can evaluate model on any historical period

2. **Randomized Period Backtesting** ⬚ PLANNED
   - ⬚ Sample non-overlapping test periods randomly from available data
   - ⬚ Each period: signal generated on first bar, outcome evaluated after horizon bars
   - ⬚ Configurable period length (default: 2x model timeframe, e.g., 2 days for daily model)
   - ⬚ Ensure no overlap with training data (strict data separation)
   - ⬚ Benchmark comparison using same random periods (e.g., index return in same windows)
   - ⬚ Statistical metrics: win rate, average return, confidence intervals, distribution of outcomes
   - ⬚ Reduces market regime bias by sampling across diverse market conditions
   - ⬚ Acceptance: Can say "model was correct X% across N independent periods, avg gain Y% vs index Z%"

3. **Performance Metrics** ⬚ PLANNED
   - ⬚ Prediction accuracy (accuracy, precision, recall, F1)
   - ⬚ Trading metrics (returns, Sharpe ratio, max drawdown)
   - ⬚ Risk metrics (volatility, VaR, Calmar ratio)
   - ⬚ Randomized period metrics (win rate across periods, return distribution, CI bounds)
   - ⬚ Acceptance: Comprehensive metrics calculated for any backtest

4. **Trade Simulation** ⬚ PLANNED
   - ⬚ Simulate trades following model signals
   - ⬚ Configurable position sizing and costs
   - ⬚ Equity curve generation
   - ⬚ Acceptance: Realistic P&L simulation with transaction costs

5. **Backtest Visualization** ⬚ PLANNED
   - ⬚ Equity curve and drawdown charts
   - ⬚ Confusion matrix and accuracy over time
   - ⬚ Returns distribution and monthly heatmap
   - ⬚ Randomized period results: histogram of outcomes, scatter plot of model vs benchmark
   - ⬚ Acceptance: Visual analysis of backtest performance

6. **Report Generation** ⬚ PLANNED
   - ⬚ Automated HTML/PDF report generation
   - ⬚ Executive summary with key metrics
   - ⬚ Trade log export to CSV
   - ⬚ Randomized period analysis section with statistical summary
   - ⬚ Acceptance: One-click professional backtest report

7. **Dashboard Integration** ⬚ PLANNED
   - ⬚ Backtesting tab in Streamlit dashboard
   - ⬚ Interactive configuration and results
   - ⬚ Toggle between chronological and randomized period modes
   - ⬚ Report download option
   - ⬚ Acceptance: Full backtesting workflow in web interface

### Non-Goals (Explicitly Out of Scope)

- **Real-time trading integration**: No broker API connections or automated trading
- **Alternative models**: No LSTM, transformer, or ensemble implementations (focus on CNN only)
- **Sentiment analysis**: No news/social media data integration
- **Options/derivatives**: Stock movement only, no options pricing
- **Mobile app**: Desktop/web only
- **Multi-GPU training**: CPU only for this implementation

---

## Technical Context

### Tech Stack

- **Language:** Python 3.12 (LTS)
- **Deep Learning:** PyTorch 2.x
- **Data Fetching:** yfinance + CSV files (Phase 6)
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Notebook:** Jupyter
- **Web Dashboard:** Streamlit
- **Environment:** venv
- **Supported Assets:** Stocks (current), Crypto/Forex/Futures (Phase 6)
- **Supported Timeframes:** Daily (current), 1m-1w (Phase 6)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│              (Jupyter Notebook / Web Dashboard)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Prediction Service                        │
│         (Load model, preprocess input, return signal)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      CNN Model                               │
│    ┌─────────────────────────────────────────────────────┐  │
│    │ Input: [batch, window_size, 5]                      │  │
│    │   ↓                                                 │  │
│    │ Conv1D_1 (5→128) + ReLU + BatchNorm                │  │
│    │   ↓                                                 │  │
│    │ Conv1D_2 (128→256) + LeakyReLU + BatchNorm         │  │
│    │   ↓                                                 │  │
│    │ Conv1D_3-4 (256→256→512) + LeakyReLU + BatchNorm   │  │
│    │   ↓                                                 │  │
│    │ Conv1D_5-8 (512→1024) + LeakyReLU + BatchNorm      │  │
│    │   ↓                                                 │  │
│    │ Flatten + FC1 (→256) + Dropout                     │  │
│    │   ↓                                                 │  │
│    │ FC2 (256→2) + Softmax                              │  │
│    │   ↓                                                 │  │
│    │ Output: [bullish_prob, bearish_prob]               │  │
│    └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │ yfinance   │→ │ Preprocess │→ │ Sliding Window Dataset │ │
│  │ (fetch)    │  │ (normalize)│  │ (PyTorch DataLoader)   │ │
│  └────────────┘  └────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Input Data Format

| Feature | Description | Notes |
|---------|-------------|-------|
| Open | Opening price | Adjusted prices for stocks |
| High | High price | |
| Low | Low price | |
| Close | Closing price | Used for label generation |
| Volume | Trading volume | Units vary by asset (shares, coins, contracts) |

**Phase 6 Flexibility:**
- Column names can be mapped (e.g., "close" → "Close")
- Works with any OHLCV data source (Yahoo Finance, Binance, custom CSVs)
- Volume units don't matter (normalized per-window)

### Model Input/Output

- **Input Shape:** `[batch_size, window_size, 5]`
  - Current: window_size = 256 (daily data)
  - Phase 6: Configurable (e.g., 168 for hourly crypto, 390 for 1-minute intraday)
- **Output Shape:** `[batch_size, 2]` (softmax probabilities for bearish/bullish)
- **Label:** 1 if price increased after T periods, 0 otherwise

### Directory Structure

```
CNNStockMarket/
├── docs/
│   ├── reference/
│   │   └── 2512.21804v1.pdf      # Original paper
│   ├── PROJECT-SPEC.md           # This file
│   ├── IMPLEMENTATION-PLAN.md    # Phased implementation
│   ├── TASKS.md                  # Current tasks
│   └── DATA-FORMATS.md           # CSV format documentation (Phase 6)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py            # Yahoo Finance data fetching
│   │   ├── preprocessor.py       # Normalization, windowing
│   │   ├── dataset.py            # PyTorch Dataset class
│   │   ├── csv_loader.py         # CSV file loading (Phase 6)
│   │   └── data_source.py        # Unified data interface (Phase 6)
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py                # CNN architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   └── metrics.py            # Accuracy, loss tracking
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py          # Inference interface
│   ├── backtesting/              # Phase 7
│   │   ├── __init__.py
│   │   ├── engine.py             # Backtesting engine
│   │   ├── metrics.py            # Performance metrics
│   │   ├── simulator.py          # Trade simulator
│   │   ├── plots.py              # Backtest visualizations
│   │   └── report.py             # Report generation
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py              # Training curves
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Hyperparameters, paths
│       └── presets.py            # Asset class presets (Phase 6)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   ├── 03_prediction.ipynb
│   ├── 04_custom_data_training.ipynb  # Phase 6
│   └── 05_backtesting.ipynb           # Phase 7
├── models/                       # Saved model weights
├── data/                         # Cached stock data & custom CSVs
├── reports/                      # Generated backtest reports (Phase 7)
├── requirements.txt
└── README.md
```

---

## Constraints

### Technical Constraints

- **CPU-only training**: No GPU available; expect slower training times
- **Memory limits**: May need to limit batch size (64-128) to fit in RAM
- **Data availability**: Yahoo Finance only provides adjusted prices (5 channels vs paper's 10)
- **API rate limits**: Yahoo Finance may throttle requests; implement caching

### Phase 6 Considerations

- **Custom data quality**: User-provided CSV data may have gaps, errors, or inconsistent formatting
- **Timeframe appropriateness**: Not all window sizes work well for all timeframes
- **Asset-specific patterns**: Models trained on stocks may not transfer to crypto/forex
- **Volume semantics**: Volume units differ across assets (shares vs coins vs contracts)

### Performance Requirements

- Training should complete within reasonable time (hours, not days) for subset of stocks
- Inference should be fast (<1 second per prediction)
- Memory usage should stay under 8GB during training

### Data Considerations

- **Survivorship bias**: Current S&P 500 list doesn't include delisted companies
- **Look-ahead bias**: Ensure no future data leaks into training
- **Data quality**: Handle missing values, stock splits already adjusted by Yahoo

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CPU training too slow | Medium | High | Start with fewer stocks (50-100), reduce batch size, implement checkpointing |
| Yahoo Finance rate limiting | Medium | Medium | Cache downloaded data locally, implement retry logic with backoff |
| Model doesn't converge | Medium | High | Start with paper's hyperparameters, implement learning rate scheduling |
| Overfitting on limited data | High | Medium | Use proper train/val/test splits, implement early stopping, use dropout |
| 5 channels insufficient vs 10 | Low | Medium | 5 adjusted channels should capture essential patterns; monitor accuracy |
| Memory issues on full S&P 500 | Medium | Medium | Implement data generators, process stocks in batches |

---

## Resolved Questions

- ✅ What is the minimum training data history needed per stock? **Answer: 256 days minimum for windowing, ideally 2+ years for training**
- ✅ Should we use the same model for all stocks or train per-sector models? **Answer: Single model trained on multiple stocks works well**
- ✅ How to handle stocks with less than 256 days of history? **Answer: Skip or pad, currently handled with error messages**
- ✅ What confidence threshold should trigger a BUY/SELL signal? **Answer: Use softmax probabilities, higher probability = signal**

---

## References

1. Original Paper: `docs/reference/2512.21804v1.pdf` - "S&P 500 Stock's Movement Prediction using CNN" by Rahul Gupta
2. Base implementation mentioned in paper: https://github.com/philipxjm/Convolutional-Neural-Stock-Market-Technical-Analyser
