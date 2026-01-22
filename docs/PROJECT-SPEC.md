# CNN Stock Movement Prediction - Project Specification

## Overview

**Project:** CNN Stock Movement Predictor
**Status:** Planning
**Last Updated:** 2026-01-22

### Problem Statement

Predicting stock price movement direction (bullish/bearish) is valuable for identifying entry points for trades. Traditional methods rely on technical indicators or fundamental analysis, but deep learning approaches can potentially identify patterns in raw price data that humans miss. This project implements a 1D CNN approach from academic research to predict S&P 500 stock movements.

### Target Users

- Primary: The developer (learning deep learning + personal trading signals)
- Secondary: Anyone interested in reproducing the paper's results

### Success Criteria

1. **Model Training**: Successfully train CNN on S&P 500 historical data on CPU
2. **Accuracy Target**: Achieve >60% validation accuracy (paper reports up to 91%, but with GPU and more data)
3. **Prediction Capability**: Generate BUY/SELL signals for any S&P 500 stock
4. **Learning Outcome**: Understand CNN architecture, training dynamics, and financial time series preprocessing

---

## Requirements

### Core Requirements (Must Have)

1. **Data Pipeline**
   - Fetch historical OHLCV data for S&P 500 stocks via Yahoo Finance
   - Preprocess data: normalization, sliding window creation, train/val/test split
   - Handle missing data and stock delistings gracefully
   - Acceptance: Can load and preprocess data for any valid ticker

2. **CNN Model Architecture**
   - Implement 1D CNN matching paper specifications (8 Conv layers + 2 FC layers)
   - Adapt for 5 channels (adjusted OHLCV) instead of paper's 10 channels
   - Binary classification output (bullish/bearish)
   - Acceptance: Model compiles and produces valid probability outputs

3. **Training Pipeline**
   - Train on CPU with configurable batch size and epochs
   - Support multiple prediction horizons (T+5, T+30)
   - Implement early stopping to prevent overfitting
   - Acceptance: Model loss decreases over training epochs

4. **Model Persistence**
   - Save trained model weights and hyperparameters
   - Load previously trained models for inference
   - Acceptance: Can save, quit, reload, and get identical predictions

5. **Training Visualization**
   - Plot loss curves (training and validation)
   - Plot accuracy curves (training and validation)
   - Display training progress in real-time or after completion
   - Acceptance: Generate plots similar to paper's Figure 9/10

6. **Prediction Interface**
   - Jupyter notebook for running predictions
   - Input: ticker symbol, prediction horizon
   - Output: BUY/SELL signal with confidence score
   - Acceptance: Can generate prediction for any S&P 500 stock

### Secondary Requirements (Should Have)

1. **Simple Web Dashboard**
   - Basic web interface to input ticker and see prediction
   - Display recent predictions and confidence levels
   - Show model training history/metrics

2. **Batch Predictions**
   - Run predictions across all S&P 500 stocks
   - Rank by confidence score
   - Export results to CSV

3. **Multiple Model Support**
   - Train separate models for different horizons (T+5 vs T+30)
   - Compare model performances

### Non-Goals (Explicitly Out of Scope)

- **Real-time trading integration**: No broker API connections or automated trading
- **Backtesting framework**: No historical performance simulation
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
- **Data Fetching:** yfinance
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Notebook:** Jupyter
- **Web Dashboard (future):** Streamlit or Gradio
- **Environment:** venv

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

| Feature | Description |
|---------|-------------|
| Open | Adjusted opening price |
| High | Adjusted high price |
| Low | Adjusted low price |
| Close | Adjusted closing price |
| Volume | Trading volume |

### Model Input/Output

- **Input Shape:** `[batch_size, window_size, 5]` where window_size = 256 days
- **Output Shape:** `[batch_size, 2]` (softmax probabilities for bearish/bullish)
- **Label:** 1 if price increased after T days, 0 otherwise

### Directory Structure

```
CNNStockMarket/
├── docs/
│   ├── reference/
│   │   └── 2512.21804v1.pdf      # Original paper
│   ├── PROJECT-SPEC.md           # This file
│   ├── IMPLEMENTATION-PLAN.md    # Phased implementation
│   └── TASKS.md                  # Current tasks
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py            # Yahoo Finance data fetching
│   │   ├── preprocessor.py       # Normalization, windowing
│   │   └── dataset.py            # PyTorch Dataset class
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py                # CNN architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   └── metrics.py            # Accuracy, loss tracking
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py              # Training curves
│   └── utils/
│       ├── __init__.py
│       └── config.py             # Hyperparameters, paths
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_prediction.ipynb
├── models/                       # Saved model weights
├── data/                         # Cached stock data
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

## Open Questions

- [ ] What is the minimum training data history needed per stock? (Paper used 8-24 years)
- [ ] Should we use the same model for all stocks or train per-sector models?
- [ ] How to handle stocks with less than 256 days of history?
- [ ] What confidence threshold should trigger a BUY/SELL signal?

---

## References

1. Original Paper: `docs/reference/2512.21804v1.pdf` - "S&P 500 Stock's Movement Prediction using CNN" by Rahul Gupta
2. Base implementation mentioned in paper: https://github.com/philipxjm/Convolutional-Neural-Stock-Market-Technical-Analyser
