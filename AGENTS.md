# AGENTS.md

Instructions for AI coding agents working on this codebase.

## Project Overview

CNNStockMarket is a stock market prediction system using 1D Convolutional Neural Networks to predict S&P 500 price movements. Based on the paper "S&P 500 Stock's Movement Prediction using CNN" (arXiv:2512.21804).

**Goal:** Predict bullish/bearish price movements and generate BUY/SELL signals.

## Directory Structure

```
CNNStockMarket/
├── docs/
│   ├── PROJECT-SPEC.md         # Complete project specification
│   ├── IMPLEMENTATION-PLAN.md  # 5-phase implementation plan
│   ├── TASKS.md                # Current task tracker
│   └── reference/
│       └── 2512.21804v1.pdf    # Research paper
├── src/
│   ├── data/                   # Data fetching and preprocessing
│   ├── models/                 # CNN model architecture
│   ├── training/               # Training pipeline
│   ├── visualization/          # Plotting utilities
│   ├── prediction/             # Inference module
│   └── utils/                  # Config and helpers
├── notebooks/                  # Jupyter notebooks for exploration
├── models/                     # Saved model weights
├── data/                       # Cached stock data
└── app.py                      # Streamlit dashboard (Phase 5)
```

## Key Documentation

Before making changes, read:
1. `docs/PROJECT-SPEC.md` - Requirements and architecture decisions
2. `docs/IMPLEMENTATION-PLAN.md` - Phase breakdown and acceptance criteria
3. `docs/TASKS.md` - Current implementation status and next actions

## Setup Commands

```bash
# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, pandas, yfinance, matplotlib; print('OK')"
```

## Key Technologies

- **Python 3.12** - Language
- **PyTorch 2.x** - Deep learning framework
- **yfinance** - Stock data from Yahoo Finance
- **pandas/numpy** - Data manipulation
- **Jupyter** - Interactive notebooks
- **Streamlit** - Web dashboard (Phase 5)

## Model Architecture

Input: `[batch, 256, 5]` - 256-day windows with 5 OHLCV channels
Output: `[batch, 2]` - Softmax probabilities (bearish/bullish)

Key hyperparameters in `src/utils/config.py`:
- `WINDOW_SIZE = 256`
- `NUM_CHANNELS = 5`
- `BATCH_SIZE = 128`
- `LEARNING_RATE = 1e-3`
- `DROPOUT = 0.4`

## Coding Conventions

1. **Organize by function** - data/, models/, training/, etc.
2. **PyTorch patterns** - Models inherit `nn.Module`, data uses `Dataset`
3. **Configuration centralized** - Hyperparameters in `src/utils/config.py`
4. **Type hints** - Use throughout for clarity
5. **Docstrings** - Document public functions

## Important Constraints

- **CPU-only training** - No GPU; batch size limited to 64-128
- **Memory limits** - Keep under 8GB during training
- **Yahoo Finance rate limits** - Implement caching and retry logic
- **Chronological splits** - No random shuffling; maintain temporal order
- **Minimum 256 days history** - Required per stock for windowing

## Common Tasks

**Fetch stock data:**
```python
from src.data.fetcher import fetch_stock_data
df = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
```

**Run training notebook:**
```bash
jupyter notebook notebooks/02_training.ipynb
```

**Run prediction:**
```python
from src.prediction.predictor import Predictor
predictor = Predictor('models/best_model.pt')
signal = predictor.predict('AAPL')
```

**Start dashboard:**
```bash
streamlit run app.py
```

## Testing

Verify each module works with inline tests in the notebooks or:
```bash
python -c "from src.models.cnn import StockCNN; import torch; \
  model = StockCNN(); x = torch.randn(32, 256, 5); \
  print(model(x).shape)"  # Should print torch.Size([32, 2])
```

## Implementation Phases

1. **Phase 1** - Data Pipeline (fetcher, preprocessor, dataset)
2. **Phase 2** - CNN Model Architecture
3. **Phase 3** - Training Pipeline (MVP)
4. **Phase 4** - Prediction & Inference
5. **Phase 5** - Web Dashboard

Check `docs/TASKS.md` for current phase and next actions.
