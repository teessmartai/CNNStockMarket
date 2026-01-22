# CNN Stock Movement Prediction - Implementation Plan

## Overview

This document outlines the implementation phases for the CNN Stock Movement Predictor.

**Total Phases:** 5
**MVP Phase:** Phase 3 (working model with training visualization)

---

## Phase 1: Project Setup & Data Pipeline

### Objective

Establish project structure, set up development environment, and build the data fetching/preprocessing pipeline.

### Prerequisites

- Python 3.12 installed
- Internet connection for fetching stock data

### Tasks

#### 1.1 Project Structure Setup

- **Description:** Create directory structure, initialize Python package, set up virtual environment
- **Files:**
  - `requirements.txt`
  - `src/__init__.py` and subpackage `__init__.py` files
  - `.gitignore`
- **Acceptance:** Can activate venv and import from `src`
- **Dependencies:** None

#### 1.2 Data Fetcher Implementation

- **Description:** Build module to fetch historical OHLCV data from Yahoo Finance for any ticker
- **Files:**
  - `src/data/fetcher.py`
- **Key Functions:**
  - `fetch_stock_data(ticker, start_date, end_date) -> pd.DataFrame`
  - `fetch_sp500_tickers() -> List[str]`
  - `fetch_multiple_stocks(tickers, start_date, end_date) -> Dict[str, pd.DataFrame]`
- **Acceptance:** Can fetch AAPL data for last 5 years, returns DataFrame with OHLCV columns
- **Dependencies:** 1.1

#### 1.3 Data Caching Layer

- **Description:** Cache fetched data locally to avoid repeated API calls
- **Files:**
  - `src/data/fetcher.py` (extend)
  - `data/` directory for cached CSV files
- **Acceptance:** Second fetch of same ticker loads from cache
- **Dependencies:** 1.2

#### 1.4 Data Preprocessor Implementation

- **Description:** Normalize data using min-max scaling, create sliding windows
- **Files:**
  - `src/data/preprocessor.py`
- **Key Functions:**
  - `normalize(df) -> pd.DataFrame`
  - `create_sliding_windows(df, window_size, horizon) -> Tuple[np.array, np.array]`
  - `create_labels(df, horizon) -> np.array` (1 if price up after T days, 0 otherwise)
- **Acceptance:** Given raw OHLCV, produces normalized windows of shape `[N, window_size, 5]`
- **Dependencies:** 1.2

#### 1.5 PyTorch Dataset Class

- **Description:** Create PyTorch Dataset and DataLoader for training
- **Files:**
  - `src/data/dataset.py`
- **Key Classes:**
  - `StockDataset(Dataset)` - handles single stock
  - `MultiStockDataset(Dataset)` - combines multiple stocks
- **Acceptance:** Can iterate DataLoader and get batches of `(X, y)` tensors
- **Dependencies:** 1.4

#### 1.6 Data Exploration Notebook

- **Description:** Jupyter notebook to explore data, visualize preprocessing steps
- **Files:**
  - `notebooks/01_data_exploration.ipynb`
- **Acceptance:** Notebook runs end-to-end, shows sample data and normalized windows
- **Dependencies:** 1.5

### Phase 1 Deliverables

- [ ] Working virtual environment with all dependencies
- [ ] Data fetcher that retrieves S&P 500 stock data
- [ ] Preprocessor that normalizes and creates sliding windows
- [ ] PyTorch Dataset ready for training
- [ ] Exploration notebook demonstrating the pipeline

### Phase 1 Verification

```bash
# Activate environment and run tests
source venv/bin/activate
python -c "from src.data import fetcher, preprocessor, dataset; print('Imports OK')"

# Fetch sample data
python -c "
from src.data.fetcher import fetch_stock_data
df = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
print(f'Fetched {len(df)} rows')
print(df.head())
"
```

---

## Phase 2: CNN Model Architecture

### Objective

Implement the 1D CNN architecture from the paper in PyTorch.

### Prerequisites

- Phase 1 complete
- Understanding of paper's model architecture (Section 4)

### Tasks

#### 2.1 Configuration Module

- **Description:** Centralize hyperparameters and model configuration
- **Files:**
  - `src/utils/config.py`
- **Key Config:**
  - `WINDOW_SIZE = 256`
  - `NUM_CHANNELS = 5`
  - `LEARNING_RATE = 1e-3`
  - `BATCH_SIZE = 128` (smaller for CPU)
  - `DROPOUT = 0.4`
  - `HORIZONS = [5, 30]`
- **Acceptance:** Config values importable and used consistently
- **Dependencies:** 1.1

#### 2.2 CNN Model Implementation

- **Description:** Implement the 8-layer Conv1D + 2 FC layer architecture
- **Files:**
  - `src/models/cnn.py`
- **Architecture:**
  ```
  Conv1D(5, 128, kernel=9) + ReLU + BatchNorm
  Conv1D(128, 256, kernel=9) + LeakyReLU + BatchNorm
  Conv1D(256, 256, kernel=9) + LeakyReLU + BatchNorm
  Conv1D(256, 512, kernel=9) + LeakyReLU + BatchNorm
  Conv1D(512, 1024, kernel=9) + LeakyReLU + BatchNorm
  Conv1D(1024, 1024, kernel=9) + LeakyReLU + BatchNorm
  Conv1D(1024, 1024, kernel=9) + LeakyReLU + BatchNorm
  Conv1D(1024, 1024, kernel=9) + LeakyReLU + BatchNorm
  Flatten
  Linear(→256) + Dropout
  Linear(256→2) + Softmax
  ```
- **Acceptance:** Model forward pass works with input shape `[batch, 256, 5]`
- **Dependencies:** 2.1

#### 2.3 Model Summary and Validation

- **Description:** Verify model architecture, count parameters, test forward pass
- **Files:**
  - Extend `src/models/cnn.py` with `summary()` method
- **Acceptance:** Print model summary showing layer shapes and param counts
- **Dependencies:** 2.2

### Phase 2 Deliverables

- [ ] Config module with all hyperparameters
- [ ] CNN model class matching paper architecture
- [ ] Model compiles and handles correct input shapes

### Phase 2 Verification

```bash
python -c "
import torch
from src.models.cnn import StockCNN
from src.utils.config import WINDOW_SIZE, NUM_CHANNELS

model = StockCNN()
x = torch.randn(32, WINDOW_SIZE, NUM_CHANNELS)
out = model(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {out.shape}')
print(f'Output sums to 1: {out[0].sum().item():.4f}')
"
```

---

## Phase 3: Training Pipeline (MVP)

### Objective

Implement complete training loop with validation, metrics tracking, visualization, and model saving.

### Prerequisites

- Phase 1 & 2 complete

### Tasks

#### 3.1 Metrics Tracking Module

- **Description:** Track and store training/validation loss and accuracy per epoch
- **Files:**
  - `src/training/metrics.py`
- **Key Classes:**
  - `MetricsTracker` - stores history, computes running averages
- **Acceptance:** Can record metrics and retrieve history as lists
- **Dependencies:** 2.2

#### 3.2 Training Loop Implementation

- **Description:** Implement training loop with validation, early stopping
- **Files:**
  - `src/training/trainer.py`
- **Key Features:**
  - Cross-entropy loss function
  - Adam optimizer
  - Train/validation split
  - Early stopping based on validation loss
  - Progress logging
- **Acceptance:** Can train for N epochs, see loss decreasing
- **Dependencies:** 3.1

#### 3.3 Model Checkpointing

- **Description:** Save model weights, optimizer state, and training history
- **Files:**
  - `src/training/trainer.py` (extend)
  - `models/` directory for saved weights
- **Key Functions:**
  - `save_checkpoint(model, optimizer, epoch, metrics, path)`
  - `load_checkpoint(path) -> (model, optimizer, epoch, metrics)`
- **Acceptance:** Can save mid-training, reload, and continue
- **Dependencies:** 3.2

#### 3.4 Visualization Module

- **Description:** Plot training curves (loss and accuracy over epochs)
- **Files:**
  - `src/visualization/plots.py`
- **Key Functions:**
  - `plot_loss_curves(train_loss, val_loss)`
  - `plot_accuracy_curves(train_acc, val_acc)`
  - `plot_training_summary(metrics)` - combined view
- **Acceptance:** Generates plots matching paper's Figure 9/10 style
- **Dependencies:** 3.1

#### 3.5 Training Notebook

- **Description:** Jupyter notebook for running and monitoring training
- **Files:**
  - `notebooks/02_training.ipynb`
- **Features:**
  - Load data for selected stocks
  - Initialize and train model
  - Display live or post-training plots
  - Save best model
- **Acceptance:** Can train model end-to-end from notebook
- **Dependencies:** 3.4

### Phase 3 Deliverables

- [ ] Complete training pipeline with validation
- [ ] Model checkpointing (save/load)
- [ ] Training visualization (loss/accuracy curves)
- [ ] Training notebook

### Phase 3 Verification

```bash
# Run a quick training test (few epochs, small dataset)
python -c "
from src.data.fetcher import fetch_stock_data
from src.data.preprocessor import prepare_data
from src.data.dataset import StockDataset
from src.models.cnn import StockCNN
from src.training.trainer import Trainer

# Quick test with minimal data
trainer = Trainer(model=StockCNN(), epochs=5)
# ... would need actual data setup
print('Trainer initialized successfully')
"
```

---

## Phase 4: Prediction & Inference

### Objective

Build prediction interface for generating BUY/SELL signals on new data.

### Prerequisites

- Phase 3 complete
- At least one trained model saved

### Tasks

#### 4.1 Prediction Service

- **Description:** Load trained model and generate predictions for any ticker
- **Files:**
  - `src/prediction/predictor.py`
- **Key Functions:**
  - `load_model(checkpoint_path) -> StockCNN`
  - `predict(model, ticker, horizon) -> (signal, confidence)`
  - `predict_batch(model, tickers, horizon) -> List[Prediction]`
- **Acceptance:** Given ticker, returns BUY/SELL with confidence %
- **Dependencies:** 3.3

#### 4.2 Prediction Notebook

- **Description:** Notebook for running predictions interactively
- **Files:**
  - `notebooks/03_prediction.ipynb`
- **Features:**
  - Load trained model
  - Input ticker symbol
  - Display prediction with confidence
  - Show recent price chart with prediction overlay
- **Acceptance:** Can predict on any S&P 500 stock
- **Dependencies:** 4.1

#### 4.3 Batch Prediction & Ranking

- **Description:** Run predictions across all S&P 500 and rank by confidence
- **Files:**
  - `src/prediction/predictor.py` (extend)
- **Key Functions:**
  - `rank_predictions(predictions) -> sorted list by confidence`
  - `export_predictions(predictions, path)` - save to CSV
- **Acceptance:** Generate ranked list of top BUY/SELL signals
- **Dependencies:** 4.1

### Phase 4 Deliverables

- [ ] Prediction service for single and batch predictions
- [ ] Prediction notebook with visualization
- [ ] CSV export of ranked predictions

### Phase 4 Verification

```bash
python -c "
from src.prediction.predictor import Predictor

predictor = Predictor('models/best_model_t5.pt')
signal, confidence = predictor.predict('AAPL', horizon=5)
print(f'AAPL T+5: {signal} ({confidence:.1%} confidence)')
"
```

---

## Phase 5: Web Dashboard (Enhancement)

### Objective

Create simple web interface for predictions (stretch goal).

### Prerequisites

- Phase 4 complete

### Tasks

#### 5.1 Streamlit Dashboard

- **Description:** Simple web UI using Streamlit
- **Files:**
  - `app.py` (root level)
- **Features:**
  - Ticker input dropdown (S&P 500 stocks)
  - Horizon selector (T+5, T+30)
  - Prediction display with confidence meter
  - Recent price chart
  - Top signals table
- **Acceptance:** `streamlit run app.py` shows working dashboard
- **Dependencies:** 4.2

#### 5.2 Dashboard Styling

- **Description:** Improve UI/UX with better styling
- **Files:**
  - `app.py` (extend)
- **Features:**
  - Color coding (green=BUY, red=SELL)
  - Confidence visualization
  - Loading states
- **Acceptance:** Dashboard looks polished
- **Dependencies:** 5.1

### Phase 5 Deliverables

- [ ] Working Streamlit dashboard
- [ ] Deployable as standalone web app

### Phase 5 Verification

```bash
streamlit run app.py
# Open browser to localhost:8501
```

---

## Dependency Graph

```
Phase 1 (Data) ──→ Phase 2 (Model) ──→ Phase 3 (Training/MVP)
                                              │
                                              ▼
                                       Phase 4 (Prediction)
                                              │
                                              ▼
                                       Phase 5 (Dashboard)
```

---

## Technical Decisions Log

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| Framework | PyTorch vs TensorFlow | PyTorch | User preference, more flexible for learning |
| Data source | Yahoo Finance vs Alpha Vantage vs Intrinio | Yahoo Finance | Free, reliable, `yfinance` library |
| Channels | 5 (adjusted only) vs 10 (raw + adjusted) | 5 | Yahoo only provides adjusted; simpler |
| Batch size | 250 (paper) vs 128 (reduced) | 128 | CPU memory constraints |
| Web framework | Streamlit vs Gradio vs Flask | Streamlit | Simple, good for ML dashboards |
| Window size | 256 (paper) | 256 | Match paper for reproducibility |

---

## CPU Training Considerations

Since training will be on CPU:

1. **Reduced batch size**: 128 instead of 250 to fit in memory
2. **Checkpointing**: Save every N epochs to recover from interruptions
3. **Subset training**: Start with 50-100 stocks, scale up once validated
4. **Patience**: Expect training to take several hours for full dataset
5. **Progress tracking**: Detailed logging so you can monitor overnight runs
