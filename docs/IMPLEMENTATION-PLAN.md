# CNN Stock Movement Prediction - Implementation Plan

## Overview

This document outlines the implementation phases for the CNN Stock Movement Predictor.

**Total Phases:** 7
**MVP Phase:** Phase 3 (working model with training visualization)
**Status:** Phases 1-5 Complete, Phases 6-7 Pending

**Completion Summary:**
- ✅ Phase 1: Data Pipeline (COMPLETE)
- ✅ Phase 2: CNN Model Architecture (COMPLETE)
- ✅ Phase 3: Training Pipeline/MVP (COMPLETE)
- ✅ Phase 4: Prediction & Inference (COMPLETE)
- ✅ Phase 5: Web Dashboard (COMPLETE)
- ⬚ Phase 6: Custom Data & Timeframes (PENDING)
- ⬚ Phase 7: Backtesting Framework (PENDING)

---

## Phase 1: Project Setup & Data Pipeline ✅ COMPLETE

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

- ✅ Working virtual environment with all dependencies
- ✅ Data fetcher that retrieves S&P 500 stock data
- ✅ Preprocessor that normalizes and creates sliding windows
- ✅ PyTorch Dataset ready for training
- ✅ Exploration notebook demonstrating the pipeline

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

## Phase 2: CNN Model Architecture ✅ COMPLETE

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

- ✅ Config module with all hyperparameters
- ✅ CNN model class matching paper architecture
- ✅ Model compiles and handles correct input shapes

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

## Phase 3: Training Pipeline (MVP) ✅ COMPLETE

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

- ✅ Complete training pipeline with validation
- ✅ Model checkpointing (save/load)
- ✅ Training visualization (loss/accuracy curves)
- ✅ Training notebook

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

## Phase 4: Prediction & Inference ✅ COMPLETE

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

- ✅ Prediction service for single and batch predictions
- ✅ Prediction notebook with visualization
- ✅ CSV export of ranked predictions

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

## Phase 5: Web Dashboard (Enhancement) ✅ COMPLETE

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

- ✅ Working Streamlit dashboard
- ✅ Deployable as standalone web app

### Phase 5 Verification

```bash
streamlit run app.py
# Open browser to localhost:8501
```

---

## Phase 6: Custom Data & Timeframes (Enhancement) ⬚ PENDING

### Objective

Enable training and prediction on arbitrary OHLCV data from CSV files with configurable timeframes, supporting assets beyond stocks (cryptocurrencies, futures, forex, commodities).

### Prerequisites

- Phase 3 complete (training pipeline)
- Phase 4 complete (prediction service)

### Background

The current implementation is limited to:
- Yahoo Finance as the only data source
- Daily OHLC data only
- Fixed 256-day window size
- Fixed T+5 and T+30 day prediction horizons

This phase removes these limitations to support:
- Any OHLCV data from CSV files
- Any timeframe (1-minute, 5-minute, hourly, daily, weekly)
- Configurable window sizes
- Configurable prediction horizons
- Any asset class (stocks, crypto, forex, futures, commodities)

### Tasks

#### 6.1 CSV Data Loader

- **Description:** Create module to load OHLCV data from user-provided CSV files
- **Files:**
  - `src/data/csv_loader.py`
- **Key Functions:**
  - `load_csv(file_path, column_mapping=None) -> pd.DataFrame`
  - `validate_ohlcv_data(df) -> bool`
  - `infer_timeframe(df) -> str` (detect data frequency from timestamps)
  - `resample_ohlcv(df, target_timeframe) -> pd.DataFrame`
- **Features:**
  - Flexible column mapping (e.g., map "close" to "Close", "vol" to "Volume")
  - Auto-detection of date/datetime columns
  - Support for common CSV formats (with/without headers, various delimiters)
  - Data validation (check for required columns, NaN handling, chronological order)
- **Acceptance:** Can load CSV with any column names and convert to standard OHLCV format
- **Dependencies:** 1.4

#### 6.2 Configurable Timeframe Support

- **Description:** Update config and preprocessor to support arbitrary timeframes
- **Files:**
  - `src/utils/config.py` (extend)
  - `src/data/preprocessor.py` (extend)
- **Key Changes:**
  - New config parameters:
    ```python
    TIMEFRAME = "1d"  # "1m", "5m", "15m", "1h", "4h", "1d", "1w"
    WINDOW_SIZE = 256  # Now means "256 periods" not "256 days"
    HORIZONS = [5, 30]  # Now means "5 periods" not "5 days"
    ```
  - Update `create_sliding_windows()` to accept configurable column names
  - Add timeframe-aware data validation
- **Acceptance:** Can create windows from 1-minute data with appropriate sizes
- **Dependencies:** 6.1

#### 6.3 Dynamic Window Size Configuration

- **Description:** Allow window size to be specified at runtime, not just in config
- **Files:**
  - `src/models/cnn.py` (extend)
  - `src/training/trainer.py` (extend)
- **Key Changes:**
  - Model accepts `window_size` parameter in constructor
  - Automatic calculation of flatten layer size based on window size
  - Training configuration includes window size
  - Model checkpoint stores window size for inference
- **Acceptance:** Can train model with window_size=128 for intraday or window_size=512 for longer-term
- **Dependencies:** 6.2

#### 6.3b Training Sample Mode (Overlapping vs Non-Overlapping)

- **Description:** Implement configurable training sample modes to support both overlapping (high sample count) and non-overlapping (independent samples) training
- **Files:**
  - `src/data/preprocessor.py` (extend)
  - `src/data/dataset.py` (extend)
  - `src/training/trainer.py` (extend)
  - `src/utils/config.py` (extend)
- **Key Classes:**
  - `SampleMode` - enum for OVERLAPPING, NON_OVERLAPPING, STRIDED
  - `SampleConfig` - configuration for sample generation
- **Key Functions:**
  - `create_sliding_windows(df, window_size, horizon, sample_mode, stride=1) -> Tuple[X, y]`
  - `estimate_sample_count(data_length, window_size, horizon, sample_mode, stride) -> int`
  - `validate_sample_sufficiency(n_samples, min_samples=100) -> bool`
- **Configuration Options:**
  ```python
  class SampleConfig:
      mode: SampleMode = SampleMode.OVERLAPPING  # or NON_OVERLAPPING, STRIDED
      stride: int = 1                             # For STRIDED mode (e.g., 10, 20, 50)
      min_samples: int = 100                      # Minimum samples required
      randomize_non_overlapping: bool = True      # Shuffle non-overlapping windows
  ```
- **Sample Modes Explained:**
  ```
  Data: [========================================] (1000 bars)
  Window size: 100, Horizon: 5

  OVERLAPPING (stride=1):
  ├─ Window 1: bars[0:100]   → label from bar[105]
  ├─ Window 2: bars[1:101]   → label from bar[106]
  ├─ Window 3: bars[2:102]   → label from bar[107]
  └─ ... (895 samples, 99% overlap between adjacent)

  STRIDED (stride=20):
  ├─ Window 1: bars[0:100]   → label from bar[105]
  ├─ Window 2: bars[20:120]  → label from bar[125]
  ├─ Window 3: bars[40:140]  → label from bar[145]
  └─ ... (~45 samples, 80% overlap between adjacent)

  NON_OVERLAPPING (stride=window_size+horizon):
  ├─ Window 1: bars[0:100]   → label from bar[105]
  ├─ Window 2: bars[105:205] → label from bar[210]
  ├─ Window 3: bars[210:310] → label from bar[315]
  └─ ... (~9 samples, 0% overlap, fully independent)
  ```
- **Data Volume Guidance:**
  ```python
  # Automatic mode suggestion based on available data
  def suggest_sample_mode(data_length, window_size, horizon, min_samples=100):
      non_overlap_samples = data_length // (window_size + horizon)
      if non_overlap_samples >= min_samples:
          return SampleMode.NON_OVERLAPPING  # Sufficient for independent samples
      elif non_overlap_samples >= min_samples // 2:
          return SampleMode.STRIDED, stride=10  # Partial overlap compromise
      else:
          return SampleMode.OVERLAPPING  # Need maximum samples

  # Example calculations:
  # Daily data, 10 years (2520 bars), window=256, horizon=5:
  #   Non-overlapping: 2520/261 ≈ 9 samples (NOT ENOUGH)
  #   → Use OVERLAPPING or add more data (more tickers, more years)

  # Minute data, 1 year (98,280 bars), window=256, horizon=5:
  #   Non-overlapping: 98280/261 ≈ 376 samples (SUFFICIENT)
  #   → Can use NON_OVERLAPPING for independent samples

  # Daily data, 50 tickers, 10 years each (126,000 bars), window=256, horizon=5:
  #   Non-overlapping: 126000/261 ≈ 482 samples (SUFFICIENT)
  #   → Can use NON_OVERLAPPING
  ```
- **Comparison Mode:**
  ```python
  # Train both modes and compare
  def compare_sample_modes(data, config):
      # Train with overlapping
      model_overlap = train(data, sample_mode=OVERLAPPING)
      metrics_overlap = evaluate(model_overlap, test_data)

      # Train with non-overlapping (if sufficient samples)
      model_non_overlap = train(data, sample_mode=NON_OVERLAPPING)
      metrics_non_overlap = evaluate(model_non_overlap, test_data)

      return ComparisonResult(
          overlapping=metrics_overlap,
          non_overlapping=metrics_non_overlap,
          recommendation=...
      )
  ```
- **Acceptance:** Can train models with different sample modes and compare performance
- **Dependencies:** 6.2, 6.3

#### 6.4 Unified Data Interface

- **Description:** Create unified interface that works with both Yahoo Finance and CSV files
- **Files:**
  - `src/data/data_source.py` (new)
- **Key Classes:**
  - `DataSource` - abstract base class
  - `YahooFinanceSource(DataSource)` - existing Yahoo Finance fetching
  - `CSVSource(DataSource)` - load from CSV files
  - `DataSourceFactory` - create appropriate source based on config
- **Key Functions:**
  - `get_data(source_config) -> pd.DataFrame`
  - `prepare_training_data(source, window_size, horizon) -> Tuple[X, y]`
- **Acceptance:** Same training code works regardless of data source
- **Dependencies:** 6.1, 6.2

#### 6.5 Multi-Asset Configuration Presets

- **Description:** Provide sensible default configurations for different asset classes
- **Files:**
  - `src/utils/presets.py` (new)
- **Key Presets:**
  ```python
  PRESETS = {
      "stock_daily": {
          "window_size": 256,
          "horizons": [5, 30],
          "timeframe": "1d",
          "trading_days_ratio": 0.67  # ~252 trading days/year
      },
      "crypto_hourly": {
          "window_size": 168,  # 1 week of hourly data
          "horizons": [6, 24],  # 6-hour and 24-hour predictions
          "timeframe": "1h",
          "trading_days_ratio": 1.0  # 24/7 trading
      },
      "crypto_daily": {
          "window_size": 256,
          "horizons": [5, 30],
          "timeframe": "1d",
          "trading_days_ratio": 1.0
      },
      "forex_4h": {
          "window_size": 180,  # ~30 days of 4h data
          "horizons": [6, 30],  # 24h and 5-day predictions
          "timeframe": "4h",
          "trading_days_ratio": 0.71  # 24/5 market
      },
      "intraday_1m": {
          "window_size": 390,  # Full trading day (6.5 hours)
          "horizons": [15, 60],  # 15-min and 1-hour predictions
          "timeframe": "1m",
          "trading_days_ratio": 0.67
      }
  }
  ```
- **Acceptance:** Can load preset and have all config values set appropriately
- **Dependencies:** 6.2

#### 6.6 Updated Training Notebook for Custom Data

- **Description:** Notebook demonstrating training on custom CSV data
- **Files:**
  - `notebooks/04_custom_data_training.ipynb`
- **Features:**
  - Load data from CSV file
  - Configure timeframe and window size
  - Train model on custom data
  - Compare results with different configurations
  - Example with cryptocurrency data
- **Acceptance:** Can train model on user-provided 1-minute crypto data
- **Dependencies:** 6.4, 6.5

#### 6.7 Updated Dashboard for Custom Data

- **Description:** Extend Streamlit dashboard to support custom data uploads
- **Files:**
  - `app.py` (extend)
- **Features:**
  - CSV file upload widget
  - Column mapping interface
  - Timeframe selection
  - Model selection (different models for different timeframes)
  - Asset type indicator (stock/crypto/forex/futures)
- **Acceptance:** Can upload CSV, configure, and get predictions in dashboard
- **Dependencies:** 6.4, 6.6

#### 6.8 Data Format Documentation

- **Description:** Document supported CSV formats and data requirements
- **Files:**
  - `docs/DATA-FORMATS.md` (new)
- **Content:**
  - Required columns and accepted variations
  - Date/time format requirements
  - Example CSV files for each asset class
  - Troubleshooting common data issues
  - Recommended data sources for different assets
- **Acceptance:** User can prepare their CSV file following documentation
- **Dependencies:** 6.1

### Phase 6 Deliverables

- ⬚ CSV data loader with flexible column mapping
- ⬚ Configurable timeframe support (1m to 1w)
- ⬚ Dynamic window size configuration (user-specified, not fixed)
- ⬚ Training sample mode selection (overlapping, strided, non-overlapping)
- ⬚ Sample count estimation and sufficiency validation
- ⬚ Mode comparison workflow (train both, compare performance)
- ⬚ Unified data interface for multiple sources
- ⬚ Asset-class presets (stocks, crypto, forex, futures)
- ⬚ Custom data training notebook
- ⬚ Enhanced dashboard with CSV upload and training mode selection
- ⬚ Data format documentation

### Phase 6 Verification

```bash
# Test CSV loading
python -c "
from src.data.csv_loader import load_csv, validate_ohlcv_data

df = load_csv('data/my_btc_1h.csv', column_mapping={
    'timestamp': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
print(f'Loaded {len(df)} rows')
print(f'Valid OHLCV: {validate_ohlcv_data(df)}')
"

# Test training with custom timeframe
python -c "
from src.data.csv_loader import load_csv
from src.data.preprocessor import create_sliding_windows
from src.models.cnn import StockCNN
from src.utils.presets import PRESETS

# Load 1-hour crypto data
df = load_csv('data/btc_1h.csv')
config = PRESETS['crypto_hourly']

X, y = create_sliding_windows(
    df,
    window_size=config['window_size'],
    horizon=config['horizons'][0]
)
print(f'Created {len(X)} samples with window_size={config[\"window_size\"]}')

model = StockCNN(window_size=config['window_size'])
print('Model initialized for custom window size')
"

# Test unified data interface
python -c "
from src.data.data_source import DataSourceFactory

# CSV source
csv_source = DataSourceFactory.create('csv', path='data/btc_1h.csv')
df_csv = csv_source.get_data()

# Yahoo Finance source (existing)
yf_source = DataSourceFactory.create('yahoo', ticker='AAPL', start='2020-01-01', end='2024-01-01')
df_yf = yf_source.get_data()

print(f'CSV data: {len(df_csv)} rows')
print(f'Yahoo data: {len(df_yf)} rows')
"
```

---

## Phase 7: Backtesting Framework (Enhancement) ⬚ PENDING

### Objective

Enable performance evaluation of trained models against historical unseen data, providing metrics to understand model effectiveness before using it for live predictions.

### Prerequisites

- Phase 3 complete (training pipeline)
- Phase 4 complete (prediction service)
- Trained model file available

### Background

After training a model, it's crucial to understand its real-world performance before relying on its predictions. Backtesting allows:
- Evaluation on data the model has never seen
- Simulation of trading performance following model signals
- Risk assessment through drawdown and volatility metrics
- Comparison between different models or configurations

### Tasks

#### 7.1 Backtesting Engine Core

- **Description:** Create the core backtesting engine to run predictions on historical data
- **Files:**
  - `src/backtesting/engine.py`
- **Key Classes:**
  - `BacktestEngine` - main backtesting orchestrator
  - `BacktestConfig` - configuration for backtest parameters
  - `BacktestResult` - container for backtest results
- **Key Functions:**
  - `run_backtest(model, data, config) -> BacktestResult`
  - `walk_forward_backtest(model, data, config) -> List[BacktestResult]`
- **Features:**
  - Chronological prediction simulation (no look-ahead bias)
  - Configurable prediction frequency (every day, every N periods)
  - Support for both single asset and portfolio backtesting
  - Walk-forward validation option
- **Acceptance:** Can run backtest on unseen data and get predictions timeline
- **Dependencies:** 4.1

#### 7.1b Randomized Period Backtesting

- **Description:** Implement randomized, non-overlapping period sampling for robust model evaluation
- **Files:**
  - `src/backtesting/engine.py` (extend)
  - `src/backtesting/period_sampler.py` (new)
- **Key Classes:**
  - `PeriodSampler` - samples non-overlapping test periods
  - `RandomizedBacktestConfig` - configuration for randomized backtesting
  - `RandomizedBacktestResult` - results with statistical analysis
- **Key Functions:**
  - `sample_periods(data, n_periods, period_length, train_end_idx, seed) -> List[Period]`
  - `run_randomized_backtest(model, data, config) -> RandomizedBacktestResult`
  - `calculate_period_metrics(results) -> PeriodMetrics`
- **Configuration Options:**
  ```python
  RandomizedBacktestConfig(
      n_periods=100,              # Number of test periods to sample
      period_length=None,         # Auto: horizon + 1 (signal bar + outcome bar)
      min_periods=50,             # Minimum periods required for valid test
      random_seed=42,             # For reproducibility
      benchmark_ticker=None,      # Compare against index (e.g., "SPY")
      confidence_level=0.95,      # For confidence interval calculation
      stratify_by_regime=False,   # Optional: ensure diverse market conditions
  )
  ```
- **Period Structure:**
  ```
  For a daily model with horizon=5:

  Period = [window_size bars for context] + [1 signal bar] + [horizon bars for outcome]

  Example with window_size=256, horizon=5:
  ├─ Bars 0-255: Historical context (input to model)
  ├─ Bar 256: Signal bar (model predicts from here)
  └─ Bar 261: Outcome bar (was prediction correct?)

  Each sampled period is fully independent (no overlap with other periods or training data)
  ```
- **Output Metrics:**
  ```python
  RandomizedBacktestResult(
      n_periods=100,
      win_rate=0.58,                    # % of correct predictions
      win_rate_ci=(0.48, 0.68),         # 95% confidence interval
      avg_return=0.008,                 # Average return when following signals
      avg_return_ci=(0.002, 0.014),     # 95% CI
      benchmark_return=0.003,           # Benchmark return over same periods
      excess_return=0.005,              # Model return - benchmark return
      periods_by_outcome={              # Distribution of outcomes
          'true_positive': 35,
          'true_negative': 23,
          'false_positive': 22,
          'false_negative': 20,
      },
      return_distribution=[...],        # List of returns for each period
      period_details=[...],             # Detailed info for each sampled period
  )
  ```
- **Data Separation:**
  - Strict separation: sampled periods must not overlap with training data
  - `train_end_idx` parameter specifies where training data ends
  - Periods sampled only from data after training window
  - Gap buffer option to ensure no label leakage (gap = horizon bars)
- **Acceptance:** Can run randomized backtest and get statistically valid performance metrics
- **Dependencies:** 7.1

#### 7.2 Performance Metrics Module

- **Description:** Calculate comprehensive performance metrics from backtest results
- **Files:**
  - `src/backtesting/metrics.py`
- **Key Metrics:**
  - **Prediction Accuracy:**
    - Overall accuracy (% correct predictions)
    - Precision (true positives / predicted positives)
    - Recall (true positives / actual positives)
    - F1 score
    - Confusion matrix
  - **Trading Performance (if following signals):**
    - Total return
    - Annualized return
    - Sharpe ratio
    - Sortino ratio
    - Maximum drawdown
    - Win rate
    - Profit factor
    - Average win/loss ratio
  - **Risk Metrics:**
    - Volatility (daily/annualized)
    - Value at Risk (VaR)
    - Calmar ratio
- **Key Functions:**
  - `calculate_prediction_metrics(predictions, actuals) -> Dict`
  - `calculate_trading_metrics(returns) -> Dict`
  - `calculate_risk_metrics(returns) -> Dict`
- **Acceptance:** All standard backtesting metrics calculated correctly
- **Dependencies:** 7.1

#### 7.3 Trade Simulator

- **Description:** Simulate trades based on model signals to calculate returns
- **Files:**
  - `src/backtesting/simulator.py`
- **Key Classes:**
  - `TradeSimulator` - simulates trading based on signals
  - `Position` - represents an open position
  - `Trade` - represents a completed trade
- **Key Functions:**
  - `simulate_trades(signals, prices, config) -> List[Trade]`
  - `calculate_equity_curve(trades, initial_capital) -> pd.Series`
- **Features:**
  - Long-only or long/short strategies
  - Configurable position sizing (fixed, percentage, Kelly criterion)
  - Transaction cost modeling (commission, slippage)
  - Configurable confidence threshold for taking trades
  - Hold period based on prediction horizon
- **Acceptance:** Can simulate realistic trading with costs
- **Dependencies:** 7.1

#### 7.4 Backtest Visualization

- **Description:** Create visualizations for backtest results
- **Files:**
  - `src/backtesting/plots.py`
- **Key Plots:**
  - Equity curve over time
  - Drawdown chart
  - Prediction accuracy over time (rolling window)
  - Confusion matrix heatmap
  - Returns distribution histogram
  - Monthly/yearly returns heatmap
  - Signal timeline with price overlay
- **Key Functions:**
  - `plot_equity_curve(equity_curve, benchmark=None)`
  - `plot_drawdown(equity_curve)`
  - `plot_confusion_matrix(predictions, actuals)`
  - `plot_returns_distribution(returns)`
  - `plot_monthly_returns(returns)`
  - `plot_signal_timeline(signals, prices)`
- **Acceptance:** Generate comprehensive visual report of backtest
- **Dependencies:** 7.2, 7.3

#### 7.4b Randomized Period Visualization

- **Description:** Create visualizations specific to randomized period backtesting
- **Files:**
  - `src/backtesting/plots.py` (extend)
- **Key Plots:**
  - **Return Distribution Histogram:** Distribution of returns across all sampled periods
  - **Model vs Benchmark Scatter:** Each point = one period, x=benchmark return, y=model return
  - **Win Rate with CI:** Bar chart showing win rate with error bars for confidence interval
  - **Period Timeline:** Show sampled periods on price chart to visualize coverage
  - **Cumulative Performance:** Simulated equity if all periods were traded sequentially
  - **Regime Analysis:** Performance breakdown by market condition (if stratified)
- **Key Functions:**
  - `plot_period_returns_histogram(results, benchmark_results=None)`
  - `plot_model_vs_benchmark_scatter(model_returns, benchmark_returns)`
  - `plot_win_rate_ci(win_rate, ci_lower, ci_upper)`
  - `plot_sampled_periods_timeline(periods, price_data)`
  - `plot_period_cumulative_return(period_returns)`
- **Acceptance:** Visual analysis of randomized period backtest results
- **Dependencies:** 7.1b, 7.4

#### 7.5 Backtest Report Generator

- **Description:** Generate comprehensive HTML/PDF reports from backtest results
- **Files:**
  - `src/backtesting/report.py`
- **Key Functions:**
  - `generate_report(backtest_result, output_path, format='html')`
  - `export_trades_csv(trades, output_path)`
  - `export_metrics_json(metrics, output_path)`
- **Report Sections:**
  - Executive summary (key metrics)
  - Prediction performance analysis
  - Trading performance analysis
  - Risk analysis
  - All visualizations
  - Trade log
- **Acceptance:** One-click generation of full backtest report
- **Dependencies:** 7.4

#### 7.6 Backtesting Notebook

- **Description:** Jupyter notebook demonstrating backtesting workflow
- **Files:**
  - `notebooks/05_backtesting.ipynb`
- **Features:**
  - Load trained model
  - Configure backtest parameters
  - Run backtest on test data
  - Analyze results with visualizations
  - Compare multiple models
  - Export report
- **Acceptance:** Complete backtesting workflow in notebook
- **Dependencies:** 7.5

#### 7.7 Dashboard Backtesting Tab

- **Description:** Add backtesting capabilities to Streamlit dashboard
- **Files:**
  - `app.py` (extend)
- **Features:**
  - Model selection for backtesting
  - Date range selection for backtest period
  - Ticker/data selection
  - Configuration options (costs, position sizing)
  - Results display with key metrics
  - Interactive charts
  - Report download option
- **Acceptance:** Can run and view backtest from web interface
- **Dependencies:** 7.5, 7.6

### Phase 7 Deliverables

- ⬚ Backtesting engine with walk-forward support
- ⬚ Randomized period backtesting with statistical analysis
- ⬚ Period sampler with configurable parameters (n_periods, period_length, seed)
- ⬚ Benchmark comparison on same sampled periods
- ⬚ Comprehensive performance metrics (accuracy, trading, risk, period-based)
- ⬚ Trade simulator with realistic cost modeling
- ⬚ Visualization suite for backtest analysis (including randomized period plots)
- ⬚ Automated report generation (HTML/PDF) with randomized period section
- ⬚ Backtesting notebook with both chronological and randomized modes
- ⬚ Dashboard integration with mode toggle

### Phase 7 Verification

```bash
# Test basic backtest
python -c "
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.prediction.predictor import Predictor

# Load model
predictor = Predictor('models/best_model_t5.pt')

# Configure backtest
config = BacktestConfig(
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_capital=100000,
    commission=0.001,  # 0.1%
    confidence_threshold=0.6
)

# Run backtest
engine = BacktestEngine(predictor)
result = engine.run_backtest('AAPL', config)

print(f'Accuracy: {result.metrics[\"accuracy\"]:.1%}')
print(f'Total Return: {result.metrics[\"total_return\"]:.1%}')
print(f'Sharpe Ratio: {result.metrics[\"sharpe_ratio\"]:.2f}')
print(f'Max Drawdown: {result.metrics[\"max_drawdown\"]:.1%}')
"

# Test report generation
python -c "
from src.backtesting.report import generate_report

# Assuming result from previous backtest
generate_report(result, 'reports/backtest_AAPL_2024.html', format='html')
print('Report generated successfully')
"

# Test via notebook
jupyter notebook notebooks/05_backtesting.ipynb
```

---

## Dependency Graph

```
Phase 1 (Data) ──→ Phase 2 (Model) ──→ Phase 3 (Training/MVP)
                                              │
                                              ▼
                                       Phase 4 (Prediction)
                                              │
                         ┌────────────────────┼──────────────────────┐
                         ▼                    ▼                      ▼
                  Phase 7 (Backtest)   Phase 5 (Dashboard)    Phase 6 (Custom Data)
                         │                    │                      │
                         │                    └──────────┬───────────┘
                         │                               ▼
                         │                    Enhanced Dashboard (6.7)
                         │                               │
                         └───────────────────────────────┘
                                             ▼
                              Dashboard with Backtesting (7.7)
```

---

## Technical Decisions Log

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| Framework | PyTorch vs TensorFlow | PyTorch | User preference, more flexible for learning |
| Data source | Yahoo Finance vs Alpha Vantage vs Intrinio | Yahoo Finance + CSV | Free API + custom data flexibility |
| Channels | 5 (adjusted only) vs 10 (raw + adjusted) | 5 | Universal OHLCV standard across assets |
| Batch size | 250 (paper) vs 128 (reduced) | 128 | CPU memory constraints |
| Web framework | Streamlit vs Gradio vs Flask | Streamlit | Simple, good for ML dashboards |
| Window size | 256 (paper) vs configurable | Configurable | Support different timeframes and use cases |
| Timeframe support | Daily only vs configurable | Configurable | Enable intraday and multi-asset support |
| Data interface | Direct API vs abstraction layer | Abstraction layer | Unified interface for Yahoo + CSV sources |
| Asset presets | Single config vs presets | Presets | Sensible defaults for stocks, crypto, forex |
| Backtesting | Custom vs existing library (backtrader) | Custom | Tighter integration, simpler for our use case |
| Performance metrics | Basic vs comprehensive | Comprehensive | Full understanding of model performance |
| Report format | Text vs HTML/PDF | HTML + PDF | Professional reports with visualizations |

---

## CPU Training Considerations

Since training will be on CPU:

1. **Reduced batch size**: 128 instead of 250 to fit in memory
2. **Checkpointing**: Save every N epochs to recover from interruptions
3. **Subset training**: Start with 50-100 stocks, scale up once validated
4. **Patience**: Expect training to take several hours for full dataset
5. **Progress tracking**: Detailed logging so you can monitor overnight runs

---

## Custom Data & Timeframe Considerations (Phase 6)

When using custom CSV data and different timeframes:

1. **Window size selection**:
   - Daily data: 256 periods = ~1 year of trading days
   - Hourly data: 168 periods = 1 week (24/7 markets) or 256 periods = ~2 weeks
   - 1-minute data: 390 periods = 1 trading day (6.5 hours)
   - Choose window sizes that capture meaningful patterns for your timeframe

2. **Horizon selection**:
   - Should be proportional to your timeframe
   - Daily: T+5 (1 week), T+30 (1 month)
   - Hourly: T+6 (6 hours), T+24 (1 day)
   - 1-minute: T+15 (15 minutes), T+60 (1 hour)

3. **Data quality**:
   - Ensure no gaps in your data (especially for intraday)
   - Handle missing values appropriately
   - Verify chronological ordering
   - Check for outliers and data errors

4. **Asset-specific considerations**:
   - **Crypto**: 24/7 trading, high volatility, volume in native currency
   - **Forex**: 24/5 trading, lower volatility, sparse volume data
   - **Futures**: Contract rollovers, varying trading hours, different volume units
   - **Stocks**: Market hours only, corporate actions (splits, dividends)

5. **Model retraining**:
   - Models trained on daily stock data won't transfer well to 1-minute crypto
   - Train separate models for different timeframes and asset classes
   - Consider transfer learning for similar assets

6. **CSV format requirements**:
   - Must have timestamp/date column
   - Must have OHLCV columns (names can be mapped)
   - Chronological order (oldest first)
   - No duplicate timestamps

---

## Backtesting Considerations (Phase 7)

When running backtests to evaluate model performance:

1. **Avoiding look-ahead bias**:
   - Never use future data for predictions
   - Ensure train/test split is chronological
   - Use walk-forward validation for robust results

2. **Realistic cost modeling**:
   - Include commission costs (typically 0.1% for stocks, varies for crypto)
   - Account for slippage (price movement between signal and execution)
   - Consider bid-ask spread for less liquid assets

3. **Position sizing**:
   - Fixed position size for simple comparison
   - Percentage-based for realistic capital allocation
   - Kelly criterion for optimal growth (advanced)

4. **Interpreting metrics**:
   - **Accuracy alone is misleading**: A 55% accuracy can be profitable with good risk management
   - **Sharpe ratio > 1.0**: Generally considered good risk-adjusted returns
   - **Max drawdown**: Critical for understanding worst-case scenarios
   - **Win rate vs profit factor**: Both matter for sustainability

5. **Common pitfalls**:
   - Overfitting to backtest period (will fail on new data)
   - Survivorship bias (only testing on stocks that still exist)
   - Ignoring transaction costs (can turn profits into losses)
   - Not accounting for market regime changes

6. **Walk-forward validation**:
   - Train on period 1, test on period 2
   - Retrain on periods 1-2, test on period 3
   - More realistic than single train/test split
   - Reveals model stability over time

---

## Randomized Period Backtesting Considerations

The randomized period backtesting approach samples independent, non-overlapping test periods to provide robust performance metrics that are less biased by specific market regimes.

1. **Why randomized periods?**
   - Sequential testing on a single time period can be biased by market conditions
   - If test period is a bull market, bullish predictions appear artificially accurate
   - Randomized sampling across diverse periods gives more honest estimates
   - Provides statistical confidence intervals, not just point estimates

2. **Period structure:**
   ```
   For daily model with horizon=5:

   [---256 bars context---][signal bar][---5 bars---][outcome bar]

   - Context: Historical data fed to model (window_size bars)
   - Signal bar: The bar where prediction is made
   - Wait period: The horizon (5 bars for T+5 prediction)
   - Outcome bar: Where we check if prediction was correct
   ```

3. **Ensuring independence:**
   - Sampled periods must not overlap with each other
   - Sampled periods must not overlap with training data
   - Add gap buffer (= horizon) between training end and first test period
   - Track which data indices are used to prevent any leakage

4. **Benchmark comparison:**
   - Use the same sampled periods for benchmark (e.g., SPY, BTC)
   - This ensures apples-to-apples comparison
   - Report: "Model returned X% vs benchmark Y% over same periods"
   - Excess return = model return - benchmark return

5. **Statistical validity:**
   - Minimum ~50 periods for meaningful statistics
   - Report confidence intervals, not just point estimates
   - Use bootstrap sampling for robust CI calculation
   - Consider stratified sampling by market regime if data allows

6. **Interpreting results:**
   - Win rate of 55% with tight CI is better than 60% with wide CI
   - Small positive excess return that's consistent is valuable
   - Large variance in period returns suggests regime sensitivity
   - Compare against random baseline (50% win rate)

---

## Training Sample Mode Considerations

The training sample mode determines how training samples are created from the time series data.

1. **The overlap problem:**
   - Default sliding window (stride=1) creates highly correlated samples
   - Adjacent samples share 99.6% of their data (255/256 bars)
   - This inflates apparent sample count and can lead to overfitting
   - Model may memorize patterns specific to the training period

2. **When to use each mode:**
   ```
   OVERLAPPING (stride=1):
   ├─ When: Limited data (daily stocks, single ticker, few years)
   ├─ Pros: Maximum sample count
   ├─ Cons: Correlated samples, potential overfitting
   └─ Mitigations: Sample weighting, larger val/test splits

   STRIDED (stride=10-50):
   ├─ When: Moderate data, want balance of volume and independence
   ├─ Pros: Reduced correlation, still reasonable sample count
   ├─ Cons: Still some overlap, not fully independent
   └─ Use case: Daily data with multiple tickers or longer history

   NON_OVERLAPPING (stride=window_size+horizon):
   ├─ When: Abundant data (intraday, multi-year, multi-asset)
   ├─ Pros: Truly independent samples, no overlap
   ├─ Cons: Fewer samples, need sufficient data volume
   └─ Use case: Minute data, or aggregated multi-ticker daily data
   ```

3. **Calculating required data:**
   ```
   To get N non-overlapping samples:
   Required bars = N × (window_size + horizon)

   Examples (window=256, horizon=5, min_samples=100):
   - Need: 100 × 261 = 26,100 bars
   - Daily data: 26,100 / 252 ≈ 104 years (single ticker) ❌
   - Daily data: 26,100 / 252 / 50 ≈ 2 years (50 tickers) ✓
   - Hourly data (24/7): 26,100 / 8760 ≈ 3 years ✓
   - Minute data: 26,100 / 98280 ≈ 0.27 years (3 months) ✓
   ```

4. **Multi-asset aggregation:**
   - Combine data from multiple tickers to increase sample count
   - Each ticker contributes independent non-overlapping samples
   - Useful for daily data where single-ticker volume is insufficient
   - Ensure tickers are somewhat independent (different sectors)

5. **Comparison workflow:**
   - Train model A with overlapping samples
   - Train model B with non-overlapping samples (if sufficient data)
   - Evaluate both on the same held-out test set
   - Compare: accuracy, generalization gap, overfitting indicators
   - Non-overlapping often shows better generalization despite fewer samples

6. **Recommendation by use case:**
   ```
   Single stock, daily, 5 years:     → OVERLAPPING (insufficient data)
   50 stocks, daily, 5 years:        → NON_OVERLAPPING or STRIDED
   Crypto, hourly, 2 years:          → NON_OVERLAPPING
   Any asset, 1-minute, 6 months:    → NON_OVERLAPPING
   ```
