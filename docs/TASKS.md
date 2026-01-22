# CNN Stock Movement Prediction - Task Tracker

## Current Status

**Phase:** 6 - Custom Data & Timeframes (PLANNING)
**Last Updated:** 2026-01-22
**Status:** Phases 1-5 complete. Phase 6 planned for custom data and arbitrary timeframe support.

---

## Active Tasks (In Progress)

*Phase 6 implementation pending*

---

## Next Tasks (Ready to Start)

### Phase 6: Custom Data & Timeframes

#### ⬚ TASK-6.1: CSV Data Loader
- Create `src/data/csv_loader.py`
- Implement `load_csv(file_path, column_mapping=None)`
- Implement `validate_ohlcv_data(df)`
- Implement `infer_timeframe(df)`
- Implement `resample_ohlcv(df, target_timeframe)`
- Support flexible column mapping
- Handle various CSV formats and delimiters

#### ⬚ TASK-6.2: Configurable Timeframe Support
- Update `src/utils/config.py` with TIMEFRAME parameter
- Update `src/data/preprocessor.py` for configurable columns
- Add timeframe-aware data validation
- Support 1m, 5m, 15m, 1h, 4h, 1d, 1w timeframes

#### ⬚ TASK-6.3: Dynamic Window Size Configuration
- Update `src/models/cnn.py` to accept window_size parameter
- Auto-calculate flatten layer size based on window size
- Update `src/training/trainer.py` to store window size in checkpoints
- Support window sizes from 128 to 512+

#### ⬚ TASK-6.4: Unified Data Interface
- Create `src/data/data_source.py`
- Implement `DataSource` abstract base class
- Implement `YahooFinanceSource(DataSource)`
- Implement `CSVSource(DataSource)`
- Implement `DataSourceFactory`

#### ⬚ TASK-6.5: Multi-Asset Configuration Presets
- Create `src/utils/presets.py`
- Define presets for: stock_daily, crypto_hourly, crypto_daily, forex_4h, intraday_1m
- Include window_size, horizons, timeframe, trading_days_ratio per preset

#### ⬚ TASK-6.6: Custom Data Training Notebook
- Create `notebooks/04_custom_data_training.ipynb`
- Demonstrate CSV loading and training
- Show configuration for different timeframes
- Include cryptocurrency example

#### ⬚ TASK-6.7: Enhanced Dashboard for Custom Data
- Update `app.py` with CSV upload widget
- Add column mapping interface
- Add timeframe and asset type selection
- Add model selection for different configurations

#### ⬚ TASK-6.8: Data Format Documentation
- Create `docs/DATA-FORMATS.md`
- Document required columns and variations
- Include example CSVs for each asset class
- Add troubleshooting guide

---

## Future Enhancements (Post Phase 6)

- Fine-tuning model hyperparameters
- Backtesting framework
- Model ensemble approaches
- Transfer learning between asset classes


---

## Completed Tasks

### ✅ TASK-1.1: Project Structure Setup
**Completed:** 2026-01-22
- Created virtual environment with Python 3.12
- Created `requirements.txt` with all dependencies
- Set up directory structure (src/, notebooks/, models/, data/)
- Created `.gitignore`
- All dependencies installed and verified

### ✅ TASK-1.2: Data Fetcher Implementation
**Completed:** 2026-01-22
- Implemented `fetch_stock_data()` using yfinance
- Implemented `fetch_sp500_tickers()`
- Implemented `fetch_multiple_stocks()` with progress bar
- Added error handling and logging

### ✅ TASK-1.3: Data Caching Layer
**Completed:** 2026-01-22
- Added caching logic to avoid repeated API calls
- Cache files stored in `data/` directory
- Significant speedup on repeated fetches

### ✅ TASK-1.4: Data Preprocessor Implementation
**Completed:** 2026-01-22
- Implemented `normalize()` for min-max scaling
- Implemented `create_labels()` for binary classification
- Implemented `create_sliding_windows()` with 256-day windows
- Implemented `train_val_test_split()` with chronological splits

### ✅ TASK-1.5: PyTorch Dataset Class
**Completed:** 2026-01-22
- Created `StockDataset` class inheriting from `Dataset`
- Implemented `create_dataloaders()` for train/val/test
- Proper shuffling and batching configured

### ✅ TASK-1.6: Data Exploration Notebook
**Completed:** 2026-01-22
- Created `notebooks/01_data_exploration.ipynb`
- Visualizations of OHLCV data, normalization, and windowing
- Label distribution analysis
- DataLoader testing

### ✅ TASK-2.1: Configuration Module
**Completed:** 2026-01-22
- Created `src/utils/config.py`
- Centralized all hyperparameters (WINDOW_SIZE=256, BATCH_SIZE=128, etc.)
- Path configurations for data, models, and cache

### ✅ TASK-2.2: CNN Model Implementation
**Completed:** 2026-01-22
- Implemented 8-layer Conv1D architecture in `src/models/cnn.py`
- 2 fully connected layers with dropout
- Softmax output for binary classification
- Matches paper architecture

### ✅ TASK-2.3: Model Summary and Validation
**Completed:** 2026-01-22
- Added model summary functionality
- Validated forward pass with correct input/output shapes
- Parameter count verification

### ✅ TASK-3.1: Metrics Tracking Module
**Completed:** 2026-01-22
- Created `src/training/metrics.py`
- `MetricsTracker` class for loss and accuracy tracking
- History storage and retrieval

### ✅ TASK-3.2: Training Loop Implementation
**Completed:** 2026-01-22
- Implemented complete training loop in `src/training/trainer.py`
- Cross-entropy loss with Adam optimizer
- Train/validation split with early stopping
- Progress logging and epoch tracking

### ✅ TASK-3.3: Model Checkpointing
**Completed:** 2026-01-22
- Implemented `save_checkpoint()` and `load_checkpoint()`
- Save model weights, optimizer state, and metrics
- Resume training capability

### ✅ TASK-3.4: Visualization Module
**Completed:** 2026-01-22
- Created `src/visualization/plots.py`
- `plot_loss_curves()` for training/validation loss
- `plot_accuracy_curves()` for training/validation accuracy
- `plot_training_summary()` for combined metrics view

### ✅ TASK-3.5: Training Notebook
**Completed:** 2026-01-22
- Created `notebooks/02_training.ipynb`
- End-to-end training workflow
- Live visualization of training progress
- Model saving functionality

### ✅ TASK-4.1: Prediction Service
**Completed:** 2026-01-22
- Created `src/prediction/predictor.py`
- `Predictor` class for loading models and generating predictions
- BUY/SELL signal generation with confidence scores
- Batch prediction support

### ✅ TASK-4.2: Prediction Notebook
**Completed:** 2026-01-22
- Created `notebooks/03_prediction.ipynb`
- Interactive prediction interface
- Price chart visualization with prediction overlay
- Confidence score display

### ✅ TASK-4.3: Batch Prediction & Ranking
**Completed:** 2026-01-22
- Batch prediction across multiple stocks
- Confidence-based ranking
- CSV export functionality

### ✅ TASK-5.1: Streamlit Dashboard
**Completed:** 2026-01-22
- Created `app.py` with Streamlit
- Four main pages: Single Prediction, Batch Analysis, Top Signals, About
- Ticker selection with S&P 500 dropdown
- Horizon selector (T+5, T+30)
- Price chart visualization
- Top signals ranking table

### ✅ TASK-5.2: Dashboard Styling
**Completed:** 2026-01-22
- Custom CSS styling with color-coded signals (green=BUY, red=SELL)
- Confidence meter visualization
- Progress bars and loading states
- Polished UI/UX with metric cards

---

## Notes

- **CPU Training Reminder**: Use batch_size=128 instead of 250 to fit in memory
- **Start Small**: Begin with 50-100 stocks, validate approach, then scale to full S&P 500
- **Checkpoint Often**: Save model every few epochs to recover from interruptions
