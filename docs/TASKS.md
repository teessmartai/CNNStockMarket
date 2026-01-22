# CNN Stock Movement Prediction - Task Tracker

## Current Status

**Phase:** 1 - Project Setup & Data Pipeline
**Last Updated:** 2026-01-22
**Next Action:** Set up project structure and virtual environment

---

## Active Tasks (In Progress)

*None currently - ready to begin Phase 1*

---

## Next Tasks (Ready to Start)

### [ ] TASK-1.1: Project Structure Setup

**Phase:** 1
**Priority:** High
**Blocked By:** None

**Context:**
Create the foundational project structure including virtual environment, directory layout, and dependency management. This enables all subsequent development.

**Implementation Steps:**

1. [ ] Create virtual environment with Python 3.12
2. [ ] Create `requirements.txt` with initial dependencies:
   - torch
   - pandas
   - numpy
   - yfinance
   - matplotlib
   - seaborn
   - jupyter
   - scikit-learn
   - tqdm
3. [ ] Create directory structure:
   ```
   src/
   ├── __init__.py
   ├── data/
   │   └── __init__.py
   ├── models/
   │   └── __init__.py
   ├── training/
   │   └── __init__.py
   ├── visualization/
   │   └── __init__.py
   ├── prediction/
   │   └── __init__.py
   └── utils/
       └── __init__.py
   notebooks/
   models/
   data/
   ```
4. [ ] Create `.gitignore` (Python, venv, data files, model weights)
5. [ ] Install dependencies and verify imports

**Files to Create:**
- `requirements.txt`
- `src/__init__.py` and all subpackage `__init__.py` files
- `.gitignore`
- Empty `notebooks/`, `models/`, `data/` directories

**Acceptance Criteria:**
- [ ] `source venv/bin/activate` works
- [ ] `pip install -r requirements.txt` succeeds
- [ ] `python -c "import torch; print(torch.__version__)"` works
- [ ] `python -c "from src.data import *"` works (no errors)

**Verification:**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import torch, pandas, yfinance, matplotlib; print('All imports OK')"
```

---

### [ ] TASK-1.2: Data Fetcher Implementation

**Phase:** 1
**Priority:** High
**Blocked By:** TASK-1.1

**Context:**
Build the module responsible for fetching historical stock data from Yahoo Finance. This is the entry point for all data into the system.

**Implementation Steps:**

1. [ ] Create `src/data/fetcher.py`
2. [ ] Implement `fetch_stock_data(ticker, start_date, end_date)`:
   - Use `yfinance.download()`
   - Return DataFrame with columns: Open, High, Low, Close, Volume
   - Handle errors (invalid ticker, no data)
3. [ ] Implement `fetch_sp500_tickers()`:
   - Fetch current S&P 500 constituents list
   - Return list of ticker symbols
4. [ ] Implement `fetch_multiple_stocks(tickers, start_date, end_date)`:
   - Fetch data for multiple tickers
   - Use progress bar (tqdm)
   - Return dict mapping ticker -> DataFrame
5. [ ] Add basic error handling and logging

**Files to Modify:**
- `src/data/fetcher.py` - Create new
- `src/data/__init__.py` - Add exports

**Acceptance Criteria:**
- [ ] `fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')` returns DataFrame with ~1000 rows
- [ ] `fetch_sp500_tickers()` returns list of ~500 tickers
- [ ] Invalid ticker returns empty DataFrame or raises informative error

**Verification:**
```bash
python -c "
from src.data.fetcher import fetch_stock_data, fetch_sp500_tickers
df = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
print(f'AAPL: {len(df)} rows, columns: {list(df.columns)}')
tickers = fetch_sp500_tickers()
print(f'S&P 500: {len(tickers)} tickers')
"
```

---

### [ ] TASK-1.3: Data Caching Layer

**Phase:** 1
**Priority:** Medium
**Blocked By:** TASK-1.2

**Context:**
Avoid repeated API calls by caching downloaded data locally. Yahoo Finance may rate-limit, and caching speeds up development iteration.

**Implementation Steps:**

1. [ ] Add caching logic to `fetch_stock_data()`:
   - Check if `data/{ticker}_{start}_{end}.csv` exists
   - If exists and recent, load from cache
   - If not, fetch and save to cache
2. [ ] Implement `clear_cache()` function
3. [ ] Add cache directory to `.gitignore`

**Files to Modify:**
- `src/data/fetcher.py` - Extend with caching

**Acceptance Criteria:**
- [ ] First fetch downloads from API
- [ ] Second fetch loads from local CSV (faster)
- [ ] Cache files appear in `data/` directory

---

### [ ] TASK-1.4: Data Preprocessor Implementation

**Phase:** 1
**Priority:** High
**Blocked By:** TASK-1.2

**Context:**
Raw stock data needs normalization and transformation into sliding windows for the CNN. This matches the paper's preprocessing approach.

**Implementation Steps:**

1. [ ] Create `src/data/preprocessor.py`
2. [ ] Implement `normalize(df)`:
   - Min-max normalization per feature: `(x - min) / (max - min)`
   - Return normalized DataFrame
3. [ ] Implement `create_labels(df, horizon)`:
   - Label = 1 if `close[t+horizon] > close[t]` else 0
   - Return numpy array of labels
4. [ ] Implement `create_sliding_windows(df, window_size, horizon)`:
   - Slide window of size `window_size` across data
   - For each window, pair with label at `horizon` days ahead
   - Return `(X, y)` where X shape is `[N, window_size, 5]`
5. [ ] Implement `train_val_test_split(X, y, val_ratio, test_ratio)`:
   - Split data chronologically (not random) to avoid look-ahead bias
   - Return train, val, test sets

**Files to Modify:**
- `src/data/preprocessor.py` - Create new
- `src/data/__init__.py` - Add exports

**Acceptance Criteria:**
- [ ] Normalized values are between 0 and 1
- [ ] Labels are binary (0 or 1)
- [ ] Windows have correct shape `[N, 256, 5]`
- [ ] No data leakage between train/val/test

**Verification:**
```bash
python -c "
from src.data.fetcher import fetch_stock_data
from src.data.preprocessor import normalize, create_sliding_windows

df = fetch_stock_data('AAPL', '2015-01-01', '2024-01-01')
X, y = create_sliding_windows(df, window_size=256, horizon=5)
print(f'X shape: {X.shape}')  # Should be [N, 256, 5]
print(f'y shape: {y.shape}')  # Should be [N,]
print(f'Labels distribution: {y.mean():.2%} bullish')
"
```

---

### [ ] TASK-1.5: PyTorch Dataset Class

**Phase:** 1
**Priority:** High
**Blocked By:** TASK-1.4

**Context:**
Wrap preprocessed data in PyTorch Dataset and DataLoader for efficient batched training.

**Implementation Steps:**

1. [ ] Create `src/data/dataset.py`
2. [ ] Implement `StockDataset(Dataset)`:
   - Takes preprocessed X, y arrays
   - Converts to torch tensors
   - Implements `__len__` and `__getitem__`
3. [ ] Implement `create_dataloaders(X_train, y_train, X_val, y_val, batch_size)`:
   - Create train DataLoader (shuffle=True)
   - Create val DataLoader (shuffle=False)
   - Return both

**Files to Modify:**
- `src/data/dataset.py` - Create new
- `src/data/__init__.py` - Add exports

**Acceptance Criteria:**
- [ ] Can iterate DataLoader and get batches
- [ ] Batch tensors have correct shapes and dtypes
- [ ] Training loader shuffles, validation loader doesn't

---

### [ ] TASK-1.6: Data Exploration Notebook

**Phase:** 1
**Priority:** Medium
**Blocked By:** TASK-1.5

**Context:**
Create Jupyter notebook to explore and visualize the data pipeline. Useful for debugging and understanding the data.

**Implementation Steps:**

1. [ ] Create `notebooks/01_data_exploration.ipynb`
2. [ ] Sections:
   - Fetch sample stock data
   - Visualize raw OHLCV
   - Show normalization effect
   - Display sliding windows
   - Show label distribution
   - Test DataLoader iteration

**Files to Create:**
- `notebooks/01_data_exploration.ipynb`

**Acceptance Criteria:**
- [ ] Notebook runs end-to-end without errors
- [ ] Contains visualizations of each preprocessing step

---

## Backlog (Future Tasks)

### [ ] TASK-2.1: Configuration Module
**Phase:** 2
**Blocked By:** TASK-1.1

### [ ] TASK-2.2: CNN Model Implementation
**Phase:** 2
**Blocked By:** TASK-2.1

### [ ] TASK-2.3: Model Summary and Validation
**Phase:** 2
**Blocked By:** TASK-2.2

### [ ] TASK-3.1: Metrics Tracking Module
**Phase:** 3
**Blocked By:** TASK-2.2

### [ ] TASK-3.2: Training Loop Implementation
**Phase:** 3
**Blocked By:** TASK-3.1

### [ ] TASK-3.3: Model Checkpointing
**Phase:** 3
**Blocked By:** TASK-3.2

### [ ] TASK-3.4: Visualization Module
**Phase:** 3
**Blocked By:** TASK-3.1

### [ ] TASK-3.5: Training Notebook
**Phase:** 3
**Blocked By:** TASK-3.4

### [ ] TASK-4.1: Prediction Service
**Phase:** 4
**Blocked By:** TASK-3.3

### [ ] TASK-4.2: Prediction Notebook
**Phase:** 4
**Blocked By:** TASK-4.1

### [ ] TASK-4.3: Batch Prediction & Ranking
**Phase:** 4
**Blocked By:** TASK-4.1

### [ ] TASK-5.1: Streamlit Dashboard
**Phase:** 5
**Blocked By:** TASK-4.2

### [ ] TASK-5.2: Dashboard Styling
**Phase:** 5
**Blocked By:** TASK-5.1

---

## Completed Tasks

*None yet - project starting*

---

## Notes

- **CPU Training Reminder**: Use batch_size=128 instead of 250 to fit in memory
- **Start Small**: Begin with 50-100 stocks, validate approach, then scale to full S&P 500
- **Checkpoint Often**: Save model every few epochs to recover from interruptions
