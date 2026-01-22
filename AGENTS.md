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

1. **Phase 1** - Data Pipeline (fetcher, preprocessor, dataset) ✅ COMPLETE
2. **Phase 2** - CNN Model Architecture ✅ COMPLETE
3. **Phase 3** - Training Pipeline (MVP) ✅ COMPLETE
4. **Phase 4** - Prediction & Inference ✅ COMPLETE
5. **Phase 5** - Web Dashboard ✅ COMPLETE

Check `docs/TASKS.md` for detailed task history and status.

## Workflow Guidelines for Agents

### Before Making a Commit

**IMPORTANT:** Always follow these steps before committing code:

1. **Mark Tasks as Done:**
   - Update `docs/TASKS.md` to move completed tasks from "Next Tasks" to "Completed Tasks"
   - Add completion date and summary of what was accomplished
   - Be specific about what was implemented

2. **Update Progress Statuses:**
   - Update the "Current Status" section in `docs/TASKS.md` with current phase
   - Update "Last Updated" date
   - Set "Next Action" to reflect what comes next

3. **Update Related Documentation:**
   - If completing a phase, update phase status markers (✅) in documentation
   - Update `README.md` if major features are added
   - Update `PROJECT-SPEC.md` if requirements or architecture changes
   - Update `IMPLEMENTATION-PLAN.md` if deliverables are completed

4. **Verify Implementation:**
   - Run the verification commands specified in the task acceptance criteria
   - Ensure all acceptance criteria are met
   - Test that the implementation works as expected

5. **Write Descriptive Commit Messages:**
   - Format: `TASK-X.Y: Brief description of what was implemented`
   - Example: `TASK-3.4: Visualization module with loss/accuracy plots`
   - Reference the task number for traceability

### Commit Message Format

```
TASK-X.Y: Brief description

- Bullet point of key changes
- Another change
- Files modified/created

Acceptance criteria met:
- Criterion 1
- Criterion 2
```

### Example Workflow

```bash
# 1. Complete implementation
# (code changes here)

# 2. Update TASKS.md
# Move TASK-3.4 to "Completed Tasks" section
# Update current status if phase completed

# 3. Test implementation
python -c "from src.visualization.plots import plot_loss_curves; print('OK')"

# 4. Commit with proper message
git add .
git commit -m "TASK-3.4: Visualization module

- Created src/visualization/plots.py
- Implemented plot_loss_curves() and plot_accuracy_curves()
- Added plot_training_summary() for combined view
- Tested with sample data

Acceptance criteria met:
- Generates plots matching paper's style
- Handles empty data gracefully
- Exports to PNG format"

# 5. Push to feature branch
git push -u origin feature/visualization
```

### Documentation Synchronization

Keep these files synchronized:
- **TASKS.md** - Current task status and history
- **IMPLEMENTATION-PLAN.md** - Phase completion status
- **PROJECT-SPEC.md** - Overall project status
- **README.md** - High-level features and setup
- **AGENTS.md** - Instructions for agents (this file)

When completing a major milestone (e.g., entire phase):
1. Mark all phase tasks as complete in TASKS.md
2. Update phase status (✅) in IMPLEMENTATION-PLAN.md
3. Update deliverables checklist in IMPLEMENTATION-PLAN.md
4. Update PROJECT-SPEC.md if it affects requirements/status
5. Update README.md with new features/capabilities
