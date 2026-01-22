# CNNStockMarket

S&P 500 stock market prediction using 1D Convolutional Neural Networks (CNN). Based on the paper ["S&P 500 Stock's Movement Prediction using CNN"](https://arxiv.org/abs/2512.21804).

## 🎯 Project Status

**Phases 1-5 Complete** | **Phases 6-7 In Planning**

- ✅ Phase 1: Data Pipeline (fetching, preprocessing, caching)
- ✅ Phase 2: CNN Model Architecture (8-layer Conv1D)
- ✅ Phase 3: Training Pipeline (metrics, visualization, checkpointing)
- ✅ Phase 4: Prediction & Inference (single/batch predictions)
- ✅ Phase 5: Web Dashboard (Streamlit interface)
- ⬚ Phase 6: Custom Data & Timeframes (planned)
- ⬚ Phase 7: Backtesting Framework (planned)

## 🚀 Features

### Current Features
- **Data Pipeline**: Fetch historical OHLCV data from Yahoo Finance with caching
- **CNN Model**: 8-layer 1D CNN with batch normalization and dropout
- **Training**: Complete training pipeline with early stopping and checkpointing
- **Visualization**: Training curves, loss/accuracy plots
- **Prediction**: Generate BUY/SELL signals for any S&P 500 stock
- **Web Dashboard**: Interactive Streamlit app for predictions
- **Batch Analysis**: Analyze multiple stocks and rank by confidence
- **Multiple Horizons**: Support for T+5 and T+30 day predictions

### Planned Features (Phase 6)
- **Custom CSV Data**: Train on your own OHLCV data from any source
- **Arbitrary Timeframes**: Support for 1m, 5m, 15m, 1h, 4h, 1d, 1w data
- **Multi-Asset Support**: Stocks, cryptocurrencies, forex, futures, commodities
- **Configurable Windows**: Dynamic window sizes for different use cases
- **Asset Presets**: Pre-configured settings for different asset classes

### Planned Features (Phase 7)
- **Backtesting Engine**: Evaluate trained models on historical unseen data
- **Performance Metrics**: Accuracy, Sharpe ratio, max drawdown, win rate
- **Trade Simulation**: Realistic P&L with transaction costs and slippage
- **Walk-Forward Validation**: Robust model evaluation over multiple periods
- **Report Generation**: Automated HTML/PDF backtest reports

## 📋 Prerequisites

- Python 3.12
- 8GB RAM recommended
- Internet connection for fetching stock data

## 🔧 Setup

```bash
# Clone the repository
git clone <repository-url>
cd CNNStockMarket

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, pandas, yfinance, matplotlib; print('All dependencies installed successfully!')"
```

## 📊 Usage

### 1. Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Explore the data pipeline, view OHLCV data, normalization, and sliding windows.

### 2. Train Model

```bash
jupyter notebook notebooks/02_training.ipynb
```

Train the CNN model on historical stock data. The notebook includes:
- Data loading and preprocessing
- Model training with validation
- Training curve visualization
- Model checkpointing

### 3. Make Predictions

```bash
jupyter notebook notebooks/03_prediction.ipynb
```

Generate predictions for individual stocks:
- Load trained model
- Input ticker symbol
- Get BUY/SELL signal with confidence score
- View price charts with predictions

### 4. Web Dashboard

```bash
streamlit run app.py
```

Launch the interactive web dashboard featuring:
- **Single Prediction**: Get predictions for individual stocks
- **Batch Analysis**: Analyze multiple stocks at once
- **Top Signals**: View highest-confidence BUY/SELL signals
- **About**: Learn about the model architecture

## 🏗️ Project Structure

```
CNNStockMarket/
├── docs/                       # Documentation
│   ├── PROJECT-SPEC.md        # Project requirements
│   ├── IMPLEMENTATION-PLAN.md # Phase breakdown
│   ├── TASKS.md               # Task tracker
│   ├── DATA-FORMATS.md        # CSV format docs (Phase 6)
│   └── reference/             # Research paper
├── src/                       # Source code
│   ├── data/                  # Data fetching & preprocessing
│   │   ├── fetcher.py         # Yahoo Finance data fetching
│   │   ├── preprocessor.py    # Normalization & windowing
│   │   ├── dataset.py         # PyTorch Dataset class
│   │   ├── csv_loader.py      # CSV file loading (Phase 6)
│   │   └── data_source.py     # Unified data interface (Phase 6)
│   ├── models/                # Model architecture
│   │   └── cnn.py             # 1D CNN implementation
│   ├── training/              # Training pipeline
│   │   ├── trainer.py         # Training loop
│   │   └── metrics.py         # Loss/accuracy tracking
│   ├── prediction/            # Prediction service
│   │   └── predictor.py       # Inference interface
│   ├── backtesting/           # Backtesting framework (Phase 7)
│   │   ├── engine.py          # Backtesting engine
│   │   ├── metrics.py         # Performance metrics
│   │   ├── simulator.py       # Trade simulator
│   │   ├── plots.py           # Backtest visualizations
│   │   └── report.py          # Report generation
│   ├── visualization/         # Plotting utilities
│   │   └── plots.py           # Training curves
│   └── utils/                 # Configuration
│       ├── config.py          # Hyperparameters
│       └── presets.py         # Asset class presets (Phase 6)
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   ├── 03_prediction.ipynb
│   ├── 04_custom_data_training.ipynb  # Phase 6
│   └── 05_backtesting.ipynb           # Phase 7
├── models/                    # Saved model weights
├── data/                      # Cached stock data & custom CSVs
├── reports/                   # Generated backtest reports (Phase 7)
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🧠 Model Architecture

**Input**: `[batch, window_size, 5]` - Configurable window with 5 OHLCV channels
- Default: 256 periods (daily data = ~1 year of trading days)
- Phase 6 will support: 128-512 periods for different timeframes

**Architecture**:
- 8 Conv1D layers (5→128→256→256→512→1024→1024→1024→1024)
- ReLU/LeakyReLU activation + Batch Normalization
- 2 Fully Connected layers (→256→2)
- Dropout (0.4) for regularization
- Softmax output

**Output**: `[batch, 2]` - Probability distribution [bearish, bullish]

**Key Hyperparameters**:
- Window Size: 256 periods (configurable in Phase 6)
- Batch Size: 128 (CPU optimized)
- Learning Rate: 1e-3
- Optimizer: Adam
- Loss: Cross-Entropy

## 📈 Quick Start Example

```python
# Fetch data
from src.data.fetcher import fetch_stock_data
df = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')

# Make prediction
from src.prediction.predictor import Predictor
predictor = Predictor('models/best_model.pt')
prediction = predictor.predict('AAPL', horizon=5)

print(f"Signal: {prediction.signal}")
print(f"Confidence: {prediction.confidence:.1%}")
print(f"Bullish: {prediction.bullish_prob:.1%}, Bearish: {prediction.bearish_prob:.1%}")
```

## 📚 Documentation

For detailed information:
- **Project Specification**: `docs/PROJECT-SPEC.md`
- **Implementation Plan**: `docs/IMPLEMENTATION-PLAN.md`
- **Task History**: `docs/TASKS.md`
- **Agent Guidelines**: `AGENTS.md`

## ⚠️ Disclaimer

**This tool is for educational purposes only.**

The predictions are based on historical patterns and should NOT be used as financial advice. Stock markets are inherently unpredictable, and past performance does not guarantee future results. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## 📖 References

- Original Paper: [S&P 500 Stock's Movement Prediction using CNN (arXiv:2512.21804)](https://arxiv.org/abs/2512.21804)
- Data Source: [Yahoo Finance](https://finance.yahoo.com/) via [yfinance](https://github.com/ranaroussi/yfinance)

## 🛠️ Technical Details

- **Framework**: PyTorch 2.x
- **Data Source**: Yahoo Finance (yfinance) + CSV files (Phase 6)
- **Supported Assets**: Stocks (current), Crypto/Forex/Futures (Phase 6)
- **Timeframes**: Daily (current), 1m-1w (Phase 6)
- **Python Version**: 3.12
- **Training**: CPU-optimized (no GPU required)
- **Web Framework**: Streamlit

## 🤝 Contributing

### Active Development (Phases 6 & 7)
The following features are currently planned:

**Phase 6 - Custom Data & Timeframes:**
- CSV file loading for custom OHLCV data
- Arbitrary timeframe support (1m to 1w)
- Multi-asset configuration presets
- Dynamic window size configuration
- Enhanced dashboard with CSV upload

**Phase 7 - Backtesting Framework:**
- Backtesting engine with walk-forward validation
- Comprehensive performance metrics
- Trade simulation with realistic costs
- Automated report generation (HTML/PDF)
- Dashboard integration for backtesting

### Future Enhancements
- Model hyperparameter tuning
- Alternative architectures (LSTM, Transformers)
- Real-time data integration

## 📝 License

See LICENSE file for details.
