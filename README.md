# CNNStockMarket

S&P 500 stock market prediction using 1D Convolutional Neural Networks (CNN). Based on the paper ["S&P 500 Stock's Movement Prediction using CNN"](https://arxiv.org/abs/2512.21804).

## 🎯 Project Status

**✅ COMPLETE** - All 5 phases implemented and functional!

- ✅ Phase 1: Data Pipeline (fetching, preprocessing, caching)
- ✅ Phase 2: CNN Model Architecture (8-layer Conv1D)
- ✅ Phase 3: Training Pipeline (metrics, visualization, checkpointing)
- ✅ Phase 4: Prediction & Inference (single/batch predictions)
- ✅ Phase 5: Web Dashboard (Streamlit interface)

## 🚀 Features

- **Data Pipeline**: Fetch historical OHLCV data from Yahoo Finance with caching
- **CNN Model**: 8-layer 1D CNN with batch normalization and dropout
- **Training**: Complete training pipeline with early stopping and checkpointing
- **Visualization**: Training curves, loss/accuracy plots
- **Prediction**: Generate BUY/SELL signals for any S&P 500 stock
- **Web Dashboard**: Interactive Streamlit app for predictions
- **Batch Analysis**: Analyze multiple stocks and rank by confidence
- **Multiple Horizons**: Support for T+5 and T+30 day predictions

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
│   ├── TASKS.md               # Task tracker (all tasks complete!)
│   └── reference/             # Research paper
├── src/                       # Source code
│   ├── data/                  # Data fetching & preprocessing
│   │   ├── fetcher.py         # Yahoo Finance data fetching
│   │   ├── preprocessor.py    # Normalization & windowing
│   │   └── dataset.py         # PyTorch Dataset class
│   ├── models/                # Model architecture
│   │   └── cnn.py             # 1D CNN implementation
│   ├── training/              # Training pipeline
│   │   ├── trainer.py         # Training loop
│   │   └── metrics.py         # Loss/accuracy tracking
│   ├── prediction/            # Prediction service
│   │   └── predictor.py       # Inference interface
│   ├── visualization/         # Plotting utilities
│   │   └── plots.py           # Training curves
│   └── utils/                 # Configuration
│       └── config.py          # Hyperparameters
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_prediction.ipynb
├── models/                    # Saved model weights
├── data/                      # Cached stock data
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🧠 Model Architecture

**Input**: `[batch, 256, 5]` - 256-day windows with 5 OHLCV channels

**Architecture**:
- 8 Conv1D layers (5→128→256→256→512→1024→1024→1024→1024)
- ReLU/LeakyReLU activation + Batch Normalization
- 2 Fully Connected layers (→256→2)
- Dropout (0.4) for regularization
- Softmax output

**Output**: `[batch, 2]` - Probability distribution [bearish, bullish]

**Key Hyperparameters**:
- Window Size: 256 days
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
- **Data Source**: Yahoo Finance (yfinance)
- **Python Version**: 3.12
- **Training**: CPU-optimized (no GPU required)
- **Web Framework**: Streamlit

## 🤝 Contributing

This project is complete but open to enhancements:
- Model hyperparameter tuning
- Additional prediction horizons
- Backtesting framework
- Alternative architectures (LSTM, Transformers)
- Real-time data integration

## 📝 License

See LICENSE file for details.
