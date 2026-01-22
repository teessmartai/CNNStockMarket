"""Prediction service for generating BUY/SELL signals."""

from typing import Dict, List, Tuple, Optional, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.models.cnn import StockCNN
from src.data.fetcher import fetch_stock_data
from src.data.preprocessor import normalize
from src.utils.config import (
    DEVICE,
    WINDOW_SIZE,
    NUM_CHANNELS,
    DEFAULT_HORIZON,
    MODELS_DIR,
)


@dataclass
class Prediction:
    """Container for a single stock prediction."""
    ticker: str
    signal: str  # 'BUY' or 'SELL'
    confidence: float  # 0-1
    bullish_prob: float  # Probability of bullish outcome
    bearish_prob: float  # Probability of bearish outcome
    horizon: int  # Prediction horizon in days
    timestamp: datetime


class Predictor:
    """
    Prediction service for stock movement prediction.

    Loads a trained model and generates BUY/SELL signals for any ticker.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: torch.device = DEVICE,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained model checkpoint.
                       If None, looks for 'best_model.pt' in MODELS_DIR.
            device: Device to run inference on.
        """
        self.device = device

        # Find model path
        if model_path is None:
            model_path = MODELS_DIR / 'best_model.pt'

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Train a model first using the training notebook."
            )

        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> StockCNN:
        """Load the trained model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        model = StockCNN()
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model

    def _prepare_input(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Prepare a DataFrame for model input.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Tensor of shape [1, WINDOW_SIZE, NUM_CHANNELS]
        """
        # Ensure we have enough data
        if len(df) < WINDOW_SIZE:
            raise ValueError(
                f"Not enough data. Need {WINDOW_SIZE} days, got {len(df)}."
            )

        # Take the most recent WINDOW_SIZE days
        df_recent = df.iloc[-WINDOW_SIZE:].copy()

        # Normalize
        df_normalized = normalize(df_recent, method='minmax')

        # Convert to numpy array
        values = df_normalized.values.astype(np.float32)

        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(values).unsqueeze(0)  # [1, 256, 5]

        return tensor.to(self.device)

    def predict(
        self,
        ticker: str,
        horizon: int = DEFAULT_HORIZON,
        end_date: Optional[str] = None,
    ) -> Prediction:
        """
        Generate a prediction for a single ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            horizon: Prediction horizon in days
            end_date: End date for data (default: today)

        Returns:
            Prediction object with signal and confidence
        """
        # Calculate start date to ensure we have enough data
        # Fetch extra days to account for market holidays
        import datetime as dt
        if end_date is None:
            end = dt.date.today()
        else:
            end = dt.datetime.strptime(end_date, '%Y-%m-%d').date()

        # Need WINDOW_SIZE trading days, so fetch ~1.5x calendar days
        start = end - dt.timedelta(days=int(WINDOW_SIZE * 1.5))
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        # Fetch data
        df = fetch_stock_data(ticker, start_str, end_str)

        if df.empty or len(df) < WINDOW_SIZE:
            raise ValueError(
                f"Insufficient data for {ticker}. "
                f"Need {WINDOW_SIZE} days, got {len(df)}."
            )

        # Prepare input tensor
        input_tensor = self._prepare_input(df)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = output.squeeze().cpu().numpy()

        bearish_prob = float(probabilities[0])
        bullish_prob = float(probabilities[1])

        # Determine signal and confidence
        if bullish_prob > bearish_prob:
            signal = 'BUY'
            confidence = bullish_prob
        else:
            signal = 'SELL'
            confidence = bearish_prob

        return Prediction(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            bullish_prob=bullish_prob,
            bearish_prob=bearish_prob,
            horizon=horizon,
            timestamp=datetime.now(),
        )

    def predict_batch(
        self,
        tickers: List[str],
        horizon: int = DEFAULT_HORIZON,
        end_date: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Prediction]:
        """
        Generate predictions for multiple tickers.

        Args:
            tickers: List of ticker symbols
            horizon: Prediction horizon in days
            end_date: End date for data
            verbose: Whether to print progress

        Returns:
            List of Prediction objects
        """
        predictions = []
        failed = []

        for i, ticker in enumerate(tickers):
            if verbose:
                print(f"Processing {ticker} ({i+1}/{len(tickers)})...", end='')

            try:
                pred = self.predict(ticker, horizon, end_date)
                predictions.append(pred)
                if verbose:
                    print(f" {pred.signal} ({pred.confidence:.1%})")
            except Exception as e:
                failed.append((ticker, str(e)))
                if verbose:
                    print(f" FAILED: {e}")

        if verbose and failed:
            print(f"\n{len(failed)} tickers failed:")
            for ticker, error in failed[:5]:
                print(f"  {ticker}: {error}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

        return predictions


def rank_predictions(
    predictions: List[Prediction],
    signal_filter: Optional[str] = None,
) -> List[Prediction]:
    """
    Rank predictions by confidence.

    Args:
        predictions: List of predictions to rank
        signal_filter: Optional filter for 'BUY' or 'SELL' signals only

    Returns:
        Sorted list of predictions (highest confidence first)
    """
    if signal_filter:
        predictions = [p for p in predictions if p.signal == signal_filter]

    return sorted(predictions, key=lambda p: p.confidence, reverse=True)


def export_predictions(
    predictions: List[Prediction],
    output_path: Path,
    format: str = 'csv',
) -> None:
    """
    Export predictions to a file.

    Args:
        predictions: List of predictions to export
        output_path: Path to save the file
        format: Output format ('csv' or 'json')
    """
    data = [
        {
            'ticker': p.ticker,
            'signal': p.signal,
            'confidence': p.confidence,
            'bullish_prob': p.bullish_prob,
            'bearish_prob': p.bearish_prob,
            'horizon': p.horizon,
            'timestamp': p.timestamp.isoformat(),
        }
        for p in predictions
    ]

    df = pd.DataFrame(data)

    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def get_top_signals(
    predictions: List[Prediction],
    n: int = 10,
    signal_type: str = 'BUY',
) -> List[Prediction]:
    """
    Get the top N predictions for a specific signal type.

    Args:
        predictions: List of all predictions
        n: Number of top predictions to return
        signal_type: 'BUY' or 'SELL'

    Returns:
        Top N predictions sorted by confidence
    """
    filtered = [p for p in predictions if p.signal == signal_type]
    sorted_preds = sorted(filtered, key=lambda p: p.confidence, reverse=True)
    return sorted_preds[:n]


def predictions_to_dataframe(predictions: List[Prediction]) -> pd.DataFrame:
    """
    Convert predictions to a pandas DataFrame.

    Args:
        predictions: List of predictions

    Returns:
        DataFrame with prediction data
    """
    return pd.DataFrame([
        {
            'Ticker': p.ticker,
            'Signal': p.signal,
            'Confidence': f"{p.confidence:.1%}",
            'Bullish %': f"{p.bullish_prob:.1%}",
            'Bearish %': f"{p.bearish_prob:.1%}",
            'Horizon': f"T+{p.horizon}",
        }
        for p in predictions
    ])
