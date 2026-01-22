"""Data preprocessing utilities for stock market data."""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize(df: pd.DataFrame, method: str = "minmax") -> pd.DataFrame:
    """
    Normalize stock data using min-max scaling per feature.

    Args:
        df: DataFrame with OHLCV columns
        method: Normalization method ('minmax' or 'zscore')

    Returns:
        Normalized DataFrame with values between 0 and 1 (for minmax)
    """
    df_norm = df.copy()

    if method == "minmax":
        for col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val - min_val != 0:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.0
    elif method == "zscore":
        for col in df_norm.columns:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val != 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
            else:
                df_norm[col] = 0.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return df_norm


def create_labels(df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
    """
    Create binary labels based on future price movement.

    Label = 1 if close[t + horizon] > close[t] (bullish)
    Label = 0 if close[t + horizon] <= close[t] (bearish)

    Args:
        df: DataFrame with 'Close' column
        horizon: Number of days ahead to predict

    Returns:
        NumPy array of binary labels (length = len(df) - horizon)
    """
    close_prices = df["Close"].values

    # Future returns: close[t+horizon] vs close[t]
    future_close = close_prices[horizon:]
    current_close = close_prices[:-horizon]

    # Binary label: 1 if price goes up, 0 otherwise
    labels = (future_close > current_close).astype(np.int64)

    return labels


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 256,
    horizon: int = 5,
    normalize_windows: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from stock data for model training.

    Each window contains `window_size` days of OHLCV data.
    The label indicates price direction `horizon` days after the window ends.

    Args:
        df: DataFrame with OHLCV columns
        window_size: Number of days in each window (default: 256)
        horizon: Days ahead to predict (default: 5)
        normalize_windows: Whether to normalize each window individually

    Returns:
        Tuple of (X, y) where:
        - X has shape [N, window_size, 5] (5 = OHLCV)
        - y has shape [N,] with binary labels
    """
    # Ensure we have enough data
    min_required = window_size + horizon
    if len(df) < min_required:
        raise ValueError(
            f"Insufficient data: need {min_required} rows, got {len(df)}"
        )

    # Get OHLCV values
    columns = ["Open", "High", "Low", "Close", "Volume"]
    data = df[columns].values

    # Create labels for the entire series
    labels = create_labels(df, horizon)

    # Number of valid windows
    n_windows = len(df) - window_size - horizon + 1

    if n_windows <= 0:
        raise ValueError(f"Not enough data for window_size={window_size}, horizon={horizon}")

    windows = []
    window_labels = []

    for i in range(n_windows):
        window = data[i : i + window_size].copy()

        if normalize_windows:
            # Min-max normalize each window independently
            for col_idx in range(window.shape[1]):
                col_data = window[:, col_idx]
                min_val = col_data.min()
                max_val = col_data.max()
                if max_val - min_val != 0:
                    window[:, col_idx] = (col_data - min_val) / (max_val - min_val)
                else:
                    window[:, col_idx] = 0.0

        windows.append(window)
        # Label at the end of the window
        window_labels.append(labels[i + window_size - 1])

    X = np.array(windows, dtype=np.float32)
    y = np.array(window_labels, dtype=np.int64)

    logger.info(
        f"Created {len(X)} windows of shape {X.shape[1:]} with {y.mean():.1%} bullish labels"
    )

    return X, y


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data chronologically into train, validation, and test sets.

    IMPORTANT: Does NOT shuffle to maintain temporal order and avoid look-ahead bias.

    Args:
        X: Feature array of shape [N, window_size, features]
        y: Label array of shape [N,]
        val_ratio: Fraction of data for validation (default: 0.15)
        test_ratio: Fraction of data for testing (default: 0.15)

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n_samples = len(X)

    train_end = int(n_samples * (1 - val_ratio - test_ratio))
    val_end = int(n_samples * (1 - test_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    logger.info(
        f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_data(
    df: pd.DataFrame,
    window_size: int = 256,
    horizon: int = 5,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to prepare data in one step.

    Combines window creation and train/val/test splitting.

    Args:
        df: Raw OHLCV DataFrame
        window_size: Days per window
        horizon: Prediction horizon
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X, y = create_sliding_windows(df, window_size, horizon)
    return train_val_test_split(X, y, val_ratio, test_ratio)


def combine_multiple_stocks(
    stock_data: dict,
    window_size: int = 256,
    horizon: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine windows from multiple stocks into a single dataset.

    Args:
        stock_data: Dict mapping ticker -> DataFrame
        window_size: Days per window
        horizon: Prediction horizon

    Returns:
        Combined (X, y) arrays from all stocks
    """
    all_X = []
    all_y = []

    for ticker, df in stock_data.items():
        try:
            X, y = create_sliding_windows(df, window_size, horizon)
            all_X.append(X)
            all_y.append(y)
            logger.debug(f"{ticker}: {len(X)} windows")
        except ValueError as e:
            logger.warning(f"Skipping {ticker}: {e}")

    if not all_X:
        raise ValueError("No valid data from any stock")

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    logger.info(
        f"Combined {len(stock_data)} stocks: {len(X_combined)} total windows"
    )

    return X_combined, y_combined
