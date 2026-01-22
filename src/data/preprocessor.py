"""Data preprocessing module for normalization and sliding window creation."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min-max normalization to OHLCV data.

    Normalizes each column independently to range [0, 1] using:
    normalized = (x - min) / (max - min)

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Normalized DataFrame with same shape and columns
    """
    if df.empty:
        return df

    df_normalized = df.copy()

    for col in df_normalized.columns:
        col_min = df_normalized[col].min()
        col_max = df_normalized[col].max()

        if col_max - col_min > 0:
            df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        else:
            df_normalized[col] = 0.0

    return df_normalized


def create_labels(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    Create binary labels based on price movement.

    Label = 1 if Close[t + horizon] > Close[t], else 0 (bullish/bearish)

    Args:
        df: DataFrame with 'Close' column
        horizon: Number of days ahead to compare

    Returns:
        Numpy array of binary labels (1 = bullish, 0 = bearish)
    """
    close_prices = df['Close'].values

    # Compare current close to close 'horizon' days ahead
    labels = np.zeros(len(close_prices) - horizon, dtype=np.int64)

    for i in range(len(labels)):
        if close_prices[i + horizon] > close_prices[i]:
            labels[i] = 1  # Bullish
        else:
            labels[i] = 0  # Bearish

    return labels


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 256,
    horizon: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from OHLCV data for CNN input.

    Each window contains 'window_size' days of normalized OHLCV data,
    paired with a label indicating if the price goes up or down after
    'horizon' days from the end of the window.

    Args:
        df: DataFrame with OHLCV columns
        window_size: Number of days in each window (default: 256)
        horizon: Days ahead for label prediction (default: 5)

    Returns:
        Tuple of (X, y) where:
        - X shape: [N, window_size, 5] - sliding windows
        - y shape: [N,] - binary labels
    """
    if df.empty or len(df) < window_size + horizon:
        logger.warning(
            f"Insufficient data: need {window_size + horizon} rows, got {len(df)}"
        )
        return np.array([]), np.array([])

    # Normalize the data
    df_normalized = normalize(df)

    # Get feature columns (OHLCV)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df_normalized[feature_cols].values

    # Create labels using non-normalized data for accurate price comparison
    labels = create_labels(df, horizon)

    # Number of windows we can create
    num_windows = len(df) - window_size - horizon + 1

    if num_windows <= 0:
        return np.array([]), np.array([])

    # Create sliding windows
    X = np.zeros((num_windows, window_size, len(feature_cols)), dtype=np.float32)
    y = np.zeros(num_windows, dtype=np.int64)

    for i in range(num_windows):
        X[i] = data[i:i + window_size]
        # Label is based on price movement from end of window
        y[i] = labels[i + window_size - 1]

    logger.info(
        f"Created {num_windows} windows: X shape {X.shape}, "
        f"y distribution: {y.mean():.2%} bullish"
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

    Uses chronological split (not random) to avoid look-ahead bias.
    Data is ordered by time, so later samples are used for validation/test.

    Args:
        X: Feature array of shape [N, window_size, features]
        y: Label array of shape [N,]
        val_ratio: Fraction of data for validation (default: 0.15)
        test_ratio: Fraction of data for test (default: 0.15)

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)

    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))

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
    End-to-end data preparation: create windows and split into train/val/test.

    Convenience function that combines create_sliding_windows and
    train_val_test_split.

    Args:
        df: DataFrame with OHLCV columns
        window_size: Number of days in each window
        horizon: Days ahead for label prediction
        val_ratio: Fraction for validation
        test_ratio: Fraction for test

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X, y = create_sliding_windows(df, window_size, horizon)

    if len(X) == 0:
        return (
            np.array([]), np.array([]),
            np.array([]), np.array([]),
            np.array([]), np.array([])
        )

    return train_val_test_split(X, y, val_ratio, test_ratio)


if __name__ == "__main__":
    # Test the preprocessor
    from src.data.fetcher import fetch_stock_data

    print("Testing preprocessor...")

    # Fetch sample data
    df = fetch_stock_data('AAPL', '2015-01-01', '2024-01-01')
    print(f"Fetched {len(df)} rows")

    # Test normalization
    print("\nTesting normalize...")
    df_norm = normalize(df)
    print(f"Normalized value ranges:")
    for col in df_norm.columns:
        print(f"  {col}: [{df_norm[col].min():.4f}, {df_norm[col].max():.4f}]")

    # Test sliding windows
    print("\nTesting create_sliding_windows...")
    X, y = create_sliding_windows(df, window_size=256, horizon=5)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Labels distribution: {y.mean():.2%} bullish, {1 - y.mean():.2%} bearish")

    # Test train/val/test split
    print("\nTesting train_val_test_split...")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val: X={X_val.shape}, y={y_val.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
