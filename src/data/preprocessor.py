"""Data preprocessing utilities for stock market data.

This module provides:
- Data normalization (min-max, z-score)
- Label generation for binary classification
- Sliding window creation with configurable stride
- Train/val/test splitting
- Multi-stock data combination
"""

import logging
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard OHLCV column names
STANDARD_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


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
    stride: int = 1,
    columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from stock data for model training.

    Each window contains `window_size` periods of OHLCV data.
    The label indicates price direction `horizon` periods after the window ends.

    Args:
        df: DataFrame with OHLCV columns
        window_size: Number of periods in each window (default: 256)
        horizon: Periods ahead to predict (default: 5)
        normalize_windows: Whether to normalize each window individually
        stride: Step size between windows (default: 1 for overlapping)
               - stride=1: Maximum samples, high correlation (overlapping)
               - stride=10-50: Balanced approach (strided)
               - stride=window_size+horizon: Independent samples (non-overlapping)
        columns: List of column names to use (default: standard OHLCV)

    Returns:
        Tuple of (X, y) where:
        - X has shape [N, window_size, num_channels]
        - y has shape [N,] with binary labels
    """
    if columns is None:
        columns = STANDARD_COLUMNS

    # Filter to available columns
    available_columns = [col for col in columns if col in df.columns]
    if len(available_columns) < len(columns):
        missing = set(columns) - set(available_columns)
        logger.warning(f"Missing columns: {missing}, using: {available_columns}")

    if not available_columns:
        raise ValueError(f"No valid columns found. Required: {columns}, Available: {list(df.columns)}")

    # Ensure we have enough data
    min_required = window_size + horizon
    if len(df) < min_required:
        raise ValueError(
            f"Insufficient data: need {min_required} rows, got {len(df)}"
        )

    # Get OHLCV values
    data = df[available_columns].values

    # Create labels for the entire series
    labels = create_labels(df, horizon)

    # Calculate number of valid windows with stride
    total_range = len(df) - window_size - horizon + 1
    if total_range <= 0:
        raise ValueError(f"Not enough data for window_size={window_size}, horizon={horizon}")

    # Generate window indices based on stride
    window_indices = list(range(0, total_range, stride))
    n_windows = len(window_indices)

    if n_windows <= 0:
        raise ValueError(f"No windows can be created with stride={stride}")

    windows = []
    window_labels = []

    for i in window_indices:
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

    mode_desc = "overlapping" if stride == 1 else f"stride={stride}"
    logger.info(
        f"Created {len(X)} windows ({mode_desc}) of shape {X.shape[1:]} "
        f"with {y.mean():.1%} bullish labels"
    )

    return X, y


def estimate_sample_count(
    data_rows: int,
    window_size: int = 256,
    horizon: int = 5,
    stride: int = 1,
) -> Dict[str, int]:
    """
    Estimate the number of training samples without loading data.

    Args:
        data_rows: Total number of data rows
        window_size: Size of the sliding window
        horizon: Prediction horizon
        stride: Step size between windows

    Returns:
        Dictionary with sample statistics
    """
    min_required = window_size + horizon
    if data_rows < min_required:
        return {
            "samples": 0,
            "min_required": min_required,
            "data_rows": data_rows,
            "sufficient": False,
            "message": f"Need at least {min_required} rows, have {data_rows}",
        }

    total_range = data_rows - window_size - horizon + 1
    samples = (total_range - 1) // stride + 1 if stride > 0 else 0

    # Estimate splits
    train_samples = int(samples * 0.7)
    val_samples = int(samples * 0.15)
    test_samples = samples - train_samples - val_samples

    return {
        "samples": samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "stride": stride,
        "min_required": min_required,
        "data_rows": data_rows,
        "sufficient": train_samples >= 100,
        "message": "OK" if train_samples >= 100 else f"Low sample count: {train_samples} training samples",
    }


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
    stride: int = 1,
    columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to prepare data in one step.

    Combines window creation and train/val/test splitting.

    Args:
        df: Raw OHLCV DataFrame
        window_size: Periods per window
        horizon: Prediction horizon
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stride: Step size between windows (1=overlapping)
        columns: List of column names to use (default: OHLCV)

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X, y = create_sliding_windows(df, window_size, horizon, stride=stride, columns=columns)
    return train_val_test_split(X, y, val_ratio, test_ratio)


def combine_multiple_stocks(
    stock_data: dict,
    window_size: int = 256,
    horizon: int = 5,
    stride: int = 1,
    columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine windows from multiple stocks into a single dataset.

    Args:
        stock_data: Dict mapping ticker -> DataFrame
        window_size: Periods per window
        horizon: Prediction horizon
        stride: Step size between windows (1=overlapping)
        columns: List of column names to use (default: OHLCV)

    Returns:
        Combined (X, y) arrays from all stocks
    """
    all_X = []
    all_y = []

    for ticker, df in stock_data.items():
        try:
            X, y = create_sliding_windows(df, window_size, horizon, stride=stride, columns=columns)
            all_X.append(X)
            all_y.append(y)
            logger.debug(f"{ticker}: {len(X)} windows")
        except ValueError as e:
            logger.warning(f"Skipping {ticker}: {e}")

    if not all_X:
        raise ValueError("No valid data from any stock")

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    mode_desc = "overlapping" if stride == 1 else f"stride={stride}"
    logger.info(
        f"Combined {len(stock_data)} stocks: {len(X_combined)} total windows ({mode_desc})"
    )

    return X_combined, y_combined


def compare_sample_modes(
    df: pd.DataFrame,
    window_size: int = 256,
    horizon: int = 5,
    custom_stride: int = 10,
) -> Dict[str, Dict]:
    """
    Compare different sample generation modes.

    Useful for deciding which mode to use based on available data.

    Args:
        df: DataFrame with OHLCV data
        window_size: Periods per window
        horizon: Prediction horizon
        custom_stride: Stride to use for strided mode

    Returns:
        Dictionary with statistics for each mode
    """
    data_rows = len(df)
    non_overlap_stride = window_size + horizon

    modes = {
        "overlapping": {
            "stride": 1,
            "description": "Maximum samples, high correlation between samples",
        },
        "strided": {
            "stride": custom_stride,
            "description": f"Stride={custom_stride}, reduced correlation",
        },
        "non_overlapping": {
            "stride": non_overlap_stride,
            "description": f"Fully independent samples (stride={non_overlap_stride})",
        },
    }

    results = {}
    for mode_name, mode_info in modes.items():
        stats = estimate_sample_count(
            data_rows,
            window_size=window_size,
            horizon=horizon,
            stride=mode_info["stride"],
        )
        stats["description"] = mode_info["description"]
        results[mode_name] = stats

    return results
