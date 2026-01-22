"""CSV data loading utilities for custom OHLCV data.

This module provides flexible CSV loading with:
- Automatic column mapping for various formats
- Date/datetime column detection
- Timeframe inference
- Data validation
- Resampling support
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard OHLCV column names (internal representation)
STANDARD_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Common column name variations (lowercase for matching)
COLUMN_MAPPINGS = {
    "open": ["open", "o", "open_price", "openprice", "first"],
    "high": ["high", "h", "high_price", "highprice", "max"],
    "low": ["low", "l", "low_price", "lowprice", "min"],
    "close": ["close", "c", "close_price", "closeprice", "last", "adj close", "adjclose", "adjusted close"],
    "volume": ["volume", "v", "vol", "qty", "quantity", "amount"],
}

# Common date/time column names
DATE_COLUMN_NAMES = [
    "date", "datetime", "time", "timestamp", "ts", "dt",
    "trade_date", "tradedate", "period", "index",
]

# Supported timeframes and their typical intervals
TIMEFRAMES = {
    "1m": {"minutes": 1, "name": "1 minute"},
    "5m": {"minutes": 5, "name": "5 minutes"},
    "15m": {"minutes": 15, "name": "15 minutes"},
    "30m": {"minutes": 30, "name": "30 minutes"},
    "1h": {"minutes": 60, "name": "1 hour"},
    "4h": {"minutes": 240, "name": "4 hours"},
    "1d": {"minutes": 1440, "name": "1 day"},
    "1w": {"minutes": 10080, "name": "1 week"},
}


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the date/datetime column in a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Name of the detected date column, or None if not found
    """
    # Check index first
    if isinstance(df.index, pd.DatetimeIndex):
        return None  # Index is already datetime

    # Check column names (case-insensitive)
    for col in df.columns:
        if col.lower() in DATE_COLUMN_NAMES:
            return col

    # Try to detect by content (first column that parses as dates)
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype == "datetime64[ns]":
            try:
                pd.to_datetime(df[col].head(10))
                return col
            except (ValueError, TypeError):
                continue

    return None


def auto_map_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically map DataFrame columns to standard OHLCV names.

    Args:
        df: DataFrame with unknown column names

    Returns:
        Dictionary mapping standard names to actual column names
    """
    mapping = {}
    df_columns_lower = {col.lower(): col for col in df.columns}

    for standard_name, variations in COLUMN_MAPPINGS.items():
        for variant in variations:
            if variant in df_columns_lower:
                mapping[standard_name.capitalize()] = df_columns_lower[variant]
                break

    return mapping


def validate_ohlcv_data(df: pd.DataFrame, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data for integrity.

    Args:
        df: DataFrame with OHLCV columns
        strict: If True, fail on any issue; if False, just warn

    Returns:
        Tuple of (is_valid, list of issues found)
    """
    issues = []

    # Check required columns
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    if issues and strict:
        return False, issues

    # Check for missing values
    for col in STANDARD_COLUMNS:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"Column '{col}' has {null_count} missing values")

    # Check chronological order
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            issues.append("Data is not in chronological order")

    # Check for duplicates
    if isinstance(df.index, pd.DatetimeIndex):
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate timestamps")

    # Check OHLC relationships (High >= Low, High >= Open/Close, Low <= Open/Close)
    if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        invalid_high = (df["High"] < df["Open"]) | (df["High"] < df["Close"])
        invalid_low = (df["Low"] > df["Open"]) | (df["Low"] > df["Close"])
        invalid_hl = df["High"] < df["Low"]

        if invalid_high.any():
            issues.append(f"Found {invalid_high.sum()} rows where High < Open or High < Close")
        if invalid_low.any():
            issues.append(f"Found {invalid_low.sum()} rows where Low > Open or Low > Close")
        if invalid_hl.any():
            issues.append(f"Found {invalid_hl.sum()} rows where High < Low")

    # Check for negative values
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"Column '{col}' has {neg_count} negative values")

    if "Volume" in df.columns:
        neg_vol = (df["Volume"] < 0).sum()
        if neg_vol > 0:
            issues.append(f"Column 'Volume' has {neg_vol} negative values")

    is_valid = len(issues) == 0 or not strict
    return is_valid, issues


def infer_timeframe(df: pd.DataFrame) -> Optional[str]:
    """
    Infer the data timeframe from timestamps.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Timeframe string (e.g., '1d', '1h', '15m') or None if unknown
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Cannot infer timeframe: index is not DatetimeIndex")
        return None

    if len(df) < 2:
        logger.warning("Cannot infer timeframe: need at least 2 data points")
        return None

    # Calculate time differences
    time_diffs = df.index.to_series().diff().dropna()

    # Get the most common difference (mode)
    mode_diff = time_diffs.mode()
    if len(mode_diff) == 0:
        return None

    median_diff = mode_diff.iloc[0]
    minutes = median_diff.total_seconds() / 60

    # Map to standard timeframes (with tolerance)
    for tf_name, tf_info in TIMEFRAMES.items():
        tf_minutes = tf_info["minutes"]
        # Allow 10% tolerance for trading hours gaps
        if abs(minutes - tf_minutes) <= tf_minutes * 0.1:
            logger.info(f"Inferred timeframe: {tf_name} ({tf_info['name']})")
            return tf_name

    # Check for daily data with trading gaps (typically ~1440 minutes for daily)
    if 1380 <= minutes <= 1500:  # ~23-25 hours
        return "1d"

    # Check for weekly data (5-7 days)
    if 7000 <= minutes <= 10500:
        return "1w"

    logger.warning(f"Could not map median interval of {minutes:.1f} minutes to standard timeframe")
    return None


def resample_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        target_timeframe: Target timeframe (e.g., '1h', '4h', '1d')

    Returns:
        Resampled DataFrame
    """
    if target_timeframe not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe: {target_timeframe}. Valid options: {list(TIMEFRAMES.keys())}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for resampling")

    # Define resampling rule
    tf_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
        "1w": "1W",
    }

    rule = tf_map[target_timeframe]

    # Resample with appropriate aggregation
    resampled = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    logger.info(f"Resampled from {len(df)} to {len(resampled)} rows ({target_timeframe})")
    return resampled


def load_csv(
    file_path: Union[str, Path],
    column_mapping: Optional[Dict[str, str]] = None,
    date_column: Optional[str] = None,
    date_format: Optional[str] = None,
    delimiter: str = ",",
    skiprows: int = 0,
    validate: bool = True,
    fill_missing: bool = True,
    target_timeframe: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file with flexible configuration.

    Args:
        file_path: Path to the CSV file
        column_mapping: Dictionary mapping standard names (Open, High, Low, Close, Volume)
                       to actual column names in the file. If None, auto-detect.
        date_column: Name of the date/datetime column. If None, auto-detect.
        date_format: Date format string (e.g., '%Y-%m-%d'). If None, auto-parse.
        delimiter: CSV delimiter (default: ',')
        skiprows: Number of header rows to skip (default: 0)
        validate: Whether to validate data integrity (default: True)
        fill_missing: Whether to forward-fill missing values (default: True)
        target_timeframe: Optionally resample to this timeframe after loading

    Returns:
        DataFrame with standardized OHLCV columns and DatetimeIndex

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    logger.info(f"Loading CSV: {file_path}")

    # Load raw CSV
    df = pd.read_csv(
        file_path,
        delimiter=delimiter,
        skiprows=skiprows,
    )

    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Handle date column
    if date_column is None:
        date_column = detect_date_column(df)

    if date_column:
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        else:
            df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError):
            logger.warning("Could not convert index to DatetimeIndex")

    # Auto-detect or apply column mapping
    if column_mapping is None:
        column_mapping = auto_map_columns(df)
        logger.info(f"Auto-detected column mapping: {column_mapping}")

    # Rename columns to standard names
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    df = df.rename(columns=reverse_mapping)

    # Keep only OHLCV columns
    available_cols = [col for col in STANDARD_COLUMNS if col in df.columns]
    df = df[available_cols]

    # Validate data
    if validate:
        is_valid, issues = validate_ohlcv_data(df, strict=False)
        if issues:
            for issue in issues:
                logger.warning(f"Data issue: {issue}")
        if not is_valid:
            raise ValueError(f"Invalid OHLCV data: {issues}")

    # Sort by date
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    # Remove duplicates
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="first")]

    # Fill missing values
    if fill_missing:
        # Forward-fill price data
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].ffill()
        # Fill volume with 0
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0)

    # Convert to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Resample if requested
    if target_timeframe:
        df = resample_ohlcv(df, target_timeframe)

    # Final validation
    if validate:
        is_valid, issues = validate_ohlcv_data(df, strict=True)
        if not is_valid:
            raise ValueError(f"Data validation failed: {issues}")

    # Infer and log timeframe
    detected_tf = infer_timeframe(df)
    if detected_tf:
        logger.info(f"Data timeframe: {detected_tf}")

    logger.info(f"Loaded {len(df)} valid OHLCV records from {df.index.min()} to {df.index.max()}")

    return df


def load_multiple_csvs(
    file_paths: List[Union[str, Path]],
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files.

    Args:
        file_paths: List of paths to CSV files
        **kwargs: Arguments passed to load_csv

    Returns:
        Dictionary mapping filename (without extension) to DataFrame
    """
    results = {}

    for file_path in file_paths:
        file_path = Path(file_path)
        name = file_path.stem

        try:
            df = load_csv(file_path, **kwargs)
            results[name] = df
            logger.info(f"Loaded {name}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    logger.info(f"Successfully loaded {len(results)}/{len(file_paths)} files")
    return results


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get a summary of OHLCV data.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "timeframe": infer_timeframe(df),
    }

    if isinstance(df.index, pd.DatetimeIndex):
        summary["start_date"] = str(df.index.min())
        summary["end_date"] = str(df.index.max())
        summary["trading_days"] = len(df)

    if "Close" in df.columns:
        summary["price_min"] = float(df["Close"].min())
        summary["price_max"] = float(df["Close"].max())
        summary["price_mean"] = float(df["Close"].mean())

    if "Volume" in df.columns:
        summary["volume_mean"] = float(df["Volume"].mean())
        summary["volume_total"] = float(df["Volume"].sum())

    return summary


def estimate_samples(
    data_rows: int,
    window_size: int = 256,
    horizon: int = 5,
    stride: int = 1,
) -> Dict[str, int]:
    """
    Estimate the number of training samples that can be generated.

    Args:
        data_rows: Total number of data rows
        window_size: Size of the sliding window
        horizon: Prediction horizon
        stride: Step size between windows (1=overlapping, window_size+horizon=non-overlapping)

    Returns:
        Dictionary with sample counts for different modes
    """
    min_required = window_size + horizon
    if data_rows < min_required:
        return {
            "overlapping": 0,
            "strided": 0,
            "non_overlapping": 0,
            "min_required": min_required,
            "data_rows": data_rows,
            "sufficient": False,
        }

    available = data_rows - window_size - horizon + 1

    overlapping = available  # stride=1
    strided = max(1, available // stride) if stride > 1 else available
    non_overlapping_stride = window_size + horizon
    non_overlapping = max(1, available // non_overlapping_stride)

    return {
        "overlapping": overlapping,
        "strided": strided,
        "non_overlapping": non_overlapping,
        "min_required": min_required,
        "data_rows": data_rows,
        "sufficient": overlapping >= 100,  # At least 100 samples recommended
    }
