"""Unified data interface for multiple data sources.

This module provides an abstraction layer for loading OHLCV data from
various sources (Yahoo Finance, CSV files, etc.) with a consistent interface.

Usage:
    # From Yahoo Finance
    source = DataSourceFactory.create("yahoo", ticker="AAPL", start="2020-01-01", end="2024-01-01")
    df = source.get_data()

    # From CSV file
    source = DataSourceFactory.create("csv", file_path="data/btc_hourly.csv")
    df = source.get_data()

    # Prepare for training
    X_train, y_train, X_val, y_val, X_test, y_test = source.prepare_training_data(
        window_size=256, horizon=5, stride=1
    )
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.preprocessor import (
    create_sliding_windows,
    train_val_test_split,
    estimate_sample_count,
)

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, name: str = "unknown"):
        """
        Initialize data source.

        Args:
            name: Name identifier for this data source
        """
        self.name = name
        self._data: Optional[pd.DataFrame] = None
        self._timeframe: Optional[str] = None

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """
        Load and return OHLCV data.

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        pass

    @property
    def data(self) -> pd.DataFrame:
        """Cached data property."""
        if self._data is None:
            self._data = self.get_data()
        return self._data

    @property
    def timeframe(self) -> Optional[str]:
        """Detected timeframe of the data."""
        return self._timeframe

    def get_summary(self) -> Dict:
        """Get summary statistics of the data."""
        df = self.data
        summary = {
            "name": self.name,
            "rows": len(df),
            "timeframe": self._timeframe,
        }

        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            summary["start_date"] = str(df.index.min())
            summary["end_date"] = str(df.index.max())

        if "Close" in df.columns:
            summary["price_min"] = float(df["Close"].min())
            summary["price_max"] = float(df["Close"].max())
            summary["price_latest"] = float(df["Close"].iloc[-1])

        return summary

    def prepare_training_data(
        self,
        window_size: int = 256,
        horizon: int = 5,
        stride: int = 1,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with train/val/test split.

        Args:
            window_size: Size of sliding windows
            horizon: Prediction horizon
            stride: Step between windows (1=overlapping)
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            columns: Columns to use (default: OHLCV)

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        df = self.data
        X, y = create_sliding_windows(
            df, window_size=window_size, horizon=horizon,
            stride=stride, columns=columns
        )
        return train_val_test_split(X, y, val_ratio, test_ratio)

    def estimate_samples(
        self,
        window_size: int = 256,
        horizon: int = 5,
        stride: int = 1,
    ) -> Dict:
        """Estimate the number of samples that will be generated."""
        return estimate_sample_count(
            len(self.data), window_size=window_size,
            horizon=horizon, stride=stride
        )


class YahooFinanceSource(DataSource):
    """Data source for Yahoo Finance data."""

    def __init__(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Initialize Yahoo Finance data source.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format (default: 10 years ago)
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            use_cache: Whether to use cached data
        """
        super().__init__(name=f"Yahoo:{ticker}")
        self.ticker = ticker

        # Default to 10 years of data
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d")

        self.start_date = start_date
        self.end_date = end_date
        self.use_cache = use_cache
        self._timeframe = "1d"  # Yahoo Finance returns daily data

    def get_data(self) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        from src.data.fetcher import fetch_stock_data

        df = fetch_stock_data(
            self.ticker,
            self.start_date,
            self.end_date,
            use_cache=self.use_cache,
        )

        if df.empty:
            raise ValueError(f"No data available for {self.ticker}")

        self._data = df
        logger.info(f"Loaded {len(df)} rows from Yahoo Finance for {self.ticker}")
        return df


class CSVSource(DataSource):
    """Data source for CSV files."""

    def __init__(
        self,
        file_path: Union[str, Path],
        column_mapping: Optional[Dict[str, str]] = None,
        date_column: Optional[str] = None,
        date_format: Optional[str] = None,
        target_timeframe: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize CSV data source.

        Args:
            file_path: Path to the CSV file
            column_mapping: Custom column name mapping
            date_column: Name of date column (auto-detected if None)
            date_format: Date format string (auto-parsed if None)
            target_timeframe: Resample to this timeframe after loading
            name: Custom name for this source
        """
        file_path = Path(file_path)
        if name is None:
            name = f"CSV:{file_path.stem}"

        super().__init__(name=name)
        self.file_path = file_path
        self.column_mapping = column_mapping
        self.date_column = date_column
        self.date_format = date_format
        self.target_timeframe = target_timeframe

    def get_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        from src.data.csv_loader import load_csv, infer_timeframe

        df = load_csv(
            self.file_path,
            column_mapping=self.column_mapping,
            date_column=self.date_column,
            date_format=self.date_format,
            target_timeframe=self.target_timeframe,
        )

        self._timeframe = infer_timeframe(df)
        self._data = df
        logger.info(f"Loaded {len(df)} rows from {self.file_path}")
        return df


class MultiSource(DataSource):
    """Combine data from multiple sources."""

    def __init__(
        self,
        sources: List[DataSource],
        name: str = "MultiSource",
    ):
        """
        Initialize multi-source data combiner.

        Args:
            sources: List of DataSource objects
            name: Name for this combined source
        """
        super().__init__(name=name)
        self.sources = sources

    def get_data(self) -> pd.DataFrame:
        """
        Get data from all sources.

        Note: For multi-source, we don't concatenate DataFrames directly.
        Instead, use prepare_training_data which handles multiple stocks correctly.
        This method returns the data from the first source for single-source operations.
        """
        if not self.sources:
            raise ValueError("No data sources configured")

        # For single operations, return first source's data
        return self.sources[0].get_data()

    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """Get data from all sources as a dictionary."""
        return {source.name: source.get_data() for source in self.sources}

    def prepare_training_data(
        self,
        window_size: int = 256,
        horizon: int = 5,
        stride: int = 1,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare combined data from all sources for training.

        Creates windows from each source and concatenates them.
        """
        from src.data.preprocessor import combine_multiple_stocks

        all_data = self.get_all_data()
        X, y = combine_multiple_stocks(
            all_data, window_size=window_size,
            horizon=horizon, stride=stride, columns=columns
        )
        return train_val_test_split(X, y, val_ratio, test_ratio)

    def get_summary(self) -> Dict:
        """Get summary for all sources."""
        summaries = {}
        total_rows = 0

        for source in self.sources:
            try:
                summary = source.get_summary()
                summaries[source.name] = summary
                total_rows += summary.get("rows", 0)
            except Exception as e:
                summaries[source.name] = {"error": str(e)}

        return {
            "name": self.name,
            "num_sources": len(self.sources),
            "total_rows": total_rows,
            "sources": summaries,
        }


class DataSourceFactory:
    """Factory for creating data sources."""

    @staticmethod
    def create(
        source_type: str,
        **kwargs,
    ) -> DataSource:
        """
        Create a data source.

        Args:
            source_type: Type of source ('yahoo', 'csv', 'multi')
            **kwargs: Arguments passed to the source constructor

        Returns:
            DataSource instance
        """
        source_type = source_type.lower()

        if source_type == "yahoo":
            return YahooFinanceSource(**kwargs)
        elif source_type == "csv":
            return CSVSource(**kwargs)
        elif source_type == "multi":
            return MultiSource(**kwargs)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    @staticmethod
    def from_config(config: Dict) -> DataSource:
        """
        Create a data source from a configuration dictionary.

        Args:
            config: Dictionary with 'type' and source-specific parameters

        Returns:
            DataSource instance
        """
        source_type = config.pop("type", "yahoo")
        return DataSourceFactory.create(source_type, **config)

    @staticmethod
    def create_yahoo(
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> YahooFinanceSource:
        """Convenience method for Yahoo Finance source."""
        return YahooFinanceSource(ticker, start_date, end_date)

    @staticmethod
    def create_csv(file_path: Union[str, Path], **kwargs) -> CSVSource:
        """Convenience method for CSV source."""
        return CSVSource(file_path, **kwargs)

    @staticmethod
    def create_multi_yahoo(
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> MultiSource:
        """Create a multi-source from multiple Yahoo Finance tickers."""
        sources = [
            YahooFinanceSource(ticker, start_date, end_date)
            for ticker in tickers
        ]
        return MultiSource(sources, name=f"Yahoo:{len(tickers)}stocks")

    @staticmethod
    def create_multi_csv(
        file_paths: List[Union[str, Path]],
        **kwargs,
    ) -> MultiSource:
        """Create a multi-source from multiple CSV files."""
        sources = [CSVSource(fp, **kwargs) for fp in file_paths]
        return MultiSource(sources, name=f"CSV:{len(file_paths)}files")


def quick_train_data(
    source: Union[str, Path, DataSource],
    window_size: int = 256,
    horizon: int = 5,
    stride: int = 1,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    **source_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick function to load data and prepare for training.

    Args:
        source: Either a DataSource, a file path (CSV), or a ticker symbol
        window_size: Size of sliding windows
        horizon: Prediction horizon
        stride: Step between windows
        val_ratio: Validation ratio
        test_ratio: Test ratio
        **source_kwargs: Additional arguments for data source

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Determine source type
    if isinstance(source, DataSource):
        data_source = source
    elif isinstance(source, (str, Path)):
        path = Path(source) if isinstance(source, str) else source
        if path.suffix.lower() == ".csv":
            data_source = CSVSource(path, **source_kwargs)
        else:
            # Assume it's a ticker symbol
            data_source = YahooFinanceSource(str(source), **source_kwargs)
    else:
        raise ValueError(f"Unknown source type: {type(source)}")

    return data_source.prepare_training_data(
        window_size=window_size,
        horizon=horizon,
        stride=stride,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
