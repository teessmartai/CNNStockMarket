"""Data fetching, preprocessing, and dataset utilities."""

from .fetcher import (
    fetch_stock_data,
    fetch_sp500_tickers,
    fetch_multiple_stocks,
    clear_cache,
)

from .preprocessor import (
    normalize,
    create_labels,
    create_sliding_windows,
    train_val_test_split,
    prepare_data,
)

__all__ = [
    # Fetcher
    'fetch_stock_data',
    'fetch_sp500_tickers',
    'fetch_multiple_stocks',
    'clear_cache',
    # Preprocessor
    'normalize',
    'create_labels',
    'create_sliding_windows',
    'train_val_test_split',
    'prepare_data',
]
