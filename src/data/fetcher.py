"""Data fetching module for retrieving stock data from Yahoo Finance."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / 'data'


def _get_cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    """Generate cache file path for a stock data request."""
    filename = f"{ticker}_{start_date}_{end_date}.csv"
    return CACHE_DIR / filename


def _load_from_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load data from cache if available."""
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
            logger.info(f"Loaded from cache: {cache_path.name}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
    return None


def _save_to_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save data to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        logger.info(f"Saved to cache: {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single stock from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        use_cache: Whether to use local caching (default: True)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Empty DataFrame if fetch fails
    """
    cache_path = _get_cache_path(ticker, start_date, end_date)

    # Try to load from cache first
    if use_cache:
        cached_df = _load_from_cache(cache_path)
        if cached_df is not None:
            return cached_df

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return pd.DataFrame()

        # Keep only OHLCV columns
        columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[columns_to_keep]

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'

        logger.info(f"Fetched {len(df)} rows for {ticker} from Yahoo Finance")

        # Save to cache
        if use_cache:
            _save_to_cache(df, cache_path)

        return df

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def fetch_sp500_tickers() -> List[str]:
    """
    Fetch current S&P 500 constituent tickers from Wikipedia.

    Returns:
        List of ticker symbols
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()

        # Clean up tickers (some have dots that need to be replaced for Yahoo Finance)
        tickers = [ticker.replace('.', '-') for ticker in tickers]

        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers

    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        return []


def fetch_multiple_stocks(
    tickers: List[str],
    start_date: str,
    end_date: str,
    show_progress: bool = True,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks.

    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        show_progress: Whether to show progress bar
        use_cache: Whether to use local caching (default: True)

    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    results = {}

    iterator = tqdm(tickers, desc="Fetching stocks") if show_progress else tickers

    for ticker in iterator:
        df = fetch_stock_data(ticker, start_date, end_date, use_cache=use_cache)
        if not df.empty:
            results[ticker] = df

    logger.info(f"Successfully fetched data for {len(results)}/{len(tickers)} stocks")
    return results


def clear_cache(ticker: Optional[str] = None) -> int:
    """
    Clear cached stock data files.

    Args:
        ticker: Specific ticker to clear, or None to clear all cache

    Returns:
        Number of files deleted
    """
    deleted = 0

    if not CACHE_DIR.exists():
        return 0

    if ticker:
        # Delete files for specific ticker
        for cache_file in CACHE_DIR.glob(f"{ticker}_*.csv"):
            try:
                cache_file.unlink()
                deleted += 1
                logger.info(f"Deleted cache: {cache_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")
    else:
        # Delete all cache files
        for cache_file in CACHE_DIR.glob("*.csv"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")
        logger.info(f"Cleared {deleted} cache files")

    return deleted


if __name__ == "__main__":
    # Test the fetcher with caching
    print("Testing fetch_stock_data with caching...")

    # First fetch (from API)
    print("\nFirst fetch (should be from API):")
    df = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
    print(f"AAPL: {len(df)} rows, columns: {list(df.columns)}")

    # Second fetch (from cache)
    print("\nSecond fetch (should be from cache):")
    df = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
    print(f"AAPL: {len(df)} rows")

    print("\nTesting fetch_sp500_tickers...")
    tickers = fetch_sp500_tickers()
    print(f"S&P 500: {len(tickers)} tickers")
    print(f"First 10: {tickers[:10]}")

    print("\nTesting clear_cache...")
    deleted = clear_cache('AAPL')
    print(f"Deleted {deleted} cache files")
