"""
Data fetching module for stock market data.

Provides functions to fetch historical OHLCV data from Yahoo Finance
for individual stocks and the S&P 500 index constituents.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single stock from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cache_dir: Optional directory to cache data. If provided, will check
                   for cached data before fetching.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Empty DataFrame if ticker is invalid or no data available.
    """
    # Check cache first if cache_dir is provided
    if cache_dir is not None:
        cache_path = _get_cache_path(cache_dir, ticker, start_date, end_date)
        if cache_path.exists():
            logger.info(f"Loading {ticker} from cache: {cache_path}")
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        logger.info(f"Fetching {ticker} from Yahoo Finance ({start_date} to {end_date})")

        # Download data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return pd.DataFrame()

        # Keep only OHLCV columns, rename if needed
        columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[columns_to_keep]

        # Remove timezone info from index for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Save to cache if cache_dir is provided
        if cache_dir is not None:
            cache_path = _get_cache_path(cache_dir, ticker, start_date, end_date)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)
            logger.info(f"Cached {ticker} to {cache_path}")

        return df

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def fetch_sp500_tickers() -> List[str]:
    """
    Fetch the current list of S&P 500 constituent tickers.

    Uses Wikipedia's S&P 500 companies table as the source.

    Returns:
        List of ticker symbols (e.g., ['AAPL', 'MSFT', ...])
    """
    try:
        # Read S&P 500 companies from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)

        # First table contains the current constituents
        sp500_table = tables[0]

        # The ticker column is named 'Symbol'
        tickers = sp500_table['Symbol'].tolist()

        # Clean up tickers (remove periods, etc. for Yahoo Finance compatibility)
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
    cache_dir: Optional[Path] = None,
    show_progress: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks.

    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cache_dir: Optional directory to cache data
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping ticker -> DataFrame
        Tickers with no data are excluded from the result.
    """
    results = {}

    iterator = tqdm(tickers, desc="Fetching stocks") if show_progress else tickers

    for ticker in iterator:
        df = fetch_stock_data(ticker, start_date, end_date, cache_dir)
        if not df.empty:
            results[ticker] = df
        else:
            logger.warning(f"Skipping {ticker} - no data available")

    logger.info(f"Successfully fetched {len(results)}/{len(tickers)} stocks")
    return results


def _get_cache_path(cache_dir: Path, ticker: str, start_date: str, end_date: str) -> Path:
    """Generate cache file path for a ticker and date range."""
    # Sanitize dates for filename
    start_clean = start_date.replace('-', '')
    end_clean = end_date.replace('-', '')
    filename = f"{ticker}_{start_clean}_{end_clean}.csv"
    return Path(cache_dir) / filename


def clear_cache(cache_dir: Path) -> int:
    """
    Clear all cached data files.

    Args:
        cache_dir: Directory containing cached files

    Returns:
        Number of files deleted
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return 0

    count = 0
    for file in cache_path.glob("*.csv"):
        file.unlink()
        count += 1

    logger.info(f"Cleared {count} cached files from {cache_dir}")
    return count
