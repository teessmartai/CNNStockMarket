"""Data fetching utilities for stock market data from Yahoo Finance."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent.parent.parent / "data"


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    cache_days: int = 1,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a stock from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        use_cache: Whether to use cached data if available
        cache_days: Number of days before cache expires

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Returns empty DataFrame if ticker is invalid or no data available
    """
    CACHE_DIR.mkdir(exist_ok=True)

    cache_file = CACHE_DIR / f"{ticker}_{start_date}_{end_date}.csv"

    # Check cache
    if use_cache and cache_file.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age < timedelta(days=cache_days):
            logger.debug(f"Loading {ticker} from cache")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

    # Fetch from Yahoo Finance
    try:
        logger.info(f"Fetching {ticker} from Yahoo Finance...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=True)

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Keep only OHLCV columns
        columns_to_keep = ["Open", "High", "Low", "Close", "Volume"]
        df = df[columns_to_keep]

        # Save to cache
        if use_cache:
            df.to_csv(cache_file)
            logger.debug(f"Cached {ticker} data to {cache_file}")

        return df

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def fetch_sp500_tickers() -> List[str]:
    """
    Fetch the current list of S&P 500 constituent tickers.

    Returns:
        List of ticker symbols for S&P 500 companies
    """
    try:
        # Fetch from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]

        # Extract ticker symbols
        tickers = df["Symbol"].tolist()

        # Clean up tickers (some have special characters)
        tickers = [t.replace(".", "-") for t in tickers]

        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers

    except Exception as e:
        logger.error(f"Error fetching S&P 500 list: {e}")
        # Return a fallback list of major tickers
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA", "META", "TSLA",
            "BRK-B", "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD", "CVX", "MA",
            "ABBV", "MRK", "LLY", "PFE", "KO", "PEP", "COST", "TMO", "AVGO",
            "MCD", "WMT", "CSCO", "ACN", "ABT", "DHR", "VZ", "ADBE", "CRM",
            "CMCSA", "NKE", "NFLX", "INTC", "TXN", "AMD", "PM", "WFC", "BMY",
            "UPS", "RTX", "MS", "T",
        ]


def fetch_multiple_stocks(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks.

    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        use_cache: Whether to use cached data
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    results = {}

    iterator = tqdm(tickers, desc="Fetching stocks") if show_progress else tickers

    for ticker in iterator:
        df = fetch_stock_data(ticker, start_date, end_date, use_cache=use_cache)
        if not df.empty:
            results[ticker] = df
        else:
            logger.warning(f"Skipping {ticker} - no data available")

    logger.info(f"Successfully fetched {len(results)}/{len(tickers)} stocks")
    return results


def clear_cache() -> int:
    """
    Clear all cached stock data files.

    Returns:
        Number of files deleted
    """
    count = 0
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob("*.csv"):
            cache_file.unlink()
            count += 1
    logger.info(f"Cleared {count} cached files")
    return count
