"""Data fetching module for retrieving stock data from Yahoo Finance."""

import logging
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single stock from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Empty DataFrame if fetch fails
    """
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

        logger.info(f"Fetched {len(df)} rows for {ticker}")
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
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks.

    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    results = {}

    iterator = tqdm(tickers, desc="Fetching stocks") if show_progress else tickers

    for ticker in iterator:
        df = fetch_stock_data(ticker, start_date, end_date)
        if not df.empty:
            results[ticker] = df

    logger.info(f"Successfully fetched data for {len(results)}/{len(tickers)} stocks")
    return results


if __name__ == "__main__":
    # Test the fetcher
    print("Testing fetch_stock_data...")
    df = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
    print(f"AAPL: {len(df)} rows, columns: {list(df.columns)}")
    print(df.head())

    print("\nTesting fetch_sp500_tickers...")
    tickers = fetch_sp500_tickers()
    print(f"S&P 500: {len(tickers)} tickers")
    print(f"First 10: {tickers[:10]}")
