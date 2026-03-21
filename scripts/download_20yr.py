#!/usr/bin/env python3
"""Download up to 20 years of OHLCV data for all largecap-stable tickers."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd

TICKERS = [
    "AAOI","ABT","AEP","AON","APD","AWK","AXP","BAX","BDX","BMY","BRK-B",
    "CAT","CL","CLX","CPB","CVX","D","DE","DTE","DUK","ECL","ED","EMR","ES",
    "EW","FE","GE","GIS","HON","HRL","HSY","ITW","JNJ","JPM","KMB","KO",
    "LIN","MA","MDT","MKC","MMC","MMM","MRK","NEE","PEP","PG","SHW","SO",
    "SYK","T","V","VZ","WEC","XEL","XOM"
]

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 20 + 5)).strftime("%Y-%m-%d")
OUT_DIR    = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

print(f"Downloading {len(TICKERS)} tickers  {START_DATE} → {END_DATE}")
print(f"Output: {OUT_DIR}\n")

ok, failed = [], []
for ticker in TICKERS:
    out_file = OUT_DIR / f"{ticker}_{START_DATE}_{END_DATE}.csv"
    if out_file.exists():
        print(f"  {ticker:8s} already exists — skip")
        ok.append(ticker)
        continue
    try:
        df = yf.Ticker(ticker).history(start=START_DATE, end=END_DATE, auto_adjust=True)
        if df.empty:
            print(f"  {ticker:8s} ✗ no data")
            failed.append(ticker)
            continue
        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
        df = df[["Open","High","Low","Close","Volume"]]
        df.to_csv(out_file)
        years = (df.index[-1] - df.index[0]).days / 365
        print(f"  {ticker:8s} ✓  {len(df):5d} rows  ({years:.1f} yrs)")
        ok.append(ticker)
    except Exception as e:
        print(f"  {ticker:8s} ✗ {e}")
        failed.append(ticker)

print(f"\nDone: {len(ok)} ok, {len(failed)} failed")
if failed:
    print(f"Failed: {failed}")
