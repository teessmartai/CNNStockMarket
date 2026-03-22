#!/usr/bin/env python3
"""Download 20yr OHLCV data for all S&P 500 tickers not yet in data/."""

import json, sys, time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
import yfinance as yf

MISSING_FILE = Path("/tmp/sp500_missing.json")
OUT_DIR      = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 20 + 5)).strftime("%Y-%m-%d")

tickers = json.loads(MISSING_FILE.read_text()) if MISSING_FILE.exists() else []
print(f"Downloading {len(tickers)} tickers  {START_DATE} → {END_DATE}")
print(f"Output: {OUT_DIR}\n")

ok, failed = [], []
for i, ticker in enumerate(tickers, 1):
    out_file = OUT_DIR / f"{ticker}_{START_DATE}_{END_DATE}.csv"
    if out_file.exists():
        print(f"  [{i:3d}/{len(tickers)}] {ticker:8s} already exists — skip")
        ok.append(ticker)
        continue
    try:
        df = yf.Ticker(ticker).history(start=START_DATE, end=END_DATE, auto_adjust=True)
        if df.empty:
            print(f"  [{i:3d}/{len(tickers)}] {ticker:8s} ✗ no data")
            failed.append(ticker)
            continue
        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
        df = df[["Open","High","Low","Close","Volume"]]
        df.to_csv(out_file)
        years = (df.index[-1] - df.index[0]).days / 365
        print(f"  [{i:3d}/{len(tickers)}] {ticker:8s} ✓  {len(df):5d} rows  ({years:.1f} yrs)")
        ok.append(ticker)
    except Exception as e:
        print(f"  [{i:3d}/{len(tickers)}] {ticker:8s} ✗ {e}")
        failed.append(ticker)
    if i % 50 == 0:
        time.sleep(1)  # gentle rate limit

print(f"\n{'='*50}")
print(f"Done: {len(ok)} ok, {len(failed)} failed")
if failed:
    print(f"Failed ({len(failed)}): {failed}")
