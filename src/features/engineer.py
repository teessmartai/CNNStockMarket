"""
Feature engineering for stock market data.

Computes additional channels beyond raw OHLCV log-returns:

  returns     — rolling log-returns at 5d, 10d, 20d horizons (3 ch)
  gap         — overnight gap: log(open_t / close_{t-1})         (1 ch)
  volatility  — high-low spread + rolling 10d realised vol        (2 ch)
  rsi         — RSI(14)                                           (1 ch)
  macd        — MACD line + histogram (12/26/9 EMA)               (2 ch)
  bbands      — price position within 20d Bollinger band          (1 ch)

All channels are z-scored across the full per-stock series before
windowing so they live in the same numerical range as the base OHLCV
log-returns.

Cross-sectional rank normalisation (xrank) is handled separately in
preprocessor.combine_multiple_stocks: after all per-stock feature
DataFrames are assembled it aligns them by date and replaces each
feature value with its cross-sectional percentile rank (0→1) on that
day.  This makes the model completely regime-agnostic — bull markets
and bear markets look identical; only relative ordering matters.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Public API ─────────────────────────────────────────────────────────────────

#: All feature group names that can be requested
ALL_GROUPS: List[str] = ["returns", "gap", "volatility", "rsi", "macd", "bbands"]

#: Channels produced by each group
GROUP_CHANNELS = {
    "returns":    ["ret_5d", "ret_10d", "ret_20d"],
    "gap":        ["gap"],
    "volatility": ["hl_spread", "real_vol_10d"],
    "rsi":        ["rsi_14"],
    "macd":       ["macd_line", "macd_hist"],
    "bbands":     ["bb_pos"],
}


def feature_channels(groups: List[str]) -> List[str]:
    """Return ordered list of channel names for the given feature groups."""
    cols: List[str] = []
    for g in groups:
        cols.extend(GROUP_CHANNELS.get(g, []))
    return cols


def compute_features(
    df: pd.DataFrame,
    groups: List[str] | None = None,
) -> pd.DataFrame:
    """
    Compute technical feature channels for a single stock.

    Args:
        df:     Raw OHLCV DataFrame (must have Open/High/Low/Close/Volume columns).
        groups: Feature groups to compute.  None → all groups.

    Returns:
        DataFrame with original OHLCV log-return columns PLUS the
        requested feature columns, one row per trading day.  The first
        ~26 rows are dropped (longest lookback is 26-day EMA for MACD).
        All columns are z-scored across the full series.
    """
    if groups is None:
        groups = ALL_GROUPS

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    open_  = df["Open"]
    volume = df["Volume"]

    feat: dict[str, pd.Series] = {}

    # ── Base: OHLCV log-returns (same as existing to_log_returns) ──────────────
    feat["Open"]   = np.log(open_  / open_.shift(1))
    feat["High"]   = np.log(high   / high.shift(1))
    feat["Low"]    = np.log(low    / low.shift(1))
    feat["Close"]  = np.log(close  / close.shift(1))
    vol_s = volume.replace(0, np.nan)
    feat["Volume"] = np.log(vol_s  / vol_s.shift(1))

    # ── Rolling log-returns ────────────────────────────────────────────────────
    if "returns" in groups:
        for k in (5, 10, 20):
            feat[f"ret_{k}d"] = np.log(close / close.shift(k))

    # ── Overnight gap ──────────────────────────────────────────────────────────
    if "gap" in groups:
        feat["gap"] = np.log(open_ / close.shift(1))

    # ── Volatility ─────────────────────────────────────────────────────────────
    if "volatility" in groups:
        # Intraday high-low spread (normalised by close)
        feat["hl_spread"] = np.log(high / low)
        # 10-day realised volatility (std of daily log-returns)
        daily_lr = np.log(close / close.shift(1))
        feat["real_vol_10d"] = daily_lr.rolling(10).std()

    # ── RSI(14) ────────────────────────────────────────────────────────────────
    if "rsi" in groups:
        feat["rsi_14"] = _rsi(close, period=14)

    # ── MACD (12/26/9) ─────────────────────────────────────────────────────────
    if "macd" in groups:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        feat["macd_line"] = macd_line / close          # normalise by price level
        feat["macd_hist"] = (macd_line - macd_signal) / close

    # ── Bollinger Band position ────────────────────────────────────────────────
    if "bbands" in groups:
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std()
        upper  = sma20 + 2 * std20
        lower  = sma20 - 2 * std20
        width  = upper - lower
        # Position in [0, 1]: 0 = at lower band, 1 = at upper band
        feat["bb_pos"] = np.where(width > 0, (close - lower) / width, 0.5)
        feat["bb_pos"] = pd.Series(feat["bb_pos"], index=close.index)

    # ── Assemble and clean ─────────────────────────────────────────────────────
    out = pd.DataFrame(feat, index=df.index)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    # Z-score each column across the full per-stock series
    for col in out.columns:
        std = out[col].std()
        if std > 1e-9:
            out[col] = (out[col] - out[col].mean()) / std
        else:
            out[col] = 0.0

    return out


# ── Cross-sectional rank normalisation ────────────────────────────────────────

def cross_sectional_rank(
    stock_features: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Replace per-stock feature values with their cross-sectional percentile
    rank on each trading day.

    For each date d and each feature channel f, the value for stock s becomes:
        rank(value_s,d,f) / count(stocks with data on d)
    so all values lie in (0, 1].

    Stocks with missing data on a given date are excluded from that day's
    ranking and keep NaN (they are later dropped when windows are created).

    Args:
        stock_features: Dict  ticker → DataFrame(date_index, feature_columns)
                        All DataFrames must share the same column names.

    Returns:
        Dict ticker → DataFrame with the same shape but rank-normalised values.
    """
    if not stock_features:
        return stock_features

    tickers = list(stock_features.keys())
    cols    = stock_features[tickers[0]].columns.tolist()

    # Collect all dates across all stocks
    all_dates = sorted(
        set().union(*[set(df.index) for df in stock_features.values()])
    )

    # Build a wide DataFrame for each feature: rows=dates, cols=tickers
    ranked: dict[str, dict[str, pd.Series]] = {t: {} for t in tickers}

    for col in cols:
        # Wide matrix: shape (n_dates, n_tickers)
        wide = pd.DataFrame(
            {t: stock_features[t][col] for t in tickers},
            index=all_dates,
        )
        # Rank across columns (stocks) for each row (date), pct=True → [0,1]
        wide_ranked = wide.rank(axis=1, pct=True, na_option="keep")
        for t in tickers:
            ranked[t][col] = wide_ranked[t]

    # Reassemble per-ticker DataFrames
    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        result[t] = pd.DataFrame(ranked[t], index=all_dates)
        # Restore original index (only dates where this stock had data)
        original_idx = stock_features[t].index
        result[t] = result[t].loc[result[t].index.isin(original_idx)].copy()
        result[t].dropna(inplace=True)

    logger.info(
        f"Cross-sectional rank applied across {len(tickers)} stocks, "
        f"{len(cols)} features, {len(all_dates)} dates"
    )
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI, returns series in [0, 1]."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    # Wilder smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 1 - (1 / (1 + rs))   # normalised to [0, 1]
    return rsi.fillna(0.5)
