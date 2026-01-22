"""Pre-configured presets for different asset classes and timeframes.

This module provides sensible default configurations for various trading scenarios:
- Stock daily trading
- Cryptocurrency (24/7 markets)
- Forex trading
- Intraday strategies

Each preset includes appropriate window sizes, prediction horizons, and trading
ratios adjusted for the market's characteristics.

Usage:
    from src.utils.presets import get_preset, list_presets

    # Get a preset configuration
    config = get_preset("crypto_hourly")
    print(config)

    # Use preset for training
    from src.utils.presets import apply_preset
    apply_preset("stock_daily")  # Updates global config
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class AssetClass(str, Enum):
    """Asset class categories."""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    COMMODITY = "commodity"


@dataclass
class Preset:
    """Configuration preset for a specific trading scenario."""

    name: str
    description: str
    asset_class: AssetClass
    timeframe: str
    window_size: int
    horizons: List[int]
    trading_days_ratio: float  # Ratio of trading days per calendar year
    min_data_rows: int  # Recommended minimum data points
    sample_mode: str = "overlapping"
    sample_stride: int = 1

    def to_dict(self) -> Dict:
        """Convert preset to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "asset_class": self.asset_class.value,
            "timeframe": self.timeframe,
            "window_size": self.window_size,
            "horizons": self.horizons,
            "trading_days_ratio": self.trading_days_ratio,
            "min_data_rows": self.min_data_rows,
            "sample_mode": self.sample_mode,
            "sample_stride": self.sample_stride,
        }

    def get_config(self) -> Dict:
        """Get configuration parameters for training."""
        return {
            "window_size": self.window_size,
            "horizons": self.horizons,
            "timeframe": self.timeframe,
            "sample_mode": self.sample_mode,
            "sample_stride": self.sample_stride,
        }


# Pre-defined presets
PRESETS: Dict[str, Preset] = {
    # =========================================================================
    # STOCK PRESETS
    # =========================================================================
    "stock_daily": Preset(
        name="stock_daily",
        description="Daily stock data (S&P 500 style, as in original paper)",
        asset_class=AssetClass.STOCK,
        timeframe="1d",
        window_size=256,
        horizons=[5, 30],
        trading_days_ratio=0.67,  # ~252 trading days / 365 calendar days
        min_data_rows=512,  # ~2 years of daily data
        sample_mode="overlapping",
    ),

    "stock_weekly": Preset(
        name="stock_weekly",
        description="Weekly stock data for longer-term predictions",
        asset_class=AssetClass.STOCK,
        timeframe="1w",
        window_size=104,  # ~2 years of weekly data
        horizons=[4, 12],  # 1 month, 3 months ahead
        trading_days_ratio=0.67,
        min_data_rows=208,  # ~4 years of weekly data
        sample_mode="overlapping",
    ),

    # =========================================================================
    # CRYPTOCURRENCY PRESETS
    # =========================================================================
    "crypto_daily": Preset(
        name="crypto_daily",
        description="Daily cryptocurrency data (24/7 markets)",
        asset_class=AssetClass.CRYPTO,
        timeframe="1d",
        window_size=256,
        horizons=[5, 30],
        trading_days_ratio=1.0,  # Crypto trades 365 days/year
        min_data_rows=512,
        sample_mode="overlapping",
    ),

    "crypto_hourly": Preset(
        name="crypto_hourly",
        description="Hourly cryptocurrency data for short-term trading",
        asset_class=AssetClass.CRYPTO,
        timeframe="1h",
        window_size=168,  # 1 week of hourly data
        horizons=[6, 24],  # 6 hours, 24 hours ahead
        trading_days_ratio=1.0,
        min_data_rows=720,  # ~1 month of hourly data
        sample_mode="overlapping",
    ),

    "crypto_4h": Preset(
        name="crypto_4h",
        description="4-hour cryptocurrency data (swing trading)",
        asset_class=AssetClass.CRYPTO,
        timeframe="4h",
        window_size=180,  # ~30 days of 4h data
        horizons=[6, 42],  # 1 day, 1 week ahead
        trading_days_ratio=1.0,
        min_data_rows=360,  # ~2 months
        sample_mode="overlapping",
    ),

    # =========================================================================
    # FOREX PRESETS
    # =========================================================================
    "forex_daily": Preset(
        name="forex_daily",
        description="Daily forex data",
        asset_class=AssetClass.FOREX,
        timeframe="1d",
        window_size=256,
        horizons=[5, 20],  # 1 week, 1 month (forex weeks)
        trading_days_ratio=0.71,  # ~260 trading days / 365
        min_data_rows=512,
        sample_mode="overlapping",
    ),

    "forex_4h": Preset(
        name="forex_4h",
        description="4-hour forex data for swing trading",
        asset_class=AssetClass.FOREX,
        timeframe="4h",
        window_size=180,  # ~30 trading days
        horizons=[6, 30],  # 1 day, 5 days ahead
        trading_days_ratio=0.71,
        min_data_rows=360,
        sample_mode="overlapping",
    ),

    "forex_1h": Preset(
        name="forex_1h",
        description="Hourly forex data for day trading",
        asset_class=AssetClass.FOREX,
        timeframe="1h",
        window_size=120,  # 5 trading days (24h each)
        horizons=[4, 24],  # 4 hours, 1 day ahead
        trading_days_ratio=0.71,
        min_data_rows=480,  # ~20 days
        sample_mode="overlapping",
    ),

    # =========================================================================
    # INTRADAY PRESETS
    # =========================================================================
    "intraday_15m": Preset(
        name="intraday_15m",
        description="15-minute data for intraday trading",
        asset_class=AssetClass.STOCK,
        timeframe="15m",
        window_size=104,  # ~1 trading day (26 bars) x 4 days
        horizons=[4, 26],  # 1 hour, 1 day ahead
        trading_days_ratio=0.67,
        min_data_rows=520,  # ~20 trading days
        sample_mode="overlapping",
    ),

    "intraday_5m": Preset(
        name="intraday_5m",
        description="5-minute data for scalping",
        asset_class=AssetClass.STOCK,
        timeframe="5m",
        window_size=156,  # ~2 trading days (78 bars each)
        horizons=[12, 78],  # 1 hour, 1 day ahead
        trading_days_ratio=0.67,
        min_data_rows=780,  # ~10 trading days
        sample_mode="strided",
        sample_stride=5,  # Reduce sample correlation
    ),

    "intraday_1m": Preset(
        name="intraday_1m",
        description="1-minute data for high-frequency strategies",
        asset_class=AssetClass.STOCK,
        timeframe="1m",
        window_size=390,  # 1 full trading day
        horizons=[15, 60],  # 15 minutes, 1 hour ahead
        trading_days_ratio=0.67,
        min_data_rows=1950,  # ~5 trading days
        sample_mode="strided",
        sample_stride=10,
    ),

    # =========================================================================
    # FUTURES PRESETS
    # =========================================================================
    "futures_daily": Preset(
        name="futures_daily",
        description="Daily futures data",
        asset_class=AssetClass.FUTURES,
        timeframe="1d",
        window_size=256,
        horizons=[5, 20],
        trading_days_ratio=0.67,
        min_data_rows=512,
        sample_mode="overlapping",
    ),

    # =========================================================================
    # NON-OVERLAPPING VARIANTS (for abundant data)
    # =========================================================================
    "crypto_hourly_independent": Preset(
        name="crypto_hourly_independent",
        description="Hourly crypto with non-overlapping samples (needs more data)",
        asset_class=AssetClass.CRYPTO,
        timeframe="1h",
        window_size=168,
        horizons=[6, 24],
        trading_days_ratio=1.0,
        min_data_rows=8760,  # 1 year of hourly data
        sample_mode="non_overlapping",
    ),

    "intraday_5m_independent": Preset(
        name="intraday_5m_independent",
        description="5-minute with non-overlapping samples",
        asset_class=AssetClass.STOCK,
        timeframe="5m",
        window_size=156,
        horizons=[12, 78],
        trading_days_ratio=0.67,
        min_data_rows=7800,  # ~100 trading days
        sample_mode="non_overlapping",
    ),
}


def list_presets() -> List[str]:
    """List all available preset names."""
    return list(PRESETS.keys())


def get_preset(name: str) -> Preset:
    """
    Get a preset by name.

    Args:
        name: Preset name

    Returns:
        Preset object

    Raises:
        KeyError: If preset doesn't exist
    """
    if name not in PRESETS:
        available = ", ".join(list_presets())
        raise KeyError(f"Unknown preset: {name}. Available: {available}")
    return PRESETS[name]


def get_preset_config(name: str) -> Dict:
    """
    Get preset configuration as dictionary.

    Args:
        name: Preset name

    Returns:
        Configuration dictionary suitable for training
    """
    return get_preset(name).get_config()


def get_presets_by_asset_class(asset_class: AssetClass) -> List[Preset]:
    """Get all presets for a specific asset class."""
    return [p for p in PRESETS.values() if p.asset_class == asset_class]


def get_presets_by_timeframe(timeframe: str) -> List[Preset]:
    """Get all presets for a specific timeframe."""
    return [p for p in PRESETS.values() if p.timeframe == timeframe]


def describe_preset(name: str) -> str:
    """Get a human-readable description of a preset."""
    preset = get_preset(name)
    return f"""
Preset: {preset.name}
{'=' * 50}
Description: {preset.description}
Asset Class: {preset.asset_class.value}
Timeframe: {preset.timeframe}

Window Size: {preset.window_size} periods
Horizons: {preset.horizons}
Trading Days Ratio: {preset.trading_days_ratio:.0%}
Minimum Data Rows: {preset.min_data_rows}

Sample Mode: {preset.sample_mode}
Sample Stride: {preset.sample_stride}
"""


def print_all_presets():
    """Print summary of all available presets."""
    print("=" * 70)
    print("Available Configuration Presets")
    print("=" * 70)

    # Group by asset class
    by_class = {}
    for preset in PRESETS.values():
        ac = preset.asset_class.value
        if ac not in by_class:
            by_class[ac] = []
        by_class[ac].append(preset)

    for asset_class, presets in by_class.items():
        print(f"\n[{asset_class.upper()}]")
        for p in presets:
            horizons_str = ", ".join(f"T+{h}" for h in p.horizons)
            print(f"  {p.name:30s} | {p.timeframe:4s} | W={p.window_size:4d} | {horizons_str}")

    print("\n" + "=" * 70)


def suggest_preset(
    timeframe: str,
    asset_class: Optional[str] = None,
    prefer_independent: bool = False,
) -> Optional[str]:
    """
    Suggest the best preset based on criteria.

    Args:
        timeframe: Desired timeframe
        asset_class: Desired asset class (optional)
        prefer_independent: Prefer non-overlapping sample modes

    Returns:
        Preset name or None if no match
    """
    candidates = []

    for name, preset in PRESETS.items():
        if preset.timeframe != timeframe:
            continue
        if asset_class and preset.asset_class.value != asset_class:
            continue

        score = 0
        if prefer_independent and preset.sample_mode == "non_overlapping":
            score += 10
        if preset.sample_mode == "overlapping":
            score += 5  # Default preference

        candidates.append((score, name))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def apply_preset(name: str):
    """
    Apply a preset to the global configuration.

    This modifies the global config module variables.

    Args:
        name: Preset name to apply
    """
    import src.utils.config as config

    preset = get_preset(name)
    config.WINDOW_SIZE = preset.window_size
    config.HORIZONS = preset.horizons
    config.TIMEFRAME = preset.timeframe
    config.SAMPLE_MODE = preset.sample_mode
    config.SAMPLE_STRIDE = preset.sample_stride

    print(f"Applied preset: {name}")
    print(f"  Window size: {preset.window_size}")
    print(f"  Horizons: {preset.horizons}")
    print(f"  Timeframe: {preset.timeframe}")
    print(f"  Sample mode: {preset.sample_mode}")
