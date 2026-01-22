# Data Formats Guide (Phase 6)

This document describes the data formats supported by the CNN Stock Market Prediction system for custom data training.

## Table of Contents

1. [CSV Format Requirements](#csv-format-requirements)
2. [Column Naming Conventions](#column-naming-conventions)
3. [Date/Time Formats](#datetime-formats)
4. [Supported Timeframes](#supported-timeframes)
5. [Example CSV Files](#example-csv-files)
6. [Data Validation](#data-validation)
7. [Troubleshooting](#troubleshooting)
8. [Recommended Data Sources](#recommended-data-sources)

---

## CSV Format Requirements

### Required Columns

Your CSV file must contain OHLCV (Open, High, Low, Close, Volume) data:

| Column | Description | Required |
|--------|-------------|----------|
| Open | Opening price for the period | Yes |
| High | Highest price during the period | Yes |
| Low | Lowest price during the period | Yes |
| Close | Closing price for the period | Yes |
| Volume | Trading volume for the period | Yes |
| Date/Time | Timestamp for the data point | Recommended |

### Basic Format Example

```csv
date,open,high,low,close,volume
2024-01-01,100.00,102.50,99.50,101.25,1000000
2024-01-02,101.25,103.00,100.00,102.50,1500000
2024-01-03,102.50,104.00,101.00,103.75,1200000
```

---

## Column Naming Conventions

The CSV loader automatically detects common column name variations. The following names are recognized (case-insensitive):

### Open Price
- `open`, `o`, `open_price`, `openprice`, `first`

### High Price
- `high`, `h`, `high_price`, `highprice`, `max`

### Low Price
- `low`, `l`, `low_price`, `lowprice`, `min`

### Close Price
- `close`, `c`, `close_price`, `closeprice`, `last`, `adj close`, `adjclose`, `adjusted close`

### Volume
- `volume`, `v`, `vol`, `qty`, `quantity`, `amount`

### Date/Time
- `date`, `datetime`, `time`, `timestamp`, `ts`, `dt`, `trade_date`, `tradedate`, `period`, `index`

### Custom Mapping

If your columns don't match these conventions, you can specify a custom mapping:

```python
from src.data.csv_loader import load_csv

df = load_csv(
    "my_data.csv",
    column_mapping={
        "Open": "price_open",
        "High": "price_high",
        "Low": "price_low",
        "Close": "price_close",
        "Volume": "trade_volume"
    },
    date_column="trading_date"
)
```

---

## Date/Time Formats

### Supported Formats

The loader auto-detects most common date formats:

| Format | Example |
|--------|---------|
| ISO 8601 | `2024-01-15T09:30:00` |
| Date only | `2024-01-15` |
| US format | `01/15/2024` |
| European format | `15/01/2024` |
| Unix timestamp | `1705312200` |
| With timezone | `2024-01-15T09:30:00-05:00` |

### Specifying Date Format

If auto-detection fails, specify the format explicitly:

```python
df = load_csv(
    "my_data.csv",
    date_format="%Y-%m-%d %H:%M:%S"
)
```

Common format codes:
- `%Y` - 4-digit year (2024)
- `%m` - 2-digit month (01-12)
- `%d` - 2-digit day (01-31)
- `%H` - Hour (00-23)
- `%M` - Minute (00-59)
- `%S` - Second (00-59)

---

## Supported Timeframes

The system supports various data timeframes:

| Timeframe | Code | Typical Use Case |
|-----------|------|------------------|
| 1 minute | `1m` | High-frequency trading, scalping |
| 5 minutes | `5m` | Day trading, scalping |
| 15 minutes | `15m` | Day trading |
| 30 minutes | `30m` | Intraday swing trading |
| 1 hour | `1h` | Swing trading |
| 4 hours | `4h` | Swing trading, position trading |
| 1 day | `1d` | Position trading, investing |
| 1 week | `1w` | Long-term investing |

### Timeframe Detection

The system automatically infers the timeframe from your data:

```python
from src.data.csv_loader import infer_timeframe

df = load_csv("my_data.csv")
timeframe = infer_timeframe(df)
print(f"Detected timeframe: {timeframe}")  # e.g., "1h"
```

### Resampling Data

You can resample data to a different timeframe:

```python
from src.data.csv_loader import load_csv, resample_ohlcv

# Load 5-minute data and resample to 1-hour
df = load_csv("5min_data.csv")
df_hourly = resample_ohlcv(df, "1h")
```

Or during loading:

```python
df = load_csv("5min_data.csv", target_timeframe="1h")
```

---

## Example CSV Files

### Daily Stock Data

```csv
Date,Open,High,Low,Close,Volume
2024-01-02,187.15,188.44,183.89,185.64,82488700
2024-01-03,184.22,185.88,183.43,184.25,58414500
2024-01-04,182.15,183.09,180.88,181.91,71983600
2024-01-05,181.99,182.76,180.17,181.18,62303300
```

### Hourly Cryptocurrency Data

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42000.00,42150.00,41900.00,42100.00,1523.45
2024-01-01 01:00:00,42100.00,42300.00,42050.00,42250.00,1842.32
2024-01-01 02:00:00,42250.00,42400.00,42200.00,42380.00,1654.21
```

### Forex 4-Hour Data

```csv
datetime,o,h,l,c,vol
2024-01-02 00:00,1.1050,1.1065,1.1040,1.1055,50000
2024-01-02 04:00,1.1055,1.1070,1.1045,1.1060,45000
2024-01-02 08:00,1.1060,1.1080,1.1055,1.1075,65000
```

### Intraday 5-Minute Data

```csv
time,open_price,high_price,low_price,close_price,quantity
2024-01-02 09:30:00,187.15,187.45,187.00,187.30,125000
2024-01-02 09:35:00,187.30,187.60,187.25,187.55,98000
2024-01-02 09:40:00,187.55,187.80,187.40,187.70,112000
```

---

## Data Validation

The loader performs automatic validation checks:

### Validation Checks

1. **Required Columns**: Verifies all OHLCV columns are present
2. **Missing Values**: Detects and reports NaN/null values
3. **Chronological Order**: Ensures data is sorted by time
4. **Duplicate Timestamps**: Identifies duplicate entries
5. **OHLC Relationships**: Validates High >= Low, High >= max(Open, Close), Low <= min(Open, Close)
6. **Negative Values**: Checks for negative prices or volumes

### Validation Example

```python
from src.data.csv_loader import load_csv, validate_ohlcv_data

df = load_csv("my_data.csv", validate=False)  # Load without validation
is_valid, issues = validate_ohlcv_data(df, strict=False)

if issues:
    print("Data issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### Handling Invalid Data

```python
# Load with automatic fixing
df = load_csv(
    "my_data.csv",
    fill_missing=True,  # Forward-fill missing prices, zero-fill volume
    validate=True       # Validate after loading
)
```

---

## Troubleshooting

### Common Issues

#### "Missing required column: Close"

**Cause**: The column name wasn't recognized.

**Solution**: Use custom column mapping:
```python
df = load_csv("data.csv", column_mapping={"Close": "last_price"})
```

#### "Could not convert index to DatetimeIndex"

**Cause**: Date format not recognized.

**Solution**: Specify the date format:
```python
df = load_csv("data.csv", date_format="%d/%m/%Y %H:%M")
```

#### "Insufficient data: need X rows, got Y"

**Cause**: Not enough data points for the configured window size.

**Solution**: Either use more data or reduce window size:
```python
from src.utils.presets import get_preset

# Use a preset with smaller window
preset = get_preset("intraday_15m")  # window_size=104
```

#### "Found X rows where High < Low"

**Cause**: Data quality issues in your source.

**Solution**: Clean the data or use a more reliable source:
```python
# Swap High/Low if inverted
df['High'], df['Low'] = df[['High', 'Low']].max(axis=1), df[['High', 'Low']].min(axis=1)
```

#### "Data is not in chronological order"

**Cause**: Data not sorted by time.

**Solution**: The loader automatically sorts data, but you can verify:
```python
df = load_csv("data.csv")  # Auto-sorted
assert df.index.is_monotonic_increasing
```

---

## Recommended Data Sources

### Free Sources

| Source | Assets | Timeframes | Notes |
|--------|--------|------------|-------|
| Yahoo Finance | Stocks, ETFs | Daily, Weekly | Built-in support via yfinance |
| Alpha Vantage | Stocks, Forex, Crypto | Various | Free API with limits |
| CoinGecko | Crypto | Daily | Free API |
| FRED | Economic data | Various | Federal Reserve data |

### Paid Sources

| Source | Assets | Timeframes | Notes |
|--------|--------|------------|-------|
| Polygon.io | US Stocks | 1min+ | Quality intraday data |
| Binance | Crypto | 1min+ | Comprehensive crypto data |
| Interactive Brokers | Various | Various | Broker data feed |
| Quandl | Various | Various | Alternative data |

### Data Preparation Tips

1. **Clean your data**: Remove corporate actions, splits, and dividends or use adjusted prices
2. **Fill gaps**: Handle market holidays and non-trading hours appropriately
3. **Normalize volume**: Different assets have vastly different volume scales
4. **Check for outliers**: Remove obviously erroneous data points
5. **Verify timestamps**: Ensure consistent timezone handling

---

## Quick Start Example

```python
from src.data.csv_loader import load_csv
from src.data.data_source import CSVSource
from src.utils.presets import get_preset

# 1. Load your CSV
df = load_csv("my_crypto_hourly_data.csv")

# 2. Create data source
source = CSVSource("my_crypto_hourly_data.csv")

# 3. Get appropriate preset
preset = get_preset("crypto_hourly")

# 4. Prepare training data
X_train, y_train, X_val, y_val, X_test, y_test = source.prepare_training_data(
    window_size=preset.window_size,
    horizon=preset.horizons[0],
    stride=1  # overlapping samples
)

print(f"Training samples: {len(X_train)}")
print(f"Ready for training!")
```

---

## Support

For additional help:
- Check the `04_custom_data_training.ipynb` notebook for examples
- Review the `src/data/csv_loader.py` source code for all options
- Open an issue on GitHub for bugs or feature requests
