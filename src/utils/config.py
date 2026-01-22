"""Configuration and hyperparameters for the CNN Stock Market Prediction model."""

from pathlib import Path
from enum import Enum
from typing import List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CUSTOM_DATA_DIR = DATA_DIR / "custom"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CUSTOM_DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# TIMEFRAME CONFIGURATION (Phase 6)
# ============================================================================

class Timeframe(str, Enum):
    """Supported data timeframes."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class SampleMode(str, Enum):
    """Training sample generation modes."""
    OVERLAPPING = "overlapping"      # stride=1, maximum samples, high correlation
    STRIDED = "strided"              # configurable stride, balanced
    NON_OVERLAPPING = "non_overlapping"  # stride=window+horizon, independent samples


# Default timeframe (daily data, as in original paper)
TIMEFRAME = Timeframe.DAY_1

# Timeframe info for display and validation
TIMEFRAME_INFO = {
    "1m": {"minutes": 1, "name": "1 minute", "bars_per_day": 390},
    "5m": {"minutes": 5, "name": "5 minutes", "bars_per_day": 78},
    "15m": {"minutes": 15, "name": "15 minutes", "bars_per_day": 26},
    "30m": {"minutes": 30, "name": "30 minutes", "bars_per_day": 13},
    "1h": {"minutes": 60, "name": "1 hour", "bars_per_day": 6.5},
    "4h": {"minutes": 240, "name": "4 hours", "bars_per_day": 1.625},
    "1d": {"minutes": 1440, "name": "1 day", "bars_per_day": 1},
    "1w": {"minutes": 10080, "name": "1 week", "bars_per_day": 0.2},
}

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Window and horizon now represent "periods" (not necessarily days)
WINDOW_SIZE = 256  # Number of periods in each input window
NUM_CHANNELS = 5   # OHLCV: Open, High, Low, Close, Volume

# Prediction horizons (periods ahead to predict)
HORIZONS = [5, 30]  # T+5 and T+30 period predictions
DEFAULT_HORIZON = 5

# ============================================================================
# SAMPLE MODE CONFIGURATION (Phase 6)
# ============================================================================

# Default sample generation mode
SAMPLE_MODE = SampleMode.OVERLAPPING

# Stride for STRIDED mode (ignored for other modes)
SAMPLE_STRIDE = 10

# Minimum recommended samples for training
MIN_TRAIN_SAMPLES = 100
MIN_VAL_SAMPLES = 20
MIN_TEST_SAMPLES = 20

# ============================================================================
# DATA SPLIT RATIOS
# ============================================================================

VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model architecture
# Following the paper's architecture with 8 Conv layers + 2 FC layers
CONV_CHANNELS = [128, 256, 256, 512, 1024, 1024, 1024, 1024]
KERNEL_SIZE = 9
FC_HIDDEN = 256
NUM_CLASSES = 2  # Binary: bearish (0) or bullish (1)

# Training hyperparameters
BATCH_SIZE = 128  # Reduced for CPU training (paper used 250)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.4

# Training schedule
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for N epochs
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
LOG_INTERVAL = 10  # Log every N batches


def get_config_dict(
    window_size: int = WINDOW_SIZE,
    horizons: List[int] = None,
    timeframe: str = None,
    sample_mode: str = None,
    sample_stride: int = SAMPLE_STRIDE,
) -> dict:
    """
    Return all configuration as a dictionary.

    Args:
        window_size: Override default window size
        horizons: Override default horizons
        timeframe: Override default timeframe
        sample_mode: Override default sample mode
        sample_stride: Override default stride for strided mode

    Returns:
        Dictionary with all configuration values
    """
    if horizons is None:
        horizons = HORIZONS
    if timeframe is None:
        timeframe = TIMEFRAME.value if isinstance(TIMEFRAME, Timeframe) else TIMEFRAME
    if sample_mode is None:
        sample_mode = SAMPLE_MODE.value if isinstance(SAMPLE_MODE, SampleMode) else SAMPLE_MODE

    return {
        # Data parameters
        "window_size": window_size,
        "num_channels": NUM_CHANNELS,
        "horizons": horizons,
        "timeframe": timeframe,
        # Sample mode
        "sample_mode": sample_mode,
        "sample_stride": sample_stride,
        "min_train_samples": MIN_TRAIN_SAMPLES,
        # Split ratios
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        # Model architecture
        "conv_channels": CONV_CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "fc_hidden": FC_HIDDEN,
        "num_classes": NUM_CLASSES,
        # Training hyperparameters
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "num_epochs": NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "device": str(DEVICE),
    }


def print_config(**kwargs):
    """Print current configuration."""
    config = get_config_dict(**kwargs)
    print("=" * 60)
    print("CNN Stock Market Prediction Configuration")
    print("=" * 60)

    # Group by category
    categories = {
        "Data": ["window_size", "num_channels", "horizons", "timeframe"],
        "Sample Mode": ["sample_mode", "sample_stride", "min_train_samples"],
        "Split Ratios": ["val_ratio", "test_ratio"],
        "Model": ["conv_channels", "kernel_size", "fc_hidden", "num_classes"],
        "Training": ["batch_size", "learning_rate", "weight_decay", "dropout",
                     "num_epochs", "early_stopping_patience", "device"],
    }

    for category, keys in categories.items():
        print(f"\n[{category}]")
        for key in keys:
            if key in config:
                print(f"  {key}: {config[key]}")

    print("=" * 60)


def calculate_stride(
    mode: SampleMode,
    window_size: int = WINDOW_SIZE,
    horizon: int = DEFAULT_HORIZON,
    custom_stride: int = SAMPLE_STRIDE,
) -> int:
    """
    Calculate the stride based on sample mode.

    Args:
        mode: Sample generation mode
        window_size: Size of the sliding window
        horizon: Prediction horizon
        custom_stride: Custom stride for STRIDED mode

    Returns:
        Stride value
    """
    if isinstance(mode, str):
        mode = SampleMode(mode)

    if mode == SampleMode.OVERLAPPING:
        return 1
    elif mode == SampleMode.STRIDED:
        return custom_stride
    elif mode == SampleMode.NON_OVERLAPPING:
        return window_size + horizon
    else:
        raise ValueError(f"Unknown sample mode: {mode}")
