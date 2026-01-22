"""Configuration and hyperparameters for the CNN Stock Market Prediction model."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data parameters
WINDOW_SIZE = 256  # Number of days in each input window
NUM_CHANNELS = 5   # OHLCV: Open, High, Low, Close, Volume

# Prediction horizons (days ahead to predict)
HORIZONS = [5, 30]  # T+5 and T+30 day predictions
DEFAULT_HORIZON = 5

# Data split ratios
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


def get_config_dict() -> dict:
    """Return all configuration as a dictionary."""
    return {
        "window_size": WINDOW_SIZE,
        "num_channels": NUM_CHANNELS,
        "horizons": HORIZONS,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "conv_channels": CONV_CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "fc_hidden": FC_HIDDEN,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "num_epochs": NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "device": str(DEVICE),
    }


def print_config():
    """Print current configuration."""
    config = get_config_dict()
    print("=" * 50)
    print("CNN Stock Market Prediction Configuration")
    print("=" * 50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
