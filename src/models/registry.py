"""
Model registry — factory for all supported architectures.

Usage in train_experiment.py:
    from src.models.registry import build_model
    model = build_model(arch=args.arch, size=args.size, window=args.window, ...)

Adding a new architecture:
    1. Create src/models/your_arch.py implementing BaseModel interface
    2. Register it in ARCH_REGISTRY below

Size presets map to (width_multiplier, num_layers) tuples per architecture.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch.nn as nn

# ── Size presets ───────────────────────────────────────────────────────────────
# (width_multiplier, num_layers)
# Chosen so parameter counts roughly match data size guidelines:
#   tiny   ~35K-150K params  → stride=133 (<15K samples)
#   small  ~1M-3M params     → stride=5 small universe (<50K samples)
#   medium ~8M-10M params    → stride=5 full S&P500 (50K-300K samples)
#   large  ~35M params       → stride=1 / T+1 large data (>300K samples)
#   xlarge ~140M params      → future: T+1 full S&P500 1.4M samples

CNN_SIZES = {
    "tiny":   (0.125, 4),   # ~35K params
    "small":  (0.25,  4),   # ~140K params
    "medium": (0.25,  8),   # ~2.2M params
    "large":  (1.0,   8),   # ~35M params  (current default)
    "xlarge": (2.0,   8),   # ~140M params
}

LSTM_SIZES = {
    "tiny":   dict(hidden=64,  num_layers=1),
    "small":  dict(hidden=128, num_layers=2),
    "medium": dict(hidden=256, num_layers=2),
    "large":  dict(hidden=512, num_layers=3),
    "xlarge": dict(hidden=1024,num_layers=4),
}

TRANSFORMER_SIZES = {
    "tiny":   dict(d_model=32,  nhead=2, num_layers=2, dim_ff=64),
    "small":  dict(d_model=64,  nhead=4, num_layers=2, dim_ff=128),
    "medium": dict(d_model=128, nhead=4, num_layers=4, dim_ff=256),
    "large":  dict(d_model=256, nhead=8, num_layers=6, dim_ff=512),
    "xlarge": dict(d_model=512, nhead=8, num_layers=8, dim_ff=2048),
}

TCN_SIZES = {
    "tiny":   dict(channels=[32,  32],           kernel_size=3),
    "small":  dict(channels=[64,  64,  64],       kernel_size=5),
    "medium": dict(channels=[128, 128, 128, 128], kernel_size=5),
    "large":  dict(channels=[256, 256, 256, 256, 256, 256], kernel_size=7),
    "xlarge": dict(channels=[512]*8,              kernel_size=7),
}


def build_model(
    arch: str,
    size: str,
    window_size: int,
    num_channels: int,
    num_classes: int,
    dropout: float,
    use_batchnorm: bool = True,
    use_residual: bool = False,
) -> nn.Module:
    """
    Factory: instantiate the requested architecture at the requested size.

    Args:
        arch:         "cnn" | "lstm" | "transformer" | "tcn"
        size:         "tiny" | "small" | "medium" | "large" | "xlarge"
        window_size:  input sequence length
        num_channels: input feature channels (5 for OHLCV)
        num_classes:  output classes (2 for buy/sell)
        dropout:      dropout rate
        use_batchnorm: CNN only — BatchNorm after each conv
        use_residual:  CNN only — residual skip connections

    Returns:
        nn.Module with .count_parameters() method
    """
    arch = arch.lower()

    if arch == "cnn":
        from src.models.cnn import StockCNN
        from src.utils.config import CONV_CHANNELS, KERNEL_SIZE
        width_mult, num_layers = CNN_SIZES[size]
        base = CONV_CHANNELS[:max(1, min(num_layers, len(CONV_CHANNELS)))]
        conv_channels = [max(1, int(c * width_mult)) for c in base]
        fc_hidden = max(1, int(256 * width_mult))
        return StockCNN(
            window_size=window_size, num_channels=num_channels,
            conv_channels=conv_channels, kernel_size=KERNEL_SIZE,
            fc_hidden=fc_hidden, num_classes=num_classes,
            dropout=dropout, use_batchnorm=use_batchnorm,
            use_residual=use_residual,
        )

    elif arch == "lstm":
        from src.models.lstm import StockLSTM
        cfg = LSTM_SIZES[size]
        return StockLSTM(
            input_size=num_channels, num_classes=num_classes,
            dropout=dropout, **cfg,
        )

    elif arch == "transformer":
        from src.models.transformer import StockTransformer
        cfg = TRANSFORMER_SIZES[size]
        return StockTransformer(
            window_size=window_size, input_size=num_channels,
            num_classes=num_classes, dropout=dropout, **cfg,
        )

    elif arch == "tcn":
        from src.models.tcn import StockTCN
        cfg = TCN_SIZES[size]
        return StockTCN(
            input_size=num_channels, num_classes=num_classes,
            dropout=dropout, **cfg,
        )

    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose: cnn, lstm, transformer, tcn")
