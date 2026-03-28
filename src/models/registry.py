"""
Model registry — factory for all supported architectures.

Model size is driven by --samples-per-param (default 100):
    target_params = train_samples / samples_per_param

A binary search finds the width multiplier for each architecture that
produces a model closest to target_params. This means architectures are
always fairly compared at the same samples-per-parameter ratio, regardless
of dataset size.

Usage in train_experiment.py:
    from src.models.registry import build_model_for_samples
    model = build_model_for_samples(
        arch=args.arch, train_samples=n_train,
        samples_per_param=args.samples_per_param, ...
    )

Adding a new architecture:
    1. Create src/models/your_arch.py
    2. Implement _build_<arch>(width, window_size, num_channels, num_classes, dropout, **kw)
    3. Register in _ARCH_BUILDERS below
"""

from __future__ import annotations
import math
import torch.nn as nn


# ── Per-architecture builders (width is a single float scaling the model) ─────
# Width=1.0 is the reference size for each arch. Binary search scales it.

def _build_cnn(width: float, window_size, num_channels, num_classes, dropout,
               use_batchnorm=True, use_residual=False, **_):
    from src.models.cnn import StockCNN
    from src.utils.config import CONV_CHANNELS, KERNEL_SIZE
    # Base shape: [128,256,256,512,1024,1024,1024,1024] at width=1.0
    channels = [max(1, round(c * width)) for c in CONV_CHANNELS]
    fc = max(1, round(256 * width))
    return StockCNN(window_size=window_size, num_channels=num_channels,
                    conv_channels=channels, kernel_size=KERNEL_SIZE,
                    fc_hidden=fc, num_classes=num_classes, dropout=dropout,
                    use_batchnorm=use_batchnorm, use_residual=use_residual)


def _build_lstm(width: float, window_size, num_channels, num_classes, dropout, **_):
    from src.models.lstm import StockLSTM
    # Base: hidden=512, num_layers=3 at width=1.0
    hidden = max(8, round(512 * width))
    nlayers = max(1, round(3 * math.log2(max(width, 0.01) + 1) + 1))
    nlayers = min(nlayers, 4)
    return StockLSTM(input_size=num_channels, hidden=hidden,
                     num_layers=nlayers, num_classes=num_classes, dropout=dropout,
                     window_size=window_size)


def _build_transformer(width: float, window_size, num_channels, num_classes, dropout, **_):
    from src.models.transformer import StockTransformer
    # Base: d_model=256, nhead=8, num_layers=6, dim_ff=512 at width=1.0
    d_model = max(8, round(256 * width))
    # nhead must divide d_model; find largest power-of-2 ≤ d_model that divides it
    nhead = 1
    for n in [1, 2, 4, 8, 16]:
        if d_model % n == 0:
            nhead = n
    dim_ff  = max(16, round(512 * width))
    nlayers = max(1, min(6, round(6 * math.sqrt(width))))
    return StockTransformer(window_size=window_size, input_size=num_channels,
                            d_model=d_model, nhead=nhead, num_layers=nlayers,
                            dim_ff=dim_ff, num_classes=num_classes, dropout=dropout)


def _build_tcn(width: float, window_size, num_channels, num_classes, dropout, **_):
    from src.models.tcn import StockTCN
    # Base: [256]*6, kernel=7 at width=1.0
    n_layers = max(2, min(8, round(6 * math.sqrt(width))))
    ch = max(8, round(256 * width))
    channels = [ch] * n_layers
    return StockTCN(input_size=num_channels, channels=channels,
                    kernel_size=7, num_classes=num_classes, dropout=dropout,
                    window_size=window_size)


_ARCH_BUILDERS = {
    "cnn":         _build_cnn,
    "lstm":        _build_lstm,
    "transformer": _build_transformer,
    "tcn":         _build_tcn,
}


# ── Fast analytical param counters (no model instantiation) ───────────────────

def _params_cnn(width: float, num_channels: int, **_) -> int:
    from src.utils.config import CONV_CHANNELS, KERNEL_SIZE
    k = KERNEL_SIZE
    channels = [max(1, round(c * width)) for c in CONV_CHANNELS]
    n = 0
    in_c = num_channels
    for out_c in channels:
        n += in_c * k * out_c + out_c   # conv weights + bias
        n += 2 * out_c                   # BN: weight + bias
        in_c = out_c
    fc = max(1, round(256 * width))
    n += channels[-1] * fc + fc         # FC1
    n += fc * 2 + 2                     # FC2
    return n


def _params_lstm(width: float, num_channels: int, num_classes: int, **_) -> int:
    hidden = max(8, round(512 * width))
    nlayers = max(1, min(4, round(3 * math.log2(max(width, 0.01) + 1) + 1)))
    # BiLSTM params per layer: 4 * (input + hidden + 1) * hidden * 2 directions
    n = 0
    input_size = num_channels
    for layer in range(nlayers):
        n += 4 * (input_size + hidden + 1) * hidden * 2  # fwd + bwd
        input_size = hidden * 2
    n += hidden * 2 * hidden + hidden   # FC1
    n += hidden * num_classes + num_classes  # FC2
    return n


def _params_transformer(width: float, window_size: int, num_channels: int,
                         num_classes: int, **_) -> int:
    d = max(8, round(256 * width))
    nhead = max(1, max(n for n in [1,2,4,8,16] if d % n == 0))
    dim_ff = max(16, round(512 * width))
    nlayers = max(1, min(6, round(6 * math.sqrt(width))))
    n = num_channels * d + d            # input_proj + bias
    n += window_size * d                # pos_embed
    n += d                              # LayerNorm
    for _ in range(nlayers):
        # Self-attn: Q,K,V projections + output
        n += 4 * d * d + 4 * d
        # FFN: linear1 + linear2
        n += d * dim_ff + dim_ff + dim_ff * d + d
        # Two LayerNorms
        n += 4 * d
    n += d * num_classes + num_classes  # final FC
    return n


def _params_tcn(width: float, num_channels: int, num_classes: int, **_) -> int:
    n_layers = max(2, min(8, round(6 * math.sqrt(width))))
    ch = max(8, round(256 * width))
    kernel = 7
    n = 0
    in_c = num_channels
    for _ in range(n_layers):
        # Two conv layers per block + 2 BNs
        n += 2 * (in_c * kernel * ch + ch) + 2 * 2 * ch
        if in_c != ch:
            n += in_c * ch + ch         # downsample 1x1
        in_c = ch
    n += ch * num_classes + num_classes  # FC
    return n


_PARAM_COUNTERS = {
    "cnn":         _params_cnn,
    "lstm":        _params_lstm,
    "transformer": _params_transformer,
    "tcn":         _params_tcn,
}


# ── Binary search on analytical param count ────────────────────────────────────

def _find_width(arch: str, target_params: int, window_size: int,
                num_channels: int, num_classes: int = 2, **kwargs) -> float:
    """Find width multiplier so param count ≈ target_params (pure math, fast)."""
    counter = _PARAM_COUNTERS[arch]

    lo, hi = 0.001, 20.0
    for _ in range(60):   # bisection: 60 iters → ~10^-18 precision, <1ms
        mid = (lo + hi) / 2.0
        if counter(mid, num_channels=num_channels, num_classes=num_classes,
                   window_size=window_size, **kwargs) < target_params:
            lo = mid
        else:
            hi = mid

    p_lo = counter(lo, num_channels=num_channels, num_classes=num_classes,
                   window_size=window_size)
    p_hi = counter(hi, num_channels=num_channels, num_classes=num_classes,
                   window_size=window_size)
    return lo if abs(p_lo - target_params) <= abs(p_hi - target_params) else hi


# ── Public API ─────────────────────────────────────────────────────────────────

def build_model_for_samples(
    arch: str,
    train_samples: int,
    samples_per_param: int,
    window_size: int,
    num_channels: int,
    num_classes: int,
    dropout: float,
    **kwargs,
) -> nn.Module:
    """
    Build a model sized so that train_samples / model.count_parameters()
    ≈ samples_per_param.

    Args:
        arch:              "cnn" | "lstm" | "transformer" | "tcn"
        train_samples:     number of training windows (after 70/15/15 split)
        samples_per_param: target ratio (default 100; higher = smaller model)
        window_size:       input sequence length
        num_channels:      input feature channels (5 for OHLCV)
        num_classes:       2 for buy/sell
        dropout:           dropout rate
        **kwargs:          arch-specific options (use_batchnorm, use_residual, …)

    Returns:
        nn.Module with .count_parameters() method
    """
    arch = arch.lower()
    if arch not in _ARCH_BUILDERS:
        raise ValueError(f"Unknown arch '{arch}'. Choose: {list(_ARCH_BUILDERS)}")

    target_params = max(100, train_samples // samples_per_param)
    width = _find_width(arch, target_params, window_size, num_channels,
                        num_classes, **kwargs)
    return _ARCH_BUILDERS[arch](width, window_size, num_channels,
                                 num_classes, dropout, **kwargs)

