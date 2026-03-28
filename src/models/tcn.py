"""Temporal Convolutional Network (TCN) for stock price movement prediction.

TCN uses dilated causal convolutions to achieve large receptive fields
without the depth of standard CNNs. Dilation doubles each layer:
  layer 0: dilation=1   (receptive field: kernel_size)
  layer 1: dilation=2   (receptive field: 2*kernel_size)
  layer 2: dilation=4
  ...
  layer N: dilation=2^N

A TCN with kernel=5 and 6 layers covers:
  (kernel-1) * sum(2^i for i in 0..5) = 4 * 63 = 252 days — full trading year.

Unlike the CNN, TCN is causal (no lookahead within the window), which is
a stricter form of correctness for time-series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    Single TCN block: two dilated causal convolutions + residual.
    Uses weight normalization for training stability (standard in TCN papers).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation   # causal padding

        self.conv1 = nn.utils.parametrize.register_parametrization if False else \
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 1×1 conv for residual when channels differ
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self._pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1 with causal truncation
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]        # remove right padding (causal)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.1)
        out = self.dropout1(out)

        # Conv2
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.bn2(out)
        out = F.leaky_relu(out, 0.1)
        out = self.dropout2(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return F.leaky_relu(out + res, 0.1)


class StockTCN(nn.Module):
    """
    Temporal Convolutional Network for stock movement prediction.

    Architecture:
        Input: [batch, window_size, input_size]
        N TemporalBlocks with exponentially increasing dilation
        Global average pooling
        FC → Softmax
    """

    def __init__(
        self,
        input_size: int = 5,
        channels: list = None,
        kernel_size: int = 5,
        num_classes: int = 2,
        dropout: float = 0.2,
        window_size: int = 128,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 128, 256]

        # Store for checkpoint compatibility (trainer.save_checkpoint accesses these)
        self.window_size = window_size
        self.num_channels = input_size

        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, window, channels] → [batch, channels, window]
        x = x.transpose(1, 2)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return F.softmax(self.fc(x), dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
