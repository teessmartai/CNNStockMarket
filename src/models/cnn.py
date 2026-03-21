"""1D Convolutional Neural Network for stock price movement prediction.

Architecture based on the paper: "S&P 500 Stock's Movement Prediction using CNN"
8 Conv1D layers + 2 Fully Connected layers

Flags:
    use_batchnorm  (default True)  — BatchNorm1d after each conv (Conv→BN→Act)
    use_residual   (default False) — residual skip connections between conv layers;
                                     1×1 projection used when channels differ
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import (
    WINDOW_SIZE,
    NUM_CHANNELS,
    CONV_CHANNELS,
    KERNEL_SIZE,
    FC_HIDDEN,
    NUM_CLASSES,
    DROPOUT,
)


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv1D → [BN] → Activation → [residual add].

    Standard order: Conv → BN → Activation (fixes original Conv → Act → BN).
    Residual: when use_residual=True, adds a skip connection from input to
    output. A 1×1 conv projection handles channel mismatches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        use_relu: bool = False,
        use_batchnorm: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels) if use_batchnorm else None
        self.activation = nn.ReLU() if use_relu else nn.LeakyReLU(0.1)

        # 1×1 projection for residual when channels differ
        if use_residual and in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.activation(out)

        if self.use_residual:
            if self.projection is not None:
                residual = self.projection(residual)
            out = out + residual

        return out


class StockCNN(nn.Module):
    """
    1D CNN for stock price movement prediction.

    Architecture:
        Input: [batch, window_size, 5] — OHLCV channels
        8 Conv1D layers with increasing channels
        Global average pooling
        2 Fully connected layers
        Output: [batch, 2] — softmax probabilities (bearish/bullish)
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        num_channels: int = NUM_CHANNELS,
        conv_channels: List[int] = None,
        kernel_size: int = KERNEL_SIZE,
        fc_hidden: int = FC_HIDDEN,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        use_batchnorm: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = CONV_CHANNELS

        self.window_size = window_size
        self.num_channels = num_channels
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()

        # First layer: ReLU (per paper), input channels → first conv channels
        self.conv_layers.append(ConvBlock(
            num_channels, conv_channels[0], kernel_size,
            use_relu=True,
            use_batchnorm=use_batchnorm,
            use_residual=use_residual,
        ))

        # Remaining layers: LeakyReLU
        for i in range(1, len(conv_channels)):
            self.conv_layers.append(ConvBlock(
                conv_channels[i - 1], conv_channels[i], kernel_size,
                use_relu=False,
                use_batchnorm=use_batchnorm,
                use_residual=use_residual,
            ))

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_channels[-1], fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, window, channels] → [batch, channels, window] for Conv1d
        x = x.transpose(1, 2)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Global average pooling → [batch, channels]
        x = self.global_pool(x).squeeze(-1)

        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.fc2(x)

        return F.softmax(x, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "StockCNN Model Summary",
            "=" * 60,
            f"Input shape:   [batch, {self.window_size}, {self.num_channels}]",
            f"BatchNorm:     {self.use_batchnorm}",
            f"Residual:      {self.use_residual}",
            "",
            "Convolutional Layers:",
        ]
        in_ch = self.num_channels
        for i, block in enumerate(self.conv_layers):
            out_ch = block.conv.out_channels
            act = "ReLU" if i == 0 else "LeakyReLU"
            res = " +residual" if self.use_residual else ""
            bn = " +BN" if self.use_batchnorm else ""
            lines.append(f"  Conv{i+1}: {in_ch:>4} → {out_ch:<4} (k={block.conv.kernel_size[0]}, {act}{bn}{res})")
            in_ch = out_ch
        lines += [
            "",
            "Pooling: Global Average Pooling",
            "",
            "Fully Connected Layers:",
            f"  FC1: {CONV_CHANNELS[-1]} → {FC_HIDDEN} (LeakyReLU, Dropout={DROPOUT})",
            f"  FC2: {FC_HIDDEN} → {NUM_CLASSES} (Softmax)",
            "",
            f"Total parameters: {self.count_parameters():,}",
            "=" * 60,
        ]
        return "\n".join(lines)


def create_model(device: Optional[torch.device] = None, **kwargs) -> "StockCNN":
    from src.utils.config import DEVICE
    if device is None:
        device = DEVICE
    model = StockCNN(**kwargs)
    return model.to(device)


def validate_model() -> bool:
    print("Validating StockCNN model (default)...")
    model = StockCNN()
    print(model.summary())

    x = torch.randn(32, WINDOW_SIZE, NUM_CHANNELS)
    model.eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape == (32, NUM_CLASSES)
    assert torch.allclose(out.sum(dim=1), torch.ones(32), atol=1e-5)
    assert (out >= 0).all() and (out <= 1).all()

    print("\nValidation passed!")

    print("\nValidating with residual + no batchnorm...")
    model2 = StockCNN(use_residual=True, use_batchnorm=False)
    print(model2.summary())
    with torch.no_grad():
        out2 = model2(x)
    assert out2.shape == (32, NUM_CLASSES)
    print("Residual validation passed!")

    return True


if __name__ == "__main__":
    validate_model()
