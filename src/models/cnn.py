"""1D Convolutional Neural Network for stock price movement prediction.

Architecture based on the paper: "S&P 500 Stock's Movement Prediction using CNN"
8 Conv1D layers + 2 Fully Connected layers
"""

from typing import Optional

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
    """Convolutional block with Conv1D, activation, and BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        use_relu: bool = False,
    ):
        """
        Initialize convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            use_relu: If True, use ReLU; otherwise use LeakyReLU
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Same padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU() if use_relu else nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        return x


class StockCNN(nn.Module):
    """
    1D CNN for stock price movement prediction.

    Architecture:
        Input: [batch, window_size, 5] - 256-day windows with 5 OHLCV channels
        8 Conv1D layers with increasing channels
        Global average pooling
        2 Fully connected layers
        Output: [batch, 2] - Softmax probabilities (bearish/bullish)
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        num_channels: int = NUM_CHANNELS,
        conv_channels: list = None,
        kernel_size: int = KERNEL_SIZE,
        fc_hidden: int = FC_HIDDEN,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        """
        Initialize the CNN model.

        Args:
            window_size: Number of time steps in input (default: 256)
            num_channels: Number of input features (default: 5 for OHLCV)
            conv_channels: List of channel sizes for conv layers
            kernel_size: Size of convolutional kernels
            fc_hidden: Size of hidden FC layer
            num_classes: Number of output classes (default: 2)
            dropout: Dropout rate for FC layers
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = CONV_CHANNELS

        self.window_size = window_size
        self.num_channels = num_channels

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()

        # First layer: use ReLU (as per paper)
        self.conv_layers.append(
            ConvBlock(num_channels, conv_channels[0], kernel_size, use_relu=True)
        )

        # Remaining layers: use LeakyReLU
        for i in range(1, len(conv_channels)):
            self.conv_layers.append(
                ConvBlock(conv_channels[i - 1], conv_channels[i], kernel_size, use_relu=False)
            )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_channels[-1], fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch, window_size, num_channels]

        Returns:
            Output tensor of shape [batch, num_classes] with softmax probabilities
        """
        # Transpose to [batch, channels, sequence] for Conv1d
        x = x.transpose(1, 2)

        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Global average pooling: [batch, channels, 1]
        x = self.global_pool(x)

        # Flatten: [batch, channels]
        x = x.squeeze(-1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.fc2(x)

        # Softmax for probability output
        x = F.softmax(x, dim=1)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (convenience method).

        Args:
            x: Input tensor

        Returns:
            Predicted class indices
        """
        with torch.no_grad():
            probs = self.forward(x)
            return torch.argmax(probs, dim=1)

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """
        Generate a summary of the model architecture.

        Returns:
            String description of the model
        """
        lines = []
        lines.append("=" * 60)
        lines.append("StockCNN Model Summary")
        lines.append("=" * 60)
        lines.append(f"Input shape: [batch, {self.window_size}, {self.num_channels}]")
        lines.append("")

        # Convolutional layers
        lines.append("Convolutional Layers:")
        for i, conv in enumerate(self.conv_layers):
            in_ch = self.num_channels if i == 0 else CONV_CHANNELS[i - 1]
            out_ch = CONV_CHANNELS[i]
            activation = "ReLU" if i == 0 else "LeakyReLU"
            lines.append(f"  Conv{i+1}: {in_ch} -> {out_ch} (kernel={KERNEL_SIZE}, {activation})")

        lines.append("")
        lines.append("Pooling: Global Average Pooling")
        lines.append("")

        # FC layers
        lines.append("Fully Connected Layers:")
        lines.append(f"  FC1: {CONV_CHANNELS[-1]} -> {FC_HIDDEN} (LeakyReLU, Dropout={DROPOUT})")
        lines.append(f"  FC2: {FC_HIDDEN} -> {NUM_CLASSES} (Softmax)")
        lines.append("")

        lines.append(f"Output shape: [batch, {NUM_CLASSES}]")
        lines.append(f"Total parameters: {self.count_parameters():,}")
        lines.append("=" * 60)

        return "\n".join(lines)


def create_model(device: Optional[torch.device] = None) -> StockCNN:
    """
    Create and initialize the CNN model.

    Args:
        device: Device to move model to (default: from config)

    Returns:
        Initialized StockCNN model
    """
    from src.utils.config import DEVICE

    if device is None:
        device = DEVICE

    model = StockCNN()
    model = model.to(device)
    return model


def validate_model() -> bool:
    """
    Validate that the model architecture is correct.

    Runs a forward pass with dummy data and checks output shapes.

    Returns:
        True if validation passes
    """
    print("Validating StockCNN model...")

    # Create model
    model = StockCNN()
    print(model.summary())

    # Test forward pass with dummy data
    batch_size = 32
    x = torch.randn(batch_size, WINDOW_SIZE, NUM_CHANNELS)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)

    # Validate output shape
    expected_shape = (batch_size, NUM_CLASSES)
    assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape}"

    # Validate output is probability distribution (sums to 1)
    output_sums = output.sum(dim=1)
    assert torch.allclose(output_sums, torch.ones(batch_size), atol=1e-5), "Output probabilities don't sum to 1"

    # Validate output values are in [0, 1]
    assert (output >= 0).all() and (output <= 1).all(), "Output values not in [0, 1]"

    print(f"\nValidation passed!")
    print(f"  Input shape: [{batch_size}, {WINDOW_SIZE}, {NUM_CHANNELS}]")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output sum (first sample): {output[0].sum().item():.4f}")
    print(f"  Sample prediction: {output[0].tolist()}")

    return True


if __name__ == "__main__":
    validate_model()
