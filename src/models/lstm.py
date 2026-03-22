"""LSTM model for stock price movement prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StockLSTM(nn.Module):
    """
    Bidirectional LSTM with a 2-layer FC head.

    Architecture:
        Input: [batch, window_size, input_size]
        BiLSTM (num_layers) → last hidden state from both directions
        FC1 → Dropout → FC2 → Softmax
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden * 2, hidden)   # *2 for bidirectional
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, channels]
        out, _ = self.lstm(x)          # [batch, window, hidden*2]
        out = out[:, -1, :]            # last timestep
        out = self.dropout(F.leaky_relu(self.fc1(out), 0.1))
        return F.softmax(self.fc2(out), dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
