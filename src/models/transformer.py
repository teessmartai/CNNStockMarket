"""Transformer (encoder-only) model for stock price movement prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StockTransformer(nn.Module):
    """
    Encoder-only Transformer with learnable positional encoding.

    Architecture:
        Input projection: input_size → d_model
        Positional encoding (learnable)
        N × TransformerEncoderLayer (self-attention + FFN)
        Global average pooling over sequence
        FC head → Softmax

    Uses causal masking is NOT applied — we have the full window at inference
    time, so bidirectional attention over the historical window is appropriate.
    """

    def __init__(
        self,
        window_size: int = 128,
        input_size: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 256,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Store for checkpoint compatibility (trainer.save_checkpoint accesses these)
        self.window_size = window_size
        self.num_channels = input_size

        # Project OHLCV channels → d_model
        self.input_proj = nn.Linear(input_size, d_model)

        # Learnable positional encoding (more flexible than sinusoidal)
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, input_size]
        x = self.input_proj(x)          # [batch, window, d_model]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)             # [batch, window, d_model]
        x = self.norm(x)
        x = x.mean(dim=1)               # global average pool over sequence
        x = self.dropout(x)
        return F.softmax(self.fc(x), dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
