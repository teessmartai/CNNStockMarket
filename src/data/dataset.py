"""PyTorch Dataset classes for stock market data."""

from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock market sliding window data.

    Each sample is a window of OHLCV data with a binary label.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the dataset.

        Args:
            X: Feature array of shape [N, window_size, features]
            y: Label array of shape [N,] with binary labels (0 or 1)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, label) tensors
        """
        return self.X[idx], self.y[idx]


class MultiStockDataset(Dataset):
    """
    PyTorch Dataset that combines data from multiple stocks.

    Useful for training on a diverse set of stocks.
    """

    def __init__(self, stock_windows: dict):
        """
        Initialize with data from multiple stocks.

        Args:
            stock_windows: Dict mapping ticker -> (X, y) tuples
        """
        all_X = []
        all_y = []

        for ticker, (X, y) in stock_windows.items():
            all_X.append(X)
            all_y.append(y)

        if not all_X:
            raise ValueError("No data provided")

        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        self.X = torch.tensor(X_combined, dtype=torch.float32)
        self.y = torch.tensor(y_combined, dtype=torch.long)
        self.n_stocks = len(stock_windows)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by index."""
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size (default: 128 for CPU training)
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=False,  # No GPU, so no pinning
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


def create_test_dataloader(
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a test DataLoader.

    Args:
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Test DataLoader
    """
    test_dataset = StockDataset(X_test, y_test)

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
