"""PyTorch Dataset and DataLoader utilities for stock data."""

import logging
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock price sliding windows.

    Wraps preprocessed X (features) and y (labels) arrays for use with
    PyTorch DataLoader.

    Args:
        X: Feature array of shape [N, window_size, features]
        y: Label array of shape [N,]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the dataset with features and labels.

        Args:
            X: Numpy array of shape [N, window_size, features]
            y: Numpy array of shape [N,]
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have same length: {len(self.X)} vs {len(self.y)}"
            )

        logger.info(
            f"Created StockDataset with {len(self)} samples, "
            f"X shape: {self.X.shape}, y shape: {self.y.shape}"
        )

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, label) tensors
        """
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
    Create DataLoaders for training and validation.

    Args:
        X_train: Training features [N_train, window_size, features]
        y_train: Training labels [N_train,]
        X_val: Validation features [N_val, window_size, features]
        y_val: Validation labels [N_val,]
        batch_size: Batch size for DataLoader (default: 128 for CPU)
        num_workers: Number of workers for data loading (default: 0)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    logger.info(
        f"Created DataLoaders: train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches (batch_size={batch_size})"
    )

    return train_loader, val_loader


def create_test_dataloader(
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader for test set.

    Args:
        X_test: Test features [N_test, window_size, features]
        y_test: Test labels [N_test,]
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading

    Returns:
        Test DataLoader
    """
    test_dataset = StockDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    logger.info(f"Created test DataLoader: {len(test_loader)} batches")

    return test_loader


if __name__ == "__main__":
    # Test the dataset and dataloaders
    from src.data.fetcher import fetch_stock_data
    from src.data.preprocessor import prepare_data

    print("Testing StockDataset and DataLoaders...")

    # Fetch and prepare data
    df = fetch_stock_data('AAPL', '2015-01-01', '2024-01-01')
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        df, window_size=256, horizon=5
    )

    # Test dataset
    print("\nTesting StockDataset...")
    train_dataset = StockDataset(X_train, y_train)
    print(f"Dataset length: {len(train_dataset)}")

    X_sample, y_sample = train_dataset[0]
    print(f"Sample X shape: {X_sample.shape}")
    print(f"Sample X dtype: {X_sample.dtype}")
    print(f"Sample y: {y_sample.item()}")
    print(f"Sample y dtype: {y_sample.dtype}")

    # Test dataloaders
    print("\nTesting create_dataloaders...")
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    # Iterate over a batch
    for batch_X, batch_y in train_loader:
        print(f"\nFirst batch:")
        print(f"  X shape: {batch_X.shape}")
        print(f"  y shape: {batch_y.shape}")
        print(f"  y values: {batch_y[:10].tolist()}")
        break

    # Test test dataloader
    print("\nTesting create_test_dataloader...")
    test_loader = create_test_dataloader(X_test, y_test, batch_size=32)
    print(f"Test loader: {len(test_loader)} batches")
