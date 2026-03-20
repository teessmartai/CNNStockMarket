"""Training loop and model training utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cnn import StockCNN
from src.training.metrics import MetricsTracker, compute_accuracy
from src.utils.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    CHECKPOINT_INTERVAL,
    LOG_INTERVAL,
    MODELS_DIR,
    WINDOW_SIZE,
    NUM_CHANNELS,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training manager for the StockCNN model.

    Handles training loop, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: StockCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        device: torch.device = DEVICE,
        checkpoint_dir: Optional[Path] = None,
        **kwargs,
    ):
        """
        Initialize the trainer.

        Args:
            model: The CNN model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir or MODELS_DIR

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        optimizer_type = kwargs.get("optimizer_type", "adam")
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        # Learning rate scheduler
        scheduler_type = kwargs.get("scheduler", "plateau")
        num_epochs = kwargs.get("num_epochs", 100)
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=1e-6
            )
        elif scheduler_type == "none":
            self.scheduler = None
        else:  # "plateau" (default)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )

        # Metrics tracker
        self.metrics = MetricsTracker()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics for this epoch
        """
        self.model.train()
        self.metrics.reset_epoch()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute metrics
            accuracy = compute_accuracy(output, target)
            batch_size = data.size(0)

            self.metrics.update("train_loss", loss.item(), batch_size)
            self.metrics.update("train_accuracy", accuracy, batch_size)

            # Update progress bar
            if batch_idx % LOG_INTERVAL == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy:.2%}"})

        return {"train_loss": self.metrics.get_history("train_loss")[-1] if self.metrics.history["train_loss"] else 0}

    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                batch_size = data.size(0)
                val_loss += loss.item() * batch_size
                val_correct += (output.argmax(dim=1) == target).sum().item()
                val_total += batch_size

        # Compute averages
        avg_loss = val_loss / val_total if val_total > 0 else 0
        avg_accuracy = val_correct / val_total if val_total > 0 else 0

        self.metrics.update("val_loss", avg_loss, val_total)
        self.metrics.update("val_accuracy", avg_accuracy, val_total)

        return {"val_loss": avg_loss, "val_accuracy": avg_accuracy}

    def train(
        self,
        num_epochs: int = NUM_EPOCHS,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        start_epoch: int = 0,
    ) -> MetricsTracker:
        """
        Run the full training loop.

        Args:
            num_epochs: Maximum number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
            checkpoint_interval: Save checkpoint every N epochs
            start_epoch: Epoch to resume from (0 = fresh start)

        Returns:
            MetricsTracker with training history
        """
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        if start_epoch > 0:
            logger.info(f"Resuming from epoch {start_epoch + 1}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate()

            # End epoch and get summary
            epoch_metrics = self.metrics.end_epoch(epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Print epoch summary
            print(
                f"\nEpoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {epoch_metrics.get('train_loss', 0):.4f}, "
                f"Train Acc: {epoch_metrics.get('train_accuracy', 0):.2%}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.2%}"
            )

            # Check for improvement
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0

                # Save best model
                self.save_checkpoint("best_model.pt", is_best=True)
                print(f"  ↳ New best model saved!")
            else:
                self.patience_counter += 1
                print(f"  ↳ No improvement for {self.patience_counter} epochs")

            # Periodic checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Final checkpoint
        self.save_checkpoint("final_model.pt")

        # Print best results
        best = self.metrics.get_best_metrics()
        print(f"\nTraining complete!")
        print(f"Best validation loss: {best['best_val_loss']:.4f} at epoch {best['best_epoch'] + 1}")
        print(f"Best validation accuracy: {best['best_val_accuracy']:.2%}")

        return self.metrics

    def save_checkpoint(
        self,
        filename: str,
        is_best: bool = False,
        extra_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Save a training checkpoint.

        Args:
            filename: Name of the checkpoint file
            is_best: Whether this is the best model so far
            extra_config: Additional configuration to save (e.g., data config)
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.checkpoint_dir / filename

        # Model configuration
        model_config = {
            "window_size": self.model.window_size,
            "num_channels": self.model.num_channels,
        }

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_loss": self.best_val_loss,
            "metrics": self.metrics.to_dict(),
            "is_best": is_best,
            "model_config": model_config,
        }

        # Add extra configuration if provided
        if extra_config:
            checkpoint["data_config"] = extra_config

        torch.save(checkpoint, filepath)
        logger.debug(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: Path) -> int:
        """
        Load a training checkpoint.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            Epoch number to resume from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.metrics = MetricsTracker.from_dict(checkpoint["metrics"])
        self.current_epoch = checkpoint["epoch"]

        logger.info(f"Loaded checkpoint from {filepath} (epoch {self.current_epoch + 1})")
        return self.current_epoch + 1


def load_model(
    filepath: Path,
    device: torch.device = DEVICE,
    window_size: Optional[int] = None,
    num_channels: Optional[int] = None,
) -> StockCNN:
    """
    Load a trained model from a checkpoint.

    Args:
        filepath: Path to the checkpoint file
        device: Device to load the model on
        window_size: Override window size (if not in checkpoint)
        num_channels: Override num channels (if not in checkpoint)

    Returns:
        Loaded StockCNN model
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Get model config from checkpoint if available
    model_config = checkpoint.get("model_config", {})

    # Use checkpoint values or provided overrides or defaults
    ws = window_size or model_config.get("window_size", WINDOW_SIZE)
    nc = num_channels or model_config.get("num_channels", NUM_CHANNELS)

    model = StockCNN(window_size=ws, num_channels=nc)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {filepath} (window_size={ws}, num_channels={nc})")
    return model


def get_checkpoint_info(filepath: Path) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.

    Args:
        filepath: Path to the checkpoint file

    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location="cpu")

    info = {
        "epoch": checkpoint["epoch"] + 1,
        "best_val_loss": checkpoint["best_val_loss"],
        "is_best": checkpoint.get("is_best", False),
        "metrics": checkpoint.get("metrics", {}),
    }

    # Include model config if available
    if "model_config" in checkpoint:
        info["model_config"] = checkpoint["model_config"]

    # Include data config if available
    if "data_config" in checkpoint:
        info["data_config"] = checkpoint["data_config"]

    return info


def create_trainer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    window_size: int = WINDOW_SIZE,
    num_channels: int = NUM_CHANNELS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    device: torch.device = DEVICE,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[StockCNN, "Trainer"]:
    """
    Convenience function to create a model and trainer.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        window_size: Size of input windows
        num_channels: Number of input channels
        learning_rate: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        checkpoint_dir: Directory for checkpoints

    Returns:
        Tuple of (model, trainer)
    """
    model = StockCNN(window_size=window_size, num_channels=num_channels)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    return model, trainer
