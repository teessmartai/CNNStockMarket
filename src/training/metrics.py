"""Metrics tracking and computation for model training."""

from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
import torch


class MetricsTracker:
    """
    Track and store training metrics across epochs.

    Stores history for loss, accuracy, and other metrics for both
    training and validation phases.
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.current_epoch_metrics: Dict[str, List[float]] = defaultdict(list)
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    def reset_epoch(self):
        """Reset metrics for the current epoch."""
        self.current_epoch_metrics = defaultdict(list)

    def update(self, metric_name: str, value: float, batch_size: int = 1):
        """
        Update a metric with a new value.

        Args:
            metric_name: Name of the metric (e.g., 'train_loss', 'val_accuracy')
            value: Metric value
            batch_size: Batch size for weighted averaging
        """
        # Store weighted value for proper averaging
        self.current_epoch_metrics[metric_name].append((value, batch_size))

    def end_epoch(self, epoch: int) -> Dict[str, float]:
        """
        End the current epoch and compute average metrics.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of averaged metrics for this epoch
        """
        epoch_metrics = {}

        for name, values in self.current_epoch_metrics.items():
            # Compute weighted average
            total_value = sum(v * n for v, n in values)
            total_count = sum(n for _, n in values)
            avg_value = total_value / total_count if total_count > 0 else 0.0

            epoch_metrics[name] = avg_value
            self.history[name].append(avg_value)

        # Track best validation metrics
        val_loss = epoch_metrics.get("val_loss", float("inf"))
        val_acc = epoch_metrics.get("val_accuracy", 0.0)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_accuracy = val_acc
            self.best_epoch = epoch

        self.reset_epoch()
        return epoch_metrics

    def get_history(self, metric_name: str) -> List[float]:
        """Get the history of a specific metric."""
        return self.history.get(metric_name, [])

    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best validation metrics achieved."""
        return {
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
        }

    def get_last_metrics(self) -> Dict[str, float]:
        """Get the most recent metrics."""
        return {name: values[-1] for name, values in self.history.items() if values}

    def to_dict(self) -> Dict:
        """Convert metrics history to a dictionary for saving."""
        return {
            "history": dict(self.history),
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MetricsTracker":
        """Create a MetricsTracker from a saved dictionary."""
        tracker = cls()
        tracker.history = defaultdict(list, data.get("history", {}))
        tracker.best_val_loss = data.get("best_val_loss", float("inf"))
        tracker.best_val_accuracy = data.get("best_val_accuracy", 0.0)
        tracker.best_epoch = data.get("best_epoch", 0)
        return tracker


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        outputs: Model outputs (probabilities), shape [batch, num_classes]
        targets: Ground truth labels, shape [batch]

    Returns:
        Accuracy as a float between 0 and 1
    """
    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def compute_precision_recall_f1(
    outputs: torch.Tensor, targets: torch.Tensor, class_idx: int = 1
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for a specific class.

    Args:
        outputs: Model outputs (probabilities), shape [batch, num_classes]
        targets: Ground truth labels, shape [batch]
        class_idx: Class index to compute metrics for (default: 1 for bullish)

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    predictions = outputs.argmax(dim=1)

    # True positives, false positives, false negatives
    tp = ((predictions == class_idx) & (targets == class_idx)).sum().item()
    fp = ((predictions == class_idx) & (targets != class_idx)).sum().item()
    fn = ((predictions != class_idx) & (targets == class_idx)).sum().item()

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_confusion_matrix(
    outputs: torch.Tensor, targets: torch.Tensor, num_classes: int = 2
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        outputs: Model outputs (probabilities), shape [batch, num_classes]
        targets: Ground truth labels, shape [batch]
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape [num_classes, num_classes]
    """
    predictions = outputs.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(predictions, targets):
        matrix[target, pred] += 1

    return matrix
