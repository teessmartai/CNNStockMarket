"""Visualization utilities for training metrics and predictions."""

from typing import List, Optional, Dict, Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.training.metrics import MetricsTracker


# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    title: str = "Training and Validation Loss",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_loss: List of training loss values per epoch
        val_loss: List of validation loss values per epoch
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Mark best validation loss
    best_epoch = np.argmin(val_loss) + 1
    best_val = min(val_loss)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
    ax.scatter([best_epoch], [best_val], color='g', s=100, zorder=5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_accuracy_curves(
    train_acc: List[float],
    val_acc: List[float],
    title: str = "Training and Validation Accuracy",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot training and validation accuracy curves.

    Args:
        train_acc: List of training accuracy values per epoch
        val_acc: List of validation accuracy values per epoch
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_acc) + 1)

    ax.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Mark best validation accuracy
    best_epoch = np.argmax(val_acc) + 1
    best_val = max(val_acc)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax.scatter([best_epoch], [best_val], color='g', s=100, zorder=5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_summary(
    metrics: MetricsTracker,
    title: str = "Training Summary",
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot combined training summary with loss and accuracy curves.

    Args:
        metrics: MetricsTracker instance with training history
        title: Overall title for the plot
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    train_loss = metrics.get_history('train_loss')
    val_loss = metrics.get_history('val_loss')
    train_acc = metrics.get_history('train_accuracy')
    val_acc = metrics.get_history('val_accuracy')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(train_loss) + 1)

    # Loss subplot
    ax1.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Mark best validation loss
    if val_loss:
        best_epoch = np.argmin(val_loss) + 1
        best_val = min(val_loss)
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
        ax1.scatter([best_epoch], [best_val], color='g', s=80, zorder=5)

    # Accuracy subplot
    ax2.plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Mark best validation accuracy
    if val_acc:
        best_epoch = np.argmax(val_acc) + 1
        best_val = max(val_acc)
        ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
        ax2.scatter([best_epoch], [best_val], color='g', s=80, zorder=5)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = ["Bearish", "Bullish"],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.

    Args:
        confusion_matrix: 2D numpy array with confusion matrix values
        class_names: Names for each class
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize confusion matrix for display
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={'label': 'Proportion'},
    )

    # Add raw counts as annotations below percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = confusion_matrix[i, j]
            ax.text(j + 0.5, i + 0.7, f'(n={count})', ha='center', va='center', fontsize=9, color='gray')

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_distribution(
    predictions: np.ndarray,
    labels: np.ndarray,
    title: str = "Prediction Confidence Distribution",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot distribution of prediction confidences by true label.

    Args:
        predictions: Model output probabilities, shape [N, 2]
        labels: True labels, shape [N]
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get bullish (class 1) probabilities
    bullish_probs = predictions[:, 1] if predictions.ndim > 1 else predictions

    # Separate by true label
    bearish_mask = labels == 0
    bullish_mask = labels == 1

    ax.hist(bullish_probs[bearish_mask], bins=30, alpha=0.6, label='True Bearish', color='red', density=True)
    ax.hist(bullish_probs[bullish_mask], bins=30, alpha=0.6, label='True Bullish', color='green', density=True)

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary')

    ax.set_xlabel('Bullish Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_stock_price(
    prices: np.ndarray,
    dates: Optional[List] = None,
    prediction: Optional[str] = None,
    confidence: Optional[float] = None,
    title: str = "Stock Price History",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot stock price with optional prediction annotation.

    Args:
        prices: Array of closing prices
        dates: Optional list of dates for x-axis
        prediction: Optional prediction label ('BUY' or 'SELL')
        confidence: Optional confidence value (0-1)
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = dates if dates is not None else range(len(prices))
    ax.plot(x, prices, 'b-', linewidth=1.5)

    ax.set_xlabel('Date' if dates is not None else 'Day', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add prediction annotation
    if prediction:
        color = 'green' if prediction == 'BUY' else 'red'
        conf_str = f" ({confidence:.1%})" if confidence else ""
        ax.annotate(
            f'{prediction}{conf_str}',
            xy=(0.98, 0.95),
            xycoords='axes fraction',
            fontsize=14,
            fontweight='bold',
            color=color,
            ha='right',
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

    # Format x-axis for dates
    if dates is not None:
        fig.autofmt_xdate()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_learning_rate_schedule(
    learning_rates: List[float],
    title: str = "Learning Rate Schedule",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot learning rate over training epochs.

    Args:
        learning_rates: List of learning rates per epoch
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(learning_rates) + 1)
    ax.plot(epochs, learning_rates, 'b-', linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
