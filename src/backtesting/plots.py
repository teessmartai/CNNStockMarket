"""
Backtest Visualization (TASK-7.4 & 7.4b)

Comprehensive visualization suite for backtesting results including
equity curves, drawdowns, confusion matrices, and more.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot equity curve over time.

    Args:
        equity_curve: Series of portfolio values with datetime index
        benchmark: Optional benchmark equity curve for comparison
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=1.5, label='Strategy')

    # Plot benchmark if provided
    if benchmark is not None:
        # Normalize benchmark to start at same value
        normalized_bench = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
        ax.plot(normalized_bench.index, normalized_bench.values, 'gray', linewidth=1,
                linestyle='--', label='Benchmark', alpha=0.7)

    # Mark peak
    peak_idx = equity_curve.idxmax()
    peak_val = equity_curve.max()
    ax.scatter([peak_idx], [peak_val], color='green', s=100, zorder=5, label=f'Peak: ${peak_val:,.0f}')

    # Mark trough (max drawdown point)
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    trough_idx = drawdown.idxmin()
    trough_val = equity_curve[trough_idx]
    ax.scatter([trough_idx], [trough_val], color='red', s=100, zorder=5, label=f'Trough: ${trough_val:,.0f}')

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    figsize: Tuple[int, int] = (12, 4),
) -> Figure:
    """
    Plot drawdown chart.

    Args:
        equity_curve: Series of portfolio values
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100  # As percentage

    # Plot
    ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)

    # Mark max drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax.scatter([max_dd_idx], [max_dd_val], color='darkred', s=100, zorder=5)
    ax.annotate(f'Max DD: {max_dd_val:.1f}%',
                xy=(max_dd_idx, max_dd_val),
                xytext=(10, -10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='darkred')

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_ylim(min(drawdown.min() * 1.2, -1), 1)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    predictions: List[int],
    actuals: List[int],
    title: str = "Prediction Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
) -> Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        predictions: List of predicted directions (1=bullish, 0=bearish)
        actuals: List of actual directions
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    from src.backtesting.metrics import calculate_confusion_matrix

    cm = calculate_confusion_matrix(predictions, actuals)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Bearish', 'Predicted Bullish'],
                yticklabels=['Actual Bearish', 'Actual Bullish'],
                ax=ax, cbar=True)

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100 if total > 0 else 0
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    plt.tight_layout()
    return fig


def plot_returns_distribution(
    returns: Union[pd.Series, List[float]],
    title: str = "Returns Distribution",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot returns distribution histogram.

    Args:
        returns: Series or list of returns
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if isinstance(returns, list):
        returns = pd.Series(returns)

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    n, bins, patches = ax.hist(returns * 100, bins=50, density=True, alpha=0.7,
                                color='steelblue', edgecolor='white')

    # Color positive/negative differently
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('salmon')

    # Add normal distribution overlay
    mu = returns.mean() * 100
    sigma = returns.std() * 100
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    ax.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
            'k--', linewidth=2, label='Normal')

    # Add vertical line at mean
    ax.axvline(mu, color='navy', linestyle='-', linewidth=2, label=f'Mean: {mu:.2f}%')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Statistics annotation
    stats_text = (
        f'Mean: {mu:.2f}%\n'
        f'Std: {sigma:.2f}%\n'
        f'Skew: {returns.skew():.2f}\n'
        f'Kurt: {returns.kurtosis():.2f}'
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Density')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_monthly_returns(
    equity_curve: pd.Series,
    title: str = "Monthly Returns Heatmap",
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot monthly returns heatmap.

    Args:
        equity_curve: Series of portfolio values with datetime index
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    from src.backtesting.metrics import calculate_monthly_returns

    monthly_df = calculate_monthly_returns(equity_curve)

    if monthly_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Insufficient data for monthly returns',
                ha='center', va='center', fontsize=12)
        ax.set_title(title)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap (exclude Year Total for color scaling)
    monthly_data = monthly_df.drop(columns=['Year Total'], errors='ignore')

    # Convert to percentage
    monthly_pct = monthly_data * 100

    # Create heatmap
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(monthly_pct, annot=True, fmt='.1f', cmap=cmap,
                center=0, linewidths=0.5, ax=ax, cbar_kws={'label': 'Return (%)'})

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Year')
    ax.set_xlabel('Month')

    plt.tight_layout()
    return fig


def plot_signal_timeline(
    predictions: List[Dict[str, Any]],
    price_data: pd.DataFrame,
    title: str = "Signal Timeline",
    figsize: Tuple[int, int] = (14, 8),
) -> Figure:
    """
    Plot signals overlaid on price chart.

    Args:
        predictions: List of prediction dictionaries
        price_data: DataFrame with price data
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1],
                                    sharex=True)

    # Price chart
    ax1.plot(price_data.index, price_data['Close'], 'b-', linewidth=1, label='Price')

    # Plot signals
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []

    for pred in predictions:
        if pred['signal'] == 'BUY':
            buy_dates.append(pred['date'])
            buy_prices.append(pred['price'])
        else:
            sell_dates.append(pred['date'])
            sell_prices.append(pred['price'])

    ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=50,
                label='BUY Signal', alpha=0.7, zorder=5)
    ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=50,
                label='SELL Signal', alpha=0.7, zorder=5)

    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Confidence chart
    dates = [p['date'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    colors = ['green' if p['signal'] == 'BUY' else 'red' for p in predictions]

    ax2.bar(dates, confidences, color=colors, alpha=0.6, width=1)
    ax2.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 63,
    title: str = "Rolling Performance Metrics",
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    Plot rolling performance metrics.

    Args:
        returns: Series of returns
        window: Rolling window size
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    from src.backtesting.metrics import calculate_rolling_metrics

    rolling = calculate_rolling_metrics(returns, window=window)

    if rolling.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Insufficient data for rolling metrics',
                ha='center', va='center', fontsize=12)
        return fig

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Rolling Return
    ax = axes[0, 0]
    ax.plot(rolling.index, rolling['rolling_return'] * 100, 'b-', linewidth=1)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.fill_between(rolling.index, rolling['rolling_return'] * 100, 0,
                    where=rolling['rolling_return'] > 0, color='green', alpha=0.3)
    ax.fill_between(rolling.index, rolling['rolling_return'] * 100, 0,
                    where=rolling['rolling_return'] < 0, color='red', alpha=0.3)
    ax.set_title(f'Rolling {window}-Day Return', fontsize=11)
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)

    # Rolling Volatility
    ax = axes[0, 1]
    ax.plot(rolling.index, rolling['rolling_volatility'] * 100, 'orange', linewidth=1)
    ax.set_title(f'Rolling {window}-Day Volatility (Annualized)', fontsize=11)
    ax.set_ylabel('Volatility (%)')
    ax.grid(True, alpha=0.3)

    # Rolling Sharpe
    ax = axes[1, 0]
    ax.plot(rolling.index, rolling['rolling_sharpe'], 'purple', linewidth=1)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(1, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(-1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=11)
    ax.set_ylabel('Sharpe')
    ax.grid(True, alpha=0.3)

    # Rolling Win Rate
    ax = axes[1, 1]
    ax.plot(rolling.index, rolling['rolling_win_rate'] * 100, 'teal', linewidth=1)
    ax.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title(f'Rolling {window}-Day Win Rate', fontsize=11)
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Rolling Max Drawdown
    ax = axes[2, 0]
    ax.plot(rolling.index, rolling['rolling_max_drawdown'] * 100, 'red', linewidth=1)
    ax.set_title(f'Rolling {window}-Day Max Drawdown', fontsize=11)
    ax.set_ylabel('Max DD (%)')
    ax.grid(True, alpha=0.3)

    # Empty last subplot - use for summary stats
    ax = axes[2, 1]
    ax.axis('off')
    stats_text = (
        f"Rolling Window: {window} days\n\n"
        f"Latest Return: {rolling['rolling_return'].iloc[-1] * 100:.1f}%\n"
        f"Latest Volatility: {rolling['rolling_volatility'].iloc[-1] * 100:.1f}%\n"
        f"Latest Sharpe: {rolling['rolling_sharpe'].iloc[-1]:.2f}\n"
        f"Latest Win Rate: {rolling['rolling_win_rate'].iloc[-1] * 100:.1f}%\n"
        f"Latest Max DD: {rolling['rolling_max_drawdown'].iloc[-1] * 100:.1f}%"
    )
    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================================
# Randomized Period Backtest Visualizations (TASK-7.4b)
# ============================================================================

def plot_period_returns_histogram(
    model_returns: List[float],
    benchmark_returns: Optional[List[float]] = None,
    title: str = "Period Returns Distribution",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot histogram of returns from sampled periods.

    Args:
        model_returns: List of model returns per period
        benchmark_returns: Optional benchmark returns
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Model returns histogram
    ax.hist(np.array(model_returns) * 100, bins=30, alpha=0.7, color='steelblue',
            edgecolor='white', label='Model')

    # Benchmark returns histogram
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        ax.hist(np.array(benchmark_returns) * 100, bins=30, alpha=0.5, color='gray',
                edgecolor='white', label='Benchmark')

    # Mean lines
    model_mean = np.mean(model_returns) * 100
    ax.axvline(model_mean, color='navy', linestyle='-', linewidth=2,
               label=f'Model Mean: {model_mean:.2f}%')

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bench_mean = np.mean(benchmark_returns) * 100
        ax.axvline(bench_mean, color='darkgray', linestyle='--', linewidth=2,
                   label=f'Benchmark Mean: {bench_mean:.2f}%')

    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_vs_benchmark_scatter(
    model_returns: List[float],
    benchmark_returns: List[float],
    title: str = "Model vs Benchmark Returns",
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    """
    Scatter plot comparing model vs benchmark returns per period.

    Args:
        model_returns: List of model returns
        benchmark_returns: List of benchmark returns
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_pct = np.array(model_returns) * 100
    bench_pct = np.array(benchmark_returns) * 100

    # Scatter plot
    ax.scatter(bench_pct, model_pct, alpha=0.6, s=50)

    # Diagonal line (equal performance)
    lim_min = min(bench_pct.min(), model_pct.min()) - 1
    lim_max = max(bench_pct.max(), model_pct.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, label='Equal')

    # Color quadrants
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)

    # Count points in each quadrant
    q1 = np.sum((model_pct > 0) & (bench_pct > 0))  # Both positive
    q2 = np.sum((model_pct > 0) & (bench_pct <= 0))  # Model positive, bench negative
    q3 = np.sum((model_pct <= 0) & (bench_pct <= 0))  # Both negative
    q4 = np.sum((model_pct <= 0) & (bench_pct > 0))  # Model negative, bench positive

    ax.text(0.95, 0.95, f'Q1: {q1}', transform=ax.transAxes, fontsize=10)
    ax.text(0.05, 0.95, f'Q2: {q2}', transform=ax.transAxes, fontsize=10)
    ax.text(0.05, 0.05, f'Q3: {q3}', transform=ax.transAxes, fontsize=10)
    ax.text(0.95, 0.05, f'Q4: {q4}', transform=ax.transAxes, fontsize=10)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Benchmark Return (%)')
    ax.set_ylabel('Model Return (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_win_rate_ci(
    win_rate: float,
    ci_lower: float,
    ci_upper: float,
    title: str = "Win Rate with Confidence Interval",
    figsize: Tuple[int, int] = (8, 5),
) -> Figure:
    """
    Plot win rate with confidence interval.

    Args:
        win_rate: Point estimate of win rate
        ci_lower: Lower bound of CI
        ci_upper: Upper bound of CI
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Bar chart
    colors = ['green' if win_rate > 0.5 else 'red']
    ax.bar(['Win Rate'], [win_rate * 100], color=colors, alpha=0.7, width=0.4)

    # Error bar for CI
    ax.errorbar(['Win Rate'], [win_rate * 100],
                yerr=[[win_rate * 100 - ci_lower * 100], [ci_upper * 100 - win_rate * 100]],
                fmt='none', color='black', capsize=10, capthick=2, elinewidth=2)

    # Reference line at 50%
    ax.axhline(50, color='black', linestyle='--', linewidth=1, label='Random (50%)')

    # Annotations
    ax.text(0, win_rate * 100 + 3, f'{win_rate:.1%}', ha='center', fontsize=14, fontweight='bold')
    ax.text(0, ci_lower * 100 - 3, f'{ci_lower:.1%}', ha='center', fontsize=10, color='gray')
    ax.text(0, ci_upper * 100 + 3, f'{ci_upper:.1%}', ha='center', fontsize=10, color='gray')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_sampled_periods_timeline(
    periods: List[Dict[str, Any]],
    price_data: pd.DataFrame,
    title: str = "Sampled Test Periods",
    figsize: Tuple[int, int] = (14, 6),
) -> Figure:
    """
    Plot sampled periods on price chart.

    Args:
        periods: List of period dictionaries with start_idx, end_idx
        price_data: DataFrame with price data
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot price
    ax.plot(price_data.index, price_data['Close'], 'b-', linewidth=1, alpha=0.7)

    # Highlight sampled periods
    for i, period in enumerate(periods):
        start_idx = period.get('start_idx', period.get('signal_idx', 0))
        end_idx = period.get('end_idx', period.get('outcome_idx', 0))

        if start_idx < len(price_data) and end_idx < len(price_data):
            start_date = price_data.index[start_idx]
            end_date = price_data.index[end_idx]

            color = 'green' if period.get('is_correct', False) else 'red'
            ax.axvspan(start_date, end_date, alpha=0.2, color=color)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


def plot_period_cumulative_return(
    period_returns: List[float],
    title: str = "Cumulative Return (Sequential Periods)",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot cumulative return as if periods were traded sequentially.

    Args:
        period_returns: List of returns per period
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Cumulative return
    cumulative = np.cumprod(1 + np.array(period_returns))
    periods = range(1, len(cumulative) + 1)

    ax.plot(periods, cumulative, 'b-', linewidth=2)
    ax.fill_between(periods, 1, cumulative, where=cumulative > 1, color='green', alpha=0.3)
    ax.fill_between(periods, 1, cumulative, where=cumulative < 1, color='red', alpha=0.3)

    # Starting point
    ax.axhline(1, color='black', linestyle='--', linewidth=1)

    # Final value annotation
    final_return = cumulative[-1] - 1
    ax.text(len(cumulative), cumulative[-1], f'{final_return:+.1%}',
            fontsize=12, fontweight='bold',
            color='green' if final_return > 0 else 'red')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Period Number')
    ax.set_ylabel('Cumulative Return (1 = break-even)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_backtest_dashboard(
    result: 'BacktestResult',
    figsize: Tuple[int, int] = (16, 12),
) -> Figure:
    """
    Create a comprehensive dashboard with multiple backtest visualizations.

    Args:
        result: BacktestResult object
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Equity Curve (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        ax1.plot(result.equity_curve.index, result.equity_curve.values, 'b-', linewidth=1.5)
        ax1.set_title('Equity Curve', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)

    # 2. Metrics Summary (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    metrics_text = (
        f"Performance Summary\n"
        f"{'─' * 25}\n"
        f"Accuracy: {result.metrics.get('accuracy', 0):.1%}\n"
        f"Win Rate: {result.metrics.get('win_rate', 0):.1%}\n"
        f"Total Return: {result.metrics.get('total_return', 0):.1%}\n"
        f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}\n"
        f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.1%}\n"
        f"Total Trades: {len(result.trades)}"
    )
    ax2.text(0.5, 0.5, metrics_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 3. Drawdown (middle left, spans 2 columns)
    ax3 = fig.add_subplot(gs[1, :2])
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        running_max = result.equity_curve.cummax()
        drawdown = (result.equity_curve - running_max) / running_max * 100
        ax3.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax3.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
        ax3.set_title('Drawdown', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)

    # 4. Confusion Matrix (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    if result.predictions:
        preds = [p['predicted_direction'] for p in result.predictions if p.get('actual_direction', -1) >= 0]
        acts = [p['actual_direction'] for p in result.predictions if p.get('actual_direction', -1) >= 0]
        if preds and acts:
            from src.backtesting.metrics import calculate_confusion_matrix
            cm = calculate_confusion_matrix(preds, acts)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                        xticklabels=['Bear', 'Bull'], yticklabels=['Bear', 'Bull'])
            ax4.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('Actual')

    # 5. Returns Distribution (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    if result.returns is not None and len(result.returns) > 0:
        ax5.hist(result.returns * 100, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
        ax5.axvline(result.returns.mean() * 100, color='navy', linestyle='-', linewidth=2)
        ax5.set_title('Returns Distribution', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Return (%)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)

    # 6. Confidence vs Accuracy (bottom middle)
    ax6 = fig.add_subplot(gs[2, 1])
    if result.predictions:
        conf_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        bin_labels = []
        bin_accuracy = []
        for low, high in conf_bins:
            in_bin = [p for p in result.predictions
                      if low <= p['confidence'] < high and p.get('is_correct') is not None]
            if in_bin:
                acc = sum(1 for p in in_bin if p['is_correct']) / len(in_bin)
                bin_labels.append(f'{low:.0%}-{high:.0%}')
                bin_accuracy.append(acc)

        if bin_accuracy:
            ax6.bar(bin_labels, [a * 100 for a in bin_accuracy], color='teal', alpha=0.7)
            ax6.axhline(50, color='black', linestyle='--', linewidth=1)
            ax6.set_title('Accuracy by Confidence', fontsize=11, fontweight='bold')
            ax6.set_xlabel('Confidence Range')
            ax6.set_ylabel('Accuracy (%)')
            ax6.set_ylim(0, 100)
            ax6.grid(True, alpha=0.3, axis='y')

    # 7. Trade P&L (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    if result.trades:
        pnls = [t.get('net_pnl', t.get('return_pct', 0) * 100) if isinstance(t, dict)
                else t.net_pnl for t in result.trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax7.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax7.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax7.set_title('Trade P&L', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Trade #')
        ax7.set_ylabel('P&L ($)')
        ax7.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Backtest Results: {result.ticker} ({result.start_date} to {result.end_date})',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig
