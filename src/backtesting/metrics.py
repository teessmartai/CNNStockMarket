"""
Performance Metrics Module (TASK-7.2)

Comprehensive performance metrics for backtesting including:
- Prediction accuracy metrics
- Trading performance metrics
- Risk metrics
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_prediction_metrics(
    predictions: List[int],
    actuals: List[int],
) -> Dict[str, float]:
    """
    Calculate prediction accuracy metrics.

    Args:
        predictions: List of predicted directions (1=bullish, 0=bearish)
        actuals: List of actual directions

    Returns:
        Dictionary with accuracy, precision, recall, F1, etc.
    """
    if not predictions or not actuals or len(predictions) != len(actuals):
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'n_predictions': 0,
        }

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Confusion matrix components
    true_positives = np.sum((predictions == 1) & (actuals == 1))
    true_negatives = np.sum((predictions == 0) & (actuals == 0))
    false_positives = np.sum((predictions == 1) & (actuals == 0))
    false_negatives = np.sum((predictions == 0) & (actuals == 1))

    total = len(predictions)

    # Accuracy
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0

    # Precision (of positive predictions)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # Recall (sensitivity, true positive rate)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Specificity (true negative rate)
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Matthews Correlation Coefficient
    mcc_num = (true_positives * true_negatives) - (false_positives * false_negatives)
    mcc_denom = np.sqrt(
        (true_positives + false_positives) *
        (true_positives + false_negatives) *
        (true_negatives + false_positives) *
        (true_negatives + false_negatives)
    )
    mcc = mcc_num / mcc_denom if mcc_denom > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'mcc': mcc,
        'true_positives': int(true_positives),
        'true_negatives': int(true_negatives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'n_predictions': total,
    }


def calculate_confusion_matrix(
    predictions: List[int],
    actuals: List[int],
) -> np.ndarray:
    """
    Calculate confusion matrix.

    Returns:
        2x2 numpy array: [[TN, FP], [FN, TP]]
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    tp = np.sum((predictions == 1) & (actuals == 1))
    tn = np.sum((predictions == 0) & (actuals == 0))
    fp = np.sum((predictions == 1) & (actuals == 0))
    fn = np.sum((predictions == 0) & (actuals == 1))

    return np.array([[tn, fp], [fn, tp]])


def calculate_trading_metrics(
    returns: Union[pd.Series, List[float], np.ndarray],
    risk_free_rate: float = 0.02,  # Annual risk-free rate (2%)
    periods_per_year: int = 252,  # Trading days per year
) -> Dict[str, float]:
    """
    Calculate trading performance metrics.

    Args:
        returns: Series or array of period returns
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        periods_per_year: Number of trading periods per year

    Returns:
        Dictionary with return metrics
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    elif isinstance(returns, pd.Series):
        returns = returns.values

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0,
            'n_trades': 0,
        }

    # Total return (compounded)
    total_return = np.prod(1 + returns) - 1

    # Annualized return
    n_periods = len(returns)
    if n_periods > 0:
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    else:
        annualized_return = 0

    # Volatility
    volatility = np.std(returns) * np.sqrt(periods_per_year)

    # Sharpe Ratio
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if np.std(excess_returns) > 0:
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = 0

    # Sortino Ratio (uses downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns) * np.sqrt(periods_per_year)
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
    else:
        sortino_ratio = sharpe_ratio if sharpe_ratio > 0 else 0

    # Maximum drawdown (for Calmar)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    n_wins = len(winning_trades)
    n_losses = len(losing_trades)
    total_trades = n_wins + n_losses

    win_rate = n_wins / total_trades if total_trades > 0 else 0

    # Average win/loss
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    avg_loss = abs(np.mean(losing_trades)) if len(losing_trades) > 0 else 0

    # Win/loss ratio
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Profit factor
    gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
    gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': abs(max_drawdown),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'expectancy': expectancy,
        'n_trades': total_trades,
        'n_wins': n_wins,
        'n_losses': n_losses,
    }


def calculate_risk_metrics(
    returns: Union[pd.Series, List[float], np.ndarray],
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Calculate risk metrics.

    Args:
        returns: Series or array of period returns
        confidence_level: Confidence level for VaR calculation
        periods_per_year: Number of trading periods per year

    Returns:
        Dictionary with risk metrics
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    elif isinstance(returns, pd.Series):
        returns = returns.values

    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {
            'volatility_daily': 0.0,
            'volatility_annual': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'cvar_95': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'downside_deviation': 0.0,
        }

    # Volatility
    volatility_daily = np.std(returns)
    volatility_annual = volatility_daily * np.sqrt(periods_per_year)

    # Value at Risk (parametric, assuming normal distribution)
    var_95 = np.percentile(returns, (1 - 0.95) * 100)
    var_99 = np.percentile(returns, (1 - 0.99) * 100)

    # Conditional VaR (Expected Shortfall)
    cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95

    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # Maximum drawdown duration
    in_drawdown = cumulative < running_max
    drawdown_duration = 0
    max_drawdown_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            drawdown_duration += 1
            max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
        else:
            drawdown_duration = 0

    # Skewness
    if volatility_daily > 0:
        skewness = np.mean(((returns - np.mean(returns)) / volatility_daily) ** 3)
    else:
        skewness = 0

    # Kurtosis (excess kurtosis)
    if volatility_daily > 0:
        kurtosis = np.mean(((returns - np.mean(returns)) / volatility_daily) ** 4) - 3
    else:
        kurtosis = 0

    # Downside deviation
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0

    return {
        'volatility_daily': volatility_daily,
        'volatility_annual': volatility_annual,
        'var_95': abs(var_95),
        'var_99': abs(var_99),
        'cvar_95': abs(cvar_95),
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'downside_deviation': downside_deviation,
    }


def calculate_equity_curve_stats(
    equity_curve: pd.Series,
    initial_capital: float = 100000.0,
) -> Dict[str, float]:
    """
    Calculate statistics from equity curve.

    Args:
        equity_curve: Series of portfolio values over time
        initial_capital: Starting capital

    Returns:
        Dictionary with equity curve statistics
    """
    if equity_curve is None or len(equity_curve) == 0:
        return {
            'final_value': initial_capital,
            'peak_value': initial_capital,
            'trough_value': initial_capital,
            'total_return': 0.0,
        }

    final_value = equity_curve.iloc[-1]
    peak_value = equity_curve.max()
    trough_value = equity_curve.min()
    total_return = (final_value - initial_capital) / initial_capital

    # Calculate drawdown series
    running_max = equity_curve.cummax()
    drawdown_series = (equity_curve - running_max) / running_max

    # Time in drawdown
    in_drawdown = equity_curve < running_max
    time_in_drawdown_pct = in_drawdown.mean()

    # Recovery factor
    max_dd = abs(drawdown_series.min())
    recovery_factor = total_return / max_dd if max_dd > 0 else 0

    return {
        'final_value': final_value,
        'peak_value': peak_value,
        'trough_value': trough_value,
        'total_return': total_return,
        'time_in_drawdown_pct': time_in_drawdown_pct,
        'recovery_factor': recovery_factor,
    }


def calculate_monthly_returns(
    equity_curve: pd.Series,
) -> pd.DataFrame:
    """
    Calculate monthly returns from equity curve.

    Args:
        equity_curve: Series of portfolio values with datetime index

    Returns:
        DataFrame with monthly returns by year
    """
    if equity_curve is None or len(equity_curve) < 2:
        return pd.DataFrame()

    # Resample to monthly
    monthly = equity_curve.resample('ME').last()
    monthly_returns = monthly.pct_change().dropna()

    # Create pivot table by year and month
    if len(monthly_returns) == 0:
        return pd.DataFrame()

    monthly_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values,
    })

    pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

    # Add yearly total
    pivot['Year Total'] = (1 + pivot.fillna(0)).prod(axis=1) - 1

    return pivot


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 63,  # ~3 months
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.

    Args:
        returns: Series of returns with datetime index
        window: Rolling window size in periods

    Returns:
        DataFrame with rolling metrics
    """
    if returns is None or len(returns) < window:
        return pd.DataFrame()

    # Rolling return
    rolling_return = returns.rolling(window).apply(
        lambda x: np.prod(1 + x) - 1, raw=True
    )

    # Rolling volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)

    # Rolling Sharpe (simplified)
    rolling_sharpe = (returns.rolling(window).mean() * 252) / (rolling_vol)

    # Rolling win rate
    rolling_win_rate = (returns > 0).rolling(window).mean()

    # Rolling max drawdown
    def calc_max_dd(x):
        cumulative = np.cumprod(1 + x)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return abs(np.min(drawdowns))

    rolling_max_dd = returns.rolling(window).apply(calc_max_dd, raw=True)

    return pd.DataFrame({
        'rolling_return': rolling_return,
        'rolling_volatility': rolling_vol,
        'rolling_sharpe': rolling_sharpe,
        'rolling_win_rate': rolling_win_rate,
        'rolling_max_drawdown': rolling_max_dd,
    })


def compare_strategies(
    results: List[Dict[str, float]],
    strategy_names: List[str],
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.

    Args:
        results: List of metric dictionaries
        strategy_names: List of strategy names

    Returns:
        DataFrame comparing strategies
    """
    if not results or not strategy_names:
        return pd.DataFrame()

    # Key metrics to compare
    key_metrics = [
        'total_return',
        'annualized_return',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'win_rate',
        'profit_factor',
        'accuracy',
    ]

    comparison_data = {}
    for name, metrics in zip(strategy_names, results):
        comparison_data[name] = {
            k: metrics.get(k, np.nan) for k in key_metrics
        }

    df = pd.DataFrame(comparison_data).T

    # Format for display
    format_as_pct = ['total_return', 'annualized_return', 'max_drawdown', 'win_rate', 'accuracy']
    for col in format_as_pct:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.1%}" if not np.isnan(x) else "N/A")

    format_as_float = ['sharpe_ratio', 'sortino_ratio', 'profit_factor']
    for col in format_as_float:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")

    return df
