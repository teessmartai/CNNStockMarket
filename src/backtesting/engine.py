"""
Backtesting Engine Core (TASK-7.1 & 7.1b)

Main backtesting orchestrator for evaluating model performance on historical data.
Supports both chronological and randomized period backtesting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from src.models.cnn import StockCNN
from src.data.fetcher import fetch_stock_data
from src.data.preprocessor import normalize
from src.utils.config import WINDOW_SIZE, DEFAULT_HORIZON, DEVICE


@dataclass
class BacktestConfig:
    """Configuration for backtest parameters."""

    # Date range
    start_date: Optional[str] = None  # Format: 'YYYY-MM-DD'
    end_date: Optional[str] = None

    # Capital and costs
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage

    # Strategy settings
    confidence_threshold: float = 0.5  # Minimum confidence to trade
    position_sizing: str = 'fixed'  # 'fixed', 'percentage', 'kelly'
    position_size: float = 1.0  # For fixed sizing
    position_pct: float = 0.1  # For percentage sizing (10%)
    allow_short: bool = False  # Long-only by default

    # Prediction settings
    horizon: int = DEFAULT_HORIZON
    prediction_frequency: int = 1  # Predict every N periods

    # Walk-forward settings
    walk_forward: bool = False
    train_periods: int = 756  # ~3 years of daily data
    test_periods: int = 252  # ~1 year of daily data
    retrain_frequency: int = 63  # Retrain every ~3 months

    # Output settings
    verbose: bool = True


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Configuration
    config: BacktestConfig
    ticker: str
    start_date: str
    end_date: str

    # Predictions
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    actuals: List[int] = field(default_factory=list)  # 1=bullish, 0=bearish

    # Trading
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    returns: Optional[pd.Series] = None

    # Metrics (calculated after backtest)
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a summary string of backtest results."""
        lines = [
            f"Backtest Results for {self.ticker}",
            f"Period: {self.start_date} to {self.end_date}",
            "-" * 40,
        ]

        if self.metrics:
            lines.extend([
                f"Accuracy: {self.metrics.get('accuracy', 0):.1%}",
                f"Total Return: {self.metrics.get('total_return', 0):.1%}",
                f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}",
                f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.1%}",
                f"Win Rate: {self.metrics.get('win_rate', 0):.1%}",
                f"Total Trades: {len(self.trades)}",
            ])

        return "\n".join(lines)


@dataclass
class RandomizedBacktestConfig:
    """Configuration for randomized period backtesting."""

    # Period sampling
    n_periods: int = 100  # Number of test periods to sample
    period_length: Optional[int] = None  # Auto: window_size + horizon + 1
    min_periods: int = 50  # Minimum periods for valid test

    # Data separation
    train_end_idx: Optional[int] = None  # Index where training data ends
    gap_buffer: Optional[int] = None  # Gap between train and test (default: horizon)

    # Randomization
    random_seed: int = 42  # For reproducibility
    stratify_by_regime: bool = False  # Ensure diverse market conditions

    # Benchmark
    benchmark_ticker: Optional[str] = None  # Compare against index (e.g., "SPY")

    # Statistics
    confidence_level: float = 0.95  # For CI calculation
    bootstrap_samples: int = 1000  # For bootstrap CI

    # Strategy settings (same as BacktestConfig)
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    confidence_threshold: float = 0.5
    horizon: int = DEFAULT_HORIZON

    verbose: bool = True


@dataclass
class SampledPeriod:
    """Represents a sampled test period."""

    start_idx: int
    end_idx: int
    signal_idx: int
    outcome_idx: int
    signal: str  # 'BUY' or 'SELL'
    confidence: float
    predicted_direction: int  # 1=bullish, 0=bearish
    actual_direction: int  # 1=bullish, 0=bearish
    is_correct: bool
    model_return: float
    benchmark_return: Optional[float] = None


@dataclass
class RandomizedBacktestResult:
    """Container for randomized backtest results."""

    config: RandomizedBacktestConfig
    ticker: str

    # Sampled periods
    n_periods: int = 0
    periods: List[SampledPeriod] = field(default_factory=list)

    # Prediction metrics
    win_rate: float = 0.0
    win_rate_ci: Tuple[float, float] = (0.0, 0.0)

    # Return metrics
    avg_return: float = 0.0
    avg_return_ci: Tuple[float, float] = (0.0, 0.0)
    total_return: float = 0.0

    # Benchmark comparison
    benchmark_return: float = 0.0
    excess_return: float = 0.0

    # Distribution of outcomes
    periods_by_outcome: Dict[str, int] = field(default_factory=dict)

    # Raw data
    return_distribution: List[float] = field(default_factory=list)

    # Full metrics dict
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"Randomized Backtest Results for {self.ticker}",
            f"Periods Sampled: {self.n_periods}",
            "-" * 40,
            f"Win Rate: {self.win_rate:.1%} (CI: {self.win_rate_ci[0]:.1%} - {self.win_rate_ci[1]:.1%})",
            f"Avg Return: {self.avg_return:.2%} (CI: {self.avg_return_ci[0]:.2%} - {self.avg_return_ci[1]:.2%})",
            f"Total Return: {self.total_return:.2%}",
        ]

        if self.benchmark_return != 0:
            lines.extend([
                f"Benchmark Return: {self.benchmark_return:.2%}",
                f"Excess Return: {self.excess_return:.2%}",
            ])

        if self.periods_by_outcome:
            lines.extend([
                "",
                "Outcome Distribution:",
                f"  True Positive: {self.periods_by_outcome.get('true_positive', 0)}",
                f"  True Negative: {self.periods_by_outcome.get('true_negative', 0)}",
                f"  False Positive: {self.periods_by_outcome.get('false_positive', 0)}",
                f"  False Negative: {self.periods_by_outcome.get('false_negative', 0)}",
            ])

        return "\n".join(lines)


class BacktestEngine:
    """
    Main backtesting orchestrator.

    Runs predictions on historical data and evaluates model performance
    without look-ahead bias.
    """

    def __init__(
        self,
        predictor=None,
        model_path: Optional[Path] = None,
        device: torch.device = DEVICE,
    ):
        """
        Initialize the backtest engine.

        Args:
            predictor: An existing Predictor instance
            model_path: Path to model checkpoint (alternative to predictor)
            device: Device for inference
        """
        self.device = device

        if predictor is not None:
            self.predictor = predictor
            self.model = predictor.model
        elif model_path is not None:
            from src.prediction.predictor import Predictor
            self.predictor = Predictor(model_path, device=device)
            self.model = self.predictor.model
        else:
            self.predictor = None
            self.model = None

    def _fetch_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical data for backtesting."""
        # Fetch extra data for the initial window
        start = datetime.strptime(start_date, '%Y-%m-%d')
        extended_start = start - timedelta(days=int(WINDOW_SIZE * 1.5))
        extended_start_str = extended_start.strftime('%Y-%m-%d')

        df = fetch_stock_data(ticker, extended_start_str, end_date)
        return df

    def _prepare_window(self, df: pd.DataFrame, end_idx: int) -> torch.Tensor:
        """
        Prepare a window of data for prediction.

        Args:
            df: Full DataFrame
            end_idx: Index of the last bar in the window

        Returns:
            Tensor of shape [1, WINDOW_SIZE, 5]
        """
        start_idx = end_idx - WINDOW_SIZE + 1
        if start_idx < 0:
            raise ValueError(f"Insufficient data: need {WINDOW_SIZE} bars")

        window_df = df.iloc[start_idx:end_idx + 1].copy()
        window_normalized = normalize(window_df, method='minmax')

        values = window_normalized.values.astype(np.float32)
        tensor = torch.from_numpy(values).unsqueeze(0)

        return tensor.to(self.device)

    def _calculate_actual_direction(
        self,
        df: pd.DataFrame,
        signal_idx: int,
        horizon: int,
    ) -> int:
        """
        Calculate actual price direction after horizon periods.

        Returns:
            1 if price increased (bullish), 0 if decreased (bearish)
        """
        outcome_idx = signal_idx + horizon
        if outcome_idx >= len(df):
            return -1  # Cannot determine

        signal_price = df.iloc[signal_idx]['Close']
        outcome_price = df.iloc[outcome_idx]['Close']

        return 1 if outcome_price > signal_price else 0

    def run_backtest(
        self,
        ticker: str,
        config: BacktestConfig,
    ) -> BacktestResult:
        """
        Run a chronological backtest on historical data.

        Args:
            ticker: Stock ticker symbol
            config: Backtest configuration

        Returns:
            BacktestResult with predictions, trades, and metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Initialize with predictor or model_path.")

        # Determine date range
        if config.end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = config.end_date

        if config.start_date is None:
            start = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)
            start_date = start.strftime('%Y-%m-%d')
        else:
            start_date = config.start_date

        # Fetch data
        if config.verbose:
            print(f"Fetching data for {ticker}...")
        df = self._fetch_data(ticker, start_date, end_date)

        if len(df) < WINDOW_SIZE + config.horizon:
            raise ValueError(
                f"Insufficient data: need at least {WINDOW_SIZE + config.horizon} bars, "
                f"got {len(df)}"
            )

        # Find the start index in the actual backtest period
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        backtest_start_idx = None
        for i, idx in enumerate(df.index):
            if idx >= start_dt:
                backtest_start_idx = i
                break

        if backtest_start_idx is None or backtest_start_idx < WINDOW_SIZE:
            backtest_start_idx = WINDOW_SIZE

        # Initialize result
        result = BacktestResult(
            config=config,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        # Run predictions chronologically
        self.model.eval()
        predictions = []
        signals = []
        confidences = []
        actuals = []

        n_predictions = 0
        total_bars = len(df) - config.horizon - backtest_start_idx

        if config.verbose:
            print(f"Running backtest from index {backtest_start_idx} to {len(df) - config.horizon}...")

        for i in range(backtest_start_idx, len(df) - config.horizon, config.prediction_frequency):
            # Prepare input window
            try:
                input_tensor = self._prepare_window(df, i)
            except ValueError:
                continue

            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = output.squeeze().cpu().numpy()

            bearish_prob = float(probs[0])
            bullish_prob = float(probs[1])

            if bullish_prob > bearish_prob:
                signal = 'BUY'
                confidence = bullish_prob
                predicted_direction = 1
            else:
                signal = 'SELL'
                confidence = bearish_prob
                predicted_direction = 0

            # Get actual outcome
            actual = self._calculate_actual_direction(df, i, config.horizon)

            # Store results
            pred_data = {
                'date': df.index[i],
                'signal': signal,
                'confidence': confidence,
                'bullish_prob': bullish_prob,
                'bearish_prob': bearish_prob,
                'predicted_direction': predicted_direction,
                'actual_direction': actual,
                'is_correct': predicted_direction == actual if actual >= 0 else None,
                'price': df.iloc[i]['Close'],
            }

            predictions.append(pred_data)
            signals.append(signal)
            confidences.append(confidence)
            if actual >= 0:
                actuals.append(actual)

            n_predictions += 1
            if config.verbose and n_predictions % 50 == 0:
                progress = (i - backtest_start_idx) / total_bars * 100
                print(f"  Progress: {progress:.1f}% ({n_predictions} predictions)")

        result.predictions = predictions
        result.signals = signals
        result.confidences = confidences
        result.actuals = actuals

        if config.verbose:
            print(f"Completed {n_predictions} predictions")

        # Run trade simulation
        from src.backtesting.simulator import TradeSimulator, SimulatorConfig

        sim_config = SimulatorConfig(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
            confidence_threshold=config.confidence_threshold,
            position_sizing=config.position_sizing,
            position_size=config.position_size,
            position_pct=config.position_pct,
            allow_short=config.allow_short,
            horizon=config.horizon,
        )

        simulator = TradeSimulator(sim_config)
        trades, equity_curve = simulator.simulate(predictions, df)

        result.trades = trades
        result.equity_curve = equity_curve
        if equity_curve is not None and len(equity_curve) > 1:
            result.returns = equity_curve.pct_change().dropna()

        # Calculate metrics
        from src.backtesting.metrics import (
            calculate_prediction_metrics,
            calculate_trading_metrics,
            calculate_risk_metrics,
        )

        pred_metrics = calculate_prediction_metrics(
            [p['predicted_direction'] for p in predictions if p['actual_direction'] >= 0],
            [p['actual_direction'] for p in predictions if p['actual_direction'] >= 0],
        )

        result.metrics.update(pred_metrics)

        if result.returns is not None and len(result.returns) > 0:
            trading_metrics = calculate_trading_metrics(result.returns)
            risk_metrics = calculate_risk_metrics(result.returns)
            result.metrics.update(trading_metrics)
            result.metrics.update(risk_metrics)

        if config.verbose:
            print("\n" + result.summary())

        return result

    def run_walk_forward_backtest(
        self,
        ticker: str,
        config: BacktestConfig,
    ) -> List[BacktestResult]:
        """
        Run walk-forward validation backtest.

        Trains on period 1, tests on period 2, retrains on 1-2, tests on 3, etc.
        Note: This requires the ability to retrain the model, which is complex.
        For simplicity, this implementation uses a fixed model but tests on
        multiple sequential periods.

        Args:
            ticker: Stock ticker symbol
            config: Backtest configuration

        Returns:
            List of BacktestResult for each test period
        """
        results = []

        # For a simplified walk-forward, we'll just run backtests on
        # sequential periods without retraining
        if config.end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = config.end_date

        if config.start_date is None:
            start = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=config.train_periods + config.test_periods * 3)
            start_date = start.strftime('%Y-%m-%d')
        else:
            start_date = config.start_date

        # Calculate period boundaries
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        current_start = start_dt + timedelta(days=config.train_periods)
        period_num = 1

        while current_start + timedelta(days=config.test_periods) <= end_dt:
            period_end = current_start + timedelta(days=config.test_periods)

            period_config = BacktestConfig(
                start_date=current_start.strftime('%Y-%m-%d'),
                end_date=period_end.strftime('%Y-%m-%d'),
                initial_capital=config.initial_capital,
                commission=config.commission,
                slippage=config.slippage,
                confidence_threshold=config.confidence_threshold,
                position_sizing=config.position_sizing,
                position_size=config.position_size,
                position_pct=config.position_pct,
                allow_short=config.allow_short,
                horizon=config.horizon,
                prediction_frequency=config.prediction_frequency,
                verbose=False,
            )

            if config.verbose:
                print(f"\nPeriod {period_num}: {period_config.start_date} to {period_config.end_date}")

            try:
                result = self.run_backtest(ticker, period_config)
                results.append(result)

                if config.verbose:
                    print(f"  Accuracy: {result.metrics.get('accuracy', 0):.1%}")
                    print(f"  Return: {result.metrics.get('total_return', 0):.1%}")
            except Exception as e:
                if config.verbose:
                    print(f"  Error: {e}")

            current_start = period_end
            period_num += 1

        if config.verbose:
            print(f"\nCompleted {len(results)} walk-forward periods")

        return results

    def run_randomized_backtest(
        self,
        ticker: str,
        data: pd.DataFrame,
        config: RandomizedBacktestConfig,
    ) -> RandomizedBacktestResult:
        """
        Run randomized period backtesting.

        Samples non-overlapping test periods for robust performance evaluation.

        Args:
            ticker: Ticker symbol (for labeling)
            data: DataFrame with OHLCV data
            config: Randomized backtest configuration

        Returns:
            RandomizedBacktestResult with statistical analysis
        """
        if self.model is None:
            raise ValueError("No model loaded.")

        # Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Calculate period length
        period_length = config.period_length
        if period_length is None:
            period_length = WINDOW_SIZE + config.horizon + 1

        # Determine valid sampling range
        train_end_idx = config.train_end_idx
        if train_end_idx is None:
            # Default: use first 70% for "training", rest for testing
            train_end_idx = int(len(data) * 0.7)

        gap = config.gap_buffer if config.gap_buffer is not None else config.horizon
        sample_start_idx = train_end_idx + gap + WINDOW_SIZE
        sample_end_idx = len(data) - config.horizon - 1

        available_range = sample_end_idx - sample_start_idx

        if available_range < period_length * config.min_periods:
            raise ValueError(
                f"Insufficient data for {config.min_periods} non-overlapping periods. "
                f"Available range: {available_range}, needed: {period_length * config.min_periods}"
            )

        # Sample non-overlapping periods
        sampled_periods = self._sample_periods(
            data,
            sample_start_idx,
            sample_end_idx,
            config.n_periods,
            period_length,
            config.horizon,
        )

        if len(sampled_periods) < config.min_periods:
            raise ValueError(
                f"Could only sample {len(sampled_periods)} periods, "
                f"need at least {config.min_periods}"
            )

        # Fetch benchmark data if specified
        benchmark_data = None
        if config.benchmark_ticker:
            try:
                start_date = data.index[0].strftime('%Y-%m-%d')
                end_date = data.index[-1].strftime('%Y-%m-%d')
                benchmark_data = fetch_stock_data(
                    config.benchmark_ticker, start_date, end_date
                )
            except Exception:
                benchmark_data = None

        # Evaluate each period
        self.model.eval()
        periods = []

        for i, (signal_idx, outcome_idx) in enumerate(sampled_periods):
            # Prepare input window
            try:
                input_tensor = self._prepare_window(data, signal_idx)
            except ValueError:
                continue

            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = output.squeeze().cpu().numpy()

            bearish_prob = float(probs[0])
            bullish_prob = float(probs[1])

            if bullish_prob > bearish_prob:
                signal = 'BUY'
                confidence = bullish_prob
                predicted_direction = 1
            else:
                signal = 'SELL'
                confidence = bearish_prob
                predicted_direction = 0

            # Calculate actual direction
            signal_price = data.iloc[signal_idx]['Close']
            outcome_price = data.iloc[outcome_idx]['Close']
            actual_direction = 1 if outcome_price > signal_price else 0

            # Calculate return
            price_change = (outcome_price - signal_price) / signal_price
            if signal == 'BUY':
                model_return = price_change
            else:
                model_return = -price_change if not config.allow_short else -price_change

            # Apply costs
            model_return -= (config.commission + config.slippage) * 2  # Entry and exit

            # Benchmark return
            benchmark_return = None
            if benchmark_data is not None:
                try:
                    bench_signal = benchmark_data.iloc[signal_idx]['Close']
                    bench_outcome = benchmark_data.iloc[outcome_idx]['Close']
                    benchmark_return = (bench_outcome - bench_signal) / bench_signal
                except (IndexError, KeyError):
                    pass

            period = SampledPeriod(
                start_idx=signal_idx - WINDOW_SIZE,
                end_idx=outcome_idx,
                signal_idx=signal_idx,
                outcome_idx=outcome_idx,
                signal=signal,
                confidence=confidence,
                predicted_direction=predicted_direction,
                actual_direction=actual_direction,
                is_correct=predicted_direction == actual_direction,
                model_return=model_return,
                benchmark_return=benchmark_return,
            )
            periods.append(period)

            if config.verbose and (i + 1) % 20 == 0:
                print(f"  Evaluated {i + 1}/{len(sampled_periods)} periods")

        # Calculate statistics
        result = self._calculate_randomized_stats(periods, config)
        result.config = config
        result.ticker = ticker
        result.periods = periods

        if config.verbose:
            print("\n" + result.summary())

        return result

    def _sample_periods(
        self,
        data: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        n_periods: int,
        period_length: int,
        horizon: int,
    ) -> List[Tuple[int, int]]:
        """
        Sample non-overlapping test periods.

        Returns list of (signal_idx, outcome_idx) tuples.
        """
        # Calculate how many non-overlapping periods we can fit
        available_range = end_idx - start_idx
        max_periods = available_range // (period_length + horizon)

        n_to_sample = min(n_periods, max_periods)

        # Generate all possible period start indices
        possible_starts = list(range(start_idx, end_idx - period_length - horizon, period_length + horizon))

        # Randomly sample without replacement
        if len(possible_starts) < n_to_sample:
            sampled_starts = possible_starts
        else:
            sampled_starts = random.sample(possible_starts, n_to_sample)

        # Convert to (signal_idx, outcome_idx) pairs
        periods = []
        for start in sampled_starts:
            signal_idx = start
            outcome_idx = start + horizon
            if outcome_idx < len(data):
                periods.append((signal_idx, outcome_idx))

        return periods

    def _calculate_randomized_stats(
        self,
        periods: List[SampledPeriod],
        config: RandomizedBacktestConfig,
    ) -> RandomizedBacktestResult:
        """Calculate statistics from sampled periods."""
        result = RandomizedBacktestResult(config=config, ticker="")

        if not periods:
            return result

        result.n_periods = len(periods)

        # Win rate
        correct_count = sum(1 for p in periods if p.is_correct)
        result.win_rate = correct_count / len(periods)

        # Return statistics
        returns = [p.model_return for p in periods]
        result.return_distribution = returns
        result.avg_return = np.mean(returns)
        result.total_return = np.sum(returns)

        # Benchmark comparison
        benchmark_returns = [p.benchmark_return for p in periods if p.benchmark_return is not None]
        if benchmark_returns:
            result.benchmark_return = np.sum(benchmark_returns)
            result.excess_return = result.total_return - result.benchmark_return

        # Outcome distribution
        result.periods_by_outcome = {
            'true_positive': sum(1 for p in periods if p.predicted_direction == 1 and p.actual_direction == 1),
            'true_negative': sum(1 for p in periods if p.predicted_direction == 0 and p.actual_direction == 0),
            'false_positive': sum(1 for p in periods if p.predicted_direction == 1 and p.actual_direction == 0),
            'false_negative': sum(1 for p in periods if p.predicted_direction == 0 and p.actual_direction == 1),
        }

        # Bootstrap confidence intervals
        win_rate_ci = self._bootstrap_ci(
            [1 if p.is_correct else 0 for p in periods],
            config.confidence_level,
            config.bootstrap_samples,
        )
        result.win_rate_ci = win_rate_ci

        return_ci = self._bootstrap_ci(
            returns,
            config.confidence_level,
            config.bootstrap_samples,
        )
        result.avg_return_ci = return_ci

        # Store metrics
        result.metrics = {
            'n_periods': result.n_periods,
            'win_rate': result.win_rate,
            'win_rate_ci_lower': win_rate_ci[0],
            'win_rate_ci_upper': win_rate_ci[1],
            'avg_return': result.avg_return,
            'avg_return_ci_lower': return_ci[0],
            'avg_return_ci_upper': return_ci[1],
            'total_return': result.total_return,
            'benchmark_return': result.benchmark_return,
            'excess_return': result.excess_return,
            'return_std': np.std(returns),
            'return_sharpe': result.avg_return / np.std(returns) if np.std(returns) > 0 else 0,
        }

        return result

    def _bootstrap_ci(
        self,
        values: List[float],
        confidence_level: float,
        n_samples: int,
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for the mean."""
        if not values:
            return (0.0, 0.0)

        values = np.array(values)
        bootstrap_means = []

        for _ in range(n_samples):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return (lower, upper)
