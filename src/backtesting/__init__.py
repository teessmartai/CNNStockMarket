"""
Backtesting Framework for CNN Stock Market Prediction (Phase 7)

This module provides comprehensive backtesting capabilities including:
- BacktestEngine: Core backtesting orchestrator
- Performance Metrics: Prediction accuracy, trading, and risk metrics
- Trade Simulator: Realistic trade simulation with costs
- Visualization: Charts and plots for backtest analysis
- Report Generator: Automated HTML report generation

Usage:
    from src.backtesting import BacktestEngine, BacktestConfig
    from src.prediction.predictor import Predictor

    predictor = Predictor('models/best_model.pt')
    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_capital=100000,
    )

    engine = BacktestEngine(predictor)
    result = engine.run_backtest('AAPL', config)
    print(f"Accuracy: {result.metrics['accuracy']:.1%}")
"""

from src.backtesting.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    RandomizedBacktestConfig,
    RandomizedBacktestResult,
)
from src.backtesting.metrics import (
    calculate_prediction_metrics,
    calculate_trading_metrics,
    calculate_risk_metrics,
)
from src.backtesting.simulator import (
    TradeSimulator,
    Trade,
    Position,
    SimulatorConfig,
)
from src.backtesting.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_confusion_matrix,
    plot_returns_distribution,
    plot_monthly_returns,
    plot_signal_timeline,
)
from src.backtesting.report import (
    generate_report,
    export_trades_csv,
    export_metrics_json,
)

__all__ = [
    # Engine
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'RandomizedBacktestConfig',
    'RandomizedBacktestResult',
    # Metrics
    'calculate_prediction_metrics',
    'calculate_trading_metrics',
    'calculate_risk_metrics',
    # Simulator
    'TradeSimulator',
    'Trade',
    'Position',
    'SimulatorConfig',
    # Plots
    'plot_equity_curve',
    'plot_drawdown',
    'plot_confusion_matrix',
    'plot_returns_distribution',
    'plot_monthly_returns',
    'plot_signal_timeline',
    # Report
    'generate_report',
    'export_trades_csv',
    'export_metrics_json',
]
