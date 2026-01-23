"""
Backtest Report Generator (TASK-7.5)

Automated HTML report generation for backtesting results.
Includes all metrics, visualizations, and analysis.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
import base64
import io

import pandas as pd
import matplotlib.pyplot as plt

from src.backtesting.engine import BacktestResult, RandomizedBacktestResult
from src.backtesting.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_confusion_matrix,
    plot_returns_distribution,
    plot_monthly_returns,
    plot_signal_timeline,
    plot_rolling_metrics,
    plot_period_returns_histogram,
    plot_win_rate_ci,
    plot_period_cumulative_return,
)


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def _format_metric(value: float, metric_type: str = 'number') -> str:
    """Format metric value for display."""
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return 'N/A'

    if metric_type == 'percent':
        return f'{value:.2%}'
    elif metric_type == 'ratio':
        return f'{value:.2f}'
    elif metric_type == 'currency':
        return f'${value:,.2f}'
    elif metric_type == 'integer':
        return f'{int(value):,}'
    else:
        return f'{value:.4f}'


def generate_html_report(
    result: Union[BacktestResult, RandomizedBacktestResult],
    output_path: Path,
    title: str = None,
    include_trades: bool = True,
) -> Path:
    """
    Generate comprehensive HTML report from backtest results.

    Args:
        result: BacktestResult or RandomizedBacktestResult object
        output_path: Path to save the HTML report
        title: Report title (auto-generated if None)
        include_trades: Whether to include trade log table

    Returns:
        Path to generated report
    """
    is_randomized = isinstance(result, RandomizedBacktestResult)

    if title is None:
        if is_randomized:
            title = f"Randomized Backtest Report - {result.ticker}"
        else:
            title = f"Backtest Report - {result.ticker}"

    # Generate charts
    charts = {}

    if is_randomized:
        charts.update(_generate_randomized_charts(result))
    else:
        charts.update(_generate_standard_charts(result))

    # Build HTML
    html = _build_html_template(result, charts, title, include_trades, is_randomized)

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


def _generate_standard_charts(result: BacktestResult) -> Dict[str, str]:
    """Generate charts for standard backtest."""
    charts = {}

    # Equity curve
    if result.equity_curve is not None and len(result.equity_curve) > 0:
        fig = plot_equity_curve(result.equity_curve, title="Portfolio Equity Curve")
        charts['equity_curve'] = _fig_to_base64(fig)

        # Drawdown
        fig = plot_drawdown(result.equity_curve, title="Portfolio Drawdown")
        charts['drawdown'] = _fig_to_base64(fig)

        # Monthly returns
        fig = plot_monthly_returns(result.equity_curve, title="Monthly Returns Heatmap")
        charts['monthly_returns'] = _fig_to_base64(fig)

    # Confusion matrix
    if result.predictions:
        preds = [p['predicted_direction'] for p in result.predictions if p.get('actual_direction', -1) >= 0]
        acts = [p['actual_direction'] for p in result.predictions if p.get('actual_direction', -1) >= 0]
        if preds and acts:
            fig = plot_confusion_matrix(preds, acts, title="Prediction Confusion Matrix")
            charts['confusion_matrix'] = _fig_to_base64(fig)

    # Returns distribution
    if result.returns is not None and len(result.returns) > 0:
        fig = plot_returns_distribution(result.returns, title="Daily Returns Distribution")
        charts['returns_dist'] = _fig_to_base64(fig)

        # Rolling metrics
        if len(result.returns) >= 63:
            fig = plot_rolling_metrics(result.returns, window=63, title="Rolling Performance")
            charts['rolling_metrics'] = _fig_to_base64(fig)

    return charts


def _generate_randomized_charts(result: RandomizedBacktestResult) -> Dict[str, str]:
    """Generate charts for randomized backtest."""
    charts = {}

    # Return distribution
    if result.return_distribution:
        benchmark_returns = None
        if result.periods:
            benchmark_returns = [p.benchmark_return for p in result.periods
                                 if p.benchmark_return is not None]

        fig = plot_period_returns_histogram(
            result.return_distribution,
            benchmark_returns,
            title="Period Returns Distribution"
        )
        charts['returns_dist'] = _fig_to_base64(fig)

    # Win rate with CI
    fig = plot_win_rate_ci(
        result.win_rate,
        result.win_rate_ci[0],
        result.win_rate_ci[1],
        title="Win Rate with 95% Confidence Interval"
    )
    charts['win_rate_ci'] = _fig_to_base64(fig)

    # Cumulative return
    if result.return_distribution:
        fig = plot_period_cumulative_return(
            result.return_distribution,
            title="Cumulative Return (Sequential Periods)"
        )
        charts['cumulative_return'] = _fig_to_base64(fig)

    return charts


def _build_html_template(
    result: Union[BacktestResult, RandomizedBacktestResult],
    charts: Dict[str, str],
    title: str,
    include_trades: bool,
    is_randomized: bool,
) -> str:
    """Build the HTML report template."""
    metrics = result.metrics

    # CSS styles
    css = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
            font-size: 2em;
        }
        .header .subtitle {
            opacity: 0.9;
            margin-top: 10px;
        }
        .section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #1e3c72;
            border-bottom: 2px solid #1e3c72;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #1e3c72;
        }
        .metric-card.positive {
            border-left-color: #28a745;
        }
        .metric-card.negative {
            border-left-color: #dc3545;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #1e3c72;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #1e3c72;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .footer {
            text-align: center;
            color: #666;
            padding: 20px;
            font-size: 0.9em;
        }
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .two-column {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """

    # Build HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'<title>{title}</title>',
        css,
        '</head>',
        '<body>',
    ]

    # Header
    if is_randomized:
        subtitle = f"Randomized Period Analysis | {result.n_periods} Periods Sampled"
    else:
        subtitle = f"{result.start_date} to {result.end_date}"

    html_parts.append(f'''
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
        <div class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>
    ''')

    # Executive Summary
    html_parts.append('<div class="section">')
    html_parts.append('<h2>Executive Summary</h2>')
    html_parts.append('<div class="metrics-grid">')

    if is_randomized:
        key_metrics = [
            ('Win Rate', metrics.get('win_rate', 0), 'percent', metrics.get('win_rate', 0) > 0.5),
            ('Avg Return', metrics.get('avg_return', 0), 'percent', metrics.get('avg_return', 0) > 0),
            ('Total Return', metrics.get('total_return', 0), 'percent', metrics.get('total_return', 0) > 0),
            ('Excess Return', metrics.get('excess_return', 0), 'percent', metrics.get('excess_return', 0) > 0),
            ('Periods Tested', metrics.get('n_periods', 0), 'integer', True),
        ]
    else:
        key_metrics = [
            ('Accuracy', metrics.get('accuracy', 0), 'percent', metrics.get('accuracy', 0) > 0.5),
            ('Total Return', metrics.get('total_return', 0), 'percent', metrics.get('total_return', 0) > 0),
            ('Sharpe Ratio', metrics.get('sharpe_ratio', 0), 'ratio', metrics.get('sharpe_ratio', 0) > 0),
            ('Max Drawdown', metrics.get('max_drawdown', 0), 'percent', False),
            ('Win Rate', metrics.get('win_rate', 0), 'percent', metrics.get('win_rate', 0) > 0.5),
        ]

    for name, value, fmt, is_positive in key_metrics:
        card_class = 'positive' if is_positive else 'negative'
        if name == 'Max Drawdown':
            card_class = 'negative'
        html_parts.append(f'''
        <div class="metric-card {card_class}">
            <div class="metric-value">{_format_metric(value, fmt)}</div>
            <div class="metric-label">{name}</div>
        </div>
        ''')

    html_parts.append('</div>')
    html_parts.append('</div>')

    # Charts Section
    if charts:
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Performance Charts</h2>')

        if 'equity_curve' in charts:
            html_parts.append('<div class="chart-container">')
            html_parts.append(f'<img src="data:image/png;base64,{charts["equity_curve"]}" alt="Equity Curve">')
            html_parts.append('</div>')

        html_parts.append('<div class="two-column">')

        if 'drawdown' in charts:
            html_parts.append(f'''
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['drawdown']}" alt="Drawdown">
            </div>
            ''')

        if 'returns_dist' in charts:
            html_parts.append(f'''
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['returns_dist']}" alt="Returns Distribution">
            </div>
            ''')

        if 'confusion_matrix' in charts:
            html_parts.append(f'''
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['confusion_matrix']}" alt="Confusion Matrix">
            </div>
            ''')

        if 'win_rate_ci' in charts:
            html_parts.append(f'''
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['win_rate_ci']}" alt="Win Rate CI">
            </div>
            ''')

        if 'cumulative_return' in charts:
            html_parts.append(f'''
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['cumulative_return']}" alt="Cumulative Return">
            </div>
            ''')

        html_parts.append('</div>')

        if 'monthly_returns' in charts:
            html_parts.append(f'''
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['monthly_returns']}" alt="Monthly Returns">
            </div>
            ''')

        if 'rolling_metrics' in charts:
            html_parts.append(f'''
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['rolling_metrics']}" alt="Rolling Metrics">
            </div>
            ''')

        html_parts.append('</div>')

    # Detailed Metrics
    html_parts.append('<div class="section">')
    html_parts.append('<h2>Detailed Metrics</h2>')
    html_parts.append('<div class="two-column">')

    # Prediction Metrics
    html_parts.append('<div>')
    html_parts.append('<h3>Prediction Performance</h3>')
    html_parts.append('<table>')
    html_parts.append('<tr><th>Metric</th><th>Value</th></tr>')

    pred_metrics = [
        ('Accuracy', 'accuracy', 'percent'),
        ('Precision', 'precision', 'percent'),
        ('Recall', 'recall', 'percent'),
        ('F1 Score', 'f1_score', 'ratio'),
        ('True Positives', 'true_positives', 'integer'),
        ('True Negatives', 'true_negatives', 'integer'),
        ('False Positives', 'false_positives', 'integer'),
        ('False Negatives', 'false_negatives', 'integer'),
    ]

    for label, key, fmt in pred_metrics:
        if key in metrics:
            html_parts.append(f'<tr><td>{label}</td><td>{_format_metric(metrics[key], fmt)}</td></tr>')

    html_parts.append('</table>')
    html_parts.append('</div>')

    # Trading Metrics
    html_parts.append('<div>')
    html_parts.append('<h3>Trading Performance</h3>')
    html_parts.append('<table>')
    html_parts.append('<tr><th>Metric</th><th>Value</th></tr>')

    trade_metrics = [
        ('Total Return', 'total_return', 'percent'),
        ('Annualized Return', 'annualized_return', 'percent'),
        ('Sharpe Ratio', 'sharpe_ratio', 'ratio'),
        ('Sortino Ratio', 'sortino_ratio', 'ratio'),
        ('Max Drawdown', 'max_drawdown', 'percent'),
        ('Win Rate', 'win_rate', 'percent'),
        ('Profit Factor', 'profit_factor', 'ratio'),
        ('Avg Win', 'avg_win', 'percent'),
        ('Avg Loss', 'avg_loss', 'percent'),
        ('Number of Trades', 'n_trades', 'integer'),
    ]

    for label, key, fmt in trade_metrics:
        if key in metrics:
            html_parts.append(f'<tr><td>{label}</td><td>{_format_metric(metrics[key], fmt)}</td></tr>')

    html_parts.append('</table>')
    html_parts.append('</div>')
    html_parts.append('</div>')
    html_parts.append('</div>')

    # Trade Log (for standard backtest)
    if include_trades and not is_randomized and hasattr(result, 'trades') and result.trades:
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Trade Log</h2>')
        html_parts.append('<table>')
        html_parts.append('''
        <tr>
            <th>#</th>
            <th>Entry Date</th>
            <th>Exit Date</th>
            <th>Signal</th>
            <th>Entry Price</th>
            <th>Exit Price</th>
            <th>Return</th>
            <th>Correct</th>
        </tr>
        ''')

        for i, trade in enumerate(result.trades[:50], 1):  # Limit to 50 trades
            if isinstance(trade, dict):
                return_pct = trade.get('return_pct', 0)
                is_correct = trade.get('prediction_correct', False)
                entry_date = trade.get('entry_date', 'N/A')
                exit_date = trade.get('exit_date', 'N/A')
                signal = trade.get('entry_signal', 'N/A')
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
            else:
                return_pct = trade.return_pct
                is_correct = trade.prediction_correct
                entry_date = trade.entry_date
                exit_date = trade.exit_date
                signal = trade.entry_signal
                entry_price = trade.entry_price
                exit_price = trade.exit_price

            return_class = 'positive' if return_pct > 0 else 'negative'
            correct_icon = '✓' if is_correct else '✗'
            correct_class = 'positive' if is_correct else 'negative'

            # Format dates
            if hasattr(entry_date, 'strftime'):
                entry_date = entry_date.strftime('%Y-%m-%d')
            if hasattr(exit_date, 'strftime'):
                exit_date = exit_date.strftime('%Y-%m-%d')

            html_parts.append(f'''
            <tr>
                <td>{i}</td>
                <td>{entry_date}</td>
                <td>{exit_date}</td>
                <td>{signal}</td>
                <td>${entry_price:.2f}</td>
                <td>${exit_price:.2f}</td>
                <td class="{return_class}">{return_pct:.2%}</td>
                <td class="{correct_class}">{correct_icon}</td>
            </tr>
            ''')

        if len(result.trades) > 50:
            html_parts.append(f'<tr><td colspan="8">... and {len(result.trades) - 50} more trades</td></tr>')

        html_parts.append('</table>')
        html_parts.append('</div>')

    # Footer
    html_parts.append('''
    <div class="footer">
        <p>Generated by CNN Stock Market Prediction - Backtesting Framework</p>
        <p>Disclaimer: Past performance does not guarantee future results. This report is for educational purposes only.</p>
    </div>
    ''')

    html_parts.extend(['</body>', '</html>'])

    return '\n'.join(html_parts)


def generate_report(
    result: Union[BacktestResult, RandomizedBacktestResult],
    output_path: Union[str, Path],
    format: str = 'html',
) -> Path:
    """
    Generate backtest report.

    Args:
        result: BacktestResult or RandomizedBacktestResult
        output_path: Path to save report
        format: Output format ('html')

    Returns:
        Path to generated report
    """
    output_path = Path(output_path)

    if format == 'html':
        return generate_html_report(result, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Only 'html' is currently supported.")


def export_trades_csv(
    trades: List[Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Export trade log to CSV.

    Args:
        trades: List of Trade objects or dicts
        output_path: Path to save CSV

    Returns:
        Path to CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not trades:
        pd.DataFrame().to_csv(output_path, index=False)
        return output_path

    # Convert to list of dicts
    if hasattr(trades[0], 'to_dict'):
        trade_dicts = [t.to_dict() for t in trades]
    else:
        trade_dicts = trades

    df = pd.DataFrame(trade_dicts)

    # Format dates
    for col in ['entry_date', 'exit_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

    df.to_csv(output_path, index=False)
    return output_path


def export_metrics_json(
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Export metrics to JSON.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON

    Returns:
        Path to JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    metrics_converted = convert(metrics)

    with open(output_path, 'w') as f:
        json.dump(metrics_converted, f, indent=2, default=str)

    return output_path


def export_predictions_csv(
    predictions: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """
    Export predictions to CSV.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save CSV

    Returns:
        Path to CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not predictions:
        pd.DataFrame().to_csv(output_path, index=False)
        return output_path

    df = pd.DataFrame(predictions)

    # Format dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    df.to_csv(output_path, index=False)
    return output_path
