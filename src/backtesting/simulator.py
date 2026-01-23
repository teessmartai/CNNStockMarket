"""
Trade Simulator (TASK-7.3)

Simulates trades based on model signals with realistic cost modeling.
Supports multiple position sizing strategies and long/short trading.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd


class PositionType(Enum):
    """Type of position."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionSizing(Enum):
    """Position sizing methods."""
    FIXED = "fixed"  # Fixed number of shares
    PERCENTAGE = "percentage"  # Percentage of equity
    KELLY = "kelly"  # Kelly criterion


@dataclass
class SimulatorConfig:
    """Configuration for trade simulation."""

    # Capital
    initial_capital: float = 100000.0

    # Costs
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage

    # Strategy
    confidence_threshold: float = 0.5  # Minimum confidence to trade
    allow_short: bool = False  # Long-only by default

    # Position sizing
    position_sizing: str = 'fixed'  # 'fixed', 'percentage', 'kelly'
    position_size: float = 1.0  # For fixed sizing (fraction of capital)
    position_pct: float = 0.1  # For percentage sizing (10% of equity)
    max_position_pct: float = 0.25  # Maximum position size (25% of equity)

    # Trade management
    horizon: int = 5  # Hold period
    allow_pyramiding: bool = False  # Add to existing positions


@dataclass
class Position:
    """Represents an open position."""

    ticker: str
    position_type: PositionType
    entry_date: datetime
    entry_price: float
    shares: float
    entry_value: float
    signal_confidence: float

    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade."""

    ticker: str
    position_type: str  # 'long' or 'short'

    # Entry
    entry_date: datetime
    entry_price: float
    entry_signal: str
    entry_confidence: float

    # Exit
    exit_date: datetime
    exit_price: float
    exit_reason: str  # 'horizon', 'signal_change', 'stop_loss'

    # Position
    shares: float
    entry_value: float
    exit_value: float

    # P&L
    gross_pnl: float
    commission_paid: float
    slippage_cost: float
    net_pnl: float
    return_pct: float

    # Prediction outcome
    predicted_direction: int  # 1=bullish, 0=bearish
    actual_direction: int  # 1=bullish, 0=bearish
    prediction_correct: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'ticker': self.ticker,
            'position_type': self.position_type,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'entry_signal': self.entry_signal,
            'entry_confidence': self.entry_confidence,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'shares': self.shares,
            'entry_value': self.entry_value,
            'exit_value': self.exit_value,
            'gross_pnl': self.gross_pnl,
            'commission_paid': self.commission_paid,
            'slippage_cost': self.slippage_cost,
            'net_pnl': self.net_pnl,
            'return_pct': self.return_pct,
            'predicted_direction': self.predicted_direction,
            'actual_direction': self.actual_direction,
            'prediction_correct': self.prediction_correct,
        }


class TradeSimulator:
    """
    Simulates trading based on model predictions.

    Handles position management, cost calculation, and equity tracking.
    """

    def __init__(self, config: SimulatorConfig):
        """
        Initialize the simulator.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.reset()

    def reset(self):
        """Reset simulator state."""
        self.capital = self.config.initial_capital
        self.equity = self.config.initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []

    def simulate(
        self,
        predictions: List[Dict[str, Any]],
        price_data: pd.DataFrame,
    ) -> Tuple[List[Dict[str, Any]], pd.Series]:
        """
        Simulate trading based on predictions.

        Args:
            predictions: List of prediction dictionaries with:
                - date: Prediction date
                - signal: 'BUY' or 'SELL'
                - confidence: Signal confidence
                - predicted_direction: 1 or 0
                - actual_direction: 1 or 0
                - price: Price at signal
            price_data: DataFrame with price history

        Returns:
            Tuple of (list of trade dicts, equity curve Series)
        """
        self.reset()

        if not predictions:
            return [], pd.Series(dtype=float)

        # Sort predictions by date
        sorted_preds = sorted(predictions, key=lambda x: x['date'])

        # Track equity at each prediction
        bars_since_entry = 0

        for pred in sorted_preds:
            current_date = pred['date']
            current_price = pred['price']
            signal = pred['signal']
            confidence = pred['confidence']
            predicted_direction = pred['predicted_direction']
            actual_direction = pred.get('actual_direction', -1)

            # Update position value if we have one
            if self.position is not None:
                self.position.current_price = current_price
                if self.position.position_type == PositionType.LONG:
                    self.position.unrealized_pnl = (
                        (current_price - self.position.entry_price) * self.position.shares
                    )
                else:  # Short
                    self.position.unrealized_pnl = (
                        (self.position.entry_price - current_price) * self.position.shares
                    )
                self.position.unrealized_pnl_pct = (
                    self.position.unrealized_pnl / self.position.entry_value
                )
                bars_since_entry += 1

            # Check if we should close position (horizon reached)
            if self.position is not None and bars_since_entry >= self.config.horizon:
                self._close_position(current_date, current_price, 'horizon', actual_direction)
                bars_since_entry = 0

            # Check if we should open a new position
            if self.position is None and confidence >= self.config.confidence_threshold:
                if signal == 'BUY':
                    self._open_position(
                        current_date, current_price, PositionType.LONG,
                        signal, confidence, predicted_direction
                    )
                    bars_since_entry = 0
                elif signal == 'SELL' and self.config.allow_short:
                    self._open_position(
                        current_date, current_price, PositionType.SHORT,
                        signal, confidence, predicted_direction
                    )
                    bars_since_entry = 0

            # Record equity
            current_equity = self._calculate_equity(current_price)
            self.equity_history.append((current_date, current_equity))

        # Close any remaining position at the end
        if self.position is not None and len(sorted_preds) > 0:
            last_pred = sorted_preds[-1]
            self._close_position(
                last_pred['date'], last_pred['price'], 'end_of_backtest',
                last_pred.get('actual_direction', -1)
            )

        # Create equity curve
        if self.equity_history:
            dates, values = zip(*self.equity_history)
            equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))
        else:
            equity_curve = pd.Series(dtype=float)

        # Convert trades to dicts
        trade_dicts = [t.to_dict() for t in self.trades]

        return trade_dicts, equity_curve

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on sizing method."""
        if self.config.position_sizing == 'fixed':
            # Use fixed fraction of initial capital
            position_value = self.config.initial_capital * self.config.position_size
        elif self.config.position_sizing == 'percentage':
            # Use percentage of current equity
            position_value = self.equity * self.config.position_pct
        elif self.config.position_sizing == 'kelly':
            # Kelly criterion (simplified)
            # Requires win rate and win/loss ratio estimates
            # For simplicity, use a fractional Kelly (half Kelly)
            kelly_pct = 0.1  # Conservative default
            position_value = self.equity * kelly_pct
        else:
            position_value = self.equity * self.config.position_pct

        # Apply maximum position limit
        max_position = self.equity * self.config.max_position_pct
        position_value = min(position_value, max_position)

        # Ensure we don't exceed available capital
        position_value = min(position_value, self.capital)

        return position_value

    def _open_position(
        self,
        date: datetime,
        price: float,
        position_type: PositionType,
        signal: str,
        confidence: float,
        predicted_direction: int,
    ):
        """Open a new position."""
        # Calculate position size
        position_value = self._calculate_position_size(price)

        if position_value <= 0:
            return  # Can't open position

        # Apply slippage to entry price
        if position_type == PositionType.LONG:
            entry_price = price * (1 + self.config.slippage)
        else:
            entry_price = price * (1 - self.config.slippage)

        # Calculate shares
        shares = position_value / entry_price

        # Calculate commission
        commission = position_value * self.config.commission

        # Update capital
        self.capital -= (position_value + commission)

        # Create position
        self.position = Position(
            ticker='',  # Will be set if needed
            position_type=position_type,
            entry_date=date,
            entry_price=entry_price,
            shares=shares,
            entry_value=position_value,
            signal_confidence=confidence,
        )

        # Store entry info for trade record
        self._pending_trade_info = {
            'entry_signal': signal,
            'entry_confidence': confidence,
            'predicted_direction': predicted_direction,
            'commission_entry': commission,
        }

    def _close_position(
        self,
        date: datetime,
        price: float,
        exit_reason: str,
        actual_direction: int,
    ):
        """Close the current position."""
        if self.position is None:
            return

        position = self.position
        pending = self._pending_trade_info

        # Apply slippage to exit price
        if position.position_type == PositionType.LONG:
            exit_price = price * (1 - self.config.slippage)
        else:
            exit_price = price * (1 + self.config.slippage)

        # Calculate exit value
        exit_value = position.shares * exit_price

        # Calculate commission
        commission_exit = exit_value * self.config.commission
        total_commission = pending['commission_entry'] + commission_exit

        # Calculate P&L
        if position.position_type == PositionType.LONG:
            gross_pnl = exit_value - position.entry_value
        else:
            gross_pnl = position.entry_value - exit_value

        # Slippage cost
        slippage_cost = position.entry_value * self.config.slippage * 2  # Entry + exit

        # Net P&L
        net_pnl = gross_pnl - total_commission

        # Return percentage
        return_pct = net_pnl / position.entry_value if position.entry_value > 0 else 0

        # Prediction outcome
        predicted_direction = pending['predicted_direction']
        prediction_correct = predicted_direction == actual_direction if actual_direction >= 0 else None

        # Create trade record
        trade = Trade(
            ticker='',
            position_type=position.position_type.value,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            entry_signal=pending['entry_signal'],
            entry_confidence=pending['entry_confidence'],
            exit_date=date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            shares=position.shares,
            entry_value=position.entry_value,
            exit_value=exit_value,
            gross_pnl=gross_pnl,
            commission_paid=total_commission,
            slippage_cost=slippage_cost,
            net_pnl=net_pnl,
            return_pct=return_pct,
            predicted_direction=predicted_direction,
            actual_direction=actual_direction,
            prediction_correct=prediction_correct,
        )

        self.trades.append(trade)

        # Update capital
        self.capital += (exit_value - commission_exit)

        # Clear position
        self.position = None
        self._pending_trade_info = None

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open position."""
        equity = self.capital

        if self.position is not None:
            # Add current position value
            if self.position.position_type == PositionType.LONG:
                position_value = self.position.shares * current_price
            else:
                # Short position: profit when price goes down
                position_value = (
                    self.position.entry_value +
                    (self.position.entry_price - current_price) * self.position.shares
                )
            equity += position_value

        return equity

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all trades."""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'total_commission': 0.0,
                'total_return': 0.0,
            }

        winning = [t for t in self.trades if t.net_pnl > 0]
        losing = [t for t in self.trades if t.net_pnl <= 0]

        total_pnl = sum(t.net_pnl for t in self.trades)
        total_commission = sum(t.commission_paid for t in self.trades)

        correct_predictions = sum(1 for t in self.trades if t.prediction_correct is True)

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(self.trades) if self.trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(self.trades) if self.trades else 0,
            'total_commission': total_commission,
            'total_return': total_pnl / self.config.initial_capital,
            'avg_winning_trade': np.mean([t.net_pnl for t in winning]) if winning else 0,
            'avg_losing_trade': np.mean([t.net_pnl for t in losing]) if losing else 0,
            'largest_win': max(t.net_pnl for t in winning) if winning else 0,
            'largest_loss': min(t.net_pnl for t in losing) if losing else 0,
            'prediction_accuracy': correct_predictions / len(self.trades) if self.trades else 0,
        }


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Convert list of trades to DataFrame."""
    if not trades:
        return pd.DataFrame()

    return pd.DataFrame([t.to_dict() for t in trades])


def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate Kelly criterion for optimal position sizing.

    Args:
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive number)

    Returns:
        Optimal fraction of capital to risk
    """
    if avg_loss == 0 or avg_win == 0:
        return 0

    # Kelly formula: f* = (bp - q) / b
    # where b = win/loss ratio, p = win rate, q = 1 - p
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b

    # Return half-Kelly for more conservative sizing
    return max(0, kelly * 0.5)
