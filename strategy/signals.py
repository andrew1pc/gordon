"""Signal generation for momentum trading strategy."""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents an open trading position.

    Attributes:
        ticker: Asset ticker symbol
        asset_type: 'stock' or 'crypto'
        entry_date: Date position was entered
        entry_price: Price at entry
        position_size: Number of shares/units (optional)
        initial_stop: Initial stop loss price
        trailing_stop: Trailing stop price (None until activated)
        highest_price: Highest price since entry (for trailing stop)
        target_price: Profit target price
        days_held: Number of days position has been open
    """
    ticker: str
    asset_type: str
    entry_date: datetime
    entry_price: float
    position_size: float = 1.0
    initial_stop: float = 0.0
    trailing_stop: Optional[float] = None
    highest_price: float = 0.0
    target_price: float = 0.0
    days_held: int = 0

    def __post_init__(self):
        """Initialize highest_price to entry_price."""
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price


class SignalGenerator:
    """
    Generates entry and exit signals for momentum trading.

    This class implements the complete signal generation logic including:
    - Entry signal detection (breakouts with volume and momentum)
    - Exit signal detection (stops, targets, trailing stops, momentum failure)
    - Position management
    """

    def __init__(self, max_holding_days: int = 30):
        """
        Initialize SignalGenerator.

        Args:
            max_holding_days: Maximum days to hold a position
        """
        self.max_holding_days = max_holding_days

    def check_entry_signals(
        self,
        df: pd.DataFrame,
        current_idx: int,
        ticker: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if entry conditions are met at current index.

        Entry Criteria (ALL must be met):
        1. Price breaks above 20-day high
        2. Volume surge (>= 1.5x average)
        3. MACD histogram > 0
        4. Price above 50-day MA
        5. 50-day MA trending up
        6. Momentum score >= 70

        Args:
            df: DataFrame with all indicators calculated
            current_idx: Current bar index to check
            ticker: Ticker symbol

        Returns:
            Signal dictionary if conditions met, None otherwise

        Example:
            >>> signal = generator.check_entry_signals(df, idx, 'AAPL')
            >>> if signal:
            ...     print(f"Entry signal: {signal['entry_price']}")
        """
        if current_idx < 20:
            return None  # Need 20 days of history

        try:
            current = df.iloc[current_idx]
            prev_20_highs = df['high'].iloc[current_idx-20:current_idx]

            # Condition 1: Price breaks above 20-day high
            breakout = current['close'] > prev_20_highs.max()

            # Condition 2: Volume surge
            volume_surge = current.get('volume_surge', False)
            if not volume_surge and 'volume_ratio' in current:
                volume_surge = current['volume_ratio'] >= 1.5

            # Condition 3: MACD confirmation
            macd_positive = False
            if 'macd_histogram' in current:
                macd_positive = current['macd_histogram'] > 0

            # Condition 4: Price above 50-day MA
            above_ma50 = False
            if 'sma_50' in current:
                above_ma50 = current['close'] > current['sma_50']

            # Condition 5: 50-day MA trending up
            ma_trending_up = False
            if 'sma_50' in current and current_idx >= 25:
                ma50_current = current['sma_50']
                ma50_5d_ago = df['sma_50'].iloc[current_idx - 5]
                ma_trending_up = ma50_current > ma50_5d_ago

            # Condition 6: Strong momentum
            strong_momentum = False
            if 'momentum_score' in current:
                strong_momentum = current['momentum_score'] >= 70

            # Check all conditions
            conditions_met = []
            if breakout:
                conditions_met.append('breakout_20d_high')
            if volume_surge:
                conditions_met.append('volume_surge')
            if macd_positive:
                conditions_met.append('macd_positive')
            if above_ma50:
                conditions_met.append('above_ma50')
            if ma_trending_up:
                conditions_met.append('ma50_trending_up')
            if strong_momentum:
                conditions_met.append('strong_momentum')

            # All conditions must be met
            all_met = (breakout and volume_surge and macd_positive and
                      above_ma50 and ma_trending_up and strong_momentum)

            if not all_met:
                return None

            # Calculate signal strength (0-100)
            signal_strength = min(100, current.get('momentum_score', 70))

            # Generate signal
            signal = {
                'signal': True,
                'ticker': ticker,
                'date': df.index[current_idx],
                'entry_price': self.calculate_entry_price(df, current_idx, 'stock'),
                'signal_strength': signal_strength,
                'conditions_met': conditions_met,
                'current_price': current['close']
            }

            logger.info(f"Entry signal for {ticker} on {signal['date'].date()}")
            return signal

        except Exception as e:
            logger.error(f"Error checking entry signal for {ticker}: {e}")
            return None

    def calculate_entry_price(
        self,
        df: pd.DataFrame,
        signal_idx: int,
        asset_type: str
    ) -> float:
        """
        Calculate realistic entry price with slippage.

        In backtesting, we enter at next bar's open + slippage.

        Args:
            df: DataFrame with price data
            signal_idx: Index where signal occurred
            asset_type: 'stock' or 'crypto'

        Returns:
            Entry price with slippage

        Example:
            >>> entry = generator.calculate_entry_price(df, idx, 'stock')
        """
        # Can't enter until next bar
        if signal_idx + 1 >= len(df):
            # Use current close if no next bar
            return df['close'].iloc[signal_idx]

        next_open = df['open'].iloc[signal_idx + 1]

        # Apply slippage
        if asset_type == 'stock':
            slippage = 0.001  # 0.1%
        else:  # crypto
            slippage = 0.003  # 0.3%

        entry_price = next_open * (1 + slippage)

        logger.debug(f"Entry price: {entry_price:.2f} (next open: {next_open:.2f}, slippage: {slippage*100}%)")
        return entry_price

    def calculate_initial_stop(
        self,
        entry_price: float,
        df: pd.DataFrame,
        entry_idx: int,
        asset_type: str
    ) -> float:
        """
        Calculate initial stop loss price.

        Uses both ATR-based and percentage-based stops, choosing the wider one.

        Args:
            entry_price: Entry price
            df: DataFrame at entry
            entry_idx: Entry index
            asset_type: 'stock' or 'crypto'

        Returns:
            Initial stop loss price

        Example:
            >>> stop = generator.calculate_initial_stop(100.0, df, idx, 'stock')
        """
        # ATR-based stop (2.5 × ATR)
        atr_stop = None
        if 'atr' in df.columns:
            atr = df['atr'].iloc[entry_idx]
            if not pd.isna(atr):
                atr_stop = entry_price - (2.5 * atr)

        # Percentage-based stop
        if asset_type == 'stock':
            pct_stop = entry_price * 0.93  # 7% stop
        else:  # crypto
            pct_stop = entry_price * 0.88  # 12% stop

        # Use wider (more forgiving) stop
        if atr_stop is not None:
            stop = min(atr_stop, pct_stop)  # Min because lower price = wider stop
        else:
            stop = pct_stop

        # Ensure stop is below recent 20-day low
        if entry_idx >= 20:
            recent_low = df['low'].iloc[entry_idx-20:entry_idx].min()
            if stop > recent_low:
                stop = recent_low * 0.99  # Slightly below low

        logger.debug(f"Initial stop: {stop:.2f} (entry: {entry_price:.2f}, {((entry_price-stop)/entry_price*100):.1f}%)")
        return stop

    def calculate_profit_target(
        self,
        entry_price: float,
        df: pd.DataFrame,
        entry_idx: int,
        asset_type: str
    ) -> float:
        """
        Calculate profit target based on volatility.

        Args:
            entry_price: Entry price
            df: DataFrame at entry
            entry_idx: Entry index
            asset_type: 'stock' or 'crypto'

        Returns:
            Profit target price

        Example:
            >>> target = generator.calculate_profit_target(100.0, df, idx, 'stock')
        """
        # Get ATR for volatility adjustment
        atr_ratio = 1.0
        if 'atr' in df.columns and entry_idx > 0:
            atr = df['atr'].iloc[entry_idx]
            price = df['close'].iloc[entry_idx]
            if not pd.isna(atr) and price > 0:
                atr_ratio = (atr / price) * 100  # ATR as % of price

        # Scale target based on volatility
        if asset_type == 'stock':
            # Stocks: 15-25% target
            if atr_ratio > 3:  # High volatility
                target_pct = 0.25
            else:
                target_pct = 0.15
        else:  # crypto
            # Crypto: 25-40% target
            if atr_ratio > 5:  # High volatility
                target_pct = 0.40
            else:
                target_pct = 0.25

        target = entry_price * (1 + target_pct)

        logger.debug(f"Profit target: {target:.2f} (entry: {entry_price:.2f}, +{target_pct*100:.0f}%)")
        return target

    def check_exit_signals(
        self,
        position: Position,
        df: pd.DataFrame,
        current_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Check all exit conditions for a position.

        Priority order:
        1. Stop loss
        2. Profit target
        3. Trailing stop
        4. Momentum failure
        5. Time exit

        Args:
            position: Open position
            df: DataFrame with current data
            current_idx: Current bar index

        Returns:
            Exit signal dictionary if exit triggered, None otherwise

        Example:
            >>> exit_signal = generator.check_exit_signals(position, df, idx)
            >>> if exit_signal:
            ...     print(f"Exit: {exit_signal['exit_type']}")
        """
        current = df.iloc[current_idx]

        # Update position days held
        position.days_held = (df.index[current_idx] - position.entry_date).days

        # 1. Check stop loss (highest priority)
        stop_signal = self.check_stop_loss(position, current)
        if stop_signal:
            return stop_signal

        # 2. Check profit target
        target_signal = self.check_profit_target(position, current)
        if target_signal:
            return target_signal

        # Update trailing stop
        self.update_trailing_stop(position, current)

        # 3. Check trailing stop
        trailing_signal = self.check_trailing_stop(position, current)
        if trailing_signal:
            return trailing_signal

        # 4. Check momentum failure
        momentum_signal = self.check_momentum_failure(position, df, current_idx)
        if momentum_signal:
            return momentum_signal

        # 5. Check time exit
        time_signal = self.check_time_exit(position, current)
        if time_signal:
            return time_signal

        return None

    def check_stop_loss(self, position: Position, current: pd.Series) -> Optional[Dict]:
        """Check if stop loss triggered."""
        stop = position.trailing_stop if position.trailing_stop else position.initial_stop

        if current['low'] <= stop:
            pnl = (stop - position.entry_price) / position.entry_price
            return {
                'exit_type': 'stop_loss',
                'exit_price': stop,
                'exit_date': current.name,
                'pnl': pnl,
                'days_held': position.days_held
            }
        return None

    def check_profit_target(self, position: Position, current: pd.Series) -> Optional[Dict]:
        """Check if profit target reached."""
        if current['high'] >= position.target_price:
            pnl = (position.target_price - position.entry_price) / position.entry_price
            return {
                'exit_type': 'profit_target',
                'exit_price': position.target_price,
                'exit_date': current.name,
                'pnl': pnl,
                'days_held': position.days_held
            }
        return None

    def update_trailing_stop(self, position: Position, current: pd.Series) -> None:
        """Update trailing stop if conditions met."""
        # Update highest price
        if current['high'] > position.highest_price:
            position.highest_price = current['high']

        # Activate trailing stop at 10% profit
        profit_pct = (position.highest_price - position.entry_price) / position.entry_price

        if profit_pct >= 0.10:
            # Set trailing stop at 8% below highest
            new_trailing = position.highest_price * 0.92

            # Only move trailing stop up, never down
            if position.trailing_stop is None or new_trailing > position.trailing_stop:
                position.trailing_stop = new_trailing

    def check_trailing_stop(self, position: Position, current: pd.Series) -> Optional[Dict]:
        """Check if trailing stop triggered."""
        if position.trailing_stop and current['low'] <= position.trailing_stop:
            pnl = (position.trailing_stop - position.entry_price) / position.entry_price
            return {
                'exit_type': 'trailing_stop',
                'exit_price': position.trailing_stop,
                'exit_date': current.name,
                'pnl': pnl,
                'days_held': position.days_held
            }
        return None

    def check_momentum_failure(
        self,
        position: Position,
        df: pd.DataFrame,
        current_idx: int
    ) -> Optional[Dict]:
        """Check if momentum has failed."""
        current = df.iloc[current_idx]

        # Price below 20-day MA on high volume
        below_ma20 = False
        if 'sma_20' in current:
            below_ma20 = current['close'] < current['sma_20']

        high_volume = current.get('volume_ratio', 0) > 1.5

        if not (below_ma20 and high_volume):
            return None

        # Check additional momentum deterioration
        macd_negative = current.get('macd_histogram', 0) < 0
        rsi_weak = current.get('rsi', 50) < 40
        momentum_weak = current.get('momentum_score', 50) < 50

        if macd_negative or rsi_weak or momentum_weak:
            pnl = (current['close'] - position.entry_price) / position.entry_price
            return {
                'exit_type': 'momentum_failure',
                'exit_price': current['close'],
                'exit_date': current.name,
                'pnl': pnl,
                'days_held': position.days_held
            }

        return None

    def check_time_exit(self, position: Position, current: pd.Series) -> Optional[Dict]:
        """Check if maximum holding period exceeded."""
        if position.days_held >= self.max_holding_days:
            pnl = (current['close'] - position.entry_price) / position.entry_price
            return {
                'exit_type': 'time_exit',
                'exit_price': current['close'],
                'exit_date': current.name,
                'pnl': pnl,
                'days_held': position.days_held
            }
        return None


class SignalTracker:
    """Tracks all signals for analysis and reporting."""

    def __init__(self):
        """Initialize signal tracker."""
        self.entry_signals: List[Dict] = []
        self.exit_signals: List[Dict] = []
        self.active_positions: Dict[str, Position] = {}

    def add_entry_signal(self, signal: Dict) -> None:
        """Record entry signal."""
        self.entry_signals.append(signal)
        logger.info(f"Entry signal recorded for {signal.get('ticker')}")

    def add_exit_signal(self, signal: Dict) -> None:
        """Record exit signal."""
        self.exit_signals.append(signal)
        logger.info(f"Exit signal recorded: {signal.get('exit_type')}")

    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of signals."""
        stats = {
            'total_entries': len(self.entry_signals),
            'total_exits': len(self.exit_signals),
            'active_positions': len(self.active_positions)
        }

        # Exit type breakdown
        if self.exit_signals:
            exit_types = {}
            for exit_sig in self.exit_signals:
                exit_type = exit_sig.get('exit_type', 'unknown')
                exit_types[exit_type] = exit_types.get(exit_type, 0) + 1

            stats['exit_breakdown'] = exit_types

            # Calculate win rate
            wins = sum(1 for sig in self.exit_signals if sig.get('pnl', 0) > 0)
            stats['win_rate'] = wins / len(self.exit_signals) if self.exit_signals else 0

        return stats

    def export_to_csv(self, filepath: str) -> None:
        """Export signals to CSV file."""
        if self.entry_signals:
            df_entries = pd.DataFrame(self.entry_signals)
            df_entries.to_csv(filepath.replace('.csv', '_entries.csv'), index=False)

        if self.exit_signals:
            df_exits = pd.DataFrame(self.exit_signals)
            df_exits.to_csv(filepath.replace('.csv', '_exits.csv'), index=False)

        logger.info(f"Signals exported to {filepath}")
