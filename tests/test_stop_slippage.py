"""Unit tests for stop loss slippage (Iteration 7)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.signals import SignalGenerator, Position


@pytest.fixture
def signal_generator():
    """Create SignalGenerator instance."""
    return SignalGenerator()


@pytest.fixture
def stock_position():
    """Create a stock position."""
    return Position(
        ticker='AAPL',
        asset_type='stock',
        entry_date=datetime(2025, 1, 1),
        entry_price=100.0,
        initial_stop=90.0,
        target_price=125.0
    )


@pytest.fixture
def crypto_position():
    """Create a crypto position."""
    return Position(
        ticker='btcusd',
        asset_type='crypto',
        entry_date=datetime(2025, 1, 1),
        entry_price=50000.0,
        initial_stop=45000.0,
        target_price=60000.0
    )


class TestStockStopSlippage:
    """Test stop loss slippage for stocks (0.5% base)."""

    def test_normal_stop_has_0_5_percent_slippage(self, signal_generator, stock_position):
        """Test that normal stock stop has 0.5% base slippage."""
        # Create current bar where stop is hit
        current = pd.Series({
            'open': 91.0,
            'high': 92.0,
            'low': 89.0,  # Below $90 stop
            'close': 89.5
        }, name=datetime(2025, 1, 5))

        # No gap (open close to previous close)
        previous = pd.Series({
            'close': 91.0
        })

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous)

        assert exit_signal is not None
        assert exit_signal['stop_price'] == 90.0

        # Expected: $90 * (1 - 0.005) = $89.55
        expected_exit = 90.0 * 0.995
        # But floored at daily low of $89
        expected_exit = max(expected_exit, 89.0)

        assert abs(exit_signal['exit_price'] - expected_exit) < 0.01
        assert exit_signal['gap_detected'] is False

    def test_stock_slippage_floored_at_daily_low(self, signal_generator, stock_position):
        """Test that exit price cannot go below daily low."""
        # Create scenario where slippage would take us below daily low
        current = pd.Series({
            'open': 91.0,
            'high': 92.0,
            'low': 89.80,  # Low is above slippage exit
            'close': 90.0
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 91.0})

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous)

        # Slippage would be $90 * 0.995 = $89.55
        # But should floor at daily low of $89.80
        assert exit_signal['exit_price'] == 89.80

    def test_stock_stop_with_gap_down(self, signal_generator, stock_position):
        """Test that gap down adds 0.5% extra slippage."""
        # Create gap down scenario
        current = pd.Series({
            'open': 88.0,  # Gap down from 91
            'high': 89.0,
            'low': 87.0,
            'close': 87.5
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({
            'close': 91.0  # Previous close
        })

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous)

        # Gap = (91 - 88) / 91 = 3.3% > 0.5% threshold
        # Should detect gap
        assert exit_signal['gap_detected'] is True

        # Expected: $90 * (1 - 0.005 - 0.005) = $90 * 0.99 = $89.10
        # Floored at $87
        expected_exit = max(90.0 * 0.99, 87.0)
        assert abs(exit_signal['exit_price'] - expected_exit) < 0.01


class TestCryptoStopSlippage:
    """Test stop loss slippage for crypto (1.0% base)."""

    def test_crypto_stop_has_1_percent_slippage(self, signal_generator, crypto_position):
        """Test that crypto stop has 1.0% base slippage."""
        # Create current bar where stop is hit
        current = pd.Series({
            'open': 45500.0,
            'high': 46000.0,
            'low': 44000.0,  # Below $45k stop
            'close': 44500.0
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 45500.0})

        exit_signal = signal_generator.check_stop_loss(crypto_position, current, previous)

        assert exit_signal is not None

        # Expected: $45000 * (1 - 0.01) = $44,550
        # Floored at daily low of $44,000
        expected_exit = max(45000.0 * 0.99, 44000.0)
        assert abs(exit_signal['exit_price'] - expected_exit) < 1.0

    def test_crypto_stop_with_gap_down(self, signal_generator, crypto_position):
        """Test that gap down adds 0.5% extra slippage for crypto."""
        # Create gap down scenario
        current = pd.Series({
            'open': 44000.0,  # Gap down from 46000
            'high': 44500.0,
            'low': 43000.0,
            'close': 43500.0
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 46000.0})

        exit_signal = signal_generator.check_stop_loss(crypto_position, current, previous)

        # Gap detected
        assert exit_signal['gap_detected'] is True

        # Expected: $45000 * (1 - 0.01 - 0.005) = $45000 * 0.985 = $44,325
        # Floored at $43,000
        expected_exit = max(45000.0 * 0.985, 43000.0)
        assert abs(exit_signal['exit_price'] - expected_exit) < 1.0


class TestGapDetection:
    """Test gap down detection logic."""

    def test_no_gap_when_open_near_close(self, signal_generator, stock_position):
        """Test that small moves are not detected as gaps."""
        current = pd.Series({
            'open': 90.3,  # Close to previous close
            'high': 91.0,
            'low': 89.0,
            'close': 89.5
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 90.5})

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous)

        # Gap = (90.5 - 90.3) / 90.5 = 0.22% < 0.5% threshold
        # Should NOT detect gap
        if exit_signal:
            assert exit_signal['gap_detected'] is False

    def test_gap_detected_when_open_below_close(self, signal_generator, stock_position):
        """Test that meaningful gaps are detected."""
        current = pd.Series({
            'open': 89.0,  # Significant gap from 91
            'high': 90.0,
            'low': 88.0,
            'close': 88.5
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 91.0})

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous)

        # Gap = (91 - 89) / 91 = 2.2% > 0.5% threshold
        # Should detect gap
        assert exit_signal['gap_detected'] is True

    def test_no_gap_when_no_previous_bar(self, signal_generator, stock_position):
        """Test that no gap is detected when previous bar is None."""
        current = pd.Series({
            'open': 89.0,
            'high': 90.0,
            'low': 88.0,
            'close': 88.5
        }, name=datetime(2025, 1, 5))

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous=None)

        # Should apply base slippage only (no gap detection)
        if exit_signal:
            assert exit_signal['gap_detected'] is False


class TestSlippageCalculation:
    """Test slippage percentage calculations."""

    def test_slippage_pct_recorded(self, signal_generator, stock_position):
        """Test that actual slippage percentage is recorded."""
        current = pd.Series({
            'open': 91.0,
            'high': 92.0,
            'low': 89.0,
            'close': 89.5
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 91.0})

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous)

        # Should have slippage_pct field
        assert 'slippage_pct' in exit_signal
        assert exit_signal['slippage_pct'] > 0

        # Actual slippage = (stop - exit) / stop
        expected_slippage = (90.0 - exit_signal['exit_price']) / 90.0
        assert abs(exit_signal['slippage_pct'] - expected_slippage) < 0.001

    def test_combined_slippage_stock_with_gap(self, signal_generator, stock_position):
        """Test combined slippage for stock with gap (0.5% + 0.5% = 1.0%)."""
        current = pd.Series({
            'open': 88.0,  # Gap down
            'high': 89.0,
            'low': 88.0,
            'close': 88.5
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 91.0})

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous)

        # Expected exit: $90 * (1 - 0.01) = $89.10
        # Slippage price is above daily low, so no flooring occurs
        assert abs(exit_signal['exit_price'] - 89.10) < 0.01
        assert exit_signal['gap_detected'] is True

    def test_combined_slippage_crypto_with_gap(self, signal_generator, crypto_position):
        """Test combined slippage for crypto with gap (1.0% + 0.5% = 1.5%)."""
        current = pd.Series({
            'open': 43000.0,  # Gap down
            'high': 44000.0,
            'low': 43000.0,
            'close': 43500.0
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 46000.0})

        exit_signal = signal_generator.check_stop_loss(crypto_position, current, previous)

        # Expected: $45000 * (1 - 0.015) = $44,325
        # Slippage price is above daily low, so no flooring occurs
        assert abs(exit_signal['exit_price'] - 44325.0) < 1.0
        assert exit_signal['gap_detected'] is True


class TestStopNotTriggered:
    """Test cases where stop is not triggered."""

    def test_no_exit_when_stop_not_hit(self, signal_generator, stock_position):
        """Test that no exit when price doesn't hit stop."""
        current = pd.Series({
            'open': 95.0,
            'high': 96.0,
            'low': 94.0,  # Above $90 stop
            'close': 95.0
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 95.0})

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous)

        assert exit_signal is None


class TestTrailingStopSlippage:
    """Test slippage with trailing stops."""

    def test_trailing_stop_has_slippage(self, signal_generator, stock_position):
        """Test that trailing stops also have slippage."""
        # Set trailing stop
        stock_position.trailing_stop = 95.0
        stock_position.highest_price = 110.0

        current = pd.Series({
            'open': 96.0,
            'high': 96.5,
            'low': 94.0,  # Below trailing stop
            'close': 94.5
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 96.0})

        exit_signal = signal_generator.check_stop_loss(stock_position, current, previous)

        # Should use trailing stop ($95) with slippage
        assert exit_signal['stop_price'] == 95.0

        # Expected: $95 * 0.995 = $94.525, floored at $94
        expected_exit = max(95.0 * 0.995, 94.0)
        assert abs(exit_signal['exit_price'] - expected_exit) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
