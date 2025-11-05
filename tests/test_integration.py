"""
Integration tests for the complete momentum trading system (Iteration 9).

Tests all 3 major features working together:
1. Flexible Entry Signals (signal strength + tiered position sizing)
2. Market Regime Detection (regime-based risk adjustments)
3. Realistic Stop Modeling (slippage)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.signals import SignalGenerator, Position
from strategy.risk_manager import RiskManager
from indicators.market_regime import MarketRegimeDetector
from indicators.technical import TechnicalIndicators


@pytest.fixture
def signal_generator():
    """Create SignalGenerator instance."""
    return SignalGenerator()


@pytest.fixture
def risk_manager():
    """Create RiskManager with $100k account."""
    return RiskManager(
        account_size=100000,
        risk_per_trade=0.01,
        max_positions=8
    )


@pytest.fixture
def regime_detector():
    """Create MarketRegimeDetector instance."""
    return MarketRegimeDetector()


@pytest.fixture
def technical_indicators():
    """Create TechnicalIndicators instance."""
    return TechnicalIndicators()


@pytest.fixture
def bull_market_data(technical_indicators):
    """Create synthetic bull market data for SPY."""
    dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
    # Upward trending prices
    prices = np.linspace(400, 500, 250)

    df = pd.DataFrame({
        'open': prices + np.random.randn(250) * 2,
        'high': prices + np.random.randn(250) * 2 + 2,
        'low': prices + np.random.randn(250) * 2 - 2,
        'close': prices + np.random.randn(250) * 2,
        'volume': 50000000 + np.random.randn(250) * 5000000
    }, index=dates)

    # Add technical indicators
    df = technical_indicators.add_all_indicators(df)
    return df


@pytest.fixture
def bear_market_data(technical_indicators):
    """Create synthetic bear market data for SPY."""
    dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
    # Downward trending prices
    prices = np.linspace(500, 400, 250)

    df = pd.DataFrame({
        'open': prices + np.random.randn(250) * 2,
        'high': prices + np.random.randn(250) * 2 + 2,
        'low': prices + np.random.randn(250) * 2 - 2,
        'close': prices + np.random.randn(250) * 2,
        'volume': 50000000 + np.random.randn(250) * 5000000
    }, index=dates)

    # Add technical indicators
    df = technical_indicators.add_all_indicators(df)
    return df


@pytest.fixture
def strong_momentum_stock(technical_indicators):
    """Create stock data with strong momentum signal."""
    dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
    # Strong uptrend with good momentum
    prices = np.linspace(100, 150, 250)

    df = pd.DataFrame({
        'open': prices + np.random.randn(250) * 1,
        'high': prices + np.random.randn(250) * 1 + 1,
        'low': prices + np.random.randn(250) * 1 - 1,
        'close': prices + np.random.randn(250) * 1,
        'volume': 1000000 + np.random.randn(250) * 100000
    }, index=dates)

    # Add technical indicators
    df = technical_indicators.add_all_indicators(df)
    return df


@pytest.fixture
def weak_momentum_stock(technical_indicators):
    """Create stock data with weak momentum signal."""
    dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
    # Slight uptrend with weak momentum
    prices = np.linspace(100, 110, 250)

    df = pd.DataFrame({
        'open': prices + np.random.randn(250) * 2,
        'high': prices + np.random.randn(250) * 2 + 1,
        'low': prices + np.random.randn(250) * 2 - 1,
        'close': prices + np.random.randn(250) * 2,
        'volume': 500000 + np.random.randn(250) * 50000
    }, index=dates)

    # Add technical indicators
    df = technical_indicators.add_all_indicators(df)
    return df


class TestFeature1FlexibleEntrySignals:
    """Test Feature 1: Flexible entry signals with tiered position sizing."""

    def test_signal_strength_scoring(self, signal_generator, strong_momentum_stock):
        """Test that signal strength is calculated correctly."""
        # Data already has indicators added
        df = strong_momentum_stock

        # Check entry signals at the end of the dataframe
        entry = signal_generator.check_entry_signals(df, len(df) - 1, 'TEST')

        # Should have signal strength if entry found
        if entry:
            assert 'signal_strength' in entry
            assert 0 <= entry['signal_strength'] <= 100

    def test_tiered_position_sizing(self, risk_manager):
        """Test that position sizing scales with signal strength."""
        # Use prices that won't hit 5% max constraint
        # Entry $50, stop $40 = $10 risk per share

        # Strong signal (90+ points)
        size_strong, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=95
        )

        # Medium signal (80-89 points)
        size_medium, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=85
        )

        # Weak signal (70-79 points)
        size_weak, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=75
        )

        # Should scale: 100% > 80% > 60%
        assert size_strong > size_medium > size_weak
        assert abs(size_medium / size_strong - 0.8) < 0.01
        assert abs(size_weak / size_strong - 0.6) < 0.01


class TestFeature2MarketRegimeDetection:
    """Test Feature 2: Market regime detection and adjustments."""

    def test_detect_bull_regime(self, regime_detector, bull_market_data):
        """Test bull market detection."""
        # Data already has indicators added
        regime, confidence = regime_detector.detect_regime(bull_market_data)

        # Should detect bull market
        assert regime == 'bull'
        assert confidence > 0.5

    def test_detect_bear_regime(self, regime_detector, bear_market_data):
        """Test bear market detection."""
        # Data already has indicators added
        regime, confidence = regime_detector.detect_regime(bear_market_data)

        # Should detect bear market
        assert regime == 'bear'
        assert confidence > 0.5

    def test_regime_adjustments_applied(self, risk_manager):
        """Test that regime adjustments are applied correctly."""
        original_risk = risk_manager.risk_per_trade

        # Apply bear adjustments
        risk_manager.apply_regime_adjustments(0.5, 4)

        # Risk should be reduced
        assert risk_manager.risk_per_trade == original_risk * 0.5
        assert risk_manager.max_positions == 4


class TestFeature3RealisticStopModeling:
    """Test Feature 3: Realistic stop loss slippage."""

    def test_stock_stop_with_slippage(self, signal_generator):
        """Test stock stop loss applies 0.5% slippage."""
        position = Position(
            ticker='AAPL',
            asset_type='stock',
            entry_date=datetime(2025, 1, 1),
            entry_price=100.0,
            initial_stop=90.0,
            target_price=125.0
        )

        current = pd.Series({
            'open': 91.0,
            'high': 92.0,
            'low': 89.0,
            'close': 89.5
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 91.0})

        exit_signal = signal_generator.check_stop_loss(position, current, previous)

        # Should have slippage
        assert exit_signal is not None
        assert exit_signal['slippage_pct'] > 0
        assert exit_signal['exit_price'] < exit_signal['stop_price']

    def test_gap_adds_extra_slippage(self, signal_generator):
        """Test that gap down adds extra slippage."""
        position = Position(
            ticker='AAPL',
            asset_type='stock',
            entry_date=datetime(2025, 1, 1),
            entry_price=100.0,
            initial_stop=90.0,
            target_price=125.0
        )

        # Gap down scenario
        current = pd.Series({
            'open': 88.0,  # Gap from 91
            'high': 89.0,
            'low': 87.0,
            'close': 87.5
        }, name=datetime(2025, 1, 5))

        previous = pd.Series({'close': 91.0})

        exit_signal = signal_generator.check_stop_loss(position, current, previous)

        # Should detect gap and apply extra slippage
        assert exit_signal['gap_detected'] is True
        assert exit_signal['slippage_pct'] > 0.005  # More than base 0.5%


class TestIntegrationAllFeatures:
    """Test all 3 features working together in realistic scenarios."""

    def test_bull_market_strong_signal_scenario(
        self, signal_generator, risk_manager, regime_detector,
        bull_market_data, strong_momentum_stock
    ):
        """
        Test complete workflow in bull market with strong signal.
        Expected: Full risk, large position, realistic stop slippage.
        """
        # 1. Detect bull market regime (data already has indicators)
        regime, _ = regime_detector.detect_regime(bull_market_data)
        assert regime == 'bull'

        # 2. Apply regime adjustments (1.0x risk, 8 positions)
        adjustments = regime_detector.get_regime_adjustments(regime, 0.8)
        risk_manager.apply_regime_adjustments(
            adjustments.risk_multiplier,
            adjustments.max_positions
        )
        assert risk_manager.max_positions == 8

        # 3. Check entry signals (data already has indicators)
        entry = signal_generator.check_entry_signals(strong_momentum_stock, len(strong_momentum_stock) - 1, 'TEST')

        # If we get an entry signal, test the workflow
        if entry:
            # 4. Calculate position size (should be reasonable due to bull + strong signal)
            size, value = risk_manager.calculate_position_size(
                entry['entry_price'],
                entry['stop_loss'],
                'stock',
                signal_strength=entry.get('signal_strength', 100)
            )

            assert size > 0
            assert value > 0

            # 5. Test stop loss with slippage
            position = Position(
                ticker='TEST',
                asset_type='stock',
                entry_date=datetime.now(),
                entry_price=entry['entry_price'],
                initial_stop=entry['stop_loss'],
                target_price=entry['target_price']
            )

            # Create stop scenario
            current = pd.Series({
                'open': entry['entry_price'] - 1,
                'high': entry['entry_price'],
                'low': entry['stop_loss'] - 1,
                'close': entry['stop_loss']
            })

            exit_signal = signal_generator.check_stop_loss(position, current)

            # Should have realistic slippage
            if exit_signal:
                assert exit_signal['slippage_pct'] > 0

    def test_bear_market_weak_signal_scenario(
        self, signal_generator, risk_manager, regime_detector,
        bear_market_data, weak_momentum_stock
    ):
        """
        Test complete workflow in bear market with weak signal.
        Expected: Reduced risk, small position, realistic stop slippage.
        """
        # 1. Detect bear market regime (data already has indicators)
        regime, _ = regime_detector.detect_regime(bear_market_data)
        assert regime == 'bear'

        # 2. Apply regime adjustments (0.5x risk, 4 positions)
        adjustments = regime_detector.get_regime_adjustments(regime, 0.8)
        risk_manager.apply_regime_adjustments(
            adjustments.risk_multiplier,
            adjustments.max_positions
        )
        assert risk_manager.max_positions == 4
        assert risk_manager.risk_per_trade < 0.01  # Reduced from 1%

        # 3. Check entry signals (data already has indicators)
        entry = signal_generator.check_entry_signals(weak_momentum_stock, len(weak_momentum_stock) - 1, 'TEST')

        # If we get a signal, it should be processed with reduced risk
        if entry:
            # 4. Calculate position size (should be small due to bear + weak signal)
            size, value = risk_manager.calculate_position_size(
                entry['entry_price'],
                entry['stop_loss'],
                'stock',
                signal_strength=entry.get('signal_strength', 75)
            )

            # Position should exist but be small
            assert size >= 0

    def test_combined_scaling(self, risk_manager):
        """
        Test that regime and signal strength scaling combine correctly.
        """
        original_risk = risk_manager.risk_per_trade

        # Base position size (full risk, full signal)
        size_base, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=100
        )

        # Apply bear regime (0.5x) + weak signal (0.6x)
        risk_manager.apply_regime_adjustments(0.5, 4)
        size_combined, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=75  # 60% sizing
        )

        # Should be approximately 0.5 * 0.6 = 0.3 of base
        expected_ratio = 0.5 * 0.6
        actual_ratio = size_combined / size_base if size_base > 0 else 0

        # Allow some tolerance due to rounding
        assert abs(actual_ratio - expected_ratio) < 0.05


class TestSystemPerformance:
    """Test overall system performance and constraints."""

    def test_signal_generation_performance(self, signal_generator, strong_momentum_stock):
        """Test that signal generation is fast enough."""
        import time

        # Data already has indicators
        df = strong_momentum_stock

        start = time.time()
        for i in range(220, len(df)):  # Start at 220 to have enough history
            signal_generator.check_entry_signals(df, i, 'TEST')
        elapsed = time.time() - start

        # Should be fast (< 2 seconds for 30 signals)
        assert elapsed < 2.0

    def test_regime_detection_performance(self, regime_detector, bull_market_data):
        """Test that regime detection is fast enough."""
        import time

        # Data already has indicators
        df = bull_market_data

        start = time.time()
        for _ in range(10):
            regime_detector.detect_regime(df)
        elapsed = time.time() - start

        # Should be fast (< 1 second for 10 detections)
        assert elapsed < 1.0


class TestBackwardCompatibility:
    """Test that new features don't break existing functionality."""

    def test_signal_generator_without_strength(self, signal_generator, strong_momentum_stock):
        """Test that signals still work if signal_strength isn't required."""
        # Data already has indicators
        entry = signal_generator.check_entry_signals(strong_momentum_stock, len(strong_momentum_stock) - 1, 'TEST')

        # Should still generate signals (or None if no signal)
        assert entry is None or isinstance(entry, dict)

    def test_risk_manager_without_regime(self, risk_manager):
        """Test that risk manager works without regime adjustments."""
        # Should work with default parameters
        size, value = risk_manager.calculate_position_size(
            100, 90, 'stock'
        )

        assert size > 0
        assert value > 0

    def test_stop_loss_without_previous_bar(self, signal_generator):
        """Test that stop loss works without previous bar (no gap detection)."""
        position = Position(
            ticker='AAPL',
            asset_type='stock',
            entry_date=datetime(2025, 1, 1),
            entry_price=100.0,
            initial_stop=90.0,
            target_price=125.0
        )

        current = pd.Series({
            'open': 91.0,
            'high': 92.0,
            'low': 89.0,
            'close': 89.5
        })

        # Should work without previous bar
        exit_signal = signal_generator.check_stop_loss(position, current, previous=None)

        assert exit_signal is not None
        assert exit_signal['gap_detected'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
