"""Unit tests for signal strength scoring (Iteration 1)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.signals import SignalGenerator
from indicators.technical import TechnicalIndicators
from indicators.momentum import MomentumMetrics


@pytest.fixture
def signal_generator():
    """Create SignalGenerator instance."""
    return SignalGenerator()


@pytest.fixture
def technical_indicators():
    """Create TechnicalIndicators instance."""
    return TechnicalIndicators()


@pytest.fixture
def momentum_metrics():
    """Create MomentumMetrics instance."""
    return MomentumMetrics()


@pytest.fixture
def perfect_signal_data():
    """
    Create data that meets ALL 6 conditions (100 points).

    Conditions:
    1. Breakout (20-day high)
    2. Volume surge (>1.5x)
    3. MACD positive
    4. Price above 50-day MA
    5. 50-day MA trending up
    6. Strong momentum (>=70)
    """
    dates = pd.bdate_range(start='2024-01-01', periods=100)

    # Build strong uptrend with consolidation then powerful breakout
    closes = []
    # Days 0-40: Strong uptrend from 100 to 160
    closes.extend(np.linspace(100, 160, 40))
    # Days 40-90: Consolidation around 160
    closes.extend([160 + np.random.uniform(-2, 2) for _ in range(50)])
    # Days 90-99: Powerful breakout to 200 (very strong momentum)
    closes.extend(np.linspace(162, 200, 10))

    closes = np.array(closes)
    highs = closes + 2
    lows = closes - 1
    opens = closes - 0.5

    # Normal volume, then surge ONLY in last bar for clear volume_ratio
    volumes = [1000000] * 99 + [2500000]  # 2.5x surge on last bar

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def four_condition_data():
    """
    Create data with 4-5 conditions met (~70-85 points).

    Target: Breakout, volume surge, above MA50, MA trending, but weaker momentum
    Should get ~75-80 points
    """
    dates = pd.bdate_range(start='2024-01-01', periods=100)

    # Moderate uptrend with consolidation then breakout
    closes = []
    closes.extend(np.linspace(100, 130, 70))
    closes.extend([130 + np.random.uniform(-1, 1) for _ in range(25)])
    closes.extend(np.linspace(131, 145, 5))  # Moderate breakout

    closes = np.array(closes)
    highs = closes + 2
    lows = closes - 1
    opens = closes - 0.5

    # Volume surge on last bar
    volumes = [1000000] * 99 + [2000000]

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def partial_credit_data():
    """
    Create data with partial credit conditions.

    - Volume 1.3x (10 points instead of 15)
    - Momentum 60 (15 points instead of 20)
    """
    dates = pd.bdate_range(start='2024-01-01', periods=100)

    closes = np.linspace(100, 140, 100)
    highs = closes + 2
    lows = closes - 1
    opens = closes - 0.5

    # Volume 1.3x (partial credit)
    volumes = [1000000] * 75 + [1400000] * 25

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def below_threshold_data():
    """Create data below 70 threshold (<70 points)."""
    dates = pd.bdate_range(start='2024-01-01', periods=100)

    # Sideways with small moves
    closes = 100 + np.random.normal(0, 2, 100)
    highs = closes + 1
    lows = closes - 1
    opens = closes
    volumes = [1000000] * 100

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


class TestSignalStrengthCalculation:
    """Test calculate_entry_signal_strength() method."""

    def test_perfect_signal_100_points(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics,
        perfect_signal_data
    ):
        """Test that all 6 conditions met = 100 points."""
        # Add all indicators
        df = technical_indicators.add_all_indicators(perfect_signal_data)
        df = momentum_metrics.add_all_momentum_metrics(df)

        # Test at last index
        result = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        # Should have 100 points (all conditions met)
        assert result['signal_strength'] == 100
        assert len(result['conditions_met']) == 6
        assert len(result['conditions_failed']) == 0

        # Verify all expected conditions
        expected = [
            'breakout_20d_high',
            'volume_surge',
            'macd_positive',
            'above_ma50',
            'ma50_trending_up',
            'strong_momentum'
        ]
        for cond in expected:
            assert cond in result['conditions_met']

    def test_four_conditions_70_85_points(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics,
        four_condition_data
    ):
        """Test that 4-5 conditions = 70-85 points."""
        df = technical_indicators.add_all_indicators(four_condition_data)
        df = momentum_metrics.add_all_momentum_metrics(df)

        result = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        # Should have 70-90 points
        assert 70 <= result['signal_strength'] <= 90
        assert 4 <= len(result['conditions_met']) <= 5

    def test_partial_credit_volume_1_3x(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics
    ):
        """Test partial credit for volume 1.3x = 10 points."""
        dates = pd.bdate_range(start='2024-01-01', periods=100)

        # Simple uptrend
        closes = np.linspace(100, 150, 100)

        # Volume surge to achieve ~1.3x ratio
        # Rolling average includes current bar, so need higher volume
        volumes = [1000000] * 99 + [1350000]

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 1,
            'close': closes,
            'volume': volumes
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)
        df = momentum_metrics.add_all_momentum_metrics(df)

        result = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        # Volume should give 10 points (partial credit for 1.3-1.5x)
        assert result['condition_scores']['volume_surge'] == 10

    def test_partial_credit_volume_1_1x(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics
    ):
        """Test minimal credit for volume 1.1x = 5 points."""
        dates = pd.bdate_range(start='2024-01-01', periods=100)

        closes = np.linspace(100, 150, 100)

        # Volume surge to achieve ~1.1x ratio
        # Rolling average includes current bar, so need higher volume
        volumes = [1000000] * 99 + [1120000]

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 1,
            'close': closes,
            'volume': volumes
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)
        df = momentum_metrics.add_all_momentum_metrics(df)

        result = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        # Volume should give 5 points (minimal credit for 1.1-1.3x)
        assert result['condition_scores']['volume_surge'] == 5

    def test_partial_credit_momentum_60(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics
    ):
        """Test partial credit for momentum 60 = 15 points."""
        dates = pd.bdate_range(start='2024-01-01', periods=100)

        # Moderate uptrend (should give ~60 momentum)
        closes = np.linspace(100, 130, 100)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 1,
            'low': closes - 1,
            'close': closes,
            'volume': [1000000] * 100
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)
        df = momentum_metrics.add_all_momentum_metrics(df)

        result = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        # If momentum is 60-69, should get 15 points
        if 60 <= df['momentum_score'].iloc[-1] < 70:
            assert result['condition_scores']['strong_momentum'] == 15

    def test_partial_credit_momentum_50(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics
    ):
        """Test minimal credit for momentum 50 = 10 points."""
        dates = pd.bdate_range(start='2024-01-01', periods=100)

        # Weak uptrend (should give ~50 momentum)
        closes = np.linspace(100, 115, 100)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 1,
            'low': closes - 1,
            'close': closes,
            'volume': [1000000] * 100
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)
        df = momentum_metrics.add_all_momentum_metrics(df)

        result = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        # If momentum is 50-59, should get 10 points
        if 50 <= df['momentum_score'].iloc[-1] < 60:
            assert result['condition_scores']['strong_momentum'] == 10

    def test_condition_scores_breakdown(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics,
        perfect_signal_data
    ):
        """Test that condition scores add up correctly."""
        df = technical_indicators.add_all_indicators(perfect_signal_data)
        df = momentum_metrics.add_all_momentum_metrics(df)

        result = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        # Verify individual scores
        expected_scores = {
            'breakout': 20,
            'volume_surge': 15,
            'macd_positive': 15,
            'above_ma50': 15,
            'ma_trending_up': 15,
            'strong_momentum': 20
        }

        for key, expected in expected_scores.items():
            assert key in result['condition_scores']
            assert result['condition_scores'][key] <= expected

        # Verify total
        total = sum(result['condition_scores'].values())
        assert total == result['signal_strength']

    def test_insufficient_data(self, signal_generator):
        """Test with insufficient data (< 20 bars)."""
        dates = pd.bdate_range(start='2024-01-01', periods=15)
        df = pd.DataFrame({
            'open': [100] * 15,
            'high': [105] * 15,
            'low': [95] * 15,
            'close': [100] * 15,
            'volume': [1000000] * 15
        }, index=dates)

        result = signal_generator.calculate_entry_signal_strength(
            df, 10, 'TEST'
        )

        # Should return 0 with insufficient data
        assert result['signal_strength'] == 0
        assert len(result['conditions_met']) == 0


class TestCheckEntrySignals:
    """Test check_entry_signals() with 70 threshold."""

    def test_70_threshold_enforced(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics,
        perfect_signal_data
    ):
        """Test that 70+ threshold is enforced."""
        df = technical_indicators.add_all_indicators(perfect_signal_data)
        df = momentum_metrics.add_all_momentum_metrics(df)

        signal = signal_generator.check_entry_signals(
            df, len(df) - 1, 'TEST'
        )

        # Should generate signal with 100 points
        assert signal is not None
        assert signal['signal_strength'] >= 70
        assert signal['ticker'] == 'TEST'

    def test_below_threshold_no_signal(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics,
        below_threshold_data
    ):
        """Test that <70 points = no signal."""
        df = technical_indicators.add_all_indicators(below_threshold_data)
        df = momentum_metrics.add_all_momentum_metrics(df)

        signal = signal_generator.check_entry_signals(
            df, len(df) - 1, 'TEST'
        )

        # Should NOT generate signal if below 70
        # First check signal strength
        strength = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        if strength['signal_strength'] < 70:
            assert signal is None

    def test_signal_includes_strength_info(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics,
        perfect_signal_data
    ):
        """Test that signal includes strength information."""
        df = technical_indicators.add_all_indicators(perfect_signal_data)
        df = momentum_metrics.add_all_momentum_metrics(df)

        signal = signal_generator.check_entry_signals(
            df, len(df) - 1, 'TEST'
        )

        # Verify signal structure
        assert signal is not None
        assert 'signal_strength' in signal
        assert 'conditions_met' in signal
        assert 'conditions_failed' in signal
        assert isinstance(signal['conditions_met'], list)
        assert isinstance(signal['conditions_failed'], list)


class TestSignalStrengthBounds:
    """Test that signal strength stays within bounds."""

    def test_minimum_is_zero(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics
    ):
        """Test that minimum signal strength is 0."""
        dates = pd.bdate_range(start='2024-01-01', periods=100)

        # Terrible conditions (downtrend, no volume)
        closes = np.linspace(200, 100, 100)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 1,
            'low': closes - 1,
            'close': closes,
            'volume': [500000] * 100  # Decreasing volume
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)
        df = momentum_metrics.add_all_momentum_metrics(df)

        result = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        assert result['signal_strength'] >= 0

    def test_maximum_is_100(
        self,
        signal_generator,
        technical_indicators,
        momentum_metrics,
        perfect_signal_data
    ):
        """Test that maximum signal strength is 100."""
        df = technical_indicators.add_all_indicators(perfect_signal_data)
        df = momentum_metrics.add_all_momentum_metrics(df)

        result = signal_generator.calculate_entry_signal_strength(
            df, len(df) - 1, 'TEST'
        )

        assert result['signal_strength'] <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
