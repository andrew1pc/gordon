"""Unit tests for MarketRegimeDetector (Iteration 4)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indicators.market_regime import MarketRegimeDetector, RegimeAdjustments
from indicators.technical import TechnicalIndicators


@pytest.fixture
def detector():
    """Create MarketRegimeDetector instance."""
    return MarketRegimeDetector()


@pytest.fixture
def technical_indicators():
    """Create TechnicalIndicators instance."""
    return TechnicalIndicators()


@pytest.fixture
def bull_market_data(technical_indicators):
    """
    Create data representing a bull market.

    Characteristics:
    - Price above MA50 and MA200
    - MA50 above MA200 (golden cross)
    - Upward trending
    """
    dates = pd.bdate_range(start='2024-01-01', periods=250)

    # Strong uptrend from 100 to 200
    closes = np.linspace(100, 200, 250)
    highs = closes + 2
    lows = closes - 2
    opens = closes - 0.5
    volumes = [1000000] * 250

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    # Add technical indicators
    df = technical_indicators.add_all_indicators(df)

    return df


@pytest.fixture
def bear_market_data(technical_indicators):
    """
    Create data representing a bear market.

    Characteristics:
    - Price below MA50 and MA200
    - MA50 below MA200 (death cross)
    - Downward trending
    """
    dates = pd.bdate_range(start='2024-01-01', periods=250)

    # Strong downtrend from 200 to 100
    closes = np.linspace(200, 100, 250)
    highs = closes + 2
    lows = closes - 2
    opens = closes + 0.5
    volumes = [1000000] * 250

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    # Add technical indicators
    df = technical_indicators.add_all_indicators(df)

    return df


@pytest.fixture
def sideways_market_data(technical_indicators):
    """
    Create data representing a sideways/ranging market.

    Characteristics:
    - Mixed signals
    - Price oscillating around MAs
    - No clear trend
    """
    dates = pd.bdate_range(start='2024-01-01', periods=250)

    # Sideways movement around 150
    closes = 150 + np.sin(np.linspace(0, 8*np.pi, 250)) * 10
    highs = closes + 2
    lows = closes - 2
    opens = closes
    volumes = [1000000] * 250

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    # Add technical indicators
    df = technical_indicators.add_all_indicators(df)

    return df


class TestRegimeDetection:
    """Test regime detection logic."""

    def test_detect_bull_market(self, detector, bull_market_data):
        """Test that bull market is detected correctly."""
        regime, confidence = detector.detect_regime(bull_market_data)

        assert regime == 'bull'
        assert confidence >= 0.6  # Should have high confidence

    def test_detect_bear_market(self, detector, bear_market_data):
        """Test that bear market is detected correctly."""
        regime, confidence = detector.detect_regime(bear_market_data)

        assert regime == 'bear'
        assert confidence >= 0.6  # Should have high confidence

    def test_detect_sideways_market(self, detector, sideways_market_data):
        """Test that sideways market is detected correctly."""
        regime, confidence = detector.detect_regime(sideways_market_data)

        assert regime == 'sideways'
        # Confidence might be lower for sideways (mixed signals)

    def test_confidence_range(self, detector, bull_market_data):
        """Test that confidence is always 0-1."""
        regime, confidence = detector.detect_regime(bull_market_data)

        assert 0.0 <= confidence <= 1.0

    def test_empty_dataframe(self, detector):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        regime, confidence = detector.detect_regime(empty_df)

        # Should default to sideways with medium confidence
        assert regime == 'sideways'
        assert confidence == 0.5

    def test_missing_columns(self, detector):
        """Test handling of DataFrame with missing columns."""
        df = pd.DataFrame({
            'close': [100, 101, 102]
        })

        regime, confidence = detector.detect_regime(df)

        # Should default to sideways
        assert regime == 'sideways'
        assert confidence == 0.5


class TestRegimeAdjustments:
    """Test regime adjustment calculations."""

    def test_bull_regime_adjustments(self, detector):
        """Test adjustments for bull regime."""
        adjustments = detector.get_regime_adjustments('bull', 0.8)

        assert adjustments.regime == 'bull'
        assert adjustments.risk_multiplier == 1.0
        assert adjustments.max_positions == 8
        assert adjustments.confidence == 0.8

    def test_bear_regime_adjustments(self, detector):
        """Test adjustments for bear regime."""
        adjustments = detector.get_regime_adjustments('bear', 0.9)

        assert adjustments.regime == 'bear'
        assert adjustments.risk_multiplier == 0.5
        assert adjustments.max_positions == 4
        assert adjustments.confidence == 0.9

    def test_sideways_regime_adjustments(self, detector):
        """Test adjustments for sideways regime."""
        adjustments = detector.get_regime_adjustments('sideways', 0.7)

        assert adjustments.regime == 'sideways'
        assert adjustments.risk_multiplier == 0.75
        assert adjustments.max_positions == 6
        assert adjustments.confidence == 0.7

    def test_invalid_regime_defaults_to_sideways(self, detector):
        """Test that invalid regime defaults to sideways."""
        adjustments = detector.get_regime_adjustments('invalid', 0.5)

        assert adjustments.regime == 'sideways'
        assert adjustments.risk_multiplier == 0.75
        assert adjustments.max_positions == 6


class TestRegimeSettings:
    """Test hardcoded regime settings."""

    def test_regime_settings_exist(self, detector):
        """Test that all regimes have settings."""
        assert 'bull' in detector.REGIME_SETTINGS
        assert 'bear' in detector.REGIME_SETTINGS
        assert 'sideways' in detector.REGIME_SETTINGS

    def test_bull_settings_correct(self, detector):
        """Test bull regime hardcoded values."""
        settings = detector.REGIME_SETTINGS['bull']
        assert settings['risk_multiplier'] == 1.0
        assert settings['max_positions'] == 8

    def test_sideways_settings_correct(self, detector):
        """Test sideways regime hardcoded values."""
        settings = detector.REGIME_SETTINGS['sideways']
        assert settings['risk_multiplier'] == 0.75
        assert settings['max_positions'] == 6

    def test_bear_settings_correct(self, detector):
        """Test bear regime hardcoded values."""
        settings = detector.REGIME_SETTINGS['bear']
        assert settings['risk_multiplier'] == 0.5
        assert settings['max_positions'] == 4


class TestRegimeClassificationCriteria:
    """Test specific regime classification criteria."""

    def test_price_above_mas_indicates_bull(self, detector, technical_indicators):
        """Test that price above MAs contributes to bull classification."""
        dates = pd.bdate_range(start='2024-01-01', periods=250)

        # Create data where price is well above MAs
        closes = np.linspace(100, 200, 250)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 2,
            'close': closes,
            'volume': [1000000] * 250
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)

        regime, confidence = detector.detect_regime(df)

        # Should detect bull (price above MAs, MA50 > MA200)
        assert regime == 'bull'

    def test_price_below_mas_indicates_bear(self, detector, technical_indicators):
        """Test that price below MAs contributes to bear classification."""
        dates = pd.bdate_range(start='2024-01-01', periods=250)

        # Create data where price is well below MAs
        closes = np.linspace(200, 100, 250)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 2,
            'close': closes,
            'volume': [1000000] * 250
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)

        regime, confidence = detector.detect_regime(df)

        # Should detect bear (price below MAs, MA50 < MA200)
        assert regime == 'bear'

    def test_golden_cross_supports_bull(self, detector, technical_indicators):
        """Test that golden cross (MA50 > MA200) supports bull regime."""
        dates = pd.bdate_range(start='2024-01-01', periods=250)

        # Strong uptrend that creates golden cross
        closes = np.linspace(100, 200, 250)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 2,
            'close': closes,
            'volume': [1000000] * 250
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)

        # Verify golden cross exists
        current = df.iloc[-1]
        assert current['sma_50'] > current['sma_200']

        regime, confidence = detector.detect_regime(df)
        assert regime == 'bull'

    def test_death_cross_supports_bear(self, detector, technical_indicators):
        """Test that death cross (MA50 < MA200) supports bear regime."""
        dates = pd.bdate_range(start='2024-01-01', periods=250)

        # Strong downtrend that creates death cross
        closes = np.linspace(200, 100, 250)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 2,
            'close': closes,
            'volume': [1000000] * 250
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)

        # Verify death cross exists
        current = df.iloc[-1]
        assert current['sma_50'] < current['sma_200']

        regime, confidence = detector.detect_regime(df)
        assert regime == 'bear'


class TestShouldTrade:
    """Test trading decision logic."""

    def test_should_trade_in_bull(self, detector):
        """Test that trading is allowed in bull market."""
        should_trade, reason = detector.should_trade('bull', 0.8)

        assert should_trade is True
        assert 'allowed' in reason.lower()

    def test_should_trade_in_bear(self, detector):
        """Test that trading is allowed in bear market (with reduced risk)."""
        should_trade, reason = detector.should_trade('bear', 0.9)

        # We allow trading but with reduced risk
        assert should_trade is True
        assert 'allowed' in reason.lower()

    def test_should_trade_in_sideways(self, detector):
        """Test that trading is allowed in sideways market."""
        should_trade, reason = detector.should_trade('sideways', 0.7)

        assert should_trade is True


class TestRegimeSummary:
    """Test comprehensive regime summary."""

    def test_get_regime_summary(self, detector, bull_market_data):
        """Test comprehensive regime summary generation."""
        summary = detector.get_regime_summary(bull_market_data)

        # Verify all expected keys present
        assert 'regime' in summary
        assert 'confidence' in summary
        assert 'risk_multiplier' in summary
        assert 'max_positions' in summary
        assert 'current_price' in summary
        assert 'ma50' in summary
        assert 'ma200' in summary
        assert 'price_vs_ma50' in summary
        assert 'price_vs_ma200' in summary
        assert 'ma50_vs_ma200' in summary

    def test_summary_bull_regime(self, detector, bull_market_data):
        """Test summary for bull regime."""
        summary = detector.get_regime_summary(bull_market_data)

        assert summary['regime'] == 'bull'
        assert summary['risk_multiplier'] == 1.0
        assert summary['max_positions'] == 8
        assert summary['ma50_vs_ma200'] == 'above'

    def test_summary_bear_regime(self, detector, bear_market_data):
        """Test summary for bear regime."""
        summary = detector.get_regime_summary(bear_market_data)

        assert summary['regime'] == 'bear'
        assert summary['risk_multiplier'] == 0.5
        assert summary['max_positions'] == 4
        assert summary['ma50_vs_ma200'] == 'below'

    def test_summary_empty_dataframe(self, detector):
        """Test summary with empty DataFrame."""
        summary = detector.get_regime_summary(pd.DataFrame())

        assert summary['regime'] == 'sideways'
        assert summary['current_price'] == 0
        assert summary['ma50'] == 0


class TestRegimeTransitions:
    """Test regime transitions and edge cases."""

    def test_bull_to_bear_transition(self, detector, technical_indicators):
        """Test detection during bull to bear transition."""
        dates = pd.bdate_range(start='2024-01-01', periods=250)

        # Bull market first 150 days, then bear
        closes = list(np.linspace(100, 200, 150)) + list(np.linspace(200, 150, 100))

        df = pd.DataFrame({
            'open': closes,
            'high': [c + 2 for c in closes],
            'low': [c - 2 for c in closes],
            'close': closes,
            'volume': [1000000] * 250
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)

        # Should detect current state (might be sideways during transition)
        regime, confidence = detector.detect_regime(df)

        assert regime in ['bull', 'bear', 'sideways']
        # Confidence might be lower during transition
        assert 0.0 <= confidence <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
