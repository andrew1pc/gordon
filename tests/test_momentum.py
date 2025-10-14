import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indicators.momentum import MomentumMetrics
from indicators.technical import TechnicalIndicators


# Fixtures

@pytest.fixture
def momentum_metrics():
    """Create a MomentumMetrics instance."""
    return MomentumMetrics()


@pytest.fixture
def technical_indicators():
    """Create a TechnicalIndicators instance."""
    return TechnicalIndicators()


@pytest.fixture
def uptrend_data():
    """Create data with strong uptrend."""
    dates = pd.bdate_range(start='2024-01-01', periods=250)
    # Strong uptrend: 100 to 200
    closes = np.linspace(100, 200, 250)
    highs = closes + np.random.uniform(1, 3, 250)
    lows = closes - np.random.uniform(1, 3, 250)
    opens = closes + np.random.uniform(-2, 2, 250)
    # Increasing volume with uptrend
    volumes = np.linspace(1000000, 2000000, 250)

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def downtrend_data():
    """Create data with downtrend."""
    dates = pd.bdate_range(start='2024-01-01', periods=250)
    closes = np.linspace(200, 100, 250)
    highs = closes + np.random.uniform(1, 3, 250)
    lows = closes - np.random.uniform(1, 3, 250)
    opens = closes + np.random.uniform(-2, 2, 250)
    volumes = np.random.uniform(1000000, 2000000, 250)

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def ranging_data():
    """Create data with ranging/sideways movement."""
    dates = pd.bdate_range(start='2024-01-01', periods=250)
    closes = 150 + np.random.normal(0, 5, 250)
    highs = closes + np.random.uniform(1, 3, 250)
    lows = closes - np.random.uniform(1, 3, 250)
    opens = closes + np.random.uniform(-2, 2, 250)
    volumes = np.random.uniform(1000000, 2000000, 250)

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def data_with_volume_spike():
    """Create data with volume spike."""
    dates = pd.bdate_range(start='2024-01-01', periods=50)
    closes = [100 + i for i in range(50)]
    # Normal volume for first 45 days, then spike
    volumes = [1000000] * 45 + [3000000] * 5

    df = pd.DataFrame({
        'open': closes,
        'high': [c + 2 for c in closes],
        'low': [c - 2 for c in closes],
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


# Tests for ROC

class TestROC:
    """Tests for Rate of Change."""

    def test_roc_calculation(self, momentum_metrics):
        """Test ROC calculation with known values."""
        dates = pd.bdate_range(start='2024-01-01', periods=25)
        # Price increases from 100 to 120 over 20 periods
        closes = [100] * 5 + [100 + i for i in range(20)]

        df = pd.DataFrame({
            'open': closes,
            'high': [c + 2 for c in closes],
            'low': [c - 2 for c in closes],
            'close': closes,
            'volume': [1000000] * 25
        }, index=dates)

        result = momentum_metrics.calculate_roc(df, 20)

        assert 'roc_20' in result.columns

        # At index 24, price is 119, 20 periods ago was 100
        # ROC = ((119 - 100) / 100) * 100 = 19%
        expected_roc = ((119 - 100) / 100) * 100
        assert abs(result['roc_20'].iloc[24] - expected_roc) < 0.1

    def test_roc_positive_momentum(self, momentum_metrics, uptrend_data):
        """Test that ROC is positive in uptrend."""
        result = momentum_metrics.calculate_roc(uptrend_data, 20)

        # Most ROC values should be positive in uptrend
        positive_count = (result['roc_20'].dropna() > 0).sum()
        total_count = result['roc_20'].dropna().count()

        assert positive_count / total_count > 0.8  # At least 80% positive

    def test_roc_negative_momentum(self, momentum_metrics, downtrend_data):
        """Test that ROC is negative in downtrend."""
        result = momentum_metrics.calculate_roc(downtrend_data, 20)

        # Most ROC values should be negative in downtrend
        negative_count = (result['roc_20'].dropna() < 0).sum()
        total_count = result['roc_20'].dropna().count()

        assert negative_count / total_count > 0.8  # At least 80% negative

    def test_roc_column_naming(self, momentum_metrics, uptrend_data):
        """Test ROC column naming."""
        result = momentum_metrics.calculate_roc(uptrend_data, 50)
        assert 'roc_50' in result.columns

    def test_roc_empty_dataframe(self, momentum_metrics):
        """Test ROC with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            momentum_metrics.calculate_roc(empty_df, 20)

    def test_roc_period_too_large(self, momentum_metrics):
        """Test ROC when period exceeds data length."""
        dates = pd.bdate_range(start='2024-01-01', periods=10)
        df = pd.DataFrame({
            'close': [100] * 10,
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'volume': [1000000] * 10
        }, index=dates)

        with pytest.raises(ValueError, match="less than data length"):
            momentum_metrics.calculate_roc(df, 20)


# Tests for Trend Strength

class TestTrendStrength:
    """Tests for trend strength score."""

    def test_trend_strength_uptrend(self, momentum_metrics, technical_indicators, uptrend_data):
        """Test that trend strength is high in uptrend."""
        # Add technical indicators first
        df = technical_indicators.add_all_indicators(uptrend_data)
        result = momentum_metrics.calculate_trend_strength(df)

        assert 'trend_strength' in result.columns
        assert 'trend_direction' in result.columns

        # Last 10 periods should have high trend strength
        avg_strength = result['trend_strength'].iloc[-10:].mean()
        assert avg_strength > 70

        # Should be classified as uptrend
        assert (result['trend_direction'].iloc[-10:] == 'up').all()

    def test_trend_strength_downtrend(self, momentum_metrics, technical_indicators, downtrend_data):
        """Test that trend strength is low in downtrend."""
        df = technical_indicators.add_all_indicators(downtrend_data)
        result = momentum_metrics.calculate_trend_strength(df)

        # Last 10 periods should have low trend strength
        avg_strength = result['trend_strength'].iloc[-10:].mean()
        assert avg_strength < 30

        # Should be classified as downtrend
        assert (result['trend_direction'].iloc[-10:] == 'down').all()

    def test_trend_strength_scoring(self, momentum_metrics, technical_indicators):
        """Test trend strength scoring criteria."""
        dates = pd.bdate_range(start='2024-01-01', periods=250)
        # Create perfect uptrend
        closes = np.linspace(100, 200, 250)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 2,
            'close': closes,
            'volume': [1000000] * 250
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)
        result = momentum_metrics.calculate_trend_strength(df)

        # In perfect uptrend, should get maximum or near-maximum score
        final_score = result['trend_strength'].iloc[-1]
        assert final_score >= 90  # Should hit most criteria

    def test_trend_strength_missing_columns(self, momentum_metrics):
        """Test trend strength with missing columns."""
        df = pd.DataFrame({'close': [100, 101, 102]})
        with pytest.raises(ValueError, match="must contain"):
            momentum_metrics.calculate_trend_strength(df)

    def test_trend_strength_bounds(self, momentum_metrics, technical_indicators, uptrend_data):
        """Test that trend strength stays within 0-100."""
        df = technical_indicators.add_all_indicators(uptrend_data)
        result = momentum_metrics.calculate_trend_strength(df)

        assert result['trend_strength'].min() >= 0
        assert result['trend_strength'].max() <= 100


# Tests for Volume Surge Detection

class TestVolumeSurge:
    """Tests for volume surge detection."""

    def test_volume_surge_detection(self, momentum_metrics, data_with_volume_spike):
        """Test that volume surges are detected."""
        result = momentum_metrics.detect_volume_surge(data_with_volume_spike, threshold=1.5)

        assert 'volume_ratio' in result.columns
        assert 'volume_surge' in result.columns

        # Last 5 days should have volume surge
        assert result['volume_surge'].iloc[-5:].all()

    def test_volume_ratio_calculation(self, momentum_metrics, data_with_volume_spike):
        """Test volume ratio calculation."""
        result = momentum_metrics.detect_volume_surge(data_with_volume_spike)

        # Last days should have high volume ratio
        assert result['volume_ratio'].iloc[-1] > 2.0

    def test_volume_surge_threshold(self, momentum_metrics, data_with_volume_spike):
        """Test different threshold values."""
        # With high threshold, fewer surges detected
        result_high = momentum_metrics.detect_volume_surge(data_with_volume_spike, threshold=2.5)
        surges_high = result_high['volume_surge'].sum()

        # With low threshold, more surges detected
        result_low = momentum_metrics.detect_volume_surge(data_with_volume_spike, threshold=1.2)
        surges_low = result_low['volume_surge'].sum()

        assert surges_low >= surges_high

    def test_volume_surge_empty_df(self, momentum_metrics):
        """Test volume surge with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            momentum_metrics.detect_volume_surge(empty_df)

    def test_volume_surge_missing_column(self, momentum_metrics):
        """Test volume surge with missing volume column."""
        df = pd.DataFrame({'close': [100, 101, 102]})
        with pytest.raises(ValueError, match="volume"):
            momentum_metrics.detect_volume_surge(df)


# Tests for Momentum Score

class TestMomentumScore:
    """Tests for composite momentum score."""

    def test_momentum_score_calculation(self, momentum_metrics, technical_indicators, uptrend_data):
        """Test momentum score calculation."""
        # Prepare data with all indicators
        df = technical_indicators.add_all_indicators(uptrend_data)
        df = momentum_metrics.calculate_roc(df, 20)
        df = momentum_metrics.calculate_trend_strength(df)
        df = momentum_metrics.detect_volume_surge(df)
        result = momentum_metrics.calculate_momentum_score(df)

        assert 'momentum_score' in result.columns

        # Uptrend should have high momentum score
        avg_score = result['momentum_score'].iloc[-10:].mean()
        assert avg_score > 60

    def test_momentum_score_bounds(self, momentum_metrics, technical_indicators, uptrend_data):
        """Test that momentum score stays within 0-100."""
        df = technical_indicators.add_all_indicators(uptrend_data)
        df = momentum_metrics.add_all_momentum_metrics(df)

        assert df['momentum_score'].min() >= 0
        assert df['momentum_score'].max() <= 100

    def test_momentum_score_downtrend(self, momentum_metrics, technical_indicators, downtrend_data):
        """Test momentum score in downtrend."""
        df = technical_indicators.add_all_indicators(downtrend_data)
        df = momentum_metrics.add_all_momentum_metrics(df)

        # Downtrend should have low momentum score
        avg_score = df['momentum_score'].iloc[-10:].mean()
        assert avg_score < 40

    def test_momentum_score_with_minimal_data(self, momentum_metrics):
        """Test momentum score with minimal indicators."""
        dates = pd.bdate_range(start='2024-01-01', periods=50)
        df = pd.DataFrame({
            'close': [100 + i for i in range(50)],
            'open': [100 + i for i in range(50)],
            'high': [102 + i for i in range(50)],
            'low': [98 + i for i in range(50)],
            'volume': [1000000] * 50
        }, index=dates)

        # Should work even without all indicators
        result = momentum_metrics.calculate_momentum_score(df)
        assert 'momentum_score' in result.columns


# Tests for add_all_momentum_metrics

class TestAddAllMomentumMetrics:
    """Tests for add_all_momentum_metrics convenience method."""

    def test_add_all_metrics(self, momentum_metrics, technical_indicators, uptrend_data):
        """Test that all momentum metrics are added."""
        df = technical_indicators.add_all_indicators(uptrend_data)
        result = momentum_metrics.add_all_momentum_metrics(df)

        expected_columns = [
            'roc_20', 'roc_50',
            'trend_strength', 'trend_direction',
            'volume_ratio', 'volume_surge',
            'momentum_score'
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_all_metrics_empty_df(self, momentum_metrics):
        """Test add_all_momentum_metrics with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            momentum_metrics.add_all_momentum_metrics(empty_df)

    def test_add_all_metrics_insufficient_data(self, momentum_metrics, technical_indicators):
        """Test with insufficient data for some metrics."""
        dates = pd.bdate_range(start='2024-01-01', periods=15)
        df = pd.DataFrame({
            'open': [100] * 15,
            'high': [105] * 15,
            'low': [95] * 15,
            'close': [100] * 15,
            'volume': [1000000] * 15
        }, index=dates)

        df = technical_indicators.add_all_indicators(df)
        result = momentum_metrics.add_all_momentum_metrics(df)

        # Should have momentum_score at minimum
        assert 'momentum_score' in result.columns


# Integration tests

class TestIntegration:
    """Integration tests combining technical indicators and momentum metrics."""

    def test_full_pipeline(self, technical_indicators, momentum_metrics, uptrend_data):
        """Test full pipeline from raw data to momentum score."""
        # Start with raw OHLCV data
        df = uptrend_data.copy()

        # Add technical indicators
        df = technical_indicators.add_all_indicators(df)

        # Add momentum metrics
        df = momentum_metrics.add_all_momentum_metrics(df)

        # Verify we have all expected columns
        expected_tech = ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
                        'rsi', 'macd', 'atr']
        expected_momentum = ['roc_20', 'trend_strength', 'volume_ratio',
                            'volume_surge', 'momentum_score']

        for col in expected_tech + expected_momentum:
            assert col in df.columns

    def test_identify_strong_momentum_stocks(self, technical_indicators, momentum_metrics):
        """Test identifying strong momentum candidates."""
        # Create data for multiple stocks
        dates = pd.bdate_range(start='2024-01-01', periods=250)

        # Stock 1: Strong uptrend
        stock1 = pd.DataFrame({
            'close': np.linspace(100, 200, 250),
            'open': np.linspace(100, 200, 250),
            'high': np.linspace(102, 202, 250),
            'low': np.linspace(98, 198, 250),
            'volume': np.linspace(1000000, 2000000, 250)
        }, index=dates)

        # Stock 2: Downtrend
        stock2 = pd.DataFrame({
            'close': np.linspace(200, 100, 250),
            'open': np.linspace(200, 100, 250),
            'high': np.linspace(202, 102, 250),
            'low': np.linspace(198, 98, 250),
            'volume': [1000000] * 250
        }, index=dates)

        # Process both stocks
        for stock in [stock1, stock2]:
            stock = technical_indicators.add_all_indicators(stock)
            stock = momentum_metrics.add_all_momentum_metrics(stock)

        # Stock 1 should have higher momentum score
        assert stock1['momentum_score'].iloc[-1] > stock2['momentum_score'].iloc[-1]
        assert stock1['momentum_score'].iloc[-1] > 70  # Strong momentum
        assert stock2['momentum_score'].iloc[-1] < 40  # Weak momentum


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
