import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indicators.technical import TechnicalIndicators


# Fixtures

@pytest.fixture
def technical_indicators():
    """Create a TechnicalIndicators instance."""
    return TechnicalIndicators()


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data with trending up pattern."""
    dates = pd.bdate_range(start='2024-01-01', periods=100)
    # Create uptrend with some noise
    closes = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
    highs = closes + np.random.uniform(1, 3, 100)
    lows = closes - np.random.uniform(1, 3, 100)
    opens = closes + np.random.uniform(-2, 2, 100)
    volumes = np.random.uniform(1000000, 2000000, 100)

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def trending_down_data():
    """Create sample data with trending down pattern."""
    dates = pd.bdate_range(start='2024-01-01', periods=100)
    closes = np.linspace(150, 100, 100) + np.random.normal(0, 2, 100)
    highs = closes + np.random.uniform(1, 3, 100)
    lows = closes - np.random.uniform(1, 3, 100)
    opens = closes + np.random.uniform(-2, 2, 100)
    volumes = np.random.uniform(1000000, 2000000, 100)

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
    """Create sample data with ranging (sideways) pattern."""
    dates = pd.bdate_range(start='2024-01-01', periods=100)
    closes = 125 + np.random.normal(0, 3, 100)
    highs = closes + np.random.uniform(1, 3, 100)
    lows = closes - np.random.uniform(1, 3, 100)
    opens = closes + np.random.uniform(-2, 2, 100)
    volumes = np.random.uniform(1000000, 2000000, 100)

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


@pytest.fixture
def simple_data():
    """Create simple data for manual verification."""
    dates = pd.bdate_range(start='2024-01-01', periods=20)
    closes = [100, 102, 101, 103, 105, 107, 106, 108, 110, 109,
              111, 113, 112, 114, 116, 115, 117, 119, 118, 120]

    df = pd.DataFrame({
        'open': closes,
        'high': [c + 2 for c in closes],
        'low': [c - 2 for c in closes],
        'close': closes,
        'volume': [1000000] * 20
    }, index=dates)

    return df


@pytest.fixture
def single_row_data():
    """Create single row DataFrame."""
    dates = pd.bdate_range(start='2024-01-01', periods=1)
    df = pd.DataFrame({
        'open': [100],
        'high': [105],
        'low': [98],
        'close': [102],
        'volume': [1000000]
    }, index=dates)

    return df


@pytest.fixture
def all_same_values_data():
    """Create data with all same values."""
    dates = pd.bdate_range(start='2024-01-01', periods=50)
    df = pd.DataFrame({
        'open': [100] * 50,
        'high': [100] * 50,
        'low': [100] * 50,
        'close': [100] * 50,
        'volume': [1000000] * 50
    }, index=dates)

    return df


# Tests for SMA

class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_calculation(self, technical_indicators, simple_data):
        """Test SMA calculation with manual verification."""
        result = technical_indicators.calculate_sma(simple_data, 5)

        assert 'sma_5' in result.columns
        assert result['sma_5'].notna().sum() == len(result) - 4  # First 4 will be NaN

        # Manual calculation for 5th value: (100+102+101+103+105)/5 = 102.2
        expected_5th = (100 + 102 + 101 + 103 + 105) / 5
        assert abs(result['sma_5'].iloc[4] - expected_5th) < 0.01

    def test_sma_column_naming(self, technical_indicators, sample_ohlcv_data):
        """Test that SMA column is named correctly."""
        result = technical_indicators.calculate_sma(sample_ohlcv_data, 20)
        assert 'sma_20' in result.columns

        result = technical_indicators.calculate_sma(sample_ohlcv_data, 50)
        assert 'sma_50' in result.columns

    def test_sma_empty_dataframe(self, technical_indicators):
        """Test SMA with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            technical_indicators.calculate_sma(empty_df, 20)

    def test_sma_missing_close(self, technical_indicators):
        """Test SMA with missing close column."""
        df = pd.DataFrame({'open': [100, 101, 102]})
        with pytest.raises(ValueError, match="close"):
            technical_indicators.calculate_sma(df, 5)

    def test_sma_period_too_large(self, technical_indicators, simple_data):
        """Test SMA when period exceeds data length."""
        with pytest.raises(ValueError, match="greater than data length"):
            technical_indicators.calculate_sma(simple_data, 100)

    def test_sma_period_invalid(self, technical_indicators, simple_data):
        """Test SMA with invalid period."""
        with pytest.raises(ValueError, match="at least 1"):
            technical_indicators.calculate_sma(simple_data, 0)


# Tests for EMA

class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_calculation(self, technical_indicators, sample_ohlcv_data):
        """Test EMA calculation."""
        result = technical_indicators.calculate_ema(sample_ohlcv_data, 12)

        assert 'ema_12' in result.columns
        assert result['ema_12'].notna().all()  # EMA should have values for all rows

    def test_ema_reacts_faster_than_sma(self, technical_indicators):
        """Test that EMA reacts faster than SMA to price changes."""
        # Create data with sudden price jump
        dates = pd.bdate_range(start='2024-01-01', periods=50)
        closes = [100] * 25 + [110] * 25  # Sudden jump at midpoint

        df = pd.DataFrame({
            'open': closes,
            'high': [c + 2 for c in closes],
            'low': [c - 2 for c in closes],
            'close': closes,
            'volume': [1000000] * 50
        }, index=dates)

        result = technical_indicators.calculate_sma(df, 10)
        result = technical_indicators.calculate_ema(result, 10)

        # After the jump, EMA should be closer to the new price than SMA
        # Check at position 30 (5 periods after jump)
        idx = 30
        assert result['ema_10'].iloc[idx] > result['sma_10'].iloc[idx]

    def test_ema_column_naming(self, technical_indicators, sample_ohlcv_data):
        """Test EMA column naming."""
        result = technical_indicators.calculate_ema(sample_ohlcv_data, 26)
        assert 'ema_26' in result.columns

    def test_ema_empty_dataframe(self, technical_indicators):
        """Test EMA with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            technical_indicators.calculate_ema(empty_df, 12)


# Tests for RSI

class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_calculation(self, technical_indicators, sample_ohlcv_data):
        """Test RSI calculation."""
        result = technical_indicators.calculate_rsi(sample_ohlcv_data, 14)

        assert 'rsi' in result.columns
        assert result['rsi'].notna().sum() > 0

    def test_rsi_bounds(self, technical_indicators, sample_ohlcv_data):
        """Test that RSI stays between 0 and 100."""
        result = technical_indicators.calculate_rsi(sample_ohlcv_data, 14)

        # Filter out NaN values
        rsi_values = result['rsi'].dropna()

        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100

    def test_rsi_uptrend_high(self, technical_indicators):
        """Test that RSI is high in strong uptrend."""
        dates = pd.bdate_range(start='2024-01-01', periods=50)
        # Strong uptrend
        closes = np.linspace(100, 150, 50)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 2,
            'close': closes,
            'volume': [1000000] * 50
        }, index=dates)

        result = technical_indicators.calculate_rsi(df, 14)

        # RSI should be high (>60) in last 10 periods
        assert result['rsi'].iloc[-10:].mean() > 60

    def test_rsi_downtrend_low(self, technical_indicators):
        """Test that RSI is low in strong downtrend."""
        dates = pd.bdate_range(start='2024-01-01', periods=50)
        # Strong downtrend
        closes = np.linspace(150, 100, 50)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 2,
            'close': closes,
            'volume': [1000000] * 50
        }, index=dates)

        result = technical_indicators.calculate_rsi(df, 14)

        # RSI should be low (<40) in last 10 periods
        assert result['rsi'].iloc[-10:].mean() < 40

    def test_rsi_flat_prices(self, technical_indicators, all_same_values_data):
        """Test RSI with flat prices (no movement)."""
        result = technical_indicators.calculate_rsi(all_same_values_data, 14)

        # RSI should be around 50 (neutral) when there's no price movement
        rsi_values = result['rsi'].dropna()
        assert rsi_values.mean() == 50


# Tests for MACD

class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_calculation(self, technical_indicators, sample_ohlcv_data):
        """Test MACD calculation."""
        result = technical_indicators.calculate_macd(sample_ohlcv_data)

        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_histogram' in result.columns

    def test_macd_components_relationship(self, technical_indicators, sample_ohlcv_data):
        """Test that MACD histogram = MACD - Signal."""
        result = technical_indicators.calculate_macd(sample_ohlcv_data)

        # Filter out NaN values
        valid_rows = result[['macd', 'macd_signal', 'macd_histogram']].dropna()

        # Verify histogram = macd - signal
        calculated_histogram = valid_rows['macd'] - valid_rows['macd_signal']
        pd.testing.assert_series_equal(
            calculated_histogram,
            valid_rows['macd_histogram'],
            check_names=False
        )

    def test_macd_uptrend(self, technical_indicators):
        """Test MACD in uptrend."""
        dates = pd.bdate_range(start='2024-01-01', periods=100)
        closes = np.linspace(100, 150, 100)

        df = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 2,
            'close': closes,
            'volume': [1000000] * 100
        }, index=dates)

        result = technical_indicators.calculate_macd(df)

        # In uptrend, MACD should eventually be positive
        assert result['macd'].iloc[-10:].mean() > 0

    def test_macd_invalid_periods(self, technical_indicators, sample_ohlcv_data):
        """Test MACD with invalid period configuration."""
        with pytest.raises(ValueError, match="Fast period must be less than slow"):
            technical_indicators.calculate_macd(sample_ohlcv_data, fast=26, slow=12)


# Tests for ATR

class TestATR:
    """Tests for Average True Range."""

    def test_atr_calculation(self, technical_indicators, sample_ohlcv_data):
        """Test ATR calculation."""
        result = technical_indicators.calculate_atr(sample_ohlcv_data, 14)

        assert 'atr' in result.columns
        assert result['atr'].notna().sum() > 0

    def test_atr_positive_values(self, technical_indicators, sample_ohlcv_data):
        """Test that ATR is always positive."""
        result = technical_indicators.calculate_atr(sample_ohlcv_data, 14)

        atr_values = result['atr'].dropna()
        assert (atr_values >= 0).all()

    def test_atr_captures_volatility(self, technical_indicators):
        """Test that ATR is higher in volatile markets."""
        dates = pd.bdate_range(start='2024-01-01', periods=100)

        # Low volatility data
        low_vol_closes = 100 + np.random.normal(0, 0.5, 100)
        df_low_vol = pd.DataFrame({
            'open': low_vol_closes,
            'high': low_vol_closes + 0.5,
            'low': low_vol_closes - 0.5,
            'close': low_vol_closes,
            'volume': [1000000] * 100
        }, index=dates)

        # High volatility data
        high_vol_closes = 100 + np.random.normal(0, 5, 100)
        df_high_vol = pd.DataFrame({
            'open': high_vol_closes,
            'high': high_vol_closes + 5,
            'low': high_vol_closes - 5,
            'close': high_vol_closes,
            'volume': [1000000] * 100
        }, index=dates)

        result_low = technical_indicators.calculate_atr(df_low_vol, 14)
        result_high = technical_indicators.calculate_atr(df_high_vol, 14)

        # High volatility should have higher ATR
        assert result_high['atr'].iloc[-10:].mean() > result_low['atr'].iloc[-10:].mean()

    def test_atr_missing_columns(self, technical_indicators):
        """Test ATR with missing columns."""
        df = pd.DataFrame({'close': [100, 101, 102]})
        with pytest.raises(ValueError, match="must contain"):
            technical_indicators.calculate_atr(df, 14)


# Tests for Volume MA

class TestVolumeMA:
    """Tests for Volume Moving Average."""

    def test_volume_ma_calculation(self, technical_indicators, sample_ohlcv_data):
        """Test volume MA calculation."""
        result = technical_indicators.calculate_volume_ma(sample_ohlcv_data, 20)

        assert 'volume_ma_20' in result.columns
        assert 'volume_ratio' in result.columns

    def test_volume_ratio_calculation(self, technical_indicators):
        """Test volume ratio calculation."""
        dates = pd.bdate_range(start='2024-01-01', periods=30)
        volumes = [1000000] * 25 + [3000000] * 5  # Volume spike at end

        df = pd.DataFrame({
            'open': [100] * 30,
            'high': [105] * 30,
            'low': [95] * 30,
            'close': [100] * 30,
            'volume': volumes
        }, index=dates)

        result = technical_indicators.calculate_volume_ma(df, 20)

        # Last 5 days should have volume ratio > 1.5 (volume spike)
        assert result['volume_ratio'].iloc[-5:].mean() > 1.5

    def test_volume_ratio_positive(self, technical_indicators, sample_ohlcv_data):
        """Test that volume ratio is always positive."""
        result = technical_indicators.calculate_volume_ma(sample_ohlcv_data, 20)

        assert (result['volume_ratio'] > 0).all()


# Tests for add_all_indicators

class TestAddAllIndicators:
    """Tests for add_all_indicators convenience method."""

    def test_add_all_indicators(self, technical_indicators, sample_ohlcv_data):
        """Test that all indicators are added."""
        result = technical_indicators.add_all_indicators(sample_ohlcv_data)

        # Check that all expected columns exist
        expected_columns = [
            'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26',
            'rsi',
            'macd', 'macd_signal', 'macd_histogram',
            'atr',
            'volume_ma_20', 'volume_ratio'
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_all_indicators_insufficient_data(self, technical_indicators):
        """Test add_all_indicators with insufficient data."""
        dates = pd.bdate_range(start='2024-01-01', periods=15)
        df = pd.DataFrame({
            'open': [100] * 15,
            'high': [105] * 15,
            'low': [95] * 15,
            'close': [100] * 15,
            'volume': [1000000] * 15
        }, index=dates)

        result = technical_indicators.add_all_indicators(df)

        # Should have at least some indicators (even if not all)
        assert 'sma_20' not in result.columns  # Not enough data for 20-day SMA
        # But should have others that fit

    def test_add_all_indicators_empty_df(self, technical_indicators):
        """Test add_all_indicators with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            technical_indicators.add_all_indicators(empty_df)


# Edge case tests

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row_handling(self, technical_indicators, single_row_data):
        """Test indicators with single row data."""
        # Most indicators should handle single row gracefully
        with pytest.raises(ValueError):
            technical_indicators.calculate_sma(single_row_data, 5)

    def test_all_same_values(self, technical_indicators, all_same_values_data):
        """Test indicators with flat prices."""
        result = technical_indicators.calculate_sma(all_same_values_data, 20)
        # SMA of flat prices should equal the price
        assert (result['sma_20'].dropna() == 100).all()

        result = technical_indicators.calculate_rsi(all_same_values_data, 14)
        # RSI of flat prices should be 50
        assert (result['rsi'].dropna() == 50).all()

    def test_nan_handling(self, technical_indicators):
        """Test handling of NaN values in input data."""
        dates = pd.bdate_range(start='2024-01-01', periods=50)
        closes = [100 + i for i in range(50)]
        closes[25] = np.nan  # Insert NaN

        df = pd.DataFrame({
            'open': closes,
            'high': [c + 2 if not pd.isna(c) else np.nan for c in closes],
            'low': [c - 2 if not pd.isna(c) else np.nan for c in closes],
            'close': closes,
            'volume': [1000000] * 50
        }, index=dates)

        # Indicators should handle NaN gracefully
        result = technical_indicators.calculate_sma(df, 10)
        assert 'sma_10' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
