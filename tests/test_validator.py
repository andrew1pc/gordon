import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.validator import DataValidator, DataValidationError


# Fixtures

@pytest.fixture
def valid_stock_data():
    """Create valid stock OHLCV data."""
    dates = pd.bdate_range(start='2024-01-01', periods=20)
    data = {
        'open': np.random.uniform(100, 110, 20),
        'high': np.random.uniform(110, 120, 20),
        'low': np.random.uniform(90, 100, 20),
        'close': np.random.uniform(100, 110, 20),
        'volume': np.random.uniform(1000000, 2000000, 20)
    }
    df = pd.DataFrame(data, index=dates)

    # Ensure OHLCV integrity
    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc('high')] = max(
            df.iloc[i]['open'],
            df.iloc[i]['close'],
            df.iloc[i]['high']
        )
        df.iloc[i, df.columns.get_loc('low')] = min(
            df.iloc[i]['open'],
            df.iloc[i]['close'],
            df.iloc[i]['low']
        )

    return df


@pytest.fixture
def valid_crypto_data():
    """Create valid crypto OHLCV data."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    data = {
        'open': np.random.uniform(40000, 45000, 30),
        'high': np.random.uniform(45000, 50000, 30),
        'low': np.random.uniform(38000, 40000, 30),
        'close': np.random.uniform(40000, 45000, 30),
        'volume': np.random.uniform(1000, 2000, 30)
    }
    df = pd.DataFrame(data, index=dates)

    # Ensure OHLCV integrity
    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc('high')] = max(
            df.iloc[i]['open'],
            df.iloc[i]['close'],
            df.iloc[i]['high']
        )
        df.iloc[i, df.columns.get_loc('low')] = min(
            df.iloc[i]['open'],
            df.iloc[i]['close'],
            df.iloc[i]['low']
        )

    return df


@pytest.fixture
def invalid_high_low_data():
    """Create data where High < Low."""
    dates = pd.bdate_range(start='2024-01-01', periods=10)
    data = {
        'open': [100] * 10,
        'high': [95] * 10,  # High < Low (invalid)
        'low': [105] * 10,
        'close': [100] * 10,
        'volume': [1000000] * 10
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def negative_price_data():
    """Create data with negative prices."""
    dates = pd.bdate_range(start='2024-01-01', periods=10)
    data = {
        'open': [100, 100, -50, 100, 100, 100, 100, 100, 100, 100],
        'high': [110] * 10,
        'low': [90] * 10,
        'close': [105] * 10,
        'volume': [1000000] * 10
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def negative_volume_data():
    """Create data with negative volume."""
    dates = pd.bdate_range(start='2024-01-01', periods=10)
    data = {
        'open': [100] * 10,
        'high': [110] * 10,
        'low': [90] * 10,
        'close': [105] * 10,
        'volume': [1000000, 1000000, -500000, 1000000, 1000000,
                   1000000, 1000000, 1000000, 1000000, 1000000]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def data_with_gaps():
    """Create data with missing dates."""
    dates = pd.bdate_range(start='2024-01-01', periods=20)
    # Remove some dates to create gaps
    dates = dates[[0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 18, 19]]

    data = {
        'open': np.random.uniform(100, 110, len(dates)),
        'high': np.random.uniform(110, 120, len(dates)),
        'low': np.random.uniform(90, 100, len(dates)),
        'close': np.random.uniform(100, 110, len(dates)),
        'volume': np.random.uniform(1000000, 2000000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)

    # Ensure OHLCV integrity
    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc('high')] = max(
            df.iloc[i]['open'],
            df.iloc[i]['close'],
            df.iloc[i]['high']
        )
        df.iloc[i, df.columns.get_loc('low')] = min(
            df.iloc[i]['open'],
            df.iloc[i]['close'],
            df.iloc[i]['low']
        )

    return df


@pytest.fixture
def stock_split_data():
    """Create data simulating a stock split."""
    dates = pd.bdate_range(start='2024-01-01', periods=30)

    # Normal data before split
    before_split = {
        'open': [200] * 15,
        'high': [210] * 15,
        'low': [195] * 15,
        'close': [205] * 15,
        'volume': [1000000] * 15
    }

    # Split day - 2:1 split
    split_day = {
        'open': [100],
        'high': [105],
        'low': [98],
        'close': [102],
        'volume': [5000000]  # 5x volume spike
    }

    # After split
    after_split = {
        'open': [100] * 14,
        'high': [105] * 14,
        'low': [98] * 14,
        'close': [102] * 14,
        'volume': [1000000] * 14
    }

    # Combine all data
    data = {
        'open': before_split['open'] + split_day['open'] + after_split['open'],
        'high': before_split['high'] + split_day['high'] + after_split['high'],
        'low': before_split['low'] + split_day['low'] + after_split['low'],
        'close': before_split['close'] + split_day['close'] + after_split['close'],
        'volume': before_split['volume'] + split_day['volume'] + after_split['volume']
    }

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def validator():
    """Create a DataValidator instance."""
    return DataValidator(log_level=logging.WARNING)


# Tests for validate_and_clean

class TestValidateAndClean:
    """Tests for validate_and_clean method."""

    def test_valid_data_passes(self, validator, valid_stock_data):
        """Test that valid data passes all checks."""
        cleaned_df, issues = validator.validate_and_clean(valid_stock_data)

        assert cleaned_df is not None
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) == len(valid_stock_data)

    def test_empty_dataframe_raises_error(self, validator):
        """Test that empty DataFrame raises DataValidationError."""
        empty_df = pd.DataFrame()

        with pytest.raises(DataValidationError, match="empty"):
            validator.validate_and_clean(empty_df)

    def test_missing_columns_raises_error(self, validator):
        """Test that missing required columns raises error."""
        dates = pd.bdate_range(start='2024-01-01', periods=10)
        df = pd.DataFrame({'open': [100] * 10, 'close': [105] * 10}, index=dates)

        with pytest.raises(DataValidationError, match="Missing required columns"):
            validator.validate_and_clean(df)

    def test_invalid_high_low_detected(self, validator, invalid_high_low_data):
        """Test that High < Low violations are detected."""
        cleaned_df, issues = validator.validate_and_clean(invalid_high_low_data)

        # Should remove all invalid rows
        assert len(cleaned_df) < len(invalid_high_low_data)

        # Check issues list
        issue_types = [issue['type'] for issue in issues]
        assert 'high_low_violation' in issue_types

    def test_negative_prices_detected(self, validator, negative_price_data):
        """Test that negative prices are detected."""
        cleaned_df, issues = validator.validate_and_clean(negative_price_data)

        # Should remove invalid rows
        assert len(cleaned_df) < len(negative_price_data)

        # Check issues list
        issue_types = [issue['type'] for issue in issues]
        assert 'negative_price' in issue_types

    def test_negative_volume_detected(self, validator, negative_volume_data):
        """Test that negative volume is detected."""
        cleaned_df, issues = validator.validate_and_clean(negative_volume_data)

        # Should remove invalid rows
        assert len(cleaned_df) < len(negative_volume_data)

        # Check issues list
        issue_types = [issue['type'] for issue in issues]
        assert 'negative_volume' in issue_types

    def test_non_datetime_index_raises_error(self, validator):
        """Test that non-DateTimeIndex raises error."""
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [110] * 10,
            'low': [90] * 10,
            'close': [105] * 10,
            'volume': [1000000] * 10
        })

        with pytest.raises(DataValidationError, match="DateTimeIndex"):
            validator.validate_and_clean(df)


# Tests for handle_missing_data

class TestHandleMissingData:
    """Tests for handle_missing_data method."""

    def test_handles_stock_gaps(self, validator, data_with_gaps):
        """Test handling missing data for stocks."""
        result = validator.handle_missing_data(data_with_gaps, asset_type='stock')

        assert result is not None
        assert len(result) > len(data_with_gaps)  # Should have filled gaps

        # Check for gap-related issues
        issue_types = [issue['type'] for issue in validator.issues]
        assert any('gap' in itype for itype in issue_types)

    def test_handles_crypto_gaps(self, validator, valid_crypto_data):
        """Test handling missing data for crypto (7 days/week)."""
        # Remove some dates
        df_with_gaps = valid_crypto_data.iloc[[0, 1, 2, 5, 6, 7, 10, 11, 12]]

        result = validator.handle_missing_data(df_with_gaps, asset_type='crypto')

        assert result is not None
        assert len(result) >= len(df_with_gaps)

    def test_forward_fill_works(self, validator, data_with_gaps):
        """Test that forward fill preserves data correctly."""
        result = validator.handle_missing_data(data_with_gaps, asset_type='stock')

        # Check that filled values match previous values
        original_dates = data_with_gaps.index
        for date in original_dates:
            assert date in result.index
            pd.testing.assert_series_equal(
                data_with_gaps.loc[date],
                result.loc[date],
                check_names=False
            )

    def test_large_gaps_flagged(self, validator):
        """Test that large gaps (3+ days) are flagged as warnings."""
        dates = pd.bdate_range(start='2024-01-01', periods=20)
        # Create a large gap (5 days)
        dates = dates[[0, 1, 2, 8, 9, 10, 11, 12, 13, 14]]

        data = {
            'open': [100] * len(dates),
            'high': [110] * len(dates),
            'low': [90] * len(dates),
            'close': [105] * len(dates),
            'volume': [1000000] * len(dates)
        }
        df = pd.DataFrame(data, index=dates)

        result = validator.handle_missing_data(df, asset_type='stock')

        # Check for large gap warning
        issue_types = [issue['type'] for issue in validator.issues]
        assert 'large_gap_warning' in issue_types


# Tests for check_corporate_actions

class TestCheckCorporateActions:
    """Tests for check_corporate_actions method."""

    def test_detects_stock_split(self, validator, stock_split_data):
        """Test that stock splits are detected."""
        suspect_dates = validator.check_corporate_actions(stock_split_data, 'AAPL')

        assert len(suspect_dates) > 0

        # Check that volume spike or potential split was detected
        event_types = [event['type'] for event in suspect_dates]
        assert any(t in ['volume_spike', 'potential_split'] for t in event_types)

    def test_normal_data_no_alerts(self, validator, valid_stock_data):
        """Test that normal data doesn't trigger false alerts."""
        suspect_dates = validator.check_corporate_actions(valid_stock_data, 'AAPL')

        # Should have no or very few alerts for normal data
        assert len(suspect_dates) == 0

    def test_requires_minimum_data(self, validator):
        """Test that insufficient data is handled gracefully."""
        dates = pd.bdate_range(start='2024-01-01', periods=5)
        data = {
            'open': [100] * 5,
            'high': [110] * 5,
            'low': [90] * 5,
            'close': [105] * 5,
            'volume': [1000000] * 5
        }
        df = pd.DataFrame(data, index=dates)

        suspect_dates = validator.check_corporate_actions(df, 'AAPL')

        # Should return empty list for insufficient data
        assert len(suspect_dates) == 0


# Tests for standardize_format

class TestStandardizeFormat:
    """Tests for standardize_format method."""

    def test_converts_column_names_to_lowercase(self, validator):
        """Test that column names are converted to lowercase."""
        dates = pd.bdate_range(start='2024-01-01', periods=10)
        data = {
            'Open': [100] * 10,
            'High': [110] * 10,
            'Low': [90] * 10,
            'Close': [105] * 10,
            'Volume': [1000000] * 10
        }
        df = pd.DataFrame(data, index=dates)

        result = validator.standardize_format(df)

        assert all(col.islower() for col in result.columns if isinstance(col, str))
        assert 'open' in result.columns
        assert 'high' in result.columns

    def test_adds_asset_type_column(self, validator, valid_stock_data):
        """Test that asset_type column is added."""
        result = validator.standardize_format(valid_stock_data, asset_type='stock')

        assert 'asset_type' in result.columns
        assert all(result['asset_type'] == 'stock')

    def test_removes_duplicate_dates(self, validator):
        """Test that duplicate dates are removed."""
        dates = pd.bdate_range(start='2024-01-01', periods=10)
        # Add duplicate dates
        dates = dates.append(dates[:3])

        data = {
            'open': [100] * 13,
            'high': [110] * 13,
            'low': [90] * 13,
            'close': [105] * 13,
            'volume': [1000000] * 13
        }
        df = pd.DataFrame(data, index=dates)

        result = validator.standardize_format(df)

        # Should have no duplicates
        assert not result.index.duplicated().any()
        assert len(result) == 10

    def test_sorts_by_date(self, validator):
        """Test that data is sorted by date ascending."""
        dates = pd.bdate_range(start='2024-01-01', periods=10)
        # Shuffle dates
        shuffled_dates = dates[[5, 2, 8, 1, 9, 0, 3, 7, 4, 6]]

        data = {
            'open': [100] * 10,
            'high': [110] * 10,
            'low': [90] * 10,
            'close': [105] * 10,
            'volume': [1000000] * 10
        }
        df = pd.DataFrame(data, index=shuffled_dates)

        result = validator.standardize_format(df)

        # Should be sorted
        assert result.index.is_monotonic_increasing

    def test_converts_date_column_to_index(self, validator):
        """Test that date column is converted to index."""
        dates = pd.bdate_range(start='2024-01-01', periods=10)
        data = {
            'date': dates,
            'open': [100] * 10,
            'high': [110] * 10,
            'low': [90] * 10,
            'close': [105] * 10,
            'volume': [1000000] * 10
        }
        df = pd.DataFrame(data)

        result = validator.standardize_format(df)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert 'date' not in result.columns or result.index.name == 'date'


# Tests for validate_all

class TestValidateAll:
    """Tests for validate_all convenience method."""

    def test_validate_all_stocks(self, validator, valid_stock_data):
        """Test full validation pipeline for stocks."""
        cleaned_df, issues = validator.validate_all(valid_stock_data, 'AAPL', 'stock')

        assert cleaned_df is not None
        assert isinstance(cleaned_df, pd.DataFrame)
        assert 'asset_type' in cleaned_df.columns
        assert all(cleaned_df['asset_type'] == 'stock')

    def test_validate_all_crypto(self, validator, valid_crypto_data):
        """Test full validation pipeline for crypto."""
        cleaned_df, issues = validator.validate_all(valid_crypto_data, 'btcusd', 'crypto')

        assert cleaned_df is not None
        assert isinstance(cleaned_df, pd.DataFrame)
        assert 'asset_type' in cleaned_df.columns
        assert all(cleaned_df['asset_type'] == 'crypto')

    def test_validate_all_with_issues(self, validator, data_with_gaps):
        """Test that validate_all detects and reports issues."""
        cleaned_df, issues = validator.validate_all(data_with_gaps, 'AAPL', 'stock')

        assert len(issues) > 0
        assert cleaned_df is not None

    def test_get_issues_summary(self, validator, data_with_gaps):
        """Test get_issues_summary method."""
        validator.validate_all(data_with_gaps, 'AAPL', 'stock')
        summary = validator.get_issues_summary()

        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_clear_issues(self, validator, valid_stock_data):
        """Test clear_issues method."""
        validator.validate_all(valid_stock_data, 'AAPL', 'stock')
        assert len(validator.issues) > 0

        validator.clear_issues()
        assert len(validator.issues) == 0


# Edge case tests

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row_data(self, validator):
        """Test handling of single row DataFrame."""
        dates = pd.bdate_range(start='2024-01-01', periods=1)
        data = {
            'open': [100],
            'high': [110],
            'low': [90],
            'close': [105],
            'volume': [1000000]
        }
        df = pd.DataFrame(data, index=dates)

        cleaned_df, issues = validator.validate_and_clean(df)

        assert len(cleaned_df) == 1

    def test_all_same_values(self, validator):
        """Test data with all same values."""
        dates = pd.bdate_range(start='2024-01-01', periods=10)
        data = {
            'open': [100] * 10,
            'high': [100] * 10,
            'low': [100] * 10,
            'close': [100] * 10,
            'volume': [1000000] * 10
        }
        df = pd.DataFrame(data, index=dates)

        cleaned_df, issues = validator.validate_and_clean(df)

        # Should pass validation even with flat prices
        assert len(cleaned_df) == 10

    def test_zero_volume(self, validator):
        """Test that zero volume is acceptable."""
        dates = pd.bdate_range(start='2024-01-01', periods=10)
        data = {
            'open': [100] * 10,
            'high': [110] * 10,
            'low': [90] * 10,
            'close': [105] * 10,
            'volume': [0] * 10  # Zero volume
        }
        df = pd.DataFrame(data, index=dates)

        cleaned_df, issues = validator.validate_and_clean(df)

        # Zero volume should be valid (>= 0)
        assert len(cleaned_df) == 10

    def test_very_large_numbers(self, validator):
        """Test handling of very large numbers."""
        dates = pd.bdate_range(start='2024-01-01', periods=10)
        data = {
            'open': [1e10] * 10,
            'high': [1.1e10] * 10,
            'low': [0.9e10] * 10,
            'close': [1.05e10] * 10,
            'volume': [1e15] * 10
        }
        df = pd.DataFrame(data, index=dates)

        cleaned_df, issues = validator.validate_and_clean(df)

        # Should handle large numbers
        assert len(cleaned_df) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
