import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from datetime import timedelta


# Set up logging
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for critical data validation issues."""
    pass


class DataValidator:
    """Validator for financial market data quality checks."""

    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize the DataValidator.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        self.issues: List[Dict] = []
        logger.setLevel(log_level)

    def validate_and_clean(
        self,
        df: pd.DataFrame,
        asset_type: str = 'stock'
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Validate and clean financial data.

        Args:
            df: DataFrame with OHLCV data and DateTimeIndex
            asset_type: Type of asset ('stock' or 'crypto')

        Returns:
            Tuple of (cleaned_df, issues_list)
            - cleaned_df: DataFrame with invalid rows removed
            - issues_list: List of dictionaries describing data quality issues

        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        self.issues = []

        # Validate input
        if df is None or df.empty:
            logger.error("DataFrame is empty or None")
            raise DataValidationError("DataFrame is empty or None")

        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise DataValidationError(f"Missing required columns: {missing_columns}")

        # Check if index is DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be a DateTimeIndex")
            raise DataValidationError("DataFrame index must be a DateTimeIndex")

        # Make a copy to avoid modifying original
        cleaned_df = df.copy()

        # Track indices to drop
        indices_to_drop = set()

        # 1. Check for missing dates/gaps in the date index
        self._check_date_gaps(cleaned_df, asset_type)

        # 2. Validate OHLCV integrity
        indices_to_drop.update(self._validate_high_low(cleaned_df))
        indices_to_drop.update(self._validate_high_vs_open_close(cleaned_df))
        indices_to_drop.update(self._validate_low_vs_open_close(cleaned_df))
        indices_to_drop.update(self._validate_volume(cleaned_df))
        indices_to_drop.update(self._validate_prices(cleaned_df))

        # Drop invalid rows
        if indices_to_drop:
            message = f'Removed {len(indices_to_drop)} rows with data quality issues'
            logger.warning(message)
            self.issues.append({
                'type': 'invalid_rows_removed',
                'count': len(indices_to_drop),
                'message': message
            })
            cleaned_df = cleaned_df.drop(index=list(indices_to_drop))

        logger.info(f"Validation complete: {len(self.issues)} issues found")
        return cleaned_df, self.issues

    def _check_date_gaps(self, df: pd.DataFrame, asset_type: str) -> None:
        """Check for missing dates/gaps in the date index."""
        if len(df) < 2:
            return

        # Calculate expected frequency based on asset type
        if asset_type == 'stock':
            # For stocks, expect business days (Mon-Fri)
            # Calculate the date range excluding weekends
            date_range = pd.bdate_range(start=df.index.min(), end=df.index.max())
            expected_dates = set(date_range.date)
            actual_dates = set(df.index.date)
            missing_dates = expected_dates - actual_dates

            if missing_dates:
                message = f'Found {len(missing_dates)} missing business days (may include holidays)'
                logger.info(message)
                self.issues.append({
                    'type': 'missing_dates',
                    'count': len(missing_dates),
                    'message': message,
                    'dates': sorted(list(missing_dates))[:10]  # Show first 10
                })
        else:
            # For crypto, expect daily data (7 days/week)
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            expected_dates = set(date_range.date)
            actual_dates = set(df.index.date)
            missing_dates = expected_dates - actual_dates

            if missing_dates:
                self.issues.append({
                    'type': 'missing_dates',
                    'count': len(missing_dates),
                    'message': f'Found {len(missing_dates)} missing dates',
                    'dates': sorted(list(missing_dates))[:10]  # Show first 10
                })

    def _validate_high_low(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """Validate that High >= Low for every row."""
        invalid_mask = df['high'] < df['low']
        invalid_indices = df[invalid_mask].index.tolist()

        if len(invalid_indices) > 0:
            message = f'Found {len(invalid_indices)} rows where High < Low'
            logger.warning(message)
            self.issues.append({
                'type': 'high_low_violation',
                'count': len(invalid_indices),
                'message': message,
                'dates': [str(idx) for idx in invalid_indices[:10]]  # Show first 10
            })

        return invalid_indices

    def _validate_high_vs_open_close(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """Validate that High >= Open and High >= Close."""
        invalid_open_mask = df['high'] < df['open']
        invalid_close_mask = df['high'] < df['close']
        invalid_mask = invalid_open_mask | invalid_close_mask
        invalid_indices = df[invalid_mask].index.tolist()

        if len(invalid_indices) > 0:
            self.issues.append({
                'type': 'high_validation',
                'count': len(invalid_indices),
                'message': f'Found {len(invalid_indices)} rows where High < Open or High < Close',
                'dates': [str(idx) for idx in invalid_indices[:10]]
            })

        return invalid_indices

    def _validate_low_vs_open_close(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """Validate that Low <= Open and Low <= Close."""
        invalid_open_mask = df['low'] > df['open']
        invalid_close_mask = df['low'] > df['close']
        invalid_mask = invalid_open_mask | invalid_close_mask
        invalid_indices = df[invalid_mask].index.tolist()

        if len(invalid_indices) > 0:
            self.issues.append({
                'type': 'low_validation',
                'count': len(invalid_indices),
                'message': f'Found {len(invalid_indices)} rows where Low > Open or Low > Close',
                'dates': [str(idx) for idx in invalid_indices[:10]]
            })

        return invalid_indices

    def _validate_volume(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """Validate that Volume >= 0."""
        invalid_mask = df['volume'] < 0
        invalid_indices = df[invalid_mask].index.tolist()

        if len(invalid_indices) > 0:
            self.issues.append({
                'type': 'negative_volume',
                'count': len(invalid_indices),
                'message': f'Found {len(invalid_indices)} rows with negative volume',
                'dates': [str(idx) for idx in invalid_indices[:10]]
            })

        return invalid_indices

    def _validate_prices(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """Validate that all prices are non-negative."""
        price_columns = ['open', 'high', 'low', 'close']
        invalid_indices = []

        for col in price_columns:
            invalid_mask = df[col] < 0
            col_invalid_indices = df[invalid_mask].index.tolist()

            if len(col_invalid_indices) > 0:
                self.issues.append({
                    'type': 'negative_price',
                    'column': col,
                    'count': len(col_invalid_indices),
                    'message': f'Found {len(col_invalid_indices)} rows with negative {col} price',
                    'dates': [str(idx) for idx in col_invalid_indices[:10]]
                })
                invalid_indices.extend(col_invalid_indices)

        return list(set(invalid_indices))  # Remove duplicates

    def handle_missing_data(
        self,
        df: pd.DataFrame,
        asset_type: str = 'stock'
    ) -> pd.DataFrame:
        """
        Handle missing data by detecting gaps and forward-filling small gaps.

        Args:
            df: DataFrame with OHLCV data and DateTimeIndex
            asset_type: Type of asset ('stock' or 'crypto')

        Returns:
            DataFrame with missing data handled

        Raises:
            ValueError: If DataFrame is empty or index is not DateTimeIndex
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DateTimeIndex")

        # Make a copy to avoid modifying original
        cleaned_df = df.copy()

        # Sort by index to ensure chronological order
        cleaned_df = cleaned_df.sort_index()

        # Generate expected date range based on asset type
        if asset_type == 'stock':
            # For stocks, expect business days (Mon-Fri)
            expected_range = pd.bdate_range(
                start=cleaned_df.index.min(),
                end=cleaned_df.index.max()
            )
        else:
            # For crypto, expect continuous daily data
            expected_range = pd.date_range(
                start=cleaned_df.index.min(),
                end=cleaned_df.index.max(),
                freq='D'
            )

        # Reindex to the expected range
        cleaned_df = cleaned_df.reindex(expected_range)

        # Find missing dates
        missing_mask = cleaned_df['close'].isna()
        missing_dates = cleaned_df[missing_mask].index

        if len(missing_dates) == 0:
            # No missing data
            return cleaned_df

        # Analyze gaps
        gap_info = self._analyze_gaps(missing_dates, expected_range)

        # Handle gaps based on size
        for gap in gap_info:
            gap_size = gap['size']
            gap_start = gap['start']
            gap_end = gap['end']

            if gap_size <= 2:
                # Small gap (1-2 days): forward fill
                self.issues.append({
                    'type': 'gap_filled',
                    'size': gap_size,
                    'start': str(gap_start),
                    'end': str(gap_end),
                    'message': f'Forward-filled {gap_size}-day gap from {gap_start.date()} to {gap_end.date()}'
                })
            else:
                # Large gap (3+ days): flag as warning
                self.issues.append({
                    'type': 'large_gap_warning',
                    'size': gap_size,
                    'start': str(gap_start),
                    'end': str(gap_end),
                    'message': f'WARNING: Large {gap_size}-day gap from {gap_start.date()} to {gap_end.date()} - forward-filled but may indicate data issue'
                })

        # Forward fill all gaps
        cleaned_df = cleaned_df.fillna(method='ffill')

        # If there are still NaN values at the beginning (no previous data to forward fill)
        # we need to backfill or drop them
        if cleaned_df.isna().any().any():
            initial_nas = cleaned_df['close'].isna().sum()
            if initial_nas > 0:
                self.issues.append({
                    'type': 'initial_missing_data',
                    'count': initial_nas,
                    'message': f'Removed {initial_nas} initial rows with no data to fill'
                })
                cleaned_df = cleaned_df.dropna()

        return cleaned_df

    def _analyze_gaps(
        self,
        missing_dates: pd.DatetimeIndex,
        expected_range: pd.DatetimeIndex
    ) -> List[Dict]:
        """
        Analyze gaps in the date sequence.

        Args:
            missing_dates: DateTimeIndex of missing dates
            expected_range: Full expected date range

        Returns:
            List of dictionaries with gap information
        """
        if len(missing_dates) == 0:
            return []

        gaps = []
        gap_start = missing_dates[0]
        gap_size = 1

        for i in range(1, len(missing_dates)):
            # Check if current date is consecutive to previous
            prev_date = missing_dates[i - 1]
            curr_date = missing_dates[i]

            # Find the index difference in expected range
            prev_idx = expected_range.get_loc(prev_date)
            curr_idx = expected_range.get_loc(curr_date)

            if curr_idx == prev_idx + 1:
                # Consecutive missing date
                gap_size += 1
            else:
                # Gap ended, record it
                gaps.append({
                    'start': gap_start,
                    'end': missing_dates[i - 1],
                    'size': gap_size
                })
                # Start new gap
                gap_start = curr_date
                gap_size = 1

        # Add the last gap
        gaps.append({
            'start': gap_start,
            'end': missing_dates[-1],
            'size': gap_size
        })

        return gaps

    def check_corporate_actions(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> List[Dict]:
        """
        Detect potential corporate actions (stock splits, dividends, etc.).

        Args:
            df: DataFrame with OHLCV data and DateTimeIndex
            ticker: Stock ticker symbol

        Returns:
            List of dictionaries with suspect dates and details

        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if len(df) < 10:
            # Need at least 10 days of data to calculate meaningful averages
            return []

        required_columns = ['close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        suspect_dates = []

        # Calculate rolling average volume (20-day window)
        df['volume_avg'] = df['volume'].rolling(window=20, min_periods=5).mean()

        # Calculate daily price and volume changes
        df['price_change_pct'] = df['close'].pct_change() * 100
        df['volume_change_pct'] = df['volume'].pct_change() * 100

        # Detect potential stock splits
        for idx in range(1, len(df)):
            date = df.index[idx]
            prev_date = df.index[idx - 1]

            current_price = df['close'].iloc[idx]
            prev_price = df['close'].iloc[idx - 1]
            current_volume = df['volume'].iloc[idx]
            volume_avg = df['volume_avg'].iloc[idx]

            price_change = df['price_change_pct'].iloc[idx]
            volume_change = df['volume_change_pct'].iloc[idx]

            # Skip if we have NaN values
            if pd.isna(price_change) or pd.isna(volume_change) or pd.isna(volume_avg):
                continue

            # Detection criteria:
            # 1. Sudden volume spike (5x+ average)
            if current_volume >= 5 * volume_avg:
                suspect_dates.append({
                    'date': str(date),
                    'ticker': ticker,
                    'type': 'volume_spike',
                    'severity': 'warning',
                    'price': current_price,
                    'volume': int(current_volume),
                    'volume_avg': int(volume_avg),
                    'volume_ratio': round(current_volume / volume_avg, 2),
                    'message': f'{ticker}: Abnormal volume spike on {date.date()} - '
                               f'volume is {round(current_volume / volume_avg, 1)}x the 20-day average'
                })

            # 2. Large price drop (>25%) with volume spike
            if price_change <= -25 and current_volume >= 2 * volume_avg:
                suspect_dates.append({
                    'date': str(date),
                    'ticker': ticker,
                    'type': 'potential_split',
                    'severity': 'high',
                    'price_change_pct': round(price_change, 2),
                    'volume_ratio': round(current_volume / volume_avg, 2),
                    'prev_price': round(prev_price, 2),
                    'current_price': round(current_price, 2),
                    'message': f'{ticker}: Potential stock split on {date.date()} - '
                               f'price dropped {abs(round(price_change, 1))}% with {round(current_volume / volume_avg, 1)}x volume'
                })

            # 3. Large price jump (>25%) with volume spike (reverse split)
            if price_change >= 25 and current_volume >= 2 * volume_avg:
                suspect_dates.append({
                    'date': str(date),
                    'ticker': ticker,
                    'type': 'potential_reverse_split',
                    'severity': 'high',
                    'price_change_pct': round(price_change, 2),
                    'volume_ratio': round(current_volume / volume_avg, 2),
                    'prev_price': round(prev_price, 2),
                    'current_price': round(current_price, 2),
                    'message': f'{ticker}: Potential reverse split on {date.date()} - '
                               f'price jumped {round(price_change, 1)}% with {round(current_volume / volume_avg, 1)}x volume'
                })

            # 4. Price jump/drop >25% with opposite volume change (unusual pattern)
            if abs(price_change) >= 25:
                # Price dropped but volume also dropped significantly
                if price_change < 0 and volume_change < -50:
                    suspect_dates.append({
                        'date': str(date),
                        'ticker': ticker,
                        'type': 'unusual_pattern',
                        'severity': 'warning',
                        'price_change_pct': round(price_change, 2),
                        'volume_change_pct': round(volume_change, 2),
                        'message': f'{ticker}: Unusual pattern on {date.date()} - '
                                   f'large price drop ({round(price_change, 1)}%) with volume drop ({round(volume_change, 1)}%)'
                    })
                # Price jumped but volume dropped significantly
                elif price_change > 0 and volume_change < -50:
                    suspect_dates.append({
                        'date': str(date),
                        'ticker': ticker,
                        'type': 'unusual_pattern',
                        'severity': 'warning',
                        'price_change_pct': round(price_change, 2),
                        'volume_change_pct': round(volume_change, 2),
                        'message': f'{ticker}: Unusual pattern on {date.date()} - '
                                   f'large price jump ({round(price_change, 1)}%) with volume drop ({round(volume_change, 1)}%)'
                    })

        # Clean up temporary columns
        df.drop(['volume_avg', 'price_change_pct', 'volume_change_pct'], axis=1, inplace=True)

        # Add to issues list
        if suspect_dates:
            self.issues.append({
                'type': 'corporate_actions_detected',
                'count': len(suspect_dates),
                'message': f'Detected {len(suspect_dates)} potential corporate actions or data anomalies',
                'details': suspect_dates
            })

        return suspect_dates

    def standardize_format(
        self,
        df: pd.DataFrame,
        asset_type: str = 'stock'
    ) -> pd.DataFrame:
        """
        Standardize DataFrame format for consistent processing.

        Args:
            df: DataFrame with OHLCV data
            asset_type: Type of asset ('stock' or 'crypto')

        Returns:
            Standardized DataFrame with consistent format

        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        # Make a copy to avoid modifying original
        standardized_df = df.copy()

        # 1. Ensure column names are lowercase
        standardized_df.columns = standardized_df.columns.str.lower()

        # 2. Check for required columns (case-insensitive now)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in standardized_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # 3. Convert date index to datetime if not already
        if not isinstance(standardized_df.index, pd.DatetimeIndex):
            # Check if there's a 'date' column
            if 'date' in standardized_df.columns:
                standardized_df['date'] = pd.to_datetime(standardized_df['date'])
                standardized_df.set_index('date', inplace=True)
            else:
                # Try to convert the index to datetime
                try:
                    standardized_df.index = pd.to_datetime(standardized_df.index)
                except Exception as e:
                    raise ValueError(f"Unable to convert index to datetime: {e}")

        # Name the index if not already named
        if standardized_df.index.name is None:
            standardized_df.index.name = 'date'

        # 4. Sort by date ascending
        standardized_df = standardized_df.sort_index()

        # 5. Remove duplicate dates (keep first)
        duplicate_count = standardized_df.index.duplicated().sum()
        if duplicate_count > 0:
            self.issues.append({
                'type': 'duplicate_dates_removed',
                'count': duplicate_count,
                'message': f'Removed {duplicate_count} duplicate dates (kept first occurrence)'
            })
            standardized_df = standardized_df[~standardized_df.index.duplicated(keep='first')]

        # 6. Add 'asset_type' column
        standardized_df['asset_type'] = asset_type

        # 7. Reorder columns to have OHLCV first, then other columns
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        other_cols = [col for col in standardized_df.columns if col not in ohlcv_cols]
        standardized_df = standardized_df[ohlcv_cols + other_cols]

        return standardized_df

    def validate_all(
        self,
        df: pd.DataFrame,
        ticker: str,
        asset_type: str = 'stock'
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Convenience method that runs all validations in sequence.

        This method performs a complete data validation pipeline:
        1. Standardizes the DataFrame format
        2. Validates and cleans OHLCV data
        3. Handles missing data with forward-filling
        4. Checks for corporate actions (stocks only)

        Args:
            df: DataFrame with OHLCV data
            ticker: Stock/crypto ticker symbol
            asset_type: Type of asset ('stock' or 'crypto')

        Returns:
            Tuple of (fully_cleaned_df, all_issues_list)
            - fully_cleaned_df: DataFrame after all validations
            - all_issues_list: Comprehensive list of all issues found

        Raises:
            DataValidationError: If critical validation issues are encountered

        Example:
            >>> validator = DataValidator()
            >>> cleaned_df, issues = validator.validate_all(raw_df, 'AAPL', 'stock')
            >>> print(f"Found {len(issues)} issues")
            >>> print(cleaned_df.head())
        """
        logger.info(f"Starting full validation for {ticker} ({asset_type})")
        self.issues = []

        try:
            # Step 1: Standardize format
            logger.info("Step 1/4: Standardizing DataFrame format")
            df = self.standardize_format(df, asset_type)

            # Step 2: Validate and clean OHLCV data
            logger.info("Step 2/4: Validating OHLCV data integrity")
            df, _ = self.validate_and_clean(df, asset_type)

            # Step 3: Handle missing data
            logger.info("Step 3/4: Handling missing data")
            df = self.handle_missing_data(df, asset_type)

            # Step 4: Check for corporate actions (stocks only)
            if asset_type == 'stock':
                logger.info("Step 4/4: Checking for corporate actions")
                self.check_corporate_actions(df, ticker)
            else:
                logger.info("Step 4/4: Skipping corporate actions check (crypto asset)")

            logger.info(f"Validation complete for {ticker}: {len(self.issues)} total issues found")

            # Log summary
            if self.issues:
                logger.info("Issue Summary:")
                issue_types = {}
                for issue in self.issues:
                    issue_type = issue.get('type', 'unknown')
                    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

                for issue_type, count in issue_types.items():
                    logger.info(f"  - {issue_type}: {count}")
            else:
                logger.info("No issues found - data is clean!")

            return df, self.issues

        except DataValidationError as e:
            logger.error(f"Critical validation error for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation for {ticker}: {e}")
            raise DataValidationError(f"Validation failed for {ticker}: {e}")

    def get_issues_summary(self) -> Dict[str, int]:
        """
        Get a summary of all issues found during validation.

        Returns:
            Dictionary mapping issue types to their counts

        Example:
            >>> validator = DataValidator()
            >>> cleaned_df, issues = validator.validate_all(df, 'AAPL')
            >>> summary = validator.get_issues_summary()
            >>> print(summary)
            {'missing_dates': 5, 'invalid_rows_removed': 2}
        """
        summary = {}
        for issue in self.issues:
            issue_type = issue.get('type', 'unknown')
            summary[issue_type] = summary.get(issue_type, 0) + 1
        return summary

    def clear_issues(self) -> None:
        """
        Clear the issues list.

        Useful when reusing the validator for multiple datasets.
        """
        self.issues = []
        logger.debug("Issues list cleared")
