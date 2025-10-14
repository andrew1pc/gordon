"""Technical indicators for financial market data analysis."""

import pandas as pd
import numpy as np
from typing import Optional


class TechnicalIndicators:
    """
    Calculate technical indicators for financial market data.

    This class provides methods to calculate common technical indicators
    used in trading and technical analysis, including moving averages,
    momentum indicators, and volatility measures.
    """

    def __init__(self):
        """Initialize the TechnicalIndicators class."""
        pass

    def calculate_sma(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate Simple Moving Average (SMA).

        The SMA is the unweighted mean of the previous n data points.
        It smooths out price data to identify trends.

        Args:
            df: DataFrame with OHLCV data (must contain 'close' column)
            period: Number of periods for the moving average

        Returns:
            DataFrame with added 'sma_{period}' column

        Raises:
            ValueError: If DataFrame is empty or missing 'close' column
            ValueError: If period is less than 1 or greater than data length

        Example:
            >>> indicators = TechnicalIndicators()
            >>> df_with_sma = indicators.calculate_sma(df, 20)
            >>> print(df_with_sma['sma_20'].tail())
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        if period < 1:
            raise ValueError("Period must be at least 1")

        if period > len(df):
            raise ValueError(f"Period ({period}) cannot be greater than data length ({len(df)})")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Calculate SMA
        column_name = f'sma_{period}'
        result[column_name] = result['close'].rolling(window=period, min_periods=period).mean()

        return result

    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average (EMA).

        EMA gives more weight to recent prices, making it more responsive
        to new information compared to SMA.

        Args:
            df: DataFrame with OHLCV data (must contain 'close' column)
            period: Number of periods for the EMA

        Returns:
            DataFrame with added 'ema_{period}' column

        Raises:
            ValueError: If DataFrame is empty or missing 'close' column
            ValueError: If period is less than 1

        Example:
            >>> indicators = TechnicalIndicators()
            >>> df_with_ema = indicators.calculate_ema(df, 12)
            >>> print(df_with_ema['ema_12'].tail())
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        if period < 1:
            raise ValueError("Period must be at least 1")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Calculate EMA using pandas ewm (exponentially weighted moving average)
        column_name = f'ema_{period}'
        result[column_name] = result['close'].ewm(span=period, adjust=False).mean()

        return result

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).

        RSI is a momentum oscillator that measures the speed and magnitude
        of price changes. Values range from 0 to 100, with readings above 70
        indicating overbought conditions and below 30 indicating oversold.

        Args:
            df: DataFrame with OHLCV data (must contain 'close' column)
            period: RSI period (default: 14)

        Returns:
            DataFrame with added 'rsi' column (values between 0-100)

        Raises:
            ValueError: If DataFrame is empty or missing 'close' column

        Example:
            >>> indicators = TechnicalIndicators()
            >>> df_with_rsi = indicators.calculate_rsi(df, 14)
            >>> print(df_with_rsi['rsi'].tail())
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        if period < 1:
            raise ValueError("Period must be at least 1")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Step 1: Calculate price changes (delta)
        delta = result['close'].diff()

        # Step 2: Separate gains (positive changes) and losses (negative changes)
        gains = delta.where(delta > 0, 0)  # Positive changes, else 0
        losses = -delta.where(delta < 0, 0)  # Absolute value of negative changes, else 0

        # Step 3: Calculate average gain and average loss using EMA
        # Using EMA with span=period gives us the same weighting as the standard RSI calculation
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()

        # Step 4: Calculate RS (Relative Strength) = average_gain / average_loss
        # Step 5: Calculate RSI = 100 - (100 / (1 + RS))
        # Combine steps 4 and 5, handling division by zero
        # When avg_loss is 0, RSI should be 100 (all gains, no losses)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle edge case where both avg_gain and avg_loss are 0 (no price movement)
        # In this case, RSI should be 50 (neutral)
        rsi = rsi.fillna(50)

        # Add RSI column
        result['rsi'] = rsi

        return result

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of prices. It consists of:
        - MACD line: fast EMA - slow EMA
        - Signal line: EMA of MACD line
        - Histogram: MACD line - signal line

        Args:
            df: DataFrame with OHLCV data (must contain 'close' column)
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)

        Returns:
            DataFrame with added columns: 'macd', 'macd_signal', 'macd_histogram'

        Raises:
            ValueError: If DataFrame is empty or missing 'close' column

        Example:
            >>> indicators = TechnicalIndicators()
            >>> df_with_macd = indicators.calculate_macd(df)
            >>> print(df_with_macd[['macd', 'macd_signal', 'macd_histogram']].tail())
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Calculate fast EMA (default 12-period)
        fast_ema = result['close'].ewm(span=fast, adjust=False).mean()

        # Calculate slow EMA (default 26-period)
        slow_ema = result['close'].ewm(span=slow, adjust=False).mean()

        # Calculate MACD line = fast_ema - slow_ema
        macd_line = fast_ema - slow_ema

        # Calculate signal line = 9-period EMA of MACD line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Calculate MACD histogram = MACD line - signal line
        histogram = macd_line - signal_line

        # Add columns
        result['macd'] = macd_line
        result['macd_signal'] = signal_line
        result['macd_histogram'] = histogram

        return result

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility by decomposing the entire range of
        an asset price for that period. It's useful for setting stop-loss
        levels and position sizing.

        The True Range is the greatest of:
        - Current High - Current Low
        - |Current High - Previous Close|
        - |Current Low - Previous Close|

        ATR is the EMA of True Range over the specified period.

        Args:
            df: DataFrame with OHLCV data (must contain 'high', 'low', 'close')
            period: ATR period (default: 14)

        Returns:
            DataFrame with added 'atr' column

        Raises:
            ValueError: If DataFrame is empty or missing required columns

        Example:
            >>> indicators = TechnicalIndicators()
            >>> df_with_atr = indicators.calculate_atr(df, 14)
            >>> # Use ATR for stop-loss: stop = entry_price - (2 * atr)
            >>> print(df_with_atr['atr'].tail())
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame must contain columns: {missing_cols}")

        if period < 1:
            raise ValueError("Period must be at least 1")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Get previous close (shifted by 1)
        prev_close = result['close'].shift(1)

        # Calculate True Range components
        tr1 = result['high'] - result['low']  # Current high - low
        tr2 = (result['high'] - prev_close).abs()  # |Current high - Previous close|
        tr3 = (result['low'] - prev_close).abs()  # |Current low - Previous close|

        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR as EMA of True Range
        # For the first row (no previous close), TR is just high - low
        atr = true_range.ewm(span=period, adjust=False).mean()

        # Add ATR column
        result['atr'] = atr

        return result

    def calculate_volume_ma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate Volume Moving Average and Volume Ratio.

        Volume MA helps identify when volume is significantly higher or lower
        than average, which can signal strong moves or breakouts.

        Args:
            df: DataFrame with OHLCV data (must contain 'volume' column)
            period: Period for volume moving average (default: 20)

        Returns:
            DataFrame with added 'volume_ma_{period}' and 'volume_ratio' columns
            - volume_ratio > 1.5 indicates significant volume surge
            - volume_ratio < 0.5 indicates unusually low volume

        Raises:
            ValueError: If DataFrame is empty or missing 'volume' column

        Example:
            >>> indicators = TechnicalIndicators()
            >>> df_with_vol = indicators.calculate_volume_ma(df, 20)
            >>> # Identify volume spikes
            >>> spikes = df_with_vol[df_with_vol['volume_ratio'] > 1.5]
            >>> print(spikes.tail())
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if 'volume' not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column")

        if period < 1:
            raise ValueError("Period must be at least 1")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Calculate volume moving average
        volume_ma_col = f'volume_ma_{period}'
        result[volume_ma_col] = result['volume'].rolling(window=period, min_periods=1).mean()

        # Calculate volume ratio (current volume / average volume)
        # This helps identify volume surges (ratio > 1.5 is significant)
        result['volume_ratio'] = result['volume'] / result[volume_ma_col]

        # Handle division by zero (if volume_ma is 0, set ratio to 1)
        result['volume_ratio'] = result['volume_ratio'].fillna(1)

        return result

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all commonly used technical indicators in one call.

        This convenience method adds:
        - SMA: 20, 50, 200-day
        - EMA: 12, 26-day
        - RSI: 14-period
        - MACD: 12, 26, 9
        - ATR: 14-period
        - Volume MA: 20-day with ratio

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicator columns added

        Raises:
            ValueError: If DataFrame is empty or missing required columns

        Example:
            >>> indicators = TechnicalIndicators()
            >>> df_full = indicators.add_all_indicators(df)
            >>> print(df_full.columns)
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        # Make a copy to avoid modifying original
        result = df.copy()

        try:
            # Add SMAs (20, 50, 200-day)
            result = self.calculate_sma(result, 20)

            if len(result) >= 50:
                result = self.calculate_sma(result, 50)

            if len(result) >= 200:
                result = self.calculate_sma(result, 200)

            # Add EMAs (12, 26-day)
            if len(result) >= 12:
                result = self.calculate_ema(result, 12)

            if len(result) >= 26:
                result = self.calculate_ema(result, 26)

            # Add RSI (14-period)
            if len(result) >= 14:
                result = self.calculate_rsi(result, 14)

            # Add MACD (12, 26, 9)
            if len(result) >= 26:
                result = self.calculate_macd(result, fast=12, slow=26, signal=9)

            # Add ATR (14-period)
            if len(result) >= 14:
                result = self.calculate_atr(result, 14)

            # Add Volume MA (20-day with ratio)
            if len(result) >= 20:
                result = self.calculate_volume_ma(result, 20)

        except Exception as e:
            raise ValueError(f"Error adding indicators: {e}")

        return result
