"""Momentum metrics for trading strategy analysis."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class MomentumMetrics:
    """
    Calculate momentum-based metrics for trading strategies.

    This class provides methods to calculate various momentum indicators
    that help identify strong trending stocks suitable for momentum trading.
    """

    def __init__(self):
        """Initialize the MomentumMetrics class."""
        pass

    def calculate_roc(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate Rate of Change (ROC).

        ROC measures the percentage change in price over a specified period.
        It's a momentum oscillator that shows the speed of price changes.

        Formula: ROC = ((current_price - price_n_periods_ago) / price_n_periods_ago) * 100

        Args:
            df: DataFrame with OHLCV data (must contain 'close' column)
            period: Number of periods to look back (typically 20 or 50 days)

        Returns:
            DataFrame with added 'roc_{period}' column

        Raises:
            ValueError: If DataFrame is empty or missing 'close' column

        Interpretation:
            ROC > 20%  : Strong upward momentum
            ROC > 0%   : Positive momentum
            ROC < 0%   : Negative momentum
            ROC < -20% : Strong downward momentum

        Example:
            >>> metrics = MomentumMetrics()
            >>> df_with_roc = metrics.calculate_roc(df, 20)
            >>> strong_momentum = df_with_roc[df_with_roc['roc_20'] > 20]
            >>> print(strong_momentum)
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        if period < 1:
            raise ValueError("Period must be at least 1")

        if period >= len(df):
            raise ValueError(f"Period ({period}) must be less than data length ({len(df)})")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Calculate ROC
        # ROC = ((current - previous) / previous) * 100
        column_name = f'roc_{period}'
        result[column_name] = ((result['close'] - result['close'].shift(period)) /
                                result['close'].shift(period)) * 100

        return result

    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength score based on moving average positioning.

        This creates a 0-100 score that quantifies the strength and direction
        of the trend by analyzing the relationship between price and various
        moving averages.

        Scoring Criteria:
        1. Price above 20-day MA: +20 points
        2. Price above 50-day MA: +20 points
        3. Price above 200-day MA: +20 points
        4. 20-day MA above 50-day MA: +15 points
        5. 50-day MA above 200-day MA: +15 points
        6. 20-day MA trending up (higher than 5 days ago): +10 points

        Args:
            df: DataFrame with OHLCV data and moving averages already calculated
                Must contain: 'close', 'sma_20', 'sma_50', 'sma_200'

        Returns:
            DataFrame with added columns:
            - 'trend_strength': Score from 0-100
            - 'trend_direction': 'up', 'down', or 'neutral'

        Raises:
            ValueError: If DataFrame is missing required columns

        Interpretation:
            >70  : Strong uptrend - ideal for momentum trading
            30-70: Neutral/mixed - avoid or use caution
            <30  : Downtrend - avoid long positions

        Example:
            >>> from indicators.technical import TechnicalIndicators
            >>> tech = TechnicalIndicators()
            >>> df = tech.add_all_indicators(df)
            >>> metrics = MomentumMetrics()
            >>> df = metrics.calculate_trend_strength(df)
            >>> strong_trends = df[df['trend_strength'] > 70]
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        required_cols = ['close', 'sma_20', 'sma_50', 'sma_200']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"DataFrame must contain columns: {missing_cols}. "
                f"Run TechnicalIndicators.add_all_indicators() first."
            )

        # Make a copy to avoid modifying original
        result = df.copy()

        # Initialize score column
        result['trend_strength'] = 0

        # Criterion 1: Price above 20-day MA (+20 points)
        result.loc[result['close'] > result['sma_20'], 'trend_strength'] += 20

        # Criterion 2: Price above 50-day MA (+20 points)
        result.loc[result['close'] > result['sma_50'], 'trend_strength'] += 20

        # Criterion 3: Price above 200-day MA (+20 points)
        result.loc[result['close'] > result['sma_200'], 'trend_strength'] += 20

        # Criterion 4: 20-day MA above 50-day MA (+15 points)
        result.loc[result['sma_20'] > result['sma_50'], 'trend_strength'] += 15

        # Criterion 5: 50-day MA above 200-day MA (+15 points)
        result.loc[result['sma_50'] > result['sma_200'], 'trend_strength'] += 15

        # Criterion 6: 20-day MA trending up (+10 points)
        # Check if SMA 20 today > SMA 20 from 5 days ago
        sma20_5d_ago = result['sma_20'].shift(5)
        result.loc[result['sma_20'] > sma20_5d_ago, 'trend_strength'] += 10

        # Determine trend direction based on score
        result['trend_direction'] = 'neutral'
        result.loc[result['trend_strength'] > 70, 'trend_direction'] = 'up'
        result.loc[result['trend_strength'] < 30, 'trend_direction'] = 'down'

        return result

    def detect_volume_surge(
        self,
        df: pd.DataFrame,
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect volume surges that may indicate strong momentum.

        Volume surges often precede or confirm strong price movements.
        This method identifies when volume is significantly above average.

        Args:
            df: DataFrame with OHLCV data (must contain 'volume' column)
            threshold: Ratio threshold for surge detection (default: 1.5)
                      1.5 means volume is 50% above average

        Returns:
            DataFrame with added columns:
            - 'volume_ratio': Current volume / 20-day volume MA
            - 'volume_surge': Boolean indicating if surge detected

        Raises:
            ValueError: If DataFrame is empty or missing required columns

        Interpretation:
            volume_ratio > 2.0: Very strong surge
            volume_ratio > 1.5: Significant surge (default threshold)
            volume_ratio > 1.0: Above average
            volume_ratio < 1.0: Below average

        Example:
            >>> metrics = MomentumMetrics()
            >>> df = metrics.detect_volume_surge(df, threshold=1.5)
            >>> surges = df[df['volume_surge'] == True]
            >>> print(f"Found {len(surges)} volume surges")
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if 'volume' not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column")

        if threshold <= 0:
            raise ValueError("Threshold must be greater than 0")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Calculate 20-day volume moving average
        volume_ma = result['volume'].rolling(window=20, min_periods=1).mean()

        # Calculate volume ratio
        result['volume_ratio'] = result['volume'] / volume_ma

        # Handle division by zero
        result['volume_ratio'] = result['volume_ratio'].fillna(1)

        # Detect surge (volume ratio exceeds threshold)
        result['volume_surge'] = result['volume_ratio'] > threshold

        return result

    def calculate_momentum_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite momentum score combining multiple metrics.

        This creates a comprehensive 0-100 momentum score that combines:
        - Rate of Change (ROC)
        - RSI positioning
        - Trend strength
        - Volume confirmation

        The score helps identify the strongest momentum candidates for trading.

        Args:
            df: DataFrame with OHLCV data and indicators
                Should have: close, rsi, roc_20, trend_strength, volume_ratio

        Returns:
            DataFrame with added 'momentum_score' column (0-100)

        Raises:
            ValueError: If DataFrame is missing required columns

        Scoring Components:
        - ROC contribution (0-30 points):
          * ROC > 20%: 30 points
          * ROC > 10%: 20 points
          * ROC > 0%: 10 points
          * ROC <= 0%: 0 points

        - RSI contribution (0-25 points):
          * RSI > 70: 25 points (strong momentum, but overbought)
          * RSI 50-70: 20 points (healthy momentum)
          * RSI 30-50: 10 points (weak momentum)
          * RSI < 30: 0 points (oversold, negative momentum)

        - Trend strength contribution (0-30 points):
          * Directly use 30% of trend_strength score

        - Volume contribution (0-15 points):
          * volume_ratio > 2.0: 15 points
          * volume_ratio > 1.5: 10 points
          * volume_ratio > 1.0: 5 points
          * volume_ratio <= 1.0: 0 points

        Interpretation:
            >80: Exceptional momentum - prime candidates
            60-80: Strong momentum - good candidates
            40-60: Moderate momentum - watch list
            <40: Weak momentum - avoid

        Example:
            >>> from indicators.technical import TechnicalIndicators
            >>> tech = TechnicalIndicators()
            >>> metrics = MomentumMetrics()
            >>>
            >>> # Prepare data with all indicators
            >>> df = tech.add_all_indicators(df)
            >>> df = metrics.calculate_roc(df, 20)
            >>> df = metrics.calculate_trend_strength(df)
            >>> df = metrics.detect_volume_surge(df)
            >>> df = metrics.calculate_momentum_score(df)
            >>>
            >>> # Find top momentum stocks
            >>> top_momentum = df[df['momentum_score'] > 80]
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        # Check for required columns
        required_cols = ['close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame must contain columns: {missing_cols}")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Initialize momentum score
        result['momentum_score'] = 0

        # Component 1: ROC contribution (0-30 points)
        if 'roc_20' in result.columns:
            result.loc[result['roc_20'] > 20, 'momentum_score'] += 30
            result.loc[(result['roc_20'] > 10) & (result['roc_20'] <= 20), 'momentum_score'] += 20
            result.loc[(result['roc_20'] > 0) & (result['roc_20'] <= 10), 'momentum_score'] += 10

        # Component 2: RSI contribution (0-25 points)
        if 'rsi' in result.columns:
            result.loc[result['rsi'] > 70, 'momentum_score'] += 25
            result.loc[(result['rsi'] >= 50) & (result['rsi'] <= 70), 'momentum_score'] += 20
            result.loc[(result['rsi'] >= 30) & (result['rsi'] < 50), 'momentum_score'] += 10

        # Component 3: Trend strength contribution (0-30 points)
        # Use 30% of the trend strength score (which is 0-100)
        if 'trend_strength' in result.columns:
            result['momentum_score'] += (result['trend_strength'] * 0.30).fillna(0)

        # Component 4: Volume contribution (0-15 points)
        if 'volume_ratio' in result.columns:
            result.loc[result['volume_ratio'] > 2.0, 'momentum_score'] += 15
            result.loc[(result['volume_ratio'] > 1.5) & (result['volume_ratio'] <= 2.0), 'momentum_score'] += 10
            result.loc[(result['volume_ratio'] > 1.0) & (result['volume_ratio'] <= 1.5), 'momentum_score'] += 5

        # Ensure score stays within 0-100 range
        result['momentum_score'] = result['momentum_score'].clip(0, 100)

        return result

    def add_all_momentum_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all momentum metrics in one call.

        This convenience method calculates all momentum metrics:
        - ROC (20 and 50 day)
        - Trend strength
        - Volume surge detection
        - Composite momentum score

        Args:
            df: DataFrame with OHLCV data and technical indicators

        Returns:
            DataFrame with all momentum metric columns added

        Raises:
            ValueError: If DataFrame is missing required columns

        Example:
            >>> from indicators.technical import TechnicalIndicators
            >>> tech = TechnicalIndicators()
            >>> metrics = MomentumMetrics()
            >>>
            >>> df = tech.add_all_indicators(df)
            >>> df = metrics.add_all_momentum_metrics(df)
            >>> print(df.columns)
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        # Make a copy to avoid modifying original
        result = df.copy()

        try:
            # Add ROC for 20 and 50 day periods
            if len(result) >= 20:
                result = self.calculate_roc(result, 20)

            if len(result) >= 50:
                result = self.calculate_roc(result, 50)

            # Add trend strength
            if all(col in result.columns for col in ['close', 'sma_20', 'sma_50', 'sma_200']):
                result = self.calculate_trend_strength(result)

            # Add volume surge detection
            if 'volume' in result.columns:
                result = self.detect_volume_surge(result)

            # Add composite momentum score
            result = self.calculate_momentum_score(result)

        except Exception as e:
            raise ValueError(f"Error adding momentum metrics: {e}")

        return result
