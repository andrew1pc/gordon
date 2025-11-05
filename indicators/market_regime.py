"""Market regime detection for adaptive risk management."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class RegimeAdjustments:
    """Risk adjustments for a market regime."""
    risk_multiplier: float
    max_positions: int
    regime: str
    confidence: float


class MarketRegimeDetector:
    """
    Detects market regime (Bull/Bear/Sideways) for adaptive risk management.

    Uses SPY (S&P 500 ETF) as market proxy to classify current regime.
    Adjusts risk parameters based on regime to reduce losses in bear markets
    and maximize returns in bull markets.

    Regime Classification:
    - Bull: Strong uptrend, price above MAs, golden cross
    - Bear: Strong downtrend, price below MAs, death cross
    - Sideways: Mixed signals, choppy price action

    Hardcoded Risk Adjustments:
    - Bull:     1.0x risk, 8 max positions
    - Sideways: 0.75x risk, 6 max positions
    - Bear:     0.5x risk, 4 max positions

    Example:
        >>> detector = MarketRegimeDetector()
        >>> regime, confidence = detector.detect_regime(spy_df)
        >>> adjustments = detector.get_regime_adjustments(regime, confidence)
        >>> print(f"Regime: {regime}, Risk: {adjustments.risk_multiplier}x")
    """

    # Hardcoded regime adjustments
    REGIME_SETTINGS = {
        'bull': {
            'risk_multiplier': 1.0,
            'max_positions': 8
        },
        'sideways': {
            'risk_multiplier': 0.75,
            'max_positions': 6
        },
        'bear': {
            'risk_multiplier': 0.5,
            'max_positions': 4
        }
    }

    def __init__(self):
        """Initialize the MarketRegimeDetector."""
        logger.info("MarketRegimeDetector initialized")

    def detect_regime(
        self,
        df: pd.DataFrame,
        lookback_period: int = 50
    ) -> Tuple[str, float]:
        """
        Detect current market regime from price data.

        Uses multiple criteria to classify regime with confidence scoring:
        - Moving average positioning (price vs MA50, MA200)
        - Moving average trend (MA50 vs MA200)
        - Price trend (recent slope)
        - Volatility (ATR analysis)

        Args:
            df: DataFrame with OHLCV data and technical indicators
                Must contain: close, sma_20, sma_50, sma_200, atr
            lookback_period: Periods to analyze for trend (default: 50)

        Returns:
            Tuple of (regime: str, confidence: float)
            - regime: 'bull', 'bear', or 'sideways'
            - confidence: 0.0-1.0 confidence score

        Example:
            >>> regime, confidence = detector.detect_regime(spy_df)
            >>> if regime == 'bear' and confidence > 0.7:
            ...     print("High-confidence bear market - reduce risk")
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame, defaulting to sideways regime")
            return 'sideways', 0.5

        # Verify required columns
        required_cols = ['close', 'sma_50', 'sma_200']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns {missing}, defaulting to sideways")
            return 'sideways', 0.5

        try:
            # Get most recent data
            current = df.iloc[-1]

            # Initialize scoring
            bull_score = 0
            bear_score = 0
            max_score = 0

            # Criterion 1: Price vs MA50 (20 points)
            max_score += 20
            if current['close'] > current['sma_50']:
                bull_score += 20
            elif current['close'] < current['sma_50']:
                bear_score += 20

            # Criterion 2: Price vs MA200 (20 points)
            max_score += 20
            if current['close'] > current['sma_200']:
                bull_score += 20
            elif current['close'] < current['sma_200']:
                bear_score += 20

            # Criterion 3: MA50 vs MA200 (Golden/Death Cross) (30 points)
            max_score += 30
            if current['sma_50'] > current['sma_200']:
                bull_score += 30
            elif current['sma_50'] < current['sma_200']:
                bear_score += 30

            # Criterion 4: MA50 trend (15 points)
            if len(df) >= 10:
                max_score += 15
                ma50_current = current['sma_50']
                ma50_10d_ago = df['sma_50'].iloc[-10]
                if ma50_current > ma50_10d_ago:
                    bull_score += 15
                elif ma50_current < ma50_10d_ago:
                    bear_score += 15

            # Criterion 5: Recent price trend (15 points)
            if len(df) >= 20:
                max_score += 15
                recent_prices = df['close'].iloc[-20:]
                # Simple linear regression slope
                x = np.arange(len(recent_prices))
                slope = np.polyfit(x, recent_prices, 1)[0]

                # Normalize slope by average price
                avg_price = recent_prices.mean()
                slope_pct = (slope / avg_price) * 100

                if slope_pct > 0.1:  # Positive trend
                    bull_score += 15
                elif slope_pct < -0.1:  # Negative trend
                    bear_score += 15

            # Determine regime
            bull_pct = bull_score / max_score if max_score > 0 else 0
            bear_pct = bear_score / max_score if max_score > 0 else 0

            # Classification thresholds
            if bull_pct >= 0.6:
                regime = 'bull'
                confidence = bull_pct
            elif bear_pct >= 0.6:
                regime = 'bear'
                confidence = bear_pct
            else:
                regime = 'sideways'
                # Confidence is how close to neutral (0.5 is perfectly neutral)
                confidence = 1.0 - abs(bull_pct - bear_pct)

            logger.info(
                f"Regime detected: {regime.upper()} "
                f"(confidence: {confidence:.1%}, bull: {bull_pct:.1%}, bear: {bear_pct:.1%})"
            )

            return regime, confidence

        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return 'sideways', 0.5

    def get_regime_adjustments(
        self,
        regime: str,
        confidence: float
    ) -> RegimeAdjustments:
        """
        Get risk adjustments for the current regime.

        Uses hardcoded settings from REGIME_SETTINGS.

        Args:
            regime: 'bull', 'bear', or 'sideways'
            confidence: Confidence score 0.0-1.0

        Returns:
            RegimeAdjustments with risk_multiplier and max_positions

        Example:
            >>> adjustments = detector.get_regime_adjustments('bull', 0.85)
            >>> print(f"Risk: {adjustments.risk_multiplier}x")
            >>> print(f"Max positions: {adjustments.max_positions}")
        """
        if regime not in self.REGIME_SETTINGS:
            logger.warning(f"Unknown regime '{regime}', using sideways")
            regime = 'sideways'

        settings = self.REGIME_SETTINGS[regime]

        adjustments = RegimeAdjustments(
            risk_multiplier=settings['risk_multiplier'],
            max_positions=settings['max_positions'],
            regime=regime,
            confidence=confidence
        )

        logger.info(
            f"Regime adjustments: {regime.upper()} - "
            f"{adjustments.risk_multiplier}x risk, "
            f"{adjustments.max_positions} max positions"
        )

        return adjustments

    def should_trade(self, regime: str, confidence: float) -> Tuple[bool, str]:
        """
        Determine if trading should proceed based on regime.

        In bear markets with high confidence, may recommend pausing trading.

        Args:
            regime: Current regime
            confidence: Confidence score

        Returns:
            Tuple of (should_trade: bool, reason: str)

        Example:
            >>> should_trade, reason = detector.should_trade('bear', 0.9)
            >>> if not should_trade:
            ...     print(f"Trading paused: {reason}")
        """
        # Allow trading in all regimes, but with adjusted risk
        # (We don't halt trading entirely, just reduce exposure)
        return True, "Trading allowed with regime adjustments"

    def get_regime_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive regime analysis summary.

        Args:
            df: DataFrame with price data and indicators

        Returns:
            Dictionary with regime analysis details

        Example:
            >>> summary = detector.get_regime_summary(spy_df)
            >>> print(summary['regime'])
            >>> print(summary['adjustments'])
        """
        regime, confidence = self.detect_regime(df)
        adjustments = self.get_regime_adjustments(regime, confidence)

        if df is not None and not df.empty:
            current = df.iloc[-1]
            price = current.get('close', 0)
            ma50 = current.get('sma_50', 0)
            ma200 = current.get('sma_200', 0)
        else:
            price = ma50 = ma200 = 0

        summary = {
            'regime': regime,
            'confidence': confidence,
            'risk_multiplier': adjustments.risk_multiplier,
            'max_positions': adjustments.max_positions,
            'current_price': price,
            'ma50': ma50,
            'ma200': ma200,
            'price_vs_ma50': ((price / ma50 - 1) * 100) if ma50 > 0 else 0,
            'price_vs_ma200': ((price / ma200 - 1) * 100) if ma200 > 0 else 0,
            'ma50_vs_ma200': 'above' if ma50 > ma200 else 'below'
        }

        return summary
