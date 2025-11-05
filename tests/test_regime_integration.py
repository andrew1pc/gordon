"""Integration tests for regime detection (Iteration 5)."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.api_config import TiingoConfig
from data.fetcher import TiingoClient
from indicators.technical import TechnicalIndicators
from indicators.market_regime import MarketRegimeDetector


class TestRegimeDetectionIntegration:
    """Integration tests using real API calls."""

    @pytest.mark.skip(reason="Uses real API call - run manually")
    def test_fetch_spy_and_detect_regime(self):
        """
        Integration test: Fetch SPY data and detect current regime.

        NOTE: This uses 1 real API call to Tiingo.
        Run manually with: pytest tests/test_regime_integration.py::TestRegimeDetectionIntegration::test_fetch_spy_and_detect_regime -v -s
        """
        # Initialize components
        api_config = TiingoConfig()
        client = TiingoClient(api_config)
        tech = TechnicalIndicators()
        detector = MarketRegimeDetector()

        # Fetch SPY data (1 API call)
        print("\nFetching SPY data...")
        spy_data = client.fetch_daily_prices('SPY', days=250)

        assert spy_data is not None
        assert not spy_data.empty
        assert len(spy_data) > 200  # Should have at least 200 days
        print(f"✓ Fetched {len(spy_data)} days of SPY data")

        # Add technical indicators
        spy_data = tech.add_all_indicators(spy_data)

        # Verify indicators added
        assert 'sma_50' in spy_data.columns
        assert 'sma_200' in spy_data.columns
        print("✓ Technical indicators calculated")

        # Detect regime
        regime, confidence = detector.detect_regime(spy_data)

        assert regime in ['bull', 'bear', 'sideways']
        assert 0.0 <= confidence <= 1.0
        print(f"✓ Regime detected: {regime.upper()} (confidence: {confidence:.1%})")

        # Get adjustments
        adjustments = detector.get_regime_adjustments(regime, confidence)

        assert adjustments.risk_multiplier > 0
        assert adjustments.max_positions > 0
        print(f"✓ Risk adjustments: {adjustments.risk_multiplier}x risk, {adjustments.max_positions} max positions")

        # Get full summary
        summary = detector.get_regime_summary(spy_data)

        print(f"\nFull regime summary:")
        print(f"  Regime: {summary['regime'].upper()}")
        print(f"  Confidence: {summary['confidence']:.1%}")
        print(f"  Risk multiplier: {summary['risk_multiplier']}x")
        print(f"  Max positions: {summary['max_positions']}")
        print(f"  Current price: ${summary['current_price']:.2f}")
        print(f"  MA50: ${summary['ma50']:.2f} ({summary['price_vs_ma50']:+.1f}%)")
        print(f"  MA200: ${summary['ma200']:.2f} ({summary['price_vs_ma200']:+.1f}%)")
        print(f"  MA50 vs MA200: {summary['ma50_vs_ma200']}")

        assert 'regime' in summary
        assert 'current_price' in summary

    def test_regime_detection_with_mock_data(self):
        """
        Test regime detection with synthetic data (no API call).

        This test can run in CI/CD without API access.
        """
        import pandas as pd
        import numpy as np

        # Create synthetic bull market data
        dates = pd.bdate_range(start='2024-01-01', periods=250)
        closes = np.linspace(100, 200, 250)  # Strong uptrend

        spy_data = pd.DataFrame({
            'open': closes,
            'high': closes + 2,
            'low': closes - 2,
            'close': closes,
            'volume': [1000000] * 250
        }, index=dates)

        # Add indicators
        tech = TechnicalIndicators()
        spy_data = tech.add_all_indicators(spy_data)

        # Detect regime
        detector = MarketRegimeDetector()
        regime, confidence = detector.detect_regime(spy_data)

        # Should detect bull market
        assert regime == 'bull'
        assert confidence >= 0.6

        # Verify adjustments
        adjustments = detector.get_regime_adjustments(regime, confidence)
        assert adjustments.risk_multiplier == 1.0  # Full risk in bull
        assert adjustments.max_positions == 8

    def test_regime_adjustments_applied_correctly(self):
        """Test that regime adjustments match expected hardcoded values."""
        detector = MarketRegimeDetector()

        # Test bull regime
        bull_adj = detector.get_regime_adjustments('bull', 0.8)
        assert bull_adj.risk_multiplier == 1.0
        assert bull_adj.max_positions == 8

        # Test sideways regime
        sideways_adj = detector.get_regime_adjustments('sideways', 0.7)
        assert sideways_adj.risk_multiplier == 0.75
        assert sideways_adj.max_positions == 6

        # Test bear regime
        bear_adj = detector.get_regime_adjustments('bear', 0.9)
        assert bear_adj.risk_multiplier == 0.5
        assert bear_adj.max_positions == 4

    def test_regime_detection_handles_errors_gracefully(self):
        """Test that regime detection handles errors without crashing."""
        import pandas as pd

        detector = MarketRegimeDetector()

        # Test with empty DataFrame
        regime, confidence = detector.detect_regime(pd.DataFrame())
        assert regime == 'sideways'  # Default fallback
        assert confidence == 0.5

        # Test with missing columns
        bad_df = pd.DataFrame({'close': [100, 101, 102]})
        regime, confidence = detector.detect_regime(bad_df)
        assert regime == 'sideways'  # Default fallback


if __name__ == '__main__':
    # Run with verbose output to see integration test results
    pytest.main([__file__, '-v', '-s'])
