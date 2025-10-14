#!/usr/bin/env python3
"""
Validation script for the AssetScanner.

This script tests the scanner functionality and displays the results.
"""

import logging
from datetime import datetime, timedelta
from config.api_config import TiingoConfig
from data.fetcher import TiingoClient
from strategy.scanner import AssetScanner
from config.strategy_config import ScannerConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """Run scanner validation."""
    print_header("Asset Scanner Validation")

    try:
        # Initialize
        api_config = TiingoConfig()
        client = TiingoClient(api_config)
        scanner_config = ScannerConfig()
        scanner = AssetScanner(client, scanner_config)

        print(f"\n✓ Scanner initialized")
        print(f"  Configuration: {scanner_config}")

    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        return

    # Test 1: Get stock universe
    print_header("Test 1: Stock Universe")
    try:
        # Use smaller subset for faster testing
        stocks = scanner._get_sp500_tickers()[:20]  # Test with 20 stocks
        print(f"✓ Retrieved {len(stocks)} stocks for testing")
        print(f"  Sample: {stocks[:10]}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    # Test 2: Get crypto universe
    print_header("Test 2: Crypto Universe")
    try:
        cryptos = scanner.get_crypto_universe(top_n=10)
        print(f"✓ Retrieved {len(cryptos)} cryptos")
        print(f"  Cryptos: {cryptos}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    # Test 3: Fetch prices for small sample
    print_header("Test 3: Fetch Historical Prices")
    try:
        # Use small sample and short timeframe for speed
        test_tickers = stocks[:5]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        print(f"Fetching data for {test_tickers} from {start_str} to {end_str}")

        prices = scanner.fetch_universe_prices(test_tickers, start_str, end_str, 'stock')

        print(f"✓ Fetched prices for {len(prices)} stocks")
        for ticker, df in list(prices.items())[:3]:
            print(f"  {ticker}: {len(df)} days")

    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    # Test 4: Prepare data with indicators
    print_header("Test 4: Add Indicators and Momentum Metrics")
    try:
        prepared = scanner.prepare_universe_data(prices)

        print(f"✓ Prepared {len(prepared)} assets with indicators")

        # Show columns for one asset
        if prepared:
            sample_ticker = list(prepared.keys())[0]
            sample_df = prepared[sample_ticker]
            print(f"\n  Columns for {sample_ticker}:")
            indicator_cols = [col for col in sample_df.columns
                            if col not in ['open', 'high', 'low', 'close', 'volume']]
            print(f"  {', '.join(indicator_cols[:10])}")
            if len(indicator_cols) > 10:
                print(f"  ... and {len(indicator_cols) - 10} more")

    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    # Test 5: Rank by momentum
    print_header("Test 5: Momentum Ranking")
    try:
        ranking = scanner.rank_universe_by_momentum(prepared)

        print(f"✓ Ranked {len(ranking)} assets")
        print(f"\nTop 10 by momentum score:")
        print(ranking[['ticker', 'momentum_score', 'rsi', 'trend_strength', 'current_price']].head(10).to_string(index=False))

    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    # Test 6: Select candidates
    print_header("Test 6: Select Top Candidates")
    try:
        candidates = scanner.select_top_candidates(ranking, max_candidates=5)

        print(f"✓ Selected {len(candidates)} candidates")

        if len(candidates) > 0:
            print(f"\nCandidates:")
            print(candidates[['ticker', 'momentum_score', 'rsi', 'trend_strength']].to_string(index=False))
        else:
            print("\n  No candidates met the criteria")
            print("  This is normal with limited test data - try with full universe")

    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    # Summary
    print_header("Validation Summary")
    print("\n✓ All tests passed!")
    print("\nNext steps:")
    print("  1. Run with full stock universe (may take 10-15 minutes)")
    print("  2. Test caching functionality")
    print("  3. Validate momentum scores against TradingView")
    print("  4. Compare selected candidates to manual analysis")
    print("\nThe scanner is working correctly!\n")


if __name__ == '__main__':
    main()
