#!/usr/bin/env python3
"""
Script to validate technical indicators against TradingView.

This script fetches AAPL data, calculates all technical indicators,
and displays the results for manual comparison with TradingView.
"""

from datetime import datetime, timedelta
from config.api_config import TiingoConfig
from data.fetcher import TiingoClient
from indicators.technical import TechnicalIndicators


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_instructions():
    """Print instructions for validating against TradingView."""
    print("\n" + "=" * 100)
    print("  INSTRUCTIONS FOR TRADINGVIEW VALIDATION")
    print("=" * 100)
    print("""
1. Go to TradingView: https://www.tradingview.com/chart/
2. Enter ticker symbol: AAPL
3. Set timeframe to: Daily (1D)
4. Add the following indicators:
   - SMA (20, 50, 200)
   - EMA (12, 26)
   - RSI (14)
   - MACD (12, 26, 9)
   - ATR (14)
   - Volume MA (20)

5. Compare the indicator values below with TradingView for the SAME DATES

NOTES:
- Small differences (<1%) are acceptable due to rounding differences
- Large differences (>5%) indicate calculation errors
- The dates shown below should match TradingView's dates exactly
- TradingView may show slightly different values due to:
  * Different data sources
  * Pre-market/after-market data inclusion
  * Adjustment methods (splits, dividends)
""")


def format_number(num, decimals=2):
    """Format number for display."""
    if pd.isna(num):
        return "N/A"
    return f"{num:.{decimals}f}"


def main():
    """Main execution function."""
    print_header("Technical Indicators Validation Script")

    # Initialize
    try:
        config = TiingoConfig()
        client = TiingoClient(config)
        indicators = TechnicalIndicators()

        print(f"\n✓ Connected to Tiingo API: {config.base_url}")

    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nMake sure to set the TIINGO_API_KEY environment variable:")
        print("  export TIINGO_API_KEY='your_api_key_here'")
        return

    # Fetch AAPL data for last 250 trading days (to ensure we have enough for 200-day SMA)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get extra data to ensure 200+ trading days

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"\nFetching AAPL data from {start_str} to {end_str}...")

    try:
        df = client.get_stock_prices('AAPL', start_str, end_str)

        if df is None or df.empty:
            print("✗ Failed to fetch data")
            return

        print(f"✓ Fetched {len(df)} days of data")

    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return

    # Calculate all indicators
    print("\nCalculating technical indicators...")

    try:
        df_with_indicators = indicators.add_all_indicators(df)
        print("✓ All indicators calculated successfully")

    except Exception as e:
        print(f"✗ Error calculating indicators: {e}")
        return

    # Print instructions
    print_instructions()

    # Display last 10 days with all indicators
    print_header("LAST 10 DAYS OF DATA WITH INDICATORS")

    # Select columns to display
    display_cols = [
        'close',
        'sma_20', 'sma_50', 'sma_200',
        'ema_12', 'ema_26',
        'rsi',
        'macd', 'macd_signal', 'macd_histogram',
        'atr',
        'volume', 'volume_ma_20', 'volume_ratio'
    ]

    # Get last 10 rows
    last_10 = df_with_indicators[display_cols].tail(10)

    print("\nDate-by-date comparison (use these to check against TradingView):")
    print("-" * 100)

    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    for date, row in last_10.iterrows():
        print(f"\n{'Date:':<20} {date.strftime('%Y-%m-%d')}")
        print("-" * 100)

        print(f"{'Close Price:':<20} ${format_number(row['close'])}")
        print()

        print(f"{'Moving Averages:':<20}")
        print(f"  {'SMA-20:':<18} ${format_number(row['sma_20'])}")
        print(f"  {'SMA-50:':<18} ${format_number(row['sma_50'])}")
        print(f"  {'SMA-200:':<18} ${format_number(row['sma_200'])}")
        print(f"  {'EMA-12:':<18} ${format_number(row['ema_12'])}")
        print(f"  {'EMA-26:':<18} ${format_number(row['ema_26'])}")
        print()

        print(f"{'RSI (14):':<20} {format_number(row['rsi'])}")
        print()

        print(f"{'MACD:':<20}")
        print(f"  {'MACD Line:':<18} {format_number(row['macd'])}")
        print(f"  {'Signal Line:':<18} {format_number(row['macd_signal'])}")
        print(f"  {'Histogram:':<18} {format_number(row['macd_histogram'])}")
        print()

        print(f"{'ATR (14):':<20} ${format_number(row['atr'])}")
        print()

        print(f"{'Volume:':<20}")
        print(f"  {'Current:':<18} {format_number(row['volume'], 0)}")
        print(f"  {'MA-20:':<18} {format_number(row['volume_ma_20'], 0)}")
        print(f"  {'Ratio:':<18} {format_number(row['volume_ratio'])}x")
        print()

    # Print summary statistics
    print_header("SUMMARY STATISTICS (Last 10 Days)")

    print(f"\n{'Indicator':<25} {'Min':<15} {'Max':<15} {'Latest':<15}")
    print("-" * 100)
    print(f"{'Close Price':<25} ${format_number(last_10['close'].min()):<14} ${format_number(last_10['close'].max()):<14} ${format_number(last_10['close'].iloc[-1]):<14}")
    print(f"{'SMA-20':<25} ${format_number(last_10['sma_20'].min()):<14} ${format_number(last_10['sma_20'].max()):<14} ${format_number(last_10['sma_20'].iloc[-1]):<14}")
    print(f"{'SMA-50':<25} ${format_number(last_10['sma_50'].min()):<14} ${format_number(last_10['sma_50'].max()):<14} ${format_number(last_10['sma_50'].iloc[-1]):<14}")
    print(f"{'SMA-200':<25} ${format_number(last_10['sma_200'].min()):<14} ${format_number(last_10['sma_200'].max()):<14} ${format_number(last_10['sma_200'].iloc[-1]):<14}")
    print(f"{'RSI':<25} {format_number(last_10['rsi'].min()):<14} {format_number(last_10['rsi'].max()):<14} {format_number(last_10['rsi'].iloc[-1]):<14}")
    print(f"{'MACD':<25} {format_number(last_10['macd'].min()):<14} {format_number(last_10['macd'].max()):<14} {format_number(last_10['macd'].iloc[-1]):<14}")
    print(f"{'ATR':<25} ${format_number(last_10['atr'].min()):<14} ${format_number(last_10['atr'].max()):<14} ${format_number(last_10['atr'].iloc[-1]):<14}")

    # Print key values to compare
    print_header("KEY VALUES TO COMPARE WITH TRADINGVIEW")

    latest = last_10.iloc[-1]
    latest_date = last_10.index[-1].strftime('%Y-%m-%d')

    print(f"\nMost Recent Date: {latest_date}")
    print("\nCopy these values to compare with TradingView:")
    print("-" * 100)
    print(f"Close:        ${format_number(latest['close'])}")
    print(f"SMA(20):      ${format_number(latest['sma_20'])}")
    print(f"SMA(50):      ${format_number(latest['sma_50'])}")
    print(f"SMA(200):     ${format_number(latest['sma_200'])}")
    print(f"EMA(12):      ${format_number(latest['ema_12'])}")
    print(f"EMA(26):      ${format_number(latest['ema_26'])}")
    print(f"RSI(14):      {format_number(latest['rsi'])}")
    print(f"MACD:         {format_number(latest['macd'])}")
    print(f"MACD Signal:  {format_number(latest['macd_signal'])}")
    print(f"MACD Hist:    {format_number(latest['macd_histogram'])}")
    print(f"ATR(14):      ${format_number(latest['atr'])}")

    # Validation notes
    print_header("VALIDATION NOTES")
    print("""
✓ If your TradingView values are within 1% of the values above, the calculations are CORRECT
✓ Small differences are normal due to:
  - Data source differences (Tiingo vs TradingView's data provider)
  - Rounding differences
  - Time zone differences

✗ If values differ by more than 5%, there may be a calculation error

Common reasons for larger differences:
- Different adjustment methods for splits/dividends
- Using adjusted vs unadjusted prices
- Different volume data (some sources exclude pre/post market)
- TradingView using intraday data for the current day

For best comparison, use completed trading days (not today's date).
""")

    print_header("Validation Complete")
    print("\n✓ Indicator calculations finished successfully!\n")


if __name__ == '__main__':
    import pandas as pd
    main()
