#!/usr/bin/env python3
"""
Simple test script to verify Tiingo API connection for crypto data.
Fetches Bitcoin (BTC) price data for the last 30 days.
"""

from datetime import datetime, timedelta
from config.api_config import TiingoConfig
from data.fetcher import TiingoClient


def main():
    print("Testing Tiingo Crypto API connection...\n")

    try:
        # Initialize config and client
        config = TiingoConfig()
        client = TiingoClient(config)
        print(f"✓ Connected to Tiingo API: {config.base_url}\n")

        # Calculate date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        print(f"Fetching Bitcoin (btcusd) data from {start_str} to {end_str}...\n")

        # Fetch crypto data
        df = client.get_crypto_prices('btcusd', start_str, end_str)

        if df is not None and not df.empty:
            print(f"✓ Successfully fetched {len(df)} rows of data\n")

            print("First 5 rows:")
            print("-" * 80)
            print(df.head())
            print()

            print("Last 5 rows:")
            print("-" * 80)
            print(df.tail())
            print()

            print("Summary statistics:")
            print("-" * 80)
            print(df[['open', 'high', 'low', 'close', 'volume']].describe())
            print()

            # Calculate price change
            first_close = df['close'].iloc[0]
            last_close = df['close'].iloc[-1]
            price_change = last_close - first_close
            price_change_pct = (price_change / first_close) * 100

            print("Price movement:")
            print("-" * 80)
            print(f"Starting price: ${first_close:,.2f}")
            print(f"Ending price:   ${last_close:,.2f}")
            print(f"Change:         ${price_change:,.2f} ({price_change_pct:+.2f}%)")
        else:
            print("✗ Failed to fetch data")

    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("\nMake sure to set the TIINGO_API_KEY environment variable:")
        print("  export TIINGO_API_KEY='your_api_key_here'")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == '__main__':
    main()
