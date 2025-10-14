#!/usr/bin/env python3
"""
Simple test script to verify Tiingo API connection.
Fetches AAPL stock data for the last 30 days.
"""

from datetime import datetime, timedelta
from config.api_config import TiingoConfig
from data.fetcher import TiingoClient


def main():
    print("Testing Tiingo API connection...\n")

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

        print(f"Fetching AAPL data from {start_str} to {end_str}...\n")

        # Fetch stock data
        df = client.get_stock_prices('AAPL', start_str, end_str)

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
