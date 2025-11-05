#!/usr/bin/env python3
"""Test if Tiingo API rate limit has reset."""

from datetime import datetime, timedelta
from config.api_config import TiingoConfig
from data.fetcher import TiingoClient

config = TiingoConfig()
client = TiingoClient(config)

end_date = datetime.now()
start_date = end_date - timedelta(days=30)
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

print("Testing Tiingo API connection...")
print(f"Attempting to fetch AAPL data from {start_str} to {end_str}\n")

try:
    df = client.get_stock_prices('AAPL', start_str, end_str)
    if df is not None and len(df) > 0:
        print(f"✓ SUCCESS! Rate limit has reset.")
        print(f"✓ Fetched {len(df)} rows of data")
        print("\nYou can now run: python find_momentum_trades.py")
    else:
        print("✗ No data returned (but no rate limit error)")
except Exception as e:
    if "Rate limit" in str(e):
        print(f"✗ Still rate limited: {e}")
        print("\nPlease wait longer and try again.")
    else:
        print(f"✗ Error: {e}")
