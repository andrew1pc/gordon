#!/usr/bin/env python3
"""Check actual API status and see what error we're getting."""

import requests
from datetime import datetime, timedelta

api_key = '1fbf176f718099036edd83ae80c1ba9e545007a2'
ticker = 'AAPL'

end_date = datetime.now()
start_date = end_date - timedelta(days=30)
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
params = {
    'startDate': start_str,
    'endDate': end_str,
    'format': 'json',
    'resampleFreq': 'daily'
}

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Token {api_key}'
}

print(f"Testing API call to: {url}")
print(f"Parameters: {params}\n")

response = requests.get(url, params=params, headers=headers, timeout=30)

print(f"Status Code: {response.status_code}")
print(f"Headers: {dict(response.headers)}\n")
print(f"Response Content Type: {response.headers.get('Content-Type')}")
print(f"Response Length: {len(response.text)} characters\n")
print(f"Raw Response Text:\n{'-'*80}")
print(response.text[:500])  # First 500 chars
print(f"{'-'*80}\n")

# Try to parse as JSON
try:
    data = response.json()
    print(f"✓ Successfully parsed as JSON")
    print(f"✓ Returned {len(data)} records")
except Exception as e:
    print(f"✗ Failed to parse as JSON: {e}")
    print(f"✗ This is the rate limit issue!")
