import time
import requests
import pandas as pd
from typing import Optional
from datetime import datetime


class TiingoClient:
    """Client for fetching data from Tiingo API."""

    def __init__(self, config):
        """
        Initialize TiingoClient with configuration.

        Args:
            config: Configuration object with api_key and base_url attributes
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.config.api_key}'
        })
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests (max 10 requests/second)

    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def get_stock_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        frequency: str = 'daily'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data from Tiingo's stock prices endpoint.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency ('daily', 'weekly', 'monthly', 'annually')

        Returns:
            pandas DataFrame with columns: date, open, high, low, close, volume
            Returns None if request fails

        Raises:
            ValueError: If date format is invalid
        """
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        # Enforce rate limiting
        self._rate_limit()

        # Build request URL
        url = f"{self.config.base_url}/tiingo/daily/{ticker}/prices"
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'format': 'json',
            'resampleFreq': frequency
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Parse response
            data = response.json()

            if not data:
                print(f"No data returned for {ticker} between {start_date} and {end_date}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Standardize column names
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'adjOpen': 'adj_open',
                'adjHigh': 'adj_high',
                'adjLow': 'adj_low',
                'adjClose': 'adj_close',
                'adjVolume': 'adj_volume'
            })

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Set date as index
            df.set_index('date', inplace=True)

            # Sort by date
            df.sort_index(inplace=True)

            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"Ticker {ticker} not found")
            elif e.response.status_code == 401:
                print("Authentication failed. Check your API key")
            elif e.response.status_code == 429:
                print("Rate limit exceeded. Please wait before making more requests")
            else:
                print(f"HTTP error occurred: {e}")
            return None

        except requests.exceptions.ConnectionError:
            print("Failed to connect to Tiingo API. Check your internet connection")
            return None

        except requests.exceptions.Timeout:
            print(f"Request timed out for {ticker}")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

        except (ValueError, KeyError) as e:
            print(f"Error parsing response data: {e}")
            return None

    def get_crypto_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        frequency: str = '1day'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch crypto OHLCV data from Tiingo's crypto endpoint.

        Args:
            ticker: Crypto ticker symbol (e.g., 'btcusd', 'ethusd')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency ('1min', '5min', '1hour', '1day')

        Returns:
            pandas DataFrame with columns: date, open, high, low, close, volume
            Returns None if request fails

        Raises:
            ValueError: If date format is invalid
        """
        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        # Enforce rate limiting
        self._rate_limit()

        # Build request URL
        url = f"{self.config.base_url}/tiingo/crypto/prices"
        params = {
            'tickers': ticker,
            'startDate': start_date,
            'endDate': end_date,
            'resampleFreq': frequency
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Parse response
            data = response.json()

            if not data or len(data) == 0:
                print(f"No data returned for {ticker} between {start_date} and {end_date}")
                return None

            # Extract price data (Tiingo crypto returns nested structure)
            if isinstance(data, list) and len(data) > 0:
                if 'priceData' in data[0]:
                    price_data = data[0]['priceData']
                else:
                    price_data = data
            else:
                print(f"Unexpected data format for {ticker}")
                return None

            if not price_data:
                print(f"No price data returned for {ticker}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(price_data)

            # Standardize column names
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'volumeNotional': 'volume_notional',
                'tradesDone': 'trades_done'
            })

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Set date as index
            df.set_index('date', inplace=True)

            # Sort by date
            df.sort_index(inplace=True)

            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"Crypto ticker {ticker} not found")
            elif e.response.status_code == 401:
                print("Authentication failed. Check your API key")
            elif e.response.status_code == 429:
                print("Rate limit exceeded. Please wait before making more requests")
            else:
                print(f"HTTP error occurred: {e}")
            return None

        except requests.exceptions.ConnectionError:
            print("Failed to connect to Tiingo API. Check your internet connection")
            return None

        except requests.exceptions.Timeout:
            print(f"Request timed out for {ticker}")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching crypto data for {ticker}: {e}")
            return None

        except (ValueError, KeyError) as e:
            print(f"Error parsing response data: {e}")
            return None

    def get_fundamentals(self, ticker: str) -> Optional[dict]:
        """
        Fetch market cap and basic metadata for a ticker from Tiingo's daily endpoint.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            Dictionary with ticker metadata including:
                - ticker: Ticker symbol
                - name: Company name
                - exchange: Exchange code
                - startDate: First available date
                - endDate: Last available date
                - description: Company description
            Returns None if request fails
        """
        # Enforce rate limiting
        self._rate_limit()

        # Build request URL
        url = f"{self.config.base_url}/tiingo/daily/{ticker}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse response
            data = response.json()

            if not data:
                print(f"No fundamentals data returned for {ticker}")
                return None

            return data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"Ticker {ticker} not found")
            elif e.response.status_code == 401:
                print("Authentication failed. Check your API key")
            elif e.response.status_code == 429:
                print("Rate limit exceeded. Please wait before making more requests")
            else:
                print(f"HTTP error occurred: {e}")
            return None

        except requests.exceptions.ConnectionError:
            print("Failed to connect to Tiingo API. Check your internet connection")
            return None

        except requests.exceptions.Timeout:
            print(f"Request timed out for {ticker}")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return None

        except (ValueError, KeyError) as e:
            print(f"Error parsing response data: {e}")
            return None
