import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fetcher import TiingoClient
from config.api_config import TiingoConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.api_key = 'test_api_key_12345'
    config.base_url = 'https://api.tiingo.com'
    return config


@pytest.fixture
def tiingo_client(mock_config):
    """Create a TiingoClient instance with mock config."""
    return TiingoClient(mock_config)


class TestTiingoClient:
    """Test suite for TiingoClient class."""

    def test_init(self, mock_config):
        """Test TiingoClient initialization."""
        client = TiingoClient(mock_config)
        assert client.config == mock_config
        assert client.session is not None
        assert client.session.headers['Authorization'] == 'Token test_api_key_12345'
        assert client.min_request_interval == 0.1

    @patch('data.fetcher.time.sleep')
    @patch('data.fetcher.time.time')
    def test_rate_limit(self, mock_time, mock_sleep, tiingo_client):
        """Test rate limiting functionality."""
        # First request - no delay needed
        mock_time.return_value = 0
        tiingo_client.last_request_time = 0
        tiingo_client._rate_limit()
        mock_sleep.assert_not_called()

        # Second request too soon - should sleep
        mock_time.side_effect = [0.05, 0.05, 0.15]
        tiingo_client.last_request_time = 0
        tiingo_client._rate_limit()
        mock_sleep.assert_called_once_with(0.05)

    @patch('data.fetcher.requests.Session.get')
    def test_get_stock_prices_success(self, mock_get, tiingo_client):
        """Test successful stock price fetching."""
        # Mock response data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'date': '2024-01-01T00:00:00.000Z',
                'open': 150.0,
                'high': 155.0,
                'low': 149.0,
                'close': 154.0,
                'volume': 1000000,
                'adjOpen': 150.0,
                'adjHigh': 155.0,
                'adjLow': 149.0,
                'adjClose': 154.0,
                'adjVolume': 1000000
            },
            {
                'date': '2024-01-02T00:00:00.000Z',
                'open': 154.0,
                'high': 158.0,
                'low': 153.0,
                'close': 157.0,
                'volume': 1200000,
                'adjOpen': 154.0,
                'adjHigh': 158.0,
                'adjLow': 153.0,
                'adjClose': 157.0,
                'adjVolume': 1200000
            }
        ]
        mock_get.return_value = mock_response

        # Call method
        df = tiingo_client.get_stock_prices('AAPL', '2024-01-01', '2024-01-02')

        # Assertions
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert df.index.name == 'date'
        assert df['close'].iloc[0] == 154.0
        assert df['close'].iloc[1] == 157.0

    @patch('data.fetcher.requests.Session.get')
    def test_get_stock_prices_invalid_date(self, mock_get, tiingo_client):
        """Test stock price fetching with invalid date format."""
        with pytest.raises(ValueError, match="Invalid date format"):
            tiingo_client.get_stock_prices('AAPL', '2024/01/01', '2024-01-02')

    @patch('data.fetcher.requests.Session.get')
    def test_get_stock_prices_404_error(self, mock_get, tiingo_client, capsys):
        """Test stock price fetching with 404 error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response

        # Mock the HTTPError
        import requests
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)

        df = tiingo_client.get_stock_prices('INVALID', '2024-01-01', '2024-01-02')

        assert df is None
        captured = capsys.readouterr()
        assert "Ticker INVALID not found" in captured.out

    @patch('data.fetcher.requests.Session.get')
    def test_get_stock_prices_auth_error(self, mock_get, tiingo_client, capsys):
        """Test stock price fetching with authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401

        import requests
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        df = tiingo_client.get_stock_prices('AAPL', '2024-01-01', '2024-01-02')

        assert df is None
        captured = capsys.readouterr()
        assert "Authentication failed" in captured.out

    @patch('data.fetcher.requests.Session.get')
    def test_get_stock_prices_rate_limit_error(self, mock_get, tiingo_client, capsys):
        """Test stock price fetching with rate limit error."""
        mock_response = Mock()
        mock_response.status_code = 429

        import requests
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        df = tiingo_client.get_stock_prices('AAPL', '2024-01-01', '2024-01-02')

        assert df is None
        captured = capsys.readouterr()
        assert "Rate limit exceeded" in captured.out

    @patch('data.fetcher.requests.Session.get')
    def test_get_stock_prices_empty_response(self, mock_get, tiingo_client, capsys):
        """Test stock price fetching with empty response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        df = tiingo_client.get_stock_prices('AAPL', '2024-01-01', '2024-01-02')

        assert df is None
        captured = capsys.readouterr()
        assert "No data returned" in captured.out

    @patch('data.fetcher.requests.Session.get')
    def test_get_crypto_prices_success(self, mock_get, tiingo_client):
        """Test successful crypto price fetching."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'ticker': 'btcusd',
                'priceData': [
                    {
                        'date': '2024-01-01T00:00:00.000Z',
                        'open': 40000.0,
                        'high': 41000.0,
                        'low': 39500.0,
                        'close': 40500.0,
                        'volume': 1000.0,
                        'volumeNotional': 40500000.0,
                        'tradesDone': 5000
                    },
                    {
                        'date': '2024-01-02T00:00:00.000Z',
                        'open': 40500.0,
                        'high': 42000.0,
                        'low': 40000.0,
                        'close': 41500.0,
                        'volume': 1200.0,
                        'volumeNotional': 49800000.0,
                        'tradesDone': 6000
                    }
                ]
            }
        ]
        mock_get.return_value = mock_response

        df = tiingo_client.get_crypto_prices('btcusd', '2024-01-01', '2024-01-02')

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert df.index.name == 'date'
        assert df['close'].iloc[0] == 40500.0
        assert df['close'].iloc[1] == 41500.0

    @patch('data.fetcher.requests.Session.get')
    def test_get_crypto_prices_invalid_date(self, mock_get, tiingo_client):
        """Test crypto price fetching with invalid date format."""
        with pytest.raises(ValueError, match="Invalid date format"):
            tiingo_client.get_crypto_prices('btcusd', '01-01-2024', '2024-01-02')

    @patch('data.fetcher.requests.Session.get')
    def test_get_crypto_prices_empty_response(self, mock_get, tiingo_client, capsys):
        """Test crypto price fetching with empty response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        df = tiingo_client.get_crypto_prices('btcusd', '2024-01-01', '2024-01-02')

        assert df is None
        captured = capsys.readouterr()
        assert "No data returned" in captured.out

    @patch('data.fetcher.requests.Session.get')
    def test_get_fundamentals_success(self, mock_get, tiingo_client):
        """Test successful fundamentals fetching."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'ticker': 'AAPL',
            'name': 'Apple Inc.',
            'exchange': 'NASDAQ',
            'startDate': '1980-12-12',
            'endDate': '2024-01-02',
            'description': 'Apple Inc. designs, manufactures, and markets smartphones.'
        }
        mock_get.return_value = mock_response

        data = tiingo_client.get_fundamentals('AAPL')

        assert data is not None
        assert isinstance(data, dict)
        assert data['ticker'] == 'AAPL'
        assert data['name'] == 'Apple Inc.'
        assert data['exchange'] == 'NASDAQ'
        assert 'startDate' in data
        assert 'endDate' in data
        assert 'description' in data

    @patch('data.fetcher.requests.Session.get')
    def test_get_fundamentals_404_error(self, mock_get, tiingo_client, capsys):
        """Test fundamentals fetching with 404 error."""
        mock_response = Mock()
        mock_response.status_code = 404

        import requests
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        data = tiingo_client.get_fundamentals('INVALID')

        assert data is None
        captured = capsys.readouterr()
        assert "Ticker INVALID not found" in captured.out

    @patch('data.fetcher.requests.Session.get')
    def test_get_fundamentals_connection_error(self, mock_get, tiingo_client, capsys):
        """Test fundamentals fetching with connection error."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()

        data = tiingo_client.get_fundamentals('AAPL')

        assert data is None
        captured = capsys.readouterr()
        assert "Failed to connect to Tiingo API" in captured.out

    @patch('data.fetcher.requests.Session.get')
    def test_get_fundamentals_timeout(self, mock_get, tiingo_client, capsys):
        """Test fundamentals fetching with timeout."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        data = tiingo_client.get_fundamentals('AAPL')

        assert data is None
        captured = capsys.readouterr()
        assert "Request timed out" in captured.out
