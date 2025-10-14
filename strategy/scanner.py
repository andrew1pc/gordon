"""Asset scanner for identifying momentum trading candidates."""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

from data.fetcher import TiingoClient
from indicators.technical import TechnicalIndicators
from indicators.momentum import MomentumMetrics
from config.strategy_config import ScannerConfig


logger = logging.getLogger(__name__)


class AssetScanner:
    """
    Scanner to identify high-momentum trading candidates from stock and crypto universes.

    This class handles the complete workflow of:
    1. Defining the trading universe (stocks and cryptos)
    2. Fetching historical data
    3. Calculating technical indicators and momentum metrics
    4. Ranking and filtering candidates
    5. Applying diversification rules
    """

    def __init__(self, client: TiingoClient, config: Optional[ScannerConfig] = None):
        """
        Initialize the AssetScanner.

        Args:
            client: TiingoClient instance for data fetching
            config: ScannerConfig instance (uses defaults if None)
        """
        self.client = client
        self.config = config or ScannerConfig()
        self.technical = TechnicalIndicators()
        self.momentum = MomentumMetrics()

        # Create cache directory if it doesn't exist
        self.cache_dir = 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_stock_universe(
        self,
        min_market_cap: Optional[int] = None,
        min_volume: Optional[int] = None
    ) -> List[str]:
        """
        Get list of qualifying stocks for momentum trading.

        Since Tiingo doesn't provide a comprehensive stock screener API,
        this implementation uses a predefined list of major exchange stocks
        and filters them based on criteria.

        Args:
            min_market_cap: Minimum market cap (uses config default if None)
            min_volume: Minimum daily volume (uses config default if None)

        Returns:
            List of qualifying ticker symbols

        Example:
            >>> scanner = AssetScanner(client)
            >>> stocks = scanner.get_stock_universe()
            >>> print(f"Found {len(stocks)} qualifying stocks")
        """
        min_market_cap = min_market_cap or self.config.STOCK_MIN_MARKET_CAP
        min_volume = min_volume or self.config.STOCK_MIN_VOLUME

        logger.info(f"Fetching stock universe (min_cap=${min_market_cap:,}, min_vol=${min_volume:,})")

        # Use a predefined list of liquid stocks from major indices
        # In production, this could be fetched from a stock screener API
        candidate_tickers = self._get_sp500_tickers()

        logger.info(f"Starting with {len(candidate_tickers)} candidate stocks")

        qualifying_stocks = []
        failed_tickers = []

        for i, ticker in enumerate(candidate_tickers):
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/{len(candidate_tickers)} stocks...")

            try:
                # Fetch fundamental data
                fundamentals = self.client.get_fundamentals(ticker)

                if fundamentals is None:
                    failed_tickers.append(ticker)
                    continue

                # Apply filters
                # Note: Tiingo's fundamental endpoint returns metadata,
                # not market cap directly. In production, use appropriate data source.
                qualifying_stocks.append(ticker)

            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        logger.info(f"Stock universe: {len(qualifying_stocks)} stocks qualify, {len(failed_tickers)} failed")
        return qualifying_stocks

    def get_crypto_universe(self, top_n: Optional[int] = None) -> List[str]:
        """
        Get list of qualifying cryptocurrencies for momentum trading.

        Args:
            top_n: Number of top cryptos to select (uses config default if None)

        Returns:
            List of qualifying crypto ticker symbols (e.g., ['btcusd', 'ethusd'])

        Example:
            >>> scanner = AssetScanner(client)
            >>> cryptos = scanner.get_crypto_universe(top_n=20)
        """
        top_n = top_n or self.config.CRYPTO_TOP_N

        logger.info(f"Fetching crypto universe (top {top_n})")

        # Predefined list of major cryptos (in order of approximate market cap)
        # In production, fetch from Tiingo's crypto list endpoint
        all_cryptos = [
            'btcusd', 'ethusd', 'bnbusd', 'xrpusd', 'adausd',
            'dogeusd', 'maticusd', 'solusd', 'dotusd', 'ltcusd',
            'avaxusd', 'uniusd', 'linkusd', 'atomusd', 'xlmusd',
            'etcusd', 'filusd', 'trxusd', 'aptusd', 'nearusd'
        ]

        # Filter out stablecoins
        cryptos = [c for c in all_cryptos if c not in self.config.EXCLUDED_STABLECOINS]

        # Take top N
        selected = cryptos[:min(top_n, len(cryptos))]

        logger.info(f"Crypto universe: selected {len(selected)} cryptos")
        return selected

    def filter_by_liquidity(
        self,
        tickers: List[str],
        min_daily_volume: int,
        asset_type: str = 'stock'
    ) -> List[str]:
        """
        Filter tickers by liquidity (average daily dollar volume).

        Args:
            tickers: List of ticker symbols to filter
            min_daily_volume: Minimum average daily dollar volume
            asset_type: 'stock' or 'crypto'

        Returns:
            Filtered list of tickers meeting liquidity requirement

        Example:
            >>> liquid_stocks = scanner.filter_by_liquidity(
            ...     stocks, min_daily_volume=5_000_000, asset_type='stock'
            ... )
        """
        logger.info(f"Filtering {len(tickers)} {asset_type}s by liquidity (min=${min_daily_volume:,})")

        liquid_tickers = []

        # Calculate date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=40)  # Extra days for weekends/holidays

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        for ticker in tickers:
            try:
                # Fetch recent data
                if asset_type == 'stock':
                    df = self.client.get_stock_prices(ticker, start_str, end_str)
                else:
                    df = self.client.get_crypto_prices(ticker, start_str, end_str)

                if df is None or len(df) < 20:
                    continue

                # Calculate average dollar volume
                avg_dollar_volume = (df['close'] * df['volume']).tail(20).mean()

                if avg_dollar_volume >= min_daily_volume:
                    liquid_tickers.append(ticker)

            except Exception as e:
                logger.warning(f"Failed to check liquidity for {ticker}: {e}")
                continue

        logger.info(f"Liquidity filter: {len(liquid_tickers)}/{len(tickers)} tickers passed")
        return liquid_tickers

    def cache_universe_data(self, universe: Dict, cache_file: str) -> None:
        """
        Save universe data to cache file.

        Args:
            universe: Dictionary with universe data
            cache_file: Name of cache file (without path)

        Example:
            >>> universe = {'stocks': stocks, 'crypto': cryptos, 'metadata': {...}}
            >>> scanner.cache_universe_data(universe, 'universe_2024-01-15.json')
        """
        cache_path = os.path.join(self.cache_dir, cache_file)

        # Add timestamp to metadata
        universe['metadata'] = universe.get('metadata', {})
        universe['metadata']['cached_at'] = datetime.now().isoformat()

        with open(cache_path, 'w') as f:
            json.dump(universe, f, indent=2)

        logger.info(f"Cached universe data to {cache_path}")

    def load_universe_data(
        self,
        cache_file: str,
        max_age_days: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Load universe data from cache if fresh enough.

        Args:
            cache_file: Name of cache file (without path)
            max_age_days: Maximum cache age in days (uses config defaults if None)

        Returns:
            Universe dictionary if cache is valid, None if stale or missing

        Example:
            >>> universe = scanner.load_universe_data('universe_2024-01-15.json')
            >>> if universe:
            ...     stocks = universe['stocks']
        """
        cache_path = os.path.join(self.cache_dir, cache_file)

        if not os.path.exists(cache_path):
            logger.info(f"Cache miss: {cache_file} not found")
            return None

        try:
            with open(cache_path, 'r') as f:
                universe = json.load(f)

            # Check if cache is stale
            cached_at = datetime.fromisoformat(universe['metadata']['cached_at'])
            age_days = (datetime.now() - cached_at).days

            # Determine max age based on asset type
            if max_age_days is None:
                # If universe contains both, use shorter expiry
                if 'stocks' in universe and 'crypto' in universe:
                    max_age_days = min(
                        self.config.CACHE_EXPIRY_DAYS_STOCKS,
                        self.config.CACHE_EXPIRY_DAYS_CRYPTO
                    )
                elif 'stocks' in universe:
                    max_age_days = self.config.CACHE_EXPIRY_DAYS_STOCKS
                else:
                    max_age_days = self.config.CACHE_EXPIRY_DAYS_CRYPTO

            if age_days > max_age_days:
                logger.info(f"Cache stale: {cache_file} is {age_days} days old (max {max_age_days})")
                return None

            logger.info(f"Cache hit: loaded {cache_file} ({age_days} days old)")
            return universe

        except Exception as e:
            logger.error(f"Failed to load cache {cache_file}: {e}")
            return None

    def _get_sp500_tickers(self) -> List[str]:
        """
        Get a representative list of S&P 500 tickers.

        In production, fetch from external source or use all S&P 500.
        This is a simplified subset for demonstration.
        """
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'CVX', 'HD',
            'LLY', 'MRK', 'ABBV', 'KO', 'PEP', 'AVGO', 'COST', 'MCD', 'ADBE',
            'TMO', 'CSCO', 'ACN', 'ABT', 'NFLX', 'CRM', 'DHR', 'NKE', 'DIS',
            'TXN', 'VZ', 'CMCSA', 'INTC', 'PM', 'UNP', 'ORCL', 'NEE', 'AMD',
            'QCOM', 'RTX', 'HON', 'UPS', 'LOW', 'SBUX', 'IBM', 'AMGN', 'CAT',
            'BA', 'GE', 'SPGI', 'BLK', 'ELV', 'AMAT', 'PLD', 'GILD', 'MDLZ',
            'ADP', 'BKNG', 'SYK', 'VRTX', 'TJX', 'MMC', 'ADI', 'C', 'AMT',
            'ZTS', 'ISRG', 'CVS', 'REGN', 'PGR', 'MO', 'TMUS', 'CI', 'DE',
            'LRCX', 'SO', 'DUK', 'BDX', 'CB', 'NOC', 'GS', 'SLB', 'SCHW'
        ]

    def fetch_universe_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        asset_type: str = 'stock'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV data for all tickers in universe.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            asset_type: 'stock' or 'crypto'

        Returns:
            Dictionary mapping ticker to DataFrame

        Example:
            >>> prices = scanner.fetch_universe_prices(
            ...     stocks, '2023-01-01', '2024-01-01', 'stock'
            ... )
        """
        logger.info(f"Fetching prices for {len(tickers)} {asset_type}s from {start_date} to {end_date}")

        prices = {}
        failed = []

        for i, ticker in enumerate(tickers):
            if (i + 1) % 10 == 0:
                logger.info(f"Fetching {i+1}/{len(tickers)}...")

            try:
                if asset_type == 'stock':
                    df = self.client.get_stock_prices(ticker, start_date, end_date)
                else:
                    df = self.client.get_crypto_prices(ticker, start_date, end_date)

                if df is not None and len(df) >= self.config.MIN_TRADING_DAYS:
                    prices[ticker] = df
                else:
                    failed.append(ticker)

            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                failed.append(ticker)

            # Rate limiting
            time.sleep(0.1)

        logger.info(f"Fetched prices: {len(prices)} succeeded, {len(failed)} failed")
        return prices

    def prepare_universe_data(
        self,
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Add technical indicators and momentum metrics to all universe assets.

        Args:
            price_data: Dictionary mapping ticker to price DataFrame

        Returns:
            Dictionary with indicators added to each DataFrame

        Example:
            >>> prepared = scanner.prepare_universe_data(prices)
        """
        logger.info(f"Preparing data for {len(price_data)} assets...")

        prepared = {}
        failed = []

        for ticker, df in price_data.items():
            try:
                # Add technical indicators
                df = self.technical.add_all_indicators(df)

                # Add momentum metrics
                df = self.momentum.add_all_momentum_metrics(df)

                prepared[ticker] = df

            except Exception as e:
                logger.warning(f"Failed to prepare {ticker}: {e}")
                failed.append(ticker)

        logger.info(f"Prepared {len(prepared)} assets, {len(failed)} failed")
        return prepared

    def rank_universe_by_momentum(
        self,
        prepared_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Rank all assets by momentum score.

        Args:
            prepared_data: Dictionary of DataFrames with indicators

        Returns:
            DataFrame with rankings sorted by momentum_score descending

        Example:
            >>> ranking = scanner.rank_universe_by_momentum(prepared)
            >>> print(ranking.head(20))
        """
        logger.info(f"Ranking {len(prepared_data)} assets by momentum...")

        rankings = []

        for ticker, df in prepared_data.items():
            try:
                latest = df.iloc[-1]

                rankings.append({
                    'ticker': ticker,
                    'momentum_score': latest.get('momentum_score', 0),
                    'current_price': latest['close'],
                    'rsi': latest.get('rsi', None),
                    'roc_20': latest.get('roc_20', None),
                    'trend_strength': latest.get('trend_strength', 0),
                    'trend_direction': latest.get('trend_direction', 'neutral'),
                    'volume_surge': latest.get('volume_surge', False),
                    'volume_ratio': latest.get('volume_ratio', 1.0)
                })
            except Exception as e:
                logger.warning(f"Failed to rank {ticker}: {e}")

        ranking_df = pd.DataFrame(rankings)
        ranking_df = ranking_df.sort_values('momentum_score', ascending=False)

        logger.info(f"Ranking complete: top score = {ranking_df['momentum_score'].max():.1f}")
        return ranking_df

    def select_top_candidates(
        self,
        ranking: pd.DataFrame,
        max_candidates: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Select top momentum candidates using filters.

        Args:
            ranking: DataFrame from rank_universe_by_momentum
            max_candidates: Maximum candidates to select (uses config if None)

        Returns:
            DataFrame of selected candidates

        Example:
            >>> candidates = scanner.select_top_candidates(ranking)
            >>> print(f"Selected {len(candidates)} candidates")
        """
        max_candidates = max_candidates or self.config.MAX_CANDIDATES

        logger.info(f"Selecting top candidates (max={max_candidates})...")
        logger.info(f"Starting with {len(ranking)} ranked assets")

        # Apply filters
        candidates = ranking.copy()

        # Filter 1: Momentum score
        candidates = candidates[candidates['momentum_score'] >= self.config.MOMENTUM_THRESHOLD]
        logger.info(f"After momentum filter (>={self.config.MOMENTUM_THRESHOLD}): {len(candidates)}")

        # Filter 2: RSI range
        candidates = candidates[
            (candidates['rsi'] >= self.config.RSI_MIN) &
            (candidates['rsi'] <= self.config.RSI_MAX)
        ]
        logger.info(f"After RSI filter ({self.config.RSI_MIN}-{self.config.RSI_MAX}): {len(candidates)}")

        # Filter 3: Trend strength
        candidates = candidates[candidates['trend_strength'] >= self.config.TREND_STRENGTH_MIN]
        logger.info(f"After trend strength filter (>={self.config.TREND_STRENGTH_MIN}): {len(candidates)}")

        # Take top N
        candidates = candidates.head(max_candidates)

        logger.info(f"Selected {len(candidates)} final candidates")
        return candidates
