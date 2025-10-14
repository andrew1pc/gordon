"""Configuration parameters for trading strategy and asset scanner."""

import os
import yaml
from typing import Optional, Dict, Any


class ScannerConfig:
    """
    Configuration parameters for the asset scanner.

    These parameters control how assets are filtered and selected
    for momentum trading.

    Attributes:
        Stock Filtering:
        - STOCK_MIN_MARKET_CAP: Minimum market capitalization ($500M default)
          Filters out small-cap stocks with higher volatility/risk
        - STOCK_MIN_VOLUME: Minimum average daily dollar volume ($5M default)
          Ensures sufficient liquidity for position entry/exit
        - STOCK_MIN_PRICE: Minimum price per share ($5.00 default)
          Avoids penny stocks which have different risk characteristics

        Crypto Filtering:
        - CRYPTO_TOP_N: Select top N cryptos by market cap (50 default)
          Focuses on established cryptos with sufficient liquidity
        - CRYPTO_MIN_VOLUME: Minimum average daily dollar volume ($1M default)
          Ensures crypto can be traded without excessive slippage

        Momentum Criteria:
        - MOMENTUM_THRESHOLD: Minimum momentum score for candidates (70 default)
          Higher = stricter selection, fewer but stronger signals
        - RSI_MIN: Minimum RSI to avoid oversold conditions (50 default)
        - RSI_MAX: Maximum RSI to avoid overbought conditions (70 default)
        - TREND_STRENGTH_MIN: Minimum trend strength score (60 default)

        Portfolio Constraints:
        - MAX_CANDIDATES: Maximum number of candidates to select (15 default)
          Limits portfolio to manageable size
        - MAX_PER_SECTOR: Maximum stocks from same sector (3 default)
          Enforces sector diversification

        Caching:
        - CACHE_EXPIRY_DAYS_STOCKS: Days before stock universe cache expires (7)
        - CACHE_EXPIRY_DAYS_CRYPTO: Days before crypto universe cache expires (1)
          Crypto markets move faster, so refresh more frequently
    """

    # Stock filtering parameters
    STOCK_MIN_MARKET_CAP: int = 500_000_000  # $500M
    STOCK_MIN_VOLUME: int = 5_000_000  # $5M average daily dollar volume
    STOCK_MIN_PRICE: float = 5.0  # Avoid penny stocks

    # Crypto filtering parameters
    CRYPTO_TOP_N: int = 50  # Top 50 by market cap
    CRYPTO_MIN_VOLUME: int = 1_000_000  # $1M average daily dollar volume

    # Momentum criteria
    MOMENTUM_THRESHOLD: int = 70  # Minimum momentum score
    RSI_MIN: int = 50  # Avoid oversold
    RSI_MAX: int = 70  # Avoid overbought
    TREND_STRENGTH_MIN: int = 60  # Minimum trend strength score

    # Portfolio constraints
    MAX_CANDIDATES: int = 15  # Total candidates to select
    MAX_PER_SECTOR: int = 3  # Max stocks from same sector

    # Caching configuration
    CACHE_EXPIRY_DAYS_STOCKS: int = 7  # Refresh weekly
    CACHE_EXPIRY_DAYS_CRYPTO: int = 1  # Refresh daily

    # Data fetching
    LOOKBACK_DAYS: int = 365  # Fetch 1 year of historical data
    MIN_TRADING_DAYS: int = 250  # Minimum trading history required

    # Stablecoins to exclude (don't have momentum)
    EXCLUDED_STABLECOINS: list = [
        'usdtusd', 'usdcusd', 'daiusd', 'busdusd', 'ustusd',
        'tusd', 'paxusd', 'gusd', 'usdsusdt'
    ]

    @classmethod
    def from_yaml(cls, file_path: str) -> 'ScannerConfig':
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            ScannerConfig instance with parameters from file

        Example YAML format:
            stock_filters:
              min_market_cap: 500000000
              min_volume: 5000000
              min_price: 5.0
            momentum_criteria:
              threshold: 70
              rsi_min: 50
              rsi_max: 70

        Example:
            >>> config = ScannerConfig.from_yaml('config/scanner.yaml')
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        # Update from YAML
        if 'stock_filters' in data:
            stock_filters = data['stock_filters']
            config.STOCK_MIN_MARKET_CAP = stock_filters.get('min_market_cap', config.STOCK_MIN_MARKET_CAP)
            config.STOCK_MIN_VOLUME = stock_filters.get('min_volume', config.STOCK_MIN_VOLUME)
            config.STOCK_MIN_PRICE = stock_filters.get('min_price', config.STOCK_MIN_PRICE)

        if 'crypto_filters' in data:
            crypto_filters = data['crypto_filters']
            config.CRYPTO_TOP_N = crypto_filters.get('top_n', config.CRYPTO_TOP_N)
            config.CRYPTO_MIN_VOLUME = crypto_filters.get('min_volume', config.CRYPTO_MIN_VOLUME)

        if 'momentum_criteria' in data:
            momentum = data['momentum_criteria']
            config.MOMENTUM_THRESHOLD = momentum.get('threshold', config.MOMENTUM_THRESHOLD)
            config.RSI_MIN = momentum.get('rsi_min', config.RSI_MIN)
            config.RSI_MAX = momentum.get('rsi_max', config.RSI_MAX)
            config.TREND_STRENGTH_MIN = momentum.get('trend_strength_min', config.TREND_STRENGTH_MIN)

        if 'portfolio' in data:
            portfolio = data['portfolio']
            config.MAX_CANDIDATES = portfolio.get('max_candidates', config.MAX_CANDIDATES)
            config.MAX_PER_SECTOR = portfolio.get('max_per_sector', config.MAX_PER_SECTOR)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary of all configuration parameters
        """
        return {
            'stock_filters': {
                'min_market_cap': self.STOCK_MIN_MARKET_CAP,
                'min_volume': self.STOCK_MIN_VOLUME,
                'min_price': self.STOCK_MIN_PRICE,
            },
            'crypto_filters': {
                'top_n': self.CRYPTO_TOP_N,
                'min_volume': self.CRYPTO_MIN_VOLUME,
            },
            'momentum_criteria': {
                'threshold': self.MOMENTUM_THRESHOLD,
                'rsi_min': self.RSI_MIN,
                'rsi_max': self.RSI_MAX,
                'trend_strength_min': self.TREND_STRENGTH_MIN,
            },
            'portfolio': {
                'max_candidates': self.MAX_CANDIDATES,
                'max_per_sector': self.MAX_PER_SECTOR,
            },
            'caching': {
                'expiry_days_stocks': self.CACHE_EXPIRY_DAYS_STOCKS,
                'expiry_days_crypto': self.CACHE_EXPIRY_DAYS_CRYPTO,
            }
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ScannerConfig(momentum_threshold={self.MOMENTUM_THRESHOLD}, max_candidates={self.MAX_CANDIDATES})"


class RiskConfig:
    """
    Configuration parameters for risk management.

    These parameters control position sizing, exposure limits, and risk controls.

    Attributes:
        Position Sizing:
        - RISK_PER_TRADE: Risk per trade as decimal (0.01 = 1% of account)
          Conservative: 0.5-1%, Moderate: 1-2%, Aggressive: 2-3%
        - MIN_POSITION_PCT: Minimum position size (0.01 = 1% of account)
          Ensures positions are meaningful
        - MAX_POSITION_PCT: Maximum position size (0.05 = 5% of account)
          Prevents over-concentration in single position
        - CRYPTO_SIZE_MULTIPLIER: Reduce crypto positions (0.7 = 70% of calculated size)
          Accounts for higher crypto volatility

        Portfolio Limits:
        - MAX_POSITIONS: Maximum concurrent positions (8 default)
          Limits complexity and ensures focused portfolio
        - MAX_PORTFOLIO_EXPOSURE: Max % of capital in positions (0.25 = 25%)
          Keeps majority in cash for new opportunities
        - MAX_SECTOR_EXPOSURE: Max % in single sector (0.15 = 15%)
          Enforces sector diversification

        Risk Controls:
        - DAILY_LOSS_LIMIT: Circuit breaker (0.03 = 3% of account)
          Stops trading for the day if hit
        - MAX_CORRELATION: Maximum correlation between positions (0.7)
          Future enhancement for correlation-based diversification
    """

    # Position sizing parameters
    RISK_PER_TRADE: float = 0.01  # 1% of account per trade
    MIN_POSITION_PCT: float = 0.01  # Minimum 1% position
    MAX_POSITION_PCT: float = 0.05  # Maximum 5% position
    CRYPTO_SIZE_MULTIPLIER: float = 0.7  # 70% of calculated size for crypto

    # Portfolio limits
    MAX_POSITIONS: int = 8  # Maximum concurrent positions
    MAX_PORTFOLIO_EXPOSURE: float = 0.25  # 25% maximum in market
    MAX_SECTOR_EXPOSURE: float = 0.15  # 15% maximum per sector

    # Risk controls
    DAILY_LOSS_LIMIT: float = 0.03  # 3% daily loss circuit breaker
    MAX_CORRELATION: float = 0.7  # Maximum correlation between positions

    @classmethod
    def from_yaml(cls, file_path: str) -> 'RiskConfig':
        """
        Load risk configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            RiskConfig instance with parameters from file

        Example YAML format:
            position_sizing:
              risk_per_trade: 0.01
              min_position_pct: 0.01
              max_position_pct: 0.05
              crypto_size_multiplier: 0.7
            portfolio_limits:
              max_positions: 8
              max_portfolio_exposure: 0.25
              max_sector_exposure: 0.15
            risk_controls:
              daily_loss_limit: 0.03

        Example:
            >>> config = RiskConfig.from_yaml('config/risk.yaml')
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        # Update from YAML
        if 'position_sizing' in data:
            ps = data['position_sizing']
            config.RISK_PER_TRADE = ps.get('risk_per_trade', config.RISK_PER_TRADE)
            config.MIN_POSITION_PCT = ps.get('min_position_pct', config.MIN_POSITION_PCT)
            config.MAX_POSITION_PCT = ps.get('max_position_pct', config.MAX_POSITION_PCT)
            config.CRYPTO_SIZE_MULTIPLIER = ps.get('crypto_size_multiplier', config.CRYPTO_SIZE_MULTIPLIER)

        if 'portfolio_limits' in data:
            pl = data['portfolio_limits']
            config.MAX_POSITIONS = pl.get('max_positions', config.MAX_POSITIONS)
            config.MAX_PORTFOLIO_EXPOSURE = pl.get('max_portfolio_exposure', config.MAX_PORTFOLIO_EXPOSURE)
            config.MAX_SECTOR_EXPOSURE = pl.get('max_sector_exposure', config.MAX_SECTOR_EXPOSURE)

        if 'risk_controls' in data:
            rc = data['risk_controls']
            config.DAILY_LOSS_LIMIT = rc.get('daily_loss_limit', config.DAILY_LOSS_LIMIT)
            config.MAX_CORRELATION = rc.get('max_correlation', config.MAX_CORRELATION)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary of all configuration parameters
        """
        return {
            'position_sizing': {
                'risk_per_trade': self.RISK_PER_TRADE,
                'min_position_pct': self.MIN_POSITION_PCT,
                'max_position_pct': self.MAX_POSITION_PCT,
                'crypto_size_multiplier': self.CRYPTO_SIZE_MULTIPLIER,
            },
            'portfolio_limits': {
                'max_positions': self.MAX_POSITIONS,
                'max_portfolio_exposure': self.MAX_PORTFOLIO_EXPOSURE,
                'max_sector_exposure': self.MAX_SECTOR_EXPOSURE,
            },
            'risk_controls': {
                'daily_loss_limit': self.DAILY_LOSS_LIMIT,
                'max_correlation': self.MAX_CORRELATION,
            }
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"RiskConfig(risk_per_trade={self.RISK_PER_TRADE*100:.1f}%, max_positions={self.MAX_POSITIONS})"
