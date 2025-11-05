"""Risk management module for position sizing and portfolio exposure control."""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Information about an open position."""
    ticker: str
    asset_type: str
    sector: Optional[str]
    entry_price: float
    current_price: float
    position_size: float  # Number of shares/units
    dollar_value: float  # Current market value
    stop_loss: float
    entry_date: datetime


class RiskManager:
    """
    Risk management system for position sizing and portfolio exposure control.

    This class enforces trading rules to protect capital:
    - Position sizing based on risk per trade
    - Maximum number of concurrent positions
    - Portfolio exposure limits
    - Sector concentration limits
    - Daily loss limits (circuit breaker)

    Example:
        >>> risk_mgr = RiskManager(account_size=100000, risk_per_trade=0.01)
        >>> size = risk_mgr.calculate_position_size(
        ...     entry_price=100, stop_loss=93, asset_type='stock'
        ... )
        >>> if risk_mgr.validate_new_trade('AAPL', 100, 93, 'Technology', 'stock'):
        ...     print(f"Trade approved: {size} shares")
    """

    def __init__(
        self,
        account_size: float,
        risk_per_trade: float = 0.01,
        max_positions: int = 8,
        max_portfolio_exposure: float = 0.25,
        max_sector_exposure: float = 0.15,
        daily_loss_limit: float = 0.03,
        crypto_size_multiplier: float = 0.7
    ):
        """
        Initialize the RiskManager.

        Args:
            account_size: Total trading capital
            risk_per_trade: Risk per trade as decimal (0.01 = 1%)
            max_positions: Maximum concurrent positions (default 8)
            max_portfolio_exposure: Max % of account in positions (0.25 = 25%)
            max_sector_exposure: Max % in single sector (0.15 = 15%)
            daily_loss_limit: Daily loss circuit breaker (0.03 = 3%)
            crypto_size_multiplier: Reduce crypto positions by this factor (0.7 = 70%)
        """
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.max_portfolio_exposure = max_portfolio_exposure
        self.max_sector_exposure = max_sector_exposure
        self.daily_loss_limit = daily_loss_limit
        self.crypto_size_multiplier = crypto_size_multiplier

        # Track open positions
        self.positions: Dict[str, PositionInfo] = {}

        # Track daily P&L
        self.daily_pnl: Dict[date, float] = {}
        self.starting_account_value = account_size
        self.today_starting_value = account_size

        logger.info(
            f"RiskManager initialized: account=${account_size:,.0f}, "
            f"risk_per_trade={risk_per_trade*100:.1f}%, "
            f"max_positions={max_positions}"
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        asset_type: str = 'stock',
        signal_strength: int = 100
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk per trade and signal strength.

        Formula: position_size = (account_size × risk_per_trade × scale_factor) / (entry_price - stop_loss)

        Signal strength scaling (hardcoded):
        - 90-100: 100% size (full position)
        - 80-89:  80% size (slightly reduced)
        - 70-79:  60% size (significantly reduced)

        Args:
            entry_price: Planned entry price
            stop_loss: Stop loss price
            asset_type: 'stock' or 'crypto'
            signal_strength: Signal strength 0-100 (default 100 for backward compatibility)

        Returns:
            Tuple of (position_size in shares/units, dollar_value)

        Example:
            >>> risk_mgr = RiskManager(100000, risk_per_trade=0.01)
            >>> size, value = risk_mgr.calculate_position_size(100, 93, 'stock', signal_strength=85)
            >>> print(f"{size:.0f} shares = ${value:,.0f}")
            113 shares = $11,360  # 80% of full size
        """
        if entry_price <= stop_loss:
            logger.error(f"Invalid prices: entry={entry_price}, stop={stop_loss}")
            raise ValueError("Entry price must be greater than stop loss")

        if entry_price <= 0 or stop_loss <= 0:
            raise ValueError("Prices must be positive")

        # Calculate dollar risk per trade
        dollar_risk = self.account_size * self.risk_per_trade

        # Apply signal strength scaling
        scale_factor = self.calculate_scaled_position_size(signal_strength)
        scaled_dollar_risk = dollar_risk * scale_factor

        logger.debug(
            f"Signal strength: {signal_strength}/100 → scale factor: {scale_factor:.0%}"
        )

        # Calculate risk per share/unit
        risk_per_unit = entry_price - stop_loss

        # Calculate position size
        position_size = scaled_dollar_risk / risk_per_unit

        # Apply crypto multiplier
        if asset_type == 'crypto':
            position_size *= self.crypto_size_multiplier
            logger.debug(f"Applied crypto multiplier: {self.crypto_size_multiplier}")

        # Calculate dollar value
        dollar_value = position_size * entry_price

        # Enforce minimum constraint: at least 1% of account
        min_position_value = self.account_size * 0.01
        if dollar_value < min_position_value:
            logger.warning(
                f"Position too small (${dollar_value:,.0f}), "
                f"increasing to minimum ${min_position_value:,.0f}"
            )
            dollar_value = min_position_value
            position_size = dollar_value / entry_price

        # Enforce maximum constraint: no more than 5% of account
        max_position_value = self.account_size * 0.05
        if dollar_value > max_position_value:
            logger.warning(
                f"Position too large (${dollar_value:,.0f}), "
                f"reducing to maximum ${max_position_value:,.0f}"
            )
            dollar_value = max_position_value
            position_size = dollar_value / entry_price

        logger.info(
            f"Position size: {position_size:.2f} units @ ${entry_price:.2f} = "
            f"${dollar_value:,.0f} (risk: ${dollar_risk:,.0f}, scaled: {scale_factor:.0%})"
        )

        return position_size, dollar_value

    def calculate_scaled_position_size(self, signal_strength: int) -> float:
        """
        Calculate position size scaling factor based on signal strength.

        Hardcoded tiers:
        - 90-100: 1.0 (100% size - strongest signals)
        - 80-89:  0.8 (80% size - good signals)
        - 70-79:  0.6 (60% size - marginal signals)

        Args:
            signal_strength: Signal strength score 0-100

        Returns:
            Scale factor (0.6, 0.8, or 1.0)

        Example:
            >>> risk_mgr = RiskManager(100000)
            >>> risk_mgr.calculate_scaled_position_size(95)  # Returns 1.0
            >>> risk_mgr.calculate_scaled_position_size(85)  # Returns 0.8
            >>> risk_mgr.calculate_scaled_position_size(75)  # Returns 0.6
        """
        if signal_strength >= 90:
            return 1.0  # Full size
        elif signal_strength >= 80:
            return 0.8  # 80% size
        elif signal_strength >= 70:
            return 0.6  # 60% size
        else:
            # Below 70 shouldn't generate signals, but handle gracefully
            logger.warning(
                f"Signal strength {signal_strength} below threshold (70), "
                f"using minimum scale factor"
            )
            return 0.6

    def get_portfolio_exposure(self) -> float:
        """
        Calculate current portfolio exposure as percentage of account.

        Returns:
            Total dollar value of all positions / account_size

        Example:
            >>> exposure = risk_mgr.get_portfolio_exposure()
            >>> print(f"Portfolio exposure: {exposure*100:.1f}%")
        """
        total_value = sum(pos.dollar_value for pos in self.positions.values())
        exposure = total_value / self.account_size

        logger.debug(f"Portfolio exposure: ${total_value:,.0f} / ${self.account_size:,.0f} = {exposure*100:.1f}%")
        return exposure

    def get_sector_exposure(self, sector: Optional[str]) -> float:
        """
        Calculate exposure to a specific sector.

        Args:
            sector: Sector name (e.g., 'Technology', 'Financial')

        Returns:
            Sector exposure as percentage of account

        Example:
            >>> tech_exposure = risk_mgr.get_sector_exposure('Technology')
        """
        if sector is None:
            return 0.0

        sector_value = sum(
            pos.dollar_value for pos in self.positions.values()
            if pos.sector == sector
        )
        exposure = sector_value / self.account_size

        logger.debug(f"{sector} exposure: ${sector_value:,.0f} = {exposure*100:.1f}%")
        return exposure

    def can_add_position(
        self,
        ticker: str,
        position_value: float,
        sector: Optional[str]
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be added without violating limits.

        Args:
            ticker: Ticker symbol
            position_value: Dollar value of proposed position
            sector: Sector classification

        Returns:
            Tuple of (can_add: bool, reason: str)

        Example:
            >>> can_add, reason = risk_mgr.can_add_position('AAPL', 5000, 'Technology')
            >>> if not can_add:
            ...     print(f"Trade rejected: {reason}")
        """
        # Check 1: Maximum positions
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Check 2: Already have this position
        if ticker in self.positions:
            return False, f"Already have position in {ticker}"

        # Check 3: Portfolio exposure
        current_exposure = self.get_portfolio_exposure()
        new_exposure = (sum(pos.dollar_value for pos in self.positions.values()) + position_value) / self.account_size

        if new_exposure > self.max_portfolio_exposure:
            return False, (
                f"Portfolio exposure would exceed limit: "
                f"{new_exposure*100:.1f}% > {self.max_portfolio_exposure*100:.1f}%"
            )

        # Check 4: Sector exposure (if sector provided)
        if sector is not None:
            current_sector_exposure = self.get_sector_exposure(sector)
            new_sector_value = sum(
                pos.dollar_value for pos in self.positions.values()
                if pos.sector == sector
            ) + position_value
            new_sector_exposure = new_sector_value / self.account_size

            if new_sector_exposure > self.max_sector_exposure:
                return False, (
                    f"Sector exposure would exceed limit: "
                    f"{sector} {new_sector_exposure*100:.1f}% > {self.max_sector_exposure*100:.1f}%"
                )

        return True, "OK"

    def check_daily_loss_limit(self) -> Tuple[bool, float]:
        """
        Check if daily loss limit has been hit (circuit breaker).

        Returns:
            Tuple of (trading_allowed: bool, daily_pnl: float)

        Example:
            >>> allowed, pnl = risk_mgr.check_daily_loss_limit()
            >>> if not allowed:
            ...     print(f"Trading halted: daily loss ${pnl:,.0f}")
        """
        today = date.today()

        # Get today's P&L
        daily_pnl = self.daily_pnl.get(today, 0.0)

        # Calculate daily loss percentage
        daily_loss_pct = abs(daily_pnl) / self.today_starting_value if daily_pnl < 0 else 0

        # Check if limit exceeded
        if daily_loss_pct > self.daily_loss_limit:
            logger.warning(
                f"CIRCUIT BREAKER: Daily loss limit hit: "
                f"{daily_loss_pct*100:.2f}% > {self.daily_loss_limit*100:.1f}%"
            )
            return False, daily_pnl

        return True, daily_pnl

    def validate_new_trade(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        sector: Optional[str],
        asset_type: str = 'stock',
        signal_strength: int = 100
    ) -> Tuple[bool, str, Optional[float], Optional[float]]:
        """
        Comprehensive validation for a new trade with signal strength scaling.

        Checks all risk management rules before approving trade.
        Position size scales based on signal strength (90+=100%, 80-89=80%, 70-79=60%).

        Args:
            ticker: Ticker symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            sector: Sector classification
            asset_type: 'stock' or 'crypto'
            signal_strength: Signal strength 0-100 (default 100 for backward compatibility)

        Returns:
            Tuple of (approved: bool, reason: str, position_size: float or None, dollar_value: float or None)

        Example:
            >>> approved, reason, size, value = risk_mgr.validate_new_trade(
            ...     'AAPL', 150, 140, 'Technology', 'stock', signal_strength=85
            ... )
            >>> if approved:
            ...     print(f"Trade approved: {size} shares = ${value:,.0f} (80% size)")
            >>> else:
            ...     print(f"Trade rejected: {reason}")
        """
        # Check 1: Daily loss limit
        trading_allowed, daily_pnl = self.check_daily_loss_limit()
        if not trading_allowed:
            return False, f"Daily loss limit exceeded: ${daily_pnl:,.0f}", None, None

        # Check 2: Calculate position size (with signal strength scaling)
        try:
            position_size, dollar_value = self.calculate_position_size(
                entry_price, stop_loss, asset_type, signal_strength
            )
        except ValueError as e:
            return False, f"Invalid position parameters: {e}", None, None

        # Check 3: Can add position?
        can_add, reason = self.can_add_position(ticker, dollar_value, sector)
        if not can_add:
            return False, reason, None, None

        # All checks passed
        scale_factor = self.calculate_scaled_position_size(signal_strength)
        logger.info(
            f"Trade validation passed: {ticker} {position_size:.2f} units @ "
            f"${entry_price:.2f} = ${dollar_value:,.0f} "
            f"(signal: {signal_strength}/100, scale: {scale_factor:.0%})"
        )
        return True, "Trade approved", position_size, dollar_value

    def add_position(
        self,
        ticker: str,
        asset_type: str,
        sector: Optional[str],
        entry_price: float,
        position_size: float,
        stop_loss: float,
        entry_date: Optional[datetime] = None
    ) -> None:
        """
        Add a new position to the portfolio.

        Args:
            ticker: Ticker symbol
            asset_type: 'stock' or 'crypto'
            sector: Sector classification
            entry_price: Entry price
            position_size: Number of shares/units
            stop_loss: Stop loss price
            entry_date: Entry date (defaults to now)

        Example:
            >>> risk_mgr.add_position('AAPL', 'stock', 'Technology', 150, 100, 140)
        """
        if entry_date is None:
            entry_date = datetime.now()

        dollar_value = position_size * entry_price

        position = PositionInfo(
            ticker=ticker,
            asset_type=asset_type,
            sector=sector,
            entry_price=entry_price,
            current_price=entry_price,
            position_size=position_size,
            dollar_value=dollar_value,
            stop_loss=stop_loss,
            entry_date=entry_date
        )

        self.positions[ticker] = position

        logger.info(
            f"Position added: {ticker} {position_size:.2f} @ ${entry_price:.2f} "
            f"(stop: ${stop_loss:.2f})"
        )

    def remove_position(self, ticker: str, exit_price: float) -> Optional[float]:
        """
        Remove a position from the portfolio and calculate P&L.

        Args:
            ticker: Ticker symbol
            exit_price: Exit price

        Returns:
            Realized P&L or None if position not found

        Example:
            >>> pnl = risk_mgr.remove_position('AAPL', 160)
            >>> print(f"Realized P&L: ${pnl:,.0f}")
        """
        if ticker not in self.positions:
            logger.warning(f"Cannot remove {ticker}: position not found")
            return None

        position = self.positions[ticker]
        pnl = (exit_price - position.entry_price) * position.position_size

        # Update daily P&L
        today = date.today()
        self.daily_pnl[today] = self.daily_pnl.get(today, 0.0) + pnl

        # Update account size
        self.account_size += pnl

        del self.positions[ticker]

        logger.info(
            f"Position closed: {ticker} {position.position_size:.2f} @ ${exit_price:.2f} "
            f"(entry: ${position.entry_price:.2f}, P&L: ${pnl:,.0f})"
        )

        return pnl

    def update_position_price(self, ticker: str, current_price: float) -> None:
        """
        Update the current price of a position.

        Args:
            ticker: Ticker symbol
            current_price: Current market price

        Example:
            >>> risk_mgr.update_position_price('AAPL', 155)
        """
        if ticker not in self.positions:
            logger.warning(f"Cannot update {ticker}: position not found")
            return

        position = self.positions[ticker]
        position.current_price = current_price
        position.dollar_value = position.position_size * current_price

        logger.debug(f"Updated {ticker} price: ${current_price:.2f}")

    def reset_daily_tracking(self) -> None:
        """
        Reset daily tracking at start of new trading day.

        Call this at the beginning of each trading day.

        Example:
            >>> risk_mgr.reset_daily_tracking()
        """
        today = date.today()

        # If we have positions from yesterday, don't reset starting value
        if today not in self.daily_pnl:
            self.today_starting_value = self.account_size
            self.daily_pnl[today] = 0.0

        logger.info(f"Daily tracking reset for {today}, starting value: ${self.today_starting_value:,.0f}")

    def get_risk_metrics(self) -> Dict:
        """
        Get comprehensive risk metrics dashboard.

        Returns:
            Dictionary with all risk metrics

        Example:
            >>> metrics = risk_mgr.get_risk_metrics()
            >>> print(f"Portfolio exposure: {metrics['portfolio_exposure']*100:.1f}%")
        """
        # Calculate unrealized P&L
        unrealized_pnl = sum(
            (pos.current_price - pos.entry_price) * pos.position_size
            for pos in self.positions.values()
        )

        # Get sector breakdown
        sector_breakdown = {}
        for pos in self.positions.values():
            if pos.sector:
                sector_breakdown[pos.sector] = sector_breakdown.get(pos.sector, 0) + pos.dollar_value

        # Today's P&L
        today = date.today()
        daily_pnl = self.daily_pnl.get(today, 0.0)

        metrics = {
            'account_size': self.account_size,
            'num_positions': len(self.positions),
            'max_positions': self.max_positions,
            'portfolio_exposure': self.get_portfolio_exposure(),
            'max_portfolio_exposure': self.max_portfolio_exposure,
            'unrealized_pnl': unrealized_pnl,
            'daily_pnl': daily_pnl,
            'daily_loss_limit': self.daily_loss_limit,
            'daily_loss_remaining': self.today_starting_value * self.daily_loss_limit + daily_pnl,
            'sector_breakdown': sector_breakdown,
            'positions': {
                ticker: {
                    'value': pos.dollar_value,
                    'unrealized_pnl': (pos.current_price - pos.entry_price) * pos.position_size,
                    'sector': pos.sector
                }
                for ticker, pos in self.positions.items()
            }
        }

        return metrics

    def apply_regime_adjustments(
        self,
        risk_multiplier: float,
        max_positions: int
    ) -> None:
        """
        Apply regime-based adjustments to risk parameters.

        Adjusts risk_per_trade and max_positions based on market regime.
        These adjustments are temporary and can be reset daily.

        Args:
            risk_multiplier: Multiplier for risk_per_trade (0.5 to 1.0)
            max_positions: Maximum concurrent positions for this regime

        Example:
            >>> # In bear market
            >>> risk_mgr.apply_regime_adjustments(0.5, 4)
            >>> # Risk reduced to 50%, max positions reduced to 4
        """
        # Store original values if not already stored
        if not hasattr(self, '_original_risk_per_trade'):
            self._original_risk_per_trade = self.risk_per_trade
            self._original_max_positions = self.max_positions

        # Apply adjustments
        self.risk_per_trade = self._original_risk_per_trade * risk_multiplier
        self.max_positions = max_positions

        logger.info(
            f"Regime adjustments applied: risk={self.risk_per_trade*100:.2f}% "
            f"(multiplier: {risk_multiplier}x), max_positions={max_positions}"
        )

    def reset_regime_adjustments(self) -> None:
        """
        Reset risk parameters to original values.

        Removes regime-based adjustments.

        Example:
            >>> risk_mgr.reset_regime_adjustments()
        """
        if hasattr(self, '_original_risk_per_trade'):
            self.risk_per_trade = self._original_risk_per_trade
            self.max_positions = self._original_max_positions
            logger.info("Regime adjustments reset to original values")
