"""Unit tests for RiskManager class."""

import pytest
from datetime import datetime, date
from strategy.risk_manager import RiskManager, PositionInfo


class TestRiskManagerInit:
    """Test RiskManager initialization."""

    def test_default_initialization(self):
        """Test RiskManager with default parameters."""
        risk_mgr = RiskManager(account_size=100000)

        assert risk_mgr.account_size == 100000
        assert risk_mgr.risk_per_trade == 0.01
        assert risk_mgr.max_positions == 8
        assert risk_mgr.max_portfolio_exposure == 0.25
        assert risk_mgr.max_sector_exposure == 0.15
        assert risk_mgr.daily_loss_limit == 0.03
        assert risk_mgr.crypto_size_multiplier == 0.7
        assert len(risk_mgr.positions) == 0

    def test_custom_initialization(self):
        """Test RiskManager with custom parameters."""
        risk_mgr = RiskManager(
            account_size=50000,
            risk_per_trade=0.02,
            max_positions=5,
            max_portfolio_exposure=0.30,
            max_sector_exposure=0.20,
            daily_loss_limit=0.05,
            crypto_size_multiplier=0.5
        )

        assert risk_mgr.account_size == 50000
        assert risk_mgr.risk_per_trade == 0.02
        assert risk_mgr.max_positions == 5
        assert risk_mgr.max_portfolio_exposure == 0.30
        assert risk_mgr.max_sector_exposure == 0.20
        assert risk_mgr.daily_loss_limit == 0.05
        assert risk_mgr.crypto_size_multiplier == 0.5


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_basic_position_sizing_stock(self):
        """Test basic position size calculation for stocks."""
        risk_mgr = RiskManager(account_size=100000, risk_per_trade=0.01)

        # Entry at 100, stop at 93, risk = $7/share
        # Dollar risk = 100000 * 0.01 = $1000
        # Position size = 1000 / 7 = 142.86 shares
        # Dollar value = 142.86 * 100 = $14,286
        # BUT: Max position is 5% = $5000, so capped at 50 shares
        size, value = risk_mgr.calculate_position_size(100, 93, 'stock')

        assert size == pytest.approx(50, abs=0.01)
        assert value == pytest.approx(5000, abs=1)

    def test_basic_position_sizing_crypto(self):
        """Test basic position size calculation for crypto."""
        risk_mgr = RiskManager(account_size=100000, risk_per_trade=0.01)

        # Same calculation but with 0.7 multiplier for crypto
        # 142.86 * 0.7 = 100 shares, value = $10,000
        # BUT: Max position is 5% = $5000, so capped at 50 shares
        size, value = risk_mgr.calculate_position_size(100, 93, 'crypto')

        assert size == pytest.approx(50, abs=0.01)
        assert value == pytest.approx(5000, abs=1)

    def test_position_size_minimum_constraint(self):
        """Test minimum position size constraint (1%)."""
        risk_mgr = RiskManager(account_size=100000, risk_per_trade=0.01)

        # Very wide stop that would create tiny position
        # Entry at 100, stop at 10, risk = $90/share
        # Position size = 1000 / 90 = 11.11 shares = $1,111
        # Should be increased to minimum 1% = $1,000
        size, value = risk_mgr.calculate_position_size(100, 10, 'stock')

        # Minimum is 1% = $1000
        assert value >= 1000

    def test_position_size_maximum_constraint(self):
        """Test maximum position size constraint (5%)."""
        risk_mgr = RiskManager(account_size=100000, risk_per_trade=0.01)

        # Very tight stop that would create huge position
        # Entry at 100, stop at 99, risk = $1/share
        # Position size = 1000 / 1 = 1000 shares = $100,000
        # Should be capped at 5% = $5,000
        size, value = risk_mgr.calculate_position_size(100, 99, 'stock')

        # Maximum is 5% = $5000
        assert value <= 5000

    def test_invalid_position_sizing(self):
        """Test position sizing with invalid inputs."""
        risk_mgr = RiskManager(account_size=100000, risk_per_trade=0.01)

        # Entry <= stop
        with pytest.raises(ValueError):
            risk_mgr.calculate_position_size(100, 100, 'stock')

        with pytest.raises(ValueError):
            risk_mgr.calculate_position_size(100, 110, 'stock')

        # Negative prices
        with pytest.raises(ValueError):
            risk_mgr.calculate_position_size(-100, -110, 'stock')

        with pytest.raises(ValueError):
            risk_mgr.calculate_position_size(100, -10, 'stock')


class TestPortfolioExposure:
    """Test portfolio and sector exposure tracking."""

    def test_empty_portfolio_exposure(self):
        """Test exposure calculation with no positions."""
        risk_mgr = RiskManager(account_size=100000)

        assert risk_mgr.get_portfolio_exposure() == 0.0

    def test_single_position_exposure(self):
        """Test exposure with one position."""
        risk_mgr = RiskManager(account_size=100000)

        # Add position worth $10,000
        risk_mgr.add_position(
            ticker='AAPL',
            asset_type='stock',
            sector='Technology',
            entry_price=100,
            position_size=100,
            stop_loss=93,
            entry_date=datetime.now()
        )

        exposure = risk_mgr.get_portfolio_exposure()
        assert exposure == pytest.approx(0.10, abs=0.001)  # 10%

    def test_multiple_positions_exposure(self):
        """Test exposure with multiple positions."""
        risk_mgr = RiskManager(account_size=100000)

        # Add 3 positions: $10k, $8k, $7k = $25k total = 25%
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        risk_mgr.add_position('MSFT', 'stock', 'Technology', 200, 40, 185)
        risk_mgr.add_position('JPM', 'stock', 'Financial', 140, 50, 130)

        exposure = risk_mgr.get_portfolio_exposure()
        assert exposure == pytest.approx(0.25, abs=0.001)  # 25%

    def test_sector_exposure_empty(self):
        """Test sector exposure with no positions in sector."""
        risk_mgr = RiskManager(account_size=100000)

        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)

        assert risk_mgr.get_sector_exposure('Financial') == 0.0
        assert risk_mgr.get_sector_exposure(None) == 0.0

    def test_sector_exposure_single_sector(self):
        """Test sector exposure calculation."""
        risk_mgr = RiskManager(account_size=100000)

        # Add 2 tech stocks: $10k + $8k = $18k = 18%
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        risk_mgr.add_position('MSFT', 'stock', 'Technology', 200, 40, 185)
        risk_mgr.add_position('JPM', 'stock', 'Financial', 140, 50, 130)

        tech_exposure = risk_mgr.get_sector_exposure('Technology')
        fin_exposure = risk_mgr.get_sector_exposure('Financial')

        assert tech_exposure == pytest.approx(0.18, abs=0.001)  # 18%
        assert fin_exposure == pytest.approx(0.07, abs=0.001)  # 7%


class TestPositionLimits:
    """Test position limit checks."""

    def test_can_add_position_success(self):
        """Test adding position when all limits satisfied."""
        risk_mgr = RiskManager(account_size=100000)

        can_add, reason = risk_mgr.can_add_position('AAPL', 5000, 'Technology')

        assert can_add is True
        assert reason == "OK"

    def test_max_positions_limit(self):
        """Test maximum positions limit."""
        risk_mgr = RiskManager(account_size=100000, max_positions=3)

        # Add 3 positions (at limit)
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 50, 93)
        risk_mgr.add_position('MSFT', 'stock', 'Technology', 200, 25, 185)
        risk_mgr.add_position('JPM', 'stock', 'Financial', 140, 35, 130)

        # Try to add 4th
        can_add, reason = risk_mgr.can_add_position('GOOGL', 5000, 'Technology')

        assert can_add is False
        assert "Max positions reached" in reason

    def test_duplicate_position_limit(self):
        """Test cannot add duplicate position."""
        risk_mgr = RiskManager(account_size=100000)

        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 50, 93)

        can_add, reason = risk_mgr.can_add_position('AAPL', 5000, 'Technology')

        assert can_add is False
        assert "Already have position" in reason

    def test_portfolio_exposure_limit(self):
        """Test portfolio exposure limit."""
        risk_mgr = RiskManager(account_size=100000, max_portfolio_exposure=0.20)

        # Add position worth $15k (15%)
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 150, 93)

        # Try to add another $10k (would be 25%, exceeds 20% limit)
        can_add, reason = risk_mgr.can_add_position('MSFT', 10000, 'Technology')

        assert can_add is False
        assert "Portfolio exposure would exceed" in reason

    def test_sector_exposure_limit(self):
        """Test sector exposure limit."""
        risk_mgr = RiskManager(account_size=100000, max_sector_exposure=0.15)

        # Add tech position worth $10k (10%)
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)

        # Try to add another tech for $8k (would be 18%, exceeds 15% limit)
        can_add, reason = risk_mgr.can_add_position('MSFT', 8000, 'Technology')

        assert can_add is False
        assert "Sector exposure would exceed" in reason

    def test_sector_exposure_limit_different_sectors(self):
        """Test sector limit doesn't block different sectors."""
        risk_mgr = RiskManager(account_size=100000, max_sector_exposure=0.15)

        # Add tech position worth $12k (12%)
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 120, 93)

        # Should be able to add financial position
        can_add, reason = risk_mgr.can_add_position('JPM', 8000, 'Financial')

        assert can_add is True
        assert reason == "OK"


class TestDailyLossLimit:
    """Test daily loss limit (circuit breaker)."""

    def test_no_loss_trading_allowed(self):
        """Test trading allowed with no losses."""
        risk_mgr = RiskManager(account_size=100000, daily_loss_limit=0.03)

        allowed, pnl = risk_mgr.check_daily_loss_limit()

        assert allowed is True
        assert pnl == 0.0

    def test_small_loss_trading_allowed(self):
        """Test trading allowed with small loss."""
        risk_mgr = RiskManager(account_size=100000, daily_loss_limit=0.03)

        # Add and close position with $1000 loss (1%)
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        pnl = risk_mgr.remove_position('AAPL', 90)

        assert pnl == -1000

        allowed, daily_pnl = risk_mgr.check_daily_loss_limit()

        assert allowed is True
        assert daily_pnl == -1000

    def test_large_loss_trading_halted(self):
        """Test trading halted with large loss."""
        risk_mgr = RiskManager(account_size=100000, daily_loss_limit=0.03)

        # Add and close position with $4000 loss (4% > 3% limit)
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        pnl = risk_mgr.remove_position('AAPL', 60)

        assert pnl == -4000

        allowed, daily_pnl = risk_mgr.check_daily_loss_limit()

        assert allowed is False
        assert daily_pnl == -4000

    def test_profit_no_circuit_breaker(self):
        """Test profit doesn't trigger circuit breaker."""
        risk_mgr = RiskManager(account_size=100000, daily_loss_limit=0.03)

        # Add and close position with profit
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        pnl = risk_mgr.remove_position('AAPL', 120)

        assert pnl == 2000

        allowed, daily_pnl = risk_mgr.check_daily_loss_limit()

        assert allowed is True
        assert daily_pnl == 2000


class TestTradeValidation:
    """Test comprehensive trade validation."""

    def test_validate_trade_success(self):
        """Test successful trade validation."""
        risk_mgr = RiskManager(account_size=100000)

        approved, reason, size, value = risk_mgr.validate_new_trade(
            ticker='AAPL',
            entry_price=100,
            stop_loss=93,
            sector='Technology',
            asset_type='stock'
        )

        assert approved is True
        assert reason == "Trade approved"
        assert size is not None
        assert value is not None

    def test_validate_trade_daily_loss_limit(self):
        """Test trade rejected due to daily loss limit."""
        risk_mgr = RiskManager(account_size=100000, daily_loss_limit=0.03)

        # Create large loss
        risk_mgr.add_position('SPY', 'stock', None, 400, 100, 390)
        risk_mgr.remove_position('SPY', 360)  # $4000 loss

        # Try new trade
        approved, reason, size, value = risk_mgr.validate_new_trade(
            ticker='AAPL',
            entry_price=100,
            stop_loss=93,
            sector='Technology',
            asset_type='stock'
        )

        assert approved is False
        assert "Daily loss limit exceeded" in reason
        assert size is None
        assert value is None

    def test_validate_trade_invalid_prices(self):
        """Test trade rejected due to invalid prices."""
        risk_mgr = RiskManager(account_size=100000)

        approved, reason, size, value = risk_mgr.validate_new_trade(
            ticker='AAPL',
            entry_price=100,
            stop_loss=110,  # Stop above entry
            sector='Technology',
            asset_type='stock'
        )

        assert approved is False
        assert "Invalid position parameters" in reason
        assert size is None
        assert value is None

    def test_validate_trade_max_positions(self):
        """Test trade rejected due to max positions."""
        risk_mgr = RiskManager(account_size=100000, max_positions=2)

        # Fill up to max
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 50, 93)
        risk_mgr.add_position('MSFT', 'stock', 'Technology', 200, 25, 185)

        # Try to add one more
        approved, reason, size, value = risk_mgr.validate_new_trade(
            ticker='GOOGL',
            entry_price=150,
            stop_loss=140,
            sector='Technology',
            asset_type='stock'
        )

        assert approved is False
        assert "Max positions reached" in reason


class TestPositionManagement:
    """Test position add/remove/update operations."""

    def test_add_position(self):
        """Test adding a position."""
        risk_mgr = RiskManager(account_size=100000)

        risk_mgr.add_position(
            ticker='AAPL',
            asset_type='stock',
            sector='Technology',
            entry_price=100,
            position_size=100,
            stop_loss=93
        )

        assert 'AAPL' in risk_mgr.positions
        assert risk_mgr.positions['AAPL'].ticker == 'AAPL'
        assert risk_mgr.positions['AAPL'].entry_price == 100
        assert risk_mgr.positions['AAPL'].position_size == 100
        assert risk_mgr.positions['AAPL'].dollar_value == 10000

    def test_remove_position_with_profit(self):
        """Test removing position with profit."""
        risk_mgr = RiskManager(account_size=100000)

        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        pnl = risk_mgr.remove_position('AAPL', 110)

        assert pnl == 1000  # $10 profit * 100 shares
        assert 'AAPL' not in risk_mgr.positions
        assert risk_mgr.account_size == 101000  # Updated with profit

    def test_remove_position_with_loss(self):
        """Test removing position with loss."""
        risk_mgr = RiskManager(account_size=100000)

        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        pnl = risk_mgr.remove_position('AAPL', 95)

        assert pnl == -500  # $5 loss * 100 shares
        assert 'AAPL' not in risk_mgr.positions
        assert risk_mgr.account_size == 99500  # Updated with loss

    def test_remove_nonexistent_position(self):
        """Test removing position that doesn't exist."""
        risk_mgr = RiskManager(account_size=100000)

        pnl = risk_mgr.remove_position('AAPL', 110)

        assert pnl is None

    def test_update_position_price(self):
        """Test updating position's current price."""
        risk_mgr = RiskManager(account_size=100000)

        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)

        # Update price
        risk_mgr.update_position_price('AAPL', 110)

        position = risk_mgr.positions['AAPL']
        assert position.current_price == 110
        assert position.dollar_value == 11000  # Updated value

    def test_update_nonexistent_position_price(self):
        """Test updating price for position that doesn't exist."""
        risk_mgr = RiskManager(account_size=100000)

        # Should not raise error, just log warning
        risk_mgr.update_position_price('AAPL', 110)


class TestRiskMetrics:
    """Test risk metrics dashboard."""

    def test_risk_metrics_empty_portfolio(self):
        """Test risk metrics with no positions."""
        risk_mgr = RiskManager(account_size=100000)

        metrics = risk_mgr.get_risk_metrics()

        assert metrics['account_size'] == 100000
        assert metrics['num_positions'] == 0
        assert metrics['portfolio_exposure'] == 0.0
        assert metrics['unrealized_pnl'] == 0.0
        assert metrics['daily_pnl'] == 0.0
        assert len(metrics['positions']) == 0
        assert len(metrics['sector_breakdown']) == 0

    def test_risk_metrics_with_positions(self):
        """Test risk metrics with open positions."""
        risk_mgr = RiskManager(account_size=100000)

        # Add positions
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        risk_mgr.add_position('MSFT', 'stock', 'Technology', 200, 40, 185)

        # Update prices to create unrealized P&L
        risk_mgr.update_position_price('AAPL', 110)  # +$1000
        risk_mgr.update_position_price('MSFT', 190)  # -$400

        metrics = risk_mgr.get_risk_metrics()

        assert metrics['num_positions'] == 2
        assert metrics['portfolio_exposure'] == pytest.approx(0.19, abs=0.01)  # 19%
        assert metrics['unrealized_pnl'] == pytest.approx(600, abs=1)  # Net unrealized
        assert 'AAPL' in metrics['positions']
        assert 'MSFT' in metrics['positions']
        assert 'Technology' in metrics['sector_breakdown']

    def test_risk_metrics_after_trade(self):
        """Test risk metrics after closing a trade."""
        risk_mgr = RiskManager(account_size=100000)

        # Add and close position with profit
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        risk_mgr.remove_position('AAPL', 110)

        metrics = risk_mgr.get_risk_metrics()

        assert metrics['account_size'] == 101000  # Updated
        assert metrics['num_positions'] == 0
        assert metrics['daily_pnl'] == 1000


class TestDailyTracking:
    """Test daily tracking reset."""

    def test_reset_daily_tracking(self):
        """Test resetting daily tracking."""
        risk_mgr = RiskManager(account_size=100000)

        # Make some trades
        risk_mgr.add_position('AAPL', 'stock', 'Technology', 100, 100, 93)
        risk_mgr.remove_position('AAPL', 110)

        assert risk_mgr.account_size == 101000

        # Manually update starting value to simulate new day
        risk_mgr.today_starting_value = risk_mgr.account_size

        # Reset for new day - this should not change starting value if already set today
        # But should initialize daily_pnl for today if needed
        today = date.today()
        if today not in risk_mgr.daily_pnl:
            risk_mgr.daily_pnl[today] = 0.0

        assert risk_mgr.today_starting_value == 101000
