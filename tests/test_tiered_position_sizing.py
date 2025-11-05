"""Unit tests for tiered position sizing (Iteration 2)."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    """Create RiskManager with $100k account."""
    return RiskManager(account_size=100000, risk_per_trade=0.01)


class TestScaledPositionSizeCalculation:
    """Test calculate_scaled_position_size() method."""

    def test_signal_strength_90_plus_returns_100_percent(self, risk_manager):
        """Test that signal strength 90-100 returns 1.0 (100% size)."""
        assert risk_manager.calculate_scaled_position_size(90) == 1.0
        assert risk_manager.calculate_scaled_position_size(95) == 1.0
        assert risk_manager.calculate_scaled_position_size(100) == 1.0

    def test_signal_strength_80_to_89_returns_80_percent(self, risk_manager):
        """Test that signal strength 80-89 returns 0.8 (80% size)."""
        assert risk_manager.calculate_scaled_position_size(80) == 0.8
        assert risk_manager.calculate_scaled_position_size(85) == 0.8
        assert risk_manager.calculate_scaled_position_size(89) == 0.8

    def test_signal_strength_70_to_79_returns_60_percent(self, risk_manager):
        """Test that signal strength 70-79 returns 0.6 (60% size)."""
        assert risk_manager.calculate_scaled_position_size(70) == 0.6
        assert risk_manager.calculate_scaled_position_size(75) == 0.6
        assert risk_manager.calculate_scaled_position_size(79) == 0.6

    def test_signal_strength_below_70_returns_60_percent(self, risk_manager):
        """Test that signal strength <70 returns 0.6 (minimum)."""
        assert risk_manager.calculate_scaled_position_size(69) == 0.6
        assert risk_manager.calculate_scaled_position_size(50) == 0.6
        assert risk_manager.calculate_scaled_position_size(0) == 0.6


class TestPositionSizeWithSignalStrength:
    """Test calculate_position_size() with signal_strength parameter."""

    def test_100_signal_strength_full_size(self, risk_manager):
        """Test that signal strength 100 gives full position size."""
        # Entry $50, stop $40, $10 risk per share
        # $100k * 1% = $1000 risk / $10 per share = 100 shares = $5000 (within 5% cap)
        size_100, value_100 = risk_manager.calculate_position_size(
            entry_price=50,
            stop_loss=40,
            asset_type='stock',
            signal_strength=100
        )

        expected_size = 1000 / 10  # 100 shares
        assert abs(size_100 - expected_size) < 0.1
        assert abs(value_100 - (size_100 * 50)) < 0.1

    def test_85_signal_strength_80_percent_size(self, risk_manager):
        """Test that signal strength 85 gives 80% position size."""
        # Same trade but with 80% scaling
        size_100, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=100
        )
        size_85, value_85 = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=85
        )

        # Should be 80% of full size
        assert abs(size_85 - (size_100 * 0.8)) < 0.1
        assert abs(value_85 - (size_85 * 50)) < 0.1

    def test_75_signal_strength_60_percent_size(self, risk_manager):
        """Test that signal strength 75 gives 60% position size."""
        size_100, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=100
        )
        size_75, value_75 = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=75
        )

        # Should be 60% of full size
        assert abs(size_75 - (size_100 * 0.6)) < 0.1
        assert abs(value_75 - (size_75 * 50)) < 0.1

    def test_backward_compatibility_default_100(self, risk_manager):
        """Test that default signal_strength=100 maintains backward compatibility."""
        # Old signature (no signal_strength)
        size_old, value_old = risk_manager.calculate_position_size(
            50, 40, 'stock'
        )

        # New signature with explicit 100
        size_new, value_new = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=100
        )

        # Should be identical
        assert abs(size_old - size_new) < 0.01
        assert abs(value_old - value_new) < 0.01

    def test_scaling_with_crypto_multiplier(self, risk_manager):
        """Test that signal strength scaling works with crypto multiplier."""
        # Crypto has 0.7x multiplier
        size_stock_100, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=100
        )
        size_crypto_100, _ = risk_manager.calculate_position_size(
            50, 40, 'crypto', signal_strength=100
        )

        # Crypto should be 70% of stock size
        assert abs(size_crypto_100 - (size_stock_100 * 0.7)) < 0.1

        # Now test with 80% signal strength scaling
        size_crypto_85, _ = risk_manager.calculate_position_size(
            50, 40, 'crypto', signal_strength=85
        )

        # Should be 70% * 80% = 56% of stock full size
        assert abs(size_crypto_85 - (size_stock_100 * 0.7 * 0.8)) < 0.1


class TestValidateNewTradeWithSignalStrength:
    """Test validate_new_trade() with signal_strength parameter."""

    def test_validate_with_90_signal_strength(self, risk_manager):
        """Test validation with strong signal (90+)."""
        approved, reason, size, value = risk_manager.validate_new_trade(
            ticker='AAPL',
            entry_price=50,
            stop_loss=40,
            sector='Technology',
            asset_type='stock',
            signal_strength=95
        )

        assert approved is True
        assert reason == "Trade approved"
        assert size is not None
        assert value is not None
        # Should be full size (1.0x)
        expected_size = (100000 * 0.01) / (50 - 40)  # 100 shares
        assert abs(size - expected_size) < 0.1

    def test_validate_with_85_signal_strength(self, risk_manager):
        """Test validation with good signal (80-89)."""
        approved, reason, size, value = risk_manager.validate_new_trade(
            ticker='MSFT',
            entry_price=50,
            stop_loss=40,
            sector='Technology',
            asset_type='stock',
            signal_strength=85
        )

        assert approved is True
        assert size is not None
        # Should be 80% of full size
        full_size = (100000 * 0.01) / (50 - 40)  # 100 shares
        expected_size = full_size * 0.8  # 80 shares
        assert abs(size - expected_size) < 0.1

    def test_validate_with_75_signal_strength(self, risk_manager):
        """Test validation with marginal signal (70-79)."""
        approved, reason, size, value = risk_manager.validate_new_trade(
            ticker='NVDA',
            entry_price=50,
            stop_loss=40,
            sector='Technology',
            asset_type='stock',
            signal_strength=75
        )

        assert approved is True
        assert size is not None
        # Should be 60% of full size
        full_size = (100000 * 0.01) / (50 - 40)  # 100 shares
        expected_size = full_size * 0.6  # 60 shares
        assert abs(size - expected_size) < 0.1

    def test_validate_backward_compatible(self, risk_manager):
        """Test that validate_new_trade works without signal_strength (default=100)."""
        # Old signature
        approved, reason, size_old, value_old = risk_manager.validate_new_trade(
            'TSLA', 50, 40, 'Automotive', 'stock'
        )

        assert approved is True
        assert size_old is not None

        # Should be same as explicitly passing 100
        # Use different ticker to avoid duplicate check
        approved2, reason2, size_new, value_new = risk_manager.validate_new_trade(
            'GM', 50, 40, 'Automotive', 'stock', signal_strength=100
        )

        # Sizes should be identical (full size)
        assert abs(size_old - size_new) < 0.1


class TestPositionSizeScalingWithConstraints:
    """Test that scaling works with min/max constraints."""

    def test_minimum_constraint_with_scaling(self):
        """Test that minimum position constraint applies after scaling."""
        # Small account to trigger minimum constraint
        risk_mgr = RiskManager(account_size=10000, risk_per_trade=0.01)

        # Wide stop that would normally give tiny position
        # $10k * 1% = $100 risk / $50 per share = 2 shares = $100 value
        # But minimum is 1% of account = $100, so should stay at 2 shares
        size, value = risk_mgr.calculate_position_size(
            entry_price=50,
            stop_loss=0.01,  # Very wide stop
            asset_type='stock',
            signal_strength=85  # 80% scaling
        )

        # Even with 80% scaling, should enforce minimum
        min_value = 10000 * 0.01  # $100
        assert value >= min_value

    def test_maximum_constraint_with_scaling(self):
        """Test that maximum position constraint applies after scaling."""
        # Create scenario that would exceed 5% max
        risk_mgr = RiskManager(account_size=100000, risk_per_trade=0.01)

        # Tight stop would normally give large position
        # $100k * 1% = $1000 risk / $1 per share = 1000 shares = $100k value
        # But max is 5% of account = $5000, so should cap at $5000
        size, value = risk_mgr.calculate_position_size(
            entry_price=100,
            stop_loss=99,  # Very tight stop
            asset_type='stock',
            signal_strength=100
        )

        # Should enforce maximum
        max_value = 100000 * 0.05  # $5000
        assert value <= max_value


class TestSignalStrengthIntegration:
    """Integration tests for signal strength scaling."""

    def test_three_tier_comparison(self, risk_manager):
        """Test that all three tiers produce different sizes."""
        entry, stop = 50, 40

        size_90, _ = risk_manager.calculate_position_size(entry, stop, 'stock', 90)
        size_85, _ = risk_manager.calculate_position_size(entry, stop, 'stock', 85)
        size_75, _ = risk_manager.calculate_position_size(entry, stop, 'stock', 75)

        # Verify ordering
        assert size_90 > size_85
        assert size_85 > size_75

        # Verify ratios
        assert abs((size_85 / size_90) - 0.8) < 0.01  # 80% of full
        assert abs((size_75 / size_90) - 0.6) < 0.01  # 60% of full

    def test_risk_amount_scales_correctly(self, risk_manager):
        """Test that dollar risk scales with signal strength."""
        entry, stop = 50, 40  # $10 risk per share

        # Full risk
        size_100, value_100 = risk_manager.calculate_position_size(
            entry, stop, 'stock', 100
        )
        risk_100 = size_100 * (entry - stop)

        # 80% risk
        size_85, value_85 = risk_manager.calculate_position_size(
            entry, stop, 'stock', 85
        )
        risk_85 = size_85 * (entry - stop)

        # 60% risk
        size_75, value_75 = risk_manager.calculate_position_size(
            entry, stop, 'stock', 75
        )
        risk_75 = size_75 * (entry - stop)

        # Verify risk amounts
        assert abs(risk_100 - 1000) < 1  # 1% of $100k
        assert abs(risk_85 - 800) < 1    # 0.8% of $100k
        assert abs(risk_75 - 600) < 1    # 0.6% of $100k


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
