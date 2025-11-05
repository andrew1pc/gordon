"""Unit tests for regime adjustments (Iteration 6)."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    """Create RiskManager with $100k account."""
    return RiskManager(
        account_size=100000,
        risk_per_trade=0.01,
        max_positions=8
    )


class TestApplyRegimeAdjustments:
    """Test applying regime adjustments to risk parameters."""

    def test_apply_bull_regime_adjustments(self, risk_manager):
        """Test bull regime adjustments (1.0x risk, 8 positions)."""
        # Original values
        original_risk = risk_manager.risk_per_trade
        original_max_pos = risk_manager.max_positions

        # Apply bull adjustments
        risk_manager.apply_regime_adjustments(
            risk_multiplier=1.0,
            max_positions=8
        )

        # Should maintain original values
        assert risk_manager.risk_per_trade == original_risk  # 1.0x
        assert risk_manager.max_positions == 8

    def test_apply_sideways_regime_adjustments(self, risk_manager):
        """Test sideways regime adjustments (0.75x risk, 6 positions)."""
        original_risk = risk_manager.risk_per_trade

        # Apply sideways adjustments
        risk_manager.apply_regime_adjustments(
            risk_multiplier=0.75,
            max_positions=6
        )

        # Should reduce risk to 75%
        assert abs(risk_manager.risk_per_trade - (original_risk * 0.75)) < 0.0001
        assert risk_manager.max_positions == 6

    def test_apply_bear_regime_adjustments(self, risk_manager):
        """Test bear regime adjustments (0.5x risk, 4 positions)."""
        original_risk = risk_manager.risk_per_trade

        # Apply bear adjustments
        risk_manager.apply_regime_adjustments(
            risk_multiplier=0.5,
            max_positions=4
        )

        # Should reduce risk to 50%
        assert abs(risk_manager.risk_per_trade - (original_risk * 0.5)) < 0.0001
        assert risk_manager.max_positions == 4

    def test_original_values_preserved(self, risk_manager):
        """Test that original values are preserved after applying adjustments."""
        original_risk = risk_manager.risk_per_trade
        original_max_pos = risk_manager.max_positions

        # Apply adjustment
        risk_manager.apply_regime_adjustments(0.5, 4)

        # Original values should be stored
        assert risk_manager._original_risk_per_trade == original_risk
        assert risk_manager._original_max_positions == original_max_pos

    def test_multiple_adjustments_use_original_baseline(self, risk_manager):
        """Test that multiple adjustments always use original baseline."""
        original_risk = risk_manager.risk_per_trade

        # First adjustment: 0.75x
        risk_manager.apply_regime_adjustments(0.75, 6)
        first_adjusted = risk_manager.risk_per_trade

        # Second adjustment: 0.5x (should be 0.5 of original, not 0.5 of first)
        risk_manager.apply_regime_adjustments(0.5, 4)
        second_adjusted = risk_manager.risk_per_trade

        # Verify it's 0.5 of original, not 0.5 of 0.75
        assert abs(second_adjusted - (original_risk * 0.5)) < 0.0001
        assert second_adjusted != first_adjusted * 0.5


class TestResetRegimeAdjustments:
    """Test resetting regime adjustments."""

    def test_reset_restores_original_values(self, risk_manager):
        """Test that reset restores original risk parameters."""
        original_risk = risk_manager.risk_per_trade
        original_max_pos = risk_manager.max_positions

        # Apply adjustment
        risk_manager.apply_regime_adjustments(0.5, 4)

        # Verify adjusted
        assert risk_manager.risk_per_trade != original_risk

        # Reset
        risk_manager.reset_regime_adjustments()

        # Should restore original
        assert risk_manager.risk_per_trade == original_risk
        assert risk_manager.max_positions == original_max_pos

    def test_reset_without_adjustments(self, risk_manager):
        """Test that reset works even if no adjustments were applied."""
        # This should not crash
        risk_manager.reset_regime_adjustments()

        # Values should remain unchanged
        assert risk_manager.risk_per_trade == 0.01
        assert risk_manager.max_positions == 8


class TestAdjustedPositionSizing:
    """Test that position sizing uses adjusted risk parameters."""

    def test_position_size_with_bear_adjustments(self, risk_manager):
        """Test that position sizing uses reduced risk in bear market."""
        # Use prices that won't hit 5% max constraint
        # Entry $50, stop $40 = $10 risk per share
        # $100k * 1% = $1000 / $10 = 100 shares = $5000 (at max)
        size_base, value_base = risk_manager.calculate_position_size(
            entry_price=50,
            stop_loss=40,
            signal_strength=100
        )

        # Apply bear adjustments (0.5x risk)
        risk_manager.apply_regime_adjustments(0.5, 4)

        # Calculate adjusted position size
        size_adjusted, value_adjusted = risk_manager.calculate_position_size(
            entry_price=50,
            stop_loss=40,
            signal_strength=100
        )

        # Should be 50% of base size
        assert abs(size_adjusted - (size_base * 0.5)) < 0.1

    def test_position_size_with_sideways_adjustments(self, risk_manager):
        """Test that position sizing uses reduced risk in sideways market."""
        # Use prices that won't hit max constraint
        size_base, _ = risk_manager.calculate_position_size(
            entry_price=50,
            stop_loss=40,
            signal_strength=100
        )

        # Apply sideways adjustments (0.75x risk)
        risk_manager.apply_regime_adjustments(0.75, 6)

        # Calculate adjusted position size
        size_adjusted, _ = risk_manager.calculate_position_size(
            entry_price=50,
            stop_loss=40,
            signal_strength=100
        )

        # Should be 75% of base size
        assert abs(size_adjusted - (size_base * 0.75)) < 0.1

    def test_max_positions_enforced_with_adjustments(self, risk_manager):
        """Test that reduced max_positions is enforced."""
        # Apply bear adjustments (max 4 positions)
        risk_manager.apply_regime_adjustments(0.5, 4)

        # Try to add positions
        for i in range(4):
            can_add, reason = risk_manager.can_add_position(
                f'STOCK{i}', 1000, 'Technology'
            )
            if can_add:
                risk_manager.add_position(
                    f'STOCK{i}', 'stock', 'Technology',
                    100, 10, 90
                )

        # Should have 4 positions
        assert len(risk_manager.positions) == 4

        # 5th position should be rejected
        can_add, reason = risk_manager.can_add_position(
            'STOCK5', 1000, 'Technology'
        )
        assert can_add is False
        assert 'Max positions' in reason


class TestRegimeAdjustmentScenarios:
    """Test realistic regime adjustment scenarios."""

    def test_bull_to_bear_transition(self, risk_manager):
        """Test transitioning from bull to bear regime."""
        original_risk = risk_manager.risk_per_trade

        # Start in bull market (full risk)
        risk_manager.apply_regime_adjustments(1.0, 8)
        assert risk_manager.risk_per_trade == original_risk

        # Transition to bear market (50% risk)
        risk_manager.apply_regime_adjustments(0.5, 4)
        assert abs(risk_manager.risk_per_trade - (original_risk * 0.5)) < 0.0001
        assert risk_manager.max_positions == 4

    def test_daily_regime_update_cycle(self, risk_manager):
        """Test daily cycle of applying regime adjustments."""
        original_risk = risk_manager.risk_per_trade

        # Day 1: Bull market
        risk_manager.apply_regime_adjustments(1.0, 8)

        # Day 2: Sideways
        risk_manager.apply_regime_adjustments(0.75, 6)
        assert abs(risk_manager.risk_per_trade - (original_risk * 0.75)) < 0.0001

        # Day 3: Bear
        risk_manager.apply_regime_adjustments(0.5, 4)
        assert abs(risk_manager.risk_per_trade - (original_risk * 0.5)) < 0.0001

        # Day 4: Back to bull
        risk_manager.apply_regime_adjustments(1.0, 8)
        assert abs(risk_manager.risk_per_trade - original_risk) < 0.0001


class TestRegimeAdjustmentsWithSignalStrength:
    """Test that regime adjustments work with signal strength scaling."""

    def test_combined_scaling(self, risk_manager):
        """Test that regime and signal strength scaling combine correctly."""
        # Base position size with full risk (use prices that avoid max constraint)
        size_base, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=100
        )

        # Apply bear regime (0.5x) with weak signal (0.6x)
        risk_manager.apply_regime_adjustments(0.5, 4)
        size_combined, _ = risk_manager.calculate_position_size(
            50, 40, 'stock', signal_strength=75  # 60% size
        )

        # Should be 0.5 (regime) * 0.6 (signal) = 0.3 of base
        expected_size = size_base * 0.5 * 0.6
        assert abs(size_combined - expected_size) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
