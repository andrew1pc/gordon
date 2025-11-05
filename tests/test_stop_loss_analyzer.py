"""Unit tests for stop loss analysis tool (Iteration 8)."""

import pytest
import json
import tempfile
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyze_stop_losses import StopLossAnalyzer


@pytest.fixture
def sample_trade_history(tmp_path):
    """Create sample trade history JSON file."""
    trades = {
        'closed_positions': [
            # Stock stop loss - no gap
            {
                'ticker': 'AAPL',
                'asset_type': 'stock',
                'exit_type': 'stop_loss',
                'exit_date': '2025-01-15',
                'stop_price': 100.0,
                'exit_price': 99.5,
                'slippage_pct': 0.005,  # 0.5%
                'gap_detected': False
            },
            # Stock stop loss - with gap
            {
                'ticker': 'TSLA',
                'asset_type': 'stock',
                'exit_type': 'stop_loss',
                'exit_date': '2025-01-20',
                'stop_price': 200.0,
                'exit_price': 198.0,
                'slippage_pct': 0.01,  # 1.0%
                'gap_detected': True
            },
            # Crypto stop loss - no gap
            {
                'ticker': 'btcusd',
                'asset_type': 'crypto',
                'exit_type': 'stop_loss',
                'exit_date': '2025-01-25',
                'stop_price': 45000.0,
                'exit_price': 44550.0,
                'slippage_pct': 0.01,  # 1.0%
                'gap_detected': False
            },
            # Crypto stop loss - with gap
            {
                'ticker': 'ethusd',
                'asset_type': 'crypto',
                'exit_type': 'stop_loss',
                'exit_date': '2025-02-01',
                'stop_price': 3000.0,
                'exit_price': 2955.0,
                'slippage_pct': 0.015,  # 1.5%
                'gap_detected': True
            },
            # Target exit (not a stop loss)
            {
                'ticker': 'MSFT',
                'asset_type': 'stock',
                'exit_type': 'target',
                'exit_date': '2025-02-05',
                'stop_price': 300.0,
                'exit_price': 350.0,
                'slippage_pct': 0.0
            }
        ]
    }

    # Write to temp file
    trade_file = tmp_path / "trade_history.json"
    with open(trade_file, 'w') as f:
        json.dump(trades, f)

    return str(trade_file)


@pytest.fixture
def empty_trade_history(tmp_path):
    """Create empty trade history file."""
    trades = {'closed_positions': []}
    trade_file = tmp_path / "empty_history.json"
    with open(trade_file, 'w') as f:
        json.dump(trades, f)
    return str(trade_file)


class TestStopLossAnalyzerInit:
    """Test analyzer initialization."""

    def test_init_with_valid_path(self, sample_trade_history):
        """Test initialization with valid path."""
        analyzer = StopLossAnalyzer(sample_trade_history)
        assert analyzer.trade_history_path == Path(sample_trade_history)
        assert analyzer.trades == []
        assert analyzer.stop_losses == []


class TestLoadTrades:
    """Test loading trade data."""

    def test_load_trades_from_file(self, sample_trade_history):
        """Test loading trades from JSON file."""
        analyzer = StopLossAnalyzer(sample_trade_history)
        analyzer.load_trades()

        assert len(analyzer.trades) == 5
        assert all(isinstance(t, dict) for t in analyzer.trades)

    def test_load_empty_history(self, empty_trade_history):
        """Test loading empty trade history."""
        analyzer = StopLossAnalyzer(empty_trade_history)
        analyzer.load_trades()

        assert len(analyzer.trades) == 0

    def test_load_nonexistent_file(self):
        """Test error when file doesn't exist."""
        analyzer = StopLossAnalyzer('nonexistent.json')

        with pytest.raises(FileNotFoundError):
            analyzer.load_trades()


class TestExtractStopLosses:
    """Test extracting stop loss exits."""

    def test_extract_stop_losses_only(self, sample_trade_history):
        """Test that only stop loss exits are extracted."""
        analyzer = StopLossAnalyzer(sample_trade_history)
        analyzer.load_trades()
        analyzer.extract_stop_losses()

        # Should have 4 stop losses (not the target exit)
        assert len(analyzer.stop_losses) == 4
        assert all(s['exit_type'] == 'stop_loss' for s in analyzer.stop_losses)

    def test_extract_with_no_stops(self, empty_trade_history):
        """Test extraction with no stop losses."""
        analyzer = StopLossAnalyzer(empty_trade_history)
        analyzer.load_trades()
        analyzer.extract_stop_losses()

        assert len(analyzer.stop_losses) == 0


class TestCalculateStatistics:
    """Test slippage statistics calculations."""

    def test_overall_statistics(self, sample_trade_history):
        """Test overall slippage statistics."""
        analyzer = StopLossAnalyzer(sample_trade_history)
        analyzer.load_trades()
        analyzer.extract_stop_losses()
        stats = analyzer.calculate_statistics()

        assert stats['total_stops'] == 4
        assert 'overall' in stats

        # Check average (0.5 + 1.0 + 1.0 + 1.5) / 4 = 1.0%
        assert abs(stats['overall']['avg_slippage'] - 1.0) < 0.01

        # Check max is 1.5%
        assert abs(stats['overall']['max_slippage'] - 1.5) < 0.01

        # Check min is 0.5%
        assert abs(stats['overall']['min_slippage'] - 0.5) < 0.01

    def test_by_asset_type(self, sample_trade_history):
        """Test breakdown by asset type."""
        analyzer = StopLossAnalyzer(sample_trade_history)
        analyzer.load_trades()
        analyzer.extract_stop_losses()
        stats = analyzer.calculate_statistics()

        assert 'by_asset_type' in stats

        # Should have 2 stocks
        assert stats['by_asset_type']['stock']['count'] == 2
        # Stock avg: (0.5 + 1.0) / 2 = 0.75%
        assert abs(stats['by_asset_type']['stock']['avg_slippage'] - 0.75) < 0.01

        # Should have 2 crypto
        assert stats['by_asset_type']['crypto']['count'] == 2
        # Crypto avg: (1.0 + 1.5) / 2 = 1.25%
        assert abs(stats['by_asset_type']['crypto']['avg_slippage'] - 1.25) < 0.01

    def test_gap_analysis(self, sample_trade_history):
        """Test gap vs non-gap analysis."""
        analyzer = StopLossAnalyzer(sample_trade_history)
        analyzer.load_trades()
        analyzer.extract_stop_losses()
        stats = analyzer.calculate_statistics()

        assert 'gap_analysis' in stats

        # Should have 2 gaps
        assert stats['gap_analysis']['gap_stops']['count'] == 2
        # Gap avg: (1.0 + 1.5) / 2 = 1.25%
        assert abs(stats['gap_analysis']['gap_stops']['avg_slippage'] - 1.25) < 0.01

        # Should have 2 non-gaps
        assert stats['gap_analysis']['non_gap_stops']['count'] == 2
        # Non-gap avg: (0.5 + 1.0) / 2 = 0.75%
        assert abs(stats['gap_analysis']['non_gap_stops']['avg_slippage'] - 0.75) < 0.01

    def test_worst_exits(self, sample_trade_history):
        """Test worst exits tracking."""
        analyzer = StopLossAnalyzer(sample_trade_history)
        analyzer.load_trades()
        analyzer.extract_stop_losses()
        stats = analyzer.calculate_statistics()

        assert 'worst_exits' in stats
        assert len(stats['worst_exits']) <= 5

        # Worst should be ethusd with 1.5% slippage
        worst = stats['worst_exits'][0]
        assert worst['ticker'] == 'ethusd'
        assert abs(worst['slippage_pct'] - 1.5) < 0.01
        assert worst['gap_detected'] is True

    def test_empty_history_statistics(self, empty_trade_history):
        """Test statistics with no stop losses."""
        analyzer = StopLossAnalyzer(empty_trade_history)
        analyzer.load_trades()
        analyzer.extract_stop_losses()
        stats = analyzer.calculate_statistics()

        assert stats['total_stops'] == 0
        assert 'message' in stats


class TestFormatReport:
    """Test report formatting."""

    def test_format_report_with_data(self, sample_trade_history):
        """Test report formatting with data."""
        analyzer = StopLossAnalyzer(sample_trade_history)
        analyzer.load_trades()
        analyzer.extract_stop_losses()
        stats = analyzer.calculate_statistics()
        report = analyzer.format_report(stats)

        # Check key sections are present
        assert "STOP LOSS SLIPPAGE ANALYSIS REPORT" in report
        assert "Total Stop Loss Exits: 4" in report
        assert "OVERALL SLIPPAGE:" in report
        assert "BY ASSET TYPE:" in report
        assert "GAP ANALYSIS:" in report
        assert "TOP 5 WORST EXITS" in report

        # Check specific values
        assert "Average:    1.00%" in report
        assert "STOCK:" in report
        assert "CRYPTO:" in report

    def test_format_report_empty(self, empty_trade_history):
        """Test report formatting with no data."""
        analyzer = StopLossAnalyzer(empty_trade_history)
        analyzer.load_trades()
        analyzer.extract_stop_losses()
        stats = analyzer.calculate_statistics()
        report = analyzer.format_report(stats)

        assert "No stop loss exits found" in report


class TestRunAnalysis:
    """Test full analysis workflow."""

    def test_run_full_analysis(self, sample_trade_history):
        """Test complete analysis workflow."""
        analyzer = StopLossAnalyzer(sample_trade_history)
        report = analyzer.run_analysis()

        # Should return formatted report
        assert isinstance(report, str)
        assert len(report) > 0
        assert "STOP LOSS SLIPPAGE ANALYSIS REPORT" in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
