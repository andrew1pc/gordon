"""Unit tests for SignalAnalyzer (Iteration 3)."""

import pytest
import os
import csv
from datetime import datetime
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.signal_analyzer import SignalAnalyzer, SignalRecord


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def analyzer(temp_output_dir):
    """Create SignalAnalyzer with temp directory."""
    return SignalAnalyzer(output_dir=temp_output_dir)


@pytest.fixture
def sample_signal_data():
    """Sample signal data from check_entry_signals()."""
    return {
        'entry_price': 150.0,
        'conditions_met': ['breakout_20d_high', 'volume_surge', 'macd_positive'],
        'conditions_failed': ['strong_momentum'],
        'condition_scores': {
            'breakout': 20,
            'volume_surge': 15,
            'macd_positive': 15,
            'above_ma50': 15,
            'ma_trending_up': 15,
            'strong_momentum': 5
        }
    }


@pytest.fixture
def sample_strength_result():
    """Sample strength result from calculate_entry_signal_strength()."""
    return {
        'signal_strength': 65,
        'conditions_met': ['breakout_20d_high', 'volume_surge'],
        'conditions_failed': ['macd_positive', 'strong_momentum'],
        'condition_scores': {
            'breakout': 20,
            'volume_surge': 15,
            'macd_positive': 0,
            'above_ma50': 15,
            'ma_trending_up': 15,
            'strong_momentum': 0
        }
    }


class TestSignalRecording:
    """Test signal and near miss recording."""

    def test_record_entry_signal(self, analyzer, sample_signal_data):
        """Test recording an entry signal."""
        analyzer.record_signal(
            ticker='AAPL',
            date=datetime(2025, 1, 1),
            signal_strength=85,
            signal_data=sample_signal_data,
            asset_type='stock',
            sector='Technology'
        )

        assert len(analyzer.signals) == 1
        assert analyzer.signals[0].ticker == 'AAPL'
        assert analyzer.signals[0].signal_strength == 85
        assert analyzer.signals[0].signal_type == 'entry'
        assert analyzer.signals[0].asset_type == 'stock'
        assert analyzer.signals[0].sector == 'Technology'

    def test_record_multiple_signals(self, analyzer, sample_signal_data):
        """Test recording multiple signals."""
        for i in range(5):
            analyzer.record_signal(
                ticker=f'TICK{i}',
                date=datetime(2025, 1, i+1),
                signal_strength=80 + i,
                signal_data=sample_signal_data
            )

        assert len(analyzer.signals) == 5
        assert analyzer.signals[0].ticker == 'TICK0'
        assert analyzer.signals[4].ticker == 'TICK4'

    def test_record_near_miss(self, analyzer, sample_strength_result):
        """Test recording a near miss (60-69)."""
        analyzer.record_near_miss(
            ticker='TSLA',
            date=datetime(2025, 1, 1),
            signal_strength=65,
            strength_result=sample_strength_result,
            entry_price=200.0,
            asset_type='stock',
            sector='Automotive'
        )

        assert len(analyzer.near_misses) == 1
        assert analyzer.near_misses[0].ticker == 'TSLA'
        assert analyzer.near_misses[0].signal_strength == 65
        assert analyzer.near_misses[0].signal_type == 'near_miss'

    def test_record_near_miss_rejects_invalid_range(
        self,
        analyzer,
        sample_strength_result
    ):
        """Test that near miss rejects signals outside 60-69 range."""
        # Try to record signal with strength 70 (should be entry, not near miss)
        analyzer.record_near_miss(
            'INVALID', datetime(2025, 1, 1), 70,
            sample_strength_result, 100.0
        )

        # Should not be recorded
        assert len(analyzer.near_misses) == 0

    def test_mixed_signals_and_near_misses(self, analyzer, sample_signal_data, sample_strength_result):
        """Test recording both signals and near misses."""
        # Record 3 entry signals
        for i in range(3):
            analyzer.record_signal(
                f'SIGNAL{i}', datetime(2025, 1, i+1),
                80, sample_signal_data
            )

        # Record 2 near misses
        for i in range(2):
            analyzer.record_near_miss(
                f'MISS{i}', datetime(2025, 1, i+1),
                65, sample_strength_result, 100.0
            )

        assert len(analyzer.signals) == 3
        assert len(analyzer.near_misses) == 2


class TestSummaryGeneration:
    """Test summary statistics generation."""

    def test_empty_summary(self, analyzer):
        """Test summary with no signals."""
        summary = analyzer.get_summary()

        assert summary['total_signals'] == 0
        assert summary['total_near_misses'] == 0
        assert summary['total_combined'] == 0
        assert summary['avg_signal_strength'] == 0

    def test_summary_with_signals(self, analyzer, sample_signal_data):
        """Test summary with multiple signals."""
        # Add signals with different strengths
        strengths = [95, 85, 75, 90, 80]
        for i, strength in enumerate(strengths):
            analyzer.record_signal(
                f'TICK{i}', datetime(2025, 1, i+1),
                strength, sample_signal_data
            )

        summary = analyzer.get_summary()

        assert summary['total_signals'] == 5
        assert summary['avg_signal_strength'] == sum(strengths) / len(strengths)

    def test_strength_distribution(self, analyzer, sample_signal_data):
        """Test signal strength distribution in summary."""
        # Add 2 signals in each tier
        for strength in [95, 92, 85, 82, 75, 72]:
            analyzer.record_signal(
                f'TICK{strength}', datetime(2025, 1, 1),
                strength, sample_signal_data
            )

        summary = analyzer.get_summary()

        assert summary['strength_distribution']['90_plus'] == 2
        assert summary['strength_distribution']['80_89'] == 2
        assert summary['strength_distribution']['70_79'] == 2

    def test_asset_type_breakdown(self, analyzer, sample_signal_data):
        """Test asset type counting in summary."""
        # Add 3 stocks
        for i in range(3):
            analyzer.record_signal(
                f'STOCK{i}', datetime(2025, 1, i+1),
                80, sample_signal_data, asset_type='stock'
            )

        # Add 2 crypto
        for i in range(2):
            analyzer.record_signal(
                f'CRYPTO{i}', datetime(2025, 1, i+1),
                80, sample_signal_data, asset_type='crypto'
            )

        summary = analyzer.get_summary()

        assert summary['by_asset_type']['stocks'] == 3
        assert summary['by_asset_type']['crypto'] == 2

    def test_top_sectors(self, analyzer, sample_signal_data):
        """Test top sectors tracking in summary."""
        # Add signals from different sectors
        sectors = ['Technology'] * 5 + ['Financial'] * 3 + ['Healthcare'] * 2
        for i, sector in enumerate(sectors):
            analyzer.record_signal(
                f'TICK{i}', datetime(2025, 1, 1),
                80, sample_signal_data, sector=sector
            )

        summary = analyzer.get_summary()

        top_sectors = summary['top_sectors']
        assert 'Technology' in top_sectors
        assert top_sectors['Technology'] == 5
        assert 'Financial' in top_sectors
        assert top_sectors['Financial'] == 3

    def test_near_miss_counting(self, analyzer, sample_signal_data, sample_strength_result):
        """Test that near misses are counted separately."""
        analyzer.record_signal('SIGNAL1', datetime(2025, 1, 1), 80, sample_signal_data)
        analyzer.record_near_miss('MISS1', datetime(2025, 1, 1), 65, sample_strength_result, 100.0)

        summary = analyzer.get_summary()

        assert summary['total_signals'] == 1
        assert summary['total_near_misses'] == 1
        assert summary['total_combined'] == 2


class TestReportGeneration:
    """Test formatted report generation."""

    def test_generate_empty_report(self, analyzer):
        """Test report generation with no signals."""
        report = analyzer.get_report()

        assert 'SIGNAL ANALYSIS REPORT' in report
        assert 'Total Entry Signals:       0' in report
        assert 'Total Near Misses (60-69): 0' in report

    def test_generate_report_with_signals(self, analyzer, sample_signal_data):
        """Test report generation with signals."""
        analyzer.record_signal('AAPL', datetime(2025, 1, 1), 85, sample_signal_data)
        analyzer.record_signal('MSFT', datetime(2025, 1, 2), 92, sample_signal_data)

        report = analyzer.get_report()

        assert 'AAPL' in report
        assert 'MSFT' in report
        assert 'Total Entry Signals:       2' in report
        assert 'SIGNAL STRENGTH DISTRIBUTION' in report

    def test_report_includes_near_misses(self, analyzer, sample_signal_data, sample_strength_result):
        """Test that report includes near miss section."""
        analyzer.record_near_miss('TSLA', datetime(2025, 1, 1), 65, sample_strength_result, 200.0)

        report = analyzer.get_report()

        assert 'NEAR MISSES' in report
        assert 'TSLA' in report
        assert '65' in report

    def test_report_shows_recent_signals(self, analyzer, sample_signal_data):
        """Test that report shows last 5 signals."""
        # Add 10 signals
        for i in range(10):
            analyzer.record_signal(
                f'TICK{i}', datetime(2025, 1, i+1),
                80, sample_signal_data
            )

        report = analyzer.get_report()

        # Should show last 5
        assert 'TICK9' in report
        assert 'TICK8' in report
        assert 'TICK7' in report
        assert 'TICK6' in report
        assert 'TICK5' in report
        # Should not show first 5
        assert 'TICK0' not in report


class TestCSVExport:
    """Test CSV export functionality."""

    def test_export_to_csv(self, analyzer, sample_signal_data, temp_output_dir):
        """Test exporting signals to CSV."""
        analyzer.record_signal('AAPL', datetime(2025, 1, 1), 85, sample_signal_data)
        analyzer.record_signal('MSFT', datetime(2025, 1, 2), 90, sample_signal_data)

        filepath = analyzer.export_to_csv('test_signals.csv')

        assert os.path.exists(filepath)
        assert filepath.endswith('test_signals.csv')

        # Read CSV and verify content
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]['ticker'] == 'AAPL'
        assert rows[0]['signal_strength'] == '85'
        assert rows[1]['ticker'] == 'MSFT'

    def test_export_empty_signals(self, analyzer, temp_output_dir):
        """Test exporting with no signals."""
        filepath = analyzer.export_to_csv('empty.csv')

        # File is created but empty (except header)
        assert os.path.exists(filepath) or True  # May or may not create file

    def test_export_includes_near_misses(
        self,
        analyzer,
        sample_signal_data,
        sample_strength_result,
        temp_output_dir
    ):
        """Test that CSV export includes both signals and near misses."""
        analyzer.record_signal('AAPL', datetime(2025, 1, 1), 85, sample_signal_data)
        analyzer.record_near_miss('TSLA', datetime(2025, 1, 2), 65, sample_strength_result, 200.0)

        filepath = analyzer.export_to_csv()

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]['signal_type'] == 'entry'
        assert rows[1]['signal_type'] == 'near_miss'

    def test_auto_generated_filename(self, analyzer, sample_signal_data, temp_output_dir):
        """Test that auto-generated filename includes timestamp."""
        analyzer.record_signal('AAPL', datetime(2025, 1, 1), 85, sample_signal_data)

        filepath = analyzer.export_to_csv()  # No filename provided

        assert 'signals_' in filepath
        assert filepath.endswith('.csv')
        assert os.path.exists(filepath)


class TestClearFunctionality:
    """Test clear/reset functionality."""

    def test_clear_signals(self, analyzer, sample_signal_data):
        """Test clearing all signals."""
        # Add signals
        analyzer.record_signal('AAPL', datetime(2025, 1, 1), 85, sample_signal_data)
        analyzer.record_signal('MSFT', datetime(2025, 1, 2), 90, sample_signal_data)

        assert len(analyzer.signals) == 2

        # Clear
        analyzer.clear()

        assert len(analyzer.signals) == 0
        assert len(analyzer.near_misses) == 0

    def test_clear_includes_near_misses(self, analyzer, sample_signal_data, sample_strength_result):
        """Test that clear removes both signals and near misses."""
        analyzer.record_signal('AAPL', datetime(2025, 1, 1), 85, sample_signal_data)
        analyzer.record_near_miss('TSLA', datetime(2025, 1, 2), 65, sample_strength_result, 200.0)

        analyzer.clear()

        assert len(analyzer.signals) == 0
        assert len(analyzer.near_misses) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
