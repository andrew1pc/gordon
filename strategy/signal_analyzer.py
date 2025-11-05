"""Signal analysis and reporting module."""

import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
import csv
import os


logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Record of a signal (entry or near miss)."""
    ticker: str
    date: datetime
    signal_strength: int
    signal_type: str  # 'entry', 'near_miss'
    entry_price: float
    conditions_met: List[str]
    conditions_failed: List[str]
    condition_scores: Dict[str, int]
    asset_type: str = 'stock'
    sector: Optional[str] = None


class SignalAnalyzer:
    """
    Analyzes and tracks signal generation statistics.

    This class records all signals (both entry signals and near misses)
    and provides reporting capabilities to understand signal frequency
    and quality.

    Features:
    - Track all entry signals with strength scores
    - Track near misses (60-69 point signals)
    - Generate summary statistics
    - Export detailed reports to CSV

    Example:
        >>> analyzer = SignalAnalyzer()
        >>> analyzer.record_signal('AAPL', datetime.now(), 85, {...})
        >>> analyzer.record_near_miss('MSFT', datetime.now(), 65, {...})
        >>> summary = analyzer.get_summary()
        >>> analyzer.export_to_csv('signals_report.csv')
    """

    def __init__(self, output_dir: str = 'output'):
        """
        Initialize the SignalAnalyzer.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.signals: List[SignalRecord] = []
        self.near_misses: List[SignalRecord] = []

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"SignalAnalyzer initialized (output_dir: {output_dir})")

    def record_signal(
        self,
        ticker: str,
        date: datetime,
        signal_strength: int,
        signal_data: Dict,
        asset_type: str = 'stock',
        sector: Optional[str] = None
    ) -> None:
        """
        Record an entry signal (strength >= 70).

        Args:
            ticker: Ticker symbol
            date: Signal date
            signal_strength: Signal strength (70-100)
            signal_data: Signal details from check_entry_signals()
            asset_type: 'stock' or 'crypto'
            sector: Sector classification

        Example:
            >>> signal = generator.check_entry_signals(df, idx, 'AAPL')
            >>> if signal:
            ...     analyzer.record_signal(
            ...         'AAPL', signal['date'], signal['signal_strength'],
            ...         signal, 'stock', 'Technology'
            ...     )
        """
        record = SignalRecord(
            ticker=ticker,
            date=date,
            signal_strength=signal_strength,
            signal_type='entry',
            entry_price=signal_data.get('entry_price', 0.0),
            conditions_met=signal_data.get('conditions_met', []),
            conditions_failed=signal_data.get('conditions_failed', []),
            condition_scores=signal_data.get('condition_scores', {}),
            asset_type=asset_type,
            sector=sector
        )

        self.signals.append(record)
        logger.debug(
            f"Recorded entry signal: {ticker} (strength: {signal_strength}/100)"
        )

    def record_near_miss(
        self,
        ticker: str,
        date: datetime,
        signal_strength: int,
        strength_result: Dict,
        entry_price: float,
        asset_type: str = 'stock',
        sector: Optional[str] = None
    ) -> None:
        """
        Record a near miss signal (strength 60-69).

        Near misses are interesting because they almost qualified
        and might be worth watching.

        Args:
            ticker: Ticker symbol
            date: Signal date
            signal_strength: Signal strength (60-69)
            strength_result: Result from calculate_entry_signal_strength()
            entry_price: Would-be entry price
            asset_type: 'stock' or 'crypto'
            sector: Sector classification

        Example:
            >>> strength_result = generator.calculate_entry_signal_strength(df, idx, 'TSLA')
            >>> if 60 <= strength_result['signal_strength'] < 70:
            ...     analyzer.record_near_miss(
            ...         'TSLA', df.index[idx], strength_result['signal_strength'],
            ...         strength_result, df['close'].iloc[idx]
            ...     )
        """
        if not (60 <= signal_strength < 70):
            logger.warning(
                f"Signal strength {signal_strength} outside near miss range (60-69)"
            )
            return

        record = SignalRecord(
            ticker=ticker,
            date=date,
            signal_strength=signal_strength,
            signal_type='near_miss',
            entry_price=entry_price,
            conditions_met=strength_result.get('conditions_met', []),
            conditions_failed=strength_result.get('conditions_failed', []),
            condition_scores=strength_result.get('condition_scores', {}),
            asset_type=asset_type,
            sector=sector
        )

        self.near_misses.append(record)
        logger.debug(
            f"Recorded near miss: {ticker} (strength: {signal_strength}/100)"
        )

    def get_summary(self) -> Dict:
        """
        Get summary statistics for all signals.

        Returns:
            Dictionary with summary statistics

        Example:
            >>> summary = analyzer.get_summary()
            >>> print(f"Total signals: {summary['total_signals']}")
            >>> print(f"Average strength: {summary['avg_signal_strength']:.1f}")
        """
        total_signals = len(self.signals)
        total_near_misses = len(self.near_misses)

        if total_signals == 0:
            avg_strength = 0
            strength_90_plus = 0
            strength_80_89 = 0
            strength_70_79 = 0
        else:
            strengths = [s.signal_strength for s in self.signals]
            avg_strength = sum(strengths) / len(strengths)
            strength_90_plus = sum(1 for s in strengths if s >= 90)
            strength_80_89 = sum(1 for s in strengths if 80 <= s < 90)
            strength_70_79 = sum(1 for s in strengths if 70 <= s < 80)

        # Count by asset type
        stocks = sum(1 for s in self.signals if s.asset_type == 'stock')
        crypto = sum(1 for s in self.signals if s.asset_type == 'crypto')

        # Count by sector (top 3)
        sector_counts = {}
        for s in self.signals:
            if s.sector:
                sector_counts[s.sector] = sector_counts.get(s.sector, 0) + 1

        top_sectors = sorted(
            sector_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        summary = {
            'total_signals': total_signals,
            'total_near_misses': total_near_misses,
            'total_combined': total_signals + total_near_misses,
            'avg_signal_strength': avg_strength,
            'strength_distribution': {
                '90_plus': strength_90_plus,
                '80_89': strength_80_89,
                '70_79': strength_70_79
            },
            'by_asset_type': {
                'stocks': stocks,
                'crypto': crypto
            },
            'top_sectors': dict(top_sectors)
        }

        return summary

    def get_report(self) -> str:
        """
        Generate a formatted text report.

        Returns:
            Multi-line string report

        Example:
            >>> report = analyzer.get_report()
            >>> print(report)
        """
        summary = self.get_summary()

        report_lines = [
            "=" * 80,
            "SIGNAL ANALYSIS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "SUMMARY",
            "-" * 80,
            f"Total Entry Signals:       {summary['total_signals']}",
            f"Total Near Misses (60-69): {summary['total_near_misses']}",
            f"Total Opportunities:       {summary['total_combined']}",
            ""
        ]

        if summary['total_signals'] > 0:
            report_lines.extend([
                "SIGNAL STRENGTH DISTRIBUTION",
                "-" * 80,
                f"Average Strength:          {summary['avg_signal_strength']:.1f}/100",
                f"  90-100 (Full size):      {summary['strength_distribution']['90_plus']} signals",
                f"  80-89  (80% size):       {summary['strength_distribution']['80_89']} signals",
                f"  70-79  (60% size):       {summary['strength_distribution']['70_79']} signals",
                ""
            ])

        report_lines.extend([
            "BY ASSET TYPE",
            "-" * 80,
            f"Stocks:                    {summary['by_asset_type']['stocks']}",
            f"Crypto:                    {summary['by_asset_type']['crypto']}",
            ""
        ])

        if summary['top_sectors']:
            report_lines.extend([
                "TOP SECTORS",
                "-" * 80
            ])
            for sector, count in summary['top_sectors'].items():
                report_lines.append(f"{sector:25}  {count} signals")
            report_lines.append("")

        if self.signals:
            report_lines.extend([
                "RECENT SIGNALS (Last 5)",
                "-" * 80
            ])
            for signal in self.signals[-5:]:
                report_lines.append(
                    f"{signal.date.strftime('%Y-%m-%d')} | {signal.ticker:6} | "
                    f"Strength: {signal.signal_strength:3}/100 | "
                    f"Price: ${signal.entry_price:8.2f}"
                )
            report_lines.append("")

        if self.near_misses:
            report_lines.extend([
                "NEAR MISSES (Last 5)",
                "-" * 80,
                "These almost qualified - might be worth watching"
            ])
            for miss in self.near_misses[-5:]:
                report_lines.append(
                    f"{miss.date.strftime('%Y-%m-%d')} | {miss.ticker:6} | "
                    f"Strength: {miss.signal_strength:3}/100 | "
                    f"Missing: {', '.join(miss.conditions_failed[:2])}"
                )
            report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export all signals to CSV file.

        Args:
            filename: Output filename (defaults to timestamped name)

        Returns:
            Path to created file

        Example:
            >>> path = analyzer.export_to_csv('daily_signals.csv')
            >>> print(f"Exported to {path}")
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"signals_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        # Combine all records
        all_records = self.signals + self.near_misses

        if not all_records:
            logger.warning("No signals to export")
            return filepath

        # Convert to list of dicts for CSV writing
        rows = []
        for record in all_records:
            row = asdict(record)
            # Convert lists/dicts to strings for CSV
            row['conditions_met'] = ', '.join(row['conditions_met'])
            row['conditions_failed'] = ', '.join(row['conditions_failed'])
            row['condition_scores'] = str(row['condition_scores'])
            row['date'] = record.date.strftime('%Y-%m-%d %H:%M:%S')
            rows.append(row)

        # Write to CSV
        with open(filepath, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Exported {len(all_records)} signals to {filepath}")
        return filepath

    def clear(self) -> None:
        """
        Clear all recorded signals.

        Useful for starting a new analysis period.

        Example:
            >>> analyzer.clear()  # Start fresh
        """
        count = len(self.signals) + len(self.near_misses)
        self.signals = []
        self.near_misses = []
        logger.info(f"Cleared {count} signals from analyzer")
