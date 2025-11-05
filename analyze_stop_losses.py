"""
Analyze stop loss exits and slippage patterns from trade history.

This tool parses trade history and generates a detailed report on:
- Average, median, and maximum slippage
- Breakdown by asset type (stocks vs crypto)
- Gap vs non-gap stop losses
- Frequency of stop loss hits
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import statistics


class StopLossAnalyzer:
    """Analyze stop loss slippage patterns from trade data."""

    def __init__(self, trade_history_path: str):
        """
        Initialize analyzer with trade history file.

        Args:
            trade_history_path: Path to trade_history.json
        """
        self.trade_history_path = Path(trade_history_path)
        self.trades = []
        self.stop_losses = []

    def load_trades(self) -> None:
        """Load trade history from JSON file."""
        if not self.trade_history_path.exists():
            raise FileNotFoundError(f"Trade history not found: {self.trade_history_path}")

        with open(self.trade_history_path, 'r') as f:
            data = json.load(f)
            self.trades = data.get('closed_positions', [])

    def extract_stop_losses(self) -> None:
        """Extract stop loss exits from trade history."""
        self.stop_losses = []

        for trade in self.trades:
            if trade.get('exit_type') == 'stop_loss':
                self.stop_losses.append(trade)

    def calculate_statistics(self) -> Dict:
        """
        Calculate slippage statistics.

        Returns:
            Dict with overall and segmented statistics
        """
        if not self.stop_losses:
            return {
                'total_stops': 0,
                'message': 'No stop loss exits found in trade history'
            }

        # Overall statistics
        all_slippage = [s.get('slippage_pct', 0) * 100 for s in self.stop_losses]

        stats = {
            'total_stops': len(self.stop_losses),
            'overall': {
                'avg_slippage': statistics.mean(all_slippage) if all_slippage else 0,
                'median_slippage': statistics.median(all_slippage) if all_slippage else 0,
                'max_slippage': max(all_slippage) if all_slippage else 0,
                'min_slippage': min(all_slippage) if all_slippage else 0
            }
        }

        # By asset type
        by_asset = defaultdict(list)
        for stop in self.stop_losses:
            asset_type = stop.get('asset_type', 'unknown')
            slippage = stop.get('slippage_pct', 0) * 100
            by_asset[asset_type].append(slippage)

        stats['by_asset_type'] = {}
        for asset_type, slippages in by_asset.items():
            stats['by_asset_type'][asset_type] = {
                'count': len(slippages),
                'avg_slippage': statistics.mean(slippages) if slippages else 0,
                'median_slippage': statistics.median(slippages) if slippages else 0,
                'max_slippage': max(slippages) if slippages else 0
            }

        # By gap vs non-gap
        gap_slippage = []
        non_gap_slippage = []

        for stop in self.stop_losses:
            slippage = stop.get('slippage_pct', 0) * 100
            if stop.get('gap_detected', False):
                gap_slippage.append(slippage)
            else:
                non_gap_slippage.append(slippage)

        stats['gap_analysis'] = {
            'gap_stops': {
                'count': len(gap_slippage),
                'avg_slippage': statistics.mean(gap_slippage) if gap_slippage else 0,
                'median_slippage': statistics.median(gap_slippage) if gap_slippage else 0
            },
            'non_gap_stops': {
                'count': len(non_gap_slippage),
                'avg_slippage': statistics.mean(non_gap_slippage) if non_gap_slippage else 0,
                'median_slippage': statistics.median(non_gap_slippage) if non_gap_slippage else 0
            }
        }

        # Top worst exits (by slippage)
        sorted_stops = sorted(self.stop_losses,
                            key=lambda x: x.get('slippage_pct', 0),
                            reverse=True)

        stats['worst_exits'] = []
        for stop in sorted_stops[:5]:  # Top 5 worst
            stats['worst_exits'].append({
                'ticker': stop.get('ticker'),
                'exit_date': stop.get('exit_date'),
                'stop_price': stop.get('stop_price'),
                'exit_price': stop.get('exit_price'),
                'slippage_pct': stop.get('slippage_pct', 0) * 100,
                'gap_detected': stop.get('gap_detected', False)
            })

        return stats

    def format_report(self, stats: Dict) -> str:
        """
        Format statistics as readable report.

        Args:
            stats: Statistics dictionary

        Returns:
            Formatted report string
        """
        if stats.get('total_stops', 0) == 0:
            return stats.get('message', 'No data to report')

        lines = []
        lines.append("=" * 70)
        lines.append("STOP LOSS SLIPPAGE ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Overall statistics
        lines.append(f"Total Stop Loss Exits: {stats['total_stops']}")
        lines.append("")
        lines.append("OVERALL SLIPPAGE:")
        lines.append(f"  Average:    {stats['overall']['avg_slippage']:.2f}%")
        lines.append(f"  Median:     {stats['overall']['median_slippage']:.2f}%")
        lines.append(f"  Maximum:    {stats['overall']['max_slippage']:.2f}%")
        lines.append(f"  Minimum:    {stats['overall']['min_slippage']:.2f}%")
        lines.append("")

        # By asset type
        lines.append("BY ASSET TYPE:")
        for asset_type, asset_stats in stats['by_asset_type'].items():
            lines.append(f"  {asset_type.upper()}:")
            lines.append(f"    Count:      {asset_stats['count']}")
            lines.append(f"    Avg:        {asset_stats['avg_slippage']:.2f}%")
            lines.append(f"    Median:     {asset_stats['median_slippage']:.2f}%")
            lines.append(f"    Max:        {asset_stats['max_slippage']:.2f}%")
            lines.append("")

        # Gap analysis
        gap = stats['gap_analysis']
        lines.append("GAP ANALYSIS:")
        lines.append(f"  Gap Downs:")
        lines.append(f"    Count:      {gap['gap_stops']['count']}")
        lines.append(f"    Avg:        {gap['gap_stops']['avg_slippage']:.2f}%")
        lines.append(f"    Median:     {gap['gap_stops']['median_slippage']:.2f}%")
        lines.append("")
        lines.append(f"  Normal Stops:")
        lines.append(f"    Count:      {gap['non_gap_stops']['count']}")
        lines.append(f"    Avg:        {gap['non_gap_stops']['avg_slippage']:.2f}%")
        lines.append(f"    Median:     {gap['non_gap_stops']['median_slippage']:.2f}%")
        lines.append("")

        # Worst exits
        if stats['worst_exits']:
            lines.append("TOP 5 WORST EXITS (by slippage):")
            for i, exit_info in enumerate(stats['worst_exits'], 1):
                gap_marker = " [GAP]" if exit_info['gap_detected'] else ""
                lines.append(f"  {i}. {exit_info['ticker']} on {exit_info['exit_date']}{gap_marker}")
                lines.append(f"     Stop: ${exit_info['stop_price']:.2f}, "
                           f"Exit: ${exit_info['exit_price']:.2f}, "
                           f"Slippage: {exit_info['slippage_pct']:.2f}%")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def run_analysis(self) -> str:
        """
        Run full analysis and return report.

        Returns:
            Formatted report string
        """
        self.load_trades()
        self.extract_stop_losses()
        stats = self.calculate_statistics()
        return self.format_report(stats)


def main():
    """Main entry point."""
    # Check for trade history path
    if len(sys.argv) > 1:
        trade_history_path = sys.argv[1]
    else:
        trade_history_path = 'data/trade_history.json'

    # Run analysis
    try:
        analyzer = StopLossAnalyzer(trade_history_path)
        report = analyzer.run_analysis()
        print(report)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUsage: python analyze_stop_losses.py [trade_history_path]")
        print("Default path: data/trade_history.json")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing stop losses: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
