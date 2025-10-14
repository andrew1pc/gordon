#!/usr/bin/env python3
"""
Run a backtest of the momentum trading strategy.

This script demonstrates the complete backtesting workflow:
1. Initialize all components
2. Run backtest on historical data
3. Calculate performance metrics
4. Generate visualizations and report
"""

import logging
from datetime import datetime, timedelta

from config.api_config import TiingoConfig
from config.strategy_config import ScannerConfig, RiskConfig
from data.fetcher import TiingoClient
from strategy.scanner import AssetScanner
from strategy.signals import SignalGenerator
from strategy.risk_manager import RiskManager
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from backtest.visualizations import generate_all_plots


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """Run backtest and generate results."""
    print_header("Momentum Trading Strategy Backtest")

    # Configuration
    INITIAL_CAPITAL = 100000
    START_DATE = '2023-01-01'
    END_DATE = '2023-06-30'  # 6 months for faster testing

    print(f"\nBacktest Parameters:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"  Period: {START_DATE} to {END_DATE}")

    # Initialize components
    print_header("Initializing Components")

    try:
        # API client
        api_config = TiingoConfig()
        client = TiingoClient(api_config)
        print("✓ Tiingo client initialized")

        # Scanner
        scanner_config = ScannerConfig()
        scanner = AssetScanner(client, scanner_config)
        print("✓ Asset scanner initialized")

        # Signal generator
        signal_generator = SignalGenerator()
        print("✓ Signal generator initialized")

        # Risk manager
        risk_config = RiskConfig()
        risk_manager = RiskManager(
            account_size=INITIAL_CAPITAL,
            risk_per_trade=risk_config.RISK_PER_TRADE,
            max_positions=risk_config.MAX_POSITIONS,
            max_portfolio_exposure=risk_config.MAX_PORTFOLIO_EXPOSURE,
            max_sector_exposure=risk_config.MAX_SECTOR_EXPOSURE,
            daily_loss_limit=risk_config.DAILY_LOSS_LIMIT,
            crypto_size_multiplier=risk_config.CRYPTO_SIZE_MULTIPLIER
        )
        print("✓ Risk manager initialized")

        # Backtest engine
        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            start_date=START_DATE,
            end_date=END_DATE,
            tiingo_client=client,
            scanner=scanner,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            slippage={'stock': 0.001, 'crypto': 0.003},
            commission={'stock': 0.0, 'crypto': 0.001},
            rescan_frequency=5
        )
        print("✓ Backtest engine initialized")

    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        return

    # Run backtest
    print_header("Running Backtest")
    print("\nThis may take several minutes...")
    print("Progress will be logged every 50 days.\n")

    try:
        results = engine.run()
        print("\n✓ Backtest completed successfully")
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        print(f"\n✗ Backtest failed: {e}")
        return

    # Calculate metrics
    print_header("Calculating Performance Metrics")

    try:
        metrics = PerformanceMetrics(results)
        print("✓ Metrics calculated")
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        return

    # Display summary
    print_header("Performance Summary")

    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-" * 52)
    print(f"{'Initial Capital':<30} ${results['summary']['initial_capital']:>19,.0f}")
    print(f"{'Final Value':<30} ${results['summary']['final_value']:>19,.0f}")
    print(f"{'Total Return':<30} {metrics.total_return():>19.2%}")
    print(f"{'CAGR':<30} {metrics.cagr():>19.2%}")
    print()
    print(f"{'Sharpe Ratio':<30} {metrics.sharpe_ratio():>20.2f}")
    print(f"{'Sortino Ratio':<30} {metrics.sortino_ratio():>20.2f}")
    print(f"{'Max Drawdown':<30} {metrics.max_drawdown():>19.2%}")
    print()
    print(f"{'Total Trades':<30} {metrics.total_trades():>20,}")
    print(f"{'Winning Trades':<30} {results['summary']['winning_trades']:>20,}")
    print(f"{'Win Rate':<30} {metrics.win_rate():>19.2%}")
    print(f"{'Profit Factor':<30} {metrics.profit_factor():>20.2f}")
    print(f"{'Expectancy':<30} ${metrics.expectancy():>19,.2f}")
    print()

    avg_win = metrics.average_win()
    avg_loss = metrics.average_loss()
    print(f"{'Average Win':<30} ${avg_win['dollars']:>19,.2f}")
    print(f"{'Average Loss':<30} ${avg_loss['dollars']:>19,.2f}")
    print()

    holding = metrics.avg_holding_period()
    print(f"{'Avg Holding Period':<30} {holding['all']:>17.1f} days")
    print(f"{'Exposure Time':<30} {metrics.exposure_time():>19.2%}")

    # Generate detailed report
    print_header("Generating Reports")

    try:
        report_file = 'backtest_report.txt'
        metrics.generate_report(report_file)
        print(f"✓ Detailed report saved to {report_file}")
    except Exception as e:
        print(f"✗ Report generation failed: {e}")

    # Generate visualizations
    print_header("Generating Visualizations")

    try:
        generate_all_plots(results, output_dir='output')
        print("✓ All visualizations generated")
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        print(f"✗ Visualization generation failed: {e}")

    # Save results to CSV
    print_header("Saving Trade Data")

    try:
        if not results['trades'].empty:
            results['trades'].to_csv('output/all_trades.csv', index=False)
            print("✓ All trades saved to output/all_trades.csv")

        if not results['exit_trades'].empty:
            results['exit_trades'].to_csv('output/completed_trades.csv', index=False)
            print("✓ Completed trades saved to output/completed_trades.csv")

        if not results['equity_curve'].empty:
            results['equity_curve'].to_csv('output/equity_curve.csv')
            print("✓ Equity curve saved to output/equity_curve.csv")

    except Exception as e:
        print(f"✗ Data export failed: {e}")

    # Final summary
    print_header("Backtest Complete")

    # Evaluate against targets
    print("\nPerformance vs Targets:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio():.2f} (target: > 1.0) {'✓' if metrics.sharpe_ratio() > 1.0 else '✗'}")
    print(f"  Max Drawdown: {metrics.max_drawdown():.1%} (target: < 20%) {'✓' if metrics.max_drawdown() < 0.20 else '✗'}")
    print(f"  Win Rate: {metrics.win_rate():.1%} (target: > 45%) {'✓' if metrics.win_rate() > 0.45 else '✗'}")
    print(f"  Profit Factor: {metrics.profit_factor():.2f} (target: > 1.5) {'✓' if metrics.profit_factor() > 1.5 else '✗'}")

    print("\nAll results saved to:")
    print("  - backtest_report.txt (detailed report)")
    print("  - output/ directory (charts and data)")
    print("\nBacktest workflow complete!\n")


if __name__ == '__main__':
    main()
