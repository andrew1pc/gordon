#!/usr/bin/env python3
"""
Main orchestration module for Momentum Trading Strategy.

This module coordinates all strategy components and provides paper trading capability.
"""

import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from config.api_config import TiingoConfig
from data.fetcher import TiingoClient
from strategy.scanner import AssetScanner
from strategy.signals import SignalGenerator, Position
from strategy.risk_manager import RiskManager
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from backtest.visualizations import generate_all_plots


logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Record of a paper trade."""
    trade_id: int
    ticker: str
    asset_type: str
    trade_type: str  # 'entry' or 'exit'
    date: datetime
    price: float
    shares: float
    value: float
    commission: float
    slippage_cost: float
    pnl_dollars: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_type: Optional[str] = None


class MomentumStrategy:
    """
    Main orchestration class for momentum trading strategy.

    Coordinates scanner, signal generator, risk manager, and trade execution
    for backtesting and paper trading.

    Example:
        >>> strategy = MomentumStrategy('config/strategy_config.yaml')
        >>> strategy.initialize_components()
        >>> strategy.run_daily(datetime.now())
    """

    def __init__(self, config_path: str = 'config/strategy_config.yaml'):
        """
        Initialize momentum strategy.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = None

        # Components (initialized later)
        self.tiingo_client = None
        self.scanner = None
        self.signal_generator = None
        self.risk_manager = None

        # Paper trading state
        self.mode = 'paper'
        self.paper_portfolio = {
            'cash': 0.0,
            'initial_capital': 0.0,
            'positions': {},
            'equity_curve': [],
            'start_date': None
        }
        self.trade_log: List[PaperTrade] = []
        self.trade_id_counter = 0

        # Caching
        self.cached_candidates = None
        self.cache_date = None
        self.cached_price_data = {}

        # Directories
        self.log_dir = 'logs'
        self.snapshot_dir = 'snapshots'
        self.report_dir = 'reports'
        self.output_dir = 'output'

    def load_config(self, config_path: Optional[str] = None) -> bool:
        """
        Load configuration from YAML file.

        Args:
            config_path: Optional override for config path

        Returns:
            True if successful

        Example:
            >>> strategy.load_config('my_config.yaml')
        """
        if config_path:
            self.config_path = config_path

        try:
            with open(self.config_path, 'r') as f:
                config_text = f.read()

            # Substitute environment variables
            config_text = os.path.expandvars(config_text)

            # Parse YAML
            self.config = yaml.safe_load(config_text)

            logger.info(f"Configuration loaded from {self.config_path}")
            self.mode = self.config.get('trading', {}).get('mode', 'paper')

            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def initialize_components(self) -> bool:
        """
        Initialize all strategy components.

        Returns:
            True if successful

        Example:
            >>> if strategy.initialize_components():
            ...     print("Ready to trade")
        """
        try:
            # Load config if not already loaded
            if self.config is None:
                if not self.load_config():
                    return False

            # Create directories
            for directory in [self.log_dir, self.snapshot_dir, self.report_dir, self.output_dir]:
                os.makedirs(directory, exist_ok=True)

            # Setup logging
            self._setup_logging()

            logger.info("Initializing components...")

            # Initialize API client
            api_config = TiingoConfig()
            self.tiingo_client = TiingoClient(api_config)
            logger.info("✓ Tiingo client initialized")

            # Initialize scanner
            from config.strategy_config import ScannerConfig
            scanner_config = ScannerConfig()
            scanner_config.MOMENTUM_THRESHOLD = self.config['scanner']['momentum_threshold']
            scanner_config.MAX_CANDIDATES = self.config['scanner']['max_candidates']

            self.scanner = AssetScanner(self.tiingo_client, scanner_config)
            logger.info("✓ Asset scanner initialized")

            # Initialize signal generator
            self.signal_generator = SignalGenerator()
            logger.info("✓ Signal generator initialized")

            # Initialize risk manager
            initial_capital = self.config['risk']['initial_capital']
            self.risk_manager = RiskManager(
                account_size=initial_capital,
                risk_per_trade=self.config['risk']['risk_per_trade'],
                max_positions=self.config['risk']['max_positions'],
                max_portfolio_exposure=self.config['risk']['max_portfolio_exposure'],
                max_sector_exposure=self.config['risk']['max_sector_exposure'],
                daily_loss_limit=self.config['risk']['max_daily_loss']
            )
            logger.info("✓ Risk manager initialized")

            # Initialize paper portfolio if in paper mode
            if self.mode == 'paper':
                self.paper_portfolio['cash'] = initial_capital
                self.paper_portfolio['initial_capital'] = initial_capital
                self.paper_portfolio['start_date'] = datetime.now()
                logger.info(f"✓ Paper portfolio initialized with ${initial_capital:,.0f}")

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Component initialization failed: {e}", exc_info=True)
            return False

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = self.config.get('logging', {}).get('log_file', 'logs/momentum_strategy.log')

        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level))

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def run_daily(self, date: datetime) -> Dict:
        """
        Execute daily trading workflow.

        Args:
            date: Trading date to process

        Returns:
            Dict with day's results

        Example:
            >>> results = strategy.run_daily(datetime.now())
            >>> print(f"Trades today: {results['entries'] + results['exits']}")
        """
        logger.info(f"{'='*80}")
        logger.info(f"Starting daily cycle for {date.date()}")
        logger.info(f"{'='*80}")

        results = {
            'date': date,
            'entries': 0,
            'exits': 0,
            'errors': []
        }

        try:
            # 1. Scan for candidates
            logger.info("Step 1: Scanning for candidates...")
            candidates = self.scan_and_select_candidates(date)
            logger.info(f"Found {len(candidates)} candidates")

            # 2. Process signals (exits first, then entries)
            logger.info("Step 2: Processing signals...")
            signals = self.process_signals(date, candidates)
            logger.info(f"Signals: {len(signals['exits'])} exits, {len(signals['entries'])} entries")

            # 3. Execute trades
            logger.info("Step 3: Executing trades...")
            self.execute_approved_trades(signals['exits'] + signals['entries'])
            results['entries'] = len(signals['entries'])
            results['exits'] = len(signals['exits'])

            # 4. Update positions
            logger.info("Step 4: Updating positions...")
            self.update_positions(date)

            # 5. Generate daily report
            logger.info("Step 5: Generating daily report...")
            report = self.generate_daily_report(date)

            # 6. Save state
            if self.config.get('logging', {}).get('save_daily_snapshots', True):
                logger.info("Step 6: Saving state...")
                self.save_state(f"{self.snapshot_dir}/state_{date.strftime('%Y-%m-%d')}.json")

            # 7. Send alerts
            if any(self.config.get('alerts', {}).values()):
                logger.info("Step 7: Sending alerts...")
                self.send_alerts(signals, results)

            logger.info(f"Daily cycle completed successfully")

        except Exception as e:
            logger.error(f"Error in daily cycle: {e}", exc_info=True)
            results['errors'].append(str(e))

        logger.info(f"{'='*80}\n")
        return results

    def scan_and_select_candidates(self, date: datetime) -> List[str]:
        """
        Scan for trading candidates with caching.

        Args:
            date: Current date

        Returns:
            List of candidate tickers
        """
        rescan_interval = self.config['scanner']['rescan_interval_days']

        # Check if we need to rescan
        need_rescan = (
            self.cached_candidates is None or
            self.cache_date is None or
            (date - self.cache_date).days >= rescan_interval
        )

        if need_rescan:
            logger.info(f"Running fresh scan (cache {'missing' if self.cached_candidates is None else 'expired'})")

            # Get universes
            stock_universe = self.scanner._get_sp500_tickers()[:50]  # Limit for speed
            crypto_universe = self.scanner.get_crypto_universe(top_n=10)

            # Fetch prices
            lookback = 90
            start_str = (date - timedelta(days=lookback)).strftime('%Y-%m-%d')
            end_str = date.strftime('%Y-%m-%d')

            stock_prices = self.scanner.fetch_universe_prices(stock_universe, start_str, end_str, 'stock')
            crypto_prices = self.scanner.fetch_universe_prices(crypto_universe, start_str, end_str, 'crypto')

            # Prepare and rank
            all_prices = {**stock_prices, **crypto_prices}
            prepared = self.scanner.prepare_universe_data(all_prices)
            ranking = self.scanner.rank_universe_by_momentum(prepared)
            candidates_df = self.scanner.select_top_candidates(ranking)

            self.cached_candidates = candidates_df['ticker'].tolist()
            self.cache_date = date
            self.cached_price_data = prepared

            logger.info(f"Scan complete: {len(self.cached_candidates)} candidates cached")
        else:
            days_since_scan = (date - self.cache_date).days
            logger.info(f"Using cached scan ({days_since_scan} days old)")

        # Filter out positions we already hold
        open_tickers = list(self.paper_portfolio['positions'].keys())
        candidates = [t for t in self.cached_candidates if t not in open_tickers]

        return candidates[:10]  # Return top 10

    def process_signals(self, date: datetime, candidates: List[str]) -> Dict[str, List]:
        """
        Process entry and exit signals.

        Args:
            date: Current date
            candidates: List of candidate tickers

        Returns:
            Dict with 'exits' and 'entries' lists
        """
        signals = {'exits': [], 'entries': []}

        # Process exits first
        for ticker, position in list(self.paper_portfolio['positions'].items()):
            try:
                if ticker not in self.cached_price_data:
                    continue

                df = self.cached_price_data[ticker]
                df_dates = pd.to_datetime(df.index)
                valid_dates = df_dates[df_dates <= date]
                if len(valid_dates) == 0:
                    continue

                current_idx = df_dates.get_loc(valid_dates[-1])
                exit_signal = self.signal_generator.check_exit_signals(position, df, current_idx)

                if exit_signal:
                    signals['exits'].append({
                        'type': 'exit',
                        'ticker': ticker,
                        'position': position,
                        'date': date,
                        'price': exit_signal['exit_price'],
                        'exit_type': exit_signal['exit_type']
                    })

            except Exception as e:
                logger.warning(f"Error checking exit for {ticker}: {e}")

        # Process entries
        for ticker in candidates:
            try:
                if ticker not in self.cached_price_data:
                    continue

                df = self.cached_price_data[ticker]
                df_dates = pd.to_datetime(df.index)
                valid_dates = df_dates[df_dates <= date]
                if len(valid_dates) == 0:
                    continue

                current_idx = df_dates.get_loc(valid_dates[-1])
                signal = self.signal_generator.check_entry_signals(df, current_idx, ticker)

                if signal:
                    asset_type = 'crypto' if ticker.endswith('usd') else 'stock'
                    entry_price = self.signal_generator.calculate_entry_price(df, current_idx, asset_type)
                    stop_loss = self.signal_generator.calculate_initial_stop(entry_price, df, current_idx, asset_type)
                    sector = 'Technology' if asset_type == 'stock' else None

                    # Validate with risk manager
                    approved, reason, position_size, dollar_value = self.risk_manager.validate_new_trade(
                        ticker, entry_price, stop_loss, sector, asset_type
                    )

                    if approved:
                        signals['entries'].append({
                            'type': 'entry',
                            'ticker': ticker,
                            'asset_type': asset_type,
                            'date': date,
                            'price': entry_price,
                            'shares': position_size,
                            'stop_loss': stop_loss,
                            'sector': sector
                        })
                    else:
                        logger.debug(f"Entry rejected for {ticker}: {reason}")

            except Exception as e:
                logger.warning(f"Error checking entry for {ticker}: {e}")

        return signals

    def execute_approved_trades(self, trades: List[Dict]) -> None:
        """
        Execute approved trades in paper trading mode.

        Args:
            trades: List of trade dicts
        """
        for trade_dict in trades:
            try:
                if trade_dict['type'] == 'entry':
                    self._execute_paper_entry(trade_dict)
                else:
                    self._execute_paper_exit(trade_dict)
            except Exception as e:
                logger.error(f"Failed to execute trade for {trade_dict.get('ticker')}: {e}")

    def _execute_paper_entry(self, trade_dict: Dict) -> None:
        """Execute paper entry trade."""
        ticker = trade_dict['ticker']
        asset_type = trade_dict['asset_type']
        entry_price = trade_dict['price']
        shares = trade_dict['shares']
        date = trade_dict['date']

        # Apply slippage
        slippage_key = f'slippage_{asset_type}'
        slippage_rate = self.config['trading'][slippage_key]
        actual_price = entry_price * (1 + slippage_rate)
        slippage_cost = (actual_price - entry_price) * shares

        # Calculate value
        trade_value = shares * actual_price

        # Apply commission
        commission_key = f'commission_{asset_type}'
        commission = trade_value * self.config['trading'][commission_key]

        # Total cost
        total_cost = trade_value + commission

        # Check cash
        if total_cost > self.paper_portfolio['cash']:
            logger.warning(f"Insufficient cash for {ticker}: need ${total_cost:,.0f}, have ${self.paper_portfolio['cash']:,.0f}")
            return

        # Deduct cash
        self.paper_portfolio['cash'] -= total_cost

        # Calculate target
        target_price = self.signal_generator.calculate_profit_target(
            entry_price,
            self.cached_price_data[ticker],
            len(self.cached_price_data[ticker]) - 1,
            asset_type
        )

        # Create position
        position = Position(
            ticker=ticker,
            asset_type=asset_type,
            entry_date=date,
            entry_price=actual_price,
            initial_stop=trade_dict['stop_loss'],
            trailing_stop=None,
            highest_price=actual_price,
            target_price=target_price,
            days_held=0
        )

        self.paper_portfolio['positions'][ticker] = position

        # Add to risk manager
        self.risk_manager.add_position(
            ticker=ticker,
            asset_type=asset_type,
            sector=trade_dict.get('sector'),
            entry_price=actual_price,
            position_size=shares,
            stop_loss=trade_dict['stop_loss'],
            entry_date=date
        )

        # Record trade
        self.trade_id_counter += 1
        paper_trade = PaperTrade(
            trade_id=self.trade_id_counter,
            ticker=ticker,
            asset_type=asset_type,
            trade_type='entry',
            date=date,
            price=actual_price,
            shares=shares,
            value=trade_value,
            commission=commission,
            slippage_cost=slippage_cost
        )
        self.trade_log.append(paper_trade)

        logger.info(
            f"ENTRY: {ticker} {shares:.2f} @ ${actual_price:.2f} = ${trade_value:,.0f} "
            f"(stop: ${trade_dict['stop_loss']:.2f}, target: ${target_price:.2f})"
        )

    def _execute_paper_exit(self, trade_dict: Dict) -> None:
        """Execute paper exit trade."""
        ticker = trade_dict['ticker']
        position = trade_dict['position']
        exit_price = trade_dict['price']
        date = trade_dict['date']
        exit_type = trade_dict['exit_type']

        if ticker not in self.risk_manager.positions:
            logger.warning(f"Cannot exit {ticker}: not in risk manager")
            return

        shares = self.risk_manager.positions[ticker].position_size

        # Apply slippage
        slippage_key = f'slippage_{position.asset_type}'
        slippage_rate = self.config['trading'][slippage_key]
        actual_price = exit_price * (1 - slippage_rate)
        slippage_cost = (exit_price - actual_price) * shares

        # Calculate proceeds
        trade_value = shares * actual_price
        commission_key = f'commission_{position.asset_type}'
        commission = trade_value * self.config['trading'][commission_key]
        net_proceeds = trade_value - commission

        # Add to cash
        self.paper_portfolio['cash'] += net_proceeds

        # Calculate P&L
        pnl_dollars = (actual_price - position.entry_price) * shares - commission
        pnl_percent = (actual_price - position.entry_price) / position.entry_price

        # Remove from risk manager and positions
        self.risk_manager.remove_position(ticker, actual_price)
        del self.paper_portfolio['positions'][ticker]

        # Record trade
        self.trade_id_counter += 1
        paper_trade = PaperTrade(
            trade_id=self.trade_id_counter,
            ticker=ticker,
            asset_type=position.asset_type,
            trade_type='exit',
            date=date,
            price=actual_price,
            shares=shares,
            value=trade_value,
            commission=commission,
            slippage_cost=slippage_cost,
            pnl_dollars=pnl_dollars,
            pnl_percent=pnl_percent,
            exit_type=exit_type
        )
        self.trade_log.append(paper_trade)

        logger.info(
            f"EXIT: {ticker} {shares:.2f} @ ${actual_price:.2f} = ${trade_value:,.0f} "
            f"({exit_type}, P&L: ${pnl_dollars:,.0f} / {pnl_percent:.1%})"
        )

    def update_positions(self, date: datetime) -> None:
        """
        Update all position states.

        Args:
            date: Current date
        """
        for ticker, position in list(self.paper_portfolio['positions'].items()):
            try:
                if ticker in self.cached_price_data:
                    df = self.cached_price_data[ticker]
                    df_dates = pd.to_datetime(df.index)
                    valid_dates = df_dates[df_dates <= date]
                    if len(valid_dates) > 0:
                        latest_idx = df_dates.get_loc(valid_dates[-1])
                        current_price = df.iloc[latest_idx]['close']

                        # Update position
                        position.days_held += 1
                        if current_price > position.highest_price:
                            position.highest_price = current_price

                        # Update in risk manager
                        if ticker in self.risk_manager.positions:
                            self.risk_manager.update_position_price(ticker, current_price)

                        # Update trailing stop
                        self.signal_generator.update_trailing_stop(position, df.iloc[latest_idx])

            except Exception as e:
                logger.warning(f"Error updating position {ticker}: {e}")

    def generate_daily_report(self, date: datetime) -> str:
        """
        Generate daily performance report.

        Args:
            date: Report date

        Returns:
            Report as string
        """
        # Calculate portfolio value
        position_value = sum(
            self.risk_manager.positions[ticker].dollar_value
            for ticker in self.paper_portfolio['positions'].keys()
            if ticker in self.risk_manager.positions
        )
        total_value = self.paper_portfolio['cash'] + position_value

        # Calculate unrealized P&L
        unrealized_pnl = sum(
            (self.risk_manager.positions[ticker].current_price -
             self.risk_manager.positions[ticker].entry_price) *
            self.risk_manager.positions[ticker].position_size
            for ticker in self.paper_portfolio['positions'].keys()
            if ticker in self.risk_manager.positions
        )

        # Calculate realized P&L today
        today_trades = [t for t in self.trade_log if t.date.date() == date.date() and t.trade_type == 'exit']
        realized_pnl_today = sum(t.pnl_dollars for t in today_trades if t.pnl_dollars)

        # Build report
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"DAILY REPORT - {date.date()}")
        lines.append(f"{'='*80}\n")

        lines.append("PORTFOLIO SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Value:        ${total_value:>20,.0f}")
        lines.append(f"Cash Balance:       ${self.paper_portfolio['cash']:>20,.0f}")
        lines.append(f"Position Value:     ${position_value:>20,.0f}")
        lines.append(f"Unrealized P&L:     ${unrealized_pnl:>20,.0f}")
        lines.append(f"Realized P&L Today: ${realized_pnl_today:>20,.0f}")
        lines.append(f"Open Positions:     {len(self.paper_portfolio['positions']):>20,}")
        lines.append("")

        # Open positions
        if self.paper_portfolio['positions']:
            lines.append("OPEN POSITIONS")
            lines.append("-" * 80)
            for ticker, position in self.paper_portfolio['positions'].items():
                if ticker in self.risk_manager.positions:
                    rm_pos = self.risk_manager.positions[ticker]
                    pnl_pct = (rm_pos.current_price - rm_pos.entry_price) / rm_pos.entry_price
                    lines.append(
                        f"{ticker:8} | Entry: ${rm_pos.entry_price:>8.2f} | "
                        f"Current: ${rm_pos.current_price:>8.2f} | "
                        f"P&L: {pnl_pct:>7.1%} | Days: {position.days_held:>3}"
                    )
            lines.append("")

        # Today's trades
        today_all_trades = [t for t in self.trade_log if t.date.date() == date.date()]
        if today_all_trades:
            lines.append("TODAY'S ACTIVITY")
            lines.append("-" * 80)
            for trade in today_all_trades:
                if trade.trade_type == 'entry':
                    lines.append(f"ENTRY: {trade.ticker} {trade.shares:.2f} @ ${trade.price:.2f}")
                else:
                    lines.append(f"EXIT:  {trade.ticker} {trade.shares:.2f} @ ${trade.price:.2f} ({trade.exit_type}, P&L: ${trade.pnl_dollars:,.0f})")
            lines.append("")

        lines.append(f"{'='*80}\n")

        report = "\n".join(lines)

        # Save to file
        report_file = f"{self.report_dir}/daily_report_{date.strftime('%Y-%m-%d')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        print(report)
        return report

    def send_alerts(self, signals: Dict, results: Dict) -> None:
        """
        Send alerts based on configuration.

        Args:
            signals: Signals dict
            results: Results dict
        """
        # For now, just log alerts
        # In production, implement email/Slack integration
        if self.config['alerts']['alert_on_entry'] and signals['entries']:
            logger.info(f"ALERT: {len(signals['entries'])} new entries today")

        if self.config['alerts']['alert_on_exit'] and signals['exits']:
            logger.info(f"ALERT: {len(signals['exits'])} exits today")

    def save_state(self, filepath: str) -> bool:
        """
        Save current state to file.

        Args:
            filepath: Path to save state

        Returns:
            True if successful
        """
        try:
            state = {
                'date': datetime.now().isoformat(),
                'paper_portfolio': {
                    'cash': self.paper_portfolio['cash'],
                    'initial_capital': self.paper_portfolio['initial_capital'],
                    'start_date': self.paper_portfolio['start_date'].isoformat() if self.paper_portfolio['start_date'] else None,
                    'positions': {
                        ticker: asdict(position)
                        for ticker, position in self.paper_portfolio['positions'].items()
                    }
                },
                'trade_log': [asdict(t) for t in self.trade_log],
                'trade_id_counter': self.trade_id_counter
            }

            # Convert datetime objects in positions
            for ticker, pos_dict in state['paper_portfolio']['positions'].items():
                if 'entry_date' in pos_dict and pos_dict['entry_date']:
                    pos_dict['entry_date'] = pos_dict['entry_date'].isoformat()

            # Convert datetime objects in trades
            for trade in state['trade_log']:
                if 'date' in trade and trade['date']:
                    trade['date'] = trade['date'].isoformat()

            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"State saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load_state(self, filepath: str) -> bool:
        """
        Load state from file.

        Args:
            filepath: Path to state file

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.paper_portfolio['cash'] = state['paper_portfolio']['cash']
            self.paper_portfolio['initial_capital'] = state['paper_portfolio']['initial_capital']

            if state['paper_portfolio']['start_date']:
                self.paper_portfolio['start_date'] = datetime.fromisoformat(state['paper_portfolio']['start_date'])

            # Reconstruct positions
            self.paper_portfolio['positions'] = {}
            for ticker, pos_dict in state['paper_portfolio']['positions'].items():
                if 'entry_date' in pos_dict and pos_dict['entry_date']:
                    pos_dict['entry_date'] = datetime.fromisoformat(pos_dict['entry_date'])
                self.paper_portfolio['positions'][ticker] = Position(**pos_dict)

            # Reconstruct trades
            self.trade_log = []
            for trade_dict in state['trade_log']:
                if 'date' in trade_dict and trade_dict['date']:
                    trade_dict['date'] = datetime.fromisoformat(trade_dict['date'])
                self.trade_log.append(PaperTrade(**trade_dict))

            self.trade_id_counter = state['trade_id_counter']

            logger.info(f"State loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Momentum Trading Strategy')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Paper trading commands
    paper_parser = subparsers.add_parser('paper', help='Paper trading mode')
    paper_parser.add_argument('--init', action='store_true', help='Initialize paper trading')
    paper_parser.add_argument('--run', action='store_true', help='Run today\'s trading cycle')
    paper_parser.add_argument('--report', action='store_true', help='Display current status')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')

    # Scan command
    subparsers.add_parser('scan', help='Run candidate scan')

    args = parser.parse_args()

    if args.command == 'paper':
        strategy = MomentumStrategy()
        if not strategy.initialize_components():
            print("Failed to initialize components")
            return 1

        if args.init:
            print(f"Paper trading initialized with ${strategy.paper_portfolio['initial_capital']:,.0f}")
            strategy.save_state(f"{strategy.snapshot_dir}/initial_state.json")
            print("Initial state saved")

        elif args.run:
            strategy.run_daily(datetime.now())

        elif args.report:
            strategy.generate_daily_report(datetime.now())

    elif args.command == 'backtest':
        print(f"Running backtest from {args.start} to {args.end}")
        print("Use run_backtest.py for full backtest functionality")

    elif args.command == 'scan':
        strategy = MomentumStrategy()
        if strategy.initialize_components():
            candidates = strategy.scan_and_select_candidates(datetime.now())
            print(f"\nTop Candidates ({len(candidates)}):")
            for ticker in candidates:
                print(f"  - {ticker}")

    else:
        parser.print_help()

    return 0


if __name__ == '__main__':
    import pandas as pd  # Import here to avoid issues if not used
    sys.exit(main())
