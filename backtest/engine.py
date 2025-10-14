"""Backtesting engine for strategy evaluation."""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from data.fetcher import TiingoClient
from strategy.scanner import AssetScanner
from strategy.signals import SignalGenerator, Position
from strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
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
    # Exit-specific fields
    entry_date: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_type: Optional[str] = None
    holding_period: Optional[int] = None
    pnl_dollars: Optional[float] = None
    pnl_percent: Optional[float] = None


@dataclass
class DailyState:
    """Portfolio state for a single day."""
    date: datetime
    total_value: float
    cash_balance: float
    position_count: int
    unrealized_pnl: float
    realized_pnl_day: float
    drawdown_pct: float


class BacktestEngine:
    """
    Backtesting engine for momentum trading strategy.

    Simulates trading with realistic execution, slippage, and commissions.
    Enforces risk management rules throughout the backtest.

    Example:
        >>> engine = BacktestEngine(
        ...     initial_capital=100000,
        ...     start_date='2022-01-01',
        ...     end_date='2023-12-31',
        ...     client=tiingo_client,
        ...     scanner=scanner,
        ...     signal_generator=signal_gen,
        ...     risk_manager=risk_mgr
        ... )
        >>> results = engine.run()
        >>> print(f"Total Return: {results['summary']['total_return']:.1%}")
    """

    def __init__(
        self,
        initial_capital: float,
        start_date: str,
        end_date: str,
        tiingo_client: TiingoClient,
        scanner: AssetScanner,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager,
        slippage: Optional[Dict[str, float]] = None,
        commission: Optional[Dict[str, float]] = None,
        rescan_frequency: int = 5
    ):
        """
        Initialize the backtesting engine.

        Args:
            initial_capital: Starting capital
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            tiingo_client: TiingoClient for data fetching
            scanner: AssetScanner for candidate selection
            signal_generator: SignalGenerator for entry/exit signals
            risk_manager: RiskManager for position sizing
            slippage: Dict with 'stock' and 'crypto' slippage rates
            commission: Dict with 'stock' and 'crypto' commission rates
            rescan_frequency: Days between universe rescans (default 5)
        """
        self.initial_capital = initial_capital
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.client = tiingo_client
        self.scanner = scanner
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.rescan_frequency = rescan_frequency

        # Trading costs
        self.slippage = slippage or {'stock': 0.001, 'crypto': 0.003}
        self.commission = commission or {'stock': 0.0, 'crypto': 0.001}

        # State tracking
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_states: List[DailyState] = []
        self.trade_id_counter = 0

        # Performance tracking
        self.peak_value = initial_capital
        self.realized_pnl_total = 0.0

        # Caching
        self.cached_candidates = None
        self.cache_date = None
        self.cached_price_data = {}

        logger.info(
            f"BacktestEngine initialized: ${initial_capital:,.0f} capital, "
            f"{self.start_date.date()} to {self.end_date.date()}"
        )

    def run(self) -> Dict:
        """
        Execute the backtest.

        Returns:
            Dict with comprehensive backtest results

        Example:
            >>> results = engine.run()
            >>> print(f"Processed {results['stats']['total_trades']} trades")
        """
        logger.info("Starting backtest...")

        # Get trading days
        trading_days = pd.bdate_range(self.start_date, self.end_date)
        total_days = len(trading_days)

        logger.info(f"Processing {total_days} trading days...")

        # Main backtest loop
        for i, date in enumerate(trading_days):
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"Progress: {i+1}/{total_days} days ({(i+1)/total_days*100:.1f}%)")

            try:
                self.run_single_day(date)
            except Exception as e:
                logger.error(f"Error processing {date.date()}: {e}")
                continue

        # Close any remaining positions at final price
        self._close_all_positions(self.end_date)

        # Compile results
        results = self.get_results()

        logger.info(
            f"Backtest complete: "
            f"Final value ${results['summary']['final_value']:,.0f}, "
            f"Return {results['summary']['total_return']:.1%}"
        )

        return results

    def run_single_day(self, date: datetime) -> Dict:
        """
        Process one trading day.

        Args:
            date: Trading date to process

        Returns:
            Dict with day's trading activity
        """
        # Update prices for all open positions
        self._update_position_prices(date)

        # Check exits first (priority: stops, targets, etc.)
        exit_trades = self.check_exits(date)
        for trade_dict in exit_trades:
            self.execute_trade(trade_dict)

        # Scan for new candidates (with caching)
        candidates = self.scan_for_candidates(date)

        # Check for entry signals
        entry_trades = self.check_entries(candidates, date)
        for trade_dict in entry_trades:
            self.execute_trade(trade_dict)

        # Update portfolio state for this day
        self.update_portfolio_state(date)

        return {
            'date': date,
            'entries': len(entry_trades),
            'exits': len(exit_trades),
            'positions': len(self.positions)
        }

    def scan_for_candidates(self, date: datetime) -> List[str]:
        """
        Scan for trading candidates with caching.

        Args:
            date: Current date

        Returns:
            List of candidate tickers
        """
        # Check if we need to rescan
        need_rescan = (
            self.cached_candidates is None or
            self.cache_date is None or
            (date - self.cache_date).days >= self.rescan_frequency
        )

        if need_rescan:
            logger.info(f"Rescanning universe on {date.date()}")

            # Get universe (use small subset for speed in backtesting)
            stock_universe = self.scanner._get_sp500_tickers()[:50]  # Top 50 for speed
            crypto_universe = self.scanner.get_crypto_universe(top_n=10)

            # Fetch prices up to current date
            lookback = 90  # days
            start_str = (date - timedelta(days=lookback)).strftime('%Y-%m-%d')
            end_str = date.strftime('%Y-%m-%d')

            stock_prices = self.scanner.fetch_universe_prices(
                stock_universe, start_str, end_str, 'stock'
            )
            crypto_prices = self.scanner.fetch_universe_prices(
                crypto_universe, start_str, end_str, 'crypto'
            )

            # Prepare data with indicators
            all_prices = {**stock_prices, **crypto_prices}
            prepared = self.scanner.prepare_universe_data(all_prices)

            # Rank by momentum
            ranking = self.scanner.rank_universe_by_momentum(prepared)

            # Select top candidates
            candidates_df = self.scanner.select_top_candidates(ranking)

            self.cached_candidates = candidates_df['ticker'].tolist()
            self.cache_date = date
            self.cached_price_data = prepared

            logger.info(f"Scan complete: {len(self.cached_candidates)} candidates")

        return self.cached_candidates[:10]  # Return top 10 for entry consideration

    def check_entries(self, candidates: List[str], date: datetime) -> List[Dict]:
        """
        Check for entry signals in candidate list.

        Args:
            candidates: List of candidate tickers
            date: Current date

        Returns:
            List of entry trade dicts
        """
        entry_trades = []

        for ticker in candidates:
            # Skip if already in position
            if ticker in self.positions:
                continue

            # Get price data
            if ticker not in self.cached_price_data:
                continue

            df = self.cached_price_data[ticker]

            # Find index for current date (or nearest prior)
            try:
                df_dates = pd.to_datetime(df.index)
                valid_dates = df_dates[df_dates <= date]
                if len(valid_dates) == 0:
                    continue
                current_idx = df_dates.get_loc(valid_dates[-1])
            except:
                continue

            # Check for entry signal
            signal = self.signal_generator.check_entry_signals(df, current_idx, ticker)

            if signal:
                # Determine asset type
                asset_type = 'crypto' if ticker.endswith('usd') else 'stock'

                # Calculate entry price (next day's open + slippage)
                entry_price = self.signal_generator.calculate_entry_price(
                    df, current_idx, asset_type
                )

                # Calculate stop loss
                stop_loss = self.signal_generator.calculate_initial_stop(
                    entry_price, df, current_idx, asset_type
                )

                # Get sector (simplified - in production, use sector mapping)
                sector = 'Technology' if asset_type == 'stock' else None

                # Validate with risk manager
                approved, reason, position_size, dollar_value = self.risk_manager.validate_new_trade(
                    ticker, entry_price, stop_loss, sector, asset_type
                )

                if approved:
                    entry_trades.append({
                        'type': 'entry',
                        'ticker': ticker,
                        'asset_type': asset_type,
                        'date': date,
                        'price': entry_price,
                        'shares': position_size,
                        'stop_loss': stop_loss,
                        'sector': sector
                    })
                    logger.debug(f"Entry signal: {ticker} @ ${entry_price:.2f}")

        return entry_trades

    def check_exits(self, date: datetime) -> List[Dict]:
        """
        Check for exit signals in open positions.

        Args:
            date: Current date

        Returns:
            List of exit trade dicts
        """
        exit_trades = []

        for ticker, position in list(self.positions.items()):
            # Get price data
            if ticker not in self.cached_price_data:
                # Try to fetch if not in cache
                try:
                    lookback = 90
                    start_str = (date - timedelta(days=lookback)).strftime('%Y-%m-%d')
                    end_str = date.strftime('%Y-%m-%d')

                    if position.asset_type == 'stock':
                        df = self.client.get_stock_prices(ticker, start_str, end_str)
                    else:
                        df = self.client.get_crypto_prices(ticker, start_str, end_str)

                    if df is None or len(df) == 0:
                        continue

                    # Add indicators
                    from indicators.technical import TechnicalIndicators
                    from indicators.momentum import MomentumMetrics
                    tech = TechnicalIndicators()
                    mom = MomentumMetrics()
                    df = tech.add_all_indicators(df)
                    df = mom.add_all_momentum_metrics(df)

                    self.cached_price_data[ticker] = df
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {ticker}: {e}")
                    continue

            df = self.cached_price_data[ticker]

            # Find current date index
            try:
                df_dates = pd.to_datetime(df.index)
                valid_dates = df_dates[df_dates <= date]
                if len(valid_dates) == 0:
                    continue
                current_idx = df_dates.get_loc(valid_dates[-1])
            except:
                continue

            # Check for exit signal
            exit_signal = self.signal_generator.check_exit_signals(position, df, current_idx)

            if exit_signal:
                exit_trades.append({
                    'type': 'exit',
                    'ticker': ticker,
                    'position': position,
                    'date': date,
                    'price': exit_signal['exit_price'],
                    'exit_type': exit_signal['exit_type']
                })
                logger.debug(f"Exit signal: {ticker} @ ${exit_signal['exit_price']:.2f} ({exit_signal['exit_type']})")

        return exit_trades

    def execute_trade(self, trade_dict: Dict) -> None:
        """
        Execute a trade (entry or exit).

        Args:
            trade_dict: Dict with trade details
        """
        if trade_dict['type'] == 'entry':
            self._execute_entry(trade_dict)
        else:
            self._execute_exit(trade_dict)

    def _execute_entry(self, trade_dict: Dict) -> None:
        """Execute an entry trade."""
        ticker = trade_dict['ticker']
        asset_type = trade_dict['asset_type']
        entry_price = trade_dict['price']
        shares = trade_dict['shares']
        date = trade_dict['date']

        # Apply slippage
        slippage_rate = self.slippage[asset_type]
        actual_price = entry_price * (1 + slippage_rate)
        slippage_cost = (actual_price - entry_price) * shares

        # Calculate value
        trade_value = shares * actual_price

        # Apply commission
        commission = trade_value * self.commission[asset_type]

        # Total cost
        total_cost = trade_value + commission

        # Check sufficient capital
        if total_cost > self.cash:
            logger.warning(f"Insufficient capital for {ticker}: need ${total_cost:,.0f}, have ${self.cash:,.0f}")
            return

        # Deduct from cash
        self.cash -= total_cost

        # Calculate profit target
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

        self.positions[ticker] = position

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
        trade = Trade(
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
        self.trades.append(trade)

        logger.info(
            f"ENTRY: {ticker} {shares:.2f} @ ${actual_price:.2f} = ${trade_value:,.0f} "
            f"(stop: ${trade_dict['stop_loss']:.2f}, target: ${target_price:.2f})"
        )

    def _execute_exit(self, trade_dict: Dict) -> None:
        """Execute an exit trade."""
        ticker = trade_dict['ticker']
        position = trade_dict['position']
        exit_price = trade_dict['price']
        date = trade_dict['date']
        exit_type = trade_dict['exit_type']

        # Get shares from risk manager
        if ticker not in self.risk_manager.positions:
            logger.warning(f"Cannot exit {ticker}: not in risk manager")
            return

        shares = self.risk_manager.positions[ticker].position_size

        # Apply slippage (negative for sells)
        slippage_rate = self.slippage[position.asset_type]
        actual_price = exit_price * (1 - slippage_rate)
        slippage_cost = (exit_price - actual_price) * shares

        # Calculate proceeds
        trade_value = shares * actual_price

        # Apply commission
        commission = trade_value * self.commission[position.asset_type]

        # Net proceeds
        net_proceeds = trade_value - commission

        # Add to cash
        self.cash += net_proceeds

        # Calculate P&L
        pnl_dollars = (actual_price - position.entry_price) * shares - commission
        pnl_percent = (actual_price - position.entry_price) / position.entry_price

        # Update realized P&L
        self.realized_pnl_total += pnl_dollars

        # Holding period
        holding_period = (date - position.entry_date).days

        # Remove from risk manager
        self.risk_manager.remove_position(ticker, actual_price)

        # Remove from positions
        del self.positions[ticker]

        # Record trade
        self.trade_id_counter += 1
        trade = Trade(
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
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_type=exit_type,
            holding_period=holding_period,
            pnl_dollars=pnl_dollars,
            pnl_percent=pnl_percent
        )
        self.trades.append(trade)

        logger.info(
            f"EXIT: {ticker} {shares:.2f} @ ${actual_price:.2f} = ${trade_value:,.0f} "
            f"({exit_type}, held {holding_period}d, P&L: ${pnl_dollars:,.0f} / {pnl_percent:.1%})"
        )

    def _update_position_prices(self, date: datetime) -> None:
        """Update all position prices for current date."""
        for ticker, position in self.positions.items():
            if ticker in self.cached_price_data:
                df = self.cached_price_data[ticker]
                try:
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
                except:
                    pass

    def update_portfolio_state(self, date: datetime) -> None:
        """
        Update and record portfolio state for the day.

        Args:
            date: Current date
        """
        # Calculate position values
        position_value = sum(
            self.risk_manager.positions[ticker].dollar_value
            for ticker in self.positions.keys()
            if ticker in self.risk_manager.positions
        )

        # Total portfolio value
        total_value = self.cash + position_value

        # Calculate unrealized P&L
        unrealized_pnl = sum(
            (self.risk_manager.positions[ticker].current_price -
             self.risk_manager.positions[ticker].entry_price) *
            self.risk_manager.positions[ticker].position_size
            for ticker in self.positions.keys()
            if ticker in self.risk_manager.positions
        )

        # Update peak for drawdown calculation
        if total_value > self.peak_value:
            self.peak_value = total_value

        # Calculate drawdown
        drawdown_pct = (self.peak_value - total_value) / self.peak_value if self.peak_value > 0 else 0

        # Get today's realized P&L
        today_trades = [t for t in self.trades if t.date == date and t.trade_type == 'exit']
        realized_pnl_day = sum(t.pnl_dollars for t in today_trades if t.pnl_dollars)

        # Record daily state
        state = DailyState(
            date=date,
            total_value=total_value,
            cash_balance=self.cash,
            position_count=len(self.positions),
            unrealized_pnl=unrealized_pnl,
            realized_pnl_day=realized_pnl_day,
            drawdown_pct=drawdown_pct
        )
        self.daily_states.append(state)

    def _close_all_positions(self, date: datetime) -> None:
        """Close all remaining positions at end of backtest."""
        logger.info(f"Closing {len(self.positions)} remaining positions at {date.date()}")

        for ticker in list(self.positions.keys()):
            position = self.positions[ticker]

            # Get final price
            if ticker in self.cached_price_data:
                df = self.cached_price_data[ticker]
                final_price = df.iloc[-1]['close']
            else:
                final_price = position.entry_price  # Fallback

            # Create exit trade
            trade_dict = {
                'type': 'exit',
                'ticker': ticker,
                'position': position,
                'date': date,
                'price': final_price,
                'exit_type': 'backtest_end'
            }

            self._execute_exit(trade_dict)

    def get_results(self) -> Dict:
        """
        Compile comprehensive backtest results.

        Returns:
            Dict with all results and statistics
        """
        # Convert trades to DataFrame
        trades_df = pd.DataFrame([asdict(t) for t in self.trades])

        # Convert daily states to DataFrame
        equity_curve = pd.DataFrame([asdict(s) for s in self.daily_states])
        if not equity_curve.empty:
            equity_curve.set_index('date', inplace=True)

        # Calculate summary statistics
        final_value = equity_curve['total_value'].iloc[-1] if not equity_curve.empty else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Count trades
        exit_trades = trades_df[trades_df['trade_type'] == 'exit'] if not trades_df.empty else pd.DataFrame()
        total_trades = len(exit_trades)
        winning_trades = len(exit_trades[exit_trades['pnl_dollars'] > 0]) if total_trades > 0 else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Max drawdown
        max_drawdown = equity_curve['drawdown_pct'].max() if not equity_curve.empty else 0

        summary = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_commission': trades_df['commission'].sum() if not trades_df.empty else 0,
            'total_slippage': trades_df['slippage_cost'].sum() if not trades_df.empty else 0
        }

        return {
            'summary': summary,
            'equity_curve': equity_curve,
            'trades': trades_df,
            'exit_trades': exit_trades
        }
