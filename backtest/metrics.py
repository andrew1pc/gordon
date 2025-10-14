"""Performance metrics calculation for backtesting results."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest results.

    Provides risk-adjusted returns, drawdown analysis, trade statistics,
    and period-based performance breakdowns.

    Example:
        >>> metrics = PerformanceMetrics(backtest_results)
        >>> print(f"Sharpe Ratio: {metrics.sharpe_ratio():.2f}")
        >>> print(f"Max Drawdown: {metrics.max_drawdown():.1%}")
    """

    def __init__(self, results: Dict):
        """
        Initialize with backtest results.

        Args:
            results: Dict from BacktestEngine.get_results()
        """
        self.results = results
        self.summary = results['summary']
        self.equity_curve = results['equity_curve']
        self.trades = results['trades']
        self.exit_trades = results['exit_trades']

        # Calculate daily returns for various metrics
        if not self.equity_curve.empty:
            self.daily_returns = self.equity_curve['total_value'].pct_change().dropna()
        else:
            self.daily_returns = pd.Series()

    def total_return(self) -> float:
        """
        Calculate total return.

        Returns:
            Total return as decimal (0.25 = 25%)

        Example:
            >>> total_ret = metrics.total_return()
        """
        return self.summary['total_return']

    def cagr(self) -> float:
        """
        Calculate Compound Annual Growth Rate.

        Returns:
            CAGR as decimal

        Example:
            >>> annual_return = metrics.cagr()
        """
        if self.equity_curve.empty:
            return 0.0

        initial_value = self.summary['initial_capital']
        final_value = self.summary['final_value']

        # Calculate years
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25

        if years <= 0 or initial_value <= 0:
            return 0.0

        cagr = (final_value / initial_value) ** (1 / years) - 1
        return cagr

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio

        Example:
            >>> sharpe = metrics.sharpe_ratio()
            >>> print(f"Sharpe: {sharpe:.2f}")
        """
        if self.daily_returns.empty or len(self.daily_returns) < 2:
            return 0.0

        # Annualize returns and volatility
        mean_daily_return = self.daily_returns.mean()
        std_daily_return = self.daily_returns.std()

        if std_daily_return == 0:
            return 0.0

        # Annualize (252 trading days)
        annualized_return = mean_daily_return * 252
        annualized_vol = std_daily_return * np.sqrt(252)

        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        return sharpe

    def sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return).

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sortino ratio

        Example:
            >>> sortino = metrics.sortino_ratio()
        """
        if self.daily_returns.empty or len(self.daily_returns) < 2:
            return 0.0

        # Annualized return
        mean_daily_return = self.daily_returns.mean()
        annualized_return = mean_daily_return * 252

        # Downside deviation (only negative returns)
        negative_returns = self.daily_returns[self.daily_returns < 0]
        if len(negative_returns) == 0:
            return np.inf  # No downside = infinite Sortino

        downside_std = negative_returns.std()
        downside_deviation = downside_std * np.sqrt(252)

        if downside_deviation == 0:
            return np.inf

        sortino = (annualized_return - risk_free_rate) / downside_deviation
        return sortino

    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.

        Returns:
            Max drawdown as positive decimal (0.15 = 15% drawdown)

        Example:
            >>> mdd = metrics.max_drawdown()
            >>> print(f"Max Drawdown: {mdd:.1%}")
        """
        if self.equity_curve.empty:
            return 0.0

        return self.equity_curve['drawdown_pct'].max()

    def max_drawdown_duration(self) -> Tuple[int, Optional[datetime], Optional[datetime]]:
        """
        Calculate maximum drawdown duration.

        Returns:
            Tuple of (duration_days, start_date, end_date)

        Example:
            >>> days, start, end = metrics.max_drawdown_duration()
            >>> print(f"Longest drawdown: {days} days")
        """
        if self.equity_curve.empty:
            return 0, None, None

        equity = self.equity_curve['total_value']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []

        start = None
        for idx, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = idx
            elif not is_dd and start is not None:
                end = idx - 1
                duration = end - start + 1
                drawdown_periods.append({
                    'duration': duration,
                    'start': equity.index[start],
                    'end': equity.index[end]
                })
                start = None

        # Check if still in drawdown at end
        if start is not None:
            duration = len(equity) - start
            drawdown_periods.append({
                'duration': duration,
                'start': equity.index[start],
                'end': equity.index[-1]
            })

        if not drawdown_periods:
            return 0, None, None

        # Find longest
        longest = max(drawdown_periods, key=lambda x: x['duration'])
        return longest['duration'], longest['start'], longest['end']

    def win_rate(self) -> float:
        """
        Calculate win rate.

        Returns:
            Win rate as decimal (0.55 = 55%)

        Example:
            >>> wr = metrics.win_rate()
        """
        return self.summary['win_rate']

    def profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Returns:
            Profit factor (1.5 means wins are 1.5x losses)

        Example:
            >>> pf = metrics.profit_factor()
        """
        if self.exit_trades.empty:
            return 0.0

        gross_profit = self.exit_trades[self.exit_trades['pnl_dollars'] > 0]['pnl_dollars'].sum()
        gross_loss = abs(self.exit_trades[self.exit_trades['pnl_dollars'] < 0]['pnl_dollars'].sum())

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def average_win(self) -> Dict[str, float]:
        """
        Calculate average winning trade.

        Returns:
            Dict with 'dollars' and 'percent'

        Example:
            >>> avg_win = metrics.average_win()
            >>> print(f"Avg Win: ${avg_win['dollars']:,.0f}")
        """
        if self.exit_trades.empty:
            return {'dollars': 0.0, 'percent': 0.0}

        winners = self.exit_trades[self.exit_trades['pnl_dollars'] > 0]

        if len(winners) == 0:
            return {'dollars': 0.0, 'percent': 0.0}

        return {
            'dollars': winners['pnl_dollars'].mean(),
            'percent': winners['pnl_percent'].mean()
        }

    def average_loss(self) -> Dict[str, float]:
        """
        Calculate average losing trade.

        Returns:
            Dict with 'dollars' and 'percent'

        Example:
            >>> avg_loss = metrics.average_loss()
        """
        if self.exit_trades.empty:
            return {'dollars': 0.0, 'percent': 0.0}

        losers = self.exit_trades[self.exit_trades['pnl_dollars'] < 0]

        if len(losers) == 0:
            return {'dollars': 0.0, 'percent': 0.0}

        return {
            'dollars': losers['pnl_dollars'].mean(),
            'percent': losers['pnl_percent'].mean()
        }

    def expectancy(self) -> float:
        """
        Calculate expectancy (expected value per trade).

        Returns:
            Expected $ per trade

        Example:
            >>> exp = metrics.expectancy()
            >>> print(f"Expectancy: ${exp:.2f} per trade")
        """
        win_rate = self.win_rate()
        avg_win = self.average_win()['dollars']
        avg_loss = abs(self.average_loss()['dollars'])

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        return expectancy

    def total_trades(self) -> int:
        """Get total number of completed trades."""
        return self.summary['total_trades']

    def avg_holding_period(self) -> Dict[str, float]:
        """
        Calculate average holding period.

        Returns:
            Dict with 'all', 'winners', 'losers' in days

        Example:
            >>> holding = metrics.avg_holding_period()
            >>> print(f"Avg hold: {holding['all']:.1f} days")
        """
        if self.exit_trades.empty:
            return {'all': 0.0, 'winners': 0.0, 'losers': 0.0}

        all_trades = self.exit_trades['holding_period'].mean()

        winners = self.exit_trades[self.exit_trades['pnl_dollars'] > 0]
        losers = self.exit_trades[self.exit_trades['pnl_dollars'] < 0]

        return {
            'all': all_trades,
            'winners': winners['holding_period'].mean() if len(winners) > 0 else 0.0,
            'losers': losers['holding_period'].mean() if len(losers) > 0 else 0.0
        }

    def exposure_time(self) -> float:
        """
        Calculate portfolio exposure time.

        Returns:
            Fraction of time capital was deployed (0.75 = 75% of days)

        Example:
            >>> exposure = metrics.exposure_time()
        """
        if self.equity_curve.empty:
            return 0.0

        days_with_positions = len(self.equity_curve[self.equity_curve['position_count'] > 0])
        total_days = len(self.equity_curve)

        return days_with_positions / total_days if total_days > 0 else 0.0

    def recovery_factor(self) -> float:
        """
        Calculate recovery factor (return / max drawdown).

        Returns:
            Recovery factor (higher is better)

        Example:
            >>> rf = metrics.recovery_factor()
        """
        total_ret = self.total_return()
        max_dd = self.max_drawdown()

        if max_dd == 0:
            return np.inf if total_ret > 0 else 0.0

        return total_ret / max_dd

    def calculate_monthly_returns(self) -> pd.Series:
        """
        Calculate monthly returns.

        Returns:
            Series of monthly returns indexed by month

        Example:
            >>> monthly = metrics.calculate_monthly_returns()
            >>> print(monthly.head())
        """
        if self.equity_curve.empty:
            return pd.Series()

        # Resample to monthly
        monthly_values = self.equity_curve['total_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()

        return monthly_returns

    def calculate_yearly_returns(self) -> pd.Series:
        """
        Calculate yearly returns.

        Returns:
            Series of yearly returns indexed by year

        Example:
            >>> yearly = metrics.calculate_yearly_returns()
        """
        if self.equity_curve.empty:
            return pd.Series()

        # Resample to yearly
        yearly_values = self.equity_curve['total_value'].resample('Y').last()
        yearly_returns = yearly_values.pct_change().dropna()

        return yearly_returns

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive performance report.

        Args:
            output_file: Optional file path to save report

        Returns:
            Report as string

        Example:
            >>> report = metrics.generate_report('backtest_report.txt')
            >>> print(report)
        """
        lines = []
        lines.append("=" * 80)
        lines.append("BACKTEST PERFORMANCE REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary Statistics
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Initial Capital:        ${self.summary['initial_capital']:>15,.0f}")
        lines.append(f"Final Value:            ${self.summary['final_value']:>15,.0f}")
        lines.append(f"Total Return:           {self.total_return():>15.2%}")
        lines.append(f"CAGR:                   {self.cagr():>15.2%}")
        lines.append(f"Sharpe Ratio:           {self.sharpe_ratio():>15.2f}")
        lines.append(f"Sortino Ratio:          {self.sortino_ratio():>15.2f}")
        lines.append(f"Max Drawdown:           {self.max_drawdown():>15.2%}")
        lines.append("")

        # Trade Analysis
        lines.append("TRADE ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Total Trades:           {self.total_trades():>15,}")
        lines.append(f"Winning Trades:         {self.summary['winning_trades']:>15,}")
        lines.append(f"Win Rate:               {self.win_rate():>15.2%}")
        lines.append(f"Profit Factor:          {self.profit_factor():>15.2f}")
        lines.append(f"Expectancy:             ${self.expectancy():>15,.2f}")
        lines.append("")

        avg_win = self.average_win()
        avg_loss = self.average_loss()
        lines.append(f"Average Win:            ${avg_win['dollars']:>15,.2f} ({avg_win['percent']:>6.2%})")
        lines.append(f"Average Loss:           ${avg_loss['dollars']:>15,.2f} ({avg_loss['percent']:>6.2%})")
        lines.append("")

        # Largest trades
        if not self.exit_trades.empty:
            best_trade = self.exit_trades.loc[self.exit_trades['pnl_dollars'].idxmax()]
            worst_trade = self.exit_trades.loc[self.exit_trades['pnl_dollars'].idxmin()]
            lines.append(f"Best Trade:             {best_trade['ticker']} ${best_trade['pnl_dollars']:,.2f} ({best_trade['pnl_percent']:.2%})")
            lines.append(f"Worst Trade:            {worst_trade['ticker']} ${worst_trade['pnl_dollars']:,.2f} ({worst_trade['pnl_percent']:.2%})")
            lines.append("")

        # Holding Period
        holding = self.avg_holding_period()
        lines.append("HOLDING PERIOD ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Avg All Trades:         {holding['all']:>15.1f} days")
        lines.append(f"Avg Winners:            {holding['winners']:>15.1f} days")
        lines.append(f"Avg Losers:             {holding['losers']:>15.1f} days")
        lines.append("")

        # Risk Metrics
        dd_duration, dd_start, dd_end = self.max_drawdown_duration()
        lines.append("RISK METRICS")
        lines.append("-" * 80)
        lines.append(f"Max DD Duration:        {dd_duration:>15,} days")
        if dd_start:
            lines.append(f"Max DD Period:          {dd_start.date()} to {dd_end.date()}")
        lines.append(f"Recovery Factor:        {self.recovery_factor():>15.2f}")
        lines.append(f"Exposure Time:          {self.exposure_time():>15.2%}")
        lines.append("")

        # Costs
        lines.append("TRANSACTION COSTS")
        lines.append("-" * 80)
        lines.append(f"Total Commission:       ${self.summary['total_commission']:>15,.2f}")
        lines.append(f"Total Slippage:         ${self.summary['total_slippage']:>15,.2f}")
        lines.append(f"Total Costs:            ${self.summary['total_commission'] + self.summary['total_slippage']:>15,.2f}")
        lines.append("")

        # Monthly returns
        monthly = self.calculate_monthly_returns()
        if not monthly.empty:
            lines.append("MONTHLY PERFORMANCE")
            lines.append("-" * 80)
            lines.append(f"Best Month:             {monthly.max():>15.2%}")
            lines.append(f"Worst Month:            {monthly.min():>15.2%}")
            lines.append(f"Avg Month:              {monthly.mean():>15.2%}")
            lines.append(f"Positive Months:        {len(monthly[monthly > 0]):>15,} / {len(monthly)}")
            lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")

        return report
