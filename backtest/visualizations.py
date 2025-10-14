"""Visualization suite for backtest results."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, Optional
import os


def plot_equity_curve(
    results: Dict,
    benchmark: Optional[pd.Series] = None,
    output_file: str = 'output/equity_curve.png'
) -> None:
    """
    Plot portfolio equity curve with entry/exit markers.

    Args:
        results: Backtest results dict
        benchmark: Optional benchmark series for comparison
        output_file: Path to save plot

    Example:
        >>> plot_equity_curve(results, output_file='equity.png')
    """
    equity_curve = results['equity_curve']
    trades = results['trades']

    if equity_curve.empty:
        print("No equity curve data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot equity curve
    ax1.plot(equity_curve.index, equity_curve['total_value'],
             label='Strategy', linewidth=2, color='#2E86AB')

    # Plot benchmark if provided
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark.values,
                 label='Benchmark', linewidth=1.5, color='#A23B72', linestyle='--', alpha=0.7)

    # Mark entry and exit points
    if not trades.empty:
        entries = trades[trades['trade_type'] == 'entry']
        exits = trades[trades['trade_type'] == 'exit']

        if not entries.empty:
            # Get portfolio value at entry dates
            for _, trade in entries.iterrows():
                if trade['date'] in equity_curve.index:
                    value = equity_curve.loc[trade['date'], 'total_value']
                    ax1.scatter(trade['date'], value, color='green', marker='^',
                               s=100, alpha=0.6, zorder=5)

        if not exits.empty:
            for _, trade in exits.iterrows():
                if trade['date'] in equity_curve.index:
                    value = equity_curve.loc[trade['date'], 'total_value']
                    color = 'darkgreen' if trade['pnl_dollars'] > 0 else 'darkred'
                    ax1.scatter(trade['date'], value, color=color, marker='v',
                               s=100, alpha=0.6, zorder=5)

    # Formatting
    ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Plot drawdown
    ax2.fill_between(equity_curve.index, 0, -equity_curve['drawdown_pct'] * 100,
                      color='#F18F01', alpha=0.5)
    ax2.plot(equity_curve.index, -equity_curve['drawdown_pct'] * 100,
             color='#C73E1D', linewidth=1.5)

    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Equity curve saved to {output_file}")
    plt.close()


def plot_trade_analysis(
    results: Dict,
    output_file: str = 'output/trade_analysis.png'
) -> None:
    """
    Plot comprehensive trade analysis.

    Args:
        results: Backtest results dict
        output_file: Path to save plot

    Example:
        >>> plot_trade_analysis(results)
    """
    exit_trades = results['exit_trades']

    if exit_trades.empty:
        print("No trades to analyze")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Trade Returns Distribution
    ax1 = axes[0, 0]
    returns_pct = exit_trades['pnl_percent'] * 100
    ax1.hist(returns_pct, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(returns_pct.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns_pct.mean():.1f}%')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_title('Distribution of Trade Returns', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Return (%)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Holding Period vs Return
    ax2 = axes[0, 1]
    colors = ['darkgreen' if r > 0 else 'darkred' for r in exit_trades['pnl_dollars']]
    ax2.scatter(exit_trades['holding_period'], exit_trades['pnl_percent'] * 100,
                c=colors, alpha=0.6, s=50)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_title('Holding Period vs Return', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Holding Period (days)', fontsize=10)
    ax2.set_ylabel('Return (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Win/Loss Size Distribution
    ax3 = axes[1, 0]
    winners = exit_trades[exit_trades['pnl_dollars'] > 0]['pnl_dollars']
    losers = exit_trades[exit_trades['pnl_dollars'] < 0]['pnl_dollars']

    ax3.hist(winners, bins=20, color='darkgreen', alpha=0.6, label=f'Winners (n={len(winners)})')
    ax3.hist(losers, bins=20, color='darkred', alpha=0.6, label=f'Losers (n={len(losers)})')
    ax3.set_title('Win/Loss Size Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('P&L ($)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Cumulative P&L
    ax4 = axes[1, 1]
    cumulative_pnl = exit_trades.sort_values('date')['pnl_dollars'].cumsum()
    ax4.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='#2E86AB')
    ax4.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                      alpha=0.3, color='#2E86AB')
    ax4.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax4.set_title('Cumulative P&L by Trade', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trade Number', fontsize=10)
    ax4.set_ylabel('Cumulative P&L ($)', fontsize=10)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Trade analysis saved to {output_file}")
    plt.close()


def plot_monthly_returns_heatmap(
    results: Dict,
    output_file: str = 'output/monthly_returns.png'
) -> None:
    """
    Plot heatmap of monthly returns.

    Args:
        results: Backtest results dict
        output_file: Path to save plot

    Example:
        >>> plot_monthly_returns_heatmap(results)
    """
    equity_curve = results['equity_curve']

    if equity_curve.empty:
        print("No data for monthly returns")
        return

    # Calculate monthly returns
    monthly_values = equity_curve['total_value'].resample('M').last()
    monthly_returns = monthly_values.pct_change().dropna() * 100

    if monthly_returns.empty:
        print("Insufficient data for monthly returns")
        return

    # Create pivot table for heatmap
    monthly_returns_df = pd.DataFrame({
        'return': monthly_returns.values,
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month
    })

    pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))

    # Create color map
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticklabels(pivot.index)

    # Add values in cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if not pd.isna(pivot.iloc[i, j]):
                text = ax.text(j, i, f'{pivot.iloc[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontsize=9)

    ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return (%)', rotation=270, labelpad=20)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Monthly returns heatmap saved to {output_file}")
    plt.close()


def plot_rolling_metrics(
    results: Dict,
    window: int = 20,
    output_file: str = 'output/rolling_metrics.png'
) -> None:
    """
    Plot rolling performance metrics.

    Args:
        results: Backtest results dict
        window: Rolling window in trades
        output_file: Path to save plot

    Example:
        >>> plot_rolling_metrics(results, window=20)
    """
    exit_trades = results['exit_trades']

    if exit_trades.empty or len(exit_trades) < window:
        print(f"Not enough trades for rolling metrics (need at least {window})")
        return

    exit_trades = exit_trades.sort_values('date').reset_index(drop=True)

    # Calculate rolling metrics
    rolling_win_rate = []
    rolling_profit_factor = []
    rolling_avg_return = []

    for i in range(window, len(exit_trades) + 1):
        window_trades = exit_trades.iloc[i-window:i]

        # Win rate
        win_rate = len(window_trades[window_trades['pnl_dollars'] > 0]) / len(window_trades)
        rolling_win_rate.append(win_rate)

        # Profit factor
        gross_profit = window_trades[window_trades['pnl_dollars'] > 0]['pnl_dollars'].sum()
        gross_loss = abs(window_trades[window_trades['pnl_dollars'] < 0]['pnl_dollars'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        rolling_profit_factor.append(pf)

        # Average return
        avg_ret = window_trades['pnl_percent'].mean()
        rolling_avg_return.append(avg_ret)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    trade_range = range(window, len(exit_trades) + 1)

    # Rolling Win Rate
    ax1 = axes[0]
    ax1.plot(trade_range, rolling_win_rate, linewidth=2, color='#2E86AB')
    ax1.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50%')
    ax1.fill_between(trade_range, 0.5, rolling_win_rate,
                      where=np.array(rolling_win_rate) >= 0.5,
                      alpha=0.3, color='green', interpolate=True)
    ax1.fill_between(trade_range, 0.5, rolling_win_rate,
                      where=np.array(rolling_win_rate) < 0.5,
                      alpha=0.3, color='red', interpolate=True)
    ax1.set_title(f'Rolling Win Rate ({window}-trade window)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Win Rate', fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Rolling Profit Factor
    ax2 = axes[1]
    ax2.plot(trade_range, rolling_profit_factor, linewidth=2, color='#F18F01')
    ax2.axhline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axhline(1.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target: 1.5')
    ax2.fill_between(trade_range, 1.0, rolling_profit_factor,
                      where=np.array(rolling_profit_factor) >= 1.0,
                      alpha=0.3, color='green', interpolate=True)
    ax2.fill_between(trade_range, 1.0, rolling_profit_factor,
                      where=np.array(rolling_profit_factor) < 1.0,
                      alpha=0.3, color='red', interpolate=True)
    ax2.set_title(f'Rolling Profit Factor ({window}-trade window)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Profit Factor', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(bottom=0)

    # Rolling Average Return
    ax3 = axes[2]
    rolling_avg_return_pct = [r * 100 for r in rolling_avg_return]
    ax3.plot(trade_range, rolling_avg_return_pct, linewidth=2, color='#A23B72')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.fill_between(trade_range, 0, rolling_avg_return_pct,
                      where=np.array(rolling_avg_return_pct) >= 0,
                      alpha=0.3, color='green', interpolate=True)
    ax3.fill_between(trade_range, 0, rolling_avg_return_pct,
                      where=np.array(rolling_avg_return_pct) < 0,
                      alpha=0.3, color='red', interpolate=True)
    ax3.set_title(f'Rolling Average Return ({window}-trade window)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trade Number', fontsize=10)
    ax3.set_ylabel('Avg Return (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Rolling metrics saved to {output_file}")
    plt.close()


def generate_all_plots(results: Dict, output_dir: str = 'output') -> None:
    """
    Generate all visualization plots.

    Args:
        results: Backtest results dict
        output_dir: Directory to save plots

    Example:
        >>> generate_all_plots(results, 'backtest_output')
    """
    print("Generating visualizations...")

    plot_equity_curve(results, output_file=f'{output_dir}/equity_curve.png')
    plot_trade_analysis(results, output_file=f'{output_dir}/trade_analysis.png')
    plot_monthly_returns_heatmap(results, output_file=f'{output_dir}/monthly_returns.png')

    if not results['exit_trades'].empty and len(results['exit_trades']) >= 20:
        plot_rolling_metrics(results, window=20, output_file=f'{output_dir}/rolling_metrics.png')

    print(f"All plots saved to {output_dir}/")
