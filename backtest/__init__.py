"""Backtesting module for strategy evaluation."""

from backtest.engine import BacktestEngine, Trade, DailyState
from backtest.metrics import PerformanceMetrics
from backtest.visualizations import (
    plot_equity_curve,
    plot_trade_analysis,
    plot_monthly_returns_heatmap,
    plot_rolling_metrics,
    generate_all_plots
)

__all__ = [
    'BacktestEngine',
    'Trade',
    'DailyState',
    'PerformanceMetrics',
    'plot_equity_curve',
    'plot_trade_analysis',
    'plot_monthly_returns_heatmap',
    'plot_rolling_metrics',
    'generate_all_plots'
]
