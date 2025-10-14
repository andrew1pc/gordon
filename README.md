# Momentum Trading Strategy

A complete momentum-based algorithmic trading system with backtesting, optimization, and paper trading capabilities.

## Overview

This is a systematic momentum trading strategy that:
- Scans stock and crypto universes for high-momentum candidates
- Generates entry signals based on technical indicators and momentum metrics
- Manages positions with dynamic stops and profit targets
- Enforces comprehensive risk management rules
- Provides backtesting and performance analysis tools
- Supports paper trading for validation

## Features

### Strategy Components

- **Asset Scanner**: Identifies high-momentum trading candidates from S&P 500 stocks and top cryptocurrencies
- **Technical Indicators**: RSI, MACD, Moving Averages, ATR, Volume analysis
- **Momentum Metrics**: Rate of Change, Trend Strength, Volume Surge detection, Composite Momentum Score
- **Signal Generation**: Multi-condition entry signals, multiple exit types (stops, targets, trailing stops, time-based)
- **Risk Management**: Position sizing (1% risk per trade), portfolio limits (25% max exposure), sector limits (15% max), daily loss circuit breaker (3%)

### Tools & Infrastructure

- **Backtesting Engine**: Walk-forward methodology, realistic slippage/commissions, complete trade history
- **Performance Analytics**: 20+ metrics including Sharpe, Sortino, profit factor, drawdown analysis
- **Parameter Optimization**: Grid search, random search, walk-forward optimization
- **Visualization Suite**: Equity curves, drawdown charts, trade analysis, monthly returns heatmaps
- **Paper Trading**: Real-time simulation with full state persistence
- **CLI Interface**: Easy command-line operation

## Installation

### Prerequisites

- Python 3.8+
- Tiingo API key (free tier available at https://www.tiingo.com/)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/andrew1pc/gordon.git
cd gordon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API key:
```bash
export TIINGO_API_KEY='your_api_key_here'
```

Or create a `.env` file:
```
TIINGO_API_KEY=your_api_key_here
```

4. Verify installation:
```bash
python -c "from config.api_config import TiingoConfig; print('Setup successful!')"
```

## Quick Start

### Run a Backtest

```bash
python run_backtest.py
```

This will:
- Run a 6-month backtest (configurable)
- Generate performance report
- Create visualizations in `output/` directory
- Export trade data to CSV

### Initialize Paper Trading

```bash
python main.py paper --init
```

### Run Daily Paper Trading Cycle

```bash
python main.py paper --run
```

### View Current Portfolio Status

```bash
python main.py paper --report
```

### Scan for Current Candidates

```bash
python main.py scan
```

## Project Structure

```
gordon/
├── backtest/           # Backtesting engine and analytics
│   ├── engine.py       # Main backtest engine
│   ├── metrics.py      # Performance metrics
│   ├── optimizer.py    # Parameter optimization
│   └── visualizations.py
├── config/             # Configuration files
│   ├── api_config.py   # API configuration
│   ├── strategy_config.py  # Strategy parameters
│   └── strategy_config.yaml
├── data/               # Data fetching and validation
│   ├── fetcher.py      # Tiingo API client
│   └── validator.py    # Data validation
├── indicators/         # Technical indicators
│   ├── technical.py    # Standard indicators (SMA, RSI, MACD, etc.)
│   └── momentum.py     # Momentum-specific metrics
├── strategy/           # Core strategy logic
│   ├── scanner.py      # Asset universe scanning
│   ├── signals.py      # Entry/exit signal generation
│   └── risk_manager.py # Position sizing and risk limits
├── tests/              # Test suite
├── logs/               # Log files
├── snapshots/          # Paper trading state files
├── reports/            # Daily reports
├── output/             # Charts and analysis
├── main.py             # Main orchestration and CLI
├── run_backtest.py     # Backtest runner script
└── README.md           # This file
```

## Configuration

Edit `config/strategy_config.yaml` to customize:

### Scanner Parameters
- `momentum_threshold`: Minimum momentum score (default: 70)
- `max_candidates`: Maximum positions to scan (default: 15)
- `rescan_interval_days`: Days between universe rescans (default: 5)

### Entry Signals
- `rsi_min/rsi_max`: RSI range (default: 50-70)
- `require_volume_surge`: Require volume confirmation (default: true)

### Exit Signals
- `profit_target_stock`: Profit target for stocks (default: 20%)
- `stop_loss_stock`: Stop loss for stocks (default: 8%)
- `trailing_stop_distance`: Trailing stop distance (default: 8%)
- `max_holding_days`: Maximum hold period (default: 30 days)

### Risk Management
- `initial_capital`: Starting capital (default: $100,000)
- `risk_per_trade`: Risk per position (default: 1%)
- `max_positions`: Maximum concurrent positions (default: 8)
- `max_portfolio_exposure`: Maximum capital deployed (default: 25%)
- `max_daily_loss`: Circuit breaker (default: 3%)

## Strategy Logic

### Entry Criteria (ALL must be met)

1. **Breakout**: Price breaks above 20-day high
2. **Volume Surge**: Volume >= 1.5x average
3. **MACD Confirmation**: MACD histogram > 0
4. **Trend Filter**: Price above 50-day MA
5. **Momentum Trend**: 50-day MA trending up
6. **Momentum Score**: Composite score >= 70

### Exit Criteria (Priority order)

1. **Stop Loss**: Price hits initial or trailing stop
2. **Profit Target**: Price reaches target (15-25% for stocks, 25-40% for crypto)
3. **Trailing Stop**: Activated at 10% profit, trails 8% below peak
4. **Momentum Failure**: Momentum score drops below threshold
5. **Time Exit**: Maximum holding period exceeded (30 days)

### Position Sizing

- Risk: 1% of account per trade
- Position size calculated to risk exactly 1% if stop is hit
- Minimum position: 1% of account
- Maximum position: 5% of account
- Crypto positions reduced to 70% of calculated size (higher volatility)

### Risk Limits

- Maximum 8 concurrent positions
- Maximum 25% of capital deployed
- Maximum 15% in any single sector
- Circuit breaker: stop trading if daily loss exceeds 3%

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total return, CAGR
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio
- **Drawdown**: Maximum drawdown, recovery time
- **Trade Statistics**: Win rate, profit factor, expectancy
- **Efficiency**: Average holding period, exposure time

### Target Performance (from backtesting)

- Sharpe Ratio: > 1.0
- Max Drawdown: < 20%
- Win Rate: > 45%
- Profit Factor: > 1.5

## Risk Disclaimer

**This is for educational and research purposes only.**

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Only trade with capital you can afford to lose
- Thoroughly test in paper trading before considering live use
- Consult with financial professionals before trading

The authors are not responsible for any losses incurred from using this software.

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- GitHub Issues: https://github.com/andrew1pc/gordon/issues

---

**Happy Trading! 📈**

Remember: Discipline and risk management are more important than finding perfect entry signals.
