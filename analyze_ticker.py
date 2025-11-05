#!/usr/bin/env python3
"""
Ultra-minimal single-ticker analyzer for entry/exit signals.
Uses only 1 API call per run - well within Tiingo free tier limits.

Usage:
    python analyze_ticker.py AAPL
    python analyze_ticker.py btcusd
"""

import sys
import os
from datetime import datetime, timedelta
from config.api_config import TiingoConfig
from data.fetcher import TiingoClient
from indicators.technical import TechnicalIndicators
from indicators.momentum import MomentumMetrics
from strategy.signals import SignalGenerator


def analyze_ticker(ticker: str):
    """
    Analyze a single ticker for entry/exit signals.

    Args:
        ticker: Stock symbol (e.g., 'AAPL') or crypto (e.g., 'btcusd')
    """
    print("=" * 80)
    print(f"ANALYZING: {ticker.upper()}")
    print("=" * 80)
    print()

    # Initialize
    config = TiingoConfig()
    client = TiingoClient(config)
    technical = TechnicalIndicators()
    momentum = MomentumMetrics()
    signal_gen = SignalGenerator()

    # Determine asset type
    asset_type = 'crypto' if ticker.lower().endswith('usd') else 'stock'

    # Date range (90 days for indicators)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Asset Type: {asset_type.upper()}")
    print(f"Date Range: {start_str} to {end_str}")
    print(f"Fetching data... (1 API call)")
    print()

    # Fetch data (SINGLE API CALL)
    try:
        if asset_type == 'stock':
            df = client.get_stock_prices(ticker, start_str, end_str)
        else:
            df = client.get_crypto_prices(ticker, start_str, end_str)

        if df is None or len(df) < 60:
            print(f"✗ Insufficient data for {ticker}")
            print(f"  Please check ticker symbol or try again later if rate limited.")
            return

        print(f"✓ Fetched {len(df)} days of data")
        print()

    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return

    # Add indicators
    print("Calculating technical indicators...")
    df = technical.add_all_indicators(df)
    df = momentum.add_all_momentum_metrics(df)
    print("✓ Indicators calculated")
    print()

    # Get latest data
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    current_idx = len(df) - 1

    # Current price info
    current_price = latest['close']
    price_change_1d = ((latest['close'] - prev['close']) / prev['close']) * 100

    # Check entry signal
    entry_signal = signal_gen.check_entry_signals(df, current_idx, ticker)

    # Calculate entry price and stops
    if entry_signal:
        entry_price = signal_gen.calculate_entry_price(df, current_idx, asset_type)
        stop_loss = signal_gen.calculate_initial_stop(entry_price, df, current_idx, asset_type)
        profit_target = signal_gen.calculate_profit_target(entry_price, df, current_idx, asset_type)
    else:
        # Calculate hypothetical levels
        entry_price = current_price
        stop_loss = signal_gen.calculate_initial_stop(current_price, df, current_idx, asset_type)
        profit_target = signal_gen.calculate_profit_target(current_price, df, current_idx, asset_type)

    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"TRADE ANALYSIS: {ticker.upper()}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Current Status
    report_lines.append("CURRENT STATUS")
    report_lines.append("-" * 80)
    report_lines.append(f"Current Price:           ${current_price:.2f}")
    report_lines.append(f"1-Day Change:            {price_change_1d:+.2f}%")
    report_lines.append(f"Asset Type:              {asset_type.upper()}")
    report_lines.append("")

    # Technical Indicators
    report_lines.append("TECHNICAL INDICATORS")
    report_lines.append("-" * 80)
    report_lines.append(f"RSI (14):                {latest.get('rsi', 0):.1f}")
    report_lines.append(f"MACD:                    {latest.get('macd', 0):.3f}")
    report_lines.append(f"MACD Signal:             {latest.get('macd_signal', 0):.3f}")
    report_lines.append(f"MACD Histogram:          {latest.get('macd_histogram', 0):.3f} {'✓ Bullish' if latest.get('macd_histogram', 0) > 0 else '✗ Bearish'}")
    report_lines.append("")
    report_lines.append(f"50-day SMA:              ${latest.get('sma_50', 0):.2f}")
    report_lines.append(f"200-day SMA:             ${latest.get('sma_200', 0):.2f}")
    report_lines.append(f"Price vs 50-day MA:      {'Above ✓' if current_price > latest.get('sma_50', 0) else 'Below ✗'}")
    report_lines.append(f"Price vs 200-day MA:     {'Above ✓' if current_price > latest.get('sma_200', 0) else 'Below ✗'}")
    report_lines.append("")
    report_lines.append(f"ATR (14):                ${latest.get('atr', 0):.2f}")
    report_lines.append(f"Volume Ratio:            {latest.get('volume_ratio', 1.0):.2f}x {'✓ Surge' if latest.get('volume_ratio', 1.0) > 1.5 else ''}")
    report_lines.append("")

    # Momentum Metrics
    report_lines.append("MOMENTUM METRICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Momentum Score:          {latest.get('momentum_score', 0):.1f}/100")
    report_lines.append(f"20-Day ROC:              {latest.get('roc_20', 0):.1f}%")
    report_lines.append(f"50-Day ROC:              {latest.get('roc_50', 0):.1f}%")
    report_lines.append(f"Trend Strength:          {latest.get('trend_strength', 0):.2f}")
    report_lines.append(f"Trend Direction:         {latest.get('trend_direction', 'neutral').upper()}")
    report_lines.append(f"Volume Surge Detected:   {'Yes ✓' if latest.get('volume_surge', False) else 'No'}")
    report_lines.append("")

    # Entry Signal Analysis
    report_lines.append("ENTRY SIGNAL ANALYSIS")
    report_lines.append("-" * 80)

    if entry_signal:
        report_lines.append("🟢 ENTRY SIGNAL DETECTED!")
        report_lines.append("")
        report_lines.append("Entry conditions met:")

        # Check which conditions are met
        conditions = []

        # Breakout
        high_20 = df['high'].rolling(20).max().iloc[-2]
        if current_price > high_20:
            conditions.append("  ✓ Breakout: Price above 20-day high")

        # MACD
        if latest.get('macd_histogram', 0) > 0:
            conditions.append("  ✓ MACD: Histogram positive (bullish)")

        # Trend
        if current_price > latest.get('sma_50', 0):
            conditions.append("  ✓ Trend: Price above 50-day MA")

        # Volume
        if latest.get('volume_surge', False):
            conditions.append("  ✓ Volume: Surge detected")

        # Momentum
        if latest.get('momentum_score', 0) >= 70:
            conditions.append(f"  ✓ Momentum: Score {latest.get('momentum_score', 0):.0f} >= 70")

        for condition in conditions:
            report_lines.append(condition)

        report_lines.append("")
        report_lines.append("TRADE SETUP:")
        report_lines.append(f"  Entry Price:           ${entry_price:.2f}")
        report_lines.append(f"  Stop Loss:             ${stop_loss:.2f} ({((stop_loss - entry_price) / entry_price * 100):.1f}%)")
        report_lines.append(f"  Profit Target:         ${profit_target:.2f} ({((profit_target - entry_price) / entry_price * 100):.1f}%)")
        report_lines.append(f"  Risk/Reward Ratio:     1:{abs((profit_target - entry_price) / (entry_price - stop_loss)):.2f}")

    else:
        report_lines.append("🔴 NO ENTRY SIGNAL")
        report_lines.append("")
        report_lines.append("Entry conditions NOT met:")

        # Check which conditions are missing
        missing = []

        # Breakout
        high_20 = df['high'].rolling(20).max().iloc[-2]
        if current_price <= high_20:
            missing.append(f"  ✗ Breakout: Price ${current_price:.2f} not above 20-day high ${high_20:.2f}")

        # MACD
        if latest.get('macd_histogram', 0) <= 0:
            missing.append(f"  ✗ MACD: Histogram {latest.get('macd_histogram', 0):.3f} not positive")

        # Trend
        if current_price <= latest.get('sma_50', 0):
            missing.append(f"  ✗ Trend: Price not above 50-day MA ${latest.get('sma_50', 0):.2f}")

        # Volume
        if not latest.get('volume_surge', False):
            missing.append(f"  ✗ Volume: No surge (ratio {latest.get('volume_ratio', 1.0):.2f}x, need >1.5x)")

        # Momentum
        if latest.get('momentum_score', 0) < 70:
            missing.append(f"  ✗ Momentum: Score {latest.get('momentum_score', 0):.0f} < 70 threshold")

        for miss in missing:
            report_lines.append(miss)

        report_lines.append("")
        report_lines.append("IF YOU ENTER AT CURRENT PRICE:")
        report_lines.append(f"  Entry Price:           ${entry_price:.2f}")
        report_lines.append(f"  Stop Loss:             ${stop_loss:.2f} ({((stop_loss - entry_price) / entry_price * 100):.1f}%)")
        report_lines.append(f"  Profit Target:         ${profit_target:.2f} ({((profit_target - entry_price) / entry_price * 100):.1f}%)")
        report_lines.append(f"  Risk/Reward Ratio:     1:{abs((profit_target - entry_price) / (entry_price - stop_loss)):.2f}")

    report_lines.append("")

    # Position sizing
    report_lines.append("POSITION SIZING (1% Risk Model)")
    report_lines.append("-" * 80)
    account_size = 100000
    risk_amount = account_size * 0.01
    risk_per_share = abs(entry_price - stop_loss)
    position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
    position_value = position_size * entry_price

    # For crypto, reduce position size by 30%
    if asset_type == 'crypto':
        position_size *= 0.7
        position_value = position_size * entry_price
        report_lines.append(f"Account Size:            ${account_size:,.0f}")
        report_lines.append(f"Risk Amount (1%):        ${risk_amount:,.0f}")
        report_lines.append(f"Risk Per Unit:           ${risk_per_share:.2f}")
        report_lines.append(f"Position Size:           {position_size:.2f} units (reduced 30% for crypto)")
        report_lines.append(f"Position Value:          ${position_value:,.0f}")
    else:
        report_lines.append(f"Account Size:            ${account_size:,.0f}")
        report_lines.append(f"Risk Amount (1%):        ${risk_amount:,.0f}")
        report_lines.append(f"Risk Per Share:          ${risk_per_share:.2f}")
        report_lines.append(f"Position Size:           {position_size:.0f} shares")
        report_lines.append(f"Position Value:          ${position_value:,.0f}")

    report_lines.append(f"% of Account:            {(position_value / account_size * 100):.1f}%")
    report_lines.append("")

    # Key Levels
    report_lines.append("KEY PRICE LEVELS")
    report_lines.append("-" * 80)
    report_lines.append(f"Current Price:           ${current_price:.2f}")
    report_lines.append(f"Resistance (20d high):   ${df['high'].rolling(20).max().iloc[-1]:.2f}")
    report_lines.append(f"Support (20d low):       ${df['low'].rolling(20).min().iloc[-1]:.2f}")
    report_lines.append(f"50-day MA:               ${latest.get('sma_50', 0):.2f}")
    report_lines.append(f"200-day MA:              ${latest.get('sma_200', 0):.2f}")
    report_lines.append("")

    # Recommendation
    report_lines.append("RECOMMENDATION")
    report_lines.append("-" * 80)

    if entry_signal:
        if latest.get('rsi', 50) > 70:
            recommendation = "⚠️  SIGNAL PRESENT but RSI OVERBOUGHT - Consider waiting for pullback"
        else:
            recommendation = "✅ CONSIDER ENTRY - All signal conditions met"
    else:
        if latest.get('momentum_score', 0) >= 60:
            recommendation = "👀 WATCH - Momentum building but entry conditions not yet met"
        elif latest.get('trend_direction', 'neutral') == 'down':
            recommendation = "❌ AVOID - Downtrend in place"
        else:
            recommendation = "⏸️  WAIT - Insufficient momentum/setup"

    report_lines.append(recommendation)
    report_lines.append("")

    # Exit strategy for existing positions
    report_lines.append("EXIT STRATEGY (If Already Holding)")
    report_lines.append("-" * 80)
    report_lines.append(f"Initial Stop:            ${stop_loss:.2f}")
    report_lines.append(f"Profit Target:           ${profit_target:.2f}")
    report_lines.append(f"Trailing Stop:           Activated at +10% profit, trails 8% below peak")
    report_lines.append(f"Time Stop:               30 days maximum hold")
    report_lines.append(f"Momentum Exit:           If momentum score drops below 50")
    report_lines.append("")

    # Notes
    report_lines.append("=" * 80)
    report_lines.append("NOTES")
    report_lines.append("-" * 80)
    report_lines.append(f"• This analysis uses {len(df)} days of historical data")
    report_lines.append(f"• Entry signals require ALL conditions to be met simultaneously")
    report_lines.append(f"• Stop loss based on ATR and recent swing lows")
    report_lines.append(f"• Position sizing risks exactly 1% of account per trade")
    if asset_type == 'crypto':
        report_lines.append(f"• Crypto positions sized at 70% of normal due to higher volatility")
    report_lines.append(f"• This is for educational purposes only - not financial advice")
    report_lines.append("=" * 80)

    # Print report
    report = "\n".join(report_lines)
    print(report)

    # Save to file
    output_file = f"{ticker.lower()}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to: {output_file}")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ticker.py <TICKER>")
        print()
        print("Examples:")
        print("  python analyze_ticker.py AAPL")
        print("  python analyze_ticker.py NVDA")
        print("  python analyze_ticker.py btcusd")
        print("  python analyze_ticker.py ethusd")
        sys.exit(1)

    ticker = sys.argv[1]
    analyze_ticker(ticker)


if __name__ == '__main__':
    main()
