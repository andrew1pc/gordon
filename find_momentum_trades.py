#!/usr/bin/env python3
"""
Script to identify top 5 current momentum trading opportunities.
Handles rate limits by processing fewer assets with delays.
"""

import os
import time
from datetime import datetime, timedelta
from config.api_config import TiingoConfig
from data.fetcher import TiingoClient
from indicators.technical import TechnicalIndicators
from indicators.momentum import MomentumMetrics


def main():
    print("=" * 80)
    print("MOMENTUM TRADE SCANNER")
    print("=" * 80)
    print()

    # Initialize
    config = TiingoConfig()
    client = TiingoClient(config)
    technical = TechnicalIndicators()
    momentum = MomentumMetrics()

    # Define smaller universe to avoid rate limits
    candidates = {
        'stocks': ['AAPL', 'NVDA', 'TSLA', 'META', 'AMD'],
        'crypto': ['btcusd', 'ethusd', 'solusd']
    }

    # Date range (90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Scanning period: {start_str} to {end_str}")
    print(f"Universe: {len(candidates['stocks'])} stocks, {len(candidates['crypto'])} cryptos")
    print()

    results = []

    # Process stocks
    print("Fetching stock data...")
    for i, ticker in enumerate(candidates['stocks']):
        try:
            print(f"  [{i+1}/{len(candidates['stocks'])}] {ticker}...", end=' ')
            df = client.get_stock_prices(ticker, start_str, end_str)

            if df is not None and len(df) >= 60:
                # Add indicators
                df = technical.add_all_indicators(df)
                df = momentum.add_all_momentum_metrics(df)

                # Get latest values
                latest = df.iloc[-1]
                prev = df.iloc[-2]

                results.append({
                    'ticker': ticker,
                    'asset_type': 'stock',
                    'current_price': latest['close'],
                    'momentum_score': latest.get('momentum_score', 0),
                    'rsi': latest.get('rsi', 0),
                    'roc_20': latest.get('roc_20', 0),
                    'macd_histogram': latest.get('macd_histogram', 0),
                    'trend_strength': latest.get('trend_strength', 0),
                    'volume_ratio': latest.get('volume_ratio', 1.0),
                    'price_change_1d': ((latest['close'] - prev['close']) / prev['close']) * 100,
                    'above_sma50': latest['close'] > latest.get('sma_50', latest['close']),
                    'above_sma200': latest['close'] > latest.get('sma_200', latest['close'])
                })
                print("✓")
            else:
                print("insufficient data")

            time.sleep(2)  # Rate limit protection

        except Exception as e:
            print(f"error: {e}")
            time.sleep(3)

    # Process crypto
    print("\nFetching crypto data...")
    for i, ticker in enumerate(candidates['crypto']):
        try:
            print(f"  [{i+1}/{len(candidates['crypto'])}] {ticker}...", end=' ')
            df = client.get_crypto_prices(ticker, start_str, end_str)

            if df is not None and len(df) >= 60:
                # Add indicators
                df = technical.add_all_indicators(df)
                df = momentum.add_all_momentum_metrics(df)

                # Get latest values
                latest = df.iloc[-1]
                prev = df.iloc[-2]

                results.append({
                    'ticker': ticker,
                    'asset_type': 'crypto',
                    'current_price': latest['close'],
                    'momentum_score': latest.get('momentum_score', 0),
                    'rsi': latest.get('rsi', 0),
                    'roc_20': latest.get('roc_20', 0),
                    'macd_histogram': latest.get('macd_histogram', 0),
                    'trend_strength': latest.get('trend_strength', 0),
                    'volume_ratio': latest.get('volume_ratio', 1.0),
                    'price_change_1d': ((latest['close'] - prev['close']) / prev['close']) * 100,
                    'above_sma50': latest['close'] > latest.get('sma_50', latest['close']),
                    'above_sma200': latest['close'] > latest.get('sma_200', latest['close'])
                })
                print("✓")
            else:
                print("insufficient data")

            time.sleep(2)  # Rate limit protection

        except Exception as e:
            print(f"error: {e}")
            time.sleep(3)

    print(f"\n{'='*80}")
    print(f"Successfully analyzed {len(results)} assets")
    print(f"{'='*80}\n")

    if len(results) == 0:
        print("No data available. Please check your API connection.")
        return

    # Sort by momentum score
    results_sorted = sorted(results, key=lambda x: x['momentum_score'], reverse=True)

    # Get top 5
    top_5 = results_sorted[:5]

    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TOP 5 MOMENTUM TRADING OPPORTUNITIES")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")

    for i, trade in enumerate(top_5, 1):
        report_lines.append(f"{i}. {trade['ticker'].upper()} ({trade['asset_type'].upper()})")
        report_lines.append("-" * 80)
        report_lines.append(f"Current Price: ${trade['current_price']:.2f}")
        report_lines.append(f"Momentum Score: {trade['momentum_score']:.1f}/100")
        report_lines.append("")

        report_lines.append("TECHNICAL INDICATORS:")
        report_lines.append(f"  • RSI (14): {trade['rsi']:.1f}")
        report_lines.append(f"  • 20-Day Rate of Change: {trade['roc_20']:.1f}%")
        report_lines.append(f"  • MACD Histogram: {trade['macd_histogram']:.3f}")
        report_lines.append(f"  • Trend Strength: {trade['trend_strength']:.2f}")
        report_lines.append(f"  • Volume Ratio: {trade['volume_ratio']:.2f}x average")
        report_lines.append(f"  • 1-Day Price Change: {trade['price_change_1d']:+.2f}%")
        report_lines.append("")

        report_lines.append("POSITION RELATIVE TO MOVING AVERAGES:")
        report_lines.append(f"  • Above 50-day MA: {'Yes ✓' if trade['above_sma50'] else 'No ✗'}")
        report_lines.append(f"  • Above 200-day MA: {'Yes ✓' if trade['above_sma200'] else 'No ✗'}")
        report_lines.append("")

        report_lines.append("TRADE JUSTIFICATION:")

        # Build justification
        justifications = []

        if trade['momentum_score'] >= 80:
            justifications.append("  • STRONG momentum score indicates powerful uptrend")
        elif trade['momentum_score'] >= 70:
            justifications.append("  • GOOD momentum score indicates solid uptrend")
        else:
            justifications.append("  • MODERATE momentum score")

        if trade['above_sma50'] and trade['above_sma200']:
            justifications.append("  • Price above both 50-day and 200-day MAs (bullish structure)")
        elif trade['above_sma50']:
            justifications.append("  • Price above 50-day MA (short-term bullish)")

        if trade['rsi'] > 50 and trade['rsi'] < 70:
            justifications.append("  • RSI in optimal range (50-70) - momentum without overbought")
        elif trade['rsi'] >= 70:
            justifications.append("  • RSI showing strong momentum (caution: overbought territory)")

        if trade['macd_histogram'] > 0:
            justifications.append("  • MACD histogram positive (bullish momentum)")

        if trade['volume_ratio'] > 1.5:
            justifications.append("  • Strong volume surge confirms price action")
        elif trade['volume_ratio'] > 1.2:
            justifications.append("  • Above-average volume supports the move")

        if trade['roc_20'] > 10:
            justifications.append(f"  • Strong 20-day return of {trade['roc_20']:.1f}%")

        for j in justifications:
            report_lines.append(j)

        report_lines.append("")

        # Risk consideration
        report_lines.append("RISK CONSIDERATIONS:")
        risks = []

        if trade['rsi'] > 70:
            risks.append("  ⚠ RSI overbought - consider waiting for pullback")
        if not trade['above_sma50']:
            risks.append("  ⚠ Below 50-day MA - weaker trend structure")
        if trade['volume_ratio'] < 1.0:
            risks.append("  ⚠ Below-average volume - lack of confirmation")
        if trade['asset_type'] == 'crypto':
            risks.append("  ⚠ Crypto asset - higher volatility, wider stops recommended")

        if risks:
            for risk in risks:
                report_lines.append(risk)
        else:
            report_lines.append("  • No major red flags identified")

        report_lines.append("")

        # Suggested action
        report_lines.append("SUGGESTED ACTION:")
        if trade['momentum_score'] >= 70 and trade['above_sma50'] and 50 <= trade['rsi'] <= 70:
            report_lines.append("  → CONSIDER ENTRY on this momentum play")
        elif trade['rsi'] > 70:
            report_lines.append("  → WATCH for pullback before entry (currently overbought)")
        else:
            report_lines.append("  → MONITOR - momentum present but confirm entry conditions")

        report_lines.append("")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("STRATEGY NOTES:")
    report_lines.append("-" * 80)
    report_lines.append("• Entry: Consider entries on pullbacks to support or consolidation breakouts")
    report_lines.append("• Stop Loss: Place 8-12% below entry (wider for crypto)")
    report_lines.append("• Position Size: Risk 1% of account per trade")
    report_lines.append("• Profit Target: 15-25% for stocks, 25-40% for crypto")
    report_lines.append("• Max Hold: 30 days or until momentum breaks")
    report_lines.append("")
    report_lines.append("DISCLAIMER: This is for educational purposes only. Not financial advice.")
    report_lines.append("Always conduct your own research and manage risk appropriately.")
    report_lines.append("=" * 80)

    # Print to console
    report = "\n".join(report_lines)
    print(report)

    # Save to file
    output_file = f"momentum_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to: {output_file}")


if __name__ == '__main__':
    main()
