# Gordon Trading System - Usage Guide

## Tiingo API Free Tier Limits

**IMPORTANT**: Understand these limits to avoid being rate-limited:

- **5 requests per minute**
- **~50 unique symbols per hour**
- **500 requests per day**
- **500 unique symbols per month**

## Recommended Tools

### 1. Single Ticker Analysis (RECOMMENDED for Free Tier)

**File**: `analyze_ticker.py`

**API Calls**: 1 per run ✅

**Usage**:
```bash
# Set API key (only needed once per session)
export TIINGO_API_KEY=1fbf176f718099036edd83ae80c1ba9e545007a2

# Analyze any stock
python analyze_ticker.py AAPL
python analyze_ticker.py NVDA
python analyze_ticker.py TSLA

# Analyze any crypto
python analyze_ticker.py btcusd
python analyze_ticker.py ethusd
python analyze_ticker.py solusd
```

**What it does**:
- Fetches 90 days of price data (1 API call)
- Calculates all technical indicators
- Evaluates entry/exit signals
- Shows current trade setup
- Provides position sizing
- Gives actionable recommendation
- Saves detailed report to file

**Safe Usage**:
- ✅ Run up to 5 times per minute
- ✅ Analyze up to 50 different tickers per hour
- ✅ Perfect for monitoring your watchlist

**Example Workflow**:
```bash
# Analyze your top 5 stocks (5 API calls)
python analyze_ticker.py AAPL
python analyze_ticker.py MSFT
python analyze_ticker.py NVDA
python analyze_ticker.py TSLA
python analyze_ticker.py AMD

# Wait 1 minute, then analyze crypto
sleep 60
python analyze_ticker.py btcusd
python analyze_ticker.py ethusd
```

---

### 2. Multi-Ticker Scanner (USE CAREFULLY)

**File**: `find_momentum_trades.py`

**API Calls**: 8 per run (5 stocks + 3 cryptos) ⚠️

**Usage**:
```bash
export TIINGO_API_KEY=1fbf176f718099036edd83ae80c1ba9e545007a2
python find_momentum_trades.py
```

**What it does**:
- Scans 5 major stocks + 3 major cryptos
- Ranks by momentum score
- Returns top 5 opportunities
- Detailed justification for each

**Safe Usage**:
- ✅ Run maximum 6 times per hour (to stay under 50 symbol limit)
- ⚠️ Do NOT run repeatedly in quick succession
- ⚠️ Wait at least 10 minutes between runs

---

### 3. Full Portfolio Scanner (PREMIUM TIER ONLY)

**File**: `main.py scan`

**API Calls**: 60+ per run ❌

**NOT RECOMMENDED for Free Tier** - Will immediately hit hourly limits

---

## Testing API Status

Check if you're rate-limited:

```bash
python test_rate_limit.py
```

Check actual API response:

```bash
python check_api_status.py
```

---

## Recommended Daily Workflow (Free Tier)

### Morning Routine (10 API calls)
```bash
# Check your top 10 stocks/crypto individually
python analyze_ticker.py AAPL
python analyze_ticker.py MSFT
# ... etc (space them out 1-2 seconds each)
```

### Midday Check (8 API calls)
```bash
# Run the multi-ticker scanner once
python find_momentum_trades.py
```

### End of Day (10 API calls)
```bash
# Re-analyze your open positions
python analyze_ticker.py NVDA
python analyze_ticker.py btcusd
# ... etc
```

**Total**: ~28 API calls per day (well within 500/day limit)

---

## Avoiding Rate Limits

### DO ✅
- Use `analyze_ticker.py` for individual analysis
- Space out requests (wait 1-2 seconds between calls)
- Monitor no more than 50 different tickers per hour
- Save output files and review them (don't re-run unnecessarily)

### DON'T ❌
- Don't run `main.py scan` (uses 60+ calls)
- Don't run scanners in rapid succession
- Don't analyze 100+ different tickers in one day
- Don't retry immediately after getting rate limited

---

## Error Messages

**"You have run over your hourly request allocation"**
- Wait 1 hour before making more requests
- When limit resets, use `analyze_ticker.py` instead of full scans

**"Rate limit exceeded"**
- Same as above - wait for the hourly window to reset

**"You have run over your 500 symbol lookup for this month"**
- You've analyzed 500 different unique symbols this month
- Either wait until next month or upgrade to paid tier
- Tip: Re-analyzing the same ticker doesn't count toward unique symbol limit

---

## Best Practices for Free Tier

1. **Create a Watchlist**: Pick 10-20 stocks/cryptos you actually want to trade
2. **Analyze Strategically**: Only analyze tickers on your watchlist
3. **Batch Your Analysis**: Analyze all watchlist tickers once in the morning
4. **Save Reports**: Review saved .txt files instead of re-running
5. **One Scan Per Day**: Run `find_momentum_trades.py` maximum once daily
6. **Track Your Usage**: Keep count of how many different symbols you've analyzed

---

## Upgrade Considerations

Consider upgrading to Tiingo paid tier if you:
- Want to scan 100+ stocks daily
- Need real-time data updates
- Want to run backtests on large universes
- Need more than 50 symbols per hour

Free tier is perfect for:
- Monitoring a focused watchlist (10-20 tickers)
- Daily analysis of key positions
- Learning the strategy
- Paper trading with limited positions

---

## Quick Reference Commands

```bash
# Single ticker analysis (SAFE - 1 API call)
python analyze_ticker.py <TICKER>

# Small momentum scan (MODERATE - 8 API calls)
python find_momentum_trades.py

# Check if rate limited (SAFE - 1 API call)
python test_rate_limit.py

# Full scan (DANGEROUS - 60+ API calls)
# DON'T USE: python main.py scan
```

---

**Remember**: Quality over quantity. Deep analysis of a few good opportunities beats superficial scanning of hundreds of tickers.
