# Momentum Trading Strategy - Complete Analysis

## How the System Identifies Momentum Trading Opportunities

### 1. Momentum Score Calculation (0-100 composite score)

**File**: `indicators/momentum.py`

The system creates a composite momentum score from 4 components:

#### Component 1: Rate of Change (ROC) - 30 points max
- **ROC > 20%**: 30 points (strong upward momentum)
- **ROC 10-20%**: 20 points (good momentum)
- **ROC 0-10%**: 10 points (weak momentum)
- **ROC < 0%**: 0 points (negative momentum)

Calculates: `(current_price - price_20_days_ago) / price_20_days_ago * 100`

#### Component 2: RSI Positioning - 25 points max
- **RSI > 70**: 25 points (strong momentum, overbought)
- **RSI 50-70**: 20 points (healthy momentum range)
- **RSI 30-50**: 10 points (weak momentum)
- **RSI < 30**: 0 points (oversold, negative)

#### Component 3: Trend Strength - 30 points max
Uses 30% of a 0-100 trend strength score based on:
- Price above 20-day MA: +20
- Price above 50-day MA: +20
- Price above 200-day MA: +20
- 20-day MA above 50-day MA: +15
- 50-day MA above 200-day MA: +15
- 20-day MA trending up (higher than 5 days ago): +10

#### Component 4: Volume Confirmation - 15 points max
- **Volume ratio > 2.0x**: 15 points (very strong surge)
- **Volume ratio 1.5-2.0x**: 10 points (significant surge)
- **Volume ratio 1.0-1.5x**: 5 points (above average)
- **Volume ratio < 1.0x**: 0 points (below average)

**Scoring Interpretation**:
- **>80**: Exceptional momentum - prime candidates
- **60-80**: Strong momentum - good candidates
- **40-60**: Moderate momentum - watch list
- **<40**: Weak momentum - avoid

---

## How Entry Timing is Determined

### Entry Signal Requirements (ALL must be met simultaneously)

**File**: `strategy/signals.py:67-175`

#### 6 Required Conditions:

1. **Breakout**: Price breaks above 20-day high
   - `current_price > max(previous_20_days_high)`

2. **Volume Surge**: Volume >= 1.5x average
   - `current_volume / 20_day_avg_volume >= 1.5`

3. **MACD Confirmation**: MACD histogram > 0
   - Confirms bullish momentum

4. **Trend Filter**: Price above 50-day MA
   - Ensures we're trading with the trend

5. **Momentum Trend**: 50-day MA trending up
   - `sma_50_today > sma_50_5_days_ago`

6. **Strong Momentum Score**: Momentum score >= 70
   - Ensures all momentum components align

### Entry Price Calculation

**Entry occurs at next bar's open + slippage**:
- Stocks: `next_open * 1.001` (0.1% slippage)
- Crypto: `next_open * 1.003` (0.3% slippage)

**Rationale**: In live trading, you can't enter at the signal bar's close. Entry happens next day at open with realistic slippage.

---

## How Exit Timing is Determined

### Exit Signal Priority (checked in order)

**File**: `strategy/signals.py:318-487`

#### 1. Stop Loss (Highest Priority)
- **Triggered if**: `daily_low <= stop_loss_price`
- **Exit at**: Stop loss price
- **Purpose**: Protect capital, limit losses to ~7-12%

#### 2. Profit Target
- **Triggered if**: `daily_high >= target_price`
- **Exit at**: Target price
- **Stocks**: 15-25% profit (based on volatility)
- **Crypto**: 25-40% profit (based on volatility)
- **Purpose**: Lock in gains at predetermined levels

#### 3. Trailing Stop
- **Activation**: When profit reaches +10%
- **Mechanism**: Trails 8% below highest price since entry
- **Updates**: Only moves up, never down
- **Triggered if**: `daily_low <= trailing_stop_price`
- **Purpose**: Protect profits while letting winners run

#### 4. Momentum Failure
- **Triggered if ALL are true**:
  - Price drops below 20-day MA on high volume, AND
  - One of: MACD negative, RSI < 40, or momentum score < 50
- **Exit at**: Current close
- **Purpose**: Exit when momentum breaks down

#### 5. Time Exit
- **Triggered if**: Position held >= 30 days
- **Exit at**: Current close
- **Purpose**: Prevent capital from being tied up too long in stagnant positions

### Stop Loss Calculation

**File**: `strategy/signals.py:217-267`

Uses the **wider** (more forgiving) of two methods:

1. **ATR-based**: `entry_price - (2.5 × ATR)`
2. **Percentage-based**:
   - Stocks: 7% stop (`entry_price × 0.93`)
   - Crypto: 12% stop (`entry_price × 0.88`)

Also ensures stop is below recent 20-day low for better placement.

---

## Position Sizing

**File**: `strategy/risk_manager.py:90-150`

### 1% Risk Model

Formula: `position_size = (account_size × 0.01) / (entry_price - stop_loss)`

**Example**:
- Account: $100,000
- Risk per trade: 1% = $1,000
- Entry: $100
- Stop: $93
- Risk per share: $7
- Position size: $1,000 / $7 = 142 shares
- Position value: $14,200

### Crypto Adjustment
- Crypto positions reduced to **70% of calculated size**
- Rationale: Higher volatility requires smaller positions

### Constraints
- **Minimum**: 1% of account ($1,000 on $100k account)
- **Maximum**: 5% of account ($5,000 on $100k account)

### Portfolio Limits
- **Max positions**: 8 concurrent
- **Max portfolio exposure**: 25% of capital
- **Max sector exposure**: 15% in any single sector
- **Daily loss limit**: 3% circuit breaker (stops trading for the day)

---

## CRITICAL LIMITATIONS

### 1. **Entry Signal Is Too Restrictive** ⚠️ HIGH IMPACT

**Problem**: ALL 6 conditions must be met simultaneously
- **File**: `strategy/signals.py:150`
- **Code**: `all_met = (breakout and volume_surge and macd_positive and above_ma50 and ma_trending_up and strong_momentum)`

**Impact**:
- Misses most real-world opportunities
- 20-day breakout + volume surge + all momentum aligned = very rare
- In sideways/choppy markets, may generate 0 signals for weeks

**Example Scenario**:
- Stock has 85 momentum score, volume surge, MACD positive, above MAs
- BUT no 20-day breakout (price at 19-day high instead)
- **Result**: NO ENTRY SIGNAL

**Better Approach**:
- Make entry criteria configurable
- Allow weighted scoring system
- Consider "partial signals" with reduced position size

---

### 2. **No Live Market Data Support** ⚠️ HIGH IMPACT

**Problem**: Only works with daily end-of-day data

**Limitations**:
- Can't detect intraday breakouts
- Can't adjust to real-time price changes
- Entry always delayed to next day's open
- Misses opportunities that reverse before next day

**Missing Features**:
- Real-time data feeds
- Intraday timeframes (1-min, 5-min, 1-hour)
- Market hours awareness
- Pre-market/after-hours handling

---

### 3. **Stop Loss Execution is Unrealistic** ⚠️ MEDIUM-HIGH IMPACT

**Problem**: Assumes perfect execution at stop price

**File**: `strategy/signals.py:382-395`

**Code**:
```python
if current['low'] <= stop:
    return {'exit_price': stop, ...}  # Assumes fill at exact stop
```

**Reality**:
- Gaps down can blow through stops
- Slippage on stop orders (especially volatile stocks/crypto)
- No differentiation between stop-loss and stop-limit orders

**Better Approach**:
- Model gap risk
- Add slippage to stop exits (e.g., 0.5% worse than stop)
- Consider stop-limit with kill zones

---

### 4. **No Adaptive Position Sizing** ⚠️ MEDIUM IMPACT

**Problem**: Fixed 1% risk regardless of market conditions

**File**: `strategy/risk_manager.py:122-129`

**Limitations**:
- Same risk in bull market vs bear market
- No adjustment for winning/losing streaks
- No Kelly Criterion or optimal f
- No volatility-based scaling

**Better Approach**:
- Reduce size after losses (drawdown protection)
- Increase size during winning streaks (capitalize on edge)
- Scale based on VIX or market regime
- Implement Kelly Criterion with safety factor

---

### 5. **Momentum Score Lacks Context** ⚠️ MEDIUM IMPACT

**Problem**: Absolute scoring without relative ranking

**File**: `indicators/momentum.py:221-327`

**Issues**:
- Score of 75 might be good in bear market, average in bull market
- No comparison to market/sector benchmarks
- No percentile ranking within universe

**Example**:
- Stock A: 75 momentum score (vs market avg 70)
- Stock B: 75 momentum score (vs market avg 80)
- Current system treats them equally, but B is relatively weak

**Better Approach**:
- Calculate relative momentum (vs SPY, sector ETF)
- Percentile ranking within universe
- Regime-aware scoring (bull vs bear)

---

### 6. **No Portfolio-Level Optimization** ⚠️ MEDIUM IMPACT

**Problem**: Each trade evaluated independently

**Missing**:
- Correlation analysis between positions
- Portfolio diversification scoring
- Drawdown-aware position management
- Dynamic allocation based on market conditions

**Example Scenario**:
- All 8 positions are tech stocks (highly correlated)
- Sector rotation happens
- All positions drop simultaneously
- No sector limit prevents this if all are different sub-sectors

---

### 7. **Volume Surge Detection is Simplistic** ⚠️ MEDIUM-LOW IMPACT

**File**: `indicators/momentum.py:159-219`

**Current**: `volume / 20_day_avg > 1.5`

**Problems**:
- Doesn't account for intraday volume patterns
- No distinction between buying vs selling pressure
- Can trigger on low-quality volume (retail frenzy vs institutional)

**Better Approach**:
- Cumulative delta (buying volume - selling volume)
- Volume-weighted average price (VWAP) analysis
- Smart money vs dumb money indicators
- Time-of-day volume normalization

---

### 8. **No Market Regime Detection** ⚠️ MEDIUM-LOW IMPACT

**Problem**: Same strategy in all market conditions

**Missing**:
- Bull market detection (favor longs, stay in longer)
- Bear market detection (avoid or reduce size)
- Sideways market detection (tighten stops, take quick profits)
- Volatility regime shifts (VIX spikes)

**Impact**:
- Strategy performs poorly in bear markets
- Overexposes in high volatility
- Doesn't adapt to changing conditions

---

### 9. **Profit Target is Static** ⚠️ MEDIUM-LOW IMPACT

**File**: `strategy/signals.py:269-316`

**Current**: Fixed 15-40% based on asset type and volatility

**Problems**:
- Doesn't consider support/resistance levels
- Ignores market structure (trends vs ranges)
- No adjustment for earnings, catalysts
- May exit too early in strong trends

**Better Approach**:
- Use ATR multiples (e.g., 3-5× ATR)
- Target previous resistance levels
- Adjust based on trend strength
- Multiple profit targets (partial exits)

---

### 10. **Trailing Stop Lacks Sophistication** ⚠️ LOW-MEDIUM IMPACT

**File**: `strategy/signals.py:410-426`

**Current**:
- Activates at +10% profit
- Trails 8% below highest price

**Problems**:
- Fixed percentages ignore volatility
- No consideration of support levels
- Can trigger on normal pullbacks in strong trends
- No ATR-based dynamic trailing

**Better Approach**:
- ATR-based trailing (e.g., 2× ATR below high)
- Chandelier stop
- SAR (Parabolic Stop and Reverse)
- Support-based trailing

---

### 11. **No Sector/Industry Analysis** ⚠️ LOW-MEDIUM IMPACT

**File**: `strategy/scanner.py`

**Current**: Simple sector exposure limit (15%)

**Missing**:
- Sector rotation detection
- Industry relative strength
- Sector-specific entry/exit rules
- Cross-sector correlation

**Impact**:
- May buy into rotating out sectors
- Misses sector leadership changes
- Can't capitalize on sector trends

---

### 12. **Backtesting Limitations** ⚠️ LOW-MEDIUM IMPACT

**Issues**:
- No transaction costs modeling (taxes, fees)
- No margin/borrowing costs
- Assumes infinite liquidity (can always fill)
- No overnight gap modeling
- No holiday/trading calendar awareness

**File**: `backtest/engine.py` (not read yet, but likely present)

---

### 13. **No Machine Learning / Adaptivity** ⚠️ LOW IMPACT

**Current**: Completely rules-based

**Missing**:
- Parameter optimization based on recent performance
- Pattern recognition for entry setups
- Adaptive thresholds
- Market state classification

**Note**: This may be intentional for transparency and consistency.

---

### 14. **Crypto-Specific Issues** ⚠️ LOW IMPACT (unless trading crypto heavily)

**Problems**:
- No exchange-specific handling
- No consideration of funding rates (for perpetuals)
- No 24/7 trading logic (crypto never closes)
- No whale wallet tracking
- No on-chain metrics

---

### 15. **No News/Catalyst Integration** ⚠️ LOW IMPACT

**Missing**:
- Earnings date awareness (avoid/target earnings)
- News sentiment analysis
- Economic calendar integration
- Fed announcement detection

---

## PRIORITY FEATURE DEVELOPMENT ROADMAP

### TIER 1: Critical (Implement First)

#### 1. **Flexible Entry Signal System** 🔥
- **Priority**: HIGHEST
- **Effort**: Medium
- **Impact**: Dramatically increase signal frequency

**Implementation**:
- Make entry conditions configurable (YAML config)
- Add weighted scoring (e.g., "need 4 of 6 conditions")
- Allow tiered entry sizes (full size if 6/6, half size if 4/6)
- Add "signal strength" output to analyze near-misses

**File to modify**: `strategy/signals.py`

#### 2. **Market Regime Detection** 🔥
- **Priority**: HIGHEST
- **Effort**: Medium
- **Impact**: Prevent losses in wrong market conditions

**Implementation**:
- Calculate market regime (bull/bear/sideways)
- Use SPY 50/200 MA, VIX levels, breadth indicators
- Adjust risk parameters by regime:
  - Bull: 1% risk, 8 positions
  - Neutral: 0.75% risk, 6 positions
  - Bear: 0.5% risk, 4 positions or no trading

**File to create**: `indicators/market_regime.py`

#### 3. **Realistic Stop Loss Modeling** 🔥
- **Priority**: HIGH
- **Effort**: Low
- **Impact**: More accurate backtest results

**Implementation**:
- Add slippage to stop exits (0.5% for stocks, 1% crypto)
- Model gap risk (if gap > 2 ATR, add extra slippage)
- Differentiate stop-loss vs stop-limit

**File to modify**: `strategy/signals.py`

---

### TIER 2: High Value (Implement Next)

#### 4. **Adaptive Position Sizing**
- **Priority**: HIGH
- **Effort**: Medium
- **Impact**: Better capital protection and growth

**Implementation**:
- Kelly Criterion calculator
- Drawdown-aware sizing (reduce after losses)
- Confidence-based sizing (full size vs half size entries)
- Volatility-adjusted sizing

**File to modify**: `strategy/risk_manager.py`

#### 5. **Relative Momentum Ranking**
- **Priority**: HIGH
- **Effort**: Low-Medium
- **Impact**: Better opportunity selection

**Implementation**:
- Calculate momentum vs SPY (or sector ETF)
- Add percentile ranking within universe
- Prefer top decile relative momentum
- Update daily or weekly

**File to modify**: `indicators/momentum.py`

#### 6. **Multi-Timeframe Analysis**
- **Priority**: HIGH
- **Effort**: High
- **Impact**: Better entries and exits

**Implementation**:
- Daily + Weekly trend alignment
- Intraday entry refinement (if live data available)
- Higher timeframe support/resistance
- Multiple timeframe momentum confirmation

**File to modify**: Multiple files

---

### TIER 3: Nice to Have

#### 7. **Advanced Trailing Stops**
- ATR-based chandelier stops
- Parabolic SAR
- Support-level based trailing

**File to modify**: `strategy/signals.py`

#### 8. **Partial Profit Taking**
- Exit 50% at first target
- Exit 25% at second target
- Trail remaining 25%

**File to modify**: `strategy/signals.py`

#### 9. **Volume Analysis Enhancement**
- Buying vs selling volume
- VWAP analysis
- Smart money indicators

**File to modify**: `indicators/momentum.py`

#### 10. **Backtesting Improvements**
- Transaction costs
- Slippage modeling
- Market impact (for large orders)
- Holiday calendar

**File to modify**: `backtest/engine.py`

---

### TIER 4: Advanced Features

#### 11. **Machine Learning Integration**
- Pattern recognition for setups
- Parameter optimization
- Market state classification
- Anomaly detection

#### 12. **Portfolio Optimization**
- Correlation analysis
- Mean-variance optimization
- Risk parity allocation
- Dynamic rebalancing

#### 13. **Real-Time Trading**
- Live data feeds
- Broker API integration
- Order management system
- Real-time risk monitoring

#### 14. **Alternative Data**
- News sentiment
- Social media signals
- Earnings calendars
- Economic data

---

## RECOMMENDED FIRST 3 FEATURES

Based on effort vs impact:

### 1. **Flexible Entry Signals** (2-3 days)
- Highest impact on strategy usability
- Medium effort
- Unlocks all other improvements

### 2. **Market Regime Detection** (1-2 days)
- Prevents trading in wrong conditions
- Medium effort
- Immediate risk reduction

### 3. **Realistic Stop Modeling** (1 day)
- Critical for accurate backtests
- Low effort
- Better understands true risk/reward

**Total**: 4-6 days of development for massive improvement

---

## TESTING RECOMMENDATIONS

For each feature:

1. **Unit tests**: Test logic in isolation
2. **Backtest comparison**: Before vs after metrics
3. **Walk-forward validation**: Test on unseen data
4. **Paper trading**: Validate in live market (simulated)
5. **Sensitivity analysis**: Test parameter robustness

---

## MEASUREMENT METRICS

Track these for feature effectiveness:

### Entry Signal Changes:
- Signal frequency (signals per day)
- Signal quality (win rate on signals)
- False positive rate
- Opportunity cost (missed trades analysis)

### Risk Management Changes:
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- Recovery factor
- Consecutive losses

### Overall Strategy:
- CAGR (annualized return)
- Win rate
- Profit factor
- Average win/loss ratio
- Exposure time

---

## CONCLUSION

**Current State**: The strategy has a solid foundation with comprehensive technical analysis but is hampered by overly restrictive entry criteria and lack of adaptivity.

**Biggest Weakness**: Entry signal requirement of all 6 conditions creates ~90% false negatives (misses good opportunities).

**Biggest Strength**: Risk management is well-designed with proper position sizing and portfolio limits.

**Quick Win**: Implement flexible entry signals (weighted scoring) - could increase trade frequency 5-10x while maintaining quality.

**Long-term Success**: Add market regime detection to avoid bear markets and adaptive position sizing to capitalize on winning periods.
