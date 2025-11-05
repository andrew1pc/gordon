# Feature Development Roadmap
## Priority Features - Iterative Implementation Plan

**Timeline**: 3-4 days
**Approach**: Iterative, test-driven, hardcoded constants

---

## Overview

This roadmap builds 3 critical features in order of dependency:

1. **Feature 1**: Flexible Entry Signals (Iteration 1-3)
2. **Feature 2**: Market Regime Detection (Iteration 4-6)
3. **Feature 3**: Realistic Stop Modeling (Iteration 7-8)

Each iteration is a small, testable increment that can be validated independently.

---

# FEATURE 1: Flexible Entry Signals

**Goal**: Increase signal frequency 5-10x while maintaining quality
**Timeline**: 1.5 days (Iterations 1-3)
**Impact**: HIGH - Unlocks actual trading opportunities

## Current Problem
```python
# signals.py:150 - ALL must be True
all_met = (breakout and volume_surge and macd_positive and
          above_ma50 and ma_trending_up and strong_momentum)
```
This creates ~90% false negatives.

---

## Iteration 1: Add Signal Strength Scoring (0.5 days)

### Goal
Calculate entry signal strength as a score (0-100) instead of binary yes/no.

### Building Blocks

#### 1.1: Create Signal Scoring Function
**File**: `strategy/signals.py`

**Task**: Add new method to `SignalGenerator` class

```python
def calculate_entry_signal_strength(
    self,
    df: pd.DataFrame,
    current_idx: int,
    ticker: str
) -> Dict[str, Any]:
    """
    Calculate entry signal strength (0-100) from conditions.

    Returns:
        {
            'signal_strength': 0-100,
            'conditions_met': ['condition1', 'condition2'],
            'conditions_failed': ['condition3'],
            'condition_scores': {
                'breakout': 20,
                'volume_surge': 15,
                ...
            }
        }
    """

    if current_idx < 20:
        return {'signal_strength': 0, 'conditions_met': [], 'conditions_failed': [], 'condition_scores': {}}

    try:
        current = df.iloc[current_idx]
        prev_20_highs = df['high'].iloc[current_idx-20:current_idx]

        scores = {}
        conditions_met = []
        conditions_failed = []

        # Condition 1: Breakout (20 points)
        breakout = current['close'] > prev_20_highs.max()
        if breakout:
            scores['breakout'] = 20
            conditions_met.append('breakout_20d_high')
        else:
            scores['breakout'] = 0
            conditions_failed.append('breakout_20d_high')

        # Condition 2: Volume surge (15 points with partial credit)
        volume_ratio = current.get('volume_ratio', 1.0)
        if volume_ratio >= 1.5:
            scores['volume_surge'] = 15
            conditions_met.append('volume_surge')
        elif volume_ratio >= 1.3:
            scores['volume_surge'] = 10  # Partial credit
            conditions_failed.append('volume_surge')
        elif volume_ratio >= 1.1:
            scores['volume_surge'] = 5   # Minimal credit
            conditions_failed.append('volume_surge')
        else:
            scores['volume_surge'] = 0
            conditions_failed.append('volume_surge')

        # Condition 3: MACD positive (15 points)
        macd_positive = current.get('macd_histogram', 0) > 0
        if macd_positive:
            scores['macd_positive'] = 15
            conditions_met.append('macd_positive')
        else:
            scores['macd_positive'] = 0
            conditions_failed.append('macd_positive')

        # Condition 4: Price above 50-day MA (15 points)
        above_ma50 = current['close'] > current.get('sma_50', current['close'])
        if above_ma50:
            scores['above_ma50'] = 15
            conditions_met.append('above_ma50')
        else:
            scores['above_ma50'] = 0
            conditions_failed.append('above_ma50')

        # Condition 5: 50-day MA trending up (15 points)
        ma_trending_up = False
        if 'sma_50' in current and current_idx >= 25:
            ma50_current = current['sma_50']
            ma50_5d_ago = df['sma_50'].iloc[current_idx - 5]
            ma_trending_up = ma50_current > ma50_5d_ago

        if ma_trending_up:
            scores['ma_trending_up'] = 15
            conditions_met.append('ma50_trending_up')
        else:
            scores['ma_trending_up'] = 0
            conditions_failed.append('ma50_trending_up')

        # Condition 6: Strong momentum (20 points with partial credit)
        momentum_score = current.get('momentum_score', 0)
        if momentum_score >= 70:
            scores['strong_momentum'] = 20
            conditions_met.append('strong_momentum')
        elif momentum_score >= 60:
            scores['strong_momentum'] = 15  # Partial credit
            conditions_failed.append('strong_momentum')
        elif momentum_score >= 50:
            scores['strong_momentum'] = 10  # Minimal credit
            conditions_failed.append('strong_momentum')
        else:
            scores['strong_momentum'] = 0
            conditions_failed.append('strong_momentum')

        # Calculate total strength
        signal_strength = sum(scores.values())

        return {
            'signal_strength': signal_strength,
            'conditions_met': conditions_met,
            'conditions_failed': conditions_failed,
            'condition_scores': scores
        }

    except Exception as e:
        logger.error(f"Error calculating signal strength for {ticker}: {e}")
        return {'signal_strength': 0, 'conditions_met': [], 'conditions_failed': [], 'condition_scores': {}}
```

**Testing**:
```python
# test_signals.py
def test_signal_strength_calculation():
    # Test case 1: All conditions met = 100
    # Test case 2: 4 of 6 conditions = ~65-70
    # Test case 3: 2 of 6 conditions = ~35
    # Test case 4: Partial credit scenarios
```

**Acceptance Criteria**:
- [ ] Function returns score 0-100
- [ ] Score 100 when all conditions met
- [ ] Score 0 when no conditions met
- [ ] Partial credit for near-misses (e.g., volume 1.4x = 10 points)
- [ ] Unit tests pass

---

#### 1.2: Add Signal Strength to check_entry_signals()
**File**: `strategy/signals.py`

**Task**: Modify existing `check_entry_signals()` to use 70 threshold

```python
def check_entry_signals(
    self,
    df: pd.DataFrame,
    current_idx: int,
    ticker: str
) -> Optional[Dict[str, Any]]:
    """
    Check if entry conditions are met using signal strength >= 70.

    This is more flexible than requiring all 6 conditions (100 points).
    Typically 4-5 strong conditions will meet the 70 point threshold.
    """

    # Calculate signal strength
    strength_result = self.calculate_entry_signal_strength(df, current_idx, ticker)

    # Require 70+ points (roughly 4-5 of 6 conditions)
    MIN_SIGNAL_STRENGTH = 70

    if strength_result['signal_strength'] < MIN_SIGNAL_STRENGTH:
        return None

    # Generate signal with strength included
    signal = {
        'signal': True,
        'ticker': ticker,
        'date': df.index[current_idx],
        'entry_price': self.calculate_entry_price(df, current_idx, 'stock'),
        'signal_strength': strength_result['signal_strength'],
        'conditions_met': strength_result['conditions_met'],
        'conditions_failed': strength_result['conditions_failed'],
        'current_price': df.iloc[current_idx]['close']
    }

    logger.info(
        f"Entry signal for {ticker} on {signal['date'].date()} "
        f"(strength: {signal['signal_strength']}/100)"
    )

    return signal
```

**Testing**:
```python
def test_flexible_threshold():
    # With 70 threshold, should accept 4-5 strong conditions
    # Should reject < 70 points
```

**Acceptance Criteria**:
- [ ] Uses 70 point threshold
- [ ] Returns signal strength in result
- [ ] Logs signal strength
- [ ] Tests pass

---

## Iteration 2: Implement Tiered Position Sizing (0.5 days)

### Goal
Adjust position size based on signal strength (stronger signal = larger position).

### Building Blocks

#### 2.1: Add Position Size Scaling to RiskManager
**File**: `strategy/risk_manager.py`

**Task**: Add new method for scaled position sizing

```python
def calculate_scaled_position_size(
    self,
    entry_price: float,
    stop_loss: float,
    asset_type: str,
    signal_strength: int
) -> Tuple[float, float]:
    """
    Calculate position size scaled by signal strength.

    Scaling:
        - signal_strength >= 90: 100% of normal size (excellent signal)
        - signal_strength 80-89: 80% of normal size (strong signal)
        - signal_strength 70-79: 60% of normal size (good signal)
        - signal_strength < 70: 0% (don't trade - should be filtered earlier)

    Returns:
        Tuple of (position_size, dollar_value)
    """

    # Calculate base position size
    base_size, base_value = self.calculate_position_size(
        entry_price, stop_loss, asset_type
    )

    # Scale by signal strength
    if signal_strength >= 90:
        scale_factor = 1.0
    elif signal_strength >= 80:
        scale_factor = 0.8
    elif signal_strength >= 70:
        scale_factor = 0.6
    else:
        # Should not happen if entry filter is working
        scale_factor = 0.0
        logger.warning(f"Signal strength {signal_strength} below 70, using 0% size")

    scaled_size = base_size * scale_factor
    scaled_value = base_value * scale_factor

    logger.info(
        f"Position scaled by signal strength {signal_strength}: "
        f"{scale_factor*100:.0f}% = {scaled_size:.2f} units (${scaled_value:,.0f})"
    )

    return scaled_size, scaled_value
```

**Testing**:
```python
def test_position_scaling():
    # Test signal_strength 95 = 100% size
    # Test signal_strength 85 = 80% size
    # Test signal_strength 75 = 60% size
    # Test signal_strength 65 = 0% size
```

**Acceptance Criteria**:
- [ ] Correctly scales position by signal strength
- [ ] Logs scaling decisions
- [ ] Tests pass
- [ ] Still respects min/max position constraints

---

#### 2.2: Update validate_new_trade() to Use Signal Strength
**File**: `strategy/risk_manager.py`

**Task**: Modify `validate_new_trade()` signature

```python
def validate_new_trade(
    self,
    ticker: str,
    entry_price: float,
    stop_loss: float,
    sector: Optional[str],
    asset_type: str,
    signal_strength: int = 100  # NEW parameter (default for backward compatibility)
) -> Tuple[bool, str, float, float]:
    """
    Validate if new trade meets risk criteria.

    Args:
        signal_strength: Entry signal strength (0-100)

    Returns:
        (approved, reason, position_size, dollar_value)
    """

    # Check if we've hit max positions
    if len(self.positions) >= self.max_positions:
        return False, "Max positions reached", 0, 0

    # Check daily loss limit
    if self._check_daily_loss_limit():
        return False, "Daily loss limit hit", 0, 0

    # Calculate position size with signal strength scaling
    position_size, dollar_value = self.calculate_scaled_position_size(
        entry_price, stop_loss, asset_type, signal_strength
    )

    # Check portfolio exposure
    current_exposure = sum(pos.dollar_value for pos in self.positions.values())
    max_exposure = self.account_size * self.max_portfolio_exposure

    if current_exposure + dollar_value > max_exposure:
        return False, "Portfolio exposure limit", 0, 0

    # Check sector exposure (if sector provided)
    if sector:
        sector_exposure = sum(
            pos.dollar_value for pos in self.positions.values()
            if pos.sector == sector
        )
        max_sector = self.account_size * self.max_sector_exposure

        if sector_exposure + dollar_value > max_sector:
            return False, f"Sector {sector} exposure limit", 0, 0

    return True, "Approved", position_size, dollar_value
```

**Testing**:
```python
def test_validate_with_signal_strength():
    # Weak signal should result in smaller position
    # Strong signal should result in full position
    # Should still respect all other limits
```

**Acceptance Criteria**:
- [ ] Accepts signal_strength parameter
- [ ] Uses scaled position sizing
- [ ] Backward compatible (default=100)
- [ ] Tests pass

---

#### 2.3: Update main.py to Pass Signal Strength
**File**: `main.py`

**Task**: Update `process_signals()` to pass signal strength

```python
# In process_signals(), around line 422-424
approved, reason, position_size, dollar_value = self.risk_manager.validate_new_trade(
    ticker,
    entry_price,
    stop_loss,
    sector,
    asset_type,
    signal_strength=signal.get('signal_strength', 100)  # NEW
)
```

**Testing**:
- Run scan with flexible signals
- Verify position sizes are scaled appropriately
- Check logs show scaling factor

**Acceptance Criteria**:
- [ ] Signal strength passed through
- [ ] Position sizes scale correctly (70-79=60%, 80-89=80%, 90+=100%)
- [ ] End-to-end test passes

---

## Iteration 3: Add Signal Analysis & Reporting (0.5 days)

### Goal
Provide visibility into signal quality and missed opportunities.

### Building Blocks

#### 3.1: Create Signal Analysis Tool
**File**: `strategy/signal_analyzer.py` (NEW FILE)

**Task**: Create analysis utility

```python
"""Signal analysis and diagnostics."""

import pandas as pd
from typing import Dict, List
from datetime import datetime


class SignalAnalyzer:
    """Analyze signal quality and missed opportunities."""

    def __init__(self):
        self.signals_analyzed = []
        self.near_misses = []  # Signals that scored 60-69

    def analyze_signal(self, signal_result: Dict) -> None:
        """Record signal for analysis."""
        self.signals_analyzed.append({
            'timestamp': datetime.now(),
            'ticker': signal_result.get('ticker'),
            'signal_strength': signal_result.get('signal_strength', 0),
            'conditions_met': signal_result.get('conditions_met', []),
            'conditions_failed': signal_result.get('conditions_failed', []),
            'triggered': signal_result.get('signal', False)
        })

        # Track near misses (60-69 score)
        strength = signal_result.get('signal_strength', 0)
        if 60 <= strength < 70 and not signal_result.get('signal', False):
            self.near_misses.append(signal_result)

    def generate_report(self) -> str:
        """Generate analysis report."""
        if not self.signals_analyzed:
            return "No signals analyzed yet."

        df = pd.DataFrame(self.signals_analyzed)

        report = []
        report.append("=" * 80)
        report.append("SIGNAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Total signals analyzed: {len(df)}")
        report.append(f"Signals triggered (>=70): {df['triggered'].sum()}")
        report.append(f"Near misses (60-69): {len(self.near_misses)}")
        report.append("")

        # Signal strength distribution
        report.append("SIGNAL STRENGTH DISTRIBUTION:")
        report.append(f"  90-100 (Excellent): {len(df[df['signal_strength'] >= 90])}")
        report.append(f"  80-89  (Strong):    {len(df[(df['signal_strength'] >= 80) & (df['signal_strength'] < 90)])}")
        report.append(f"  70-79  (Good):      {len(df[(df['signal_strength'] >= 70) & (df['signal_strength'] < 80)])}")
        report.append(f"  60-69  (Moderate):  {len(df[(df['signal_strength'] >= 60) & (df['signal_strength'] < 70)])}")
        report.append(f"  < 60   (Weak):      {len(df[df['signal_strength'] < 60])}")
        report.append("")

        # Most common failing conditions
        if self.near_misses:
            report.append("NEAR MISSES - Most Common Missing Conditions:")
            all_failed = []
            for nm in self.near_misses:
                all_failed.extend(nm.get('conditions_failed', []))

            from collections import Counter
            failed_counts = Counter(all_failed)
            for condition, count in failed_counts.most_common(5):
                report.append(f"  {condition}: {count} times")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def export_to_csv(self, filepath: str) -> None:
        """Export signal analysis to CSV."""
        if self.signals_analyzed:
            df = pd.DataFrame(self.signals_analyzed)
            df.to_csv(filepath, index=False)

        if self.near_misses:
            df_near = pd.DataFrame(self.near_misses)
            df_near.to_csv(filepath.replace('.csv', '_near_misses.csv'), index=False)
```

**Testing**:
```python
def test_signal_analyzer():
    # Test recording signals
    # Test report generation
    # Test near-miss tracking
```

**Acceptance Criteria**:
- [ ] Records all signals analyzed
- [ ] Tracks near misses (60-69)
- [ ] Generates readable report
- [ ] Exports to CSV
- [ ] Tests pass

---

#### 3.2: Integrate Signal Analyzer into main.py
**File**: `main.py`

**Task**: Add signal analyzer to strategy

```python
# In __init__
from strategy.signal_analyzer import SignalAnalyzer
self.signal_analyzer = SignalAnalyzer()

# In scan_and_select_candidates(), after checking signals
for ticker in candidates:
    # ... existing code to get df ...

    # NEW: Always calculate strength for analysis
    strength_result = self.signal_generator.calculate_entry_signal_strength(
        df, current_idx, ticker
    )
    self.signal_analyzer.analyze_signal({
        **strength_result,
        'ticker': ticker,
        'signal': strength_result['signal_strength'] >= 70
    })

    # Check for entry signal
    signal = self.signal_generator.check_entry_signals(df, current_idx, ticker)

    if signal:
        # ... existing entry logic ...

# At end of run_daily()
report = self.signal_analyzer.generate_report()
logger.info(report)
self.signal_analyzer.export_to_csv(
    f"{self.report_dir}/signals_{date.strftime('%Y-%m-%d')}.csv"
)
```

**Testing**:
- Run daily cycle
- Check for signal analysis report in logs
- Verify CSV export

**Acceptance Criteria**:
- [ ] Analyzer integrated
- [ ] Reports generated daily
- [ ] CSVs exported
- [ ] No performance degradation

---

## Feature 1 Summary

**Deliverables**:
- [x] Iteration 1: Signal strength scoring system (70 point threshold hardcoded)
- [x] Iteration 2: Tiered position sizing by signal strength
- [x] Iteration 3: Signal analysis & reporting

**Files Modified**:
- `strategy/signals.py`
- `strategy/risk_manager.py`
- `main.py`

**Files Created**:
- `strategy/signal_analyzer.py`

**Testing**:
- Unit tests for all new methods
- Integration test (run full scan)

**Expected Outcome**:
- 5-10x increase in signal frequency
- Position sizing aligned with signal quality
- Full visibility into signal performance
- 70-79 signals = 60% size, 80-89 = 80% size, 90+ = 100% size

---

# FEATURE 2: Market Regime Detection

**Goal**: Prevent losses in bear markets, optimize for bull markets
**Timeline**: 1.5 days (Iterations 4-6)
**Impact**: HIGH - Protects capital during adverse conditions

## Current Problem
Strategy trades identically in bull and bear markets, leading to:
- Drawdowns during bear markets
- Over-exposure in high volatility
- No adaptation to changing conditions

---

## Iteration 4: Implement Regime Detection Logic (0.5 days)

### Goal
Classify market as Bull, Bear, or Sideways based on technical indicators.

### Building Blocks

#### 4.1: Create MarketRegime Module
**File**: `indicators/market_regime.py` (NEW FILE)

**Task**: Create market regime detection system

```python
"""Market regime detection for adaptive trading."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from enum import Enum


class Regime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class MarketRegimeDetector:
    """
    Detect current market regime based on multiple indicators.

    Uses SPY (S&P 500) as market proxy and considers:
    - Moving average alignment (50/200)
    - Trend strength
    - Price position vs MAs
    """

    def __init__(self, benchmark_ticker: str = 'SPY'):
        """
        Initialize regime detector.

        Args:
            benchmark_ticker: Market benchmark (default SPY for stocks)
        """
        self.benchmark_ticker = benchmark_ticker
        self.current_regime = Regime.UNKNOWN
        self.regime_confidence = 0.0
        self.regime_since = None

    def detect_regime(self, benchmark_df: pd.DataFrame) -> Dict:
        """
        Detect current market regime.

        Args:
            benchmark_df: Price data for market benchmark (SPY)

        Returns:
            {
                'regime': Regime enum,
                'confidence': 0-100,
                'since': datetime,
                'scores': {'bull': 0-100, 'bear': 0-100, 'sideways': 0-100},
                'indicators': {...}
            }
        """

        if benchmark_df is None or len(benchmark_df) < 200:
            return self._unknown_regime()

        latest = benchmark_df.iloc[-1]

        # Get required indicators
        sma_50 = latest.get('sma_50')
        sma_200 = latest.get('sma_200')

        if pd.isna(sma_50) or pd.isna(sma_200):
            return self._unknown_regime()

        price = latest['close']
        trend_strength = latest.get('trend_strength', 50)

        # Scoring system (0-100 for bull vs bear)
        bull_score = 0
        bear_score = 0

        # Component 1: MA alignment (40 points)
        if sma_50 > sma_200:
            bull_score += 40  # Golden cross
        else:
            bear_score += 40  # Death cross

        # Component 2: Price position (30 points)
        price_vs_ma50 = price > sma_50
        price_vs_ma200 = price > sma_200

        if price_vs_ma50 and price_vs_ma200:
            bull_score += 30
        elif not price_vs_ma50 and not price_vs_ma200:
            bear_score += 30
        else:
            # Mixed - split the points
            bull_score += 15
            bear_score += 15

        # Component 3: Trend strength (30 points)
        if trend_strength > 70:
            bull_score += 30
        elif trend_strength < 30:
            bear_score += 30
        else:
            # Sideways - distribute based on proximity to 50
            sideways_component = 30 - abs(trend_strength - 50) * 0.6
            bull_score += sideways_component / 2
            bear_score += sideways_component / 2

        # Classify regime
        regime = self._classify_regime(bull_score, bear_score)
        confidence = max(bull_score, bear_score)

        # Update tracking
        if regime != self.current_regime:
            self.regime_since = datetime.now()

        self.current_regime = regime
        self.regime_confidence = confidence

        return {
            'regime': regime,
            'confidence': confidence,
            'since': self.regime_since,
            'scores': {
                'bull': bull_score,
                'bear': bear_score,
                'sideways': 100 - bull_score - bear_score
            },
            'indicators': {
                'sma_50_200_cross': 'bull' if sma_50 > sma_200 else 'bear',
                'price_vs_ma50': 'above' if price_vs_ma50 else 'below',
                'price_vs_ma200': 'above' if price_vs_ma200 else 'below',
                'trend_strength': trend_strength
            }
        }

    def _classify_regime(self, bull_score: float, bear_score: float) -> Regime:
        """Classify regime based on scores."""

        # Strong bull: bull_score >= 70
        if bull_score >= 70:
            return Regime.BULL

        # Strong bear: bear_score >= 70
        if bear_score >= 70:
            return Regime.BEAR

        # Otherwise sideways
        return Regime.SIDEWAYS

    def _unknown_regime(self) -> Dict:
        """Return unknown regime result."""
        return {
            'regime': Regime.UNKNOWN,
            'confidence': 0,
            'since': None,
            'scores': {'bull': 0, 'bear': 0, 'sideways': 0},
            'indicators': {}
        }

    def get_regime_adjustments(self, regime: Regime) -> Dict:
        """
        Get strategy adjustments for regime (hardcoded).

        Returns:
            {
                'risk_multiplier': 0.5-1.0,
                'max_positions': 4-8,
                'should_trade': True/False,
                'signal_threshold': 70-90
            }
        """

        # Hardcoded adjustments per regime
        REGIME_SETTINGS = {
            Regime.BULL: {
                'risk_multiplier': 1.0,
                'max_positions': 8,
                'should_trade': True,
                'signal_threshold': 70,
                'description': 'Normal trading - full risk'
            },
            Regime.SIDEWAYS: {
                'risk_multiplier': 0.75,
                'max_positions': 6,
                'should_trade': True,
                'signal_threshold': 80,
                'description': 'Reduced trading - require stronger signals'
            },
            Regime.BEAR: {
                'risk_multiplier': 0.5,
                'max_positions': 4,
                'should_trade': False,  # Don't trade in bear markets
                'signal_threshold': 90,
                'description': 'Defensive mode - no trading'
            },
            Regime.UNKNOWN: {
                'risk_multiplier': 0.75,
                'max_positions': 6,
                'should_trade': True,
                'signal_threshold': 80,
                'description': 'Cautious mode - insufficient data'
            }
        }

        return REGIME_SETTINGS.get(regime, REGIME_SETTINGS[Regime.UNKNOWN])
```

**Testing**:
```python
# test_market_regime.py
def test_bull_regime_detection():
    # Mock data: price above 50/200 MA, strong trend
    # Should classify as BULL with high confidence

def test_bear_regime_detection():
    # Mock data: price below 50/200 MA, weak trend
    # Should classify as BEAR

def test_sideways_regime_detection():
    # Mock data: mixed signals, moderate trend
    # Should classify as SIDEWAYS
```

**Acceptance Criteria**:
- [ ] Correctly classifies bull markets (score >= 70)
- [ ] Correctly classifies bear markets (score >= 70)
- [ ] Correctly classifies sideways markets
- [ ] Returns confidence score
- [ ] Provides hardcoded regime-specific adjustments
- [ ] Unit tests pass

---

## Iteration 5: Integrate Regime Detection (0.5 days)

### Goal
Fetch benchmark data and detect regime during daily cycle.

### Building Blocks

#### 5.1: Add Regime Detector to Main Strategy
**File**: `main.py`

**Task**: Initialize and update regime detector

```python
# In __init__
from indicators.market_regime import MarketRegimeDetector, Regime

self.regime_detector = MarketRegimeDetector(benchmark_ticker='SPY')
self.current_regime = Regime.UNKNOWN

# Add new method
def update_market_regime(self, date: datetime) -> Dict:
    """
    Update current market regime.

    Returns:
        Regime detection result
    """
    logger.info("Updating market regime...")

    # Fetch benchmark data (need 200+ days for 200 MA)
    lookback = 250
    start_str = (date - timedelta(days=lookback)).strftime('%Y-%m-%d')
    end_str = date.strftime('%Y-%m-%d')

    benchmark_df = self.tiingo_client.get_stock_prices(
        self.regime_detector.benchmark_ticker,
        start_str,
        end_str
    )

    if benchmark_df is None:
        logger.warning("Could not fetch benchmark data for regime detection")
        return self.regime_detector._unknown_regime()

    # Add indicators to benchmark
    from indicators.technical import TechnicalIndicators
    from indicators.momentum import MomentumMetrics

    tech = TechnicalIndicators()
    mom = MomentumMetrics()

    benchmark_df = tech.add_all_indicators(benchmark_df)
    benchmark_df = mom.add_all_momentum_metrics(benchmark_df)

    # Detect regime
    regime_result = self.regime_detector.detect_regime(benchmark_df)

    self.current_regime = regime_result['regime']

    logger.info(
        f"Market Regime: {regime_result['regime'].value.upper()} "
        f"(confidence: {regime_result['confidence']:.0f}%)"
    )

    return regime_result
```

**In `run_daily()`**: Add regime update step

```python
def run_daily(self, date: datetime) -> Dict:
    # ... existing setup ...

    try:
        # NEW: Step 0 - Update market regime
        logger.info("Step 0: Detecting market regime...")
        regime_result = self.update_market_regime(date)
        results['regime'] = regime_result['regime'].value
        results['regime_confidence'] = regime_result['confidence']

        # Check if we should trade in this regime
        adjustments = self.regime_detector.get_regime_adjustments(
            regime_result['regime']
        )

        if not adjustments['should_trade']:
            logger.warning(
                f"Skipping trading today - {regime_result['regime'].value} regime detected"
            )
            results['skipped'] = True
            results['skip_reason'] = f"{regime_result['regime'].value} regime"
            return results

        # Continue with normal trading...
        # 1. Scan for candidates
        # ... existing code ...
```

**Acceptance Criteria**:
- [ ] Fetches SPY data (1 API call per day)
- [ ] Detects regime daily
- [ ] Logs regime to console
- [ ] Skips trading in bear markets
- [ ] Integration test passes

---

## Iteration 6: Apply Regime-Based Adjustments (0.5 days)

### Goal
Dynamically adjust risk parameters based on detected regime.

### Building Blocks

#### 6.1: Update RiskManager to Accept Regime Adjustments
**File**: `strategy/risk_manager.py`

**Task**: Add method to apply regime adjustments

```python
def apply_regime_adjustments(self, regime_adjustments: Dict) -> None:
    """
    Adjust risk parameters based on market regime.

    Args:
        regime_adjustments: Dict with risk_multiplier, max_positions, etc.
    """

    # Store original values on first call
    if not hasattr(self, '_original_risk_per_trade'):
        self._original_risk_per_trade = self.risk_per_trade
        self._original_max_positions = self.max_positions

    # Apply multiplier to risk per trade
    risk_multiplier = regime_adjustments.get('risk_multiplier', 1.0)
    self.risk_per_trade = self._original_risk_per_trade * risk_multiplier

    # Adjust max positions
    self.max_positions = regime_adjustments.get('max_positions', self._original_max_positions)

    logger.info(
        f"Applied regime adjustments: "
        f"risk={self.risk_per_trade*100:.2f}% (×{risk_multiplier}), "
        f"max_positions={self.max_positions}"
    )

def reset_to_defaults(self) -> None:
    """Reset risk parameters to original values."""
    if hasattr(self, '_original_risk_per_trade'):
        self.risk_per_trade = self._original_risk_per_trade
        self.max_positions = self._original_max_positions
        logger.info("Reset risk parameters to defaults")
```

**Testing**:
```python
def test_regime_adjustments():
    # Test applying bull adjustments (1.0x, 8 pos)
    # Test applying sideways adjustments (0.75x, 6 pos)
    # Test applying bear adjustments (0.5x, 4 pos)
    # Test reset to defaults
```

**Acceptance Criteria**:
- [ ] Adjusts risk per trade by multiplier
- [ ] Adjusts max positions
- [ ] Can reset to defaults
- [ ] Tests pass

---

#### 6.2: Apply Adjustments in main.py
**File**: `main.py`

**Task**: Apply regime adjustments after detection

```python
def run_daily(self, date: datetime) -> Dict:
    # ... after regime detection ...

    # Get regime-specific adjustments
    adjustments = self.regime_detector.get_regime_adjustments(
        regime_result['regime']
    )

    # Apply to risk manager
    self.risk_manager.apply_regime_adjustments(adjustments)

    # Note the signal threshold (used in scanning)
    signal_threshold = adjustments['signal_threshold']
    logger.info(f"Using signal threshold: {signal_threshold} for {regime_result['regime'].value} regime")

    # Continue with scanning...
    # NOTE: We still use hardcoded 70 in check_entry_signals
    # In a real implementation, you'd pass signal_threshold through
    # For simplicity, we're keeping 70 hardcoded and using regime to skip trading entirely
```

**Acceptance Criteria**:
- [ ] Adjustments applied each day
- [ ] Risk parameters adjusted based on regime
- [ ] Logs show regime and adjustments
- [ ] End-to-end test passes

---

## Feature 2 Summary

**Deliverables**:
- [x] Iteration 4: Regime detection logic (hardcoded thresholds)
- [x] Iteration 5: Integration with strategy
- [x] Iteration 6: Dynamic parameter adjustment

**Files Modified**:
- `main.py`
- `strategy/risk_manager.py`

**Files Created**:
- `indicators/market_regime.py`

**Testing**:
- Unit tests for regime detection
- Integration test (full daily cycle)

**Expected Outcome**:
- Automatic regime detection
- Skip trading in bear markets
- Reduced risk (0.75x, 6 pos) in sideways markets
- Full risk (1.0x, 8 pos) in bull markets
- Drawdown protection

---

# FEATURE 3: Realistic Stop Loss Modeling

**Goal**: Accurate stop loss execution modeling with slippage and gaps
**Timeline**: 1 day (Iterations 7-8)
**Impact**: MEDIUM-HIGH - Better risk understanding and backtest accuracy

## Current Problem
```python
# signals.py:382-395
if current['low'] <= stop:
    return {'exit_price': stop, ...}  # Assumes perfect fill
```

Reality: Slippage on stops, gaps can blow through stops.

---

## Iteration 7: Implement Stop Loss Slippage (0.5 days)

### Goal
Model realistic stop loss execution with slippage.

### Building Blocks

#### 7.1: Add Stop Loss Slippage Calculation
**File**: `strategy/signals.py`

**Task**: Update `check_stop_loss()` method

```python
def check_stop_loss(
    self,
    position: Position,
    current: pd.Series,
    previous: Optional[pd.Series] = None
) -> Optional[Dict]:
    """
    Check if stop loss triggered with realistic execution modeling.

    Applies hardcoded slippage:
    - Stocks: 0.5% slippage
    - Crypto: 1.0% slippage
    - Extra slippage for gaps

    Args:
        position: Open position
        current: Current bar data
        previous: Previous bar (for gap detection)

    Returns:
        Exit signal if stop triggered
    """

    stop = position.trailing_stop if position.trailing_stop else position.initial_stop

    if current['low'] > stop:
        return None  # Stop not triggered

    # Stop triggered - calculate realistic exit price with slippage

    # Hardcoded base slippage
    if position.asset_type == 'stock':
        base_slippage = 0.005  # 0.5%
    else:
        base_slippage = 0.010  # 1.0%

    # Check for gap down
    gap_slippage = 0
    gap_size = 0
    gap_down = False

    if previous is not None:
        # Detect gap: open significantly below previous close
        gap_size = previous['close'] - current['open']

        if gap_size > 0:
            # Check if it's a significant gap (> 0.5%)
            gap_pct = gap_size / previous['close']

            # Also check vs ATR if available
            if 'atr' in current and not pd.isna(current['atr']):
                gap_threshold = current['atr'] * 1.0  # 1 ATR
                gap_down = gap_size > gap_threshold
            else:
                gap_down = gap_pct > 0.005  # 0.5%

            if gap_down:
                gap_slippage = 0.005  # Additional 0.5% for gaps

                # If gap blows way past stop, add even more slippage
                if current['open'] < stop:
                    blowthrough = (stop - current['open']) / stop
                    if blowthrough > 0.02:  # Blows through by >2%
                        gap_slippage += blowthrough / 2  # Add half the blowthrough
                        logger.warning(
                            f"Large gap blowthrough: ${gap_size:.2f} "
                            f"({blowthrough*100:.1f}% past stop)"
                        )

    # Total slippage
    total_slippage = base_slippage + gap_slippage

    # Realistic exit price (worse than stop)
    realistic_exit = stop * (1 - total_slippage)

    # Floor at daily low (can't exit better than low)
    realistic_exit = max(realistic_exit, current['low'])

    pnl = (realistic_exit - position.entry_price) / position.entry_price

    log_msg = f"Stop loss triggered: "
    if gap_down:
        log_msg += f"GAP DOWN ${gap_size:.2f}, "
    log_msg += (
        f"stop=${stop:.2f}, exit=${realistic_exit:.2f} "
        f"(slippage: {total_slippage*100:.2f}%)"
    )
    logger.info(log_msg)

    return {
        'exit_type': 'stop_loss',
        'exit_price': realistic_exit,
        'stop_price': stop,  # Original stop for reference
        'slippage': total_slippage,
        'gap_down': gap_down,
        'gap_size': gap_size if gap_down else 0,
        'exit_date': current.name,
        'pnl': pnl,
        'days_held': position.days_held
    }
```

**Testing**:
```python
def test_stop_loss_with_slippage():
    # Test normal stop (no gap) - should have 0.5% or 1% slippage
    # Test gap down through stop - should have extra slippage
    # Test different asset types (stock vs crypto)
```

**Acceptance Criteria**:
- [ ] Applies base slippage (stocks 0.5%, crypto 1.0%)
- [ ] Detects gaps using previous bar
- [ ] Adds extra slippage for gaps
- [ ] Never exits better than daily low
- [ ] Logs slippage and gaps
- [ ] Tests pass

---

#### 7.2: Update check_exit_signals to Pass Previous Bar
**File**: `strategy/signals.py`

**Task**: Modify to include previous bar for gap detection

```python
def check_exit_signals(
    self,
    position: Position,
    df: pd.DataFrame,
    current_idx: int
) -> Optional[Dict[str, Any]]:
    """Check all exit conditions for a position."""

    current = df.iloc[current_idx]

    # Get previous bar for gap detection
    previous = df.iloc[current_idx - 1] if current_idx > 0 else None

    # Update position days held
    position.days_held = (df.index[current_idx] - position.entry_date).days

    # 1. Check stop loss with gap detection
    stop_signal = self.check_stop_loss(position, current, previous)
    if stop_signal:
        return stop_signal

    # 2. Check profit target
    target_signal = self.check_profit_target(position, current)
    if target_signal:
        return target_signal

    # ... rest of exit checks ...
```

**Acceptance Criteria**:
- [ ] Previous bar passed to stop check
- [ ] Gap detection works in live flow
- [ ] Integration test passes

---

## Iteration 8: Add Stop Loss Analysis Tool (0.5 days)

### Goal
Create tool to analyze stop loss execution quality.

### Building Blocks

#### 8.1: Create Analysis Script
**File**: `analyze_stop_losses.py` (NEW FILE)

**Task**: Create stop loss analysis utility

```python
#!/usr/bin/env python3
"""
Analyze stop loss execution quality from backtest results.
Shows impact of slippage and gap modeling.
"""

import pandas as pd
import sys


def analyze_stop_losses(trade_log_csv: str):
    """
    Analyze stop loss exits from trade log.

    Args:
        trade_log_csv: Path to trade log CSV
    """

    print("=" * 80)
    print("STOP LOSS EXECUTION ANALYSIS")
    print("=" * 80)

    # Load trades
    df = pd.read_csv(trade_log_csv)

    # Filter to stop loss exits
    stops = df[df['exit_type'] == 'stop_loss'].copy()

    if len(stops) == 0:
        print("\nNo stop loss exits found in trade log.")
        return

    print(f"\nTotal stop loss exits: {len(stops)}")
    print(f"Of {len(df)} total exits ({len(stops)/len(df)*100:.1f}%)")
    print()

    # Analyze slippage
    if 'slippage' in stops.columns:
        print("SLIPPAGE ANALYSIS:")
        print(f"  Average slippage:   {stops['slippage'].mean()*100:.2f}%")
        print(f"  Median slippage:    {stops['slippage'].median()*100:.2f}%")
        print(f"  Max slippage:       {stops['slippage'].max()*100:.2f}%")
        print(f"  Min slippage:       {stops['slippage'].min()*100:.2f}%")
        print()

    # Analyze gaps
    if 'gap_down' in stops.columns:
        gaps = stops[stops['gap_down'] == True]
        print("GAP DOWN ANALYSIS:")
        print(f"  Stops with gaps:    {len(gaps)} ({len(gaps)/len(stops)*100:.1f}%)")

        if len(gaps) > 0 and 'gap_size' in gaps.columns:
            print(f"  Average gap size:   ${gaps['gap_size'].mean():.2f}")
            print(f"  Max gap size:       ${gaps['gap_size'].max():.2f}")
        print()

    # P&L impact
    print("P&L IMPACT:")
    print(f"  Average stop loss:  {stops['pnl'].mean()*100:.2f}%")
    print(f"  Median stop loss:   {stops['pnl'].median()*100:.2f}%")
    print(f"  Worst stop loss:    {stops['pnl'].min()*100:.2f}%")
    print()

    # Asset type breakdown
    if 'asset_type' in stops.columns:
        print("BY ASSET TYPE:")
        for asset_type in stops['asset_type'].unique():
            subset = stops[stops['asset_type'] == asset_type]
            avg_slip = subset['slippage'].mean() if 'slippage' in subset.columns else 0
            print(f"  {asset_type}:")
            print(f"    Count:            {len(subset)}")
            print(f"    Avg slippage:     {avg_slip*100:.2f}%")
            print(f"    Avg P&L:          {subset['pnl'].mean()*100:.2f}%")
        print()

    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_stop_losses.py <trade_log.csv>")
        sys.exit(1)

    analyze_stop_losses(sys.argv[1])
```

**Testing**:
```bash
# After running a backtest
python analyze_stop_losses.py output/trades.csv
```

**Acceptance Criteria**:
- [ ] Analyzes stop loss execution
- [ ] Shows slippage statistics
- [ ] Identifies gap events
- [ ] Calculates P&L impact
- [ ] Works with backtest output

---

## Feature 3 Summary

**Deliverables**:
- [x] Iteration 7: Stop loss slippage modeling (hardcoded 0.5%/1.0%)
- [x] Iteration 8: Gap down detection and analysis tool

**Files Modified**:
- `strategy/signals.py`

**Files Created**:
- `analyze_stop_losses.py`

**Testing**:
- Unit tests for slippage calculation
- Unit tests for gap detection
- Integration test with backtest
- Analysis of stop loss execution

**Expected Outcome**:
- More realistic backtest results
- Better understanding of actual risk
- Gap risk awareness
- Stop losses executed at: stop_price * (1 - slippage)
- Stocks: ~0.5% worse, Crypto: ~1.0% worse, Gaps: additional 0.5%+

---

# INTEGRATION & TESTING

## Integration Phase (0.5 days)

### Goal
Ensure all 3 features work together harmoniously.

### Tasks

#### 1. End-to-End Integration Test

**File**: `tests/test_integration.py` (NEW FILE)

```python
"""Integration tests for all 3 features."""

import pytest
from datetime import datetime, timedelta
from main import MomentumStrategy


class TestFeatureIntegration:
    """Test all 3 features working together."""

    def test_full_workflow(self):
        """
        Test complete workflow:
        1. Detect market regime
        2. Generate flexible signals (70 threshold)
        3. Size positions by signal strength
        4. Execute with realistic stops (slippage)
        """

        strategy = MomentumStrategy('config/strategy_config.yaml')
        assert strategy.initialize_components()

        # Run one daily cycle
        results = strategy.run_daily(datetime.now())

        # Verify regime was detected
        assert 'regime' in results

        # If not skipped, verify signals were generated
        if not results.get('skipped', False):
            assert 'entries' in results
            assert 'exits' in results

    def test_signal_strength_affects_sizing(self):
        """Test that weak signals get smaller positions."""
        # Would need to mock signal generation with different strengths
        # Verify position sizes scale: 70-79=60%, 80-89=80%, 90+=100%

    def test_regime_skips_bear_trading(self):
        """Test that bear regime skips trading."""
        # Mock bear regime detection
        # Verify trading is skipped
```

**Run**: `pytest tests/test_integration.py -v`

---

## Testing Strategy

### Unit Tests
- Each new method has dedicated test
- Test edge cases and error handling
- Mock external dependencies (API calls)
- Target: 80%+ code coverage

### Integration Tests
- Test features working together
- Test configuration loading
- Test backward compatibility (if old defaults used)

### Validation Scripts
- `analyze_stop_losses.py` - analyze stop execution
- Manual testing tools

### Backtest Validation
- Run backtest before and after
- Compare key metrics
- Ensure improvement in:
  - Signal frequency (up 5-10x)
  - Drawdown (down 20-30%)
  - Risk-adjusted returns (Sharpe up)

---

## Success Metrics

### Feature 1: Flexible Signals
- [ ] Signal frequency increases 5-10x
- [ ] Win rate stays >40%
- [ ] Signals have measurable strength scores
- [ ] Position sizing correlates with strength (60%/80%/100%)

### Feature 2: Market Regime
- [ ] Correctly identifies bull/bear/sideways regimes
- [ ] Skips trading in bear markets
- [ ] Reduces risk in sideways (0.75x, 6 pos)
- [ ] Maximum drawdown improves >20%

### Feature 3: Realistic Stops
- [ ] Stop losses show slippage (0.5% stocks, 1% crypto)
- [ ] Gap events are detected and tracked
- [ ] Backtest results more conservative
- [ ] Risk calculations more accurate

---

## FINAL CHECKLIST

Before marking complete:

### Feature 1: Flexible Signals
- [ ] Signal strength calculation implemented
- [ ] 70-point threshold hardcoded
- [ ] Tiered position sizing working (60%/80%/100%)
- [ ] Signal analyzer generating reports
- [ ] All tests pass
- [ ] Documentation complete

### Feature 2: Market Regime
- [ ] Regime detection logic implemented
- [ ] Integration with daily cycle
- [ ] Dynamic parameter adjustment (hardcoded settings)
- [ ] All tests pass
- [ ] Documentation complete

### Feature 3: Realistic Stops
- [ ] Slippage modeling implemented (hardcoded 0.5%/1.0%)
- [ ] Gap detection working
- [ ] Analysis tool created
- [ ] All tests pass
- [ ] Documentation complete

### Integration
- [ ] All features work together
- [ ] Performance acceptable
- [ ] Backtest shows improvement
- [ ] Ready for paper trading

---

## ESTIMATED TIMELINE

| Iteration | Feature | Tasks | Time |
|-----------|---------|-------|------|
| 1 | Flexible Signals | Signal strength scoring | 0.5d |
| 2 | Flexible Signals | Tiered sizing | 0.5d |
| 3 | Flexible Signals | Analysis & reporting | 0.5d |
| 4 | Market Regime | Detection logic | 0.5d |
| 5 | Market Regime | Integration | 0.5d |
| 6 | Market Regime | Dynamic adjustments | 0.5d |
| 7 | Realistic Stops | Slippage modeling | 0.5d |
| 8 | Realistic Stops | Analysis tool | 0.5d |
| - | Integration | Testing & validation | 0.5d |

**Total: 4.5 days**

With buffer for debugging: **3-4 days**

---

## GETTING STARTED

1. **Read this roadmap completely**
2. **Set up development environment**
   ```bash
   git checkout -b feature/flexible-signals-regime-stops
   ```

3. **Start with Iteration 1**
   - Read the iteration goals
   - Implement building blocks in order
   - Test each building block
   - Commit when iteration complete

4. **Proceed sequentially**
   - Don't skip iterations
   - Each builds on previous
   - Test thoroughly before moving on

5. **Track progress**
   - Check off tasks as completed
   - Update this file with notes
   - Document issues encountered

---

**This roadmap uses hardcoded constants throughout - no YAML configuration complexity. Values are embedded directly in the code for simplicity.**
