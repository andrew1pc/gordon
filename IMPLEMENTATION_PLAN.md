# Implementation Execution Plan

## Overview
Building 3 features across 8 iterations with testing at each step.

**Timeline**: 3-4 days
**Current Status**: Planning phase

---

## Execution Strategy

### Phase 1: Feature 1 - Flexible Entry Signals (Iterations 1-3)
**Goal**: Increase signal frequency 5-10x with tiered position sizing

#### Iteration 1: Signal Strength Scoring (0.5 days)
**Build Steps**:
1. Add `calculate_entry_signal_strength()` to `strategy/signals.py`
2. Modify `check_entry_signals()` to use 70 threshold
3. Write unit tests for signal strength calculation
4. Run tests to verify

**Test Plan**:
- Unit test: All conditions met = 100 points
- Unit test: 4-5 conditions = 70-85 points
- Unit test: Partial credit for volume/momentum
- Integration test: Run analyze_ticker.py on one ticker

**Success Criteria**:
- [ ] Signal strength method returns 0-100
- [ ] 70 threshold enforced
- [ ] All tests pass
- [ ] Can generate signals with strength scores

---

#### Iteration 2: Tiered Position Sizing (0.5 days)
**Build Steps**:
1. Add `calculate_scaled_position_size()` to `strategy/risk_manager.py`
2. Update `validate_new_trade()` to accept signal_strength parameter
3. Update `main.py` to pass signal_strength through
4. Write unit tests for position scaling
5. Run tests to verify

**Test Plan**:
- Unit test: 90+ = 100% size
- Unit test: 80-89 = 80% size
- Unit test: 70-79 = 60% size
- Integration test: Run full scan, verify position sizes scale

**Success Criteria**:
- [ ] Position sizes scale by signal strength
- [ ] Backward compatible (default=100)
- [ ] All tests pass
- [ ] End-to-end test shows scaling

---

#### Iteration 3: Signal Analysis & Reporting (0.5 days)
**Build Steps**:
1. Create `strategy/signal_analyzer.py`
2. Integrate analyzer into `main.py`
3. Write unit tests for analyzer
4. Run full daily cycle and check report

**Test Plan**:
- Unit test: Records signals correctly
- Unit test: Tracks near misses (60-69)
- Unit test: Generates report
- Integration test: Run daily cycle, verify CSV export

**Success Criteria**:
- [ ] Analyzer tracks all signals
- [ ] Report generated
- [ ] CSV exports work
- [ ] No performance degradation

---

### Phase 2: Feature 2 - Market Regime Detection (Iterations 4-6)
**Goal**: Prevent losses in bear markets, adjust risk by regime

#### Iteration 4: Regime Detection Logic (0.5 days)
**Build Steps**:
1. Create `indicators/market_regime.py` with `MarketRegimeDetector` class
2. Implement `detect_regime()` method with scoring
3. Implement `get_regime_adjustments()` with hardcoded settings
4. Write unit tests for regime classification

**Test Plan**:
- Unit test: Bull regime (SMA50>SMA200, price above MAs)
- Unit test: Bear regime (SMA50<SMA200, price below MAs)
- Unit test: Sideways regime (mixed signals)
- Unit test: Returns correct adjustments per regime

**Success Criteria**:
- [ ] Classifies regimes correctly
- [ ] Returns confidence scores
- [ ] Hardcoded adjustments work
- [ ] All tests pass

---

#### Iteration 5: Integrate Regime Detection (0.5 days)
**Build Steps**:
1. Add regime detector initialization to `main.py`
2. Add `update_market_regime()` method
3. Add regime check to `run_daily()`
4. Test with live API call to SPY

**Test Plan**:
- Integration test: Fetch SPY data (1 API call)
- Integration test: Detect current regime
- Integration test: Skip trading in bear market
- Log verification: Check regime logged correctly

**Success Criteria**:
- [ ] SPY data fetches successfully
- [ ] Regime detected and logged
- [ ] Trading skipped if bear market
- [ ] Integration test passes

---

#### Iteration 6: Apply Regime Adjustments (0.5 days)
**Build Steps**:
1. Add `apply_regime_adjustments()` to `strategy/risk_manager.py`
2. Wire adjustments into `run_daily()` in `main.py`
3. Write unit tests for adjustment application
4. Run full cycle and verify risk parameters change

**Test Plan**:
- Unit test: Bull = 1.0x risk, 8 positions
- Unit test: Sideways = 0.75x risk, 6 positions
- Unit test: Bear = 0.5x risk, 4 positions
- Integration test: Verify risk adjusted in logs

**Success Criteria**:
- [ ] Risk parameters adjust by regime
- [ ] Logs show adjustments
- [ ] Tests pass
- [ ] End-to-end test works

---

### Phase 3: Feature 3 - Realistic Stop Modeling (Iterations 7-8)
**Goal**: Model stop loss execution with slippage and gaps

#### Iteration 7: Stop Loss Slippage (0.5 days)
**Build Steps**:
1. Update `check_stop_loss()` in `strategy/signals.py` with slippage
2. Update `check_exit_signals()` to pass previous bar
3. Write unit tests for slippage calculation
4. Test gap detection logic

**Test Plan**:
- Unit test: Normal stop = 0.5%/1.0% slippage
- Unit test: Gap detected correctly
- Unit test: Gap adds extra slippage
- Unit test: Exit floored at daily low

**Success Criteria**:
- [ ] Base slippage applied
- [ ] Gap detection works
- [ ] Extra gap slippage applied
- [ ] Tests pass

---

#### Iteration 8: Stop Loss Analysis Tool (0.5 days)
**Build Steps**:
1. Create `analyze_stop_losses.py`
2. Test with sample trade data
3. Verify output formatting

**Test Plan**:
- Create mock trade CSV
- Run analysis tool
- Verify statistics calculated correctly
- Check output readability

**Success Criteria**:
- [ ] Tool analyzes stop losses
- [ ] Shows slippage stats
- [ ] Identifies gaps
- [ ] Output is clear

---

### Phase 4: Integration & Final Testing (0.5 days)

**Build Steps**:
1. Create `tests/test_integration.py`
2. Run full end-to-end test
3. Run full backtest (if time permits)
4. Generate final report

**Test Plan**:
- Integration test: All 3 features work together
- Integration test: Regime detected, signals generated, stops have slippage
- Performance test: No major slowdown
- Validation: Signal frequency increased 5-10x

**Success Criteria**:
- [ ] All features integrated
- [ ] All tests pass
- [ ] Performance acceptable
- [ ] Metrics improved

---

## Testing Approach

### Unit Tests
- Test individual methods in isolation
- Mock external dependencies (API calls)
- Focus on edge cases
- Fast execution (<1 second per test)

### Integration Tests
- Test multiple components together
- Use real API calls sparingly (rate limits)
- Test full workflows
- Verify data flows correctly

### Manual Validation
- Run analyze_ticker.py on real ticker
- Check logs for expected behavior
- Verify CSV exports
- Validate against known scenarios

---

## Rollback Plan

If any iteration fails:
1. Review error logs
2. Fix issue
3. Re-run tests
4. If unfixable, document and move to next iteration
5. Circle back after completing other features

---

## Progress Tracking

I'll update this file after each iteration with:
- ✅ Completed iterations
- 🔄 Current iteration
- ⏸️ Blocked items
- 📝 Notes/issues encountered

---

## Ready to Begin

Starting with **Iteration 1: Signal Strength Scoring**
