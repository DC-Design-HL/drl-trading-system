# Decision-Making System Analysis & Recommendations

**Date:** 2026-03-15
**Current Status:** System is very conservative, not trading frequently

---

## Current Decision-Making Architecture

### 1. Signal Weights (live_trading_multi.py)

```python
Whale:      30.0%  ⚠️ ISSUE: Data 4-5 days stale!
Regime:     21.0%  ✅
TFT:        21.0%  ✅
Funding:    15.0%  ✅
Order Flow: 15.0%  ✅
News:        0.0%  ✅ Disabled (correct)
────────────────
Total:     102.0% (auto-normalized to 100%)
```

### 2. Decision Thresholds

**Market Action Determination:**
```python
if composite_score > 0.10:    market_action = BUY
elif composite_score < -0.10: market_action = SELL
else:                         market_action = NEUTRAL/HOLD
```

**Confidence Tiers (3-tier system):**
```python
Tier 1 (conf >= 0.65): SIGNAL OVERRIDE - Ignore PPO, trade on signals
Tier 2 (conf >= 0.45): CONSENSUS - PPO must agree with signals
Tier 3 (conf <  0.45): HOLD - Too uncertain, don't trade
```

### 3. Composite → Confidence Mapping

```python
For BUY action:
  confidence = (composite + 1) / 2

  Examples:
  composite = +0.50 → confidence = 0.75 (Tier 1: Signal Override)
  composite = +0.30 → confidence = 0.65 (Tier 1: Signal Override)
  composite = +0.10 → confidence = 0.55 (Tier 2: Consensus needed)
  composite = +0.05 → confidence = 0.525 (Tier 2: Consensus needed)
  composite = -0.10 → confidence = 0.45 (Tier 2: Consensus needed)
```

---

## Issues Identified

### 🚨 Issue #1: Whale Weight Too High for Stale Data

**Problem:**
- Whale signals have 30% weight (highest of all signals)
- But whale data is 4-5 days old (last updated March 10-11)
- Trading decisions based on outdated whale positions

**Impact:**
```
Example: BTC bullish scenario
- Regime: +0.8 (strong uptrend) × 21% = 0.168
- TFT: +0.6 (bullish forecast) × 21% = 0.126
- Whale: -0.09 (stale bearish) × 30% = -0.027 ← PROBLEM!
- Composite: 0.267 → confidence 0.63 (barely Tier 2)

If whale weight was 15% instead:
- Whale: -0.09 × 15% = -0.014
- Composite: 0.308 → confidence 0.65 (Tier 1!)
```

**Recommendation:** Reduce whale weight from 30% → 15-20%

---

### ⚠️ Issue #2: Market Action Threshold Too Conservative

**Problem:**
- Composite score must be >0.10 or <-0.10 to be considered bullish/bearish
- Scores between -0.10 and +0.10 are "neutral"
- This is too strict - missing borderline opportunities

**Impact:**
```
Scenario: Moderate bullish signals
- Composite: +0.08 → Considered NEUTRAL (market_action = 0)
- But this is actually slightly bullish!
- System won't trade even if PPO says BUY

Scenario: With lower threshold (0.05)
- Composite: +0.08 → Considered BULLISH (market_action = 1)
- System can trade if PPO agrees
```

**Current Thresholds:**
- Neutral zone: -0.10 to +0.10 (20% range)
- Requires strong signals to escape neutral

**Recommendation:** Lower threshold from 0.10 → 0.05 or 0.07

---

### ⚠️ Issue #3: Confidence Tiers May Be Too Strict

**Problem:**
- Need confidence >= 0.65 for Signal Override (Tier 1)
- Need confidence >= 0.45 for Consensus (Tier 2)
- With conservative composite→confidence mapping, rarely hitting Tier 1

**Impact:**
```
To reach Tier 1 (conf >= 0.65):
  For BUY: composite must be >= +0.30
  For SELL: composite must be <= -0.30

Current reality (from logs):
  Typical composite: -0.10 to +0.20
  Typical confidence: 0.45 to 0.60 (stuck in Tier 2)

Result: PPO model always has veto power, signals can't override
```

**Recommendation:**
- Lower Tier 1 threshold: 0.65 → 0.60
- Lower Tier 2 threshold: 0.45 → 0.40

---

### ⚠️ Issue #4: No Asset-Specific Tuning

**Problem:**
- All assets (BTC, ETH, SOL, XRP) use same weights and thresholds
- But they have very different characteristics:
  - BTC: Reliable, stable (win rate ~40-50%)
  - ETH: Good performer
  - SOL: Volatile
  - XRP: 75% loss rate (already disabled)

**Recommendation:**
Add asset-specific weight adjustments:
```python
if symbol == 'BTCUSDT':
    # BTC: Increase regime weight (trends are reliable)
    regime_weight = 0.25  # +4% from 21%
    whale_weight = 0.15   # -15% from 30%

elif symbol == 'ETHUSDT':
    # ETH: Balanced
    # Use default weights

elif symbol == 'SOLUSDT':
    # SOL: Increase funding/order flow (captures volatility better)
    funding_weight = 0.18  # +3% from 15%
    order_flow_weight = 0.18  # +3% from 15%
    whale_weight = 0.20  # -10% from 30%
```

---

### ⚠️ Issue #5: Whale Tracker Internal Weights Inconsistency

**Current State:**
- Inside `whale_tracker.py`: whale_patterns = 10% (we reduced it today)
- In `live_trading_multi.py`: whale (composite) = 30%

**Issue:**
- These are two different weighting systems
- But having 30% on a stale whale composite is still too high

**Recommendation:**
Reduce whale weight in live_trading_multi.py to match reduced importance

---

## Recommended Changes

### 📊 Priority 1: Adjust Signal Weights (HIGH IMPACT)

**Current:**
```python
scores = {
    'whale': (whale_score, 0.30),      # ← TOO HIGH for stale data
    'regime': (regime_score, 0.21),
    'tft': (tft_score, 0.21),
    'funding': (funding_score, 0.15),
    'order_flow': (flow_score, 0.15),
}
```

**Recommended (Option A - Conservative):**
```python
scores = {
    'whale': (whale_score, 0.20),      # Reduced 30% → 20%
    'regime': (regime_score, 0.22),    # Increased
    'tft': (tft_score, 0.22),          # Increased
    'funding': (funding_score, 0.18),  # Increased
    'order_flow': (flow_score, 0.18),  # Increased
}
```

**Recommended (Option B - Aggressive, for more trades):**
```python
scores = {
    'whale': (whale_score, 0.15),      # Reduced 30% → 15%
    'regime': (regime_score, 0.25),    # Increased (most reliable)
    'tft': (tft_score, 0.25),          # Increased (ML forecast)
    'funding': (funding_score, 0.17),  # Increased
    'order_flow': (flow_score, 0.18),  # Increased
}
```

---

### 📊 Priority 2: Lower Market Action Threshold (MEDIUM IMPACT)

**Current:**
```python
if composite_score > 0.10:
    market_action = 1  # BUY
elif composite_score < -0.10:
    market_action = 2  # SELL
else:
    market_action = 0  # NEUTRAL
```

**Recommended:**
```python
if composite_score > 0.05:  # Lowered from 0.10
    market_action = 1  # BUY
elif composite_score < -0.05:  # Lowered from -0.10
    market_action = 2  # SELL
else:
    market_action = 0  # NEUTRAL
```

**Impact:** System becomes more responsive, catches borderline opportunities

---

### 📊 Priority 3: Adjust Confidence Tiers (MEDIUM IMPACT)

**Current:**
```python
if confidence >= 0.65:  # Tier 1: Signal Override
    return market_action, ...
if confidence >= 0.45:  # Tier 2: Consensus
    ...
```

**Recommended:**
```python
if confidence >= 0.60:  # Tier 1: Signal Override (lowered from 0.65)
    return market_action, ...
if confidence >= 0.40:  # Tier 2: Consensus (lowered from 0.45)
    ...
```

**Impact:** Signals have more authority, PPO has less veto power

---

### 📊 Priority 4: Add Asset-Specific Weights (LOW IMPACT, COMPLEX)

**Current:**
- All assets use same weights

**Recommended:**
```python
def get_signal_weights(symbol: str) -> dict:
    """Get asset-specific signal weights."""

    # Default weights
    weights = {
        'whale': 0.20,
        'regime': 0.22,
        'tft': 0.22,
        'funding': 0.18,
        'order_flow': 0.18,
    }

    # Asset-specific adjustments
    if symbol == 'BTCUSDT':
        # BTC: Strong trend follower, reliable regime signals
        weights['regime'] = 0.25
        weights['tft'] = 0.25
        weights['whale'] = 0.15
        weights['funding'] = 0.17
        weights['order_flow'] = 0.18

    elif symbol == 'SOLUSDT':
        # SOL: Volatile, better with real-time signals
        weights['funding'] = 0.20
        weights['order_flow'] = 0.20
        weights['whale'] = 0.18
        weights['regime'] = 0.21
        weights['tft'] = 0.21

    return weights
```

---

## Expected Impact of Changes

### Before (Current):
```
Typical composite scores: -0.10 to +0.20
Typical confidence: 0.45 to 0.60
Tier distribution:
  - Tier 1 (Signal Override): ~10% of time
  - Tier 2 (Consensus): ~60% of time
  - Tier 3 (Hold): ~30% of time

Result: Very conservative, few trades
```

### After (with Priority 1 + 2 + 3):
```
Expected composite scores: -0.15 to +0.30 (wider range)
Expected confidence: 0.50 to 0.70 (higher)
Tier distribution:
  - Tier 1 (Signal Override): ~25% of time ↑
  - Tier 2 (Consensus): ~55% of time
  - Tier 3 (Hold): ~20% of time ↓

Result: More responsive, increased trade frequency
```

---

## Implementation Plan

### Phase 1: Quick Fixes (Immediate)
1. ✅ **Reduce whale weight to 20%**
   - File: `live_trading_multi.py` line 490
   - Change: `scores['whale'] = (whale_score, 0.20)`  # was 0.30

2. ✅ **Lower market action threshold to 0.05**
   - File: `live_trading_multi.py` line 399
   - Change: `if composite_score > 0.05:`  # was 0.10

3. ✅ **Lower confidence tiers**
   - File: `live_trading_multi.py` line 415
   - Change: `if confidence >= 0.60:`  # was 0.65
   - Line 423: `if confidence >= 0.40:`  # was 0.45

### Phase 2: Weight Redistribution (Next)
4. **Redistribute 10% from whale to other signals**
   - Regime: 21% → 23%
   - TFT: 21% → 23%
   - Funding: 15% → 17%
   - Order Flow: 15% → 17%

### Phase 3: Asset-Specific (Optional, Later)
5. **Add asset-specific weight function**
   - Create `get_signal_weights(symbol)` method
   - Apply different weights for BTC/ETH/SOL

---

## Testing Strategy

### Before deploying changes:
1. **Backtest with new weights** on 180-day period
2. **Compare metrics:**
   - Win rate (target: >55%)
   - Sharpe ratio (target: >1.2)
   - Trade frequency (target: 2-3x current)
   - Max drawdown (target: <20%)

3. **Deploy to dev Space first**
4. **Monitor for 48 hours**
5. **If successful, deploy to production**

---

## Risk Assessment

**Low Risk:**
- Priority 1, 2, 3 (weight and threshold adjustments)
- Easy to revert if issues occur
- Incremental changes, not radical redesign

**Medium Risk:**
- Priority 4 (asset-specific weights)
- More complex logic
- Needs thorough testing per asset

**Mitigation:**
- Deploy to dev Space first
- Monitor closely for 48h
- Keep Quick Wins safety features active
- Can quickly revert by reverting git commit

---

## Recommendation Summary

**For immediate deployment (safe, high impact):**

1. **Reduce whale weight: 30% → 20%**
   - Rationale: Data is stale, shouldn't dominate decisions

2. **Lower market threshold: 0.10 → 0.05**
   - Rationale: Catch more opportunities, less conservative

3. **Lower confidence tiers: 0.65/0.45 → 0.60/0.40**
   - Rationale: Give signals more authority

**Expected result:**
- ✅ 2-3x more trades
- ✅ Better responsiveness to market conditions
- ✅ Still conservative (Quick Wins active)
- ✅ Easy to revert if needed

These changes make the system less dependent on stale whale data and more responsive to real-time signals while maintaining safety through Quick Wins.
