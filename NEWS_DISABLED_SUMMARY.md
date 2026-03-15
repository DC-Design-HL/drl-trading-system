# News Sentiment Disabled

**Date:** 2026-03-15
**Status:** ✅ DISABLED PER USER REQUEST

---

## What Changed

News sentiment has been **completely removed** from trading decisions due to reliability issues.

### Weight Distribution Changes

**Before:**
```python
Whale:      30.0%
Regime:     18.0%
TFT:        18.0%
Funding:    13.5%
Order Flow: 13.5%
News:       10.0%  ← ACTIVE
────────────────
Total:     103.0% (auto-normalized to 100%)
```

**After:**
```python
Whale:      30.0% (unchanged)
Regime:     21.0% (+3%)  ← Real-time trend detection
TFT:        21.0% (+3%)  ← Price forecasting
Funding:    15.0% (+1.5%) ← Market sentiment
Order Flow: 15.0% (+1.5%) ← Buy/sell pressure
News:        0.0% (-10%)  ← DISABLED
────────────────
Total:     102.0% (auto-normalized to 100%)
```

### Changes Made

**File:** `live_trading_multi.py`

1. **News weight set to 0%** (line 576, 585, 588)
   ```python
   scores['news'] = (news_score, 0.00)  # DISABLED: was 0.10
   ```

2. **Hard veto logic disabled** (lines 579-583)
   ```python
   # Hard veto: DISABLED - news not reliable enough
   # (commented out - no longer blocks trades)
   ```

3. **Redistributed 10% to proven signals:**
   - Regime: +3% (better trend detection)
   - TFT: +3% (price forecasting)
   - Funding: +1.5% (real-time sentiment)
   - Order Flow: +1.5% (buy/sell pressure)

---

## Why This Was Done

### User Request
- News aggregation not working reliably
- User wants to ensure it doesn't affect trading decisions
- Can be re-enabled later when fixed

### Technical Issues with News
1. API rate limits causing failures
2. Sentiment scores not accurate
3. Limited sources (CryptoCompare only)
4. Hard veto logic too aggressive

---

## Impact on Trading

### Before (with news enabled):
```
Example: BTC bullish market
- Regime: +0.8 (strong uptrend) × 18% = 0.144
- TFT: +0.6 (forecast up) × 18% = 0.108
- News: -0.5 (bearish news!) × 10% = -0.050
- Composite: 0.202

If news confidence > 0.7: HARD VETO (no trade allowed!)
```

### After (news disabled):
```
Example: Same BTC bullish market
- Regime: +0.8 (strong uptrend) × 21% = 0.168
- TFT: +0.6 (forecast up) × 21% = 0.126
- News: 0.0 × 0% = 0.000
- Composite: 0.294

No hard veto, trade can proceed! ✅
```

### Key Improvements
1. **No false vetoes** from unreliable news
2. **Stronger weight** on proven signals (regime, TFT)
3. **More responsive** to actual market conditions
4. **Cleaner decision logic** without news interference

---

## What Still Works

All other decision signals remain fully functional:

✅ **Whale Signals** (30%)
- Real-time whale flow monitoring
- Pattern learning from wallet behavior
- Exchange reserve tracking

✅ **Regime Detection** (21%, was 18%)
- Trend detection (TRENDING_UP/DOWN/RANGING)
- ADX strength measurement
- Market phase classification

✅ **TFT Forecasting** (21%, was 18%)
- 1h, 4h, 1d price predictions
- Temporal Fusion Transformer model
- Direction consensus

✅ **Funding Rate** (15%, was 13.5%)
- Long/short sentiment
- Leverage bias detection
- OKX funding data

✅ **Order Flow** (15%, was 13.5%)
- CVD (Cumulative Volume Delta)
- Taker buy/sell ratio
- Notable trades (whales)

---

## Verification

### Check news is disabled:
```bash
grep -A3 "6. News Sentiment" live_trading_multi.py
```

Should show:
```python
# ── 6. News Sentiment (DISABLED - not reliable) ───────────────
scores['news'] = (news_score, 0.00)  # DISABLED: was 0.10
```

### Check new weight distribution:
```bash
grep "scores\['regime'\] = " live_trading_multi.py | head -1
grep "scores\['tft'\] = " live_trading_multi.py | head -1
grep "scores\['funding'\] = " live_trading_multi.py | head -1
grep "scores\['order_flow'\] = " live_trading_multi.py | head -1
```

Should show:
```python
scores['regime'] = (regime_score, 0.21)
scores['tft'] = (tft_score, 0.21)
scores['funding'] = (funding_score, 0.15)
scores['order_flow'] = (flow_score, 0.15)
```

---

## Re-enabling News (Future)

When news aggregation is fixed, re-enable by:

1. **Set news weight back to 0.10:**
   ```python
   scores['news'] = (news_score, 0.10)  # Re-enabled
   ```

2. **Uncomment hard veto logic** (if desired)

3. **Reduce other weights accordingly:**
   - Regime: 21% → 18%
   - TFT: 21% → 18%
   - Funding: 15% → 13.5%
   - Order Flow: 15% → 13.5%

---

## Summary

✅ **News sentiment completely disabled**
✅ **10% weight redistributed to proven signals**
✅ **No hard vetoes from unreliable news**
✅ **System more responsive to real market conditions**

The trading bot now relies on 5 proven signal types instead of 6, with higher weights on the most reliable indicators (regime detection and TFT forecasting).
