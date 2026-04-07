# Trade Analysis — Apr 7, 2026

**Dataset:** 16 closed testnet trades (Apr 6–7, 2026)
**Balance:** $5,060 | **Risk per trade:** ~$25 (0.5% of balance)

---

## Summary Stats

| Metric | Value |
|--------|-------|
| Total closed trades | 16 |
| Win rate | 62.5% (10W / 6L) |
| Total realized PnL | +$81.99 |
| Avg win | $13.21 |
| Avg loss | -$8.35 |
| Reward/Risk ratio | 1.58x |

### By Direction
| Direction | Trades | Win Rate | PnL |
|-----------|--------|----------|-----|
| LONG | 6 | **17%** (1/6) | **-$32.83** |
| SHORT | 10 | **90%** (9/10) | **+$114.81** |

### By Asset
| Asset | Closed | Win% | PnL |
|-------|--------|------|-----|
| BTC | 3 | 67% | +$26.75 |
| ETH | 7 | 43% | -$22.89 |
| SOL | 3 | 100% | +$40.91 |
| XRP | 3 | 33% | +$37.22 |

### Exit Reasons
| Reason | Count | PnL |
|--------|-------|-----|
| SL (trailing stop) | 10 | +$86.25 |
| REVERSE_CLOSE | 3 | -$20.75 |
| STAGNANT_EXIT (6h) | 2 | -$3.65 |
| Direct SL hit | 1 | -$25.70 |

---

## Root Cause Analysis

### Root Cause #1 — Tier 1 Confidence Bypasses Regime Check ⚠️ CRITICAL

**The bug:** When model confidence ≥ 0.80 ("Tier 1 autonomous"), the signal gate is
completely skipped — including the regime/trend veto.

During Apr 6 21:00 – Apr 7 04:00 UTC, market was TRENDING_DOWN (ADX 39-41).
Three high-confidence LONG entries slipped through:
- ETH LONG conf=0.992 → Tier 1 autonomous → gate skipped → LOSS
- ETH LONG conf=0.870 → Tier 1 autonomous → gate skipped → LOSS  
- XRP LONG conf=0.816 → Tier 1 autonomous → gate skipped → LOSS

The XRP log explicitly showed the gate blocking the SAME signals at lower confidence:
`"Signal gate BLOCK: LONG conf=0.68 | MTF=❌ bearish vs LONG | REG=❌ TRENDING_DOWN ADX=63"`

**Estimated impact:** -$35 to -$40 from these 3 trades alone.

---

### Root Cause #2 — REVERSE_CLOSE: 0 for 3, -$20.75

Every position flip lost:
- ETH LONG→SHORT (conf=0.863): -$5.98 after only -0.32% adverse move
- BTC LONG→SHORT (conf=0.686): -$5.69 after -0.30% adverse move
- ETH SHORT→LONG (conf=0.849): -$9.08 after -0.50% adverse move

None hit their 1.5% SL. The reversal fires at conf≥0.70 and atomically closes the
existing position AND opens the opposite — double-losing on both sides.

---

### Root Cause #3 — Trailing Stop Closes Winners Too Early

All 10 wins exited via trailing SL, not TP. Avg capture: ~0.5% vs TP target of 3-5%.
Examples:
- SOL SHORT: exited at +0.21%, TP was 5.4% away
- ETH SHORT: exited at +0.37%, TP was 3.0% away

Current settings: `TRAILING_BREAKEVEN_PCT = 0.5%`, `TRAILING_DISTANCE_PCT = 0.3%`
These activate too aggressively, locking profit before the trade has room to run.

**Impact:** Avg win ($13.21) should be materially higher given $25 risk (should be >$25 for positive EV at 60% WR).

---

### Root Cause #4 — ETH LONG Bias in Downtrend

ETH: 7 trades, 43% WR, -$22.89.
The ETH model (fold_08) took 4 LONG entries during a sustained downtrend, all lost.
When it switched to SHORTs on Apr 7 10:48+, it won 2/2.

This is NOT a permanent ETH model problem — it's the same Tier 1 regime bypass issue
(Root Cause #1) showing up disproportionately in ETH because ETH ran longer during the
downtrend period.

---

## Improvement Plan

> **Status: PENDING — waiting for 50+ trades before implementing**
> Reassess on: **Apr 9, 2026**

### Fix #1 — Regime Hard-Veto at ALL Confidence Tiers [HIGHEST IMPACT]

**File:** `live_trading_htf.py`, `_check_signal_gate()` around line 985

**Change:** Before the Tier 1 early-return, add regime veto that applies to all tiers:
```python
# NEW: Regime veto applies even at Tier 1 (conf ≥ 0.80)
regime = self._last_market_signals.get('regime', {})
regime_type = (regime.get('type') or '').upper()
adx = regime.get('adx', 0) or 0
if adx >= 25 and adx < 60:  # Strong trend, not exhaustion
    if direction == 'LONG' and 'DOWN' in regime_type:
        logger.info("🚫 Tier 1 REGIME VETO: LONG vs %s ADX=%.0f", regime_type, adx)
        return False
    elif direction == 'SHORT' and 'UP' in regime_type:
        logger.info("🚫 Tier 1 REGIME VETO: SHORT vs %s ADX=%.0f", regime_type, adx)
        return False
if confidence >= SIGNAL_GATE_AUTONOMOUS:
    return True
```

**Expected benefit:** Blocks ~3/5 losing LONGs. Est. +$35-40 PnL per similar market cycle.

---

### Fix #2 — Reversal Gate: Close Flat, Don't Flip [HIGH IMPACT]

**File:** `live_trading_htf.py`, `execute_trade()`

**Change:** When reversal signal fails gate, close existing position but do NOT open
the opposite. Currently both happen atomically.

```python
if is_reversal and not self._check_signal_gate(action, confidence, is_reversal=True):
    # Close only — don't open opposite
    return self._close_position(current_price, "REVERSAL_BLOCKED_CLOSE", confidence)
```

Also raise reversal threshold: `conf ≥ 0.70` → `conf ≥ 0.82`

**Expected benefit:** Eliminates -$20.75 from 3 double-loss reversals.

---

### Fix #3 — Widen Trailing SL [MEDIUM IMPACT]

**File:** `live_trading_htf.py`, constants section (~line 81)

**Change:**
```python
TRAILING_BREAKEVEN_PCT = 0.008   # was 0.005
TRAILING_DISTANCE_PCT  = 0.005   # was 0.003
```

**Expected benefit:** +$3-5 per winning trade. At 10 wins per 16 trades, +$30-50 per cycle.

---

### Fix #4 — ETH LONG Confidence Floor (Temporary) [MEDIUM IMPACT]

**File:** `live_trading_htf.py`, `execute_trade()`

Add per-symbol directional floor until Fix #1 is validated in production:
```python
SYMBOL_DIRECTIONAL_CONF = {"ETHUSDT": {"LONG": 0.95}}
```

Remove once Fix #1 has 50+ trades of data proving it works.

---

### Fix #5 — Raise Reversal Threshold [LOWER IMPACT]

**File:** `live_trading_htf.py`, line ~1976

Change: `conf ≥ 0.70` → `conf ≥ 0.82`

---

## Notes

- The 1.5% SL is NOT too tight — the one full SL hit was a legitimate loss on a bad entry
- The trailing stop is the issue for winners, not the initial SL placement
- ETH's poor WR is the same problem as Root Cause #1 — fix that, ETH recovers
- Need 50+ closed trades minimum before these fixes can be validated statistically

---

*Analysis generated: 2026-04-07*
*Next review: 2026-04-09*
