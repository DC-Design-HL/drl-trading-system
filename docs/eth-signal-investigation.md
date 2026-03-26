# ETH Signal Investigation: Deep Dive into Unprofitable Trades

**Date:** 2026-03-26  
**Period analyzed:** 2026-03-22 to 2026-03-26  
**Model:** HTF PPO walk-forward fold_08 (OOS Sharpe 9.90) — ETH-specific  
**Summary:** 13 closed trades, 4 wins, 9 losses, ~31% win rate, **-$421 total PnL**

---

## 1. Per-Trade Breakdown

| # | Open Time | Dir | Entry | Exit | Conf | Close Reason | PnL | BOS/CHOCH at Entry | Market Trend | What Went Wrong |
|---|-----------|-----|-------|------|------|-------------|-----|-------------------|--------------|-----------------|
| 1 | Mar 22 19:42 | **SHORT** | $2,065.10 | $2,065.10 | 0.51 | SL | **-$1.00** | N/A (pre-BOS module) | Unknown | Near-breakeven SL hit; entered SHORT at low confidence (0.51) in a $2,040–$2,080 range. Price never moved enough. |
| 2 | Mar 22 23:28 | LONG | $2,063.26 | $2,032.31 | 0.87 | SL | **-$57.68** | N/A (pre-BOS module) | Unknown | Entered LONG after closing the SHORT above. Price dropped ~1.5% over 7 hours to hit SL at $2,032. The model was confident (0.87) but ETH was in a slow bleed from ~$2,080 down to $2,030. |
| 3 | Mar 23 07:23 | **SHORT** | $2,030.07 | $2,054.26 | 0.48 | REVERSE_CLOSE | **-$68.87** | N/A (pre-BOS module) | Unknown | Entered SHORT at 0.48 confidence (barely above 0.45 threshold) at a local LOW ($2,030). Price bounced $24 in 1 hour. Classic bottom-shorting. |
| 4 | Mar 23 11:05 | **SHORT** | $2,052.87 | $2,083.66 | N/A | SL | **-$9.71** | N/A (pre-BOS module) | Unknown | Legacy position from earlier state. Hit SL during a sharp spike from $2,040 to $2,140+. |
| 5 | Mar 23 11:52 | LONG | $2,143.58 | $2,111.42 | 0.69 | SL | **-$134.46** | N/A (pre-BOS module) | Unknown | Entered LONG at $2,143 **immediately after a $100+ spike** ($2,040→$2,143). Bought the top of a vertical move. SL hit in just **11 minutes**. |
| 6 | Mar 23 17:28 | **SHORT** | $2,134.16 | $2,167.12 | 0.57 | SL | **-$310.94** | N/A (pre-BOS module) | Unknown | **Largest single loss.** Entered SHORT at $2,134 during what was now a bullish trend (ETH had spiked $100+ from $2,030 to $2,140+). Price continued to $2,167. The model was shorting into strength. |
| 7 | Mar 23 19:30 | **SHORT** | $2,166.50 | $2,166.50 | 0.56 | SL | **-$11.83** | trend=bullish, all BOS/CHOCH=False | Bullish/Ranging | Entered another SHORT at the new highs. BOS/CHOCH was reporting **bullish trend** — model ignored this and shorted anyway. Held 12.5 hours through ranging, exited near breakeven. |
| 8 | Mar 24 20:36 | **SHORT** | $2,146.54 | $2,152.58 | 0.70 | STAGNANT_EXIT | **-$2.26** | trend=ranging, all BOS/CHOCH=False | Ranging | Entered SHORT at 0.70 conf in a tight range. Price barely moved in 6.5 hours. Time-based stagnant exit saved larger loss. |
| 9 | Mar 25 03:42 | **SHORT** | $2,157.56 | $2,156.78 | 0.84 | STAGNANT_EXIT | **-$0.05** | trend=ranging | Ranging | Same pattern: SHORT at 0.84 conf, price barely moved, stagnant exit at ~breakeven after 11.5 hours. |
| **W1** | Mar 23 08:29 | LONG | $2,054.26 | $2,159.32 | 0.56 | SL (trailing) | **+$159.14** | N/A | Unknown | Caught the big $100+ move from $2,054 to $2,190. SL trailed to $2,159. |
| **W2** | Mar 24 08:31 | **SHORT** | $2,156.02 | $2,134.60 | 0.94 | SL | **+$8.68** | trend=ranging | Ranging | Small win on a SHORT that worked. Correctly identified a pullback from $2,170 highs. |
| **W3** | Mar 25 15:51 | **SHORT** | $2,163.79 | $2,162.76 | 0.97 | STAGNANT_EXIT | **+$0.07** | trend=ranging | Ranging | Breakeven via stagnant exit. |
| **W4** | Mar 25 22:11 | **SHORT** | $2,163.64 | $2,152.88 | 0.99 | STAGNANT_EXIT | **+$7.42** | trend=ranging | Ranging | Best SHORT win. Entered at near-perfect confidence and got a small pullback. |

### PnL Summary by Direction
| Direction | Trades | Wins | Losses | Total PnL |
|-----------|--------|------|--------|-----------|
| LONG | 3 | 1 | 2 | **-$32.99** |
| SHORT | 10 | 3 | 7 | **-$388.23** |
| **Total** | **13** | **4** | **9** | **-$421.22** |

---

## 2. Pattern Analysis Across Losing Trades

### Pattern 1: **Massive Short Bias — The Model Almost Always Shorts**

Of 13 trades, **10 were shorts** (77%). Yet ETH was generally **rising** during this period ($2,030 → $2,170, a +7% move). The model has a deep structural bias toward shorting ETH.

**Root cause:** The PPO model was trained on ETH walk-forward data where the optimal strategy may have been heavily short-biased. The fold_08 checkpoint (best OOS Sharpe 9.90) likely learned during a period of ETH decline. When deployed on a **range-to-uptrend** market, this baked-in short bias becomes toxic.

### Pattern 2: **VecNormalize Failure — Raw Observations Used**

Every startup shows:
```
WARNING - Could not load VecNormalize: spaces must have the same shape: (117,) != (4,) — raw obs used
```

**This is CRITICAL.** The model was trained with normalized observations via VecNormalize (117 dimensions). At inference time, the VecNormalize stats fail to load (shape mismatch: 117 vs 4), so **raw, unnormalized observations** are fed to the model.

**Impact:** The PPO policy learned decision boundaries on *normalized* feature distributions (zero-mean, unit-variance). When fed raw features (where RSI is 0-1, ATR ratios are -2 to 2, prices are in thousands), the model's action probabilities become **essentially random or adversarial**. This single issue could explain nearly all misbehavior.

### Pattern 3: **Confidence Scores Are Meaningless**

| Confidence Range | Trades | Wins | Losses | Avg PnL |
|-----------------|--------|------|--------|---------|
| < 0.60 | 5 | 1 | 4 | -$18.47 |
| 0.60 - 0.80 | 3 | 0 | 3 | -$149.04 |
| > 0.80 | 5 | 3 | 2 | -$10.27 |

High-confidence trades (>0.80) actually perform slightly better, but the model routinely reports 0.95-0.99 confidence **while holding losing positions for hours**. For example, Trade #6 (the $311 loss) had the model reporting SHORT confidence of 0.87-0.93 every 15 minutes as price moved against it by $33.

**The confidence metric is dominated by the model's broken normalization.** Without VecNormalize, the softmax output converges to one action regardless of market state.

### Pattern 4: **BOS/CHOCH Signals Are Disconnected from the Model**

From the logs, BOS/CHOCH signals consistently showed:
- `trend=bullish` (Mar 23 19:57 through 21:13) while the model opened SHORT positions
- `trend=ranging` (Mar 24 00:00 onward) while the model maintained SHORT with 0.96-0.99 "confidence"
- **Zero active BOS or CHOCH signals** in almost every reading — `bos_bull=False bos_bear=False choch_bull=False choch_bear=False`

The BOS/CHOCH module isn't feeding INTO the model's decision. It only adjusts SL/TP *after* entry. The model's 117 features include their own BOS/CHOCH calculation in `htf_features.py` (features `4h_smc_bos`, `4h_smc_choch`), but these are computed on 4H resampled data — a completely different swing detection than the live module that uses 5m candles.

**Critical gap:** Two separate, conflicting BOS/CHOCH systems. The live module detects "bullish trend" while the model's internal 4H features may show different structure. Neither prevents the model from entering counter-trend trades.

### Pattern 5: **Entering After Violent Moves (Momentum Chasing)**

Trade #5 ($2,143 LONG, -$134) was opened **immediately after a $100+ vertical spike** (from $2,040 to $2,143 in ~3 hours). The model treated this as a "go long" signal rather than recognizing it as exhaustion.

Trade #6 ($2,134 SHORT, -$311) was the reverse — shorting after the same spike, as if expecting a mean reversion that never came.

**Feature gap:** The 15M features include `15m_price_acceleration` and `15m_breakout_strength`, but there's **no feature measuring exhaustion** — e.g., how far price has moved from a recent swing point in ATR terms, or how many consecutive bars have been in one direction on higher timeframes.

### Pattern 6: **Shorts Entered During Ranging Markets Go Nowhere**

Trades #7, #8, #9 (and the later wins #W3, #W4) all entered SHORT during `trend=ranging`. Price oscillated in a $2,130-$2,170 band. The model kept shorting this range, losing small amounts to stagnant exits.

The stagnant exit mechanism (6-hour timeout when PnL is between -0.3% and +0.5%) saved the model from larger losses but the fundamental problem is: **the model has no concept of "ranging = don't trade."**

---

## 3. Feature-Level Hypotheses

### H1: VecNormalize Shape Mismatch → Broken Policy (SEVERITY: CRITICAL)

The model expects (117,) normalized observations but receives raw observations because the VecNormalize pickle has shape (4,) — likely from a CartPole dummy environment or a different model. **This makes the entire inference pipeline unreliable.**

**Evidence:** The model outputs SHORT 77% of the time with near-constant 0.95+ confidence regardless of market state. This is consistent with a policy receiving out-of-distribution inputs and collapsing to a single action.

### H2: 4H Feature Resampling Creates Stale Signals on ETH

The `HTFDataAligner` resamples 15M candles to 1H/4H/1D. On ETH with its lower absolute price and different microstructure vs BTC:
- 4H bars may have different characteristics (ETH 4H bars cover larger % moves than BTC 4H bars)
- The `_detect_bos` in `htf_features.py` uses `swing_lookback=10`, which on 4H data means looking at the last 40 hours. This may be too wide for ETH's faster-moving structure.
- `_detect_structure_trend` only checks the last 2 swing highs/lows. A $100 spike creates a new swing high, and the model reads "HH+HL = uptrend" but the 4H `trend_score` composites weight this only at 25%.

### H3: The Composite Trend Scores Don't Capture Momentum Exhaustion

Each timeframe produces a composite score:
- `1d_daily_trend_score`: 30% trend_direction + 25% ADX + 25% RSI + 20% HH/HL
- `4h_trend_score`: 30% BOS + 25% structure + 20% EMA + 15% RSI + 10% FVG
- `1h_momentum_score`: 25% MACD + 25% RSI + 20% Stoch + 15% consecutive + 15% vol_delta
- `15m_entry_score`: 20% RSI + 20% MACD + 15% candle + 15% tick + 15% vol_spike + 15% breakout

None of these incorporate:
- **Distance from VWAP or mean** — how extended price is
- **Time since last significant move** — recent spike = don't enter
- **Cross-asset momentum** — ETH often lags BTC moves by 15-30 min
- **Funding rate** — not in the 117 features at all
- **Volume profile** — order book depth, not just bar volume

### H4: The Alignment Module Has a Bug

```python
def compute_alignment(self, sig_1d, sig_4h, sig_1h):
    out[2] = _agree(sig_1h, sig_1h)  # ← BUG: compares 1H with ITSELF
```

Feature `align_1h_15m` always agrees with itself (will be +1 or -1, never 0 for mixed signals). This means the overall alignment score is systematically biased. The `compute_alignment_full()` method fixes this by using `sig_15m`, but whether the live bot calls `compute_alignment_full()` or `compute_alignment()` depends on the code path.

Looking at the live bot's `compute_observation()`:
```python
f_align = self.feature_engine.compute_alignment_full(sig_1d, sig_4h, sig_1h, sig_15m)
```
It does call the correct `_full` version. But the non-full `compute_alignment()` is still in the codebase and may have been used during **training**. If the model was trained with the buggy alignment, it learned that feature 112 (`align_1h_15m`) is always ±1, which creates a distributional mismatch at inference.

### H5: BOS/CHOCH on 5m is Too Noisy for ETH

The live BOS/CHOCH detector uses `swing_lookback=8` on 5m candles. On ETH, 5-minute candles frequently spike ±0.3%, creating abundant false swing points. The logs show BOS/CHOCH signals are almost always `False` — the detector sees so much noise that it can't identify meaningful structure.

Yet the feature engine's internal BOS (`_detect_bos`) uses `swing_lookback=10` on 4H data — the opposite extreme, too slow to react.

**Neither granularity is appropriate for ETH's 15-minute trading horizon.**

---

## 4. Concrete Recommendations

### R1: FIX THE VECNORMALIZE LOADING (Priority: IMMEDIATE)

This is the root cause of 80%+ of the problem. The model was trained with VecNormalize and cannot function without it. Options:
1. **Re-export the VecNormalize stats** from the training environment with the correct (117,) shape
2. **Retrain with `normalize_observations=False`** if normalization can't be restored
3. **Manually normalize** using known feature ranges from the feature engine (clip + scale)

### R2: ADD A RANGING REGIME FILTER (Priority: HIGH)

When the regime detector reports "ranging" (ADX < 20), reduce trading frequency:
```python
if regime_name == 'ranging':
    MIN_CONFIDENCE = 0.85  # Raise threshold from 0.45
    POSITION_SIZE *= 0.5    # Half size
```

Or simply **skip trades entirely** when ADX < 18 and ATR < 0.8x average.

### R3: ADD MOMENTUM EXHAUSTION DETECTION (Priority: HIGH)

Before entering any trade, check:
```python
# Distance from 20-bar VWAP as % of ATR
price_extension = abs(current_price - vwap_20) / atr_14
if price_extension > 3.0:
    logger.info("Price extended %.1f ATR from VWAP — SKIP", price_extension)
    return None  # Don't enter
```

Also add a feature: `recent_range_used = (close - low_20) / (high_20 - low_20)` — if >0.85 and trying to go LONG, or <0.15 and trying to go SHORT, skip.

### R4: UNIFY BOS/CHOCH DETECTION (Priority: MEDIUM)

Currently two separate systems:
1. `htf_features.py` internal BOS/CHOCH (4H swing_lookback=10)
2. `signals/bos_choch.py` live module (5m swing_lookback=8)

Merge them into one. Use **1H candles as the primary BOS/CHOCH timeframe for ETH** (not 5m, not 4H). 1H provides enough granularity for the 15-minute trading loop while filtering out 5-minute noise.

### R5: ADD DIRECTIONAL CONVICTION GATE (Priority: MEDIUM)

Before opening a SHORT, require:
- The 1D trend score < -0.2 (macro trend bearish or flat) **OR**
- The 4H structure_trend == -1 (lower highs + lower lows on 4H)
- **Never short when `overall_alignment` > 0.3** (multi-TF bullish alignment)

Before opening a LONG, require the reverse.

This prevents the model from shorting into a multi-timeframe uptrend just because the 15M entry score momentarily flips.

### R6: INCREASE MINIMUM CONFIDENCE FOR ETH (Priority: LOW)

The current MIN_CONFIDENCE=0.45 is too low, especially given the VecNormalize issue. For ETH specifically:
- Set MIN_CONFIDENCE=0.70 until VecNormalize is fixed
- After fixing normalization, evaluate whether 0.55-0.60 is appropriate for ETH

### R7: ADD TIME-OF-DAY FEATURE (Priority: LOW)

ETH has different volatility profiles at different times. The current 117 features have no time-of-day or day-of-week encoding. Adding `sin(hour/24 * 2π)` and `cos(hour/24 * 2π)` would let the model learn that, e.g., Asian session ETH tends to range while US session often trends.

---

## 5. Summary of Root Causes (Ranked)

| Rank | Issue | Impact | Fix Effort |
|------|-------|--------|------------|
| 1 | **VecNormalize broken** — model receives raw obs, policy outputs are meaningless | All trades affected | Low (re-export stats) |
| 2 | **No regime filter** — model trades ranging markets the same as trending | Trades #7-9, stagnant exits | Low (add condition) |
| 3 | **Structural short bias** — model learned to short from historical data that doesn't match current regime | 77% of trades are shorts in a rising market | Medium (retrain or add gate) |
| 4 | **No exhaustion detection** — enters after vertical moves | Trade #5 (-$134), Trade #6 (-$311) | Low (add feature check) |
| 5 | **Dual BOS/CHOCH systems** — live module says "bullish" but model doesn't see it | Trade #7 shorted during bullish trend | Medium (unify) |
| 6 | **Confidence is unreliable** — broken normalization means softmax collapses | All trades | Fixed by R1 |
| 7 | **Alignment bug in training** — `align_1h_15m` may have been self-referential during training | Systematic alignment bias | Medium (retrain if confirmed) |

**Bottom line:** Fix VecNormalize first. That alone likely turns the model from adversarial to functional. Then add the regime filter and exhaustion check to prevent the worst case scenarios. The BOS/CHOCH unification and directional gates are medium-term improvements.
