# SL/TP Redesign Plan

**Date**: 2026-03-24
**Status**: Draft — analysis complete, awaiting implementation decision

---

## 1. Executive Summary

Historical live trading data reveals a **85% stop-loss hit rate** (11 of 13 closed trades) with a 15.4% win rate in the standard HTF strategy, resulting in -381.88 USDT realized loss. The only profitable configuration tested is the **Hybrid strategy** (partial exits + trailing SL), which achieved a 57.1% win rate and +230.23 USDT profit.

The core problem is not just "SLs too tight" or "TPs too far" in isolation — it is a combination of:
1. SLs set mechanically at 1.5% without structural context, getting clipped by normal volatility
2. Training environments that never experienced partial exits or trailing stops
3. A significant train/live parameter mismatch
4. Missing MFE/MAE data that prevents precise calibration

This plan outlines a phased approach to fix SL/TP configuration, align training with live execution, and adopt partial take-profit as a first-class exit mechanism.

---

## 2. Current State Analysis

### 2.1 Parameter Overview

| Component | SL | TP | R:R | Notes |
|---|---|---|---|---|
| `ultimate_env.py` (training) | 2.5% | 5.0% | 2:1 | Hardcoded, flat |
| `htf_env.py` (training) | 1.5% | 3.0% | 2:1 | Default, flat |
| `analyze_trades.py` | 2.0% | 4.0% | 2:1 | Yet another set |
| Live BTC | 1.5% | 3.5% | 2.3:1 | Structural |
| Live ETH | 2.0% | 5.0% | 2.5:1 | Structural |
| Live SOL | 2.5% | 6.0% | 2.4:1 | Structural |
| Live XRP | 2.0% | 5.5% | 2.75:1 | Structural |
| Live (high vol regime) | base×2.0 | base×1.5 | varies | Regime scaled |
| Live (ranging regime) | base×1.8 | base×1.0 | varies | Widened SL |

Every layer of the system uses **different numbers**. The model was trained under conditions that will never match live execution.

### 2.2 Historical Trade Performance

Data sourced from local JSONL logs (Mar 22-24, 2026):

| Strategy | Closed Trades | SL Hits | TP Hits | Win Rate | Total PnL |
|---|---|---|---|---|---|
| HTF Standard | 13 | 11 (85%) | 1 (8%) | 15.4% | -381.88 USDT |
| HTF Partial | 5 | 2 (40%) | 0 (0%) | 40.0% | -303.16 USDT |
| **HTF Hybrid** | 7 | 2 (29%) | 1 (14%) | **57.1%** | **+230.23 USDT** |

**The Hybrid strategy (partial TP + trailing SL) is the only configuration generating positive PnL.**

### 2.3 Symbol-Level Breakdown (HTF Standard)

| Symbol | Closed | Win Rate | PnL |
|---|---|---|---|
| BTCUSDT | 5 | 20.0% | +53.48 USDT |
| ETHUSDT | 8 | 12.5% | -435.36 USDT |

ETH is the primary loss driver. It also exhibits a severe position-size compounding pattern: in 36 hours, ETH position size grew from 1.21 units (~$2,500) to 13.65 units (~$29,578) — an 11.8x increase — resembling inadvertent martingale scaling. This compounds losses catastrophically when the model is wrong.

### 2.4 Core Issues Identified

#### Issue 1: SLs are Too Tight for Live Volatility
- 85% SL hit rate on the standard strategy is unsustainable. At 2:1 R:R, break-even requires >33.3% win rate. 15.4% is less than half that.
- Fixed-percentage SLs (1.5% for BTC) do not account for ATR or market structure. A 1.5% move is within normal noise on a 15-minute BTC chart.
- The regime multiplier (up to 2.0x in high-vol) addresses this partially, but only after a structural SL is calculated — the base is still too tight.

#### Issue 2: TPs Are Too Ambitious (or Never Reached)
- Only 1 of 13 standard trades hit TP (8% hit rate). This suggests TPs at 3.0% may be too far for the typical move the model captures, **or** SLs are hit before price has time to reach TP.
- Partial takes at intermediate levels (1R = 1.5%) would have captured profit on several trades that ultimately hit SL after briefly going positive.

#### Issue 3: Training/Live Mismatch
- `ultimate_env.py` trains with flat 2.5% SL / 5.0% TP.
- `htf_env.py` trains with flat 1.5% SL / 3.0% TP.
- Live execution uses structural SL (VWAP + swing points + ATR buffer) with regime multipliers.
- The model learns to hold through -1.5% drawdowns because training taught it "SL is at -1.5%." But in live trading under high-vol regime, the SL might be at -3.0%. The model has no knowledge of this broader tolerance window and will signal exits (or panic-reverse) before the live SL triggers.
- Conversely, the model may have learned to hold all the way to 5% TP, which live trading rarely achieves in one move.

#### Issue 4: No MFE/MAE Tracking
- Neither MongoDB nor JSONL logs record Maximum Favorable Excursion (how far price moved in our favor before reversing) or Maximum Adverse Excursion (how far against us before recovering).
- Without this data, SL/TP calibration is guesswork. We cannot determine if we are being stopped out at -1.5% on trades that were ultimately +3% favorable, or whether our TP targets are achievable.

#### Issue 5: Position Sizing Compounding (ETH)
- The ETH bot's position size grew 11.8x in 36 hours. This suggests the position sizing formula (`balance × 0.25`) is compounding on a balance that includes unrealized gains, or the Kelly criterion is miscalculating under repeated losses.
- This is not strictly a SL/TP issue, but it amplifies SL losses dramatically and should be fixed as part of this redesign.

---

## 3. Proposed SL/TP Strategy

The redesign has three tiers:

### Tier 1: Immediate Fixes (No Retraining Required)
Changes to live execution only — these take effect immediately without model changes.

#### 3.1 Wider Base SL with ATR Floor

Replace the flat asset-specific SL base with an **ATR-gated minimum**:

```
sl_pct = max(asset_base_sl, 1.5 × ATR_14_pct)
tp_pct = max(asset_base_tp, 3.0 × ATR_14_pct)
```

Where `ATR_14_pct` = 14-period ATR / current price.

**Rationale**: A 1.5% SL is meaningless if 14-period ATR is already 2%. The SL should always be outside normal noise. On low-volatility days, the tighter asset-base value kicks in.

#### 3.2 Regime Multiplier Recalibration

Current multipliers are too aggressive in the `ranging` regime (SL×1.8 widens the SL but TP stays at 1.0×, killing R:R). Proposed:

| Regime | SL Mult | TP Mult | R:R Effect |
|---|---|---|---|
| high_volatility | 1.5× (was 2.0×) | 1.3× (was 1.5×) | Preserved |
| trending_up/down | 1.0× | 1.8× (was 1.5×) | Let runners run |
| ranging | 1.2× (was 1.8×) | 0.8× | Tighter TP for mean reversion |
| neutral | 1.0× | 1.0× | Unchanged |

**Rationale**: High-vol SL of 2.0× was causing SL prices to be so wide that losses were massive when hit. Reducing to 1.5× limits damage while still clearing noise. For ranging, reduce TP (mean reversion moves are smaller by definition) rather than widening SL.

#### 3.3 Implement Partial Take Profit (Mandatory)

The Hybrid strategy's key advantage is partial exits. Make this standard across all live bots:

**Partial TP Schedule:**
- **Level 1 (1R)**: Close 40% of position at `entry ± 1.0 × sl_pct` (i.e., 1:1 R:R)
  - Move SL to break-even immediately after Level 1 fills
- **Level 2 (2R)**: Close 35% of position at `entry ± 2.0 × sl_pct`
  - Move trailing stop to lock 50% of Level 2 profit
- **Level 3 (Trailing)**: Let remaining 25% trail with `0.8 × sl_pct` trailing distance

**Exchange implementation:**
- Level 1 and 2 are placed as `TAKE_PROFIT_MARKET` orders with `quantity` (not `closePosition=true`) so they are partial
- Trailing SL for Level 3 is managed bot-side (cancel + replace on new highs/lows)
- SL for full position remains as safety net until Level 1 fills, then moves to break-even

**Expected outcome**: Even on trades that ultimately hit SL (after break-even move), we exit at 0 loss after Level 1. Trades that reach Level 2 lock in ~1R profit regardless of Level 3 outcome.

#### 3.4 Time-Based Exit for Stagnant Trades

Add a time-based exit rule:
- If a position has been open for `>6 hours` AND unrealized PnL is between `-0.3%` and `+0.5%` (going nowhere), close at market.
- Rationale: Stagnant trades tie up capital and often resolve against the original signal once it degrades.
- Current min hold is 4 hours — make time-based exit trigger at 6h only if position hasn't moved beyond ±0.5% of entry.

#### 3.5 Cap Position Size Compounding

Add an absolute cap on position size independent of Kelly/balance calculation:

```python
max_notional_per_trade = min(balance × position_size_pct, FIXED_MAX_NOTIONAL)
FIXED_MAX_NOTIONAL = 3000  # USDT, adjustable
```

This prevents the martingale-like compounding observed in the ETH bot. Kelly sizing should be calculated on **initial session balance**, not running balance, to prevent runaway sizing during drawdown periods.

---

### Tier 2: Training Alignment (Requires Retraining)
Changes to the training environments to match Tier 1 live parameters.

#### 3.6 Unify SL/TP Across Environments

Replace hardcoded flat percentages with a unified configuration:

```python
# Proposed training defaults (to match live asset averages)
stop_loss_pct = 0.020   # 2.0% (was 2.5% in ultimate_env, 1.5% in htf_env)
take_profit_pct = 0.040  # 4.0% (was 5.0% / 3.0%)
```

These represent the midpoint of live asset params weighted by expected trade frequency:
- BTC: 1.5%/3.5% (low vol, high frequency)
- ETH: 2.0%/5.0% (medium)
- SOL: 2.5%/6.0% (high vol)

A 2.0% SL / 4.0% TP represents a reasonable average that the model can be trained on without being too far from any single asset's live params.

#### 3.7 Add Partial TP to Training Reward

Modify the reward function in both `ultimate_env.py` and `htf_env.py` to reward partial exits:

```python
# Current: binary SL/TP reward
# Proposed: tiered reward matching partial TP schedule

if pnl_pct >= tp_pct * 0.5:   # Level 1 (1R equivalent)
    partial_bonus = 0.04 * reward_scaling
    reward += partial_bonus

if pnl_pct >= tp_pct:          # Level 2 (full TP)
    full_tp_bonus = 0.10 * reward_scaling
    reward += full_tp_bonus

# Intermediate reward shaping: incentivize holding past 1R
if 0 < pnl_pct < tp_pct:
    reward += pnl_pct * 0.3 * reward_scaling  # Scale up from existing 0.3×
```

The model should learn that reaching 1R is good, reaching 2R is better, and not reverse prematurely.

#### 3.8 Add ATR-Aware SL/TP to Training Observations

Add ATR-as-percentage to the observation space so the model can learn ATR-gated SL/TP:

```python
# Add to feature set
atr_pct = df['atr_14'] / df['close']   # Normalized ATR
obs = np.append(obs, [atr_pct, current_sl_pct, current_tp_pct])
```

This allows the model to understand the current SL/TP context and make exit decisions accordingly, rather than learning a fixed-distance exit policy that won't transfer to live.

#### 3.9 Add MFE/MAE Tracking to Both Training and Live

**Training env changes:**
```python
# Track per-episode MFE and MAE
self.mfe = 0.0  # max favorable pnl_pct seen this trade
self.mae = 0.0  # max adverse pnl_pct seen this trade

# In step():
self.mfe = max(self.mfe, current_pnl_pct)
self.mae = min(self.mae, current_pnl_pct)  # mae is negative

# On trade close: log {mfe, mae, outcome, duration}
```

**Live trading changes:**
```python
# In position state tracking
self.mfe_pct = 0.0
self.mae_pct = 0.0

# In main loop price update
unrealized_pct = (current_price - entry_price) / entry_price * direction
self.mfe_pct = max(self.mfe_pct, unrealized_pct)
self.mae_pct = min(self.mae_pct, unrealized_pct)

# Add to trade close log record
trade_record['mfe_pct'] = self.mfe_pct
trade_record['mae_pct'] = self.mae_pct
```

**Why this matters**: After 50-100 trades with MFE/MAE data, we can precisely answer: "Did we get stopped out at -1.5% on trades where MFE was +3%?" If yes, the SL is definitively too tight. This data is the foundation for all future calibration.

---

### Tier 3: Advanced Enhancements (Phase 2, Post-Retraining)

#### 3.10 BOS/CHOCH-Driven Dynamic SL/TP

The `skills/bos-choch-dynamic-sltp/` framework is already designed for this. Once Tier 1 partial exits are stable:
- Use BOS/CHOCH signals to move SL to recent structure (not fixed %) after Level 1 fills
- Extend TP target on confirmed BOS in trade direction
- Use multi-TF confidence scoring to decide whether to trail aggressively or take Level 2 quickly

#### 3.11 Regime-Specific TP Strategies

| Regime | TP Strategy |
|---|---|
| Trending | Let 25% trail to 3R+; don't force Level 2 |
| Ranging | Take Level 1 at 0.8R, skip Level 3 (mean-reversion trades are small) |
| High Vol | Widen Level 1 to 1.5R (avoid whipsaw taking you out too early) |

#### 3.12 Time-to-TP Distribution Analysis

Once MFE/MAE is tracked (§3.9), analyze: "Given a trade that reaches Level 1, what % eventually reach Level 2 within N hours?" This informs whether the 25% trailing position should be time-capped (e.g., close Level 3 after 12 hours regardless of trailing stop status).

---

## 4. Implementation Order

### Phase 1 — Immediate (No Retraining, This Week)

1. **Fix position size cap** (§3.5) — prevent ETH compounding disaster
2. **Add MFE/MAE tracking** to live bots (§3.9 live portion) — start collecting data NOW
3. **Implement partial TP** in `live_trading_multi.py` and `futures_executor.py` (§3.3)
4. **Recalibrate regime multipliers** (§3.2) — simple constant changes
5. **Add ATR floor to SL** (§3.1) — structural change to `risk_manager.py`
6. **Add time-based exit** (§3.4) — simple timer check in main loop

### Phase 2 — Training Alignment (After 2 Weeks of Phase 1 Data)

7. **Unify SL/TP in training envs** (§3.6)
8. **Add partial TP reward shaping** (§3.7)
9. **Add ATR obs to feature set** (§3.8)
10. **Retrain models** with new environment — run parallel paper trading comparison
11. **Add MFE/MAE tracking** to training env (§3.9 training portion)

### Phase 3 — Advanced (After Retraining, 4-6 Weeks Out)

12. **BOS/CHOCH dynamic SL/TP** (§3.10)
13. **Regime-specific TP strategies** (§3.11)
14. **Time-to-TP distribution analysis** (§3.12) and iterative calibration

---

## 5. Success Metrics

After Phase 1 implementation, evaluate over 50+ trades:

| Metric | Current | Target |
|---|---|---|
| SL hit rate | 85% | <50% |
| Win rate | 15.4% | >40% |
| Average R:R realized | ~0.5:1 | >1.5:1 |
| Monthly PnL | -381 USDT | Positive |
| MFE on SL trades | Unknown | <1.5× SL distance |
| ETH max position | 13.65 units | Capped (≤$3k notional) |

If SL hit rate drops below 50% and win rate exceeds 33% (break-even for 2:1 R:R), the strategy is viable. Win rate above 40% with 1.5:1 realized R:R would produce consistent profitability.

---

## 6. Key Risks and Considerations

### Risk: Wider SLs = Larger Losses Per Trade
Widening SL from 1.5% to 2.0-3.0% means each losing trade loses more. This is only acceptable if win rate improves proportionally. **Mitigation**: Position size must be reduced proportionally when SL is wider. If SL doubles, position size halves to maintain constant risk per trade in dollar terms.

### Risk: Partial TP Reduces Upside on Big Moves
Taking 40% profit at 1R means you capture less of a trend. **Mitigation**: The 25% trailing position retains exposure to large moves. The tradeoff (consistent small wins vs. occasional large wins) is favorable given current 15.4% win rate.

### Risk: Retraining May Produce Worse Models
New environment parameters don't guarantee better model behavior. **Mitigation**: Run old and new models in parallel paper trading for 2+ weeks before switching live allocation.

### Risk: Exchange Order Complexity (Partial TP)
Binance Futures partial close orders require quantity-specific orders rather than `closePosition=true`. Canceling and replacing SL orders when partial TP fills is operationally complex and introduces timing risk. **Mitigation**: Implement with fill event monitoring; always keep full-position safety SL until break-even is confirmed.

---

## 7. Files to Modify

| File | Change | Phase |
|---|---|---|
| `live_trading_multi.py` | Partial TP logic, time-based exit, position cap | 1 |
| `src/features/risk_manager.py` | ATR floor for SL, regime multiplier values | 1 |
| `src/api/futures_executor.py` | Quantity-specific TP orders, cancel/replace SL on fill | 1 |
| `live_trading_multi.py` | MFE/MAE tracking and logging | 1 |
| `src/env/ultimate_env.py` | Unified SL/TP params, partial reward, ATR obs | 2 |
| `src/env/htf_env.py` | Unified SL/TP params, partial reward, ATR obs | 2 |
| `analyze_trades.py` | Add MFE/MAE columns to analysis | 2 |

---

## 8. Open Questions

1. **Should we retrain from scratch or fine-tune?** Fine-tuning existing PPO checkpoint with new environment parameters may be faster but risks mode collapse if the new params are too different from training distribution.

2. **What is the right partial TP split?** The 40%/35%/25% split proposed here is untested. Literature suggests 50%/50% is simpler and nearly as effective for most trend-following setups.

3. **Should SL be exchange-order or bot-monitored?** Currently exchange-side (`STOP_MARKET`). For partial TP implementation, SL cancel/replace events create race conditions. Consider moving SL to bot-side monitoring with exchange SL as a "catastrophic" backstop only.

4. **How to handle ETH position compounding root cause?** The 11.8x position growth in 36h needs a definitive root-cause analysis before Phase 1 deployment. Is it Kelly miscalculation? Balance sync issues? Leverage interacting with position_size_pct?

5. **Should ultimate_env.py and htf_env.py be merged?** They are diverging in parameters. A single unified environment with configurable asset profiles (passed at init) would reduce maintenance burden and ensure training consistency.
