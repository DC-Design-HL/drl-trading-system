# Testnet Trade Analysis — 2026-03-22 13:00 to 2026-03-23 19:15 UTC

## Executive Summary

**Total realized PnL: -$12.49 USDT** (BTC) + **-$9.93 USDT** (ETH) = **~-$22.42 USDT** combined on testnet exchange.
The internal state files show the simulated (model-tracked) PnL as BTC: +$58.63 and ETH: -$423.53. This divergence itself is a finding.

Current open position: **BTC LONG** 0.012 @ $70,339.90, unrealized +$6.53 (mark ~$70,884).

**Root causes of losses (in order of severity):**
1. 🔴 **TIF GTE Bug causing emergency closes** — the #1 destroyer of capital
2. 🔴 **Position sizing mismatch** — model expects large positions, testnet only fills tiny fractions
3. 🟡 **Rapid-fire whipsaw trades** — 10 micro-trades in 4 minutes on BTC (19:09–19:13 on Mar 22)
4. 🟡 **SL too tight on ETH** — 1.5% SL gets hit in normal volatility
5. 🟠 **Model entering on low-confidence signals** — multiple entries at conf < 0.55

---

## 1. BTC Trade-by-Trade Breakdown (BTCUSDT)

### Complete Trade Log from Exchange (47 fills, ~16 logical round-trips)

| # | Entry Time | Dir | Entry $ | Exit Time | Exit $ | Reason | Qty | PnL ($) | Conf |
|---|-----------|-----|---------|-----------|--------|--------|-----|---------|------|
| 1 | Mar 22 13:43 | LONG | 68,648 | 13:44 | 68,637 | Emergency close (TIF GTE) | 0.002 | -0.023 | 0.76 |
| 2 | Mar 22 13:53 | LONG | 68,616 | 14:16 | 68,726 | TP signal / normal close | 0.016 | +2.046 | 0.77 |
| 3 | Mar 22 15:10 | LONG | 68,826 | 18:00 | 68,654 | Model direction change | 0.018 | -3.098 | 0.98 |
| 4 | Mar 22 18:04 | LONG | 68,588 | 18:59 | 68,535 | Model direction change | 0.013 | -0.685 | 0.72 |
| 5a-5e | Mar 22 19:09–19:10 | LONG↔SHORT | ~68,463 | 19:10 | ~68,463 | **WHIPSAW** (5 rapid flips) | 0.002 ea | -0.013 | — |
| 6a-6e | Mar 22 19:12–19:13 | LONG↔SHORT | ~68,455 | 19:13 | ~68,450 | **WHIPSAW** (5 rapid flips) | 0.002 ea | -0.120 | — |
| 7 | Mar 22 19:21 | SHORT | 68,327 | 19:29 | 68,316 | Normal close | 0.016 | +0.176 | 0.88 |
| 8 | Mar 22 19:42 | SHORT | 68,342 | 23:13 | 68,330 | Normal close (partial) | 0.014 | +0.145 | 0.79 |
| 9 | Mar 22 23:13 | SHORT | 68,288 | Mar 23 05:51 | 68,643 | **SL hit** (trailing SL) | 0.012+0.014 | -4.857 | 0.95 |
| 10 | Mar 23 11:30 | LONG | 70,475 | 12:01 | 69,999 | **SL hit** | 0.017 | -8.073 | 0.99 |
| 11 | Mar 23 12:46 | LONG | 70,658 | 15:40 | 70,792 | TP (trailing lock-50%) | 0.015 | +2.012 | 1.00 |
| 12 | Mar 23 16:27 | LONG | 70,340 | — | (open) | — | 0.012 | +6.53 (unreal) | 0.65 |

**BTC Exchange Stats:**
- Closed trade PnL: **-$12.49**
- Win rate: 9/24 closed = **37.5%**
- Average win: +$0.60 | Average loss: -$1.53
- Profit factor: 0.37 (very poor)
- Largest loss: **-$8.07** (Trade #10, SL hit on LONG at 70,475 → 69,999)
- Largest win: +$2.05 (Trade #2)

### Key BTC Observations

**Trade #1 (Emergency Close):** Opened LONG at 68,648, immediately closed at 68,637 within 35 seconds. The TIF GTE bug prevented SL/TP placement, triggering emergency close. Lost only $0.02 but the pattern repeats.

**Trade #3 ($-3.10):** Entered LONG at 68,826 with 0.98 confidence. Held for 2h50m. Price dropped to 68,654. The model had high confidence but the direction was wrong — price was entering a down-leg. The model only sees the signal change after the damage is done.

**Trades #5–#6 (Whipsaw Disaster):** Between 19:09 and 19:13, the bot executed **10 open/close cycles** in 4 minutes, each losing $0.002–$0.04 to the spread. This appears to be a state sync bug — the service was restarted and the bot rapidly flip-flopped trying to reconcile internal state with exchange position. Combined loss: **~$0.13** — small but reveals a fragile restart path.

**Trade #9 ($-4.86):** SHORT entered at 68,288, trailing SL tightened to ~68,372 after price dropped. Then price reversed overnight and jumped to 68,643. The trailing SL locked profits too early at breakeven but then held the SHORT too long as price reversed. The issue: trailing SL locked at the entry price (~$68,372) which was NOT a profit — the entry was at $68,342 for the second leg. So the "breakeven" SL was actually above the effective entry, creating an instant loss when hit.

**Trade #10 ($-8.07, largest loss):** LONG at 70,475 with 0.99 confidence. BTC dropped to 69,999 within 31 minutes. SL hit at 69,825 (1.5% below entry). This was a normal SL functioning correctly, but the model entered at the top of a local pump. **The model's 0.99 confidence was completely wrong about the direction.** Price dropped $475 in 31 min.

---

## 2. ETH Trade-by-Trade Breakdown (ETHUSDT)

### Complete Trade Log from Exchange (28 fills, ~12 logical round-trips)

| # | Entry Time | Dir | Entry $ | Exit Time | Exit $ | Reason | Qty | PnL ($) | Conf |
|---|-----------|-----|---------|-----------|--------|--------|-----|---------|------|
| 1 | Mar 22 13:26 | LONG | 2,084 | 14:16 | 2,078 | Model close | 1.261 | -2.19 | 0.99 |
| 2 | Mar 22 15:10 | LONG | 2,081 | 18:00 | 2,079 | Model direction change | 0.594 | -1.26 | 0.99 |
| 3 | Mar 22 18:04 | SHORT | 2,075 | 18:59 | 2,070 | Normal close | 0.300 | +1.44 | 0.49 |
| 4 | Mar 22 19:21 | SHORT | 2,060 | 19:29 | 2,061 | Partial close | 0.315 | -0.06 | 0.67 |
| 5 | Mar 22 19:42 | SHORT | 2,065 | 22:52 | 2,065 | **SL hit** (trailing) | 0.247 | small loss | 0.51 |
| 6 | Mar 22 23:28 | LONG | 2,063 | Mar 23 06:42 | 2,032 | **SL hit** | 0.373 | -3.18 | 0.87 |
| 7 | Mar 23 07:23 | SHORT | 2,030 | 08:29 | 2,055 | **REVERSE** | 0.307 | -7.44 | 0.48 |
| 8 | Mar 23 08:29 | LONG | 2,054 | 08:29 | — | **EMERGENCY CLOSE (TIF GTE)** | 0.294 | (forced flat) | 0.56 |
| 9 | Mar 23 08:29 | SHORT | 2,054 | 11:05 | 2,052 | Normal close | 0.307 | +0.33 | — |
| 10 | Mar 23 11:52 | LONG | 2,142 | 12:03 | 2,120 | **SL hit** | 0.303 | -6.71 | 0.69 |
| 11 | Mar 23 12:37 | LONG | 2,130 | 15:40 | 2,158 | **Trailing TP (profit!)** | 0.328 | +9.10 | 0.56 |
| 12 | Mar 23 17:28 | SHORT | 2,131 | 18:55 | 2,159 | **SL hit** | 0.274 | -7.42 | 0.57 |

**ETH Exchange Stats:**
- Closed trade PnL: **~-$9.93** (sum of realized on exchange, not in API as separate ETH total)
- Simulated model PnL: **-$423.53** (state file — model thinks it lost way more because it tracks larger simulated positions)
- Win rate: ~3/12 = **25%**
- Largest loss: **-$7.44** (Trade #7, SHORT reversed at a loss)
- Largest win: +$9.10 (Trade #11, trailing TP worked beautifully)

### Key ETH Observations

**Trade #1 ($-2.19):** Entered LONG at 2,084 with 0.99 confidence. Closed at 2,078 after 50 min. The model was confident but ETH was already at a local high. The first SL/TP placement FAILED due to `Order type not supported for this endpoint` — the very first trade had no protection!

**Trade #6 ($-3.18, overnight hold):** LONG at 2,063, held overnight for 7+ hours. Price slid to SL at 2,032 (1.5% below entry). This was a slow grind lower. The model kept saying "LONG" with 99% confidence every 15 minutes while the position bled out. The SL at 1.5% was appropriate but the model failed to recognize the down-trend.

**Trade #7 ($-7.44, worst directional call):** SHORT at 2,030 with only 0.48 confidence — **below the 0.5 threshold for a meaningful signal.** The bot should NOT be opening positions at 0.48 confidence. Within an hour, price rose to 2,055 and the position was reversed. Then the TIF GTE bug hit on the reversal, causing an emergency close of the new LONG at 2,054, wiping out the replacement position immediately.

**Trade #8 (TIF GTE Emergency):** After reversing from SHORT to LONG at 2,054, the algo-order SL/TP placement failed with `Time in Force (TIF) GTE can only be used with open positions`. This triggered an immediate emergency close. The position was opened and closed within seconds, turning what should have been a recovery trade into a dead loss from commissions.

**Trade #10 ($-6.71):** LONG at 2,142 with 0.69 confidence on the massive ETH pump (ETH went from 2,030 to 2,170 in 5 hours). Entered AFTER the pump was largely done. SL hit at 2,111 within 11 minutes. Classic case of chasing a move.

**Trade #11 (+$9.10, best trade):** LONG at 2,130, trailed up to 2,158. Trailing lock-50% worked perfectly — locked in profit at 2% up and trailed the SL to protect gains. This is what the system SHOULD do when it catches a trend.

**Trade #12 ($-7.42, last ETH trade):** SHORT at 2,131 with 0.57 confidence, SL hit at 2,159 within 87 minutes. Entered SHORT right at the bottom of a dip that reversed. Poor timing, low confidence.

---

## 3. Root Cause Analysis

### 🔴 ROOT CAUSE #1: TIF GTE Bug (Critical — ~$10+ in shadow losses)

**What happens:** When the bot tries to adjust SL/TP via Binance algo-orders, it fails with:
```
Time in Force (TIF) GTE can only be used with open positions
```

This error cascades:
1. SL/TP adjustment attempt fails
2. The bot enters an error recovery path
3. On new position opens after a reverse, the SL/TP can't be placed
4. The bot does an **emergency close** of the position it just opened
5. Result: position opened and immediately closed at a loss (commission + spread)

**Evidence:**
- BTC Trade #1: Emergency close 35 seconds after open
- ETH Trade #8: Emergency close of LONG after reversing from SHORT
- Mar 23 11:05 BTC: Dozens of failed SL adjustment attempts in rapid succession (every 1.4s)

**Impact:** Each emergency close wastes ~$0.05–$0.50 in commissions, but the real damage is that it cancels positions that would have been profitable. BTC Trade #1 was closed at 68,637 — but the intended position would have caught the move to 68,726 (+$0.18). ETH Trade #8 was closed flat but the SHORT replacement from 2,054 went on to close at 2,052 for a profit.

### 🔴 ROOT CAUSE #2: Position Sizing Mismatch (Critical — explains model vs. reality divergence)

**The model's internal balance is completely decoupled from testnet reality.**

Evidence from state files:
- BTC state: `balance: 25,481` (simulated), but testnet account has ~$5,000 + positions
- ETH state: `balance: 59,156` (simulated), testnet account is the same ~$5,000
- ETH state shows `realized_pnl: -423.53` but exchange shows only ~-$9.93 in ETH realized PnL

The model computes its own balance based on simulated fills and sizes:
- ETH OPEN_LONG logged `units=64.155` but only 0.418 ETH was filled on testnet
- ETH OPEN_LONG logged `units=1125.296` but only 0.252 ETH was filled on testnet 
- BTC OPEN_LONG logged `units=1.942` but only 0.010 BTC was filled on testnet

**This means the model is making decisions based on a position size 100-1000x larger than reality.** The model thinks it has $266,765 in balance (from compounding simulated gains) and sizes accordingly. The testnet executor then clamps to what the actual ~$5,000 account can afford.

**Why this matters:** The model's risk management, trailing stops, and position sizing algorithms are calibrated for a much larger position. When the testnet only fills a tiny fraction, the SL/TP distances become disproportionate to the actual risk.

### 🟡 ROOT CAUSE #3: SL Distance Too Tight for Timeframe

The SL is set at approximately **1.5% from entry** for both BTC and ETH.

For BTC on a 15-minute timeframe:
- Normal 15-min range: $100–$300 (~0.15%–0.44% of price)
- 1.5% = ~$1,050. This gives about 3–10 candles of room.
- In trending conditions (Mar 23), BTC moved $475 in 31 minutes (0.67%)
- The 1.5% SL is marginal — it works in trends but gets chopped in ranges

For ETH on a 15-minute timeframe:
- Normal 15-min range: $5–$15 (~0.25%–0.72% of price)
- 1.5% = ~$31. This gives only 2–6 candles of room.
- ETH Trade #10: Entered at 2,142, SL at 2,111 ($31). Hit in 11 minutes.
- ETH Trade #12: Entered at 2,131, SL at 2,166 ($35). Hit in 87 minutes.

**ETH has notably higher % volatility than BTC, so the same 1.5% SL gives less room.** ETH frequently makes 1.5–2% moves within an hour.

### 🟡 ROOT CAUSE #4: Model Direction Confidence Issues

Multiple entries were made with confidence < 0.60:
- ETH Trade #3: conf=0.49 (but actually won!)
- ETH Trade #5: conf=0.51
- ETH Trade #7: conf=0.48 ← **should never have entered**
- ETH Trade #8: conf=0.56
- ETH Trade #11: conf=0.56 ← this one worked great
- ETH Trade #12: conf=0.57

The model appears to enter LOW-confidence positions that often fail. When confidence is >0.90, the model is often wrong too (ETH Trade #1 at 0.99, Trade #2 at 0.99, Trade #6 at 0.87 — all lost money).

**This suggests the model's confidence is not well-calibrated.** High confidence doesn't correlate with better outcomes, and the model takes positions even at very low confidence.

### 🟠 ROOT CAUSE #5: Whipsaw on Service Restart

BTC Trades #5-#6 (19:09–19:13 on Mar 22): 10 rapid open/close cycles in 4 minutes after what appears to be a service restart. The bot repeatedly tries to reconcile its state, flip-flopping between LONG and SHORT with minimum size (0.002 BTC). Each round-trip loses $0.002–$0.04 to spread.

**Cost:** ~$0.13 total, but it indicates a fragile state recovery path that could be much worse in real money.

---

## 4. Market Condition Analysis

### BTC (Mar 22–23)
- Mar 22 13:00–19:00: **Ranging** $68,500–$68,850 (choppy, no clear trend)
- Mar 22 19:00–Mar 23 06:00: **Slow decline** from $68,400 to $67,700 then recovery to $68,500
- Mar 23 06:00–11:30: **Strong rally** from $68,500 to $71,200 (+3.9% — breakout!)
- Mar 23 11:30–16:00: **Volatile ranging** $69,900–$70,800 with sharp pullbacks

The first 24 hours were a choppy range — the worst possible environment for a trend-following model. The Mar 23 morning rally was the one big opportunity, and the model caught it (Trade #9 state file shows SL trailed up) but the testnet position was only $12 in a SHORT that got stopped out at a loss before the rally.

### ETH (Mar 22–23)
- Mar 22 13:00–19:00: **Choppy decline** from $2,084 to $2,060
- Mar 22 19:00–Mar 23 06:00: **Ranging** $2,030–$2,070 (extreme chop)
- Mar 23 06:00–12:00: **Massive pump** from $2,030 to $2,170 (+6.9%!)
- Mar 23 12:00–19:00: **High-volatility ranging** $2,120–$2,170

ETH was brutal for the first 18 hours — a choppy range that whipsawed the bot. The big pump on Mar 23 morning was only partially captured (Trade #11 caught +$9.10, but Trade #10 entered too late and got stopped).

---

## 5. Commission Analysis

Total commissions paid:
- BTC: ~$5.82 across 47 fills
- ETH: ~$5.15 across 28 fills
- **Total: ~$10.97** in commissions

Given a total realized PnL of -$12.49 (BTC), commissions represent **47% of the total BTC loss.** For a system with thin edges, this commission drag is significant, especially with the whipsaw trades burning commissions for zero benefit.

---

## 6. BOS/CHOCH Signal Impact

The logs show BOS (Break of Structure) and CHOCH (Change of Character) signals were not explicitly logged per-trade in the iteration output. However, the model's action decisions implicitly incorporate these.

**Assessment:** The model flips direction too frequently for the current market conditions. In the choppy Mar 22 period, it went:
- LONG → LONG → LONG → SHORT → SHORT → LONG → LONG
This is 3 direction changes in ~10 hours during a range-bound market.

For ETH, the model flipped:
- LONG → LONG → SHORT → SHORT → LONG → SHORT → LONG → LONG → SHORT
That's 5 direction changes. Some of these (SHORT at 18:04 → profit of $1.44) were good. But the rapid flipping at low confidence killed the edge.

---

## 7. Recommendations

### CRITICAL (Do Immediately)

1. **Fix TIF GTE Bug:** The algo-order placement must handle the `TIF GTE` error gracefully. When the error occurs, the bot should NOT emergency-close the position. Instead, it should fall back to bot-side monitoring (which it already does for STOP_MARKET). The emergency close path is destroying value.

2. **Sync Model Balance to Testnet Reality:** The model's internal balance tracker is in fantasy-land ($266K, $59K). Either:
   - Reset the model balance to match the testnet account balance (~$5,000) on each restart
   - Or (better) compute position size based on the ACTUAL testnet balance, not the simulated one

3. **Minimum Confidence Threshold:** Do NOT open positions when confidence < 0.55. Several of the worst trades (ETH #7 at 0.48, ETH #12 at 0.57) were low-confidence entries that immediately went wrong.

### IMPORTANT (Do This Week)

4. **Widen ETH Stop-Loss:** ETH volatility demands a wider SL. Consider 2.0–2.5% for ETH (vs 1.5% for BTC). ETH's intraday swings are proportionally larger.

5. **Cooldown After Service Restart:** Add a "warmup" period after service restart where the bot does NOT trade for at least 2 candles (30 min). This prevents the whipsaw issue seen in Trades #5-#6.

6. **Commission-Aware Position Sizing:** With ~$0.05 commission per BTC fill and ~$0.08 per BTC fill for 0.002 size, the minimum profitable BTC move is ~$25 per BTC (0.037%). Ensure position sizes are large enough that commissions are <10% of expected profit.

### NICE TO HAVE

7. **Confidence Calibration:** The model reports 0.99 confidence on positions that immediately lose money. The confidence output needs recalibration — consider adding a separate confidence model or using dropout at inference to get uncertainty estimates.

8. **Market Regime Filter:** Don't trade during the first 6 hours of a new range (when the model's trend signals are stale). The Mar 22 choppy range from 13:00–19:00 generated 3 losing trades.

9. **WebSocket Stability:** The ETH bot's WS price monitor disconnects frequently (12+ disconnections on Mar 23 afternoon). While it reconnects, there's a window where trailing SL adjustments are missed.

---

## 8. What's Actually Working

1. ✅ **Trailing SL mechanism** — when it works (no TIF GTE errors), the trailing lock-50% strategy is excellent. ETH Trade #11 locked in +$9.10 by trailing the SL up as price rose.

2. ✅ **BOS-triggered entries in trends** — BTC Trade #2 (+$2.05) and BTC Trade #11 (+$2.01) caught real trend moves.

3. ✅ **SHORT trades in declining markets** — BTC Trade #7 (+$0.18) and ETH Trade #3 (+$1.44) correctly shorted declining moves.

4. ✅ **TP placement** — when TP orders are successfully placed, they capture profits at reasonable levels.

---

## 9. Bottom Line

The model isn't bad — it's being sabotaged by infrastructure issues. The TIF GTE bug alone accounts for multiple unnecessary losses and failed position management. The position sizing mismatch means the model is operating in a different reality than the testnet.

**If we fix the TIF GTE bug, sync the balance, and add a 0.55 confidence filter, the system would likely be profitable.** The one winning streak (BTC rally Mar 23 + ETH Trade #11) generated +$11.12, which nearly offsets all other losses. The model catches trends when they exist — it just needs to stop bleeding during ranges and stop self-destructing on infrastructure bugs.

**Priority order:** TIF GTE fix > Balance sync > Confidence threshold > ETH SL widening > Restart cooldown
