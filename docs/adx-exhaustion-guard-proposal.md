# ADX Exhaustion Guard — Proposal (Apr 2, 2026)

## Status: PENDING — Collecting data until Apr 4 18:00 UTC

## Problem
Trades entered when ADX > 60 have a **29% win rate** (same as ranging ADX < 20). High ADX means the trend is overextended and likely to reverse — our trend-following model enters right at exhaustion.

## Data (53 trades, Mar 29 - Apr 2)

### Win Rate & PnL by ADX Band

| ADX Band | Trades | WR | Total PnL | Avg PnL |
|----------|--------|-----|-----------|---------|
| 0-20 | 7 | 29% | -$92 | -$13 |
| 20-30 | 15 | 53% | -$85 | -$6 |
| **30-40** | **7** | **71%** | **+$53** | **+$8** |
| **40-50** | **9** | **67%** | **+$33** | **+$4** |
| 50-60 | 8 | 62% | -$26 | -$3 |
| 60+ | 7 | 29% | -$67 | -$10 |

**Sweet spot: ADX 30-50** — only profitable bands (71% and 67% WR).

### Simulated Block Results

| Threshold | Blocked | Blocked WR | PnL Saved | Winners Lost | Net |
|-----------|---------|------------|-----------|-------------|-----|
| > 55 | 10 | 30% | +$109 | 3 wins ($22) | +$87 |
| > 60 | 8 | 25% | +$93 | 2 wins ($12) | +$81 |
| > 65 | 5 | 20% | +$75 | 1 win ($11) | +$64 |

### Specific ADX > 60 Trades

| Asset | Dir | ADX | PnL | MFE |
|-------|-----|-----|-----|-----|
| BTC LONG | 65.6 | +$10.88 | +1.2% | ✅ winner (would be blocked) |
| BTC SHORT | 61.8 | +$1.59 | +1.0% | ✅ stagnant win |
| BTC LONG | 65.6 | -$7.70 | +0.0% | ❌ immediate reversal |
| ETH SHORT | 67.0 | -$26.04 | +0.18% | ❌ barely moved, SL hit |
| ETH SHORT | 67.0 | -$26.04 | +0.18% | ❌ same pattern |
| SOL LONG | 69.0 | -$25.70 | +0.11% | ❌ immediate reversal |
| SOL LONG | 61.1 | -$19.56 | +0.32% | ❌ barely moved |
| SOL SHORT | 60.2 | -$0.25 | +0.13% | ❌ stagnant exit |

**6/8 losers, 2/8 winners. The winners were small (+$12 combined), the losers were big (-$105 combined).**

## Proposed Change

```python
ADX_EXHAUSTION_ENABLED = True
ADX_EXHAUSTION_MAX = 60  # Block all trades when ADX above this
```

Add to `_check_rsi_adx_guard()` in `live_trading_htf.py`:
- If ADX > 60, block the trade (trend exhaustion — likely to reverse)
- Combined with existing ADX < 20 block, effective trading range = ADX 20-60

## Current Guards (deployed)

| Guard | Threshold | Status |
|-------|-----------|--------|
| ADX Ranging | ADX < 20 → block | ✅ Deployed Apr 2 |
| RSI Extreme | RSI > 70 (OB) / RSI < 30 (OS) → block | ✅ Deployed |
| Rescue Override | Disabled | ✅ Disabled Apr 2 |
| Orderbook Guard | OB bias contradicts direction → block | ✅ Deployed Mar 31 |
| Exhaustion Filter | Price > 3 ATR from VWAP → skip | ✅ Deployed |
| Signal Gate | Tier 2 (conf < 0.80) needs 2/4 signal confirms | ✅ Deployed |
| **ADX Exhaustion** | **ADX > 60 → block** | **⏳ PENDING** |

## Changes Made Today (Apr 2)

1. **Rescue Override disabled** — RSI/ADX blocks are final, no override
2. **ADX Guard raised to 20** — blocks ranging market trades
3. **Trailing activation: +1% → +0.5%** — catches smaller moves
4. **Trailing distance: 0.5% → 0.3%** — tighter trail captures more profit
5. **SQLite local storage** — replaced broken MongoDB Atlas

## Other Findings (Apr 2 analysis)

### MFE Profit Capture
- Average winning trade peaks at +1.4% MFE
- We captured only ~60% of peak profit (old trailing settings)
- New trailing settings should improve to ~70-75% capture
- Trades with MFE < 0.2% are bad entries (no trailing can save them)

### Whale Signal
- Whale SELL confidence ranged 23-34% across ALL 71 trades — never hit 40%
- No correlation between whale signal and trade outcomes
- Robinhood wallet data stale since Feb 14
- Model needs retraining (scheduled for Chen's Mac M3)
- **Verdict: display-only, not actionable yet**

## Decision Point (Apr 4)
After 48 more hours of data, re-evaluate:
1. How many ADX > 60 entries occurred?
2. What was their win rate and PnL?
3. Did the new trailing settings improve profit capture?
4. Are the existing guards (rescue disabled, ADX<20) performing as expected?

If ADX > 60 still shows < 35% WR after 100+ total trades → deploy the guard.
