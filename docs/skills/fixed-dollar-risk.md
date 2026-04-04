---
name: fixed-dollar-risk
description: Fixed-Dollar-Risk position sizing model for the DRL trading system. Use when reviewing, modifying, or debugging position sizing, leverage, margin, or risk calculations. Covers the full risk chain from balance → risk pool → dollar risk → notional → leverage (from liq buffer) → margin.
---

# Fixed-Dollar-Risk Position Sizing

## The Risk Chain

```
Balance ($5,000)
  → Risk Pool = 10% of balance ($500)
    → Dollar Risk = pool ÷ 20 parts ($25)
      → Notional = risk ÷ SL% ($25 ÷ 0.015 = $1,667)
        → Liq distance = SL% + 1% buffer = 2.5%
          → Leverage = 1 / liq_distance = 40x       ← derived from liq buffer
            → Margin = notional / leverage = $42     ← derived, NOT chosen
```

## Constants (live_trading_htf.py)

```python
RISK_POOL_PCT = 0.10       # 10% of balance is risk pool
RISK_BUDGET_PARTS = 20     # 20 equal risk slots
LIQ_BUFFER_PCT = 0.01      # Liquidation must be 1% beyond SL from entry
MAX_LEVERAGE = 50           # Hard cap
FIXED_MAX_NOTIONAL = 3000.0 # Safety cap on notional (USDT)
```

## Key Rules

1. **Liquidation = SL + 1% buffer** — this is the anchor. Everything derives from it.
2. **Leverage = 1 / (SL% + buffer%)** — at 1.5% SL: 1/0.025 = 40x.
3. **Margin = notional / leverage** — derived, not chosen.
4. **Isolated margin** per symbol — set via `set_margin_type(symbol, "ISOLATED")` before every trade.
5. **Validation**: if actual liq distance < required, reduce leverage until safe.
6. **Risk scales both ways** — balance drops → risk drops. Balance grows → risk grows.

## Example at $5,000 Balance

| Field | Value |
|-------|-------|
| Risk pool | $500 (10%) |
| Dollar risk/trade | $25 |
| Notional | $1,667 |
| Liq distance | 2.5% (SL 1.5% + buffer 1%) |
| Leverage | 40x |
| Margin | $42 |
| SL distance | 1.5% |
| Max loss per trade | $25 (0.5% of balance) |
| Max simultaneous | 20 (risk budget) |

## Scaling Examples

| Balance | Risk/Trade | Notional | Leverage | Margin |
|---------|-----------|----------|----------|--------|
| $5,000 | $25 | $1,667 | 40x | $42 |
| $10,000 | $50 | $3,000* | 40x | $75 |
| $3,000 | $15 | $1,000 | 40x | $25 |

*Capped at FIXED_MAX_NOTIONAL=$3,000. Leverage stays 40x because it's derived from SL%+buffer%, not from balance.

## BTC Example @ $68,400

```
Entry:  $68,400
SL:     $67,374 (1.5% below)
Liq:    $66,690 (2.5% below)
Buffer: $684 (1% between SL and liq) ✅
```

## Files Modified

| File | What Changed |
|------|-------------|
| `live_trading_htf.py` | `_open_position()` — risk chain, liq-buffer-derived leverage |
| `src/api/futures_executor.py` | `open_long/open_short` — set ISOLATED margin before trade |
| `src/api/binance_futures.py` | Added `set_margin_type()` API method |
| `src/api/testnet_executor.py` | `_execute_futures_open()` — pass leverage + trade_value from bot |
| `trade_alerter.py` | Shows leverage/risk/margin in open trade alerts |

## Alert Format

Open trade alerts include:
```
⚡ Leverage: 40x | Risk: $25.14 | Margin: $41.90
```

## Regime-Adaptive SL Interaction

The regime detector can widen SL (e.g., high vol → SL × 1.5 = 2.25%). The risk chain uses **base SL% for sizing** (notional calculation), then the actual SL price uses the regime-adjusted percentage. This means:
- Notional stays consistent regardless of regime
- Wider SL → actual dollar risk may exceed $25 slightly
- Wider SL also increases liq distance (liq = wider_SL + 1%), so leverage auto-adjusts

## Troubleshooting

**"Margin exceeds balance"** — notional scaled down automatically. Check if balance dropped significantly.

**"set_margin_type failed"** — can't change margin type with open position. Will succeed on next trade after position closes. Error is non-fatal.

**"Liquidation too close"** — leverage auto-reduced. Check logs for "Adjusted leverage to Xx for safety".
