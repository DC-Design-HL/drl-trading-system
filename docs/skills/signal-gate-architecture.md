# Signal Gate Architecture

## Overview
The DRL trading system uses a multi-layer filtering architecture between the model's decision and actual trade execution. Each layer can block or modify the trade independently.

## Current Architecture (as of 2026-03-31)

```
Model Decision (LONG/SHORT/HOLD)
       │
       ▼
┌──────────────────────────┐
│ Layer 0: EXHAUSTION      │ Block if price > 3 ATR from VWAP
│ (always active)          │ 
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Layer 1: ORDERBOOK GUARD │ ✅ LIVE
│ (Golden Guard)           │ Block when orderbook bias contradicts direction
│                          │ Applies to ALL tiers
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Layer 2: SIGNAL GATE     │ ✅ LIVE  
│ (Tier 1/2 system)        │ Tier 1 (conf≥0.80): autonomous
│                          │ Tier 2 (conf<0.80): needs 2/4 confirmations
│                          │ Signals: MTF, Order Flow, Regime, Orderbook Imbalance
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ POSITION SIZING          │ Fixed-dollar-risk
│                          │ dollar_risk = (balance × risk_pool%) / budget_parts
└──────────┬───────────────┘
           │
           ▼
     OPEN POSITION

```

## Planned Architecture (pending approval)

```
Model Decision
       │
       ▼
  Layer 0: Exhaustion Filter (existing)
       │
       ▼  
  Layer 1: Orderbook Guard ← LIVE ✅
       │
       ▼
  Layer 1b: Macro Trend Guard (SMA+RSI) ← PLANNED (toggle)
       │
       ▼
  Layer 2: Signal Gate (existing)
       │
       ▼
  Layer 3: OB Tier Sizing ← PLANNED
       │  T1 (in OB zone) = 1.5x
       │  T2 (near OB) = 1.0x  
       │  T3 (no OB) = 0.5x
       │
       ▼
  OPEN POSITION (sized by tier)
```

## Key Files
- `live_trading_htf.py` — Main trading loop, all guards and gate logic
- `src/features/orderbook_imbalance.py` — Orderbook data fetcher (futures API, 3-sample avg)
- `src/signals/order_blocks.py` — Order Block detector (standalone, not yet integrated)
- `src/signals/order_blocks_config.py` — OB config
- `src/features/order_flow.py` — Order flow analysis (used by signal gate)
- `src/features/whale_tracker.py` — Whale tracking signals
- `src/features/mtf_analyzer.py` — Multi-timeframe analysis
- `src/features/regime_detector.py` — Market regime detection

## Signal Gate Details (Layer 2)

### Tier 1: Autonomous (confidence ≥ 0.80)
- Model decides alone
- Still subject to Layer 0 (exhaustion) and Layer 1 (orderbook guard)
- Constant: `SIGNAL_GATE_AUTONOMOUS = 0.80`

### Tier 2: Consensus (confidence < 0.80)
- Needs ≥ 2 out of 4 market signals to agree:
  1. **MTF Alignment** — all timeframes agree with direction
  2. **Order Flow Score** — net buying/selling pressure (threshold: ±0.20)
  3. **Regime** — not fighting a strong trend (ADX ≥ 25)
  4. **Orderbook Imbalance** — bid/ask pressure (threshold: ±0.30)
- Constant: `SIGNAL_GATE_MIN_CONFIRMS = 2`

## Constants (live_trading_htf.py)
```python
SIGNAL_GATE_AUTONOMOUS = 0.80
SIGNAL_GATE_MIN_CONFIRMS = 2
SIGNAL_GATE_OF_THRESHOLD = 0.20
SIGNAL_GATE_OB_THRESHOLD = 0.30
SIGNAL_GATE_REGIME_ADX_MIN = 25.0
ORDERBOOK_GUARD_ENABLED = True
```

## Backtest Comparison (Mar 24-31)
| Strategy | Trades | WR | PnL |
|---|---|---|---|
| No guards | 72 | 45.7% | $45 |
| Orderbook guard only | 59 | 51.8% | $170 |
| OB guard + SMA combo | 46 | 56.5% | $239 |
| Full stack (guard + OB tiers) | 40 | 62.5% | $221 |

## Rules
- **NEVER push new guard logic without Chen's approval**
- Each new layer should be tested in shadow mode first (log only, don't affect trades)
- All guards are fail-open (if data unavailable, trade proceeds)
- Every guard has a kill switch constant (ENABLED = True/False)
