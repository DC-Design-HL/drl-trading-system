---
name: wyckoff-implementation
description: Wyckoff Market Cycle detector for the DRL trading system. Use when resuming Wyckoff work — phase detection, event identification, integrating signals, or training the LSTM model. Status ON HOLD as of Apr 2, 2026.
---

# Wyckoff Implementation — DRL Trading System

## Status: ON HOLD (Apr 2, 2026)

Detector is built and validated. Backtest showed it's not ready as a guard/filter yet. Parked until we have enough labeled data (500+ trades, ~2-3 months).

## What's Done
- ✅ `src/features/wyckoff_detector.py` (32KB) — 14/14 events, all 4 symbols
- ✅ `tests/test_wyckoff_detector.py` — 12/12 tests passing
- ✅ `scripts/wyckoff_validation.py` — real data validation
- ✅ Full backtest against 47 trades across 3 timeframes (4H, 1H, 15m)
- ✅ Detailed analysis doc: `docs/wyckoff-analysis-summary.md`

## Why On Hold
- **15m too noisy** — everything Phase B/D, zero Phase A/C/E
- **4H too aggressive** as filter — blocks 80% of trades including winners
- **Event-level conflicts inverted** — "conflict" trades had 62% WR vs "aligned" 39%
- **Need more data** — 47 trades over 4 days is not enough

## Backtest Key Numbers
| Timeframe | Allowed (A/C/E) | WR | PnL | Blocked Winners Lost |
|-----------|------------------|----|-----|---------------------|
| 4H | 9 trades | 56% | +$9 | +$240 |
| 1H | 5 trades | 60% | +$11 | +$261 |
| 15m | 0 trades | - | $0 | +$299 |

## Resume Plan (3 phases)

### Phase 1: Start Logging (when Chen says go)
- Add Wyckoff to trade alerts (display only, like whale signal)
- Log to SQLite per trade: phase, scheme, confidence, events
- Use **4H candles** for detection
- Zero risk, just data collection

### Phase 2: Soft Feature (after 2-4 weeks of logged data)
- Feed Wyckoff features into DRL model as inputs
- Or use as confidence multiplier: A/C/E = 1.0x, B = 0.7x, D = directional

### Phase 3: LSTM Model (after 2-3 months, 500+ trades)
- Train on Chen's Mac M3 Pro
- Input: event sequences + volume ratios
- Output: trade profitability probability

## Key Files
| File | Purpose |
|------|---------|
| `src/features/wyckoff_detector.py` | Main detector (32KB) |
| `tests/test_wyckoff_detector.py` | 12 tests, all passing |
| `scripts/wyckoff_validation.py` | Real data validation |
| `docs/wyckoff-analysis-summary.md` | Full analysis + backtest results |
| `docs/wyckoff-validation-report.md` | Event validation report |
| `src/features/ultimate_features.py` | OLD detector (to replace later) |
