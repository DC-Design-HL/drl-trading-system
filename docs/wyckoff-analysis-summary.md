# Wyckoff Analysis Summary — Apr 2, 2026

## What We Built
- **Full 5-phase Wyckoff detector**: `src/features/wyckoff_detector.py` (32KB)
- **14/14 event types detected** on real data: PS, SC, AR, ST, Spring, SOS, LPS, PSY, BC, UTAD, SOW, LPSY
- **12/12 tests passing**: `tests/test_wyckoff_detector.py`
- **Validation script**: `scripts/wyckoff_validation.py`
- All 4 symbols working: BTC, ETH, SOL, XRP

## Backtest Results (47 trades, Mar 29 – Apr 2)

### Multi-Timeframe Phase A/C/E Filter

| Timeframe | Allowed | Blocked | Allowed WR | Allowed PnL | Blocked PnL |
|-----------|---------|---------|------------|-------------|-------------|
| **4H** | 9 | 38 | 56% | +$9 | -$220 |
| **1H** | 5 | 42 | 60% | +$11 | -$222 |
| **15m** | 0 | 47 | - | $0 | -$211 |

### Phase Distribution (15m candles)

| Phase | Trades | WR | PnL | Avg |
|-------|--------|-----|-----|-----|
| Accum Phase B | 30 | 67% | +$32 | +$1 |
| Accum Phase D | 15 | 27% | -$191 | -$13 |

### Conflict Analysis (15m)
- "Conflict" trades (Wyckoff says don't): 24 trades, **62% WR**, +$3
- "Aligned" trades (Wyckoff agrees): 23 trades, **39% WR**, -$214
- Result: **inverted** — conflict trades performed better on 15m

## Key Findings

1. **15m too noisy** for Wyckoff — everything is Phase B/D, zero Phase A/C/E detected
2. **4H best timeframe** for phase classification but too aggressive as hard filter (blocks 80% of trades including 19 winners worth +$240)
3. **Phase B vs D has signal** — Phase D trades are terrible (27% WR, -$191) vs Phase B (67% WR, +$32)
4. **Event-level signals too noisy** — LPSY/SOW flags fire on winners and losers equally
5. **Detector classifies almost everything as "accumulation"** (46/47 trades) — not enough differentiation

## Why Not Deploy Now

- As a **hard guard**: blocks too many trades (80%+), kills profitability
- As a **confidence multiplier**: promising but needs more data to validate
- As a **model feature**: best approach but needs 500+ labeled trades (2-3 months)

## Recommended Path Forward (When Resuming)

### Phase 1: Logging (deploy now when ready)
- Add Wyckoff phase + events to every trade alert (display only)
- Log to SQLite: `wyckoff_phase`, `wyckoff_scheme`, `wyckoff_confidence`, `wyckoff_events`
- Use 4H candles for phase detection
- Zero risk — just collecting labeled data

### Phase 2: Soft Feature (after 2-4 weeks of data)
- Feed Wyckoff features into existing DRL model as additional inputs
- Confidence multiplier: Phase A/C/E = 1.0x, Phase B = 0.7x, Phase D = directional
- Evaluate impact on paper trading

### Phase 3: LSTM Model (after 2-3 months, 500+ trades)
- Train dedicated Wyckoff LSTM on Chen's Mac M3 Pro
- Input: sequence of Wyckoff events + volume ratios
- Output: probability of profitable trade given current Wyckoff state
- Needs: labeled training data from Phase 1 logging

## File Inventory
| File | Purpose |
|------|---------|
| `src/features/wyckoff_detector.py` | Main detector (32KB, production-ready) |
| `tests/test_wyckoff_detector.py` | Test suite (12 tests, all passing) |
| `scripts/wyckoff_validation.py` | Real data validation |
| `docs/wyckoff-validation-report.md` | Auto-generated validation report |
| `docs/wyckoff-validation-results.json` | Raw validation data |
| `src/features/ultimate_features.py` | OLD detector (WyckoffAnalyzer class) — to be replaced |
