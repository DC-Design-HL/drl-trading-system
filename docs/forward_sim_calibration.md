# Forward-Simulator Calibration Report

**Generated:** 2026-06-13T08:29:25.285131+00:00  
**Window:** 2026-05-30 → 2026-06-13 (2 weeks)  
**Sim runtime:** 449.1s  

## Gate criteria (PROFITABILITY_PLAN.md §3/P2)

  1. Entry count per (symbol, side) within ±30%
  2. Directional agreement on overlapping entries ≥ 80%
  3. Net PnL same sign

## 1. Entry counts

| Combo | Live | Sim | Δ% | Within ±30% |
|---|---:|---:|---:|:--:|
| BTCUSDT LONG | 4 | 9 | 125.0% | ❌ |
| BTCUSDT SHORT | 46 | 48 | 4.3% | ✅ |
| ETHUSDT LONG | 8 | 14 | 75.0% | ❌ |
| ETHUSDT SHORT | 41 | 44 | 7.3% | ✅ |
| SOLUSDT SHORT | 40 | 69 | 72.5% | ❌ |

**Entry-count gate: FAIL**

## 2. Directional agreement

_Time-match window: ±30 minutes_

| Combo | Live entries | Matched in sim | Agreement |
|---|---:|---:|---:|
| BTCUSDT LONG | 4 | 1 | 25.0% |
| BTCUSDT SHORT | 46 | 26 | 56.5% |
| ETHUSDT LONG | 8 | 4 | 50.0% |
| ETHUSDT SHORT | 41 | 26 | 63.4% |
| SOLUSDT SHORT | 40 | 21 | 52.5% |
| **overall** | — | — | **56.1%** |

**Directional-agreement gate: FAIL** (threshold ≥ 80%)

## 3. Net PnL

- Live: **$+518.06**
- Sim:  **$+701.46**
- Ratio (sim / live): **1.35**
- Sign match: **✅**

## Verdict

**Overall: FAIL ❌**

_The orchestrator may use forward-sim results as a promotion gate only after this report PASSES and Chen acknowledges it on Telegram (PROFITABILITY_PLAN.md §3/P2)._

## Known limitations of v1

* S5 symbol filters (OB-proximity + ADX-directional) ARE now   replicated (P2.D part 1) — the ETH-zero-entries bug is fixed.
* Stateful post-close gates (cooldown / anti-whipsaw) ARE simulated   (P2.D). ADX + trend-aware RSI bands now come from the live   MarketRegimeDetector (P2.D #1), not a kline approximation — this   lifted directional agreement ~40% → ~56%.
* Per-decision diagnostic (forward_sim_calibration_diagnosis.md)   shows the entry LOGIC is faithful: among live entries where the   sim was free to decide, agreement is ~90%. The residual headline   gap is occupancy drift (sim busy in a different trade, ~23%) +   cadence/data gaps (~16%), NOT entry-logic disagreement (~5%).
* Occupancy drift is driven by over-production: the sim takes   entries live skipped because live's orderbook / whale / news /   USDT.D guards cannot be replayed offline (assumed-pass). This is   a STRUCTURAL ceiling on timestamp-matched agreement — see   docs/forward_sim_gate_redefinition.md (proposed Option B).
* Residuals vs live: RSI *value* is a kline proxy (live reads it   from the API signals bundle) and the conf≥0.90 rescue override   is not replayed.
* Funding accrual not yet wired (P2.E); fees + slippage only.
* No BOS/CHOCH profitable-overlay on exits.
