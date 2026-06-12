# Forward-Simulator Calibration Report

**Generated:** 2026-06-12T21:10:40.424066+00:00  
**Window:** 2026-05-29 → 2026-06-12 (2 weeks)  
**Sim runtime:** 470.2s  

## Gate criteria (PROFITABILITY_PLAN.md §3/P2)

  1. Entry count per (symbol, side) within ±30%
  2. Directional agreement on overlapping entries ≥ 80%
  3. Net PnL same sign

## 1. Entry counts

| Combo | Live | Sim | Δ% | Within ±30% |
|---|---:|---:|---:|:--:|
| BTCUSDT LONG | 4 | 10 | 150.0% | ❌ |
| BTCUSDT SHORT | 46 | 45 | 2.2% | ✅ |
| ETHUSDT LONG | 8 | 14 | 75.0% | ❌ |
| ETHUSDT SHORT | 42 | 31 | 26.2% | ✅ |
| SOLUSDT SHORT | 41 | 53 | 29.3% | ✅ |

**Entry-count gate: FAIL**

## 2. Directional agreement

_Time-match window: ±30 minutes_

| Combo | Live entries | Matched in sim | Agreement |
|---|---:|---:|---:|
| BTCUSDT LONG | 4 | 2 | 50.0% |
| BTCUSDT SHORT | 46 | 22 | 47.8% |
| ETHUSDT LONG | 8 | 1 | 12.5% |
| ETHUSDT SHORT | 42 | 19 | 45.2% |
| SOLUSDT SHORT | 41 | 12 | 29.3% |
| **overall** | — | — | **39.7%** |

**Directional-agreement gate: FAIL** (threshold ≥ 80%)

## 3. Net PnL

- Live: **$+504.31**
- Sim:  **$+344.28**
- Ratio (sim / live): **0.68**
- Sign match: **✅**

## Verdict

**Overall: FAIL ❌**

_The orchestrator may use forward-sim results as a promotion gate only after this report PASSES and Chen acknowledges it on Telegram (PROFITABILITY_PLAN.md §3/P2)._

## Known limitations of v1

* S5 symbol filters (OB-proximity + ADX-directional) ARE now   replicated (P2.D part 1) — the ETH-zero-entries bug is fixed.
* Replayable pre-trade guards (structure-first ADX, exhaustion,   RSI) ARE now applied (P2.D part 2). Guards that cannot be   replayed offline (USDT.D proxy, ext-pos-news, orderbook) are   still assumed-pass — sim may over-count where these block live.
* Entry TIMING still diverges: even where aggregate counts match,   directional agreement on overlapping timestamps is low. Leading   suspects: stateful cooldown / min-hold / anti-whipsaw not yet   simulated, and live's continuous eval cadence vs the sim's   per-bar-close cadence. This is the dominant remaining gap.
* LONG-side over-production: sim emits more LONG entries than live   on BTC/ETH — a LONG-side gate present in live is not replicated.
* Funding accrual not yet wired (P2.E); fees + slippage only.
* No BOS/CHOCH profitable-overlay on exits.
