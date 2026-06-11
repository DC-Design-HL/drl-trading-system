# Forward-Simulator Calibration Report

**Generated:** 2026-06-11T21:30:08.676225+00:00  
**Window:** 2026-05-28 → 2026-06-11 (2 weeks)  
**Sim runtime:** 567.7s  

## Gate criteria (PROFITABILITY_PLAN.md §3/P2)

  1. Entry count per (symbol, side) within ±30%
  2. Directional agreement on overlapping entries ≥ 80%
  3. Net PnL same sign

## 1. Entry counts

| Combo | Live | Sim | Δ% | Within ±30% |
|---|---:|---:|---:|:--:|
| BTCUSDT LONG | 4 | 8 | 100.0% | ❌ |
| BTCUSDT SHORT | 46 | 56 | 21.7% | ✅ |
| ETHUSDT LONG | 6 | 0 | 100.0% | ❌ |
| ETHUSDT SHORT | 43 | 0 | 100.0% | ❌ |
| SOLUSDT SHORT | 42 | 77 | 83.3% | ❌ |

**Entry-count gate: FAIL**

## 2. Directional agreement

_Time-match window: ±30 minutes_

| Combo | Live entries | Matched in sim | Agreement |
|---|---:|---:|---:|
| BTCUSDT LONG | 4 | 1 | 25.0% |
| BTCUSDT SHORT | 46 | 24 | 52.2% |
| ETHUSDT LONG | 6 | 0 | 0.0% |
| ETHUSDT SHORT | 43 | 0 | 0.0% |
| SOLUSDT SHORT | 42 | 22 | 52.4% |
| **overall** | — | — | **33.3%** |

**Directional-agreement gate: FAIL** (threshold ≥ 80%)

## 3. Net PnL

- Live: **$+445.79**
- Sim:  **$+388.08**
- Ratio (sim / live): **0.87**
- Sign match: **✅**

## Verdict

**Overall: FAIL ❌**

_The orchestrator may use forward-sim results as a promotion gate only after this report PASSES and Chen acknowledges it on Telegram (PROFITABILITY_PLAN.md §3/P2)._

## Known limitations of v1

* S5 symbol filters (OB-proximity + ADX-directional) are NOT   replicated yet — ETH entries will be undercounted.
* Pre-trade guards (RSI, ADX, exhaustion, USDT.D, ext-pos-news,   anti-whipsaw, cooldown, min-hold) NOT applied — sim entries   may be over-counted vs live where these guards block.
* Funding accrual not yet wired (P2.E); fees + slippage only.
* No BOS/CHOCH profitable-overlay on exits.
