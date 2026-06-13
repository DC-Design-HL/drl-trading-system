# Forward-Simulator Calibration Report

**Generated:** 2026-06-13T01:43:09.045110+00:00  
**Window:** 2026-05-30 → 2026-06-13 (2 weeks)  
**Sim runtime:** 516.2s  

## Gate criteria (PROFITABILITY_PLAN.md §3/P2)

  1. Entry count per (symbol, side) within ±30%
  2. Directional agreement on overlapping entries ≥ 80%
  3. Net PnL same sign

## 1. Entry counts

| Combo | Live | Sim | Δ% | Within ±30% |
|---|---:|---:|---:|:--:|
| BTCUSDT LONG | 4 | 11 | 175.0% | ❌ |
| BTCUSDT SHORT | 46 | 41 | 10.9% | ✅ |
| ETHUSDT LONG | 8 | 13 | 62.5% | ❌ |
| ETHUSDT SHORT | 42 | 30 | 28.6% | ✅ |
| SOLUSDT SHORT | 40 | 50 | 25.0% | ✅ |

**Entry-count gate: FAIL**

## 2. Directional agreement

_Time-match window: ±30 minutes_

| Combo | Live entries | Matched in sim | Agreement |
|---|---:|---:|---:|
| BTCUSDT LONG | 4 | 2 | 50.0% |
| BTCUSDT SHORT | 46 | 21 | 45.7% |
| ETHUSDT LONG | 8 | 1 | 12.5% |
| ETHUSDT SHORT | 42 | 19 | 45.2% |
| SOLUSDT SHORT | 40 | 14 | 35.0% |
| **overall** | — | — | **40.7%** |

**Directional-agreement gate: FAIL** (threshold ≥ 80%)

## 3. Net PnL

- Live: **$+517.49**
- Sim:  **$+430.55**
- Ratio (sim / live): **0.83**
- Sign match: **✅**

## Verdict

**Overall: FAIL ❌**

_The orchestrator may use forward-sim results as a promotion gate only after this report PASSES and Chen acknowledges it on Telegram (PROFITABILITY_PLAN.md §3/P2)._

## Known limitations of v1

* S5 symbol filters (OB-proximity + ADX-directional) ARE now   replicated (P2.D part 1) — the ETH-zero-entries bug is fixed.
* Replayable pre-trade guards (structure-first ADX, exhaustion,   RSI) ARE now applied (P2.D part 2). Guards that cannot be   replayed offline (USDT.D proxy, ext-pos-news, orderbook) are   still assumed-pass — sim may over-count where these block live.
* Stateful post-close gates (cooldown after loss, anti-whipsaw   reversal block) ARE now simulated (P2.D). This improved the net   PnL ratio (0.68→0.83) and tightened entry counts but did NOT move   directional agreement (~40%) — so post-close timing was NOT the main driver.
* DOMINANT REMAINING GAP — entry-signal divergence: only ~40% of   live entries have a same-side sim entry within ±30min, and ~half   of sim entries land at times live did not trade. The mismatch is   at the entry-decision level, not the post-close gates. Suspects:   (a) the MarketStructure BOS/CHOCH signal firing on different bars   (data-window / cadence-phase differences); (b) live guards that   cannot be replayed offline (orderbook, whale, news, USDT.D) — this   may cap achievable agreement below 80%. Needs a per-decision   diagnostic (sim vs live logs) to localise before the next fix.
* LONG-side over-production: sim emits more LONG entries than live   on BTC/ETH — a LONG-side gate present in live is not replicated.
* Funding accrual not yet wired (P2.E); fees + slippage only.
* No BOS/CHOCH profitable-overlay on exits.
