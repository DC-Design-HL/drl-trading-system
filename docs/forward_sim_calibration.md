# Forward-Simulator Calibration Report

**Generated:** 2026-06-13T09:38:03.282502+00:00  
**Window:** 2026-05-30 → 2026-06-13 (2 weeks)  
**Sim runtime:** 456.1s  

## Gate criteria — Option B (PROFITABILITY_PLAN.md §3/P2,
## redefined 2026-06-13, see docs/forward_sim_gate_redefinition.md)

  GATE  1. Co-decided directional agreement ≥ 80% (live entries
           where the sim was free to decide — excludes occupancy
           drift and cadence gaps).
  GATE  2. Net PnL same sign.
  WATCH 3. Entry counts, all-live agreement, over-production ratio
           — reported for monitoring, do NOT block promotion.

## GATE 1 — Co-decided directional agreement

_Time-match window: ±30 min. Denominator excludes entries where the sim was holding another trade or had no decision bar._

| Combo | Co-decided | Matched | Agreement | Excluded |
|---|---:|---:|---:|---:|
| BTCUSDT LONG | 2 | 2 | 100.0% | 2 |
| BTCUSDT SHORT | 27 | 22 | 81.5% | 19 |
| ETHUSDT LONG | 4 | 4 | 100.0% | 4 |
| ETHUSDT SHORT | 25 | 23 | 92.0% | 16 |
| SOLUSDT SHORT | 27 | 24 | 88.9% | 13 |
| **overall** | 85 | 75 | **88.2%** | 54 |

**GATE 1: PASS ✅** (threshold ≥ 80%)

## GATE 2 — Net PnL

- Live: **$+518.06**
- Sim:  **$+487.91**
- Ratio (sim / live): **0.94**
- Sign match: **✅**

**GATE 2: PASS ✅**

## Watched metrics (non-gating)

### Entry counts (all live entries)

| Combo | Live | Sim | Δ% | Within ±30% |
|---|---:|---:|---:|:--:|
| BTCUSDT LONG | 4 | 8 | 100.0% | ❌ |
| BTCUSDT SHORT | 46 | 41 | 10.9% | ✅ |
| ETHUSDT LONG | 8 | 13 | 62.5% | ❌ |
| ETHUSDT SHORT | 41 | 37 | 9.8% | ✅ |
| SOLUSDT SHORT | 40 | 64 | 60.0% | ❌ |

- All-live directional agreement (incl. occupancy/cadence gaps): **54.0%**
- Over-production: **93** sim entries with no live match / 139 live = **0.67×** (sim total 163). Driven by non-replayable live guards; watch for growth.

## Verdict

**Overall: PASS ✅** (GATE 1 co-decided agreement + GATE 2 PnL sign)

_The orchestrator may use forward-sim results as a promotion gate only after this report PASSES and Chen acknowledges it on Telegram (PROFITABILITY_PLAN.md §3/P2)._

## Known limitations of v1

* S5 symbol filters (OB-proximity + ADX-directional) ARE now   replicated (P2.D part 1) — the ETH-zero-entries bug is fixed.
* Replayable pre-trade guards (structure-first ADX, exhaustion,   RSI) ARE now applied (P2.D part 2). Guards that cannot be   replayed offline (USDT.D proxy, ext-pos-news, orderbook) are   still assumed-pass — sim may over-count where these block live.
* Stateful post-close gates (cooldown / anti-whipsaw) ARE simulated   (P2.D). ADX + trend-aware RSI bands now come from the live   MarketRegimeDetector (P2.D #1), not a kline approximation — this   lifted directional agreement ~40% -> ~56%.
* Per-decision diagnostic (forward_sim_calibration_diagnosis.md)   shows the entry LOGIC is faithful: among live entries where the   sim was free to decide, agreement is ~90%. The residual headline   gap is occupancy drift (sim busy in a different trade, ~23%) +   cadence/data gaps (~16%), NOT entry-logic disagreement (~5%).
* Occupancy drift is driven by over-production: the sim takes   entries live skipped because live's orderbook / whale / news /   USDT.D guards cannot be replayed offline (assumed-pass). This is   a STRUCTURAL ceiling on timestamp-matched agreement — handled by   the Option B gate (adopted 2026-06-13): co-decided agreement is   the gate, over-production is a watched metric. See   docs/forward_sim_gate_redefinition.md.
* Residuals vs live: RSI *value* is a kline proxy (live reads it   from the API signals bundle) and the conf>=0.90 rescue override   is not replayed (needs model conf + order-flow/whale/mtf signals).
* Funding IS accrued (P2.E): per 8h boundary on entry notional,   LONG pays / SHORT receives. Magnitude is small (sub-dollar per   trade); the PnL ratio moves more from the rolling now() window   pulling a slightly different live+sim trade set each run.
* No BOS/CHOCH profitable-overlay on exits.
