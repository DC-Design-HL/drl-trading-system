# Forward-Sim Gate Redefinition — options for Chen

**Status:** proposal, 2026-06-13. Decision needed before P2 can be
declared "calibrated". Author: profitability-p2 work.

## The problem with the current gate

PROFITABILITY_PLAN.md P2 gates the simulator on **directional agreement
≥ 80%** — for each live entry, did the sim enter the same side within
±30 min? After the P2.D fixes this sits at ~55% (was ~40%). The
per-decision diagnostic (`forward_sim_calibration_diagnosis.md`) shows
*why* it is stuck, and it is **not** because the sim's entry logic is
wrong:

| Why a live entry isn't matched | Share | Nature |
|---|---:|---|
| matched ✅ | ~55% | — |
| sim busy in a different trade (occupancy) | ~23% | timing drift |
| no sim decision bar near that time | ~16% | cadence / data gap |
| sim's entry logic actually disagreed (signal/guard) | **~5%** | real |

**Among live entries where the sim was actually free to decide, it agrees
~90% of the time.** The headline number is dragged down by *occupancy
drift* — the sim takes entries live didn't (it can't see live's
orderbook / whale / news / USDT.D guards, so it over-produces ~98 extra
trades), which keeps it busy when live's real entries arrive. That is a
**structural ceiling**: those guards depend on real-time data the offline
sim cannot replay, so 80% timestamp-matched recall is likely unreachable —
and chasing it would mean overfitting the sim, not improving it.

## What the gate is really for

The sim is a tool to **validate config changes** before the autonomous
loop applies them. What we need to trust is: *does the sim's entry +
exit + PnL logic faithfully reproduce live decisions when given the same
inputs?* — not *does the sim stay bar-for-bar in lockstep with live over
two weeks of compounding occupancy drift?* The current gate measures the
second; we care about the first.

## Options

### Option A — Lower the single threshold (simplest)
Keep directional-agreement-on-all-live-entries, set the bar at the
measured ceiling minus a margin (e.g. ≥ 55%). **Con:** arbitrary, and
conflates entry-logic fidelity with occupancy drift — a future regression
in either is indistinguishable.

### Option B — Co-decided agreement + separate drift bounds (recommended)
Split the one number into what it actually measures:
1. **Co-decided directional agreement ≥ 80%** — restrict to live entries
   where the sim had a free decision bar within ±30 min (exclude
   occupancy + no-bar). This directly tests the shared entry logic.
   Currently ~90% → already passes.
2. **Net PnL sign match + ratio within band** (already in the gate;
   currently sign ✅, ratio 0.83).
3. **Over-production bound** (new, looser): sim-only entries / live
   entries ≤ a documented multiple (currently ~0.7×). Tracks the
   non-replayable-guard drift without blocking on it; alert if it grows.

Promotion eligibility requires (1) + (2); (3) is a watched health metric,
not a hard block.

### Option C — Co-evaluated confusion matrix (most rigorous)
For every decision timestamp where **both** sim and live evaluated while
flat, build a LONG/SHORT/FLAT confusion matrix and gate on agreement
(e.g. Cohen's κ ≥ threshold). Most statistically honest, but needs live
per-decision logs (HOLD reasons), which aren't in `trading.db` today —
would require capturing live decision traces first. Larger build.

## Recommendation

**Option B.** It isolates the question we actually care about (entry-logic
fidelity, ~90%) from the structural occupancy drift, keeps the PnL check,
and demotes over-production to a monitored metric instead of an
unreachable hard gate. Option C is the right long-term target once live
decision-trace logging exists (a natural P4/attribution add-on).

## If approved, the plan edit
Update PROFITABILITY_PLAN.md P2 acceptance to Option B's three checks,
and update `calibrate_forward_sim.py` to compute co-decided agreement
(it already has the sim entries + live pairs; needs the sim decision
trace, which `run_forward_sim(trace=...)` now provides).
