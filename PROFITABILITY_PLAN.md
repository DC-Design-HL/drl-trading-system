# PROFITABILITY_PLAN.md — From Loss-Suppression to Edge

**Status:** Approved by Chen 2026-06-11 for implementation.
**Authored:** 2026-06-11, from a full code review of `feature/autonomous-loop`
(HEAD `08512ea`) conducted on Chen's Mac.
**Audience:** a Claude Code session running on the Hetzner server with no
memory of the authoring conversation. Everything you need is in this file
plus the referenced code. Read `CLAUDE.md` and `PLAN.md` first — their rules
apply on top of this plan.

---

## 0. Mission

The system currently runs 4 bots (BTC/ETH/SOL/XRP perp futures, Binance
testnet) with a structure-first entry engine and an autonomous
self-improvement loop that can only *suppress* entries. **Goal: make the
system profitable on rolling 30-day net PnL after fees** by (a) fixing a
correctness bug in the loop's apply surface, (b) building the forward
simulation that lets the loop validate *improving* changes, not only
suppressing ones, and (c) widening the loop's apply surface safely.

Optimization order (unchanged from PLAN.md §7): 30d net PnL after fees →
max DD ≤ 8% → Sharpe ≥ 0.8 → Sortino ≥ 1.2. Win rate is tracked, never
targeted.

### Branch & workflow rules (Chen's explicit instruction)

* **All implementation work goes on a new branch created from
  `feature/autonomous-loop`**, named `feature/profitability-p<N>` per phase
  (e.g. `feature/profitability-p1`). Do NOT commit implementation directly
  to `feature/autonomous-loop`, `dev`, or `main`.
* One phase per branch. Merge back into `feature/autonomous-loop` only
  after the phase's acceptance criteria pass and Chen confirms on Telegram.
* Conventional commits (`fix:`, `feat:`, `test:`, `docs:`). Every
  behavioral change ships with tests in the same commit.
* Run the relevant test suite before every commit:
  `python -m pytest tests/test_self_improve/ tests/` (skip
  `requires_model` / `requires_api` markers if assets are missing).
* After any change to live-bot code: `./start_services.sh` (CLAUDE.md rule 6)
  — but only once the phase branch is merged and Chen has approved deploy.
* Telegram protocol applies (CLAUDE.md rule 9): report completion of every
  phase to the primary chat.

### Hard rules that bound everything below (do not relax)

1. Testnet only. No mainnet, ever.
2. No model training on the server (2 CPU / 3.7 GB RAM). Training happens
   on Chen's Mac.
3. Never open a position without SL and TP.
4. Sizing, leverage, base SL/TP percentages, `FIXED_MAX_NOTIONAL`,
   daily-loss/max-DD halts, symbols, and venue remain **Chen-only**
   decisions. Nothing in this plan gives the autonomous loop access to them.
5. Kill switch (`data/self_improve/AUTONOMY_DISABLED`) and circuit breaker
   (`src/self_improve/live_apply.py`: 5% realized loss / 8% DD since apply)
   keep working exactly as they do today through every phase.

---

## 1. System snapshot (as of `08512ea`, verified by code reading)

How a live trade happens (`live_trading_htf.py`, one process per symbol):

1. `run_iteration()` (line ~3416): fetch 15m candles → manage open position
   (partial TPs at 1R/2R, MFE/MAE, trailing, stagnant exit) → compute 5m
   BOS/CHOCH structure signals (`src/signals/bos_choch.py::MarketStructure`,
   with 1h/4h confirmation).
2. `STRUCTURE_FIRST_MODE = True` (hardcoded, line ~390): entry direction
   comes from `_get_structure_direction()` (line ~2340) — trend +
   last-signal alignment, per-symbol S1/S5 filter config, and the
   `SYMBOL_SIDE_BLOCKLIST` check. Confidence = the structure signal's
   confidence (`sig.get("confidence")`), **not** PPO model confidence.
3. `execute_trade()` (line ~2849) runs the guard chain: loss cooldown,
   anti-whipsaw, min-hold, ranging/ADX block, exhaustion (3 ATR from VWAP),
   USDT.D proxy (blocks LONGs), extreme-positive-news fade (blocks LONGs),
   orderbook guard, RSI/ADX guard. **In structure-first mode the
   confidence-floor guards and the market-signal gate are skipped** (the
   `if not STRUCTURE_FIRST_MODE:` blocks at ~2919, ~2927, ~3031).
4. `_open_position()` (line ~3066): fixed-dollar-risk sizing — 10% risk
   pool / 20 parts = 0.5% of balance risked per trade, notional =
   risk / 1.5% SL, capped at `FIXED_MAX_NOTIONAL=$3000`, halved if recent
   symbol WR < 40%. Regime-adaptive SL/TP multipliers; partial-TP schedule.

Current entry universe after the blocklist (`SYMBOL_SIDE_BLOCKLIST`, line
~182): BTC LONG/SHORT, ETH LONG/SHORT, SOL SHORT. SOL LONG and both XRP
sides are blocked. **The XRP bot runs but can never enter.**

The self-improvement loop (`src/self_improve/`): performance monitor →
triggers (T1–T7) → Researcher (Opus, proposes one config change) → Risk
Officer → backtest harness (**replay-only**: re-filters historical trades
from `data/trading.db`) → 7-day paper stage (same replay, forward window,
needs ≥15 kept closes) → canary (override file applied live if
`AUTONOMY_ARMED`, 48h ambient-PnL watch) → live (circuit-breaker
monitored). Apply surface (`runtime_overrides.APPLYABLE_KEYS`):
`MIN_CONFIDENCE`, `SYMBOL_MIN_CONFIDENCE`, `SYMBOL_DIRECTIONAL_CONF`,
`SYMBOL_SIDE_BLOCKLIST(_ADD)` — **monotonic tightening only**.

---

## 2. Findings (what's wrong, with evidence)

### F1 — BUG: 3 of the 4 autonomous knobs have zero live effect

`execute_trade()` skips `MIN_CONFIDENCE` / `SYMBOL_MIN_CONFIDENCE` /
`SYMBOL_DIRECTIONAL_CONF` checks when `STRUCTURE_FIRST_MODE` is true
(live_trading_htf.py ~2919, ~2927) — and structure-first is hardcoded on.
Only the blocklist is consulted (inside `_get_structure_direction`,
~2444). Meanwhile the backtest harness (`backtest_harness._block_reason`)
and the paper evaluation **do** block historical trades by those floors.
Consequence: the loop can validate a confidence-floor experiment end-to-end
and "apply" it live as a no-op; the 48h canary then attributes ambient
market PnL to it. The researcher realignment in commit `08512ea` points it
at these same dead knobs.

Aggravating detail: the `confidence` recorded on OPEN rows in
structure-first mode is BOS/CHOCH confidence — a different quantity and
scale from the PPO model confidence that the existing floor values (ETH
0.80, ETH/SOL LONG 0.95) were calibrated against.

### F2 — Structural ceiling: the loop can only shrink the system

Monotonic tightening + replay-only validation = the loop can only remove
trades. It cannot tune exits, restore a block that has stopped being
justified, or improve entries. Its fixed point is "don't trade." The
profit-side levers (entry signal quality, exit schedule, sizing) are all
outside its reach, and replay mode structurally can't evaluate them
(it cannot invent trades the bot didn't take, nor re-time exits).

### F3 — The loop starves itself statistically

The paper gate requires ≥15 candidate closes in 7 days
(`paper_trader.MIN_PAPER_CLOSES`). Each promoted suppression lowers the
close rate; with the universe already down to 5 (symbol, side) combos,
candidate experiments will increasingly fail on "insufficient closes"
regardless of merit.

### F4 — Canary attribution was never implemented

PLAN.md §12.3 requires canary trades to be attribution-tagged.
`live_apply.measure_since()` actually sums **all** closes after the apply
timestamp. A 48h ambient window with ≥3 closes is dominated by market
noise; promotions and rollbacks are weakly causal at best.

### F5 — The system's alpha sources are sidelined and unmeasured

The PPO/DRL models pick no entries (used only in the stagnant-exit keep
check, ~3701). Whale flow, news sentiment, MTF, order flow act only as
veto guards. The single entry signal (5m BOS/CHOCH) has no measured
standalone expectancy, and nothing in the system measures whether each
guard/signal adds or subtracts money. Exits are static constants.

### F6 — Funding rates are absent from all accounting

Perp positions held hours-to-days pay/receive funding every 8h. Funding
appears nowhere: not in live PnL, not in the harness, not in metrics. At
the system's hold times this can flip the sign of thin edges.

---

## 3. Phases

Execute in order. P0 and P1 are independent of each other and can be done
in either order (or in parallel branches). P2 blocks P3; P4 and P5 can
proceed in parallel with P2 after P1 lands.

---

### P0 — Ground-truth report (no behavior change)

**Goal:** replace assumptions with data before tuning anything.

**Branch:** `feature/profitability-p0`

**Build** `scripts/self_improve/ground_truth_report.py` producing
`docs/ground_truth/<YYYY-MM-DD>-report.md` from `data/trading.db`
(always `is_testnet=1`, paired OPEN/CLOSE via the same FIFO pairing as
`backtest_harness.pair_open_close`). Sections:

1. **Headline:** net PnL, Sharpe, Sortino, PF, max DD, n trades — 7d / 30d
   / since 2026-05-01, portfolio and per symbol-side.
2. **Exit-reason breakdown:** PnL and count by close reason (SL_HIT,
   TP_HIT, PARTIAL_TP*, TRAILING, STAGNANT_EXIT, REVERSE_CLOSE_*,
   REVERSAL_BLOCKED_CLOSE) — this tells us whether the exit stack or the
   entry signal is losing the money.
3. **Confidence calibration:** structure-confidence deciles vs win rate
   and expectancy (only trades after structure-first go-live). This
   decides the floor values P1 enables.
4. **MFE/MAE analysis:** for losing trades, how far did they go in our
   favor first (was there a better exit)? For TP_HIT winners, how much
   further did price run (are TPs too tight)? Uses the mfe_pct/mae_pct
   fields logged since Phase 1.
5. **Guard counterfactuals where logs allow:** count of blocked entries
   per guard from bot logs (`logs/*_live.log` grep for the 🚫 markers),
   to size how much each guard suppresses.
6. **Self-improve audit:** dump of `experiments` (id, proposal, stage,
   rejection reason), `decisions` outcomes, `agent_runs` cost totals.
7. **Funding estimate:** for each closed trade, estimated funding paid
   (fetch historical funding rates via ccxt `fetch_funding_rate_history`;
   if testnet doesn't serve it, use mainnet rates as proxy — flag which).

**Acceptance:** report generated and committed; Telegram summary to Chen
with the 5 most actionable numbers. **No thresholds are changed in P0.**

---

### P1 — Make the loop's apply surface real (bug fix)

**Goal:** every key the loop can deploy must demonstrably change live
behavior, and live behavior must match what the harness simulates.

**Branch:** `feature/profitability-p1`

**Design decision (made, implement as specified):** do NOT apply the
legacy model-confidence floors to structure confidence — values like 0.95
would instantly strangle entries. Instead introduce parallel
structure-confidence floors with no-op baselines, and make them the
loop's apply surface in structure-first mode.

**Changes:**

1. `live_trading_htf.py`: add module-level constants (near the existing
   confidence block ~149):
   ```python
   STRUCT_MIN_CONFIDENCE = 0.0           # global structure-confidence floor
   STRUCT_SYMBOL_MIN_CONFIDENCE: dict = {}        # {"ETHUSDT": 0.6}
   STRUCT_SYMBOL_DIRECTIONAL_CONF: dict = {}      # {"ETHUSDT": {"LONG": 0.7}}
   ```
   Enforce them at the end of `_get_structure_direction()` (after the
   blocklist check, before returning a direction): resolve the effective
   floor as directional → per-symbol → global (first defined wins, same
   precedence as the harness `_block_reason`), compare against the
   structure signal's confidence, log a distinct marker
   (`🚫 STRUCT_CONF floor`) on block. Baselines of 0.0/{} mean **zero
   behavior change at deploy**.
2. `src/self_improve/runtime_overrides.py`: add the three `STRUCT_*` keys
   to `APPLYABLE_KEYS` and to `tighten_overrides` /
   `check_tightening_only` (raise-only, identical mechanics to their
   legacy twins). The legacy keys STAY applyable (they matter if
   structure-first is ever switched off) but see item 5.
3. `live_trading_htf._apply_runtime_overrides()`: wire the three new
   globals through the override loader.
4. `backtest_harness.py`: recognize the `STRUCT_*` keys. In replay mode
   they filter on the recorded OPEN confidence **only for trades after
   the structure-first go-live timestamp** (add module constant
   `STRUCTURE_FIRST_LIVE_SINCE` — recover the actual date from git
   history / DB and document it). Same for `paper_trader`. Warn loudly
   when a `STRUCT_*` override is evaluated over a window that includes
   pre-structure-first trades.
5. `researcher.py::ALLOWED_AREAS_HINT`: rewrite the apply-boundary text —
   the four real knobs in the current live mode are
   `STRUCT_MIN_CONFIDENCE`, `STRUCT_SYMBOL_MIN_CONFIDENCE`,
   `STRUCT_SYMBOL_DIRECTIONAL_CONF`, `SYMBOL_SIDE_BLOCKLIST`. State
   explicitly that the legacy `MIN_CONFIDENCE`-family keys are inert
   while `STRUCTURE_FIRST_MODE=True` and proposing them wastes the
   experiment slot. Update `_load_config_fingerprint()` in
   `orchestrator.py` to expose the `STRUCT_*` values.
6. **Entry-time dual-confidence logging:** in `_open_position()` /
   `_log_trade()`, record both `confidence` (structure, unchanged — keeps
   schema compatibility) and a new `model_confidence` field (best-effort:
   compute the PPO observation and prediction at entry; on failure store
   null — never block an entry on this). Additive DB change only.
7. **Tests** (`tests/test_self_improve/` + a new
   `tests/test_struct_conf_floor.py`):
   * floor enforced in `_get_structure_direction` at each precedence level;
   * 0.0/{} baselines provably no-op on a fixture of entries;
   * override file round-trip: write `STRUCT_*` override → loader applies →
     entry blocked (use `DRL_SKIP_RUNTIME_OVERRIDES` escape hatch pattern);
   * harness blocks/keeps the same trades the live check would (shared
     fixture asserting live-vs-harness agreement — this is the regression
     test for F1);
   * `check_tightening_only` accepts raises and refuses lowers for the
     new keys.

**Acceptance:** all tests green; deploying the branch with empty overrides
produces byte-identical entry decisions (verify by replaying a week of
logged signals if practical, otherwise by the no-op fixture tests); a
hand-written `STRUCT_SYMBOL_MIN_CONFIDENCE` override demonstrably blocks a
matching entry on the server (observe one 🚫 STRUCT_CONF log line in
dry-run or live).

---

### P2 — Forward-simulation harness (the deferred M3; prerequisite for everything edge-side)

**Goal:** answer "what would config X have done over window D" by
simulating entries AND exits bar-by-bar from cached klines — including
trades the live bot never took. This is what unlocks validating exit
tuning, guard-threshold changes, and (in P3) loosening.

**Branch:** `feature/profitability-p2`

**Module:** `src/self_improve/forward_sim.py` + CLI
`scripts/self_improve/run_forward_sim.py`.

**Data layer:** `data/kline_cache/` parquet, 5m/15m/1h/4h for the 4
symbols. Build `scripts/self_improve/refresh_kline_cache.py` (resumable,
uses the existing fetcher util the bots use; respect rate limits; cron it
daily). Also cache funding rates per symbol (8h grid) —
`data/kline_cache/funding_<symbol>.parquet`.

**Simulation spec — reuse live logic, do not reimplement heuristics:**

* Entry: instantiate `MarketStructure(swing_lookback=8)` from
  `src/signals/bos_choch.py` and walk the 5m candles with 1h/4h frames,
  reproducing `_get_structure_direction` (trend+last-signal alignment,
  S1/S5 per-symbol filters, blocklist, P1's structure-confidence floors).
  Factor the live decision code into pure functions where needed so the
  sim and the bot **call the same code** (move the S5 OB-proximity and
  ADX-directional checks into `src/signals/` pure helpers; the bot
  delegates to them — behavior-preserving refactor with tests).
* Guards replayable from klines: ADX guard, ranging, exhaustion, RSI
  guard, USDT.D proxy (computable from the 4-symbol basket),
  cooldown/min-hold/anti-whipsaw (stateful, simulate). Guards NOT
  replayable offline (orderbook, news, whale): configurable
  `assume_pass=True` default, and report sensitivity (n entries that
  would be subject to those guards).
* Exits: full fidelity — SL/TP with regime multipliers, partial TP1/TP2
  with SL-to-breakeven and lock, trailing (breakeven activation,
  distance, regime-tightened distance), stagnant exit (including its
  keep-checks degraded to "no model" mode — document), reverse-close.
  Intrabar ordering rule: on a bar where both SL and TP are inside the
  range, count SL first (conservative). Document this.
* Costs: taker fee 0.04% both sides + funding accrued per 8h boundary
  crossed (from the funding cache) + a slippage parameter (default 1 bp,
  configurable).
* Sizing: replicate fixed-dollar-risk sizing with a constant equity base
  (no compounding) so metrics are comparable across windows.
* Output: the same JSON shape as `backtest_harness` results (portfolio +
  per-symbol metrics, trade log, config echo, git head) with
  `"mode": "forward"`.
* Determinism: same inputs → identical output. No network in the sim path.

**Calibration gate (critical — the sim is useless if it doesn't match
reality):** run the sim with the live baseline config over the most recent
4 live weeks and compare to actual logged trades:
  * entry count per (symbol, side) within ±30%;
  * directional agreement on overlapping entry timestamps ≥ 80%;
  * net PnL same sign and within a documented band.
Commit the calibration report to `docs/forward_sim_calibration.md`. **The
orchestrator may not use forward-sim results as a promotion gate until
this calibration is committed and Chen has acknowledged it on Telegram.**
Re-run calibration weekly via cron; if it drifts out of band, the
orchestrator falls back to replay-only and pings Chen.

**Integration:** `orchestrator._advance_proposed` routes experiments whose
keys are replay-expressible to the replay harness (cheap) and everything
else to forward sim. `experiments.backtest_result_json` stores which
engine validated it.

**Tests:** golden-window tests (3 fixed historical windows with committed
expected metrics), unit tests per exit mechanism (synthetic candle
fixtures: SL hit, TP hit, partial→trail, stagnant, whipsaw chain),
funding accrual math, determinism test (two runs byte-identical).

**Acceptance:** calibration gate passed; golden tests green; a
demonstration sweep over one exit knob (e.g. `TRAILING_DISTANCE_PCT` ∈
{0.3%, 0.5%, 0.8%}) produces a ranked report — **not deployed**, just
proving the capability.

---

### P3 — Two-sided apply surface with bounded safety envelopes

**Goal:** let the loop move knobs in BOTH directions inside hard,
Chen-approved ranges — turning it from a loss-suppressor into an
optimizer — without giving it the keys it must never hold.

**Branch:** `feature/profitability-p3` (requires P2 calibration passed)

**Build** `src/self_improve/safety_envelopes.py`:

```python
ENVELOPES = {
    # key: (min, max, validation_engine)
    "STRUCT_MIN_CONFIDENCE":        (0.0,   0.95, "replay"),
    "STRUCT_SYMBOL_MIN_CONFIDENCE": (0.0,   0.95, "replay"),      # per-symbol values
    "STRUCT_SYMBOL_DIRECTIONAL_CONF": (0.0, 0.95, "replay"),
    "TRAILING_DISTANCE_PCT":        (0.003, 0.010, "forward"),
    "TRAILING_BREAKEVEN_PCT":       (0.005, 0.015, "forward"),
    "STAGNANT_HOURS":               (4.0,   12.0,  "forward"),
    "COOLDOWN_SECONDS":             (900,   7200,  "forward"),
    "MIN_HOLD_SECONDS":             (1800,  7200,  "forward"),
    "WHIPSAW_COOLDOWN_HOURS":       (1.0,   6.0,   "forward"),
    "ADX_GUARD_MIN":                (15.0,  30.0,  "forward"),
    "EXHAUSTION_ATR_THRESHOLD":     (2.0,   4.0,   "forward"),
}
```

(Ranges above are proposals — **Chen must approve the final table before
this phase merges**; ping him with the table on Telegram.)

**Rules:**

* Keys not in `ENVELOPES` (and not blocklist-add) → reject at apply time.
  Sizing/leverage/SL-TP/notional/halts are structurally absent, same as
  today.
* Envelope keys may move in either direction **within range**; the
  apply-time guard checks range, not direction. Floors-family keys keep
  their envelope but note: lowering a `STRUCT_*` floor below its committed
  baseline is now allowed *within envelope* because forward sim + paper +
  canary + circuit breaker validate it — this supersedes
  monotonic-tightening for envelope keys. `runtime_overrides` gains
  `check_envelope()`; `tighten_overrides` survives for blocklist semantics
  (add-only stays; **blocklist REMOVAL stays Chen-only** — the researcher
  may propose it, the orchestrator escalates it to Telegram for YES/NO).
* Bot startup loader (`_apply_runtime_overrides`) extended to set the new
  globals, with envelope re-validation at load time (a corrupt file
  outside range is ignored + alerts).
* Validation pipeline for envelope experiments: forward-sim backtest gate
  (beats baseline net PnL after costs AND max DD not worsened >20% — same
  shape as PLAN.md §6) → paper stage evaluated by **forward-sim over the
  forward window** (replay can't re-time exits) → canary/live unchanged
  (48h, circuit breaker). Max 1 envelope experiment in flight (unchanged
  concurrency cap).
* Researcher prompt: new allowed-areas text generated FROM
  `safety_envelopes.ENVELOPES` (single source of truth — never hand-list
  knobs in the prompt again; that's what caused the dead-ends fixed in
  `08512ea`).

**Tests:** envelope guard (in-range pass, out-of-range reject, unknown key
reject), loader round-trip per key, blocklist-removal escalation path,
prompt-generation snapshot test.

**Acceptance:** an end-to-end dry-run experiment moving
`TRAILING_DISTANCE_PCT` inside its envelope passes proposed→backtest→
paper→(held at canary if disarmed) with all decisions logged; Chen has
approved the envelope table.

---

### P4 — Canary attribution & honest evaluation

**Goal:** know what an applied change actually did, instead of reading
ambient PnL tea leaves.

**Branch:** `feature/profitability-p4` (independent of P2/P3; needs P1's
key alignment only)

**Changes:**

1. Additive schema: `ALTER TABLE trades ADD COLUMN experiment_id INTEGER`
   (nullable). The bot reads `experiment_id` from
   `active_overrides.json` at startup (already parsed in
   `_apply_runtime_overrides`) and stamps every trade row it logs while
   that override is active.
2. **Suppression event log:** new table `suppressed_entries
   (ts, symbol, side, confidence, gate, experiment_id)`. Every time an
   entry is blocked by a gate the loop controls (struct-conf floor,
   blocklist, and P3 envelope knobs where detectable), log a row. This
   is the counterfactual record replay mode can't reconstruct.
3. **Canary evaluation v2** (`live_apply.measure_since` + a new
   `evaluate_canary()`): keep the existing ambient circuit breaker
   untouched (it's a safety net, not a measurement), but base
   promotion on:
   * for suppression changes: PnL-avoided of suppressed entries,
     estimated by forward-simming the suppressed entries' hypothetical
     exits (P2) — promote if avoided PnL ≤ 0 (the blocked trades would
     have lost) with n ≥ 5, else extend canary up to 7d, else reject;
   * for envelope changes: realized PnL of stamped trades vs the
     forward-sim counterfactual of baseline config over the same window.
   If P2 is not yet merged, land items 1–2 anyway (they're pure logging)
   and keep promotion logic as today; wire evaluation v2 in a follow-up
   commit once P2 exists.
4. Reviewer nightly post-mortem gains a section: per active experiment,
   stamped-trade PnL vs baseline counterfactual.

**Tests:** stamping round-trip (override active → trade rows carry id),
suppression rows written on each gate, evaluation v2 decision matrix on
fixtures.

**Acceptance:** after one canary cycle on the server, the decision log
shows an evidence-based promote/reject with the counterfactual numbers in
`rationale`.

---

### P5 — Measure the sidelined alpha + funding-aware accounting

**Goal:** stop guessing which signals deserve to gate or drive entries;
make fees+funding first-class in every metric.

**Branch:** `feature/profitability-p5` (parallel with P2+; logging parts
can start immediately after P1)

**Changes:**

1. **Entry-time signal snapshot:** at every OPEN (and every *suppressed*
   entry from P4), capture a JSON snapshot into a new `entry_signals`
   table: PPO model action+confidence (already computed for
   `model_confidence` in P1), MTF bias/alignment, order-flow score,
   orderbook imbalance, regime+ADX, whale signal
   (`_get_whale_behavior_signal` — shadow logging exists, extend it),
   news sentiment, USDT.D proxy state, structure-signal fields
   (trend, last_signal_direction, confidence, fake_bos/fake_choch).
   Reuse `_build_signal_summary` / `_fetch_market_signals`; never let
   snapshot failure block an entry.
2. **Signal value report:** `scripts/self_improve/signal_value_report.py`
   — after ≥100 snapshotted closes, per-signal conditional expectancy
   (e.g. trades where model agreed with structure vs disagreed; whale
   aligned vs opposed; each guard's would-have-blocked set). Output
   markdown to `docs/ground_truth/`, Telegram digest of the top finding.
   This data — not opinion — decides whether the PPO model returns to the
   entry path (as a sizing/confidence input first, per the conviction
   question in §5) and whether any guard should be retired into the
   envelope set.
3. **Funding-aware PnL:** record funding payments against open positions
   (poll income history via the executor where testnet supports it;
   otherwise compute from cached funding rates × position notional at
   each 8h boundary, clearly labeled as estimated). Add
   `funding_paid` to trade close rows; include it in
   `src/self_improve/metrics.summarize` net-PnL, in the ground-truth
   report, and (already specced) in P2's sim costs.
4. **Reviewer upgrade:** nightly post-mortem consumes the snapshots —
   "of the last 24h losers, N had model disagreement at entry" style
   patterns, feeding T7 hypothesis seeds with actual columns to cite.

**Tests:** snapshot write path (entry + suppressed), report math on
fixtures, funding accrual unit tests (boundary crossing, sign by side).

**Acceptance:** snapshots flowing for all 4 bots; first signal-value
report generated after data accumulates; net-PnL metrics include funding
everywhere PnL is shown (UI column can lag — flag it if so).

---

## 4. Sequencing summary

```
P0 (report)  ──┐
P1 (bug fix) ──┼──► P2 (forward sim) ──► P3 (envelopes)
               ├──► P4 (attribution; eval-v2 waits for P2)
               └──► P5 (signal snapshots + funding; sim costs need P2 only for reuse)
```

Recommended order on the server: **P0 → P1 → P4(logging) + P5(logging) →
P2 → P4(eval v2) → P3 → P5(report)**. Logging earlier = more data
accumulated by the time the analysis steps need it.

---

## 5. Decisions reserved for Chen (escalate on Telegram, do not decide)

1. Final envelope table values (P3) — proposal is in §3/P3.
2. Any blocklist REMOVAL (e.g. re-enabling XRP after retrain) — always.
3. Re-introducing the PPO model into the live entry path (P5 data may
   recommend it; deployment is Chen's call, likely needs Mac retraining
   first — that's the M6 trainer-handoff from PLAN.md, still unbuilt).
4. Initial values for the `STRUCT_*` floors beyond 0.0 baselines — only
   with P0's confidence-decile evidence attached.
5. Any change to sizing/leverage/SL-TP/notional/halts — forever.
6. Arming/disarming autonomy and merging any phase branch.

---

## 6. Verification checklist per phase (gate before Telegram-reporting done)

- [ ] All new/changed code has tests; `python -m pytest tests/` green.
- [ ] No new top-level dependency without justification in the commit body.
- [ ] Behavior-preserving claims proven by a no-op test, not asserted.
- [ ] `decisions` rows written for anything the loop does differently.
- [ ] Kill switch + circuit breaker paths re-tested if touched (run
      `tests/test_self_improve/test_live_apply.py`).
- [ ] `./start_services.sh` after merge+deploy; confirm all 4 bots alive
      via `logs/running_services.json` and one clean iteration per bot in
      the logs.
- [ ] Telegram report to the primary chat with: phase, branch, merged
      commit, what changed in one paragraph, and the acceptance evidence.
