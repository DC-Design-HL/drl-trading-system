# PLAN.md — Autonomous Self-Improving Trading Service

**Status:** Draft, awaiting Chen's approval before any implementation.
**Authored:** 2026-05-22
**Scope:** Extend the existing DRL trading system into a self-directing
loop that proposes, validates, deploys, and (when needed) rolls back its
own changes — without Chen having to prompt it.

> Note: the prior `PLAN.md` (Mar 2026, HuggingFace → local migration —
> long since completed) was archived to
> `docs/PLAN-archive-2026-03-19-hf-migration.md`.

---

## 0. TL;DR

Build a cron-orchestrated multi-agent loop on top of the existing 7-service
DRL cluster. Performance metrics trigger a Researcher → Implementer →
Backtester → Risk Officer → Paper Trader → Live Canary pipeline. Every
behavioral change is logged with reasoning and the triggering metric.
Risk Officer enforces hard caps with no override. Chen is escalated to on
guardrail trips, capital/risk changes, and two consecutive underperforming
iterations.

I picked default values for every blank placeholder in the spec (max
position size, daily-loss halt, paper days, backtest tolerance, live
order cap) — see [§8 Guardrails](#8-guardrails). Two open questions are
flagged in [§12](#12-open-questions); everything else I'm going to call
unless you push back.

**Nothing gets built until you approve this plan.**

---

## 1. Existing Assets (Brief)

A full inventory was just compiled — captured here at the level the plan
references it. Anything not listed I'm treating as out-of-scope.

| Layer | Asset |
|---|---|
| Live processes | `live_trading_all.py` (consolidated bot, 4 symbol threads), `trade_alerter.py`, `start_local_server.py`, `news_sentinel.py`, `news_alerter.py`, `whale_behavior_ws.py`, `src/ui/app.py` |
| Supervision | `start_services.sh`, `watchdog.sh` (2-min cron), `@reboot` cron, in-process per-thread watchdog |
| Strategy | PPO-LSTM agents per symbol (`data/models/htf_walkforward_*/`), structure-first signals (BOS/CHOCH), MTF analyzer, regime detector, whale-flow predictor, news sentiment |
| Guards | ADX, RSI, orderbook imbalance, USDT.D proxy, EXT_POS_NEWS, REVERSE_CLOSE_LONG canary, symbol-side blocklist, anti-whipsaw, ranging-min-confidence, exhaustion |
| Risk | Per-trade $3k notional cap (will be lowered — see §8), 1.5% ATR-floored SL, 3% ATR-floored TP, partial-TP 40/35/25, trailing stop, 30-min cooldown, 1h min hold |
| Storage | `data/trading.db` (SQLite): `trades`, `state`, `news_events`. `is_testnet=1` filter enforced at storage layer. |
| Data | Binance Futures testnet (klines, orders), Alchemy WS (whales), RSS feeds (news), `data/kline_cache/` Parquet |
| Backtest | `src/backtest/engine.py` + 7 ad-hoc `backtest_*.py` scripts (no unified harness — Phase 2 builds it) |
| Existing agent role docs | `.agent/agents/{quantitative-researcher,risk-officer,professional-trader,data-scientist,developer-ml-engineer,mlops-engineer,qa-engineer,architect}.md` — I'll reuse these as agent personas rather than write fresh ones |
| Telegram | `luigiAlertBot` (8609279031) → chats `-5243679323` (primary trading) and `-5233405100` (extra) |
| Cron | `watchdog.sh` every 2m, `healthcheck.py` hourly, `@reboot start_services.sh`, May-1 whale training bundle prep |
| Hard rules | Train on Mac only; testnet only in DB; UI changes → `restart_ui.sh`; trading channel English-only |

**Scope of trading**: BTC, ETH, SOL, XRP — perpetual futures on Binance
testnet. **No mainnet, no new instruments without escalation.**

---

## 2. High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         Live Trading Plane (unchanged)                 │
│  live_trading_all.py  ──►  Binance Futures testnet  ◄── orders, fills │
│         │                                                              │
│         └──► writes ──► data/trading.db (trades) ──► logs/bots_live    │
└───────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ reads
                                  ▼
┌───────────────────────────────────────────────────────────────────────┐
│                   Self-Improvement Plane (new)                         │
│                                                                        │
│   ┌────────────────┐  hourly cron   ┌─────────────────────┐           │
│   │ Performance    │ ─────────────► │  Orchestrator       │           │
│   │ Monitor (5min) │   triggers     │  (Python script)    │           │
│   └────────────────┘                └──────────┬──────────┘           │
│                                                │ spawns                │
│                                                ▼                       │
│   ┌──────────────────────────────────────────────────────────┐        │
│   │  Agent pipeline (each = headless Claude Code run):       │        │
│   │                                                           │        │
│   │   Researcher ─► Risk Officer (veto?) ─► Implementer ─►   │        │
│   │   Backtester ─► Risk Officer ─► Paper Trader ─►          │        │
│   │   Reviewer (post-mortem, runs nightly independently)     │        │
│   └─────────────────────┬─────────────────────────────────────┘        │
│                         │ writes                                       │
│                         ▼                                              │
│   ┌──────────────────────────────────────────┐                        │
│   │  data/trading.db (new tables):           │                        │
│   │    decisions, experiments,               │                        │
│   │    metrics_snapshots, agent_runs         │                        │
│   └──────────┬───────────────────────────────┘                        │
│              │ exports                                                 │
│              ▼                                                         │
│   ┌──────────────────────────────────────────┐                        │
│   │  Observability:                          │                        │
│   │    logs/decisions/decisions.md (human)   │                        │
│   │    Streamlit "Self-Improvement" page     │                        │
│   │    Telegram digests (daily + on-event)   │                        │
│   └──────────────────────────────────────────┘                        │
└───────────────────────────────────────────────────────────────────────┘
```

**Key isolation property**: the live trading plane is untouched. The
self-improvement plane only reads from `trading.db` and writes to its
own new tables. The only mutation point against the live system is when
the orchestrator promotes a paper-tested change → that's a guarded,
PR-equivalent step described in §6.

---

## 3. Component Design

### 3.1 Orchestrator

- **What**: Single Python script `scripts/self_improve/orchestrator.py`.
- **Run mode**: Triggered by cron, not a long-running daemon. Each
  invocation reads state, decides what to do, optionally spawns agents,
  exits.
- **Cron schedule**:
  - `*/5 * * * *` — performance monitor (compute rolling metrics, write
    snapshot, check triggers)
  - `0 * * * *` — orchestrator main tick (if any trigger fires, spawn
    Researcher; otherwise no-op)
  - `0 3 * * *` — nightly Reviewer post-mortem (last 24h)
- **State**: persists everything in `data/trading.db`. No in-memory
  state across runs.

### 3.2 Agents

Each agent is a **headless Claude Code invocation** (`claude --print`
non-interactive). Why Claude Code over raw API: agents inherit the
repo's CLAUDE.md rules, skills, and tool definitions automatically. Each
agent gets a focused system prompt sourced from the existing
`.agent/agents/<role>.md` SOUL docs, plus a tight task-specific brief
that the orchestrator generates from current state.

Model choice per agent (opinionated):

| Agent | Model | Why |
|---|---|---|
| Researcher | `claude-opus-4-7` | Needs reasoning + signal synthesis across performance + market + news. Worth the cost (~1-2 invocations/day). |
| Implementer | `claude-opus-4-7` | Writes/modifies trading code — wrong edits = real losses. |
| Backtester | `claude-haiku-4-5` | Runs deterministic scripts and summarizes outputs. Fast, cheap. |
| Risk Officer | `claude-haiku-4-5` | Deterministic guardrail checks — quick, cheap, frequent invocation. |
| Reviewer | `claude-opus-4-7` | Pattern-spotting in post-mortems; 1 invocation/day. |

Each agent run is logged to the `agent_runs` table with: agent name,
model, input context size, tokens used, duration, outputs (decision +
artifacts).

**Agent role docs** (existing, will be augmented):
- `.agent/agents/quantitative-researcher.md` → Researcher
- `.agent/agents/developer-ml-engineer.md` → Implementer
- `.agent/agents/data-scientist.md` → Backtester
- `.agent/agents/risk-officer.md` → Risk Officer (extended with hard
  guardrails from §8)
- `.agent/agents/professional-trader.md` → Reviewer (post-mortem)

### 3.3 State store schema

New tables in `data/trading.db` (additive, no migration of existing
tables):

```sql
-- Every behavioral change the system makes to itself
CREATE TABLE decisions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            TEXT    NOT NULL,             -- ISO8601 UTC
  agent         TEXT    NOT NULL,             -- 'researcher' | 'implementer' | ...
  decision_type TEXT    NOT NULL,             -- 'config_change' | 'param_tune' | 'strategy_propose' | 'rollback' | 'halt'
  summary       TEXT    NOT NULL,             -- 1-line human-readable
  rationale     TEXT    NOT NULL,             -- agent's reasoning
  trigger_metric TEXT,                        -- which trigger fired (e.g. 'sharpe_7d<0.5')
  trigger_value REAL,
  expected_impact TEXT,                       -- what the agent expects to change
  diff_or_config_blob TEXT,                   -- the actual change (patch text or JSON config blob)
  experiment_id INTEGER,                      -- FK → experiments.id when this kicks off an experiment
  outcome       TEXT,                         -- 'pending' | 'approved' | 'rejected' | 'rolled_back' | 'kept'
  outcome_metric REAL,                        -- post-deploy metric (set later)
  notes         TEXT,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Proposed changes going through validation pipeline
CREATE TABLE experiments (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_created    TEXT    NOT NULL,
  proposal      TEXT    NOT NULL,             -- short description
  branch        TEXT,                         -- git branch with the change
  stage         TEXT    NOT NULL,             -- 'proposed' | 'backtest' | 'paper' | 'canary' | 'live' | 'rolled_back' | 'rejected'
  backtest_result_json TEXT,
  paper_result_json    TEXT,
  canary_result_json   TEXT,
  ts_promoted_paper    TEXT,
  ts_promoted_canary   TEXT,
  ts_promoted_live     TEXT,
  ts_rolled_back       TEXT,
  rollback_reason      TEXT
);

-- Periodic performance snapshots — what triggers fire from
CREATE TABLE metrics_snapshots (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            TEXT    NOT NULL,
  window        TEXT    NOT NULL,             -- '7d' | '30d' | '24h'
  symbol        TEXT,                         -- NULL = portfolio-wide
  net_pnl_usd   REAL,
  num_closes    INTEGER,
  win_rate      REAL,
  profit_factor REAL,
  sharpe        REAL,
  sortino       REAL,
  max_drawdown_pct REAL,
  metadata_json TEXT
);

-- Audit trail of every agent invocation
CREATE TABLE agent_runs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            TEXT    NOT NULL,
  agent         TEXT    NOT NULL,
  model         TEXT    NOT NULL,
  duration_s    REAL,
  input_tokens  INTEGER,
  output_tokens INTEGER,
  decision_id   INTEGER,                      -- FK → decisions.id if this run produced a decision
  context_summary TEXT,
  output_summary TEXT,
  error         TEXT
);

CREATE INDEX idx_decisions_ts ON decisions(ts);
CREATE INDEX idx_metrics_ts_window ON metrics_snapshots(ts, window);
CREATE INDEX idx_experiments_stage ON experiments(stage);
```

### 3.4 Backtest Harness

The current state is 7+ ad-hoc backtest scripts with inconsistent
assumptions. We need a **unified harness** before the loop is useful;
otherwise the Backtester agent can't validate anything reproducibly.

- **Module**: `src/self_improve/backtest_harness.py`
- **Inputs**: a candidate config blob OR a git ref + a date range
- **Outputs**: structured JSON (Sharpe, Sortino, max DD, PF, WR, net PnL,
  trade count, per-symbol breakdown) + a deterministic trade log
- **Data**: reuses `data/kline_cache/` (Parquet) — no Binance calls
- **Determinism**: fixed seeds, no randomization, walk-forward windows
  with documented splits
- **Wraps**: existing `src/backtest/engine.py` (already has the core
  loop); the harness adds config-injection + standardized output

### 3.5 Paper Trader

Reuses the existing `championship_shadow.py` substrate — it already
runs a shadow process alongside the live bot. The paper trader is a
configured shadow process running the candidate strategy against
real-time price data but with **simulated fills** (no orders sent).
Writes to a separate state file (`DRL_STATE_DIR=/tmp/paper_<expid>/`).

### 3.6 Decision Log

Two surfaces, same source of truth:

1. **SQLite `decisions` table** — machine-readable, joinable to
   experiments and agent_runs.
2. **`logs/decisions/decisions.md`** — append-only human-readable log.
   Each entry: timestamp, agent, summary, trigger, rationale, link to
   `experiments.id`, outcome. Generated from the `decisions` table by a
   small exporter (`scripts/self_improve/export_decisions.py`) run on
   cron after each decision.

This dual surface satisfies "writes a decision log every time it
changes its own behavior, with the reasoning and the metric that
triggered the change" — the Markdown is the human view; the table is
the structured source.

---

## 4. Data Flow

```
trades + market_data + news_events ──► PerformanceMonitor (every 5m)
                                              │
                                              ▼
                                       metrics_snapshots
                                              │
                                              ▼ (any trigger fires?)
                                     Orchestrator (hourly tick)
                                              │
              ┌───────────────────────────────┴───┐
              │                                    │
              ▼                                    ▼
        spawn Researcher                  (no trigger → no-op)
              │
              │ produces: hypothesis + proposed change (JSON)
              ▼
        Risk Officer (precheck) ───veto──► log + halt experiment
              │
              ▼ approved
        spawn Implementer
              │
              │ produces: git branch with patch
              ▼
        spawn Backtester (on the branch)
              │
              │ produces: backtest_result_json
              ▼
        Risk Officer (backtest review) ───fail──► log + close experiment
              │
              ▼ pass
        deploy to PaperTrader (7 days)
              │
              │ produces: paper_result_json
              ▼
        Risk Officer + Reviewer (paper review) ──fail──► log + close experiment
              │
              ▼ pass (within tolerance vs backtest)
        promote to Canary (small sizing, 3 days)
              │
              ▼ pass
        promote to Live (full sizing)
              │
              ▼
        watch — if outcome metric degrades → rollback + log
```

Every transition writes a `decisions` row. Every agent call writes an
`agent_runs` row. Every experiment has a row tracking its stage.

---

## 5. Agent Roles (Detail)

### 5.1 Researcher
- **Input**: latest `metrics_snapshots`, recent trades (last 100 closes),
  recent news/whale events, last 5 decisions, current config diff vs
  baseline.
- **Output (structured JSON)**: `{ hypothesis: str, proposed_change:
  {file: str, diff: str} | {config: {key: val}}, expected_impact: {metric,
  delta}, confidence: float, alternatives_considered: [...] }`
- **Forbidden topics**: capital allocation, new instruments/venues, risk
  logic — escalate to Chen if hypothesis touches these (per §9).

### 5.2 Implementer
- **Input**: Researcher's proposal + repo state.
- **Output**: a git branch (`auto/experiment-<id>`) with the change
  committed. Branch is **not merged** to dev.
- **Tests**: must add or update tests for the change (per Chen's
  "always write tests with bug fixes" rule). Backtester checks tests
  pass before running the strategy backtest.

### 5.3 Backtester
- **Input**: branch + date-range spec.
- **Action**: checkout branch (in a worktree, doesn't disturb the live
  checkout), run `backtest_harness.py`.
- **Output**: structured metrics JSON saved to
  `experiments.backtest_result_json`.

### 5.4 Risk Officer
- **Input**: any proposal/result + the hard guardrails (§8).
- **Action**: deterministic checks first (constants compared); LLM
  judgment second (e.g. "does this proposal indirectly increase
  exposure?").
- **Output**: `{verdict: 'approve'|'veto', reasons: [...]}`. Veto
  immediately halts the experiment.
- **No override**. If the Risk Officer vetoes, the orchestrator records
  a decision row with `outcome='rejected'` and the loop ends for that
  experiment.

### 5.5 Reviewer (Post-Mortem)
- **Cadence**: nightly cron at 03:00 UTC.
- **Input**: all closes from the last 24h (or last 7d on Sundays for a
  weekly retro), grouped by symbol and exit reason.
- **Output**: a markdown post-mortem appended to
  `logs/decisions/post-mortems/YYYY-MM-DD.md`, plus a Telegram digest if
  a recurring pattern is identified (e.g. "5 of 7 ETH SHORT losses had
  same regime+news pattern — propose investigation").
- **May escalate to Researcher** with a hypothesis seed.

---

## 6. Self-Improvement Loop — Triggers, Gates, Rollback

### Triggers (any of these → spawn Researcher; "OR" semantics)

| ID | Metric | Threshold | Window |
|---|---|---|---|
| T1 | Rolling Sharpe | < 0.5 | 7d |
| T2 | Net PnL | < −3% of capital | 7d |
| T3 | Per-symbol PF | < 0.7 (with n≥10 closes) | last 20 closes |
| T4 | Consecutive losing trades | ≥ 3 on same symbol/side | (any) |
| T5 | Drawdown from peak | > 5% | (rolling) |
| T6 | Quiet trigger | — | every 24h |
| T7 | Reviewer-surfaced pattern | (qualitative) | nightly |

Trigger evaluation happens in the PerformanceMonitor every 5 min; the
orchestrator reads the snapshot table on its hourly tick.

### Validation gates

| Gate | Pass criteria |
|---|---|
| Backtest | Sharpe ≥ current-baseline Sharpe **AND** max DD not worsened by more than 20% **AND** no per-symbol PF drops below 0.5 |
| Paper (7 days) | Sharpe within **±25%** of backtest Sharpe **AND** zero daily-loss-limit breaches **AND** ≥ 15 closes for statistical signal |
| Canary (3 days, **25% sizing**, single primary symbol) | Net PnL ≥ −1% of capital **AND** no SL cluster (≥3 in a row) |
| Live | (promoted) — monitored continuously; auto-rollback if DD from canary-start > 3% |

### Rollback

- **Automatic rollback triggers** (no agent involvement):
  - Live drawdown > 3% from canary-start within 48h of promotion → rollback
  - Daily-loss-limit breach during canary → rollback + halt
  - Two consecutive canary days with PnL < −1% capital → rollback
- **Mechanism**: `git revert <experiment_commit>` on dev,
  `./start_services.sh` to redeploy, Telegram alert, `decisions` row
  with `outcome='rolled_back'`.
- All rollbacks notify Chen on Telegram with the rollback reason and a
  link to the decision log entry.

---

## 7. Success Metrics

Optimize in this order (matches Chen's spec exactly):

1. **Net profit after fees, rolling 30 days** (primary)
2. **Max drawdown ≤ 8%** (hard ceiling — see §8)
3. **Sharpe ≥ 0.8** (informational target; below 0.5 triggers T1)
4. **Sortino ≥ 1.2** (informational target)
5. **Win rate** — tracked but never optimized for

Baseline is whatever the live config produced over the trailing 30 days
at the moment a new experiment proposes. The Researcher sees both the
baseline and the current snapshot.

---

## 8. Guardrails

Chen left every numeric placeholder blank. Here are my opinionated picks
with explicit justification. Risk Officer enforces all of them.

| Guardrail | Picked Value | Justification |
|---|---|---|
| **Max position size** | **20% of capital per trade** | Current `FIXED_MAX_NOTIONAL=$3000` on a $5K wallet = 60% per trade, which is too high to call "risk-managed". 20% is conservative (and 5× lower than the leverage permits). At $5K = $1000 notional. |
| **Max daily loss → halt** | **5% of capital** | Tight enough that two bad days don't compound into a 10%+ DD; loose enough to allow one rough Asian session. $250 on $5K. After breach: halt new entries, manage open positions, ping Chen. |
| **Max drawdown ceiling** | **8% from peak** | Hard ceiling: at 8% DD from peak, halt new entries and require Chen sign-off to resume. Spec only said "under X%" — I'm picking 8% as a number that protects the wallet while leaving enough headroom for normal variance. |
| **Paper trading days** | **7 days** | At current activity rate (~3-5 closes/day across 4 symbols) this gives ~25-35 closes — enough for a noisy-but-informative Sharpe read. <7d would let noise dominate; >14d would slow the loop too much. |
| **Backtest→paper tolerance** | **Sharpe within ±25%, max DD within ±30%** | Live noise vs backtest is inherently high; tighter would reject ideas due to variance. Loose enough to admit imperfect-but-real edges. |
| **Live order cap per order** | **$1000 notional** | = current max position size. Below this is the canary 25% sizing = $250 per canary order. Chen explicitly raises before going higher. |
| **Max concurrent experiments** | **1 in canary, 2 in paper** | Avoid resource contention and confounded results. |
| **Max changes per experiment** | **One config/code touch-point** | Forces clean attribution; "kitchen sink" PRs are not allowed by the Implementer. |
| **Symbols in scope** | BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT | Same as today. Adding a new symbol is an escalation. |
| **Venue** | Binance Futures testnet | Same as today. Switching to mainnet is an escalation (and currently forbidden by hard rule). |
| **Training site** | Chen's Mac M3 Pro | Hard rule. The orchestrator may **propose** a retraining job but never execute one server-side; it ships a training bundle and Telegram-pings Chen. |

### Risk Officer enforcement order

1. **Deterministic preflight** (no LLM): compare numeric constants in
   the proposed config against the table above. Anything out-of-bounds
   = automatic veto with reason text.
2. **LLM judgment pass**: only if preflight passes. Used to catch
   indirect violations ("this proposal doesn't change MAX_NOTIONAL but
   it disables the orderbook guard, effectively increasing
   slippage-driven exposure").
3. **Append-only logging**: every Risk Officer decision is logged with
   the inputs and the reasoning. Vetoes can't be silently retried.

---

## 9. Escalation to Chen

The orchestrator pings Telegram (chat `-5243679323`) and **halts the
relevant pipeline branch** when any of these fire:

| Condition | Action |
|---|---|
| Guardrail breached (daily loss, max DD, position size) | Halt new entries, manage open positions, ping Chen |
| Proposal touches risk logic / capital allocation / position sizing | Pause experiment, ping Chen with proposal + rationale |
| Proposal would add a new symbol or venue | Halt, ping Chen |
| Two consecutive strategy iterations underperform baseline | Halt the loop, ping Chen with both post-mortems |
| Backtest harness or any agent crashes 3 times in a row | Halt orchestrator, ping Chen |
| Watchdog detects orchestrator stalled > 6h | Page Chen |
| Researcher requests training data refresh | Ping Chen (training is on Mac, not server) |

Escalations include: a 1-line summary, the decision/experiment ID, and
a link/path to the full log entry. Chen can reply YES/NO/DEFER on
Telegram and the orchestrator (next tick) will resume or close.

---

## 10. Phased Rollout

Each phase is a discrete deliverable. **I'd ship M1 first and pause for
your eyeballs** before kicking off M2.

| Milestone | Deliverable | Estimated effort |
|---|---|---|
| **M1: Foundation** | New SQLite tables, PerformanceMonitor cron, decisions exporter (Markdown), Streamlit "Self-Improvement" page (read-only), baseline metrics measurement script. No agents yet. | 1-2 days |
| **M2: Backtest harness** | `src/self_improve/backtest_harness.py` wrapping the engine + standardized JSON output. Tests covering 3 known historical scenarios. Integration with kline cache. | 2-3 days |
| **M3: Risk Officer + Reviewer** | Risk Officer agent (deterministic + LLM) callable from orchestrator. Nightly Reviewer cron writing post-mortems. Plus tests for the Risk Officer's veto logic. | 2 days |
| **M4: Researcher + Implementer + first end-to-end loop (manual gate)** | Orchestrator can complete a full pipeline up to paper trader, but **Chen must approve canary promotion** — manual gate at the last step. | 3 days |
| **M5: Autonomous canary + auto-rollback** | Remove the manual canary gate. Full autonomy within guardrails. Telegram digest. | 2 days |
| **M6 (optional, post-launch)**: trainer-handoff workflow | When Researcher proposes a retrain, package a training bundle (similar to existing whale prep) and ping Chen on Telegram with download command. Chen runs training on Mac, pushes the artifact, orchestrator picks it up. | 1-2 days |

After M3 we have a functioning *advisory* system (Reviewer + Risk Officer
working, no changes deployed automatically). After M4 we have a guarded
loop with Chen as the deploy gate. After M5 we have the autonomous
system the spec asks for.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **Agent makes a destructive change** | All changes flow through git branches + Backtester + Risk Officer + paper before they touch live. Live is reached only after passing all gates. Rollback is automated. |
| **Backtest overfit → paper fails → loop churns** | "Paper passes within ±25% of backtest" is the gate. Two consecutive paper-vs-backtest mismatches → halt and Telegram (escalation per §9). |
| **Agent invocation budget blowout** | Per-day budgets: max 24 Researcher runs, max 12 Backtester runs, unlimited Risk Officer (cheap). Tracked in `agent_runs`. Orchestrator refuses to spawn beyond cap. |
| **Researcher proposes "disable a guard"** | Risk Officer's LLM pass is briefed to flag any guard-disabling as a risk-logic change → escalates to Chen. |
| **Orchestrator silently dies** | Each tick writes a heartbeat row in `metrics_snapshots` with a marker; watchdog cron checks heartbeat freshness; alert if > 6h stale. |
| **Concurrent self-improve + manual dev work** | Self-improve operates on `auto/experiment-*` branches. Merge to `dev` happens only after canary success and only via a tracked merge commit. Conflicts with manual dev work are surfaced and escalated. |
| **Training would help but can't run on server** | Researcher knows the rule; when it identifies a retrain need, it packages a bundle and escalates to Chen (M6). |
| **Confidence calibration drift in agents** | The Reviewer post-mortem includes a meta-question: "did past Researcher confidence correlate with actual outcomes?" → tracked over time. |

---

## 12. Open Questions

These are decisions I can't make alone — flagging them, **none block
M1**, but I need answers before M4.

1. **Canary promotion**: should it require a Chen-approved PR merge to
   `dev`, or auto-merge after passing paper? My recommendation: **manual
   PR approval through M5; auto-merge only after the first 3 successful
   end-to-end loops to build trust.** Acceptable?

2. **Agent cost cap**: I'll instrument token-cost tracking but I don't
   know your spend tolerance. Default proposal: **$15/day soft cap, $30
   hard cap → halt agent spawning and Telegram-ping**. Want to set
   different numbers?

3. **Capital wallet for canary**: do we run canary on the live testnet
   wallet alongside the production strategy (multiplexed), or on a
   separate testnet wallet? My recommendation: **separate testnet
   wallet** to avoid confounding. Need you to create one or confirm I
   should.

I'll proceed with my recommendations on (1) and (2) if you say "go"
without addressing them. (3) requires you to create the wallet or say
multiplex.

---

## 13. What I Will NOT Do Without Approval

- Touch `live_trading_all.py` or `live_trading_htf.py` directly (the
  Implementer will, via experiment branches, only after PR approval
  through M5).
- Modify `data/trading.db` schema beyond additive table creation.
- Open positions on mainnet, ever. (Hard rule.)
- Run model training on the server. (Hard rule.)
- Disable any existing guard without escalation.
- Change `MAX_NOTIONAL`, `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, or any
  risk constant without explicit Chen approval (Risk Officer will flag
  any proposal touching these).
- Auto-merge to `main` (only `dev`, per hard rule).
- Send Telegram messages to chats not in the allowlist.

---

## 14. Approval Checklist

Reply with one of:
- ✅ **Approve as written** — I start M1 immediately.
- ✏️ **Approve with changes** — call out the lines you want changed.
- ⛔ **Hold** — name the blockers; I'll redraft.

If you say "approve as written", here's exactly what happens next:

1. I create the new SQLite tables (`decisions`, `experiments`,
   `metrics_snapshots`, `agent_runs`) via a migration script.
2. I write `scripts/self_improve/performance_monitor.py` + add the cron
   line.
3. I write the decision-log exporter + the Streamlit "Self-Improvement"
   page (read-only).
4. I write the baseline metrics measurement script and run it once to
   capture the current 30-day baseline.
5. No agents are spawned yet. No code changes go through any automated
   pipeline. M1 is purely instrumentation.
6. I Telegram-confirm M1 complete with the baseline numbers and ask for
   M2 go-ahead.

Cumulative downtime risk of M1: zero (additive only, no live-process
changes).
