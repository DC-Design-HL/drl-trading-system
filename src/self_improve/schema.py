"""SQLite schema for the self-improvement plane.

Tables are additive to data/trading.db (the live trading DB). They
never reference the live `trades` table by FK because trades is
authoritative and we don't want any constraint to block the live bot.
The Performance Monitor reads from `trades` independently.

Schema is defined here as plain SQL strings rather than via ORM so it
matches PLAN.md §3.3 verbatim and so migrations can be reasoned about
without loading the rest of the codebase.
"""

from __future__ import annotations

DDL_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    agent         TEXT    NOT NULL,
    decision_type TEXT    NOT NULL,
    summary       TEXT    NOT NULL,
    rationale     TEXT    NOT NULL,
    trigger_metric TEXT,
    trigger_value REAL,
    expected_impact TEXT,
    diff_or_config_blob TEXT,
    experiment_id INTEGER,
    outcome       TEXT,
    outcome_metric REAL,
    notes         TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
)
"""

DDL_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS experiments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_created    TEXT    NOT NULL,
    proposal      TEXT    NOT NULL,
    branch        TEXT,
    stage         TEXT    NOT NULL,
    backtest_result_json TEXT,
    paper_result_json    TEXT,
    canary_result_json   TEXT,
    ts_promoted_paper    TEXT,
    ts_promoted_canary   TEXT,
    ts_promoted_live     TEXT,
    ts_rolled_back       TEXT,
    rollback_reason      TEXT
)
"""

DDL_METRICS_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS metrics_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    window        TEXT    NOT NULL,
    symbol        TEXT,
    net_pnl_usd   REAL,
    num_closes    INTEGER,
    win_rate      REAL,
    profit_factor REAL,
    sharpe        REAL,
    sortino       REAL,
    max_drawdown_pct REAL,
    metadata_json TEXT
)
"""

DDL_AGENT_RUNS = """
CREATE TABLE IF NOT EXISTS agent_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    agent         TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    duration_s    REAL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    decision_id   INTEGER,
    context_summary TEXT,
    output_summary TEXT,
    error         TEXT
)
"""

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_outcome ON decisions(outcome)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_ts_window ON metrics_snapshots(ts, window)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_symbol ON metrics_snapshots(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_stage ON experiments(stage)",
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_ts ON agent_runs(ts)",
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_agent ON agent_runs(agent)",
)

ALL_TABLES = ("decisions", "experiments", "metrics_snapshots", "agent_runs")

# Experiments stage state machine — used by Risk Officer to validate transitions.
EXPERIMENT_STAGES = (
    "proposed",
    "backtest",
    "paper",
    "canary",
    "live",
    "rolled_back",
    "rejected",
)

DECISION_OUTCOMES = (
    "pending",
    "approved",
    "rejected",
    "rolled_back",
    "kept",
)
