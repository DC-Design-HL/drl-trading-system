"""Read-only Streamlit view onto the self-improvement plane.

Rendered as a tab inside `src/ui/app.py`. Strictly read-only — this
page never mutates the live config, never triggers an agent, never
modifies any database row. Its job is observability:

  * latest metrics_snapshots (per window, per symbol)
  * decisions log (most recent N)
  * experiments pipeline (which experiments are in which stage)
  * current baseline (loaded from data/self_improve/baseline_metrics.json)

If the self_improve tables don't exist yet (M1 not yet migrated against
this DB) the page renders a help message instead of crashing.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _REPO_ROOT / "data" / "trading.db"
_BASELINE_PATH = _REPO_ROOT / "data" / "self_improve" / "baseline_metrics.json"
_CONFIG_PATH = _REPO_ROOT / "data" / "self_improve" / "baseline_config_fingerprint.json"

_REQUIRED_TABLES = ("decisions", "experiments", "metrics_snapshots", "agent_runs")


def _tables_present(conn: sqlite3.Connection) -> bool:
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    return all(t in existing for t in _REQUIRED_TABLES)


def _latest_snapshots(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One row per (window, symbol) — the most recent snapshot."""
    rows = conn.execute(
        """
        SELECT s.window, s.symbol, s.ts, s.net_pnl_usd, s.num_closes,
               s.win_rate, s.profit_factor, s.sharpe, s.sortino,
               s.max_drawdown_pct
        FROM metrics_snapshots s
        JOIN (
            SELECT window, COALESCE(symbol, '') AS sym, MAX(ts) AS max_ts
            FROM metrics_snapshots
            WHERE window != 'heartbeat'
            GROUP BY window, sym
        ) latest
        ON s.window = latest.window
           AND COALESCE(s.symbol, '') = latest.sym
           AND s.ts = latest.max_ts
        ORDER BY
            CASE s.window WHEN '24h' THEN 0 WHEN '7d' THEN 1
                         WHEN '30d' THEN 2 ELSE 3 END,
            COALESCE(s.symbol, '')
        """
    ).fetchall()
    cols = [
        "window", "symbol", "ts", "net_pnl_usd", "num_closes", "win_rate",
        "profit_factor", "sharpe", "sortino", "max_drawdown_pct",
    ]
    return [dict(zip(cols, row, strict=False)) for row in rows]


def _recent_decisions(conn: sqlite3.Connection, limit: int = 25):
    rows = conn.execute(
        """
        SELECT id, ts, agent, decision_type, summary, outcome,
               trigger_metric, trigger_value, experiment_id
        FROM decisions
        ORDER BY ts DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return rows


def _experiments_by_stage(conn: sqlite3.Connection):
    rows = conn.execute(
        """
        SELECT stage, COUNT(*) AS n
        FROM experiments
        GROUP BY stage
        ORDER BY n DESC
        """
    ).fetchall()
    return dict(rows)


def _heartbeat_age(conn: sqlite3.Connection) -> float | None:
    """Seconds since the most recent monitor heartbeat. None if never run."""
    row = conn.execute(
        "SELECT MAX(ts) FROM metrics_snapshots WHERE window = 'heartbeat'"
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        last = datetime.fromisoformat(row[0])
    except ValueError:
        return None
    return (datetime.now(last.tzinfo) - last).total_seconds()


def _format_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%"


def _format_money(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${v:+,.2f}"


def render() -> None:
    """Top-level render — called from app.py inside `with tab_self_improve`."""
    st.subheader("🤖 Self-Improvement Plane")
    st.caption(
        "Read-only view onto the self-improving loop's state. "
        "Plan reference: `PLAN.md` at repo root."
    )

    if not _DB_PATH.exists():
        st.error(f"Trading DB not found at `{_DB_PATH}`.")
        return

    with sqlite3.connect(str(_DB_PATH)) as conn:
        if not _tables_present(conn):
            st.warning(
                "Self-improvement tables not yet present in this DB. "
                "Run `python -m scripts.self_improve.migrate` to create "
                "them. (M1 setup step.)"
            )
            return

        heartbeat = _heartbeat_age(conn)
        snapshots = _latest_snapshots(conn)
        decisions = _recent_decisions(conn)
        experiments = _experiments_by_stage(conn)

    # --- Status row ----------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        if heartbeat is None:
            st.metric("Monitor heartbeat", "never run")
        elif heartbeat < 600:
            st.metric("Monitor heartbeat", f"{heartbeat:.0f}s ago", delta="OK ✅")
        elif heartbeat < 1800:
            st.metric("Monitor heartbeat", f"{heartbeat / 60:.0f}m ago", delta="degraded ⚠")
        else:
            st.metric("Monitor heartbeat", f"{heartbeat / 60:.0f}m ago", delta="STALLED ❌")
    with col2:
        st.metric("Decisions logged", len(decisions))
    with col3:
        st.metric(
            "Experiments running",
            sum(v for k, v in experiments.items() if k in ("backtest", "paper", "canary")),
        )

    st.divider()

    # --- Latest metrics ------------------------------------------------
    st.markdown("### Latest metrics snapshots")
    if not snapshots:
        st.info(
            "No snapshots yet. The performance monitor will create them "
            "once the cron entry is enabled (or run "
            "`python -m scripts.self_improve.performance_monitor` "
            "manually)."
        )
    else:
        portfolio_rows = [s for s in snapshots if not s["symbol"]]
        per_symbol_rows = [s for s in snapshots if s["symbol"]]

        if portfolio_rows:
            st.markdown("**Portfolio-wide**")
            st.dataframe(
                [
                    {
                        "window": r["window"],
                        "ts": r["ts"],
                        "closes": r["num_closes"],
                        "net pnl": _format_money(r["net_pnl_usd"]),
                        "WR": _format_pct(r["win_rate"]),
                        "PF": f"{r['profit_factor']:.2f}",
                        "Sharpe": f"{r['sharpe']:.2f}",
                        "Sortino": f"{r['sortino']:.2f}",
                        "max DD %": f"{r['max_drawdown_pct']:.2f}",
                    }
                    for r in portfolio_rows
                ],
                hide_index=True,
                use_container_width=True,
            )

        if per_symbol_rows:
            st.markdown("**Per-symbol (30d)**")
            st.dataframe(
                [
                    {
                        "symbol": r["symbol"],
                        "closes": r["num_closes"],
                        "net pnl": _format_money(r["net_pnl_usd"]),
                        "WR": _format_pct(r["win_rate"]),
                        "PF": f"{r['profit_factor']:.2f}",
                        "Sharpe": f"{r['sharpe']:.2f}",
                        "max DD %": f"{r['max_drawdown_pct']:.2f}",
                    }
                    for r in per_symbol_rows
                ],
                hide_index=True,
                use_container_width=True,
            )

    st.divider()

    # --- Baseline ------------------------------------------------------
    st.markdown("### Baseline (comparison point for proposals)")
    if _BASELINE_PATH.exists():
        baseline = json.loads(_BASELINE_PATH.read_text())
        st.caption(
            f"Captured: `{baseline.get('captured_at', '?')}` · "
            f"git HEAD: `{baseline.get('git_head', '?')[:10]}` · "
            f"window since: `{baseline.get('window_since', '?')}`"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Portfolio 30d**")
            st.json(baseline.get("portfolio_30d", {}))
        with col_b:
            st.markdown("**Portfolio 7d**")
            st.json(baseline.get("portfolio_7d", {}))
    else:
        st.info(
            "No baseline yet. Run `python -m scripts.self_improve.measure_baseline` "
            "to capture one."
        )

    st.divider()

    # --- Decisions -----------------------------------------------------
    st.markdown("### Decisions log (most recent 25)")
    if not decisions:
        st.info(
            "No decisions logged yet — expected during M1 (instrumentation "
            "only, no agents spawned)."
        )
    else:
        st.dataframe(
            [
                {
                    "id": d[0],
                    "ts": d[1],
                    "agent": d[2],
                    "type": d[3],
                    "summary": d[4],
                    "outcome": d[5],
                    "trigger": d[6],
                    "exp_id": d[8],
                }
                for d in decisions
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    # --- Experiments pipeline ------------------------------------------
    st.markdown("### Experiments pipeline")
    if not experiments:
        st.info("No experiments recorded yet.")
    else:
        st.bar_chart(experiments)
