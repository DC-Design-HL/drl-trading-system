"""Orchestrator — drives the self-improvement loop end-to-end.

State machine (see PLAN.md §6):

  proposed → backtest → paper → awaiting_canary_approval → canary → live
                                                                   |
              rejected ←──────────────── (anywhere, if Risk Officer veto,
                                          backtest fail, paper fail, or
                                          test fail)
              rolled_back ←──── (post-live, if metrics degrade)

In M4 the loop stops at `awaiting_canary_approval` — that's the manual
canary gate. The orchestrator posts a Telegram message with the full
proposal + backtest + paper results and **does nothing else** until
Chen approves out-of-band (e.g. by replying on Telegram, which the
human operator translates into a SQL update on the experiments row).

The orchestrator is cron-triggered. Each tick:

  1. Load any in-progress experiments (stage in {'proposed', 'backtest',
     'paper', 'awaiting_canary_approval'}) — advance them.
  2. If no in-progress experiments AND recent triggers fired AND today's
     LLM budget hasn't been blown AND we haven't already proposed today,
     spawn a new Researcher run.

Pipeline functions are idempotent — they update one experiment per
tick. If a tick fails partway, the next tick picks up where it left off.

This module is intentionally LIGHT — the heavy lifting lives in the
agent modules. The orchestrator is the conductor, not a player.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .backtest_harness import BacktestRequest, run_backtest
from .llm_client import LLMClient, default_client
from .paper_trader import evaluate_paper_period
from .researcher import ResearcherContext, propose
from .risk_officer import Proposal, evaluate as risk_evaluate

UTC = timezone.utc

# How long an experiment stays in the 'paper' stage before evaluation
PAPER_DAYS = 7

# Look back this far when assembling Researcher context
RECENT_CLOSES_WINDOW_DAYS = 30
RECENT_CLOSES_LIMIT = 100

# Telegram channel for escalations and canary-gate prompts
ESCALATION_CHAT_ID = "-5243679323"

# Capital base for backtests (default May-1 reset baseline)
CAPITAL_BASE = 5000.0


# ─────────────────────────────────────────────────────────────────────────
# Telegram helper — best-effort, never crashes the orchestrator
# ─────────────────────────────────────────────────────────────────────────


def _telegram_post(text: str) -> None:
    import os
    import urllib.parse
    import urllib.request
    token = os.environ.get("TELEGRAM_ALERT_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", ESCALATION_CHAT_ID)
    if not token:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        urllib.request.urlopen(url, data=data, timeout=10).read()
    except Exception:
        pass  # never crash the loop on a Telegram outage


# ─────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _log_decision(
    conn: sqlite3.Connection,
    *,
    agent: str,
    decision_type: str,
    summary: str,
    rationale: str,
    trigger_metric: str | None = None,
    trigger_value: float | None = None,
    expected_impact: str = "",
    diff_or_config_blob: str = "",
    experiment_id: int | None = None,
    outcome: str = "pending",
    notes: str = "",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO decisions(ts, agent, decision_type, summary, rationale,
                              trigger_metric, trigger_value, expected_impact,
                              diff_or_config_blob, experiment_id, outcome, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now_iso(),
            agent,
            decision_type,
            summary,
            rationale,
            trigger_metric,
            trigger_value,
            expected_impact,
            diff_or_config_blob,
            experiment_id,
            outcome,
            notes,
        ),
    )
    return int(cur.lastrowid or 0)


def _set_experiment_stage(
    conn: sqlite3.Connection,
    experiment_id: int,
    stage: str,
    *,
    rollback_reason: str | None = None,
    backtest_result_json: str | None = None,
    paper_result_json: str | None = None,
) -> None:
    updates = ["stage = ?"]
    params: list[Any] = [stage]
    now = _now_iso()
    if stage == "paper":
        updates.append("ts_promoted_paper = ?")
        params.append(now)
    elif stage == "canary":
        updates.append("ts_promoted_canary = ?")
        params.append(now)
    elif stage == "live":
        updates.append("ts_promoted_live = ?")
        params.append(now)
    elif stage in ("rolled_back", "rejected"):
        updates.append("ts_rolled_back = ?")
        params.append(now)
    if rollback_reason is not None:
        updates.append("rollback_reason = ?")
        params.append(rollback_reason)
    if backtest_result_json is not None:
        updates.append("backtest_result_json = ?")
        params.append(backtest_result_json)
    if paper_result_json is not None:
        updates.append("paper_result_json = ?")
        params.append(paper_result_json)
    params.append(experiment_id)
    conn.execute(
        f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?",
        params,
    )


# ─────────────────────────────────────────────────────────────────────────
# Context assembly for the Researcher
# ─────────────────────────────────────────────────────────────────────────


def _load_recent_triggers(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Read the most recent monitor tick's fired triggers. Triggers are
    stored in metrics_snapshots.metadata_json by performance_monitor —
    but since we currently log the triggers to stdout rather than to a
    table, M4 reads the live counter instead by re-running the trigger
    evaluator on the latest closes."""
    from .triggers import evaluate as eval_triggers
    from .metrics import TradeClose, parse_ts
    rows = conn.execute(
        """
        SELECT timestamp, symbol, action, pnl
        FROM trades
        WHERE is_testnet=1
          AND timestamp >= ?
          AND action IN ('CLOSE_LONG','CLOSE_SHORT','REVERSE_CLOSE_LONG',
                         'REVERSE_CLOSE_SHORT','SL_HIT','TP_HIT')
        ORDER BY timestamp
        """,
        ((datetime.now(UTC) - timedelta(days=30)).isoformat(),),
    ).fetchall()
    closes = []
    for ts_str, symbol, action, pnl in rows:
        if pnl is None:
            continue
        side = "LONG" if "LONG" in action else "SHORT"
        closes.append(TradeClose(parse_ts(ts_str), symbol, side, float(pnl)))
    hits = eval_triggers(closes, capital_base=CAPITAL_BASE)
    return [
        {
            "id": h.id,
            "metric": h.metric,
            "value": h.value,
            "threshold": h.threshold,
            "window": h.window,
            "symbol": h.symbol,
            "rationale": h.rationale,
        }
        for h in hits
    ]


def _load_latest_snapshot(
    conn: sqlite3.Connection, window: str = "30d", symbol: str | None = None
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT net_pnl_usd, num_closes, win_rate, profit_factor,
               sharpe, sortino, max_drawdown_pct, ts
        FROM metrics_snapshots
        WHERE window = ? AND COALESCE(symbol, '') = COALESCE(?, '')
        ORDER BY ts DESC LIMIT 1
        """,
        (window, symbol),
    ).fetchone()
    if not row:
        return {}
    return {
        "net_pnl_usd": row[0],
        "num_closes": row[1],
        "win_rate": row[2],
        "profit_factor": row[3],
        "sharpe": row[4],
        "sortino": row[5],
        "max_drawdown_pct": row[6],
        "captured_at": row[7],
    }


def _load_per_symbol_30d(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT symbol, net_pnl_usd, num_closes, win_rate, profit_factor,
               sharpe, sortino, max_drawdown_pct
        FROM metrics_snapshots
        WHERE window = '30d' AND symbol IS NOT NULL
          AND ts = (
            SELECT MAX(ts) FROM metrics_snapshots
            WHERE window='30d' AND symbol=metrics_snapshots.symbol
          )
        """
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r[0]] = {
            "net_pnl_usd": r[1],
            "num_closes": r[2],
            "win_rate": r[3],
            "profit_factor": r[4],
            "sharpe": r[5],
            "sortino": r[6],
            "max_drawdown_pct": r[7],
        }
    return out


def _load_recent_closes(
    conn: sqlite3.Connection, limit: int = RECENT_CLOSES_LIMIT
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT timestamp, symbol, action, pnl, confidence, reason
        FROM trades
        WHERE is_testnet=1
          AND action IN ('CLOSE_LONG','CLOSE_SHORT','REVERSE_CLOSE_LONG',
                         'REVERSE_CLOSE_SHORT','SL_HIT','TP_HIT')
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    out = []
    for ts, symbol, action, pnl, conf, reason in rows:
        side = "LONG" if "LONG" in action else "SHORT"
        out.append({
            "ts": ts, "symbol": symbol, "side": side,
            "pnl": pnl, "confidence": conf, "reason": reason,
        })
    return list(reversed(out))


def _load_recent_decisions(conn: sqlite3.Connection, limit: int = 5):
    rows = conn.execute(
        """
        SELECT ts, agent, decision_type, summary, outcome, trigger_metric
        FROM decisions
        ORDER BY ts DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "ts": r[0], "agent": r[1], "decision_type": r[2],
            "summary": r[3], "outcome": r[4], "trigger_metric": r[5],
        }
        for r in rows
    ]


def _load_config_fingerprint() -> dict[str, Any]:
    """Reload the live config fingerprint."""
    try:
        from scripts.self_improve.measure_baseline import fingerprint_config
        return fingerprint_config()
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────
# Pipeline state-machine — one experiment, one stage transition per tick
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class TickResult:
    actions_taken: list[str]
    n_experiments_advanced: int = 0
    n_experiments_proposed: int = 0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "actions_taken": self.actions_taken,
            "n_experiments_advanced": self.n_experiments_advanced,
            "n_experiments_proposed": self.n_experiments_proposed,
            "error": self.error,
        }


def _check_existing_pipeline(conn: sqlite3.Connection) -> list[tuple]:
    """Return experiments still in the pipeline (not terminal)."""
    return conn.execute(
        """
        SELECT id, stage, proposal, branch, ts_promoted_paper,
               backtest_result_json
        FROM experiments
        WHERE stage IN ('proposed', 'backtest', 'paper',
                        'awaiting_canary_approval')
        ORDER BY id
        """
    ).fetchall()


def _advance_proposed(
    conn: sqlite3.Connection,
    exp_row: tuple,
    *,
    client: LLMClient,
) -> str:
    """proposed → backtest. Run the harness on the proposal, advance or
    reject based on backtest gate."""
    exp_id, stage, proposal_text, branch, _, _ = exp_row
    # Read the original proposal payload from decisions table
    payload = conn.execute(
        "SELECT diff_or_config_blob FROM decisions WHERE experiment_id=? "
        "AND decision_type='strategy_propose' ORDER BY id LIMIT 1",
        (exp_id,),
    ).fetchone()
    if not payload or not payload[0]:
        _set_experiment_stage(
            conn, exp_id, "rejected",
            rollback_reason="no proposal payload found in decisions",
        )
        return f"exp {exp_id}: rejected — no proposal payload"
    try:
        proposal_blob = json.loads(payload[0])
    except json.JSONDecodeError:
        _set_experiment_stage(
            conn, exp_id, "rejected",
            rollback_reason="proposal payload is not valid JSON",
        )
        return f"exp {exp_id}: rejected — bad payload"

    config_overrides = proposal_blob.get("config_changes", {})
    # Run backtest over the last 30 days with this proposed override
    end = datetime.now(UTC)
    start = end - timedelta(days=30)
    req = BacktestRequest(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        config_overrides=config_overrides,
        label=f"experiment-{exp_id}-backtest",
        capital_base=CAPITAL_BASE,
    )
    result = run_backtest(req)
    # Risk Officer review after backtest
    proposal = Proposal(
        description=proposal_blob.get("description", ""),
        config_changes=config_overrides,
        category=proposal_blob.get("category", "config_tune"),
        rationale=proposal_blob.get("rationale", ""),
        expected_impact=proposal_blob.get("expected_impact", ""),
    )
    verdict = risk_evaluate(proposal, client=client, db_path=str(_DB))
    if verdict.is_veto:
        _set_experiment_stage(
            conn, exp_id, "rejected",
            rollback_reason="risk officer veto: " + "; ".join(verdict.reasons),
            backtest_result_json=json.dumps(result.to_json()),
        )
        _log_decision(
            conn, agent="risk-officer",
            decision_type="veto",
            summary=f"Vetoed experiment #{exp_id}",
            rationale="; ".join(verdict.reasons),
            experiment_id=exp_id, outcome="rejected",
        )
        return f"exp {exp_id}: rejected — RO veto"

    # Gate: backtest must be Sharpe ≥ baseline AND DD not worsened > 20%
    baseline_req = BacktestRequest(
        start_date=start.isoformat(), end_date=end.isoformat(),
        config_overrides={}, label="baseline", capital_base=CAPITAL_BASE,
    )
    baseline = run_backtest(baseline_req)
    candidate_sharpe = result.portfolio_metrics["sharpe"]
    baseline_sharpe = baseline.portfolio_metrics["sharpe"]
    candidate_dd = result.portfolio_metrics["max_drawdown_pct"]
    baseline_dd = baseline.portfolio_metrics["max_drawdown_pct"]
    gate_reasons: list[str] = []
    if candidate_sharpe < baseline_sharpe - 0.01:
        gate_reasons.append(
            f"backtest Sharpe {candidate_sharpe:.2f} < baseline "
            f"{baseline_sharpe:.2f}"
        )
    # DD must not be worse than +20% relative to baseline
    if baseline_dd > 0 and candidate_dd > baseline_dd * 1.2:
        gate_reasons.append(
            f"max DD {candidate_dd:.2f}% > 1.2× baseline {baseline_dd:.2f}%"
        )
    if gate_reasons:
        _set_experiment_stage(
            conn, exp_id, "rejected",
            rollback_reason="backtest gate failed: " + "; ".join(gate_reasons),
            backtest_result_json=json.dumps(result.to_json()),
        )
        return f"exp {exp_id}: rejected — backtest gate"

    # Promote to paper
    _set_experiment_stage(
        conn, exp_id, "paper",
        backtest_result_json=json.dumps(result.to_json()),
    )
    _log_decision(
        conn, agent="orchestrator", decision_type="promote",
        summary=f"Experiment #{exp_id} → paper",
        rationale=f"backtest passed: Sharpe {candidate_sharpe:.2f} vs baseline {baseline_sharpe:.2f}",
        experiment_id=exp_id, outcome="approved",
    )
    return f"exp {exp_id}: promoted → paper"


def _advance_paper(
    conn: sqlite3.Connection, exp_row: tuple
) -> str:
    """paper → awaiting_canary_approval. Run paper-trader evaluation."""
    exp_id, stage, _, _, ts_promoted_paper, backtest_json = exp_row
    if not ts_promoted_paper:
        return f"exp {exp_id}: no ts_promoted_paper — skipping"
    promoted = datetime.fromisoformat(ts_promoted_paper)
    now = datetime.now(UTC)
    if (now - promoted).total_seconds() < PAPER_DAYS * 86400:
        return f"exp {exp_id}: still in paper window ({(now - promoted).days}d/{PAPER_DAYS}d)"

    # Load the proposal payload
    payload = conn.execute(
        "SELECT diff_or_config_blob FROM decisions WHERE experiment_id=? "
        "AND decision_type='strategy_propose' ORDER BY id LIMIT 1",
        (exp_id,),
    ).fetchone()
    proposal_blob = json.loads(payload[0]) if payload and payload[0] else {}
    config_overrides = proposal_blob.get("config_changes", {})

    backtest_sharpe = 0.0
    if backtest_json:
        try:
            backtest_sharpe = json.loads(backtest_json)["portfolio_metrics"]["sharpe"]
        except (json.JSONDecodeError, KeyError, TypeError):
            backtest_sharpe = 0.0

    result = evaluate_paper_period(
        paper_start=promoted,
        paper_end=now,
        config_overrides=config_overrides,
        backtest_sharpe_reference=backtest_sharpe,
        capital_base=CAPITAL_BASE,
    )
    next_stage = "awaiting_canary_approval" if result.pass_gate else "rejected"
    _set_experiment_stage(
        conn, exp_id, next_stage,
        paper_result_json=json.dumps(result.to_json()),
        rollback_reason=("; ".join(result.reasons) if not result.pass_gate else None),
    )
    _log_decision(
        conn,
        agent="orchestrator",
        decision_type="paper_evaluation",
        summary=f"Experiment #{exp_id} paper {'PASSED' if result.pass_gate else 'FAILED'}",
        rationale=f"delta_pnl=${result.delta_pnl:+.2f}, "
                  f"candidate_sharpe={result.candidate_sharpe:.2f}, "
                  f"n_closes={result.n_closes_kept}, "
                  f"reasons={'; '.join(result.reasons) or 'none'}",
        experiment_id=exp_id,
        outcome="approved" if result.pass_gate else "rejected",
    )

    if result.pass_gate:
        _telegram_post(
            f"🤖 Experiment #{exp_id} PASSED paper gate. "
            f"delta_pnl ${result.delta_pnl:+.2f} over {PAPER_DAYS}d, "
            f"Sharpe {result.candidate_sharpe:.2f}. "
            f"Awaiting your canary approval — proposal in "
            f"`experiments` row #{exp_id} in trading.db, branch: "
            f"`auto/experiment-{exp_id}`."
        )
        return f"exp {exp_id}: paper passed → awaiting canary approval"
    return f"exp {exp_id}: paper failed → rejected"


# ─────────────────────────────────────────────────────────────────────────
# Researcher trigger gate
# ─────────────────────────────────────────────────────────────────────────


def _should_propose(
    conn: sqlite3.Connection, *, triggers: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Return (should, reason). Decide whether to spawn the Researcher."""
    # 1. If any pipeline is already running, defer
    in_flight = _check_existing_pipeline(conn)
    if in_flight:
        return (False, f"{len(in_flight)} experiment(s) in flight; defer")

    # 2. If no triggers fired AND no 24h-tick rule, skip
    if not triggers:
        # 24h tick rule: if we haven't run the Researcher in 24h, run it
        # anyway (T6 in PLAN.md §6)
        row = conn.execute(
            "SELECT MAX(ts) FROM decisions WHERE agent='researcher'"
        ).fetchone()
        last = row[0] if row else None
        if last:
            last_dt = datetime.fromisoformat(last)
            if (datetime.now(UTC) - last_dt).total_seconds() < 86400:
                return (False, "no triggers + last researcher < 24h ago")
        return (True, "T6 quiet tick: no triggers but >24h since last run")

    # 3. Rate limit: max 1 proposal per 6h to avoid spam
    row = conn.execute(
        "SELECT MAX(ts) FROM decisions WHERE agent='researcher' "
        "AND decision_type='strategy_propose'"
    ).fetchone()
    last = row[0] if row else None
    if last:
        last_dt = datetime.fromisoformat(last)
        if (datetime.now(UTC) - last_dt).total_seconds() < 6 * 3600:
            return (False, "last proposal < 6h ago; rate-limit")
    return (True, f"triggers fired: {[t['id'] for t in triggers]}")


# ─────────────────────────────────────────────────────────────────────────
# Tick entry point
# ─────────────────────────────────────────────────────────────────────────


_DB = Path("data/trading.db")


def run_tick(
    *,
    db_path: str | Path = _DB,
    client: LLMClient | None = None,
) -> TickResult:
    actions: list[str] = []
    advanced = 0
    proposed = 0
    db = Path(db_path)
    if not db.exists():
        return TickResult(actions, error=f"db not found at {db}")

    cli = client or default_client(db_path)
    with sqlite3.connect(str(db)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # 1. Advance any in-flight experiments
        in_flight = _check_existing_pipeline(conn)
        for row in in_flight:
            exp_id, stage = row[0], row[1]
            try:
                if stage == "proposed":
                    msg = _advance_proposed(conn, row, client=cli)
                elif stage == "paper":
                    msg = _advance_paper(conn, row)
                elif stage == "awaiting_canary_approval":
                    msg = f"exp {exp_id}: held at canary gate (manual)"
                else:
                    msg = f"exp {exp_id}: stage={stage} (no auto-advance)"
            except Exception as exc:  # noqa: BLE001
                msg = f"exp {exp_id}: ERROR in advance({stage}): {exc}"
                _set_experiment_stage(
                    conn, exp_id, "rejected",
                    rollback_reason=f"orchestrator exception: {exc}",
                )
            actions.append(msg)
            advanced += 1

        # 2. Decide whether to propose anything new
        triggers = _load_recent_triggers(conn)
        should, reason = _should_propose(conn, triggers=triggers)
        actions.append(f"researcher gate: {reason}")
        if not should:
            return TickResult(actions, n_experiments_advanced=advanced)

        # 3. Assemble Researcher context
        ctx = ResearcherContext(
            triggers_fired=triggers,
            portfolio_metrics=_load_latest_snapshot(conn, "30d"),
            per_symbol_metrics=_load_per_symbol_30d(conn),
            recent_closes=_load_recent_closes(conn),
            recent_decisions=_load_recent_decisions(conn),
            config_fingerprint=_load_config_fingerprint(),
        )

        # 4. Spawn Researcher
        out = propose(ctx, client=cli, db_path=str(db))
        if out.verdict == "no_change":
            _log_decision(
                conn, agent="researcher", decision_type="no_change",
                summary=out.hypothesis or "(no hypothesis)",
                rationale=out.error or "researcher returned no_change",
                outcome="kept",
            )
            actions.append(f"researcher: no_change ({out.error or 'opinionated abstain'})")
            return TickResult(actions, n_experiments_advanced=advanced)

        if out.verdict == "escalate":
            _log_decision(
                conn, agent="researcher", decision_type="escalate",
                summary=out.hypothesis,
                rationale=out.escalation_reason,
                outcome="pending",
            )
            _telegram_post(
                f"🚨 Researcher ESCALATION:\n{out.hypothesis}\n\n"
                f"Reason: {out.escalation_reason}\n\n"
                f"Awaiting Chen's review."
            )
            actions.append(f"researcher: escalate ({out.escalation_reason[:80]})")
            return TickResult(actions, n_experiments_advanced=advanced)

        # 5. Propose verdict — Risk Officer Phase 1 preflight
        proposal = out.proposal
        if proposal is None:
            actions.append("researcher: propose verdict but no proposal object")
            return TickResult(actions, n_experiments_advanced=advanced)

        verdict = risk_evaluate(proposal, client=cli, db_path=str(db), skip_llm=False)
        if verdict.is_veto:
            _log_decision(
                conn, agent="risk-officer", decision_type="veto",
                summary=f"Pre-implementation veto: {proposal.description}",
                rationale="; ".join(verdict.reasons),
                outcome="rejected",
                diff_or_config_blob=json.dumps(_proposal_to_blob(proposal)),
            )
            actions.append(f"risk-officer: VETO ({verdict.reasons[0] if verdict.reasons else '?'})")
            return TickResult(actions, n_experiments_advanced=advanced)

        # 6. Create the experiment row + log the strategy_propose decision
        cur = conn.execute(
            """
            INSERT INTO experiments(ts_created, proposal, stage)
            VALUES (?, ?, ?)
            """,
            (_now_iso(), proposal.description, "proposed"),
        )
        exp_id = int(cur.lastrowid or 0)
        _log_decision(
            conn, agent="researcher", decision_type="strategy_propose",
            summary=proposal.description,
            rationale=out.hypothesis,
            trigger_metric=triggers[0]["id"] if triggers else "T6",
            trigger_value=float(triggers[0]["value"]) if triggers else 0.0,
            expected_impact=str(out.expected_impact),
            diff_or_config_blob=json.dumps(_proposal_to_blob(proposal)),
            experiment_id=exp_id,
            outcome="pending",
        )
        proposed += 1
        actions.append(f"researcher: proposed experiment #{exp_id}: {proposal.description[:80]}")

    return TickResult(
        actions_taken=actions,
        n_experiments_advanced=advanced,
        n_experiments_proposed=proposed,
    )


def _proposal_to_blob(p: Proposal) -> dict[str, Any]:
    return {
        "description": p.description,
        "config_changes": _jsonable(p.config_changes),
        "category": p.category,
        "rationale": p.rationale,
        "expected_impact": p.expected_impact,
    }


def _jsonable(v: Any) -> Any:
    """Recursively convert sets to lists for JSON serialization."""
    if isinstance(v, set):
        return sorted(list(v), key=str)
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v
