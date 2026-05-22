"""Tests for the LLM client wrapper.

We never make a real API call here — the client's graceful-degradation
path is what we exercise. Token accounting + budget tracking are
verified by inserting synthetic agent_runs rows.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.self_improve.migrate import migrate
from src.self_improve.llm_client import (
    MODEL_HAIKU,
    BudgetExceeded,
    CallContext,
    LLMClient,
    _calc_cost_usd,
    load_persona,
)


def _seed_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        migrate(conn)


def _insert_synthetic_run(
    db: Path, *, day_iso: str, cost_usd: float, agent: str = "test"
) -> None:
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """
            INSERT INTO agent_runs(ts, agent, model, duration_s,
                                   input_tokens, output_tokens,
                                   output_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{day_iso}T12:00:00Z",
                agent,
                MODEL_HAIKU,
                1.0,
                100, 200,
                json.dumps({"cost_usd": cost_usd, "degraded": False}),
            ),
        )


def test_no_api_key_returns_degraded_response(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    cli = LLMClient(db_path=db, api_key="")
    assert not cli.has_api_key

    resp = cli.call(
        ctx=CallContext(agent="risk-officer"),
        model=MODEL_HAIKU,
        system="you are a test",
        user="hello",
    )
    assert resp.degraded is True
    assert "ANTHROPIC_API_KEY" in (resp.error or "")
    assert resp.cost_usd == 0.0


def test_call_records_audit_row_even_when_degraded(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    cli = LLMClient(db_path=db, api_key="")
    cli.call(
        ctx=CallContext(agent="risk-officer", context_summary="ctx1"),
        model=MODEL_HAIKU,
        system="s", user="u",
    )
    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT agent, error, output_summary FROM agent_runs WHERE agent=?",
            ("risk-officer",),
        ).fetchone()
    assert row is not None
    assert row[0] == "risk-officer"
    assert "ANTHROPIC_API_KEY" in (row[1] or "")
    out = json.loads(row[2])
    assert out["degraded"] is True
    assert out["cost_usd"] == 0.0


def test_budget_state_sums_todays_calls(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    import time as _t
    today = _t.strftime("%Y-%m-%d", _t.gmtime())
    yesterday = "2026-05-21"
    _insert_synthetic_run(db, day_iso=today, cost_usd=3.50)
    _insert_synthetic_run(db, day_iso=today, cost_usd=1.25)
    _insert_synthetic_run(db, day_iso=yesterday, cost_usd=99.0)  # not today

    cli = LLMClient(db_path=db, api_key="", soft_cap_usd=5.0, hard_cap_usd=10.0)
    state = cli.budget_state()
    assert state.n_calls == 2
    assert state.spend_usd == 4.75
    assert state.soft_cap_usd == 5.0
    assert state.hard_cap_usd == 10.0
    assert not state.over_soft
    assert not state.over_hard


def test_hard_cap_blocks_real_call_but_not_degraded(tmp_path: Path) -> None:
    """When over the hard cap, even real calls return a degraded response
    instead of hitting the API."""
    db = tmp_path / "trades.db"
    _seed_db(db)
    import time as _t
    today = _t.strftime("%Y-%m-%d", _t.gmtime())
    _insert_synthetic_run(db, day_iso=today, cost_usd=50.0)  # over hard cap

    cli = LLMClient(db_path=db, api_key="fake", soft_cap_usd=5.0, hard_cap_usd=10.0)
    resp = cli.call(
        ctx=CallContext(agent="risk-officer"),
        model=MODEL_HAIKU,
        system="s", user="u",
    )
    assert resp.degraded is True
    assert "budget" in (resp.error or "").lower()


def test_check_budget_raises_when_over_hard(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    import time as _t
    today = _t.strftime("%Y-%m-%d", _t.gmtime())
    _insert_synthetic_run(db, day_iso=today, cost_usd=50.0)

    cli = LLMClient(db_path=db, soft_cap_usd=5.0, hard_cap_usd=10.0)
    try:
        cli.check_budget(raise_on_hard=True)
        raised = False
    except BudgetExceeded:
        raised = True
    assert raised


def test_cost_calculation_haiku() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 500_000,
    }
    # Haiku rates: $1/Mt input, $5/Mt output, $0.10/Mt cache_read
    expected = 1.0 + 0.5 + 0.05  # 1.0 + 100k*$5/M + 500k*$0.10/M
    cost = _calc_cost_usd(MODEL_HAIKU, usage)
    assert abs(cost - expected) < 1e-9


def test_load_persona_returns_empty_for_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_persona("does-not-exist") == ""
