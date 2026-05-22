"""Orchestrator tests — end-to-end with mocked LLM.

The orchestrator threads together Researcher → Risk Officer → state
machine. We verify the gating logic (researcher gate, in-flight defer,
rate limit), the no-op paths (no API key, no triggers, recent
proposal), and the trigger-fired pipeline.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.self_improve.migrate import migrate
from src.self_improve import orchestrator as orch
from src.self_improve.llm_client import LLMResponse


UTC = timezone.utc


def _seed_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT,
                action TEXT,
                data TEXT,
                price REAL,
                pnl REAL,
                confidence REAL,
                reason TEXT,
                is_testnet INTEGER DEFAULT 0
            )
            """
        )
        migrate(conn)


def _add_pair(
    db: Path,
    *,
    open_ts: datetime,
    close_ts: datetime,
    symbol: str,
    side: str,
    pnl: float,
    confidence: float = 0.7,
) -> None:
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO trades(timestamp, symbol, action, data, "
            "confidence, is_testnet) VALUES (?, ?, ?, '{}', ?, 1)",
            (open_ts.isoformat(), symbol, f"OPEN_{side}", confidence),
        )
        conn.execute(
            "INSERT INTO trades(timestamp, symbol, action, data, "
            "pnl, is_testnet) VALUES (?, ?, ?, '{}', ?, 1)",
            (close_ts.isoformat(), symbol, f"CLOSE_{side}", pnl),
        )


@dataclass
class _StubClient:
    response_text: str = ""
    degraded: bool = False
    has_api_key: bool = True
    calls_seen: int = 0

    def call(self, *, ctx, model, system, user, max_tokens=1024, **kw):
        self.calls_seen += 1
        return LLMResponse(
            text=self.response_text,
            model=model, input_tokens=10, output_tokens=10,
            cache_read_tokens=0, cache_write_tokens=0,
            cost_usd=0.001, duration_s=0.05,
            degraded=self.degraded,
        )

    def check_budget(self, raise_on_hard: bool = True):
        return None


# ─────────────────────────────────────────────────────────────────────────
# Gate logic
# ─────────────────────────────────────────────────────────────────────────


def test_should_propose_blocked_by_in_flight(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO experiments(ts_created, proposal, stage) "
            "VALUES (?, ?, ?)",
            ("2026-05-22T00:00:00", "test", "proposed"),
        )
        ok, reason = orch._should_propose(conn, triggers=[{"id": "T1", "value": -1}])
    assert not ok
    assert "in flight" in reason


def test_should_propose_rate_limit(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    with sqlite3.connect(str(db)) as conn:
        # Researcher proposed 1h ago — rate limit should block
        recent = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        conn.execute(
            "INSERT INTO decisions(ts, agent, decision_type, summary, "
            "rationale) VALUES (?, ?, ?, ?, ?)",
            (recent, "researcher", "strategy_propose", "x", "y"),
        )
        ok, reason = orch._should_propose(
            conn, triggers=[{"id": "T1", "value": -1}]
        )
    assert not ok
    assert "rate" in reason.lower()


def test_should_propose_triggers_fire(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    with sqlite3.connect(str(db)) as conn:
        ok, reason = orch._should_propose(
            conn, triggers=[{"id": "T1", "value": -1}]
        )
    assert ok
    assert "T1" in reason


def test_should_propose_t6_quiet_tick(tmp_path: Path) -> None:
    """No triggers + no recent researcher run → T6 fires."""
    db = tmp_path / "trades.db"
    _seed_db(db)
    with sqlite3.connect(str(db)) as conn:
        ok, reason = orch._should_propose(conn, triggers=[])
    assert ok
    assert "quiet tick" in reason


# ─────────────────────────────────────────────────────────────────────────
# Tick — end-to-end
# ─────────────────────────────────────────────────────────────────────────


def test_run_tick_with_no_api_key_writes_no_change_decision(tmp_path: Path) -> None:
    """When the Researcher LLM is degraded, the tick must NOT create an
    experiments row, must NOT advance any stages, and must log a
    no_change decision row."""
    db = tmp_path / "trades.db"
    _seed_db(db)
    # Seed some real trades so the trigger evaluator sees data
    base = datetime.now(UTC)
    for i in range(15):
        _add_pair(
            db,
            open_ts=base - timedelta(hours=i * 4 + 1),
            close_ts=base - timedelta(hours=i * 4),
            symbol="BTCUSDT", side="LONG",
            pnl=-5.0,  # all losing → triggers fire
        )

    stub = _StubClient(degraded=True)
    result = orch.run_tick(db_path=db, client=stub)
    assert result.n_experiments_proposed == 0
    with sqlite3.connect(str(db)) as conn:
        n_exp = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        # A no_change decision should be logged
        n_dec = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE agent='researcher' AND decision_type='no_change'"
        ).fetchone()[0]
    assert n_exp == 0
    assert n_dec == 1


def test_run_tick_creates_experiment_when_researcher_proposes(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    base = datetime.now(UTC)
    for i in range(15):
        _add_pair(
            db,
            open_ts=base - timedelta(hours=i * 4 + 1),
            close_ts=base - timedelta(hours=i * 4),
            symbol="BTCUSDT", side="LONG", pnl=-5.0,
        )

    # Configure the stub: first call (Researcher) returns a propose,
    # subsequent calls (Risk Officer LLM phase) return approve.
    proposal_payload = {
        "verdict": "propose",
        "hypothesis": "BTC LONGs losing — raise confidence floor",
        "proposal": {
            "description": "Raise BTC confidence floor to 0.70",
            "config_changes": {"SYMBOL_MIN_CONFIDENCE": {"BTCUSDT": 0.70}},
            "category": "config_tune",
            "rationale": "low-conf BTC entries cluster",
            "expected_impact": "block ~half BTC entries",
        },
        "expected_impact": {"metric": "net_pnl_usd", "delta_estimate": 50},
        "confidence": 0.7,
        "alternatives_considered": ["wait", "blocklist"],
    }
    ro_approve = '{"verdict": "approve", "concerns": []}'

    class _MultiStub:
        has_api_key = True
        def __init__(self) -> None:
            self.idx = 0
            self.responses = [
                LLMResponse(
                    text=json.dumps(proposal_payload),
                    model="claude-opus-4-7",
                    input_tokens=100, output_tokens=100,
                    cache_read_tokens=0, cache_write_tokens=0,
                    cost_usd=0.01, duration_s=0.1,
                ),
                LLMResponse(
                    text=ro_approve,
                    model="claude-haiku-4-5",
                    input_tokens=50, output_tokens=10,
                    cache_read_tokens=0, cache_write_tokens=0,
                    cost_usd=0.001, duration_s=0.05,
                ),
            ]
        def call(self, **kw):
            if self.idx < len(self.responses):
                r = self.responses[self.idx]
                self.idx += 1
                return r
            return self.responses[-1]
        def check_budget(self, raise_on_hard=True):
            return None

    stub = _MultiStub()
    result = orch.run_tick(db_path=db, client=stub)
    assert result.n_experiments_proposed == 1
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT id, stage, proposal FROM experiments"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "proposed"
    assert "Raise BTC confidence floor" in rows[0][2]


def test_run_tick_vetoes_when_risk_officer_says_no(tmp_path: Path) -> None:
    """Researcher proposes but Risk Officer Phase 1 vetoes (out-of-cap
    constant): no experiment row, decision logged as 'veto'."""
    db = tmp_path / "trades.db"
    _seed_db(db)
    base = datetime.now(UTC)
    for i in range(15):
        _add_pair(
            db,
            open_ts=base - timedelta(hours=i * 4 + 1),
            close_ts=base - timedelta(hours=i * 4),
            symbol="BTCUSDT", side="LONG", pnl=-5.0,
        )

    # Researcher proposes a config change that VIOLATES Risk Officer
    # caps (STAGNANT_HOURS=100h is above the 24h cap)
    bad_payload = {
        "verdict": "propose",
        "hypothesis": "stagnant exit too aggressive",
        "proposal": {
            "description": "Widen stagnant_hours to 100",
            "config_changes": {"STAGNANT_HOURS": 100.0},
            "category": "config_tune",
            "rationale": "give trades more room",
            "expected_impact": "fewer false-exit losses",
        },
        "expected_impact": {"metric": "net_pnl_usd", "delta_estimate": 30},
        "confidence": 0.6,
        "alternatives_considered": ["narrow band", "remove altogether"],
    }
    stub = _StubClient(response_text=json.dumps(bad_payload))
    result = orch.run_tick(db_path=db, client=stub)

    assert result.n_experiments_proposed == 0  # vetoed
    with sqlite3.connect(str(db)) as conn:
        n_exp = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        n_veto = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE decision_type='veto'"
        ).fetchone()[0]
    assert n_exp == 0
    assert n_veto >= 1
