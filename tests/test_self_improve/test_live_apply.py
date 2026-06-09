"""Tests for the autonomous-apply pipeline (2026-05-28).

Covers the four pieces that let the loop change live trading on its own:

  * runtime_overrides — monotonic-tightening merge + the tightening guard
  * live_apply — arm/kill gating, apply/revert (dry-run), override merge,
    pristine-baseline parse, and the circuit-breaker measurement
  * orchestrator — awaiting_canary→canary→live auto-promotion and the
    circuit-breaker auto-rollback, all under dry_run so no bot restarts.

Safety invariant under test: an autonomous live change can only TIGHTEN
(block more entries); a loosening or unknown-key override is refused, and a
§8 PnL/DD breach reverts to the committed baseline automatically.
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
from src.self_improve import live_apply as la
from src.self_improve.llm_client import LLMResponse
from src.self_improve.runtime_overrides import (
    check_tightening_only,
    tighten_overrides,
)

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[2]

# Pristine baseline mirroring live_trading_htf.py constants.
BASELINE = dict(
    min_confidence=0.45,
    symbol_min_confidence={"ETHUSDT": 0.80},
    symbol_directional_conf={"ETHUSDT": {"LONG": 0.95}, "SOLUSDT": {"LONG": 0.95}},
    symbol_side_blocklist={("SOLUSDT", "LONG"), ("XRPUSDT", "LONG"), ("XRPUSDT", "SHORT")},
)


@dataclass
class _StubClient:
    response_text: str = ""
    degraded: bool = True

    def call(self, *, ctx, model, system, user, max_tokens=1024, **kw):
        return LLMResponse(
            text=self.response_text, model=model, input_tokens=1,
            output_tokens=1, cache_read_tokens=0, cache_write_tokens=0,
            cost_usd=0.0, duration_s=0.0, degraded=self.degraded,
        )

    def check_budget(self, raise_on_hard: bool = True):
        return None


# ─────────────────────────────────────────────────────────────────────────
# runtime_overrides — tightening logic
# ─────────────────────────────────────────────────────────────────────────


def test_tighten_applies_new_floor_and_noops_equal_restatement():
    ov = {"SYMBOL_DIRECTIONAL_CONF": {
        "ETHUSDT": {"LONG": 0.95}, "SOLUSDT": {"LONG": 0.95},
        "BTCUSDT": {"SHORT": 0.9},
    }}
    r = tighten_overrides(overrides=ov, **BASELINE)
    assert r["symbol_directional_conf"]["BTCUSDT"]["SHORT"] == 0.9
    assert any("BTCUSDT" in a for a in r["applied"])
    # Equal restatements are benign no-ops, not violations.
    assert check_tightening_only(overrides=ov, **BASELINE) == []


def test_tighten_refuses_to_lower_a_floor():
    bad = {"MIN_CONFIDENCE": 0.1}
    r = tighten_overrides(overrides=bad, **BASELINE)
    assert r["min_confidence"] == 0.45  # unchanged
    assert r["applied"] == []
    violations = check_tightening_only(overrides=bad, **BASELINE)
    assert violations and "would loosen" in violations[0]


def test_tighten_refuses_unknown_key():
    unk = {"TRAILING_DISTANCE_PCT": 0.01}
    violations = check_tightening_only(overrides=unk, **BASELINE)
    assert violations and "unknown key" in violations[0]


def test_tighten_unions_blocklist_add():
    ov = {"SYMBOL_SIDE_BLOCKLIST_ADD": [["BTCUSDT", "SHORT"]]}
    r = tighten_overrides(overrides=ov, **BASELINE)
    assert ("BTCUSDT", "SHORT") in r["symbol_side_blocklist"]
    # existing entries preserved
    assert ("SOLUSDT", "LONG") in r["symbol_side_blocklist"]


def test_tighten_accepts_blocklist_naked_key_alias():
    """SYMBOL_SIDE_BLOCKLIST (live bot's variable name) aliases _ADD."""
    ov = {"SYMBOL_SIDE_BLOCKLIST": [["BTCUSDT", "SHORT"]]}
    r = tighten_overrides(overrides=ov, **BASELINE)
    assert ("BTCUSDT", "SHORT") in r["symbol_side_blocklist"]
    assert any("SYMBOL_SIDE_BLOCKLIST+" in line for line in r["applied"])
    # No "unknown key" entry for the alias.
    assert not any("unknown key" in s and "SYMBOL_SIDE_BLOCKLIST" in s
                   for s in r["skipped"])


def test_pure_noop_is_not_a_valid_promotion():
    noop = {"SYMBOL_DIRECTIONAL_CONF": {"ETHUSDT": {"LONG": 0.95}}}
    violations = check_tightening_only(overrides=noop, **BASELINE)
    assert violations == ["override has no applicable tightening effect"]


# ─────────────────────────────────────────────────────────────────────────
# live_apply — gating, apply/revert, merge, baseline
# ─────────────────────────────────────────────────────────────────────────


CC = {"SYMBOL_DIRECTIONAL_CONF": {"BTCUSDT": {"SHORT": 0.9}}}


def test_apply_refused_when_disarmed(tmp_path: Path):
    r = la.apply_live(experiment_id=1, config_changes=CC, base_dir=tmp_path,
                      repo=REPO, dry_run=True)
    assert not r.ok and "not armed" in r.reason


def test_apply_writes_override_when_armed(tmp_path: Path):
    la.armed_flag(tmp_path).touch()
    r = la.apply_live(experiment_id=2, config_changes=CC, base_dir=tmp_path,
                      repo=REPO, dry_run=True)
    assert r.ok and r.override_written
    payload = json.loads(la.override_path(tmp_path).read_text())
    assert payload["config_changes"]["SYMBOL_DIRECTIONAL_CONF"]["BTCUSDT"]["SHORT"] == 0.9
    assert payload["experiment_id"] == 2


def test_kill_switch_blocks_apply_even_when_armed(tmp_path: Path):
    la.armed_flag(tmp_path).touch()
    la.kill_flag(tmp_path).touch()
    r = la.apply_live(experiment_id=3, config_changes=CC, base_dir=tmp_path,
                      repo=REPO, dry_run=True)
    assert not r.ok and "kill switch" in r.reason
    assert not la.override_path(tmp_path).exists()


def test_apply_refuses_loosening_override(tmp_path: Path):
    la.armed_flag(tmp_path).touch()
    r = la.apply_live(experiment_id=4, config_changes={"MIN_CONFIDENCE": 0.1},
                      base_dir=tmp_path, repo=REPO, dry_run=True)
    assert not r.ok and r.violations
    assert not la.override_path(tmp_path).exists()


def test_revert_clears_override(tmp_path: Path):
    la.armed_flag(tmp_path).touch()
    la.apply_live(experiment_id=5, config_changes=CC, base_dir=tmp_path,
                  repo=REPO, dry_run=True)
    assert la.override_path(tmp_path).exists()
    r = la.revert_live(reason="test", base_dir=tmp_path, repo=REPO, dry_run=True)
    assert r.ok and not la.override_path(tmp_path).exists()


def test_apply_merges_with_existing_override(tmp_path: Path):
    la.armed_flag(tmp_path).touch()
    la.apply_live(experiment_id=6, config_changes=CC, base_dir=tmp_path,
                  repo=REPO, dry_run=True)
    # second experiment adds a different gate
    cc2 = {"SYMBOL_SIDE_BLOCKLIST_ADD": [["ETHUSDT", "SHORT"]]}
    la.apply_live(experiment_id=7, config_changes=cc2, base_dir=tmp_path,
                  repo=REPO, dry_run=True)
    merged = json.loads(la.override_path(tmp_path).read_text())["config_changes"]
    assert merged["SYMBOL_DIRECTIONAL_CONF"]["BTCUSDT"]["SHORT"] == 0.9
    assert ["ETHUSDT", "SHORT"] in merged["SYMBOL_SIDE_BLOCKLIST_ADD"]


def test_baseline_parse_matches_live_constants():
    b = la.read_baseline_constants(REPO)
    assert b is not None
    assert b["min_confidence"] == 0.45
    assert b["symbol_directional_conf"]["ETHUSDT"]["LONG"] == 0.95
    assert ("XRPUSDT", "LONG") in b["symbol_side_blocklist"]


# ─────────────────────────────────────────────────────────────────────────
# Circuit breaker
# ─────────────────────────────────────────────────────────────────────────


def _mkdb(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT, action TEXT, data TEXT,
                price REAL, pnl REAL, confidence REAL, reason TEXT,
                created_at TEXT, is_testnet INTEGER DEFAULT 0)"""
        )
        migrate(conn)


def _close(db: Path, *, ts: datetime, pnl: float, symbol="BTCUSDT", side="LONG"):
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO trades(timestamp, created_at, symbol, action, data, "
            "pnl, is_testnet) VALUES (?, ?, ?, ?, '{}', ?, 1)",
            (ts.isoformat(), ts.isoformat(), symbol, f"CLOSE_{side}", pnl),
        )


def test_breaker_clean_does_not_trip(tmp_path: Path):
    db = tmp_path / "t.db"
    _mkdb(db)
    base = datetime.now(UTC)
    for i in range(5):
        _close(db, ts=base + timedelta(minutes=i), pnl=+10.0)
    with sqlite3.connect(str(db)) as conn:
        r = la.measure_since(conn, since_iso="2026-01-01T00:00:00", capital=5000.0)
    assert not r.tripped and r.realized_pnl == 50.0


def test_breaker_trips_on_loss(tmp_path: Path):
    db = tmp_path / "t.db"
    _mkdb(db)
    base = datetime.now(UTC)
    for i in range(5):
        _close(db, ts=base + timedelta(minutes=i), pnl=-60.0)  # -300 total
    with sqlite3.connect(str(db)) as conn:
        r = la.measure_since(conn, since_iso="2026-01-01T00:00:00", capital=5000.0)
    assert r.tripped and "realized loss" in r.reason  # -300 ≥ 5% of 5000=250


def test_breaker_respects_min_closes(tmp_path: Path):
    db = tmp_path / "t.db"
    _mkdb(db)
    base = datetime.now(UTC)
    _close(db, ts=base, pnl=-500.0)  # one big loss, but below min closes
    with sqlite3.connect(str(db)) as conn:
        r = la.measure_since(conn, since_iso="2026-01-01T00:00:00", capital=5000.0)
    assert not r.tripped


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator — canary / live transitions (dry_run, no real restart)
# ─────────────────────────────────────────────────────────────────────────


def _seed_experiment(db: Path, *, stage: str, config_changes: dict,
                     ts_canary: str | None = None, ts_live: str | None = None) -> int:
    created = datetime.now(UTC).isoformat()
    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute(
            "INSERT INTO experiments(ts_created, proposal, stage, "
            "ts_promoted_canary, ts_promoted_live) VALUES (?, ?, ?, ?, ?)",
            (created, "test exp", stage, ts_canary, ts_live),
        )
        exp_id = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO decisions(ts, agent, decision_type, summary, "
            "rationale, diff_or_config_blob, experiment_id) "
            "VALUES (?, 'researcher', 'strategy_propose', 'x', 'y', ?, ?)",
            (created, json.dumps({"config_changes": config_changes}), exp_id),
        )
    return exp_id


def _stage(db: Path, exp_id: int) -> str:
    with sqlite3.connect(str(db)) as conn:
        return conn.execute(
            "SELECT stage FROM experiments WHERE id=?", (exp_id,)
        ).fetchone()[0]


def test_canary_gate_holds_when_disarmed(tmp_path: Path):
    db = tmp_path / "t.db"
    _mkdb(db)
    exp_id = _seed_experiment(db, stage="awaiting_canary_approval", config_changes=CC)
    sd = tmp_path / "sd"  # no AUTONOMY_ARMED flag
    orch.run_tick(db_path=db, client=_StubClient(), base_dir=sd, repo=REPO,
                  dry_run=True)
    assert _stage(db, exp_id) == "awaiting_canary_approval"
    assert not la.override_path(sd).exists()


def test_armed_canary_applies_and_promotes_to_canary(tmp_path: Path):
    db = tmp_path / "t.db"
    _mkdb(db)
    exp_id = _seed_experiment(db, stage="awaiting_canary_approval", config_changes=CC)
    sd = tmp_path / "sd"
    sd.mkdir()
    la.armed_flag(sd).touch()
    orch.run_tick(db_path=db, client=_StubClient(), base_dir=sd, repo=REPO,
                  dry_run=True)
    assert _stage(db, exp_id) == "canary"
    assert la.override_path(sd).exists()


def test_canary_promotes_to_live_after_window(tmp_path: Path):
    db = tmp_path / "t.db"
    _mkdb(db)
    old = (datetime.now(UTC) - timedelta(hours=la.CANARY_HOURS + 1)).isoformat()
    exp_id = _seed_experiment(db, stage="canary", config_changes=CC, ts_canary=old)
    sd = tmp_path / "sd"
    sd.mkdir()
    la.armed_flag(sd).touch()
    orch.run_tick(db_path=db, client=_StubClient(), base_dir=sd, repo=REPO,
                  dry_run=True)
    assert _stage(db, exp_id) == "live"


def test_canary_circuit_breaker_reverts(tmp_path: Path):
    db = tmp_path / "t.db"
    _mkdb(db)
    recent = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    exp_id = _seed_experiment(db, stage="canary", config_changes=CC, ts_canary=recent)
    # bleed since apply: 5 closes of -60 = -300 ≥ 5% of 5000
    base = datetime.now(UTC)
    for i in range(5):
        _close(db, ts=base + timedelta(minutes=i), pnl=-60.0)
    sd = tmp_path / "sd"
    sd.mkdir()
    la.armed_flag(sd).touch()
    la.apply_live(experiment_id=exp_id, config_changes=CC, base_dir=sd,
                  repo=REPO, dry_run=True)  # simulate the live override present
    orch.run_tick(db_path=db, client=_StubClient(), base_dir=sd, repo=REPO,
                  dry_run=True)
    assert _stage(db, exp_id) == "rolled_back"
    assert not la.override_path(sd).exists()  # reverted


def test_live_circuit_breaker_reverts(tmp_path: Path):
    db = tmp_path / "t.db"
    _mkdb(db)
    recent = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    exp_id = _seed_experiment(db, stage="live", config_changes=CC, ts_live=recent)
    base = datetime.now(UTC)
    for i in range(5):
        _close(db, ts=base + timedelta(minutes=i), pnl=-60.0)
    sd = tmp_path / "sd"
    sd.mkdir()
    la.armed_flag(sd).touch()
    la.apply_live(experiment_id=exp_id, config_changes=CC, base_dir=sd,
                  repo=REPO, dry_run=True)
    orch.run_tick(db_path=db, client=_StubClient(), base_dir=sd, repo=REPO,
                  dry_run=True)
    assert _stage(db, exp_id) == "rolled_back"


def test_live_healthy_stays_live(tmp_path: Path):
    db = tmp_path / "t.db"
    _mkdb(db)
    recent = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    exp_id = _seed_experiment(db, stage="live", config_changes=CC, ts_live=recent)
    base = datetime.now(UTC)
    for i in range(5):
        _close(db, ts=base + timedelta(minutes=i), pnl=+10.0)
    sd = tmp_path / "sd"
    sd.mkdir()
    la.armed_flag(sd).touch()
    orch.run_tick(db_path=db, client=_StubClient(), base_dir=sd, repo=REPO,
                  dry_run=True)
    assert _stage(db, exp_id) == "live"
