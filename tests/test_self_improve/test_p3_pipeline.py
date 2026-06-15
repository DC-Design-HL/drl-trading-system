"""P3 forward-validation pipeline tests (PROFITABILITY_PLAN.md §3/P3).

Two levels:

  * forward_sim.forward_config_from_overrides + evaluate_forward_gate over a
    synthetic kline cache — the override→sim-config mapping and the
    net-PnL/DD gate compute end-to-end on real sim machinery.
  * orchestrator routing — an envelope/"forward" experiment travels
    proposed → paper → awaiting_canary_approval and is HELD at the canary
    gate while autonomy is disarmed (the P3 acceptance dry-run), plus a
    blocklist-removal proposal is escalated instead of piped.

The orchestrator level monkeypatches the forward gate + risk officer so the
DB state machine is exercised deterministically without running the sim or
calling an LLM.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.self_improve.migrate import migrate
from src.self_improve import forward_sim as fs
from src.self_improve import orchestrator as orch
from src.self_improve.forward_sim import (
    ForwardGateResult,
    ForwardSimConfig,
    forward_config_from_overrides,
)

UTC = timezone.utc
_5M_NS = 5 * 60 * 1_000_000_000


# ─────────────────────────────────────────────────────────────────────────
# forward_config_from_overrides — the override→sim-config mapping
# ─────────────────────────────────────────────────────────────────────────


def test_forward_config_maps_exit_knobs() -> None:
    cfg, unexpressible = forward_config_from_overrides({
        "TRAILING_DISTANCE_PCT": 0.008,
        "STAGNANT_HOURS": 10.0,
        "ADX_GUARD_MIN": 25.0,
        "COOLDOWN_SECONDS": 3600,
    })
    assert cfg.trailing_distance_pct == 0.008
    assert cfg.stagnant_hours == 10.0
    assert cfg.adx_guard_min == 25.0
    assert cfg.cooldown_seconds == 3600
    assert unexpressible == []


def test_forward_config_flags_min_hold_as_unexpressible() -> None:
    # MIN_HOLD_SECONDS is forward-validated by table but the sim can't model
    # an in-position min-hold (it only decides while flat).
    cfg, unexpressible = forward_config_from_overrides(
        {"MIN_HOLD_SECONDS": 5400})
    assert unexpressible == ["MIN_HOLD_SECONDS"]


def test_forward_config_unions_blocklist_add() -> None:
    base = ForwardSimConfig()
    cfg, _ = forward_config_from_overrides(
        {"SYMBOL_SIDE_BLOCKLIST_ADD": [["BTCUSDT", "SHORT"]]}, base=base)
    assert ("BTCUSDT", "SHORT") in cfg.blocklist
    # existing baseline blocks preserved.
    assert all(b in cfg.blocklist for b in base.blocklist)


# ─────────────────────────────────────────────────────────────────────────
# evaluate_forward_gate — real sim over a synthetic cache
# ─────────────────────────────────────────────────────────────────────────


def _seed_synthetic_cache(tmp_path: Path, symbol: str = "BTCUSDT", seed: int = 42) -> None:
    start_ns = int(datetime(2026, 5, 25, tzinfo=UTC).timestamp() * 1_000_000_000)
    n_5m = 14 * 24 * 12
    ts = np.arange(n_5m, dtype=np.int64) * _5M_NS + start_ns
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.standard_normal(n_5m) * 0.5)
    opens = np.roll(closes, 1); opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n_5m))
    lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n_5m))
    df = pd.DataFrame({
        "ts": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.ones(n_5m),
    })
    tmp_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_path / f"{symbol}_5m.parquet", engine="pyarrow")
    for tf, hours in (("15m", 0.25), ("1h", 1.0), ("4h", 4.0)):
        step = int(hours * 12)
        df.iloc[::step].reset_index(drop=True).to_parquet(
            tmp_path / f"{symbol}_{tf}.parquet", engine="pyarrow")


def test_evaluate_forward_gate_runs_and_is_deterministic(tmp_path: Path) -> None:
    _seed_synthetic_cache(tmp_path)
    start = datetime(2026, 6, 6, tzinfo=UTC)
    end = datetime(2026, 6, 7, tzinfo=UTC)
    kw = dict(
        config_overrides={"TRAILING_DISTANCE_PCT": 0.008},
        start=start, end=end, capital=5000.0,
        symbols=("BTCUSDT",), cache_base=tmp_path,
    )
    r1 = fs.evaluate_forward_gate(**kw)
    r2 = fs.evaluate_forward_gate(**kw)
    assert isinstance(r1, ForwardGateResult)
    assert r1.to_json()["engine"] == "forward"
    # Deterministic: same cache + same config → identical verdict + numbers.
    assert r1.to_json() == r2.to_json()
    # to_json carries the keys downstream readers expect.
    assert "portfolio_metrics" in r1.to_json()


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator routing — the P3 acceptance dry-run + escalation
# ─────────────────────────────────────────────────────────────────────────


class _NoVetoVerdict:
    is_veto = False
    reasons: list[str] = []


def _seed_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT, action TEXT,
                data TEXT, price REAL, pnl REAL, confidence REAL,
                reason TEXT, is_testnet INTEGER DEFAULT 0
            )
            """
        )
        migrate(conn)


def _insert_proposed(db: Path, config_changes: dict, *, desc: str) -> int:
    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute(
            "INSERT INTO experiments(ts_created, proposal, stage) "
            "VALUES (?, ?, 'proposed')",
            (datetime.now(UTC).isoformat(), desc),
        )
        exp_id = int(cur.lastrowid)
        blob = json.dumps({
            "description": desc, "config_changes": config_changes,
            "category": "config_tune", "rationale": "test", "expected_impact": "",
        })
        conn.execute(
            "INSERT INTO decisions(ts, agent, decision_type, summary, "
            "rationale, diff_or_config_blob, experiment_id, outcome) "
            "VALUES (?, 'researcher', 'strategy_propose', ?, 'r', ?, ?, 'pending')",
            (datetime.now(UTC).isoformat(), desc, blob, exp_id),
        )
        return exp_id


def _stage(db: Path, exp_id: int) -> str:
    with sqlite3.connect(str(db)) as conn:
        return conn.execute(
            "SELECT stage FROM experiments WHERE id=?", (exp_id,)
        ).fetchone()[0]


def _passing_gate(**kwargs) -> ForwardGateResult:
    return ForwardGateResult(
        pass_gate=True, candidate_pnl=120.0, baseline_pnl=100.0,
        delta_pnl=20.0, candidate_max_dd_pct=2.0, baseline_max_dd_pct=2.0,
        n_candidate_trades=12, reasons=[], unexpressible_keys=[],
    )


def test_p3_dry_run_proposed_to_held_at_canary(tmp_path: Path, monkeypatch) -> None:
    """Acceptance dry-run: a TRAILING_DISTANCE_PCT change inside its envelope
    goes proposed → paper (forward gate) → awaiting_canary_approval, and is
    HELD at the canary gate because autonomy is disarmed."""
    db = tmp_path / "trading.db"
    _seed_db(db)
    base_dir = tmp_path / "self_improve"   # no AUTONOMY_ARMED → disarmed
    base_dir.mkdir()

    # Deterministic gate + RO so we test the state machine, not the sim/LLM.
    monkeypatch.setattr(orch, "evaluate_forward_gate", _passing_gate)
    monkeypatch.setattr(orch, "risk_evaluate", lambda *a, **k: _NoVetoVerdict())

    exp_id = _insert_proposed(
        db, {"TRAILING_DISTANCE_PCT": 0.008}, desc="loosen trailing")

    # Tick 1: proposed → paper (routed through the forward gate).
    orch.run_tick(db_path=db, client=None, base_dir=base_dir,
                  repo=tmp_path, dry_run=True)
    assert _stage(db, exp_id) == "paper"

    # Backdate the paper promotion so the paper window has elapsed.
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "UPDATE experiments SET ts_promoted_paper=? WHERE id=?",
            ((datetime.now(UTC) - timedelta(days=orch.PAPER_DAYS + 1)).isoformat(),
             exp_id),
        )

    # Tick 2: paper → awaiting_canary_approval (forward paper gate).
    orch.run_tick(db_path=db, client=None, base_dir=base_dir,
                  repo=tmp_path, dry_run=True)
    assert _stage(db, exp_id) == "awaiting_canary_approval"

    # Tick 3: held at canary because autonomy is disarmed (no live apply).
    res = orch.run_tick(db_path=db, client=None, base_dir=base_dir,
                        repo=tmp_path, dry_run=True)
    assert _stage(db, exp_id) == "awaiting_canary_approval"
    assert any("held at canary" in a for a in res.actions_taken), res.actions_taken


def test_p3_blocklist_removal_is_escalated_not_piped(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "trading.db"
    _seed_db(db)
    base_dir = tmp_path / "self_improve"
    base_dir.mkdir()

    posted: list[str] = []
    monkeypatch.setattr(orch, "_telegram_post", lambda msg: posted.append(msg))

    from src.self_improve.safety_envelopes import BLOCKLIST_REMOVE_KEY
    exp_id = _insert_proposed(
        db, {BLOCKLIST_REMOVE_KEY: [["XRPUSDT", "LONG"]]}, desc="unblock XRP")

    orch.run_tick(db_path=db, client=None, base_dir=base_dir,
                  repo=tmp_path, dry_run=True)

    # Not advanced into backtest/paper — rejected + escalated to Chen.
    assert _stage(db, exp_id) == "rejected"
    assert any("BLOCKLIST REMOVAL" in m for m in posted), posted
