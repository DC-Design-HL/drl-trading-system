"""P5 — sidelined-alpha measurement + funding-aware accounting (§3/P5).

Covers: entry_signals snapshot round-trip (storage), funding estimate
(boundary crossing + sign by side) and funding-aware metrics, the
signal-value report math, and the reviewer entry-signal pattern section.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from scripts.self_improve.migrate import migrate
from scripts.self_improve import signal_value_report as svr
from src.data.storage import SQLiteStorage
from src.self_improve import metrics as M
from src.self_improve.reviewer import entry_signal_patterns_section

UTC = timezone.utc


def _mkdb(tmp_path):
    db = tmp_path / "trading.db"
    SQLiteStorage(str(db))
    with sqlite3.connect(str(db)) as conn:
        migrate(conn)
    return db


# ── funding estimate: boundary crossing + sign ──────────────────────────


def test_estimate_funding_boundary_and_sign():
    e = datetime(2026, 6, 18, 0, 0, tzinfo=UTC)
    x = datetime(2026, 6, 18, 20, 0, tzinfo=UTC)
    # boundaries at 00:00 (==entry, excluded), 08:00, 16:00 (both in), next day (out)
    fts = [datetime(2026, 6, 18, h, 0, tzinfo=UTC).timestamp() for h in (0, 8, 16)]
    fts.append(datetime(2026, 6, 19, 0, 0, tzinfo=UTC).timestamp())
    frt = [0.0001] * 4
    long_cost = M.estimate_funding_usd(
        entry_ts=e, exit_ts=x, notional=3000, side="LONG",
        funding_ts=fts, funding_rates=frt)
    short_cost = M.estimate_funding_usd(
        entry_ts=e, exit_ts=x, notional=3000, side="SHORT",
        funding_ts=fts, funding_rates=frt)
    assert round(long_cost, 4) == 0.6   # 2 boundaries × 0.0001 × 3000, LONG pays
    assert round(short_cost, 4) == -0.6  # SHORT receives


def test_estimate_funding_empty_inputs():
    e = datetime(2026, 6, 18, tzinfo=UTC)
    assert M.estimate_funding_usd(entry_ts=e, exit_ts=e + timedelta(hours=4),
                                  notional=3000, side="LONG",
                                  funding_ts=[], funding_rates=[]) == 0.0
    assert M.estimate_funding_usd(entry_ts=e, exit_ts=e, notional=3000,
                                  side="LONG", funding_ts=[1.0], funding_rates=[0.1]) == 0.0


def test_funding_aware_metrics():
    t = [
        M.TradeClose(ts=datetime(2026, 6, 18, tzinfo=UTC), symbol="BTCUSDT",
                     side="LONG", pnl=20.0, funding_usd=2.0),
        M.TradeClose(ts=datetime(2026, 6, 18, tzinfo=UTC), symbol="ETHUSDT",
                     side="SHORT", pnl=-5.0, funding_usd=-1.0),
    ]
    assert M.net_pnl(t) == 15.0                    # gross
    assert M.funding_total(t) == 1.0
    assert M.net_pnl_after_funding(t) == 14.0      # 15 - 1
    s = M.summarize(t)
    assert s["net_pnl_usd"] == 15.0
    assert s["funding_usd_total"] == 1.0
    assert s["net_pnl_after_funding_usd"] == 14.0


def test_tradeclose_backcompat_without_funding():
    # Pre-P5 construction (no funding_usd) still works, funding defaults 0.
    t = M.TradeClose(ts=datetime(2026, 6, 18, tzinfo=UTC), symbol="BTCUSDT",
                     side="LONG", pnl=10.0)
    assert t.funding_usd == 0.0
    assert M.net_pnl_after_funding([t]) == 10.0


# ── entry_signals storage round-trip ────────────────────────────────────


def test_entry_signals_roundtrip(tmp_path):
    db = _mkdb(tmp_path)
    s = SQLiteStorage(str(db))
    s.log_entry_signal(ts="2026-06-18T10:00:00", symbol="ETHUSDT", side="SHORT",
                       snapshot_type="entry", signals={"whale": {"direction": 0.2}},
                       structure_conf=0.6, model_action="LONG", model_confidence=0.5,
                       experiment_id=14)
    with sqlite3.connect(str(db)) as c:
        row = c.execute(
            "SELECT symbol,side,snapshot_type,model_action,experiment_id FROM entry_signals"
        ).fetchone()
        assert row == ("ETHUSDT", "SHORT", "entry", "LONG", 14)


# ── signal-value report ─────────────────────────────────────────────────


def _seed_paired(db, ts, symbol, side, outcome_pnl, *, model_action, whale_dir):
    """Seed an OPEN (with entry snapshot) + a CLOSE with the outcome pnl."""
    s = SQLiteStorage(str(db))
    s.log_trade({"timestamp": ts, "symbol": symbol, "action": f"OPEN_{side}",
                 "is_testnet": True, "pnl": 0.0})
    s.log_entry_signal(ts=ts, symbol=symbol, side=side, snapshot_type="entry",
                       signals={"whale": {"direction": whale_dir, "intent": "x"}},
                       model_action=model_action)
    s.log_trade({"timestamp": ts, "symbol": symbol, "action": f"CLOSE_{side}",
                 "is_testnet": True, "pnl": outcome_pnl})


def test_signal_value_report_math(tmp_path):
    db = _mkdb(tmp_path)
    base = datetime(2026, 6, 18, tzinfo=UTC)
    # 4 trades: model-agree winners, model-disagree losers.
    rows = [
        ("BTCUSDT", "LONG", +10.0, "LONG", 0.9),   # agree, whale aligned, win
        ("ETHUSDT", "LONG", +6.0, "LONG", 0.8),    # agree, win
        ("BTCUSDT", "SHORT", -8.0, "LONG", 0.9),   # disagree, whale opposed, loss
        ("ETHUSDT", "SHORT", -4.0, "LONG", 0.7),   # disagree, loss
    ]
    for i, (sym, side, pnl, ma, wd) in enumerate(rows):
        _seed_paired(db, (base + timedelta(minutes=i)).isoformat(), sym, side, pnl,
                     model_action=ma, whale_dir=wd)
    with sqlite3.connect(str(db)) as conn:
        res = svr.compute_signal_value(conn, min_closes=4)
    assert res["ready"] and res["n"] == 4
    assert res["model_agree"]["avg_pnl"] == 8.0      # (10+6)/2
    assert res["model_disagree"]["avg_pnl"] == -6.0  # (-8-4)/2
    assert res["model_agree"]["win_rate"] == 1.0
    assert res["model_disagree"]["win_rate"] == 0.0


def test_signal_value_report_not_ready_below_threshold(tmp_path):
    db = _mkdb(tmp_path)
    _seed_paired(db, datetime(2026, 6, 18, tzinfo=UTC).isoformat(), "BTCUSDT",
                 "LONG", 5.0, model_action="LONG", whale_dir=0.9)
    with sqlite3.connect(str(db)) as conn:
        res = svr.compute_signal_value(conn, min_closes=100)
    assert res["ready"] is False and res["n"] == 1


# ── reviewer entry-signal pattern section ───────────────────────────────


def test_entry_signal_patterns_section(tmp_path):
    db = _mkdb(tmp_path)
    base = datetime(2026, 6, 18, 12, 0, tzinfo=UTC)
    # 3 losers, all with model disagreement at entry.
    for i in range(3):
        _seed_paired(db, (base + timedelta(minutes=i)).isoformat(), "BTCUSDT",
                     "SHORT", -5.0, model_action="LONG", whale_dir=0.9)
    with sqlite3.connect(str(db)) as conn:
        section = entry_signal_patterns_section(
            conn, since=base - timedelta(hours=1), until=base + timedelta(hours=1))
    assert "Entry-signal patterns" in section
    assert "3" in section and "DISAGREE" in section.upper()


def test_entry_signal_patterns_empty_without_schema_data(tmp_path):
    db = _mkdb(tmp_path)
    base = datetime(2026, 6, 18, tzinfo=UTC)
    with sqlite3.connect(str(db)) as conn:
        assert entry_signal_patterns_section(
            conn, since=base, until=base + timedelta(days=1)) == ""
