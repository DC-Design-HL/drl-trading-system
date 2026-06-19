"""P4 — canary attribution & honest evaluation (PROFITABILITY_PLAN.md §3/P4).

Covers: experiment_id stamping + suppressed_entries round-trip (storage),
canary_eval.classify_change, evaluate_canary's promote/extend/reject matrix
for both suppression and envelope changes (sim functions monkeypatched so the
decision logic is tested deterministically), and the reviewer's
experiment-attribution section.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from scripts.self_improve.migrate import migrate
from src.data.storage import SQLiteStorage
from src.self_improve import canary_eval as ce

UTC = timezone.utc
_CL = ("CLOSE_LONG", "CLOSE_SHORT", "SL_HIT", "TP_HIT")


def _mkdb(tmp_path):
    """A DB with the bot's trades+suppressed_entries schema AND the
    self_improve experiments/decisions schema."""
    db = tmp_path / "trading.db"
    SQLiteStorage(str(db))           # creates trades + suppressed_entries
    with sqlite3.connect(str(db)) as conn:
        migrate(conn)                # adds experiments + decisions + ...
    return db


# ── Storage: stamping + suppression round-trip ──────────────────────────


def test_trade_stamping_and_suppression_roundtrip(tmp_path):
    db = _mkdb(tmp_path)
    s = SQLiteStorage(str(db))
    s.log_trade({"timestamp": "t", "symbol": "ETHUSDT", "action": "CLOSE_SHORT",
                 "pnl": -5.0, "is_testnet": True, "experiment_id": 14})
    s.log_trade({"timestamp": "t", "symbol": "BTCUSDT", "action": "CLOSE_LONG",
                 "pnl": 3.0, "is_testnet": True})  # no exp → NULL
    s.log_suppressed_entry(symbol="ETHUSDT", side="SHORT", confidence=0.52,
                           gate="struct_floor", experiment_id=14)
    with sqlite3.connect(str(db)) as c:
        assert c.execute(
            "SELECT experiment_id FROM trades WHERE symbol='ETHUSDT'").fetchone()[0] == 14
        assert c.execute(
            "SELECT experiment_id FROM trades WHERE symbol='BTCUSDT'").fetchone()[0] is None
        row = c.execute(
            "SELECT symbol,side,confidence,gate,experiment_id FROM suppressed_entries"
        ).fetchone()
        assert row == ("ETHUSDT", "SHORT", 0.52, "struct_floor", 14)


# ── classify_change ─────────────────────────────────────────────────────


def test_classify_change():
    assert ce.classify_change({"STRUCT_SYMBOL_DIRECTIONAL_CONF": {"ETHUSDT": {"SHORT": 0.6}}}) == "suppression"
    assert ce.classify_change({"SYMBOL_SIDE_BLOCKLIST_ADD": [["BTCUSDT", "SHORT"]]}) == "suppression"
    assert ce.classify_change({"TRAILING_DISTANCE_PCT": 0.008}) == "envelope"
    assert ce.classify_change({}) == "unknown"


# ── evaluate_canary: suppression matrix ─────────────────────────────────


def _seed_suppressed(db, exp_id, n, since):
    with sqlite3.connect(str(db)) as c:
        for i in range(n):
            c.execute(
                "INSERT INTO suppressed_entries (ts,symbol,side,confidence,gate,experiment_id) "
                "VALUES (?,?,?,?,?,?)",
                ((since + timedelta(minutes=i)).isoformat(), "ETHUSDT", "SHORT",
                 0.5, "struct_floor", exp_id),
            )
        c.commit()


def test_suppression_promote_when_blocked_trades_would_lose(tmp_path, monkeypatch):
    db = _mkdb(tmp_path)
    since = datetime(2026, 6, 18, tzinfo=UTC)
    _seed_suppressed(db, 14, 6, since)
    # Blocked trades would have lost $40 net → suppression earns its keep.
    monkeypatch.setattr(ce, "_simulate_suppressed", lambda rows, **k: (-40.0, 6))
    with sqlite3.connect(str(db)) as conn:
        v = ce.evaluate_canary(
            conn, experiment_id=14,
            config_changes={"STRUCT_SYMBOL_DIRECTIONAL_CONF": {"ETHUSDT": {"SHORT": 0.6}}},
            since_iso=since.isoformat(), capital=5000.0,
            now=since + timedelta(days=2))
    assert v.decision == "promote" and v.change_type == "suppression"
    assert v.avoided_pnl == -40.0


def test_suppression_reject_when_blocked_trades_would_win(tmp_path, monkeypatch):
    db = _mkdb(tmp_path)
    since = datetime(2026, 6, 18, tzinfo=UTC)
    _seed_suppressed(db, 14, 6, since)
    monkeypatch.setattr(ce, "_simulate_suppressed", lambda rows, **k: (+55.0, 6))
    with sqlite3.connect(str(db)) as conn:
        v = ce.evaluate_canary(
            conn, experiment_id=14, config_changes={"STRUCT_MIN_CONFIDENCE": 0.6},
            since_iso=since.isoformat(), capital=5000.0,
            now=since + timedelta(days=2))
    assert v.decision == "reject"


def test_suppression_extend_then_reject_on_low_samples(tmp_path, monkeypatch):
    db = _mkdb(tmp_path)
    since = datetime(2026, 6, 18, tzinfo=UTC)
    monkeypatch.setattr(ce, "_simulate_suppressed", lambda rows, **k: (-10.0, 2))  # < MIN_SAMPLES
    cc = {"STRUCT_MIN_CONFIDENCE": 0.6}
    with sqlite3.connect(str(db)) as conn:
        within = ce.evaluate_canary(conn, experiment_id=14, config_changes=cc,
                                    since_iso=since.isoformat(), capital=5000.0,
                                    now=since + timedelta(days=2))
        capped = ce.evaluate_canary(conn, experiment_id=14, config_changes=cc,
                                    since_iso=since.isoformat(), capital=5000.0,
                                    now=since + timedelta(days=8))
    assert within.decision == "extend"
    assert capped.decision == "reject"


# ── evaluate_canary: envelope matrix ────────────────────────────────────


def _seed_stamped_closes(db, exp_id, pnls, since):
    s = SQLiteStorage(str(db))
    for i, p in enumerate(pnls):
        s.log_trade({"timestamp": (since + timedelta(minutes=i)).isoformat(),
                     "symbol": "BTCUSDT", "action": "CLOSE_LONG", "pnl": p,
                     "is_testnet": True, "experiment_id": exp_id})


def test_envelope_promote_when_beats_baseline(tmp_path, monkeypatch):
    db = _mkdb(tmp_path)
    since = datetime(2026, 6, 18, tzinfo=UTC)
    _seed_stamped_closes(db, 20, [10, -5, 8, 12, -3, 6], since)  # realized +28
    monkeypatch.setattr(ce, "_forward_baseline_pnl", lambda **k: 15.0)
    with sqlite3.connect(str(db)) as conn:
        v = ce.evaluate_canary(conn, experiment_id=20,
                               config_changes={"TRAILING_DISTANCE_PCT": 0.008},
                               since_iso=since.isoformat(), capital=5000.0,
                               now=since + timedelta(days=2))
    assert v.decision == "promote" and v.change_type == "envelope"
    assert v.realized_pnl == 28.0 and v.baseline_pnl == 15.0


def test_envelope_reject_when_underperforms_baseline(tmp_path, monkeypatch):
    db = _mkdb(tmp_path)
    since = datetime(2026, 6, 18, tzinfo=UTC)
    _seed_stamped_closes(db, 20, [1, -5, 2, -8, 1, -2], since)  # realized -11
    monkeypatch.setattr(ce, "_forward_baseline_pnl", lambda **k: 20.0)
    with sqlite3.connect(str(db)) as conn:
        v = ce.evaluate_canary(conn, experiment_id=20,
                               config_changes={"STAGNANT_HOURS": 9.0},
                               since_iso=since.isoformat(), capital=5000.0,
                               now=since + timedelta(days=2))
    assert v.decision == "reject"


def test_envelope_extend_on_few_stamped_closes(tmp_path, monkeypatch):
    db = _mkdb(tmp_path)
    since = datetime(2026, 6, 18, tzinfo=UTC)
    _seed_stamped_closes(db, 20, [5, -2], since)  # only 2 closes < MIN_SAMPLES
    # baseline shouldn't even be consulted; guard anyway
    monkeypatch.setattr(ce, "_forward_baseline_pnl", lambda **k: 0.0)
    with sqlite3.connect(str(db)) as conn:
        v = ce.evaluate_canary(conn, experiment_id=20,
                               config_changes={"TRAILING_DISTANCE_PCT": 0.008},
                               since_iso=since.isoformat(), capital=5000.0,
                               now=since + timedelta(days=2))
    assert v.decision == "extend"


# ── reviewer attribution section ────────────────────────────────────────


def test_experiment_attribution_section_renders(tmp_path):
    from src.self_improve.reviewer import experiment_attribution_section
    db = _mkdb(tmp_path)
    since = datetime(2026, 6, 18, tzinfo=UTC)
    now = since + timedelta(days=1)
    with sqlite3.connect(str(db)) as conn:
        conn.execute("INSERT INTO experiments(id,ts_created,proposal,stage) VALUES (14,?,?, 'canary')",
                     (since.isoformat(), "ETH SHORT floor"))
        conn.execute(
            "INSERT INTO suppressed_entries (ts,symbol,side,confidence,gate,experiment_id) "
            "VALUES (?, 'ETHUSDT','SHORT',0.5,'struct_floor',14)",
            ((since + timedelta(hours=1)).isoformat(),))
        conn.commit()
        SQLiteStorage(str(db)).log_trade({
            "timestamp": (since + timedelta(hours=2)).isoformat(), "symbol": "ETHUSDT",
            "action": "CLOSE_SHORT", "pnl": -4.0, "is_testnet": True, "experiment_id": 14})
        section = experiment_attribution_section(conn, since=since, until=now)
    assert "Active-experiment attribution" in section
    assert "#14" in section and "canary" in section


def test_attribution_section_empty_when_no_experiments(tmp_path):
    from src.self_improve.reviewer import experiment_attribution_section
    db = _mkdb(tmp_path)
    since = datetime(2026, 6, 18, tzinfo=UTC)
    with sqlite3.connect(str(db)) as conn:
        assert experiment_attribution_section(
            conn, since=since, until=since + timedelta(days=1)) == ""
