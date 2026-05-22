"""End-to-end test for the performance monitor against a tmp DB."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.self_improve.migrate import migrate
from scripts.self_improve.performance_monitor import run_monitor


UTC = timezone.utc


def _seed_db(db: Path) -> None:
    """Build a tmp DB with the live schema *and* the self-improve schema."""
    with sqlite3.connect(str(db)) as conn:
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
                created_at TEXT,
                is_testnet INTEGER DEFAULT 0
            )
            """
        )
        migrate(conn)


def _add_trade(db: Path, *, ts: datetime, symbol: str, action: str, pnl: float) -> None:
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """
            INSERT INTO trades(timestamp, symbol, action, data, price,
                               pnl, confidence, reason, is_testnet)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts.isoformat(),
                symbol,
                action,
                "{}",
                100.0,
                pnl,
                0.5,
                "test",
                1,
            ),
        )


def test_run_monitor_writes_snapshots(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    _seed_db(db)

    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    for i in range(5):
        _add_trade(
            db,
            ts=now - timedelta(hours=i + 1),
            symbol="BTCUSDT",
            action="CLOSE_LONG",
            pnl=10.0 if i % 2 == 0 else -5.0,
        )

    summary = run_monitor(db, since="2026-05-01T00:00:00", now=now)

    # 24h + 7d + 30d portfolio + 1 per-symbol (BTC) + 1 heartbeat = 5
    assert summary["snapshots_written"] >= 4
    with sqlite3.connect(str(db)) as conn:
        windows = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT window FROM metrics_snapshots"
            ).fetchall()
        }
    assert {"24h", "7d", "30d", "heartbeat"} <= windows


def test_run_monitor_ignores_non_testnet(tmp_path: Path) -> None:
    """Hard rule: only is_testnet=1 trades feed the metrics."""
    db = tmp_path / "test.db"
    _seed_db(db)

    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    # One testnet trade, one mainnet — only the testnet should count.
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO trades(timestamp, symbol, action, pnl, is_testnet) "
            "VALUES (?, ?, ?, ?, ?)",
            ((now - timedelta(hours=1)).isoformat(), "BTCUSDT", "CLOSE_LONG", 10.0, 1),
        )
        conn.execute(
            "INSERT INTO trades(timestamp, symbol, action, pnl, is_testnet) "
            "VALUES (?, ?, ?, ?, ?)",
            ((now - timedelta(hours=1)).isoformat(), "BTCUSDT", "CLOSE_LONG", 999.0, 0),
        )

    run_monitor(db, since="2026-05-01T00:00:00", now=now)

    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT net_pnl_usd FROM metrics_snapshots WHERE window='24h' AND symbol IS NULL"
        ).fetchone()
    assert row is not None
    assert row[0] == 10.0, (
        f"Mainnet trade leaked into metrics — got pnl={row[0]}, expected 10.0"
    )


def test_run_monitor_fires_triggers(tmp_path: Path) -> None:
    """Construct a losing-streak scenario and confirm T4 fires."""
    db = tmp_path / "test.db"
    _seed_db(db)

    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    # 4 consecutive BTC LONG losses, most recent
    for hours_ago in (8, 6, 4, 2):
        _add_trade(
            db,
            ts=now - timedelta(hours=hours_ago),
            symbol="BTCUSDT",
            action="CLOSE_LONG",
            pnl=-10.0,
        )

    summary = run_monitor(db, since="2026-05-01T00:00:00", now=now)

    hit_ids = {h["id"] for h in summary["triggers_fired"]}
    assert "T4" in hit_ids, f"T4 should have fired, got: {hit_ids}"


def test_heartbeat_row_written(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    _seed_db(db)
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)

    run_monitor(db, since="2026-05-01T00:00:00", now=now)

    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT ts FROM metrics_snapshots WHERE window='heartbeat' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    assert row[0].startswith("2026-05-22T12:00")
