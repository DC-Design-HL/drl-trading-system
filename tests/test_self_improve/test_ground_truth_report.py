"""End-to-end test for the P0 ground-truth report.

Asserts the report renders all 7 sections on a tmp DB with a small seed —
without depending on the live trading.db. Funding section is run with
network disabled to keep the test offline.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.self_improve import ground_truth_report
from scripts.self_improve.migrate import migrate


UTC = timezone.utc


def _seed_schema(db: Path) -> None:
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


def _add_open(
    db: Path, *, ts: datetime, symbol: str, side: str,
    price: float, units: float, confidence: float,
) -> None:
    action = f"OPEN_{side}"
    data = json.dumps({
        "action": action,
        "symbol": symbol,
        "price": price,
        "units": units,
        "trade_value": price * units,
        "confidence": confidence,
    })
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """INSERT INTO trades(timestamp,symbol,action,data,price,
                pnl,confidence,reason,is_testnet)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ts.isoformat(), symbol, action, data, price,
             None, confidence, None, 1),
        )


def _add_close(
    db: Path, *, ts: datetime, symbol: str, side: str,
    price: float, pnl: float, reason: str,
    mfe_pct: float = 0.0, mae_pct: float = 0.0,
) -> None:
    action = f"CLOSE_{side}"
    data = json.dumps({
        "action": action,
        "symbol": symbol,
        "exit_price": price,
        "pnl": pnl,
        "reason": reason,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
    })
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """INSERT INTO trades(timestamp,symbol,action,data,price,
                pnl,confidence,reason,is_testnet)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ts.isoformat(), symbol, action, data, price,
             pnl, 1.0, reason, 1),
        )


@pytest.fixture
def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "trading.db"
    _seed_schema(db)
    now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
    # 3 trades inside since-reset window
    for i, (sym, side, pnl, reason, conf) in enumerate([
        ("BTCUSDT", "LONG", 50.0, "TP_HIT", 0.85),
        ("ETHUSDT", "SHORT", -20.0, "SL", 0.55),
        ("SOLUSDT", "SHORT", 12.0, "TRAILING", 0.65),
    ]):
        open_ts = now - timedelta(hours=24 + i)
        close_ts = now - timedelta(hours=12 + i)
        _add_open(db, ts=open_ts, symbol=sym, side=side,
                  price=100.0, units=1.0, confidence=conf)
        _add_close(db, ts=close_ts, symbol=sym, side=side,
                   price=110.0, pnl=pnl, reason=reason,
                   mfe_pct=0.01 if pnl > 0 else 0.008,
                   mae_pct=-0.003 if pnl > 0 else -0.02)

    monkeypatch.setattr(ground_truth_report, "DB_PATH", db)
    return db


def test_report_renders_all_sections(seeded_db: Path) -> None:
    report = ground_truth_report.build_report(
        enable_funding=False,
        now=datetime(2026, 6, 11, 13, 0, tzinfo=UTC),
    )
    # Each numbered section header present
    for header in [
        "## 1. Headline metrics",
        "## 2. Exit-reason breakdown",
        "## 3. Confidence calibration",
        "## 4. MFE/MAE analysis",
        "## 5. Guard counterfactuals",
        "## 6. Self-improve audit",
        "## 7. Funding estimate",
    ]:
        assert header in report, f"missing section: {header}"


def test_headline_totals_match_seed(seeded_db: Path) -> None:
    """The 'since-reset' row should show n=3 and net=+42 ($50-$20+$12)."""
    report = ground_truth_report.build_report(
        enable_funding=False,
        now=datetime(2026, 6, 11, 13, 0, tzinfo=UTC),
    )
    # Portfolio table row for since-reset
    lines = [ln for ln in report.split("\n") if ln.startswith("| since-reset |")]
    assert lines, "no since-reset row in portfolio table"
    assert "| 3 |" in lines[0]
    assert "+42.00" in lines[0]


def test_no_funding_flag_skips_network(seeded_db: Path) -> None:
    """With enable_funding=False the section must not try to reach ccxt."""
    report = ground_truth_report.build_report(enable_funding=False)
    assert "Skipped (--no-funding)" in report


def test_exit_reason_buckets_separated(seeded_db: Path) -> None:
    report = ground_truth_report.build_report(enable_funding=False)
    # Each reason appears as a row
    assert "| TP_HIT |" in report
    assert "| SL |" in report
    assert "| TRAILING |" in report


def test_only_testnet_trades_counted(tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard rule: is_testnet=0 rows must not affect the report."""
    db = tmp_path / "trading.db"
    _seed_schema(db)
    now = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)
    # One testnet trade, one (forbidden) mainnet trade — only the first counts
    _add_open(db, ts=now - timedelta(hours=10), symbol="BTCUSDT", side="LONG",
              price=100.0, units=1.0, confidence=0.8)
    _add_close(db, ts=now - timedelta(hours=5), symbol="BTCUSDT", side="LONG",
               price=110.0, pnl=10.0, reason="TP_HIT")
    # Manually insert a non-testnet row that we expect to be ignored
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """INSERT INTO trades(timestamp,symbol,action,data,price,
                pnl,confidence,reason,is_testnet)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ((now - timedelta(hours=4)).isoformat(),
             "BTCUSDT", "CLOSE_LONG", "{}", 100.0, 999.0, 1.0, "MAINNET", 0),
        )
    monkeypatch.setattr(ground_truth_report, "DB_PATH", db)
    report = ground_truth_report.build_report(enable_funding=False, now=now)
    # The +999 mainnet PnL must not appear — only +10
    assert "+999" not in report
    # 7d row should show net 10.00
    seven_d_lines = [ln for ln in report.split("\n") if ln.startswith("| 7d |")]
    assert seven_d_lines
    assert "+10.00" in seven_d_lines[0]
