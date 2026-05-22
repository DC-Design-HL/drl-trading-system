"""Paper Trader tests — replay-validate gates."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.self_improve.migrate import migrate
from src.self_improve.paper_trader import (
    DD_DAILY_LIMIT_PCT,
    MIN_PAPER_CLOSES,
    evaluate_paper_period,
)

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
    is_testnet: int = 1,
) -> None:
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO trades(timestamp, symbol, action, data, price, "
            "confidence, is_testnet) VALUES (?, ?, ?, '{}', 100.0, ?, ?)",
            (open_ts.isoformat(), symbol, f"OPEN_{side}", confidence, is_testnet),
        )
        conn.execute(
            "INSERT INTO trades(timestamp, symbol, action, data, price, pnl, "
            "is_testnet) VALUES (?, ?, ?, '{}', 100.0, ?, ?)",
            (close_ts.isoformat(), symbol, f"CLOSE_{side}", pnl, is_testnet),
        )


def test_insufficient_closes_fails_gate(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    end = datetime(2026, 5, 17, tzinfo=UTC)

    # Add only 5 closes — below the MIN_PAPER_CLOSES bar
    for i in range(5):
        _add_pair(
            db,
            open_ts=start + timedelta(hours=i),
            close_ts=start + timedelta(hours=i + 1),
            symbol="BTCUSDT", side="LONG", pnl=2.0,
        )

    result = evaluate_paper_period(
        paper_start=start, paper_end=end,
        config_overrides={},
        backtest_sharpe_reference=1.0,
        db_path=str(db),
    )
    assert not result.pass_gate
    assert any("insufficient closes" in r for r in result.reasons)


def test_passes_when_metrics_close_to_backtest(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    end = datetime(2026, 5, 17, tzinfo=UTC)

    # 20 winning closes, spread across days for non-zero Sharpe
    for i in range(20):
        _add_pair(
            db,
            open_ts=start + timedelta(hours=i * 6),
            close_ts=start + timedelta(hours=i * 6 + 1),
            symbol="BTCUSDT", side="LONG",
            pnl=5.0 if i % 2 == 0 else 6.0,
        )

    # Compute Sharpe of this scenario from a no-override backtest, then
    # pass that same value back as the reference — gate must pass.
    from src.self_improve.backtest_harness import BacktestRequest, run_backtest
    probe = run_backtest(BacktestRequest(
        start_date=start.isoformat(), end_date=end.isoformat(),
        config_overrides={}, db_path=str(db), capital_base=5000.0,
        label="probe",
    ))
    actual_sharpe = probe.portfolio_metrics["sharpe"]

    result = evaluate_paper_period(
        paper_start=start, paper_end=end,
        config_overrides={},
        backtest_sharpe_reference=actual_sharpe,
        db_path=str(db),
    )
    assert result.pass_gate, f"expected pass; got reasons: {result.reasons}"


def test_fails_when_pnl_significantly_worse(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    end = datetime(2026, 5, 17, tzinfo=UTC)

    # 20 winning trades (baseline +$100), but with high conf to be picked up
    for i in range(20):
        _add_pair(
            db,
            open_ts=start + timedelta(hours=i * 6),
            close_ts=start + timedelta(hours=i * 6 + 1),
            symbol="BTCUSDT", side="LONG", pnl=5.0,
            confidence=0.4,
        )

    # Candidate filter blocks every trade → net_pnl = 0 (vs baseline +$100)
    # → delta = -$100, which is below the -$50 threshold
    result = evaluate_paper_period(
        paper_start=start, paper_end=end,
        config_overrides={"MIN_CONFIDENCE": 0.99},
        backtest_sharpe_reference=2.0,
        db_path=str(db),
    )
    assert not result.pass_gate


def test_min_paper_closes_constant_matches_plan() -> None:
    # PLAN.md §6 sets ≥15 closes
    assert MIN_PAPER_CLOSES == 15


def test_dd_limit_matches_plan() -> None:
    # PLAN.md §8 — 5% daily loss halt
    assert DD_DAILY_LIMIT_PCT == 5.0
