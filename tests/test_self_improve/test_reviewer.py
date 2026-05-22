"""Reviewer tests — synthetic 24h windows + mocked LLM."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.self_improve.migrate import migrate
from src.self_improve.llm_client import LLMResponse
from src.self_improve.reviewer import (
    WindowSummary,
    load_window,
    render_markdown,
    run_review,
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
                created_at TEXT,
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
    side: str,  # 'LONG' | 'SHORT'
    pnl: float,
    confidence: float = 0.7,
    reason: str = "",
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
            "reason, is_testnet) VALUES (?, ?, ?, '{}', 100.0, ?, ?, ?)",
            (close_ts.isoformat(), symbol, f"CLOSE_{side}", pnl, reason, is_testnet),
        )


def test_load_window_filters_outside(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)

    _add_pair(
        db,
        open_ts=now - timedelta(hours=10),
        close_ts=now - timedelta(hours=8),
        symbol="BTCUSDT", side="LONG", pnl=5.0,
    )
    # Outside the 24h window (closed > 24h ago)
    _add_pair(
        db,
        open_ts=now - timedelta(hours=50),
        close_ts=now - timedelta(hours=48),
        symbol="ETHUSDT", side="LONG", pnl=999.0,
    )

    with sqlite3.connect(str(db)) as conn:
        rows = load_window(
            conn,
            since=now - timedelta(hours=24),
            until=now,
        )
    assert len(rows) == 1
    assert rows[0].symbol == "BTCUSDT"


def test_load_window_only_testnet(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    _add_pair(
        db,
        open_ts=now - timedelta(hours=5),
        close_ts=now - timedelta(hours=4),
        symbol="BTCUSDT", side="LONG", pnl=999.0,
        is_testnet=0,  # mainnet — must not leak
    )
    with sqlite3.connect(str(db)) as conn:
        rows = load_window(conn, since=now - timedelta(hours=24), until=now)
    assert rows == []


def test_summary_overall_metrics(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    for i, pnl in enumerate([10.0, -5.0, 7.0, -3.0]):
        _add_pair(
            db,
            open_ts=now - timedelta(hours=20 - i),
            close_ts=now - timedelta(hours=19 - i),
            symbol="BTCUSDT", side="LONG", pnl=pnl,
        )
    with sqlite3.connect(str(db)) as conn:
        closes = load_window(conn, since=now - timedelta(hours=24), until=now)
    s = WindowSummary(since=now - timedelta(hours=24), until=now, closes=closes)
    o = s.overall()
    assert o["net_pnl_usd"] == pytest.approx(9.0)
    assert o["num_closes"] == 4
    assert o["win_rate"] == pytest.approx(0.5)


def test_trailing_streaks(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    # BTC SHORT: 3 most recent are losses (chronological)
    for hours_ago, pnl in [(20, 5.0), (15, -1.0), (10, -1.0), (5, -1.0)]:
        _add_pair(
            db,
            open_ts=now - timedelta(hours=hours_ago + 1),
            close_ts=now - timedelta(hours=hours_ago),
            symbol="BTCUSDT", side="SHORT", pnl=pnl,
        )
    with sqlite3.connect(str(db)) as conn:
        closes = load_window(conn, since=now - timedelta(hours=24), until=now)
    s = WindowSummary(since=now - timedelta(hours=24), until=now, closes=closes)
    streaks = s.trailing_streaks()
    assert streaks.get(("BTCUSDT", "SHORT")) == 3


def test_render_markdown_has_required_sections(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    _add_pair(
        db,
        open_ts=now - timedelta(hours=5),
        close_ts=now - timedelta(hours=4),
        symbol="BTCUSDT", side="LONG", pnl=10.0, reason="TP",
    )
    with sqlite3.connect(str(db)) as conn:
        closes = load_window(conn, since=now - timedelta(hours=24), until=now)
    s = WindowSummary(since=now - timedelta(hours=24), until=now, closes=closes)
    md = render_markdown(s)
    for marker in ("# Post-Mortem", "## Overall", "## By symbol", "## By side", "## By exit reason"):
        assert marker in md


# ─────────────────────────────────────────────────────────────────────────
# End-to-end run_review with mocked client
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _StubClient:
    text: str = "**No patterns** No high-confidence next step."
    has_api_key: bool = True

    def call(self, *, ctx, model, system, user, max_tokens=1024, **kw) -> LLMResponse:
        return LLMResponse(
            text=self.text,
            model=model,
            input_tokens=10,
            output_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=0.005,
            duration_s=0.1,
            degraded=False,
        )


def test_run_review_writes_markdown(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    out_dir = tmp_path / "post-mortems"
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)

    _add_pair(
        db,
        open_ts=now - timedelta(hours=5),
        close_ts=now - timedelta(hours=4),
        symbol="BTCUSDT", side="LONG", pnl=10.0,
    )

    result = run_review(
        db_path=db,
        out_dir=out_dir,
        window_hours=24,
        now=now,
        client=_StubClient(),
        enable_llm=True,
    )
    assert result.markdown_path.exists()
    text = result.markdown_path.read_text()
    assert "# Post-Mortem" in text
    assert "Pattern analysis (LLM)" in text


def test_run_review_handles_empty_window(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    out_dir = tmp_path / "post-mortems"
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)

    result = run_review(
        db_path=db,
        out_dir=out_dir,
        window_hours=24,
        now=now,
        enable_llm=False,
    )
    assert result.markdown_path.exists()
    assert result.summary.n == 0
    assert result.telegram_digest is not None


def test_run_review_telegram_digest_flags_streaks(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    _seed_db(db)
    out_dir = tmp_path / "post-mortems"
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    for hours_ago in (8, 6, 4, 2):
        _add_pair(
            db,
            open_ts=now - timedelta(hours=hours_ago + 1),
            close_ts=now - timedelta(hours=hours_ago),
            symbol="ETHUSDT", side="LONG", pnl=-5.0,
        )

    result = run_review(
        db_path=db,
        out_dir=out_dir,
        window_hours=24,
        now=now,
        enable_llm=False,
    )
    assert "⚠" in (result.telegram_digest or "")
    assert "ETHUSDT" in (result.telegram_digest or "")
