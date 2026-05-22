"""Tests for the display-only whale + news signals added to trade alerts.

These signals are STATISTICS ONLY — the bot's decision path is not affected.
Purpose of the tests:
  * whale basic heuristic correctly aggregates tail of jsonl + respects window
  * news signal correctly filters by asset + 'ALL' tag and weights by confidence
  * format_whale_news_lines always returns both blocks (whale + news)
"""
import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import trade_alerter as ta


# ──────────────────────────────────────────────────────────────────────────
# _basic_whale_flow_signal
# ──────────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, events: list):
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_whale_flow_bullish_when_exchange_out_dominates(tmp_path, monkeypatch):
    """ETH leaving exchange wallets = buyers accumulating = LONG signal."""
    whale_dir = tmp_path / "eth"
    whale_dir.mkdir()
    now = time.time()
    _write_jsonl(whale_dir / "binance_hot_wallet.jsonl", [
        {"timestamp": now - 300, "action": "LARGE_TRANSFER_OUT", "value_eth": 200, "direction": "out"},
        {"timestamp": now - 200, "action": "LARGE_TRANSFER_OUT", "value_eth": 150, "direction": "out"},
        {"timestamp": now - 100, "action": "LARGE_TRANSFER_IN",  "value_eth": 50,  "direction": "in"},
    ])
    monkeypatch.setattr(ta, "_WHALE_DATA_DIR", whale_dir)
    r = ta._basic_whale_flow_signal(window_minutes=30)
    assert r["direction"] == "LONG"
    assert r["bullish_eth"] == pytest.approx(350.0)
    assert r["bearish_eth"] == pytest.approx(50.0)
    assert r["tx_count"] == 3
    assert r["confidence"] > 0.7  # strong skew


def test_whale_flow_bearish_when_exchange_in_dominates(tmp_path, monkeypatch):
    whale_dir = tmp_path / "eth"
    whale_dir.mkdir()
    now = time.time()
    _write_jsonl(whale_dir / "coinbase_institutional.jsonl", [
        {"timestamp": now - 120, "action": "LARGE_TRANSFER_IN", "value_eth": 500, "direction": "in"},
        {"timestamp": now - 60,  "action": "LARGE_TRANSFER_IN", "value_eth": 300, "direction": "in"},
        {"timestamp": now - 30,  "action": "LARGE_TRANSFER_OUT", "value_eth": 50, "direction": "out"},
    ])
    monkeypatch.setattr(ta, "_WHALE_DATA_DIR", whale_dir)
    r = ta._basic_whale_flow_signal(window_minutes=30)
    assert r["direction"] == "SHORT"
    assert r["bearish_eth"] == pytest.approx(800.0)
    assert r["bullish_eth"] == pytest.approx(50.0)


def test_whale_flow_window_excludes_old_events(tmp_path, monkeypatch):
    whale_dir = tmp_path / "eth"
    whale_dir.mkdir()
    now = time.time()
    _write_jsonl(whale_dir / "binance_hot_wallet.jsonl", [
        {"timestamp": now - 3600*24, "action": "LARGE_TRANSFER_OUT", "value_eth": 10000, "direction": "out"},
        {"timestamp": now - 120,     "action": "LARGE_TRANSFER_IN",  "value_eth": 50,    "direction": "in"},
    ])
    monkeypatch.setattr(ta, "_WHALE_DATA_DIR", whale_dir)
    r = ta._basic_whale_flow_signal(window_minutes=30)
    # Only the recent IN event counts
    assert r["tx_count"] == 1
    assert r["bullish_eth"] == 0
    assert r["bearish_eth"] == 50


def test_whale_flow_ignores_dust_and_contract_calls(tmp_path, monkeypatch):
    whale_dir = tmp_path / "eth"
    whale_dir.mkdir()
    now = time.time()
    _write_jsonl(whale_dir / "binance_hot_wallet.jsonl", [
        {"timestamp": now - 60, "action": "LARGE_TRANSFER_IN",  "value_eth": 1e-18, "direction": "in"},
        {"timestamp": now - 50, "action": "CONTRACT_CALL",      "value_eth": 1e-18, "direction": "out"},
        {"timestamp": now - 40, "action": "CONTRACT_CALL",      "value_eth": 5,     "direction": "in"},  # still skipped
    ])
    monkeypatch.setattr(ta, "_WHALE_DATA_DIR", whale_dir)
    r = ta._basic_whale_flow_signal(window_minutes=30)
    assert r["tx_count"] == 0
    assert r["direction"] == "NEUTRAL"


def test_whale_flow_handles_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ta, "_WHALE_DATA_DIR", tmp_path / "does_not_exist")
    r = ta._basic_whale_flow_signal(window_minutes=30)
    assert r["direction"] == "NEUTRAL"
    assert r["tx_count"] == 0


def test_whale_flow_known_whale_direction_inverted(tmp_path, monkeypatch):
    """For known whale wallets (not exchanges), IN=bullish (whale accumulated)
    and OUT=bearish (whale sold)."""
    whale_dir = tmp_path / "eth"
    whale_dir.mkdir()
    now = time.time()
    _write_jsonl(whale_dir / "jump_trading.jsonl", [
        {"timestamp": now - 120, "action": "LARGE_TRANSFER_IN",  "value_eth": 100, "direction": "in"},
        {"timestamp": now - 60,  "action": "LARGE_TRANSFER_OUT", "value_eth": 20,  "direction": "out"},
    ])
    monkeypatch.setattr(ta, "_WHALE_DATA_DIR", whale_dir)
    r = ta._basic_whale_flow_signal(window_minutes=30)
    assert r["direction"] == "LONG"
    assert r["bullish_eth"] == 100  # IN to whale
    assert r["bearish_eth"] == 20   # OUT from whale


# ──────────────────────────────────────────────────────────────────────────
# _basic_news_signal
# ──────────────────────────────────────────────────────────────────────────

def _seed_news_db(tmp_path: Path, rows: list) -> Path:
    db = tmp_path / "trading.db"
    con = sqlite3.connect(str(db))
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE news_events (id INTEGER PRIMARY KEY, sentiment_score REAL, "
        "confidence REAL, assets TEXT, created_at TEXT)"
    )
    for r in rows:
        cur.execute(
            "INSERT INTO news_events (sentiment_score, confidence, assets, created_at) "
            "VALUES (?, ?, ?, ?)", r)
    con.commit()
    con.close()
    return db


def test_news_signal_bearish_majority(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    db = _seed_news_db(tmp_path, [
        (-0.8, 0.9, '["BTC"]',       now),
        (-0.6, 0.8, '["ALL", "BTC"]', now),
        (+0.2, 0.5, '["BTC"]',       now),
    ])
    monkeypatch.chdir(tmp_path)
    Path(tmp_path / "data").mkdir()
    (tmp_path / "data" / "trading.db").symlink_to(db)
    r = ta._basic_news_signal("BTCUSDT", window_minutes=60)
    assert r["event_count"] == 3
    assert r["direction"] == "SHORT"
    assert r["weighted_score"] < -0.2


def test_news_signal_filters_by_asset(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    db = _seed_news_db(tmp_path, [
        (-0.9, 0.9, '["ETH"]', now),   # should be EXCLUDED for BTC query
        (+0.5, 0.7, '["BTC"]', now),
        (+0.4, 0.6, '["ALL"]', now),   # market-wide, included
    ])
    monkeypatch.chdir(tmp_path)
    Path(tmp_path / "data").mkdir()
    (tmp_path / "data" / "trading.db").symlink_to(db)
    r = ta._basic_news_signal("BTCUSDT", window_minutes=60)
    assert r["event_count"] == 2  # the ETH one is excluded
    assert r["direction"] == "LONG"


def test_news_signal_excludes_old_events(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    recent = now.isoformat()
    ancient = (now - timedelta(days=2)).isoformat()
    db = _seed_news_db(tmp_path, [
        (-0.9, 0.9, '["BTC"]', ancient),   # outside window
        (+0.5, 0.7, '["BTC"]', recent),
    ])
    monkeypatch.chdir(tmp_path)
    Path(tmp_path / "data").mkdir()
    (tmp_path / "data" / "trading.db").symlink_to(db)
    r = ta._basic_news_signal("BTCUSDT", window_minutes=60)
    assert r["event_count"] == 1
    assert r["direction"] == "LONG"


def test_news_signal_no_events_returns_neutral(tmp_path, monkeypatch):
    db = _seed_news_db(tmp_path, [])
    monkeypatch.chdir(tmp_path)
    Path(tmp_path / "data").mkdir()
    (tmp_path / "data" / "trading.db").symlink_to(db)
    r = ta._basic_news_signal("BTCUSDT", window_minutes=60)
    assert r["event_count"] == 0
    assert r["direction"] == "NEUTRAL"
    assert r["confidence"] == 0.0


def test_news_signal_handles_missing_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no data/trading.db exists here
    r = ta._basic_news_signal("BTCUSDT", window_minutes=60)
    assert r["direction"] == "NEUTRAL"
    assert r["event_count"] == 0


# ──────────────────────────────────────────────────────────────────────────
# _format_whale_news_lines — the thing the alerter actually calls
# ──────────────────────────────────────────────────────────────────────────

def test_format_always_returns_both_blocks(monkeypatch):
    """The formatter must always emit both a whale line and a news line,
    even when data is missing — this is what makes the alert stream
    analyzable later (every trade has a signal row to compare against)."""
    # Point to nonexistent data so both paths fall to NEUTRAL
    monkeypatch.setattr(ta, "_WHALE_DATA_DIR", Path("/tmp/does_not_exist_whale"))
    monkeypatch.setattr(ta, "_get_whale_signal", lambda: {})
    lines = ta._format_whale_news_lines("BTCUSDT")
    # At minimum: 1 whale headline + 1 news headline (plus possibly detail lines)
    whale_line = [l for l in lines if "Whale" in l]
    news_line = [l for l in lines if "News" in l]
    assert len(whale_line) == 1
    assert len(news_line) == 1
    assert "BTC" in news_line[0] or "BTCUSDT" in news_line[0]
