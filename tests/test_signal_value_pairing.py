"""Regression test for the P5 signal-value OPEN->outcome pairing fix (2026-07-03).

The old FIFO pairing drifted under the live trade log's OPEN/CLOSE imbalance
(re-affirmation opens + server-side closes), pairing only ~4/68 snapshots.
The fix attributes the next full CLOSE's PnL to every unresolved OPEN before it.
"""

import sqlite3

from scripts.self_improve.signal_value_report import _pnl_by_open


def _db(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT, "
        "action TEXT, pnl REAL, is_testnet INTEGER)"
    )
    conn.executemany(
        "INSERT INTO trades (timestamp, symbol, action, pnl, is_testnet) "
        "VALUES (?,?,?,?,1)",
        rows,
    )
    return conn


def test_reaffirmation_opens_all_pair_to_next_close():
    # 3 back-to-back opens (re-affirmations), then one close at +5.0
    conn = _db([
        ("t1", "SOLUSDT", "OPEN_SHORT", None),
        ("t2", "SOLUSDT", "OPEN_SHORT", None),
        ("t3", "SOLUSDT", "OPEN_SHORT", None),
        ("t4", "SOLUSDT", "CLOSE_SHORT", 5.0),
    ])
    out = _pnl_by_open(conn)
    assert out[("SOLUSDT", "t1")] == 5.0
    assert out[("SOLUSDT", "t2")] == 5.0
    assert out[("SOLUSDT", "t3")] == 5.0


def test_partial_close_keeps_position_open():
    conn = _db([
        ("t1", "BTCUSDT", "OPEN_LONG", None),
        ("t2", "BTCUSDT", "PARTIAL_CLOSE_LONG", 2.0),  # position stays open
        ("t3", "BTCUSDT", "CLOSE_LONG", 7.0),
    ])
    out = _pnl_by_open(conn)
    # the entry maps to the FULL close's pnl, partial does not resolve it
    assert out[("BTCUSDT", "t1")] == 7.0


def test_open_without_logged_close_does_not_pair():
    # server-side close never logged -> open stays unresolved (SOL logging gap)
    conn = _db([
        ("t1", "SOLUSDT", "OPEN_SHORT", None),
        ("t2", "SOLUSDT", "OPEN_SHORT", None),
    ])
    out = _pnl_by_open(conn)
    assert out == {}


def test_symbols_are_independent():
    conn = _db([
        ("t1", "BTCUSDT", "OPEN_LONG", None),
        ("t2", "ETHUSDT", "OPEN_SHORT", None),
        ("t3", "BTCUSDT", "CLOSE_LONG", 3.0),
        ("t4", "ETHUSDT", "CLOSE_SHORT", -1.0),
    ])
    out = _pnl_by_open(conn)
    assert out[("BTCUSDT", "t1")] == 3.0
    assert out[("ETHUSDT", "t2")] == -1.0


def test_second_position_pairs_to_its_own_close():
    conn = _db([
        ("t1", "BTCUSDT", "OPEN_LONG", None),
        ("t2", "BTCUSDT", "CLOSE_LONG", 4.0),
        ("t3", "BTCUSDT", "OPEN_SHORT", None),
        ("t4", "BTCUSDT", "CLOSE_SHORT", -2.0),
    ])
    out = _pnl_by_open(conn)
    assert out[("BTCUSDT", "t1")] == 4.0
    assert out[("BTCUSDT", "t3")] == -2.0
