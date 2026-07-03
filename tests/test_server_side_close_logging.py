"""Tests for the server-side close logger (SOL opens-without-closes fix, 2026-07-03).

When a position closes on the exchange (SL/TP) between bot iterations, the bot
must log the close with the REAL fill PnL instead of silently dropping it.
"""

import types

from live_trading_htf import HTFLiveBot


class _FakeConn:
    def __init__(self, fills):
        self._fills = fills

    def get_trade_history(self, symbol, limit=200):
        return self._fills


class _FakeExec:
    def __init__(self, fills):
        self.connector = _FakeConn(fills)


def _fake_self(position=-1, entry=100.0, units=10.0, entry_time=1000.0):
    s = types.SimpleNamespace(
        symbol="SOLUSDT", position=position, position_price=entry,
        position_units=units, position_entry_time=entry_time,
        realized_pnl=0.0, last_close_pnl=0.0, last_loss_time=0.0,
        last_close_direction=0, last_close_time=0.0, logged=[],
    )
    s._log_trade = s.logged.append
    return s


def test_short_close_logs_real_pnl_and_tp():
    # short from 100, bought back at 95 -> profit; entry_ms = 1000*1000
    fills = [
        {"side": "SELL", "price": "100", "qty": "10", "realizedPnl": "0",   "commission": "0.4", "time": 1_000_001},
        {"side": "BUY",  "price": "95",  "qty": "10", "realizedPnl": "5.0", "commission": "0.5", "time": 2_000_000},
    ]
    s = _fake_self()
    HTFLiveBot._log_server_side_close(s, _FakeExec(fills))

    assert len(s.logged) == 1
    t = s.logged[0]
    assert t["action"] == "CLOSE_SHORT"
    assert t["reason"] == "SERVER_TP"
    assert abs(t["pnl"] - (5.0 - 0.9)) < 1e-9   # realizedPnl - total commission
    assert abs(t["exit_price"] - 95.0) < 1e-9
    assert t["server_side"] is True  # is_testnet is stamped downstream by _log_trade
    assert abs(s.realized_pnl - 4.1) < 1e-9


def test_long_close_marks_sl_and_loss_time():
    # long from 100, sold at 98.5 -> loss
    fills = [
        {"side": "BUY",  "price": "100",  "qty": "5", "realizedPnl": "0",    "commission": "0.2",  "time": 1_000_001},
        {"side": "SELL", "price": "98.5", "qty": "5", "realizedPnl": "-7.5", "commission": "0.25", "time": 1_500_000},
    ]
    s = _fake_self(position=1, entry=100.0, units=5.0)
    HTFLiveBot._log_server_side_close(s, _FakeExec(fills))

    t = s.logged[0]
    assert t["action"] == "CLOSE_LONG"
    assert t["reason"] == "SERVER_SL"
    assert abs(t["exit_price"] - 98.5) < 1e-9
    assert abs(t["pnl"] - (-7.5 - 0.45)) < 1e-9
    assert s.last_loss_time > 0


def test_close_still_logged_when_fill_fetch_fails():
    class _Boom:
        def get_trade_history(self, *a, **k):
            raise RuntimeError("api down")

    s = _fake_self()
    HTFLiveBot._log_server_side_close(s, types.SimpleNamespace(connector=_Boom()))

    assert len(s.logged) == 1, "close row must never be dropped"
    assert s.logged[0]["pnl"] is None
    assert s.logged[0]["action"] == "CLOSE_SHORT"


def test_fills_before_entry_are_ignored():
    # an older fill (before entry_ms) must not be counted
    fills = [
        {"side": "BUY", "price": "50", "qty": "10", "realizedPnl": "99.0", "commission": "0", "time": 500_000},  # stale
        {"side": "BUY", "price": "95", "qty": "10", "realizedPnl": "5.0",  "commission": "0.5", "time": 2_000_000},
    ]
    s = _fake_self()
    HTFLiveBot._log_server_side_close(s, _FakeExec(fills))
    assert abs(s.logged[0]["pnl"] - (5.0 - 0.5)) < 1e-9   # stale fill excluded
