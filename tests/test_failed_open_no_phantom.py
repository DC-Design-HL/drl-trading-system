"""Tests for the phantom-open fix (SOL -2019 margin churn, 2026-07-12).

When the exchange REJECTS an open order (e.g. Binance -2019 "Margin is
insufficient"), the bot must NOT record a phantom OPEN position. Before the
fix, _open_position set self.position optimistically and logged the OPEN
regardless of whether the exchange order executed — producing a fake OPEN row
plus a spurious SERVER_CLOSE on the next sync, every iteration (29 phantom SOL
pairs observed since Jul 8).
"""

import types

from live_trading_htf import HTFLiveBot


# ── _mirror_testnet must surface executed=False back to the caller ──────────

class _RejectingExec:
    """Testnet executor whose mirror_trade reports a rejected open."""
    def mirror_trade(self, trade, _):
        return {"executed": False, "error": "Binance Futures API 400: "
                "{'code': -2019, 'msg': 'Margin is insufficient.'}"}


class _OkExec:
    def mirror_trade(self, trade, _):
        return {"executed": True, "order_id": 123, "sl_order_id": 1, "tp_order_id": 2}


def _mirror_self(executor):
    s = types.SimpleNamespace(testnet_executor=executor, symbol="SOLUSDT")
    s._write_system_alert = lambda *a, **k: None
    return s


def test_mirror_testnet_returns_rejected_result():
    s = _mirror_self(_RejectingExec())
    result = HTFLiveBot._mirror_testnet(s, {"action": "OPEN_SHORT", "symbol": "SOLUSDT", "price": 76.0})
    assert result is not None
    assert result.get("executed") is False
    assert "-2019" in result.get("error", "")


def test_mirror_testnet_returns_executed_result():
    s = _mirror_self(_OkExec())
    result = HTFLiveBot._mirror_testnet(s, {"action": "OPEN_SHORT", "symbol": "SOLUSDT", "price": 76.0})
    assert result is not None and result.get("executed") is True


def test_mirror_testnet_no_executor_returns_none():
    s = types.SimpleNamespace(testnet_executor=None)
    assert HTFLiveBot._mirror_testnet(s, {"action": "OPEN_SHORT", "symbol": "SOLUSDT"}) is None


# ── _revert_failed_open must refund the fee and reset to FLAT ────────────────

def _open_state(fee_deducted_balance=1000.0):
    """A bot mid-open: position already set optimistically, fee deducted."""
    return types.SimpleNamespace(
        balance=fee_deducted_balance,
        position=-1, position_price=76.0, position_units=20.0,
        sl_price=77.0, tp_price=74.0, peak_price=76.0,
        mfe_pct=0.1, mae_pct=0.2, partial_tp_level=1,
        partial_tp1_price=75.0, partial_tp2_price=74.5, sl_pct=0.015,
        initial_position_units=20.0, position_entry_time=1234.0,
    )


def test_revert_failed_open_resets_to_flat_and_refunds_fee():
    s = _open_state(fee_deducted_balance=1000.0)
    HTFLiveBot._revert_failed_open(s, fee=1.5)
    # fee refunded
    assert abs(s.balance - 1001.5) < 1e-9
    # fully flat
    assert s.position == 0
    assert s.position_units == 0.0
    assert s.position_price == 0.0
    assert s.sl_price == 0.0 and s.tp_price == 0.0
    assert s.partial_tp_level == 0
    assert s.partial_tp1_price == 0.0 and s.partial_tp2_price == 0.0
    assert s.sl_pct == 0.0
    assert s.initial_position_units == 0.0
    assert s.position_entry_time == 0.0
