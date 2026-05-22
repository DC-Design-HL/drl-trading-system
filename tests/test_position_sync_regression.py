"""Regression tests for the PnL-inflation bug fix in _sync_position_from_exchange.

Bug (2026-04-23): _sync_position_from_exchange unconditionally overwrote
self.position_units with exchange-reported size. When a partial-TP order
hadn't yet settled on the exchange, this re-added the partially-closed units,
causing the eventual full-close pnl to double-count.

Fix: the sync now (a) refuses upward corrections, (b) skips units sync for a
60s grace window after any partial close.
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import live_trading_htf as m


def _fake_exchange_position(symbol: str, units: float, side: int = -1) -> dict:
    """Build a fake Binance Futures positionRisk entry."""
    return {
        "symbol": symbol,
        "positionAmt": str(units if side == 1 else -units),
        "entryPrice": "1.4472",
    }


def _bot_stub(**overrides):
    """Build the minimum stub for _sync_position_from_exchange."""
    defaults = dict(
        symbol="XRPUSDT",
        dry_run=False,
        position=-1,
        position_price=1.4472,
        position_units=725.2,          # already decremented by a partial TP
        sl_price=1.4732,
        tp_price=1.4124,
        peak_price=1.4472,
        partial_tp_level=1,
        partial_tp1_price=0.0,
        partial_tp2_price=0.0,
        mfe_pct=0.0,
        mae_pct=0.0,
        last_partial_close_time=0.0,
    )
    defaults.update(overrides)
    stub = SimpleNamespace(**defaults)
    stub._save_state = MagicMock()
    return stub


def _run_sync(stub, exchange_units: float):
    """Invoke the real _sync_position_from_exchange bound to `stub` with a
    mocked executor that reports `exchange_units` units for XRP."""
    executor = MagicMock()
    executor.connector.get_positions.return_value = [
        _fake_exchange_position("XRPUSDT", exchange_units, side=-1),
    ]
    with patch("src.api.futures_executor.get_futures_executor", return_value=executor):
        m.HTFLiveBot._sync_position_from_exchange(stub)


# ──────────────────────────────────────────────────────────────────────────
# The core fix: no upward correction
# ──────────────────────────────────────────────────────────────────────────

def test_exchange_reports_more_units_does_not_inflate():
    """The canonical bug: partial close decremented bot to 725; exchange still
    reports 1208; old code would overwrite bot to 1208 (re-adding 483 units)
    and cause the next close_position to double-count pnl."""
    stub = _bot_stub(position_units=725.2)
    _run_sync(stub, exchange_units=1208.6)
    assert stub.position_units == pytest.approx(725.2), \
        "bot-tracked units must NOT be inflated by a stale exchange read"
    stub._save_state.assert_not_called()


def test_exchange_reports_fewer_units_accepts_downward_sync():
    """Downward corrections still work — e.g. an SL fired on exchange while
    the bot was mid-tick. Outside the grace window this must still apply."""
    stub = _bot_stub(position_units=1208.6, last_partial_close_time=0.0)
    _run_sync(stub, exchange_units=600.0)
    assert stub.position_units == pytest.approx(600.0), \
        "downward corrections outside grace must still take effect"
    stub._save_state.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────
# Grace-period behaviour
# ──────────────────────────────────────────────────────────────────────────

def test_grace_window_blocks_downward_sync_too():
    """Inside the 60s grace window after a partial close, any units sync is
    suppressed — even downward. Partial-close reduce-only orders frequently
    settle at slightly different sizes and we don't want to chase those."""
    stub = _bot_stub(position_units=725.2, last_partial_close_time=time.time())
    _run_sync(stub, exchange_units=720.0)  # slightly smaller — still inside grace
    assert stub.position_units == pytest.approx(725.2)
    stub._save_state.assert_not_called()


def test_downward_sync_accepted_after_grace_expires():
    stub = _bot_stub(
        position_units=725.2,
        last_partial_close_time=time.time() - 120,  # 2 min ago
    )
    _run_sync(stub, exchange_units=600.0)
    assert stub.position_units == pytest.approx(600.0)
    stub._save_state.assert_called_once()


def test_upward_sync_blocked_inside_and_outside_grace():
    """Upward corrections are NEVER accepted regardless of grace window."""
    for partial_ts in (time.time(), time.time() - 3600):
        stub = _bot_stub(position_units=725.2, last_partial_close_time=partial_ts)
        _run_sync(stub, exchange_units=1500.0)
        assert stub.position_units == pytest.approx(725.2), \
            f"upward sync must be blocked (partial_ts age={time.time()-partial_ts:.0f}s)"
        stub._save_state.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
# Non-regression: other code paths still work
# ──────────────────────────────────────────────────────────────────────────

def test_position_adoption_when_bot_is_flat():
    """Case 2 (bot flat, exchange has position) must still adopt exchange
    state — this branch is unrelated to the partial-TP bug."""
    stub = _bot_stub(position=0, position_price=0.0, position_units=0.0)
    _run_sync(stub, exchange_units=1000.0)
    assert stub.position == -1  # adopted SHORT from exchange
    assert stub.position_units == pytest.approx(1000.0)
    stub._save_state.assert_called()


def test_stale_position_resets_when_exchange_flat():
    """Case 1 (bot has position but exchange is flat) must still reset to flat."""
    stub = _bot_stub(position=-1, position_units=725.2)
    _run_sync(stub, exchange_units=0.0)
    assert stub.position == 0
    assert stub.position_units == 0.0
    stub._save_state.assert_called()


def test_dry_run_skips_entirely():
    stub = _bot_stub(dry_run=True, position_units=725.2)
    _run_sync(stub, exchange_units=1208.6)
    assert stub.position_units == pytest.approx(725.2)
    stub._save_state.assert_not_called()
