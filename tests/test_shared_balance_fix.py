"""Regression test for the shared-wallet balance bug.

Bug: each per-symbol bot tracked self.balance independently and the
saved total_balance was `self.balance + sum(asset.pnl)`. With 4 symbols
sharing one Binance wallet this produced inconsistent total_balance
values that depended on which bot saved last.

Fix (live_trading_htf.py:_update_shared_state, 2026-05-03):
    total_balance = self._get_real_balance()  # exchange-sourced
    fallback = self.balance + sum(asset.pnl)  # only if exchange fails

Tests assert:
  * happy path: total_balance comes from exchange
  * fallback: if _get_real_balance returns 0/None, legacy compute kicks in
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _build_bot(real_balance, self_balance=4500.0, realized_pnl=12.34, position=0):
    """Build a minimal HTFLiveBot stand-in that exercises _save_state."""
    import live_trading_htf as m
    bot = MagicMock()  # not spec=… — _save_state pokes lots of attrs
    bot.symbol = "BTCUSDT"
    bot.position = position
    bot.position_price = 60000.0
    bot.position_units = 0.0
    bot.balance = self_balance
    bot.realized_pnl = realized_pnl
    bot.sl_price = 0.0
    bot.tp_price = 0.0
    bot.peak_price = 0.0
    bot.last_loss_time = 0.0
    bot.last_entry_time = 0.0
    bot.start_time = "2026-05-03T00:00:00"
    bot.session_balance = self_balance
    bot.mfe_pct = 0.0
    bot.mae_pct = 0.0
    bot.sl_pct = 0.0
    bot.partial_tp_level = 0
    bot.initial_position_units = 0.0
    bot.partial_tp1_price = 0.0
    bot.partial_tp2_price = 0.0
    bot.position_entry_time = 0.0
    bot.last_close_direction = 0
    bot.last_close_pnl = 0.0
    bot.last_close_time = 0.0
    bot._state_file = Path("/tmp/bogus_state_for_test.json")
    bot._get_real_balance = MagicMock(return_value=real_balance)

    # Storage: capture saved state for inspection
    saved = {}
    storage = MagicMock()
    storage.load_state = MagicMock(return_value={"assets": {
        "ETHUSDT": {"position": 0, "pnl": 5.02},
        "SOLUSDT": {"position": 0, "pnl": 272.74},
        "XRPUSDT": {"position": 0, "pnl": -55.42},
    }})
    def _capture(state):
        saved.update(state)
    storage.save_state = MagicMock(side_effect=_capture)
    bot.storage = storage

    return bot, saved


def test_total_balance_uses_exchange_when_available(tmp_path):
    import live_trading_htf as m
    bot, saved = _build_bot(real_balance=4941.09, self_balance=4500.0,
                              realized_pnl=-133.49)
    bot._state_file = tmp_path / "bogus.json"
    # Call the real method on the mock — bind it so `self` resolves
    m.HTFLiveBot._save_state(bot)
    assert "total_balance" in saved
    # Must be exchange-sourced, not the buggy `self_balance + sum(pnl)`
    assert saved["total_balance"] == pytest.approx(4941.09)


def test_total_balance_falls_back_when_exchange_returns_zero(tmp_path):
    import live_trading_htf as m
    bot, saved = _build_bot(real_balance=0.0, self_balance=5000.0,
                              realized_pnl=-133.49)
    bot._state_file = tmp_path / "bogus.json"
    m.HTFLiveBot._save_state(bot)
    # Legacy path: self.balance + sum(asset pnl)
    # = 5000 + (-133.49 + 5.02 + 272.74 + -55.42)
    # The current bot's pnl is realized_pnl=-133.49 and that gets put into assets[BTCUSDT].pnl
    expected = 5000.0 + (-133.49 + 5.02 + 272.74 + -55.42)
    assert saved["total_balance"] == pytest.approx(expected, abs=0.01)


def test_total_balance_falls_back_when_exchange_returns_none(tmp_path):
    import live_trading_htf as m
    bot, saved = _build_bot(real_balance=None, self_balance=5000.0,
                              realized_pnl=-133.49)
    bot._state_file = tmp_path / "bogus.json"
    m.HTFLiveBot._save_state(bot)
    expected = 5000.0 + (-133.49 + 5.02 + 272.74 + -55.42)
    assert saved["total_balance"] == pytest.approx(expected, abs=0.01)


def test_total_balance_falls_back_when_get_real_balance_raises(tmp_path):
    import live_trading_htf as m
    bot, saved = _build_bot(real_balance=4941.09, self_balance=5000.0,
                              realized_pnl=-133.49)
    bot._get_real_balance = MagicMock(side_effect=RuntimeError("API down"))
    bot._state_file = tmp_path / "bogus.json"
    m.HTFLiveBot._save_state(bot)
    # Should still write something, falling back to legacy
    expected = 5000.0 + (-133.49 + 5.02 + 272.74 + -55.42)
    assert saved["total_balance"] == pytest.approx(expected, abs=0.01)
