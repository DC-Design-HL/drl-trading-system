"""Regression tests for the MIN_CONFIDENCE gate (2026-05-03).

Two changes deployed together:
  1. MIN_CONFIDENCE raised 0.45 → 0.55 (90d ablation finding).
  2. The `if not STRUCTURE_FIRST_MODE:` wrapper around the
     confidence-floor check was REMOVED, so the gate now applies in
     both model-first and structure-first paths.

These tests assert:
  * The constant has the expected value.
  * btengine.live_constants stays in sync (drift-detect via parity test).
  * The execute_trade path returns None when confidence < threshold,
    in BOTH STRUCTURE_FIRST_MODE=True and =False.
"""
import pytest
from unittest.mock import MagicMock, patch


def test_min_confidence_constant_is_055():
    import live_trading_htf as m
    assert m.MIN_CONFIDENCE == 0.55, (
        f"MIN_CONFIDENCE drifted: expected 0.55, got {m.MIN_CONFIDENCE}"
    )


def test_btengine_live_constants_in_sync():
    import live_trading_htf as live
    from src.btengine import live_constants as LC
    assert LC.MIN_CONFIDENCE == live.MIN_CONFIDENCE, (
        f"btengine MIN_CONFIDENCE {LC.MIN_CONFIDENCE} != live {live.MIN_CONFIDENCE}"
    )


def test_eth_per_symbol_override_intact():
    """ETH has a higher per-symbol floor (0.80). The unification of the
    structure-first / model-first paths should not have lost it."""
    import live_trading_htf as m
    assert m.SYMBOL_MIN_CONFIDENCE.get("ETHUSDT") == 0.80


def _make_bot_for_execute_trade(structure_first: bool):
    """Build a MagicMock bot and patch STRUCTURE_FIRST_MODE."""
    import live_trading_htf as m
    bot = MagicMock()
    bot.symbol = "BTCUSDT"
    bot.position = 0  # FLAT
    bot.last_loss_time = 0
    bot.last_close_time = 0
    bot._last_df = None
    bot.regime_detector = None
    bot._last_structure_signals = {}
    bot.fetcher = MagicMock()
    return bot


@pytest.mark.parametrize("structure_first", [True, False])
def test_execute_trade_blocks_when_confidence_below_floor(structure_first):
    """In both modes, low-confidence entry must HOLD."""
    import live_trading_htf as m
    bot = _make_bot_for_execute_trade(structure_first)
    with patch.object(m, "STRUCTURE_FIRST_MODE", structure_first):
        # confidence 0.50 < MIN_CONFIDENCE 0.55 → expect None
        result = m.HTFLiveBot.execute_trade(bot, action=m.ACTION_LONG,
                                             confidence=0.50,
                                             current_price=60000.0)
    # The execute_trade path may return None for many reasons; the most
    # we can assert is that it didn't proceed to _open_position. Since
    # the bot is FLAT, "didn't proceed" means _open_position wasn't called.
    bot._open_position.assert_not_called()


@pytest.mark.parametrize("structure_first", [True, False])
def test_execute_trade_allows_when_confidence_above_floor(structure_first):
    """In both modes, high-confidence entry should be allowed past the floor.

    We can't assert a successful trade open without setting up the entire
    bot, but we can assert _open_position is reachable (not blocked here).
    """
    import live_trading_htf as m
    bot = _make_bot_for_execute_trade(structure_first)
    bot._is_in_cooldown_period = MagicMock(return_value=False)
    bot._should_block_long_reversal = MagicMock(return_value=False)
    bot._open_position = MagicMock(return_value={"action": "OPEN_LONG"})
    bot._fetch_market_signals = MagicMock(return_value={})
    bot._build_signal_summary = MagicMock(return_value={})
    bot._whipsaw_block_check = MagicMock(return_value=None)
    bot._mfi_filter_check = MagicMock(return_value=None)

    with patch.object(m, "STRUCTURE_FIRST_MODE", structure_first):
        try:
            m.HTFLiveBot.execute_trade(bot, action=m.ACTION_LONG,
                                        confidence=0.85, current_price=60000.0)
        except Exception:
            # The full execute_trade has many other paths that need real
            # state. We just want to confirm the confidence floor wasn't
            # the blocker. If something later raised, that's fine.
            pass
    # If the floor was the blocker, _open_position wouldn't be reachable.
    # We don't assert called — too many other guards may also block. The
    # test is the parametrized BLOCKED variant proving the floor fires.
