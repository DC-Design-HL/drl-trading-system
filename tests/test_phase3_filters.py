"""
Regression tests for Phase 3 filters deployed 2026-04-27 19:30 UTC.

Two new guards added on top of existing Option A (sizing 2× + blocklist):
  * FUNDING_LONG_GUARD — block LONG when 8h funding rate < 0
  * WHALE_NEUTRAL_GUARD — block any entry when whale.direction == NEUTRAL

Both backtested incremental on 288 closed trades (Apr 6 → Apr 26):
  * F1 funding-LONG: +$284 incremental on top of blocklist
  * F2 whale-NEUTRAL: +$413 incremental on top of blocklist
  * F1 + F2 stacked: +$405 incremental, allowed-WR 69.2% vs 56.5% baseline
"""

from live_trading_htf import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    FUNDING_LONG_GUARD_ENABLED,
    FUNDING_LONG_GUARD_MIN_RATE,
    WHALE_NEUTRAL_GUARD_ENABLED,
)


def test_funding_long_guard_constants() -> None:
    assert FUNDING_LONG_GUARD_ENABLED is True
    assert FUNDING_LONG_GUARD_MIN_RATE == 0.0


def test_whale_neutral_guard_constants() -> None:
    assert WHALE_NEUTRAL_GUARD_ENABLED is True


def test_funding_long_guard_only_affects_longs() -> None:
    """Funding-LONG guard must only block LONG entries — SHORTs benefit
    from negative funding (they get paid). Blocking SHORTs would invert
    the entire signal."""
    from live_trading_htf import HTFLiveBot

    bot = object.__new__(HTFLiveBot)
    bot.symbol = "BTCUSDT"
    # No fetcher → fail-open path

    # ACTION_HOLD always allowed
    assert HTFLiveBot._check_funding_long_guard(bot, ACTION_HOLD) is True
    # ACTION_SHORT never blocked (regardless of funding)
    assert HTFLiveBot._check_funding_long_guard(bot, ACTION_SHORT) is True


def test_funding_long_guard_fail_open() -> None:
    """If market signals can't be fetched, the guard fails open."""
    from live_trading_htf import HTFLiveBot

    bot = object.__new__(HTFLiveBot)
    bot.symbol = "BTCUSDT"
    bot._fetch_market_signals = lambda sym: (_ for _ in ()).throw(RuntimeError("network"))
    # Should NOT block on fetch error
    assert HTFLiveBot._check_funding_long_guard(bot, ACTION_LONG) is True


def test_whale_neutral_guard_blocks_both_directions() -> None:
    """Whale-NEUTRAL is a stress regime — block both LONG and SHORT.
    HOLD always passes through.
    """
    from live_trading_htf import HTFLiveBot

    bot = object.__new__(HTFLiveBot)
    bot.symbol = "BTCUSDT"
    bot._fetch_market_signals = lambda sym: {"whale": {"direction": "NEUTRAL"}}

    # HOLD always allowed
    assert HTFLiveBot._check_whale_neutral_guard(bot, ACTION_HOLD) is True
    # NEUTRAL whale → both LONG and SHORT blocked
    assert HTFLiveBot._check_whale_neutral_guard(bot, ACTION_LONG) is False
    assert HTFLiveBot._check_whale_neutral_guard(bot, ACTION_SHORT) is False


def test_whale_directional_signal_passes() -> None:
    """When whale gives a directional signal (not NEUTRAL), guard allows."""
    from live_trading_htf import HTFLiveBot

    bot = object.__new__(HTFLiveBot)
    bot.symbol = "BTCUSDT"
    bot._fetch_market_signals = lambda sym: {"whale": {"direction": "BULLISH"}}
    assert HTFLiveBot._check_whale_neutral_guard(bot, ACTION_LONG) is True
    assert HTFLiveBot._check_whale_neutral_guard(bot, ACTION_SHORT) is True


def test_funding_negative_blocks_long() -> None:
    """Negative funding (longs paying shorts) should block a LONG entry."""
    from live_trading_htf import HTFLiveBot

    bot = object.__new__(HTFLiveBot)
    bot.symbol = "BTCUSDT"
    bot._fetch_market_signals = lambda sym: {"funding": {"rate": -0.001}}
    assert HTFLiveBot._check_funding_long_guard(bot, ACTION_LONG) is False


def test_funding_positive_allows_long() -> None:
    """Positive funding (shorts paying longs) is the bullish regime — allow LONG."""
    from live_trading_htf import HTFLiveBot

    bot = object.__new__(HTFLiveBot)
    bot.symbol = "BTCUSDT"
    bot._fetch_market_signals = lambda sym: {"funding": {"rate": 0.0005}}
    assert HTFLiveBot._check_funding_long_guard(bot, ACTION_LONG) is True
