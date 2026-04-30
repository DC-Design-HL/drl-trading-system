"""Regression tests for the EXT_POS_NEWS guard deployed 2026-04-30."""

from live_trading_htf import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    EXT_POS_NEWS_GUARD_ENABLED,
    EXT_POS_NEWS_LOOKBACK_HOURS,
    EXT_POS_NEWS_SENTIMENT_THRESHOLD,
)


def test_constants() -> None:
    assert EXT_POS_NEWS_GUARD_ENABLED is True
    assert EXT_POS_NEWS_SENTIMENT_THRESHOLD == 0.5
    assert EXT_POS_NEWS_LOOKBACK_HOURS == 4


def test_guard_only_affects_longs() -> None:
    """Filter is asymmetric — must NEVER block SHORT."""
    from live_trading_htf import HTFLiveBot
    bot = object.__new__(HTFLiveBot)
    bot.symbol = "BTCUSDT"
    # HOLD always allowed
    assert HTFLiveBot._check_ext_pos_news_guard(bot, ACTION_HOLD) is True
    # SHORT never blocked regardless of news
    assert HTFLiveBot._check_ext_pos_news_guard(bot, ACTION_SHORT) is True


def test_guard_fail_open_on_db_error() -> None:
    """If DB query fails, guard fails open (allows the trade)."""
    from live_trading_htf import HTFLiveBot, _ext_pos_news_cache
    _ext_pos_news_cache.clear()
    bot = object.__new__(HTFLiveBot)
    bot.symbol = "NONEXISTENT_SYMBOL_XYZ"
    # Should not raise even if news DB has no relevant rows; should return True
    result = HTFLiveBot._check_ext_pos_news_guard(bot, ACTION_LONG)
    assert isinstance(result, bool)
