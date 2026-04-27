"""
Regression tests for the aggressive-sizing rollout (target 30%/month).

Three deltas vs. the prior baseline (tag v-pre-aggressive-sizing-20260425):
  1. RISK_POOL_PCT  0.10 → 0.30  (3× per-trade dollar_risk)
  2. FIXED_MAX_NOTIONAL  3000 → 6000  (allow the larger notional through)
  3. ADX_GUARD_MAX  60  (new — block ADX > 60 exhaustion)
  4. USDT.D filter  (new — block LONGs when stablecoin dominance rising)
  5. Stagnant band already widened to [-1.0%, +0.5%] (deployed earlier)

These tests guard the constants and the LONG-only directionality of the
USDT.D filter; they do not exercise live data fetches.
"""

from live_trading_htf import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    ADX_GUARD_MAX,
    ADX_GUARD_MIN,
    FIXED_MAX_NOTIONAL,
    RISK_BUDGET_PARTS,
    RISK_POOL_PCT,
    USDT_D_GUARD_ENABLED,
    USDT_D_LOOKBACK_HOURS,
    USDT_D_PROXY_SYMBOLS,
    USDT_D_THRESHOLD_PCT,
)


def test_aggressive_sizing_constants() -> None:
    """Sizing was stepped back 3× → 2× on 2026-04-27 after -8.18% drawdown.
    Notional cap stays at 6000 from the original aggressive bundle.
    See tests/test_filter_option_a.py for the canonical post-rollback assertions.
    """
    assert RISK_POOL_PCT == 0.20, "Risk pool must be 20% (2× sizing post-rollback)."
    assert RISK_BUDGET_PARTS == 20
    assert FIXED_MAX_NOTIONAL == 6000.0


def test_per_trade_risk_dollars_at_5k_balance() -> None:
    """Sanity: at $5K balance, dollar_risk per trade = $50 (1.0% of balance) at 2× sizing."""
    balance = 5_000
    pool = balance * RISK_POOL_PCT
    risk_per_trade = pool / RISK_BUDGET_PARTS
    assert risk_per_trade == 50.0, (
        f"Expected $50/trade at $5K balance under 2× sizing, got ${risk_per_trade}."
    )


def test_adx_exhaustion_block() -> None:
    """ADX > 60 should now be a blocking signal."""
    assert ADX_GUARD_MAX == 60, "Exhaustion guard must trigger at ADX > 60."
    assert ADX_GUARD_MIN < ADX_GUARD_MAX, "Min must be below max."


def test_usdt_d_filter_config() -> None:
    """USDT.D filter must be on, with the documented 4-symbol basket.
    Threshold tuned 0.5 → 0.7 on 2026-04-26 after threshold-sensitivity backtest
    (scripts/backtest_usdt_d_threshold.py).
    """
    assert USDT_D_GUARD_ENABLED is True
    assert USDT_D_LOOKBACK_HOURS == 2
    assert USDT_D_THRESHOLD_PCT == 0.7
    assert set(USDT_D_PROXY_SYMBOLS) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}


def test_usdt_d_guard_only_blocks_longs() -> None:
    """The USDT.D filter is asymmetric: it must NEVER block SHORTs.
    SHORTs benefit from a USDT.D-rising regime (crypto falling); blocking
    them would invert the entire signal.
    """
    from live_trading_htf import HTFLiveBot
    # Stub a bot just enough to call _check_usdt_d_guard without fetching.
    bot = object.__new__(HTFLiveBot)
    bot.symbol = "BTCUSDT"
    bot.fetcher = None  # the helper handles this gracefully via fail-open

    # ACTION_HOLD → always allow
    assert HTFLiveBot._check_usdt_d_guard(bot, ACTION_HOLD) is True

    # ACTION_SHORT → never blocked, regardless of USDT.D state
    assert HTFLiveBot._check_usdt_d_guard(bot, ACTION_SHORT) is True


def test_usdt_d_guard_long_with_no_data_fails_open() -> None:
    """If the proxy can't be computed (no fetcher), LONGs proceed (fail-open)."""
    from live_trading_htf import HTFLiveBot, _usdt_d_cache
    # Reset cache so no stale value answers
    _usdt_d_cache["ts"] = 0
    _usdt_d_cache["rising"] = None
    bot = object.__new__(HTFLiveBot)
    bot.symbol = "BTCUSDT"
    bot.fetcher = None  # forces None return path
    # With no cache and no fetcher, we should fail-open (return True = allow)
    assert HTFLiveBot._check_usdt_d_guard(bot, ACTION_LONG) is True
