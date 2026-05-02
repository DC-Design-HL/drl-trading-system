"""M4 tests — guards individually + chain composition."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ctx(symbol="BTCUSDT", n_bars=80, with_atr_data=True, **extras):
    """Build a minimal Ctx with enough primary bars for ADX guards."""
    from src.btengine.sim.context import Ctx
    rng = np.random.default_rng(0)
    closes = 60000 + np.cumsum(rng.normal(50, 50, n_bars))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + 100
    lows = np.minimum(opens, closes) - 100
    df = pd.DataFrame({
        "open_time": np.arange(n_bars) * 900_000,
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.full(n_bars, 100.0),
    })
    return Ctx(symbol=symbol,
               now_ms=int(df["open_time"].iloc[-1]),
               cursor_index=len(df) - 1,
               primary=df, htf={}, extras=dict(extras))


# ──────────────────────────────────────────────────────────────────
# SymbolBlocklistGuard
# ──────────────────────────────────────────────────────────────────

def test_symbol_blocklist_blocks_default_pairs():
    from src.btengine.guards import SymbolBlocklistGuard
    from src.btengine.strategy.base import Intent
    g = SymbolBlocklistGuard()
    # Default blocklist contains (BTC, SHORT)
    r = g(Intent(action="OPEN_SHORT"), _ctx(symbol="BTCUSDT"))
    assert not r.allowed
    assert "blocked" in r.reason


def test_symbol_blocklist_allows_unblocked_pair():
    from src.btengine.guards import SymbolBlocklistGuard
    from src.btengine.strategy.base import Intent
    g = SymbolBlocklistGuard()
    # Default does NOT block (BTC, LONG)
    assert g(Intent(action="OPEN_LONG"), _ctx(symbol="BTCUSDT")).allowed
    # And not (XRP, *) either
    assert g(Intent(action="OPEN_LONG"), _ctx(symbol="XRPUSDT")).allowed
    assert g(Intent(action="OPEN_SHORT"), _ctx(symbol="XRPUSDT")).allowed


def test_symbol_blocklist_allows_non_open_intents():
    from src.btengine.guards import SymbolBlocklistGuard
    from src.btengine.strategy.base import Intent
    g = SymbolBlocklistGuard()
    assert g(Intent(action="HOLD"), _ctx()).allowed
    assert g(Intent(action="REVERSE_CLOSE_LONG"), _ctx()).allowed


# ──────────────────────────────────────────────────────────────────
# ADXGuard
# ──────────────────────────────────────────────────────────────────

def test_adx_guard_allows_in_warmup():
    """Below 3*period bars → fail open."""
    from src.btengine.guards import ADXGuard
    from src.btengine.strategy.base import Intent
    g = ADXGuard(min_adx=20, period=14)
    short_ctx = _ctx(n_bars=20)
    assert g(Intent(action="OPEN_LONG"), short_ctx).allowed


def test_adx_guard_blocks_below_min():
    """Range-bound oscillating data → low non-NaN ADX → blocked.

    Note: pure-flat data produces NaN ADX (degenerate division). We test
    a realistic range-bound regime where ADX is computable but low. The
    NaN path is exercised by test_adx_guard_allows_in_warmup (insufficient
    bars) and is also covered by the explicit fail-open in adx.py.
    """
    from src.btengine.guards import ADXGuard
    from src.btengine.strategy.base import Intent
    from src.btengine.sim.context import Ctx
    n = 200
    # Sine-wave oscillation around 100 → no directional bias → low ADX
    t = np.arange(n)
    closes = 100.0 + 0.5 * np.sin(t * 0.4)  # oscillates ±0.5%
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + 0.05
    lows = np.minimum(opens, closes) - 0.05
    df = pd.DataFrame({
        "open_time": np.arange(n) * 900_000,
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.full(n, 10.0),
    })
    ctx = Ctx(symbol="BTCUSDT", now_ms=int(df["open_time"].iloc[-1]),
              cursor_index=len(df) - 1, primary=df, htf={})
    g = ADXGuard(min_adx=20, period=14)
    r = g(Intent(action="OPEN_LONG"), ctx)
    assert not r.allowed, f"expected ADX block on range-bound data, got allow: {r}"
    assert "ranging" in r.reason or "adx" in r.reason


# ──────────────────────────────────────────────────────────────────
# FundingLongGuard
# ──────────────────────────────────────────────────────────────────

def test_funding_long_blocks_above_threshold():
    from src.btengine.guards import FundingLongGuard
    from src.btengine.strategy.base import Intent
    g = FundingLongGuard(max_funding=0.05)  # 0.05%
    # Funding 0.001 (decimal) = 0.1% — above threshold
    r = g(Intent(action="OPEN_LONG"), _ctx(funding_rate=0.001))
    assert not r.allowed


def test_funding_long_allows_below_threshold_or_short():
    from src.btengine.guards import FundingLongGuard
    from src.btengine.strategy.base import Intent
    g = FundingLongGuard(max_funding=0.05)
    # 0.0001 (decimal) = 0.01% < 0.05%
    assert g(Intent(action="OPEN_LONG"), _ctx(funding_rate=0.0001)).allowed
    # SHORT entry — guard never blocks
    assert g(Intent(action="OPEN_SHORT"), _ctx(funding_rate=0.001)).allowed
    # Missing funding rate → fail-open
    assert g(Intent(action="OPEN_LONG"), _ctx()).allowed


# ──────────────────────────────────────────────────────────────────
# WhaleNeutralGuard
# ──────────────────────────────────────────────────────────────────

def test_whale_neutral_blocks_on_neutral():
    from src.btengine.guards import WhaleNeutralGuard
    from src.btengine.strategy.base import Intent
    g = WhaleNeutralGuard()
    assert not g(Intent(action="OPEN_LONG"), _ctx(whale_direction="NEUTRAL")).allowed
    assert not g(Intent(action="OPEN_SHORT"), _ctx(whale_direction="NEUTRAL")).allowed


def test_whale_neutral_allows_on_bull_or_bear():
    from src.btengine.guards import WhaleNeutralGuard
    from src.btengine.strategy.base import Intent
    g = WhaleNeutralGuard()
    assert g(Intent(action="OPEN_LONG"), _ctx(whale_direction="BULL")).allowed
    assert g(Intent(action="OPEN_LONG"), _ctx(whale_direction="BEAR")).allowed
    assert g(Intent(action="OPEN_LONG"), _ctx()).allowed  # no data → allow


# ──────────────────────────────────────────────────────────────────
# ExtPosNewsGuard
# ──────────────────────────────────────────────────────────────────

def test_ext_pos_news_blocks_long_on_high_sentiment():
    from src.btengine.guards import ExtPosNewsGuard
    from src.btengine.strategy.base import Intent
    g = ExtPosNewsGuard(sentiment_threshold=0.5)
    r = g(Intent(action="OPEN_LONG"), _ctx(recent_max_news_sentiment=0.7))
    assert not r.allowed


def test_ext_pos_news_allows_short_or_low_sentiment():
    from src.btengine.guards import ExtPosNewsGuard
    from src.btengine.strategy.base import Intent
    g = ExtPosNewsGuard(sentiment_threshold=0.5)
    # SHORT entry — no block
    assert g(Intent(action="OPEN_SHORT"), _ctx(recent_max_news_sentiment=0.9)).allowed
    # Below threshold
    assert g(Intent(action="OPEN_LONG"), _ctx(recent_max_news_sentiment=0.3)).allowed
    # No data
    assert g(Intent(action="OPEN_LONG"), _ctx()).allowed


# ──────────────────────────────────────────────────────────────────
# USDTDGuard
# ──────────────────────────────────────────────────────────────────

def test_usdtd_blocks_long_when_basket_drops():
    from src.btengine.guards import USDTDGuard
    from src.btengine.strategy.base import Intent
    g = USDTDGuard(threshold_pct=0.7)
    # Basket dropped 1% (>0.7% threshold) → USDT.D rising → block LONG
    r = g(Intent(action="OPEN_LONG"), _ctx(basket_change_pct=-1.0))
    assert not r.allowed


def test_usdtd_allows_short_or_calm():
    from src.btengine.guards import USDTDGuard
    from src.btengine.strategy.base import Intent
    g = USDTDGuard(threshold_pct=0.7)
    assert g(Intent(action="OPEN_SHORT"), _ctx(basket_change_pct=-1.0)).allowed
    assert g(Intent(action="OPEN_LONG"), _ctx(basket_change_pct=-0.3)).allowed
    assert g(Intent(action="OPEN_LONG"), _ctx()).allowed


# ──────────────────────────────────────────────────────────────────
# ReverseCloseLongGuard
# ──────────────────────────────────────────────────────────────────

def test_reverse_close_long_only_acts_on_rcl_intent():
    from src.btengine.guards import ReverseCloseLongGuard
    from src.btengine.strategy.base import Intent
    g = ReverseCloseLongGuard()
    # OPEN_LONG/OPEN_SHORT/HOLD all pass
    assert g(Intent(action="OPEN_LONG"), _ctx(btc_4h_slope_pct=2.0)).allowed
    assert g(Intent(action="OPEN_SHORT"), _ctx(btc_4h_slope_pct=2.0)).allowed
    assert g(Intent(action="HOLD"), _ctx()).allowed


def test_reverse_close_long_blocks_in_uptrend_for_canary_symbol():
    from src.btengine.guards import ReverseCloseLongGuard
    from src.btengine.strategy.base import Intent
    g = ReverseCloseLongGuard()
    # XRP is in canary; BTC slope +2% > -0.5% gate → block flip
    r = g(Intent(action="REVERSE_CLOSE_LONG"),
          _ctx(symbol="XRPUSDT", btc_4h_slope_pct=2.0))
    assert not r.allowed
    assert "canary" in r.reason


def test_reverse_close_long_allows_in_downtrend():
    from src.btengine.guards import ReverseCloseLongGuard
    from src.btengine.strategy.base import Intent
    g = ReverseCloseLongGuard()
    # Slope -2% < -0.5% gate → allow flip
    r = g(Intent(action="REVERSE_CLOSE_LONG"),
          _ctx(symbol="XRPUSDT", btc_4h_slope_pct=-2.0))
    assert r.allowed


def test_reverse_close_long_allows_non_canary_symbol():
    """Default canary is all 4 production symbols. A hypothetical new symbol
    should NOT trigger the guard."""
    from src.btengine.guards import ReverseCloseLongGuard
    from src.btengine.strategy.base import Intent
    g = ReverseCloseLongGuard()
    r = g(Intent(action="REVERSE_CLOSE_LONG"),
          _ctx(symbol="HYPOTHETICAL", btc_4h_slope_pct=2.0))
    assert r.allowed


def test_reverse_close_long_allows_when_slope_unavailable():
    """Fail-open path matches live's transient-error behavior."""
    from src.btengine.guards import ReverseCloseLongGuard
    from src.btengine.strategy.base import Intent
    g = ReverseCloseLongGuard()
    r = g(Intent(action="REVERSE_CLOSE_LONG"),
          _ctx(symbol="XRPUSDT"))  # no btc_4h_slope_pct
    assert r.allowed


# ──────────────────────────────────────────────────────────────────
# Guard chain composition
# ──────────────────────────────────────────────────────────────────

def test_build_guard_chain_from_config():
    from src.btengine.guards import build_guard_chain
    chain = build_guard_chain(
        enabled=["symbol_blocklist", "adx"],
        params={"adx": {"min_adx": 25, "max_adx": 55}},
    )
    assert len(chain) == 2


def test_build_guard_chain_rejects_unknown_name():
    from src.btengine.guards import build_guard_chain
    with pytest.raises(ValueError, match="Unknown guard"):
        build_guard_chain(enabled=["does_not_exist"], params={})


def test_chain_first_blocker_wins():
    from src.btengine.guards import build_guard_chain
    from src.btengine.strategy.base import Intent
    chain = build_guard_chain(
        enabled=["symbol_blocklist", "adx"],
        params={},
    )
    # symbol_blocklist will fire on (BTC, SHORT) — adx never even runs
    r = chain(Intent(action="OPEN_SHORT"), _ctx(symbol="BTCUSDT"))
    assert not r.allowed
    assert "symbol_blocklist" in r.reason or "blocked" in r.reason
