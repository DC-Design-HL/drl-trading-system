"""Tests for the asymmetric REVERSE_CLOSE_LONG canary guard.

Guard lives in live_trading_htf.py and fires before _close_position() at the
reversal decision point. It should:
  * block REVERSE_CLOSE_LONG for symbols in REVERSAL_BLOCK_LONG_CANARY_SYMBOLS
    when BTC 4h EMA slope > REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT
  * allow it otherwise (downtrend, non-canary symbol, slope fetch failure)
  * never touch SHORT reversals
  * be based on a cached BTC slope that respects _BTC_REGIME_CACHE_TTL_SECONDS
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import live_trading_htf as m


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _bot_stub(symbol: str) -> SimpleNamespace:
    """Minimal stand-in exposing the two attributes the guard reads."""
    return SimpleNamespace(symbol=symbol, fetcher=MagicMock())


def _reset_cache():
    m._btc_regime_cache["slope_pct"] = None
    m._btc_regime_cache["ts"] = 0.0


def _fake_4h_df(slope_pct: float) -> pd.DataFrame:
    """Build a 4h OHLCV frame where closes grow linearly such that
    _compute_btc_4h_ema_slope_pct returns approximately `slope_pct`.

    The helper compares EMA(10) over closes[-10:] vs EMA(10) over closes[-15:-5].
    Those windows are offset by 5 bars, so a linear ramp of per-bar ratio `r`
    yields ≈ 5*r percent difference. We calibrate r = slope_pct/5 per bar.
    """
    base = 60000.0
    per_bar_pct = slope_pct / 5.0
    closes = [base * (1 + per_bar_pct / 100.0 * i) for i in range(20)]
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1.0] * len(closes),
    })


def _force_cache(slope_pct: float) -> None:
    """Bypass the slope-helper path entirely for tests that only care
    about the bot-side branch behaviour."""
    m._btc_regime_cache["slope_pct"] = slope_pct
    m._btc_regime_cache["ts"] = time.time()


# ──────────────────────────────────────────────────────────────────────────
# Config sanity
# ──────────────────────────────────────────────────────────────────────────

def test_config_defaults_are_conservative():
    """Canary expanded 2026-05-01 to all 4 symbols after 8 days of clean
    XRP validation (XRP REVERSE_CLOSE_LONG events: 11 pre → 0 post).
    Original XRP-only safety check superseded; the four symbols match
    the asymmetric LONG-only Bonferroni-validated finding from 2026-04-23
    (95% CI [+$1,408, +$1,976]/month)."""
    assert m.REVERSAL_BLOCK_LONG_CANARY_SYMBOLS == {"XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert m.REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT == -0.5
    assert m._BTC_REGIME_CACHE_TTL_SECONDS >= 60


# ──────────────────────────────────────────────────────────────────────────
# Guard behavior
# ──────────────────────────────────────────────────────────────────────────

def test_non_canary_symbol_never_blocks():
    """No symbol is currently outside the canary set, but the safety
    behavior must hold: a hypothetical non-canary symbol returns False
    without invoking the fetcher."""
    _reset_cache()
    bot = _bot_stub("HYPOTHETICAL_NEW_SYMBOL")  # not in {BTC,ETH,SOL,XRP}
    bot.fetcher.fetch_asset.return_value = _fake_4h_df(+2.0)
    assert m.HTFLiveBot._should_block_long_reversal(bot) is False
    bot.fetcher.fetch_asset.assert_not_called()


def test_canary_symbol_uptrend_blocks():
    _reset_cache()
    _force_cache(+2.0)
    bot = _bot_stub("XRPUSDT")
    assert m.HTFLiveBot._should_block_long_reversal(bot) is True


def test_canary_symbol_range_blocks():
    _reset_cache()
    # 0% slope is > -0.5% gate → still block (range regime was validated)
    _force_cache(0.0)
    bot = _bot_stub("XRPUSDT")
    assert m.HTFLiveBot._should_block_long_reversal(bot) is True


def test_canary_symbol_clear_downtrend_allows_reversal():
    _reset_cache()
    _force_cache(-2.0)
    bot = _bot_stub("XRPUSDT")
    # Slope -2% <= gate -0.5% → downtrend, let reversal fire (untested regime)
    assert m.HTFLiveBot._should_block_long_reversal(bot) is False


def test_canary_symbol_at_exact_threshold_allows():
    _reset_cache()
    _force_cache(-0.5)
    bot = _bot_stub("XRPUSDT")
    # Boundary: slope == gate → NOT a block (we require strictly above gate)
    assert m.HTFLiveBot._should_block_long_reversal(bot) is False


def test_fetch_failure_fails_open():
    _reset_cache()
    bot = _bot_stub("XRPUSDT")
    bot.fetcher.fetch_asset.side_effect = RuntimeError("API down")
    # Cache empty + fetch failure → slope None → allow reversal
    assert m.HTFLiveBot._should_block_long_reversal(bot) is False


def test_fetch_failure_uses_stale_cache_if_present():
    _reset_cache()
    # Seed a fresh cache value (uptrend)
    m._btc_regime_cache["slope_pct"] = 1.5
    m._btc_regime_cache["ts"] = time.time()

    bot = _bot_stub("XRPUSDT")
    bot.fetcher.fetch_asset.side_effect = RuntimeError("API down")
    # Cache was fresh and uptrend — should still block
    assert m.HTFLiveBot._should_block_long_reversal(bot) is True


# ──────────────────────────────────────────────────────────────────────────
# Slope helper — caching behavior
# ──────────────────────────────────────────────────────────────────────────

def test_slope_helper_caches_within_ttl():
    _reset_cache()
    fetcher = MagicMock()
    fetcher.fetch_asset.return_value = _fake_4h_df(+1.0)

    first = m._compute_btc_4h_ema_slope_pct(fetcher)
    second = m._compute_btc_4h_ema_slope_pct(fetcher)

    assert first is not None
    assert first == second
    # Second call must hit cache, not the fetcher
    assert fetcher.fetch_asset.call_count == 1


def test_slope_helper_refreshes_after_ttl(monkeypatch):
    _reset_cache()
    fetcher = MagicMock()
    fetcher.fetch_asset.return_value = _fake_4h_df(+1.0)

    m._compute_btc_4h_ema_slope_pct(fetcher)
    assert fetcher.fetch_asset.call_count == 1

    # Pretend the cache aged out
    m._btc_regime_cache["ts"] = time.time() - m._BTC_REGIME_CACHE_TTL_SECONDS - 1
    m._compute_btc_4h_ema_slope_pct(fetcher)
    assert fetcher.fetch_asset.call_count == 2


def test_slope_helper_returns_none_on_first_failure():
    _reset_cache()
    fetcher = MagicMock()
    fetcher.fetch_asset.side_effect = RuntimeError("nope")
    assert m._compute_btc_4h_ema_slope_pct(fetcher) is None


def test_slope_helper_handles_empty_dataframe():
    _reset_cache()
    fetcher = MagicMock()
    fetcher.fetch_asset.return_value = pd.DataFrame(columns=["close"])
    assert m._compute_btc_4h_ema_slope_pct(fetcher) is None
