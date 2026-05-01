"""Tests for the Phase 2 OI observational logger.

The helper `_compute_oi_features(symbol)` is module-level on
`live_trading_htf` and:
  * fetches 1h OI hist + 1h klines from Binance Futures public API
  * caches per-symbol with _OI_CACHE_TTL_SECONDS TTL
  * returns dict with oi_roc_4h, oi_roc_24h, oi_z_7d, oi_value_roc_4h,
    px_roc_4h, computed_at
  * fails open: returns None (or stale cache) on any error
  * never raises

We mock urllib.request.urlopen to return canned payloads so tests are
deterministic and don't hit the network.
"""
import io
import json
import time
from unittest.mock import patch, MagicMock

import pytest

import live_trading_htf as m


def _fake_response(payload):
    """Build a context-manager-compatible mock for urllib.request.urlopen."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read = MagicMock(return_value=json.dumps(payload).encode())
    return cm


def _build_oi_payload(n=200, base_oi=100000.0, ramp_per_bar=10.0):
    """Build a synthetic Binance OI hist response with a clear linear ramp."""
    base_ts = 1700000000000
    out = []
    for i in range(n):
        oi = base_oi + ramp_per_bar * i
        oi_v = oi * 60000  # arbitrary mark-price multiplier
        out.append({
            "symbol": "BTCUSDT",
            "sumOpenInterest": f"{oi:.6f}",
            "sumOpenInterestValue": f"{oi_v:.6f}",
            "CMCCirculatingSupply": "20000000.00000000",
            "timestamp": base_ts + i * 3600 * 1000,
        })
    return out


def _build_kline_payload(n=200, base_close=60000.0, ramp_per_bar=20.0):
    """Build synthetic kline response: list-of-lists with [openTime, open, high, low, close, ...]."""
    base_ts = 1700000000000
    out = []
    for i in range(n):
        c = base_close + ramp_per_bar * i
        out.append([base_ts + i * 3600 * 1000, c, c, c, c, "0", base_ts + (i+1)*3600*1000 - 1, "0", 0, "0", "0", "0"])
    return out


def _reset_cache():
    m._oi_features_cache.clear()


# ──────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────

def test_compute_oi_features_returns_expected_keys():
    _reset_cache()
    oi = _build_oi_payload()
    kl = _build_kline_payload()

    def _side_effect(url, *args, **kwargs):
        u = url if isinstance(url, str) else url.full_url
        return _fake_response(oi if "openInterestHist" in u else kl)

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        feats = m._compute_oi_features("BTCUSDT")

    assert feats is not None
    for k in ("oi_roc_4h", "oi_roc_24h", "oi_z_7d", "oi_value_roc_4h", "px_roc_4h", "computed_at"):
        assert k in feats, f"missing key: {k}"
    # Linear ramp → oi_roc_4h should be > 0 and finite
    assert feats["oi_roc_4h"] > 0
    assert feats["oi_roc_24h"] > feats["oi_roc_4h"]  # 24h > 4h on a monotone ramp
    assert feats["oi_z_7d"] is not None  # 200 bars > 168 window


def test_caches_within_ttl():
    _reset_cache()
    oi = _build_oi_payload()
    kl = _build_kline_payload()
    call_count = {"n": 0}

    def _side_effect(url, *args, **kwargs):
        call_count["n"] += 1
        u = url if isinstance(url, str) else url.full_url
        return _fake_response(oi if "openInterestHist" in u else kl)

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        first = m._compute_oi_features("BTCUSDT")
        second = m._compute_oi_features("BTCUSDT")

    assert first == second
    # Should hit network exactly twice on first call (oi + kline), zero on second
    assert call_count["n"] == 2


def test_cache_is_per_symbol():
    _reset_cache()
    oi = _build_oi_payload()
    kl = _build_kline_payload()
    call_count = {"n": 0}

    def _side_effect(url, *args, **kwargs):
        call_count["n"] += 1
        u = url if isinstance(url, str) else url.full_url
        return _fake_response(oi if "openInterestHist" in u else kl)

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        m._compute_oi_features("BTCUSDT")
        m._compute_oi_features("ETHUSDT")

    # Two symbols → 4 fetches total (2 endpoints × 2 symbols)
    assert call_count["n"] == 4


def test_refresh_after_ttl():
    _reset_cache()
    oi = _build_oi_payload()
    kl = _build_kline_payload()
    call_count = {"n": 0}

    def _side_effect(url, *args, **kwargs):
        call_count["n"] += 1
        u = url if isinstance(url, str) else url.full_url
        return _fake_response(oi if "openInterestHist" in u else kl)

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        m._compute_oi_features("BTCUSDT")
        # Age out the cache
        m._oi_features_cache["BTCUSDT"]["ts"] = time.time() - m._OI_CACHE_TTL_SECONDS - 1
        m._compute_oi_features("BTCUSDT")

    # 4 calls: first oi+kl, post-TTL oi+kl
    assert call_count["n"] == 4


# ──────────────────────────────────────────────────────────────────────────
# Failure modes — fail-open contract
# ──────────────────────────────────────────────────────────────────────────

def test_returns_none_on_hard_failure_with_empty_cache():
    _reset_cache()
    with patch("urllib.request.urlopen", side_effect=RuntimeError("nope")):
        assert m._compute_oi_features("BTCUSDT") is None


def test_returns_stale_cache_when_fetch_fails():
    _reset_cache()
    # Seed cache manually
    seeded = {"oi_roc_4h": 1.5, "oi_roc_24h": 2.5, "oi_z_7d": 0.3,
              "oi_value_roc_4h": 1.7, "px_roc_4h": 0.5, "computed_at": 0}
    m._oi_features_cache["BTCUSDT"] = {
        "features": seeded,
        "ts": time.time() - m._OI_CACHE_TTL_SECONDS - 1,  # stale
    }
    with patch("urllib.request.urlopen", side_effect=RuntimeError("nope")):
        result = m._compute_oi_features("BTCUSDT")
    assert result == seeded


def test_returns_cache_or_none_on_short_response():
    _reset_cache()
    short_oi = _build_oi_payload(n=10)  # below 25 minimum
    short_kl = _build_kline_payload(n=10)

    def _side_effect(url, *args, **kwargs):
        u = url if isinstance(url, str) else url.full_url
        return _fake_response(short_oi if "openInterestHist" in u else short_kl)

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        assert m._compute_oi_features("BTCUSDT") is None


def test_returns_cache_or_none_on_empty_response():
    _reset_cache()

    def _side_effect(url, *args, **kwargs):
        return _fake_response([])

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        assert m._compute_oi_features("BTCUSDT") is None


def test_helper_never_raises():
    """Even with truly broken inputs, the helper must not propagate exceptions."""
    _reset_cache()

    def _side_effect(url, *args, **kwargs):
        # Raise from inside the with-block context to simulate mid-stream failure
        cm = MagicMock()
        cm.__enter__ = MagicMock(side_effect=ValueError("boom"))
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    with patch("urllib.request.urlopen", side_effect=_side_effect):
        result = m._compute_oi_features("BTCUSDT")
    assert result is None  # never raised, returned None
