"""Tests for src/signals/structure_filters.py (PROFITABILITY_PLAN.md P2.D).

The helpers were extracted from live_trading_htf._get_structure_direction
so the bot and the forward simulator can share entry logic. These tests
assert the pure-function contract and the live-vs-helper bit equivalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.structure_filters import (
    passes_adx_directional,
    passes_exhaustion_filter,
    passes_ob_proximity,
    passes_rsi_guard,
    passes_structure_first_adx,
)


def _ohlc(opens, highs, lows, closes) -> pd.DataFrame:
    return pd.DataFrame({"open": opens, "high": highs,
                         "low": lows, "close": closes,
                         "volume": np.ones(len(opens))})


# ─── OB proximity ──────────────────────────────────────────────────────


def test_ob_proximity_fail_open_too_few_bars() -> None:
    """Live behaviour: filter is skipped when < 40 bars are available."""
    df = _ohlc([100]*30, [101]*30, [99]*30, [100]*30)
    assert passes_ob_proximity(
        df, direction_long=True, current_price=100.0,
        proximity_pct=0.01,
    ) is True


def test_ob_proximity_far_from_ob_blocks() -> None:
    """When OBs exist but current price is nowhere near one, the filter
    must return False (live: the bot does NOT enter)."""
    n = 50
    rng = np.random.default_rng(7)
    opens = 100 + rng.standard_normal(n) * 0.5
    highs = opens + 0.5
    lows = opens - 0.5
    closes = opens + rng.standard_normal(n) * 0.1
    # Force a clear bull OB at idx 35 (down candle), 36 (up), 37 (impulsive up)
    opens[35], closes[35] = 100.0, 99.0   # down
    opens[36], closes[36] = 99.0, 100.0   # up
    opens[37], closes[37] = 100.0, 102.0  # impulsive up
    highs[35], lows[35] = 100.5, 98.5
    df = _ohlc(opens, highs, lows, closes)
    # The OB mid-price is around 99.5; query way above proximity (101.5,
    # > 1% from 99.5).
    assert passes_ob_proximity(
        df, direction_long=True, current_price=110.0,
        proximity_pct=0.005,  # 0.5%
    ) is False


def test_ob_proximity_near_ob_passes() -> None:
    n = 50
    opens = np.full(n, 100.0)
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    closes = np.full(n, 100.0)
    # Carve a clear bull OB at idx 35-37
    opens[35], closes[35] = 100.0, 99.0
    opens[36], closes[36] = 99.0, 100.0
    opens[37], closes[37] = 100.0, 102.0
    highs[35], lows[35] = 100.0, 99.0
    df = _ohlc(opens, highs, lows, closes)
    # Current price right at the OB mid → should pass
    assert passes_ob_proximity(
        df, direction_long=True, current_price=99.5,
        proximity_pct=0.02,
    ) is True


# ─── ADX directional ───────────────────────────────────────────────────


def test_adx_fail_open_too_few_bars() -> None:
    df = _ohlc([100]*10, [101]*10, [99]*10, [100]*10)
    assert passes_adx_directional(
        df, direction_long=True, adx_guard_min=20.0,
    ) is True


def test_adx_below_guard_passes() -> None:
    """If ADX < guard, directional check is skipped (live behaviour)."""
    n = 35
    # Flat price series → low ADX
    df = _ohlc([100]*n, [100.2]*n, [99.8]*n, [100]*n)
    assert passes_adx_directional(
        df, direction_long=True, adx_guard_min=20.0,
    ) is True


def test_adx_directional_block_long_in_downtrend() -> None:
    """A clearly bearish 30-bar window (lows trending down, highs flat)
    should block a LONG entry (-DI > +DI, ADX above guard)."""
    n = 35
    highs = np.full(n, 105.0)
    lows = np.linspace(100, 95, n)
    closes = lows + 0.5
    opens = closes - 0.2
    df = _ohlc(opens, highs, lows, closes)
    assert passes_adx_directional(
        df, direction_long=True, adx_guard_min=10.0,
    ) is False


def test_adx_directional_passes_long_in_uptrend() -> None:
    n = 35
    lows = np.full(n, 100.0)
    highs = np.linspace(105, 110, n)
    closes = highs - 0.5
    opens = closes - 0.2
    df = _ohlc(opens, highs, lows, closes)
    assert passes_adx_directional(
        df, direction_long=True, adx_guard_min=10.0,
    ) is True


# ─── Live bot still uses the helpers (smoke import) ────────────────────


def test_live_bot_imports_helpers() -> None:
    """Refactor invariant: the live module references the same helpers."""
    import live_trading_htf as live
    assert live.passes_adx_directional is passes_adx_directional
    assert live.passes_ob_proximity is passes_ob_proximity


# ─── Structure-first ADX hard block ────────────────────────────────────


def test_struct_first_adx_passes_strong_trend() -> None:
    n = 35
    lows = np.full(n, 100.0)
    highs = np.linspace(105, 115, n)
    closes = highs - 0.5
    opens = closes - 0.2
    df = _ohlc(opens, highs, lows, closes)
    assert passes_structure_first_adx(df, adx_guard_min=20.0) is True


def test_struct_first_adx_blocks_flat() -> None:
    n = 35
    df = _ohlc([100]*n, [100.1]*n, [99.9]*n, [100]*n)
    assert passes_structure_first_adx(df, adx_guard_min=20.0) is False


def test_struct_first_adx_fail_open_too_few_bars() -> None:
    df = _ohlc([100]*10, [101]*10, [99]*10, [100]*10)
    assert passes_structure_first_adx(df, adx_guard_min=20.0) is True


# ─── Exhaustion filter ────────────────────────────────────────────────


def test_exhaustion_blocks_far_from_vwap() -> None:
    n = 25
    # Tight 100±0.5 series → VWAP≈100, ATR small. Probe price 110 →
    # extension huge → should block.
    df = _ohlc([100]*n, [100.5]*n, [99.5]*n, [100]*n)
    assert passes_exhaustion_filter(
        df, current_price=110.0, threshold_atr=3.0,
    ) is False


def test_exhaustion_passes_near_vwap() -> None:
    n = 25
    df = _ohlc([100]*n, [100.5]*n, [99.5]*n, [100]*n)
    assert passes_exhaustion_filter(
        df, current_price=100.2, threshold_atr=3.0,
    ) is True


def test_exhaustion_fail_open_too_few_bars() -> None:
    df = _ohlc([100]*5, [101]*5, [99]*5, [100]*5)
    assert passes_exhaustion_filter(
        df, current_price=200.0, threshold_atr=3.0,
    ) is True


def _legacy_exhaustion_allows(df, current_price, threshold) -> bool:
    """Verbatim copy of the OLD inline execute_trade exhaustion block
    (live_trading_htf lines ~3026-3055, pre-delegation). Returns True if
    the trade was ALLOWED (i.e. NOT blocked) — used to prove the helper
    is a bit-for-bit no-op replacement of the inlined logic.
    """
    if df is None or len(df) < 20:
        return True  # too few bars → block never ran → allowed
    try:
        closes = df["close"].values[-20:]
        volumes = df["volume"].values[-20:]
        typical_price = (df["high"].values[-20:] + df["low"].values[-20:] + closes) / 3.0
        vwap_20 = float(np.sum(typical_price * volumes) / (np.sum(volumes) + 1e-10))
        highs = df["high"].values[-20:]
        lows = df["low"].values[-20:]
        tr = np.maximum(highs - lows, np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1)),
        ))
        atr_14 = float(np.mean(tr[-14:]))
        if atr_14 > 0:
            extension = abs(current_price - vwap_20) / atr_14
            if extension > threshold:
                return False  # blocked
        return True
    except Exception:
        return True  # error → block skipped → allowed


def test_exhaustion_matches_legacy_inline() -> None:
    """The helper must equal the old inline live formula on every fixture
    (behavior-preserving proof, not assertion — PROFITABILITY_PLAN.md §6)."""
    rng = np.random.RandomState(7)
    cases = []
    # Randomized realistic OHLCV windows of varying length.
    for n in (19, 20, 21, 30, 50):
        for _ in range(40):
            base = rng.uniform(10, 50000)
            closes = base + rng.normal(0, base * 0.01, n)
            highs = closes + rng.uniform(0, base * 0.01, n)
            lows = closes - rng.uniform(0, base * 0.01, n)
            opens = closes + rng.normal(0, base * 0.005, n)
            vols = rng.uniform(1, 1000, n)
            df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                               "close": closes, "volume": vols})
            for probe in (base, base * 1.05, base * 0.95, base * 2.0):
                cases.append((df, float(probe)))
    # Plus the degenerate edge cases the live block fail-opens on.
    cases.append((_ohlc([100]*20, [100]*20, [100]*20, [100]*20), 100.0))  # ATR=0
    cases.append((None, 100.0))
    for df, probe in cases:
        assert passes_exhaustion_filter(
            df, current_price=probe, threshold_atr=3.0,
        ) is _legacy_exhaustion_allows(df, probe, 3.0)


# ─── RSI band guard ────────────────────────────────────────────────────


def test_rsi_blocks_long_when_overbought() -> None:
    n = 30
    closes = np.linspace(100, 110, n)  # consistently rising → RSI > 70
    df = _ohlc(closes - 0.1, closes + 0.2, closes - 0.2, closes)
    assert passes_rsi_guard(
        df, direction_long=True, ob_threshold=70.0, os_threshold=30.0,
    ) is False


def test_rsi_blocks_short_when_oversold() -> None:
    n = 30
    closes = np.linspace(100, 90, n)
    df = _ohlc(closes + 0.1, closes + 0.2, closes - 0.2, closes)
    assert passes_rsi_guard(
        df, direction_long=False, ob_threshold=70.0, os_threshold=30.0,
    ) is False


def test_rsi_neutral_passes_both_sides() -> None:
    n = 30
    rng = np.random.default_rng(0)
    closes = 100 + rng.standard_normal(n) * 0.1
    df = _ohlc(closes - 0.05, closes + 0.1, closes - 0.1, closes)
    # RSI near 50 → neither side blocks
    assert passes_rsi_guard(
        df, direction_long=True, ob_threshold=70.0, os_threshold=30.0,
    ) is True
    assert passes_rsi_guard(
        df, direction_long=False, ob_threshold=70.0, os_threshold=30.0,
    ) is True
