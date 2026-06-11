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
    passes_ob_proximity,
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
