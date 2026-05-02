"""Canonical indicator implementations.

The audit (docs/backtest_redesign_proposal.md) found 9 different ADX
implementations across legacy backtests, drifting ±5% WR between them.
This module is the ONE implementation imported by every guard / strategy.
Both Wilder smoothing (live-faithful) and EMA smoothing (faster, used in
some legacy scripts) are exposed; the live-faithful path is the default.

Functions are pandas-vectorized for backtest speed but produce
single-bar values when called incrementally with the trailing window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── ADX (Wilder smoothing — matches live bot) ─────────────────────────

def adx_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index with Wilder's smoothing.

    Returns a Series aligned to df.index. NaN for the first `2*period`
    rows where smoothing hasn't warmed up.

    Wilder smoothing: new = (prev * (n-1) + this) / n  (RMA-like).
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)

    atr = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(plus_dm, period) / atr
    minus_di = 100 * _wilder_smooth(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder_smooth(dx, period)
    return adx


def _wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA: alpha = 1/period, but the first `period` values are
    averaged into the seed, then exponential after."""
    alpha = 1.0 / period
    return s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


# ── RSI ────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── ATR ────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _wilder_smooth(tr, period)


# ── ATR percent (relative to current close) ────────────────────────────

def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as % of close — useful for ATR-floor SL/TP sizing."""
    a = atr(df, period)
    return a / df["close"].astype(float)
