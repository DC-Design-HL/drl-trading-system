"""Pure helpers for structure-first entry filters.

Extracted from live_trading_htf.py per PROFITABILITY_PLAN.md §3/P2 so
the live bot and the forward simulator call the same code. Every
helper is deterministic — same DataFrame in → same answer out — and
has no logging side effects.

These helpers MIRROR the live logic so the forward simulator can call
them today; the live bot still inlines its own copies. Wiring the live
bot to delegate here (so there is a single source of truth) is a
pending step tracked in PROFITABILITY_PLAN.md §3/P2 — until that lands,
the calibration gate is what proves the mirror has not drifted from:
  * the S5 block at live_trading_htf._get_structure_direction
    (``passes_ob_proximity`` / ``passes_adx_directional``);
  * the pre-trade guards in ``execute_trade``
    (``passes_structure_first_adx`` / ``passes_exhaustion_filter`` /
    ``passes_rsi_guard``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def passes_ob_proximity(
    df_15m: pd.DataFrame,
    *,
    direction_long: bool,
    current_price: float,
    proximity_pct: float,
) -> bool:
    """Return True if the current price sits within ``proximity_pct`` of an
    order block matching ``direction_long``.

    Order blocks are detected over the last 30 bars (after a 10-bar
    burn-in) of the supplied 15m frame: an OB is the candle preceding a
    body-flip + impulsive move ≥ 0.5 × ATR-proxy. Bull OB = a down
    candle followed by an up candle followed by an impulsive move;
    bear OB = the opposite. The price-vs-OB check uses each OB's
    mid-price.

    Fail-open: if there are not enough bars to compute, the function
    returns True (matches the bot's "if df_15m is not None and len(...)
    >= 40" guard — the filter only blocks when it can confidently say
    no nearby OB exists).
    """
    if df_15m is None or len(df_15m) < 40:
        return True

    ob_window = df_15m.tail(40)
    opn = ob_window["open"].values
    close = ob_window["close"].values
    high = ob_window["high"].values
    low = ob_window["low"].values
    n = len(close)
    bull_obs: list[float] = []
    bear_obs: list[float] = []
    atr_proxy = float(np.mean(high - low) + 1e-10)
    for idx in range(max(0, n - 30), n - 2):
        body_i = close[idx] - opn[idx]
        body_i1 = close[idx + 1] - opn[idx + 1]
        move = abs(close[idx + 2] - close[idx + 1]) if (idx + 2) < n else 0.0
        if body_i < 0 and body_i1 > 0 and move > atr_proxy * 0.5:
            bull_obs.append((high[idx] + low[idx]) / 2.0)
        if body_i > 0 and body_i1 < 0 and move > atr_proxy * 0.5:
            bear_obs.append((high[idx] + low[idx]) / 2.0)

    ob_levels = bull_obs if direction_long else bear_obs
    return any(
        abs(current_price - lvl) / (lvl + 1e-10) < proximity_pct
        for lvl in ob_levels
    )


def passes_adx_directional(
    df_15m: pd.DataFrame,
    *,
    direction_long: bool,
    adx_guard_min: float,
    period: int = 14,
) -> bool:
    """Return True if ADX/DI on the last 30 15m bars confirms the trade
    direction (or if ADX is below the guard threshold, in which case
    the directional check is skipped — matches live behaviour).

    Computed exactly as in the live bot (live_trading_htf
    ``_get_structure_direction`` S5 branch):
      * +DM = max(high_diff, 0) when high_diff > -low_diff
      * -DM = max(-low_diff, 0) when -low_diff > high_diff
      * ATR = mean of true ranges
      * +DI, -DI as 100×mean(DM)/ATR
      * ADX = 100 × |+DI − −DI| / (+DI + −DI)

    Fail-open: too few bars, ATR = 0, or any computation error returns
    True (live bot logs and swallows the same exceptions).
    """
    if df_15m is None or len(df_15m) < 30:
        return True
    try:
        _close = df_15m["close"].values
        _high = df_15m["high"].values
        _low = df_15m["low"].values

        _plus_dm = np.diff(_high[-30:])
        _minus_dm = -np.diff(_low[-30:])
        _plus_dm = np.where(
            (_plus_dm > _minus_dm) & (_plus_dm > 0), _plus_dm, 0,
        )
        _minus_dm = np.where(
            (_minus_dm > _plus_dm) & (_minus_dm > 0), _minus_dm, 0,
        )
        _tr = np.maximum(
            _high[-29:] - _low[-29:],
            np.maximum(
                np.abs(_high[-29:] - _close[-30:-1]),
                np.abs(_low[-29:] - _close[-30:-1]),
            ),
        )
        _atr = float(np.mean(_tr[-period:]))
        if _atr <= 0:
            return True
        _plus_di = 100.0 * float(np.mean(_plus_dm[-period:])) / _atr
        _minus_di = 100.0 * float(np.mean(_minus_dm[-period:])) / _atr
        _adx_val = (
            100.0 * abs(_plus_di - _minus_di)
            / (_plus_di + _minus_di + 1e-10)
        )
        if _adx_val < adx_guard_min:
            return True  # below guard → directional check skipped
        if direction_long and _minus_di > _plus_di:
            return False
        if (not direction_long) and _plus_di > _minus_di:
            return False
        return True
    except Exception:  # noqa: BLE001 — live bot uses bare try/except too
        return True


# ─── Structure-first ADX hard block ────────────────────────────────────


def passes_structure_first_adx(
    df_15m: pd.DataFrame,
    *,
    adx_guard_min: float,
    period: int = 14,
) -> bool:
    """Return True if 15m ADX is at or above ``adx_guard_min``.

    Mirrors the structure-first ADX block in live execute_trade
    (live_trading_htf line ~3010). ADX is computed exactly as the
    live regime_detector does — Wilder's smoothing collapsed to a
    simple mean over the last `period` bars of DM/TR (the live
    regime_detector uses the same approximation in non-trending mode).
    """
    if df_15m is None or len(df_15m) < 30:
        return True  # fail-open, matches live behaviour
    try:
        _close = df_15m["close"].values
        _high = df_15m["high"].values
        _low = df_15m["low"].values
        _plus_dm = np.diff(_high[-30:])
        _minus_dm = -np.diff(_low[-30:])
        _plus_dm = np.where(
            (_plus_dm > _minus_dm) & (_plus_dm > 0), _plus_dm, 0,
        )
        _minus_dm = np.where(
            (_minus_dm > _plus_dm) & (_minus_dm > 0), _minus_dm, 0,
        )
        _tr = np.maximum(
            _high[-29:] - _low[-29:],
            np.maximum(
                np.abs(_high[-29:] - _close[-30:-1]),
                np.abs(_low[-29:] - _close[-30:-1]),
            ),
        )
        _atr = float(np.mean(_tr[-period:]))
        if _atr <= 0:
            return True
        _plus_di = 100.0 * float(np.mean(_plus_dm[-period:])) / _atr
        _minus_di = 100.0 * float(np.mean(_minus_dm[-period:])) / _atr
        _adx_val = (
            100.0 * abs(_plus_di - _minus_di)
            / (_plus_di + _minus_di + 1e-10)
        )
        return _adx_val >= adx_guard_min
    except Exception:  # noqa: BLE001
        return True


# ─── Exhaustion / momentum-extension filter ────────────────────────────


def passes_exhaustion_filter(
    df_15m: pd.DataFrame,
    *,
    current_price: float,
    threshold_atr: float,
    period: int = 14,
    window: int = 20,
) -> bool:
    """Return True if price is within ``threshold_atr`` ATRs of recent VWAP.

    Mirrors live execute_trade lines ~3030–3055. The 20-bar VWAP is
    computed from typical-price × volume; ATR is the mean of the last
    14 true-range bars within that 20-bar slice.

    Fail-open if fewer than ``window`` 15m bars are available, or if
    ATR collapses to 0 (zero-volume edge case).
    """
    if df_15m is None or len(df_15m) < window:
        return True
    try:
        closes = df_15m["close"].values[-window:]
        volumes = df_15m["volume"].values[-window:]
        highs = df_15m["high"].values[-window:]
        lows = df_15m["low"].values[-window:]

        typical_price = (highs + lows + closes) / 3.0
        vwap = float(np.sum(typical_price * volumes)
                     / (np.sum(volumes) + 1e-10))
        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - np.roll(closes, 1)),
                np.abs(lows - np.roll(closes, 1)),
            ),
        )
        atr = float(np.mean(tr[-period:]))
        if atr <= 0:
            return True
        extension = abs(current_price - vwap) / atr
        return extension <= threshold_atr
    except Exception:  # noqa: BLE001
        return True


# ─── RSI band guard ────────────────────────────────────────────────────


def passes_rsi_guard(
    df_15m: pd.DataFrame,
    *,
    direction_long: bool,
    ob_threshold: float,
    os_threshold: float,
    period: int = 14,
) -> bool:
    """Return True if 15m RSI does NOT block the trade.

    Live (live_trading_htf line ~1636): LONG is blocked when 15m
    RSI > ob_threshold (overbought); SHORT is blocked when RSI <
    os_threshold (oversold). Live computes RSI from the mtf signals
    bundle; the sim computes it directly from the 15m closes using
    Wilder's smoothing approximation (mean of gain / loss over period).

    Fail-open if fewer than `period+1` bars available.
    """
    if df_15m is None or len(df_15m) < period + 1:
        return True
    try:
        closes = df_15m["close"].values
        delta = np.diff(closes)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = float(np.mean(gain[-period:]))
        avg_loss = float(np.mean(loss[-period:]))
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        if direction_long and rsi > ob_threshold:
            return False
        if (not direction_long) and rsi < os_threshold:
            return False
        return True
    except Exception:  # noqa: BLE001
        return True
