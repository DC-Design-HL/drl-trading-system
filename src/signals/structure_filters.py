"""Pure helpers for the structure-first S5 entry filters.

Extracted from live_trading_htf.py per PROFITABILITY_PLAN.md §3/P2 so
the live bot and the forward simulator call the same code. Both
helpers are deterministic — same DataFrame in → same answer out — and
have no logging side effects.

Live behaviour preserved exactly: the bot's S5 block at
live_trading_htf._get_structure_direction now delegates to
``passes_ob_proximity`` and ``passes_adx_directional`` instead of
inlining the computations.
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
