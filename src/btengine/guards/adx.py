"""ADXGuard — block entries outside [min_adx, max_adx] range.

Replaces 9 different ADX implementations the audit found. Single source
of truth via strategy.indicators.adx_wilder().
"""

from __future__ import annotations

import numpy as np

from .. import live_constants as LC
from ..strategy.base import Guard, GuardResult, Intent
from ..strategy.indicators import adx_wilder


class ADXGuard(Guard):
    name = "adx"

    def __init__(self, min_adx: float = LC.ADX_GUARD_MIN,
                 max_adx: float = LC.ADX_GUARD_MAX,
                 period: int = 14):
        self.min_adx = float(min_adx)
        self.max_adx = float(max_adx)
        self.period = int(period)

    def __call__(self, intent: Intent, ctx) -> GuardResult:
        if intent.action not in ("OPEN_LONG", "OPEN_SHORT"):
            return GuardResult.allow()
        if len(ctx.primary) < self.period * 3:
            # Not enough history to compute reliable ADX → fail-open (allow)
            return GuardResult.allow()
        try:
            series = adx_wilder(ctx.primary, period=self.period).dropna()
            if len(series) == 0:
                return GuardResult.allow()
            cur = float(series.iloc[-1])
        except Exception:
            return GuardResult.allow()
        if cur < self.min_adx:
            return GuardResult.block(
                f"adx: {cur:.1f} < min {self.min_adx} (ranging)",
                adx=cur,
            )
        if cur > self.max_adx:
            return GuardResult.block(
                f"adx: {cur:.1f} > max {self.max_adx} (overheated)",
                adx=cur,
            )
        return GuardResult.allow()
