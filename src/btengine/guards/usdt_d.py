"""USDTDGuard — block LONG when USDT dominance is rising.

Live's logic: synthetic proxy via the 4-symbol crypto basket. If the
basket dropped more than USDT_D_THRESHOLD_PCT over the lookback window,
USDT.D is treated as rising (capital fleeing crypto into stables) and
LONG entries are suppressed.

This guard requires the runner to populate `ctx.extras['basket_change_pct']`
from a portfolio-level computation each bar (since each ctx is per-symbol
but the basket is across symbols). If that key is absent, the guard
fails open (allows).
"""

from __future__ import annotations

from .. import live_constants as LC
from ..strategy.base import Guard, GuardResult, Intent


class USDTDGuard(Guard):
    name = "usdtd"

    def __init__(self, threshold_pct: float = LC.USDT_D_THRESHOLD_PCT,
                 lookback_hours: int = LC.USDT_D_LOOKBACK_HOURS):
        self.threshold_pct = float(threshold_pct)
        self.lookback_hours = int(lookback_hours)

    def __call__(self, intent: Intent, ctx) -> GuardResult:
        # Only blocks LONG entries
        if intent.action != "OPEN_LONG":
            return GuardResult.allow()
        change_pct = ctx.extras.get("basket_change_pct")
        if change_pct is None:
            return GuardResult.allow()
        if change_pct < -self.threshold_pct:
            return GuardResult.block(
                f"usdtd: basket dropped {change_pct:+.2f}% (>{self.threshold_pct}% rise) — LONG suppressed",
                basket_change_pct=change_pct,
            )
        return GuardResult.allow()
