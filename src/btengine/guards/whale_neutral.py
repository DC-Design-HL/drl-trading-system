"""WhaleNeutralGuard — block entries when whale flow is NEUTRAL.

Live's WHALE_NEUTRAL_GUARD reads `whale.direction` from the whale flow
service. NEUTRAL means no clear directional bias from on-chain whale
behavior in the trailing window; live blocks ALL entries (LONG or SHORT)
in that regime.

Backtest path: ctx.extras['whale_direction'] populated by the runner.
If absent, fail-open. Historical depth is limited (~Apr 6 onward), so
long-window backtests will mostly fail-open and not invoke this guard.
"""

from __future__ import annotations

from ..strategy.base import Guard, GuardResult, Intent


class WhaleNeutralGuard(Guard):
    name = "whale_neutral"

    def __call__(self, intent: Intent, ctx) -> GuardResult:
        if intent.action not in ("OPEN_LONG", "OPEN_SHORT"):
            return GuardResult.allow()
        direction = ctx.extras.get("whale_direction")
        if direction is None:
            return GuardResult.allow()
        if str(direction).upper() == "NEUTRAL":
            return GuardResult.block(
                f"whale_neutral: whale flow direction is NEUTRAL",
                whale_direction=direction,
            )
        return GuardResult.allow()
