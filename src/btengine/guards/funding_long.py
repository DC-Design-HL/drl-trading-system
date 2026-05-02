"""FundingLongGuard — block LONG when funding rate is too positive.

Funding rate > threshold means LONG positions pay SHORTs each funding
window. The live bot blocks LONG entries above FUNDING_LONG_GUARD_MAX.

For backtest, funding history is provided via `ctx.extras['funding_rate']`
(populated by the runner from a per-symbol funding cache). If absent,
fail-open.
"""

from __future__ import annotations

from .. import live_constants as LC
from ..strategy.base import Guard, GuardResult, Intent


class FundingLongGuard(Guard):
    name = "funding_long"

    def __init__(self, max_funding: float = LC.FUNDING_LONG_GUARD_MAX):
        self.max_funding = float(max_funding)

    def __call__(self, intent: Intent, ctx) -> GuardResult:
        if intent.action != "OPEN_LONG":
            return GuardResult.allow()
        rate = ctx.extras.get("funding_rate")
        if rate is None:
            return GuardResult.allow()
        # Live constant is in % units; if extras provides raw decimal, normalize
        # The raw funding rate from Binance is already a small decimal (e.g. 0.0001)
        # Live's max is 0.05 (% — 0.05%). To compare apples-to-apples treat both as %.
        rate_pct = float(rate) * 100.0  # decimal → %
        if rate_pct > self.max_funding:
            return GuardResult.block(
                f"funding_long: rate {rate_pct:.4f}% > max {self.max_funding}%",
                funding_rate_pct=rate_pct,
            )
        return GuardResult.allow()
