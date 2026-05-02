"""SymbolBlocklistGuard — blocks specific (symbol, side) entries.

Matches live's SYMBOL_SIDE_BLOCKLIST (deployed Apr 27). The audit found
this guard was MISSING from 17/35 legacy backtests. By driving it from
live_constants, every btengine run automatically respects the same set.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from .. import live_constants as LC
from ..strategy.base import Guard, GuardResult, Intent


class SymbolBlocklistGuard(Guard):
    name = "symbol_blocklist"

    def __init__(self, blocklist: Iterable[Tuple[str, str]] | None = None):
        # Default: live's blocklist. Tests/configs can override.
        self.blocklist = frozenset(
            blocklist if blocklist is not None else LC.SYMBOL_SIDE_BLOCKLIST
        )

    def __call__(self, intent: Intent, ctx) -> GuardResult:
        if intent.action not in ("OPEN_LONG", "OPEN_SHORT"):
            return GuardResult.allow()
        side = "LONG" if intent.action == "OPEN_LONG" else "SHORT"
        if (ctx.symbol, side) in self.blocklist:
            return GuardResult.block(
                f"symbol_blocklist: ({ctx.symbol}, {side}) is blocked",
                symbol=ctx.symbol, side=side,
            )
        return GuardResult.allow()
