"""ReverseCloseLongGuard — block REVERSE_CLOSE_LONG outside downtrends.

Live's asymmetric canary (deployed 2026-04-23, expanded to all 4 symbols
2026-05-01): when a LONG position is open and the strategy emits an
opposite-direction signal (OPEN_SHORT), the live bot would normally
flip via REVERSE_CLOSE_LONG. The canary BLOCKS this flip when:

  1. The symbol is in REVERSAL_BLOCK_LONG_CANARY_SYMBOLS
  2. BTC 4h EMA slope > REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT
     (i.e., not in a clear downtrend)

When blocked, the position stays open (still SL/TP-protected). Validated
on 8 days of XRP-only data (11 → 0 RC_LONG events) before expansion.

This is technically not an entry guard — it's an exit-side guard that
intercepts the reversal flow. But the guard interface is the same: it's
called when the strategy WOULD trigger a force_close. The runner is
responsible for invoking it at the right point.

Backtest path: the runner pre-computes BTC 4h slope at each ctx and
puts it on ctx.extras['btc_4h_slope_pct']. If absent (e.g., insufficient
BTC history for slope computation), fail-open with a 'no_slope' reason
matching live's degraded-mode behavior.
"""

from __future__ import annotations

from .. import live_constants as LC
from ..strategy.base import Guard, GuardResult, Intent


class ReverseCloseLongGuard(Guard):
    """Apply on REVERSE_CLOSE_LONG candidate intents only.

    The runner constructs an Intent(action='REVERSE_CLOSE_LONG') when it
    decides to flip a LONG into a SHORT. This guard either allows that
    flip to happen, or blocks it (causing the position to remain open).
    """
    name = "reverse_close_long"

    def __init__(self,
                 canary_symbols=None,
                 min_slope_pct: float = LC.REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT):
        self.canary_symbols = frozenset(
            canary_symbols if canary_symbols is not None
            else LC.REVERSAL_BLOCK_LONG_CANARY_SYMBOLS
        )
        self.min_slope_pct = float(min_slope_pct)

    def __call__(self, intent: Intent, ctx) -> GuardResult:
        if intent.action != "REVERSE_CLOSE_LONG":
            return GuardResult.allow()
        if ctx.symbol not in self.canary_symbols:
            return GuardResult.allow()
        slope = ctx.extras.get("btc_4h_slope_pct")
        if slope is None:
            # Fail-open on missing slope (matches live's transient-error path)
            return GuardResult.allow()
        if float(slope) > self.min_slope_pct:
            return GuardResult.block(
                f"reverse_close_long_canary: BTC slope {slope:+.2f}% > gate "
                f"{self.min_slope_pct:+.2f}% — flip blocked, position kept",
                btc_4h_slope_pct=float(slope),
            )
        return GuardResult.allow()
