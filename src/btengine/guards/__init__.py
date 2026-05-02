"""Guard components — block entries that violate a rule.

Each guard is a callable `f(ctx) -> GuardResult` where GuardResult is
either Allow() or Block(reason, signal_snapshot).

Loaded by the runner via `guards.enabled: [adx, usdtd, ...]` in config.

The audit (docs/backtest_redesign_proposal.md) found that 5 production
guards were missing from most legacy backtests:
  * symbol_blocklist (Apr 27 deploy)         — only 2/35 scripts
  * funding_long     (late Apr deploy)       — only 1/35 scripts
  * whale_neutral    (late Apr deploy)       — only 2/35 scripts
  * ext_pos_news     (Apr 30 deploy)         — only 4/35 scripts
  * reverse_close_long canary (Apr 23, exp May 1) — only 2/35 scripts

This package implements all 5 plus adx + usdtd (already covered by
several legacy scripts but in 9 different ways). Driven from
live_constants, so they stay in sync with live by construction.
"""

from .adx import ADXGuard
from .ext_pos_news import ExtPosNewsGuard
from .funding_long import FundingLongGuard
from .reverse_close_long import ReverseCloseLongGuard
from .symbol_blocklist import SymbolBlocklistGuard
from .usdt_d import USDTDGuard
from .whale_neutral import WhaleNeutralGuard


# Registry mapping config-string → Guard class
GUARD_CLASSES = {
    "adx": ADXGuard,
    "ext_pos_news": ExtPosNewsGuard,
    "funding_long": FundingLongGuard,
    "reverse_close_long": ReverseCloseLongGuard,
    "symbol_blocklist": SymbolBlocklistGuard,
    "usdtd": USDTDGuard,
    "whale_neutral": WhaleNeutralGuard,
}


def build_guard_chain(enabled, params):
    """Build a GuardChain from a list of enabled names and per-guard params."""
    from ..strategy.base import GuardChain
    guards = []
    for name in enabled:
        if name not in GUARD_CLASSES:
            raise ValueError(f"Unknown guard {name!r}. Known: {list(GUARD_CLASSES)}")
        cls = GUARD_CLASSES[name]
        kwargs = (params or {}).get(name, {}) or {}
        guards.append(cls(**kwargs))
    return GuardChain(guards)


__all__ = [
    "ADXGuard", "ExtPosNewsGuard", "FundingLongGuard", "ReverseCloseLongGuard",
    "SymbolBlocklistGuard", "USDTDGuard", "WhaleNeutralGuard",
    "GUARD_CLASSES", "build_guard_chain",
]
