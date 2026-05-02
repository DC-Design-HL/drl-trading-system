"""Guard components — block entries that violate a rule.

Each guard is a callable `f(ctx) -> GuardResult` where GuardResult is
either Allow() or Block(reason, signal_snapshot).

Loaded by the runner via `guards.enabled: [adx, usdtd, ...]` in config.
"""
