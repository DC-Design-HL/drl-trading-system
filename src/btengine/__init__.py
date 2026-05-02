"""
btengine — unified backtest framework for the DRL trading system.

Replaces the sprawl of ~45 one-off backtest_*.py scripts with a single
composable engine. See docs/backtest_redesign_proposal.md for the
audit findings + full design rationale.

Naming note: this lives at `src/btengine/` (not `src/backtest/`) because
the latter is occupied by the legacy DRL-agent backtest engine used by
training scripts. The two coexist; over time the legacy engine may be
absorbed here, but Phase 1 keeps them isolated.

Public surface:
    from src.btengine import BacktestRunner, BacktestConfig
    from src.btengine.strategy import Strategy, EntryRule, GuardChain, ExitPolicy
"""

__all__ = ["BacktestRunner", "BacktestConfig"]


def __getattr__(name):
    # Lazy import so importing the package doesn't pull in pandas/pyarrow
    # for callers that just want a type symbol.
    if name == "BacktestRunner":
        from .runner import BacktestRunner
        return BacktestRunner
    if name == "BacktestConfig":
        from .config import BacktestConfig
        return BacktestConfig
    raise AttributeError(name)
