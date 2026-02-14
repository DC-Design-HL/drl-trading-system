"""
Backtest Layer Package
Historical backtesting engine.
"""

from .data_loader import DataLoader

def get_backtest_engine():
    """Lazy import to avoid pulling in heavy env dependencies."""
    from .engine import BacktestEngine
    return BacktestEngine

__all__ = ['DataLoader', 'get_backtest_engine']
