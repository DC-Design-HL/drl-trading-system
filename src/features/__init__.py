"""
Features Module

Advanced feature engineering for trading:
- Ultimate Feature Engine (Wyckoff, SMC, Market Structure)
- Multi-Asset Correlation Engine
"""

from .ultimate_features import (
    UltimateFeatureEngine,
    WyckoffAnalyzer,
    SMCAnalyzer,
    MarketStructureAnalyzer,
    VolumeProfileAnalyzer,
)

from .correlation_engine import (
    CorrelationEngine,
    SimulatedDominanceEngine,
)

__all__ = [
    'UltimateFeatureEngine',
    'WyckoffAnalyzer',
    'SMCAnalyzer',
    'MarketStructureAnalyzer',
    'VolumeProfileAnalyzer',
    'CorrelationEngine',
    'SimulatedDominanceEngine',
]
