"""
Confidence Engine (Phase 11.6)

Scales trade position sizes based on the predictive uncertainty
of the ensemble agents. 

Calculates a Confidence Multiplier based on:
1. Ensemble Agreement (HMM probability alignment)
2. Historical Win Rate of similar confidence trades
"""

import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """
    Translates model confidence into position sizing multipliers.
    """
    
    def __init__(
        self,
        min_multiplier: float = 0.25,
        max_multiplier: float = 2.0,
        baseline_confidence: float = 0.5,
    ):
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.baseline_confidence = baseline_confidence
        
        # Track historical confidence vs outcome
        self.confidence_history: List[float] = []
        self.outcome_history: List[float] = []
        
    def get_position_multiplier(self, raw_confidence: float) -> float:
        """
        Convert raw model confidence [0.0 - 1.0] to a position multiplier.
        
        Args:
            raw_confidence: The agreement score from the EnsembleOrchestrator
            
        Returns:
            multiplier (float): to be multiplied with base position size
        """
        # If confidence is below baseline, scale down linearly
        if raw_confidence < self.baseline_confidence:
            scale = raw_confidence / self.baseline_confidence
            multiplier = self.min_multiplier + (1.0 - self.min_multiplier) * scale
        
        # If confidence is above baseline, scale up
        else:
            scale = (raw_confidence - self.baseline_confidence) / (1.0 - self.baseline_confidence)
            multiplier = 1.0 + (self.max_multiplier - 1.0) * (scale ** 1.5)  # Exponential scale up
            
        return np.clip(multiplier, self.min_multiplier, self.max_multiplier)
        
    def apply_confidence(self, base_size: float, raw_confidence: float) -> float:
        """Calculate final position size."""
        multiplier = self.get_position_multiplier(raw_confidence)
        
        # Base size * multiplier
        final_size = base_size * multiplier
        
        logger.info(
            f"🧠 Confidence Engine: Raw={raw_confidence:.2f} -> "
            f"Mult={multiplier:.2f}x -> Size={base_size:.2%} to {final_size:.2%}"
        )
        return final_size

    def record_outcome(self, confidence: float, pnl_pct: float):
        """Record trade outcome to map confidence correlations later."""
        self.confidence_history.append(confidence)
        self.outcome_history.append(pnl_pct)
        
        if len(self.confidence_history) > 1000:
            self.confidence_history.pop(0)
            self.outcome_history.pop(0)
            
    def get_confidence_reliability(self) -> float:
        """Calculate Pearson correlation between confidence and P&L."""
        if len(self.confidence_history) < 20:
            return 0.0
            
        try:
            corr = np.corrcoef(self.confidence_history, self.outcome_history)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0


# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ConfidenceEngine()
    
    print("-" * 40)
    print("Confidence Multiplier Curve")
    print("-" * 40)
    for conf in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        mult = engine.get_position_multiplier(conf)
        print(f"Conf: {conf:.2f} -> Mult: {mult:.2f}x")
