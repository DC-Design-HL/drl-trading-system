"""
Global Portfolio Manager
Enforces cross-asset correlation limits and portfolio-wide risk rules.
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GlobalPortfolioManager:
    """
    Manages global portfolio risk across multiple independent trading agents.
    Prevents correlated wipeouts by limiting concurrent directional exposure.
    """
    
    def __init__(self, max_correlated_positions: int = 2):
        """
        Initialize the global portfolio manager.
        
        Args:
            max_correlated_positions: Maximum number of concurrent positions 
                                      allowed in the exact same direction.
        """
        self.max_correlated_positions = max_correlated_positions
        
        # Track active positions across all agents. Format: {symbol: direction (1 or -1)}
        self.active_positions: Dict[str, int] = {}
        
    def register_position(self, symbol: str, direction: int):
        """Register that an agent successfully opened a position."""
        if direction not in [1, -1]:
            return
            
        self.active_positions[symbol] = direction
        logger.info(
            f"🌐 GlobalPortfolioManager: Registered {symbol} {'LONG' if direction == 1 else 'SHORT'}. "
            f"Active Portfolio: {self.get_portfolio_summary()}"
        )
        
    def clear_position(self, symbol: str):
        """Clear a position when an agent exits a trade."""
        if symbol in self.active_positions:
            direction = self.active_positions.pop(symbol)
            logger.info(
                f"🌐 GlobalPortfolioManager: Cleared {symbol} {'LONG' if direction == 1 else 'SHORT'}. "
                f"Active Portfolio: {self.get_portfolio_summary()}"
            )
            
    def can_open_position(self, symbol: str, proposed_direction: int) -> bool:
        """
        Check if an agent is allowed to open a new position.
        
        Args:
            symbol: The asset wanting to trade (e.g. BTCUSDT)
            proposed_direction: 1 for LONG, -1 for SHORT
            
        Returns:
            True if the position is allowed under portfolio correlation limits.
        """
        if proposed_direction not in [1, -1]:
            return False
            
        # Count how many existing positions are in this exact same direction
        correlated_count = sum(
            1 for s, d in self.active_positions.items() 
            if d == proposed_direction and s != symbol
        )
        
        if correlated_count >= self.max_correlated_positions:
            logger.warning(
                f"🌐 GlobalPortfolioManager: BLOCKED proposed {symbol} {'LONG' if proposed_direction == 1 else 'SHORT'}. "
                f"Correlation Limit Reached ({correlated_count}/{self.max_correlated_positions} active). "
                f"Active Portfolio: {self.get_portfolio_summary()}"
            )
            return False
            
        return True
        
    def get_portfolio_summary(self) -> str:
        """Returns a string describing the current directional exposure."""
        if not self.active_positions:
            return "Flat"
            
        longs = [s for s, d in self.active_positions.items() if d == 1]
        shorts = [s for s, d in self.active_positions.items() if d == -1]
        
        summary = []
        if longs:
            summary.append(f"LONGs: {','.join(longs)}")
        if shorts:
            summary.append(f"SHORTs: {','.join(shorts)}")
            
        return " | ".join(summary)
