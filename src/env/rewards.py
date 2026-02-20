"""
Reward Functions Module
Risk-adjusted reward calculations using Sharpe and Sortino ratios.
"""

import numpy as np
from typing import List, Optional
from collections import deque


class RewardCalculator:
    """
    Calculates risk-adjusted rewards for the trading environment.
    Combines Sharpe and Sortino ratios to punish volatility and drawdowns.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        risk_free_rate: float = 0.0,
        sharpe_weight: float = 0.6,
        sortino_weight: float = 0.4,
        drawdown_penalty: float = 2.0,
        holding_cost: float = 0.0001,
    ):
        """
        Initialize the reward calculator.
        
        Args:
            window_size: Rolling window for ratio calculations
            risk_free_rate: Daily risk-free rate (default 0)
            sharpe_weight: Weight for Sharpe ratio component
            sortino_weight: Weight for Sortino ratio component
            drawdown_penalty: Multiplier for drawdown penalty
            holding_cost: Cost per step for holding a position (encourages action)
        """
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight
        self.drawdown_penalty = drawdown_penalty
        self.holding_cost = holding_cost
        
        # Rolling returns buffer
        self.returns_buffer: deque = deque(maxlen=window_size)
        self.peak_value: float = 0.0
        self.current_drawdown: float = 0.0
        
    def reset(self, initial_value: float = 10000.0):
        """Reset the calculator for a new episode."""
        self.returns_buffer.clear()
        self.peak_value = initial_value
        self.current_drawdown = 0.0
        
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio from returns array.
        
        Sharpe = (mean_return - rf) / std_return
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-8:
            return 0.0
            
        return (mean_return - self.risk_free_rate) / std_return
    
    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino ratio from returns array.
        Only penalizes downside volatility.
        
        Sortino = (mean_return - rf) / downside_std
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) < 2:
            # No downside volatility - return high score
            return mean_return * 10 if mean_return > 0 else 0.0
            
        downside_std = np.std(negative_returns)
        
        if downside_std < 1e-8:
            return 0.0
            
        return (mean_return - self.risk_free_rate) / downside_std
    
    def update_drawdown(self, current_value: float) -> float:
        """
        Update and return current drawdown.
        """
        if current_value > self.peak_value:
            self.peak_value = current_value
            
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        return self.current_drawdown
    
    def calculate_reward(
        self,
        step_return: float,
        portfolio_value: float,
        position: int,
        action_taken: int,
        trade_pnl: Optional[float] = None,
    ) -> float:
        """
        Calculate the total reward for a step.
        
        Args:
            step_return: The return for this step (pct change)
            portfolio_value: Current total portfolio value
            position: Current position (-1, 0, 1)
            action_taken: Action taken this step (0=hold, 1=buy, 2=sell)
            trade_pnl: P&L from closed trade if any
            
        Returns:
            Total reward value
        """
        # Add return to buffer
        self.returns_buffer.append(step_return)
        
        # Update drawdown
        drawdown = self.update_drawdown(portfolio_value)
        
        # Get returns array
        returns = np.array(self.returns_buffer)
        
        # Calculate risk-adjusted metrics
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        
        # Combine ratios
        risk_adjusted_reward = (
            self.sharpe_weight * sharpe + 
            self.sortino_weight * sortino
        )
        
        # Apply drawdown penalty
        drawdown_penalty = self.drawdown_penalty * drawdown
        
        # Holding cost (small penalty for holding positions)
        holding_penalty = self.holding_cost if position != 0 else 0
        
        # Bonus for profitable closed trades
        trade_bonus = 0.0
        if trade_pnl is not None:
            # Scale by magnitude of profit/loss
            trade_bonus = np.sign(trade_pnl) * np.log1p(abs(trade_pnl) * 100)
        
        # Raw return component (small weight to maintain gradient)
        return_component = step_return * 10  # Scale up small returns
        
        # Total reward
        total_reward = (
            risk_adjusted_reward +
            return_component +
            trade_bonus -
            drawdown_penalty -
            holding_penalty
        )
        
        # Clip to prevent extreme values
        return np.clip(total_reward, -10.0, 10.0)
    
    def get_episode_metrics(self) -> dict:
        """
        Get summary metrics for the episode.
        """
        returns = np.array(self.returns_buffer) if self.returns_buffer else np.array([0.0])
        
        return {
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.current_drawdown,
            'total_return': np.prod(1 + returns) - 1 if len(returns) > 0 else 0.0,
            'volatility': np.std(returns) if len(returns) > 1 else 0.0,
            'num_steps': len(self.returns_buffer),
        }


def create_reward_calculator(config: dict) -> RewardCalculator:
    """Factory function to create RewardCalculator from config."""
    return RewardCalculator(
        window_size=config.get('window_size', 20),
        risk_free_rate=config.get('risk_free_rate', 0.0),
        sharpe_weight=config.get('sharpe_weight', 0.6),
        sortino_weight=config.get('sortino_weight', 0.4),
        drawdown_penalty=config.get('drawdown_penalty', 2.0),
        holding_cost=config.get('holding_cost', 0.0001),
    )
