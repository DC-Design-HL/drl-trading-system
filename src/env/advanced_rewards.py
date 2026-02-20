"""
Advanced Reward Function
Profit-factor optimized, regime-aware reward calculation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AdvancedRewardCalculator:
    """
    Sophisticated reward function designed to maximize profitability.
    
    Key innovations:
    1. Profit factor reward (gross profit / gross loss)
    2. Win rate bonus
    3. Regime-aware rewards (trend-following vs mean-reversion)
    4. Trade quality scoring
    5. Holding period optimization
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        profit_factor_weight: float = 0.4,
        win_rate_weight: float = 0.3,
        sharpe_weight: float = 0.3,
        min_trades_for_stats: int = 10,
        holding_bonus_steps: int = 4,  # Encourage holding for at least 4 hours
    ):
        self.initial_balance = initial_balance
        self.profit_factor_weight = profit_factor_weight
        self.win_rate_weight = win_rate_weight
        self.sharpe_weight = sharpe_weight
        self.min_trades_for_stats = min_trades_for_stats
        self.holding_bonus_steps = holding_bonus_steps
        
        self.reset(initial_balance)
        
    def reset(self, initial_balance: float = None):
        """Reset for new episode."""
        if initial_balance:
            self.initial_balance = initial_balance
            
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        
        # Trade tracking
        self.trades: List[Dict] = []
        self.current_position_start = None
        self.current_position_type = 0
        self.steps_in_position = 0
        
        # Running statistics
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Regime tracking
        self.recent_returns = deque(maxlen=24)  # Last 24 hours
        
    def calculate_reward(
        self,
        step_return: float,
        portfolio_value: float,
        position: int,
        action_taken: int,
        trade_pnl: Optional[float] = None,
        market_features: Optional[Dict] = None,
    ) -> float:
        """
        Calculate sophisticated reward.
        
        Args:
            step_return: Return this step from holding position
            portfolio_value: Current portfolio value
            position: Current position (-1, 0, 1)
            action_taken: Action taken (0=hold, 1=buy, 2=sell)
            trade_pnl: P&L from closed trade if any
            market_features: Optional market features for regime-aware rewards
            
        Returns:
            Reward value
        """
        self.portfolio_values.append(portfolio_value)
        self.returns.append(step_return)
        self.recent_returns.append(step_return)
        
        reward = 0.0
        
        # === 1. BASE RETURN REWARD ===
        # Reward for making money on the step
        reward += step_return * 100  # Scale up returns
        
        # === 2. TRADE P&L REWARD ===
        if trade_pnl is not None:
            self._record_trade(trade_pnl)
            reward += self._calculate_trade_reward(trade_pnl)
            
        # === 3. POSITION MANAGEMENT REWARD ===
        reward += self._calculate_position_reward(position, action_taken, step_return, market_features)
        
        # === 4. RISK MANAGEMENT REWARD ===
        reward += self._calculate_risk_reward(portfolio_value)
        
        # === 5. PROFIT FACTOR BONUS ===
        if len(self.trades) >= self.min_trades_for_stats:
            profit_factor = self._calculate_profit_factor()
            if profit_factor > 1.5:
                reward += 0.1 * (profit_factor - 1)  # Bonus for high profit factor
            elif profit_factor < 0.5:
                reward -= 0.2  # Penalty for poor profit factor
                
        # === 6. WIN RATE BONUS ===
        win_rate = self._calculate_win_rate()
        if self.winning_trades + self.losing_trades >= self.min_trades_for_stats:
            if win_rate > 0.55:
                reward += 0.1 * (win_rate - 0.5)
            elif win_rate < 0.4:
                reward -= 0.1
                
        # Track position duration
        if position != 0:
            self.steps_in_position += 1
        else:
            self.steps_in_position = 0
            
        return reward
    
    def _record_trade(self, pnl: float):
        """Record a completed trade."""
        self.trades.append({
            'pnl': pnl,
            'steps_held': self.steps_in_position,
        })
        
        if pnl > 0:
            self.gross_profit += pnl
            self.winning_trades += 1
        else:
            self.gross_loss += abs(pnl)
            self.losing_trades += 1
            
    def _calculate_trade_reward(self, pnl: float) -> float:
        """Calculate reward for a closed trade."""
        reward = 0.0
        
        # Base P&L reward (normalized by initial balance)
        pnl_pct = pnl / self.initial_balance
        reward += pnl_pct * 50  # Strong weight on trade outcomes
        
        # Bonus for held positions (encourage patience)
        if pnl > 0 and self.steps_in_position >= self.holding_bonus_steps:
            reward += 0.2  # Bonus for profitable patient trades
            
        # Penalty for very short losing trades (overtrading)
        if pnl < 0 and self.steps_in_position < 2:
            reward -= 0.3  # Discourage quick losing trades
            
        # Quality trade bonus (good risk/reward)
        if pnl > 0:
            # Estimate risk taken (assume 2% stop loss)
            estimated_risk = 0.02 * self.initial_balance
            risk_reward = pnl / estimated_risk
            if risk_reward > 2:
                reward += 0.2  # Bonus for 2:1+ R:R trades
                
        return reward
    
    def _calculate_position_reward(
        self,
        position: int,
        action: int,
        step_return: float,
        market_features: Optional[Dict],
    ) -> float:
        """Calculate reward for position management."""
        reward = 0.0
        
        # === SMART ENTRY REWARD ===
        if action in [1, 2]:  # Buy or Sell
            # Entering a position
            if market_features:
                # Reward for trading with the trend
                trend = market_features.get('trend_strength', 0.5)
                if action == 1 and trend > 0.6:  # Long in uptrend
                    reward += 0.15
                elif action == 2 and trend < 0.4:  # Short in downtrend
                    reward += 0.15
                    
                # Reward for trading at good levels
                position_in_range = market_features.get('position_in_range', 0.5)
                if action == 1 and position_in_range < 0.3:  # Long near support
                    reward += 0.1
                elif action == 2 and position_in_range > 0.7:  # Short near resistance
                    reward += 0.1
                    
                # Reward for trading on volume confirmation
                vol_ratio = market_features.get('vol_ratio_24h', 1.0)
                if vol_ratio > 1.5:
                    reward += 0.05  # Good volume
                    
        # === HOLDING REWARD ===
        if action == 0 and position != 0:  # Holding a position
            # Reward for profitable holds
            if step_return > 0:
                reward += 0.05
            # Small penalty for holding through losses (stop loss should trigger)
            elif step_return < -0.01:  # 1% loss in a single step
                reward -= 0.1
                
        # === FLAT CASH PENALTY ===
        if position == 0:
            # Small penalty for being flat (opportunity cost)
            # But only if market is trending
            if market_features:
                adx = market_features.get('adx', 0)
                if adx > 30:  # Strong trend, should be in a position
                    reward -= 0.02
                    
        return reward
    
    def _calculate_risk_reward(self, portfolio_value: float) -> float:
        """Calculate reward for risk management."""
        reward = 0.0
        
        # Drawdown penalty
        peak_value = max(self.portfolio_values)
        current_drawdown = (peak_value - portfolio_value) / peak_value
        
        if current_drawdown > 0.10:  # 10% drawdown
            reward -= 0.5 * current_drawdown  # Heavy penalty
        elif current_drawdown > 0.05:  # 5% drawdown
            reward -= 0.2 * current_drawdown
            
        # Equity curve bonus (making new highs)
        if portfolio_value > peak_value:
            reward += 0.1
            
        return reward
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if self.gross_loss == 0:
            return 2.0 if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.5
        return self.winning_trades / total
    
    def get_episode_metrics(self) -> Dict:
        """Get comprehensive metrics for the episode."""
        portfolio_values = np.array(self.portfolio_values)
        returns = np.array(self.returns) if self.returns else np.array([0])
        
        # Sharpe ratio (annualized for hourly data)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)  # Annualized
        else:
            sharpe = 0.0
            
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(8760)
        else:
            sortino = sharpe
            
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
        
        # Total return
        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'profit_factor': self._calculate_profit_factor(),
            'win_rate': self._calculate_win_rate(),
            'total_trades': len(self.trades),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'avg_trade_pnl': sum(t['pnl'] for t in self.trades) / max(len(self.trades), 1),
            'avg_hold_time': sum(t['steps_held'] for t in self.trades) / max(len(self.trades), 1),
        }
