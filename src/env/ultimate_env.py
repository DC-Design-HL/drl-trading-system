"""
Ultimate Trading Environment

Enhanced trading environment with:
- Ultimate Feature Engine (150+ features)
- Multi-asset correlation
- Advanced reward shaping
- Regime-aware training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any
import logging

from src.features.ultimate_features import UltimateFeatureEngine
from src.features.correlation_engine import CorrelationEngine, SimulatedDominanceEngine

logger = logging.getLogger(__name__)


class UltimateTradingEnv(gym.Env):
    """
    Ultimate Trading Environment for DRL.
    
    Features:
    - 150+ advanced features (Wyckoff, SMC, correlations)
    - Enhanced reward shaping
    - Position sizing
    - Risk management integration
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        lookback_window: int = 48,
        trading_fee: float = 0.0004,
        position_size: float = 0.25,
        max_position: int = 1,
        use_correlations: bool = False,
        reward_scaling: float = 1.0,
        stop_loss_pct: float = 0.025,  # 2.5% stop loss (matches live)
        take_profit_pct: float = 0.05,  # 5% take profit (matches live, 2:1 R:R)
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.trading_fee = trading_fee
        self.position_size = position_size
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Feature engines
        self.feature_engine = UltimateFeatureEngine()
        self.correlation_engine = CorrelationEngine() if use_correlations else None
        self.dominance_engine = SimulatedDominanceEngine()
        
        # Precompute all features
        logger.info("Computing ultimate features...")
        self._precompute_features()
        
        # Action space: 0=Hold, 1=Buy/Long, 2=Sell/Short
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.num_features = self.features.shape[1]
        # Features + position info (3 values: position, unrealized_pnl, balance_ratio)
        obs_dim = self.num_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        logger.info(f"Ultimate Environment initialized with {self.num_features} features")
        
        # Episode state
        self.reset()
        
    def _precompute_features(self):
        """Precompute all features for the dataset."""
        # Get ultimate features
        all_features = self.feature_engine.get_all_features(self.df)
        
        # Add simulated dominance features
        dominance_features = self.dominance_engine.compute_simulated_dominance(self.df)
        all_features.update(dominance_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Handle NaN and inf
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Clip extreme values
        for col in features_df.columns:
            if features_df[col].dtype in [np.float64, np.float32]:
                features_df[col] = features_df[col].clip(-10, 10)
        
        self.features = features_df.values.astype(np.float32)
        self.feature_names = list(features_df.columns)
        
        logger.info(f"Precomputed {len(self.feature_names)} features")
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Random or fixed start
        if options and options.get('random_start', True):
            max_start = len(self.df) - self.lookback_window - 100
            self.current_step = self.np_random.integers(self.lookback_window, max(self.lookback_window + 1, max_start))
        else:
            self.current_step = self.lookback_window
            
        # Trading state
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: flat, 1: long
        self.position_price = 0.0
        self.position_size_units = 0.0
        self.steps_since_trade = 0  # Counter to penalize frequent trading
        self.position_entry_step = 0  # Track when position was opened
        
        # Metrics
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.returns = []
        self.max_balance = self.initial_balance
        
        return self._get_observation(), {}
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get precomputed features for current step
        features = self.features[self.current_step].copy()
        
        # Add position info
        current_price = self.df.iloc[self.current_step]['close']
        
        # Unrealized P&L (normalized)
        if self.position != 0:
            if self.position == 1:  # Long
                unrealized_pnl = (current_price - self.position_price) / self.position_price
            else:  # Short
                unrealized_pnl = (self.position_price - current_price) / self.position_price
        else:
            unrealized_pnl = 0.0
            
        # Balance ratio (normalized change from initial)
        balance_ratio = (self.balance - self.initial_balance) / self.initial_balance
        
        # Position info: [position, unrealized_pnl, balance_ratio]
        position_info = np.array([
            self.position,
            np.clip(unrealized_pnl, -0.5, 0.5),  # Clip extreme unrealized P&L
            np.clip(balance_ratio, -0.5, 0.5),
        ], dtype=np.float32)
        
        # Combine features and position info
        observation = np.concatenate([features, position_info])
        
        return observation.astype(np.float32)
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        current_price = self.df.iloc[self.current_step]['close']
        previous_balance = self.balance
        
        reward = 0.0
        trade_made = False
        
        # Detect trend using simple price change over lookback
        lookback = min(24, self.current_step)  # 24-period trend
        if lookback > 0 and self.current_step >= lookback:
            past_price = self.df.iloc[self.current_step - lookback]['close']
            trend_pct = (current_price - past_price) / past_price
            trend = 1 if trend_pct > 0.01 else (-1 if trend_pct < -0.01 else 0)
        else:
            trend = 0
        
        # Calculate minimum hold penalty (penalize flipping too fast)
        steps_in_position = self.current_step - self.position_entry_step
        min_hold_steps = 6  # Minimum 6 hours hold to reduce churn
        
        # Execute action
        if action == 1:  # Buy / Go Long
            if self.position == -1:  # Close short first
                # Stronger penalty for closing position too early
                if steps_in_position < min_hold_steps:
                    reward -= 0.02 * self.reward_scaling
                pnl = self._close_position(current_price)
                reward += pnl * self.reward_scaling
                trade_made = True
                
            if self.position == 0:  # Open long
                # Stronger trend alignment bonus
                if trend == 1:  # Bullish trend - good long entry
                    reward += 0.005 * self.reward_scaling
                elif trend == -1:  # Bearish trend - significant penalty
                    reward -= 0.01 * self.reward_scaling
                    
                self._open_position(current_price, 1)
                self.position_entry_step = self.current_step
                reward -= 0.003 * self.reward_scaling  # Trade cost
                trade_made = True
                self.steps_since_trade = 0
                
        elif action == 2:  # Sell / Go Short
            if self.position == 1:  # Close long first
                # Stronger penalty for closing position too early
                if steps_in_position < min_hold_steps:
                    reward -= 0.02 * self.reward_scaling
                pnl = self._close_position(current_price)
                reward += pnl * self.reward_scaling
                trade_made = True
                
            if self.position == 0:  # Open short
                # Stronger trend alignment bonus
                if trend == -1:  # Bearish trend - good short entry
                    reward += 0.005 * self.reward_scaling
                elif trend == 1:  # Bullish trend - significant penalty
                    reward -= 0.01 * self.reward_scaling
                    
                self._open_position(current_price, -1)
                self.position_entry_step = self.current_step
                reward -= 0.003 * self.reward_scaling  # Trade cost
                trade_made = True
                self.steps_since_trade = 0
        
        # Increment step counter
        self.steps_since_trade += 1
                
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        # Check SL/TP if in position
        if self.position != 0 and not done:
            new_price = self.df.iloc[self.current_step]['close']
            
            # Calculate P&L percentage
            if self.position == 1:  # Long
                pnl_pct = (new_price - self.position_price) / self.position_price
            else:  # Short
                pnl_pct = (self.position_price - new_price) / self.position_price
            
            # Check Stop Loss
            if pnl_pct <= -self.stop_loss_pct:
                pnl = self._close_position(new_price)
                reward += pnl * self.reward_scaling
                reward -= 0.05 * self.reward_scaling  # STRONG penalty for hitting SL
                trade_made = True
                
            # Check Take Profit
            elif pnl_pct >= self.take_profit_pct:
                pnl = self._close_position(new_price)
                reward += pnl * self.reward_scaling
                reward += 0.10 * self.reward_scaling  # Big bonus for hitting TP!
                trade_made = True
                
            else:
                # Unrealized P&L tracking (encourages holding winners)
                price_change = pnl_pct
                reward += price_change * self.position_size * 0.3 * self.reward_scaling
        
        # Track equity
        equity = self._calculate_equity()
        self.equity_curve.append(equity)
        
        if equity > self.max_balance:
            self.max_balance = equity
            
        # Drawdown penalty (earlier and stronger)
        drawdown = (self.max_balance - equity) / self.max_balance
        if drawdown > 0.05:  # Penalize > 5% drawdown (was 15%)
            reward -= drawdown * 0.1 * self.reward_scaling
            
        # Get observation
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            'balance': self.balance,
            'equity': equity,
            'position': self.position,
            'trade_made': trade_made,
        }
        
        return obs, reward, done, False, info
        
    def _open_position(self, price: float, direction: int):
        """Open a position."""
        trade_amount = self.balance * self.position_size
        fee = trade_amount * self.trading_fee
        
        self.position = direction
        self.position_price = price
        self.position_size_units = (trade_amount - fee) / price
        self.balance -= fee
        
    def _close_position(self, price: float) -> float:
        """Close position and return P&L ratio."""
        if self.position == 0:
            return 0.0
            
        if self.position == 1:  # Long
            pnl = (price - self.position_price) * self.position_size_units
        else:  # Short
            pnl = (self.position_price - price) * self.position_size_units
            
        # Apply trading fee
        fee = abs(pnl) * self.trading_fee if pnl > 0 else 0
        pnl -= fee
        
        self.balance += pnl
        
        # Record trade
        self.trades.append({
            'direction': 'long' if self.position == 1 else 'short',
            'entry': self.position_price,
            'exit': price,
            'pnl': pnl,
            'pnl_pct': pnl / (self.position_price * self.position_size_units),
        })
        
        # Reset position
        self.position = 0
        self.position_price = 0.0
        self.position_size_units = 0.0
        
        return pnl / self.initial_balance  # Return normalized P&L
        
    def _calculate_equity(self) -> float:
        """Calculate current equity (balance + unrealized P&L)."""
        if self.position == 0:
            return self.balance
            
        current_price = self.df.iloc[self.current_step]['close']
        
        if self.position == 1:  # Long
            unrealized = (current_price - self.position_price) * self.position_size_units
        else:  # Short
            unrealized = (self.position_price - current_price) * self.position_size_units
            
        return self.balance + unrealized
        
    def get_episode_metrics(self) -> Dict[str, float]:
        """Get episode performance metrics."""
        equity_curve = np.array(self.equity_curve)
        
        # Total return
        total_return = (equity_curve[-1] - self.initial_balance) / self.initial_balance
        
        # Returns for Sharpe calculation
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
        else:
            returns = np.array([0.0])
            
        # Sharpe ratio (annualized for hourly data)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0
            
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(24 * 365)
        else:
            sortino = 0.0
            
        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if t['pnl'] > 0)
            win_rate = wins / len(self.trades)
            avg_trade_pnl = np.mean([t['pnl'] for t in self.trades])
        else:
            win_rate = 0.0
            avg_trade_pnl = 0.0
            
        # Profit factor
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_pnl': avg_trade_pnl,
            'final_balance': equity_curve[-1],
        }


def create_ultimate_env(df: pd.DataFrame, **kwargs) -> UltimateTradingEnv:
    """Factory function to create ultimate environment."""
    return UltimateTradingEnv(df, **kwargs)
