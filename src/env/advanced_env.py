"""
Advanced Trading Environment
Uses sophisticated features and reward function for competitive performance.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from enum import IntEnum
import logging

from .advanced_features import AdvancedFeatureEngine
from .advanced_rewards import AdvancedRewardCalculator

logger = logging.getLogger(__name__)


class Actions(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


class Positions(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class AdvancedTradingEnv(gym.Env):
    """
    Advanced trading environment with sophisticated features.
    
    Key innovations:
    1. 50+ advanced features (multi-timeframe, regime, patterns)
    2. Profit-factor optimized reward function
    3. Market feature context for smart rewards
    4. Dynamic position sizing based on confidence
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        lookback_window: int = 48,  # 48 hours of context
        trading_fee: float = 0.0004,  # 0.04% (Binance maker fee)
        position_size: float = 0.25,  # 25% of portfolio per trade
        max_position: int = 1,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Data and features
        self.raw_df = df.copy()
        self.feature_engine = AdvancedFeatureEngine()
        self._prepare_data()
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.trading_fee = trading_fee
        self.position_size = position_size
        self.max_position = max_position
        
        # Advanced reward calculator
        self.reward_calculator = AdvancedRewardCalculator(initial_balance=initial_balance)
        
        # Feature dimensions
        self.feature_columns = self.feature_engine.get_feature_columns()
        available_features = [f for f in self.feature_columns if f in self.df.columns]
        self.feature_columns = available_features
        self.n_features = len(self.feature_columns)
        self.n_ohlcv = 5
        self.n_agent_state = 8  # Extended agent state
        
        # Observation space
        obs_dim = self.lookback_window * (self.n_features + self.n_ohlcv) + self.n_agent_state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space
        self.action_space = spaces.Discrete(3)
        
        self._reset_state()
        
    def _prepare_data(self):
        """Compute all advanced features."""
        logger.info("Computing advanced features...")
        self.df = self.feature_engine.compute_all(self.raw_df)
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(method='ffill').fillna(0)
        
        # Ensure enough data after feature computation
        min_lookback = 250  # Need 250 bars for features like EMA 200
        self.df = self.df.iloc[min_lookback:]
        
        self.prices = self.df['close'].values
        self.highs = self.df['high'].values
        self.lows = self.df['low'].values
        
        logger.info(f"Prepared {len(self.df)} candles with {self.n_features if hasattr(self, 'n_features') else 'N/A'} features")
        
    def _reset_state(self):
        """Reset all state variables."""
        self.balance = self.initial_balance
        self.position = Positions.FLAT
        self.position_price = 0.0
        self.position_size_units = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.trade_count = 0
        self.current_step = self.lookback_window
        self.done = False
        self.trades: List[Dict] = []
        self.steps_since_trade = 0
        
    def _get_market_features(self) -> Dict:
        """Get current market features for reward calculation."""
        if self.current_step >= len(self.df):
            return {}
            
        row = self.df.iloc[self.current_step]
        
        features = {}
        for col in ['trend_strength', 'position_in_range', 'vol_ratio_24h', 
                    'adx', 'rsi_14', 'trending_regime', 'zscore_100']:
            if col in row.index:
                features[col] = float(row[col])
                
        return features
        
    def _get_observation(self) -> np.ndarray:
        """Construct observation with advanced features."""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        window_data = self.df.iloc[start_idx:end_idx]
        
        # OHLCV (normalized)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        ohlcv = window_data[ohlcv_cols].values.copy()
        
        # Normalize prices by first close
        price_norm = ohlcv[0, 3]
        if price_norm > 0:
            ohlcv[:, :4] = ohlcv[:, :4] / price_norm - 1
        
        # Normalize volume
        vol_mean = ohlcv[:, 4].mean()
        if vol_mean > 0:
            ohlcv[:, 4] = ohlcv[:, 4] / vol_mean - 1
            
        # Advanced features (already normalized/scaled)
        features = window_data[self.feature_columns].values.copy()
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip extreme values
        features = np.clip(features, -5, 5)
        
        # Combine
        combined = np.concatenate([ohlcv, features], axis=1)
        flat_history = combined.flatten()
        
        # Enhanced agent state
        portfolio_value = self._get_portfolio_value()
        market_features = self._get_market_features()
        
        agent_state = np.array([
            float(self.position),  # Position: -1, 0, 1
            self.unrealized_pnl / self.initial_balance,  # Unrealized P&L ratio
            self.balance / self.initial_balance,  # Balance ratio
            self.current_drawdown,  # Current drawdown
            min(self.trade_count / 100, 1.0),  # Normalized trade count
            min(self.steps_since_trade / 48, 1.0),  # Time since last trade (normalized)
            market_features.get('trend_strength', 0.5),  # Current trend
            market_features.get('trending_regime', 0.5),  # Trend regime
        ], dtype=np.float32)
        
        observation = np.concatenate([flat_history, agent_state]).astype(np.float32)
        
        return observation
        
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return self.balance + self.unrealized_pnl
        
    def _update_unrealized_pnl(self):
        """Update unrealized P&L based on current position."""
        if self.position == Positions.FLAT:
            self.unrealized_pnl = 0.0
            return
            
        current_price = self.prices[self.current_step]
        
        if self.position == Positions.LONG:
            self.unrealized_pnl = (current_price - self.position_price) * self.position_size_units
        elif self.position == Positions.SHORT:
            self.unrealized_pnl = (self.position_price - current_price) * self.position_size_units
            
    def _update_drawdown(self):
        """Update peak balance and current drawdown."""
        portfolio_value = self._get_portfolio_value()
        if portfolio_value > self.peak_balance:
            self.peak_balance = portfolio_value
        self.current_drawdown = (self.peak_balance - portfolio_value) / self.peak_balance
        
    def _execute_trade(self, action: int) -> Tuple[float, Optional[float]]:
        """Execute trading action."""
        current_price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1]
        trade_pnl = None
        
        # Calculate step return
        if self.position == Positions.LONG:
            step_return = (current_price - prev_price) / prev_price
        elif self.position == Positions.SHORT:
            step_return = (prev_price - current_price) / prev_price
        else:
            step_return = 0.0
            
        # Execute action
        if action == Actions.BUY and self.position != Positions.LONG:
            if self.position == Positions.SHORT:
                trade_pnl = self._close_position(current_price)
            self._open_position(current_price, Positions.LONG)
            self.steps_since_trade = 0
            
        elif action == Actions.SELL and self.position != Positions.SHORT:
            if self.position == Positions.LONG:
                trade_pnl = self._close_position(current_price)
            self._open_position(current_price, Positions.SHORT)
            self.steps_since_trade = 0
            
        else:
            self.steps_since_trade += 1
            
        return step_return, trade_pnl
        
    def _open_position(self, price: float, position_type: Positions):
        """Open new position."""
        trade_amount = self.balance * self.position_size
        fee = trade_amount * self.trading_fee
        
        self.position = position_type
        self.position_price = price
        self.position_size_units = (trade_amount - fee) / price
        self.balance -= fee
        self.trade_count += 1
        
    def _close_position(self, price: float) -> float:
        """Close current position."""
        if self.position == Positions.LONG:
            pnl = (price - self.position_price) * self.position_size_units
        elif self.position == Positions.SHORT:
            pnl = (self.position_price - price) * self.position_size_units
        else:
            return 0.0
            
        # Apply fee
        fee = abs(pnl) * self.trading_fee
        pnl -= fee
        
        # Record trade
        self.trades.append({
            'entry_price': self.position_price,
            'exit_price': price,
            'position': int(self.position),
            'pnl': pnl,
            'step': self.current_step,
        })
        
        # Update balance
        self.balance += pnl
        self.realized_pnl += pnl
        
        # Reset position
        self.position = Positions.FLAT
        self.position_price = 0.0
        self.position_size_units = 0.0
        
        return pnl
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        if self.done:
            raise RuntimeError("Episode done, call reset()")
            
        # Get market features for reward calculation
        market_features = self._get_market_features()
        
        # Execute trade
        step_return, trade_pnl = self._execute_trade(action)
        
        # Update state
        self._update_unrealized_pnl()
        self._update_drawdown()
        
        # Calculate advanced reward
        portfolio_value = self._get_portfolio_value()
        reward = self.reward_calculator.calculate_reward(
            step_return=step_return,
            portfolio_value=portfolio_value,
            position=int(self.position),
            action_taken=action,
            trade_pnl=trade_pnl,
            market_features=market_features,
        )
        
        # Move to next step
        self.current_step += 1
        
        # Check termination
        terminated = False
        truncated = False
        
        if self.current_step >= len(self.df) - 1:
            truncated = True
            self.done = True
            
        if self.balance <= 0 or self.current_drawdown > 0.25:  # Stop at 25% drawdown
            terminated = True
            self.done = True
            reward -= 5.0  # Heavy penalty for blowing up
            
        observation = self._get_observation()
        
        info = {
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'position': int(self.position),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'drawdown': self.current_drawdown,
            'trade_count': self.trade_count,
            'step': self.current_step,
            'price': self.prices[min(self.current_step, len(self.prices)-1)],
        }
        
        return observation, reward, terminated, truncated, info
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self._reset_state()
        self.reward_calculator.reset(self.initial_balance)
        
        # Random start for training diversity
        if options and options.get('random_start', True):
            max_start = len(self.df) - self.lookback_window - 500
            if max_start > self.lookback_window:
                self.current_step = self.np_random.integers(self.lookback_window, max_start)
                
        observation = self._get_observation()
        info = {'balance': self.balance, 'portfolio_value': self.initial_balance}
        
        return observation, info
        
    def render(self):
        """Render environment."""
        if self.render_mode == 'human':
            print(self._render_ansi())
        elif self.render_mode == 'ansi':
            return self._render_ansi()
            
    def _render_ansi(self) -> str:
        """String representation."""
        pos_str = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}[int(self.position)]
        return (
            f"Step: {self.current_step} | "
            f"Price: ${self.prices[self.current_step]:.2f} | "
            f"Position: {pos_str} | "
            f"Portfolio: ${self._get_portfolio_value():.2f} | "
            f"Trades: {self.trade_count} | "
            f"DD: {self.current_drawdown:.2%}"
        )
        
    def get_episode_metrics(self) -> Dict:
        """Get episode metrics."""
        metrics = self.reward_calculator.get_episode_metrics()
        metrics['trades'] = self.trades
        metrics['final_balance'] = self.balance
        metrics['final_portfolio_value'] = self._get_portfolio_value()
        return metrics


def create_advanced_env(df: pd.DataFrame, config: Optional[Dict] = None) -> AdvancedTradingEnv:
    """Factory function to create advanced environment."""
    config = config or {}
    return AdvancedTradingEnv(
        df=df,
        initial_balance=config.get('initial_balance', 10000.0),
        lookback_window=config.get('lookback_window', 48),
        trading_fee=config.get('trading_fee', 0.0004),
        position_size=config.get('position_size', 0.25),
    )
