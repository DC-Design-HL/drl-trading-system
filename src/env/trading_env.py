"""
Crypto Trading Environment
Custom Gymnasium environment for reinforcement learning-based crypto trading.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from enum import IntEnum

from .indicators import TechnicalIndicators
from .rewards import RewardCalculator


class Actions(IntEnum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2


class Positions(IntEnum):
    """Position states."""
    SHORT = -1
    FLAT = 0
    LONG = 1


class CryptoTradingEnv(gym.Env):
    """
    A Gymnasium environment for cryptocurrency trading.
    
    Observation Space:
        - OHLCV data for lookback_window periods
        - Technical indicators (RSI, MACD, BB, etc.)
        - Agent state (position, unrealized P&L, balance, etc.)
    
    Action Space:
        - 0: Hold
        - 1: Buy/Long
        - 2: Sell/Short
    
    Reward:
        - Risk-adjusted returns using Sharpe/Sortino ratios
        - Drawdown penalties
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        lookback_window: int = 30,
        trading_fee: float = 0.001,
        position_size: float = 0.1,
        max_position: int = 1,
        normalize_obs: bool = True,
        render_mode: Optional[str] = None,
        feature_engine: Optional[Any] = None,
        symbol: str = "BTCUSDT",
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV data and timestamp index
            initial_balance: Starting balance in quote currency (USDT)
            lookback_window: Number of historical candles in observation
            trading_fee: Trading fee as fraction (0.001 = 0.1%)
            position_size: Fraction of balance per trade (0.1 = 10%)
            max_position: Maximum position size (-1, 0, 1)
            normalize_obs: Whether to normalize observations
            render_mode: Rendering mode ('human' or 'ansi')
            feature_engine: Optional external feature engine (e.g. MultiAssetFeatureEngine)
            symbol: Trading pair symbol (for feature engine)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.symbol = symbol
        self.feature_engine = feature_engine
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.trading_fee = trading_fee
        self.position_size = position_size
        self.max_position = max_position
        self.normalize_obs = normalize_obs
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Data preparation (compute features)
        self.df = df.copy()
        
        if self.feature_engine:
            # Use external feature engine
            self._prepare_data_external()
        else:
            # Use default indicators
            self.indicators = TechnicalIndicators()
            self._prepare_data_internal()
            
        # Calculate observation dimensions
        self.n_features = len(self.feature_columns)
        self.n_ohlcv = 5  # open, high, low, close, volume (normalized)
        self.n_agent_state = 5  # position, unrealized_pnl, balance_ratio, drawdown, trade_count
        
        # Total observation size per timestep (OHLCV + features + state)
        # Note: If feature engine includes OHLCV, we might double count, but that's okay for now
        # The external engine usually returns unified feature vector
        
        if self.feature_engine:
            # External engine returns full feature vector
            # We assume lookback is handled by the model (LSTM) or by flattening here
            obs_dim = self.lookback_window * self.n_features + self.n_agent_state
        else:
             # Default: flattening window of (features + OHLCV)
            obs_dim = self.lookback_window * (self.n_features + self.n_ohlcv) + self.n_agent_state
            
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)
        
        # State variables (will be set in reset)
        self._reset_state()
        
    def _prepare_data_internal(self):
        """Compute technical indicators using internal engine."""
        self.df = self.indicators.compute_all(self.df)
        self.feature_columns = self.indicators.get_feature_columns()
        self._finalize_data()
        
    def _prepare_data_external(self):
        """Compute features using external engine."""
        # Compute features for all rows
        # Assumes feature_engine has compute_features_batch
        features = self.feature_engine.compute_features_batch(self.df, self.symbol)
        
        # Convert to DataFrame columns
        feat_cols = [f"feat_{i}" for i in range(features.shape[1])]
        feat_df = pd.DataFrame(features, columns=feat_cols, index=self.df.index)
        
        # Add to main DF
        self.df = pd.concat([self.df, feat_df], axis=1)
        self.feature_columns = feat_cols
        self._finalize_data()

    def _finalize_data(self):
        """Common data finalization steps."""
        # Ensure we have enough data
        self.df = self.df.dropna()
        
        # Store price data separately for easy access
        self.prices = self.df['close'].values
        self.highs = self.df['high'].values
        self.lows = self.df['low'].values
        
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
        
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation array.
        
        Returns:
            Flattened observation array
        """
        # Get lookback window of data
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        window_data = self.df.iloc[start_idx:end_idx]
        
        # OHLCV features (normalized by first close in window)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        ohlcv = window_data[ohlcv_cols].values.copy()  # Make copy to avoid read-only error
        
        # Normalize OHLCV
        if self.normalize_obs:
            price_norm = ohlcv[0, 3]  # First close price
            if price_norm > 0:  # Avoid division by zero
                ohlcv[:, :4] = ohlcv[:, :4] / price_norm - 1  # Normalize prices
            vol_mean = ohlcv[:, 4].mean()
            ohlcv[:, 4] = ohlcv[:, 4] / (vol_mean + 1e-8) - 1  # Normalize volume
        
        # Technical indicator features
        features = window_data[self.feature_columns].values.copy()  # Make copy
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Combine OHLCV and features for each timestep
        if self.feature_engine:
            # If using external engine, features already contain everything needed
            combined = features
        else:
            # Default behavior: combine OHLCV + indicators
            combined = np.concatenate([ohlcv, features], axis=1)
        
        # Flatten the lookback window
        flat_history = combined.flatten()
        
        # Agent state
        portfolio_value = self._get_portfolio_value()
        agent_state = np.array([
            float(self.position),  # Position: -1, 0, 1
            self.unrealized_pnl / self.initial_balance,  # Unrealized P&L ratio
            self.balance / self.initial_balance,  # Balance ratio
            self.current_drawdown,  # Current drawdown
            min(self.trade_count / 100, 1.0),  # Normalized trade count
        ], dtype=np.float32)
        
        # Combine all
        observation = np.concatenate([flat_history, agent_state]).astype(np.float32)
        
        return observation
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value including unrealized P&L."""
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
        """
        Execute a trading action.
        
        Args:
            action: The action to take (0=hold, 1=buy, 2=sell)
            
        Returns:
            Tuple of (step_return, trade_pnl or None)
        """
        current_price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1]
        trade_pnl = None
        
        # Calculate step return regardless of action
        if self.position == Positions.LONG:
            step_return = (current_price - prev_price) / prev_price
        elif self.position == Positions.SHORT:
            step_return = (prev_price - current_price) / prev_price
        else:
            step_return = 0.0
        
        # Execute action
        if action == Actions.BUY and self.position != Positions.LONG:
            # Close short if exists
            if self.position == Positions.SHORT:
                trade_pnl = self._close_position(current_price)
            
            # Open long
            self._open_position(current_price, Positions.LONG)
            
        elif action == Actions.SELL and self.position != Positions.SHORT:
            # Close long if exists
            if self.position == Positions.LONG:
                trade_pnl = self._close_position(current_price)
            
            # Open short
            self._open_position(current_price, Positions.SHORT)
            
        elif action == Actions.HOLD:
            # Just update unrealized P&L
            pass
            
        return step_return, trade_pnl
    
    def _open_position(self, price: float, position_type: Positions):
        """Open a new position."""
        trade_amount = self.balance * self.position_size
        fee = trade_amount * self.trading_fee
        
        self.position = position_type
        self.position_price = price
        self.position_size_units = (trade_amount - fee) / price
        self.balance -= fee  # Only deduct fee, not the position value
        self.trade_count += 1
        
    def _close_position(self, price: float) -> float:
        """Close current position and return P&L."""
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
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
            
        # Execute trade
        step_return, trade_pnl = self._execute_trade(action)
        
        # Update state
        self._update_unrealized_pnl()
        self._update_drawdown()
        
        # Calculate reward
        portfolio_value = self._get_portfolio_value()
        reward = self.reward_calculator.calculate_reward(
            step_return=step_return,
            portfolio_value=portfolio_value,
            position=int(self.position),
            action_taken=action,
            trade_pnl=trade_pnl,
        )
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        terminated = False
        truncated = False
        
        if self.current_step >= len(self.df) - 1:
            truncated = True
            self.done = True
            
        if self.balance <= 0:
            terminated = True
            self.done = True
            
        # Get new observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'position': int(self.position),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'drawdown': self.current_drawdown,
            'trade_count': self.trade_count,
            'step': self.current_step,
            'price': self.prices[self.current_step],
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        self._reset_state()
        self.reward_calculator.reset(self.initial_balance)
        
        # Option to start at random point
        if options and options.get('random_start', False):
            max_start = len(self.df) - self.lookback_window - 100  # Leave room for episode
            if max_start > self.lookback_window:
                self.current_step = self.np_random.integers(
                    self.lookback_window, max_start
                )
        
        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'portfolio_value': self._get_portfolio_value(),
        }
        
        return observation, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'ansi':
            return self._render_ansi()
            
    def _render_human(self):
        """Print current state to console."""
        print(self._render_ansi())
        
    def _render_ansi(self) -> str:
        """Return string representation of current state."""
        portfolio = self._get_portfolio_value()
        pos_str = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}[int(self.position)]
        
        return (
            f"Step: {self.current_step} | "
            f"Price: ${self.prices[self.current_step]:.2f} | "
            f"Position: {pos_str} | "
            f"Balance: ${self.balance:.2f} | "
            f"Portfolio: ${portfolio:.2f} | "
            f"P&L: ${self.realized_pnl:+.2f} | "
            f"Drawdown: {self.current_drawdown:.2%}"
        )
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get summary metrics for the episode."""
        reward_metrics = self.reward_calculator.get_episode_metrics()
        
        return {
            **reward_metrics,
            'final_balance': self.balance,
            'final_portfolio_value': self._get_portfolio_value(),
            'total_pnl': self.realized_pnl,
            'total_return': (self._get_portfolio_value() - self.initial_balance) / self.initial_balance,
            'trade_count': self.trade_count,
            'trades': self.trades,
        }
    
    def get_trade_signals(self) -> List[Dict]:
        """Get list of trade signals for visualization."""
        return self.trades.copy()


def create_env_from_df(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
) -> CryptoTradingEnv:
    """
    Factory function to create environment from dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        config: Optional configuration dictionary
        
    Returns:
        CryptoTradingEnv instance
    """
    config = config or {}
    
    return CryptoTradingEnv(
        df=df,
        initial_balance=config.get('initial_balance', 10000.0),
        lookback_window=config.get('lookback_window', 30),
        trading_fee=config.get('trading_fee', 0.001),
        position_size=config.get('position_size', 0.1),
        max_position=config.get('max_position', 1),
        normalize_obs=config.get('normalize_obs', True),
    )
