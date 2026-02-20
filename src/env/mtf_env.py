"""
Multi-Timeframe Trading Environment

Extends AdvancedTradingEnv with real multi-timeframe observations:
- 15-minute data: micro order flow, entry timing, short-term momentum
- 1-hour data: main trading timeframe (existing)
- 4-hour data: macro bias, trend direction, support/resistance

The observation is extended with compact MTF summary features
(not full bars), keeping the observation space manageable.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import logging

from .advanced_env import AdvancedTradingEnv, Actions, Positions
from .advanced_features import AdvancedFeatureEngine
from .advanced_rewards import AdvancedRewardCalculator

logger = logging.getLogger(__name__)


def resample_to_higher_tf(df_1h: pd.DataFrame, target_tf: str = '4h') -> pd.DataFrame:
    """
    Resample 1h OHLCV data to a higher timeframe.

    Args:
        df_1h: DataFrame with 1h OHLCV data (DatetimeIndex)
        target_tf: Target timeframe ('4h', '1d', etc.)

    Returns:
        Resampled DataFrame
    """
    # Map our labels to pandas resample rules
    tf_map = {'4h': '4h', '1d': '1D', '1w': '1W'}
    rule = tf_map.get(target_tf, target_tf)

    resampled = df_1h.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    return resampled


def compute_mtf_features_at(
    df_4h: pd.DataFrame,
    timestamp: pd.Timestamp,
    lookback: int = 30,
) -> np.ndarray:
    """
    Compute compact 4h timeframe features at a given 1h timestamp.

    Returns 12 features:
    - 4h trend (EMA 7 vs 14 slope)
    - 4h RSI
    - 4h momentum (return over 5, 10, 20 bars)
    - 4h volatility (ATR ratio)
    - 4h volume trend
    - 4h MACD state
    - 4h Bollinger position
    - 4h support/resistance distance
    - 4h candle body ratio
    - 4h trend strength (ADX proxy)
    """
    # Find the most recent 4h bar at or before this timestamp
    mask = df_4h.index <= timestamp
    available = df_4h[mask]

    if len(available) < lookback:
        return np.zeros(12, dtype=np.float32)

    window = available.tail(lookback)
    close = window['close'].values
    high = window['high'].values
    low = window['low'].values
    volume = window['volume'].values
    opn = window['open'].values

    features = np.zeros(12, dtype=np.float32)

    try:
        # 1. EMA trend: EMA7 vs EMA14 normalized
        ema7 = pd.Series(close).ewm(span=7).mean().values
        ema14 = pd.Series(close).ewm(span=14).mean().values
        features[0] = np.clip((ema7[-1] - ema14[-1]) / (close[-1] * 0.01 + 1e-10), -3, 3)

        # 2. RSI 14
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values[-1]
        avg_loss = pd.Series(loss).rolling(14).mean().values[-1]
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            features[1] = (100 - 100 / (1 + rs)) / 100  # Normalized 0-1
        else:
            features[1] = 0.5

        # 3-5. Momentum (returns over 5, 10, 20 bars)
        for i, period in enumerate([5, 10, 20]):
            if len(close) > period:
                features[2 + i] = np.clip((close[-1] / close[-period] - 1) * 10, -3, 3)

        # 6. ATR ratio (current vs average)
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        atr_14 = pd.Series(tr).rolling(14).mean().values[-1]
        atr_avg = pd.Series(tr).rolling(min(len(tr), 50)).mean().values[-1]
        features[5] = np.clip(atr_14 / (atr_avg + 1e-10) - 1, -2, 2)

        # 7. Volume trend (current vs 20-bar average)
        vol_avg = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        features[6] = np.clip(volume[-1] / (vol_avg + 1e-10) - 1, -3, 3)

        # 8. MACD state (sign of MACD histogram)
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=9).mean().values
        hist = macd[-1] - signal[-1]
        features[7] = np.clip(hist / (close[-1] * 0.001 + 1e-10), -3, 3)

        # 9. Bollinger position
        bb_ma = np.mean(close[-20:])
        bb_std = np.std(close[-20:])
        if bb_std > 0:
            features[8] = np.clip((close[-1] - bb_ma) / (2 * bb_std + 1e-10), -1.5, 1.5)

        # 10. Support/resistance distance (distance to recent high/low)
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        price_range = recent_high - recent_low
        if price_range > 0:
            features[9] = (close[-1] - recent_low) / price_range  # 0-1 position

        # 11. Candle body ratio (last bar)
        hl_range = high[-1] - low[-1]
        if hl_range > 0:
            features[10] = (close[-1] - opn[-1]) / hl_range  # -1 to 1

        # 12. Trend strength (directional movement proxy)
        if len(close) >= 14:
            up_moves = np.diff(high[-15:])
            down_moves = -np.diff(low[-15:])
            up_moves = np.where(up_moves > 0, up_moves, 0)
            down_moves = np.where(down_moves > 0, down_moves, 0)
            di_plus = np.mean(up_moves[-14:])
            di_minus = np.mean(down_moves[-14:])
            if (di_plus + di_minus) > 0:
                features[11] = np.clip((di_plus - di_minus) / (di_plus + di_minus), -1, 1)

    except Exception as e:
        logger.debug(f"MTF feature error: {e}")

    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


N_MTF_FEATURES = 12  # per higher timeframe
N_MTF_TOTAL = N_MTF_FEATURES  # just 4h for now (15m would need separate data)


class MultiTimeframeTradingEnv(AdvancedTradingEnv):
    """
    Multi-Timeframe Trading Environment.

    Extends AdvancedTradingEnv by appending compact 4h timeframe features
    to the observation space. The agent sees:

    [flattened 1h OHLCV+features | agent state | 4h MTF features]

    This is backward-compatible — the original observation is preserved
    and the MTF features are appended at the end.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_4h: pd.DataFrame = None,
        initial_balance: float = 10000.0,
        lookback_window: int = 48,
        trading_fee: float = 0.0004,
        position_size: float = 0.25,
        max_position: int = 1,
        render_mode: Optional[str] = None,
    ):
        # Store 4h data before parent init
        if df_4h is not None:
            self._df_4h = df_4h.copy()
        else:
            # Auto-resample from 1h if not provided
            self._df_4h = resample_to_higher_tf(df, '4h')

        # Parent init (computes features, sets observation space)
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            lookback_window=lookback_window,
            trading_fee=trading_fee,
            position_size=position_size,
            max_position=max_position,
            render_mode=render_mode,
        )

        # Extend observation space with MTF features
        base_obs_dim = self.observation_space.shape[0]
        new_obs_dim = base_obs_dim + N_MTF_TOTAL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32
        )

        logger.info(
            f"🔀 MTF env initialized: base={base_obs_dim}, "
            f"mtf={N_MTF_TOTAL}, total={new_obs_dim}, "
            f"4h bars={len(self._df_4h)}"
        )

    def _get_observation(self) -> np.ndarray:
        """Construct observation with base features + MTF features."""
        # Get base observation from parent
        base_obs = super()._get_observation()

        # Compute 4h MTF features at current timestamp
        current_idx = min(self.current_step, len(self.df) - 1)
        current_ts = self.df.index[current_idx]

        mtf_4h = compute_mtf_features_at(self._df_4h, current_ts)

        # Concatenate
        observation = np.concatenate([base_obs, mtf_4h]).astype(np.float32)

        return observation


def create_mtf_env(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame = None,
    config: Optional[Dict] = None,
) -> MultiTimeframeTradingEnv:
    """Factory function to create MTF environment."""
    config = config or {}

    return MultiTimeframeTradingEnv(
        df=df_1h,
        df_4h=df_4h,
        initial_balance=config.get('initial_balance', 10000.0),
        lookback_window=config.get('lookback_window', 48),
        trading_fee=config.get('trading_fee', 0.0004),
        position_size=config.get('position_size', 0.25),
    )
