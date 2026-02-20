"""
Technical Indicators Module
Wrapper around pandas_ta for normalized technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class TechnicalIndicators:
    """
    Computes and normalizes technical indicators for the trading environment.
    All indicators are normalized to [-1, 1] or [0, 1] range for neural network input.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize technical indicators with configuration.
        
        Args:
            config: Dictionary with indicator parameters
        """
        self.config = config or {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'ema_periods': [9, 21, 50],
        }
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and add them to the dataframe.
        Using pure pandas for reliability without external TA libraries.
        """
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI
        rsi_period = self.config['rsi_period']
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.ewm(com=rsi_period-1, adjust=False).mean()
        avg_loss = loss.ewm(com=rsi_period-1, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = df['rsi'] / 100.0
        
        # MACD
        ema_fast = close.ewm(span=self.config['macd_fast'], adjust=False).mean()
        ema_slow = close.ewm(span=self.config['macd_slow'], adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config['macd_signal'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        # Normalize MACD by price
        df['macd_norm'] = df['macd'] / close
        df['macd_signal_norm'] = df['macd_signal'] / close
        df['macd_hist_norm'] = df['macd_hist'] / close
        
        # Bollinger Bands
        bb_period = self.config['bb_period']
        bb_std = self.config['bb_std']
        sma = close.rolling(window=bb_period).mean()
        std = close.rolling(window=bb_period).std()
        
        df['bb_lower'] = sma - bb_std * std
        df['bb_mid'] = sma
        df['bb_upper'] = sma + bb_std * std
        bb_range = df['bb_upper'] - df['bb_lower']
        
        # Avoid division by zero
        bb_range = bb_range.replace(0, 1e-8)
        df['bb_bandwidth'] = bb_range / df['bb_mid']
        df['bb_position'] = 2 * (close - df['bb_lower']) / bb_range - 1
        df['bb_position'] = df['bb_position'].clip(-2, 2)
        
        # ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.config['atr_period']).mean()
        df['atr_norm'] = df['atr'] / close
        
        # EMAs
        for period in self.config['ema_periods']:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            df[f'ema_{period}_dist'] = (close - df[f'ema_{period}']) / close
            
        # Volume indicators
        df['volume_sma'] = volume.rolling(window=20).mean()
        df['volume_ratio'] = volume / df['volume_sma'].replace(0, 1e-8)
        df['volume_ratio'] = df['volume_ratio'].clip(0, 5)
        
        # Price momentum
        df['returns'] = close.pct_change()
        df['returns_5'] = close.pct_change(5)
        df['returns_10'] = close.pct_change(10)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """
        Returns the list of feature columns to use for the observation space.
        """
        return [
            'rsi_norm',
            'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
            'bb_position', 'bb_bandwidth',
            'atr_norm',
            'ema_9_dist', 'ema_21_dist', 'ema_50_dist',
            'volume_ratio',
            'returns', 'returns_5', 'returns_10',
            'volatility',
        ]
    
    def get_normalized_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract normalized feature array from dataframe.
        """
        features = df[self.get_feature_columns()].values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        return features


def compute_indicators(ohlcv_data: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to compute all indicators.
    
    Args:
        ohlcv_data: DataFrame with OHLCV columns
        config: Optional indicator configuration
        
    Returns:
        DataFrame with all indicators computed
    """
    indicators = TechnicalIndicators(config)
    return indicators.compute_all(ohlcv_data)
