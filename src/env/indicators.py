"""
Technical Indicators Module
Wrapper around pandas_ta for normalized technical indicators.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
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
        
        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
            
        Returns:
            DataFrame with additional indicator columns
        """
        df = df.copy()
        
        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()
        
        # RSI - Relative Strength Index [0, 100] -> normalized to [0, 1]
        df['rsi'] = ta.rsi(df['close'], length=self.config['rsi_period'])
        df['rsi_norm'] = df['rsi'] / 100.0
        
        # MACD - Moving Average Convergence Divergence
        macd = ta.macd(
            df['close'],
            fast=self.config['macd_fast'],
            slow=self.config['macd_slow'],
            signal=self.config['macd_signal']
        )
        if macd is not None:
            df['macd'] = macd.iloc[:, 0]
            df['macd_signal'] = macd.iloc[:, 1]
            df['macd_hist'] = macd.iloc[:, 2]
            # Normalize MACD by price for scale-independence
            df['macd_norm'] = df['macd'] / df['close']
            df['macd_signal_norm'] = df['macd_signal'] / df['close']
            df['macd_hist_norm'] = df['macd_hist'] / df['close']
        
        # Bollinger Bands
        bb = ta.bbands(
            df['close'],
            length=self.config['bb_period'],
            std=self.config['bb_std']
        )
        if bb is not None:
            df['bb_lower'] = bb.iloc[:, 0]
            df['bb_mid'] = bb.iloc[:, 1]
            df['bb_upper'] = bb.iloc[:, 2]
            df['bb_bandwidth'] = bb.iloc[:, 3] if bb.shape[1] > 3 else (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
            # BB position: where is price relative to bands [-1 = lower, 0 = mid, 1 = upper]
            bb_range = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = 2 * (df['close'] - df['bb_lower']) / bb_range - 1
            df['bb_position'] = df['bb_position'].clip(-2, 2)  # Clip extreme values
        
        # ATR - Average True Range (for volatility)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.config['atr_period'])
        df['atr_norm'] = df['atr'] / df['close']  # Normalize by price
        
        # EMAs - Exponential Moving Averages
        for period in self.config['ema_periods']:
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)
            # EMA distance from price (normalized)
            df[f'ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df['close']
        
        # Volume indicators
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_ratio'] = df['volume_ratio'].clip(0, 5)  # Clip extreme spikes
        
        # Price momentum
        df['returns'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        
        # Volatility (rolling std of returns)
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
        
        Args:
            df: DataFrame with computed indicators
            
        Returns:
            Numpy array of shape (n_samples, n_features)
        """
        features = df[self.get_feature_columns()].values
        # Replace NaN with 0
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
