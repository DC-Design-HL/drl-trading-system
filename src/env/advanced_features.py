"""
Advanced Feature Engineering
Multi-timeframe, regime detection, and sophisticated market features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatureEngine:
    """
    Advanced feature engineering for crypto trading.
    
    Features:
    - Multi-timeframe momentum
    - Volatility regime detection
    - Price action patterns
    - Market microstructure
    - Trend strength indicators
    """
    
    def __init__(self, lookback: int = 200):
        self.lookback = lookback
        
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all advanced features."""
        df = df.copy()
        
        # === PRICE ACTION FEATURES ===
        df = self._add_returns(df)
        df = self._add_volatility_features(df)
        df = self._add_trend_features(df)
        
        # === MOMENTUM FEATURES ===
        df = self._add_momentum_features(df)
        
        # === PATTERN RECOGNITION ===
        df = self._add_candlestick_patterns(df)
        df = self._add_support_resistance(df)
        
        # === REGIME DETECTION ===
        df = self._add_regime_features(df)
        
        # === VOLUME ANALYSIS ===
        df = self._add_volume_features(df)
        
        # === HIGHER TIMEFRAME CONTEXT ===
        df = self._add_multi_timeframe_features(df)
        
        return df
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        # Log returns at different horizons
        for period in [1, 4, 12, 24, 48]:  # 1h, 4h, 12h, 24h, 48h
            df[f'return_{period}h'] = np.log(df['close'] / df['close'].shift(period))
            
        # Cumulative returns
        df['cumret_24h'] = df['return_1h'].rolling(24).sum()
        df['cumret_7d'] = df['return_1h'].rolling(168).sum()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime features."""
        # Realized volatility at different scales
        for window in [12, 24, 72, 168]:  # 12h, 1d, 3d, 7d
            df[f'volatility_{window}h'] = df['return_1h'].rolling(window).std() * np.sqrt(window)
            
        # Volatility ratio (short-term vs long-term)
        df['vol_ratio'] = df['volatility_24h'] / (df['volatility_168h'] + 1e-8)
        
        # ATR-based volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        df['atr_14'] = true_range.rolling(14).mean()
        df['atr_percent'] = df['atr_14'] / df['close']
        
        # Volatility regime (high/low)
        df['vol_regime'] = (df['volatility_24h'] > df['volatility_24h'].rolling(168).mean()).astype(float)
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend detection features."""
        # EMAs for trend
        for period in [8, 21, 55, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
            
        # Trend direction using EMA stack
        df['trend_strength'] = (
            (df['close'] > df['ema_21']).astype(float) +
            (df['ema_21'] > df['ema_55']).astype(float) +
            (df['ema_55'] > df['ema_100']).astype(float) +
            (df['ema_100'] > df['ema_200']).astype(float)
        ) / 4  # 0 to 1 scale
        
        # Linear regression slope
        for window in [20, 50, 100]:
            df[f'slope_{window}'] = self._rolling_slope(df['close'], window)
            
        # ADX-style trend strength
        df['dx'] = self._compute_dx(df)
        df['adx'] = df['dx'].rolling(14).mean()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI at multiple timeframes
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._compute_rsi(df['close'], period)
            
        # RSI divergence (price vs RSI)
        df['rsi_divergence'] = (
            df['return_24h'].rolling(24).corr(df['rsi_14'].diff(24))
        )
        
        # Stochastic
        for period in [14, 21]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
            
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_hist_change'] = df['macd_hist'].diff()
        
        # Rate of Change
        for period in [12, 24, 48]:
            df[f'roc_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
            
        return df
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        # Body size relative to range
        body = abs(df['close'] - df['open'])
        range_ = df['high'] - df['low']
        df['body_ratio'] = body / (range_ + 1e-8)
        
        # Wick ratios
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_ratio'] = upper_wick / (range_ + 1e-8)
        df['lower_wick_ratio'] = lower_wick / (range_ + 1e-8)
        
        # Bullish/Bearish candle
        df['bullish'] = (df['close'] > df['open']).astype(float)
        
        # Consecutive candles
        df['consec_bullish'] = df['bullish'].rolling(5).sum()
        df['consec_bearish'] = (1 - df['bullish']).rolling(5).sum()
        
        # Engulfing patterns
        prev_body = abs(df['close'].shift() - df['open'].shift())
        df['engulfing'] = ((body > prev_body * 1.5) & (df['bullish'] != df['bullish'].shift())).astype(float)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support/resistance features."""
        # Recent high/low distance
        for period in [24, 72, 168]:  # 1d, 3d, 7d
            df[f'dist_high_{period}h'] = (df['high'].rolling(period).max() - df['close']) / df['close']
            df[f'dist_low_{period}h'] = (df['close'] - df['low'].rolling(period).min()) / df['close']
            
        # Breakout detection
        df['breakout_high'] = (df['close'] > df['high'].rolling(48).max().shift()).astype(float)
        df['breakout_low'] = (df['close'] < df['low'].rolling(48).min().shift()).astype(float)
        
        # Position in range
        high_72 = df['high'].rolling(72).max()
        low_72 = df['low'].rolling(72).min()
        df['position_in_range'] = (df['close'] - low_72) / (high_72 - low_72 + 1e-8)
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        # Trend vs Range regime
        # High ADX = trending, Low ADX = ranging
        df['trending_regime'] = (df['adx'] > 25).astype(float)
        
        # Volatility expansion/contraction
        df['vol_expanding'] = (df['volatility_24h'] > df['volatility_24h'].shift(24)).astype(float)
        
        # Mean reversion signal (price far from mean)
        mean_100 = df['close'].rolling(100).mean()
        std_100 = df['close'].rolling(100).std()
        df['zscore_100'] = (df['close'] - mean_100) / (std_100 + 1e-8)
        
        # Trend consistency (how aligned are short/medium/long trends)
        short_trend = np.sign(df['return_4h'])
        medium_trend = np.sign(df['return_24h'])
        long_trend = np.sign(df['return_48h'])
        df['trend_alignment'] = (short_trend + medium_trend + long_trend) / 3
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis features."""
        # Volume moving averages
        df['vol_ma_24h'] = df['volume'].rolling(24).mean()
        df['vol_ma_72h'] = df['volume'].rolling(72).mean()
        df['vol_ma_168h'] = df['volume'].rolling(168).mean()
            
        # Volume ratio (current vs average)
        df['vol_ratio_24h'] = df['volume'] / (df['vol_ma_24h'] + 1e-8)
        
        # Volume trend
        df['vol_trend'] = df['vol_ma_24h'] / (df['vol_ma_72h'] + 1e-8)
        
        # Price-Volume confirmation
        # High volume on up moves = bullish, high volume on down moves = bearish
        df['pv_confirm'] = np.sign(df['return_1h']) * df['vol_ratio_24h']
        
        # Accumulation/Distribution
        money_flow_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
        money_flow_vol = money_flow_mult * df['volume']
        df['ad_line'] = money_flow_vol.cumsum()
        df['ad_line_norm'] = (df['ad_line'] - df['ad_line'].rolling(168).mean()) / (df['ad_line'].rolling(168).std() + 1e-8)
        
        return df
    
    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add higher timeframe context (simulated from hourly)."""
        # 4-hour perspective
        df['close_4h'] = df['close'].rolling(4).apply(lambda x: x.iloc[-1] if len(x) == 4 else np.nan)
        df['high_4h'] = df['high'].rolling(4).max()
        df['low_4h'] = df['low'].rolling(4).min()
        
        # Daily perspective
        df['close_1d'] = df['close'].rolling(24).apply(lambda x: x.iloc[-1] if len(x) == 24 else np.nan)
        df['high_1d'] = df['high'].rolling(24).max()
        df['low_1d'] = df['low'].rolling(24).min()
        
        # Higher timeframe RSI
        df['rsi_4h'] = self._compute_rsi(df['close'].rolling(4).mean(), 14)
        df['rsi_1d'] = self._compute_rsi(df['close'].rolling(24).mean(), 14)
        
        # Higher timeframe trend
        ema_20_4h = df['close'].rolling(4).mean().ewm(span=20, adjust=False).mean()
        df['trend_4h'] = ((df['close'].rolling(4).mean() > ema_20_4h) * 2 - 1).astype(float)
        
        return df
    
    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rolling linear regression slope."""
        def calc_slope(x):
            if len(x) < window:
                return np.nan
            y = np.array(x)
            x_arr = np.arange(len(y))
            slope = np.polyfit(x_arr, y, 1)[0]
            return slope / y.mean()  # Normalize by price level
            
        return series.rolling(window).apply(calc_slope, raw=False)
    
    def _compute_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Compute RSI indicator."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _compute_dx(self, df: pd.DataFrame) -> pd.Series:
        """Compute Directional Index for ADX."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-8))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        return dx
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return [
            # Returns
            'return_1h', 'return_4h', 'return_12h', 'return_24h', 'return_48h',
            'cumret_24h', 'cumret_7d',
            # Volatility
            'volatility_24h', 'volatility_168h', 'vol_ratio', 'atr_percent', 'vol_regime',
            # Trend
            'ema_21_dist', 'ema_55_dist', 'ema_100_dist', 'ema_200_dist',
            'trend_strength', 'slope_20', 'slope_50', 'adx',
            # Momentum
            'rsi_7', 'rsi_14', 'rsi_21', 'rsi_divergence',
            'stoch_14', 'stoch_21',
            'macd_hist', 'macd_hist_change',
            'roc_12', 'roc_24', 'roc_48',
            # Candlestick
            'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
            'consec_bullish', 'consec_bearish', 'engulfing',
            # Support/Resistance
            'dist_high_24h', 'dist_low_24h', 'dist_high_72h', 'dist_low_72h',
            'breakout_high', 'breakout_low', 'position_in_range',
            # Regime
            'trending_regime', 'vol_expanding', 'zscore_100', 'trend_alignment',
            # Volume
            'vol_ratio_24h', 'vol_trend', 'pv_confirm', 'ad_line_norm',
            # Multi-timeframe
            'rsi_4h', 'rsi_1d', 'trend_4h',
        ]
