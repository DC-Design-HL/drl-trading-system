"""
Market Regime Detection for HTF Trading

Detects market regimes (trending_up, trending_down, ranging, volatile) using
technical indicators and statistical measures. Provides regime labels with
confidence scores for regime-conditional trading strategies.

Key features:
- Four regime types: trending_up, trending_down, ranging, volatile
- Uses ADX, volatility percentile, price momentum for detection
- Confidence scores based on signal strength and consistency
- Optimized for crypto market characteristics
- Fast computation suitable for real-time trading
"""

import logging
from typing import Dict, Tuple, Optional, List
from enum import Enum

import numpy as np
import pandas as pd
import talib

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


class RegimeDetector:
    """
    Market regime detector using multiple technical indicators.
    
    Combines trend strength (ADX), directional movement, volatility analysis,
    and price momentum to classify market conditions into four regime types.
    """
    
    def __init__(
        self,
        lookback_periods: int = 96,  # 24 hours of 15m bars
        adx_period: int = 14,
        volatility_lookback: int = 48,  # 12 hours for volatility
        trend_threshold: float = 0.6,
        volatility_threshold: float = 0.8,
        adx_threshold: float = 25.0,
    ):
        """
        Initialize the regime detector.
        
        Args:
            lookback_periods: Number of periods to analyze for regime detection
            adx_period: Period for ADX calculation
            volatility_lookback: Lookback for volatility percentile calculation
            trend_threshold: Threshold for trend strength (0-1)
            volatility_threshold: Threshold for high volatility regime (0-1)
            adx_threshold: Minimum ADX for trending regime consideration
        """
        self.lookback_periods = lookback_periods
        self.adx_period = adx_period
        self.volatility_lookback = volatility_lookback
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.adx_threshold = adx_threshold
        
        # Cache for recent calculations
        self._cache = {}
        self._last_index = None
    
    def detect_regime(
        self,
        df: pd.DataFrame,
        current_idx: Optional[int] = None
    ) -> Tuple[RegimeType, float]:
        """
        Detect the current market regime.
        
        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            current_idx: Index to analyze (default: last row)
            
        Returns:
            Tuple of (regime_type, confidence_score)
        """
        if current_idx is None:
            current_idx = len(df) - 1
        
        # Ensure we have enough data
        if current_idx < self.lookback_periods:
            return RegimeType.RANGING, 0.5
        
        # Check cache
        cache_key = (id(df), current_idx)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Extract data slice
        start_idx = max(0, current_idx - self.lookback_periods + 1)
        data_slice = df.iloc[start_idx:current_idx + 1].copy()
        
        # Calculate regime indicators
        indicators = self._calculate_indicators(data_slice)
        
        # Determine regime
        regime, confidence = self._classify_regime(indicators)
        
        # Cache result
        self._cache[cache_key] = (regime, confidence)
        
        # Clean old cache entries
        if len(self._cache) > 100:
            old_keys = list(self._cache.keys())[:-50]
            for key in old_keys:
                del self._cache[key]
        
        return regime, confidence
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all regime detection indicators."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # ADX and directional movement
        adx = self._safe_talib(talib.ADX, high, low, close, timeperiod=self.adx_period)
        plus_di = self._safe_talib(talib.PLUS_DI, high, low, close, timeperiod=self.adx_period)
        minus_di = self._safe_talib(talib.MINUS_DI, high, low, close, timeperiod=self.adx_period)
        
        # Current values (last valid)
        current_adx = self._get_last_valid(adx, default=20.0)
        current_plus_di = self._get_last_valid(plus_di, default=20.0)
        current_minus_di = self._get_last_valid(minus_di, default=20.0)
        
        # Trend direction and strength
        di_diff = current_plus_di - current_minus_di
        trend_strength = current_adx / 100.0  # Normalize to 0-1
        trend_direction = np.tanh(di_diff / 10.0)  # -1 to 1
        
        # Price momentum (multiple timeframes)
        price_momentum_short = self._calculate_momentum(close, periods=[3, 6, 12])
        price_momentum_medium = self._calculate_momentum(close, periods=[12, 24, 48])
        
        # Volatility analysis
        returns = np.diff(np.log(close))
        recent_vol = np.std(returns[-self.volatility_lookback:]) if len(returns) >= self.volatility_lookback else np.std(returns)
        historical_vol = np.std(returns) if len(returns) > 0 else recent_vol
        volatility_ratio = recent_vol / max(historical_vol, 1e-6)
        
        # Volatility percentile
        if len(returns) >= self.volatility_lookback:
            vol_window = returns[-self.volatility_lookback:]
            vol_percentile = np.percentile(np.abs(vol_window), 90)
            current_vol_pct = np.mean(np.abs(vol_window[-5:])) / max(vol_percentile, 1e-6)
        else:
            current_vol_pct = 0.5
        
        # Range vs trend analysis
        price_range = (np.max(close[-24:]) - np.min(close[-24:])) / close[-1] if len(close) >= 24 else 0.02
        trend_consistency = self._calculate_trend_consistency(close)
        
        indicators = {
            'adx': current_adx,
            'plus_di': current_plus_di,
            'minus_di': current_minus_di,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'di_diff': di_diff,
            'momentum_short': price_momentum_short,
            'momentum_medium': price_momentum_medium,
            'volatility_ratio': volatility_ratio,
            'volatility_percentile': current_vol_pct,
            'price_range': price_range,
            'trend_consistency': trend_consistency,
        }
        
        return indicators
    
    def _classify_regime(self, indicators: Dict[str, float]) -> Tuple[RegimeType, float]:
        """Classify regime based on calculated indicators."""
        adx = indicators['adx']
        trend_direction = indicators['trend_direction']
        trend_strength = indicators['trend_strength']
        volatility_percentile = indicators['volatility_percentile']
        momentum_short = indicators['momentum_short']
        momentum_medium = indicators['momentum_medium']
        trend_consistency = indicators['trend_consistency']
        
        # Initialize scores for each regime
        scores = {
            RegimeType.TRENDING_UP: 0.0,
            RegimeType.TRENDING_DOWN: 0.0,
            RegimeType.RANGING: 0.0,
            RegimeType.VOLATILE: 0.0,
        }
        
        # Trending Up conditions
        trending_up_strength = (
            max(0, trend_direction) * 0.3 +  # Positive DI difference
            max(0, momentum_short) * 0.25 +  # Positive short-term momentum  
            max(0, momentum_medium) * 0.25 + # Positive medium-term momentum
            (trend_strength if trend_direction > 0 else 0) * 0.2  # Strong ADX with positive direction
        )
        
        if adx >= self.adx_threshold and trend_direction > 0.1:
            trending_up_strength *= (1.0 + trend_consistency * 0.5)
        
        scores[RegimeType.TRENDING_UP] = trending_up_strength
        
        # Trending Down conditions
        trending_down_strength = (
            max(0, -trend_direction) * 0.3 +  # Negative DI difference
            max(0, -momentum_short) * 0.25 +  # Negative short-term momentum
            max(0, -momentum_medium) * 0.25 + # Negative medium-term momentum
            (trend_strength if trend_direction < 0 else 0) * 0.2  # Strong ADX with negative direction
        )
        
        if adx >= self.adx_threshold and trend_direction < -0.1:
            trending_down_strength *= (1.0 + trend_consistency * 0.5)
        
        scores[RegimeType.TRENDING_DOWN] = trending_down_strength
        
        # Ranging conditions (low ADX, low momentum, high consistency)
        ranging_strength = (
            max(0, 1.0 - trend_strength) * 0.4 +  # Low trend strength
            max(0, 1.0 - abs(momentum_short)) * 0.3 +  # Low short momentum
            max(0, 1.0 - volatility_percentile) * 0.3  # Low volatility
        )
        
        if adx < self.adx_threshold:
            ranging_strength *= 1.5
        
        scores[RegimeType.RANGING] = ranging_strength
        
        # Volatile conditions (high volatility, inconsistent direction)
        volatile_strength = (
            volatility_percentile * 0.5 +  # High volatility
            max(0, 1.0 - trend_consistency) * 0.3 +  # Low trend consistency
            min(1.0, abs(momentum_short) * 2.0) * 0.2  # High absolute momentum
        )
        
        if volatility_percentile > self.volatility_threshold:
            volatile_strength *= 1.3
        
        scores[RegimeType.VOLATILE] = volatile_strength
        
        # Find the regime with highest score
        best_regime = max(scores.keys(), key=lambda x: scores[x])
        max_score = scores[best_regime]
        
        # Calculate confidence based on score separation
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            separation = sorted_scores[0] - sorted_scores[1]
            confidence = min(0.95, max(0.1, max_score * (1.0 + separation)))
        else:
            confidence = max_score
        
        # Apply minimum confidence thresholds per regime
        min_confidence = {
            RegimeType.TRENDING_UP: 0.3,
            RegimeType.TRENDING_DOWN: 0.3,
            RegimeType.RANGING: 0.2,
            RegimeType.VOLATILE: 0.25,
        }
        
        if confidence < min_confidence[best_regime]:
            # Fall back to ranging if confidence is too low
            best_regime = RegimeType.RANGING
            confidence = 0.4
        
        return best_regime, min(0.95, max(0.1, confidence))
    
    def _safe_talib(self, func, *args, **kwargs):
        """Safely call TA-Lib function with error handling."""
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.debug("TA-Lib function %s failed: %s", func.__name__, e)
            return np.full(len(args[0]), np.nan)
    
    def _get_last_valid(self, array: np.ndarray, default: float = 0.0) -> float:
        """Get the last valid (non-NaN) value from array."""
        valid_mask = ~np.isnan(array)
        if np.any(valid_mask):
            return float(array[valid_mask][-1])
        return default
    
    def _calculate_momentum(self, prices: np.ndarray, periods: List[int]) -> float:
        """Calculate weighted momentum across multiple periods."""
        if len(prices) < max(periods) + 1:
            return 0.0
        
        momentums = []
        weights = []
        
        for period in periods:
            if len(prices) > period:
                mom = (prices[-1] / prices[-1 - period] - 1.0) * 100
                momentums.append(mom)
                weights.append(1.0 / period)  # Shorter periods get higher weight
        
        if not momentums:
            return 0.0
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        weighted_momentum = np.average(momentums, weights=weights)
        return float(np.tanh(weighted_momentum / 5.0))  # Normalize to roughly -1 to 1
    
    def _calculate_trend_consistency(self, prices: np.ndarray, window: int = 24) -> float:
        """Calculate trend consistency over a rolling window."""
        if len(prices) < window + 1:
            return 0.5
        
        # Calculate rolling returns
        returns = np.diff(np.log(prices[-window-1:]))
        
        if len(returns) == 0:
            return 0.5
        
        # Measure consistency as the ratio of same-sign returns
        positive_returns = np.sum(returns > 0)
        negative_returns = np.sum(returns < 0)
        
        consistency = max(positive_returns, negative_returns) / len(returns)
        return float(consistency)
    
    def get_regime_features(
        self,
        df: pd.DataFrame,
        current_idx: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get numerical features for the detected regime.
        
        Returns a dictionary with regime probabilities and key indicators
        that can be used as input features for trading models.
        """
        if current_idx is None:
            current_idx = len(df) - 1
        
        if current_idx < self.lookback_periods:
            return {
                'regime_trending_up': 0.25,
                'regime_trending_down': 0.25,
                'regime_ranging': 0.25,
                'regime_volatile': 0.25,
                'regime_confidence': 0.1,
                'adx_strength': 0.2,
                'trend_direction': 0.0,
                'volatility_percentile': 0.5,
            }
        
        # Extract data slice
        start_idx = max(0, current_idx - self.lookback_periods + 1)
        data_slice = df.iloc[start_idx:current_idx + 1].copy()
        
        # Calculate indicators
        indicators = self._calculate_indicators(data_slice)
        
        # Get regime probabilities
        regime_scores = self._calculate_regime_probabilities(indicators)
        
        # Detected regime and confidence
        detected_regime, confidence = self._classify_regime(indicators)
        
        features = {
            'regime_trending_up': regime_scores[RegimeType.TRENDING_UP],
            'regime_trending_down': regime_scores[RegimeType.TRENDING_DOWN],
            'regime_ranging': regime_scores[RegimeType.RANGING],
            'regime_volatile': regime_scores[RegimeType.VOLATILE],
            'regime_confidence': confidence,
            'adx_strength': indicators['trend_strength'],
            'trend_direction': indicators['trend_direction'],
            'volatility_percentile': indicators['volatility_percentile'],
        }
        
        return features
    
    def _calculate_regime_probabilities(self, indicators: Dict[str, float]) -> Dict[RegimeType, float]:
        """Calculate probability scores for all regime types."""
        # Use the same logic as classification but return all scores
        adx = indicators['adx']
        trend_direction = indicators['trend_direction']
        trend_strength = indicators['trend_strength']
        volatility_percentile = indicators['volatility_percentile']
        momentum_short = indicators['momentum_short']
        momentum_medium = indicators['momentum_medium']
        trend_consistency = indicators['trend_consistency']
        
        scores = {}
        
        # Trending Up
        trending_up = (
            max(0, trend_direction) * 0.3 +
            max(0, momentum_short) * 0.25 +
            max(0, momentum_medium) * 0.25 +
            (trend_strength if trend_direction > 0 else 0) * 0.2
        )
        if adx >= self.adx_threshold and trend_direction > 0.1:
            trending_up *= (1.0 + trend_consistency * 0.5)
        scores[RegimeType.TRENDING_UP] = trending_up
        
        # Trending Down
        trending_down = (
            max(0, -trend_direction) * 0.3 +
            max(0, -momentum_short) * 0.25 +
            max(0, -momentum_medium) * 0.25 +
            (trend_strength if trend_direction < 0 else 0) * 0.2
        )
        if adx >= self.adx_threshold and trend_direction < -0.1:
            trending_down *= (1.0 + trend_consistency * 0.5)
        scores[RegimeType.TRENDING_DOWN] = trending_down
        
        # Ranging
        ranging = (
            max(0, 1.0 - trend_strength) * 0.4 +
            max(0, 1.0 - abs(momentum_short)) * 0.3 +
            max(0, 1.0 - volatility_percentile) * 0.3
        )
        if adx < self.adx_threshold:
            ranging *= 1.5
        scores[RegimeType.RANGING] = ranging
        
        # Volatile
        volatile = (
            volatility_percentile * 0.5 +
            max(0, 1.0 - trend_consistency) * 0.3 +
            min(1.0, abs(momentum_short) * 2.0) * 0.2
        )
        if volatility_percentile > self.volatility_threshold:
            volatile *= 1.3
        scores[RegimeType.VOLATILE] = volatile
        
        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            for regime in scores:
                scores[regime] = scores[regime] / total
        else:
            # Equal probabilities if no signal
            for regime in scores:
                scores[regime] = 0.25
        
        return scores
    
    def __repr__(self) -> str:
        return (
            f"RegimeDetector("
            f"lookback={self.lookback_periods}, "
            f"adx_period={self.adx_period}, "
            f"trend_threshold={self.trend_threshold})"
        )


def create_regime_detector(config: Optional[Dict[str, Any]] = None) -> RegimeDetector:
    """
    Factory function to create a RegimeDetector.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        RegimeDetector instance
    """
    default_config = {
        'lookback_periods': 96,
        'adx_period': 14,
        'volatility_lookback': 48,
        'trend_threshold': 0.6,
        'volatility_threshold': 0.8,
        'adx_threshold': 25.0,
    }
    
    if config:
        default_config.update(config)
    
    return RegimeDetector(**default_config)