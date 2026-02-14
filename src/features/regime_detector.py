"""
Market Regime Detection Module

Identifies market conditions to adapt trading strategy:
- TRENDING_UP: Strong uptrend (ADX > 25, +DI > -DI)
- TRENDING_DOWN: Strong downtrend (ADX > 25, -DI > +DI)
- RANGING: No clear trend (ADX < 20)
- HIGH_VOLATILITY: Large price swings (ATR > 1.5x average)
- LOW_VOLATILITY: Quiet market (ATR < 0.5x average)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeInfo:
    """Container for regime analysis results."""
    regime: MarketRegime
    trend_strength: float  # ADX value (0-100)
    trend_direction: float  # +1 bullish, -1 bearish, 0 neutral
    volatility_ratio: float  # Current ATR / Average ATR
    confidence: float  # 0-1 confidence in the regime
    recommendation: str  # Trading recommendation


class MarketRegimeDetector:
    """
    Detect market regime using ADX, ATR, and price action.
    
    Adapts trading behavior:
    - Trending: Follow trend, wider stops
    - Ranging: Mean reversion, tight stops
    - High Vol: Reduce size, use momentum
    - Low Vol: Skip or use breakout
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        trend_threshold: float = 25.0,
        range_threshold: float = 20.0,
        vol_lookback: int = 50
    ):
        """
        Initialize regime detector.
        
        Args:
            adx_period: Period for ADX calculation
            atr_period: Period for ATR calculation
            trend_threshold: ADX above this = trending
            range_threshold: ADX below this = ranging
            vol_lookback: Bars to compare current volatility
        """
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.vol_lookback = vol_lookback
        
        logger.info(f"📊 MarketRegimeDetector initialized (ADX>{trend_threshold}=trend, ADX<{range_threshold}=range)")
    
    def calculate_adx(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX, +DI, and -DI.
        
        Returns:
            Tuple of (ADX, +DI, -DI) series
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.adx_period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        # Smoothed DM
        plus_dm_smooth = plus_dm.rolling(self.adx_period).mean()
        minus_dm_smooth = minus_dm.rolling(self.adx_period).mean()
        
        # DI calculations
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx, plus_di, minus_di
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(self.atr_period).mean()
    
    def detect_regime(self, df: pd.DataFrame) -> RegimeInfo:
        """
        Detect current market regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            RegimeInfo with current regime and metrics
        """
        if len(df) < self.vol_lookback + self.adx_period:
            return RegimeInfo(
                regime=MarketRegime.UNKNOWN,
                trend_strength=0,
                trend_direction=0,
                volatility_ratio=1.0,
                confidence=0,
                recommendation="Insufficient data"
            )
        
        # Calculate indicators
        adx, plus_di, minus_di = self.calculate_adx(df)
        atr = self.calculate_atr(df)
        
        # Get current values
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-self.vol_lookback:].mean()
        
        # Volatility ratio
        vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Trend direction: +1 bullish, -1 bearish
        trend_direction = 0.0
        if current_plus_di > current_minus_di:
            trend_direction = min((current_plus_di - current_minus_di) / 20, 1.0)
        else:
            trend_direction = max((current_plus_di - current_minus_di) / 20, -1.0)
        
        # Determine regime
        regime = MarketRegime.UNKNOWN
        confidence = 0.0
        recommendation = ""
        
        # Check volatility first
        if vol_ratio > 1.5:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min((vol_ratio - 1.5) / 0.5 + 0.5, 1.0)
            recommendation = "High volatility - reduce position size, use momentum"
        elif vol_ratio < 0.5:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min((0.5 - vol_ratio) / 0.25 + 0.5, 1.0)
            recommendation = "Low volatility - wait for breakout or skip"
        # Then check trend
        elif current_adx >= self.trend_threshold:
            if trend_direction > 0:
                regime = MarketRegime.TRENDING_UP
                confidence = min(current_adx / 40, 1.0)
                recommendation = "Uptrend - follow trend, buy dips"
            else:
                regime = MarketRegime.TRENDING_DOWN
                confidence = min(current_adx / 40, 1.0)
                recommendation = "Downtrend - follow trend, sell rallies"
        elif current_adx <= self.range_threshold:
            regime = MarketRegime.RANGING
            confidence = min((self.range_threshold - current_adx) / 10 + 0.5, 1.0)
            recommendation = "Ranging - mean reversion, tight stops"
        else:
            # Between range and trend thresholds
            regime = MarketRegime.RANGING
            confidence = 0.3
            recommendation = "Weak trend - be cautious"
        
        result = RegimeInfo(
            regime=regime,
            trend_strength=current_adx,
            trend_direction=trend_direction,
            volatility_ratio=vol_ratio,
            confidence=confidence,
            recommendation=recommendation
        )
        
        logger.info(
            f"📊 Regime: {regime.value.upper()} | "
            f"ADX: {current_adx:.1f} | "
            f"Direction: {'↑' if trend_direction > 0 else '↓' if trend_direction < 0 else '→'} | "
            f"Vol: {vol_ratio:.2f}x"
        )
        
        return result
    
    def should_trade(self, df: pd.DataFrame, trade_type: str = "any") -> Tuple[bool, str, float]:
        """
        Check if trading is advisable given current regime.
        
        Args:
            df: DataFrame with OHLCV data
            trade_type: "long", "short", or "any"
            
        Returns:
            Tuple of (should_trade, reason, position_size_multiplier)
        """
        regime_info = self.detect_regime(df)
        
        # Default position size multiplier
        size_mult = 1.0
        
        if regime_info.regime == MarketRegime.UNKNOWN:
            return True, "Unknown regime - using default", 1.0
        
        # High volatility: reduce size
        if regime_info.regime == MarketRegime.HIGH_VOLATILITY:
            size_mult = 0.5  # Half position size
            return True, f"High vol ({regime_info.volatility_ratio:.1f}x) - reduced size", size_mult
        
        # Low volatility: skip or small size
        if regime_info.regime == MarketRegime.LOW_VOLATILITY:
            if regime_info.confidence > 0.7:
                return False, "Very low volatility - skipping", 0.0
            size_mult = 0.75
            return True, "Low vol - smaller size", size_mult
        
        # Trending up
        if regime_info.regime == MarketRegime.TRENDING_UP:
            if trade_type == "short":
                return False, f"Uptrend (ADX={regime_info.trend_strength:.0f}) - blocking SHORT", 0.0
            size_mult = 1.2  # Larger size in trend
            return True, f"Uptrend - favor LONG", size_mult
        
        # Trending down
        if regime_info.regime == MarketRegime.TRENDING_DOWN:
            if trade_type == "long":
                return False, f"Downtrend (ADX={regime_info.trend_strength:.0f}) - blocking LONG", 0.0
            size_mult = 1.2
            return True, f"Downtrend - favor SHORT", size_mult
        
        # Ranging
        if regime_info.regime == MarketRegime.RANGING:
            size_mult = 0.8  # Smaller size in range
            return True, f"Ranging market - use mean reversion", size_mult
        
        return True, "Default regime rules", 1.0
    
    def get_regime_summary(self, df: pd.DataFrame) -> Dict:
        """Get a summary dict of current regime for UI/logging."""
        regime_info = self.detect_regime(df)
        
        return {
            'regime': regime_info.regime.value,
            'trend_strength': regime_info.trend_strength,
            'trend_direction': regime_info.trend_direction,
            'volatility_ratio': regime_info.volatility_ratio,
            'confidence': regime_info.confidence,
            'recommendation': regime_info.recommendation,
            'emoji': self._get_regime_emoji(regime_info.regime)
        }
    
    def _get_regime_emoji(self, regime: MarketRegime) -> str:
        """Get emoji for regime."""
        emojis = {
            MarketRegime.TRENDING_UP: "📈",
            MarketRegime.TRENDING_DOWN: "📉",
            MarketRegime.RANGING: "↔️",
            MarketRegime.HIGH_VOLATILITY: "🌊",
            MarketRegime.LOW_VOLATILITY: "😴",
            MarketRegime.UNKNOWN: "❓"
        }
        return emojis.get(regime, "❓")
