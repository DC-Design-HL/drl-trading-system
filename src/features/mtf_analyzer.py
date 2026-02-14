"""
Multi-Timeframe Confluence Analyzer

Requires alignment across multiple timeframes before trading:
- 4H: Overall trend direction (higher timeframe bias)
- 1H: Primary trading timeframe (model operates here)
- 15m: Entry timing (fine-tune entries)

Only trades when all timeframes agree on direction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str
    direction: TrendDirection
    strength: float  # 0-1
    price: float
    ema_fast: float
    ema_slow: float
    rsi: float


@dataclass
class ConfluenceResult:
    """Result of multi-timeframe analysis."""
    aligned: bool
    direction: TrendDirection
    strength: float  # Average strength across timeframes
    signals: Dict[str, TimeframeSignal]
    recommendation: str


class MultiTimeframeAnalyzer:
    """
    Analyze multiple timeframes for trade confluence.
    
    Strategy:
    - 4H determines overall bias (trend direction)
    - 1H is the primary trading timeframe
    - 15m provides entry timing confirmation
    
    Only enter trades when all timeframes align.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30
    ):
        """
        Initialize multi-timeframe analyzer.
        
        Args:
            symbol: Trading symbol
            ema_fast: Fast EMA period
            ema_slow: Slow EMA period
            rsi_period: RSI period
            rsi_overbought: RSI overbought level
            rsi_oversold: RSI oversold level
        """
        self.symbol = symbol
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
        # Timeframes to analyze
        self.timeframes = ["4h", "1h", "15m"]
        
        # Cache for fetched data
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        
        logger.info(f"📊 MTF Analyzer initialized for {symbol} ({', '.join(self.timeframes)})")
    
    def _fetch_klines(self, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch klines from Binance for a specific timeframe."""
        # Check cache (valid for 1 minute)
        cache_key = f"{self.symbol}_{timeframe}"
        if cache_key in self._cache_time:
            if datetime.now() - self._cache_time[cache_key] < timedelta(minutes=1):
                return self._data_cache.get(cache_key)
        
        try:
            import os
            url = os.environ.get("BINANCE_FUTURES_URL", "https://data-api.binance.vision") + "/api/v3/klines"
            params = {
                "symbol": self.symbol,
                "interval": timeframe,
                "limit": limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Cache the result
            self._data_cache[cache_key] = df
            self._cache_time[cache_key] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {timeframe} klines: {e}")
            return None
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def analyze_timeframe(self, timeframe: str, df: pd.DataFrame = None) -> Optional[TimeframeSignal]:
        """
        Analyze a single timeframe.
        
        Args:
            timeframe: Timeframe to analyze (e.g., "4h", "1h", "15m")
            df: Optional pre-fetched dataframe
            
        Returns:
            TimeframeSignal with direction and strength
        """
        if df is None:
            df = self._fetch_klines(timeframe)
        
        if df is None or len(df) < self.ema_slow + 5:
            return None
        
        close = df['close']
        current_price = close.iloc[-1]
        
        # Calculate indicators
        ema_fast = self._calculate_ema(close, self.ema_fast)
        ema_slow = self._calculate_ema(close, self.ema_slow)
        rsi = self._calculate_rsi(close, self.rsi_period)
        
        current_ema_fast = ema_fast.iloc[-1]
        current_ema_slow = ema_slow.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Determine direction
        direction = TrendDirection.NEUTRAL
        strength = 0.0
        
        # EMA crossover direction
        ema_diff_pct = (current_ema_fast - current_ema_slow) / current_ema_slow * 100
        
        if ema_diff_pct > 0.1:  # Fast EMA above slow = bullish
            direction = TrendDirection.BULLISH
            strength = min(abs(ema_diff_pct) / 2, 1.0)  # Cap at 1.0
        elif ema_diff_pct < -0.1:  # Fast EMA below slow = bearish
            direction = TrendDirection.BEARISH
            strength = min(abs(ema_diff_pct) / 2, 1.0)
        
        # RSI confirmation
        if current_rsi > self.rsi_overbought:
            if direction == TrendDirection.BEARISH:
                strength = min(strength * 1.3, 1.0)  # Boost bearish
            elif direction == TrendDirection.BULLISH:
                strength *= 0.7  # Reduce bullish (overbought warning)
        elif current_rsi < self.rsi_oversold:
            if direction == TrendDirection.BULLISH:
                strength = min(strength * 1.3, 1.0)  # Boost bullish
            elif direction == TrendDirection.BEARISH:
                strength *= 0.7  # Reduce bearish (oversold warning)
        
        return TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            price=current_price,
            ema_fast=current_ema_fast,
            ema_slow=current_ema_slow,
            rsi=current_rsi
        )
    
    def get_confluence(self, primary_df: pd.DataFrame = None) -> ConfluenceResult:
        """
        Analyze all timeframes and check for confluence.
        
        Args:
            primary_df: Optional 1H dataframe if already fetched
            
        Returns:
            ConfluenceResult with alignment status and recommendation
        """
        signals: Dict[str, TimeframeSignal] = {}
        
        # Analyze each timeframe
        for tf in self.timeframes:
            df = primary_df if tf == "1h" else None
            signal = self.analyze_timeframe(tf, df)
            if signal:
                signals[tf] = signal
        
        # Check if we have all signals
        if len(signals) < len(self.timeframes):
            missing = set(self.timeframes) - set(signals.keys())
            return ConfluenceResult(
                aligned=False,
                direction=TrendDirection.NEUTRAL,
                strength=0,
                signals=signals,
                recommendation=f"Missing data for: {', '.join(missing)}"
            )
        
        # Count directions
        bullish_count = sum(1 for s in signals.values() if s.direction == TrendDirection.BULLISH)
        bearish_count = sum(1 for s in signals.values() if s.direction == TrendDirection.BEARISH)
        
        # All aligned?
        all_bullish = bullish_count == len(self.timeframes)
        all_bearish = bearish_count == len(self.timeframes)
        aligned = all_bullish or all_bearish
        
        # Determine overall direction
        if all_bullish:
            direction = TrendDirection.BULLISH
        elif all_bearish:
            direction = TrendDirection.BEARISH
        elif bullish_count > bearish_count:
            direction = TrendDirection.BULLISH
        elif bearish_count > bullish_count:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL
        
        # Calculate average strength
        avg_strength = np.mean([s.strength for s in signals.values()])
        
        # Generate recommendation
        if aligned:
            if direction == TrendDirection.BULLISH:
                recommendation = f"✅ ALL BULLISH - Strong LONG setup (strength: {avg_strength:.0%})"
            else:
                recommendation = f"✅ ALL BEARISH - Strong SHORT setup (strength: {avg_strength:.0%})"
        else:
            # Describe the conflict
            tf_dirs = [f"{tf}={s.direction.value}" for tf, s in signals.items()]
            recommendation = f"⚠️ CONFLICTING: {', '.join(tf_dirs)}"
        
        result = ConfluenceResult(
            aligned=aligned,
            direction=direction,
            strength=avg_strength,
            signals=signals,
            recommendation=recommendation
        )
        
        # Log the analysis
        emoji = "✅" if aligned else "⚠️"
        logger.info(
            f"📊 MTF Analysis: {emoji} {direction.value.upper()} | "
            f"4H={signals['4h'].direction.value if '4h' in signals else '?'} "
            f"1H={signals['1h'].direction.value if '1h' in signals else '?'} "
            f"15m={signals['15m'].direction.value if '15m' in signals else '?'}"
        )
        
        return result
    
    def should_trade(self, trade_type: str, primary_df: pd.DataFrame = None) -> Tuple[bool, str]:
        """
        Check if trade is allowed based on MTF confluence.
        
        Args:
            trade_type: "long" or "short"
            primary_df: Optional 1H dataframe
            
        Returns:
            Tuple of (should_trade, reason)
        """
        result = self.get_confluence(primary_df)
        
        # If not aligned, don't trade
        if not result.aligned:
            return False, f"MTF not aligned: {result.recommendation}"
        
        # Check direction matches trade type
        if trade_type == "long" and result.direction == TrendDirection.BULLISH:
            return True, f"MTF confirms LONG: {result.recommendation}"
        elif trade_type == "short" and result.direction == TrendDirection.BEARISH:
            return True, f"MTF confirms SHORT: {result.recommendation}"
        else:
            return False, f"MTF opposes {trade_type.upper()}: Market is {result.direction.value}"
    
    def get_summary(self, primary_df: pd.DataFrame = None) -> Dict:
        """Get a summary dict for UI/logging."""
        result = self.get_confluence(primary_df)
        
        signal_summary = {}
        for tf, signal in result.signals.items():
            signal_summary[tf] = {
                'direction': signal.direction.value,
                'strength': signal.strength,
                'rsi': signal.rsi,
                'emoji': '📈' if signal.direction == TrendDirection.BULLISH else 
                         '📉' if signal.direction == TrendDirection.BEARISH else '➡️'
            }
        
        return {
            'aligned': result.aligned,
            'direction': result.direction.value,
            'strength': result.strength,
            'signals': signal_summary,
            'recommendation': result.recommendation
        }
