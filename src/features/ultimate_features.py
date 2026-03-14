"""
Ultimate Feature Engine
Comprehensive feature engineering for professional-grade trading.

Features:
- Wyckoff Phase Detection
- Smart Money Concepts (Order Blocks, FVG, BOS, CHOCH)
- Market Structure Analysis
- Multi-Timeframe Analysis
- Volume Profile Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class WyckoffAnalyzer:
    """
    Wyckoff Phase Analysis
    
    Detects accumulation/distribution phases and key Wyckoff events:
    - Selling Climax (SC) / Buying Climax (BC)
    - Automatic Rally (AR) / Automatic Reaction
    - Secondary Test (ST)
    - Spring / Upthrust
    - Sign of Strength (SOS) / Sign of Weakness (SOW)
    - Last Point of Support (LPS) / Last Point of Supply (LPSY)
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        
    def detect_climax(self, df: pd.DataFrame) -> pd.Series:
        """Detect Selling/Buying Climax based on volume and price action."""
        volume_ma = df['volume'].rolling(20).mean()
        volume_std = df['volume'].rolling(20).std()
        
        # High volume spike (> 2 std above mean)
        volume_spike = df['volume'] > (volume_ma + 2 * volume_std)
        
        # Price range expansion
        price_range = df['high'] - df['low']
        range_ma = price_range.rolling(20).mean()
        wide_range = price_range > (range_ma * 1.5)
        
        # Selling climax: high volume + wide range + closes near low
        close_near_low = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10) < 0.3
        selling_climax = volume_spike & wide_range & close_near_low
        
        # Buying climax: high volume + wide range + closes near high
        close_near_high = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10) > 0.7
        buying_climax = volume_spike & wide_range & close_near_high
        
        # 1 = selling climax, -1 = buying climax, 0 = none
        climax = pd.Series(0, index=df.index)
        climax[selling_climax] = 1
        climax[buying_climax] = -1
        
        return climax
    
    def detect_spring_upthrust(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect Spring (false breakdown) and Upthrust (false breakout)."""
        # Find local lows and highs
        window = 10
        
        local_low = df['low'].rolling(window, center=True).min()
        local_high = df['high'].rolling(window, center=True).max()
        
        # Support/Resistance as rolling min/max of last N periods
        support = df['low'].rolling(self.lookback).min()
        resistance = df['high'].rolling(self.lookback).max()
        
        # Spring: breaks below support but closes above it
        spring = (df['low'] < support.shift(1)) & (df['close'] > support.shift(1))
        
        # Upthrust: breaks above resistance but closes below it
        upthrust = (df['high'] > resistance.shift(1)) & (df['close'] < resistance.shift(1))
        
        return spring.astype(float), upthrust.astype(float)
    
    def detect_phase(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify Wyckoff phase (simplified):
        0 = Unknown/Ranging
        1 = Accumulation (Phase A-C)
        2 = Markup
        3 = Distribution (Phase A-C)
        4 = Markdown
        """
        # Calculate trend
        sma20 = df['close'].rolling(20).mean()
        sma50 = df['close'].rolling(50).mean()
        
        # Volume analysis
        vol_ma = df['volume'].rolling(20).mean()
        vol_increasing = df['volume'] > vol_ma
        
        # Price momentum
        roc = df['close'].pct_change(10)
        
        # Volatility (shrinking = accumulation/distribution)
        atr = self._calculate_atr(df, 14)
        atr_ma = atr.rolling(20).mean()
        low_volatility = atr < atr_ma * 0.8
        
        phase = pd.Series(0, index=df.index)
        
        # Markup: uptrend with expanding volume
        markup = (sma20 > sma50) & (roc > 0.02)
        phase[markup] = 2
        
        # Markdown: downtrend
        markdown = (sma20 < sma50) & (roc < -0.02)
        phase[markdown] = 4
        
        # Accumulation: low volatility after downtrend, volume increasing
        accum = low_volatility & (sma20 < sma50) & vol_increasing
        phase[accum] = 1
        
        # Distribution: low volatility after uptrend
        distrib = low_volatility & (sma20 > sma50) & vol_increasing
        phase[distrib] = 3
        
        return phase
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def get_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get all Wyckoff features."""
        climax = self.detect_climax(df)
        spring, upthrust = self.detect_spring_upthrust(df)
        phase = self.detect_phase(df)
        
        return {
            'wyckoff_climax': climax,
            'wyckoff_spring': spring,
            'wyckoff_upthrust': upthrust,
            'wyckoff_phase': phase,
            'wyckoff_accumulation': (phase == 1).astype(float),
            'wyckoff_distribution': (phase == 3).astype(float),
            'wyckoff_markup': (phase == 2).astype(float),
            'wyckoff_markdown': (phase == 4).astype(float),
        }


class SMCAnalyzer:
    """
    Smart Money Concepts (SMC) Analysis
    
    Detects:
    - Order Blocks (demand/supply zones)
    - Fair Value Gaps (FVG)
    - Break of Structure (BOS)
    - Change of Character (CHOCH)
    - Liquidity pools
    """
    
    def __init__(self, swing_lookback: int = 5):
        self.swing_lookback = swing_lookback
        
    def detect_swing_points(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect swing highs and lows."""
        window = self.swing_lookback
        
        # Swing high: high is highest in window
        swing_high = df['high'] == df['high'].rolling(window * 2 + 1, center=True).max()
        
        # Swing low: low is lowest in window
        swing_low = df['low'] == df['low'].rolling(window * 2 + 1, center=True).min()
        
        return swing_high.astype(float), swing_low.astype(float)
    
    def detect_order_blocks(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect Order Blocks (OB):
        - Bullish OB: last bearish candle before impulsive bullish move
        - Bearish OB: last bullish candle before impulsive bearish move
        """
        # Calculate candle direction
        bullish = df['close'] > df['open']
        bearish = df['close'] < df['open']
        
        # Impulsive move: large body relative to ATR
        body = np.abs(df['close'] - df['open'])
        atr = self._calculate_atr(df, 14)
        impulsive = body > atr * 1.5
        
        # Bullish OB: bearish candle followed by impulsive bullish
        bullish_ob = bearish.shift(1) & bullish & impulsive
        
        # Bearish OB: bullish candle followed by impulsive bearish
        bearish_ob = bullish.shift(1) & bearish & impulsive
        
        return bullish_ob.astype(float), bearish_ob.astype(float)
    
    def detect_fvg(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect Fair Value Gaps (FVG):
        - Bullish FVG: gap between candle 1 high and candle 3 low
        - Bearish FVG: gap between candle 1 low and candle 3 high
        """
        # Bullish FVG: candle[i-2].high < candle[i].low
        bullish_fvg = df['high'].shift(2) < df['low']
        
        # Bearish FVG: candle[i-2].low > candle[i].high
        bearish_fvg = df['low'].shift(2) > df['high']
        
        return bullish_fvg.astype(float), bearish_fvg.astype(float)
    
    def detect_bos_choch(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect Break of Structure (BOS) and Change of Character (CHOCH):
        - BOS: break of previous swing high/low in trend direction
        - CHOCH: break of previous swing in opposite direction (trend reversal)
        """
        swing_high, swing_low = self.detect_swing_points(df)
        
        # Track previous swing levels
        prev_swing_high = df['high'].where(swing_high.astype(bool)).ffill()
        prev_swing_low = df['low'].where(swing_low.astype(bool)).ffill()
        
        # BOS bullish: break above previous swing high
        bos_bullish = df['close'] > prev_swing_high.shift(1)
        
        # BOS bearish: break below previous swing low
        bos_bearish = df['close'] < prev_swing_low.shift(1)
        
        # Simple trend tracking
        sma20 = df['close'].rolling(20).mean()
        uptrend = df['close'] > sma20
        downtrend = df['close'] < sma20
        
        # CHOCH: BOS in opposite direction of trend
        choch_bullish = bos_bullish & downtrend.shift(1)
        choch_bearish = bos_bearish & uptrend.shift(1)
        
        bos = pd.Series(0, index=df.index)
        bos[bos_bullish] = 1
        bos[bos_bearish] = -1
        
        choch = pd.Series(0, index=df.index)
        choch[choch_bullish] = 1
        choch[choch_bearish] = -1
        
        return bos, choch
    
    def detect_liquidity(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect liquidity pools (clusters of swing highs/lows)."""
        swing_high, swing_low = self.detect_swing_points(df)
        
        # Count nearby swing points
        window = 20
        liquidity_above = swing_high.rolling(window).sum()
        liquidity_below = swing_low.rolling(window).sum()
        
        return liquidity_above, liquidity_below
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def get_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get all SMC features."""
        swing_high, swing_low = self.detect_swing_points(df)
        bullish_ob, bearish_ob = self.detect_order_blocks(df)
        bullish_fvg, bearish_fvg = self.detect_fvg(df)
        bos, choch = self.detect_bos_choch(df)
        liq_above, liq_below = self.detect_liquidity(df)
        
        return {
            'smc_swing_high': swing_high,
            'smc_swing_low': swing_low,
            'smc_bullish_ob': bullish_ob,
            'smc_bearish_ob': bearish_ob,
            'smc_bullish_fvg': bullish_fvg,
            'smc_bearish_fvg': bearish_fvg,
            'smc_bos': bos,
            'smc_choch': choch,
            'smc_liquidity_above': liq_above,
            'smc_liquidity_below': liq_below,
        }


class MarketStructureAnalyzer:
    """
    Market Structure Analysis
    
    Detects:
    - Higher Highs / Higher Lows (HH/HL) - Uptrend
    - Lower Highs / Lower Lows (LH/LL) - Downtrend
    - Support / Resistance levels
    - Trend strength
    """
    
    def __init__(self, pivot_lookback: int = 5):
        self.pivot_lookback = pivot_lookback
        
    def detect_structure(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Detect market structure (HH, HL, LH, LL)."""
        window = self.pivot_lookback
        
        # Find pivot points
        pivot_high = df['high'] == df['high'].rolling(window * 2 + 1, center=True).max()
        pivot_low = df['low'] == df['low'].rolling(window * 2 + 1, center=True).min()
        
        # Track previous pivots
        prev_pivot_high = df['high'].where(pivot_high).ffill()
        prev_pivot_low = df['low'].where(pivot_low).ffill()
        
        # Higher High: current pivot high > previous pivot high
        hh = pivot_high & (df['high'] > prev_pivot_high.shift(1))
        
        # Higher Low: current pivot low > previous pivot low
        hl = pivot_low & (df['low'] > prev_pivot_low.shift(1))
        
        # Lower High: current pivot high < previous pivot high
        lh = pivot_high & (df['high'] < prev_pivot_high.shift(1))
        
        # Lower Low: current pivot low < previous pivot low
        ll = pivot_low & (df['low'] < prev_pivot_low.shift(1))
        
        return {
            'structure_hh': hh.astype(float),
            'structure_hl': hl.astype(float),
            'structure_lh': lh.astype(float),
            'structure_ll': ll.astype(float),
        }
    
    def detect_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect trend based on structure:
        1 = Uptrend (HH + HL)
        -1 = Downtrend (LH + LL)
        0 = Ranging
        """
        structure = self.detect_structure(df)
        
        # Count HH/HL vs LH/LL in rolling window
        window = 20
        
        bullish_count = (
            structure['structure_hh'].rolling(window).sum() +
            structure['structure_hl'].rolling(window).sum()
        )
        
        bearish_count = (
            structure['structure_lh'].rolling(window).sum() +
            structure['structure_ll'].rolling(window).sum()
        )
        
        trend = pd.Series(0, index=df.index)
        trend[bullish_count > bearish_count + 1] = 1
        trend[bearish_count > bullish_count + 1] = -1
        
        return trend
    
    def detect_sr_levels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect Support/Resistance levels."""
        window = 50
        
        # Rolling support/resistance
        support = df['low'].rolling(window).min()
        resistance = df['high'].rolling(window).max()
        
        # Distance to S/R (normalized)
        range_size = resistance - support + 1e-10
        dist_to_support = (df['close'] - support) / range_size
        dist_to_resistance = (resistance - df['close']) / range_size
        
        return dist_to_support, dist_to_resistance
    
    def get_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get all market structure features."""
        structure = self.detect_structure(df)
        trend = self.detect_trend(df)
        dist_support, dist_resistance = self.detect_sr_levels(df)
        
        features = structure.copy()
        features['structure_trend'] = trend
        features['structure_dist_support'] = dist_support
        features['structure_dist_resistance'] = dist_resistance
        
        # Trend strength
        sma_short = df['close'].rolling(10).mean()
        sma_long = df['close'].rolling(50).mean()
        features['structure_trend_strength'] = (sma_short - sma_long) / sma_long
        
        return features


class VolumeProfileAnalyzer:
    """
    Volume Profile Analysis
    
    Features:
    - Point of Control (POC)
    - Value Area High/Low (VAH/VAL)
    - VWAP
    - Volume at price zones
    """
    
    def __init__(self, lookback: int = 50, num_bins: int = 20):
        self.lookback = lookback
        self.num_bins = num_bins
        
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_vol = (typical_price * df['volume']).cumsum()
        cumulative_vol = df['volume'].cumsum()
        
        vwap = cumulative_tp_vol / cumulative_vol
        return vwap
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume profile metrics."""
        vwap = self.calculate_vwap(df)
        
        # Price position relative to VWAP
        vwap_distance = (df['close'] - vwap) / vwap
        above_vwap = (df['close'] > vwap).astype(float)
        
        # Volume moving averages
        vol_sma = df['volume'].rolling(20).mean()
        vol_ratio = df['volume'] / vol_sma
        
        # On-balance volume
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        obv_normalized = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-10)
        
        # Volume price trend
        vpt = (df['close'].pct_change() * df['volume']).cumsum()
        vpt_normalized = (vpt - vpt.rolling(50).mean()) / (vpt.rolling(50).std() + 1e-10)
        
        # Accumulation/Distribution Line
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        adl = (mfm * df['volume']).cumsum()
        adl_normalized = (adl - adl.rolling(50).mean()) / (adl.rolling(50).std() + 1e-10)
        
        return {
            'volume_vwap_distance': vwap_distance,
            'volume_above_vwap': above_vwap,
            'volume_ratio': vol_ratio.clip(0, 5),  # Cap extreme values
            'volume_obv': obv_normalized,
            'volume_vpt': vpt_normalized,
            'volume_adl': adl_normalized,
        }
    
    def get_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get all volume profile features."""
        return self.calculate_volume_profile(df)


class UltimateFeatureEngine:
    """
    Ultimate Feature Engine combining all analysis methods.
    
    Total features: ~150+
    """
    
    def __init__(self):
        self.wyckoff = WyckoffAnalyzer()
        self.smc = SMCAnalyzer()
        self.structure = MarketStructureAnalyzer()
        self.volume = VolumeProfileAnalyzer()
        
    def compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute all features and return as numpy array."""
        features_dict = self.get_all_features(df)
        
        # Convert to DataFrame and then numpy array
        features_df = pd.DataFrame(features_dict)
        
        # Fill NaN with 0
        features_df = features_df.fillna(0)
        
        # Replace inf with large values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        return features_df.values
    
    def get_all_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get all features as dictionary."""
        all_features = {}
        
        # 1. Basic price features
        all_features.update(self._get_price_features(df))
        
        # 2. Technical indicators
        all_features.update(self._get_technical_features(df))
        
        # 3. Wyckoff features
        all_features.update(self.wyckoff.get_features(df))
        
        # 4. SMC features
        all_features.update(self.smc.get_features(df))
        
        # 5. Market structure features
        all_features.update(self.structure.get_features(df))
        
        # 6. Volume profile features
        all_features.update(self.volume.get_features(df))
        
        # 7. Whale-proxy features (approximate institutional behavior)
        all_features.update(self._get_whale_proxy_features(df))
        
        # 8. NEW: Explicit Whale Action Vectors (On-Chain Proxies)
        all_features.update(self._get_whale_action_vectors(df))
        
        return all_features
    
    def _get_whale_action_vectors(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculates explicit Whale Action Vectors simulating on-chain network data.
        These directly mimic the signals from the WhalePatternPredictor model.
        """
        features = {}
        
        vol_ma = df['volume'].rolling(20).mean()
        price_range = df['high'] - df['low']
        
        # 1. Stealth Accumulation (Whales buying quietly without pumping price)
        # Condition: High volume, but very small price range and close near open
        small_body = np.abs(df['close'] - df['open']) < (price_range * 0.2)
        stealth_vol = df['volume'] > vol_ma
        features['whale_stealth_accumulation'] = (small_body & stealth_vol).astype(float).rolling(10).mean()
        
        # NOTE: Removed whale_capitulation_index (Sharpe -8.5, toxic feature)
        # NOTE: Removed whale_fomo_index (Sharpe -5.4, toxic feature)
        
        # 4. Large Tx Ratio Proxy (Percentage of volume happening in macro-moves)
        large_move = price_range > price_range.rolling(20).mean() * 1.5
        macro_vol = df['volume'].where(large_move, 0)
        features['whale_large_tx_ratio'] = (macro_vol.rolling(10).sum() / (df['volume'].rolling(10).sum() + 1e-10)).clip(0, 1)
        
        # 5. Net Flow Proxy (Institutional Net Buying/Selling Pressure)
        # Buy volume proxy = total volume * (close - low) / range
        buy_vol_proxy = df['volume'] * ((df['close'] - df['low']) / (price_range + 1e-10))
        sell_vol_proxy = df['volume'] * ((df['high'] - df['close']) / (price_range + 1e-10))
        net_flow_raw = buy_vol_proxy - sell_vol_proxy
        features['whale_net_flow_proxy'] = (net_flow_raw - net_flow_raw.rolling(50).mean()) / (net_flow_raw.rolling(50).std() + 1e-10)
        
        return features
    
    def _get_whale_proxy_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Whale-proxy features that approximate institutional/whale behavior.
        These are derived from OHLCV data to identify large player activity.
        """
        features = {}
        
        # 1. Whale Volume Spike - large player activity detection
        vol_ma = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        features['whale_volume_spike'] = (df['volume'] / (vol_ma + 1e-10)).clip(0, 5)
        features['whale_volume_zscore'] = ((df['volume'] - vol_ma) / (vol_std + 1e-10)).clip(-3, 3)
        
        # 2. Accumulation/Distribution Detection
        # Accumulation: close > open on high volume (institutional buying)
        # Distribution: close < open on high volume (institutional selling)
        body_up = (df['close'] > df['open']).astype(float)
        body_down = (df['close'] < df['open']).astype(float)
        high_volume = (df['volume'] > vol_ma * 1.5).astype(float)

        # NOTE: Removed whale_accumulation (redundant) and whale_distribution (Sharpe -6.4, toxic)
        # Compute accumulation_dist_ratio directly (this feature is kept - statistically significant)
        whale_acc = (body_up * high_volume).rolling(5).sum() / 5
        whale_dist = (body_down * high_volume).rolling(5).sum() / 5
        features['whale_accumulation_dist_ratio'] = (whale_acc - whale_dist).clip(-1, 1)
        
        # 3. Crowd Sentiment Proxy (approximates Fear & Greed)
        # Combines RSI extremes with volume for sentiment signal
        rsi = self._calculate_rsi(df['close'], 14)
        rsi_extreme_greed = (rsi > 70).astype(float)

        # NOTE: Removed whale_crowd_fear (Sharpe -7.4, toxic feature)
        # Keep whale_crowd_greed - Win Rate 54.5% (best whale feature!)
        features['whale_crowd_greed'] = (rsi_extreme_greed * high_volume).rolling(5).mean()
        
        # Sentiment score: -1 (fear) to +1 (greed)
        features['whale_sentiment_score'] = (
            (rsi / 100 - 0.5) * 2
        ).rolling(10).mean().clip(-1, 1)
        
        # 4. Smart Money Divergence
        # Price vs volume divergence detection
        price_direction = np.sign(df['close'].diff(5))
        volume_direction = np.sign(df['volume'].diff(5))
        
        # Positive divergence: price down but volume up (smart money buying)
        features['whale_positive_divergence'] = (
            ((price_direction < 0) & (volume_direction > 0)).astype(float)
        ).rolling(5).mean()
        
        # Negative divergence: price up but volume down (distribution)
        features['whale_negative_divergence'] = (
            ((price_direction > 0) & (volume_direction < 0)).astype(float)
        ).rolling(5).mean()
        
        # 5. Open Interest Proxy (volatility expansion/contraction)
        # High vol expansion = positions being opened, contraction = being closed
        returns = df['close'].pct_change()
        volatility = returns.rolling(14).std()
        vol_expanding = volatility > volatility.rolling(50).mean()
        vol_contracting = volatility < volatility.rolling(50).mean() * 0.7
        
        features['whale_oi_proxy_expanding'] = vol_expanding.astype(float)
        features['whale_oi_proxy_contracting'] = vol_contracting.astype(float)
        
        # 6. Long/Short Ratio Proxy
        # Based on candle close position and volume
        # Close in upper half = more longs, lower half = more shorts
        close_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        volume_weighted_position = close_position * df['volume']
        
        features['whale_ls_ratio_proxy'] = (
            volume_weighted_position.rolling(10).mean() / 
            (df['volume'].rolling(10).mean() + 1e-10) - 0.5
        ) * 2  # Scale to -1 to +1
        
        # 7. Top Trader Proxy (large candle with high volume)
        # Big players move price significantly with volume
        price_range = df['high'] - df['low']
        range_ma = price_range.rolling(20).mean()
        large_range = price_range > range_ma * 1.5
        
        features['whale_large_player_activity'] = (
            (large_range & (df['volume'] > vol_ma * 1.5)).astype(float)
        ).rolling(5).mean()
        
        # 8. Institutional vs Retail Flow Proxy
        # Large moves during "smart money" hours approximation
        # Using volume patterns as proxy
        features['whale_institutional_flow'] = (
            (df['volume'] / (vol_ma + 1e-10)) * 
            np.abs(df['close'] - df['open']) / (price_range + 1e-10)
        ).rolling(5).mean().clip(0, 3)
        
        # ===== FUNDING RATE PROXY FEATURES =====
        # Funding rate correlates with price premium and market imbalance
        
        # 9. Price Premium Proxy (basis approximation)
        # When price is above recent average, funding tends to be positive
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        features['funding_premium_proxy'] = ((df['close'] - ema_20) / ema_20 * 100).clip(-2, 2)
        
        # 10. Momentum-based funding proxy
        # Strong uptrends = positive funding, strong downtrends = negative
        momentum_10 = df['close'].pct_change(10)
        momentum_20 = df['close'].pct_change(20)
        features['funding_momentum_proxy'] = (momentum_10 + momentum_20).clip(-0.1, 0.1) * 10
        
        # 11. Funding rate extreme detector
        # Extreme premium = potential reversal
        extreme_premium = np.abs(features['funding_premium_proxy']) > 1.5
        features['funding_extreme'] = extreme_premium.astype(float)
        
        # ===== ORDER FLOW PROXY FEATURES =====
        
        # 12. CVD Proxy from candle analysis
        body = df['close'] - df['open']
        range_total = df['high'] - df['low']
        body_ratio = body / (range_total + 1e-10)
        raw_cvd = (body_ratio * df['volume']).cumsum()
        # Normalize to recent range
        features['orderflow_cvd'] = ((raw_cvd - raw_cvd.rolling(20).mean()) / 
                                      (raw_cvd.rolling(20).std() + 1e-10)).clip(-3, 3)
        
        # 13. Buying/Selling Pressure
        upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
        lower_wick = df[['close', 'open']].min(axis=1) - df['low']
        
        # More lower wick = buying pressure (rejected lows)
        # More upper wick = selling pressure (rejected highs)
        features['orderflow_buy_pressure'] = (lower_wick / (range_total + 1e-10)).rolling(5).mean()
        features['orderflow_sell_pressure'] = (upper_wick / (range_total + 1e-10)).rolling(5).mean()
        features['orderflow_pressure_diff'] = (features['orderflow_buy_pressure'] - 
                                                features['orderflow_sell_pressure']).clip(-0.5, 0.5)
        
        # 14. Large Order Proxy (volume spikes with directional moves)
        vol_spike = df['volume'] > vol_ma * 2
        up_move = df['close'] > df['open']
        down_move = df['close'] < df['open']
        
        features['orderflow_large_buys'] = (vol_spike & up_move).astype(float).rolling(10).mean()
        features['orderflow_large_sells'] = (vol_spike & down_move).astype(float).rolling(10).mean()
        features['orderflow_large_bias'] = (features['orderflow_large_buys'] - 
                                             features['orderflow_large_sells']).clip(-1, 1)
        
        # 15. Market Regime Confidence
        # Combine multiple signals for overall market state
        trend_signal = (ema_20 > ema_50).astype(float) * 2 - 1  # 1 or -1
        volume_confirms = (df['volume'] > vol_ma).astype(float)
        features['regime_confidence'] = (trend_signal * volume_confirms * 
                                          np.abs(features['funding_momentum_proxy'])).clip(-1, 1)
        
        return features
    
    def _get_price_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Basic price-based features."""
        features = {}
        
        # Returns
        for period in [1, 5, 10, 20]:
            features[f'return_{period}'] = df['close'].pct_change(period)
        
        # Log returns
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        features['volatility_10'] = df['close'].pct_change().rolling(10).std()
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # Price position in range
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Body ratio
        features['body_ratio'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        
        # Candle direction
        features['candle_direction'] = np.sign(df['close'] - df['open'])
        
        # Gap
        features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return features
    
    def _get_technical_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Technical indicator features."""
        features = {}
        
        # RSI
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self._calculate_rsi(df['close'], period) / 100
        
        # MACD
        macd, signal, hist = self._calculate_macd(df['close'])
        features['macd'] = macd / df['close']  # Normalize
        features['macd_signal'] = signal / df['close']
        features['macd_hist'] = hist / df['close']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger(df['close'])
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR
        atr = self._calculate_atr(df)
        features['atr_normalized'] = atr / df['close']
        
        # Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(df)
        features['stoch_k'] = stoch_k / 100
        features['stoch_d'] = stoch_d / 100
        
        # Moving averages
        for period in [10, 20, 50, 100]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}_dist'] = (df['close'] - sma) / sma
        
        # EMA crossovers
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['ema_cross'] = (ema_12 > ema_26).astype(float)
        
        # ADX (trend strength)
        features['adx'] = self._calculate_adx(df) / 100
        
        # CCI
        features['cci'] = self._calculate_cci(df) / 200  # Normalize around 0
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        return macd, signal, hist
    
    def _calculate_bollinger(self, prices: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std_dev = prices.rolling(period).std()
        upper = middle + std * std_dev
        lower = middle - std * std_dev
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator."""
        lowest_low = df['low'].rolling(k_period).min()
        highest_high = df['high'].rolling(k_period).max()
        stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(d_period).mean()
        return stoch_k, stoch_d
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX."""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._calculate_atr(df, 1)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / (tr.rolling(period).mean() + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (tr.rolling(period).mean() + 1e-10))
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate CCI."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (typical_price - sma) / (0.015 * mad + 1e-10)
        return cci
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        # Create a dummy DataFrame to get feature names
        dummy_df = pd.DataFrame({
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.rand(100),
        })
        dummy_df['high'] = dummy_df[['open', 'close', 'high']].max(axis=1)
        dummy_df['low'] = dummy_df[['open', 'close', 'low']].min(axis=1)
        
        features = self.get_all_features(dummy_df)
        return list(features.keys())
