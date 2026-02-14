"""
Multi-Asset Correlation Engine

Fetches and analyzes correlation across multiple assets:
- USDT Dominance (risk-on/risk-off)
- BTC Dominance (altcoin season indicator)
- ETH/BTC ratio
- Total crypto market cap
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import requests
import time

logger = logging.getLogger(__name__)


class CorrelationEngine:
    """
    Multi-Asset Correlation Engine for enhanced market analysis.
    
    Features derived:
    - USDT dominance trend and level
    - BTC dominance for altcoin rotation
    - ETH/BTC for ETH relative strength
    - Cross-asset divergences
    """
    
    BINANCE_BASE = os.environ.get("BINANCE_API_URL", "https://data-api.binance.vision/api/v3/klines")
    
    # Symbols for correlation analysis
    CORRELATION_PAIRS = {
        'ETH': 'ETHUSDT',
        'ETH_BTC': 'ETHBTC',  # ETH/BTC ratio
        'BNB': 'BNBUSDT',
        'SOL': 'SOLUSDT',
    }
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.session = requests.Session()
        self._cache = {}
        
    def fetch_correlation_data(
        self,
        primary_df: pd.DataFrame,
        timeframe: str = '1h',
    ) -> pd.DataFrame:
        """
        Fetch correlation assets and align with primary DataFrame.
        
        Args:
            primary_df: Primary asset DataFrame (e.g., BTC/USDT)
            timeframe: Timeframe to fetch
            
        Returns:
            DataFrame with correlation features added
        """
        result_df = primary_df.copy()
        
        # Get date range from primary data
        start_time = int(primary_df.index[0].timestamp() * 1000)
        end_time = int(primary_df.index[-1].timestamp() * 1000)
        
        # Fetch each correlation pair
        for name, symbol in self.CORRELATION_PAIRS.items():
            try:
                pair_df = self._fetch_asset(symbol, timeframe, start_time, end_time)
                if pair_df is not None and len(pair_df) > 0:
                    # Align with primary index
                    pair_df = pair_df.reindex(primary_df.index, method='ffill')
                    
                    # Add correlation features
                    result_df[f'{name}_close'] = pair_df['close']
                    result_df[f'{name}_return'] = pair_df['close'].pct_change()
                    
                    logger.debug(f"Added {name} correlation data")
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")
                # Fill with NaN which will be handled later
                result_df[f'{name}_close'] = np.nan
                result_df[f'{name}_return'] = np.nan
                
        return result_df
    
    def _fetch_asset(
        self,
        symbol: str,
        timeframe: str,
        start_time: int,
        end_time: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch single asset from Binance."""
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d',
        }
        
        interval = interval_map.get(timeframe, '1h')
        
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_time,
                'limit': 1000,
            }
            
            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(self.BINANCE_BASE, params=params, timeout=30)
                    response.raise_for_status()
                    klines = response.json()
                    
                    if not klines:
                        break
                        
                    all_klines.extend(klines)
                    current_start = klines[-1][0] + 1
                    
                    time.sleep(0.1)  # Rate limiting
                    break
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                    else:
                        raise
                        
            if not klines:
                break
                
        if not all_klines:
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def compute_correlation_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Compute correlation-based features from multi-asset data.
        
        Args:
            df: DataFrame with primary asset + correlation assets
            
        Returns:
            Dictionary of correlation features
        """
        features = {}
        primary_return = df['close'].pct_change()
        
        # ETH/BTC ratio features
        if 'ETH_BTC_close' in df.columns:
            eth_btc = df['ETH_BTC_close']
            features['corr_eth_btc_ratio'] = eth_btc / eth_btc.rolling(20).mean()
            features['corr_eth_btc_trend'] = np.sign(eth_btc.diff(5))
            features['corr_eth_strength'] = (eth_btc - eth_btc.rolling(50).min()) / (
                eth_btc.rolling(50).max() - eth_btc.rolling(50).min() + 1e-10
            )
        
        # ETH correlation
        if 'ETH_return' in df.columns:
            eth_return = df['ETH_return'].fillna(0)
            # Rolling correlation with BTC
            features['corr_eth_btc_corr'] = primary_return.rolling(20).corr(eth_return)
            # Divergence: BTC up, ETH down = potential weakness
            features['corr_eth_divergence'] = np.sign(primary_return) != np.sign(eth_return)
            features['corr_eth_divergence'] = features['corr_eth_divergence'].astype(float)
        
        # SOL correlation (high beta asset)
        if 'SOL_return' in df.columns:
            sol_return = df['SOL_return'].fillna(0)
            features['corr_sol_btc_corr'] = primary_return.rolling(20).corr(sol_return)
            # SOL beta: measures how much SOL moves relative to BTC
            features['corr_sol_beta'] = sol_return.rolling(20).cov(primary_return) / (
                primary_return.rolling(20).var() + 1e-10
            )
        
        # BNB correlation
        if 'BNB_return' in df.columns:
            bnb_return = df['BNB_return'].fillna(0)
            features['corr_bnb_btc_corr'] = primary_return.rolling(20).corr(bnb_return)
        
        # Cross-asset momentum
        altcoin_returns = []
        for col in ['ETH_return', 'SOL_return', 'BNB_return']:
            if col in df.columns:
                altcoin_returns.append(df[col].fillna(0))
                
        if altcoin_returns:
            avg_alt_return = pd.concat(altcoin_returns, axis=1).mean(axis=1)
            features['corr_alt_momentum'] = avg_alt_return.rolling(10).sum()
            # BTC vs Alts: BTC outperforming alts = BTC dominance rising
            features['corr_btc_vs_alts'] = primary_return.rolling(10).sum() - avg_alt_return.rolling(10).sum()
        
        # Simulate USDT dominance (approximation based on market behavior)
        # When BTC and alts dump, USDT.D rises (flight to safety)
        if len(altcoin_returns) > 0:
            avg_market_return = (primary_return + avg_alt_return) / 2
            features['corr_usdt_d_proxy'] = -avg_market_return.rolling(10).sum()  # Inverse of market
            features['corr_risk_off'] = (avg_market_return < -0.01).astype(float).rolling(5).mean()
        
        # Market regime based on correlations
        features['corr_regime'] = self._detect_regime(df, features)
        
        # Fill NaN values
        for key in features:
            if isinstance(features[key], pd.Series):
                features[key] = features[key].fillna(0)
                
        return features
    
    def _detect_regime(self, df: pd.DataFrame, features: Dict[str, pd.Series]) -> pd.Series:
        """
        Detect market regime:
        0 = Neutral
        1 = Risk-On (strong uptrend, high correlation)
        2 = Risk-Off (downtrend, flight to safety)
        3 = Rotation (BTC dominance changing)
        """
        regime = pd.Series(0, index=df.index)
        
        # Price trend
        sma20 = df['close'].rolling(20).mean()
        sma50 = df['close'].rolling(50).mean()
        
        uptrend = (sma20 > sma50) & (df['close'] > sma20)
        downtrend = (sma20 < sma50) & (df['close'] < sma20)
        
        regime[uptrend] = 1  # Risk-On
        regime[downtrend] = 2  # Risk-Off
        
        # Check for rotation (ETH/BTC changing significantly)
        if 'corr_eth_btc_trend' in features:
            eth_rotating = np.abs(features['corr_eth_btc_trend'].rolling(5).sum()) > 3
            regime[eth_rotating & ~uptrend & ~downtrend] = 3
            
        return regime
    
    def get_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Main method to get all correlation features.
        
        Args:
            df: Primary asset DataFrame
            
        Returns:
            Dictionary of all correlation features
        """
        # First fetch correlation data
        enriched_df = self.fetch_correlation_data(df)
        
        # Then compute features
        return self.compute_correlation_features(enriched_df)


class SimulatedDominanceEngine:
    """
    Simulated Dominance metrics when real dominance data is unavailable.
    
    Approximates:
    - USDT.D behavior (inverse of risk sentiment)
    - BTC.D behavior (BTC vs alts performance)
    """
    
    def __init__(self):
        pass
        
    def compute_simulated_dominance(
        self,
        btc_df: pd.DataFrame,
        alt_returns: Optional[pd.Series] = None,
    ) -> Dict[str, pd.Series]:
        """Compute simulated dominance features."""
        features = {}
        
        btc_return = btc_df['close'].pct_change()
        
        # Simulated USDT Dominance
        # When market drops, USDT.D rises (flight to safety)
        market_momentum = btc_return.rolling(10).sum()
        features['sim_usdt_d'] = -market_momentum  # Inverse
        features['sim_usdt_d_rising'] = (features['sim_usdt_d'].diff(5) > 0).astype(float)
        
        # Simulated BTC Dominance
        if alt_returns is not None:
            # BTC.D rises when BTC outperforms alts
            btc_vs_alt = btc_return - alt_returns
            features['sim_btc_d'] = btc_vs_alt.rolling(10).sum()
            features['sim_btc_d_rising'] = (features['sim_btc_d'].diff(5) > 0).astype(float)
        else:
            # Without alt data, use volatility as proxy
            # High volatility periods often see BTC.D rise (flight to BTC)
            vol = btc_return.rolling(10).std()
            vol_ma = vol.rolling(20).mean()
            features['sim_btc_d'] = (vol > vol_ma).astype(float)
            features['sim_btc_d_rising'] = features['sim_btc_d'].diff(5) > 0
        
        # Risk sentiment
        features['sim_risk_sentiment'] = np.tanh(market_momentum * 10)  # -1 to 1
        features['sim_fear'] = (features['sim_risk_sentiment'] < -0.3).astype(float)
        features['sim_greed'] = (features['sim_risk_sentiment'] > 0.3).astype(float)
        
        return features
