"""
Multi-Asset Feature Engine

Extends UltimateFeatureEngine to support multiple assets with:
- Asset embeddings (unique features per asset)
- Cross-asset correlation features (BTC dominance effect)
- Asset-specific risk scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from src.features.ultimate_features import UltimateFeatureEngine
from src.data.multi_asset_fetcher import SUPPORTED_ASSETS, get_asset_embedding

logger = logging.getLogger(__name__)


class MultiAssetFeatureEngine:
    """
    Feature engine for multi-asset trading.
    
    Extends base features with:
    - Asset ID embedding (4 features)
    - Cross-asset features (BTC influence, correlation regimes)
    - Asset-specific volatility scaling
    
    Total features: 94 (base) + 4 (asset) + 6 (cross-asset) = 104
    """
    
    # Asset embedding features
    ASSET_FEATURES = [
        'asset_id_norm',       # Normalized asset ID
        'base_volatility',     # Expected volatility vs BTC
        'liquidity_score',     # Relative liquidity (1.0 = BTC)
        'btc_correlation',     # Typical correlation with BTC
    ]
    
    # Cross-asset features (when BTC data is available)
    CROSS_ASSET_FEATURES = [
        'btc_return_1h',       # BTC return last 1h (leader signal)
        'btc_return_4h',       # BTC return last 4h
        'btc_momentum',        # BTC momentum (EMA slope)
        'btc_volatility',      # BTC current volatility
        'relative_strength',   # Asset return vs BTC return (RS)
        'correlation_regime',  # Current correlation regime
    ]
    
    # Alternative Data Features (Phase 11.4)
    ALT_DATA_FEATURES = [
        'fear_greed_value',
        'fear_greed_class',
        'btc_dominance',
        'altcoin_season_index',
    ]
    
    def __init__(
        self,
        include_cross_asset: bool = True,
        btc_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize multi-asset feature engine.
        
        Args:
            include_cross_asset: Whether to include BTC cross-asset features
            btc_data: BTC price data for cross-asset calculations
        """
        self.base_engine = UltimateFeatureEngine()
        self.include_cross_asset = include_cross_asset
        self.btc_data = btc_data
        
        from src.features.alternative_data import AlternativeDataCollector
        self.alt_collector = AlternativeDataCollector()
        self.current_alt_features = None
        
        # Calculate feature count
        self.n_asset_features = len(self.ASSET_FEATURES)
        self.n_cross_features = len(self.CROSS_ASSET_FEATURES) if include_cross_asset else 0
        self.n_alt_features = len(self.ALT_DATA_FEATURES)
        self.n_base_features = 94  # From UltimateFeatureEngine
        self.n_total_features = self.n_base_features + self.n_asset_features + self.n_cross_features + self.n_alt_features
        
        logger.info(
            f"📊 MultiAssetFeatureEngine: {self.n_total_features} features "
            f"(base={self.n_base_features}, asset={self.n_asset_features}, cross={self.n_cross_features}, alt={self.n_alt_features})"
        )
    
    def set_btc_data(self, btc_data: pd.DataFrame):
        """Set BTC data for cross-asset feature calculation."""
        self.btc_data = btc_data.copy()
        self.btc_data['btc_timestamp'] = self.btc_data['timestamp']
        logger.info(f"📊 BTC data set: {len(btc_data)} candles")
    
    def compute_asset_features(self, symbol: str) -> np.ndarray:
        """
        Compute asset embedding features.
        
        Returns:
            Array of 4 asset-specific features
        """
        if symbol not in SUPPORTED_ASSETS:
            # Default features for unknown assets
            return np.array([0.5, 1.5, 0.5, 0.7])
        
        return get_asset_embedding(symbol)
    
    def compute_cross_asset_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        idx: int,
    ) -> np.ndarray:
        """
        Compute cross-asset features (BTC influence on alts).
        
        Returns:
            Array of 6 cross-asset features
        """
        if not self.include_cross_asset or self.btc_data is None or symbol == "BTCUSDT":
            return np.zeros(len(self.CROSS_ASSET_FEATURES))
        
        try:
            # Find corresponding BTC timestamp
            current_time = df['timestamp'].iloc[idx]
            
            # Find BTC data at same timestamp
            btc_subset = self.btc_data[self.btc_data['timestamp'] <= current_time].tail(24)
            
            if len(btc_subset) < 4:
                return np.zeros(len(self.CROSS_ASSET_FEATURES))
            
            # BTC returns
            btc_close = btc_subset['close'].values
            btc_return_1h = (btc_close[-1] / btc_close[-2] - 1) if len(btc_close) >= 2 else 0
            btc_return_4h = (btc_close[-1] / btc_close[-4] - 1) if len(btc_close) >= 4 else 0
            
            # BTC momentum (EMA slope)
            if len(btc_close) >= 8:
                ema_fast = btc_close[-4:].mean()
                ema_slow = btc_close[-8:].mean()
                btc_momentum = (ema_fast / ema_slow - 1) * 10
            else:
                btc_momentum = 0
            
            # BTC volatility
            btc_returns = np.diff(btc_close) / btc_close[:-1]
            btc_volatility = np.std(btc_returns) * np.sqrt(24) if len(btc_returns) > 1 else 0
            
            # Asset relative strength
            asset_close = df['close'].values
            if idx >= 1:
                asset_return = asset_close[idx] / asset_close[idx-1] - 1
                relative_strength = asset_return - btc_return_1h
            else:
                relative_strength = 0
            
            # Correlation regime (rolling correlation with BTC)
            if idx >= 20:
                asset_returns = np.diff(asset_close[idx-20:idx+1]) / asset_close[idx-20:idx]
                btc_returns_align = np.diff(btc_close[-21:]) / btc_close[-21:-1] if len(btc_close) >= 21 else np.zeros(20)
                
                if len(asset_returns) == len(btc_returns_align) == 20:
                    correlation_regime = np.corrcoef(asset_returns, btc_returns_align)[0, 1]
                else:
                    correlation_regime = SUPPORTED_ASSETS.get(symbol, SUPPORTED_ASSETS["BTCUSDT"]).btc_correlation
            else:
                correlation_regime = SUPPORTED_ASSETS.get(symbol, SUPPORTED_ASSETS["BTCUSDT"]).btc_correlation
            
            features = np.array([
                np.clip(btc_return_1h * 100, -10, 10),  # Scale to reasonable range
                np.clip(btc_return_4h * 100, -20, 20),
                np.clip(btc_momentum, -5, 5),
                np.clip(btc_volatility, 0, 1),
                np.clip(relative_strength * 100, -10, 10),
                np.clip(correlation_regime, -1, 1),
            ])
            
            return np.nan_to_num(features, nan=0.0)
            
        except Exception as e:
            logger.warning(f"Cross-asset feature error: {e}")
            return np.zeros(len(self.CROSS_ASSET_FEATURES))
    
    def compute_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        idx: int = -1,
    ) -> np.ndarray:
        """
        Compute full feature vector for multi-asset trading.
        
        Args:
            df: OHLCV DataFrame (must have timestamp, open, high, low, close, volume)
            symbol: Trading pair (e.g., "BTCUSDT")
            idx: Row index to compute features for (-1 for last row)
            
        Returns:
            Feature vector of length n_total_features
        """
        if idx == -1:
            idx = len(df) - 1
        
        # 1. Base features (94)
    def compute_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        idx: int = -1,
    ) -> np.ndarray:
        """
        Compute full feature vector for multi-asset trading.
        """
        if idx == -1:
            idx = len(df) - 1
            
        # 1. Base features (94)
        # Optimization: UltimateFeatureEngine computes for whole DF. 
        # If we only need one row, this is expensive but necessary for indicators.
        # Ideally we cache this or use a stateful engine.
        all_base_features = self.base_engine.compute_features(df)
        base_features = all_base_features[idx]
        
        # 2. Asset embedding features (4)
        asset_features = self.compute_asset_features(symbol)
        
        # 3. Cross-asset features (6, optional)
        if self.include_cross_asset:
            cross_features = self.compute_cross_asset_features(df, symbol, idx)
        else:
            cross_features = np.array([])
            
        # 4. Alternative Data features (4)
        if self.current_alt_features is None:
            raw_alt = self.alt_collector.get_current_features()
            self.current_alt_features = np.array([
                raw_alt['fear_greed_value'],
                raw_alt['fear_greed_class'],
                raw_alt['btc_dominance'],
                raw_alt['altcoin_season_index']
            ], dtype=np.float32)
        alt_features = self.current_alt_features
        
        # Combine all features
        all_features = np.concatenate([base_features, asset_features, cross_features, alt_features])
        
        return all_features.astype(np.float32)
    
    def compute_features_batch(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_idx: int = 0,
    ) -> np.ndarray:
        """
        Compute features for all rows from start_idx to end.
        Vectorized implementation for speed.
        """
        # 1. Base features (N x 94)
        all_base_features = self.base_engine.compute_features(df)
        base_batch = all_base_features[start_idx:]
        n_rows = len(base_batch)
        
        # 2. Asset features (N x 4)
        asset_feat = self.compute_asset_features(symbol)
        asset_batch = np.tile(asset_feat, (n_rows, 1))
        
        # 3. Cross-asset features (N x 6)
        if self.include_cross_asset:
            # We still compute these in a loop for now as they depend on specific logic
            # or we could optimize later. For now, loop is okay as it's just 6 features.
            cross_batch = np.zeros((n_rows, self.n_cross_features))
            for i in range(n_rows):
                cross_batch[i] = self.compute_cross_asset_features(df, symbol, start_idx + i)
        else:
            cross_batch = np.zeros((n_rows, 0))
            
        # 4. Alternative Data features (N x 4)
        if self.current_alt_features is None:
            raw_alt = self.alt_collector.get_current_features()
            self.current_alt_features = np.array([
                raw_alt['fear_greed_value'],
                raw_alt['fear_greed_class'],
                raw_alt['btc_dominance'],
                raw_alt['altcoin_season_index']
            ], dtype=np.float32)
            
        alt_batch = np.tile(self.current_alt_features, (n_rows, 1))
            
        # Combine
        combined = np.hstack([base_batch, asset_batch, cross_batch, alt_batch])
        
        return combined.astype(np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        base_names = self.base_engine.get_feature_names()
        all_names = base_names + self.ASSET_FEATURES
        
        if self.include_cross_asset:
            all_names += self.CROSS_ASSET_FEATURES
        
        return all_names


def create_multi_asset_env_features(
    asset_data: Dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Create feature arrays for multiple assets.
    
    Args:
        asset_data: Dict mapping symbol to OHLCV DataFrame
        btc_df: BTC data for cross-asset features
        
    Returns:
        Dict mapping symbol to feature array
    """
    engine = MultiAssetFeatureEngine(include_cross_asset=True)
    engine.set_btc_data(btc_df)
    
    result = {}
    
    for symbol, df in asset_data.items():
        if df.empty:
            continue
        
        # Compute features for all rows (skip first 50 for warmup)
        start_idx = min(50, len(df) - 1)
        features = engine.compute_features_batch(df, symbol, start_idx)
        result[symbol] = features
        
        logger.info(f"📊 {symbol}: {features.shape[0]} samples, {features.shape[1]} features")
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from src.data.multi_asset_fetcher import MultiAssetDataFetcher
    
    # Fetch data
    fetcher = MultiAssetDataFetcher()
    btc_df = fetcher.fetch_asset("BTCUSDT", "1h", days=7)
    eth_df = fetcher.fetch_asset("ETHUSDT", "1h", days=7)
    
    # Create feature engine
    engine = MultiAssetFeatureEngine(include_cross_asset=True)
    engine.set_btc_data(btc_df)
    
    # Compute features
    features = engine.compute_features(eth_df, "ETHUSDT", -1)
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature names: {engine.get_feature_names()[-10:]}")  # Last 10
