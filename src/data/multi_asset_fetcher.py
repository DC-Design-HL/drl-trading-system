"""
Multi-Asset Data Fetcher

Fetches historical and real-time data for multiple crypto assets.
Supports: BTC, ETH, SOL, XRP (and more)

Features:
- Parallel data fetching for efficiency
- Asset-specific metadata (volatility characteristics, typical spread)
- Unified DataFrame format across all assets
"""

import os
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class AssetConfig:
    """Configuration for a tradeable asset."""
    symbol: str              # e.g., "BTCUSDT"
    name: str               # e.g., "Bitcoin"
    asset_id: int           # Unique ID for embedding (0=BTC, 1=ETH, etc.)
    base_volatility: float  # Historical volatility multiplier relative to BTC
    liquidity_score: float  # 1.0 = highest (BTC), lower = less liquid
    btc_correlation: float  # Typical correlation with BTC
    
    def to_features(self) -> np.ndarray:
        """Convert to feature vector for embedding."""
        return np.array([
            self.asset_id / 10.0,  # Normalized asset ID
            self.base_volatility,
            self.liquidity_score,
            self.btc_correlation,
        ])


# Predefined asset configurations
SUPPORTED_ASSETS: Dict[str, AssetConfig] = {
    "BTCUSDT": AssetConfig(
        symbol="BTCUSDT",
        name="Bitcoin",
        asset_id=0,
        base_volatility=1.0,
        liquidity_score=1.0,
        btc_correlation=1.0,
    ),
    "ETHUSDT": AssetConfig(
        symbol="ETHUSDT", 
        name="Ethereum",
        asset_id=1,
        base_volatility=1.15,  # ~15% more volatile than BTC
        liquidity_score=0.9,
        btc_correlation=0.85,
    ),
    "SOLUSDT": AssetConfig(
        symbol="SOLUSDT",
        name="Solana",
        asset_id=2,
        base_volatility=1.8,   # Much more volatile
        liquidity_score=0.6,
        btc_correlation=0.75,
    ),
    "XRPUSDT": AssetConfig(
        symbol="XRPUSDT",
        name="XRP",
        asset_id=3,
        base_volatility=1.5,
        liquidity_score=0.7,
        btc_correlation=0.60,
    ),
    "BNBUSDT": AssetConfig(
        symbol="BNBUSDT",
        name="BNB",
        asset_id=4,
        base_volatility=1.2,
        liquidity_score=0.8,
        btc_correlation=0.80,
    ),
    "DOGEUSDT": AssetConfig(
        symbol="DOGEUSDT",
        name="Dogecoin",
        asset_id=5,
        base_volatility=2.5,   # Very volatile, meme-driven
        liquidity_score=0.5,
        btc_correlation=0.50,
    ),
}


class MultiAssetDataFetcher:
    """
    Fetches data for multiple crypto assets in parallel.
    
    Usage:
        fetcher = MultiAssetDataFetcher()
        
        # Fetch single asset
        df = fetcher.fetch_asset("BTCUSDT", "1h", days=30)
        
        # Fetch multiple assets
        data = fetcher.fetch_multiple(["BTCUSDT", "ETHUSDT"], "1h", days=30)
    """
    
    def __init__(
        self,
        base_url: str = None,
        max_workers: int = 4,
    ):
        self.base_url = base_url or os.environ.get("BINANCE_FUTURES_URL", "https://data-api.binance.vision")
        self.max_workers = max_workers
        
        logger.info(f"📊 MultiAssetDataFetcher initialized for {len(SUPPORTED_ASSETS)} assets")
    
    def get_asset_config(self, symbol: str) -> AssetConfig:
        """Get configuration for an asset."""
        if symbol not in SUPPORTED_ASSETS:
            raise ValueError(f"Unsupported asset: {symbol}. Supported: {list(SUPPORTED_ASSETS.keys())}")
        return SUPPORTED_ASSETS[symbol]
    
    def fetch_asset(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 30,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a single asset.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            days: Number of days to fetch
            limit: Max candles per request (Binance max: 1000)
            
        Returns:
            DataFrame with OHLCV data + asset metadata
        """
        all_data = []
        
        # Calculate timestamps
        end_time = int(time.time() * 1000)
        
        # Map interval to milliseconds
        interval_ms = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }.get(interval, 60 * 60 * 1000)
        
        candles_needed = int(days * 24 * 60 * 60 * 1000 / interval_ms)
        
        # Fetch data in chunks
        current_end = end_time
        remaining = candles_needed
        
        while remaining > 0:
            chunk_size = min(remaining, limit)
            
            try:
                url = f"{self.base_url}/api/v3/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": chunk_size,
                    "endTime": current_end,
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Move to earlier time
                current_end = int(data[0][0]) - 1
                remaining -= len(data)
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        
        # Add asset metadata
        config = self.get_asset_config(symbol)
        df['symbol'] = symbol
        df['asset_id'] = config.asset_id
        df['base_volatility'] = config.base_volatility
        df['liquidity_score'] = config.liquidity_score
        df['btc_correlation'] = config.btc_correlation
        
        # Keep essential columns
        df = df[[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume',
            'symbol', 'asset_id', 'base_volatility', 'liquidity_score', 'btc_correlation'
        ]]
        
        logger.info(f"📊 Fetched {len(df)} candles for {symbol} ({interval})")
        
        return df
    
    def fetch_multiple(
        self,
        symbols: List[str],
        interval: str = "1h",
        days: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple assets in parallel.
        
        Args:
            symbols: List of trading pairs
            interval: Candle interval
            days: Number of days
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_asset, symbol, interval, days): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    results[symbol] = df
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
                    results[symbol] = pd.DataFrame()
        
        return results
    
    def fetch_all_supported(
        self,
        interval: str = "1h",
        days: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for all supported assets."""
        return self.fetch_multiple(list(SUPPORTED_ASSETS.keys()), interval, days)
    
    def create_combined_dataset(
        self,
        symbols: List[str],
        interval: str = "1h",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Create a combined dataset with all assets for multi-asset training.
        
        Returns:
            Single DataFrame with all assets, marked by symbol/asset_id
        """
        data = self.fetch_multiple(symbols, interval, days)
        
        dfs = []
        for symbol, df in data.items():
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"📊 Combined dataset: {len(combined)} rows across {len(dfs)} assets")
        
        return combined


# Convenience functions
def get_asset_embedding(symbol: str) -> np.ndarray:
    """Get the feature embedding for an asset."""
    if symbol not in SUPPORTED_ASSETS:
        raise ValueError(f"Unknown asset: {symbol}")
    return SUPPORTED_ASSETS[symbol].to_features()


def get_all_supported_symbols() -> List[str]:
    """Get list of all supported trading symbols."""
    return list(SUPPORTED_ASSETS.keys())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    fetcher = MultiAssetDataFetcher()
    
    # Test single asset
    df = fetcher.fetch_asset("BTCUSDT", "1h", days=7)
    print(f"BTC: {len(df)} candles")
    
    # Test multiple assets
    data = fetcher.fetch_multiple(["BTCUSDT", "ETHUSDT", "SOLUSDT"], "1h", days=7)
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} candles")
    
    # Test combined dataset
    combined = fetcher.create_combined_dataset(["BTCUSDT", "ETHUSDT"], "1h", days=7)
    print(f"Combined: {len(combined)} total rows")
