"""
Alternative Data Collector

Fetches and caches external market signals that aren't available
in standard OHLCV price/volume data.

Signals collected:
1. Fear & Greed Index (alternative.me API)
2. BTC Dominance (CoinGecko API)

All APIs used are free and don't require API keys. Responses are
cached to disk to avoid rate limits during training loops.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AlternativeDataCollector:
    """Collects and caches alternative market data."""
    
    CACHE_DIR = Path('./data/alternative_cache')
    CACHE_TTL_HOURS = 12  # How long to consider cache valid for live trading
    
    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.fear_greed_cache_file = self.CACHE_DIR / 'fear_greed.json'
        self.btc_dom_cache_file = self.CACHE_DIR / 'btc_dominance.json'
        
        # Headers to prevent 403 Forbidden from strict APIs
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        }
    
    def get_current_features(self) -> Dict[str, float]:
        """
        Get latest alternative features for live trading.
        Returns a dictionary of normalized features ready for the RL agent.
        """
        features = {
            'fear_greed_value': 0.0,
            'fear_greed_class': 0.0,
            'btc_dominance': 0.0,
            'altcoin_season_index': 0.0,
        }
        
        try:
            # 1. Fear & Greed (Normalized -1 to +1)
            fng = self.fetch_fear_greed(limit=1)
            if fng and len(fng) > 0:
                val = float(fng[0]['value'])
                # 0 = Extreme Fear, 100 = Extreme Greed -> map to -1 to +1
                features['fear_greed_value'] = (val - 50) / 50.0
                
                # Classes: 0: Ext Fear, 1: Fear, 2: Neutral, 3: Greed, 4: Ext Greed
                # Map to -1.0 to 1.0
                class_str = fng[0]['value_classification']
                class_map = {
                    'Extreme Fear': -1.0,
                    'Fear': -0.5,
                    'Neutral': 0.0,
                    'Greed': 0.5,
                    'Extreme Greed': 1.0
                }
                features['fear_greed_class'] = class_map.get(class_str, 0.0)
                
            # 2. BTC Dominance (Requires CoinGecko global data)
            dom_data = self.fetch_btc_dominance()
            if dom_data:
                # Usually between 35% and 65% -> normalize
                btc_d = float(dom_data.get('btc_dominance', 50.0))
                features['btc_dominance'] = np.clip((btc_d - 50) / 20.0, -1, 1)
                
                # "Altcoin season" proxy: When ETH dominance is high relative to BTC
                eth_d = float(dom_data.get('eth_dominance', 15.0))
                if btc_d > 0:
                    ratio = eth_d / btc_d  # e.g. 15/50 = 0.3
                    features['altcoin_season_index'] = np.clip((ratio - 0.3) * 5, -1, 1)
                    
        except Exception as e:
            logger.error(f"Error getting alt data features: {e}")
            
        return features

    def fetch_fear_greed(self, limit: int = 30) -> Optional[list]:
        """Fetch Crypto Fear & Greed Index."""
        # Check cache first if we only need 1 (live trading)
        if limit == 1 and self._is_cache_valid(self.fear_greed_cache_file):
            return self._load_cache(self.fear_greed_cache_file)
            
        url = f"https://api.alternative.me/fng/?limit={limit}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data:
                    # Save to cache if just getting the latest
                    if limit == 1:
                        self._save_cache(self.fear_greed_cache_file, data['data'])
                    return data['data']
        except Exception as e:
            logger.warning(f"Fear & Greed API failed: {e}")
            
        # Fallback to cache if API fails
        if self.fear_greed_cache_file.exists():
            return self._load_cache(self.fear_greed_cache_file)
            
        return None

    def fetch_btc_dominance(self) -> Optional[Dict]:
        """Fetch BTC and ETH dominance from CoinGecko global metrics."""
        if self._is_cache_valid(self.btc_dom_cache_file):
            return self._load_cache(self.btc_dom_cache_file)
            
        url = "https://api.coingecko.com/api/v3/global"
        try:
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data and 'market_cap_percentage' in data['data']:
                    result = {
                        'btc_dominance': data['data']['market_cap_percentage'].get('btc', 50),
                        'eth_dominance': data['data']['market_cap_percentage'].get('eth', 15),
                        'timestamp': time.time()
                    }
                    self._save_cache(self.btc_dom_cache_file, result)
                    return result
            elif resp.status_code == 429:
                logger.warning("CoinGecko API rate limited")
        except Exception as e:
            logger.warning(f"CoinGecko global API failed: {e}")
            
        # Fallback to cache
        if self.btc_dom_cache_file.exists():
            return self._load_cache(self.btc_dom_cache_file)
            
        return {'btc_dominance': 50.0, 'eth_dominance': 15.0}

    # ─── Cache Helpers ────────────────────────────────────────────────────────
    
    def _is_cache_valid(self, filepath: Path) -> bool:
        if not filepath.exists():
            return False
        
        mtime = filepath.stat().st_mtime
        valid_time = time.time() - (self.CACHE_TTL_HOURS * 3600)
        return mtime > valid_time
        
    def _save_cache(self, filepath: Path, data: Any):
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save cache to {filepath}: {e}")
            
    def _load_cache(self, filepath: Path) -> Any:
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = AlternativeDataCollector()
    features = collector.get_current_features()
    print("\nCurrent Alternative Features:")
    for k, v in features.items():
        print(f"  {k}: {v:.3f}")
