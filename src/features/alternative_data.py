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
        self.stablecoin_cache_file = self.CACHE_DIR / 'stablecoin_supply.json'
        
        # Headers to prevent 403 Forbidden from strict APIs
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        }
        
        # Stablecoin cache (1 hour — supply changes slowly)
        self._stablecoin_cache = None
        self._stablecoin_cache_time = 0
        self._STABLECOIN_CACHE_TTL = 3600  # 1 hour
    
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

    def fetch_stablecoin_supply(self) -> Optional[Dict]:
        """
        Fetch total stablecoin supply (USDT + USDC) from DeFiLlama.
        
        API: https://stablecoins.llama.fi/stablecoins (free, no key required)
        
        Computes:
          - total_supply: combined USDT + USDC circulating supply (USD)
          - change_7d_pct: 7-day percentage change in supply
          - signal: "bullish" if supply growing >1%/week,
                    "bearish" if shrinking <-0.5%/week,
                    "neutral" otherwise
        
        Returns None on error.
        """
        # Check in-memory cache (1 hour TTL)
        now = time.time()
        if self._stablecoin_cache and (now - self._stablecoin_cache_time) < self._STABLECOIN_CACHE_TTL:
            return self._stablecoin_cache
        
        # Check disk cache 
        if self._is_cache_valid(self.stablecoin_cache_file):
            cached = self._load_cache(self.stablecoin_cache_file)
            if cached:
                self._stablecoin_cache = cached
                self._stablecoin_cache_time = now
                return cached
        
        try:
            url = "https://stablecoins.llama.fi/stablecoins"
            resp = requests.get(url, headers=self.headers, timeout=15)
            
            if resp.status_code != 200:
                logger.warning(f"DeFiLlama stablecoins API returned {resp.status_code}")
                return self._stablecoin_fallback()
            
            data = resp.json()
            peg_assets = data.get('peggedAssets', [])
            
            if not peg_assets:
                logger.warning("DeFiLlama returned empty peggedAssets")
                return self._stablecoin_fallback()
            
            # Find USDT and USDC
            target_symbols = {'USDT', 'USDC'}
            total_supply = 0.0
            total_supply_7d_ago = 0.0
            found = []
            
            for asset in peg_assets:
                symbol = asset.get('symbol', '').upper()
                if symbol in target_symbols:
                    # Current circulating supply
                    chains_circ = asset.get('chainCirculating', {})
                    current = 0.0
                    for chain_data in chains_circ.values():
                        current += chain_data.get('current', {}).get('peggedUSD', 0) or 0
                    
                    if current == 0:
                        # Fallback to top-level circulating
                        current = asset.get('circulating', {}).get('peggedUSD', 0) or 0
                    
                    total_supply += current
                    found.append(symbol)
                    
                    # 7d ago supply for change calculation
                    supply_7d_ago = current  # default: no change
                    chains_circ_prev = asset.get('chainCirculating', {})
                    prev_7d = 0.0
                    for chain_data in chains_circ_prev.values():
                        prev_7d += chain_data.get('circulatingPrevDay', {}).get('peggedUSD', 0) or 0
                    
                    # DeFiLlama's circulatingPrevDay is ~1 day ago, not 7 days
                    # Use it as a proxy — actual 7d would need historical endpoint
                    # We'll approximate: if we can get circulatingPrevDay, extrapolate
                    if prev_7d > 0:
                        daily_change = current - prev_7d
                        supply_7d_ago = current - (daily_change * 7)  # rough 7d extrapolation
                    
                    total_supply_7d_ago += supply_7d_ago
            
            if total_supply == 0:
                logger.warning(f"Could not find USDT/USDC in DeFiLlama data (found: {found})")
                return self._stablecoin_fallback()
            
            # Compute 7-day change
            if total_supply_7d_ago > 0:
                change_7d_pct = ((total_supply - total_supply_7d_ago) / total_supply_7d_ago) * 100
            else:
                change_7d_pct = 0.0
            
            # Determine signal
            if change_7d_pct > 1.0:
                signal = 'bullish'  # Supply growing >1%/week = new buying power
            elif change_7d_pct < -0.5:
                signal = 'bearish'  # Supply shrinking = capital leaving
            else:
                signal = 'neutral'
            
            result = {
                'total_supply': round(total_supply, 0),
                'change_7d_pct': round(change_7d_pct, 3),
                'signal': signal,
                'tokens_found': found,
                'timestamp': time.time(),
            }
            
            # Cache to memory and disk
            self._stablecoin_cache = result
            self._stablecoin_cache_time = now
            self._save_cache(self.stablecoin_cache_file, result)
            
            logger.info(
                f"💲 Stablecoin Supply: ${total_supply/1e9:.1f}B "
                f"(7d Δ={change_7d_pct:+.2f}%) → {signal}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Stablecoin supply fetch error: {e}")
            return self._stablecoin_fallback()
    
    def _stablecoin_fallback(self) -> Optional[Dict]:
        """Return cached stablecoin data if available."""
        if self.stablecoin_cache_file.exists():
            cached = self._load_cache(self.stablecoin_cache_file)
            if cached:
                self._stablecoin_cache = cached
                self._stablecoin_cache_time = time.time()
                return cached
        return None

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
