"""
Order Book Imbalance Signal

Fetches order book depth from Binance and computes bid/ask imbalance
at multiple depth levels. Order book imbalance is one of the strongest
short-term directional predictors documented in academic literature
(Cont, Kukanov & Stoikov, 2014).

Signal range: [-1, +1]
  +1 = heavily bid-dominated (bullish)
  -1 = heavily ask-dominated (bearish)
   0 = balanced

Computed at 3 depth levels (5, 10, 20) for varying time horizons.
"""

import logging
import time
from typing import Dict, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────────────
_ob_cache: Dict[str, dict] = {}
_ob_cache_time: Dict[str, float] = {}
_OB_CACHE_TTL = 10  # 10 seconds — order book changes fast


def get_orderbook_imbalance(symbol: str = "BTCUSDT") -> Optional[Dict]:
    """
    Fetch order book depth from Binance and compute bid/ask imbalance.
    
    Uses https://data-api.binance.vision/api/v3/depth (free, 2400 req/min).
    
    Args:
        symbol: Trading pair (e.g. BTCUSDT)
        
    Returns:
        dict with:
            - imbalance_5:  imbalance at top 5 levels  [-1, +1]
            - imbalance_10: imbalance at top 10 levels [-1, +1]
            - imbalance_20: imbalance at top 20 levels [-1, +1]
            - bid_volume_5/10/20: total bid volume at each depth
            - ask_volume_5/10/20: total ask volume at each depth
            - ratio_5/10/20: bid_volume / ask_volume
            - bias: "bullish" / "bearish" / "neutral"
            - score: composite imbalance score [-1, +1] (weighted avg of all levels)
        Returns None on error.
    """
    global _ob_cache, _ob_cache_time
    
    clean_symbol = symbol.replace('/', '').upper()
    
    # Check cache
    if clean_symbol in _ob_cache:
        if time.time() - _ob_cache_time.get(clean_symbol, 0) < _OB_CACHE_TTL:
            return _ob_cache[clean_symbol]
    
    try:
        url = "https://data-api.binance.vision/api/v3/depth"
        response = requests.get(
            url,
            params={"symbol": clean_symbol, "limit": 20},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        
        bids = data.get('bids', [])  # [[price, qty], ...]
        asks = data.get('asks', [])  # [[price, qty], ...]
        
        if not bids or not asks:
            logger.warning(f"Empty order book for {clean_symbol}")
            return None
        
        # Convert to numpy for fast computation
        bid_qtys = np.array([float(b[1]) for b in bids])
        ask_qtys = np.array([float(a[1]) for a in asks])
        
        result = {}
        
        # Compute imbalance at 3 depth levels
        for level in [5, 10, 20]:
            bid_vol = float(bid_qtys[:level].sum()) if len(bid_qtys) >= level else float(bid_qtys.sum())
            ask_vol = float(ask_qtys[:level].sum()) if len(ask_qtys) >= level else float(ask_qtys.sum())
            
            total = bid_vol + ask_vol
            if total > 0:
                imbalance = (bid_vol - ask_vol) / total  # Range: [-1, +1]
            else:
                imbalance = 0.0
            
            ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
            
            result[f'imbalance_{level}'] = round(float(np.clip(imbalance, -1, 1)), 4)
            result[f'bid_volume_{level}'] = round(bid_vol, 4)
            result[f'ask_volume_{level}'] = round(ask_vol, 4)
            result[f'ratio_{level}'] = round(ratio, 4)
        
        # Composite score: weighted average across levels
        # Deeper levels get less weight (more noise)
        composite = (
            0.50 * result['imbalance_5'] +
            0.30 * result['imbalance_10'] +
            0.20 * result['imbalance_20']
        )
        composite = float(np.clip(composite, -1, 1))
        result['score'] = round(composite, 4)
        
        # Determine bias
        if composite > 0.10:
            result['bias'] = 'bullish'
        elif composite < -0.10:
            result['bias'] = 'bearish'
        else:
            result['bias'] = 'neutral'
        
        # Cache result
        _ob_cache[clean_symbol] = result
        _ob_cache_time[clean_symbol] = time.time()
        
        logger.info(
            f"📖 OrderBook [{clean_symbol}]: "
            f"imb5={result['imbalance_5']:+.3f} "
            f"imb10={result['imbalance_10']:+.3f} "
            f"imb20={result['imbalance_20']:+.3f} "
            f"→ {result['bias']} (score={composite:+.3f})"
        )
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Order book fetch error for {clean_symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Order book imbalance error for {clean_symbol}: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = get_orderbook_imbalance("BTCUSDT")
    if result:
        print("\n📖 Order Book Imbalance:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("Failed to fetch order book data")
