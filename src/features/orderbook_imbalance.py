"""
Order Book Imbalance Signal

Fetches order book depth from Binance FUTURES and computes bid/ask imbalance
at multiple depth levels. Order book imbalance is one of the strongest
short-term directional predictors documented in academic literature
(Cont, Kukanov & Stoikov, 2014).

Signal range: [-1, +1]
  +1 = heavily bid-dominated (bullish)
  -1 = heavily ask-dominated (bearish)
   0 = balanced

Computed at 3 depth levels (5, 10, 20) for varying time horizons.

v2 improvements (2026-03-31):
  - Uses FUTURES orderbook (fapi.binance.com) instead of SPOT — matches our trading venue
  - Averages 3 snapshots to reduce noise/spoofing
  - Bias threshold raised to 0.25 on imbalance_10 (backtested: catches 3/3 whipsaw, net +4)
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

# ── Config ────────────────────────────────────────────────────────────────────
_OB_SAMPLES = 3          # Number of snapshots to average
_OB_SAMPLE_DELAY = 0.3   # Seconds between samples
_OB_BIAS_THRESHOLD = 0.25  # imbalance_10 threshold for bullish/bearish bias

# Futures API endpoint (primary) and spot fallback
_FUTURES_URL = "https://fapi.binance.com/fapi/v1/depth"
_SPOT_URL = "https://data-api.binance.vision/api/v3/depth"


def _fetch_single_snapshot(url: str, symbol: str) -> Optional[Dict]:
    """Fetch a single orderbook snapshot and compute imbalances."""
    try:
        response = requests.get(
            url,
            params={"symbol": symbol, "limit": 20},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        bids = data.get('bids', [])
        asks = data.get('asks', [])

        if not bids or not asks:
            return None

        bid_qtys = np.array([float(b[1]) for b in bids])
        ask_qtys = np.array([float(a[1]) for a in asks])

        result = {}
        for level in [5, 10, 20]:
            bid_vol = float(bid_qtys[:level].sum()) if len(bid_qtys) >= level else float(bid_qtys.sum())
            ask_vol = float(ask_qtys[:level].sum()) if len(ask_qtys) >= level else float(ask_qtys.sum())

            total = bid_vol + ask_vol
            imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
            ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0

            result[f'imbalance_{level}'] = float(np.clip(imbalance, -1, 1))
            result[f'bid_volume_{level}'] = bid_vol
            result[f'ask_volume_{level}'] = ask_vol
            result[f'ratio_{level}'] = ratio

        return result
    except Exception:
        return None


def get_orderbook_imbalance(symbol: str = "BTCUSDT") -> Optional[Dict]:
    """
    Fetch order book depth from Binance FUTURES and compute bid/ask imbalance.

    Takes multiple snapshots and averages them to reduce noise from spoofing
    and momentary order placement/cancellation.

    Args:
        symbol: Trading pair (e.g. BTCUSDT)

    Returns:
        dict with:
            - imbalance_5/10/20: averaged imbalance at each depth [-1, +1]
            - bid_volume_5/10/20: total bid volume at each depth (last snapshot)
            - ask_volume_5/10/20: total ask volume at each depth (last snapshot)
            - ratio_5/10/20: bid_volume / ask_volume (last snapshot)
            - bias: "bullish" / "bearish" / "neutral" (based on imbalance_10 threshold)
            - score: composite imbalance score [-1, +1] (weighted avg of all levels)
            - samples: number of successful snapshots averaged
            - source: "futures" or "spot"
        Returns None on error.
    """
    global _ob_cache, _ob_cache_time

    clean_symbol = symbol.replace('/', '').upper()

    # Check cache
    if clean_symbol in _ob_cache:
        if time.time() - _ob_cache_time.get(clean_symbol, 0) < _OB_CACHE_TTL:
            return _ob_cache[clean_symbol]

    # Collect multiple snapshots
    snapshots = []
    source = "futures"

    for i in range(_OB_SAMPLES):
        snap = _fetch_single_snapshot(_FUTURES_URL, clean_symbol)
        if snap is None and i == 0:
            # Futures failed on first try — fall back to spot
            snap = _fetch_single_snapshot(_SPOT_URL, clean_symbol)
            if snap is not None:
                source = "spot"
        if snap is not None:
            snapshots.append(snap)
        if i < _OB_SAMPLES - 1 and _OB_SAMPLE_DELAY > 0:
            time.sleep(_OB_SAMPLE_DELAY)

    if not snapshots:
        logger.warning(f"No orderbook snapshots obtained for {clean_symbol}")
        return None

    # Average imbalances across snapshots
    result = {}
    for level in [5, 10, 20]:
        avg_imb = np.mean([s[f'imbalance_{level}'] for s in snapshots])
        result[f'imbalance_{level}'] = round(float(np.clip(avg_imb, -1, 1)), 4)
        # Use last snapshot for volume/ratio (most recent)
        last = snapshots[-1]
        result[f'bid_volume_{level}'] = round(last[f'bid_volume_{level}'], 4)
        result[f'ask_volume_{level}'] = round(last[f'ask_volume_{level}'], 4)
        result[f'ratio_{level}'] = round(last[f'ratio_{level}'], 4)

    # Composite score: weighted average across levels
    composite = (
        0.50 * result['imbalance_5'] +
        0.30 * result['imbalance_10'] +
        0.20 * result['imbalance_20']
    )
    composite = float(np.clip(composite, -1, 1))
    result['score'] = round(composite, 4)

    # Determine bias using imbalance_10 with tuned threshold
    # Backtested Mar 28-31: threshold 0.25 catches 3/3 whipsaw, net +4 losers blocked
    imb10 = result['imbalance_10']
    if imb10 > _OB_BIAS_THRESHOLD:
        result['bias'] = 'bullish'
    elif imb10 < -_OB_BIAS_THRESHOLD:
        result['bias'] = 'bearish'
    else:
        result['bias'] = 'neutral'

    result['samples'] = len(snapshots)
    result['source'] = source

    # Cache result
    _ob_cache[clean_symbol] = result
    _ob_cache_time[clean_symbol] = time.time()

    logger.info(
        f"📖 OrderBook [{clean_symbol}] ({source}, {len(snapshots)} samples): "
        f"imb5={result['imbalance_5']:+.3f} "
        f"imb10={result['imbalance_10']:+.3f} "
        f"imb20={result['imbalance_20']:+.3f} "
        f"→ {result['bias']} (score={composite:+.3f})"
    )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
        result = get_orderbook_imbalance(sym)
        if result:
            print(f"\n📖 {sym} ({result['source']}, {result['samples']} samples):")
            print(f"  imb10: {result['imbalance_10']:+.4f} | bias: {result['bias']} | score: {result['score']:+.4f}")
        else:
            print(f"\n❌ {sym}: Failed")
