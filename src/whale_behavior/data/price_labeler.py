"""
Price Outcome Labeler

For each whale wallet action, labels what ETH price did in the following
1h, 4h, 12h, and 24h windows. This creates the training labels for the
behavioral prediction model.

Uses Binance OHLCV data for price history.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

PRICE_CACHE_DIR = Path("data/whale_behavior/price_cache")
LABELED_DIR = Path("data/whale_behavior/labeled")

# Label windows (in hours)
LABEL_WINDOWS = [1, 4, 12, 24]

# Price change thresholds for classification
# > threshold = BUY_SIGNAL, < -threshold = SELL_SIGNAL, else = NEUTRAL
SIGNAL_THRESHOLD = 0.015  # 1.5%


class PriceLabeler:
    """
    Labels whale actions with subsequent price outcomes.

    For each action timestamp, computes:
    - price_change_1h, 4h, 12h, 24h (percentage)
    - label_1h, 4h, 12h, 24h (BUY_SIGNAL / SELL_SIGNAL / NEUTRAL)
    """

    BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

    def __init__(self):
        PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        LABELED_DIR.mkdir(parents=True, exist_ok=True)
        self._price_cache: Optional[pd.DataFrame] = None

    def _fetch_eth_hourly(
        self, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        """Fetch ETH/USDT hourly candles from Binance."""
        all_candles = []
        current_start = start_ms

        while current_start < end_ms:
            params = {
                "symbol": "ETHUSDT",
                "interval": "1h",
                "startTime": str(current_start),
                "endTime": str(end_ms),
                "limit": "1000",
            }
            try:
                resp = requests.get(self.BINANCE_KLINES_URL, params=params, timeout=15)
                data = resp.json()
                if not isinstance(data, list) or len(data) == 0:
                    break

                all_candles.extend(data)
                # Move start to after last candle
                current_start = int(data[-1][0]) + 3600001
                time.sleep(0.2)  # Rate limit
            except Exception as exc:
                logger.error("Binance klines error: %s", exc)
                break

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = df["open_time"].astype(int) // 1000
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        return df

    def load_or_fetch_prices(
        self, min_timestamp: int, max_timestamp: int
    ) -> pd.DataFrame:
        """Load cached prices or fetch from Binance."""
        cache_file = PRICE_CACHE_DIR / "eth_hourly.parquet"

        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            cached_min = df["timestamp"].min()
            cached_max = df["timestamp"].max()

            # Check if we need to extend
            need_earlier = min_timestamp < cached_min - 3600
            need_later = max_timestamp > cached_max + 3600

            if need_earlier or need_later:
                logger.info("Extending price cache...")
                new_parts = [df]
                if need_earlier:
                    earlier = self._fetch_eth_hourly(
                        min_timestamp * 1000 - 86400000,
                        cached_min * 1000,
                    )
                    if not earlier.empty:
                        new_parts.insert(0, earlier)
                if need_later:
                    later = self._fetch_eth_hourly(
                        cached_max * 1000,
                        (max_timestamp + 86400) * 1000,
                    )
                    if not later.empty:
                        new_parts.append(later)
                df = pd.concat(new_parts).drop_duplicates("timestamp").sort_values("timestamp")
                df.to_parquet(cache_file)
        else:
            logger.info("Fetching ETH hourly prices from Binance...")
            df = self._fetch_eth_hourly(
                (min_timestamp - 86400) * 1000,
                (max_timestamp + 86400 * 2) * 1000,
            )
            if not df.empty:
                df.to_parquet(cache_file)

        self._price_cache = df
        return df

    def get_price_at(self, timestamp: int) -> Optional[float]:
        """Get ETH price closest to a timestamp."""
        if self._price_cache is None or self._price_cache.empty:
            return None
        idx = (self._price_cache["timestamp"] - timestamp).abs().idxmin()
        return float(self._price_cache.loc[idx, "close"])

    def get_price_change(
        self, timestamp: int, hours_ahead: int
    ) -> Optional[float]:
        """Get percentage price change from timestamp to timestamp + hours."""
        if self._price_cache is None or self._price_cache.empty:
            return None

        target_time = timestamp + hours_ahead * 3600

        # Find closest price at both timestamps
        df = self._price_cache
        start_idx = (df["timestamp"] - timestamp).abs().idxmin()
        end_idx = (df["timestamp"] - target_time).abs().idxmin()

        start_price = float(df.loc[start_idx, "close"])
        end_price = float(df.loc[end_idx, "close"])

        if start_price <= 0:
            return None

        return (end_price - start_price) / start_price

    def label_timeline(self, actions: List[Dict]) -> List[Dict]:
        """
        Add price outcome labels to a wallet's action timeline.

        For each action, adds:
        - price_at_action: ETH price when the action occurred
        - price_change_Xh: percentage change X hours later
        - label_Xh: BUY_SIGNAL / SELL_SIGNAL / NEUTRAL
        """
        if not actions:
            return []

        # Ensure prices are loaded
        min_ts = min(a["timestamp"] for a in actions)
        max_ts = max(a["timestamp"] for a in actions)
        self.load_or_fetch_prices(min_ts, max_ts)

        labeled = []
        for action in actions:
            ts = action["timestamp"]
            record = dict(action)  # Copy

            # Price at action time
            record["price_at_action"] = self.get_price_at(ts)

            # Price changes and labels for each window
            for hours in LABEL_WINDOWS:
                change = self.get_price_change(ts, hours)
                record[f"price_change_{hours}h"] = change

                if change is not None:
                    if change > SIGNAL_THRESHOLD:
                        record[f"label_{hours}h"] = "BUY_SIGNAL"
                    elif change < -SIGNAL_THRESHOLD:
                        record[f"label_{hours}h"] = "SELL_SIGNAL"
                    else:
                        record[f"label_{hours}h"] = "NEUTRAL"
                else:
                    record[f"label_{hours}h"] = None

            labeled.append(record)

        return labeled

    def label_wallet(self, wallet_label: str) -> List[Dict]:
        """
        Load a wallet's timeline, add price labels, and save.
        """
        from .eth_collector import EthWhaleHistoryCollector

        actions = EthWhaleHistoryCollector.load_wallet_timeline(wallet_label)
        if not actions:
            logger.warning("No data for wallet: %s", wallet_label)
            return []

        logger.info("Labeling %d actions for %s...", len(actions), wallet_label)
        labeled = self.label_timeline(actions)

        # Save labeled data
        safe_label = wallet_label.lower().replace(" ", "_").replace("/", "_")
        out_file = LABELED_DIR / f"{safe_label}_labeled.jsonl"
        with open(out_file, "w") as f:
            for record in labeled:
                f.write(json.dumps(record, default=str) + "\n")

        # Stats
        total = len(labeled)
        for hours in LABEL_WINDOWS:
            key = f"label_{hours}h"
            buys = sum(1 for r in labeled if r.get(key) == "BUY_SIGNAL")
            sells = sum(1 for r in labeled if r.get(key) == "SELL_SIGNAL")
            neutral = sum(1 for r in labeled if r.get(key) == "NEUTRAL")
            logger.info(
                "  %s %dh: BUY=%d (%.1f%%) SELL=%d (%.1f%%) NEUTRAL=%d (%.1f%%)",
                wallet_label, hours,
                buys, buys/total*100 if total else 0,
                sells, sells/total*100 if total else 0,
                neutral, neutral/total*100 if total else 0,
            )

        return labeled

    def label_all_wallets(self) -> Dict[str, int]:
        """Label all collected wallets. Returns {label: count}."""
        from .eth_collector import EthWhaleHistoryCollector

        results = {}
        for label, count in EthWhaleHistoryCollector.list_collected_wallets():
            if count == 0:
                continue
            try:
                labeled = self.label_wallet(label)
                results[label] = len(labeled)
            except Exception as exc:
                logger.error("Failed to label %s: %s", label, exc)
                results[label] = -1

        return results
