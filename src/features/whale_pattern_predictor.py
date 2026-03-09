"""
Whale Pattern Predictor

Real-time signal generator that uses the trained whale pattern models
to produce trading signals from recent whale wallet activity.

Designed to be called from the main trading loop via WhaleTracker.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.whale_wallet_collector import WhaleWalletCollector, WHALE_DATA_DIR
from src.features.whale_wallet_registry import get_wallets_by_chain
from src.models.whale_pattern_learner import WhalePatternLearner

logger = logging.getLogger(__name__)


class WhalePatternPredictor:
    """
    Real-time whale pattern signal generator.

    Workflow:
    1. Load trained models for each chain
    2. On each call, fetch recent wallet data (from cache)
    3. Compute flow features
    4. Run prediction
    5. Return aggregated signal
    """

    # How often to refresh wallet data from the blockchain (seconds)
    COLLECTION_INTERVAL = 3600  # 1 hour

    # Cache TTL for predictions (avoid recomputing every iteration)
    PREDICTION_CACHE_TTL = 300  # 5 minutes

    def __init__(self):
        self.learners: Dict[str, WhalePatternLearner] = {}
        self.collector = WhaleWalletCollector()
        self.last_collection_time = 0
        self.prediction_cache = {}
        self.prediction_cache_time = 0

        # Wallet accuracy weights (computed from trained models)
        self.wallet_accuracy: Dict[str, Dict[str, float]] = {}  # chain -> {wallet_idx: accuracy}

        # Asset-to-chain mapping
        self.asset_chain_map = {
            "ETHUSDT": "ETH",
            "SOLUSDT": "SOL",
            "XRPUSDT": "XRP",
            "BTCUSDT": "ETH",  # BTC uses ETH whale signals as proxy
        }

        # Load models
        self._load_models()
        # Compute accuracy weights from loaded models
        self._compute_wallet_accuracy_weights()

    def _load_models(self):
        """Load trained whale pattern models for each chain."""
        for chain in ["ETH", "SOL", "XRP"]:
            learner = WhalePatternLearner(chain)
            if learner.load_model():
                self.learners[chain] = learner
                logger.info(f"🐋 Whale pattern model loaded for {chain}")
            else:
                logger.warning(
                    f"⚠️ No whale pattern model for {chain} — "
                    f"run train_whale_patterns.py first"
                )

    def _maybe_collect(self):
        """Collect new wallet data if enough time has passed."""
        now = time.time()
        if now - self.last_collection_time < self.COLLECTION_INTERVAL:
            return

        try:
            # Only collect recent data (1 page) to minimize API usage
            for chain in ["ETH", "SOL", "XRP"]:
                wallets = get_wallets_by_chain(chain)
                for wallet in wallets:
                    try:
                        if chain == "ETH":
                            self.collector.eth_collector.collect_wallet(
                                wallet, max_pages=1
                            )
                        elif chain == "SOL":
                            self.collector.sol_collector.collect_wallet(
                                wallet, max_sigs=10
                            )
                        elif chain == "XRP":
                            self.collector.xrp_collector.collect_wallet(
                                wallet, max_pages=1
                            )
                    except Exception as e:
                        logger.warning(
                            f"Collection failed for {wallet.label}: {e}"
                        )

            self.last_collection_time = now
            logger.info("🐋 Whale wallet data refreshed")

        except Exception as e:
            logger.error(f"Whale collection error: {e}")

    def _compute_wallet_accuracy_weights(self):
        """
        Rank wallets by historical prediction accuracy using impact features
        from trained models. Wallets with >55% hit rate get boosted weight.
        """
        for chain, learner in self.learners.items():
            try:
                wallets = learner._load_wallet_data()
                if not wallets:
                    continue

                # Use impact features if model has them cached
                impact = getattr(learner, '_cached_impact_features', {})
                if not impact:
                    # Try to compute from stored data
                    price_data = learner._fetch_price_data(days=30)
                    if price_data is not None and not price_data.empty:
                        price_hourly = price_data['close'].resample('1h').last().dropna()
                        if hasattr(price_hourly.index, 'tz') and price_hourly.index.tz is not None:
                            price_hourly.index = price_hourly.index.tz_convert(None)
                        impact = learner._compute_price_impact_features(wallets, price_hourly)

                accuracy = {}
                for w_idx in range(5):
                    hit_rate_key = f"w{w_idx}_hit_rate"
                    if hit_rate_key in impact:
                        hr = impact[hit_rate_key]
                        # Scale: 0.50 (random) → weight 1.0, 0.70 → weight 2.0
                        weight = max(0.5, 1.0 + (hr - 0.50) * 5.0)
                        accuracy[f"w{w_idx}"] = min(weight, 3.0)  # Cap at 3x
                    else:
                        accuracy[f"w{w_idx}"] = 1.0  # Default weight

                self.wallet_accuracy[chain] = accuracy
                logger.info(
                    f"🎯 Wallet accuracy computed for {chain}: "
                    f"{', '.join(f'{k}={v:.2f}x' for k, v in accuracy.items())}"
                )
            except Exception as e:
                logger.warning(f"Wallet accuracy computation failed for {chain}: {e}")
                self.wallet_accuracy[chain] = {}

    def get_signal(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Get whale pattern signal for a trading symbol.

        Args:
            symbol: Trading pair (e.g., "ETHUSDT")

        Returns:
            Dict with:
            - signal: float from -1 to +1
            - confidence: float from 0 to 1
            - chain: which chain was used
            - status: ok/no_model/error
        """
        # Check prediction cache
        now = time.time()
        cache_key = symbol
        if cache_key in self.prediction_cache:
            cached_time, cached_result = self.prediction_cache[cache_key]
            if now - cached_time < self.PREDICTION_CACHE_TTL:
                return cached_result

        # Maybe collect new data
        self._maybe_collect()

        # Determine chain
        chain = self.asset_chain_map.get(symbol, "ETH")

        # Check if we have a model
        if chain not in self.learners:
            result = {
                "signal": 0.0,
                "confidence": 0.0,
                "chain": chain,
                "status": "no_model",
            }
            self.prediction_cache[cache_key] = (now, result)
            return result

        learner = self.learners[chain]

        try:
            # Load cached wallet data
            wallets = learner._load_wallet_data()
            if not wallets:
                result = {
                    "signal": 0.0,
                    "confidence": 0.0,
                    "chain": chain,
                    "status": "no_data",
                }
                self.prediction_cache[cache_key] = (now, result)
                return result

            # Convert to hourly flow features
            hourly = learner._transactions_to_hourly(wallets)
            if hourly.empty or len(hourly) < 2:
                result = {
                    "signal": 0.0,
                    "confidence": 0.0,
                    "chain": chain,
                    "status": "insufficient_data",
                }
                self.prediction_cache[cache_key] = (now, result)
                return result

            # Predict
            prediction = learner.predict(hourly)
            raw_signal = prediction.get("signal", 0.0)
            raw_confidence = prediction.get("confidence", 0.0)

            # Apply wallet accuracy weighting
            # If accurate wallets drove this signal, boost it; if random wallets, dampen it
            accuracy_weights = self.wallet_accuracy.get(chain, {})
            if accuracy_weights and raw_signal != 0:
                # Compute average accuracy multiplier across tracked wallets
                weights_list = list(accuracy_weights.values())
                avg_accuracy_mult = sum(weights_list) / len(weights_list) if weights_list else 1.0
                # Apply: accurate wallets (>1.0x) boost the signal
                adjusted_signal = float(np.clip(raw_signal * avg_accuracy_mult, -1.0, 1.0))
                adjusted_confidence = min(raw_confidence * min(avg_accuracy_mult, 1.5), 1.0)
            else:
                adjusted_signal = raw_signal
                adjusted_confidence = raw_confidence

            result = {
                "signal": adjusted_signal,
                "confidence": adjusted_confidence,
                "chain": chain,
                "status": prediction.get("status", "unknown"),
                "n_wallets": len(wallets),
                "n_hours": len(hourly),
                "raw_signal": raw_signal,
                "accuracy_multiplier": avg_accuracy_mult if accuracy_weights else 1.0,
            }

            self.prediction_cache[cache_key] = (now, result)

            logger.info(
                f"🐋 Whale pattern signal for {symbol}: "
                f"signal={result['signal']:.3f}, "
                f"confidence={result['confidence']:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Whale pattern prediction error: {e}")
            result = {
                "signal": 0.0,
                "confidence": 0.0,
                "chain": chain,
                "status": f"error: {e}",
            }
            self.prediction_cache[cache_key] = (now, result)
            return result

    def get_all_signals(self) -> Dict[str, Dict]:
        """Get whale pattern signals for all tracked assets."""
        results = {}
        for symbol in ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BTCUSDT"]:
            results[symbol] = self.get_signal(symbol)
        return results
