#!/usr/bin/env python3
"""
Championship Model Shadow Comparator

Runs the championship ensemble model in shadow mode alongside the live HTF bot.
Does NOT trade — only logs what the championship model WOULD have done and compares
with the actual live bot decisions.

Reads the same market data as the live bot, feeds it to the championship ensemble,
and logs comparison results to MongoDB and a JSONL file.

Usage:
    python championship_shadow.py --symbol BTCUSDT --interval 15
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("championship_shadow")

# Actions
ACTION_HOLD = 0
ACTION_LONG = 1
ACTION_SHORT = 2
ACTION_NAMES = {0: "HOLD", 1: "LONG", 2: "SHORT"}


class ChampionshipShadow:
    """Shadow evaluator for the championship ensemble model."""

    def __init__(self, symbol: str = "BTCUSDT", interval: int = 15):
        self.symbol = symbol
        self.interval = interval
        self.asset = symbol.replace("USDT", "").lower()

        # Paths
        self.ensemble_path = f"data/models/championship/fold_0/phase3/final_ensemble"
        self.log_path = Path(f"logs/championship_shadow_{self.asset}.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # State
        self.ensemble = None
        self.htf_env = None
        self.vec_normalize = None
        self.live_model = None
        self.live_vec_normalize = None
        self.fetcher = None
        self.storage = None

        # Comparison stats
        self.stats = {
            "total_comparisons": 0,
            "agreements": 0,
            "disagreements": 0,
            "championship_signals": {"HOLD": 0, "LONG": 0, "SHORT": 0},
            "live_signals": {"HOLD": 0, "LONG": 0, "SHORT": 0},
        }

    def initialize(self):
        """Load models and set up data pipeline."""
        logger.info("Initializing Championship Shadow Comparator...")

        # 1. Load championship ensemble
        self._load_championship_ensemble()

        # 2. Load live model (same one the HTF bot uses)
        self._load_live_model()

        # 3. Set up data fetcher
        self._setup_data_fetcher()

        # 4. Set up MongoDB storage
        self._setup_storage()

        logger.info("Shadow comparator ready | symbol=%s", self.symbol)

    def _load_championship_ensemble(self):
        """Load the championship ensemble model."""
        from src.brain.ensemble_agent import EnsembleAgent
        from src.env.htf_env import HTFTradingEnv

        # Create a dummy env for model initialization
        # We need real data to create the env
        logger.info("Loading championship ensemble from %s", self.ensemble_path)

        try:
            # We'll defer full loading until we have data
            self._ensemble_path = self.ensemble_path
            logger.info("Championship ensemble path set (will load with first data)")
        except Exception as e:
            logger.error("Failed to set up championship ensemble: %s", e)
            raise

    def _load_live_model(self):
        """Load the same model the live HTF bot uses."""
        from live_trading_htf import find_best_htf_model
        from stable_baselines3 import PPO
        import pickle

        model_path, vecnorm_path = find_best_htf_model(self.symbol)
        if model_path is None:
            logger.error("No live model found for %s", self.symbol)
            return

        logger.info("Loading live model from %s", model_path)
        self.live_model = PPO.load(str(model_path), device="cpu")

        if vecnorm_path and vecnorm_path.exists():
            # Load VecNormalize stats directly (without venv) for obs normalization
            with open(str(vecnorm_path), "rb") as f:
                vn = pickle.load(f)
            self.live_vec_normalize = vn
            self.live_vec_normalize.training = False
            self.live_vec_normalize.norm_reward = False
            logger.info("Loaded live VecNormalize from %s", vecnorm_path)

    def _setup_data_fetcher(self):
        """Set up the multi-asset data fetcher."""
        from src.data.multi_asset_fetcher import MultiAssetDataFetcher
        self.fetcher = MultiAssetDataFetcher()

    def _setup_storage(self):
        """Set up MongoDB storage for comparison logs."""
        try:
            from src.data.storage import get_storage
            self.storage = get_storage()
            logger.info("MongoDB storage connected")
        except Exception as e:
            logger.warning("MongoDB not available, using JSONL only: %s", e)

    def _build_observation(self) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Build the same 117-dim observation vector the live bot uses."""
        from src.features.htf_features import HTFFeatureEngine, HTFDataAligner

        try:
            # Fetch 15m data
            df_15m = self.fetcher.fetch_asset(self.symbol, "15m", days=12)
            if df_15m is None or len(df_15m) < 100:
                logger.warning("Insufficient 15m data for %s", self.symbol)
                return None, None

            # Ensure datetime index
            if not isinstance(df_15m.index, pd.DatetimeIndex):
                if "open_time" in df_15m.columns:
                    df_15m = df_15m.set_index("open_time")
                elif "timestamp" in df_15m.columns:
                    df_15m = df_15m.set_index("timestamp")
                df_15m.index = pd.to_datetime(df_15m.index)

            # Use the same aligner as the live bot
            aligner = HTFDataAligner()
            frames = aligner.align_timestamps(df_15m)
            df_1d = frames["1d"]
            df_4h = frames["4h"]
            df_1h = frames["1h"]
            df_15 = frames["15m"]

            if len(df_1d) < 5 or len(df_4h) < 10 or len(df_1h) < 20 or len(df_15) < 30:
                logger.warning("Insufficient aligned bars")
                return None, None

            # Compute features exactly like the live bot
            feature_engine = HTFFeatureEngine()
            f1d = feature_engine.compute_1d_features(df_1d, len(df_1d) - 1)
            f4h = feature_engine.compute_4h_features(df_4h, len(df_4h) - 1)
            f1h = feature_engine.compute_1h_features(df_1h, len(df_1h) - 1)
            f15m = feature_engine.compute_15m_features(df_15, len(df_15) - 1)

            sig_1d = float(f1d[-1])
            sig_4h = float(f4h[-1])
            sig_1h = float(f1h[-1])
            sig_15m = float(f15m[-1])
            f_align = feature_engine.compute_alignment_full(sig_1d, sig_4h, sig_1h, sig_15m)

            feats_114 = np.concatenate([f1d, f4h, f1h, f15m, f_align])

            # Position state: shadow has no position (neutral observer)
            pos_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            obs = np.concatenate([feats_114, pos_state]).astype(np.float32)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

            return obs, df_15m

        except Exception as e:
            logger.error("Failed to build observation: %s", e)
            return None, None

    def _get_live_prediction(self, obs: np.ndarray) -> Tuple[int, float]:
        """Get prediction from the live model."""
        if self.live_model is None:
            return ACTION_HOLD, 0.0

        try:
            obs_2d = obs.reshape(1, -1)

            if self.live_vec_normalize is not None:
                try:
                    obs_2d = self.live_vec_normalize.normalize_obs(obs_2d)
                except Exception:
                    pass

            action, _ = self.live_model.predict(obs_2d, deterministic=True)
            action = int(action.item() if hasattr(action, "item") else action)

            # Confidence from policy distribution
            try:
                import torch
                with torch.no_grad():
                    obs_tensor = self.live_model.policy.obs_to_tensor(obs_2d)[0]
                    dist = self.live_model.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.detach().cpu().numpy()[0]
                confidence = float(np.max(probs))
            except Exception:
                confidence = 1.0 / 3.0

            return action, confidence

        except Exception as e:
            logger.error("Live prediction failed: %s", e)
            return ACTION_HOLD, 0.0

    def _get_championship_prediction(self, obs: np.ndarray, price_data: pd.DataFrame) -> Tuple[int, float, float]:
        """Get prediction from the championship ensemble."""
        try:
            from src.brain.ensemble_agent import EnsembleAgent
            from src.env.htf_env import HTFTradingEnv
            import gymnasium as gym

            # Lazy-load ensemble on first call
            if self.ensemble is None:
                # Create a proper HTF env with matching observation space (117-dim)
                env = HTFTradingEnv(
                    df_15m=price_data,
                    df_1h=price_data.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(),
                    df_4h=price_data.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(),
                    df_1d=price_data.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna(),
                )
                self.ensemble = EnsembleAgent(env)
                self.ensemble.load(self._ensemble_path)
                logger.info("Championship ensemble loaded successfully")

            # Get close prices for regime detection
            close_prices = price_data["close"].values[-100:].astype(np.float64)

            action, confidence, pos_size = self.ensemble.predict(
                obs, price_data=close_prices, deterministic=True
            )
            action = int(action.item() if hasattr(action, "item") else action)

            return action, confidence, pos_size

        except Exception as e:
            logger.error("Championship prediction failed: %s (type: %s)", e, type(e).__name__, exc_info=True)
            return ACTION_HOLD, 0.0, 0.0

    def _cleanup_models(self):
        """Free model memory between comparison ticks."""
        import gc
        if self.ensemble is not None:
            del self.ensemble
            self.ensemble = None
        gc.collect()

    def compare_once(self) -> Optional[Dict]:
        """Run a single comparison between live and championship models."""
        obs, df_15m = self._build_observation()
        if obs is None:
            return None

        # Get both predictions
        live_action, live_confidence = self._get_live_prediction(obs)
        champ_action, champ_confidence, champ_pos_size = self._get_championship_prediction(obs, df_15m)

        # Free championship model memory immediately
        self._cleanup_models()

        # Current price
        current_price = float(df_15m["close"].iloc[-1])

        # Build comparison record
        now = datetime.now(timezone.utc)
        record = {
            "timestamp": now.isoformat(),
            "symbol": self.symbol,
            "price": current_price,
            "live": {
                "action": ACTION_NAMES[live_action],
                "action_id": live_action,
                "confidence": round(live_confidence, 4),
            },
            "championship": {
                "action": ACTION_NAMES[champ_action],
                "action_id": champ_action,
                "confidence": round(champ_confidence, 4),
                "position_size_multiplier": round(champ_pos_size, 4),
            },
            "agreement": live_action == champ_action,
        }

        # Update stats
        self.stats["total_comparisons"] += 1
        if live_action == champ_action:
            self.stats["agreements"] += 1
        else:
            self.stats["disagreements"] += 1
        self.stats["championship_signals"][ACTION_NAMES[champ_action]] += 1
        self.stats["live_signals"][ACTION_NAMES[live_action]] += 1

        agreement_pct = (self.stats["agreements"] / self.stats["total_comparisons"] * 100
                         if self.stats["total_comparisons"] > 0 else 0)

        # Log
        agree_emoji = "✅" if record["agreement"] else "❌"
        logger.info(
            "%s %s @ $%.2f | Live: %s (%.2f) | Champ: %s (%.2f, pos=%.3f) | Agreement: %.1f%% (%d/%d)",
            agree_emoji, self.symbol, current_price,
            record["live"]["action"], live_confidence,
            record["championship"]["action"], champ_confidence, champ_pos_size,
            agreement_pct, self.stats["agreements"], self.stats["total_comparisons"],
        )

        # Save to JSONL
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Save to MongoDB
        if self.storage:
            try:
                # Copy record to avoid ObjectId mutation
                mongo_record = dict(record)
                self.storage.db["championship_shadow"].insert_one(mongo_record)
            except Exception as e:
                logger.debug("MongoDB insert failed: %s", e)

        return record

    def run_loop(self):
        """Run continuous shadow comparison loop."""
        self.initialize()

        logger.info("Starting shadow comparison loop (every %d min)", self.interval)

        while True:
            try:
                self.compare_once()
            except KeyboardInterrupt:
                logger.info("Shadow comparator stopped by user")
                break
            except Exception as e:
                logger.error("Comparison error: %s", e)

            # Wait for next interval
            # Align to candle boundaries
            now = datetime.now(timezone.utc)
            minutes_past = now.minute % self.interval
            seconds_past = now.second
            wait_seconds = (self.interval - minutes_past) * 60 - seconds_past + 5  # +5s buffer
            if wait_seconds <= 0:
                wait_seconds = self.interval * 60

            logger.info("Next comparison in %d seconds", wait_seconds)
            time.sleep(wait_seconds)

    def get_historical_comparison(self) -> Dict:
        """Compare championship model signals against historical testnet trades."""
        logger.info("Running historical comparison against testnet trades...")

        if not self.storage:
            logger.error("MongoDB required for historical comparison")
            return {}

        # Get all testnet trades for this symbol
        trades = list(self.storage.db["testnet_trades"].find(
            {"symbol": self.symbol},
            sort=[("timestamp", 1)]
        ))

        logger.info("Found %d historical testnet trades for %s", len(trades), self.symbol)

        results = {
            "symbol": self.symbol,
            "total_trades": len(trades),
            "comparisons": [],
            "summary": {},
        }

        # For each trade, we'd need the observation at that point in time
        # This requires historical feature reconstruction — complex but doable
        # For now, log the trade history and start forward comparison

        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Championship Model Shadow Comparator")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=int, default=15, help="Comparison interval in minutes")
    parser.add_argument("--historical", action="store_true", help="Run historical comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    shadow = ChampionshipShadow(symbol=args.symbol, interval=args.interval)

    if args.historical:
        shadow.initialize()
        results = shadow.get_historical_comparison()
        print(json.dumps(results, indent=2, default=str))
    else:
        shadow.run_loop()
