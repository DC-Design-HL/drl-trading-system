"""
Whale Pattern Learner

Learns buy/sell patterns from whale wallet transaction history
and correlates them with price movements to generate predictive signals.

Algorithm:
1. Convert transaction history into hourly flow time-series
2. Compute rolling features (24h net flow, 7d accumulation, flow acceleration)
3. Cross-correlate with OHLCV price data
4. Train a lightweight model (sklearn) to predict price direction
"""

import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models/whale_patterns")
WHALE_DATA_DIR = Path("data/whale_wallets")


class WhalePatternLearner:
    """
    Learns whale trading patterns and their correlation with price.

    Features extracted per hourly bucket:
    - net_flow: net inflow/outflow in native units
    - tx_count: number of transactions
    - avg_size: average transaction size
    - direction_ratio: ratio of in vs out transactions
    - flow_24h: rolling 24h net flow
    - flow_7d: rolling 7d net flow
    - flow_acceleration: rate of change of flow
    - hour_of_day: cyclical time feature
    - day_of_week: cyclical time feature
    """

    def __init__(self, chain: str):
        self.chain = chain.upper()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.model_path = MODEL_DIR / f"{self.chain.lower()}_whale_model.pkl"
        self.feature_names = []

    def _load_wallet_data(self) -> List[Dict]:
        """Load all cached wallet data for this chain."""
        chain_dir = WHALE_DATA_DIR / self.chain.lower()
        wallets = []
        if chain_dir.exists():
            for filepath in chain_dir.glob("*.json"):
                try:
                    with open(filepath, "r") as f:
                        wallets.append(json.load(f))
                except (json.JSONDecodeError, IOError):
                    pass
        return wallets

    def _transactions_to_hourly(
        self, wallets: List[Dict]
    ) -> pd.DataFrame:
        """
        Convert all wallet transactions into an hourly flow time-series.

        Each row = 1 hour, columns = aggregate flow metrics.
        """
        all_txns = []
        for w in wallets:
            for tx in w.get("transactions", []):
                ts = tx.get("timestamp", 0)
                if ts <= 0:
                    continue
                value = tx.get("value", 0)
                direction = tx.get("direction", "unknown")

                # Signed flow: positive = inflow, negative = outflow
                signed_value = value if direction == "in" else -value

                all_txns.append(
                    {
                        "timestamp": pd.Timestamp.utcfromtimestamp(ts),
                        "value": value,
                        "signed_value": signed_value,
                        "direction": direction,
                    }
                )

        if not all_txns:
            return pd.DataFrame()

        df = pd.DataFrame(all_txns)
        df = df.set_index("timestamp").sort_index()

        # Resample to hourly buckets
        hourly = pd.DataFrame()
        hourly["net_flow"] = df["signed_value"].resample("1h").sum().fillna(0)
        hourly["tx_count"] = df["signed_value"].resample("1h").count().fillna(0)
        hourly["avg_size"] = df["value"].resample("1h").mean().fillna(0)
        hourly["inflow_count"] = (
            df[df["direction"] == "in"]["value"]
            .resample("1h").count()
            .reindex(hourly.index, fill_value=0)
        )
        hourly["outflow_count"] = (
            df[df["direction"] == "out"]["value"]
            .resample("1h").count()
            .reindex(hourly.index, fill_value=0)
        )

        # Direction ratio: >0.5 = more inflows, <0.5 = more outflows
        total_count = hourly["inflow_count"] + hourly["outflow_count"]
        hourly["direction_ratio"] = np.where(
            total_count > 0,
            hourly["inflow_count"] / total_count,
            0.5,
        )

        # Rolling features
        hourly["flow_24h"] = hourly["net_flow"].rolling(24, min_periods=1).sum()
        hourly["flow_7d"] = hourly["net_flow"].rolling(168, min_periods=1).sum()
        hourly["flow_acceleration"] = hourly["flow_24h"].diff(6).fillna(0)

        # Time features (cyclical encoding)
        hourly["hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24)
        hourly["hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24)
        hourly["dow_sin"] = np.sin(2 * np.pi * hourly.index.dayofweek / 7)
        hourly["dow_cos"] = np.cos(2 * np.pi * hourly.index.dayofweek / 7)

        return hourly

    def _fetch_price_data(self, days: int = 90) -> Optional[pd.DataFrame]:
        """Fetch OHLCV price data for the corresponding asset."""
        try:
            from src.data.multi_asset_fetcher import MultiAssetDataFetcher

            fetcher = MultiAssetDataFetcher()

            # Map chain to trading symbol
            symbol_map = {
                "ETH": "ETHUSDT",
                "SOL": "SOLUSDT",
                "XRP": "XRPUSDT",
            }
            symbol = symbol_map.get(self.chain, "BTCUSDT")
            df = fetcher.fetch_asset(symbol, "1h", days=days)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
        return None

    def _build_training_data(
        self, hourly_flow: pd.DataFrame, price_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Merge whale flow features with price data and create targets.

        Target: price direction in next 4 hours (+1 = up, -1 = down)
        """
        # Ensure price_df has a proper datetime index
        if "timestamp" in price_df.columns:
            price_df = price_df.set_index("timestamp")
        elif not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)

        price_df = price_df.sort_index()

        # Strip timezone info from both to ensure matching
        try:
            if hasattr(price_df.index, 'tz') and price_df.index.tz is not None:
                price_df.index = price_df.index.tz_convert(None)
        except TypeError:
            price_df.index = price_df.index.tz_localize(None)
        try:
            if hasattr(hourly_flow.index, 'tz') and hourly_flow.index.tz is not None:
                hourly_flow.index = hourly_flow.index.tz_convert(None)
        except TypeError:
            hourly_flow.index = hourly_flow.index.tz_localize(None)

        # Debug: show date ranges
        logger.info(
            f"📊 Whale flow range: {hourly_flow.index.min()} → {hourly_flow.index.max()}"
        )
        logger.info(
            f"📊 Price data range: {price_df.index.min()} → {price_df.index.max()}"
        )

        # Resample price to hourly close
        price_hourly = price_df["close"].resample("1h").last().ffill()
        logger.info(f"📊 Price hourly: {len(price_hourly)} rows")

        # Create target: price change 4h ahead
        price_change_4h = price_hourly.pct_change(4).shift(-4)

        # Filter whale flow to only the period covered by price data
        overlap_start = max(hourly_flow.index.min(), price_hourly.index.min())
        overlap_end = min(hourly_flow.index.max(), price_hourly.index.max())

        logger.info(f"📊 Overlap period: {overlap_start} → {overlap_end}")

        if overlap_start >= overlap_end:
            logger.warning("No overlapping time period between whale flow and price data")
            return pd.DataFrame(), pd.Series()

        # Filter to overlap period
        flow_overlap = hourly_flow.loc[overlap_start:overlap_end].copy()
        logger.info(f"📊 Flow rows in overlap: {len(flow_overlap)}")

        if flow_overlap.empty:
            return pd.DataFrame(), pd.Series()

        # Use merge_asof for robust join (nearest hour match)
        flow_reset = flow_overlap.reset_index().rename(columns={"index": "timestamp"})
        if flow_reset.columns[0] != "timestamp":
            # The index name might differ
            flow_reset = flow_overlap.reset_index()
            flow_reset.columns = ["timestamp"] + list(flow_overlap.columns)

        price_reset = pd.DataFrame({
            "timestamp": price_hourly.index,
            "price": price_hourly.values,
            "price_change_4h": price_change_4h.values,
        })

        merged = pd.merge_asof(
            flow_reset.sort_values("timestamp"),
            price_reset.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )

        # Fill NaN in flow features (sparse whale data is normal)
        flow_cols = [c for c in merged.columns if c not in ["price", "price_change_4h", "timestamp"]]
        merged[flow_cols] = merged[flow_cols].fillna(0)
        # Drop only rows missing price/target data
        merged = merged.dropna(subset=["price", "price_change_4h"])
        logger.info(f"📊 Merged rows after dropna: {len(merged)}")

        if merged.empty:
            return pd.DataFrame(), pd.Series()

        # Target: +1 if price goes up, -1 if down
        target = np.sign(merged["price_change_4h"])

        # Features (exclude target, price, and timestamp)
        feature_cols = [
            c
            for c in merged.columns
            if c not in ["price", "price_change_4h", "timestamp"]
        ]
        features = merged[feature_cols]

        return features, target

    def train(self, days: int = 90) -> bool:
        """
        Train the whale pattern model.

        1. Load wallet transaction data
        2. Convert to hourly flow features
        3. Fetch price data
        4. Build training data
        5. Train a Random Forest
        """
        logger.info(f"🧠 Training whale pattern model for {self.chain}...")

        # Step 1: Load wallet data
        wallets = self._load_wallet_data()
        if not wallets:
            logger.warning(f"No wallet data for {self.chain}")
            return False

        total_txns = sum(len(w.get("transactions", [])) for w in wallets)
        logger.info(f"📊 Loaded {len(wallets)} wallets, {total_txns} total txns")

        if total_txns < 50:
            logger.warning(
                f"Not enough transactions ({total_txns}) for training. "
                f"Need at least 50."
            )
            return False

        # Step 2: Convert to hourly flow
        hourly = self._transactions_to_hourly(wallets)
        if hourly.empty:
            logger.warning("No hourly flow data generated")
            return False

        logger.info(f"📊 Hourly flow: {len(hourly)} rows, {hourly.columns.tolist()}")

        # Step 3: Fetch price data
        price_df = self._fetch_price_data(days=days)
        if price_df is None or price_df.empty:
            logger.warning("No price data available")
            return False

        # Step 4: Build training data
        features, target = self._build_training_data(hourly, price_df)
        if features.empty:
            logger.warning("No valid training samples after merge")
            return False

        logger.info(f"📊 Training data: {len(features)} samples, {features.shape[1]} features")

        # Step 5: Train model
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score

            self.feature_names = features.columns.tolist()

            # Handle class imbalance
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            # Cross-validate
            if len(features) >= 20:
                cv_folds = min(5, len(features) // 4)
                if cv_folds >= 2:
                    scores = cross_val_score(
                        model, features, target, cv=cv_folds, scoring="accuracy"
                    )
                    logger.info(
                        f"📊 CV Accuracy: {scores.mean():.3f} "
                        f"(+/- {scores.std():.3f})"
                    )

            # Train on full data
            model.fit(features, target)
            self.model = model

            # Save model
            model_data = {
                "model": model,
                "feature_names": self.feature_names,
                "chain": self.chain,
                "trained_at": datetime.utcnow().isoformat(),
                "n_samples": len(features),
                "n_wallets": len(wallets),
                "n_transactions": total_txns,
            }

            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(
                f"✅ Whale pattern model saved: {self.model_path} "
                f"({len(features)} samples)"
            )

            # Feature importance
            if hasattr(model, "feature_importances_"):
                importances = dict(
                    zip(self.feature_names, model.feature_importances_)
                )
                top_features = sorted(
                    importances.items(), key=lambda x: x[1], reverse=True
                )[:5]
                logger.info("📊 Top features:")
                for name, imp in top_features:
                    logger.info(f"   {name}: {imp:.3f}")

            return True

        except ImportError:
            logger.error(
                "sklearn not installed. Run: pip install scikit-learn"
            )
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def load_model(self) -> bool:
        """Load a previously trained model."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    model_data = pickle.load(f)
                self.model = model_data["model"]
                self.feature_names = model_data.get("feature_names", [])
                logger.info(
                    f"✅ Loaded whale pattern model: {self.chain} "
                    f"(trained on {model_data.get('n_samples', '?')} samples)"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        return False

    def predict(self, hourly_flow: pd.DataFrame) -> Dict:
        """
        Predict price direction from current whale flow features.

        Args:
            hourly_flow: DataFrame with whale flow features (last few rows)

        Returns:
            Dict with signal (-1 to +1) and confidence
        """
        if self.model is None:
            return {"signal": 0.0, "confidence": 0.0, "status": "no_model"}

        try:
            # Use only the feature columns the model was trained on
            available = [c for c in self.feature_names if c in hourly_flow.columns]
            if len(available) < len(self.feature_names) * 0.5:
                return {"signal": 0.0, "confidence": 0.0, "status": "missing_features"}

            # Fill missing features with 0
            features = pd.DataFrame()
            for col in self.feature_names:
                if col in hourly_flow.columns:
                    features[col] = hourly_flow[col]
                else:
                    features[col] = 0.0

            # Use the last row (most recent)
            latest = features.iloc[[-1]]

            # Predict with probabilities
            proba = self.model.predict_proba(latest)[0]
            classes = self.model.classes_

            # Convert to signal: weighted sum of class probabilities
            signal = 0.0
            for cls, prob in zip(classes, proba):
                signal += cls * prob

            # Confidence = max probability
            confidence = float(max(proba))

            return {
                "signal": np.clip(signal, -1.0, 1.0),
                "confidence": confidence,
                "status": "ok",
                "probabilities": dict(zip([str(c) for c in classes], proba.tolist())),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"signal": 0.0, "confidence": 0.0, "status": f"error: {e}"}
