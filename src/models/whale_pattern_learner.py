"""
Whale Pattern Learner v2

Enhanced learning engine that analyzes per-wallet transaction impact on
price movements to generate predictive signals.

Algorithm:
1. Convert per-wallet transaction history into hourly flow time-series
2. Compute per-wallet price impact features (1h, 4h, 24h after large txns)
3. Add transaction size classification (large vs small)
4. Cross-correlate with OHLCV price data
5. Train GradientBoosting classifier to predict price direction
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

    v2 features:
    - Per-wallet flow features (not just aggregate)
    - Per-transaction price impact analysis
    - Transaction size classification (large vs small)
    - GradientBoosting with multi-horizon targets
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

    # ─────────────────────────────────────────────
    # Feature Engineering
    # ─────────────────────────────────────────────

    def _transactions_to_hourly(
        self, wallets: List[Dict]
    ) -> pd.DataFrame:
        """
        Convert all wallet transactions into an hourly flow time-series.

        Includes aggregate features + per-wallet features +
        transaction size classification.
        """
        all_txns = []
        for w_idx, w in enumerate(wallets):
            wallet_label = w.get("label", f"wallet_{w_idx}")
            for tx in w.get("transactions", []):
                ts = tx.get("timestamp", 0)
                if ts <= 0:
                    continue
                value = tx.get("value", 0)
                direction = tx.get("direction", "unknown")
                context = tx.get("context", "unknown")

                # Signed flow: positive = inflow, negative = outflow
                signed_value = value if direction == "in" else -value

                all_txns.append({
                    "timestamp": pd.Timestamp.utcfromtimestamp(ts),
                    "value": value,
                    "signed_value": signed_value,
                    "direction": direction,
                    "context": context,
                    "wallet_idx": w_idx,
                    "wallet_label": wallet_label,
                })

        if not all_txns:
            return pd.DataFrame()

        df = pd.DataFrame(all_txns)
        df = df.set_index("timestamp").sort_index()

        # ── Aggregate flow features ──
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

        # ── Time-Decay Weighted Flow (Whale Velocity) ──
        # Recent whale moves are MUCH more predictive than older ones.
        # Exponential decay: half-life = 6 hours → recent 2h have ~5x weight of 12h-old
        decay_half_life = 6  # hours
        decay_lambda = np.log(2) / decay_half_life

        if len(hourly) > 1:
            hours_ago = np.arange(len(hourly))[::-1].astype(float)
            decay_weights = np.exp(-decay_lambda * hours_ago)

            hourly["decay_weighted_flow"] = hourly["net_flow"] * decay_weights
            hourly["decay_flow_cum"] = hourly["decay_weighted_flow"].rolling(24, min_periods=1).sum()
            # Whale velocity: rate of change of time-decayed flow
            hourly["whale_velocity"] = hourly["decay_flow_cum"].diff(3).fillna(0)
        else:
            hourly["decay_weighted_flow"] = hourly["net_flow"]
            hourly["decay_flow_cum"] = hourly["net_flow"]
            hourly["whale_velocity"] = 0

        # ── Transaction size classification ──
        if len(df) > 10:
            value_75th = df["value"].quantile(0.75)
            large_mask = df["value"] >= value_75th

            hourly["large_tx_flow"] = (
                df.loc[large_mask, "signed_value"]
                .resample("1h").sum()
                .reindex(hourly.index, fill_value=0)
            )
            hourly["large_tx_count"] = (
                df.loc[large_mask, "value"]
                .resample("1h").count()
                .reindex(hourly.index, fill_value=0)
            )
            # Large tx ratio
            hourly["large_tx_ratio"] = np.where(
                hourly["tx_count"] > 0,
                hourly["large_tx_count"] / hourly["tx_count"],
                0,
            )
        else:
            hourly["large_tx_flow"] = 0
            hourly["large_tx_count"] = 0
            hourly["large_tx_ratio"] = 0

        # ── Contextual Flow Features ──
        for ctx in ["exchange", "institution", "accumulator"]:
            # Money sent TO ctx (bearish if exchange, bullish if cold storage/staking)
            mask_out = (df["direction"] == "out") & (df["context"] == ctx)
            hourly[f"to_{ctx}_flow"] = (
                df.loc[mask_out, "value"]
                .resample("1h").sum()
                .reindex(hourly.index, fill_value=0)
            )
            # Money received FROM ctx
            mask_in = (df["direction"] == "in") & (df["context"] == ctx)
            hourly[f"from_{ctx}_flow"] = (
                df.loc[mask_in, "value"]
                .resample("1h").sum()
                .reindex(hourly.index, fill_value=0)
            )

        # ── Contextual Flow Ratio Features ──
        # Provide the ML model with the direct percentage splits (Exchange Dominance vs Accumulators)
        total_ctx_flow = sum(hourly.get(f"to_{ctx}_flow", 0) for ctx in ["exchange", "institution", "accumulator"]) + 1e-9
        hourly["exchange_dump_ratio"] = hourly.get("to_exchange_flow", 0) / total_ctx_flow
        hourly["accumulator_hoard_ratio"] = hourly.get("to_accumulator_flow", 0) / total_ctx_flow

        # ── Per-wallet flow features (top 5 wallets) ──
        unique_wallets = sorted(df["wallet_idx"].unique())[:5]
        for w_idx in unique_wallets:
            w_df = df[df["wallet_idx"] == w_idx]
            prefix = f"w{w_idx}"

            hourly[f"{prefix}_flow"] = (
                w_df["signed_value"].resample("1h").sum()
                .reindex(hourly.index, fill_value=0)
            )
            hourly[f"{prefix}_count"] = (
                w_df["value"].resample("1h").count()
                .reindex(hourly.index, fill_value=0)
            )
            hourly[f"{prefix}_flow_24h"] = (
                hourly[f"{prefix}_flow"].rolling(24, min_periods=1).sum()
            )

        # ── Whale consensus ──
        # What % of wallets are flowing in the same direction?
        if len(unique_wallets) > 1:
            wallet_directions = []
            for w_idx in unique_wallets:
                col = f"w{w_idx}_flow_24h"
                if col in hourly.columns:
                    wallet_directions.append(
                        np.sign(hourly[col]).fillna(0)
                    )
            if wallet_directions:
                direction_df = pd.concat(wallet_directions, axis=1)
                # Consensus: fraction of wallets agreeing on direction
                hourly["whale_consensus"] = (
                    direction_df.apply(
                        lambda row: abs(row.sum()) / max(len(row), 1),
                        axis=1
                    )
                )
            else:
                hourly["whale_consensus"] = 0
        else:
            hourly["whale_consensus"] = 0

        # ── Time features (cyclical) ──
        hourly["hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24)
        hourly["hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24)
        hourly["dow_sin"] = np.sin(2 * np.pi * hourly.index.dayofweek / 7)
        hourly["dow_cos"] = np.cos(2 * np.pi * hourly.index.dayofweek / 7)

        return hourly

    def _compute_price_impact_features(
        self,
        wallets: List[Dict],
        price_hourly: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute per-wallet price impact statistics.

        For each wallet's large transactions, measure what the price
        did at +1h, +4h, +24h. This tells us which wallets are
        predictive and in which direction.

        Returns a dict of per-wallet impact stats to be used as
        static features during training (added to every row).
        """
        impact_features = {}

        for w_idx, w in enumerate(wallets[:5]):
            txns = w.get("transactions", [])
            if len(txns) < 5:
                continue

            # Find large transactions (top 25%)
            values = [tx.get("value", 0) for tx in txns]
            if not values:
                continue
            threshold = np.percentile(values, 75)

            impacts_1h = []
            impacts_4h = []
            impacts_24h = []
            correct_predictions = 0
            total_predictions = 0

            for tx in txns:
                value = tx.get("value", 0)
                if value < threshold:
                    continue

                ts = tx.get("timestamp", 0)
                if ts <= 0:
                    continue

                try:
                    tx_time = pd.Timestamp.utcfromtimestamp(ts)
                    if tx_time.tzinfo is not None:
                        tx_time = tx_time.tz_convert(None)
                except Exception:
                    continue

                direction = tx.get("direction", "unknown")

                # Find price at transaction time and after
                try:
                    # Find nearest price
                    idx = price_hourly.index.get_indexer(
                        [tx_time], method="nearest"
                    )[0]
                    if idx < 0 or idx >= len(price_hourly):
                        continue

                    price_at_tx = price_hourly.iloc[idx]
                    if price_at_tx <= 0:
                        continue

                    # Price change at +1h, +4h, +24h
                    for offset, impacts_list in [
                        (1, impacts_1h),
                        (4, impacts_4h),
                        (24, impacts_24h),
                    ]:
                        future_idx = idx + offset
                        if future_idx < len(price_hourly):
                            pct_change = (
                                (price_hourly.iloc[future_idx] - price_at_tx)
                                / price_at_tx
                            )
                            impacts_list.append(pct_change)

                    # Hit rate: did inflow precede price increase?
                    if idx + 4 < len(price_hourly):
                        price_4h = price_hourly.iloc[idx + 4]
                        price_went_up = price_4h > price_at_tx
                        total_predictions += 1
                        if direction == "in" and price_went_up:
                            correct_predictions += 1
                        elif direction == "out" and not price_went_up:
                            correct_predictions += 1

                except Exception:
                    continue

            prefix = f"w{w_idx}"
            impact_features[f"{prefix}_impact_1h"] = (
                float(np.mean(impacts_1h)) if impacts_1h else 0
            )
            impact_features[f"{prefix}_impact_4h"] = (
                float(np.mean(impacts_4h)) if impacts_4h else 0
            )
            impact_features[f"{prefix}_impact_24h"] = (
                float(np.mean(impacts_24h)) if impacts_24h else 0
            )
            impact_features[f"{prefix}_hit_rate"] = (
                correct_predictions / max(total_predictions, 1)
            )
            impact_features[f"{prefix}_predictive"] = (
                1.0 if impact_features[f"{prefix}_hit_rate"] > 0.55 else 0.0
            )

        return impact_features

    # ─────────────────────────────────────────────
    # Price Data
    # ─────────────────────────────────────────────

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

    # ─────────────────────────────────────────────
    # Training Data Construction
    # ─────────────────────────────────────────────

    def _build_training_data(
        self,
        hourly_flow: pd.DataFrame,
        price_df: pd.DataFrame,
        impact_features: Dict[str, float],
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
        flow_reset = flow_overlap.reset_index()
        if flow_reset.columns[0] != "timestamp":
            flow_reset = flow_reset.rename(
                columns={flow_reset.columns[0]: "timestamp"}
            )

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

        # Use per-wallet impact features to create weighted flow signals
        # Instead of static features, weight each wallet's flow by its hit rate
        if impact_features:
            weighted_flow = pd.Series(0.0, index=merged.index)
            for w_idx in range(5):
                flow_col = f"w{w_idx}_flow"
                hit_rate_key = f"w{w_idx}_hit_rate"
                if flow_col in merged.columns and hit_rate_key in impact_features:
                    hit_rate = impact_features[hit_rate_key]
                    # Weight: center around 0.5 (no-info), scale by deviation
                    weight = (hit_rate - 0.5) * 2  # range: -1 to +1
                    weighted_flow += merged[flow_col].fillna(0) * weight
            merged["weighted_smart_flow"] = weighted_flow
            merged["weighted_smart_flow_24h"] = (
                merged["weighted_smart_flow"].rolling(24, min_periods=1).sum()
            )

        # Fill NaN in flow features (sparse whale data is normal)
        flow_cols = [
            c for c in merged.columns
            if c not in ["price", "price_change_4h", "timestamp"]
        ]
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

    # ─────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────

    def train(self, days: int = 90) -> bool:
        """
        Train the whale pattern model.

        1. Load wallet transaction data
        2. Convert to hourly flow features
        3. Fetch price data
        4. Compute per-wallet price impact
        5. Build training data
        6. Train GradientBoosting + RandomForest ensemble
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

        # Step 2: Convert to hourly flow (with per-wallet features)
        hourly = self._transactions_to_hourly(wallets)
        if hourly.empty:
            logger.warning("No hourly flow data generated")
            return False

        logger.info(
            f"📊 Hourly flow: {len(hourly)} rows, "
            f"{len(hourly.columns)} features"
        )

        # Step 3: Fetch price data
        price_df = self._fetch_price_data(days=days)
        if price_df is None or price_df.empty:
            logger.warning("No price data available")
            return False

        # Step 4: Compute per-wallet price impact features
        # Need price as hourly series for impact computation
        _price = price_df.copy()
        if "timestamp" in _price.columns:
            _price = _price.set_index("timestamp")
        if not isinstance(_price.index, pd.DatetimeIndex):
            _price.index = pd.to_datetime(_price.index)
        _price = _price.sort_index()
        try:
            if hasattr(_price.index, 'tz') and _price.index.tz is not None:
                _price.index = _price.index.tz_convert(None)
        except TypeError:
            pass

        price_hourly_series = _price["close"].resample("1h").last().ffill()
        impact_features = self._compute_price_impact_features(
            wallets, price_hourly_series
        )
        if impact_features:
            logger.info(f"📊 Price impact features: {len(impact_features)}")
            for k, v in impact_features.items():
                if "hit_rate" in k:
                    logger.info(f"   {k}: {v:.2%}")

        # Step 5: Build training data
        features, target = self._build_training_data(
            hourly, price_df, impact_features
        )
        if features.empty:
            logger.warning("No valid training samples after merge")
            return False

        logger.info(
            f"📊 Training data: {len(features)} samples, "
            f"{features.shape[1]} features"
        )

        # Step 6: Train model
        try:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                RandomForestClassifier,
                VotingClassifier,
            )
            from sklearn.model_selection import cross_val_score

            self.feature_names = features.columns.tolist()

            # GradientBoosting — better at learning patterns
            gb_model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42,
            )

            # RandomForest — robust baseline
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            # Ensemble via soft voting
            model = VotingClassifier(
                estimators=[
                    ("gb", gb_model),
                    ("rf", rf_model),
                ],
                voting="soft",
                weights=[0.6, 0.4],  # GB weighted higher
            )

            # Cross-validate
            if len(features) >= 20:
                cv_folds = min(5, len(features) // 4)
                if cv_folds >= 2:
                    scores = cross_val_score(
                        model, features, target,
                        cv=cv_folds, scoring="accuracy"
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
                "n_features": features.shape[1],
                "impact_features": impact_features,
                "version": "v2",
            }

            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(
                f"✅ Whale pattern model saved: {self.model_path} "
                f"({len(features)} samples, {features.shape[1]} features)"
            )

            # Feature importance from the GB model
            try:
                gb_fitted = model.named_estimators_["gb"]
                if hasattr(gb_fitted, "feature_importances_"):
                    importances = dict(
                        zip(self.feature_names, gb_fitted.feature_importances_)
                    )
                    top_features = sorted(
                        importances.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:8]
                    logger.info("📊 Top features (GradientBoosting):")
                    for name, imp in top_features:
                        logger.info(f"   {name}: {imp:.3f}")
            except Exception:
                pass

            return True

        except ImportError:
            logger.error(
                "sklearn not installed. Run: pip install scikit-learn"
            )
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_model(self) -> bool:
        """Load a previously trained model."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    model_data = pickle.load(f)
                self.model = model_data["model"]
                self.feature_names = model_data.get("feature_names", [])
                version = model_data.get("version", "v1")
                logger.info(
                    f"✅ Loaded whale pattern model: {self.chain} "
                    f"({version}, {model_data.get('n_samples', '?')} samples, "
                    f"{model_data.get('n_features', '?')} features)"
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
            if len(available) < len(self.feature_names) * 0.3:
                return {
                    "signal": 0.0,
                    "confidence": 0.0,
                    "status": "missing_features",
                }

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
                "probabilities": dict(
                    zip([str(c) for c in classes], proba.tolist())
                ),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"signal": 0.0, "confidence": 0.0, "status": f"error: {e}"}
