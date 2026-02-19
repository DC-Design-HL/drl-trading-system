"""
Hidden Markov Model Regime Classifier

Unsupervised regime detection using HMM:
- State 0: BULL_TREND  — Strong upward momentum
- State 1: BEAR_TREND  — Strong downward momentum
- State 2: RANGE_CHOP  — Sideways/choppy market
- State 3: HIGH_VOL_BREAKOUT — High volatility transition

Key innovation: the HMM's transition matrix tells us the PROBABILITY
of switching regimes BEFORE it happens, giving a predictive edge.

Usage:
    python -m src.models.regime_classifier --asset BTCUSDT --days 730
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Regime labels
REGIME_NAMES = {
    0: 'BULL_TREND',
    1: 'BEAR_TREND',
    2: 'RANGE_CHOP',
    3: 'HIGH_VOL_BREAKOUT',
}


class RegimeClassifier:
    """
    HMM-based market regime classifier.

    Features used for regime detection:
    - Returns (1h, 4h, 24h)
    - Realized volatility (24h, 168h/1w)
    - Volume change ratio
    - RSI deviation from 50
    - ADX-proxy (directional strength)
    """

    MODEL_DIR = './data/models/regime'

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.feature_scaler = None  # Store mean/std for normalization

    def _compute_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute features optimized for regime detection.

        Returns (n_samples, 8) array.
        """
        close = df['close'].astype(float)
        volume = df['volume'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)

        features = pd.DataFrame(index=df.index)

        # 1. Returns at multiple scales
        features['ret_1h'] = close.pct_change(1).fillna(0)
        features['ret_4h'] = close.pct_change(4).fillna(0)
        features['ret_24h'] = close.pct_change(24).fillna(0)

        # 2. Realized volatility
        features['vol_24h'] = close.pct_change().rolling(24).std().fillna(0)
        features['vol_168h'] = close.pct_change().rolling(168).std().fillna(0)

        # 3. Volume change
        vol_ma = volume.rolling(24).mean()
        features['vol_ratio'] = (volume / (vol_ma + 1e-10) - 1).fillna(0)

        # 4. RSI deviation from 50
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features['rsi_dev'] = ((rsi - 50) / 50).fillna(0)

        # 5. Directional strength (simplified ADX)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        features['dir_strength'] = (close.diff(14).abs() / (atr * 14 + 1e-10)).fillna(0)

        result = features.values.astype(np.float64)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip extreme values
        result = np.clip(result, -5, 5)

        return result

    def fit(self, df: pd.DataFrame) -> 'RegimeClassifier':
        """
        Fit HMM on historical data.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
        """
        from hmmlearn.hmm import GaussianHMM

        features = self._compute_regime_features(df)

        # Skip initial NaN rows
        valid_start = 170  # After all rolling windows are filled
        features = features[valid_start:]

        # Normalize features
        self.feature_scaler = {
            'mean': features.mean(axis=0),
            'std': features.std(axis=0) + 1e-10,
        }
        features_norm = (features - self.feature_scaler['mean']) / self.feature_scaler['std']

        # Fit HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        self.model.fit(features_norm)

        # Decode states and label them
        states = self.model.predict(features_norm)
        self._label_regimes(df.iloc[valid_start:], states)

        logger.info(
            f"✅ HMM fitted with {self.n_regimes} regimes, "
            f"score={self.model.score(features_norm):.2f}"
        )

        return self

    def _label_regimes(self, df: pd.DataFrame, states: np.ndarray):
        """
        Auto-label HMM states based on their characteristics.

        Assigns labels (BULL, BEAR, RANGE, BREAKOUT) to state indices
        based on average return and volatility in each state.
        """
        close = df['close'].values
        returns = np.diff(close) / close[:-1]
        returns = np.append(returns, 0)  # Pad to match length

        vol = pd.Series(returns).rolling(24).std().fillna(0).values

        state_stats = {}
        for s in range(self.n_regimes):
            mask = states == s
            if mask.sum() > 0:
                state_stats[s] = {
                    'mean_return': np.mean(returns[mask]),
                    'mean_vol': np.mean(vol[mask]),
                    'count': int(mask.sum()),
                    'pct': mask.sum() / len(states) * 100,
                }

        # Sort states by return to assign labels
        sorted_states = sorted(state_stats.keys(), key=lambda s: state_stats[s]['mean_return'])

        self.state_labels = {}
        if len(sorted_states) >= 4:
            # Most bearish → BEAR, most bullish → BULL
            self.state_labels[sorted_states[0]] = 'BEAR_TREND'
            self.state_labels[sorted_states[-1]] = 'BULL_TREND'

            # Of remaining 2, higher vol → BREAKOUT, lower → RANGE
            remaining = sorted_states[1:-1]
            if state_stats[remaining[0]]['mean_vol'] > state_stats[remaining[1]]['mean_vol']:
                self.state_labels[remaining[0]] = 'HIGH_VOL_BREAKOUT'
                self.state_labels[remaining[1]] = 'RANGE_CHOP'
            else:
                self.state_labels[remaining[0]] = 'RANGE_CHOP'
                self.state_labels[remaining[1]] = 'HIGH_VOL_BREAKOUT'
        else:
            for i, s in enumerate(sorted_states):
                self.state_labels[s] = list(REGIME_NAMES.values())[i]

        # Log regime distribution
        for s, label in self.state_labels.items():
            stats = state_stats.get(s, {})
            logger.info(
                f"  Regime {s} ({label}): "
                f"return={stats.get('mean_return', 0)*100:.3f}%/h, "
                f"vol={stats.get('mean_vol', 0)*100:.3f}%, "
                f"freq={stats.get('pct', 0):.1f}%"
            )

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict current regime and transition probabilities.

        Returns:
            dict with:
                - current_regime: str (e.g. 'BULL_TREND')
                - regime_id: int
                - transition_probs: dict mapping regime_name -> probability
                - regime_history: list of last 24 regime states
        """
        if self.model is None:
            return self._empty_prediction()

        features = self._compute_regime_features(df)

        # Use last 200 bars for prediction
        features = features[-200:]

        # Normalize
        features_norm = (features - self.feature_scaler['mean']) / self.feature_scaler['std']

        # Predict
        states = self.model.predict(features_norm)
        current_state = states[-1]

        # Get transition probabilities from current state
        trans_probs = self.model.transmat_[current_state]

        # Label transitions
        transition_dict = {}
        for s, prob in enumerate(trans_probs):
            label = self.state_labels.get(s, f'STATE_{s}')
            transition_dict[label] = round(float(prob), 4)

        return {
            'current_regime': self.state_labels.get(current_state, 'UNKNOWN'),
            'regime_id': int(current_state),
            'transition_probs': transition_dict,
            'regime_history': [self.state_labels.get(s, 'UNKNOWN') for s in states[-24:]],
            'confidence': float(max(trans_probs)),
        }

    def get_regime_filtered_indices(
        self,
        df: pd.DataFrame,
        regime: str,
    ) -> np.ndarray:
        """
        Get indices of bars belonging to a specific regime.

        Used by train_specialist.py to filter training data per regime.

        Args:
            df: OHLCV DataFrame
            regime: One of 'BULL_TREND', 'BEAR_TREND', 'RANGE_CHOP', 'HIGH_VOL_BREAKOUT'

        Returns:
            Array of integer indices where the regime is active
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        features = self._compute_regime_features(df)
        valid_start = 170
        features = features[valid_start:]
        features_norm = (features - self.feature_scaler['mean']) / self.feature_scaler['std']
        states = self.model.predict(features_norm)

        # Find state ID for this regime label
        target_state = None
        for s, label in self.state_labels.items():
            if label == regime:
                target_state = s
                break

        if target_state is None:
            logger.warning(f"Regime '{regime}' not found. Available: {list(self.state_labels.values())}")
            return np.arange(len(df))

        # Get indices (offset by valid_start)
        regime_mask = states == target_state
        indices = np.where(regime_mask)[0] + valid_start

        logger.info(f"Regime '{regime}': {len(indices)} bars ({len(indices)/len(df)*100:.1f}% of data)")
        return indices

    def save(self, symbol: str = 'BTCUSDT'):
        """Save model to disk."""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        path = os.path.join(self.MODEL_DIR, f'regime_{symbol.lower()}.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.feature_scaler,
                'labels': self.state_labels,
                'n_regimes': self.n_regimes,
            }, f)
        logger.info(f"💾 Regime classifier saved: {path}")

    def load(self, symbol: str = 'BTCUSDT') -> bool:
        """Load model from disk."""
        path = os.path.join(self.MODEL_DIR, f'regime_{symbol.lower()}.pkl')
        if not os.path.exists(path):
            logger.warning(f"No regime model at {path}")
            return False

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.feature_scaler = data['scaler']
        self.state_labels = data['labels']
        self.n_regimes = data['n_regimes']
        logger.info(f"✅ Regime classifier loaded: {path}")
        return True

    @staticmethod
    def _empty_prediction() -> Dict:
        return {
            'current_regime': 'UNKNOWN',
            'regime_id': -1,
            'transition_probs': {},
            'regime_history': [],
            'confidence': 0.0,
        }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Train Regime Classifier')
    parser.add_argument('--asset', type=str, default='BTCUSDT')
    parser.add_argument('--days', type=int, default=730)
    args = parser.parse_args()

    from src.backtest.data_loader import download_binance_data

    symbol = args.asset.replace('USDT', '/USDT')
    df = download_binance_data(symbol=symbol, timeframe='1h', days=args.days)

    classifier = RegimeClassifier(n_regimes=4)
    classifier.fit(df)
    classifier.save(args.asset)

    # Test prediction
    result = classifier.predict(df)
    print(f"\nCurrent regime: {result['current_regime']}")
    print(f"Transition probabilities: {result['transition_probs']}")
