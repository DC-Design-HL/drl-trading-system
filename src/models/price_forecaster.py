"""
Temporal Fusion Transformer (TFT) for Multi-Horizon Price Forecasting

A simplified but effective TFT implementation that predicts:
- Price returns at 1h, 4h, 12h, 24h horizons
- Confidence intervals (10th, 50th, 90th percentile via quantile loss)
- Direction consensus across horizons

Architecture:
1. Variable Selection Network (VSN) — learns which features matter
2. LSTM Encoder — captures temporal patterns
3. Multi-Head Attention — focuses on relevant past timesteps
4. Gated Residual Networks (GRN) — controls information flow
5. Quantile Output — predicts distribution, not just point estimate

References:
- Lim et al., 2021. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
import json

logger = logging.getLogger(__name__)


# ─── Building Blocks ──────────────────────────────────────────────────────────

class GatedLinearUnit(nn.Module):
    """GLU activation — controls information flow."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """
    GRN — the core building block of TFT.
    Applies non-linear processing with skip connection and gating.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, context_dim: int = None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional context vector (e.g., static features)
        self.context_fc = nn.Linear(context_dim, hidden_dim) if context_dim else None

        # Skip connection projection (if dims differ)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, context=None):
        residual = self.skip_proj(x) if self.skip_proj else x

        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.dropout(self.fc2(hidden))
        hidden = self.glu(hidden)

        return self.layer_norm(residual + hidden)


class VariableSelectionNetwork(nn.Module):
    """
    VSN — learns which input features are relevant.
    Outputs per-feature importance weights (softmax) and transformed features.
    """
    def __init__(self, n_features: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Per-feature GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
            for _ in range(n_features)
        ])

        # Feature importance weights
        self.importance_grn = GatedResidualNetwork(
            n_features * hidden_dim, hidden_dim, n_features, dropout
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            selected: (batch, seq_len, hidden_dim) - weighted feature combination
            weights: (batch, seq_len, n_features) - feature importance
        """
        batch, seq_len, _ = x.shape

        # Process each feature independently
        processed = []
        for i in range(self.n_features):
            feat = x[:, :, i:i+1]  # (batch, seq_len, 1)
            processed.append(self.feature_grns[i](feat))  # (batch, seq_len, hidden_dim)

        # Stack and flatten for importance computation
        stacked = torch.stack(processed, dim=2)  # (batch, seq_len, n_features, hidden_dim)
        flat = stacked.reshape(batch, seq_len, -1)  # (batch, seq_len, n_features * hidden_dim)

        # Compute importance weights
        weights = self.softmax(self.importance_grn(flat))  # (batch, seq_len, n_features)

        # Weighted combination
        selected = (stacked * weights.unsqueeze(-1)).sum(dim=2)  # (batch, seq_len, hidden_dim)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable per-head weights.
    Unlike standard MHA, this produces a single attention pattern per head for visualization.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch, seq_len, _ = query.shape

        Q = self.q_proj(query).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.out_proj(attn_output), attn_weights


# ─── Main TFT Model ──────────────────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for multi-horizon crypto price forecasting.

    Input: (batch, lookback, n_features) — past OHLCV + technical features
    Output: (batch, n_horizons, n_quantiles) — quantile predictions at each horizon
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        horizons: List[int] = None,
        quantiles: List[float] = None,
    ):
        super().__init__()

        self.horizons = horizons or [1, 4, 12, 24]  # 1h, 4h, 12h, 24h ahead
        self.quantiles = quantiles or [0.1, 0.5, 0.9]  # 10th, 50th, 90th percentile
        self.n_horizons = len(self.horizons)
        self.n_quantiles = len(self.quantiles)
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        # 1. Variable Selection
        self.vsn = VariableSelectionNetwork(n_features, hidden_dim, dropout)

        # 2. LSTM Encoder (bidirectional for richer representations)
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # 3. Gated skip connection over LSTM
        self.lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # 4. Multi-head self-attention
        self.attention = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout)
        self.attn_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 5. Position-wise feedforward
        self.ff_grn = GatedResidualNetwork(hidden_dim, hidden_dim * 2, hidden_dim, dropout)

        # 6. Quantile output heads — one per horizon
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.n_quantiles),
            )
            for _ in range(self.n_horizons)
        ])

        logger.info(
            f"🔮 TFT initialized: {n_features} features → {hidden_dim}d, "
            f"horizons={self.horizons}, quantiles={self.quantiles}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            predictions: (batch, n_horizons, n_quantiles) — quantile forecasts
            intermediates: dict with attention weights, feature importances for interpretability
        """
        # 1. Variable selection — learn which features matter
        selected, feature_weights = self.vsn(x)  # (batch, seq_len, hidden_dim)

        # 2. LSTM encoding
        lstm_out, _ = self.lstm_encoder(selected)  # (batch, seq_len, hidden_dim)
        lstm_out = self.lstm_norm(selected + self.lstm_gate(lstm_out))  # Gated skip

        # 3. Self-attention over temporal dimension
        # Create causal mask (can't look into the future)
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1) == 0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out, causal_mask)
        attn_out = self.attn_norm(lstm_out + self.attn_gate(attn_out))

        # 4. Feedforward
        ff_out = self.ff_grn(attn_out)  # (batch, seq_len, hidden_dim)

        # 5. Use the LAST hidden state for predictions
        last_hidden = ff_out[:, -1, :]  # (batch, hidden_dim)

        # 6. Multi-horizon quantile predictions
        predictions = []
        for head in self.output_heads:
            pred = head(last_hidden)  # (batch, n_quantiles)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)  # (batch, n_horizons, n_quantiles)

        intermediates = {
            'feature_weights': feature_weights,  # (batch, seq_len, n_features)
            'attention_weights': attn_weights,     # (batch, n_heads, seq_len, seq_len)
        }

        return predictions, intermediates

    def predict(self, x: torch.Tensor) -> Dict:
        """
        Convenience method for inference.

        Returns dict with predictions, confidence, and direction consensus.
        """
        self.eval()
        with torch.no_grad():
            preds, intermediates = self(x)

        # Extract median predictions (50th percentile = index 1)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        medians = preds[:, :, median_idx].cpu().numpy()  # (batch, n_horizons)

        # Confidence = inverse of prediction interval width
        if len(self.quantiles) >= 2:
            low_idx = 0   # 10th percentile
            high_idx = -1  # 90th percentile
            intervals = (preds[:, :, high_idx] - preds[:, :, low_idx]).cpu().numpy()
            confidences = 1.0 / (1.0 + np.abs(intervals))
        else:
            confidences = np.ones_like(medians)

        # Direction consensus: do all horizons agree on direction?
        directions = np.sign(medians)  # -1, 0, or 1 for each horizon
        consensus = np.mean(directions, axis=1)  # -1 = all bearish, +1 = all bullish

        # Feature importance (average across batch and time)
        feat_weights = intermediates['feature_weights'].mean(dim=(0, 1)).cpu().numpy()

        return {
            'predictions': medians,          # (batch, n_horizons)
            'confidences': confidences,        # (batch, n_horizons)
            'direction_consensus': consensus,  # (batch,)
            'feature_importance': feat_weights,  # (n_features,)
            'quantiles': preds.cpu().numpy(),    # (batch, n_horizons, n_quantiles)
        }


# ─── Quantile Loss ───────────────────────────────────────────────────────────

class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic forecasting.
    Penalizes under-predictions for high quantiles and over-predictions for low quantiles.
    """
    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, n_horizons, n_quantiles)
            targets: (batch, n_horizons)
        """
        targets = targets.unsqueeze(-1)  # (batch, n_horizons, 1)
        errors = targets - predictions   # (batch, n_horizons, n_quantiles)

        losses = []
        for i, q in enumerate(self.quantiles):
            e = errors[:, :, i]
            loss = torch.max(q * e, (q - 1) * e)
            losses.append(loss)

        return torch.stack(losses, dim=-1).mean()


# ─── Feature Preparation ─────────────────────────────────────────────────────

class TFTFeaturePreprocessor:
    """
    Prepare features for TFT from raw OHLCV data.

    Creates a standardized feature set optimized for time-series forecasting:
    - Price returns (multiple periods)
    - Volume changes
    - Technical indicators (RSI, MACD, BB)
    - Temporal features (hour, day_of_week)
    - Volatility measures
    """

    FEATURE_NAMES = [
        # Returns
        'return_1', 'return_2', 'return_4', 'return_8', 'return_12', 'return_24',
        # Price ratios
        'hl_range', 'close_position', 'body_ratio',
        # Volume
        'volume_change', 'volume_ma_ratio',
        # Moving averages
        'ma_7_ratio', 'ma_14_ratio', 'ma_50_ratio',
        # RSI
        'rsi_14',
        # MACD
        'macd', 'macd_signal', 'macd_hist',
        # Bollinger
        'bb_position', 'bb_width',
        # Volatility
        'atr_14', 'realized_vol_24',
        # Temporal
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]

    N_FEATURES = len(FEATURE_NAMES)

    @staticmethod
    def prepare_features(df: pd.DataFrame) -> np.ndarray:
        """
        Convert OHLCV DataFrame to TFT feature array.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Must have a DatetimeIndex or 'timestamp' column

        Returns:
            features: (n_samples, N_FEATURES) numpy array
        """
        features = pd.DataFrame(index=df.index)

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        opn = df['open'].astype(float)
        volume = df['volume'].astype(float)

        # ── Returns ──
        for period in [1, 2, 4, 8, 12, 24]:
            features[f'return_{period}'] = close.pct_change(period).fillna(0).clip(-0.2, 0.2)

        # ── Price ratios ──
        hl_range = high - low
        features['hl_range'] = (hl_range / close).fillna(0).clip(0, 0.2)
        features['close_position'] = ((close - low) / (hl_range + 1e-10)).fillna(0.5).clip(0, 1)
        features['body_ratio'] = ((close - opn) / (hl_range + 1e-10)).fillna(0).clip(-1, 1)

        # ── Volume ──
        features['volume_change'] = volume.pct_change().fillna(0).clip(-5, 5)
        vol_ma = volume.rolling(20).mean()
        features['volume_ma_ratio'] = ((volume / (vol_ma + 1e-10)) - 1).fillna(0).clip(-3, 3)

        # ── Moving averages ──
        for period in [7, 14, 50]:
            ma = close.rolling(period).mean()
            features[f'ma_{period}_ratio'] = ((close / (ma + 1e-10)) - 1).fillna(0).clip(-0.2, 0.2)

        # ── RSI ──
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50) / 100  # Normalize 0-1

        # ── MACD ──
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_norm = close.rolling(50).std() + 1e-10
        features['macd'] = (macd_line / macd_norm).fillna(0).clip(-3, 3)
        features['macd_signal'] = (macd_signal / macd_norm).fillna(0).clip(-3, 3)
        features['macd_hist'] = ((macd_line - macd_signal) / macd_norm).fillna(0).clip(-3, 3)

        # ── Bollinger Bands ──
        bb_ma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        features['bb_position'] = ((close - bb_lower) / (bb_upper - bb_lower + 1e-10)).fillna(0.5).clip(-0.5, 1.5)
        features['bb_width'] = ((bb_upper - bb_lower) / (bb_ma + 1e-10)).fillna(0).clip(0, 0.3)

        # ── Volatility ──
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        features['atr_14'] = (tr.rolling(14).mean() / close).fillna(0).clip(0, 0.1)
        features['realized_vol_24'] = close.pct_change().rolling(24).std().fillna(0).clip(0, 0.1)

        # ── Temporal features (cyclical encoding) ──
        if hasattr(df.index, 'hour'):
            hour = df.index.hour
            dow = df.index.dayofweek
        elif 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            hour = ts.dt.hour
            dow = ts.dt.dayofweek
        else:
            hour = pd.Series(0, index=df.index)
            dow = pd.Series(0, index=df.index)

        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        # Final cleanup
        result = features[TFTFeaturePreprocessor.FEATURE_NAMES].values.astype(np.float32)
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

        return result

    @staticmethod
    def prepare_targets(df: pd.DataFrame, horizons: List[int] = None) -> np.ndarray:
        """
        Create multi-horizon return targets.

        Args:
            df: DataFrame with 'close' column
            horizons: list of forward-looking periods (in candles)

        Returns:
            targets: (n_samples, n_horizons) numpy array of future returns
        """
        horizons = horizons or [1, 4, 12, 24]
        close = df['close'].astype(float)

        targets = np.zeros((len(df), len(horizons)), dtype=np.float32)
        for i, h in enumerate(horizons):
            future_return = close.pct_change(h).shift(-h)
            targets[:, i] = future_return.fillna(0).clip(-0.2, 0.2).values

        return targets


# ─── Model Manager ───────────────────────────────────────────────────────────

class TFTForecaster:
    """
    High-level wrapper for TFT model management.

    Handles:
    - Model creation and loading
    - Feature preprocessing
    - Inference with proper normalization
    - Model saving/loading
    """

    def __init__(
        self,
        model_dir: str = './data/models/tft',
        lookback: int = 72,  # 72 hours of context
        hidden_dim: int = 64,
        device: str = None,
    ):
        self.model_dir = model_dir
        self.lookback = lookback
        self.hidden_dim = hidden_dim

        # Auto-detect device
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.preprocessor = TFTFeaturePreprocessor()
        self.model: Optional[TemporalFusionTransformer] = None

        logger.info(f"🔮 TFTForecaster initialized (device={self.device}, lookback={lookback}h)")

    def create_model(self) -> TemporalFusionTransformer:
        """Create a new TFT model."""
        self.model = TemporalFusionTransformer(
            n_features=TFTFeaturePreprocessor.N_FEATURES,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        return self.model

    def load_model(self, symbol: str = 'BTCUSDT') -> bool:
        """Load a trained model for the given symbol."""
        model_path = os.path.join(self.model_dir, f'tft_{symbol.lower()}.pt')
        if not os.path.exists(model_path):
            logger.warning(f"No TFT model found at {model_path}")
            return False

        self.model = TemporalFusionTransformer(
            n_features=TFTFeaturePreprocessor.N_FEATURES,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"✅ TFT model loaded: {model_path}")
        return True

    def save_model(self, symbol: str = 'BTCUSDT'):
        """Save the trained model."""
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f'tft_{symbol.lower()}.pt')
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"💾 TFT model saved: {model_path}")

    def forecast(self, df: pd.DataFrame) -> Dict:
        """
        Generate multi-horizon forecast from OHLCV data.

        Args:
            df: DataFrame with OHLCV data (at least `lookback` rows)

        Returns:
            dict with keys:
                - return_1h, return_4h, return_12h, return_24h (median predictions)
                - confidence_1h, confidence_4h, etc.
                - direction_consensus (-1 to 1)
                - feature_importance (array)
        """
        if self.model is None:
            return self._empty_forecast()

        if len(df) < self.lookback:
            logger.warning(f"Need {self.lookback} rows, got {len(df)}")
            return self._empty_forecast()

        # Prepare features from last `lookback` candles
        features = self.preprocessor.prepare_features(df.tail(self.lookback))
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Run inference
        result = self.model.predict(x)

        horizons = self.model.horizons
        output = {
            'direction_consensus': float(result['direction_consensus'][0]),
            'feature_importance': result['feature_importance'].tolist(),
        }

        for i, h in enumerate(horizons):
            output[f'return_{h}h'] = float(result['predictions'][0, i])
            output[f'confidence_{h}h'] = float(result['confidences'][0, i])

        return output

    def get_rl_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get TFT predictions as a feature vector for the RL agent.

        Returns:
            Array of shape (9,):
            [return_1h, return_4h, return_12h, return_24h,
             confidence_1h, confidence_4h,
             direction_consensus,
             trend_strength, predicted_volatility]
        """
        forecast = self.forecast(df)

        horizons = [1, 4, 12, 24]
        returns = [forecast.get(f'return_{h}h', 0.0) for h in horizons]
        confs = [forecast.get(f'confidence_{h}h', 0.0) for h in [1, 4]]
        consensus = forecast.get('direction_consensus', 0.0)

        # Derived features
        trend_strength = abs(np.mean(returns))  # How strongly directional
        predicted_vol = np.std(returns) if len(returns) > 1 else 0.0  # Disagreement = volatility

        features = np.array(
            returns + confs + [consensus, trend_strength, predicted_vol],
            dtype=np.float32
        )
        return np.clip(features, -1.0, 1.0)

    @staticmethod
    def _empty_forecast() -> Dict:
        return {
            'return_1h': 0.0, 'return_4h': 0.0, 'return_12h': 0.0, 'return_24h': 0.0,
            'confidence_1h': 0.0, 'confidence_4h': 0.0,
            'confidence_12h': 0.0, 'confidence_24h': 0.0,
            'direction_consensus': 0.0,
            'feature_importance': [],
        }
