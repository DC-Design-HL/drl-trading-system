"""model_first_v1 — port of the legacy model-first PPO entry path.

Reproduces what the live bot did BEFORE STRUCTURE_FIRST_MODE became the
default. Each bar:
  1. Build a 117-dim observation (HTF features 1d/4h/1h/15m + alignment + pos)
  2. Apply VecNormalize stats from the trained fold
  3. Run model.predict(obs, deterministic=True) → action ∈ {HOLD, LONG, SHORT}
  4. Compute confidence = max action probability
  5. Emit Intent

Per-symbol model selection mirrors live's `_find_htf_model_path`:
  data/models/htf_walkforward_<asset>/ → best Sharpe fold
  → fall back to data/models/htf_walkforward_50pct_v2/

Caveats:
  * Only useful for backtesting "what would the dev / model-first config
    have produced". Not for new development — STRUCTURE_FIRST_MODE has
    been live since well before May 2026.
  * Inference is per-bar, single-batch. ~10-30 ms per call on CPU.
    A 90d × 4-symbol × 15m run = ~35k inferences = ~10-15 min just for
    model calls (plus feature engine). Acceptable for ablation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.features.htf_features import HTFFeatureEngine, HTFDataAligner

from ..base import EntryRule, Intent, Strategy, register_strategy

logger = logging.getLogger(__name__)


_HOLD = 0
_LONG = 1
_SHORT = 2


def _find_model_path(symbol: str, models_root: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Mirror of live_trading_htf._find_htf_model_path."""
    asset = symbol.replace("USDT", "").lower()

    # 1. Per-symbol walk-forward dir
    symbol_dir = models_root / f"htf_walkforward_{asset}"
    if symbol_dir.exists():
        m, v = _best_fold(symbol_dir)
        if m:
            return m, v

    # 2. Default walk-forward 50pct
    wfv_dir = models_root / "htf_walkforward_50pct_v2"
    if wfv_dir.exists():
        m, v = _best_fold(wfv_dir)
        if m:
            return m, v

    return None, None


def _best_fold(wf_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Pick the fold with highest test_metrics.sharpe_ratio.

    Live's _find_best_fold_model uses out-of-sample test sharpe (the
    OOS metric, not in-sample). Mirror that here: prefer
    test_metrics.sharpe_ratio, then val_metrics.sharpe_ratio, else 0.
    """
    best = None
    best_sharpe = -1e9
    for fold_dir in sorted(wf_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        model_zip = fold_dir / "best_model.zip"
        if not model_zip.exists():
            continue
        sharpe = -1e9
        result_json = fold_dir / "fold_result.json"
        if result_json.exists():
            try:
                d = json.loads(result_json.read_text())
                sharpe = float(
                    (d.get("test_metrics") or {}).get("sharpe_ratio")
                    or (d.get("val_metrics") or {}).get("sharpe_ratio")
                    or d.get("oos_sharpe")
                    or d.get("sharpe")
                    or 0.0
                )
            except Exception:
                pass
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            vn = fold_dir / "best_model_vecnorm.pkl"
            if not vn.exists():
                vn = fold_dir / "fold_model_vecnorm.pkl"
            best = (model_zip, vn if vn.exists() else None)
    return best if best else (None, None)


@dataclass
class ModelFirstEntry(EntryRule):
    """Per-symbol PPO model + VecNormalize + feature engine."""
    min_confidence: float = 0.0    # if >0, gate intents below threshold
    models_root: str = "data/models"
    cache_per_symbol: bool = True   # cache loaded models per symbol

    _per_symbol: Dict[str, Dict] = field(default_factory=dict, init=False)
    _aligner: Optional[HTFDataAligner] = field(default=None, init=False)
    _feature_engine: Optional[HTFFeatureEngine] = field(default=None, init=False)

    def __post_init__(self):
        self._aligner = HTFDataAligner()
        self._feature_engine = HTFFeatureEngine()

    def _load_for_symbol(self, symbol: str) -> Optional[Dict]:
        if symbol in self._per_symbol:
            return self._per_symbol[symbol] or None

        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import VecNormalize
            import pickle
        except Exception as exc:
            logger.error("stable_baselines3 unavailable: %s", exc)
            self._per_symbol[symbol] = None
            return None

        models_root = Path(self.models_root)
        model_path, vecnorm_path = _find_model_path(symbol, models_root)
        if model_path is None:
            logger.warning("No model found for %s under %s", symbol, models_root)
            self._per_symbol[symbol] = None
            return None

        try:
            model = PPO.load(str(model_path))
        except Exception as exc:
            logger.error("PPO.load failed for %s: %s", symbol, exc)
            self._per_symbol[symbol] = None
            return None

        vec_normalize = None
        if vecnorm_path and Path(vecnorm_path).exists():
            try:
                with open(vecnorm_path, "rb") as f:
                    vec_normalize = pickle.load(f)
            except Exception as exc:
                logger.warning("VecNormalize load failed for %s: %s", symbol, exc)

        info = {"model": model, "vec_normalize": vec_normalize, "model_path": str(model_path)}
        self._per_symbol[symbol] = info
        logger.info("Loaded %s model: %s", symbol, model_path)
        return info

    def _compute_observation(self, ctx) -> Optional[np.ndarray]:
        """Reproduce live_trading_htf.compute_observation() against ctx.

        ctx.primary is the 15m bars; the aligner builds 1d/4h/1h from
        the same df_15m using its `align_timestamps` method.
        """
        try:
            df_15m = ctx.primary
            # Aligner requires a DatetimeIndex on df_15m; cache stores open_time as int64.
            if not isinstance(df_15m.index, pd.DatetimeIndex):
                df_15m = df_15m.copy()
                df_15m.index = pd.to_datetime(df_15m["open_time"], unit="ms", utc=True)
            frames = self._aligner.align_timestamps(df_15m)
            df_1d = frames["1d"]
            df_4h = frames["4h"]
            df_1h = frames["1h"]
            df_15 = frames["15m"]

            if len(df_1d) < 5 or len(df_4h) < 10 or len(df_1h) < 20 or len(df_15) < 30:
                return None

            f1d = self._feature_engine.compute_1d_features(df_1d, len(df_1d) - 1)
            f4h = self._feature_engine.compute_4h_features(df_4h, len(df_4h) - 1)
            f1h = self._feature_engine.compute_1h_features(df_1h, len(df_1h) - 1)
            f15m = self._feature_engine.compute_15m_features(df_15, len(df_15) - 1)

            sig_1d = float(f1d[-1])
            sig_4h = float(f4h[-1])
            sig_1h = float(f1h[-1])
            sig_15m = float(f15m[-1])
            f_align = self._feature_engine.compute_alignment_full(
                sig_1d, sig_4h, sig_1h, sig_15m
            )

            feats_114 = np.concatenate([f1d, f4h, f1h, f15m, f_align])

            # Position state — derive from ctx
            current_price = float(df_15.iloc[-1]["close"])
            if ctx.position_state == "LONG" and ctx.entry_price > 0:
                unrealized_pnl = (current_price - ctx.entry_price) / (ctx.entry_price + 1e-10)
                pos_int = 1.0
            elif ctx.position_state == "SHORT" and ctx.entry_price > 0:
                unrealized_pnl = (ctx.entry_price - current_price) / (ctx.entry_price + 1e-10)
                pos_int = -1.0
            else:
                unrealized_pnl = 0.0
                pos_int = 0.0

            initial_balance = 5000.0
            balance_ratio = (ctx.balance - initial_balance) / (initial_balance + 1e-10)

            pos_state = np.array([
                pos_int,
                float(np.clip(unrealized_pnl, -0.5, 0.5)),
                float(np.clip(balance_ratio, -0.5, 0.5)),
            ], dtype=np.float32)

            obs = np.concatenate([feats_114, pos_state]).astype(np.float32)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            return obs
        except Exception as exc:
            logger.debug("compute_observation failed: %s", exc)
            return None

    def __call__(self, ctx) -> Intent:
        info = self._load_for_symbol(ctx.symbol)
        if info is None:
            return Intent(action="HOLD", reason="no_model")

        obs = self._compute_observation(ctx)
        if obs is None:
            return Intent(action="HOLD", reason="warmup")

        try:
            obs_2d = obs.reshape(1, -1)
            if info["vec_normalize"] is not None:
                try:
                    obs_2d = info["vec_normalize"].normalize_obs(obs_2d)
                except Exception:
                    pass
            action, _ = info["model"].predict(obs_2d, deterministic=True)
            action = int(action.item() if hasattr(action, "item") else action)

            # Confidence = max action probability
            confidence = self._model_confidence(info["model"], obs_2d)
        except Exception as exc:
            logger.debug("Model inference failed for %s: %s", ctx.symbol, exc)
            return Intent(action="HOLD", reason="model_error")

        if action == _LONG:
            act_str = "OPEN_LONG"
        elif action == _SHORT:
            act_str = "OPEN_SHORT"
        else:
            return Intent(action="HOLD", confidence=confidence, reason="model_hold")

        if confidence < self.min_confidence:
            return Intent(action="HOLD", confidence=confidence,
                          reason=f"low_conf<{self.min_confidence:.2f}")

        return Intent(action=act_str, confidence=confidence, reason="model_first")

    @staticmethod
    def _model_confidence(model, obs_2d) -> float:
        try:
            import torch
            with torch.no_grad():
                obs_tensor = model.policy.obs_to_tensor(obs_2d)[0]
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]
            return float(np.max(probs))
        except Exception:
            return 1.0 / 3.0


@register_strategy("model_first_v1")
class ModelFirstV1(Strategy):
    """Legacy model-first entry — for ablation against structure_first_v3."""

    def __init__(self, **overrides):
        self.entry = ModelFirstEntry(
            min_confidence=float(overrides.get("min_confidence", 0.0)),
            models_root=str(overrides.get("models_root", "data/models")),
        )
