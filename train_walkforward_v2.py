#!/usr/bin/env python3
"""
Walk-Forward Cross-Validation Training Pipeline v2

Anti-overfitting DRL retraining script for the crypto trading system.

Key design choices:
  - Strict chronological splits — no shuffling, ever
  - Walk-forward windows: 12mo train / 3mo val (in-window) / 3mo OOS test
  - VecNormalize stats frozen from training set, applied to val + test
  - Early stopping on VALIDATION Sharpe (not training reward)
  - Observation noise injection during training only
  - Train/val divergence logging to detect overfitting early
  - Ensemble of per-fold best models
  - Reduced model complexity (128→64 vs 256→256→128)
  - Multi-algorithm support: PPO (baseline), RecurrentPPO (LSTM), QRDQN (distributional)

Usage:
    python train_walkforward_v2.py --asset BTCUSDT --data-dir data/historical
    python train_walkforward_v2.py --asset BTCUSDT --algorithm recurrent_ppo
    python train_walkforward_v2.py --asset BTCUSDT --algorithm qrdqn
    python train_walkforward_v2.py --asset BTCUSDT --compare
    python train_walkforward_v2.py --asset BTCUSDT --eval-only --ensemble
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import json
import logging
import numbers
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO, QRDQN
    _SB3_CONTRIB_AVAILABLE = True
except ImportError:
    _SB3_CONTRIB_AVAILABLE = False
    RecurrentPPO = None
    QRDQN = None

from src.env.ultimate_env import UltimateTradingEnv
from src.backtest.data_loader import BinanceHistoricalDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalar types that the standard encoder can't handle."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Walk-forward windows
    "train_months": 12,
    "val_months": 3,   # Last 3 months of train window used for early stopping
    "test_months": 3,  # Strictly out-of-sample
    "slide_months": 3, # How far to advance per fold

    # PPO hyperparameters — conservative, regularized
    "ppo": {
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,       # Lower than default to reduce memorization speed
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.15,          # Conservative: was 0.2
        "ent_coef": 0.05,            # Higher entropy: more exploration, less policy collapse
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            # Smaller network: 128→64 vs original 256→256→128
            # ~30K params vs ~250K — much harder to memorize
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
            "activation_fn": "torch.nn.Tanh",  # resolved at init
        },
    },

    # RecurrentPPO (LSTM) — captures temporal dependencies in sequential market data
    # PRIMARY recommendation over plain PPO for time-series financial data
    "recurrent_ppo": {
        "policy": "MlpLstmPolicy",
        "learning_rate": 3e-5,       # Lower than PPO — LSTM training is more sensitive
        "n_steps": 512,              # Shorter rollouts — LSTM state propagated within episodes
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,           # More conservative clipping for stability
        "ent_coef": 0.02,            # Less entropy needed — LSTM explores via temporal memory
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "lstm_hidden_size": 64,
            "n_lstm_layers": 1,
            "net_arch": [64],        # Feedforward layers AFTER lstm
            "activation_fn": "torch.nn.Tanh",
        },
    },

    # QRDQN — distributional RL for risk-aware trading decisions
    # SECONDARY recommendation: models return distribution, not just mean
    "qrdqn": {
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "learning_starts": 5_000,    # Collect experience before first update
        "batch_size": 256,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "policy_kwargs": {
            "n_quantiles": 50,       # Distributional: 50 quantile heads
            "net_arch": [128, 64],
            "activation_fn": "torch.nn.ReLU",
        },
    },

    # Training
    "total_timesteps": 300_000,    # Per fold
    "eval_freq": 25_000,           # Validate every N steps
    "patience": 6,                 # Early stop after N checks without improvement
    "min_val_sharpe_to_save": 0.5, # Don't save model unless val Sharpe > this

    # Environment
    "initial_balance": 10_000.0,
    "lookback_window": 48,
    "trading_fee": 0.0004,
    "position_size": 0.25,
    "stop_loss_pct": 0.025,
    "take_profit_pct": 0.05,

    # Regularization
    "obs_noise_std": 0.01,          # Std of Gaussian noise relative to feature std
    "vecnorm_clip": 10.0,           # VecNormalize clip_obs value
}


# ---------------------------------------------------------------------------
# Algorithm helpers
# ---------------------------------------------------------------------------

def _check_algo(algorithm: str):
    """Raise if algorithm requires sb3-contrib but it's not installed."""
    if algorithm in ("recurrent_ppo", "qrdqn") and not _SB3_CONTRIB_AVAILABLE:
        raise ImportError(
            f"Algorithm '{algorithm}' requires sb3-contrib. "
            "Install it with: pip install sb3-contrib"
        )


def _is_recurrent(model) -> bool:
    """Return True if model is RecurrentPPO (needs LSTM state management)."""
    if RecurrentPPO is not None and isinstance(model, RecurrentPPO):
        return True
    return False


def _is_off_policy(model) -> bool:
    """Return True if model is an off-policy algorithm (e.g. QRDQN)."""
    if QRDQN is not None and isinstance(model, QRDQN):
        return True
    return False


def _resolve_activation(policy_kwargs: Dict) -> Dict:
    """Resolve activation_fn string to actual class."""
    import torch.nn as nn
    pk = dict(policy_kwargs)
    act = pk.pop("activation_fn", None)
    if act == "torch.nn.Tanh":
        pk["activation_fn"] = nn.Tanh
    elif act == "torch.nn.ReLU":
        pk["activation_fn"] = nn.ReLU
    return pk


def _build_model_kwargs(algorithm: str, config: Dict) -> Dict:
    """Return resolved keyword arguments for a given algorithm."""
    algo_cfg = dict(config[algorithm])
    pk = _resolve_activation(algo_cfg.pop("policy_kwargs", {}))
    algo_cfg["policy_kwargs"] = pk
    return algo_cfg


def build_model(algorithm: str, env, config: Dict):
    """Construct and return the model for the requested algorithm."""
    _check_algo(algorithm)
    kwargs = _build_model_kwargs(algorithm, config)
    policy = kwargs.pop("policy")

    if algorithm == "ppo":
        return PPO(policy=policy, env=env, verbose=0, **kwargs)
    elif algorithm == "recurrent_ppo":
        return RecurrentPPO(policy=policy, env=env, verbose=0, **kwargs)
    elif algorithm == "qrdqn":
        return QRDQN(policy=policy, env=env, verbose=0, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose: ppo, recurrent_ppo, qrdqn")


def load_model(algorithm: str, path: str):
    """Load a saved model for the given algorithm."""
    _check_algo(algorithm)
    if algorithm == "ppo":
        return PPO.load(path)
    elif algorithm == "recurrent_ppo":
        return RecurrentPPO.load(path)
    elif algorithm == "qrdqn":
        return QRDQN.load(path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# Observation Noise Wrapper
# ---------------------------------------------------------------------------

import gymnasium as gym


class ObservationNoiseWrapper(gym.Wrapper):
    """
    Adds Gaussian noise to observations during training.
    Forces the network to learn robust features, not memorize specific patterns.
    Applied ONLY during training — NOT during val/test evaluation.
    """
    def __init__(self, env: gym.Env, noise_std: float = 0.01):
        super().__init__(env)
        self.noise_std = noise_std

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_noise(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_noise(obs), reward, terminated, truncated, info

    def _add_noise(self, obs: np.ndarray) -> np.ndarray:
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, obs.shape).astype(obs.dtype)
            return obs + noise
        return obs

    def get_episode_metrics(self):
        return self.env.get_episode_metrics()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ValidationSharpCallback(BaseCallback):
    """
    Evaluates on validation set every eval_freq steps.
    Implements:
      - Early stopping on validation Sharpe
      - Train/val divergence detection
      - Best model saving

    Works with on-policy (PPO, RecurrentPPO) and off-policy (QRDQN) algorithms.
    For off-policy: ep_info_buffer may be empty during learning_starts; handled gracefully.
    For RecurrentPPO: LSTM states are reset per evaluation episode.
    """

    def __init__(
        self,
        val_env: UltimateTradingEnv,
        vec_norm: VecNormalize,
        check_freq: int = 25_000,
        patience: int = 6,
        min_sharpe_to_save: float = 0.5,
        model_save_path: str = "./best_model",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.val_env = val_env
        self.vec_norm = vec_norm
        self.check_freq = check_freq
        self.patience = patience
        self.min_sharpe_to_save = min_sharpe_to_save
        self.model_save_path = model_save_path

        self.best_val_sharpe = -np.inf
        self.no_improve_count = 0
        self.history: List[Dict] = []

    def _run_validation(self) -> Dict:
        """Run a full episode on validation data, return metrics.
        Handles LSTM state reset for RecurrentPPO automatically.
        """
        obs, _ = self.val_env.reset(options={"random_start": False})
        done = False
        step_count = 0

        # LSTM state management for RecurrentPPO
        lstm_states = None
        episode_start = np.array([True])

        while not done and step_count < 50_000:
            # Normalize observation using FROZEN train stats
            obs_t = obs.reshape(1, -1).astype(np.float32)
            obs_norm = self.vec_norm.normalize_obs(obs_t)[0]

            if _is_recurrent(self.model):
                action, lstm_states = self.model.predict(
                    obs_norm,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start = np.array([False])
            else:
                action, _ = self.model.predict(obs_norm, deterministic=True)

            obs, _, terminated, truncated, _ = self.val_env.step(int(action))
            done = terminated or truncated
            step_count += 1

        metrics = self.val_env.get_episode_metrics()
        return metrics

    def _get_train_mean_reward(self) -> float:
        # ep_info_buffer may be empty during QRDQN learning_starts — return 0 gracefully
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            return float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
        return 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        val_metrics = self._run_validation()
        val_sharpe = val_metrics.get("sharpe_ratio", 0.0)
        val_return = val_metrics.get("total_return", 0.0)
        val_trades = val_metrics.get("total_trades", 0)
        train_reward = self._get_train_mean_reward()

        record = {
            "step": self.n_calls,
            "val_sharpe": round(val_sharpe, 4),
            "val_return_pct": round(val_return * 100, 2),
            "val_trades": val_trades,
            "train_mean_reward": round(train_reward, 4),
            "overfit_ratio": round(abs(train_reward) / max(abs(val_sharpe), 0.01), 2),
        }
        self.history.append(record)

        if self.verbose:
            logger.info(
                f"  [Val@{self.n_calls:,}] Sharpe={val_sharpe:.3f} "
                f"Return={val_return*100:.1f}% Trades={val_trades} "
                f"TrainRwd={train_reward:.3f} "
                f"Overfit={record['overfit_ratio']:.1f}×"
            )

        # Overfitting alert
        if record["overfit_ratio"] > 5.0 and self.n_calls > 50_000:
            logger.warning(
                f"  [OVERFIT WARNING] train/val ratio={record['overfit_ratio']:.1f}× "
                f"— training diverging from validation performance"
            )

        # Save best model
        if val_sharpe > self.best_val_sharpe:
            improvement = val_sharpe - self.best_val_sharpe
            self.best_val_sharpe = val_sharpe
            self.no_improve_count = 0
            if val_sharpe >= self.min_sharpe_to_save:
                self.model.save(self.model_save_path)
                if self.verbose:
                    logger.info(
                        f"  [SAVED] Best model at step {self.n_calls:,} "
                        f"(Sharpe={val_sharpe:.3f}, +{improvement:.3f})"
                    )
        else:
            self.no_improve_count += 1
            if self.verbose:
                logger.info(
                    f"  [No improve] {self.no_improve_count}/{self.patience} "
                    f"(best={self.best_val_sharpe:.3f})"
                )

        # Early stopping
        if self.no_improve_count >= self.patience:
            logger.info(
                f"  [EARLY STOP] No validation improvement for {self.patience} checks. "
                f"Best val Sharpe: {self.best_val_sharpe:.3f}"
            )
            return False  # Stop training

        return True


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_episode_metrics(equity_curve: np.ndarray, trades: List[Dict]) -> Dict:
    """Compute comprehensive performance metrics from equity curve and trade list."""
    if len(equity_curve) < 2:
        return {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 1.0}

    initial = equity_curve[0]
    final = equity_curve[-1]
    total_return = (final - initial) / initial

    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
              if np.std(returns) > 0 else 0.0)

    downside = returns[returns < 0]
    sortino = (np.mean(returns) / np.std(downside) * np.sqrt(24 * 365)
               if len(downside) > 0 and np.std(downside) > 0 else 0.0)

    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.maximum(peak, 1e-9)
    max_drawdown = float(np.max(dd))

    if trades:
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] < 0]
        win_rate = len(wins) / len(trades)
        profit_factor = sum(wins) / max(abs(sum(losses)), 1e-9)
        avg_pnl = float(np.mean([t["pnl"] for t in trades]))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_pnl = 0.0

    return {
        "total_return": round(total_return, 4),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_drawdown, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_trades": len(trades) if trades else 0,
        "avg_trade_pnl": round(avg_pnl, 4),
        "final_equity": round(float(final), 2),
    }


def evaluate_model_on_env(
    model,
    env: UltimateTradingEnv,
    vec_norm: VecNormalize,
    deterministic: bool = True,
) -> Dict:
    """
    Evaluate model on an environment using frozen VecNormalize stats.
    Handles LSTM state management for RecurrentPPO automatically.
    Returns metrics dict from the environment.
    """
    obs, _ = env.reset(options={"random_start": False})
    done = False
    step_count = 0

    # LSTM state management for RecurrentPPO
    lstm_states = None
    episode_start = np.array([True])

    while not done and step_count < 100_000:
        obs_t = obs.reshape(1, -1).astype(np.float32)
        obs_norm = vec_norm.normalize_obs(obs_t)[0]

        if _is_recurrent(model):
            action, lstm_states = model.predict(
                obs_norm,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            episode_start = np.array([False])
        else:
            action, _ = model.predict(obs_norm, deterministic=deterministic)

        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        step_count += 1

    return env.get_episode_metrics()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_asset_data(symbol: str, data_dir: Path) -> pd.DataFrame:
    """
    Load historical data for an asset from CSV files in data_dir.
    Picks the file with the most data (longest date range).
    Falls back to Binance API download if no local file found.
    """
    pattern = f"{symbol}_1h_*.csv"
    candidates = sorted(data_dir.glob(pattern))

    if candidates:
        # Pick the file covering the longest period
        best = max(candidates, key=lambda p: p.stat().st_size)
        logger.info(f"Loading {symbol} from {best.name}")
        df = pd.read_csv(best, parse_dates=["timestamp"])
        if "timestamp" not in df.columns:
            df = df.reset_index()
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"  {len(df):,} rows, {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
        return df

    # Fallback: download 3 years from Binance
    logger.warning(f"No local data for {symbol}. Downloading 3 years from Binance...")
    fetcher = BinanceHistoricalDataFetcher()
    end_dt = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt.replace(year=end_dt.year - 3)
    df = fetcher.fetch_historical_data(symbol, "1h", start_dt, end_dt)
    if df.empty:
        raise ValueError(f"Could not fetch data for {symbol}")
    # Reset index to get timestamp as column
    df = df.reset_index()
    if "timestamp" not in df.columns:
        df = df.rename(columns={"index": "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{symbol}_1h_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df):,} rows to {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Window creation
# ---------------------------------------------------------------------------

def create_walk_forward_windows(
    df: pd.DataFrame,
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
    slide_months: int = 3,
) -> List[Dict]:
    """
    Create non-overlapping walk-forward windows.

    Within each fold:
      train_df: first (train_months - val_months) months
      val_df:   last val_months of the train window (for early stopping)
      test_df:  next test_months after the train window (strictly OOS)

    The val_df and test_df are NEVER used during training weight updates.
    """
    ts = pd.to_datetime(df["timestamp"])
    df = df.copy()
    df["timestamp"] = ts

    start = ts.iloc[0]
    end = ts.iloc[-1]

    windows = []
    fold = 0
    current = start

    train_delta = pd.DateOffset(months=train_months)
    val_delta = pd.DateOffset(months=val_months)
    test_delta = pd.DateOffset(months=test_months)
    slide_delta = pd.DateOffset(months=slide_months)

    while True:
        train_end = current + train_delta
        test_end = train_end + test_delta

        if test_end > end:
            break

        # Val is carved from the end of the train window — still "in-sample" for data,
        # but used only for early stopping signal, NOT for weight updates.
        pure_train_end = train_end - val_delta

        train_df = df[(df["timestamp"] >= current) & (df["timestamp"] < pure_train_end)].copy()
        val_df = df[(df["timestamp"] >= pure_train_end) & (df["timestamp"] < train_end)].copy()
        test_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < test_end)].copy()

        min_train_bars = 500
        min_val_bars = 200
        min_test_bars = 200

        if len(train_df) >= min_train_bars and len(val_df) >= min_val_bars and len(test_df) >= min_test_bars:
            windows.append({
                "fold": fold,
                "train_start": train_df["timestamp"].iloc[0].isoformat(),
                "train_end": train_df["timestamp"].iloc[-1].isoformat(),
                "val_start": val_df["timestamp"].iloc[0].isoformat(),
                "val_end": val_df["timestamp"].iloc[-1].isoformat(),
                "test_start": test_df["timestamp"].iloc[0].isoformat(),
                "test_end": test_df["timestamp"].iloc[-1].isoformat(),
                "train_bars": len(train_df),
                "val_bars": len(val_df),
                "test_bars": len(test_df),
                "train_df": train_df.reset_index(drop=True),
                "val_df": val_df.reset_index(drop=True),
                "test_df": test_df.reset_index(drop=True),
            })
            fold += 1

        current = current + slide_delta

    logger.info(f"Created {len(windows)} walk-forward folds")
    for w in windows:
        logger.info(
            f"  Fold {w['fold']}: "
            f"train [{w['train_start'][:10]} → {w['train_end'][:10]}, {w['train_bars']} bars] "
            f"val [{w['val_start'][:10]} → {w['val_end'][:10]}, {w['val_bars']} bars] "
            f"test [{w['test_start'][:10]} → {w['test_end'][:10]}, {w['test_bars']} bars]"
        )
    return windows


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_env_fn(df: pd.DataFrame, config: Dict, noise_std: float = 0.0, seed: int = 0):
    """Return a callable that creates a training environment."""
    def _make():
        env = UltimateTradingEnv(
            df=df,
            initial_balance=config["initial_balance"],
            lookback_window=config["lookback_window"],
            trading_fee=config["trading_fee"],
            position_size=config["position_size"],
            stop_loss_pct=config["stop_loss_pct"],
            take_profit_pct=config["take_profit_pct"],
            training_mode=True,  # Disable live API calls during training
        )
        if noise_std > 0:
            env = ObservationNoiseWrapper(env, noise_std=noise_std)
        return Monitor(env)
    return _make


def build_training_envs(
    train_df: pd.DataFrame,
    config: Dict,
    n_envs: int = 4,
    algorithm: str = "ppo",
) -> VecNormalize:
    """
    Build vectorized + normalized training environments.
    Uses n_envs=1 for off-policy algorithms (QRDQN) where parallelism provides no benefit.
    Returns VecNormalize wrapper (with training=True).
    """
    # Off-policy algorithms work best with a single environment
    effective_n_envs = 1 if algorithm == "qrdqn" else n_envs
    fns = [make_env_fn(train_df, config, noise_std=config["obs_noise_std"], seed=i)
           for i in range(effective_n_envs)]
    vec_env = DummyVecEnv(fns)
    vec_norm = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,   # Do NOT normalize reward — it distorts the signal
        clip_obs=config["vecnorm_clip"],
        training=True,
    )
    return vec_norm


def build_eval_env(df: pd.DataFrame, config: Dict, train_vec_norm: VecNormalize) -> Tuple:
    """
    Build evaluation environment with FROZEN VecNormalize stats from training.
    Returns (raw_env, frozen_vec_norm) — use frozen_vec_norm to normalize obs.
    """
    raw_env = UltimateTradingEnv(
        df=df,
        initial_balance=config["initial_balance"],
        lookback_window=config["lookback_window"],
        trading_fee=config["trading_fee"],
        position_size=config["position_size"],
        stop_loss_pct=config["stop_loss_pct"],
        take_profit_pct=config["take_profit_pct"],
        training_mode=True,  # Consistent with training envs; no live API calls
    )
    # Create a single-env VecNormalize clone with frozen stats
    single_vec = DummyVecEnv([make_env_fn(df, config, noise_std=0.0)])
    frozen_norm = VecNormalize(single_vec, norm_obs=True, norm_reward=False,
                               clip_obs=config["vecnorm_clip"], training=False)
    # Copy running statistics from the training env
    frozen_norm.obs_rms = copy.deepcopy(train_vec_norm.obs_rms)
    frozen_norm.ret_rms = copy.deepcopy(train_vec_norm.ret_rms)
    frozen_norm.training = False

    return raw_env, frozen_norm


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class WalkForwardTrainerV2:
    """
    Production-grade walk-forward training pipeline with overfitting controls.
    Supports PPO (baseline), RecurrentPPO (LSTM), and QRDQN (distributional RL).
    """

    def __init__(
        self,
        asset: str,
        config: Dict,
        output_dir: Path,
        data_dir: Path,
        algorithm: str = "ppo",
    ):
        self.asset = asset
        self.config = config
        self.algorithm = algorithm
        self.output_dir = output_dir / asset / algorithm
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir

        _check_algo(algorithm)

        # Set up per-run file handler
        log_path = self.output_dir / "training.log"
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

        self.fold_results: List[Dict] = []
        self.fold_models: List[Path] = []

    def train_fold(self, window: Dict) -> Dict:
        """Train on one walk-forward fold. Returns fold metrics."""
        fold = window["fold"]
        fold_dir = self.output_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold} — {self.asset} [{self.algorithm.upper()}]")
        logger.info(f"  Train: {window['train_start'][:10]} → {window['train_end'][:10]} ({window['train_bars']} bars)")
        logger.info(f"  Val:   {window['val_start'][:10]} → {window['val_end'][:10]} ({window['val_bars']} bars)")
        logger.info(f"  Test:  {window['test_start'][:10]} → {window['test_end'][:10]} ({window['test_bars']} bars)")
        logger.info(f"{'='*70}")

        train_df = window["train_df"]
        val_df = window["val_df"]
        test_df = window["test_df"]

        # --- Build environments ---
        logger.info("Building training environments...")
        try:
            train_vec_norm = build_training_envs(
                train_df, self.config, n_envs=4, algorithm=self.algorithm
            )
        except Exception as exc:
            logger.error(f"  Failed to build training env: {exc}", exc_info=True)
            return self._empty_fold_result(fold, window, str(exc))

        # Eval environments use frozen train stats
        try:
            val_raw_env, val_frozen_norm = build_eval_env(val_df, self.config, train_vec_norm)
            test_raw_env, test_frozen_norm = build_eval_env(test_df, self.config, train_vec_norm)
        except Exception as exc:
            logger.error(f"  Failed to build eval envs: {exc}", exc_info=True)
            return self._empty_fold_result(fold, window, str(exc))

        # --- Build model ---
        logger.info(f"Initializing {self.algorithm.upper()} model...")
        try:
            model = build_model(self.algorithm, train_vec_norm, self.config)
        except Exception as exc:
            logger.error(f"  Failed to build model: {exc}", exc_info=True)
            return self._empty_fold_result(fold, window, str(exc))

        # --- Validation callback (early stopping on val Sharpe) ---
        model_save_path = str(fold_dir / "best_model")
        val_callback = ValidationSharpCallback(
            val_env=val_raw_env,
            vec_norm=val_frozen_norm,
            check_freq=self.config["eval_freq"],
            patience=self.config["patience"],
            min_sharpe_to_save=self.config["min_val_sharpe_to_save"],
            model_save_path=model_save_path,
            verbose=1,
        )

        # --- Train ---
        logger.info(f"Training for up to {self.config['total_timesteps']:,} steps...")
        train_start = time.time()
        try:
            model.learn(
                total_timesteps=self.config["total_timesteps"],
                callback=val_callback,
                progress_bar=False,
                reset_num_timesteps=True,
            )
        except Exception as exc:
            logger.error(f"  Training failed: {exc}", exc_info=True)

        train_elapsed = time.time() - train_start
        logger.info(f"  Training done in {train_elapsed:.0f}s. Steps: {model.num_timesteps:,}")

        # --- Load best model (by validation Sharpe) ---
        best_model_path = model_save_path + ".zip"
        if Path(best_model_path).exists():
            logger.info(f"  Loading best model from {best_model_path}")
            model = load_model(self.algorithm, model_save_path)
        else:
            logger.warning("  No best model saved (val Sharpe never exceeded threshold). Using final model.")
            model.save(model_save_path)

        # Save VecNormalize stats alongside model
        train_vec_norm.save(str(fold_dir / "vecnorm.pkl"))
        # Save algorithm name so ensemble can load correctly
        with open(fold_dir / "algorithm.txt", "w") as f:
            f.write(self.algorithm)

        # --- Evaluate on VALIDATION set ---
        logger.info("Evaluating on validation set...")
        try:
            val_metrics = evaluate_model_on_env(model, val_raw_env, val_frozen_norm)
        except Exception as exc:
            logger.error(f"  Val evaluation failed: {exc}", exc_info=True)
            return self._empty_fold_result(fold, window, str(exc))
        logger.info(
            f"  Val  → Sharpe={val_metrics['sharpe_ratio']:.3f} "
            f"Return={val_metrics['total_return_pct']:.1f}% "
            f"MaxDD={val_metrics['max_drawdown_pct']:.1f}% "
            f"Trades={val_metrics['total_trades']}"
        )

        # --- Evaluate on TEST set (strictly out-of-sample) ---
        logger.info("Evaluating on TEST (out-of-sample) set...")
        try:
            test_metrics = evaluate_model_on_env(model, test_raw_env, test_frozen_norm)
        except Exception as exc:
            logger.error(f"  Test evaluation failed: {exc}", exc_info=True)
            return self._empty_fold_result(fold, window, str(exc))
        logger.info(
            f"  Test → Sharpe={test_metrics['sharpe_ratio']:.3f} "
            f"Return={test_metrics['total_return_pct']:.1f}% "
            f"MaxDD={test_metrics['max_drawdown_pct']:.1f}% "
            f"Trades={test_metrics['total_trades']}"
        )

        # --- Overfitting ratio ---
        best_val_sharpe = val_callback.best_val_sharpe
        overfit_ratio = abs(best_val_sharpe) / max(abs(test_metrics["sharpe_ratio"]), 0.01)
        overfit_flag = overfit_ratio > 3.0
        logger.info(
            f"  Overfit ratio (val/test Sharpe): {overfit_ratio:.2f}× "
            f"{'[OVERFIT WARNING]' if overfit_flag else '[OK]'}"
        )

        fold_result = {
            "fold": fold,
            "asset": self.asset,
            "algorithm": self.algorithm,
            "train_start": window["train_start"][:10],
            "train_end": window["train_end"][:10],
            "val_start": window["val_start"][:10],
            "val_end": window["val_end"][:10],
            "test_start": window["test_start"][:10],
            "test_end": window["test_end"][:10],
            "train_bars": window["train_bars"],
            "val_bars": window["val_bars"],
            "test_bars": window["test_bars"],
            "steps_trained": model.num_timesteps,
            "train_elapsed_s": round(train_elapsed, 1),
            "best_val_sharpe": round(best_val_sharpe, 4),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "overfit_ratio_val_test": round(overfit_ratio, 3),
            "overfit_flag": overfit_flag,
            "callback_history": val_callback.history,
            "model_path": str(fold_dir / "best_model.zip"),
            "vecnorm_path": str(fold_dir / "vecnorm.pkl"),
        }

        # Save fold result JSON
        with open(fold_dir / "fold_result.json", "w") as f:
            json.dump({k: v for k, v in fold_result.items()
                       if k not in ("train_df", "val_df", "test_df")}, f, indent=2, cls=_NumpyEncoder)

        # Clean up
        train_vec_norm.close()
        val_frozen_norm.close()
        test_frozen_norm.close()

        return fold_result

    def _empty_fold_result(self, fold: int, window: Dict, error: str) -> Dict:
        return {
            "fold": fold,
            "asset": self.asset,
            "algorithm": self.algorithm,
            "error": error,
            "test_metrics": {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 1.0,
                             "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "total_trades": 0},
            "val_metrics": {"sharpe_ratio": 0.0},
            "best_val_sharpe": 0.0,
            "overfit_flag": False,
        }

    def run(self, windows: List[Dict]) -> Dict:
        """Run all walk-forward folds and aggregate results."""
        logger.info(f"\n{'#'*70}")
        logger.info(f"WALK-FORWARD TRAINING: {self.asset} [{self.algorithm.upper()}] ({len(windows)} folds)")
        logger.info(f"{'#'*70}\n")

        for window in windows:
            result = self.train_fold(window)
            self.fold_results.append(result)
            model_path = Path(result.get("model_path", ""))
            if model_path.exists():
                self.fold_models.append(model_path)

        return self.aggregate_results()

    def aggregate_results(self) -> Dict:
        """Compute aggregate OOS metrics across all folds."""
        valid = [r for r in self.fold_results if "error" not in r]
        if not valid:
            logger.error("No valid fold results to aggregate")
            return {}

        oos_sharpes = [r["test_metrics"]["sharpe_ratio"] for r in valid]
        oos_returns = [r["test_metrics"]["total_return_pct"] for r in valid]
        oos_drawdowns = [r["test_metrics"]["max_drawdown_pct"] for r in valid]
        oos_trades = [r["test_metrics"]["total_trades"] for r in valid]
        overfit_flags = [r["overfit_flag"] for r in valid]

        summary = {
            "asset": self.asset,
            "algorithm": self.algorithm,
            "total_folds": len(self.fold_results),
            "valid_folds": len(valid),
            "oos_sharpe_mean": round(float(np.mean(oos_sharpes)), 4),
            "oos_sharpe_std": round(float(np.std(oos_sharpes)), 4),
            "oos_sharpe_min": round(float(np.min(oos_sharpes)), 4),
            "oos_sharpe_max": round(float(np.max(oos_sharpes)), 4),
            "oos_return_mean_pct": round(float(np.mean(oos_returns)), 2),
            "oos_return_std_pct": round(float(np.std(oos_returns)), 2),
            "oos_drawdown_mean_pct": round(float(np.mean(oos_drawdowns)), 2),
            "positive_fold_pct": round(sum(r > 0 for r in oos_returns) / len(oos_returns) * 100, 1),
            "avg_trades_per_fold": round(float(np.mean(oos_trades)), 1),
            "overfit_flags_count": sum(overfit_flags),
            "per_fold": [
                {
                    "fold": r["fold"],
                    "test_period": f"{r.get('test_start','')} → {r.get('test_end','')}",
                    "oos_sharpe": r["test_metrics"]["sharpe_ratio"],
                    "oos_return_pct": r["test_metrics"]["total_return_pct"],
                    "oos_drawdown_pct": r["test_metrics"]["max_drawdown_pct"],
                    "oos_trades": r["test_metrics"]["total_trades"],
                    "overfit": r["overfit_flag"],
                }
                for r in valid
            ],
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"WALK-FORWARD SUMMARY — {self.asset} [{self.algorithm.upper()}]")
        logger.info(f"{'='*70}")
        logger.info(f"  OOS Sharpe:  {summary['oos_sharpe_mean']:.3f} ± {summary['oos_sharpe_std']:.3f}")
        logger.info(f"  OOS Return:  {summary['oos_return_mean_pct']:.1f}% ± {summary['oos_return_std_pct']:.1f}%")
        logger.info(f"  Max DD:      {summary['oos_drawdown_mean_pct']:.1f}%")
        logger.info(f"  Positive:    {summary['positive_fold_pct']:.0f}% of folds")
        logger.info(f"  Overfit:     {summary['overfit_flags_count']}/{summary['valid_folds']} folds flagged")
        logger.info(f"{'='*70}")

        # Save summary
        summary_path = self.output_dir / "fold_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=_NumpyEncoder)
        logger.info(f"Summary saved to {summary_path}")

        # Save overfit report
        overfit_report = self._build_overfit_report(valid)
        with open(self.output_dir / "overfit_report.json", "w") as f:
            json.dump(overfit_report, f, indent=2, cls=_NumpyEncoder)

        return summary

    def _build_overfit_report(self, valid_folds: List[Dict]) -> Dict:
        """Analyze train vs val vs test performance gaps to quantify overfitting."""
        val_sharpes = [r["best_val_sharpe"] for r in valid_folds]
        test_sharpes = [r["test_metrics"]["sharpe_ratio"] for r in valid_folds]

        val_test_ratios = [
            abs(v) / max(abs(t), 0.01)
            for v, t in zip(val_sharpes, test_sharpes)
        ]

        report = {
            "asset": self.asset,
            "algorithm": self.algorithm,
            "interpretation": {
                "val_test_ratio < 1.5": "Excellent generalization",
                "val_test_ratio 1.5-3.0": "Mild overfitting (acceptable)",
                "val_test_ratio 3.0-5.0": "Moderate overfitting (review features)",
                "val_test_ratio > 5.0": "Severe overfitting (do not deploy)",
            },
            "avg_val_sharpe": round(float(np.mean(val_sharpes)), 4),
            "avg_test_sharpe": round(float(np.mean(test_sharpes)), 4),
            "avg_val_test_ratio": round(float(np.mean(val_test_ratios)), 3),
            "max_val_test_ratio": round(float(np.max(val_test_ratios)), 3),
            "per_fold": [
                {
                    "fold": r["fold"],
                    "val_sharpe": r["best_val_sharpe"],
                    "test_sharpe": r["test_metrics"]["sharpe_ratio"],
                    "ratio": round(vr, 3),
                    "verdict": (
                        "excellent" if vr < 1.5 else
                        "mild" if vr < 3.0 else
                        "moderate" if vr < 5.0 else
                        "severe"
                    ),
                }
                for r, vr in zip(valid_folds, val_test_ratios)
            ],
        }

        avg_ratio = report["avg_val_test_ratio"]
        if avg_ratio < 1.5:
            report["overall_verdict"] = "EXCELLENT — model generalizes well to unseen data"
        elif avg_ratio < 3.0:
            report["overall_verdict"] = "ACCEPTABLE — mild overfitting, monitor live performance"
        elif avg_ratio < 5.0:
            report["overall_verdict"] = "CAUTION — moderate overfitting, reduce model complexity"
        else:
            report["overall_verdict"] = "DO NOT DEPLOY — severe overfitting detected"

        logger.info(f"\nOverfit Report: {report['overall_verdict']}")
        logger.info(f"  Avg val/test Sharpe ratio: {avg_ratio:.2f}×")

        return report

    def build_ensemble(self, windows: List[Dict]) -> Optional[Dict]:
        """
        Load all fold models and evaluate their ensemble on each test period.

        Uses majority voting across models, which works across all algorithm types
        (PPO, RecurrentPPO, QRDQN). LSTM states are tracked per model for RecurrentPPO.
        """
        if not self.fold_models:
            logger.warning("No fold models available for ensemble")
            return None

        logger.info(f"\nBuilding ensemble from {len(self.fold_models)} fold models [{self.algorithm.upper()}]...")

        # Load all models and detect their algorithms
        models = []
        model_algos = []
        for model_path in self.fold_models:
            if model_path.exists():
                try:
                    # Read algorithm from saved metadata if available
                    algo_file = model_path.parent / "algorithm.txt"
                    algo = self.algorithm
                    if algo_file.exists():
                        algo = algo_file.read_text().strip()
                    m = load_model(algo, str(model_path))
                    models.append(m)
                    model_algos.append(algo)
                    logger.info(f"  Loaded: {model_path.name} [{algo}]")
                except Exception as exc:
                    logger.warning(f"  Could not load {model_path}: {exc}")

        if not models:
            logger.error("No models loaded for ensemble")
            return None

        # Evaluate ensemble on each fold's test set
        ensemble_metrics = []
        for window in windows:
            fold = window["fold"]
            fold_dir = self.output_dir / f"fold_{fold:02d}"
            vecnorm_path = fold_dir / "vecnorm.pkl"

            if not vecnorm_path.exists():
                continue

            test_df = window["test_df"]
            config = self.config

            test_env = UltimateTradingEnv(
                df=test_df,
                initial_balance=config["initial_balance"],
                lookback_window=config["lookback_window"],
                trading_fee=config["trading_fee"],
                position_size=config["position_size"],
            )

            # Load frozen VecNorm for this fold
            single_vec = DummyVecEnv([make_env_fn(test_df, config)])
            frozen_norm = VecNormalize.load(str(vecnorm_path), single_vec)
            frozen_norm.training = False
            frozen_norm.norm_reward = False

            # Per-model LSTM states (None for non-recurrent models)
            lstm_states_per_model = [None] * len(models)
            episode_starts_per_model = [np.array([True])] * len(models)

            # Run ensemble evaluation — majority voting
            obs, _ = test_env.reset(options={"random_start": False})
            done = False
            step_count = 0

            while not done and step_count < 100_000:
                obs_t = obs.reshape(1, -1).astype(np.float32)
                obs_norm = frozen_norm.normalize_obs(obs_t)[0]

                # Collect action from each model
                actions = []
                for i, (m, algo) in enumerate(zip(models, model_algos)):
                    if _is_recurrent(m):
                        act, lstm_states_per_model[i] = m.predict(
                            obs_norm,
                            state=lstm_states_per_model[i],
                            episode_start=episode_starts_per_model[i],
                            deterministic=True,
                        )
                        episode_starts_per_model[i] = np.array([False])
                    else:
                        act, _ = m.predict(obs_norm, deterministic=True)
                    actions.append(int(act))

                # Majority vote
                from collections import Counter
                action = Counter(actions).most_common(1)[0][0]

                obs, _, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                step_count += 1

            metrics = test_env.get_episode_metrics()
            metrics["fold"] = fold
            metrics["test_period"] = f"{window['test_start'][:10]} → {window['test_end'][:10]}"
            ensemble_metrics.append(metrics)
            frozen_norm.close()

            logger.info(
                f"  Fold {fold} ensemble test → "
                f"Sharpe={metrics['sharpe_ratio']:.3f} "
                f"Return={metrics['total_return_pct']:.1f}% "
                f"Trades={metrics['total_trades']}"
            )

        if not ensemble_metrics:
            return None

        sharpes = [m["sharpe_ratio"] for m in ensemble_metrics]
        returns = [m["total_return_pct"] for m in ensemble_metrics]

        ensemble_summary = {
            "asset": self.asset,
            "algorithm": self.algorithm,
            "num_models": len(models),
            "num_folds_evaluated": len(ensemble_metrics),
            "oos_sharpe_mean": round(float(np.mean(sharpes)), 4),
            "oos_sharpe_std": round(float(np.std(sharpes)), 4),
            "oos_return_mean_pct": round(float(np.mean(returns)), 2),
            "positive_fold_pct": round(sum(r > 0 for r in returns) / len(returns) * 100, 1),
            "per_fold": ensemble_metrics,
        }

        with open(self.output_dir / "ensemble_metrics.json", "w") as f:
            json.dump(ensemble_summary, f, indent=2, cls=_NumpyEncoder)

        logger.info(f"\nEnsemble OOS Sharpe: {ensemble_summary['oos_sharpe_mean']:.3f} "
                    f"± {ensemble_summary['oos_sharpe_std']:.3f}")
        logger.info(f"Ensemble OOS Return: {ensemble_summary['oos_return_mean_pct']:.1f}%")
        logger.info(f"Positive folds: {ensemble_summary['positive_fold_pct']:.0f}%")

        return ensemble_summary


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    asset: str,
    config: Dict,
    output_dir: Path,
    data_dir: Path,
    windows: List[Dict],
    algorithms: List[str] = None,
) -> Dict:
    """
    Train all algorithms on the same walk-forward windows and output a comparison.
    Returns dict keyed by algorithm name with their aggregate summaries.
    """
    if algorithms is None:
        algorithms = ["ppo", "recurrent_ppo", "qrdqn"]

    results = {}
    for algo in algorithms:
        if algo in ("recurrent_ppo", "qrdqn") and not _SB3_CONTRIB_AVAILABLE:
            logger.warning(f"Skipping {algo} — sb3-contrib not available")
            continue

        logger.info(f"\n{'#'*70}")
        logger.info(f"COMPARISON RUN: {algo.upper()}")
        logger.info(f"{'#'*70}")

        trainer = WalkForwardTrainerV2(
            asset=asset,
            config=config,
            output_dir=output_dir,
            data_dir=data_dir,
            algorithm=algo,
        )
        summary = trainer.run(windows)
        results[algo] = summary

    # Print comparison table
    _print_comparison_table(results)

    # Save comparison
    comparison_path = output_dir / asset / "algorithm_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    logger.info(f"\nComparison saved to {comparison_path}")

    return results


def _print_comparison_table(results: Dict):
    """Print a formatted comparison table of algorithm results."""
    if not results:
        return

    header = f"\n{'='*72}"
    logger.info(header)
    logger.info("ALGORITHM COMPARISON — OOS WALK-FORWARD RESULTS")
    logger.info(f"{'='*72}")
    logger.info(
        f"{'Algorithm':<18} {'Sharpe':>8} {'±':>6} {'Return%':>9} {'MaxDD%':>8} "
        f"{'Pos%':>6} {'Overfit':>8}"
    )
    logger.info(f"{'-'*72}")

    # Sort by OOS Sharpe descending
    ranked = sorted(results.items(), key=lambda kv: kv[1].get("oos_sharpe_mean", -99), reverse=True)
    for algo, r in ranked:
        logger.info(
            f"{algo:<18} "
            f"{r.get('oos_sharpe_mean', 0):>8.3f} "
            f"{r.get('oos_sharpe_std', 0):>6.3f} "
            f"{r.get('oos_return_mean_pct', 0):>9.1f} "
            f"{r.get('oos_drawdown_mean_pct', 0):>8.1f} "
            f"{r.get('positive_fold_pct', 0):>6.0f}% "
            f"{r.get('overfit_flags_count', 0):>3}/{r.get('valid_folds', 0):<3}"
        )

    logger.info(f"{'='*72}\n")
    if ranked:
        best_algo = ranked[0][0]
        logger.info(f"Best algorithm by OOS Sharpe: {best_algo.upper()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Cross-Validation Training v2"
    )
    parser.add_argument("--asset", type=str, default="BTCUSDT",
                        help="Asset to train (e.g. BTCUSDT)")
    parser.add_argument("--data-dir", type=str, default="data/historical",
                        help="Directory containing historical CSV files")
    parser.add_argument("--output-dir", type=str, default="data/models/wfv2",
                        help="Base directory for model outputs")
    parser.add_argument("--algorithm", type=str, default="ppo",
                        choices=["ppo", "recurrent_ppo", "qrdqn"],
                        help="RL algorithm: ppo (default), recurrent_ppo (LSTM), qrdqn (distributional)")
    parser.add_argument("--compare", action="store_true",
                        help="Run all 3 algorithms and output a comparison table")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--slide-months", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Observation noise std (0 to disable)")
    parser.add_argument("--min-val-sharpe", type=float, default=0.5,
                        help="Min val Sharpe required to save a model")
    parser.add_argument("--ensemble", action="store_true",
                        help="Build ensemble after training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing models")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Limit number of folds (useful for quick testing)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Merge CLI args into config
    config = dict(DEFAULT_CONFIG)
    config["train_months"] = args.train_months
    config["val_months"] = args.val_months
    config["test_months"] = args.test_months
    config["slide_months"] = args.slide_months
    config["total_timesteps"] = args.total_timesteps
    config["eval_freq"] = args.eval_freq
    config["patience"] = args.patience
    config["obs_noise_std"] = args.noise_std
    config["min_val_sharpe_to_save"] = args.min_val_sharpe

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_display = "ALL (comparison)" if args.compare else args.algorithm.upper()
    logger.info(f"\nWalk-Forward Trainer v2")
    logger.info(f"  Asset:       {args.asset}")
    logger.info(f"  Algorithm:   {algo_display}")
    logger.info(f"  Data dir:    {data_dir.resolve()}")
    logger.info(f"  Output dir:  {output_dir.resolve()}")
    logger.info(f"  Window:      {args.train_months}mo train / {args.val_months}mo val / {args.test_months}mo test")
    logger.info(f"  Timesteps:   {args.total_timesteps:,} per fold")
    logger.info(f"  Patience:    {args.patience} checks ({args.patience * args.eval_freq:,} steps)")
    logger.info(f"  Noise std:   {args.noise_std}")

    # Load data
    df = load_asset_data(args.asset, data_dir)

    # Create windows
    windows = create_walk_forward_windows(
        df,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        slide_months=args.slide_months,
    )

    if not windows:
        logger.error(
            "Not enough data for walk-forward windows. "
            "Run: python download_historical_data.py --years 3"
        )
        return 1

    if args.max_folds:
        windows = windows[:args.max_folds]
        logger.info(f"Limiting to {args.max_folds} folds (--max-folds)")

    # --- Comparison mode: run all 3 algorithms ---
    if args.compare:
        run_comparison(
            asset=args.asset,
            config=config,
            output_dir=output_dir,
            data_dir=data_dir,
            windows=windows,
        )
        logger.info("\nDone.")
        return 0

    # Save config
    asset_dir = output_dir / args.asset / args.algorithm
    asset_dir.mkdir(parents=True, exist_ok=True)
    with open(asset_dir / "config.json", "w") as f:
        safe_config = {k: v for k, v in config.items()
                       if k not in ("ppo", "recurrent_ppo", "qrdqn")}
        safe_config["algorithm"] = args.algorithm
        safe_config[args.algorithm] = {
            k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
            for k, v in config[args.algorithm].items()
        }
        json.dump(safe_config, f, indent=2, cls=_NumpyEncoder)

    # Initialize trainer
    trainer = WalkForwardTrainerV2(
        asset=args.asset,
        config=config,
        output_dir=output_dir,
        data_dir=data_dir,
        algorithm=args.algorithm,
    )

    if not args.eval_only:
        # Train all folds
        summary = trainer.run(windows)
        if not summary:
            logger.error("Training produced no results")
            return 1
    else:
        # Reload existing fold results
        for window in windows:
            fold = window["fold"]
            result_path = output_dir / args.asset / args.algorithm / f"fold_{fold:02d}" / "fold_result.json"
            model_path = output_dir / args.asset / args.algorithm / f"fold_{fold:02d}" / "best_model.zip"
            if result_path.exists():
                with open(result_path) as f:
                    result = json.load(f)
                trainer.fold_results.append(result)
                if model_path.exists():
                    trainer.fold_models.append(model_path)
            else:
                logger.warning(f"No result found for fold {fold}")

    # Build ensemble if requested
    if args.ensemble:
        # Need test_dfs in windows — reload them
        windows_with_data = create_walk_forward_windows(
            df,
            train_months=args.train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            slide_months=args.slide_months,
        )
        if args.max_folds:
            windows_with_data = windows_with_data[:args.max_folds]
        trainer.build_ensemble(windows_with_data)

    logger.info("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
