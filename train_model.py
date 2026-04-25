#!/usr/bin/env python3
"""
Unified Walk-Forward Training Script for Structure-Gated DRL Filter
====================================================================

Trains a RecurrentPPO (LSTM) model per symbol that acts as a binary
ACCEPT/REJECT gate on BOS/CHOCH market structure signals.

Key design choices from DRL research review:
  1. Model = Structure-Gated Filter (binary ACCEPT/REJECT), NOT entry maker
  2. Reward = Differential Sharpe Ratio (Markovian)
  3. ~50 features (curated, no overfitting)
  4. Strict 3-way split with 48h embargo
  5. Early stopping on validation Sharpe

Usage:
    python train_model.py --symbol BTCUSDT
    python train_model.py --symbol BTCUSDT --download-data
    python train_model.py --symbol SOLUSDT --folds 5   # fewer folds for testing

Author:  CEO bot
Date:    2026-04-13
"""

import sys
import os
import gc
import json
import signal as sig_module
import time
import copy
import shutil
import argparse
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("train_model")

# ---------------------------------------------------------------------------
# Lazy imports (fail fast with clear messages)
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    logger.error("PyTorch not installed. Run: pip install -r requirements-training.txt")
    sys.exit(1)

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    logger.error("stable-baselines3 not installed. Run: pip install -r requirements-training.txt")
    sys.exit(1)

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    logger.error("sb3-contrib not installed. Run: pip install sb3-contrib>=2.3.0")
    sys.exit(1)

from src.env.structure_filter_env import StructureFilterEnv

# ---------------------------------------------------------------------------
# Device detection (Mac M3 MPS / CUDA / CPU)
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS (Metal) device")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("Using CUDA device")
        return "cuda"
    else:
        logger.info("Using CPU device")
        return "cpu"

# RecurrentPPO in sb3-contrib currently does not support MPS well.
# Force CPU for training stability. MPS can be used for inference later.
DEVICE = "cpu"
logger.info("Training device: %s (RecurrentPPO forced to CPU for stability)", DEVICE)

# ---------------------------------------------------------------------------
# Per-Symbol Configuration
# ---------------------------------------------------------------------------
SYMBOL_CONFIGS = {
    "BTCUSDT": {
        "lstm_hidden": 48,
        "post_lstm": [32],
        "learning_rate": 3e-5,
        "total_steps": 500_000,
        "obs_noise_std": 0.01,
        "n_seeds": 3,
    },
    "ETHUSDT": {
        "lstm_hidden": 48,
        "post_lstm": [32],
        "learning_rate": 3e-5,
        "total_steps": 500_000,
        "obs_noise_std": 0.01,
        "n_seeds": 3,
    },
    "SOLUSDT": {
        "lstm_hidden": 64,
        "post_lstm": [48],
        "learning_rate": 2e-5,
        "total_steps": 500_000,
        "obs_noise_std": 0.02,
        "n_seeds": 3,
    },
    "XRPUSDT": {
        "lstm_hidden": 48,
        "post_lstm": [32],
        "learning_rate": 3e-5,
        "total_steps": 500_000,
        "obs_noise_std": 0.01,
        "n_seeds": 3,
    },
}

# Curriculum step allocation (must sum to total_steps)
CURRICULUM_STEPS = {
    "stage1": 100_000,   # Trending only (ADX > 30)
    "stage2": 200_000,   # Add ranging (ADX > 15)
    "stage3": 200_000,   # Full data
}

# ---------------------------------------------------------------------------
# Graceful interrupt handler
# ---------------------------------------------------------------------------
_interrupted = False

def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    logger.warning("\n>>> KeyboardInterrupt caught. Saving progress and exiting after current step...")

sig_module.signal(sig_module.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------
def linear_schedule(initial_lr: float):
    """Linear decay to 10% of initial LR."""
    def schedule(progress_remaining: float) -> float:
        return initial_lr * (0.1 + 0.9 * progress_remaining)
    return schedule

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(symbol: str, data_dir: str = "data/historical") -> pd.DataFrame:
    """Load 15m OHLCV data for a symbol. Returns DataFrame with DatetimeIndex."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Try multiple file patterns
    patterns = [
        f"{symbol}_15m_*.csv",
        f"{symbol}_15m.csv",
    ]
    csv_file = None
    for pattern in patterns:
        matches = sorted(data_path.glob(pattern))
        if matches:
            csv_file = matches[-1]  # most recent
            break

    if csv_file is None:
        raise FileNotFoundError(
            f"No 15m data found for {symbol} in {data_path}. "
            f"Run: python train_model.py --symbol {symbol} --download-data"
        )

    logger.info("Loading data from %s", csv_file)
    df = pd.read_csv(csv_file)

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Handle timestamp column
    time_col = None
    for col in ("timestamp", "open_time", "datetime", "date"):
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    else:
        # Try converting existing index
        df.index = pd.to_datetime(df.index)

    # Ensure required columns
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        df[col] = df[col].astype(float)

    df = df.sort_index().drop_duplicates()
    logger.info("Loaded %d bars from %s to %s", len(df), df.index[0], df.index[-1])
    return df


def download_data(symbol: str):
    """Download 3 years of 15m data from Binance."""
    from download_historical_data import download_asset
    output_dir = Path("data/historical")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading 3 years of 15m data for %s ...", symbol)
    download_asset(symbol, years=3, output_dir=output_dir, interval="15m")
    logger.info("Download complete for %s", symbol)

# ---------------------------------------------------------------------------
# ADX computation (for curriculum filtering)
# ---------------------------------------------------------------------------
def compute_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX series for a DataFrame."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0

    alpha = 1.0 / period
    atr = pd.Series(tr).ewm(alpha=alpha, min_periods=period).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(alpha=alpha, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(alpha=alpha, min_periods=period).mean().values / (atr + 1e-10)
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=alpha, min_periods=period).mean()
    return pd.Series(adx.values, index=df.index)

# ---------------------------------------------------------------------------
# Walk-forward fold generation
# ---------------------------------------------------------------------------
def generate_folds(
    df: pd.DataFrame,
    train_months: int = 6,
    val_months: int = 2,
    test_months: int = 2,
    slide_months: int = 2,
    embargo_bars: int = 192,
) -> List[Dict[str, Any]]:
    """
    Generate walk-forward folds with strict splits and 48h embargo.

    Returns list of dicts with 'train', 'val', 'test' DataFrames.
    """
    folds = []
    start = df.index[0]
    end = df.index[-1]

    fold_idx = 0
    current_start = start

    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        # Embargo 1: 192 bars = 192 * 15min = 48 hours
        embargo_td = pd.Timedelta(minutes=15 * embargo_bars)
        val_start = train_end + embargo_td
        val_end = val_start + pd.DateOffset(months=val_months)
        # Embargo 2
        test_start = val_end + embargo_td
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        train_df = df.loc[current_start:train_end].copy()
        val_df = df.loc[val_start:val_end].copy()
        test_df = df.loc[test_start:test_end].copy()

        # Require minimum data
        if len(train_df) < 500 or len(val_df) < 100 or len(test_df) < 100:
            logger.warning("Fold %d has insufficient data, skipping", fold_idx)
            current_start += pd.DateOffset(months=slide_months)
            fold_idx += 1
            continue

        folds.append({
            "fold_idx": fold_idx,
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "train_range": f"{train_df.index[0].date()} to {train_df.index[-1].date()}",
            "val_range": f"{val_df.index[0].date()} to {val_df.index[-1].date()}",
            "test_range": f"{test_df.index[0].date()} to {test_df.index[-1].date()}",
        })

        logger.info(
            "Fold %02d: train=%d bars (%s), val=%d bars (%s), test=%d bars (%s)",
            fold_idx, len(train_df), folds[-1]["train_range"],
            len(val_df), folds[-1]["val_range"],
            len(test_df), folds[-1]["test_range"],
        )

        current_start += pd.DateOffset(months=slide_months)
        fold_idx += 1

    logger.info("Generated %d folds", len(folds))
    return folds

# ---------------------------------------------------------------------------
# Curriculum data filtering
# ---------------------------------------------------------------------------
def filter_by_adx(df: pd.DataFrame, min_adx: float) -> pd.DataFrame:
    """Return only bars where ADX >= min_adx."""
    adx = compute_adx_series(df)
    mask = adx >= min_adx
    filtered = df[mask].copy()
    if len(filtered) < 200:
        logger.warning(
            "ADX filter (>= %.0f) left only %d bars. Using full data instead.",
            min_adx, len(filtered)
        )
        return df.copy()
    return filtered

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(
    df: pd.DataFrame,
    training_mode: bool = True,
    symbol: str = "BTCUSDT",
) -> StructureFilterEnv:
    """Create a StructureFilterEnv from a DataFrame."""
    env = StructureFilterEnv(
        df_15m=df,
        training_mode=training_mode,
        symbol=symbol,
    )
    return env


def make_vec_env(
    df: pd.DataFrame,
    training_mode: bool = True,
    symbol: str = "BTCUSDT",
    normalize: bool = True,
) -> Tuple[Any, Optional[VecNormalize]]:
    """Create a VecNormalize-wrapped DummyVecEnv."""
    def _make():
        return Monitor(make_env(df, training_mode=training_mode, symbol=symbol))

    vec_env = DummyVecEnv([_make])

    if normalize:
        vec_norm = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        return vec_norm, vec_norm
    return vec_env, None

# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------
class EarlyStoppingCallback(BaseCallback):
    """
    Stop training if val Sharpe degrades for `patience` consecutive evals.
    Saves the best model checkpoint.
    """

    def __init__(
        self,
        val_df: pd.DataFrame,
        symbol: str,
        eval_freq: int = 50_000,
        patience: int = 3,
        save_dir: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.val_df = val_df
        self.symbol = symbol
        self.eval_freq = eval_freq
        self.patience = patience
        self.save_dir = save_dir

        self.best_val_sharpe = -np.inf
        self.no_improve_count = 0
        self.best_model_path: Optional[str] = None
        self.eval_history: List[Dict] = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if _interrupted:
            return False

        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        # Evaluate on validation set
        val_metrics = evaluate_model(
            self.model, self.val_df, self.symbol, training_norm=self.training_env
        )
        val_sharpe = val_metrics.get("sharpe", 0.0)
        val_wr = val_metrics.get("win_rate", 0.0)

        self.eval_history.append({
            "step": self.num_timesteps,
            "val_sharpe": val_sharpe,
            "val_win_rate": val_wr,
            "val_trades": val_metrics.get("total_trades", 0),
        })

        if self.verbose:
            logger.info(
                "  [Eval @ %dk] val_sharpe=%.3f val_wr=%.1f%% trades=%d",
                self.num_timesteps // 1000,
                val_sharpe, val_wr * 100,
                val_metrics.get("total_trades", 0),
            )

        if val_sharpe > self.best_val_sharpe:
            self.best_val_sharpe = val_sharpe
            self.no_improve_count = 0
            # Save best model
            if self.save_dir:
                self.best_model_path = os.path.join(self.save_dir, "best_model")
                self.model.save(self.best_model_path)
                # Save VecNormalize stats
                if hasattr(self.training_env, "save"):
                    self.training_env.save(
                        os.path.join(self.save_dir, "best_vecnormalize.pkl")
                    )
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.patience:
                logger.info(
                    "  Early stopping triggered: %d evals without improvement "
                    "(best val_sharpe=%.3f)",
                    self.no_improve_count, self.best_val_sharpe,
                )
                return False

        return True

# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model,
    df: pd.DataFrame,
    symbol: str,
    training_norm=None,
    n_episodes: int = 1,
) -> Dict[str, float]:
    """
    Run the model on a dataset and return metrics.
    Uses frozen VecNormalize stats from training.
    """
    env = make_env(df, training_mode=False, symbol=symbol)

    all_trades = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False

        while not done:
            # Normalize observation if we have training stats
            if training_norm is not None and hasattr(training_norm, "normalize_obs"):
                obs_norm = training_norm.normalize_obs(obs.reshape(1, -1))
                obs_input = obs_norm.flatten()
            else:
                obs_input = obs

            action, lstm_states = model.predict(
                obs_input,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            episode_starts = np.zeros((1,), dtype=bool)

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        all_trades.extend(env._trade_returns)

    # Compute metrics
    metrics = env.get_episode_metrics()
    return metrics

# ---------------------------------------------------------------------------
# S1 Baseline (structure-only, no model filtering)
# ---------------------------------------------------------------------------
def compute_s1_baseline(df: pd.DataFrame, symbol: str) -> Dict[str, float]:
    """
    Run structure-only strategy: accept ALL BOS/CHOCH signals.
    Returns performance metrics.
    """
    env = make_env(df, training_mode=False, symbol=symbol)
    obs, _ = env.reset()
    done = False

    while not done:
        # Always ACCEPT (action=1)
        obs, reward, terminated, truncated, _ = env.step(1)
        done = terminated or truncated

    metrics = env.get_episode_metrics()
    return metrics

# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------
def validate_model(
    aggregate_metrics: Dict[str, float],
    s1_baseline: Dict[str, float],
) -> Tuple[bool, Dict[str, bool]]:
    """Check if model passes deployment gates."""
    checks = {
        "min_trades": aggregate_metrics.get("total_trades", 0) >= 200,
        "win_rate_above_40": aggregate_metrics.get("win_rate", 0) >= 0.40,
        "beats_s1_wr": aggregate_metrics.get("win_rate", 0) > s1_baseline.get("win_rate", 1.0),
        "beats_s1_sharpe": aggregate_metrics.get("sharpe", -99) > s1_baseline.get("sharpe", 99),
        "oos_sharpe_positive": aggregate_metrics.get("sharpe", -1) > 0.0,
        "max_drawdown_under_20": aggregate_metrics.get("max_drawdown_pct", 100) < 20.0,
    }
    passed = all(checks.values())
    return passed, checks

# ---------------------------------------------------------------------------
# Create RecurrentPPO model
# ---------------------------------------------------------------------------
def create_model(
    env,
    config: Dict[str, Any],
    seed: int,
) -> RecurrentPPO:
    """Create a fresh RecurrentPPO model."""
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=linear_schedule(config["learning_rate"]),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        max_grad_norm=0.3,
        policy_kwargs={
            "lstm_hidden_size": config["lstm_hidden"],
            "n_lstm_layers": 1,
            "net_arch": config["post_lstm"],
        },
        seed=seed,
        device=DEVICE,
        verbose=0,
    )
    return model

# ---------------------------------------------------------------------------
# Train one seed on one fold (with curriculum)
# ---------------------------------------------------------------------------
def train_one_seed(
    fold: Dict,
    config: Dict[str, Any],
    symbol: str,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Train one seed through 3 curriculum stages.
    Returns dict with model path, metrics, eval history.
    """
    global _interrupted

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_df = fold["train"]
    val_df = fold["val"]
    test_df = fold["test"]

    # Compute ADX for curriculum filtering
    adx_series = compute_adx_series(train_df)

    result = {
        "seed": seed,
        "stages": [],
        "val_sharpe": -np.inf,
        "test_metrics": {},
        "model_path": None,
        "vecnorm_path": None,
    }

    # Curriculum stages
    stages = [
        ("stage1", 30.0, CURRICULUM_STEPS["stage1"]),   # ADX > 30 (trending)
        ("stage2", 15.0, CURRICULUM_STEPS["stage2"]),   # ADX > 15 (add ranging)
        ("stage3", 0.0,  CURRICULUM_STEPS["stage3"]),   # All data
    ]

    model = None
    vec_norm = None

    for stage_name, min_adx, stage_steps in stages:
        if _interrupted:
            logger.warning("Interrupted at %s", stage_name)
            break

        logger.info("    [Seed %d] %s: ADX >= %.0f, %dk steps",
                     seed, stage_name, min_adx, stage_steps // 1000)

        # Filter training data by ADX regime
        if min_adx > 0:
            stage_df = filter_by_adx(train_df, min_adx)
        else:
            stage_df = train_df.copy()

        logger.info("    [Seed %d] %s: %d training bars", seed, stage_name, len(stage_df))

        # Create environment
        vec_env, vec_norm_new = make_vec_env(
            stage_df, training_mode=True, symbol=symbol, normalize=True
        )

        if model is None:
            # First stage: create new model
            model = create_model(vec_env, config, seed)
            vec_norm = vec_norm_new
        else:
            # Subsequent stages: set new environment
            model.set_env(vec_env)
            vec_norm = vec_norm_new

        # Early stopping callback
        es_callback = EarlyStoppingCallback(
            val_df=val_df,
            symbol=symbol,
            eval_freq=50_000,
            patience=3,
            save_dir=str(seed_dir),
        )

        # Train
        t0 = time.time()
        try:
            model.learn(
                total_timesteps=stage_steps,
                callback=es_callback,
                reset_num_timesteps=False,
                progress_bar=True,
            )
        except KeyboardInterrupt:
            _interrupted = True
            logger.warning("Training interrupted at %s", stage_name)

        elapsed = time.time() - t0

        result["stages"].append({
            "name": stage_name,
            "steps": model.num_timesteps,
            "elapsed_sec": elapsed,
            "best_val_sharpe": es_callback.best_val_sharpe,
            "eval_history": es_callback.eval_history,
        })

        logger.info(
            "    [Seed %d] %s complete: %dk steps, %.1f min, best_val_sharpe=%.3f",
            seed, stage_name, model.num_timesteps // 1000,
            elapsed / 60, es_callback.best_val_sharpe,
        )

        # Clean up stage env
        vec_env.close()
        gc.collect()

    if model is None:
        return result

    # Load best model if early stopping saved one
    best_model_file = seed_dir / "best_model.zip"
    if best_model_file.exists():
        logger.info("    [Seed %d] Loading best checkpoint from early stopping", seed)
        model = RecurrentPPO.load(str(best_model_file), device=DEVICE)

    # Save final model
    final_model_path = seed_dir / "model.zip"
    model.save(str(final_model_path))
    result["model_path"] = str(final_model_path)

    # Save VecNormalize stats
    if vec_norm is not None:
        vecnorm_path = seed_dir / "vecnormalize.pkl"
        vec_norm.save(str(vecnorm_path))
        result["vecnorm_path"] = str(vecnorm_path)

    # Evaluate on val set (for seed selection)
    logger.info("    [Seed %d] Evaluating on validation set...", seed)
    val_metrics = evaluate_model(model, val_df, symbol, training_norm=vec_norm)
    result["val_sharpe"] = val_metrics.get("sharpe", 0.0)
    result["val_metrics"] = val_metrics

    # Evaluate on test set (frozen normalization)
    logger.info("    [Seed %d] Evaluating on test set...", seed)
    test_metrics = evaluate_model(model, test_df, symbol, training_norm=vec_norm)
    result["test_metrics"] = test_metrics

    logger.info(
        "    [Seed %d] val_sharpe=%.3f val_wr=%.1f%% | test_sharpe=%.3f test_wr=%.1f%%",
        seed,
        val_metrics.get("sharpe", 0), val_metrics.get("win_rate", 0) * 100,
        test_metrics.get("sharpe", 0), test_metrics.get("win_rate", 0) * 100,
    )

    del model
    gc.collect()
    return result

# ---------------------------------------------------------------------------
# Train one fold (all seeds)
# ---------------------------------------------------------------------------
def train_one_fold(
    fold: Dict,
    config: Dict[str, Any],
    symbol: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Train all seeds for one fold, select best seed."""
    fold_idx = fold["fold_idx"]
    fold_dir = output_dir / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    logger.info("\n%s", sep)
    logger.info(
        "FOLD %02d | train: %s | val: %s | test: %s",
        fold_idx, fold["train_range"], fold["val_range"], fold["test_range"],
    )
    logger.info("%s", sep)

    n_seeds = config.get("n_seeds", 3)
    seed_results = []

    for seed in range(n_seeds):
        if _interrupted:
            break
        logger.info("  --- Seed %d/%d ---", seed + 1, n_seeds)
        result = train_one_seed(fold, config, symbol, seed, fold_dir)
        seed_results.append(result)

    if not seed_results:
        return {"fold_idx": fold_idx, "seeds": [], "best_seed": -1}

    # Select best seed by val Sharpe
    best_idx = max(range(len(seed_results)), key=lambda i: seed_results[i]["val_sharpe"])
    best = seed_results[best_idx]

    logger.info(
        "  Fold %02d best seed: %d (val_sharpe=%.3f)",
        fold_idx, best["seed"], best["val_sharpe"],
    )

    # Copy best model to fold root
    if best["model_path"] and Path(best["model_path"]).exists():
        shutil.copy2(best["model_path"], fold_dir / "best_model.zip")
    if best.get("vecnorm_path") and Path(best["vecnorm_path"]).exists():
        shutil.copy2(best["vecnorm_path"], fold_dir / "best_vecnormalize.pkl")

    # Save fold metrics
    fold_metrics = {
        "fold_idx": fold_idx,
        "train_range": fold["train_range"],
        "val_range": fold["val_range"],
        "test_range": fold["test_range"],
        "best_seed": best["seed"],
        "best_val_sharpe": best["val_sharpe"],
        "val_metrics": best.get("val_metrics", {}),
        "test_metrics": best.get("test_metrics", {}),
        "all_seeds": [
            {
                "seed": r["seed"],
                "val_sharpe": r["val_sharpe"],
                "stages": r["stages"],
            }
            for r in seed_results
        ],
    }
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(fold_metrics, f, indent=2, default=str)

    return fold_metrics

# ---------------------------------------------------------------------------
# Build ensemble from top folds
# ---------------------------------------------------------------------------
def build_ensemble(
    fold_results: List[Dict],
    output_dir: Path,
    top_k: int = 3,
) -> Dict:
    """Select top-K folds by val Sharpe and copy models to ensemble dir."""
    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Sort by val Sharpe
    valid_folds = [f for f in fold_results if f.get("best_val_sharpe", -np.inf) > -np.inf]
    valid_folds.sort(key=lambda f: f.get("best_val_sharpe", -np.inf), reverse=True)
    top_folds = valid_folds[:top_k]

    if not top_folds:
        logger.warning("No valid folds for ensemble")
        return {"top_folds": [], "models": []}

    top_info = []
    for i, fold in enumerate(top_folds):
        fold_idx = fold["fold_idx"]
        src_model = output_dir / f"fold_{fold_idx:02d}" / "best_model.zip"
        src_norm = output_dir / f"fold_{fold_idx:02d}" / "best_vecnormalize.pkl"

        dst_model = ensemble_dir / f"final_model_{i}.zip"
        dst_norm = ensemble_dir / f"final_vecnorm_{i}.pkl"

        if src_model.exists():
            shutil.copy2(src_model, dst_model)
        if src_norm.exists():
            shutil.copy2(src_norm, dst_norm)

        top_info.append({
            "rank": i,
            "fold_idx": fold_idx,
            "val_sharpe": fold.get("best_val_sharpe", 0),
            "test_metrics": fold.get("test_metrics", {}),
            "model_file": str(dst_model),
            "vecnorm_file": str(dst_norm),
        })

    # Save manifest
    manifest = {"top_folds": top_info}
    with open(ensemble_dir / "top3_models.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest

# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def train_symbol(symbol: str, max_folds: Optional[int] = None):
    """Full training pipeline for one symbol."""
    global _interrupted

    if symbol not in SYMBOL_CONFIGS:
        raise ValueError(f"Unknown symbol: {symbol}. Valid: {list(SYMBOL_CONFIGS.keys())}")

    config = SYMBOL_CONFIGS[symbol]
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING: %s", symbol)
    logger.info("Config: %s", json.dumps(config, indent=2))
    logger.info("=" * 70)

    # Output directory
    output_dir = REPO / "data" / "models" / "v3" / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(symbol)

    # Generate folds
    folds = generate_folds(df)
    if max_folds is not None:
        folds = folds[:max_folds]
        logger.info("Limiting to %d folds", max_folds)

    if not folds:
        logger.error("No valid folds generated. Need at least 10 months of data.")
        return

    # Training log
    training_log = {
        "symbol": symbol,
        "config": config,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "n_folds": len(folds),
        "data_range": f"{df.index[0]} to {df.index[-1]}",
        "total_bars": len(df),
        "folds": [],
    }

    # Train each fold
    fold_results = []
    all_test_metrics = []

    for fold in folds:
        if _interrupted:
            logger.warning("Training interrupted, saving progress...")
            break

        fold_result = train_one_fold(fold, config, symbol, output_dir)
        fold_results.append(fold_result)
        training_log["folds"].append(fold_result)

        if fold_result.get("test_metrics"):
            all_test_metrics.append(fold_result["test_metrics"])

        # Save progress after each fold
        with open(output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2, default=str)

    # Aggregate test metrics across folds
    if all_test_metrics:
        aggregate = aggregate_metrics(all_test_metrics)
    else:
        aggregate = {}

    # Compute S1 baseline on test folds
    logger.info("\nComputing S1 baseline (structure-only) on test folds...")
    s1_results = []
    for fold in folds:
        if _interrupted:
            break
        s1 = compute_s1_baseline(fold["test"], symbol)
        s1_results.append(s1)

    if s1_results:
        s1_aggregate = aggregate_metrics(s1_results)
    else:
        s1_aggregate = {}

    # Save S1 baseline
    with open(output_dir / "s1_baseline.json", "w") as f:
        json.dump({"per_fold": s1_results, "aggregate": s1_aggregate}, f, indent=2, default=str)

    # Validation gates
    passed, checks = validate_model(aggregate, s1_aggregate)
    validation_report = {
        "passed": passed,
        "checks": checks,
        "aggregate_metrics": aggregate,
        "s1_baseline": s1_aggregate,
    }
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2, default=str)

    # Build ensemble from top folds
    ensemble = build_ensemble(fold_results, output_dir)

    # Finalize training log
    training_log["end_time"] = datetime.now(timezone.utc).isoformat()
    training_log["aggregate_metrics"] = aggregate
    training_log["s1_baseline"] = s1_aggregate
    training_log["validation"] = validation_report
    training_log["ensemble"] = ensemble
    training_log["interrupted"] = _interrupted

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2, default=str)

    # Print summary
    print_summary(symbol, fold_results, aggregate, s1_aggregate, passed, checks)

    return validation_report


def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
    """Aggregate metrics across multiple folds."""
    if not metrics_list:
        return {}

    keys = ["total_trades", "win_rate", "sharpe", "total_return_pct",
            "max_drawdown_pct", "profit_factor"]

    result = {}
    for key in keys:
        values = [m.get(key, 0) for m in metrics_list if key in m]
        if values:
            if key == "total_trades":
                result[key] = sum(values)
            elif key == "max_drawdown_pct":
                result[key] = max(values)
            else:
                result[key] = float(np.mean(values))

    return result


def print_summary(
    symbol: str,
    fold_results: List[Dict],
    aggregate: Dict,
    s1_baseline: Dict,
    passed: bool,
    checks: Dict,
):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print(f"  TRAINING SUMMARY: {symbol}")
    print("=" * 80)

    # Per-fold results
    print(f"\n{'Fold':>6} {'Val Sharpe':>12} {'Test Sharpe':>12} {'Test WR%':>10} {'Test Trades':>12}")
    print("-" * 60)
    for fr in fold_results:
        fold_idx = fr.get("fold_idx", "?")
        val_s = fr.get("best_val_sharpe", 0)
        tm = fr.get("test_metrics", {})
        test_s = tm.get("sharpe", 0)
        test_wr = tm.get("win_rate", 0) * 100
        test_t = tm.get("total_trades", 0)
        print(f"  {fold_idx:>4}   {val_s:>10.3f}   {test_s:>10.3f}   {test_wr:>8.1f}%   {test_t:>10}")

    # Aggregate
    print(f"\n{'AGGREGATE':>10}")
    print("-" * 40)
    for k, v in aggregate.items():
        if isinstance(v, float):
            print(f"  {k:>25}: {v:.4f}")
        else:
            print(f"  {k:>25}: {v}")

    # S1 Baseline comparison
    print(f"\n{'S1 BASELINE':>12}")
    print("-" * 40)
    for k, v in s1_baseline.items():
        if isinstance(v, float):
            print(f"  {k:>25}: {v:.4f}")
        else:
            print(f"  {k:>25}: {v}")

    # Validation gates
    print(f"\n{'VALIDATION GATES':>18}")
    print("-" * 40)
    for check_name, check_passed in checks.items():
        status = "PASS" if check_passed else "FAIL"
        print(f"  {check_name:>30}: {status}")

    overall = "PASS" if passed else "FAIL"
    print(f"\n  {'OVERALL':>30}: {overall}")
    print("=" * 80)

    if passed:
        print(f"\n  Model PASSED all gates. Ready for deployment.")
        print(f"  Models saved to: data/models/v3/{symbol}/")
    else:
        print(f"\n  Model FAILED validation. Review metrics and consider retraining.")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train Structure-Gated Filter model for DRL trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py --symbol BTCUSDT
  python train_model.py --symbol BTCUSDT --download-data
  python train_model.py --symbol SOLUSDT --folds 3
  python train_model.py --symbol ETHUSDT --folds 5 --seeds 2
        """,
    )
    parser.add_argument(
        "--symbol", required=True,
        choices=list(SYMBOL_CONFIGS.keys()),
        help="Trading pair to train",
    )
    parser.add_argument(
        "--download-data", action="store_true",
        help="Download 3 years of 15m data before training",
    )
    parser.add_argument(
        "--folds", type=int, default=None,
        help="Limit number of folds (default: all available, ~14)",
    )
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Override number of seeds per fold",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/historical",
        help="Directory containing historical CSV files",
    )

    args = parser.parse_args()

    # Download data if requested
    if args.download_data:
        download_data(args.symbol)

    # Override seeds if specified
    if args.seeds is not None:
        SYMBOL_CONFIGS[args.symbol]["n_seeds"] = args.seeds

    # Print system info
    logger.info("System info:")
    logger.info("  Python: %s", sys.version.split()[0])
    logger.info("  PyTorch: %s", torch.__version__)
    logger.info("  MPS available: %s", torch.backends.mps.is_available())
    logger.info("  CUDA available: %s", torch.cuda.is_available())
    logger.info("  Training device: %s", DEVICE)

    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info("  RAM: %.1f GB total, %.1f GB available",
                     mem.total / 1e9, mem.available / 1e9)
    except ImportError:
        pass

    # Start training
    t0 = time.time()
    try:
        result = train_symbol(args.symbol, max_folds=args.folds)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception:
        logger.error("Training failed:\n%s", traceback.format_exc())
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("Total training time: %.1f hours (%.0f minutes)",
                elapsed / 3600, elapsed / 60)

    if _interrupted:
        logger.warning("Training was interrupted but progress was saved.")
        sys.exit(130)


if __name__ == "__main__":
    main()
