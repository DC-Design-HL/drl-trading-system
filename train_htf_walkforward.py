#!/usr/bin/env python3
"""
HTF Walk-Forward Validation Script

Walk-forward cross-validation for the HTF multi-timeframe trading system.
Adapted from train_walkforward_v2.py methodology, but uses HTFTradingEnv,
HTFFeatureEngine (via env), and HTFTradingAgent with curriculum training.

Key design choices:
  - Strict chronological splits — no shuffling, ever
  - Walk-forward windows: 6mo train / 2mo val / 2mo test (shorter windows for 2yr data)
  - Val is carved from end of train window — used for anti-overfitting detection only
  - Curriculum training per fold: Phase 1 (200K steps) + Phase 2 (400K steps)
  - OOS metrics per fold: Sharpe, return, max drawdown, win rate, trades
  - Anti-overfitting: compare val vs test Sharpe ratio (flag if >3x gap)
  - All 4 TFs (15m/1h/4h/1d) split consistently using open_time cutoffs

Data requirements:
  - BTCUSDT 15M CSV at data/historical/BTCUSDT_15m.csv (~70K bars, 2 years)
  - Columns: open_time, open, high, low, close, volume (standard OHLCV)

Usage:
    python train_htf_walkforward.py
    python train_htf_walkforward.py --data-path data/historical/BTCUSDT_15m.csv
    python train_htf_walkforward.py --phase1-steps 100000 --phase2-steps 200000
    python train_htf_walkforward.py --max-folds 2
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("htf_walkforward")


# ---------------------------------------------------------------------------
# JSON encoder for numpy types
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
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
# HTF data alignment (resampling) — matches train_htf.py
# ---------------------------------------------------------------------------

class HTFDataAligner:
    """
    Resample 15-minute base OHLCV data to higher timeframes.
    Matches the implementation in train_htf.py exactly.
    """

    @staticmethod
    def resample(df_15m: pd.DataFrame, rule: str) -> pd.DataFrame:
        df = df_15m.set_index("open_time").sort_index()
        resampled = df.resample(rule, label="left", closed="left").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        resampled = resampled.dropna(subset=["close"])
        resampled = resampled.reset_index().rename(columns={"open_time": "open_time"})
        return resampled

    @staticmethod
    def align_to_15m(
        df_15m: pd.DataFrame,
        df_htf: pd.DataFrame,
        suffix: str,
    ) -> pd.DataFrame:
        htf_renamed = df_htf.rename(
            columns={c: f"{c}{suffix}" for c in df_htf.columns if c != "open_time"}
        )
        merged = pd.merge_asof(
            df_15m.sort_values("open_time"),
            htf_renamed.sort_values("open_time"),
            on="open_time",
            direction="backward",
        )
        return merged.reset_index(drop=True)


def build_htf_dataframes(
    df_15m: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all 4 TF DataFrames from base 15M data. Returns (df_merged, df_1h, df_4h, df_1d)."""
    aligner = HTFDataAligner()

    df_1h = aligner.resample(df_15m, "1h")
    df_4h = aligner.resample(df_15m, "4h")
    df_1d = aligner.resample(df_15m, "1D")

    df_merged = aligner.align_to_15m(df_15m, df_1h, "_1h")
    df_merged = aligner.align_to_15m(df_merged, df_4h, "_4h")
    df_merged = aligner.align_to_15m(df_merged, df_1d, "_1d")

    logger.info(
        "Timeframes built: 15M=%d rows, 1H=%d rows, 4H=%d rows, 1D=%d rows",
        len(df_15m), len(df_1h), len(df_4h), len(df_1d),
    )

    return df_merged, df_1h, df_4h, df_1d


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_15m_csv(data_path: str) -> pd.DataFrame:
    """
    Load BTCUSDT 15M OHLCV data from a local CSV file.

    The CSV must have an open_time column (datetime or parseable string).
    Timezone is set to UTC if not already present.
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(
            f"15M data not found at {data_path}. "
            "Run: python download_historical_data.py --symbol BTCUSDT --interval 15m"
        )

    logger.info("Loading 15M data from %s", p)
    df = pd.read_csv(p, parse_dates=["open_time"])

    # Ensure UTC-aware datetime
    if df["open_time"].dt.tz is None:
        df["open_time"] = df["open_time"].dt.tz_localize("UTC")
    else:
        df["open_time"] = df["open_time"].dt.tz_convert("UTC")

    # Ensure numeric OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("open_time").reset_index(drop=True)
    logger.info(
        "Loaded %d rows: %s → %s (%.1f months)",
        len(df),
        df["open_time"].iloc[0].strftime("%Y-%m-%d"),
        df["open_time"].iloc[-1].strftime("%Y-%m-%d"),
        len(df) / (4 * 24 * 30),
    )
    return df


# ---------------------------------------------------------------------------
# Environment factory — matches train_htf.py _create_env
# ---------------------------------------------------------------------------

def _create_env(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    initial_balance: float = 10_000.0,
    training: bool = True,
    position_size: float = 0.25,
):
    """
    Instantiate an HTFTradingEnv with the four timeframe DataFrames.
    The env's _ensure_datetime_index handles both DatetimeIndex and open_time columns.
    """
    from src.env.htf_env import HTFTradingEnv  # type: ignore

    env = HTFTradingEnv(
        df_15m=df_15m,
        df_1h=df_1h,
        df_4h=df_4h,
        df_1d=df_1d,
        initial_balance=initial_balance,
        position_size=position_size,
        training_mode=training,
    )
    return env


# ---------------------------------------------------------------------------
# Evaluation — matches train_htf.py evaluate_agent
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env, n_episodes: int = 3) -> Dict:
    """
    Run the trained agent on a held-out env and collect episode metrics.

    Returns a dict with mean_ prefixed aggregate metrics (Sharpe, return, etc.).
    """
    all_metrics: List[Dict] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action, _, _ = agent.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)

        if hasattr(env, "get_episode_metrics"):
            ep_metrics = env.get_episode_metrics()
            all_metrics.append(ep_metrics)
        else:
            logger.warning("env.get_episode_metrics() not available")

    if not all_metrics:
        return {}

    agg: Dict = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
        if vals:
            agg[f"mean_{key}"] = float(np.mean(vals))

    return agg


def _extract_metrics(agg: Dict) -> Dict:
    """Extract key scalar metrics from an evaluate_agent result dict."""
    return {
        "sharpe_ratio": agg.get("mean_sharpe_ratio", 0.0),
        "total_return_pct": agg.get("mean_total_return_pct", 0.0),
        "max_drawdown_pct": agg.get("mean_max_drawdown_pct", 0.0),
        "win_rate": agg.get("mean_win_rate", 0.0),
        "total_trades": agg.get("mean_total_trades", 0.0),
        "profit_factor": agg.get("mean_profit_factor", 0.0),
    }


# ---------------------------------------------------------------------------
# Walk-forward window creation — adapted for HTF (open_time column)
# ---------------------------------------------------------------------------

def create_walk_forward_windows(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    train_months: int = 6,
    val_months: int = 2,
    test_months: int = 2,
    slide_months: int = 2,
) -> List[Dict]:
    """
    Create walk-forward windows over the HTF data.

    Within each fold:
      train_*: first (train_months - val_months) months (used for weight updates)
      val_*:   last val_months of the train window   (used for early stopping only)
      test_*:  next test_months after train_end        (strictly out-of-sample)

    All 4 TFs are split consistently using open_time timestamp cutoffs on the
    15M frame, matching the approach in train_htf.py line 554-565.

    Returns a list of window dicts, each containing 12 DataFrames (3 splits × 4 TFs)
    plus metadata.
    """
    ts = df_15m["open_time"]
    start = ts.iloc[0]
    end = ts.iloc[-1]

    train_delta = pd.DateOffset(months=train_months)
    val_delta = pd.DateOffset(months=val_months)
    test_delta = pd.DateOffset(months=test_months)
    slide_delta = pd.DateOffset(months=slide_months)

    def _split_df(df: pd.DataFrame, t_start, t_end) -> pd.DataFrame:
        """Return rows where open_time is in [t_start, t_end)."""
        mask = (df["open_time"] >= t_start) & (df["open_time"] < t_end)
        return df[mask].reset_index(drop=True)

    windows = []
    fold = 0
    current = start

    while True:
        train_end = current + train_delta
        test_end = train_end + test_delta

        if test_end > end:
            break

        pure_train_end = train_end - val_delta

        # Apply same cutoffs to all 4 TFs
        train_15m = _split_df(df_15m, current, pure_train_end)
        val_15m   = _split_df(df_15m, pure_train_end, train_end)
        test_15m  = _split_df(df_15m, train_end, test_end)

        train_1h = _split_df(df_1h, current, pure_train_end)
        val_1h   = _split_df(df_1h, pure_train_end, train_end)
        test_1h  = _split_df(df_1h, train_end, test_end)

        train_4h = _split_df(df_4h, current, pure_train_end)
        val_4h   = _split_df(df_4h, pure_train_end, train_end)
        test_4h  = _split_df(df_4h, train_end, test_end)

        train_1d = _split_df(df_1d, current, pure_train_end)
        val_1d   = _split_df(df_1d, pure_train_end, train_end)
        test_1d  = _split_df(df_1d, train_end, test_end)

        # Minimum bars sanity checks (15M has ~2880 bars/month)
        min_train_15m = 500
        min_val_15m   = 200
        min_test_15m  = 200
        min_htf_bars  = 10   # 1D may have few bars — just ensure non-empty

        ok = (
            len(train_15m) >= min_train_15m
            and len(val_15m) >= min_val_15m
            and len(test_15m) >= min_test_15m
            and len(train_1h) >= min_htf_bars
            and len(val_1h) >= min_htf_bars
            and len(test_1h) >= min_htf_bars
            and len(train_4h) >= min_htf_bars
            and len(val_4h) >= min_htf_bars
            and len(test_4h) >= min_htf_bars
            and len(train_1d) >= min_htf_bars
            and len(val_1d) >= min_htf_bars
            and len(test_1d) >= min_htf_bars
        )

        if ok:
            windows.append({
                "fold": fold,
                "train_start": train_15m["open_time"].iloc[0].isoformat(),
                "train_end":   train_15m["open_time"].iloc[-1].isoformat(),
                "val_start":   val_15m["open_time"].iloc[0].isoformat(),
                "val_end":     val_15m["open_time"].iloc[-1].isoformat(),
                "test_start":  test_15m["open_time"].iloc[0].isoformat(),
                "test_end":    test_15m["open_time"].iloc[-1].isoformat(),
                "train_bars":  len(train_15m),
                "val_bars":    len(val_15m),
                "test_bars":   len(test_15m),
                # 15M splits
                "train_15m": train_15m,
                "val_15m":   val_15m,
                "test_15m":  test_15m,
                # 1H splits
                "train_1h":  train_1h,
                "val_1h":    val_1h,
                "test_1h":   test_1h,
                # 4H splits
                "train_4h":  train_4h,
                "val_4h":    val_4h,
                "test_4h":   test_4h,
                # 1D splits
                "train_1d":  train_1d,
                "val_1d":    val_1d,
                "test_1d":   test_1d,
            })
            fold += 1

        current = current + slide_delta

    logger.info("Created %d walk-forward folds", len(windows))
    for w in windows:
        logger.info(
            "  Fold %d: train [%s → %s, %d bars] val [%s → %s, %d bars] "
            "test [%s → %s, %d bars]",
            w["fold"],
            w["train_start"][:10], w["train_end"][:10], w["train_bars"],
            w["val_start"][:10],   w["val_end"][:10],   w["val_bars"],
            w["test_start"][:10],  w["test_end"][:10],  w["test_bars"],
        )

    return windows


# ---------------------------------------------------------------------------
# Per-fold trainer
# ---------------------------------------------------------------------------

class HTFWalkForwardTrainer:
    """
    Walk-forward validation trainer for the HTF multi-timeframe system.

    Each fold:
      1. Creates train/val/test envs from pre-split 4-TF DataFrames
      2. Creates a fresh HTFTradingAgent
      3. Runs curriculum: Phase 1 (phase1_steps) + Phase 2 (phase2_steps)
      4. Evaluates on validation and test sets
      5. Computes overfitting ratio (val Sharpe / test Sharpe)
    """

    def __init__(
        self,
        output_dir: Path,
        initial_balance: float = 10_000.0,
        phase1_steps: int = 200_000,
        phase2_steps: int = 400_000,
        n_eval_episodes: int = 3,
        position_size: float = 0.25,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_balance = initial_balance
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.n_eval_episodes = n_eval_episodes
        self.position_size = position_size

        self.fold_results: List[Dict] = []

        # Set up file logging
        fh = logging.FileHandler(self.output_dir / "walkforward.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        logger.addHandler(fh)

    def train_fold(self, window: Dict) -> Dict:
        """Train and evaluate one walk-forward fold. Returns fold metrics dict."""
        fold = window["fold"]
        fold_dir = self.output_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        sep = "=" * 70
        logger.info("\n%s", sep)
        logger.info(
            "FOLD %d  |  train [%s → %s, %d bars]  val [%s → %s, %d bars]  "
            "test [%s → %s, %d bars]",
            fold,
            window["train_start"][:10], window["train_end"][:10], window["train_bars"],
            window["val_start"][:10],   window["val_end"][:10],   window["val_bars"],
            window["test_start"][:10],  window["test_end"][:10],  window["test_bars"],
        )
        logger.info("%s", sep)

        # --- Build environments ---
        logger.info("Building environments...")
        try:
            train_env = _create_env(
                window["train_15m"], window["train_1h"],
                window["train_4h"], window["train_1d"],
                initial_balance=self.initial_balance,
                training=True,
                position_size=self.position_size,
            )
            val_env = _create_env(
                window["val_15m"], window["val_1h"],
                window["val_4h"], window["val_1d"],
                initial_balance=self.initial_balance,
                training=False,
                position_size=self.position_size,
            )
            test_env = _create_env(
                window["test_15m"], window["test_1h"],
                window["test_4h"], window["test_1d"],
                initial_balance=self.initial_balance,
                training=False,
                position_size=self.position_size,
            )
        except Exception as exc:
            logger.error("Failed to build environments for fold %d: %s", fold, exc, exc_info=True)
            return self._empty_result(fold, window, str(exc))

        logger.info(
            "Env obs shape: %s  |  train=%d bars  val=%d bars  test=%d bars",
            train_env.observation_space.shape,
            window["train_bars"], window["val_bars"], window["test_bars"],
        )

        # --- Build agent ---
        logger.info("Initialising HTFTradingAgent...")
        try:
            from src.brain.htf_agent import create_htf_agent  # noqa: PLC0415
            agent = create_htf_agent(env=train_env, config_path=None, model_path=None)
            agent.model.verbose = 0  # quiet during walk-forward
        except Exception as exc:
            logger.error("Failed to create agent for fold %d: %s", fold, exc, exc_info=True)
            return self._empty_result(fold, window, str(exc))

        # --- Curriculum training ---
        save_path = str(fold_dir)
        t_start = time.time()

        logger.info("Phase 1: HTF Alignment Focus (%d steps)...", self.phase1_steps)
        try:
            metrics_p1 = agent.train_phase1(
                timesteps=self.phase1_steps,
                eval_env=val_env,
                save_path=save_path,
            )
            logger.info("Phase 1 done: %s", {k: v for k, v in metrics_p1.items()
                                              if not isinstance(v, list)})
        except Exception as exc:
            logger.error("Phase 1 failed for fold %d: %s", fold, exc, exc_info=True)
            return self._empty_result(fold, window, str(exc))

        logger.info("Phase 2: Full 4-TF Cascade Execution (%d steps)...", self.phase2_steps)
        try:
            metrics_p2 = agent.train_phase2(
                timesteps=self.phase2_steps,
                eval_env=val_env,
                save_path=save_path,
            )
            logger.info("Phase 2 done: %s", {k: v for k, v in metrics_p2.items()
                                              if not isinstance(v, list)})
        except Exception as exc:
            logger.error("Phase 2 failed for fold %d: %s", fold, exc, exc_info=True)
            return self._empty_result(fold, window, str(exc))

        train_elapsed = time.time() - t_start
        total_steps_trained = self.phase1_steps + self.phase2_steps
        logger.info(
            "Training complete in %.0fs (%.1f min)  |  %d steps",
            train_elapsed, train_elapsed / 60, total_steps_trained,
        )

        # --- Save fold model ---
        model_path = str(fold_dir / "fold_model.zip")
        try:
            agent.save(model_path)
            logger.info("Saved fold model to %s", model_path)
        except Exception as exc:
            logger.warning("Could not save fold model: %s", exc)

        # --- Evaluate on validation set ---
        logger.info("Evaluating on validation set (%d episodes)...", self.n_eval_episodes)
        try:
            val_agg = evaluate_agent(agent, val_env, n_episodes=self.n_eval_episodes)
            val_metrics = _extract_metrics(val_agg)
        except Exception as exc:
            logger.error("Val evaluation failed for fold %d: %s", fold, exc, exc_info=True)
            val_metrics = {"sharpe_ratio": 0.0, "total_return_pct": 0.0,
                           "max_drawdown_pct": 0.0, "win_rate": 0.0, "total_trades": 0.0}

        logger.info(
            "  Val  →  Sharpe=%.3f  Return=%.1f%%  MaxDD=%.1f%%  "
            "WinRate=%.1f%%  Trades=%.0f",
            val_metrics["sharpe_ratio"],
            val_metrics["total_return_pct"],
            val_metrics["max_drawdown_pct"],
            val_metrics["win_rate"] * 100,
            val_metrics["total_trades"],
        )

        # --- Evaluate on test set (strictly OOS) ---
        logger.info("Evaluating on test set (%d episodes)...", self.n_eval_episodes)
        try:
            test_agg = evaluate_agent(agent, test_env, n_episodes=self.n_eval_episodes)
            test_metrics = _extract_metrics(test_agg)
        except Exception as exc:
            logger.error("Test evaluation failed for fold %d: %s", fold, exc, exc_info=True)
            test_metrics = {"sharpe_ratio": 0.0, "total_return_pct": 0.0,
                            "max_drawdown_pct": 0.0, "win_rate": 0.0, "total_trades": 0.0}

        logger.info(
            "  Test →  Sharpe=%.3f  Return=%.1f%%  MaxDD=%.1f%%  "
            "WinRate=%.1f%%  Trades=%.0f",
            test_metrics["sharpe_ratio"],
            test_metrics["total_return_pct"],
            test_metrics["max_drawdown_pct"],
            test_metrics["win_rate"] * 100,
            test_metrics["total_trades"],
        )

        # --- Anti-overfitting detection ---
        val_sharpe = val_metrics["sharpe_ratio"]
        test_sharpe = test_metrics["sharpe_ratio"]
        overfit_ratio = abs(val_sharpe) / max(abs(test_sharpe), 0.01)
        overfit_flag = overfit_ratio > 3.0
        logger.info(
            "  Overfit ratio (val/test Sharpe): %.2f×  %s",
            overfit_ratio,
            "[OVERFIT WARNING]" if overfit_flag else "[OK]",
        )

        fold_result = {
            "fold": fold,
            "train_start": window["train_start"][:10],
            "train_end":   window["train_end"][:10],
            "val_start":   window["val_start"][:10],
            "val_end":     window["val_end"][:10],
            "test_start":  window["test_start"][:10],
            "test_end":    window["test_end"][:10],
            "train_bars":  window["train_bars"],
            "val_bars":    window["val_bars"],
            "test_bars":   window["test_bars"],
            "steps_trained": total_steps_trained,
            "train_elapsed_s": round(train_elapsed, 1),
            "val_metrics":  val_metrics,
            "test_metrics": test_metrics,
            "overfit_ratio_val_test": round(overfit_ratio, 3),
            "overfit_flag": overfit_flag,
            "model_path": model_path,
        }

        # Save per-fold JSON
        result_json = {k: v for k, v in fold_result.items()
                       if not isinstance(v, pd.DataFrame)}
        with open(fold_dir / "fold_result.json", "w") as f:
            json.dump(result_json, f, indent=2, cls=_NumpyEncoder)

        return fold_result

    def _empty_result(self, fold: int, window: Dict, error: str) -> Dict:
        empty_metrics = {
            "sharpe_ratio": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0.0,
            "profit_factor": 0.0,
        }
        return {
            "fold": fold,
            "error": error,
            "train_start": window.get("train_start", "")[:10],
            "test_start":  window.get("test_start", "")[:10],
            "test_end":    window.get("test_end", "")[:10],
            "train_bars":  window.get("train_bars", 0),
            "val_bars":    window.get("val_bars", 0),
            "test_bars":   window.get("test_bars", 0),
            "val_metrics":  dict(empty_metrics),
            "test_metrics": dict(empty_metrics),
            "overfit_ratio_val_test": 0.0,
            "overfit_flag": False,
        }

    def run(self, windows: List[Dict]) -> Dict:
        """Run all walk-forward folds and return aggregated results."""
        logger.info("\n%s", "#" * 70)
        logger.info(
            "HTF WALK-FORWARD VALIDATION  |  %d folds  |  "
            "Phase1=%d steps  Phase2=%d steps",
            len(windows), self.phase1_steps, self.phase2_steps,
        )
        logger.info("#" * 70)

        for window in windows:
            result = self.train_fold(window)
            self.fold_results.append(result)

        return self.aggregate_results()

    def aggregate_results(self) -> Dict:
        """Aggregate OOS metrics across all folds and print summary."""
        valid = [r for r in self.fold_results if "error" not in r]
        if not valid:
            logger.error("No valid fold results to aggregate")
            return {}

        oos_sharpes    = [r["test_metrics"]["sharpe_ratio"]      for r in valid]
        oos_returns    = [r["test_metrics"]["total_return_pct"]  for r in valid]
        oos_drawdowns  = [r["test_metrics"]["max_drawdown_pct"]  for r in valid]
        oos_winrates   = [r["test_metrics"]["win_rate"]          for r in valid]
        oos_trades     = [r["test_metrics"]["total_trades"]      for r in valid]
        overfit_flags  = [r["overfit_flag"]                      for r in valid]

        val_sharpes    = [r["val_metrics"]["sharpe_ratio"]  for r in valid]
        overfit_ratios = [r["overfit_ratio_val_test"]       for r in valid]

        summary = {
            "total_folds": len(self.fold_results),
            "valid_folds": len(valid),
            "failed_folds": len(self.fold_results) - len(valid),
            # OOS test metrics
            "oos_sharpe_mean":    round(float(np.mean(oos_sharpes)),   4),
            "oos_sharpe_std":     round(float(np.std(oos_sharpes)),    4),
            "oos_sharpe_min":     round(float(np.min(oos_sharpes)),    4),
            "oos_sharpe_max":     round(float(np.max(oos_sharpes)),    4),
            "oos_return_mean_pct":  round(float(np.mean(oos_returns)),  2),
            "oos_return_std_pct":   round(float(np.std(oos_returns)),   2),
            "oos_drawdown_mean_pct": round(float(np.mean(oos_drawdowns)), 2),
            "oos_winrate_mean":   round(float(np.mean(oos_winrates)),  4),
            "avg_trades_per_fold": round(float(np.mean(oos_trades)),   1),
            "positive_fold_pct":  round(sum(r > 0 for r in oos_returns) / len(oos_returns) * 100, 1),
            # Overfitting analysis
            "overfit_flags_count":    sum(overfit_flags),
            "avg_val_sharpe":         round(float(np.mean(val_sharpes)),   4),
            "avg_overfit_ratio":      round(float(np.mean(overfit_ratios)), 3),
            "max_overfit_ratio":      round(float(np.max(overfit_ratios)),  3),
            # Per-fold breakdown
            "per_fold": [
                {
                    "fold":        r["fold"],
                    "test_period": f"{r['test_start']} → {r['test_end']}",
                    "oos_sharpe":        r["test_metrics"]["sharpe_ratio"],
                    "oos_return_pct":    r["test_metrics"]["total_return_pct"],
                    "oos_drawdown_pct":  r["test_metrics"]["max_drawdown_pct"],
                    "oos_win_rate":      r["test_metrics"]["win_rate"],
                    "oos_trades":        r["test_metrics"]["total_trades"],
                    "val_sharpe":        r["val_metrics"]["sharpe_ratio"],
                    "overfit_ratio":     r["overfit_ratio_val_test"],
                    "overfit_flag":      r["overfit_flag"],
                }
                for r in valid
            ],
        }

        # Overfitting verdict
        avg_ratio = summary["avg_overfit_ratio"]
        if avg_ratio < 1.5:
            verdict = "EXCELLENT — model generalizes well to unseen data"
        elif avg_ratio < 3.0:
            verdict = "ACCEPTABLE — mild overfitting, monitor live performance"
        elif avg_ratio < 5.0:
            verdict = "CAUTION — moderate overfitting, reduce model complexity"
        else:
            verdict = "DO NOT DEPLOY — severe overfitting detected"
        summary["overfit_verdict"] = verdict

        # Print summary table
        sep = "=" * 70
        print(f"\n{sep}")
        print("  HTF WALK-FORWARD VALIDATION SUMMARY")
        print(sep)
        print(f"  Folds:          {summary['valid_folds']}/{summary['total_folds']} valid")
        print(f"  OOS Sharpe:     {summary['oos_sharpe_mean']:.3f} ± {summary['oos_sharpe_std']:.3f}")
        print(f"  OOS Return:     {summary['oos_return_mean_pct']:.1f}% ± {summary['oos_return_std_pct']:.1f}%")
        print(f"  OOS MaxDD:      {summary['oos_drawdown_mean_pct']:.1f}%")
        print(f"  OOS Win Rate:   {summary['oos_winrate_mean']:.1%}")
        print(f"  Avg Trades:     {summary['avg_trades_per_fold']:.1f} per fold")
        print(f"  Positive Folds: {summary['positive_fold_pct']:.0f}%")
        print(f"  Overfit Flags:  {summary['overfit_flags_count']}/{summary['valid_folds']} folds")
        print(f"  Avg Val/Test Sharpe Ratio: {summary['avg_overfit_ratio']:.2f}×")
        print(f"  Overfitting:    {verdict}")
        print(sep)

        # Per-fold table
        print(f"\n  {'Fold':>4}  {'Test Period':>23}  {'OOS Sharpe':>10}  "
              f"{'Return%':>8}  {'MaxDD%':>7}  {'WinRate':>8}  {'Trades':>7}  "
              f"{'ValSh':>7}  {'OFRatio':>8}  {'Flag':>5}")
        print("  " + "-" * 104)
        for pf in summary["per_fold"]:
            flag_str = "OVER" if pf["overfit_flag"] else "ok"
            print(
                f"  {pf['fold']:>4}  {pf['test_period']:>23}  "
                f"{pf['oos_sharpe']:>10.3f}  {pf['oos_return_pct']:>8.1f}  "
                f"{pf['oos_drawdown_pct']:>7.1f}  {pf['oos_win_rate']:>8.1%}  "
                f"{pf['oos_trades']:>7.0f}  {pf['val_sharpe']:>7.3f}  "
                f"{pf['overfit_ratio']:>8.2f}x  {flag_str:>5}"
            )
        print()

        # Save summary
        summary_path = self.output_dir / "walk_forward_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=_NumpyEncoder)
        logger.info("Summary saved to %s", summary_path)

        return summary


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HTF Walk-Forward Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-path",
        default="data/historical/BTCUSDT_15m.csv",
        help="Path to BTCUSDT 15M CSV file",
    )
    p.add_argument(
        "--output-dir",
        default="data/models/htf_walkforward",
        help="Base directory for fold outputs",
    )
    p.add_argument(
        "--phase1-steps",
        type=int,
        default=200_000,
        dest="phase1_steps",
        help="Steps for curriculum Phase 1 (HTF alignment focus)",
    )
    p.add_argument(
        "--phase2-steps",
        type=int,
        default=400_000,
        dest="phase2_steps",
        help="Steps for curriculum Phase 2 (full 4-TF cascade)",
    )
    p.add_argument(
        "--train-months",
        type=int,
        default=6,
        dest="train_months",
        help="Train window length in months",
    )
    p.add_argument(
        "--val-months",
        type=int,
        default=2,
        dest="val_months",
        help="Validation slice length (carved from end of train window)",
    )
    p.add_argument(
        "--test-months",
        type=int,
        default=2,
        dest="test_months",
        help="Test (OOS) window length in months",
    )
    p.add_argument(
        "--slide-months",
        type=int,
        default=2,
        dest="slide_months",
        help="How far to advance per fold",
    )
    p.add_argument(
        "--initial-balance",
        type=float,
        default=10_000.0,
        dest="initial_balance",
        help="Simulated starting balance (USD)",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        dest="eval_episodes",
        help="Episodes to run per val/test evaluation",
    )
    p.add_argument(
        "--max-folds",
        type=int,
        default=None,
        dest="max_folds",
        help="Limit number of folds (useful for quick testing)",
    )
    p.add_argument(
        "--position-size",
        type=float,
        default=0.25,
        dest="position_size",
        help="Fraction of capital per trade (default: 0.25)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = args.phase1_steps + args.phase2_steps

    print("\n" + "=" * 70)
    print("  HTF WALK-FORWARD CROSS-VALIDATION")
    print("=" * 70)
    print(f"  Data:           {args.data_path}")
    print(f"  Output:         {output_dir.resolve()}")
    print(f"  Window:         {args.train_months}mo train / {args.val_months}mo val / {args.test_months}mo test")
    print(f"  Slide:          {args.slide_months} months per fold")
    print(f"  Steps/fold:     {args.phase1_steps:,} Phase1 + {args.phase2_steps:,} Phase2 = {total_steps:,} total")
    print(f"  Eval episodes:  {args.eval_episodes}")
    if args.max_folds:
        print(f"  Max folds:      {args.max_folds}")
    print("=" * 70 + "\n")

    # -----------------------------------------------------------------------
    # Step 1: Load 15M data
    # -----------------------------------------------------------------------
    logger.info("[1/4] Loading 15M OHLCV data...")
    df_15m = load_15m_csv(args.data_path)

    # -----------------------------------------------------------------------
    # Step 2: Resample to HTF frames
    # -----------------------------------------------------------------------
    logger.info("[2/4] Resampling to 1H / 4H / 1D...")
    _, df_1h, df_4h, df_1d = build_htf_dataframes(df_15m)

    # -----------------------------------------------------------------------
    # Step 3: Create walk-forward windows
    # -----------------------------------------------------------------------
    logger.info("[3/4] Creating walk-forward windows...")
    windows = create_walk_forward_windows(
        df_15m=df_15m,
        df_1h=df_1h,
        df_4h=df_4h,
        df_1d=df_1d,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        slide_months=args.slide_months,
    )

    if not windows:
        logger.error(
            "No valid walk-forward windows generated. "
            "Need at least %d months of data (have %.1f months).",
            args.train_months + args.test_months,
            len(df_15m) / (4 * 24 * 30),
        )
        return 1

    if args.max_folds:
        windows = windows[:args.max_folds]
        logger.info("Limited to %d folds (--max-folds)", args.max_folds)

    # -----------------------------------------------------------------------
    # Step 4: Run walk-forward training and evaluation
    # -----------------------------------------------------------------------
    logger.info("[4/4] Running walk-forward training (%d folds)...", len(windows))

    trainer = HTFWalkForwardTrainer(
        output_dir=output_dir,
        initial_balance=args.initial_balance,
        phase1_steps=args.phase1_steps,
        phase2_steps=args.phase2_steps,
        n_eval_episodes=args.eval_episodes,
        position_size=args.position_size,
    )

    summary = trainer.run(windows)

    if not summary:
        logger.error("Walk-forward produced no results")
        return 1

    logger.info("\nHTF walk-forward validation complete.")
    logger.info("Results saved to %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
