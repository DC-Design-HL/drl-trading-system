#!/usr/bin/env python3
"""
HTF (Hierarchical Multi-Timeframe) DRL Training Script

Trains a PPO agent on the 4-timeframe cascade system:
  1D (macro) -> 4H (structure) -> 1H (momentum) -> 15M (execution)

The script handles the full pipeline:
  1. Load or download 15M OHLCV data from Binance public API
  2. Resample to 1H, 4H and 1D using HTFDataAligner
  3. Train/validation split (last 20% held out)
  4. Curriculum training: Phase 1 (HTF alignment) -> Phase 2 (cascade execution)
  5. Evaluation on held-out set: Sharpe, win_rate, max_drawdown, total_return
  6. Timestamped model checkpoint

Usage:
    python train_htf.py --symbol BTCUSDT --phase1-steps 500000 --phase2-steps 1000000
    python train_htf.py --symbol ETHUSDT --model-path data/models/htf/htf_eth_latest.zip
    python train_htf.py --symbol BTCUSDT --no-curriculum
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_htf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the HTF 4-timeframe DRL trading agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Binance trading pair (e.g. BTCUSDT, ETHUSDT)",
    )
    p.add_argument(
        "--phase1-steps",
        type=int,
        default=500_000,
        dest="phase1_steps",
        help="Environment steps for Phase 1 (HTF alignment focus)",
    )
    p.add_argument(
        "--phase2-steps",
        type=int,
        default=1_000_000,
        dest="phase2_steps",
        help="Environment steps for Phase 2 (full cascade execution)",
    )
    p.add_argument(
        "--no-curriculum",
        action="store_true",
        dest="no_curriculum",
        help="Skip curriculum; run a single flat training phase",
    )
    p.add_argument(
        "--model-path",
        default=None,
        dest="model_path",
        help="Path to an existing model zip to resume training from",
    )
    p.add_argument(
        "--data-path",
        default="data/historical/",
        dest="data_path",
        help="Directory for cached CSV OHLCV data files",
    )
    p.add_argument(
        "--save-dir",
        default="data/models/htf/",
        dest="save_dir",
        help="Directory to save trained model checkpoints",
    )
    p.add_argument(
        "--initial-balance",
        type=float,
        default=10_000.0,
        dest="initial_balance",
        help="Simulated starting balance (USD)",
    )
    p.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level passed to PPO",
    )
    return p


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _fetch_klines_batch(
    symbol: str,
    interval: str,
    start_ms: int,
    limit: int = 1000,
) -> List[list]:
    """Fetch one page of raw kline data from the Binance public API."""
    import requests

    params: Dict = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": limit,
    }
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_ohlcv_from_binance(
    symbol: str,
    interval: str = "15m",
    years: int = 2,
) -> pd.DataFrame:
    """
    Page through the Binance public klines endpoint to retrieve approximately
    ``years`` years of OHLCV data at the requested interval.

    No API key is required; the /api/v3/klines endpoint is public.

    Args:
        symbol:   Binance symbol string, e.g. "BTCUSDT".
        interval: Kline interval string, e.g. "15m", "1h", "4h", "1d".
        years:    How many years of history to request (approximate).

    Returns:
        DataFrame with columns [open_time, open, high, low, close, volume, ...],
        sorted ascending by open_time. open_time is a UTC-aware datetime index.
    """
    ms_per_year = 365 * 24 * 60 * 60 * 1000
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - years * ms_per_year

    all_rows: List[list] = []
    batch_limit = 1000
    page = 0

    logger.info(
        "Fetching %s %s data (~%d years) from Binance ...", symbol, interval, years
    )

    while True:
        batch = _fetch_klines_batch(symbol, interval, start_ms, limit=batch_limit)
        if not batch:
            break

        all_rows.extend(batch)
        page += 1

        # The last row's open_time becomes the next startTime
        last_open_ms = int(batch[-1][0])
        start_ms = last_open_ms + 1  # +1 ms to avoid duplicate

        logger.debug("  Page %d: fetched %d rows (total %d)", page, len(batch), len(all_rows))

        # If fewer than limit rows returned, we've hit the end
        if len(batch) < batch_limit:
            break

        # Small courtesy delay to avoid hammering the endpoint
        time.sleep(0.1)

    logger.info("Total raw klines fetched: %d", len(all_rows))

    df = pd.DataFrame(all_rows, columns=KLINES_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def load_or_fetch_data(
    symbol: str,
    data_path: str,
    interval: str = "15m",
    years: int = 2,
) -> pd.DataFrame:
    """
    Load OHLCV data from a local CSV cache, or fetch from Binance if not found.

    Args:
        symbol:    Trading pair, e.g. "BTCUSDT".
        data_path: Directory to look for / write CSV files.
        interval:  Kline interval (default "15m").
        years:     Years of history to fetch if downloading.

    Returns:
        DataFrame with datetime index at the given interval.
    """
    Path(data_path).mkdir(parents=True, exist_ok=True)
    csv_path = Path(data_path) / f"{symbol}_{interval}.csv"

    if csv_path.exists():
        logger.info("Loading cached data from %s", csv_path)
        df = pd.read_csv(csv_path, parse_dates=["open_time"])
        if df["open_time"].dt.tz is None:
            df["open_time"] = df["open_time"].dt.tz_localize("UTC")
        df = df.sort_values("open_time").reset_index(drop=True)
        logger.info("Loaded %d rows (%.1f months)", len(df), len(df) / (4 * 24 * 30))
    else:
        df = fetch_ohlcv_from_binance(symbol, interval=interval, years=years)
        df.to_csv(csv_path, index=False)
        logger.info("Cached data to %s", csv_path)

    return df


# ---------------------------------------------------------------------------
# HTF data alignment (resampling)
# ---------------------------------------------------------------------------

class HTFDataAligner:
    """
    Resample 15-minute base OHLCV data to higher timeframes.

    Output DataFrames are aligned so that each row in a higher-TF frame
    corresponds to the 15M candle that *closes* it (forward-fill join), giving
    the agent a consistent view across all four timeframes.
    """

    @staticmethod
    def resample(df_15m: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample a 15M OHLCV DataFrame to a coarser timeframe.

        Args:
            df_15m: DataFrame with an open_time column (UTC-aware datetime).
            rule:   Pandas offset alias, e.g. "1H", "4H", "1D".

        Returns:
            Resampled DataFrame with the same column set, indexed by bar open.
        """
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
        """
        Left-merge HTF columns onto the 15M frame using an as-of merge
        (each 15M bar inherits the most recently *closed* HTF bar's values).

        Args:
            df_15m: Base 15M DataFrame sorted by open_time.
            df_htf: Higher-TF DataFrame sorted by open_time.
            suffix: Column suffix, e.g. "_1h", "_4h", "_1d".

        Returns:
            df_15m with additional columns appended for the HTF bar.
        """
        htf_renamed = df_htf.rename(
            columns={
                c: f"{c}{suffix}"
                for c in df_htf.columns
                if c != "open_time"
            }
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
    """
    Build all four timeframe DataFrames from the base 15M data.

    Returns:
        Tuple of (df_15m, df_1h, df_4h, df_1d) — all individually usable
        DataFrames plus the 15M frame enriched with aligned HTF columns.
    """
    aligner = HTFDataAligner()

    df_1h = aligner.resample(df_15m, "1h")
    df_4h = aligner.resample(df_15m, "4h")
    df_1d = aligner.resample(df_15m, "1D")

    # Enrich the 15M frame with aligned HTF data for environments that
    # consume a single merged DataFrame.
    df_merged = aligner.align_to_15m(df_15m, df_1h, "_1h")
    df_merged = aligner.align_to_15m(df_merged, df_4h, "_4h")
    df_merged = aligner.align_to_15m(df_merged, df_1d, "_1d")

    logger.info(
        "Timeframes built: 15M=%d rows, 1H=%d rows, 4H=%d rows, 1D=%d rows",
        len(df_15m), len(df_1h), len(df_4h), len(df_1d),
    )

    return df_merged, df_1h, df_4h, df_1d


# ---------------------------------------------------------------------------
# Environment factory (HTFTradingEnv)
# ---------------------------------------------------------------------------

def _create_env(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    initial_balance: float,
    training: bool,
):
    """
    Attempt to instantiate an HTFTradingEnv; fall back gracefully to
    UltimateTradingEnv if the HTF env is not yet implemented, so that this
    script remains runnable while the env is being developed.

    Args:
        df_15m:          Base 15-minute OHLCV DataFrame.
        df_1h:           1-hour resampled OHLCV DataFrame.
        df_4h:           4-hour resampled OHLCV DataFrame.
        df_1d:           Daily resampled OHLCV DataFrame.
        initial_balance: Starting portfolio value.
        training:        True when used for training (disables live API calls).

    Returns:
        A Gymnasium-compatible trading environment.
    """
    # Ensure all DataFrames have DatetimeIndex (required by HTFTradingEnv)
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "open_time" in df.columns:
                df = df.copy()
                df["open_time"] = pd.to_datetime(df["open_time"])
                df = df.set_index("open_time")
        return df

    df_15m = _ensure_datetime_index(df_15m)
    df_1h = _ensure_datetime_index(df_1h)
    df_4h = _ensure_datetime_index(df_4h)
    df_1d = _ensure_datetime_index(df_1d)

    try:
        from src.env.htf_env import HTFTradingEnv  # type: ignore

        env = HTFTradingEnv(
            df_15m=df_15m,
            df_1h=df_1h,
            df_4h=df_4h,
            df_1d=df_1d,
            initial_balance=initial_balance,
            training_mode=training,
        )
        logger.info("Using HTFTradingEnv (%d-dim obs)", env.observation_space.shape[0])
        return env

    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "HTFTradingEnv not found — falling back to UltimateTradingEnv. "
            "Implement src/env/htf_env.py for full HTF capability."
        )
        from src.env.ultimate_env import UltimateTradingEnv  # type: ignore

        env = UltimateTradingEnv(
            df=df_15m,
            initial_balance=initial_balance,
            training_mode=training,
        )
        logger.info(
            "Using UltimateTradingEnv (%d-dim obs)", env.observation_space.shape[0]
        )
        return env


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(agent, env, n_episodes: int = 5) -> Dict[str, float]:
    """
    Run the trained agent on a held-out evaluation environment and collect
    episode-level performance metrics.

    Metrics returned:
        - mean_total_return / std_total_return  (fractional)
        - mean_sharpe / std_sharpe
        - mean_win_rate
        - mean_max_drawdown
        - mean_total_trades
        - mean_profit_factor

    Args:
        agent:      A trained HTFTradingAgent (or compatible).
        env:        Evaluation Gymnasium environment.
        n_episodes: How many full episodes to run.

    Returns:
        Dict of aggregated metric floats.
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
            logger.debug("  Episode %d: %s", ep + 1, ep_metrics)
        else:
            logger.warning("  env.get_episode_metrics() not available; skipping episode.")

    if not all_metrics:
        return {}

    agg: Dict[str, float] = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
        if vals:
            agg[f"mean_{key}"] = float(np.mean(vals))
            if len(vals) > 1:
                agg[f"std_{key}"] = float(np.std(vals))

    return agg


def print_eval_report(metrics: Dict[str, float], symbol: str) -> None:
    """Pretty-print the evaluation results."""
    print()
    print("=" * 60)
    print(f"  EVALUATION RESULTS — {symbol}")
    print("=" * 60)

    display_keys = [
        ("mean_total_return_pct",  "Total Return (%)",      ".2f"),
        ("mean_sharpe_ratio",      "Sharpe Ratio",          ".3f"),
        ("mean_sortino_ratio",     "Sortino Ratio",         ".3f"),
        ("mean_win_rate",          "Win Rate",              ".2%"),
        ("mean_max_drawdown_pct",  "Max Drawdown (%)",      ".2f"),
        ("mean_total_trades",      "Avg Trades / Episode",  ".1f"),
        ("mean_profit_factor",     "Profit Factor",         ".3f"),
        ("mean_avg_trade_pnl",     "Avg Trade PnL ($)",     ".2f"),
    ]

    for key, label, fmt in display_keys:
        if key in metrics:
            value = metrics[key]
            formatted = format(value, fmt)
            print(f"  {label:<30s}  {formatted}")
        else:
            # Try generic lookup for custom envs
            generic = key.replace("mean_", "")
            if f"mean_{generic}" in metrics:
                print(f"  {label:<30s}  {metrics[f'mean_{generic}']:.4g}")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    symbol: str = args.symbol.upper()
    use_curriculum: bool = not args.no_curriculum
    total_steps: int = args.phase1_steps + args.phase2_steps

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  HTF MULTI-TIMEFRAME DRL TRAINING")
    print(f"  Symbol       : {symbol}")
    print(f"  Curriculum   : {use_curriculum}")
    if use_curriculum:
        print(f"  Phase 1 steps: {args.phase1_steps:,}")
        print(f"  Phase 2 steps: {args.phase2_steps:,}")
    print(f"  Total steps  : {total_steps:,}")
    print(f"  Save dir     : {args.save_dir}")
    print("=" * 60 + "\n")
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # STEP 1: Load / download 15M data
    # -----------------------------------------------------------------------
    print("[1/7]  Loading OHLCV data ...")
    df_15m = load_or_fetch_data(
        symbol=symbol,
        data_path=args.data_path,
        interval="15m",
        years=2,
    )
    logger.info("15M data: %d rows spanning %s to %s",
                len(df_15m),
                df_15m["open_time"].iloc[0].strftime("%Y-%m-%d"),
                df_15m["open_time"].iloc[-1].strftime("%Y-%m-%d"))

    # -----------------------------------------------------------------------
    # STEP 2: Resample to HTF frames
    # -----------------------------------------------------------------------
    print("[2/7]  Resampling to 1H / 4H / 1D ...")
    df_merged, df_1h, df_4h, df_1d = build_htf_dataframes(df_15m)

    # -----------------------------------------------------------------------
    # STEP 3: Train / validation split (last 20%) across all 4 timeframes
    # -----------------------------------------------------------------------
    print("[3/7]  Splitting train / validation sets ...")
    n_total = len(df_15m)
    n_val = max(1, int(n_total * 0.20))
    n_train = n_total - n_val

    # Use the 15M cutoff timestamp so HTF frames are split consistently
    cutoff_ts = df_15m["open_time"].iloc[n_train]

    def _split_by_ts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df["open_time"] < cutoff_ts].reset_index(drop=True)
        val   = df[df["open_time"] >= cutoff_ts].reset_index(drop=True)
        return train, val

    df_15m_train, df_15m_val = _split_by_ts(df_15m)
    df_1h_train,  df_1h_val  = _split_by_ts(df_1h)
    df_4h_train,  df_4h_val  = _split_by_ts(df_4h)
    df_1d_train,  df_1d_val  = _split_by_ts(df_1d)

    logger.info(
        "Split (15M): train=%d rows (%.0f%%), val=%d rows (%.0f%%)",
        len(df_15m_train), 100 * len(df_15m_train) / n_total,
        len(df_15m_val),   100 * len(df_15m_val)   / n_total,
    )
    logger.info(
        "Split (1H/4H/1D): train=%d/%d/%d rows, val=%d/%d/%d rows",
        len(df_1h_train), len(df_4h_train), len(df_1d_train),
        len(df_1h_val),   len(df_4h_val),   len(df_1d_val),
    )

    # -----------------------------------------------------------------------
    # STEP 4: Create train and eval environments
    # -----------------------------------------------------------------------
    print("[4/7]  Creating environments ...")
    train_env = _create_env(
        df_15m_train, df_1h_train, df_4h_train, df_1d_train,
        initial_balance=args.initial_balance, training=True,
    )
    eval_env = _create_env(
        df_15m_val, df_1h_val, df_4h_val, df_1d_val,
        initial_balance=args.initial_balance, training=False,
    )

    # -----------------------------------------------------------------------
    # STEP 5: Create agent
    # -----------------------------------------------------------------------
    print("[5/7]  Initialising HTFTradingAgent ...")

    # Import here so the script still parses even if SB3 is not installed
    # (error surfaces at runtime rather than import time).
    from src.brain.htf_agent import create_htf_agent  # noqa: PLC0415

    agent_config = {"verbose": args.verbose}
    agent = create_htf_agent(
        env=train_env,
        model_path=args.model_path,
        config_path=None,
    )
    # Apply CLI-level verbose setting
    agent.model.verbose = args.verbose

    logger.info("Agent: %s", agent)

    # -----------------------------------------------------------------------
    # STEP 6: Curriculum training
    # -----------------------------------------------------------------------
    print("[6/7]  Starting training ...\n")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if use_curriculum:
        print("--- Phase 1: HTF Alignment Focus ---")
        metrics_p1 = agent.train_phase1(
            timesteps=args.phase1_steps,
            eval_env=eval_env,
            save_path=args.save_dir,
        )
        print(f"Phase 1 done. Summary: {metrics_p1}\n")

        print("--- Phase 2: Full 4-TF Cascade Execution ---")
        metrics_p2 = agent.train_phase2(
            timesteps=args.phase2_steps,
            eval_env=eval_env,
            save_path=args.save_dir,
        )
        print(f"Phase 2 done. Summary: {metrics_p2}\n")

        train_metrics = {"phase1": metrics_p1, "phase2": metrics_p2}
    else:
        print("--- Single-phase training ---")
        train_metrics = agent.train(
            total_timesteps=total_steps,
            eval_env=eval_env,
            save_path=args.save_dir,
            use_curriculum=False,
        )
        print(f"Training done. Summary: {train_metrics}\n")

    elapsed = time.time() - t_start
    logger.info("Training wall-clock time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    # -----------------------------------------------------------------------
    # STEP 7: Evaluate on validation set
    # -----------------------------------------------------------------------
    print("[7/7]  Evaluating on validation set ...")
    eval_metrics = evaluate_agent(agent, eval_env, n_episodes=5)
    print_eval_report(eval_metrics, symbol)

    # -----------------------------------------------------------------------
    # Save model with timestamp
    # -----------------------------------------------------------------------
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_filename = f"htf_{symbol.lower()}_{timestamp}.zip"
    model_save_path = str(Path(args.save_dir) / model_filename)

    agent.save(model_save_path)

    # Also write a "latest" symlink / copy for easy resumption
    latest_path = str(Path(args.save_dir) / f"htf_{symbol.lower()}_latest.zip")
    import shutil
    shutil.copy2(model_save_path, latest_path)

    # Companion VecNorm for latest
    src_vecnorm = model_save_path.replace(".zip", "_vecnorm.pkl")
    dst_vecnorm = latest_path.replace(".zip", "_vecnorm.pkl")
    if Path(src_vecnorm).exists():
        shutil.copy2(src_vecnorm, dst_vecnorm)

    print(f"\nModel saved:")
    print(f"  Timestamped : {model_save_path}")
    print(f"  Latest      : {latest_path}")
    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
