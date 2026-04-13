#!/usr/bin/env python3
"""
RSI Guard Backtest — Compare 3 guard configurations on 30 days of data.

Scenarios:
  A) RSI 70 flat     — strict, no regime awareness (original guard)
  B) RSI 75 in trend — moderate loosening (TRENDING_UP/DOWN + ADX >= 25)
  C) RSI 80 in trend — aggressive loosening (current deployed config)

Uses the SAME models, features, and inference as the live bots.
Processes one symbol at a time to stay within 1.6GB RAM headroom.

Usage:
    python3 backtest_rsi_guard.py
    python3 backtest_rsi_guard.py --days 30 --symbol BTCUSDT
"""
import sys
import os
import gc
import json
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project root
REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

# Suppress noisy logs during backtest
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("bt_rsi")
logger.setLevel(logging.INFO)

# Limit threads (same as live)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.features.htf_features import HTFFeatureEngine
from src.features.regime_detector import MarketRegimeDetector, MarketRegime

# ── Guard configs to compare ──
# Each scenario has RSI guard params + per-symbol confidence overrides
SCENARIOS = {
    "A_strict_70": {
        "ob": 70, "os": 30, "ob_trend": 70, "os_trend": 30, "adx_min": 25,
        "conf": {},  # use defaults
        "dir_conf": {},  # no directional overrides
    },
    "B_rsi75": {
        "ob": 70, "os": 30, "ob_trend": 75, "os_trend": 25, "adx_min": 25,
        "conf": {},
        "dir_conf": {},
    },
    "C_rsi80": {
        "ob": 70, "os": 30, "ob_trend": 80, "os_trend": 20, "adx_min": 25,
        "conf": {},
        "dir_conf": {},
    },
    "D_rsi75+conf": {
        "ob": 70, "os": 30, "ob_trend": 75, "os_trend": 25, "adx_min": 25,
        "conf": {"ETHUSDT": 0.90, "BTCUSDT": 0.60, "XRPUSDT": 0.55},
        "dir_conf": {"XRPUSDT": {"SHORT": 0.70}},  # XRP shorts need 0.70+
    },
    "E_rsi75+conf_tight": {
        "ob": 70, "os": 30, "ob_trend": 75, "os_trend": 25, "adx_min": 25,
        "conf": {"ETHUSDT": 0.90, "BTCUSDT": 0.65, "XRPUSDT": 0.60},
        "dir_conf": {"XRPUSDT": {"SHORT": 0.80}, "BTCUSDT": {"SHORT": 0.65}},
    },
}

# ── Constants matching live ──
ACTION_HOLD, ACTION_LONG, ACTION_SHORT = 0, 1, 2
STOP_LOSS_PCT = 0.015
TAKE_PROFIT_PCT = 0.030
MIN_CONFIDENCE = 0.45
SYMBOL_MIN_CONFIDENCE = {"ETHUSDT": 0.80}  # base defaults (overridden per scenario)
ADX_GUARD_MIN = 20  # Low-ADX ranging block (same across all scenarios)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def fetch_candles(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch historical candles from Binance public API."""
    import urllib.request
    limit = min(days * (1440 // {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}[interval]), 1500)
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={symbol}&interval={interval}&limit={limit}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx, plus_di, minus_di


def detect_regime(adx_val: float, plus_di: float, minus_di: float) -> str:
    """Simplified regime detection matching the API's regime output."""
    if adx_val >= 25:
        if plus_di > minus_di:
            return "TRENDING_UP"
        else:
            return "TRENDING_DOWN"
    return "RANGING"


def guard_allows(direction: str, rsi: float, adx: float, regime: str, cfg: dict) -> bool:
    """Apply RSI + ADX guard with the given config. Returns True if trade is allowed."""
    # ADX ranging block (same for all scenarios)
    if adx < ADX_GUARD_MIN:
        return False

    # Determine effective RSI threshold
    ob_threshold = cfg["ob"]
    os_threshold = cfg["os"]
    if adx >= cfg["adx_min"]:
        if direction == "LONG" and regime == "TRENDING_UP":
            ob_threshold = cfg["ob_trend"]
        elif direction == "SHORT" and regime == "TRENDING_DOWN":
            os_threshold = cfg["os_trend"]

    if direction == "LONG" and rsi > ob_threshold:
        return False
    if direction == "SHORT" and rsi < os_threshold:
        return False
    return True


class Position:
    def __init__(self, direction: str, entry_price: float, sl: float, tp: float):
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp


def load_model(symbol: str):
    """Load the HTF model for a symbol. Returns (model, vec_normalize) or (None, None)."""
    from live_trading_htf import find_best_htf_model
    model_path, vecnorm_path = find_best_htf_model(symbol)
    if model_path is None:
        return None, None

    model = PPO.load(str(model_path))
    vec_normalize = None
    if vecnorm_path and vecnorm_path.exists():
        try:
            import gymnasium as gym
            from gymnasium import spaces as gym_spaces
            n_features = 117
            dummy_env = gym.Env()
            dummy_env.observation_space = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
            dummy_env.action_space = gym_spaces.Discrete(3)
            dummy_env.reset = lambda **kw: (np.zeros(n_features, dtype=np.float32), {})
            dummy_env.step = lambda a: (np.zeros(n_features, dtype=np.float32), 0.0, True, False, {})
            dummy_venv = DummyVecEnv([lambda: dummy_env])
            vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_venv)
            vec_normalize.training = False
            vec_normalize.norm_reward = False
        except Exception:
            pass
    return model, vec_normalize


def get_model_action(model, vec_normalize, obs: np.ndarray) -> Tuple[int, float]:
    """Run PPO inference, return (action, confidence)."""
    import torch
    obs_2d = obs.reshape(1, -1)
    if vec_normalize is not None:
        try:
            obs_2d = vec_normalize.normalize_obs(obs_2d)
        except Exception:
            pass
    action, _ = model.predict(obs_2d, deterministic=True)
    action = int(action.item() if hasattr(action, "item") else action)
    try:
        with torch.no_grad():
            obs_tensor = model.policy.obs_to_tensor(obs_2d)[0]
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
        confidence = float(np.max(probs))
    except Exception:
        confidence = 1.0 / 3.0
    return action, confidence


def compute_htf_obs(feature_engine, df_15m: pd.DataFrame, position: int,
                    position_price: float, balance: float, initial_balance: float) -> Optional[np.ndarray]:
    """Compute 117-dim observation matching live bot's compute_observation."""
    try:
        # Resample to multiple timeframes
        df = df_15m.set_index("timestamp") if "timestamp" in df_15m.columns else df_15m
        if not isinstance(df.index, pd.DatetimeIndex):
            return None

        # Resample
        ohlcv_cols = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df_1h = df.resample("1h").agg(ohlcv_cols).dropna()
        df_4h = df.resample("4h").agg(ohlcv_cols).dropna()
        df_1d = df.resample("1D").agg(ohlcv_cols).dropna()
        df_15 = df.copy()

        if len(df_1d) < 5 or len(df_4h) < 10 or len(df_1h) < 20 or len(df_15) < 30:
            return None

        f1d = feature_engine.compute_1d_features(df_1d.reset_index(), len(df_1d) - 1)
        f4h = feature_engine.compute_4h_features(df_4h.reset_index(), len(df_4h) - 1)
        f1h = feature_engine.compute_1h_features(df_1h.reset_index(), len(df_1h) - 1)
        f15m = feature_engine.compute_15m_features(df_15.reset_index(), len(df_15) - 1)

        sig_1d, sig_4h, sig_1h, sig_15m = float(f1d[-1]), float(f4h[-1]), float(f1h[-1]), float(f15m[-1])
        f_align = feature_engine.compute_alignment_full(sig_1d, sig_4h, sig_1h, sig_15m)

        feats_114 = np.concatenate([f1d, f4h, f1h, f15m, f_align])

        current_price = float(df_15.iloc[-1]["close"])
        if position != 0 and position_price > 0:
            if position == 1:
                unrealized_pnl = (current_price - position_price) / (position_price + 1e-10)
            else:
                unrealized_pnl = (position_price - current_price) / (position_price + 1e-10)
        else:
            unrealized_pnl = 0.0
        balance_ratio = (balance - initial_balance) / (initial_balance + 1e-10)

        pos_state = np.array([float(position), np.clip(unrealized_pnl, -0.5, 0.5),
                              np.clip(balance_ratio, -0.5, 0.5)], dtype=np.float32)
        obs = np.concatenate([feats_114, pos_state]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
    except Exception as e:
        return None


def backtest_symbol(symbol: str, days: int) -> Dict:
    """Run the 3-scenario backtest for one symbol. Returns per-scenario results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Backtesting {symbol} — {days} days")
    logger.info(f"{'='*60}")

    # Fetch data
    logger.info(f"Fetching 15m candles...")
    df_15m = fetch_candles(symbol, "15m", days)
    logger.info(f"  Got {len(df_15m)} bars ({df_15m['timestamp'].iloc[0]} → {df_15m['timestamp'].iloc[-1]})")

    # Compute RSI and ADX on 15m
    df_15m["rsi"] = compute_rsi(df_15m["close"])
    adx_series, plus_di_series, minus_di_series = compute_adx(df_15m)
    df_15m["adx"] = adx_series
    df_15m["plus_di"] = plus_di_series
    df_15m["minus_di"] = minus_di_series

    # Load model
    logger.info(f"Loading model for {symbol}...")
    model, vec_normalize = load_model(symbol)
    if model is None:
        logger.warning(f"No model for {symbol} — skipping")
        return {}

    # Feature engine
    try:
        feature_engine = HTFFeatureEngine()
    except Exception:
        from src.features.ultimate_features import UltimateFeatureEngine
        feature_engine = UltimateFeatureEngine()

    initial_balance = 5000.0
    step_interval = 1  # Evaluate every bar (15m). For speed, can increase.

    # Run backtest for each scenario
    results = {}
    for scenario_name, cfg in SCENARIOS.items():
        position: Optional[Position] = None
        balance = initial_balance
        trades: List[Dict] = []
        position_int = 0  # 0=flat, 1=long, -1=short
        position_price = 0.0

        # Step through data — start at bar 200 to have enough history for features
        start_bar = 200
        for i in range(start_bar, len(df_15m), step_interval):
            bar = df_15m.iloc[i]
            price = bar["close"]
            high = bar["high"]
            low = bar["low"]
            rsi = bar["rsi"]
            adx = bar["adx"]
            plus_di = bar["plus_di"]
            minus_di = bar["minus_di"]

            if np.isnan(rsi) or np.isnan(adx):
                continue

            regime = detect_regime(adx, plus_di, minus_di)

            # Check SL/TP on existing position
            if position is not None:
                hit_sl = False
                hit_tp = False
                if position.direction == "LONG":
                    if low <= position.sl:
                        hit_sl = True
                        exit_price = position.sl
                    elif high >= position.tp:
                        hit_tp = True
                        exit_price = position.tp
                else:  # SHORT
                    if high >= position.sl:
                        hit_sl = True
                        exit_price = position.sl
                    elif low <= position.tp:
                        hit_tp = True
                        exit_price = position.tp

                if hit_sl or hit_tp:
                    if position.direction == "LONG":
                        pnl_pct = (exit_price - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - exit_price) / position.entry_price
                    trade_size = balance * 0.10  # 10% risk pool
                    pnl_usd = trade_size * pnl_pct * 40  # leveraged
                    balance += pnl_usd
                    trades.append({
                        "entry": position.entry_price,
                        "exit": exit_price,
                        "direction": position.direction,
                        "pnl_pct": pnl_pct,
                        "pnl_usd": pnl_usd,
                        "reason": "SL" if hit_sl else "TP",
                        "bar_idx": i,
                    })
                    position = None
                    position_int = 0
                    position_price = 0.0
                    continue

            # Compute observation and get model action
            window = df_15m.iloc[max(0, i-500):i+1].copy()
            obs = compute_htf_obs(feature_engine, window, position_int,
                                  position_price, balance, initial_balance)
            if obs is None:
                continue

            action, confidence = get_model_action(model, vec_normalize, obs)

            if action == ACTION_HOLD:
                continue

            direction = "LONG" if action == ACTION_LONG else "SHORT"

            # Confidence check — scenario can override per-symbol and per-direction
            base_conf = SYMBOL_MIN_CONFIDENCE.get(symbol, MIN_CONFIDENCE)
            scenario_conf = cfg.get("conf", {}).get(symbol, base_conf)
            dir_conf = cfg.get("dir_conf", {}).get(symbol, {}).get(direction, 0)
            min_conf = max(scenario_conf, dir_conf)
            if confidence < min_conf:
                continue

            # If we already have a position in same direction, skip
            if position is not None:
                if (position.direction == "LONG" and action == ACTION_LONG) or \
                   (position.direction == "SHORT" and action == ACTION_SHORT):
                    continue
                # Close existing position at current price (signal reversal)
                if position.direction == "LONG":
                    pnl_pct = (price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - price) / position.entry_price
                trade_size = balance * 0.10
                pnl_usd = trade_size * pnl_pct * 40
                balance += pnl_usd
                trades.append({
                    "entry": position.entry_price,
                    "exit": price,
                    "direction": position.direction,
                    "pnl_pct": pnl_pct,
                    "pnl_usd": pnl_usd,
                    "reason": "REVERSAL",
                    "bar_idx": i,
                })
                position = None
                position_int = 0
                position_price = 0.0

            # Apply RSI/ADX guard
            if not guard_allows(direction, rsi, adx, regime, cfg):
                continue

            # Open new position
            if direction == "LONG":
                sl = price * (1 - STOP_LOSS_PCT)
                tp = price * (1 + TAKE_PROFIT_PCT)
                position_int = 1
            else:
                sl = price * (1 + STOP_LOSS_PCT)
                tp = price * (1 - TAKE_PROFIT_PCT)
                position_int = -1
            position = Position(direction, price, sl, tp)
            position_price = price

        # Close any remaining position at last price
        if position is not None:
            last_price = df_15m.iloc[-1]["close"]
            if position.direction == "LONG":
                pnl_pct = (last_price - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - last_price) / position.entry_price
            trade_size = balance * 0.10
            pnl_usd = trade_size * pnl_pct * 40
            balance += pnl_usd
            trades.append({
                "entry": position.entry_price,
                "exit": last_price,
                "direction": position.direction,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "reason": "END",
                "bar_idx": len(df_15m) - 1,
            })

        # Compute stats
        wins = [t for t in trades if t["pnl_usd"] > 0]
        losses = [t for t in trades if t["pnl_usd"] <= 0]
        longs = [t for t in trades if t["direction"] == "LONG"]
        shorts = [t for t in trades if t["direction"] == "SHORT"]
        long_wins = [t for t in longs if t["pnl_usd"] > 0]
        short_wins = [t for t in shorts if t["pnl_usd"] > 0]
        net_pnl = sum(t["pnl_usd"] for t in trades)
        sl_count = len([t for t in trades if t["reason"] == "SL"])
        tp_count = len([t for t in trades if t["reason"] == "TP"])

        results[scenario_name] = {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / max(len(trades), 1) * 100,
            "net_pnl": net_pnl,
            "final_balance": balance,
            "return_pct": (balance - initial_balance) / initial_balance * 100,
            "longs": len(longs),
            "long_wins": len(long_wins),
            "shorts": len(shorts),
            "short_wins": len(short_wins),
            "sl_exits": sl_count,
            "tp_exits": tp_count,
            "avg_win": np.mean([t["pnl_usd"] for t in wins]) if wins else 0,
            "avg_loss": np.mean([t["pnl_usd"] for t in losses]) if losses else 0,
        }

    # Cleanup
    del model, vec_normalize, feature_engine
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="RSI Guard Backtest — 3 scenario comparison")
    parser.add_argument("--days", type=int, default=30, help="Days of history (default 30)")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol (default: all 4)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS

    all_results = {}
    for symbol in symbols:
        try:
            res = backtest_symbol(symbol, args.days)
            all_results[symbol] = res
        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")
            import traceback; traceback.print_exc()

    # ── Aggregate across all symbols ──
    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATE RESULTS (all symbols combined)")
    logger.info(f"{'='*80}")

    agg = {}
    for scenario in SCENARIOS:
        agg[scenario] = {
            "trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0,
            "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0,
            "sl_exits": 0, "tp_exits": 0, "avg_win_sum": 0, "avg_loss_sum": 0,
            "win_count": 0, "loss_count": 0,
        }
        for symbol in symbols:
            if symbol in all_results and scenario in all_results[symbol]:
                r = all_results[symbol][scenario]
                agg[scenario]["trades"] += r["trades"]
                agg[scenario]["wins"] += r["wins"]
                agg[scenario]["losses"] += r["losses"]
                agg[scenario]["net_pnl"] += r["net_pnl"]
                agg[scenario]["longs"] += r["longs"]
                agg[scenario]["long_wins"] += r["long_wins"]
                agg[scenario]["shorts"] += r["shorts"]
                agg[scenario]["short_wins"] += r["short_wins"]
                agg[scenario]["sl_exits"] += r["sl_exits"]
                agg[scenario]["tp_exits"] += r["tp_exits"]

    # Print comparison table
    header = f"{'Scenario':<20} {'Trades':>7} {'W/L':>7} {'WR%':>6} {'Net P&L':>10} {'Longs':>7} {'L-WR%':>6} {'Shorts':>7} {'S-WR%':>6} {'SL':>4} {'TP':>4}"
    logger.info(header)
    logger.info("-" * len(header))
    for scenario in SCENARIOS:
        a = agg[scenario]
        wr = a["wins"] / max(a["trades"], 1) * 100
        lwr = a["long_wins"] / max(a["longs"], 1) * 100
        swr = a["short_wins"] / max(a["shorts"], 1) * 100
        logger.info(
            f"{scenario:<20} {a['trades']:>7} {a['wins']:>3}/{a['losses']:<3} {wr:>5.1f}% "
            f"{a['net_pnl']:>+10.2f} {a['longs']:>7} {lwr:>5.1f}% {a['shorts']:>7} {swr:>5.1f}% "
            f"{a['sl_exits']:>4} {a['tp_exits']:>4}"
        )

    # Per-symbol breakdown
    for symbol in symbols:
        if symbol not in all_results:
            continue
        logger.info(f"\n── {symbol} ──")
        for scenario in SCENARIOS:
            if scenario not in all_results[symbol]:
                continue
            r = all_results[symbol][scenario]
            wr = r["win_rate"]
            logger.info(
                f"  {scenario:<20} {r['trades']:>4} trades | {r['wins']}W/{r['losses']}L ({wr:.1f}%) | "
                f"P&L {r['net_pnl']:>+8.2f} | L:{r['longs']}({r['long_wins']}W) S:{r['shorts']}({r['short_wins']}W) | "
                f"SL:{r['sl_exits']} TP:{r['tp_exits']}"
            )

    # Save results
    out_path = REPO / "data" / "backtest_rsi_guard_results.json"
    with open(out_path, "w") as f:
        json.dump({"symbols": {s: all_results.get(s, {}) for s in symbols}, "aggregate": agg}, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
