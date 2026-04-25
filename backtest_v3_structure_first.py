#!/usr/bin/env python3
"""
Structure-First Backtest v3

FLIPPED APPROACH: BOS/CHOCH + Order Blocks are the PRIMARY entry signals.
The DRL model is demoted to a confirmation/tiebreaker role.

Scenarios:
  S1) Structure only: BOS/CHOCH triggers entry, no model needed
  S2) Structure + OB: BOS/CHOCH + near order block
  S3) Structure + model agree: BOS triggers, model must not disagree (HOLD ok)
  S4) Structure + OB + model agree
  S5) Structure + OB + RSI filter + ADX trending
  B)  Baseline: model-first (current live approach) for comparison

Uses 31 days of 15m data from Binance.
"""
import sys, os, gc, json, logging, time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("bt_v3")
logger.setLevel(logging.INFO)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.features.htf_features import HTFFeatureEngine
from src.signals.bos_choch import MarketStructure

ACTION_HOLD, ACTION_LONG, ACTION_SHORT = 0, 1, 2
STOP_LOSS_PCT = 0.015
TAKE_PROFIT_PCT = 0.030
ADX_GUARD_MIN = 20
OB_PROXIMITY_PCT = 0.010  # 1% proximity (relaxed from 0.5%)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
MIN_BARS_BETWEEN_TRADES = 4  # Don't spam entries — wait at least 1h between trades


def fetch_candles(symbol: str, interval: str, total_bars: int = 3000) -> pd.DataFrame:
    import urllib.request
    all_data = []
    end_time = None
    remaining = total_bars
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    while remaining > 0:
        batch = min(remaining, 1000)
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={batch}"
        if end_time is not None:
            url += f"&endTime={end_time}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        if not data:
            break
        all_data = data + all_data
        end_time = data[0][0] - 1
        remaining -= len(data)
        if len(data) < batch:
            break
        _time.sleep(0.2)
    df = pd.DataFrame(all_data, columns=cols)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
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


def compute_adx(df: pd.DataFrame, period: int = 14):
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


def find_order_blocks(df_slice: pd.DataFrame) -> Tuple[List[float], List[float]]:
    opn = df_slice["open"].values
    close = df_slice["close"].values
    high = df_slice["high"].values
    low = df_slice["low"].values
    n = len(close)
    bull_obs, bear_obs = [], []
    lookback = min(30, n - 2)
    start = max(0, n - lookback)
    atr_proxy = np.mean(high[start:] - low[start:]) + 1e-10
    for i in range(start, n - 2):
        body_i = close[i] - opn[i]
        body_i1 = close[i + 1] - opn[i + 1]
        move = abs(close[i + 2] - close[i + 1]) if (i + 2) < n else 0.0
        if body_i < 0 and body_i1 > 0 and move > atr_proxy * 0.5:
            bull_obs.append((high[i] + low[i]) / 2.0)
        if body_i > 0 and body_i1 < 0 and move > atr_proxy * 0.5:
            bear_obs.append((high[i] + low[i]) / 2.0)
    return bull_obs, bear_obs


def near_order_block(price: float, ob_levels: List[float], proximity_pct: float = OB_PROXIMITY_PCT) -> bool:
    for lvl in ob_levels:
        if abs(price - lvl) / (lvl + 1e-10) < proximity_pct:
            return True
    return False


def get_bos_signals(df_window: pd.DataFrame, df_1h: pd.DataFrame = None) -> Dict:
    try:
        analyzer = MarketStructure()
        result = analyzer.get_signals(df_window, df_1h)
        return result.to_dict() if hasattr(result, 'to_dict') else result
    except Exception:
        return {"bos_bullish": False, "bos_bearish": False, "trend": "ranging",
                "confidence": 0.5, "choch_bullish": False, "choch_bearish": False,
                "fake_bos": False, "fake_choch": False}


def load_model(symbol: str):
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


def get_model_action(model, vec_normalize, obs: np.ndarray) -> int:
    obs_2d = obs.reshape(1, -1)
    if vec_normalize is not None:
        try:
            obs_2d = vec_normalize.normalize_obs(obs_2d)
        except Exception:
            pass
    action, _ = model.predict(obs_2d, deterministic=True)
    return int(action.item() if hasattr(action, "item") else action)


def compute_htf_obs(feature_engine, df_15m: pd.DataFrame, position: int,
                    position_price: float, balance: float, initial_balance: float) -> Optional[np.ndarray]:
    try:
        df = df_15m.set_index("timestamp") if "timestamp" in df_15m.columns else df_15m
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
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
            unrealized_pnl = ((current_price - position_price) / (position_price + 1e-10)
                              if position == 1 else
                              (position_price - current_price) / (position_price + 1e-10))
        else:
            unrealized_pnl = 0.0
        balance_ratio = (balance - initial_balance) / (initial_balance + 1e-10)
        pos_state = np.array([float(position), np.clip(unrealized_pnl, -0.5, 0.5),
                              np.clip(balance_ratio, -0.5, 0.5)], dtype=np.float32)
        obs = np.concatenate([feats_114, pos_state]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
    except Exception:
        return None


class Position:
    def __init__(self, direction: str, entry_price: float, sl: float, tp: float):
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp


def backtest_symbol(symbol: str) -> Dict:
    logger.info(f"\n{'='*60}\nBacktesting {symbol}\n{'='*60}")

    df_15m = fetch_candles(symbol, "15m", 3000)
    logger.info(f"  {len(df_15m)} 15m bars ({df_15m['timestamp'].iloc[0]} → {df_15m['timestamp'].iloc[-1]})")
    df_1h = fetch_candles(symbol, "1h", 1000)

    # Technical indicators
    df_15m["rsi"] = compute_rsi(df_15m["close"])
    adx_s, plus_di_s, minus_di_s = compute_adx(df_15m)
    df_15m["adx"] = adx_s
    df_15m["plus_di"] = plus_di_s
    df_15m["minus_di"] = minus_di_s

    # Load model
    model, vec_normalize = load_model(symbol)
    if model is None:
        logger.warning(f"  No model — skip")
        return {}

    feature_engine = HTFFeatureEngine()
    initial_balance = 5000.0
    start_bar = 200

    # ── PRECOMPUTE all signals once ──
    logger.info(f"  Precomputing signals for {len(df_15m) - start_bar} bars...")
    bar_data = {}

    for i in range(start_bar, len(df_15m)):
        bar = df_15m.iloc[i]
        price = bar["close"]
        rsi = bar["rsi"]
        adx_val = bar["adx"]

        if np.isnan(rsi) or np.isnan(adx_val):
            continue

        # BOS/CHOCH
        bos_window = df_15m.iloc[max(0, i-100):i+1].copy()
        bar_ts = bar["timestamp"]
        h1_window = df_1h[df_1h["timestamp"] <= bar_ts].tail(50)
        bos_result = get_bos_signals(bos_window, h1_window if len(h1_window) > 10 else None)

        # Order blocks
        ob_window = df_15m.iloc[max(0, i-40):i+1]
        bull_obs, bear_obs = find_order_blocks(ob_window)

        # Model action (position-independent for precompute)
        window = df_15m.iloc[max(0, i-500):i+1].copy()
        obs = compute_htf_obs(feature_engine, window, 0, 0.0, initial_balance, initial_balance)
        model_action = None
        if obs is not None:
            model_action = get_model_action(model, vec_normalize, obs)

        # Structure-derived direction
        bos_bull = bos_result.get("bos_bullish", False)
        bos_bear = bos_result.get("bos_bearish", False)
        choch_bull = bos_result.get("choch_bullish", False)
        choch_bear = bos_result.get("choch_bearish", False)
        fake_bos = bos_result.get("fake_bos", False)
        fake_choch = bos_result.get("fake_choch", False)

        # Structure signal: direction from BOS/CHOCH
        struct_long = (bos_bull or choch_bull) and not fake_bos and not fake_choch
        struct_short = (bos_bear or choch_bear) and not fake_bos and not fake_choch

        # OB proximity
        ob_long = near_order_block(price, bull_obs, OB_PROXIMITY_PCT)
        ob_short = near_order_block(price, bear_obs, OB_PROXIMITY_PCT)

        bar_data[i] = {
            "struct_long": struct_long, "struct_short": struct_short,
            "ob_long": ob_long, "ob_short": ob_short,
            "model_action": model_action,
            "rsi": rsi, "adx": adx_val,
            "plus_di": bar["plus_di"], "minus_di": bar["minus_di"],
        }

        if (i - start_bar) % 500 == 0 and i > start_bar:
            logger.info(f"    ... bar {i - start_bar}/{len(df_15m) - start_bar}")

    logger.info(f"  Precompute done. Running scenarios...")

    # ── SCENARIOS ──
    scenarios = {
        "S1_struct_only": {
            "need_struct": True, "need_ob": False, "need_model": False,
            "need_rsi": False, "need_adx": False,
        },
        "S2_struct+ob": {
            "need_struct": True, "need_ob": True, "need_model": False,
            "need_rsi": False, "need_adx": False,
        },
        "S3_struct+model": {
            "need_struct": True, "need_ob": False, "need_model": True,
            "need_rsi": False, "need_adx": False,
        },
        "S4_struct+ob+model": {
            "need_struct": True, "need_ob": True, "need_model": True,
            "need_rsi": False, "need_adx": False,
        },
        "S5_struct+ob+rsi+adx": {
            "need_struct": True, "need_ob": True, "need_model": False,
            "need_rsi": True, "need_adx": True,
        },
        "S6_struct+ob+rsi+model": {
            "need_struct": True, "need_ob": True, "need_model": True,
            "need_rsi": True, "need_adx": False,
        },
        "B_model_first": {
            "need_struct": False, "need_ob": False, "need_model": "primary",
            "need_rsi": True, "need_adx": False,
        },
    }

    results = {}
    for scenario_name, cfg in scenarios.items():
        position: Optional[Position] = None
        balance = initial_balance
        trades: List[Dict] = []
        last_trade_bar = -999
        blocked_by: Dict[str, int] = {"struct": 0, "ob": 0, "model": 0, "rsi": 0, "adx": 0, "cooldown": 0}

        for i in range(start_bar, len(df_15m)):
            if i not in bar_data:
                continue

            bar = df_15m.iloc[i]
            price = bar["close"]
            high_val = bar["high"]
            low_val = bar["low"]
            sig = bar_data[i]

            # Check SL/TP on existing position
            if position is not None:
                hit_sl, hit_tp = False, False
                if position.direction == "LONG":
                    if low_val <= position.sl:
                        hit_sl, exit_price = True, position.sl
                    elif high_val >= position.tp:
                        hit_tp, exit_price = True, position.tp
                else:
                    if high_val >= position.sl:
                        hit_sl, exit_price = True, position.sl
                    elif low_val <= position.tp:
                        hit_tp, exit_price = True, position.tp

                if hit_sl or hit_tp:
                    if position.direction == "LONG":
                        pnl_pct = (exit_price - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - exit_price) / position.entry_price
                    trade_size = balance * 0.10
                    pnl_usd = trade_size * pnl_pct * 40
                    balance += pnl_usd
                    trades.append({"direction": position.direction, "pnl_usd": pnl_usd,
                                   "reason": "SL" if hit_sl else "TP"})
                    position = None
                    continue

            # Skip if already in position
            if position is not None:
                continue

            # Cooldown between trades
            if i - last_trade_bar < MIN_BARS_BETWEEN_TRADES:
                blocked_by["cooldown"] += 1
                continue

            # ── Determine direction based on scenario ──
            direction = None

            if cfg["need_model"] == "primary":
                # MODEL-FIRST (baseline): model decides direction
                if sig["model_action"] == ACTION_LONG:
                    direction = "LONG"
                elif sig["model_action"] == ACTION_SHORT:
                    direction = "SHORT"
                else:
                    continue

                # RSI guard for baseline
                if cfg["need_rsi"]:
                    rsi = sig["rsi"]
                    if direction == "LONG" and rsi > 75:
                        blocked_by["rsi"] += 1
                        continue
                    if direction == "SHORT" and rsi < 25:
                        blocked_by["rsi"] += 1
                        continue
            else:
                # STRUCTURE-FIRST: BOS/CHOCH decides direction
                if cfg["need_struct"]:
                    if sig["struct_long"]:
                        direction = "LONG"
                    elif sig["struct_short"]:
                        direction = "SHORT"
                    else:
                        blocked_by["struct"] += 1
                        continue

                # OB confirmation
                if cfg["need_ob"]:
                    if direction == "LONG" and not sig["ob_long"]:
                        blocked_by["ob"] += 1
                        continue
                    if direction == "SHORT" and not sig["ob_short"]:
                        blocked_by["ob"] += 1
                        continue

                # Model confirmation (not primary — just veto)
                if cfg["need_model"]:
                    ma = sig["model_action"]
                    if ma is None:
                        blocked_by["model"] += 1
                        continue
                    # Model must agree OR hold (not contradict)
                    if direction == "LONG" and ma == ACTION_SHORT:
                        blocked_by["model"] += 1
                        continue
                    if direction == "SHORT" and ma == ACTION_LONG:
                        blocked_by["model"] += 1
                        continue

                # RSI guard
                if cfg["need_rsi"]:
                    rsi = sig["rsi"]
                    if direction == "LONG" and rsi > 75:
                        blocked_by["rsi"] += 1
                        continue
                    if direction == "SHORT" and rsi < 25:
                        blocked_by["rsi"] += 1
                        continue

                # ADX trending filter
                if cfg["need_adx"]:
                    adx_val = sig["adx"]
                    if adx_val < ADX_GUARD_MIN:
                        blocked_by["adx"] += 1
                        continue
                    # ADX must confirm direction
                    plus_di = sig["plus_di"]
                    minus_di = sig["minus_di"]
                    if direction == "LONG" and plus_di < minus_di:
                        blocked_by["adx"] += 1
                        continue
                    if direction == "SHORT" and minus_di < plus_di:
                        blocked_by["adx"] += 1
                        continue

            if direction is None:
                continue

            # Open position
            if direction == "LONG":
                sl = price * (1 - STOP_LOSS_PCT)
                tp = price * (1 + TAKE_PROFIT_PCT)
            else:
                sl = price * (1 + STOP_LOSS_PCT)
                tp = price * (1 - TAKE_PROFIT_PCT)
            position = Position(direction, price, sl, tp)
            last_trade_bar = i

        # Close remaining
        if position is not None:
            last_price = df_15m.iloc[-1]["close"]
            pnl_pct = ((last_price - position.entry_price) / position.entry_price
                       if position.direction == "LONG" else
                       (position.entry_price - last_price) / position.entry_price)
            pnl_usd = balance * 0.10 * pnl_pct * 40
            balance += pnl_usd
            trades.append({"direction": position.direction, "pnl_usd": pnl_usd, "reason": "END"})

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
            "trades": len(trades), "wins": len(wins), "losses": len(losses),
            "win_rate": len(wins) / max(len(trades), 1) * 100,
            "net_pnl": net_pnl, "final_balance": balance,
            "return_pct": (balance - initial_balance) / initial_balance * 100,
            "longs": len(longs), "long_wins": len(long_wins),
            "shorts": len(shorts), "short_wins": len(short_wins),
            "sl_exits": sl_count, "tp_exits": tp_count,
            "blocked_by": blocked_by.copy(),
        }

    del model, vec_normalize, feature_engine
    gc.collect()
    return results


def main():
    all_results = {}
    for symbol in SYMBOLS:
        try:
            all_results[symbol] = backtest_symbol(symbol)
        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")
            import traceback; traceback.print_exc()

    # Aggregate
    scenarios = ["S1_struct_only", "S2_struct+ob", "S3_struct+model",
                 "S4_struct+ob+model", "S5_struct+ob+rsi+adx", "S6_struct+ob+rsi+model",
                 "B_model_first"]

    logger.info(f"\n{'='*90}")
    logger.info("AGGREGATE RESULTS — Structure-First vs Model-First")
    logger.info(f"{'='*90}")

    header = f"{'Scenario':<24} {'Trades':>7} {'W/L':>7} {'WR%':>6} {'Net P&L':>10} {'Longs':>6} {'L-WR%':>6} {'Shorts':>7} {'S-WR%':>6} {'SL':>4} {'TP':>4}"
    logger.info(header)
    logger.info("-" * len(header))

    for scenario in scenarios:
        a = {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0,
             "longs": 0, "long_wins": 0, "shorts": 0, "short_wins": 0,
             "sl_exits": 0, "tp_exits": 0}
        for symbol in SYMBOLS:
            if symbol in all_results and scenario in all_results[symbol]:
                r = all_results[symbol][scenario]
                for k in a:
                    a[k] += r[k]
        wr = a["wins"] / max(a["trades"], 1) * 100
        lwr = a["long_wins"] / max(a["longs"], 1) * 100
        swr = a["short_wins"] / max(a["shorts"], 1) * 100
        logger.info(
            f"{scenario:<24} {a['trades']:>7} {a['wins']:>3}/{a['losses']:<3} {wr:>5.1f}% "
            f"{a['net_pnl']:>+10.2f} {a['longs']:>6} {lwr:>5.1f}% {a['shorts']:>7} {swr:>5.1f}% "
            f"{a['sl_exits']:>4} {a['tp_exits']:>4}"
        )

    # Per-symbol
    for symbol in SYMBOLS:
        if symbol not in all_results:
            continue
        logger.info(f"\n── {symbol} ──")
        for scenario in scenarios:
            if scenario not in all_results[symbol]:
                continue
            r = all_results[symbol][scenario]
            blk = r.get("blocked_by", {})
            blk_str = " ".join(f"{k}={v}" for k, v in blk.items() if v > 0)
            logger.info(
                f"  {scenario:<24} {r['trades']:>3}T | {r['wins']}W/{r['losses']}L ({r['win_rate']:.1f}%) | "
                f"P&L {r['net_pnl']:>+8.2f} | L:{r['longs']}({r['long_wins']}W) S:{r['shorts']}({r['short_wins']}W) | "
                f"SL:{r['sl_exits']} TP:{r['tp_exits']} | blocked: {blk_str}"
            )

    # Save
    out_path = REPO / "data" / "backtest_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
