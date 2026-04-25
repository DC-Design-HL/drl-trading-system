#!/usr/bin/env python3
"""
Multi-Signal Confirmation Backtest v2

Compares 4 scenarios:
  B)  RSI 75 only (baseline from v1)
  D)  RSI 75 + per-symbol confidence (best from v1)
  F)  D + BOS/CHOCH confirmation + Order Block proximity
  G)  F + whale direction + funding rate tiebreaker

Uses the same HTF models as live. 15m candles from Binance.
"""
import sys, os, gc, json, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("bt_v2")
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
MIN_CONFIDENCE = 0.45
SYMBOL_MIN_CONFIDENCE = {"ETHUSDT": 0.80}
ADX_GUARD_MIN = 20
OB_PROXIMITY_PCT = 0.005  # 0.5% from order block level
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# ── Scenarios ──
CONF_TUNED = {"ETHUSDT": 0.90, "BTCUSDT": 0.60, "XRPUSDT": 0.55}
DIR_CONF_TUNED = {"XRPUSDT": {"SHORT": 0.70}}

SCENARIOS = {
    "B_rsi75": {
        "rsi_ob_trend": 75, "rsi_os_trend": 25, "adx_min": 25,
        "conf": {}, "dir_conf": {},
        "use_bos": False, "use_ob": False, "bos_mode": "hard",
        "ob_proximity": OB_PROXIMITY_PCT, "use_whale_funding": False,
    },
    "D_rsi75+conf": {
        "rsi_ob_trend": 75, "rsi_os_trend": 25, "adx_min": 25,
        "conf": CONF_TUNED, "dir_conf": DIR_CONF_TUNED,
        "use_bos": False, "use_ob": False, "bos_mode": "hard",
        "ob_proximity": OB_PROXIMITY_PCT, "use_whale_funding": False,
    },
    "F1_bos_only": {
        "rsi_ob_trend": 75, "rsi_os_trend": 25, "adx_min": 25,
        "conf": CONF_TUNED, "dir_conf": DIR_CONF_TUNED,
        "use_bos": True, "use_ob": False, "bos_mode": "hard",
        "ob_proximity": OB_PROXIMITY_PCT, "use_whale_funding": False,
    },
    "F2_ob_only": {
        "rsi_ob_trend": 75, "rsi_os_trend": 25, "adx_min": 25,
        "conf": CONF_TUNED, "dir_conf": DIR_CONF_TUNED,
        "use_bos": False, "use_ob": True, "bos_mode": "hard",
        "ob_proximity": 0.010, "use_whale_funding": False,  # 1% proximity = looser
    },
    "F3_bos_or_ob": {
        "rsi_ob_trend": 75, "rsi_os_trend": 25, "adx_min": 25,
        "conf": CONF_TUNED, "dir_conf": DIR_CONF_TUNED,
        "use_bos": True, "use_ob": True, "bos_mode": "either",  # BOS OR OB passes
        "ob_proximity": 0.010, "use_whale_funding": False,
    },
    "F4_bos_and_ob": {
        "rsi_ob_trend": 75, "rsi_os_trend": 25, "adx_min": 25,
        "conf": CONF_TUNED, "dir_conf": DIR_CONF_TUNED,
        "use_bos": True, "use_ob": True, "bos_mode": "both",  # BOS AND OB required
        "ob_proximity": 0.010, "use_whale_funding": False,
    },
    "G_full_stack": {
        "rsi_ob_trend": 75, "rsi_os_trend": 25, "adx_min": 25,
        "conf": CONF_TUNED, "dir_conf": DIR_CONF_TUNED,
        "use_bos": True, "use_ob": True, "bos_mode": "either",
        "ob_proximity": 0.010, "use_whale_funding": True,
    },
}

def fetch_candles(symbol: str, interval: str, total_bars: int = 3000) -> pd.DataFrame:
    """Fetch candles with pagination to get more than 1000 bars."""
    import urllib.request, time as _time
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
        all_data = data + all_data  # prepend older data
        end_time = data[0][0] - 1  # before first bar of this batch
        remaining -= len(data)
        if len(data) < batch:
            break  # no more historical data
        _time.sleep(0.2)  # rate limit

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


def detect_regime(adx_val: float, plus_di: float, minus_di: float) -> str:
    if adx_val >= 25:
        return "TRENDING_UP" if plus_di > minus_di else "TRENDING_DOWN"
    return "RANGING"


def guard_allows_rsi(direction: str, rsi: float, adx: float, regime: str, cfg: dict) -> bool:
    if adx < ADX_GUARD_MIN:
        return False
    ob_threshold = 70
    os_threshold = 30
    if adx >= cfg["adx_min"]:
        if direction == "LONG" and regime == "TRENDING_UP":
            ob_threshold = cfg["rsi_ob_trend"]
        elif direction == "SHORT" and regime == "TRENDING_DOWN":
            os_threshold = cfg["rsi_os_trend"]
    if direction == "LONG" and rsi > ob_threshold:
        return False
    if direction == "SHORT" and rsi < os_threshold:
        return False
    return True


def find_order_blocks(df_slice: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """Detect order block levels from recent candles."""
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


def get_bos_signals(df_5m: pd.DataFrame, df_1h: pd.DataFrame = None, df_4h: pd.DataFrame = None) -> Dict:
    """Run BOS/CHOCH analysis. Returns dict with bos_bullish, bos_bearish, trend, confidence."""
    try:
        analyzer = MarketStructure()
        result = analyzer.get_signals(df_5m, df_1h, df_4h)
        return result.to_dict() if hasattr(result, 'to_dict') else result
    except Exception:
        return {"bos_bullish": False, "bos_bearish": False, "trend": "ranging", "confidence": 0.5,
                "choch_bullish": False, "choch_bearish": False, "fake_bos": False}


def simulate_whale_funding(df_15m: pd.DataFrame, i: int) -> Tuple[str, str]:
    """
    Simulate whale direction and funding bias from price action.
    Uses taker volume ratio and recent momentum as proxies.
    Not as good as real API data, but directionally useful for backtest.
    """
    if i < 20:
        return "NEUTRAL", "neutral"

    window = df_15m.iloc[i-20:i+1]
    # Whale proxy: large candle bodies = institutional activity
    bodies = (window["close"] - window["open"]).abs()
    volumes = window["volume"]
    weighted_dir = ((window["close"] - window["open"]) * volumes).sum()
    whale_dir = "BULLISH" if weighted_dir > 0 else "BEARISH" if weighted_dir < 0 else "NEUTRAL"

    # Funding proxy: if price trending up with high volume, longs are crowded
    recent_return = (window["close"].iloc[-1] / window["close"].iloc[0]) - 1
    if recent_return > 0.01:
        funding_bias = "long_crowded"  # fade longs
    elif recent_return < -0.01:
        funding_bias = "short_crowded"  # fade shorts
    else:
        funding_bias = "neutral"

    return whale_dir, funding_bias


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


def get_model_action(model, vec_normalize, obs: np.ndarray) -> Tuple[int, float]:
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

    # Fetch data — 3000 bars = ~31 days of 15m candles
    df_15m = fetch_candles(symbol, "15m", 3000)
    logger.info(f"  {len(df_15m)} 15m bars ({df_15m['timestamp'].iloc[0]} → {df_15m['timestamp'].iloc[-1]})")

    # Also fetch 1h for BOS/CHOCH higher-TF confirmation
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

    # ── PRECOMPUTE expensive signals once for all bars ──
    logger.info(f"  Precomputing model actions + signals for {len(df_15m) - start_bar} bars...")
    bar_signals = {}  # i -> {action, confidence, bos_result, bull_obs, bear_obs, whale_dir, funding_bias}

    for i in range(start_bar, len(df_15m)):
        bar = df_15m.iloc[i]
        price = bar["close"]
        rsi = bar["rsi"]
        adx_val = bar["adx"]

        if np.isnan(rsi) or np.isnan(adx_val):
            continue

        # Model inference (position-independent for precompute — use FLAT)
        window = df_15m.iloc[max(0, i-500):i+1].copy()
        obs = compute_htf_obs(feature_engine, window, 0, 0.0, initial_balance, initial_balance)
        if obs is None:
            continue
        action, confidence = get_model_action(model, vec_normalize, obs)
        if action == ACTION_HOLD:
            bar_signals[i] = {"action": ACTION_HOLD}
            continue

        # BOS/CHOCH (expensive — compute once)
        bos_window = df_15m.iloc[max(0, i-100):i+1].copy()
        bar_ts = bar["timestamp"]
        h1_window = df_1h[df_1h["timestamp"] <= bar_ts].tail(50)
        bos_result = get_bos_signals(bos_window, h1_window if len(h1_window) > 10 else None)

        # Order blocks (compute once with wider window for 1% proximity)
        ob_window = df_15m.iloc[max(0, i-40):i+1]
        bull_obs, bear_obs = find_order_blocks(ob_window)

        # Whale + funding (compute once)
        whale_dir, funding_bias = simulate_whale_funding(df_15m, i)

        bar_signals[i] = {
            "action": action, "confidence": confidence,
            "bos_result": bos_result,
            "bull_obs": bull_obs, "bear_obs": bear_obs,
            "whale_dir": whale_dir, "funding_bias": funding_bias,
        }

        if (i - start_bar) % 500 == 0 and i > start_bar:
            logger.info(f"    ... bar {i - start_bar}/{len(df_15m) - start_bar}")

    logger.info(f"  Precompute done. Running {len(SCENARIOS)} scenarios...")

    # ── Run each scenario using precomputed signals ──
    results = {}
    for scenario_name, cfg in SCENARIOS.items():
        position: Optional[Position] = None
        balance = initial_balance
        trades: List[Dict] = []
        position_int = 0
        position_price = 0.0
        blocked_by: Dict[str, int] = {"rsi": 0, "conf": 0, "bos": 0, "ob": 0, "whale_fund": 0}

        for i in range(start_bar, len(df_15m)):
            bar = df_15m.iloc[i]
            price = bar["close"]
            high_val = bar["high"]
            low_val = bar["low"]
            rsi = bar["rsi"]
            adx_val = bar["adx"]
            plus_di = bar["plus_di"]
            minus_di = bar["minus_di"]

            if i not in bar_signals:
                continue

            regime = detect_regime(adx_val, plus_di, minus_di)

            # Check SL/TP
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
                    position, position_int, position_price = None, 0, 0.0
                    continue

            sig = bar_signals[i]
            action = sig["action"]
            if action == ACTION_HOLD:
                continue

            confidence = sig["confidence"]
            direction = "LONG" if action == ACTION_LONG else "SHORT"

            # ── Layer 2: Confidence check ──
            base_conf = SYMBOL_MIN_CONFIDENCE.get(symbol, MIN_CONFIDENCE)
            scenario_conf = cfg.get("conf", {}).get(symbol, base_conf)
            dir_conf = cfg.get("dir_conf", {}).get(symbol, {}).get(direction, 0)
            min_conf = max(scenario_conf, dir_conf)
            if confidence < min_conf:
                blocked_by["conf"] += 1
                continue

            # ── Layer 3: RSI guard ──
            if not guard_allows_rsi(direction, rsi, adx_val, regime, cfg):
                blocked_by["rsi"] += 1
                continue

            # ── Layer 4+5: BOS/CHOCH + Order Block ──
            bos_mode = cfg.get("bos_mode", "hard")
            bos_pass = True
            ob_pass = True

            if cfg["use_bos"]:
                bos_result = sig["bos_result"]
                bos_bull = bos_result.get("bos_bullish", False)
                bos_bear = bos_result.get("bos_bearish", False)
                choch_bull = bos_result.get("choch_bullish", False)
                choch_bear = bos_result.get("choch_bearish", False)
                fake_bos = bos_result.get("fake_bos", False)

                if direction == "LONG":
                    bos_pass = (bos_bull or choch_bull) and not fake_bos
                else:
                    bos_pass = (bos_bear or choch_bear) and not fake_bos

            if cfg["use_ob"]:
                ob_prox = cfg.get("ob_proximity", OB_PROXIMITY_PCT)
                bull_obs = sig["bull_obs"]
                bear_obs = sig["bear_obs"]
                if direction == "LONG":
                    ob_pass = near_order_block(price, bull_obs, ob_prox)
                else:
                    ob_pass = near_order_block(price, bear_obs, ob_prox)

            if cfg["use_bos"] and cfg["use_ob"]:
                if bos_mode == "both":
                    if not bos_pass:
                        blocked_by["bos"] += 1
                    if not ob_pass:
                        blocked_by["ob"] += 1
                    if not (bos_pass and ob_pass):
                        continue
                elif bos_mode == "either":
                    if not bos_pass:
                        blocked_by["bos"] += 1
                    if not ob_pass:
                        blocked_by["ob"] += 1
                    if not (bos_pass or ob_pass):
                        continue
            elif cfg["use_bos"]:
                if not bos_pass:
                    blocked_by["bos"] += 1
                    continue
            elif cfg["use_ob"]:
                if not ob_pass:
                    blocked_by["ob"] += 1
                    continue

            # ── Layer 6: Whale + funding tiebreaker ──
            if cfg["use_whale_funding"]:
                whale_dir = sig["whale_dir"]
                funding_bias = sig["funding_bias"]
                whale_disagrees = (direction == "LONG" and whale_dir == "BEARISH") or \
                                  (direction == "SHORT" and whale_dir == "BULLISH")
                funding_disagrees = (direction == "LONG" and funding_bias == "long_crowded") or \
                                    (direction == "SHORT" and funding_bias == "short_crowded")
                if whale_disagrees and funding_disagrees:
                    blocked_by["whale_fund"] += 1
                    continue

            # Close existing position if opposite direction
            if position is not None:
                if (position.direction == "LONG" and action == ACTION_LONG) or \
                   (position.direction == "SHORT" and action == ACTION_SHORT):
                    continue
                if position.direction == "LONG":
                    pnl_pct = (price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - price) / position.entry_price
                trade_size = balance * 0.10
                pnl_usd = trade_size * pnl_pct * 40
                balance += pnl_usd
                trades.append({"direction": position.direction, "pnl_usd": pnl_usd, "reason": "REVERSAL"})
                position, position_int, position_price = None, 0, 0.0

            # Open position
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
    logger.info(f"\n{'='*90}")
    logger.info("AGGREGATE RESULTS")
    logger.info(f"{'='*90}")

    header = f"{'Scenario':<20} {'Trades':>7} {'W/L':>7} {'WR%':>6} {'Net P&L':>10} {'Longs':>6} {'L-WR%':>6} {'Shorts':>7} {'S-WR%':>6} {'SL':>4} {'TP':>4}"
    logger.info(header)
    logger.info("-" * len(header))

    for scenario in SCENARIOS:
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
            f"{scenario:<20} {a['trades']:>7} {a['wins']:>3}/{a['losses']:<3} {wr:>5.1f}% "
            f"{a['net_pnl']:>+10.2f} {a['longs']:>6} {lwr:>5.1f}% {a['shorts']:>7} {swr:>5.1f}% "
            f"{a['sl_exits']:>4} {a['tp_exits']:>4}"
        )

    # Per-symbol
    for symbol in SYMBOLS:
        if symbol not in all_results:
            continue
        logger.info(f"\n── {symbol} ──")
        for scenario in SCENARIOS:
            if scenario not in all_results[symbol]:
                continue
            r = all_results[symbol][scenario]
            blk = r.get("blocked_by", {})
            blk_str = " ".join(f"{k}={v}" for k, v in blk.items() if v > 0)
            logger.info(
                f"  {scenario:<20} {r['trades']:>3}T | {r['wins']}W/{r['losses']}L ({r['win_rate']:.1f}%) | "
                f"P&L {r['net_pnl']:>+8.2f} | L:{r['longs']}({r['long_wins']}W) S:{r['shorts']}({r['short_wins']}W) | "
                f"SL:{r['sl_exits']} TP:{r['tp_exits']} | blocked: {blk_str}"
            )

    # Save
    out_path = REPO / "data" / "backtest_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
