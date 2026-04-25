"""
Structure-Gated Filter Environment for DRL Trading
====================================================

A Gymnasium environment where BOS/CHOCH signals generate trade candidates
and the model only decides ACCEPT (1) or REJECT (0) for each candidate.

Key design:
  - Binary action space: REJECT=0, ACCEPT=1
  - Action masking for sb3-contrib MaskablePPO
  - When no signal fires, only REJECT is valid (effectively HOLD)
  - Differential Sharpe Ratio reward (Moody & Saffell 2001)
  - ~50 observation features covering price, structure, MTF, vol regime, temporal

Author:  CEO bot
Date:    2026-04-13
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully for testing
# ---------------------------------------------------------------------------
_HAS_BOS_CHOCH = False
try:
    from src.signals.bos_choch import MarketStructure, MarketStructureResult, StructureSignal
    _HAS_BOS_CHOCH = True
except ImportError:
    logger.warning("src.signals.bos_choch not available — signals will be empty")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REJECT = 0
ACCEPT = 1

SL_PCT = 0.015       # 1.5% stop loss
TP_PCT = 0.030       # 3.0% take profit
FEE_PCT = 0.0004     # 0.04% per side
POSITION_SIZE = 0.25  # 25% of balance

N_OBS = 51  # observation dimensionality (13+8+12+4+4+5+3+2)


# ---------------------------------------------------------------------------
# Helper: pure-numpy / pandas indicator functions (no lookahead)
# ---------------------------------------------------------------------------

def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).rolling(period, min_periods=1).mean().values


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorised RSI returning array same length as input."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(com=period - 1, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(com=period - 1, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 1e-12, avg_gain / avg_loss, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_hist(close: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> np.ndarray:
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd_line = ema_f - ema_s
    signal_line = _ema(macd_line, sig)
    return macd_line - signal_line


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ADX, +DI, -DI) arrays."""
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
    atr = _ema(tr, period)
    plus_di = 100.0 * _ema(plus_dm, period) / np.where(atr > 1e-12, atr, 1e-12)
    minus_di = 100.0 * _ema(minus_dm, period) / np.where(atr > 1e-12, atr, 1e-12)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 1e-12, plus_di + minus_di, 1e-12)
    adx = _ema(dx, period)
    return adx, plus_di, minus_di


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return _ema(tr, period)


def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    mf = tp * volume
    n = len(close)
    pos_mf = np.zeros(n)
    neg_mf = np.zeros(n)
    for i in range(1, n):
        if tp[i] > tp[i - 1]:
            pos_mf[i] = mf[i]
        else:
            neg_mf[i] = mf[i]
    pos_sum = pd.Series(pos_mf).rolling(period, min_periods=1).sum().values
    neg_sum = pd.Series(neg_mf).rolling(period, min_periods=1).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(neg_sum > 1e-12, pos_sum / neg_sum, 100.0)
    return 100.0 - 100.0 / (1.0 + ratio)


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    sign = np.sign(np.diff(close, prepend=close[0]))
    return np.cumsum(sign * volume)


def _bollinger_pct_b(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> np.ndarray:
    mid = _sma(close, period)
    std = pd.Series(close).rolling(period, min_periods=1).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = upper - lower
    return np.where(width > 1e-12, (close - lower) / width, 0.5)


# ---------------------------------------------------------------------------
# Pre-computation of all features for the entire DataFrame
# ---------------------------------------------------------------------------

@dataclass
class _PrecomputedFeatures:
    """All pre-computed feature arrays aligned to df_15m index."""
    # Price features (12)
    rsi: np.ndarray
    macd_hist: np.ndarray
    adx: np.ndarray
    plus_di: np.ndarray
    minus_di: np.ndarray
    atr_close_ratio: np.ndarray
    atr_pctile: np.ndarray
    vol_ratio: np.ndarray
    dist_ema20: np.ndarray
    dist_ema50: np.ndarray
    boll_pct_b: np.ndarray
    mfi: np.ndarray
    obv_slope: np.ndarray

    # Multi-timeframe (12) — resampled
    htf_1h_rsi: np.ndarray
    htf_1h_adx: np.ndarray
    htf_1h_macd: np.ndarray
    htf_1h_vr: np.ndarray
    htf_4h_rsi: np.ndarray
    htf_4h_adx: np.ndarray
    htf_4h_macd: np.ndarray
    htf_4h_vr: np.ndarray
    htf_1d_rsi: np.ndarray
    htf_1d_adx: np.ndarray
    htf_1d_macd: np.ndarray
    htf_1d_vr: np.ndarray

    # Volatility regime (4)
    vol_regime_ratio: np.ndarray
    vol_of_vol: np.ndarray
    atr_pctile_100: np.ndarray
    realised_vs_expected: np.ndarray

    # Temporal (5)
    sin_hour: np.ndarray
    cos_hour: np.ndarray
    sin_weekday: np.ndarray
    cos_weekday: np.ndarray
    hours_to_funding: np.ndarray

    n_bars: int


def _compute_htf_features(
    df_15m: pd.DataFrame,
    rule: str,
    n_15m_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample df_15m to a higher timeframe and compute RSI, ADX, MACD_hist, vol_ratio.

    Returns arrays of length n_15m_bars, forward-filled to align with 15m index.
    """
    zeros = np.zeros(n_15m_bars)
    if len(df_15m) < 30:
        return zeros, zeros, zeros, zeros

    try:
        ohlcv = df_15m[["open", "high", "low", "close", "volume"]].copy()
        resampled = ohlcv.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        if len(resampled) < 20:
            return zeros, zeros, zeros, zeros

        c = resampled["close"].values
        h = resampled["high"].values
        lo = resampled["low"].values
        v = resampled["volume"].values

        rsi_arr = _rsi(c) / 100.0
        adx_arr, _, _ = _adx(h, lo, c)
        adx_arr = adx_arr / 100.0
        macd_arr = _macd_hist(c)
        macd_std = np.std(macd_arr) if np.std(macd_arr) > 1e-12 else 1.0
        macd_arr = macd_arr / macd_std
        vol_sma = _sma(v, 20)
        vr_arr = np.where(vol_sma > 1e-12, v / vol_sma, 1.0)

        # Map back to 15m index via forward fill
        htf_df = pd.DataFrame({
            "rsi": rsi_arr, "adx": adx_arr, "macd": macd_arr, "vr": vr_arr,
        }, index=resampled.index)
        htf_aligned = htf_df.reindex(df_15m.index, method="ffill").fillna(0.0)

        return (
            htf_aligned["rsi"].values,
            htf_aligned["adx"].values,
            htf_aligned["macd"].values,
            htf_aligned["vr"].values,
        )
    except Exception as e:
        logger.debug("HTF resample (%s) failed: %s", rule, e)
        return zeros, zeros, zeros, zeros


def _precompute_all(df: pd.DataFrame) -> _PrecomputedFeatures:
    """Compute every feature array once for the full DataFrame."""
    n = len(df)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # --- Price features ---
    rsi = _rsi(close) / 100.0  # [0,1]
    macd_h = _macd_hist(close)
    macd_std = np.std(macd_h)
    if macd_std > 1e-12:
        macd_h = macd_h / macd_std
    macd_h = np.clip(macd_h, -3.0, 3.0)

    adx_arr, plus_di, minus_di = _adx(high, low, close)
    adx_norm = adx_arr / 100.0
    plus_di_norm = plus_di / 100.0
    minus_di_norm = minus_di / 100.0

    atr_arr = _atr(high, low, close)
    atr_close = np.where(close > 1e-12, atr_arr / close, 0.0)

    # ATR percentile over rolling 100 bars
    atr_pctile = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        window = atr_arr[start:i + 1]
        if len(window) > 1:
            atr_pctile[i] = np.searchsorted(np.sort(window), atr_arr[i]) / len(window)
        else:
            atr_pctile[i] = 0.5

    vol_sma20 = _sma(volume, 20)
    vol_ratio = np.where(vol_sma20 > 1e-12, volume / vol_sma20, 1.0)
    vol_ratio = np.clip(vol_ratio, 0.0, 5.0) / 5.0  # normalize

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    dist_ema20 = np.where(close > 1e-12, (close - ema20) / close, 0.0)
    dist_ema50 = np.where(close > 1e-12, (close - ema50) / close, 0.0)
    dist_ema20 = np.clip(dist_ema20, -0.1, 0.1) / 0.1
    dist_ema50 = np.clip(dist_ema50, -0.1, 0.1) / 0.1

    boll_b = _bollinger_pct_b(close)
    boll_b = np.clip(boll_b, -0.5, 1.5)

    mfi = _mfi(high, low, close, volume) / 100.0

    obv = _obv(close, volume)
    obv_slope = np.zeros(n)
    for i in range(5, n):
        obv_slope[i] = obv[i] - obv[i - 5]
    obv_std = np.std(obv_slope)
    if obv_std > 1e-12:
        obv_slope = obv_slope / obv_std
    obv_slope = np.clip(obv_slope, -3.0, 3.0)

    # --- Multi-timeframe ---
    htf_1h = _compute_htf_features(df, "1h", n)
    htf_4h = _compute_htf_features(df, "4h", n)
    htf_1d = _compute_htf_features(df, "1D", n)

    # --- Volatility regime ---
    atr_ewma20 = _ema(atr_arr, 20)
    vol_regime_ratio = np.where(atr_ewma20 > 1e-12, atr_arr / atr_ewma20, 1.0)
    vol_regime_ratio = np.clip(vol_regime_ratio, 0.0, 3.0) / 3.0

    vol_of_vol = np.zeros(n)
    for i in range(20, n):
        window = atr_arr[i - 19:i + 1]
        m = np.mean(window)
        if m > 1e-12:
            vol_of_vol[i] = np.std(window) / m
    vol_of_vol = np.clip(vol_of_vol, 0.0, 2.0) / 2.0

    # Realised vol / expected vol
    returns = np.diff(close, prepend=close[0]) / np.where(close > 1e-12, close, 1.0)
    realised_vol = pd.Series(np.abs(returns)).rolling(20, min_periods=1).std().values
    expected_vol = atr_close
    rv_ev = np.where(expected_vol > 1e-12, realised_vol / expected_vol, 1.0)
    rv_ev = np.clip(rv_ev, 0.0, 3.0) / 3.0

    # --- Temporal ---
    if hasattr(df.index, 'hour'):
        hours = df.index.hour.values.astype(np.float64)
        weekdays = df.index.weekday.values.astype(np.float64)
    else:
        hours = np.zeros(n)
        weekdays = np.zeros(n)

    sin_hour = np.sin(hours / 24.0 * 2.0 * np.pi)
    cos_hour = np.cos(hours / 24.0 * 2.0 * np.pi)
    sin_weekday = np.sin(weekdays / 7.0 * 2.0 * np.pi)
    cos_weekday = np.cos(weekdays / 7.0 * 2.0 * np.pi)
    # Hours to funding: funding at 00:00, 08:00, 16:00 UTC
    hours_to_funding = np.zeros(n)
    for i in range(n):
        h = hours[i]
        next_funding = 8.0 * math.ceil(h / 8.0) if h % 8 != 0 else h + 8.0
        if next_funding > 24.0:
            next_funding = 8.0
        hours_to_funding[i] = (next_funding - h) / 8.0  # normalize [0,1]

    return _PrecomputedFeatures(
        rsi=rsi, macd_hist=macd_h, adx=adx_norm,
        plus_di=plus_di_norm, minus_di=minus_di_norm,
        atr_close_ratio=atr_close, atr_pctile=atr_pctile,
        vol_ratio=vol_ratio, dist_ema20=dist_ema20, dist_ema50=dist_ema50,
        boll_pct_b=boll_b, mfi=mfi, obv_slope=obv_slope,
        htf_1h_rsi=htf_1h[0], htf_1h_adx=htf_1h[1],
        htf_1h_macd=htf_1h[2], htf_1h_vr=htf_1h[3],
        htf_4h_rsi=htf_4h[0], htf_4h_adx=htf_4h[1],
        htf_4h_macd=htf_4h[2], htf_4h_vr=htf_4h[3],
        htf_1d_rsi=htf_1d[0], htf_1d_adx=htf_1d[1],
        htf_1d_macd=htf_1d[2], htf_1d_vr=htf_1d[3],
        vol_regime_ratio=vol_regime_ratio, vol_of_vol=vol_of_vol,
        atr_pctile_100=atr_pctile, realised_vs_expected=rv_ev,
        sin_hour=sin_hour, cos_hour=cos_hour,
        sin_weekday=sin_weekday, cos_weekday=cos_weekday,
        hours_to_funding=hours_to_funding,
        n_bars=n,
    )


# ---------------------------------------------------------------------------
# Cached signal per bar
# ---------------------------------------------------------------------------

@dataclass
class _BarSignal:
    """Pre-computed BOS/CHOCH signal for a single bar."""
    has_signal: bool = False
    signal_type: float = 0.0    # 0=none, 0.5=BOS, 1.0=CHOCH (normalized)
    direction: float = 0.0      # -1=bearish, +1=bullish
    confidence: float = 0.0
    is_fake: float = 0.0
    bars_since_swing_high: float = 0.0
    bars_since_swing_low: float = 0.0
    swing_range_norm: float = 0.0
    trend: float = 0.0          # -1=bearish, 0=ranging, +1=bullish
    demand_zone_dist: float = 0.0
    supply_zone_dist: float = 0.0


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class StructureFilterEnv(gym.Env):
    """
    Gymnasium environment for structure-gated DRL trading.

    The BOS/CHOCH signals decide WHEN a trade candidate appears.
    The model only decides whether to ACCEPT or REJECT each candidate.

    Action space: Discrete(2) — 0=REJECT, 1=ACCEPT
    Observation space: Box(N_OBS,) with float32 features.

    Supports sb3-contrib MaskablePPO via action_masks() method.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_15m: pd.DataFrame,
        initial_balance: float = 10_000.0,
        training_mode: bool = True,
        episode_length: int = 2000,
        symbol: str = "BTCUSDT",
        btc_df: Optional[pd.DataFrame] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            df_15m: DataFrame with OHLCV columns and DatetimeIndex (or open_time/timestamp col).
            initial_balance: Starting balance in USDT.
            training_mode: If True, episodes start at random positions.
            episode_length: Max bars per episode.
            symbol: Trading pair (used for cross-asset features).
            btc_df: Optional BTC 15m DataFrame for cross-asset features (alts only).
            render_mode: Gymnasium render mode.
        """
        super().__init__()

        # --- Normalise DataFrame ---
        self._df = self._prepare_df(df_15m)
        self._btc_df = self._prepare_df(btc_df) if btc_df is not None else None
        self._n_bars = len(self._df)
        self._symbol = symbol
        self._is_btc = "BTC" in symbol.upper()
        self._training_mode = training_mode
        self._episode_length = episode_length
        self._initial_balance = initial_balance
        self.render_mode = render_mode

        assert self._n_bars >= 200, f"Need >= 200 bars, got {self._n_bars}"

        # --- Pre-compute features ---
        logger.info("Pre-computing features for %d bars ...", self._n_bars)
        self._features = _precompute_all(self._df)

        # --- Pre-compute BOS/CHOCH signals for every bar ---
        logger.info("Pre-computing BOS/CHOCH signals ...")
        self._bar_signals: List[_BarSignal] = self._precompute_signals()

        # --- Pre-compute cross-asset features ---
        self._btc_3bar_ret = np.zeros(self._n_bars)
        self._btc_corr = np.zeros(self._n_bars)
        self._btc_dom_mom = np.zeros(self._n_bars)
        self._alt_beta = np.zeros(self._n_bars)
        if not self._is_btc and self._btc_df is not None:
            self._compute_cross_asset()

        # --- Spaces ---
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(N_OBS,), dtype=np.float32,
        )

        # --- Episode state (set in reset) ---
        self._step_idx = 0
        self._start_idx = 0
        self._balance = initial_balance
        self._position = 0       # 0=flat, 1=long, -1=short
        self._entry_price = 0.0
        self._entry_bar = 0
        self._total_trades = 0
        self._wins = 0
        self._losses = 0
        self._pnl_history: List[float] = []
        self._trade_returns: List[float] = []

        # Differential Sharpe Ratio state
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        self._dsr_eta = 0.01

        logger.info(
            "StructureFilterEnv ready: %d bars, %d signal bars, symbol=%s",
            self._n_bars,
            sum(1 for s in self._bar_signals if s.has_signal),
            symbol,
        )

    # ------------------------------------------------------------------
    # DataFrame preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has lowercase columns and DatetimeIndex."""
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Handle time column → DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("open_time", "timestamp", "datetime", "date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    break
            else:
                # Try converting the existing index
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass

        # Ensure required columns
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # BOS/CHOCH pre-computation
    # ------------------------------------------------------------------

    def _precompute_signals(self) -> List[_BarSignal]:
        """Run MarketStructure on the full DataFrame and map signals to bars.

        Strategy: analyse the entire chart once to get all BOS/CHOCH signals
        and swing points, then assign each signal to its bar_index. Structure
        context (swing distances, trend, zones) is computed per-bar from the
        swing point sequence.
        """
        bar_signals = [_BarSignal() for _ in range(self._n_bars)]

        if not _HAS_BOS_CHOCH:
            logger.warning("BOS/CHOCH module unavailable — no signals pre-computed")
            return bar_signals

        ms = MarketStructure(swing_lookback=5, min_swing_pct=0.005)
        close = self._df["close"].values

        # --- Run full analysis once ---
        result: MarketStructureResult = ms._analyze_single_tf(self._df)
        swing_points = result.swing_points
        all_signals = result.signals  # List[StructureSignal]

        logger.info(
            "Full-chart analysis: %d swings, %d signals (trend=%s)",
            len(swing_points), len(all_signals), result.trend,
        )

        # --- Build signal lookup: bar_index → StructureSignal ---
        signal_map: Dict[int, StructureSignal] = {}
        for sig in all_signals:
            # If multiple signals on same bar, keep the latest (CHOCH > BOS)
            if sig.bar_index not in signal_map or sig.kind == "choch":
                signal_map[sig.bar_index] = sig

        # --- Build swing point timeline for per-bar context ---
        # Sort swing points by index and build arrays for fast lookup
        swings_sorted = sorted(swing_points, key=lambda sp: sp.index)

        # Pre-compute trend at each bar using a progressive scan
        # We determine trend from the most recent 4+ swings up to each bar
        bar_trend = np.zeros(self._n_bars)  # 0=ranging
        recent_highs: List[float] = []
        recent_lows: List[float] = []
        swing_idx = 0
        for i in range(self._n_bars):
            # Add any swings at or before this bar
            while swing_idx < len(swings_sorted) and swings_sorted[swing_idx].index <= i:
                sp = swings_sorted[swing_idx]
                if sp.kind == "high":
                    recent_highs.append(sp.price)
                    if len(recent_highs) > 3:
                        recent_highs = recent_highs[-3:]
                else:
                    recent_lows.append(sp.price)
                    if len(recent_lows) > 3:
                        recent_lows = recent_lows[-3:]
                swing_idx += 1

            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                hh = recent_highs[-1] > recent_highs[-2]
                hl = recent_lows[-1] > recent_lows[-2]
                lh = recent_highs[-1] < recent_highs[-2]
                ll = recent_lows[-1] < recent_lows[-2]
                if hh and hl:
                    bar_trend[i] = 1.0
                elif lh and ll:
                    bar_trend[i] = -1.0
            elif i > 0:
                bar_trend[i] = bar_trend[i - 1]

        # --- Pre-compute per-bar last swing high/low ---
        last_sh_bar = np.zeros(self._n_bars, dtype=np.int64)
        last_sl_bar = np.zeros(self._n_bars, dtype=np.int64)
        last_sh_price = np.full(self._n_bars, close[0])
        last_sl_price = np.full(self._n_bars, close[0])

        cur_sh_bar = 0
        cur_sl_bar = 0
        cur_sh_price = close[0]
        cur_sl_price = close[0]
        swing_ptr = 0
        for i in range(self._n_bars):
            while swing_ptr < len(swings_sorted) and swings_sorted[swing_ptr].index <= i:
                sp = swings_sorted[swing_ptr]
                if sp.kind == "high":
                    cur_sh_bar = sp.index
                    cur_sh_price = sp.price
                else:
                    cur_sl_bar = sp.index
                    cur_sl_price = sp.price
                swing_ptr += 1
            last_sh_bar[i] = cur_sh_bar
            last_sl_bar[i] = cur_sl_bar
            last_sh_price[i] = cur_sh_price
            last_sl_price[i] = cur_sl_price

        # --- Fill _BarSignal for every bar ---
        for i in range(self._n_bars):
            bs = _BarSignal()

            # Structure context
            bars_sh = max(0, i - last_sh_bar[i]) / 100.0
            bars_sl = max(0, i - last_sl_bar[i]) / 100.0
            swing_range = (
                (last_sh_price[i] - last_sl_price[i]) / close[i]
                if close[i] > 0 else 0.0
            )

            bs.trend = float(bar_trend[i])
            bs.bars_since_swing_high = min(bars_sh, 1.0)
            bs.bars_since_swing_low = min(bars_sl, 1.0)
            bs.swing_range_norm = float(np.clip(swing_range, -0.5, 0.5))

            # Order block proximity (swing levels as proxy)
            if last_sl_price[i] > 0 and close[i] > 0:
                bs.demand_zone_dist = float(np.clip(
                    (close[i] - last_sl_price[i]) / close[i], -0.1, 0.1
                ))
            if last_sh_price[i] > 0 and close[i] > 0:
                bs.supply_zone_dist = float(np.clip(
                    (last_sh_price[i] - close[i]) / close[i], -0.1, 0.1
                ))

            # Check for signal on this bar
            if i in signal_map:
                sig = signal_map[i]
                bs.has_signal = True
                bs.signal_type = 0.5 if sig.kind == "bos" else 1.0
                bs.direction = 1.0 if sig.direction == "bullish" else -1.0
                bs.confidence = result.confidence
                bs.is_fake = 1.0 if sig.is_fake else 0.0

            bar_signals[i] = bs

        return bar_signals

    # ------------------------------------------------------------------
    # Cross-asset features
    # ------------------------------------------------------------------

    def _compute_cross_asset(self) -> None:
        """Compute BTC-relative features for altcoin pairs."""
        if self._btc_df is None or self._is_btc:
            return

        btc_close = self._btc_df["close"].values
        alt_close = self._df["close"].values
        n = min(len(btc_close), self._n_bars)

        # BTC 3-bar return
        for i in range(3, n):
            self._btc_3bar_ret[i] = (btc_close[i] / btc_close[i - 3]) - 1.0
        self._btc_3bar_ret = np.clip(self._btc_3bar_ret, -0.1, 0.1) / 0.1

        # Rolling 20-bar correlation
        btc_ret = np.diff(btc_close[:n], prepend=btc_close[0]) / np.where(btc_close[:n] > 0, btc_close[:n], 1.0)
        alt_ret = np.diff(alt_close[:n], prepend=alt_close[0]) / np.where(alt_close[:n] > 0, alt_close[:n], 1.0)
        btc_s = pd.Series(btc_ret)
        alt_s = pd.Series(alt_ret)
        corr = btc_s.rolling(20, min_periods=5).corr(alt_s).fillna(0.0).values
        self._btc_corr[:n] = corr

        # Alt beta to BTC (20-bar rolling)
        btc_var = btc_s.rolling(20, min_periods=5).var().fillna(1e-12).values
        cov = btc_s.rolling(20, min_periods=5).cov(alt_s).fillna(0.0).values
        beta = np.where(btc_var > 1e-12, cov / btc_var, 1.0)
        self._alt_beta[:n] = np.clip(beta, -3.0, 3.0)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self, idx: int) -> np.ndarray:
        """Build the observation vector for bar `idx`."""
        f = self._features
        bs = self._bar_signals[idx]
        close = self._df["close"].values[idx]

        obs = np.zeros(N_OBS, dtype=np.float32)
        k = 0

        # Price features (13)
        obs[k] = f.rsi[idx];                k += 1
        obs[k] = f.macd_hist[idx];          k += 1
        obs[k] = f.adx[idx];                k += 1
        obs[k] = f.plus_di[idx];            k += 1
        obs[k] = f.minus_di[idx];           k += 1
        obs[k] = f.atr_close_ratio[idx];    k += 1
        obs[k] = f.atr_pctile[idx];         k += 1
        obs[k] = f.vol_ratio[idx];          k += 1
        obs[k] = f.dist_ema20[idx];         k += 1
        obs[k] = f.dist_ema50[idx];         k += 1
        obs[k] = f.boll_pct_b[idx];         k += 1
        obs[k] = f.mfi[idx];                k += 1
        obs[k] = f.obv_slope[idx];          k += 1

        # Structure features (8)
        obs[k] = bs.signal_type;             k += 1
        obs[k] = bs.direction;               k += 1
        obs[k] = bs.confidence;              k += 1
        obs[k] = bs.is_fake;                k += 1
        obs[k] = bs.bars_since_swing_high;   k += 1
        obs[k] = bs.bars_since_swing_low;    k += 1
        obs[k] = bs.swing_range_norm;        k += 1
        obs[k] = bs.trend;                   k += 1

        # Multi-timeframe (12)
        obs[k] = f.htf_1h_rsi[idx];         k += 1
        obs[k] = f.htf_1h_adx[idx];         k += 1
        obs[k] = f.htf_1h_macd[idx];        k += 1
        obs[k] = f.htf_1h_vr[idx];          k += 1
        obs[k] = f.htf_4h_rsi[idx];         k += 1
        obs[k] = f.htf_4h_adx[idx];         k += 1
        obs[k] = f.htf_4h_macd[idx];        k += 1
        obs[k] = f.htf_4h_vr[idx];          k += 1
        obs[k] = f.htf_1d_rsi[idx];         k += 1
        obs[k] = f.htf_1d_adx[idx];         k += 1
        obs[k] = f.htf_1d_macd[idx];        k += 1
        obs[k] = f.htf_1d_vr[idx];          k += 1

        # Cross-asset (4)
        obs[k] = self._btc_3bar_ret[idx];   k += 1
        obs[k] = self._btc_corr[idx];       k += 1
        obs[k] = self._btc_dom_mom[idx];    k += 1
        obs[k] = self._alt_beta[idx];       k += 1

        # Volatility regime (4)
        obs[k] = f.vol_regime_ratio[idx];    k += 1
        obs[k] = f.vol_of_vol[idx];          k += 1
        obs[k] = f.atr_pctile_100[idx];      k += 1
        obs[k] = f.realised_vs_expected[idx]; k += 1

        # Temporal (5)
        obs[k] = f.sin_hour[idx];            k += 1
        obs[k] = f.cos_hour[idx];            k += 1
        obs[k] = f.sin_weekday[idx];         k += 1
        obs[k] = f.cos_weekday[idx];         k += 1
        obs[k] = f.hours_to_funding[idx];    k += 1

        # Order block proximity (2)
        obs[k] = bs.demand_zone_dist;        k += 1
        obs[k] = bs.supply_zone_dist;        k += 1

        # Position state (3)
        obs[k] = float(self._position);      k += 1
        unrealised = 0.0
        if self._position != 0 and self._entry_price > 0:
            unrealised = self._position * (close / self._entry_price - 1.0)
        obs[k] = np.clip(unrealised, -0.5, 0.5);  k += 1
        balance_ratio = (self._balance / self._initial_balance) - 1.0
        obs[k] = np.clip(balance_ratio, -0.5, 0.5); k += 1

        assert k == N_OBS, f"Expected {N_OBS} features, wrote {k}"
        return obs

    # ------------------------------------------------------------------
    # Action masking (MaskablePPO interface)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions for current state.

        Shape: (2,) — [REJECT_valid, ACCEPT_valid]
        MaskablePPO expects True for VALID actions.
        """
        mask = np.array([True, False], dtype=bool)  # REJECT always valid

        bs = self._bar_signals[self._step_idx]
        if bs.has_signal:
            # Signal fired — both actions valid unless same-direction position
            if self._position == 0:
                mask[ACCEPT] = True
            elif self._position == 1 and bs.direction < 0:
                # In long, bearish signal → can accept (close & reverse)
                mask[ACCEPT] = True
            elif self._position == -1 and bs.direction > 0:
                # In short, bullish signal → can accept (close & reverse)
                mask[ACCEPT] = True
            # Same-direction signal while already in position → mask ACCEPT
            # (already handled by default False)

        return mask

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)

        # Choose start index
        warmup = 110  # enough for 100-bar lookback + indicators
        max_start = self._n_bars - self._episode_length
        if self._training_mode and max_start > warmup:
            self._start_idx = self.np_random.integers(warmup, max_start)
        else:
            self._start_idx = warmup

        self._step_idx = self._start_idx
        self._balance = self._initial_balance
        self._position = 0
        self._entry_price = 0.0
        self._entry_bar = 0
        self._total_trades = 0
        self._wins = 0
        self._losses = 0
        self._pnl_history = []
        self._trade_returns = []
        self._dsr_A = 0.0
        self._dsr_B = 0.0

        obs = self._get_obs(self._step_idx)
        info = {"signal": self._bar_signals[self._step_idx].has_signal}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step (one 15m bar).

        Args:
            action: 0=REJECT, 1=ACCEPT

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        info: Dict[str, Any] = {}
        close = self._df["close"].values
        high = self._df["high"].values
        low = self._df["low"].values
        current_close = close[self._step_idx]
        bs = self._bar_signals[self._step_idx]

        # --- Check SL/TP on existing position ---
        if self._position != 0:
            sl_hit, tp_hit, exit_price = self._check_sl_tp(self._step_idx)
            if sl_hit or tp_hit:
                pnl_pct = self._close_position(exit_price)
                reward = self._compute_dsr_reward(pnl_pct)
                info["trade_closed"] = True
                info["pnl_pct"] = pnl_pct
                info["sl_hit"] = sl_hit
                info["tp_hit"] = tp_hit

        # --- Process action ---
        if bs.has_signal and action == ACCEPT and self._position == 0:
            # Open new position in signal direction
            direction = 1 if bs.direction > 0 else -1
            self._position = direction
            self._entry_price = current_close
            self._entry_bar = self._step_idx
            # Deduct entry fee
            notional = self._balance * POSITION_SIZE
            self._balance -= notional * FEE_PCT
            info["opened"] = "long" if direction == 1 else "short"

        elif bs.has_signal and action == ACCEPT and self._position != 0:
            # Close existing position and open in new direction
            pnl_pct = self._close_position(current_close)
            reward += self._compute_dsr_reward(pnl_pct)

            direction = 1 if bs.direction > 0 else -1
            self._position = direction
            self._entry_price = current_close
            self._entry_bar = self._step_idx
            notional = self._balance * POSITION_SIZE
            self._balance -= notional * FEE_PCT
            info["reversed"] = "long" if direction == 1 else "short"
            info["pnl_pct"] = pnl_pct

        elif bs.has_signal and action == REJECT:
            # Hindsight reward for rejection quality
            reward += self._hindsight_reject_reward(self._step_idx, bs)

        # No reward for non-signal bars (HOLD)

        # --- Advance ---
        self._step_idx += 1
        bars_elapsed = self._step_idx - self._start_idx
        terminated = self._balance <= 0
        truncated = (
            self._step_idx >= self._n_bars - 1
            or bars_elapsed >= self._episode_length
        )

        # Force close at episode end
        if (terminated or truncated) and self._position != 0:
            exit_p = close[min(self._step_idx, self._n_bars - 1)]
            pnl_pct = self._close_position(exit_p)
            reward += self._compute_dsr_reward(pnl_pct)
            info["force_closed"] = True
            info["pnl_pct"] = pnl_pct

        obs = self._get_obs(min(self._step_idx, self._n_bars - 1))
        info["balance"] = self._balance
        info["position"] = self._position
        info["step"] = self._step_idx

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Trade mechanics
    # ------------------------------------------------------------------

    def _check_sl_tp(self, idx: int) -> Tuple[bool, bool, float]:
        """Check if SL or TP was hit on bar `idx`.

        Returns (sl_hit, tp_hit, exit_price).
        """
        high_val = self._df["high"].values[idx]
        low_val = self._df["low"].values[idx]

        if self._position == 1:  # Long
            sl_price = self._entry_price * (1.0 - SL_PCT)
            tp_price = self._entry_price * (1.0 + TP_PCT)
            if low_val <= sl_price:
                return True, False, sl_price
            if high_val >= tp_price:
                return False, True, tp_price
        elif self._position == -1:  # Short
            sl_price = self._entry_price * (1.0 + SL_PCT)
            tp_price = self._entry_price * (1.0 - TP_PCT)
            if high_val >= sl_price:
                return True, False, sl_price
            if low_val <= tp_price:
                return False, True, tp_price

        return False, False, 0.0

    def _close_position(self, exit_price: float) -> float:
        """Close current position at exit_price. Returns PnL as fraction of balance."""
        if self._position == 0 or self._entry_price <= 0:
            return 0.0

        notional = self._balance * POSITION_SIZE
        raw_return = self._position * (exit_price / self._entry_price - 1.0)
        pnl = notional * raw_return
        fee = notional * FEE_PCT  # exit fee
        net_pnl = pnl - fee

        self._balance += net_pnl
        pnl_pct = net_pnl / self._initial_balance

        self._total_trades += 1
        if net_pnl > 0:
            self._wins += 1
        else:
            self._losses += 1
        self._pnl_history.append(net_pnl)
        self._trade_returns.append(pnl_pct)

        self._position = 0
        self._entry_price = 0.0
        self._entry_bar = 0

        return pnl_pct

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_dsr_reward(self, pnl_pct: float) -> float:
        """Differential Sharpe Ratio reward (Moody & Saffell 2001)."""
        delta = pnl_pct
        new_A = self._dsr_A + self._dsr_eta * (delta - self._dsr_A)
        new_B = self._dsr_B + self._dsr_eta * (delta ** 2 - self._dsr_B)
        denom = (self._dsr_B - self._dsr_A ** 2) ** 1.5

        if abs(denom) > 1e-10:
            dsr = (self._dsr_B * delta - 0.5 * self._dsr_A * delta ** 2) / denom
        else:
            dsr = delta  # fallback to raw return when insufficient history

        self._dsr_A = new_A
        self._dsr_B = new_B

        return float(np.clip(dsr, -2.0, 2.0))

    def _hindsight_reject_reward(self, idx: int, bs: _BarSignal) -> float:
        """Small hindsight reward/penalty for rejecting a signal.

        Looks ahead to see if the signal would have been profitable.
        """
        close = self._df["close"].values
        high = self._df["high"].values
        low = self._df["low"].values
        entry = close[idx]

        if entry <= 0:
            return 0.0

        direction = 1.0 if bs.direction > 0 else -1.0

        # Scan forward up to 20 bars to check SL/TP outcome
        sl_price = entry * (1.0 - SL_PCT) if direction > 0 else entry * (1.0 + SL_PCT)
        tp_price = entry * (1.0 + TP_PCT) if direction > 0 else entry * (1.0 - TP_PCT)

        for j in range(idx + 1, min(idx + 21, self._n_bars)):
            if direction > 0:  # Would have been long
                if low[j] <= sl_price:
                    return 0.001   # Good reject — would have hit SL
                if high[j] >= tp_price:
                    return -0.001  # Bad reject — missed a winner
            else:  # Would have been short
                if high[j] >= sl_price:
                    return 0.001   # Good reject
                if low[j] <= tp_price:
                    return -0.001  # Bad reject

        return 0.0  # No SL/TP hit within window — neutral

    # ------------------------------------------------------------------
    # Episode metrics
    # ------------------------------------------------------------------

    def get_episode_metrics(self) -> Dict[str, float]:
        """Return summary metrics for the completed episode."""
        metrics: Dict[str, float] = {
            "total_trades": self._total_trades,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": self._wins / max(self._total_trades, 1),
            "final_balance": self._balance,
            "total_return_pct": (self._balance / self._initial_balance - 1.0) * 100.0,
            "total_pnl": sum(self._pnl_history),
        }

        if len(self._trade_returns) >= 2:
            returns = np.array(self._trade_returns)
            mean_r = np.mean(returns)
            std_r = np.std(returns)

            # Sharpe (annualised assuming ~96 bars/day × 365 = 35040 bars/year)
            trades_per_year = 35040.0 / max(self._episode_length, 1) * self._total_trades
            if std_r > 1e-12:
                metrics["sharpe"] = (mean_r / std_r) * np.sqrt(trades_per_year)
            else:
                metrics["sharpe"] = 0.0

            # Sortino
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else 1e-12
            if downside_std > 1e-12:
                metrics["sortino"] = (mean_r / downside_std) * np.sqrt(trades_per_year)
            else:
                metrics["sortino"] = 0.0

            # Max drawdown
            cumulative = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative)
            dd = peak - cumulative
            metrics["max_drawdown_pct"] = float(np.max(dd)) * 100.0 if len(dd) > 0 else 0.0

            # Profit factor
            gross_profit = float(np.sum(returns[returns > 0]))
            gross_loss = float(np.abs(np.sum(returns[returns < 0])))
            metrics["profit_factor"] = gross_profit / max(gross_loss, 1e-12)

            # Average winner / loser
            winners = returns[returns > 0]
            losers = returns[returns < 0]
            metrics["avg_win_pct"] = float(np.mean(winners)) * 100.0 if len(winners) > 0 else 0.0
            metrics["avg_loss_pct"] = float(np.mean(losers)) * 100.0 if len(losers) > 0 else 0.0
        else:
            metrics["sharpe"] = 0.0
            metrics["sortino"] = 0.0
            metrics["max_drawdown_pct"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["avg_win_pct"] = 0.0
            metrics["avg_loss_pct"] = 0.0

        # Signal stats
        start = self._start_idx
        end = min(self._step_idx, self._n_bars)
        signals_seen = sum(1 for i in range(start, end) if self._bar_signals[i].has_signal)
        metrics["signals_seen"] = signals_seen
        metrics["accept_rate"] = self._total_trades / max(signals_seen, 1)

        return metrics

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        if self.render_mode == "human":
            idx = self._step_idx
            bs = self._bar_signals[min(idx, self._n_bars - 1)]
            close = self._df["close"].values[min(idx, self._n_bars - 1)]
            sig_str = "SIGNAL" if bs.has_signal else "      "
            pos_str = {0: "FLAT", 1: "LONG", -1: "SHORT"}.get(self._position, "?")
            msg = (
                f"[{idx:>6d}] {sig_str} | close={close:>10.2f} | "
                f"pos={pos_str} | bal={self._balance:>10.2f} | "
                f"trades={self._total_trades} W={self._wins} L={self._losses}"
            )
            print(msg)
            return msg
        return None
