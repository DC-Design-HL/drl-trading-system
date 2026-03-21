"""
Multi-Timeframe Hierarchical Feature Engine (HTFFeatureEngine)

Computes 117 observation dimensions across 4 timeframes for a DRL trading agent:
  - 1D  (20 features): macro trend & regime
  - 4H  (25 features): swing structure & Smart Money Concepts
  - 1H  (30 features): momentum & divergence
  - 15M (35 features): micro entry triggers & candle patterns
  - alignment (4 features): cross-TF cascade hierarchy signals
  - position state (3 features): handled externally by env

Also provides HTFDataAligner for resampling a 15M DataFrame to all required TFs.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_BARS = 30  # minimum bars required before computing features

N_1D = 20
N_4H = 25
N_1H = 30
N_15M = 35
N_ALIGN = 4
N_TOTAL = N_1D + N_4H + N_1H + N_15M + N_ALIGN  # 114; env adds 3 for position


# ---------------------------------------------------------------------------
# Low-level helpers (pure numpy / pandas, no lookahead)
# ---------------------------------------------------------------------------

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average via pandas EWM."""
    return pd.Series(series).ewm(span=span, adjust=False).mean().values


def _sma(series: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(series).rolling(period).mean().values


def _rsi(close: np.ndarray, period: int = 14) -> float:
    """Return RSI value at the last bar."""
    if len(close) < period + 1:
        return 50.0
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean().values[-1]
    avg_loss = pd.Series(loss).rolling(period).mean().values[-1]
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range series."""
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    # Pad with first value so length matches close
    atr_series = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return np.concatenate([[atr_series[0]], atr_series])


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
    """Return (ADX, DI+, DI-) at last bar."""
    n = len(close)
    if n < period + 2:
        return 20.0, 20.0, 20.0

    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    # Directional movement
    up_move = np.diff(high)
    down_move = -np.diff(low)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    smooth_tr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    smooth_plus = pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values
    smooth_minus = pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values

    di_plus = 100.0 * smooth_plus[-1] / (smooth_tr[-1] + 1e-10)
    di_minus = 100.0 * smooth_minus[-1] / (smooth_tr[-1] + 1e-10)
    dx = 100.0 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)

    # Smooth DX to get ADX
    dx_series = 100.0 * np.abs(
        pd.Series(smooth_plus).values - pd.Series(smooth_minus).values
    ) / (pd.Series(smooth_plus).values + pd.Series(smooth_minus).values + 1e-10)
    adx_val = pd.Series(dx_series).ewm(span=period, adjust=False).mean().values[-1]

    return float(adx_val), float(di_plus), float(di_minus)


def _macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """Return (macd_line, signal_line, histogram) at last bar."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    hist = macd_line[-1] - signal_line[-1]
    return float(macd_line[-1]), float(signal_line[-1]), float(hist)


def _stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
    """Return (K%, D%) at last bar, 0-100."""
    if len(close) < k_period:
        return 50.0, 50.0
    roll_high = pd.Series(high).rolling(k_period).max().values
    roll_low = pd.Series(low).rolling(k_period).min().values
    k = 100.0 * (close - roll_low) / (roll_high - roll_low + 1e-10)
    d = pd.Series(k).rolling(d_period).mean().values
    return float(k[-1]), float(d[-1])


def _bollinger(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[float, float, float]:
    """Return (upper, mid, lower) at last bar."""
    mid = np.nanmean(close[-period:])
    std = np.nanstd(close[-period:])
    return mid + std_mult * std, mid, mid - std_mult * std


def _keltner(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, mult: float = 1.5) -> Tuple[float, float, float]:
    """Return (upper, mid, lower) Keltner Channel at last bar."""
    mid = _ema(close, period)[-1]
    atr_val = _atr(high, low, close, period)[-1]
    return mid + mult * atr_val, mid, mid - mult * atr_val


def _compact_12_features(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                          volume: np.ndarray, opn: np.ndarray) -> np.ndarray:
    """
    Compute the 12 compact structural features shared by 4H, 1H, and 15M methods.
    Identical logic to compute_mtf_features_at in mtf_env.py.
    """
    feats = np.zeros(12, dtype=np.float32)

    # 1. EMA trend: EMA7 vs EMA14
    ema7 = _ema(close, 7)
    ema14 = _ema(close, 14)
    feats[0] = np.clip((ema7[-1] - ema14[-1]) / (close[-1] * 0.01 + 1e-10), -3.0, 3.0)

    # 2. RSI 14 (normalised 0-1)
    feats[1] = _rsi(close, 14) / 100.0

    # 3-5. Momentum (returns over 5, 10, 20 bars)
    for i, period in enumerate([5, 10, 20]):
        if len(close) > period:
            feats[2 + i] = np.clip((close[-1] / (close[-1 - period] + 1e-12) - 1.0) * 10.0, -3.0, 3.0)

    # 6. ATR ratio (current vs 50-bar avg)
    if len(close) > 2:
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values[-1]
        atr_avg = np.nanmean(tr[-50:]) if len(tr) >= 50 else np.nanmean(tr)
        feats[5] = np.clip(atr_14 / (atr_avg + 1e-10) - 1.0, -2.0, 2.0)

    # 7. Volume trend (current vs 20-bar average)
    vol_avg = np.nanmean(volume[-20:]) if len(volume) >= 20 else np.nanmean(volume)
    feats[6] = np.clip(volume[-1] / (vol_avg + 1e-10) - 1.0, -3.0, 3.0)

    # 8. MACD state (normalised histogram)
    _, _, hist = _macd(close)
    feats[7] = np.clip(hist / (close[-1] * 0.001 + 1e-10), -3.0, 3.0)

    # 9. Bollinger position
    bb_upper, bb_mid, bb_lower = _bollinger(close)
    bb_half_width = (bb_upper - bb_mid) + 1e-10
    feats[8] = np.clip((close[-1] - bb_mid) / (2.0 * bb_half_width), -1.5, 1.5)

    # 10. Support/resistance position (within 20-bar range)
    recent_high = np.max(high[-20:])
    recent_low = np.min(low[-20:])
    price_range = recent_high - recent_low
    if price_range > 0:
        feats[9] = (close[-1] - recent_low) / price_range  # 0-1

    # 11. Candle body ratio (last bar)
    hl_range = high[-1] - low[-1]
    if hl_range > 0:
        feats[10] = (close[-1] - opn[-1]) / hl_range  # -1 to 1

    # 12. Trend strength (DI+ - DI- proxy)
    if len(close) >= 14:
        up_moves = np.diff(high[-15:])
        dn_moves = -np.diff(low[-15:])
        di_plus_raw = np.nanmean(np.where(up_moves > 0, up_moves, 0.0)[-14:])
        di_minus_raw = np.nanmean(np.where(dn_moves > 0, dn_moves, 0.0)[-14:])
        if (di_plus_raw + di_minus_raw) > 0:
            feats[11] = np.clip(
                (di_plus_raw - di_minus_raw) / (di_plus_raw + di_minus_raw), -1.0, 1.0
            )

    return feats


# ---------------------------------------------------------------------------
# HTFFeatureEngine
# ---------------------------------------------------------------------------

class HTFFeatureEngine:
    """
    Hierarchical Multi-Timeframe Feature Engine.

    Computes features for four timeframes (1D, 4H, 1H, 15M) plus cross-TF
    alignment signals.  Total: 114 features (env appends 3 position features).

    Usage
    -----
    engine = HTFFeatureEngine()
    f_1d  = engine.compute_1d_features(df_1d,   at_idx)   # 20
    f_4h  = engine.compute_4h_features(df_4h,   at_idx)   # 25
    f_1h  = engine.compute_1h_features(df_1h,   at_idx)   # 30
    f_15m = engine.compute_15m_features(df_15m, at_idx)   # 35
    align = engine.compute_alignment(f_1d[19], f_4h[24], f_1h[29])  # 4
    obs   = np.concatenate([f_1d, f_4h, f_1h, f_15m, align])        # 114
    """

    def __init__(self, swing_lookback: int = 10, ob_proximity_pct: float = 0.005):
        """
        Parameters
        ----------
        swing_lookback : int
            Bars used to detect local swing highs/lows.
        ob_proximity_pct : float
            Fractional distance to consider price "near" an order block.
        """
        self.swing_lookback = swing_lookback
        self.ob_proximity_pct = ob_proximity_pct

    # ------------------------------------------------------------------
    # 1D Features (20)
    # ------------------------------------------------------------------

    def compute_1d_features(self, df_1d: pd.DataFrame, at_idx: int) -> np.ndarray:
        """
        Compute 20 daily (macro trend) features.

        Parameters
        ----------
        df_1d   : Full daily OHLCV DataFrame (DatetimeIndex).
        at_idx  : Index of the current bar (inclusive).

        Returns
        -------
        np.ndarray of shape (20,), dtype float32.
        """
        out = np.zeros(N_1D, dtype=np.float32)
        d = df_1d.iloc[: at_idx + 1]
        if len(d) < MIN_BARS:
            return out

        close = d["close"].values.astype(np.float64)
        high = d["high"].values.astype(np.float64)
        low = d["low"].values.astype(np.float64)
        opn = d["open"].values.astype(np.float64)
        volume = d["volume"].values.astype(np.float64)

        try:
            # 1. sma_trend: (sma20 - sma50) / sma50 * 100
            sma20 = _sma(close, 20)
            sma50 = _sma(close, 50)
            if not np.isnan(sma50[-1]) and sma50[-1] > 0:
                out[0] = np.clip((sma20[-1] - sma50[-1]) / sma50[-1] * 100.0, -3.0, 3.0)

            # 2. sma200_dist: (close - sma200) / sma200 * 100
            sma200 = _sma(close, 200)
            if not np.isnan(sma200[-1]) and sma200[-1] > 0:
                out[1] = np.clip((close[-1] - sma200[-1]) / sma200[-1] * 100.0, -5.0, 5.0)

            # 3. adx: ADX(14) / 100
            adx_val, di_plus, di_minus = _adx(high, low, close, 14)
            out[2] = np.clip(adx_val / 100.0, 0.0, 1.0)

            # 4. adx_trend: (DI+ - DI-) / (DI+ + DI-)
            denom = di_plus + di_minus + 1e-10
            out[3] = np.clip((di_plus - di_minus) / denom, -1.0, 1.0)

            # 5. macro_rsi: RSI(21) / 100
            out[4] = np.clip(_rsi(close, 21) / 100.0, 0.0, 1.0)

            # 6. monthly_return: pct change over 20 bars
            if len(close) > 20:
                out[5] = np.clip(close[-1] / (close[-21] + 1e-12) - 1.0, -0.5, 0.5)

            # 7. weekly_return: pct change over 5 bars
            if len(close) > 5:
                out[6] = np.clip(close[-1] / (close[-6] + 1e-12) - 1.0, -0.3, 0.3)

            # 8. vol_regime: current volume / 50-bar avg - 1
            vol_avg_50 = np.nanmean(volume[-50:]) if len(volume) >= 50 else np.nanmean(volume)
            out[7] = np.clip(volume[-1] / (vol_avg_50 + 1e-10) - 1.0, -2.0, 2.0)

            # 9. atr_regime: current ATR / 50-bar avg ATR - 1
            atr_series = _atr(high, low, close, 14)
            atr_avg_50 = np.nanmean(atr_series[-50:]) if len(atr_series) >= 50 else np.nanmean(atr_series)
            out[8] = np.clip(atr_series[-1] / (atr_avg_50 + 1e-10) - 1.0, -2.0, 2.0)

            # 10. higher_high: 1 if current high > prev 5-bar high else -1
            prev5_high = np.max(high[-6:-1]) if len(high) >= 6 else high[-1]
            out[9] = 1.0 if high[-1] > prev5_high else -1.0

            # 11. higher_low: 1 if current low > prev 5-bar low else -1
            prev5_low = np.min(low[-6:-1]) if len(low) >= 6 else low[-1]
            out[10] = 1.0 if low[-1] > prev5_low else -1.0

            # 12. price_vs_range: position within 20-bar range, centered on 0
            lo20 = np.min(low[-20:])
            hi20 = np.max(high[-20:])
            rng20 = hi20 - lo20
            if rng20 > 0:
                out[11] = np.clip((close[-1] - lo20) / rng20 - 0.5, -0.5, 0.5)

            # 13. ema_stack: sign of (ema9 - ema21), normalised to {-1, 0, 1}
            ema9 = _ema(close, 9)
            ema21 = _ema(close, 21)
            out[12] = float(np.sign(ema9[-1] - ema21[-1]))

            # 14. trend_maturity: bars since last EMA9/EMA21 crossover, normalised
            crossovers = np.where(np.diff(np.sign(ema9[1:] - ema21[1:])))[0]
            if len(crossovers) > 0:
                bars_since = len(close) - 1 - crossovers[-1]
                out[13] = np.clip(bars_since / 60.0, 0.0, 1.0)
            else:
                out[13] = 1.0  # very mature trend — no recent crossover

            # 15. ichimoku_cloud_pos: simplified cloud position
            # Tenkan: (9-bar high + 9-bar low) / 2
            # Kijun : (26-bar high + 26-bar low) / 2
            # Senkou A: (tenkan + kijun) / 2 (lagged 26)
            if len(close) >= 52:
                tenkan = (np.max(high[-9:]) + np.min(low[-9:])) / 2.0
                kijun = (np.max(high[-26:]) + np.min(low[-26:])) / 2.0
                senkou_a = (tenkan + kijun) / 2.0
                senkou_b = (np.max(high[-52:]) + np.min(low[-52:])) / 2.0
                cloud_top = max(senkou_a, senkou_b)
                cloud_bot = min(senkou_a, senkou_b)
                if close[-1] > cloud_top:
                    out[14] = 0.0       # above cloud (bullish)
                elif close[-1] < cloud_bot:
                    out[14] = -1.0      # below cloud (bearish)
                else:
                    out[14] = -0.5      # inside cloud (neutral)

            # 16. wyckoff_phase: simplified phase detection
            out[15] = self._detect_wyckoff_phase_simple(close, volume, high, low)

            # 17. vol_expansion: volume > 1.5x 20-bar avg → mapped to {0, 1}
            vol_avg_20 = np.nanmean(volume[-20:]) if len(volume) >= 20 else np.nanmean(volume)
            out[16] = 1.0 if volume[-1] > 1.5 * vol_avg_20 else 0.0

            # 18. doji_day: abs(close-open)/(high-low) < 0.1
            body_frac = abs(close[-1] - opn[-1]) / (high[-1] - low[-1] + 1e-10)
            out[17] = 1.0 if body_frac < 0.1 else 0.0

            # 19. range_compression: ATR / 20-day price range, 0-1
            price_range_20 = np.max(high[-20:]) - np.min(low[-20:])
            if price_range_20 > 0:
                out[18] = np.clip(atr_series[-1] / price_range_20, 0.0, 1.0)

            # 20. daily_trend_score: weighted macro bias [-1, 1]
            trend_direction = float(np.sign(sma20[-1] - sma50[-1])) if not np.isnan(sma50[-1]) else 0.0
            rsi_bias = (out[4] - 0.5) * 2.0  # normalise RSI to [-1,1]
            adx_directional = out[3]
            hh_hl_score = (out[9] + out[10]) / 2.0  # avg of HH and HL signals
            out[19] = np.clip(
                0.30 * trend_direction
                + 0.25 * adx_directional
                + 0.25 * rsi_bias
                + 0.20 * hh_hl_score,
                -1.0,
                1.0,
            )

        except Exception as exc:
            logger.debug("1D feature error: %s", exc)

        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 4H Features (25)
    # ------------------------------------------------------------------

    def compute_4h_features(self, df_4h: pd.DataFrame, at_idx: int) -> np.ndarray:
        """
        Compute 25 4-hourly (swing structure) features.

        Features 1-12 follow the compact structure of compute_mtf_features_at.
        Features 13-25 cover Smart Money Concepts: BOS, CHOCH, Order Blocks,
        FVGs, swing distances, structure trend, liquidity pools.

        Parameters
        ----------
        df_4h  : Full 4H OHLCV DataFrame.
        at_idx : Current bar index (inclusive).

        Returns
        -------
        np.ndarray of shape (25,), dtype float32.
        """
        out = np.zeros(N_4H, dtype=np.float32)
        d = df_4h.iloc[: at_idx + 1]
        if len(d) < MIN_BARS:
            return out

        close = d["close"].values.astype(np.float64)
        high = d["high"].values.astype(np.float64)
        low = d["low"].values.astype(np.float64)
        volume = d["volume"].values.astype(np.float64)
        opn = d["open"].values.astype(np.float64)

        try:
            # Features 0-11: compact 12 structural features
            out[:12] = _compact_12_features(close, high, low, volume, opn)

            # ATR for normalisation
            atr_series = _atr(high, low, close, 14)
            atr_val = atr_series[-1]

            # Swing points
            sh_idx, sl_idx = self._find_swing_points(high, low)

            # 13. smc_bos: Break of Structure
            out[12] = self._detect_bos(close, high, low, sh_idx, sl_idx)

            # 14. smc_choch: Change of Character
            out[13] = self._detect_choch(close, high, low, sh_idx, sl_idx)

            # Order blocks
            bull_obs, bear_obs = self._find_order_blocks(opn, close, high, low)

            # 15. bullish_ob: near bullish OB within ob_proximity_pct
            out[14] = self._near_order_block(close[-1], bull_obs, self.ob_proximity_pct)

            # 16. bearish_ob: near bearish OB
            out[15] = self._near_order_block(close[-1], bear_obs, self.ob_proximity_pct)

            # FVGs (Fair Value Gaps)
            bull_fvgs, bear_fvgs = self._find_fvgs(high, low)

            # 17. bullish_fvg
            out[16] = 1.0 if len(bull_fvgs) > 0 else 0.0

            # 18. bearish_fvg
            out[17] = 1.0 if len(bear_fvgs) > 0 else 0.0

            # 19. swing_high_dist: normalised by ATR
            if len(sh_idx) > 0:
                last_sh = high[sh_idx[-1]]
                out[18] = np.clip((last_sh - close[-1]) / (atr_val + 1e-10), -5.0, 5.0)
            # else remains 0

            # 20. swing_low_dist
            if len(sl_idx) > 0:
                last_sl = low[sl_idx[-1]]
                out[19] = np.clip((close[-1] - last_sl) / (atr_val + 1e-10), -5.0, 5.0)

            # 21. structure_trend: HH/HL=1, LH/LL=-1, ranging=0
            out[20] = self._detect_structure_trend(high, low, sh_idx, sl_idx)

            # 22. ob_zone_strength: count of OBs in current zone (0-1)
            total_obs = len(bull_obs) + len(bear_obs)
            out[21] = np.clip(total_obs / 10.0, 0.0, 1.0)

            # 23. liquidity_above: swing high cluster count (normalised)
            out[22] = np.clip(len(sh_idx) / 10.0, 0.0, 1.0)

            # 24. liquidity_below: swing low cluster count (normalised)
            out[23] = np.clip(len(sl_idx) / 10.0, 0.0, 1.0)

            # 25. 4h_trend_score: weighted composite [-1,1]
            bos_bias = out[12]           # -1/0/1
            structure_bias = out[20]     # -1/0/1
            ema_bias = np.clip(out[0] / 3.0, -1.0, 1.0)
            rsi_bias = (out[1] - 0.5) * 2.0
            fvg_bias = out[16] - out[17]  # bull fvg - bear fvg
            out[24] = np.clip(
                0.30 * bos_bias
                + 0.25 * structure_bias
                + 0.20 * ema_bias
                + 0.15 * rsi_bias
                + 0.10 * fvg_bias,
                -1.0,
                1.0,
            )

        except Exception as exc:
            logger.debug("4H feature error: %s", exc)

        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 1H Features (30)
    # ------------------------------------------------------------------

    def compute_1h_features(self, df_1h: pd.DataFrame, at_idx: int) -> np.ndarray:
        """
        Compute 30 hourly (momentum) features.

        Features 1-12: compact structural features shared across TFs.
        Features 13-30: divergence, Wyckoff events, Stochastic, pivot levels,
                         momentum composite.

        Parameters
        ----------
        df_1h  : Full 1H OHLCV DataFrame.
        at_idx : Current bar index (inclusive).

        Returns
        -------
        np.ndarray of shape (30,), dtype float32.
        """
        out = np.zeros(N_1H, dtype=np.float32)
        d = df_1h.iloc[: at_idx + 1]
        if len(d) < MIN_BARS:
            return out

        close = d["close"].values.astype(np.float64)
        high = d["high"].values.astype(np.float64)
        low = d["low"].values.astype(np.float64)
        volume = d["volume"].values.astype(np.float64)
        opn = d["open"].values.astype(np.float64)

        try:
            # 0-11: compact structural features
            out[:12] = _compact_12_features(close, high, low, volume, opn)

            # RSI series for divergence
            rsi_vals = self._rsi_series(close, 14)

            # 13. macd_divergence
            _, _, macd_hist_series = self._macd_hist_series(close)
            out[12] = self._detect_momentum_divergence(close, macd_hist_series, lookback=20)

            # 14. rsi_divergence: price lower low but RSI higher low
            out[13] = self._detect_rsi_divergence(close, rsi_vals, lookback=20)

            # 15. volume_climax: volume > 3x 20-bar avg
            vol_avg_20 = np.nanmean(volume[-20:]) if len(volume) >= 20 else np.nanmean(volume)
            out[14] = 1.0 if volume[-1] > 3.0 * vol_avg_20 else 0.0

            # Wyckoff events
            spring, upthrust, climax = self._detect_wyckoff_events(close, high, low, volume)

            # 16. wyckoff_spring
            out[15] = 1.0 if spring else 0.0

            # 17. wyckoff_upthrust
            out[16] = 1.0 if upthrust else 0.0

            # 18. wyckoff_climax: 1=selling climax, -1=buying climax
            out[17] = float(climax)

            # 19-20. Stochastic K, D (normalised 0-1)
            k, d_val = _stochastic(high, low, close, k_period=14, d_period=3)
            out[18] = np.clip(k / 100.0, 0.0, 1.0)
            out[19] = np.clip(d_val / 100.0, 0.0, 1.0)

            # 21. stoch_state: OB=1, OS=-1, neutral=0
            if k > 80.0:
                out[20] = 1.0
            elif k < 20.0:
                out[20] = -1.0
            else:
                out[20] = 0.0

            # ATR for pivot proximity
            atr_series = _atr(high, low, close, 14)
            atr_val = atr_series[-1]

            # 22-23. Pivot high/low proximity (classical pivot = prev high/low midpoints)
            if len(close) >= 3:
                prev_high = np.max(high[-10:-1]) if len(high) >= 10 else high[-1]
                prev_low = np.min(low[-10:-1]) if len(low) >= 10 else low[-1]
                dist_to_res = abs(close[-1] - prev_high) / (atr_val + 1e-10)
                dist_to_sup = abs(close[-1] - prev_low) / (atr_val + 1e-10)
                out[21] = 1.0 if dist_to_res < 0.5 else 0.0  # within 0.5 ATR of resistance
                out[22] = 1.0 if dist_to_sup < 0.5 else 0.0  # within 0.5 ATR of support

            # 24. momentum_1h: ROC(5) clipped [-3, 3]
            if len(close) > 5:
                out[23] = np.clip((close[-1] / (close[-6] + 1e-12) - 1.0) * 100.0, -3.0, 3.0)

            # 25. ema_ribbon: (ema9 - ema21 - ema55) / close * 100
            ema9 = _ema(close, 9)
            ema21 = _ema(close, 21)
            ema55 = _ema(close, 55)
            out[24] = np.clip(
                (ema9[-1] - ema21[-1] - ema55[-1]) / (close[-1] + 1e-10) * 100.0, -5.0, 5.0
            )

            # 26. vol_delta_proxy: candle direction strength -1 to 1
            hl_rng = high[-1] - low[-1]
            out[25] = (close[-1] - opn[-1]) / (hl_rng + 1e-10) if hl_rng > 0 else 0.0
            out[25] = np.clip(out[25], -1.0, 1.0)

            # 27. trend_strength: abs(sma20 slope) normalised by price
            sma20 = _sma(close, 20)
            if not np.isnan(sma20[-1]) and not np.isnan(sma20[-2]):
                slope = abs(sma20[-1] - sma20[-2]) / (close[-1] + 1e-10) * 100.0
                out[26] = np.clip(slope, 0.0, 1.0)

            # 28. consecutive_bars: count of same-direction closes, normalised [-1,1]
            out[27] = self._consecutive_direction(close, max_count=10)

            # 29. bb_squeeze: BB width < historical avg → 0/1
            _, bb_mid, _ = _bollinger(close, 20)
            bb_std_now = np.nanstd(close[-20:])
            bb_width_hist = pd.Series(close).rolling(20).std().values
            bb_avg_width = np.nanmean(bb_width_hist[-100:]) if len(bb_width_hist) >= 100 else np.nanmean(bb_width_hist)
            out[28] = 1.0 if bb_std_now < bb_avg_width * 0.75 else 0.0

            # 30. 1h_momentum_score: weighted composite [-1,1]
            macd_bias = np.clip(out[12], -1.0, 1.0)
            stoch_bias = out[20]            # -1/0/1
            rsi_bias = (out[1] - 0.5) * 2.0
            consec_bias = out[27]
            vol_bias = out[25]
            out[29] = np.clip(
                0.25 * macd_bias
                + 0.25 * rsi_bias
                + 0.20 * stoch_bias
                + 0.15 * consec_bias
                + 0.15 * vol_bias,
                -1.0,
                1.0,
            )

        except Exception as exc:
            logger.debug("1H feature error: %s", exc)

        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 15M Features (35)
    # ------------------------------------------------------------------

    def compute_15m_features(self, df_15m: pd.DataFrame, at_idx: int) -> np.ndarray:
        """
        Compute 35 15-minute (entry trigger) features.

        Features 1-12: compact structural features.
        Features 13-35: micro RSI, MACD histogram, candle patterns, wick ratios,
                         volume spikes, scalp momentum, bar type flags, Keltner
                         position, price acceleration, ATR percentile, entry score.

        Parameters
        ----------
        df_15m  : Full 15M OHLCV DataFrame.
        at_idx  : Current bar index (inclusive).

        Returns
        -------
        np.ndarray of shape (35,), dtype float32.
        """
        out = np.zeros(N_15M, dtype=np.float32)
        d = df_15m.iloc[: at_idx + 1]
        if len(d) < MIN_BARS:
            return out

        close = d["close"].values.astype(np.float64)
        high = d["high"].values.astype(np.float64)
        low = d["low"].values.astype(np.float64)
        volume = d["volume"].values.astype(np.float64)
        opn = d["open"].values.astype(np.float64)

        try:
            # 0-11: compact structural features
            out[:12] = _compact_12_features(close, high, low, volume, opn)

            # ATR
            atr_series = _atr(high, low, close, 14)
            atr_val = atr_series[-1]

            # 13. micro_rsi: RSI(9)/100
            out[12] = np.clip(_rsi(close, 9) / 100.0, 0.0, 1.0)

            # 14. micro_macd_hist: MACD histogram normalised by price
            _, _, hist = _macd(close, fast=5, slow=13, signal=4)
            out[13] = np.clip(hist / (close[-1] * 0.001 + 1e-10), -3.0, 3.0)

            # 15. candle_pattern: composite [-1,1] (normalised from multi-class)
            out[14] = self._detect_candle_pattern(opn, high, low, close)

            # 16. wick_ratio_up: upper wick / (high - low)
            hl_rng = high[-1] - low[-1]
            if hl_rng > 0:
                body_top = max(opn[-1], close[-1])
                body_bot = min(opn[-1], close[-1])
                upper_wick = high[-1] - body_top
                lower_wick = body_bot - low[-1]
                out[15] = np.clip(upper_wick / (hl_rng + 1e-10), 0.0, 1.0)
                out[16] = np.clip(lower_wick / (hl_rng + 1e-10), 0.0, 1.0)
                out[17] = np.clip((close[-1] - opn[-1]) / (hl_rng + 1e-10), -1.0, 1.0)
            # else all remain 0

            # 19. volume_spike: volume / 20-bar avg - 1
            vol_avg_20 = np.nanmean(volume[-20:]) if len(volume) >= 20 else np.nanmean(volume)
            out[18] = np.clip(volume[-1] / (vol_avg_20 + 1e-10) - 1.0, -2.0, 3.0)

            # 20. micro_sr_pos: position between nearest micro S/R (0-1)
            micro_res = np.max(high[-10:])
            micro_sup = np.min(low[-10:])
            micro_rng = micro_res - micro_sup
            if micro_rng > 0:
                out[19] = np.clip((close[-1] - micro_sup) / micro_rng, 0.0, 1.0)

            # 21. breakout_strength: (close - prev range high) / ATR
            prev_range_high = np.max(high[-11:-1]) if len(high) >= 11 else high[-1]
            prev_range_low = np.min(low[-11:-1]) if len(low) >= 11 else low[-1]
            if close[-1] > prev_range_high:
                out[20] = np.clip((close[-1] - prev_range_high) / (atr_val + 1e-10), 0.0, 5.0)

            # 22. breakdown_strength
            if close[-1] < prev_range_low:
                out[21] = np.clip((prev_range_low - close[-1]) / (atr_val + 1e-10), 0.0, 5.0)

            # 23. scalp_momentum: ema3/ema8 - 1 normalised
            ema3 = _ema(close, 3)
            ema8 = _ema(close, 8)
            out[22] = np.clip((ema3[-1] / (ema8[-1] + 1e-10) - 1.0) * 100.0, -3.0, 3.0)

            # 24. recent_range: ATR / close * 100 (%)
            out[23] = np.clip(atr_val / (close[-1] + 1e-10) * 100.0, 0.0, 5.0)

            # 25. open_interest_proxy: institutional accumulation proxy
            # Approximated by trending volume with price consolidation
            out[24] = self._oi_proxy(close, volume)

            # 26. tick_direction: last 3 bar close directions, normalised [-1,1]
            if len(close) >= 4:
                dirs = np.sign(np.diff(close[-4:]))  # 3 diffs
                out[25] = np.clip(np.mean(dirs), -1.0, 1.0)

            # 27. pin_bar_bull: close near high + long lower wick
            if hl_rng > 0:
                close_vs_high = (high[-1] - close[-1]) / (hl_rng + 1e-10)
                body_size = abs(close[-1] - opn[-1]) / (hl_rng + 1e-10)
                lower_wick_ratio = (min(opn[-1], close[-1]) - low[-1]) / (hl_rng + 1e-10)
                out[26] = 1.0 if (close_vs_high < 0.2 and lower_wick_ratio > 0.5 and body_size < 0.4) else 0.0

                # 28. pin_bar_bear: close near low + long upper wick
                close_vs_low = (close[-1] - low[-1]) / (hl_rng + 1e-10)
                upper_wick_ratio = (high[-1] - max(opn[-1], close[-1])) / (hl_rng + 1e-10)
                out[27] = 1.0 if (close_vs_low < 0.2 and upper_wick_ratio > 0.5 and body_size < 0.4) else 0.0

            # 29. inside_bar
            if len(high) >= 2:
                out[28] = 1.0 if (high[-1] < high[-2] and low[-1] > low[-2]) else 0.0

            # 30. outside_bar
            if len(high) >= 2:
                out[29] = 1.0 if (high[-1] > high[-2] and low[-1] < low[-2]) else 0.0

            # 31. keltner_pos: (close - mid) / (upper - mid)
            kelt_upper, kelt_mid, _ = _keltner(high, low, close, 20, 1.5)
            kelt_half = (kelt_upper - kelt_mid) + 1e-10
            out[30] = np.clip((close[-1] - kelt_mid) / kelt_half, -1.5, 1.5)

            # 32. micro_trend: avg (close-open)/ATR for last 5 bars
            if len(close) >= 5:
                micro_trend_vals = (close[-5:] - opn[-5:]) / (atr_val + 1e-10)
                out[31] = np.clip(np.mean(micro_trend_vals), -3.0, 3.0)

            # 33. price_acceleration: change in ROC
            if len(close) >= 7:
                roc_now = close[-1] / (close[-4] + 1e-12) - 1.0
                roc_prev = close[-4] / (close[-7] + 1e-12) - 1.0
                out[32] = np.clip((roc_now - roc_prev) * 100.0, -3.0, 3.0)

            # 34. atr_percentile: current ATR vs 100-bar history, 0-1
            if len(atr_series) >= 10:
                hist_atr = atr_series[-min(100, len(atr_series)):]
                pct = np.sum(hist_atr <= atr_val) / len(hist_atr)
                out[33] = float(pct)

            # 35. 15m_entry_score: weighted entry trigger composite [-1,1]
            rsi_bias = (out[12] - 0.5) * 2.0
            macd_bias = np.clip(out[13] / 3.0, -1.0, 1.0)
            candle_bias = np.clip(out[14], -1.0, 1.0)
            vol_spike_bias = np.clip(out[18] / 3.0, -1.0, 1.0)
            tick_bias = out[25]
            breakout_net = out[20] - out[21]  # breakout - breakdown
            out[34] = np.clip(
                0.20 * rsi_bias
                + 0.20 * macd_bias
                + 0.15 * candle_bias
                + 0.15 * tick_bias
                + 0.15 * vol_spike_bias
                + 0.15 * breakout_net,
                -1.0,
                1.0,
            )

        except Exception as exc:
            logger.debug("15M feature error: %s", exc)

        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Alignment Features (4)
    # ------------------------------------------------------------------

    def compute_alignment(
        self, sig_1d: float, sig_4h: float, sig_1h: float
    ) -> np.ndarray:
        """
        Compute 4 cross-TF alignment features.

        Encodes the cascade hierarchy: when all TF trend scores align bullish
        (each = +1), overall_alignment = 1.0 (strongest entry signal).

        Parameters
        ----------
        sig_1d : Daily trend score (feature index 19 of compute_1d_features).
        sig_4h : 4H trend score (feature index 24 of compute_4h_features).
        sig_1h : 1H momentum score (feature index 29 of compute_1h_features).

        Returns
        -------
        np.ndarray of shape (4,), dtype float32.
        """
        out = np.zeros(N_ALIGN, dtype=np.float32)

        def _agree(a: float, b: float) -> float:
            """Agreement metric: +1 both bull, -1 both bear, 0 mixed."""
            if a > 0.1 and b > 0.1:
                return 1.0
            if a < -0.1 and b < -0.1:
                return -1.0
            return 0.0

        out[0] = _agree(sig_1d, sig_4h)   # align_1d_4h
        out[1] = _agree(sig_4h, sig_1h)   # align_4h_1h
        out[2] = _agree(sig_1h, sig_1h)   # align_1h_15m placeholder (15m score not passed here)
        out[3] = float(np.clip(np.mean(out[:3]), -1.0, 1.0))  # overall_alignment

        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def compute_alignment_full(
        self, sig_1d: float, sig_4h: float, sig_1h: float, sig_15m: float
    ) -> np.ndarray:
        """
        Compute 4 cross-TF alignment features with all four signals.

        Parameters
        ----------
        sig_1d  : Daily trend score.
        sig_4h  : 4H trend score.
        sig_1h  : 1H momentum score.
        sig_15m : 15M entry score.

        Returns
        -------
        np.ndarray of shape (4,), dtype float32.
        """
        out = np.zeros(N_ALIGN, dtype=np.float32)

        def _agree(a: float, b: float) -> float:
            if a > 0.1 and b > 0.1:
                return 1.0
            if a < -0.1 and b < -0.1:
                return -1.0
            return 0.0

        out[0] = _agree(sig_1d, sig_4h)    # align_1d_4h
        out[1] = _agree(sig_4h, sig_1h)    # align_4h_1h
        out[2] = _agree(sig_1h, sig_15m)   # align_1h_15m
        out[3] = float(np.clip(np.mean(out[:3]), -1.0, 1.0))

        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature name catalogue
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return ordered list of all 114 feature names (excludes 3 env position feats)."""
        names_1d = [
            "1d_sma_trend", "1d_sma200_dist", "1d_adx", "1d_adx_trend",
            "1d_macro_rsi", "1d_monthly_return", "1d_weekly_return",
            "1d_vol_regime", "1d_atr_regime", "1d_higher_high", "1d_higher_low",
            "1d_price_vs_range", "1d_ema_stack", "1d_trend_maturity",
            "1d_ichimoku_cloud_pos", "1d_wyckoff_phase", "1d_vol_expansion",
            "1d_doji_day", "1d_range_compression", "1d_daily_trend_score",
        ]
        names_4h = [
            "4h_ema_trend", "4h_rsi", "4h_mom_5", "4h_mom_10", "4h_mom_20",
            "4h_atr_ratio", "4h_vol_trend", "4h_macd_hist", "4h_bb_pos",
            "4h_sr_pos", "4h_body_ratio", "4h_trend_strength",
            "4h_smc_bos", "4h_smc_choch", "4h_bullish_ob", "4h_bearish_ob",
            "4h_bullish_fvg", "4h_bearish_fvg", "4h_swing_high_dist",
            "4h_swing_low_dist", "4h_structure_trend", "4h_ob_zone_strength",
            "4h_liquidity_above", "4h_liquidity_below", "4h_trend_score",
        ]
        names_1h = [
            "1h_ema_trend", "1h_rsi", "1h_mom_5", "1h_mom_10", "1h_mom_20",
            "1h_atr_ratio", "1h_vol_trend", "1h_macd_hist", "1h_bb_pos",
            "1h_sr_pos", "1h_body_ratio", "1h_trend_strength",
            "1h_macd_divergence", "1h_rsi_divergence", "1h_volume_climax",
            "1h_wyckoff_spring", "1h_wyckoff_upthrust", "1h_wyckoff_climax",
            "1h_stoch_k", "1h_stoch_d", "1h_stoch_state",
            "1h_pivot_high", "1h_pivot_low", "1h_momentum_roc5",
            "1h_ema_ribbon", "1h_vol_delta_proxy", "1h_trend_strength_slope",
            "1h_consecutive_bars", "1h_bb_squeeze", "1h_momentum_score",
        ]
        names_15m = [
            "15m_ema_trend", "15m_rsi", "15m_mom_5", "15m_mom_10", "15m_mom_20",
            "15m_atr_ratio", "15m_vol_trend", "15m_macd_hist", "15m_bb_pos",
            "15m_sr_pos", "15m_body_ratio", "15m_trend_strength",
            "15m_micro_rsi", "15m_micro_macd_hist", "15m_candle_pattern",
            "15m_wick_ratio_up", "15m_wick_ratio_down", "15m_body_strength",
            "15m_volume_spike", "15m_micro_sr_pos",
            "15m_breakout_strength", "15m_breakdown_strength",
            "15m_scalp_momentum", "15m_recent_range", "15m_oi_proxy",
            "15m_tick_direction", "15m_pin_bar_bull", "15m_pin_bar_bear",
            "15m_inside_bar", "15m_outside_bar", "15m_keltner_pos",
            "15m_micro_trend", "15m_price_acceleration", "15m_atr_percentile",
            "15m_entry_score",
        ]
        names_align = [
            "align_1d_4h", "align_4h_1h", "align_1h_15m", "overall_alignment",
        ]
        return names_1d + names_4h + names_1h + names_15m + names_align

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_swing_points(
        self, high: np.ndarray, low: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return indices of recent swing highs and lows."""
        lb = self.swing_lookback
        n = len(high)
        if n < 2 * lb + 1:
            return np.array([], dtype=int), np.array([], dtype=int)

        sh_idx: List[int] = []
        sl_idx: List[int] = []
        for i in range(lb, n - lb):
            if high[i] == np.max(high[max(0, i - lb): i + lb + 1]):
                sh_idx.append(i)
            if low[i] == np.min(low[max(0, i - lb): i + lb + 1]):
                sl_idx.append(i)
        return np.array(sh_idx, dtype=int), np.array(sl_idx, dtype=int)

    def _detect_bos(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        sh_idx: np.ndarray,
        sl_idx: np.ndarray,
    ) -> float:
        """Detect Break of Structure: +1 bullish, -1 bearish, 0 none."""
        if len(sh_idx) < 2 or len(sl_idx) < 2:
            return 0.0
        # Bullish BOS: close breaks above last swing high
        last_sh = high[sh_idx[-1]]
        if close[-1] > last_sh:
            return 1.0
        # Bearish BOS: close breaks below last swing low
        last_sl = low[sl_idx[-1]]
        if close[-1] < last_sl:
            return -1.0
        return 0.0

    def _detect_choch(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        sh_idx: np.ndarray,
        sl_idx: np.ndarray,
    ) -> float:
        """
        Change of Character: reversal of prevailing structure.
        Bullish (+1): price in downtrend creates LH/LL, then breaks a prior SH.
        Bearish (-1): price in uptrend creates HH/HL, then breaks a prior SL.
        """
        if len(sh_idx) < 3 or len(sl_idx) < 3:
            return 0.0
        # Bullish CHOCH: last two swing lows are lower (downtrend) but price now
        # breaks above the most recent swing high (shift in character)
        sl_vals = low[sl_idx[-2:]]
        if sl_vals[-1] < sl_vals[-2]:  # lower lows = downtrend
            if close[-1] > high[sh_idx[-1]]:
                return 1.0
        # Bearish CHOCH
        sh_vals = high[sh_idx[-2:]]
        if sh_vals[-1] > sh_vals[-2]:  # higher highs = uptrend
            if close[-1] < low[sl_idx[-1]]:
                return -1.0
        return 0.0

    def _find_order_blocks(
        self,
        opn: np.ndarray,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        lookback: int = 30,
    ) -> Tuple[List[float], List[float]]:
        """
        Identify bullish and bearish order blocks as mid-points.

        A bullish OB is the last bearish candle before a strong bullish move.
        A bearish OB is the last bullish candle before a strong bearish move.
        Returns lists of mid-price levels.
        """
        bull_obs: List[float] = []
        bear_obs: List[float] = []
        n = len(close)
        start = max(0, n - lookback)

        for i in range(start, n - 2):
            body_i = close[i] - opn[i]
            body_i1 = close[i + 1] - opn[i + 1]
            move = abs(close[i + 2] - close[i + 1]) if (i + 2) < n else 0.0
            atr_proxy = np.mean(high[start:] - low[start:]) + 1e-10

            # Bullish OB: bearish candle (i) then strong bullish push (i+1)
            if body_i < 0 and body_i1 > 0 and move > atr_proxy * 0.5:
                bull_obs.append((high[i] + low[i]) / 2.0)

            # Bearish OB: bullish candle (i) then strong bearish push (i+1)
            if body_i > 0 and body_i1 < 0 and move > atr_proxy * 0.5:
                bear_obs.append((high[i] + low[i]) / 2.0)

        return bull_obs, bear_obs

    def _near_order_block(
        self, price: float, ob_levels: List[float], proximity_pct: float
    ) -> float:
        """Return 1 if price is within proximity_pct of any OB level."""
        for lvl in ob_levels:
            if abs(price - lvl) / (lvl + 1e-10) < proximity_pct:
                return 1.0
        return 0.0

    def _find_fvgs(
        self, high: np.ndarray, low: np.ndarray, lookback: int = 20
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Fair Value Gaps (imbalance zones).

        Bullish FVG: low[i+2] > high[i]  → gap up (unfilled bullish imbalance).
        Bearish FVG: high[i+2] < low[i]  → gap down.
        Returns recent unfilled FVGs as (top, bottom) tuples.
        """
        bull_fvgs: List[Tuple[float, float]] = []
        bear_fvgs: List[Tuple[float, float]] = []
        n = len(high)
        start = max(0, n - lookback)

        for i in range(start, n - 2):
            if low[i + 2] > high[i]:
                # Check if current price is inside (unfilled)
                gap_top = low[i + 2]
                gap_bot = high[i]
                bull_fvgs.append((gap_top, gap_bot))
            if high[i + 2] < low[i]:
                gap_top = low[i]
                gap_bot = high[i + 2]
                bear_fvgs.append((gap_top, gap_bot))

        return bull_fvgs, bear_fvgs

    def _detect_structure_trend(
        self,
        high: np.ndarray,
        low: np.ndarray,
        sh_idx: np.ndarray,
        sl_idx: np.ndarray,
    ) -> float:
        """Return +1 (HH/HL), -1 (LH/LL), 0 (ranging)."""
        if len(sh_idx) < 2 or len(sl_idx) < 2:
            return 0.0
        last_sh = high[sh_idx[-1]]
        prev_sh = high[sh_idx[-2]]
        last_sl = low[sl_idx[-1]]
        prev_sl = low[sl_idx[-2]]

        higher_high = last_sh > prev_sh
        higher_low = last_sl > prev_sl
        lower_high = last_sh < prev_sh
        lower_low = last_sl < prev_sl

        if higher_high and higher_low:
            return 1.0
        if lower_high and lower_low:
            return -1.0
        return 0.0

    def _detect_wyckoff_phase_simple(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> float:
        """
        Simplified Wyckoff phase detection normalised to [-1, 1].

        0=unknown(0), 1=accumulation(0.5), 2=markup(1), 3=distribution(-0.5), 4=markdown(-1).
        Heuristic based on price trend + volume trend.
        """
        if len(close) < 20:
            return 0.0

        price_trend = (close[-1] - close[-20]) / (abs(close[-20]) + 1e-10)
        vol_trend = np.nanmean(volume[-5:]) / (np.nanmean(volume[-20:]) + 1e-10) - 1.0
        price_range = np.max(high[-20:]) - np.min(low[-20:])
        price_mid = (np.max(high[-20:]) + np.min(low[-20:])) / 2.0
        pos_in_range = (close[-1] - np.min(low[-20:])) / (price_range + 1e-10)

        if price_trend > 0.03 and close[-1] > price_mid:
            return 1.0   # markup
        if price_trend < -0.03 and close[-1] < price_mid:
            return -1.0  # markdown
        if pos_in_range < 0.4 and vol_trend < 0:
            return 0.5   # accumulation (low in range, decreasing volume)
        if pos_in_range > 0.6 and vol_trend < 0:
            return -0.5  # distribution (high in range, decreasing volume)
        return 0.0

    def _detect_wyckoff_events(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> Tuple[bool, bool, int]:
        """
        Detect spring, upthrust, and climax at the current bar.

        Returns
        -------
        (spring, upthrust, climax) where climax ∈ {-1, 0, 1}.
        """
        if len(close) < 20:
            return False, False, 0

        support = np.min(low[-20:-1])
        resistance = np.max(high[-20:-1])
        vol_avg = np.nanmean(volume[-20:])
        vol_spike = volume[-1] > (vol_avg * 2.0)

        price_range = high[-1] - low[-1]
        range_avg = np.nanmean(high[-20:] - low[-20:])

        close_near_low = (close[-1] - low[-1]) / (price_range + 1e-10) < 0.3
        close_near_high = (close[-1] - low[-1]) / (price_range + 1e-10) > 0.7
        wide_range = price_range > range_avg * 1.5

        # Spring: breaks below support but closes above it
        spring = bool(low[-1] < support and close[-1] > support)

        # Upthrust: breaks above resistance but closes below it
        upthrust = bool(high[-1] > resistance and close[-1] < resistance)

        # Climax: high volume + wide range
        climax = 0
        if vol_spike and wide_range:
            if close_near_low:
                climax = 1   # selling climax
            elif close_near_high:
                climax = -1  # buying climax

        return spring, upthrust, climax

    def _rsi_series(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI series (one value per bar)."""
        n = len(close)
        rsi_out = np.full(n, 50.0)
        if n < period + 1:
            return rsi_out
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-12)
        rsi_vals = 100.0 - 100.0 / (1.0 + rs)
        rsi_out[1:] = rsi_vals
        return rsi_out

    def _macd_hist_series(
        self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full MACD series arrays."""
        ema_fast = _ema(close, fast)
        ema_slow = _ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    def _detect_momentum_divergence(
        self, close: np.ndarray, hist: np.ndarray, lookback: int = 20
    ) -> float:
        """
        Detect MACD divergence over lookback bars.
        Bullish: price lower low, MACD histogram higher low → +1.
        Bearish: price higher high, MACD histogram lower high → -1.
        """
        if len(close) < lookback:
            return 0.0
        price_window = close[-lookback:]
        hist_window = hist[-lookback:]

        price_min_idx = int(np.argmin(price_window))
        price_max_idx = int(np.argmax(price_window))

        # Bullish divergence: recent bar's price is near the low, hist is not
        if price_min_idx < lookback - 5:
            if price_window[-1] < price_window[price_min_idx] * 1.005:
                if hist_window[-1] > hist_window[price_min_idx]:
                    return 1.0

        # Bearish divergence
        if price_max_idx < lookback - 5:
            if price_window[-1] > price_window[price_max_idx] * 0.995:
                if hist_window[-1] < hist_window[price_max_idx]:
                    return -1.0

        return 0.0

    def _detect_rsi_divergence(
        self, close: np.ndarray, rsi: np.ndarray, lookback: int = 20
    ) -> float:
        """
        Detect RSI divergence.
        Bullish: price lower low but RSI higher low → +1.
        Bearish: price higher high but RSI lower high → -1.
        """
        if len(close) < lookback:
            return 0.0
        price_window = close[-lookback:]
        rsi_window = rsi[-lookback:]

        # Find prior significant low
        price_low_idx = int(np.argmin(price_window[:-3]))
        if price_window[-1] < price_window[price_low_idx] * 1.002:
            if rsi_window[-1] > rsi_window[price_low_idx]:
                return 1.0

        # Find prior significant high
        price_high_idx = int(np.argmax(price_window[:-3]))
        if price_window[-1] > price_window[price_high_idx] * 0.998:
            if rsi_window[-1] < rsi_window[price_high_idx]:
                return -1.0

        return 0.0

    def _detect_candle_pattern(
        self,
        opn: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> float:
        """
        Detect candle pattern at last bar, returning normalised value in [-1, 1].

        hammer=1, shooting_star=-1, doji=0, engulfing_bull=0.75, engulfing_bear=-0.75.
        Multiple patterns → use highest absolute priority.
        """
        if len(close) < 2:
            return 0.0

        hl = high[-1] - low[-1]
        if hl < 1e-10:
            return 0.0
        body = close[-1] - opn[-1]
        body_abs = abs(body)
        body_frac = body_abs / hl
        upper_wick = high[-1] - max(close[-1], opn[-1])
        lower_wick = min(close[-1], opn[-1]) - low[-1]

        # Engulfing patterns (highest priority)
        if len(close) >= 2:
            prev_body = close[-2] - opn[-2]
            prev_body_abs = abs(prev_body)
            # Bullish engulfing
            if prev_body < 0 and body > 0 and body_abs > prev_body_abs * 1.1:
                return 0.75
            # Bearish engulfing
            if prev_body > 0 and body < 0 and body_abs > prev_body_abs * 1.1:
                return -0.75

        # Doji
        if body_frac < 0.1:
            return 0.0

        # Hammer: small body at top, long lower wick
        if lower_wick > 2.0 * body_abs and upper_wick < 0.3 * body_abs and close[-1] > opn[-1]:
            return 1.0

        # Shooting star: small body at bottom, long upper wick
        if upper_wick > 2.0 * body_abs and lower_wick < 0.3 * body_abs and close[-1] < opn[-1]:
            return -1.0

        # Bullish / bearish candle by body direction
        return float(np.sign(body)) * body_frac

    def _consecutive_direction(self, close: np.ndarray, max_count: int = 10) -> float:
        """
        Count consecutive same-direction closes, normalised to [-1, 1].

        Positive: consecutive up bars, negative: consecutive down bars.
        """
        if len(close) < 2:
            return 0.0
        diffs = np.sign(np.diff(close))
        direction = diffs[-1]
        if direction == 0.0:
            return 0.0
        count = 0
        for d in reversed(diffs):
            if d == direction:
                count += 1
            else:
                break
        return float(np.clip(direction * count / max_count, -1.0, 1.0))

    def _oi_proxy(self, close: np.ndarray, volume: np.ndarray, lookback: int = 20) -> float:
        """
        Institutional accumulation proxy via OBV-like volume-price agreement.

        Rising price + rising volume = accumulation (+1).
        Falling price + rising volume = distribution (-1).
        """
        if len(close) < lookback + 1:
            return 0.0
        price_change = close[-1] - close[-lookback]
        vol_trend = np.nanmean(volume[-5:]) / (np.nanmean(volume[-lookback:]) + 1e-10) - 1.0
        if price_change > 0 and vol_trend > 0:
            return np.clip(vol_trend, 0.0, 1.0)
        if price_change < 0 and vol_trend > 0:
            return np.clip(-vol_trend, -1.0, 0.0)
        return 0.0


# ---------------------------------------------------------------------------
# HTFDataAligner
# ---------------------------------------------------------------------------

class HTFDataAligner:
    """
    Resamples a 15-minute OHLCV DataFrame to 1H, 4H, and 1D timeframes.

    The 15M DataFrame must have a DatetimeIndex.  All resampled frames use
    standard OHLCV aggregation (open=first, high=max, low=min, close=last,
    volume=sum) and bars with all-NaN OHLCV are dropped.

    Usage
    -----
    aligner = HTFDataAligner()
    frames = aligner.align_timestamps(df_15m)
    # frames['15m'], frames['1h'], frames['4h'], frames['1d']

    parent_idx = aligner.get_parent_idx(frames['15m'], frames['1h'], child_idx=100)
    """

    _RESAMPLE_RULES: Dict[str, str] = {
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }

    _OHLCV_AGG = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    def align_timestamps(self, df_15m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Resample df_15m to all four standard timeframes.

        Parameters
        ----------
        df_15m : DataFrame with DatetimeIndex and OHLCV columns.

        Returns
        -------
        dict with keys '15m', '1h', '4h', '1d' → DataFrames.
        """
        if not isinstance(df_15m.index, pd.DatetimeIndex):
            raise ValueError("df_15m must have a DatetimeIndex.")

        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df_15m.columns]
        if not cols:
            raise ValueError("df_15m must contain at least one OHLCV column.")

        agg = {c: self._OHLCV_AGG[c] for c in cols if c in self._OHLCV_AGG}

        frames: Dict[str, pd.DataFrame] = {"15m": df_15m.copy()}

        for key in ("1h", "4h", "1d"):
            rule = self._RESAMPLE_RULES[key]
            resampled = df_15m.resample(rule).agg(agg).dropna(how="all")
            frames[key] = resampled

        logger.debug(
            "HTFDataAligner: 15m=%d 1h=%d 4h=%d 1d=%d bars",
            len(frames["15m"]),
            len(frames["1h"]),
            len(frames["4h"]),
            len(frames["1d"]),
        )
        return frames

    def get_parent_idx(
        self,
        df_child: pd.DataFrame,
        df_parent: pd.DataFrame,
        child_idx: int,
    ) -> int:
        """
        Return the parent-frame index corresponding to df_child.iloc[child_idx].

        Uses a searchsorted lookup — O(log n) per call.

        Parameters
        ----------
        df_child  : Child timeframe DataFrame (e.g. 15M).
        df_parent : Parent timeframe DataFrame (e.g. 1H).
        child_idx : Integer position in df_child.

        Returns
        -------
        int : Integer position in df_parent of the most recent parent bar
              whose timestamp is <= df_child.index[child_idx].
              Returns 0 if no parent bar precedes the child timestamp.
        """
        child_ts = df_child.index[child_idx]
        # searchsorted gives insertion point; subtract 1 to get last bar <= child_ts
        pos = df_parent.index.searchsorted(child_ts, side="right") - 1
        return int(max(0, pos))
