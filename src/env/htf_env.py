"""
HTF Trading Environment
=======================

Hierarchical Multi-Timeframe (HTF) Gymnasium trading environment for DRL.

Executes at 15-minute resolution.  At each step it reads pre-computed
hierarchical features across 1D / 4H / 1H / 15M timeframes plus cross-TF
alignment signals, building a 117-dimensional observation:

    [20 1D feats | 25 4H feats | 30 1H feats | 35 15M feats |
     4 align feats | 3 pos state]

Reward shaping rewards entries that align with the HTF cascade and penalises
counter-trend positions, idle inaction during strong setups, and drawdowns.
"""

import logging
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

try:
    from src.features.htf_features import HTFDataAligner, HTFFeatureEngine
    _HTF_AVAILABLE = True
except ImportError:
    _HTF_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observation dimension constants (must match htf_features.py)
# ---------------------------------------------------------------------------
N_1D = 20
N_4H = 25
N_1H = 30
N_15M = 35
N_ALIGN = 4
N_POS = 3
N_OBS = N_1D + N_4H + N_1H + N_15M + N_ALIGN + N_POS  # 117

# Indices of the per-TF "summary score" features inside each TF block.
# These are used to compute overall HTF alignment.
IDX_1D_SCORE = N_1D - 1          # feature index 19 within 1D block
IDX_4H_SCORE = N_4H - 1          # feature index 24 within 4H block
IDX_1H_SCORE = N_1H - 1          # feature index 29 within 1H block
IDX_15M_SCORE = N_15M - 1         # feature index 34 within 15M block

# Within the concatenated 114-feature vector (before pos state):
#   1D:  [0 .. 19]
#   4H:  [20 .. 44]
#   1H:  [45 .. 74]
#   15M: [75 .. 109]
#   align: [110 .. 113]
OFFSET_1D = 0
OFFSET_4H = N_1D
OFFSET_1H = N_1D + N_4H
OFFSET_15M = N_1D + N_4H + N_1H
OFFSET_ALIGN = N_1D + N_4H + N_1H + N_15M

# Position of the overall_alignment scalar inside the concatenated feature vec
# (last of the 4 alignment features)
IDX_OVERALL_ALIGN = OFFSET_ALIGN + 3  # index 113 in the 114-feature prefix


class HTFTradingEnv(gym.Env):
    """
    Hierarchical Multi-Timeframe Trading Environment.

    Parameters
    ----------
    df_15m : pd.DataFrame
        Required.  OHLCV DataFrame with DatetimeIndex at 15-minute resolution.
    df_1h : pd.DataFrame, optional
        1-hour OHLCV data.  Auto-resampled from df_15m when not provided.
    df_4h : pd.DataFrame, optional
        4-hour OHLCV data.  Auto-resampled from df_15m when not provided.
    df_1d : pd.DataFrame, optional
        Daily OHLCV data.  Auto-resampled from df_15m when not provided.
    initial_balance : float
        Starting portfolio balance (default 10 000).
    position_size : float
        Fraction of balance committed per trade (default 0.25).
    stop_loss_pct : float
        Stop-loss distance as a fraction of entry price (default 0.015 = 1.5%).
    take_profit_pct : float
        Take-profit distance as a fraction of entry price (default 0.03 = 3%).
    trading_fee : float
        Taker fee applied on open and close (default 0.0004 = 0.04%).
    lookback_window : int
        Minimum bars consumed before the first tradeable step (default 96).
    training_mode : bool
        When True randomises the episode start position for diversity.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_15m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None,
        initial_balance: float = 10_000.0,
        position_size: float = 0.25,
        stop_loss_pct: float = 0.015,
        take_profit_pct: float = 0.03,
        trading_fee: float = 0.0004,
        lookback_window: int = 96,
        training_mode: bool = True,
    ) -> None:
        super().__init__()

        if not _HTF_AVAILABLE:
            raise ImportError(
                "HTFFeatureEngine / HTFDataAligner could not be imported from "
                "src.features.htf_features.  Ensure the module is on the Python path."
            )

        # ---- hyperparameters ------------------------------------------------
        self.initial_balance = float(initial_balance)
        self.position_size = float(position_size)
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.trading_fee = float(trading_fee)
        self.lookback_window = int(lookback_window)
        self.training_mode = bool(training_mode)

        # ---- build / align timeframe DataFrames -----------------------------
        def _to_datetime_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
            """Ensure a DataFrame has a DatetimeIndex."""
            if isinstance(df.index, pd.DatetimeIndex):
                return df.copy()
            if "open_time" in df.columns:
                df = df.copy()
                df["open_time"] = pd.to_datetime(df["open_time"])
                return df.set_index("open_time")
            raise ValueError(f"{label} must have a DatetimeIndex or 'open_time' column.")

        self.df_15m = _to_datetime_index(df_15m, "df_15m")

        aligner = HTFDataAligner()
        aligned = aligner.align_timestamps(self.df_15m)

        self.df_15m = aligned["15m"]
        self.df_1h = _to_datetime_index(df_1h, "df_1h") if df_1h is not None else aligned["1h"]
        self.df_4h = _to_datetime_index(df_4h, "df_4h") if df_4h is not None else aligned["4h"]
        self.df_1d = _to_datetime_index(df_1d, "df_1d") if df_1d is not None else aligned["1d"]

        self._n_15m = len(self.df_15m)

        # ---- pre-cache parent-bar index arrays ------------------------------
        logger.info("HTFTradingEnv: caching parent-bar index arrays (%d 15M bars).", self._n_15m)
        self._idx_1h = np.empty(self._n_15m, dtype=np.int32)
        self._idx_4h = np.empty(self._n_15m, dtype=np.int32)
        self._idx_1d = np.empty(self._n_15m, dtype=np.int32)

        for i in range(self._n_15m):
            self._idx_1h[i] = aligner.get_parent_idx(self.df_15m, self.df_1h, i)
            self._idx_4h[i] = aligner.get_parent_idx(self.df_15m, self.df_4h, i)
            self._idx_1d[i] = aligner.get_parent_idx(self.df_15m, self.df_1d, i)

        # ---- pre-compute all features (critical for training speed) ---------
        logger.info("HTFTradingEnv: pre-computing features for all %d bars.", self._n_15m)
        self._engine = HTFFeatureEngine()
        self._precompute_all_features()

        # ---- action / observation spaces ------------------------------------
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Long, 2=Short
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(N_OBS,),
            dtype=np.float32,
        )

        # ---- initialise episode state (calls reset logic) -------------------
        self.balance: float = self.initial_balance
        self.position: int = 0
        self.position_price: float = 0.0
        self.position_size_units: float = 0.0
        self.position_entry_step: int = 0
        self.current_step: int = self.lookback_window
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.max_balance: float = self.initial_balance

    # =========================================================================
    # Pre-computation
    # =========================================================================

    def _precompute_all_features(self) -> None:
        """
        Pre-compute and cache feature arrays for every 15M bar.

        Stores:
            _feat_1d   : (N, 20) float32
            _feat_4h   : (N, 25) float32
            _feat_1h   : (N, 30) float32
            _feat_15m  : (N, 35) float32
            _feat_align: (N,  4) float32
            _htf_align_at: (N,) float32  — overall_alignment scalar per bar
        """
        n = self._n_15m
        feat_1d = np.zeros((n, N_1D), dtype=np.float32)
        feat_4h = np.zeros((n, N_4H), dtype=np.float32)
        feat_1h = np.zeros((n, N_1H), dtype=np.float32)
        feat_15m = np.zeros((n, N_15M), dtype=np.float32)
        feat_align = np.zeros((n, N_ALIGN), dtype=np.float32)

        eng = self._engine

        for i in range(n):
            idx_1d = int(self._idx_1d[i])
            idx_4h = int(self._idx_4h[i])
            idx_1h = int(self._idx_1h[i])

            f1d = eng.compute_1d_features(self.df_1d, idx_1d)
            f4h = eng.compute_4h_features(self.df_4h, idx_4h)
            f1h = eng.compute_1h_features(self.df_1h, idx_1h)
            f15m = eng.compute_15m_features(self.df_15m, i)
            align = eng.compute_alignment_full(
                float(f1d[IDX_1D_SCORE]),
                float(f4h[IDX_4H_SCORE]),
                float(f1h[IDX_1H_SCORE]),
                float(f15m[IDX_15M_SCORE]),
            )

            feat_1d[i] = f1d
            feat_4h[i] = f4h
            feat_1h[i] = f1h
            feat_15m[i] = f15m
            feat_align[i] = align

        self._feat_1d = feat_1d
        self._feat_4h = feat_4h
        self._feat_1h = feat_1h
        self._feat_15m = feat_15m
        self._feat_align = feat_align
        # overall_alignment is the 4th alignment feature (index 3)
        self._htf_align_at = feat_align[:, 3].astype(np.float32)

        logger.info("HTFTradingEnv: feature pre-computation complete.")

    # =========================================================================
    # Gymnasium interface
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if self.training_mode:
            max_start = self._n_15m - self.lookback_window - 100
            low_start = self.lookback_window
            if max_start <= low_start:
                self.current_step = low_start
            else:
                self.current_step = int(
                    self.np_random.integers(low_start, max_start)
                )
        else:
            self.current_step = self.lookback_window

        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0.0
        self.position_size_units = 0.0
        self.position_entry_step = 0

        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.max_balance = self.initial_balance

        return self._get_observation(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step (one 15M bar).

        Parameters
        ----------
        action : int
            0 = Hold, 1 = Long, 2 = Short.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        current_price: float = float(self.df_15m.iloc[self.current_step]["close"])
        reward: float = 0.0
        trade_made: bool = False

        # ---- HTF alignment at this bar (pre-computed) -----------------------
        htf_align: float = float(self._htf_align_at[self.current_step])

        # Derive higher-TF directional consensus from 1D + 4H trend scores
        sig_1d: float = float(self._feat_1d[self.current_step, IDX_1D_SCORE])
        sig_4h: float = float(self._feat_4h[self.current_step, IDX_4H_SCORE])
        htf_bearish: bool = sig_1d < -0.1 and sig_4h < -0.1
        htf_bullish: bool = sig_1d > 0.1 and sig_4h > 0.1

        steps_in_position: int = self.current_step - self.position_entry_step
        min_hold_steps: int = 4  # ~1 hour minimum hold for 15M bars

        # ---- execute action -------------------------------------------------
        if action == 1:  # Long
            if self.position == -1:  # close short first
                if steps_in_position < min_hold_steps:
                    reward -= 0.015
                pnl = self._close_position(current_price)
                reward += pnl
                trade_made = True

            if self.position == 0:
                # Counter-trend penalty: going long when 1D+4H are bearish
                if htf_bearish:
                    reward -= 0.015
                # HTF alignment multiplier on trade cost
                htf_mult = 1.0 + 0.5 * max(0.0, htf_align)
                self._open_position(current_price, 1)
                self.position_entry_step = self.current_step
                reward -= 0.003 * htf_mult  # weighted trade cost signal
                trade_made = True

        elif action == 2:  # Short
            if self.position == 1:  # close long first
                if steps_in_position < min_hold_steps:
                    reward -= 0.015
                pnl = self._close_position(current_price)
                reward += pnl
                trade_made = True

            if self.position == 0:
                # Counter-trend penalty: going short when 1D+4H are bullish
                if htf_bullish:
                    reward -= 0.015
                htf_mult = 1.0 + 0.5 * max(0.0, -htf_align)
                self._open_position(current_price, -1)
                self.position_entry_step = self.current_step
                reward -= 0.003 * htf_mult
                trade_made = True

        # ---- advance step ---------------------------------------------------
        self.current_step += 1
        done: bool = self.current_step >= self._n_15m - 1

        # ---- SL / TP check at the new bar -----------------------------------
        if self.position != 0 and not done:
            new_price: float = float(self.df_15m.iloc[self.current_step]["close"])

            if self.position == 1:
                pnl_pct: float = (new_price - self.position_price) / self.position_price
            else:
                pnl_pct = (self.position_price - new_price) / self.position_price

            if pnl_pct <= -self.stop_loss_pct:
                pnl = self._close_position(new_price)
                reward += pnl
                reward -= 0.05  # strong SL penalty
                trade_made = True

            elif pnl_pct >= self.take_profit_pct:
                pnl = self._close_position(new_price)
                reward += pnl
                # Bonus scaled by HTF alignment: aligned TP hit earns more
                align_bonus = 0.10 * (1.0 + 0.5 * abs(htf_align))
                reward += align_bonus
                trade_made = True

            else:
                # Unrealized PnL credit (0.2x) — encourages holding winners
                # Position-size-invariant: pure percentage, no scaling
                reward += pnl_pct * 0.2

        # ---- idle penalty when strong setup is being ignored ----------------
        if self.position == 0 and abs(htf_align) > 0.6:
            reward -= 0.0005

        # ---- equity tracking and drawdown penalty ---------------------------
        equity: float = self._calculate_equity()
        self.equity_curve.append(equity)
        if equity > self.max_balance:
            self.max_balance = equity

        drawdown: float = (self.max_balance - equity) / (self.max_balance + 1e-10)
        if drawdown > 0.05:
            reward -= drawdown * 0.1

        # ---- build observation ----------------------------------------------
        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        info: Dict = {
            "balance": self.balance,
            "equity": equity,
            "position": self.position,
            "htf_alignment": htf_align,
            "trade_made": trade_made,
        }

        return obs, float(reward), done, False, info

    # =========================================================================
    # Observation
    # =========================================================================

    def _get_observation(self) -> np.ndarray:
        """
        Build and return the 117-dim observation for the current step.

        Layout: [20 1D | 25 4H | 30 1H | 35 15M | 4 align | 3 pos]
        Reads directly from the pre-computed cache — no recomputation.
        """
        i = self.current_step
        feats_114 = np.concatenate([
            self._feat_1d[i],
            self._feat_4h[i],
            self._feat_1h[i],
            self._feat_15m[i],
            self._feat_align[i],
        ])  # shape (114,)

        # Position state
        current_price: float = float(self.df_15m.iloc[i]["close"])
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (current_price - self.position_price) / (self.position_price + 1e-10)
            else:
                unrealized_pnl = (self.position_price - current_price) / (self.position_price + 1e-10)
        else:
            unrealized_pnl = 0.0

        balance_ratio = (self.balance - self.initial_balance) / (self.initial_balance + 1e-10)

        pos_state = np.array([
            float(self.position),
            float(np.clip(unrealized_pnl, -0.5, 0.5)),
            float(np.clip(balance_ratio, -0.5, 0.5)),
        ], dtype=np.float32)

        obs = np.concatenate([feats_114, pos_state])  # (117,)
        return obs.astype(np.float32)

    # =========================================================================
    # HTF alignment helper
    # =========================================================================

    def _compute_htf_alignment(self) -> float:
        """
        Return the overall HTF alignment score [-1, 1] at the current step.

        Reads from the pre-computed cache.
        """
        return float(self._htf_align_at[self.current_step])

    # =========================================================================
    # Position management
    # =========================================================================

    def _open_position(self, price: float, direction: int) -> None:
        """
        Open a long (direction=+1) or short (direction=-1) position.

        Deducts the opening taker fee from balance.
        """
        trade_amount: float = self.balance * self.position_size
        fee: float = trade_amount * self.trading_fee
        self.balance -= fee
        self.position = direction
        self.position_price = price
        self.position_size_units = (trade_amount - fee) / (price + 1e-10)

    def _close_position(self, price: float) -> float:
        """
        Close the current position and credit balance.

        Returns
        -------
        float
            Normalised PnL: raw_pnl / initial_balance.
        """
        if self.position == 0:
            return 0.0

        if self.position == 1:
            raw_pnl: float = (price - self.position_price) * self.position_size_units
        else:
            raw_pnl = (self.position_price - price) * self.position_size_units

        # Taker fee on the closing leg (applied only to proceeds when profitable)
        fee: float = abs(price * self.position_size_units) * self.trading_fee
        raw_pnl -= fee

        self.balance += raw_pnl

        # Record trade for metrics
        entry_val: float = self.position_price * self.position_size_units + 1e-10
        self.trades.append({
            "direction": "long" if self.position == 1 else "short",
            "entry": self.position_price,
            "exit": price,
            "pnl": raw_pnl,
            "pnl_pct": raw_pnl / entry_val,
            "entry_step": self.position_entry_step,
            "exit_step": self.current_step,
        })

        self.position = 0
        self.position_price = 0.0
        self.position_size_units = 0.0

        # Position-size-invariant reward: percentage return on position value
        return raw_pnl / (entry_val + 1e-10)

    # =========================================================================
    # Equity
    # =========================================================================

    def _calculate_equity(self) -> float:
        """Return balance plus unrealised PnL."""
        if self.position == 0:
            return self.balance

        current_price: float = float(self.df_15m.iloc[self.current_step]["close"])
        if self.position == 1:
            unrealised: float = (current_price - self.position_price) * self.position_size_units
        else:
            unrealised = (self.position_price - current_price) * self.position_size_units

        return self.balance + unrealised

    # =========================================================================
    # Episode metrics
    # =========================================================================

    def get_episode_metrics(self) -> Dict:
        """
        Return a dictionary of episode performance statistics.

        Metrics include total return, Sharpe, Sortino, max drawdown, win rate,
        profit factor, trade count, and final balance.  Annualisation uses
        4 × 24 × 365 = 35 040 fifteen-minute bars per year.
        """
        equity_curve = np.array(self.equity_curve, dtype=np.float64)
        _BARS_PER_YEAR = 4 * 24 * 365  # 15-minute bars

        total_return: float = (
            (equity_curve[-1] - self.initial_balance) / (self.initial_balance + 1e-10)
        )

        returns: np.ndarray = (
            np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
            if len(equity_curve) > 1
            else np.zeros(1)
        )

        # Sharpe ratio
        ret_std = float(np.std(returns))
        if ret_std > 1e-12:
            sharpe: float = float(np.mean(returns)) / ret_std * np.sqrt(_BARS_PER_YEAR)
        else:
            sharpe = 0.0

        # Sortino ratio
        downside = returns[returns < 0]
        down_std = float(np.std(downside)) if len(downside) > 0 else 0.0
        if down_std > 1e-12:
            sortino: float = float(np.mean(returns)) / down_std * np.sqrt(_BARS_PER_YEAR)
        else:
            sortino = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown_series = (peak - equity_curve) / (peak + 1e-10)
        max_drawdown: float = float(np.max(drawdown_series))

        # Trade statistics
        n_trades: int = len(self.trades)
        if n_trades > 0:
            pnl_list = [t["pnl"] for t in self.trades]
            wins = sum(1 for p in pnl_list if p > 0)
            win_rate: float = wins / n_trades
            avg_trade_pnl: float = float(np.mean(pnl_list))
            gross_profit: float = sum(p for p in pnl_list if p > 0)
            gross_loss: float = abs(sum(p for p in pnl_list if p < 0))
            profit_factor: float = gross_profit / (gross_loss + 1e-10)
            avg_win: float = (
                float(np.mean([p for p in pnl_list if p > 0])) if wins > 0 else 0.0
            )
            avg_loss: float = (
                float(np.mean([p for p in pnl_list if p < 0]))
                if (n_trades - wins) > 0
                else 0.0
            )
        else:
            win_rate = 0.0
            avg_trade_pnl = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0

        return {
            "total_return": total_return,
            "total_return_pct": round(total_return * 100.0, 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "total_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": round(profit_factor, 4),
            "avg_trade_pnl": avg_trade_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "final_balance": float(equity_curve[-1]),
            "initial_balance": self.initial_balance,
        }

    # =========================================================================
    # Render
    # =========================================================================

    def render(self, mode: str = "human") -> None:
        equity = self._calculate_equity()
        step = self.current_step
        pos_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.position, "?")
        align = self._compute_htf_alignment()
        logger.info(
            "Step %d | %s @ %.4f | equity=%.2f | htf_align=%.3f",
            step,
            pos_str,
            self.position_price,
            equity,
            align,
        )


# =============================================================================
# Factory function
# =============================================================================

def create_htf_env(
    df_15m: pd.DataFrame,
    df_1h: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    df_1d: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None,
) -> HTFTradingEnv:
    """
    Factory function for HTFTradingEnv.

    Parameters
    ----------
    df_15m : pd.DataFrame
        Required 15-minute OHLCV data (DatetimeIndex).
    df_1h, df_4h, df_1d : pd.DataFrame, optional
        Higher timeframe data; auto-resampled from df_15m when omitted.
    config : dict, optional
        Override any HTFTradingEnv constructor keyword argument.
        Recognised keys: initial_balance, position_size, stop_loss_pct,
        take_profit_pct, trading_fee, lookback_window, training_mode.

    Returns
    -------
    HTFTradingEnv
    """
    kwargs: Dict = {}
    if config is not None:
        for key in (
            "initial_balance",
            "position_size",
            "stop_loss_pct",
            "take_profit_pct",
            "trading_fee",
            "lookback_window",
            "training_mode",
        ):
            if key in config:
                kwargs[key] = config[key]

    return HTFTradingEnv(
        df_15m=df_15m,
        df_1h=df_1h,
        df_4h=df_4h,
        df_1d=df_1d,
        **kwargs,
    )
