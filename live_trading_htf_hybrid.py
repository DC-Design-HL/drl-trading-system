#!/usr/bin/env python3
"""
HTF Hybrid Exit Strategy Paper Trade Bot (Strategy 3)
======================================================
Identical to live_trading_htf.py for ENTRY logic.
ONLY the exit logic differs: hybrid partial + trailing stop exits.

Exit rules:
  - At +1.5% unrealized → close 50% of position  (PARTIAL_CLOSE)
    - Move SL to break-even on remaining 50%
  - After partial close: trail remaining with 1% trailing stop behind peak price
    - Track peak_price (highest for LONG, lowest for SHORT)
    - trailing_sl = peak - 1% for LONG, peak + 1% for SHORT
    - SL only tightens, never loosens
  - Hard TP at +3.0% for remaining position
  - SL at -1.5% until the +1.5% threshold is hit

This bot runs ONLY in dry-run mode (paper trade).

Usage:
    python live_trading_htf_hybrid.py --dry-run --interval 900
"""

import sys
import os
import time
import json
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.features.htf_features import HTFFeatureEngine, HTFDataAligner
from src.data.multi_asset_fetcher import MultiAssetDataFetcher
from src.data.storage import get_storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("htf_hybrid")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
STATE_FILE = Path("logs/htf_hybrid_state.json")
TRADES_FILE = Path("logs/htf_hybrid_trades.json")

# Validated walk-forward configuration (50% position, Sharpe 3.85)
POSITION_SIZE = 0.50   # 50% of balance per trade
STOP_LOSS_PCT = 0.015  # 1.5% stop loss
TAKE_PROFIT_PCT = 0.030  # 3.0% take profit (final exit for remaining)
TRADING_FEE = 0.0004   # 0.04% taker fee

# Hybrid exit thresholds
PARTIAL_CLOSE_PCT = 0.015   # +1.5% → close 50%, move SL to break-even
TRAILING_STOP_PCT = 0.01    # 1% trailing stop behind peak (after partial close)
HARD_TP_PCT = 0.030         # +3.0% → close remaining position

# Anti-overtrading guards
COOLDOWN_SECONDS = 1800   # 30 min after a stopped-out trade
MIN_HOLD_SECONDS = 3600   # 1 hour minimum hold (HTF signal is slower)

# Actions from PPO (must match HTFTradingEnv)
ACTION_HOLD = 0
ACTION_LONG = 1
ACTION_SHORT = 2
ACTION_LABELS = {ACTION_HOLD: "HOLD", ACTION_LONG: "LONG", ACTION_SHORT: "SHORT"}

# Minimum confidence to act on a signal
MIN_CONFIDENCE = 0.45

# How many 15M bars to fetch (10 days = ~960 bars)
FETCH_DAYS = 12


# ---------------------------------------------------------------------------
# Model path resolution (identical to original)
# ---------------------------------------------------------------------------

def find_best_htf_model() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (model_path, vecnorm_path) for the best available HTF model.

    Search order:
      1. data/models/htf_walkforward_50pct_v2/ — best OOS Sharpe fold
      2. data/models/htf/best_model.zip        — any saved HTF model
      3. data/models/wfv2/BTCUSDT/ppo/fold_00/ — walk-forward fallback
    """
    root = Path("data/models")

    # 1. Walk-forward 50pct directory (8 folds)
    wfv_dir = root / "htf_walkforward_50pct_v2"
    if wfv_dir.exists():
        best_sharpe = -999.0
        best_model_path: Optional[Path] = None
        best_vecnorm_path: Optional[Path] = None

        for fold_dir in sorted(wfv_dir.iterdir()):
            result_file = fold_dir / "fold_result.json"
            model_zip = fold_dir / "best_model.zip"
            if not (result_file.exists() and model_zip.exists()):
                continue
            try:
                result = json.loads(result_file.read_text())
                oos_sharpe = float(result.get("oos_sharpe", result.get("sharpe", -999)))
                if oos_sharpe > best_sharpe:
                    best_sharpe = oos_sharpe
                    best_model_path = model_zip
                    vn = fold_dir / "vecnorm.pkl"
                    if not vn.exists():
                        vn = fold_dir / "best_model_vecnorm.pkl"
                    best_vecnorm_path = vn if vn.exists() else None
            except Exception:
                continue

        if best_model_path:
            logger.info(
                "HTF model: walk-forward best fold (OOS Sharpe %.2f) → %s",
                best_sharpe, best_model_path,
            )
            return best_model_path, best_vecnorm_path

    # 2. Generic HTF model directory
    htf_best = root / "htf" / "best_model.zip"
    if htf_best.exists():
        vn = root / "htf" / "best_model_vecnorm.pkl"
        logger.info("HTF model: data/models/htf/best_model.zip")
        return htf_best, vn if vn.exists() else None

    # 3. Walk-forward v2 fold_00
    wfv2_model = root / "wfv2" / "BTCUSDT" / "ppo" / "fold_00" / "best_model.zip"
    wfv2_vecnorm = root / "wfv2" / "BTCUSDT" / "ppo" / "fold_00" / "vecnorm.pkl"
    if wfv2_model.exists():
        logger.warning(
            "HTF-specific model not found. Falling back to wfv2/fold_00 — "
            "this model was NOT trained on 117-dim HTF features; inference "
            "will run in heuristic-only mode."
        )
        return wfv2_model, wfv2_vecnorm if wfv2_vecnorm.exists() else None

    logger.error("No usable model found. Bot will HOLD on every step.")
    return None, None


# ---------------------------------------------------------------------------
# HTF Hybrid Exit Bot
# ---------------------------------------------------------------------------

class HTFHybridBot:
    """
    Paper trading bot using Hierarchical Multi-Timeframe PPO agent
    with hybrid exit strategy: partial close + trailing stop.

    Same entry logic as HTFLiveBot; only exit logic differs.
    """

    def __init__(
        self,
        dry_run: bool = True,
        initial_balance: float = 10_000.0,
        interval_minutes: int = 15,
    ):
        self.symbol = SYMBOL
        self.dry_run = dry_run
        self.initial_balance = initial_balance
        self.interval_minutes = interval_minutes

        # Portfolio state
        self.balance = initial_balance
        self.position = 0        # 1 = LONG, -1 = SHORT, 0 = FLAT
        self.position_price = 0.0
        self.position_units = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.peak_price = 0.0
        self.current_price = 0.0
        self.realized_pnl = 0.0

        # Hybrid exit state
        self.partial_exits = 0       # 0 = no partial close yet, 1 = partial done
        self.original_units = 0.0
        self.trailing_active = False  # True after partial close

        # Anti-overtrading
        self.last_loss_time = 0.0
        self.last_entry_time = 0.0

        # Bookkeeping
        self.trades: List[Dict] = []
        self.start_time = datetime.now().isoformat()
        self._lock = threading.Lock()

        # Feature pipeline
        self.aligner = HTFDataAligner()
        self.feature_engine = HTFFeatureEngine()
        self.fetcher = MultiAssetDataFetcher()

        # Storage (shared with main bot)
        self.storage = get_storage()

        # Model
        self.model: Optional[PPO] = None
        self.vec_normalize: Optional[VecNormalize] = None
        self._model_path: Optional[Path] = None
        self._load_model()

        # Restore persisted state
        self._load_state()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "HTFHybridBot ready | symbol=%s dry_run=%s balance=%.2f",
            self.symbol, self.dry_run, self.balance,
        )

    # ------------------------------------------------------------------
    # Model loading (identical to original)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        model_path, vecnorm_path = find_best_htf_model()
        if model_path is None:
            logger.warning("No model loaded — bot will HOLD on every step.")
            return

        try:
            self.model = PPO.load(str(model_path))
            self._model_path = model_path
            logger.info("PPO model loaded from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load PPO model: %s", exc)
            self.model = None
            return

        if vecnorm_path and vecnorm_path.exists():
            try:
                import gymnasium as gym
                dummy_venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])
                self.vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_venv)
                self.vec_normalize.training = False
                self.vec_normalize.norm_reward = False
                logger.info("VecNormalize stats loaded from %s", vecnorm_path)
            except Exception as exc:
                logger.warning("Could not load VecNormalize: %s — raw obs used", exc)
                self.vec_normalize = None

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        state = {
            "symbol": self.symbol,
            "balance": self.balance,
            "position": self.position,
            "position_price": self.position_price,
            "position_units": self.position_units,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "peak_price": self.peak_price,
            "realized_pnl": self.realized_pnl,
            "last_loss_time": self.last_loss_time,
            "last_entry_time": self.last_entry_time,
            "start_time": self.start_time,
            "partial_exits": self.partial_exits,
            "original_units": self.original_units,
            "trailing_active": self.trailing_active,
            "model_path": str(self._model_path) if self._model_path else None,
            "updated_at": datetime.now().isoformat(),
            "strategy": "hybrid",
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            logger.error("Failed to save state: %s", exc)

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            state = json.loads(STATE_FILE.read_text())
            self.balance = float(state.get("balance", self.initial_balance))
            self.position = int(state.get("position", 0))
            self.position_price = float(state.get("position_price", 0.0))
            self.position_units = float(state.get("position_units", 0.0))
            self.sl_price = float(state.get("sl_price", 0.0))
            self.tp_price = float(state.get("tp_price", 0.0))
            self.peak_price = float(state.get("peak_price", 0.0))
            self.realized_pnl = float(state.get("realized_pnl", 0.0))
            self.last_loss_time = float(state.get("last_loss_time", 0.0))
            self.last_entry_time = float(state.get("last_entry_time", 0.0))
            self.start_time = state.get("start_time", self.start_time)
            self.partial_exits = int(state.get("partial_exits", 0))
            self.original_units = float(state.get("original_units", 0.0))
            self.trailing_active = bool(state.get("trailing_active", False))
            logger.info(
                "State restored: pos=%d price=%.2f balance=%.2f partial_exits=%d trailing=%s peak=%.2f",
                self.position, self.position_price, self.balance,
                self.partial_exits, self.trailing_active, self.peak_price,
            )
        except Exception as exc:
            logger.warning("Could not restore state: %s", exc)

    def _log_trade(self, trade: Dict) -> None:
        """Append trade to line-delimited JSON file and shared storage."""
        trade["strategy"] = "hybrid"
        self.trades.append(trade)
        try:
            with open(TRADES_FILE, "a") as f:
                f.write(json.dumps(trade) + "\n")
        except Exception as exc:
            logger.error("Failed to write trade log: %s", exc)
        try:
            self.storage.log_trade(trade)
        except Exception as exc:
            logger.debug("storage.log_trade failed: %s", exc)
        # Send alert
        self._send_alert(trade)

    def _fetch_market_signals(self, symbol: str = "BTCUSDT") -> Dict:
        """Fetch current market analysis signals from the local API server."""
        try:
            import requests as req
            resp = req.get(f"http://127.0.0.1:5001/api/market?symbol={symbol}", timeout=10)
            if resp.ok:
                return resp.json()
        except Exception as exc:
            logger.debug("Failed to fetch market signals: %s", exc)
        return {}

    def _build_signal_summary(self, market: Dict) -> Dict:
        """Extract a compact signal summary from market analysis data."""
        summary = {}
        regime = market.get("regime", {})
        if regime:
            summary["regime"] = {
                "state": regime.get("regime", "unknown"),
                "adx": regime.get("adx"),
                "trend": regime.get("trend_strength"),
            }
        whale = market.get("whale", {})
        if whale:
            summary["whale"] = {
                "direction": whale.get("direction", "NEUTRAL"),
                "score": whale.get("score", 0),
                "confidence": whale.get("confidence", 0),
            }
        funding = market.get("funding", {})
        if funding:
            summary["funding"] = {
                "rate": funding.get("rate"),
                "bias": funding.get("bias"),
            }
        of = market.get("order_flow", {})
        if of:
            summary["order_flow"] = {
                "bias": of.get("bias", "neutral"),
                "large_buys": of.get("large_buys", 0),
                "large_sells": of.get("large_sells", 0),
            }
        if market.get("price"):
            summary["price"] = market["price"]
        return summary

    def _send_alert(self, trade: Dict) -> None:
        """Write trade alert to SHARED htf_pending_alerts.jsonl with strategy tag."""
        try:
            market = self._fetch_market_signals(trade.get("symbol", "BTCUSDT"))
            signal_summary = self._build_signal_summary(market)

            alert_file = Path("logs/htf_pending_alerts.jsonl")
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            with open(alert_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "strategy": "hybrid",
                    "trade": trade,
                    "signals": signal_summary,
                    "position": {
                        "entry_price": self.position_price,
                        "sl_price": self.sl_price,
                        "tp_price": self.tp_price,
                        "units": self.position_units,
                        "original_units": self.original_units,
                        "partial_exits": self.partial_exits,
                        "trailing_active": self.trailing_active,
                        "peak_price": self.peak_price,
                        "direction": "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT",
                    },
                }) + "\n")
            logger.info("Trade alert queued: %s %s @ $%.2f [strategy=hybrid]",
                        trade.get("action", "?"), trade.get("symbol", "?"),
                        trade.get("price", 0))
        except Exception as exc:
            logger.warning("Failed to queue trade alert: %s", exc)

    # ------------------------------------------------------------------
    # Data fetching (identical to original)
    # ------------------------------------------------------------------

    def fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch recent 15M OHLCV data for BTCUSDT."""
        try:
            df = self.fetcher.fetch_asset(self.symbol, "15m", FETCH_DAYS)
            if df is None or df.empty:
                logger.warning("Empty 15M data returned")
                return None
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                else:
                    df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            logger.info("Fetched %d 15M bars for %s", len(df), self.symbol)
            return df
        except Exception as exc:
            logger.error("fetch_data failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Observation building (identical to original)
    # ------------------------------------------------------------------

    def compute_observation(self, df_15m: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Compute the 117-dim HTF observation from a 15M DataFrame.
        Layout: [20 1D | 25 4H | 30 1H | 35 15M | 4 align | 3 pos]
        """
        try:
            frames = self.aligner.align_timestamps(df_15m)
            df_1d = frames["1d"]
            df_4h = frames["4h"]
            df_1h = frames["1h"]
            df_15 = frames["15m"]

            if len(df_1d) < 5 or len(df_4h) < 10 or len(df_1h) < 20 or len(df_15) < 30:
                logger.warning(
                    "Insufficient bars: 1d=%d 4h=%d 1h=%d 15m=%d",
                    len(df_1d), len(df_4h), len(df_1h), len(df_15),
                )
                return None

            f1d = self.feature_engine.compute_1d_features(df_1d, len(df_1d) - 1)
            f4h = self.feature_engine.compute_4h_features(df_4h, len(df_4h) - 1)
            f1h = self.feature_engine.compute_1h_features(df_1h, len(df_1h) - 1)
            f15m = self.feature_engine.compute_15m_features(df_15, len(df_15) - 1)

            sig_1d = float(f1d[-1])
            sig_4h = float(f4h[-1])
            sig_1h = float(f1h[-1])
            sig_15m = float(f15m[-1])
            f_align = self.feature_engine.compute_alignment_full(
                sig_1d, sig_4h, sig_1h, sig_15m
            )

            feats_114 = np.concatenate([f1d, f4h, f1h, f15m, f_align])

            current_price = float(df_15.iloc[-1]["close"])
            self.current_price = current_price

            if self.position != 0 and self.position_price > 0:
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

            obs = np.concatenate([feats_114, pos_state]).astype(np.float32)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            return obs

        except Exception as exc:
            logger.error("compute_observation failed: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Inference (identical to original)
    # ------------------------------------------------------------------

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Run PPO inference on a 117-dim observation."""
        if self.model is None:
            return ACTION_HOLD, 0.0

        try:
            obs_2d = obs.reshape(1, -1)
            if self.vec_normalize is not None:
                try:
                    obs_2d = self.vec_normalize.normalize_obs(obs_2d)
                except Exception as exc:
                    logger.debug("VecNormalize failed: %s", exc)

            action, _ = self.model.predict(obs_2d, deterministic=True)
            action = int(action.item() if hasattr(action, "item") else action)
            confidence = self._compute_confidence(obs_2d)
            return action, confidence

        except Exception as exc:
            logger.error("get_action failed: %s", exc)
            return ACTION_HOLD, 0.0

    def _compute_confidence(self, obs_2d: np.ndarray) -> float:
        """Return max action probability from the policy distribution."""
        try:
            import torch
            with torch.no_grad():
                obs_tensor = self.model.policy.obs_to_tensor(obs_2d)[0]
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]
            return float(np.max(probs))
        except Exception:
            return 1.0 / 3.0

    # ------------------------------------------------------------------
    # HYBRID EXIT LOGIC
    # ------------------------------------------------------------------

    def _check_hybrid_exit(self, current_price: float) -> Optional[str]:
        """
        Hybrid exit logic combining partial close + trailing stop.

        Phase 1 (before partial close, partial_exits == 0):
          - SL at -1.5% from entry
          - At +1.5% → PARTIAL_CLOSE (close 50%), move SL to break-even
          - Hard TP at +3.0%

        Phase 2 (after partial close, trailing_active == True):
          - Track peak_price (highest for LONG, lowest for SHORT)
          - trailing_sl = peak - 1% for LONG, peak + 1% for SHORT
          - SL only tightens, never loosens
          - Hard TP at +3.0%

        Returns exit reason string or None.
        """
        if self.position == 0:
            return None

        entry = self.position_price
        if entry <= 0:
            return None

        # Compute unrealized profit %
        if self.position == 1:  # LONG
            profit_pct = (current_price - entry) / entry
        else:  # SHORT
            profit_pct = (entry - current_price) / entry

        # --- Hard TP: +3.0% on remaining → close all ---
        if profit_pct >= HARD_TP_PCT:
            return "TP"

        # --- Phase 2: Trailing stop after partial close ---
        if self.trailing_active:
            # Update peak price
            if self.position == 1:  # LONG — track highest
                if current_price > self.peak_price:
                    self.peak_price = current_price
                    logger.debug("Peak price updated (LONG): $%.2f", self.peak_price)

                # Trailing SL = peak - 1%
                trailing_sl = self.peak_price * (1.0 - TRAILING_STOP_PCT)

                # SL only tightens (moves up for LONG), never loosens
                if trailing_sl > self.sl_price:
                    old_sl = self.sl_price
                    self.sl_price = trailing_sl
                    logger.info(
                        "Trailing SL tightened (LONG): $%.2f → $%.2f (peak=$%.2f)",
                        old_sl, self.sl_price, self.peak_price,
                    )
                    self._save_state()

                # Check trailing SL hit
                if current_price <= self.sl_price:
                    return "TRAILING_SL"

            else:  # SHORT — track lowest
                if current_price < self.peak_price:
                    self.peak_price = current_price
                    logger.debug("Peak price updated (SHORT): $%.2f", self.peak_price)

                # Trailing SL = peak + 1%
                trailing_sl = self.peak_price * (1.0 + TRAILING_STOP_PCT)

                # SL only tightens (moves down for SHORT), never loosens
                if trailing_sl < self.sl_price:
                    old_sl = self.sl_price
                    self.sl_price = trailing_sl
                    logger.info(
                        "Trailing SL tightened (SHORT): $%.2f → $%.2f (peak=$%.2f)",
                        old_sl, self.sl_price, self.peak_price,
                    )
                    self._save_state()

                # Check trailing SL hit
                if current_price >= self.sl_price:
                    return "TRAILING_SL"

            return None

        # --- Phase 1: Before partial close ---

        # Partial close: +1.5% and no partial done yet
        if profit_pct >= PARTIAL_CLOSE_PCT and self.partial_exits == 0:
            return "PARTIAL_CLOSE"

        # Stop loss: -1.5% on full position (pre-partial)
        if self.position == 1:
            if self.sl_price > 0 and current_price <= self.sl_price:
                return "SL"
        elif self.position == -1:
            if self.sl_price > 0 and current_price >= self.sl_price:
                return "SL"

        return None

    def _execute_partial_close(self, current_price: float) -> Dict:
        """
        Execute a 50% partial position close.
        Move SL to break-even and activate trailing stop.

        Returns a trade dict for the partial close.
        """
        # Close 50% of original units
        close_units = self.original_units * 0.50

        # Safety: don't close more than remaining
        close_units = min(close_units, self.position_units)

        if close_units <= 0:
            return {}

        # Calculate PnL for closed portion
        if self.position == 1:
            raw_pnl = (current_price - self.position_price) * close_units
            action_str = "PARTIAL_CLOSE_LONG"
        else:
            raw_pnl = (self.position_price - current_price) * close_units
            action_str = "PARTIAL_CLOSE_SHORT"

        fee = current_price * close_units * TRADING_FEE
        net_pnl = raw_pnl - fee

        # Update balance: return capital from closed portion
        self.balance += (current_price * close_units) + net_pnl - raw_pnl
        self.realized_pnl += net_pnl

        # Reduce position units
        self.position_units -= close_units

        # Update partial exit state
        self.partial_exits = 1
        self.trailing_active = True

        # Move SL to break-even on remaining position
        old_sl = self.sl_price
        self.sl_price = self.position_price  # break-even

        # Initialize peak price tracking for trailing stop
        self.peak_price = current_price

        trade = {
            "action": action_str,
            "reason": "PARTIAL_CLOSE",
            "symbol": self.symbol,
            "entry_price": self.position_price,
            "exit_price": current_price,
            "units": close_units,
            "remaining_units": self.position_units,
            "original_units": self.original_units,
            "partial_exit_num": 1,
            "pnl": net_pnl,
            "new_sl": self.sl_price,
            "old_sl": old_sl,
            "peak_price": self.peak_price,
            "trailing_active": True,
            "confidence": 1.0,
            "timestamp": datetime.now().isoformat(),
            "agent": "htf",
            "strategy": "hybrid",
        }

        logger.info(
            "📊 %s (PARTIAL_CLOSE) @ $%.2f | closed=%.5f remaining=%.5f | "
            "pnl=$%.2f | SL moved to break-even $%.2f | trailing active | balance=$%.2f",
            action_str, current_price, close_units, self.position_units,
            net_pnl, self.sl_price, self.balance,
        )

        self._log_trade(trade)
        self._save_state()

        return trade

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_trade(self, action: int, confidence: float, current_price: float) -> Optional[Dict]:
        """Execute a trading decision. Returns a trade record dict if executed."""
        now = time.time()

        # ── Guard: cooldown after loss ──
        if self.last_loss_time > 0 and (now - self.last_loss_time) < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - (now - self.last_loss_time)
            logger.info("Cooldown active: %.0fs remaining — HOLD", remaining)
            return None

        # ── Guard: minimum hold time ──
        if self.position != 0 and (now - self.last_entry_time) < MIN_HOLD_SECONDS:
            remaining = MIN_HOLD_SECONDS - (now - self.last_entry_time)
            logger.info("Min hold: %.0fs remaining — HOLD", remaining)
            return None

        # ── Guard: confidence threshold ──
        if action != ACTION_HOLD and confidence < MIN_CONFIDENCE:
            logger.info("Low confidence %.2f < %.2f — HOLD", confidence, MIN_CONFIDENCE)
            return None

        trade: Optional[Dict] = None

        # ── CLOSE existing position if direction reverses ──
        if self.position == 1 and action == ACTION_SHORT:
            trade = self._close_position(current_price, "REVERSE_CLOSE_LONG", confidence)
        elif self.position == -1 and action == ACTION_LONG:
            trade = self._close_position(current_price, "REVERSE_CLOSE_SHORT", confidence)
        elif self.position != 0 and action == ACTION_HOLD:
            pass

        # ── OPEN new position ──
        if action == ACTION_LONG and self.position == 0:
            trade = self._open_position(current_price, 1, confidence)
        elif action == ACTION_SHORT and self.position == 0:
            trade = self._open_position(current_price, -1, confidence)

        return trade

    def _open_position(self, price: float, direction: int, confidence: float) -> Dict:
        trade_value = self.balance * POSITION_SIZE
        fee = trade_value * TRADING_FEE
        self.balance -= fee
        self.position = direction
        self.position_price = price
        self.position_units = (trade_value - fee) / (price + 1e-10)
        self.peak_price = price
        self.last_entry_time = time.time()

        # Reset hybrid exit state for new position
        self.partial_exits = 0
        self.original_units = self.position_units
        self.trailing_active = False

        if direction == 1:
            self.sl_price = price * (1.0 - STOP_LOSS_PCT)
            self.tp_price = price * (1.0 + TAKE_PROFIT_PCT)
            action_str = "OPEN_LONG"
        else:
            self.sl_price = price * (1.0 + STOP_LOSS_PCT)
            self.tp_price = price * (1.0 - TAKE_PROFIT_PCT)
            action_str = "OPEN_SHORT"

        trade = {
            "action": action_str,
            "symbol": self.symbol,
            "price": price,
            "units": self.position_units,
            "original_units": self.original_units,
            "trade_value": trade_value,
            "confidence": confidence,
            "sl": self.sl_price,
            "tp": self.tp_price,
            "pnl": 0.0,
            "timestamp": datetime.now().isoformat(),
            "agent": "htf",
            "strategy": "hybrid",
        }

        logger.info(
            "📈 %s @ $%.2f | units=%.5f | SL=$%.2f | TP=$%.2f | conf=%.2f [strategy=hybrid]",
            action_str, price, self.position_units, self.sl_price, self.tp_price, confidence,
        )

        self._log_trade(trade)
        self._save_state()

        return trade

    def _close_position(self, price: float, reason: str, confidence: float) -> Dict:
        """Close the ENTIRE remaining position."""
        if self.position == 0:
            return {}

        if self.position == 1:
            raw_pnl = (price - self.position_price) * self.position_units
            action_str = "CLOSE_LONG"
        else:
            raw_pnl = (self.position_price - price) * self.position_units
            action_str = "CLOSE_SHORT"

        fee = price * self.position_units * TRADING_FEE
        net_pnl = raw_pnl - fee
        self.balance += (price * self.position_units) + net_pnl - raw_pnl
        self.realized_pnl += net_pnl

        if net_pnl < 0:
            self.last_loss_time = time.time()

        trade = {
            "action": action_str,
            "reason": reason,
            "symbol": self.symbol,
            "entry_price": self.position_price,
            "exit_price": price,
            "units": self.position_units,
            "original_units": self.original_units,
            "partial_exits_completed": self.partial_exits,
            "trailing_was_active": self.trailing_active,
            "peak_price": self.peak_price,
            "pnl": net_pnl,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "agent": "htf",
            "strategy": "hybrid",
        }

        logger.info(
            "📉 %s @ $%.2f | pnl=$%.2f | reason=%s | balance=$%.2f [strategy=hybrid]",
            action_str, price, net_pnl, reason, self.balance,
        )

        # Reset position
        self.position = 0
        self.position_price = 0.0
        self.position_units = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.peak_price = 0.0
        self.partial_exits = 0
        self.original_units = 0.0
        self.trailing_active = False

        self._log_trade(trade)
        self._save_state()

        return trade

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def run_iteration(self) -> Dict:
        """Execute one full decision cycle. Returns a status dict."""
        t0 = time.time()
        status: Dict = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "action": "HOLD",
            "price": 0.0,
            "confidence": 0.0,
            "position": self.position,
            "balance": self.balance,
            "realized_pnl": self.realized_pnl,
            "partial_exits": self.partial_exits,
            "trailing_active": self.trailing_active,
            "peak_price": self.peak_price,
            "strategy": "hybrid",
            "error": None,
        }

        try:
            # 1. Fetch data
            df_15m = self.fetch_data()
            if df_15m is None or df_15m.empty:
                status["error"] = "No data"
                return status

            current_price = float(df_15m.iloc[-1]["close"])
            self.current_price = current_price
            status["price"] = current_price

            # 2. Check hybrid exit logic before computing new action
            exit_reason = self._check_hybrid_exit(current_price)
            if exit_reason and self.position != 0:
                logger.info("Exit triggered (%s) @ $%.2f", exit_reason, current_price)

                if exit_reason == "PARTIAL_CLOSE":
                    # Partial close — don't fully close position
                    trade = self._execute_partial_close(current_price)
                    status["action"] = trade.get("action", exit_reason)
                else:
                    # Full close (TP, SL, or TRAILING_SL)
                    trade = self._close_position(current_price, exit_reason, 1.0)
                    status["action"] = trade.get("action", "CLOSE")

                status["position"] = self.position
                status["balance"] = self.balance
                status["realized_pnl"] = self.realized_pnl
                status["partial_exits"] = self.partial_exits
                status["trailing_active"] = self.trailing_active
                status["peak_price"] = self.peak_price
                return status

            # 3. Compute observation
            obs = self.compute_observation(df_15m)
            if obs is None:
                status["error"] = "Observation failed"
                return status

            # 4. Get action from model
            action, confidence = self.get_action(obs)
            status["action"] = ACTION_LABELS[action]
            status["confidence"] = confidence

            logger.info(
                "Step: price=$%.2f action=%s conf=%.2f pos=%d bal=$%.2f "
                "partial=%d trailing=%s peak=$%.2f [strategy=hybrid]",
                current_price, ACTION_LABELS[action], confidence, self.position,
                self.balance, self.partial_exits, self.trailing_active, self.peak_price,
            )

            # 5. Execute trade
            trade = self.execute_trade(action, confidence, current_price)
            if trade:
                status["action"] = trade.get("action", ACTION_LABELS[action])

        except Exception as exc:
            logger.error("run_iteration error: %s", exc, exc_info=True)
            status["error"] = str(exc)

        status["position"] = self.position
        status["balance"] = self.balance
        status["realized_pnl"] = self.realized_pnl
        status["partial_exits"] = self.partial_exits
        status["trailing_active"] = self.trailing_active
        status["peak_price"] = self.peak_price
        status["elapsed_s"] = round(time.time() - t0, 2)
        return status

    # ------------------------------------------------------------------
    # Continuous loop
    # ------------------------------------------------------------------

    def run_loop(self, stop_event: Optional[threading.Event] = None) -> None:
        """Run the trading loop at `interval_minutes` cadence until stopped."""
        interval_secs = self.interval_minutes * 60
        logger.info(
            "HTF Hybrid exit loop started | interval=%dmin dry_run=%s",
            self.interval_minutes, self.dry_run,
        )

        while True:
            if stop_event and stop_event.is_set():
                logger.info("Stop event received — shutting down.")
                break

            with self._lock:
                status = self.run_iteration()

            logger.info("Iteration complete: %s", json.dumps(status, default=str))

            if stop_event:
                stop_event.wait(timeout=interval_secs)
            else:
                time.sleep(interval_secs)

    # ------------------------------------------------------------------
    # Status summary (for API)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Return current agent status."""
        with self._lock:
            unrealized_pnl = 0.0
            if self.position != 0 and self.position_price > 0 and self.current_price > 0:
                if self.position == 1:
                    unrealized_pnl = (self.current_price - self.position_price) * self.position_units
                else:
                    unrealized_pnl = (self.position_price - self.current_price) * self.position_units

            return {
                "symbol": self.symbol,
                "position": self.position,
                "position_label": {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.position, "FLAT"),
                "position_price": self.position_price,
                "position_units": self.position_units,
                "original_units": self.original_units,
                "partial_exits": self.partial_exits,
                "trailing_active": self.trailing_active,
                "peak_price": self.peak_price,
                "sl_price": self.sl_price,
                "tp_price": self.tp_price,
                "current_price": self.current_price,
                "balance": self.balance,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": self.realized_pnl + unrealized_pnl,
                "model_path": str(self._model_path) if self._model_path else None,
                "start_time": self.start_time,
                "trade_count": len(self.trades),
                "dry_run": self.dry_run,
                "strategy": "hybrid",
            }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HTF Hybrid Exit Strategy Paper Trade Bot")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Paper trading only (default: True)")
    parser.add_argument("--balance", type=float, default=10_000.0,
                        help="Initial balance in USDT (default: 10000)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Decision interval in minutes (default: 15)")
    parser.add_argument("--once", action="store_true",
                        help="Run a single iteration and exit")
    args = parser.parse_args()

    # This bot is ALWAYS dry-run (paper trade only)
    bot = HTFHybridBot(
        dry_run=True,
        initial_balance=args.balance,
        interval_minutes=args.interval,
    )

    if args.once:
        status = bot.run_iteration()
        print(json.dumps(status, indent=2, default=str))
        return

    stop_event = threading.Event()
    try:
        bot.run_loop(stop_event=stop_event)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping.")
        stop_event.set()


if __name__ == "__main__":
    main()
