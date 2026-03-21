#!/usr/bin/env python3
"""
HTF Live Trading Bot
====================
Runs the Hierarchical Multi-Timeframe (HTF) PPO agent for BTCUSDT.

Features across 4 timeframes (1D, 4H, 1H, 15M) → 117-dim observation.
Mirrors all decisions to Binance testnet when TESTNET_MIRROR=true.
Saves trades to the same MongoDB/JSON storage as the main system.

Usage:
    python live_trading_htf.py --dry-run              # Paper trading only
    python live_trading_htf.py --interval 15          # 15-minute decision loop
    python live_trading_htf.py --balance 5000         # Custom starting balance
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
logger = logging.getLogger("htf_live")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
STATE_FILE = Path("logs/htf_trading_state.json")
TRADES_FILE = Path("logs/htf_trades.json")

# Validated walk-forward configuration (50% position, Sharpe 3.85)
POSITION_SIZE = 0.50   # 50% of balance per trade
STOP_LOSS_PCT = 0.015  # 1.5% stop loss
TAKE_PROFIT_PCT = 0.030  # 3.0% take profit
TRADING_FEE = 0.0004   # 0.04% taker fee

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
# Model path resolution
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
                    # Companion vecnorm (two naming conventions)
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

    # 3. Walk-forward v2 fold_00 (oldest fallback — older PPO, not HTF-specific)
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
# HTF Live Bot
# ---------------------------------------------------------------------------

class HTFLiveBot:
    """
    Live trading bot using the Hierarchical Multi-Timeframe PPO agent.

    One instance trades BTCUSDT at 15-minute resolution.
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
        self.current_price = 0.0
        self.realized_pnl = 0.0

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

        # Testnet executor
        self.testnet_executor = None
        if os.getenv("TESTNET_MIRROR", "").lower() in ("1", "true", "yes"):
            self._init_testnet()

        # Restore persisted state
        self._load_state()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "HTFLiveBot ready | symbol=%s dry_run=%s balance=%.2f",
            self.symbol, self.dry_run, self.balance,
        )

    # ------------------------------------------------------------------
    # Model loading
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
    # Testnet executor
    # ------------------------------------------------------------------

    def _init_testnet(self) -> None:
        try:
            from src.api.testnet_executor import get_testnet_executor
            self.testnet_executor = get_testnet_executor()
            if self.testnet_executor:
                logger.info("Testnet mirror enabled")
            else:
                logger.warning("Testnet mirror: executor not available (keys missing?)")
        except Exception as exc:
            logger.warning("Testnet init failed: %s", exc)

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
            "realized_pnl": self.realized_pnl,
            "last_loss_time": self.last_loss_time,
            "last_entry_time": self.last_entry_time,
            "start_time": self.start_time,
            "model_path": str(self._model_path) if self._model_path else None,
            "updated_at": datetime.now().isoformat(),
        }
        # Save to local HTF state file
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            logger.error("Failed to save state: %s", exc)
        # Also update shared MongoDB state so Live Portfolio reflects HTF positions
        try:
            pos_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.position, "FLAT")
            shared_state = self.storage.load_state()
            if not shared_state:
                shared_state = {}
            assets = shared_state.get("assets", {})
            assets[self.symbol] = {
                "position": self.position,
                "position_label": pos_label,
                "entry_price": self.position_price,
                "units": self.position_units,
                "pnl": self.realized_pnl,
                "sl_price": self.sl_price,
                "tp_price": self.tp_price,
                "source": "htf_agent",
            }
            shared_state["assets"] = assets
            shared_state["total_balance"] = self.balance + sum(
                a.get("pnl", 0) for a in assets.values()
            )
            self.storage.save_state(shared_state)
        except Exception as exc:
            logger.debug("Failed to update shared state: %s", exc)

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
            self.realized_pnl = float(state.get("realized_pnl", 0.0))
            self.last_loss_time = float(state.get("last_loss_time", 0.0))
            self.last_entry_time = float(state.get("last_entry_time", 0.0))
            self.start_time = state.get("start_time", self.start_time)
            logger.info(
                "State restored: pos=%d price=%.2f balance=%.2f",
                self.position, self.position_price, self.balance,
            )
        except Exception as exc:
            logger.warning("Could not restore state: %s", exc)

    def _log_trade(self, trade: Dict) -> None:
        """Append trade to line-delimited JSON file and shared storage."""
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
        # Send Telegram alert immediately
        self._send_telegram_alert(trade)

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

        # Regime
        regime = market.get("regime", {})
        if regime:
            summary["regime"] = {
                "state": regime.get("regime", "unknown"),
                "adx": regime.get("adx"),
                "trend": regime.get("trend_strength"),
            }

        # Whale signals
        whale = market.get("whale", {})
        if whale:
            summary["whale"] = {
                "direction": whale.get("direction", "NEUTRAL"),
                "score": whale.get("score", 0),
                "confidence": whale.get("confidence", 0),
            }

        # Funding rate
        funding = market.get("funding", {})
        if funding:
            summary["funding"] = {
                "rate": funding.get("rate"),
                "bias": funding.get("bias"),
            }

        # Order flow
        of = market.get("order_flow", {})
        if of:
            summary["order_flow"] = {
                "bias": of.get("bias", "neutral"),
                "large_buys": of.get("large_buys", 0),
                "large_sells": of.get("large_sells", 0),
            }

        # Price
        if market.get("price"):
            summary["price"] = market["price"]

        return summary

    def _send_telegram_alert(self, trade: Dict) -> None:
        """Write trade alert with market signal context to pending alerts file."""
        try:
            # Fetch current market signals for context
            market = self._fetch_market_signals(trade.get("symbol", "BTCUSDT"))
            signal_summary = self._build_signal_summary(market)

            alert_file = Path("logs/htf_pending_alerts.jsonl")
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            with open(alert_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "trade": trade,
                    "signals": signal_summary,
                }) + "\n")
            logger.info("Trade alert queued: %s %s @ $%.2f",
                        trade.get("action", "?"), trade.get("symbol", "?"),
                        trade.get("price", 0))
        except Exception as exc:
            logger.warning("Failed to queue trade alert: %s", exc)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch recent 15M OHLCV data for BTCUSDT."""
        try:
            df = self.fetcher.fetch_asset(self.symbol, "15m", FETCH_DAYS)
            if df is None or df.empty:
                logger.warning("Empty 15M data returned")
                return None
            # Ensure DatetimeIndex
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
    # Observation building
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

            # Need at least MIN_BARS in each timeframe
            if len(df_1d) < 5 or len(df_4h) < 10 or len(df_1h) < 20 or len(df_15) < 30:
                logger.warning(
                    "Insufficient bars: 1d=%d 4h=%d 1h=%d 15m=%d",
                    len(df_1d), len(df_4h), len(df_1h), len(df_15),
                )
                return None

            # Compute features at the last available bar
            f1d = self.feature_engine.compute_1d_features(df_1d, len(df_1d) - 1)
            f4h = self.feature_engine.compute_4h_features(df_4h, len(df_4h) - 1)
            f1h = self.feature_engine.compute_1h_features(df_1h, len(df_1h) - 1)
            f15m = self.feature_engine.compute_15m_features(df_15, len(df_15) - 1)

            # Cross-TF alignment (uses last scores from each TF feature vector)
            sig_1d = float(f1d[-1])
            sig_4h = float(f4h[-1])
            sig_1h = float(f1h[-1])
            sig_15m = float(f15m[-1])
            f_align = self.feature_engine.compute_alignment_full(
                sig_1d, sig_4h, sig_1h, sig_15m
            )

            # Concatenate 114 market features
            feats_114 = np.concatenate([f1d, f4h, f1h, f15m, f_align])

            # Position state (3 dims — must match HTFTradingEnv._get_observation)
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
    # Inference
    # ------------------------------------------------------------------

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Run PPO inference on a 117-dim observation.

        Returns (action, confidence) where action ∈ {0, 1, 2}.
        Falls back to HOLD if no model is loaded.
        """
        if self.model is None:
            return ACTION_HOLD, 0.0

        try:
            obs_2d = obs.reshape(1, -1)

            # Apply VecNormalize if available
            if self.vec_normalize is not None:
                try:
                    obs_2d = self.vec_normalize.normalize_obs(obs_2d)
                except Exception as exc:
                    logger.debug("VecNormalize failed: %s", exc)

            action, _ = self.model.predict(obs_2d, deterministic=True)
            action = int(action.item() if hasattr(action, "item") else action)

            # Compute confidence (max action probability)
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
    # SL/TP check
    # ------------------------------------------------------------------

    def _check_sl_tp(self, current_price: float) -> Optional[str]:
        """Return 'SL' or 'TP' if the current position should be closed, else None."""
        if self.position == 0:
            return None
        if self.position == 1:  # LONG
            if self.sl_price > 0 and current_price <= self.sl_price:
                return "SL"
            if self.tp_price > 0 and current_price >= self.tp_price:
                return "TP"
        elif self.position == -1:  # SHORT
            if self.sl_price > 0 and current_price >= self.sl_price:
                return "SL"
            if self.tp_price > 0 and current_price <= self.tp_price:
                return "TP"
        return None

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_trade(self, action: int, confidence: float, current_price: float) -> Optional[Dict]:
        """
        Execute a trading decision.

        Returns a trade record dict if a trade was executed, else None.
        """
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
            # Just hold; SL/TP handled separately
            pass

        # ── OPEN new position ──
        if action == ACTION_LONG and self.position == 0:
            if trade:  # Already closed — open after brief gap
                pass
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
        self.last_entry_time = time.time()

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
            "trade_value": trade_value,
            "confidence": confidence,
            "sl": self.sl_price,
            "tp": self.tp_price,
            "pnl": 0.0,
            "timestamp": datetime.now().isoformat(),
            "agent": "htf",
        }

        logger.info(
            "📈 %s @ $%.2f | units=%.5f | SL=$%.2f | TP=$%.2f | conf=%.2f",
            action_str, price, self.position_units, self.sl_price, self.tp_price, confidence,
        )

        if not self.dry_run:
            self._log_trade(trade)
            self._save_state()
            self._mirror_testnet(trade)
        else:
            logger.info("[DRY RUN] Trade not executed")

        return trade

    def _close_position(self, price: float, reason: str, confidence: float) -> Dict:
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

        # Track loss time for cooldown
        if net_pnl < 0:
            self.last_loss_time = time.time()

        trade = {
            "action": action_str,
            "reason": reason,
            "symbol": self.symbol,
            "entry_price": self.position_price,
            "exit_price": price,
            "units": self.position_units,
            "pnl": net_pnl,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "agent": "htf",
        }

        logger.info(
            "📉 %s @ $%.2f | pnl=$%.2f | reason=%s | balance=$%.2f",
            action_str, price, net_pnl, reason, self.balance,
        )

        # Reset position
        self.position = 0
        self.position_price = 0.0
        self.position_units = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0

        if not self.dry_run:
            self._log_trade(trade)
            self._save_state()
            self._mirror_testnet(trade)
        else:
            logger.info("[DRY RUN] Close not recorded")

        return trade

    # ------------------------------------------------------------------
    # Testnet mirroring
    # ------------------------------------------------------------------

    def _mirror_testnet(self, trade: Dict) -> None:
        if not self.testnet_executor:
            return
        try:
            result = self.testnet_executor.mirror_trade(trade, {})
            if result and result.get("executed"):
                logger.info("Testnet mirror: %s @ $%.2f", trade["action"], trade.get("price", 0))
        except Exception as exc:
            logger.warning("Testnet mirror failed: %s", exc)

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

            # 2. Check SL/TP before computing new action
            exit_reason = self._check_sl_tp(current_price)
            if exit_reason and self.position != 0:
                logger.info("SL/TP triggered (%s) @ $%.2f", exit_reason, current_price)
                trade = self._close_position(current_price, exit_reason, 1.0)
                status["action"] = trade.get("action", "CLOSE")
                status["position"] = self.position
                status["balance"] = self.balance
                status["realized_pnl"] = self.realized_pnl
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
                "Step: price=$%.2f action=%s conf=%.2f pos=%d bal=$%.2f",
                current_price, ACTION_LABELS[action], confidence, self.position, self.balance,
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
        status["elapsed_s"] = round(time.time() - t0, 2)
        return status

    # ------------------------------------------------------------------
    # Continuous loop
    # ------------------------------------------------------------------

    def run_loop(self, stop_event: Optional[threading.Event] = None) -> None:
        """Run the trading loop at `interval_minutes` cadence until stopped."""
        interval_secs = self.interval_minutes * 60
        logger.info(
            "HTF trading loop started | interval=%dmin dry_run=%s",
            self.interval_minutes, self.dry_run,
        )

        while True:
            if stop_event and stop_event.is_set():
                logger.info("Stop event received — shutting down.")
                break

            with self._lock:
                status = self.run_iteration()

            logger.info("Iteration complete: %s", json.dumps(status, default=str))

            # Sleep until next bar
            if stop_event:
                stop_event.wait(timeout=interval_secs)
            else:
                time.sleep(interval_secs)

    # ------------------------------------------------------------------
    # Status summary (for API)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Return current agent status (safe to call from API thread)."""
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
            }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HTF Live Trading Bot")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Paper trading only (default: True)")
    parser.add_argument("--live", action="store_true",
                        help="Enable real trade execution (overrides --dry-run)")
    parser.add_argument("--balance", type=float, default=10_000.0,
                        help="Initial balance in USDT (default: 10000)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Decision interval in minutes (default: 15)")
    parser.add_argument("--once", action="store_true",
                        help="Run a single iteration and exit")
    args = parser.parse_args()

    dry_run = not args.live  # live flag disables dry_run

    bot = HTFLiveBot(
        dry_run=dry_run,
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
