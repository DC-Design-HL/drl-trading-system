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
from src.signals.bos_choch import MarketStructure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("htf_live")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = os.environ.get("HTF_SYMBOL", "BTCUSDT")
# STATE_FILE is set dynamically after --symbol is parsed (see _get_state_file())
TRADES_FILE = Path("logs/htf_trades.json")


def _get_state_file(symbol: str = None) -> Path:
    """Return symbol-specific state file path."""
    sym = symbol or SYMBOL
    if sym == "BTCUSDT":
        return Path("logs/htf_trading_state.json")  # backwards compat
    return Path(f"logs/htf_trading_state_{sym}.json")

# ── Risk Management (Fixed-Dollar-Risk Model) ──
# Risk pool = RISK_POOL_PCT of balance, divided into RISK_BUDGET_PARTS
# Each trade risks exactly (risk_pool / budget_parts) dollars.
RISK_POOL_PCT = 0.10       # 10% of balance is the risk pool
RISK_BUDGET_PARTS = 20     # Pool divided into 20 equal risk slots
MARGIN_MULTIPLIER = 3.0    # Margin per trade = 3× dollar risk (Dvir's rule)
LIQ_SL_BUFFER = 1.5        # Liquidation must be ≥1.5× further than SL from entry
MAX_LEVERAGE = 50          # Hard cap on leverage (Binance testnet limit)

STOP_LOSS_PCT = 0.015  # 1.5% stop loss
TAKE_PROFIT_PCT = 0.030  # 3.0% take profit
TRADING_FEE = 0.0004   # 0.04% taker fee

# Trailing stop configuration
TRAILING_BREAKEVEN_PCT = 0.01    # At +1% profit, move SL to break-even
TRAILING_LOCK_PCT = 0.02         # At +2% profit, lock 50% of profit
TRAILING_DISTANCE_PCT = 0.01     # Trail 1% behind peak

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

# Ranging regime filter: raise confidence threshold when ADX is low
RANGING_MIN_CONFIDENCE = 0.80  # Need higher conviction in ranging markets
RANGING_ADX_THRESHOLD = 20.0   # ADX below this = ranging

# Momentum exhaustion: skip entries when price is extended
EXHAUSTION_ATR_THRESHOLD = 3.0  # Skip if price > 3 ATR from 20-bar VWAP

# Phase 1 §3.5: Hard cap on notional per trade to prevent martingale compounding
FIXED_MAX_NOTIONAL = 3000.0  # USDT

# How many 15M bars to fetch (10 days = ~960 bars)
FETCH_DAYS = 12


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

def _find_best_fold_model(wf_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the best model (highest OOS Sharpe) in a walk-forward directory."""
    best_sharpe = -999.0
    best_model_path: Optional[Path] = None
    best_vecnorm_path: Optional[Path] = None

    for fold_dir in sorted(wf_dir.iterdir()):
        result_file = fold_dir / "fold_result.json"
        model_zip = fold_dir / "best_model.zip"
        if not fold_dir.is_dir():
            continue
        if not (result_file.exists() and model_zip.exists()):
            continue
        try:
            result = json.loads(result_file.read_text())
            oos_sharpe = float(result.get("oos_sharpe", result.get("sharpe", result.get("test_metrics", {}).get("sharpe_ratio", -999))))
            if oos_sharpe > best_sharpe:
                best_sharpe = oos_sharpe
                best_model_path = model_zip
                vn = fold_dir / "vecnorm.pkl"
                if not vn.exists():
                    vn = fold_dir / "best_model_vecnorm.pkl"
                if not vn.exists():
                    vn = fold_dir / "fold_model_vecnorm.pkl"
                best_vecnorm_path = vn if vn.exists() else None
        except Exception:
            continue

    if best_model_path:
        logger.info(
            "HTF model: walk-forward best fold (OOS Sharpe %.2f) → %s",
            best_sharpe, best_model_path,
        )
    return best_model_path, best_vecnorm_path


def find_best_htf_model(symbol: str = "BTCUSDT") -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (model_path, vecnorm_path) for the best available HTF model.

    Search order (symbol-aware):
      1. data/models/htf_walkforward_<asset>/ — best OOS Sharpe fold (e.g. htf_walkforward_eth)
      2. data/models/htf_walkforward_50pct_v2/ — BTC default walk-forward
      3. data/models/htf/best_model.zip        — any saved HTF model
      4. data/models/wfv2/BTCUSDT/ppo/fold_00/ — walk-forward fallback
    """
    root = Path("data/models")

    # Derive asset name from symbol (e.g. BTCUSDT -> btc, ETHUSDT -> eth)
    asset = symbol.replace("USDT", "").lower()

    # 1. Symbol-specific walk-forward directory (e.g. htf_walkforward_eth)
    symbol_wf_dir = root / f"htf_walkforward_{asset}"
    if symbol_wf_dir.exists():
        model_path, vecnorm_path = _find_best_fold_model(symbol_wf_dir)
        if model_path:
            return model_path, vecnorm_path

    # 2. Walk-forward 50pct directory (8 folds) — BTC default
    wfv_dir = root / "htf_walkforward_50pct_v2"
    if wfv_dir.exists():
        model_path, vecnorm_path = _find_best_fold_model(wfv_dir)
        if model_path:
            return model_path, vecnorm_path

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
        initial_balance: float = 5_000.0,
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

        # Phase 1 §3.5: Session balance — captured once at startup for stable position sizing
        self.session_balance = initial_balance

        # Phase 1 §3.9: MFE/MAE tracking per trade
        self.mfe_pct = 0.0   # Max Favorable Excursion (best unrealized %) since open
        self.mae_pct = 0.0   # Max Adverse Excursion (worst unrealized %, negative) since open

        # Phase 1 §3.3: Partial take-profit state
        self.sl_pct = 0.0               # SL % at open for R-multiple calculations
        self.initial_position_units = 0.0  # Total units at open (for partial size math)
        self.partial_tp_level = 0       # 0=none, 1=Level1 filled (40%), 2=Level2 filled (35%)
        self.partial_tp1_price = 0.0    # Level 1 TP price: entry ± 1.0 × sl_pct
        self.partial_tp2_price = 0.0    # Level 2 TP price: entry ± 2.0 × sl_pct

        # Phase 1 §3.4: Entry time tracking for time-based stagnant exit
        self.position_entry_time = 0.0

        # Cache of last fetched 15m DataFrame (used by _open_position for regime detection)
        self._last_df = None

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

        # BOS/CHOCH market structure detector
        # swing_lookback=8 for 5m candles (noisier than 15m, needs wider window)
        self.market_structure = MarketStructure(swing_lookback=8)
        self._last_structure_signals: Dict = {}

        # Phase 1 §3.2: Regime detector for adaptive SL/TP multipliers
        try:
            from src.features.regime_detector import MarketRegimeDetector
            self.regime_detector = MarketRegimeDetector()
        except Exception as _exc:
            logger.warning("Regime detector unavailable: %s — using fixed SL/TP", _exc)
            self.regime_detector = None

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

        # Symbol-specific state file
        self._state_file = _get_state_file(self.symbol)

        # Restore persisted state
        self._load_state()
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Sync balance from exchange on startup for testnet bots
        # (handles case where no state file exists yet)
        if not self.dry_run and not self._state_file.exists():
            self._sync_balance_from_exchange()

        logger.info(
            "HTFLiveBot ready | symbol=%s dry_run=%s balance=%.2f",
            self.symbol, self.dry_run, self.balance,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        model_path, vecnorm_path = find_best_htf_model(self.symbol)
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
                from gymnasium import spaces as gym_spaces
                # Create a dummy env whose observation space matches the model's
                # (117 dims for HTF features). Using CartPole (4 dims) caused a
                # shape mismatch that silently broke normalization.
                n_features = 117  # HTF feature count
                dummy_env = gym.Env()
                dummy_env.observation_space = gym_spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32,
                )
                dummy_env.action_space = gym_spaces.Discrete(3)  # HOLD/LONG/SHORT
                dummy_env.reset = lambda **kw: (np.zeros(n_features, dtype=np.float32), {})
                dummy_env.step = lambda a: (np.zeros(n_features, dtype=np.float32), 0.0, True, False, {})
                dummy_venv = DummyVecEnv([lambda: dummy_env])
                self.vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_venv)
                self.vec_normalize.training = False
                self.vec_normalize.norm_reward = False
                logger.info("✅ VecNormalize stats loaded from %s (obs_shape=%s)",
                            vecnorm_path, self.vec_normalize.observation_space.shape)
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
                mode = "futures (real longs/shorts)" if getattr(self.testnet_executor, '_futures_executor', None) else "spot"
                logger.info("Testnet mirror enabled (%s)", mode)

                # On startup: verify SL/TP orders exist for all open exchange positions.
                # The FuturesTestnetExecutor already does this in __init__, but log it
                # explicitly so the HTF bot startup log shows the verification happened.
                futures_exec = getattr(self.testnet_executor, '_futures_executor', None)
                if futures_exec:
                    logger.info(
                        "Startup SL/TP verification: SL orders=%s, TP orders=%s",
                        dict(futures_exec._sl_orders),
                        dict(futures_exec._tp_orders),
                    )
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
            "peak_price": self.peak_price,
            "realized_pnl": self.realized_pnl,
            "last_loss_time": self.last_loss_time,
            "last_entry_time": self.last_entry_time,
            "start_time": self.start_time,
            "model_path": str(self._model_path) if self._model_path else None,
            "session_balance": self.session_balance,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
            "sl_pct": self.sl_pct,
            "initial_position_units": self.initial_position_units,
            "partial_tp_level": self.partial_tp_level,
            "partial_tp1_price": self.partial_tp1_price,
            "partial_tp2_price": self.partial_tp2_price,
            "position_entry_time": self.position_entry_time,
            "updated_at": datetime.now().isoformat(),
        }
        # Save to local HTF state file
        try:
            self._state_file.write_text(json.dumps(state, indent=2))
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
        if not self._state_file.exists():
            return
        try:
            state = json.loads(self._state_file.read_text())
            # Guard: only load state that belongs to THIS symbol
            saved_symbol = state.get("symbol", self.symbol)
            if saved_symbol != self.symbol:
                logger.warning(
                    "State file symbol mismatch: file has %s, bot is %s — ignoring stale state",
                    saved_symbol, self.symbol,
                )
                return
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
            # Phase 1 new fields — defaults keep existing state files loading correctly
            self.session_balance = float(state.get("session_balance", self.balance))
            self.mfe_pct = float(state.get("mfe_pct", 0.0))
            self.mae_pct = float(state.get("mae_pct", 0.0))
            self.sl_pct = float(state.get("sl_pct", 0.0))
            self.initial_position_units = float(state.get("initial_position_units", self.position_units))
            self.partial_tp_level = int(state.get("partial_tp_level", 0))
            self.partial_tp1_price = float(state.get("partial_tp1_price", 0.0))
            self.partial_tp2_price = float(state.get("partial_tp2_price", 0.0))
            self.position_entry_time = float(state.get("position_entry_time", 0.0))
            logger.info(
                "State restored: pos=%d price=%.2f balance=%.2f",
                self.position, self.position_price, self.balance,
            )
            # Sync balance from exchange for testnet bots (not dry_run)
            if not self.dry_run:
                self._sync_balance_from_exchange()
            # If in a position but SL/TP are missing, recalculate from entry price
            if self.position != 0 and self.position_price > 0:
                if self.sl_price <= 0:
                    if self.position == 1:  # LONG: SL below entry
                        self.sl_price = round(self.position_price * (1 - STOP_LOSS_PCT), 2)
                    else:  # SHORT: SL above entry
                        self.sl_price = round(self.position_price * (1 + STOP_LOSS_PCT), 2)
                    logger.warning(
                        "⚠️ SL was $0 for %s position — recalculated from entry: SL=$%.2f",
                        "LONG" if self.position == 1 else "SHORT", self.sl_price,
                    )
                if self.tp_price <= 0:
                    if self.position == 1:  # LONG: TP above entry
                        self.tp_price = round(self.position_price * (1 + TAKE_PROFIT_PCT), 2)
                    else:  # SHORT: TP below entry
                        self.tp_price = round(self.position_price * (1 - TAKE_PROFIT_PCT), 2)
                    logger.warning(
                        "⚠️ TP was $0 for %s position — recalculated from entry: TP=$%.2f",
                        "LONG" if self.position == 1 else "SHORT", self.tp_price,
                    )
                if self.peak_price <= 0:
                    self.peak_price = self.position_price
                self._save_state()
            # Check liquidation safety on state restore (if in a position)
            if self.position != 0:
                self._check_liquidation_safety()
        except Exception as exc:
            logger.warning("Could not restore state: %s", exc)

    def _sync_balance_from_exchange(self) -> None:
        """
        Fetch real USDT balance from Binance Futures testnet and override
        the internally tracked balance.

        This prevents the bot from using inflated balances accumulated during
        training/simulation. Only called for testnet bots (not dry_run).
        """
        try:
            from src.api.futures_executor import get_futures_executor
            executor = get_futures_executor()
            if executor is None:
                logger.warning("Balance sync: futures executor unavailable, keeping state balance")
                return

            real_balance = executor.get_account_balance('USDT')
            if real_balance <= 0:
                logger.warning(
                    "Balance sync: exchange returned $%.2f — keeping state balance $%.2f",
                    real_balance, self.balance,
                )
                return

            old_balance = self.balance
            if abs(old_balance - real_balance) > 1.0:  # Only log if meaningful difference
                logger.info(
                    "💰 Balance sync: state=$%.2f → exchange=$%.2f (corrected)",
                    old_balance, real_balance,
                )
            else:
                logger.info(
                    "💰 Balance sync: state=$%.2f ≈ exchange=$%.2f (in sync)",
                    old_balance, real_balance,
                )

            self.balance = real_balance
            # Keep initial_balance as the original testnet starting balance ($5,000)
            # so that PnL % in alerts reflects total performance, not per-session.
            # session_balance is used for position sizing only.
            self.session_balance = real_balance  # Update session balance for position sizing cap
            self._save_state()
        except Exception as exc:
            logger.warning("Balance sync failed: %s — keeping state balance $%.2f", exc, self.balance)

    def _sync_position_from_exchange(self) -> None:
        """
        Sync position state with the actual exchange position.
        The exchange is the source of truth — if it shows no position for
        this symbol, the bot resets to flat (fixes ghost SL/TP alerts).
        """
        if self.dry_run or self.position == 0:
            return
        try:
            from src.api.futures_executor import get_futures_executor
            executor = get_futures_executor()
            if executor is None:
                return
            positions = executor.connector.get_positions()
            # Find our symbol's position on the exchange
            exchange_pos = None
            for p in positions:
                if p.get("symbol") == self.symbol:
                    exchange_pos = p
                    break

            if exchange_pos is None:
                # Symbol not even in exchange response — position is flat
                real_amt = 0.0
            else:
                real_amt = abs(float(exchange_pos.get("positionAmt", 0)))

            if real_amt == 0.0 and self.position != 0:
                # Exchange says flat but bot thinks it has a position — reset
                logger.warning(
                    "⚠️ STALE POSITION DETECTED: bot has %s %s but exchange is FLAT. "
                    "Resetting to flat (likely closed on exchange side).",
                    "LONG" if self.position == 1 else "SHORT",
                    self.symbol,
                )
                # Record a synthetic close trade for tracking
                close_action = "CLOSE_LONG" if self.position == 1 else "CLOSE_SHORT"
                self.position = 0
                self.position_price = 0.0
                self.position_units = 0.0
                self.sl_price = 0.0
                self.tp_price = 0.0
                self.peak_price = 0.0
                self.partial_tp_level = 0
                self.partial_tp1_price = 0.0
                self.partial_tp2_price = 0.0
                self.mfe_pct = 0.0
                self.mae_pct = 0.0
                self._save_state()
                logger.info("✅ Bot state reset to FLAT for %s", self.symbol)
                return

            if real_amt > 0:
                # Check direction mismatch (bot says LONG but exchange has SHORT, or vice versa)
                raw_amt = float(exchange_pos.get("positionAmt", 0))
                exchange_dir = 1 if raw_amt > 0 else -1  # positive = LONG, negative = SHORT
                if exchange_dir != self.position:
                    exchange_entry = float(exchange_pos.get("entryPrice", 0))
                    logger.warning(
                        "⚠️ DIRECTION MISMATCH: bot has %s but exchange has %s %s (entry=$%.2f, qty=%.6f). "
                        "Syncing to exchange state.",
                        "LONG" if self.position == 1 else "SHORT",
                        "LONG" if exchange_dir == 1 else "SHORT",
                        self.symbol, exchange_entry, real_amt,
                    )
                    self.position = exchange_dir
                    self.position_price = exchange_entry if exchange_entry > 0 else self.position_price
                    self.position_units = real_amt
                    self.peak_price = self.position_price
                    # Recalculate SL/TP for the correct direction
                    if exchange_dir == 1:  # LONG
                        self.sl_price = round(self.position_price * (1 - STOP_LOSS_PCT), 2)
                        self.tp_price = round(self.position_price * (1 + TAKE_PROFIT_PCT), 2)
                    else:  # SHORT
                        self.sl_price = round(self.position_price * (1 + STOP_LOSS_PCT), 2)
                        self.tp_price = round(self.position_price * (1 - TAKE_PROFIT_PCT), 2)
                    self._save_state()
                    logger.info(
                        "✅ Synced to exchange: %s %s @ $%.2f, SL=$%.2f, TP=$%.2f",
                        "LONG" if exchange_dir == 1 else "SHORT",
                        self.symbol, self.position_price, self.sl_price, self.tp_price,
                    )
                elif abs(real_amt - self.position_units) > 0.0001:
                    logger.info(
                        "📐 Position sync: units=%.6f → exchange=%.6f (corrected)",
                        self.position_units, real_amt,
                    )
                    self.position_units = real_amt
                    self._save_state()
        except Exception as exc:
            logger.warning("Position sync failed: %s", exc)

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
                    "strategy": "htf",
                    "trade": trade,
                    "signals": signal_summary,
                    "position": {
                        "entry_price": self.position_price,
                        "sl_price": self.sl_price,
                        "tp_price": self.tp_price,
                        "units": self.position_units,
                        "direction": "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT",
                    },
                }, default=str) + "\n")
            logger.info("Trade alert queued: %s %s @ $%.2f",
                        trade.get("action", "?"), trade.get("symbol", "?"),
                        trade.get("price", 0))
        except Exception as exc:
            logger.warning("Failed to queue trade alert: %s", exc)

    def _write_sltp_update_alert(
        self, sl_changed: bool, tp_changed: bool,
        old_sl: float, new_sl: float,
        old_tp: float, new_tp: float,
        reason: str,
    ) -> None:
        """Write SL/TP update alert to the shared alerts file."""
        try:
            alert_file = Path("logs/htf_pending_alerts.jsonl")
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            direction = "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT"

            # Compute current profit %
            profit_pct = 0.0
            if self.position_price > 0 and self.current_price > 0:
                if self.position == 1:
                    profit_pct = ((self.current_price - self.position_price) / self.position_price) * 100
                elif self.position == -1:
                    profit_pct = ((self.position_price - self.current_price) / self.position_price) * 100

            with open(alert_file, "a") as f:
                if sl_changed and old_sl != new_sl:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "strategy": "htf",
                        "trade": {
                            "type": "SL_UPDATE",
                            "update_type": "SL",
                            "symbol": self.symbol,
                            "old_price": old_sl,
                            "new_price": new_sl,
                            "reason": reason,
                            "direction": direction,
                            "profit_pct": round(profit_pct, 2),
                            "current_price": self.current_price,
                        },
                        "signals": {},
                        "position": {
                            "entry_price": self.position_price,
                            "sl_price": new_sl,
                            "tp_price": self.tp_price,
                            "units": self.position_units,
                            "direction": direction,
                        },
                    }) + "\n")

                if tp_changed and old_tp != new_tp:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "strategy": "htf",
                        "trade": {
                            "type": "TP_UPDATE",
                            "update_type": "TP",
                            "symbol": self.symbol,
                            "old_price": old_tp,
                            "new_price": new_tp,
                            "reason": reason,
                            "direction": direction,
                            "profit_pct": round(profit_pct, 2),
                            "current_price": self.current_price,
                        },
                        "signals": {},
                        "position": {
                            "entry_price": self.position_price,
                            "sl_price": self.sl_price,
                            "tp_price": new_tp,
                            "units": self.position_units,
                            "direction": direction,
                        },
                    }) + "\n")

            logger.info("SL/TP update alert queued for %s", self.symbol)
        except Exception as exc:
            logger.warning("Failed to queue SL/TP update alert: %s", exc)

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
    # BOS/CHOCH market structure data
    # ------------------------------------------------------------------

    def _fetch_structure_candles(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch 5m, 1H and 4H OHLCV candles for BOS/CHOCH multi-timeframe analysis.

        Returns (df_5m, df_1h, df_4h).  Any may be None on fetch failure.
        """
        df_5m = None
        df_1h = None
        df_4h = None

        # 5m candles — primary BOS/CHOCH timeframe (faster signal detection)
        try:
            df_5m = self.fetcher.fetch_asset(self.symbol, "5m", days=FETCH_DAYS)
            if df_5m is not None and not df_5m.empty:
                if not isinstance(df_5m.index, pd.DatetimeIndex):
                    if "timestamp" in df_5m.columns:
                        df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])
                        df_5m = df_5m.set_index("timestamp")
                    else:
                        df_5m.index = pd.to_datetime(df_5m.index)
                df_5m = df_5m.sort_index()
                logger.info("Fetched %d 5m bars for BOS/CHOCH", len(df_5m))
        except Exception as exc:
            logger.warning("Failed to fetch 5m candles for BOS/CHOCH: %s", exc)

        try:
            df_1h = self.fetcher.fetch_asset(self.symbol, "1h", days=FETCH_DAYS)
            if df_1h is not None and not df_1h.empty:
                if not isinstance(df_1h.index, pd.DatetimeIndex):
                    if "timestamp" in df_1h.columns:
                        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"])
                        df_1h = df_1h.set_index("timestamp")
                    else:
                        df_1h.index = pd.to_datetime(df_1h.index)
                df_1h = df_1h.sort_index()
                logger.debug("Fetched %d 1H bars for BOS/CHOCH", len(df_1h))
        except Exception as exc:
            logger.debug("Failed to fetch 1H candles: %s", exc)

        try:
            df_4h = self.fetcher.fetch_asset(self.symbol, "4h", days=FETCH_DAYS)
            if df_4h is not None and not df_4h.empty:
                if not isinstance(df_4h.index, pd.DatetimeIndex):
                    if "timestamp" in df_4h.columns:
                        df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"])
                        df_4h = df_4h.set_index("timestamp")
                    else:
                        df_4h.index = pd.to_datetime(df_4h.index)
                df_4h = df_4h.sort_index()
                logger.debug("Fetched %d 4H bars for BOS/CHOCH", len(df_4h))
        except Exception as exc:
            logger.debug("Failed to fetch 4H candles: %s", exc)

        return df_5m, df_1h, df_4h

    def _get_structure_signals(self, trigger: bool = False) -> Dict:
        """
        Get the latest BOS/CHOCH signals.  Caches the result so it is only
        recomputed once per iteration (not on every WS tick).

        If *trigger* is True, a fresh analysis is run using 5m candles as
        the primary timeframe and 1H/4H for multi-timeframe confirmation.
        Otherwise the cached result from the last iteration is returned.
        """
        if trigger:
            try:
                df_5m, df_1h, df_4h = self._fetch_structure_candles()
                if df_5m is None or df_5m.empty:
                    logger.warning("BOS/CHOCH: no 5m data available — skipping")
                    self._last_structure_signals = {}
                else:
                    self._last_structure_signals = self.market_structure.get_signals(
                        df_5m, df_1h=df_1h, df_4h=df_4h,
                    )
                    # Log BOS/CHOCH signal state every iteration at INFO level
                    sig = self._last_structure_signals
                    logger.info(
                        "BOS/CHOCH signals: bos_bull=%s bos_bear=%s choch_bull=%s "
                        "choch_bear=%s fake_bos=%s fake_choch=%s trend=%s "
                        "swing_high=$%.2f swing_low=$%.2f conf=%.2f",
                        sig.get("bos_bullish", False),
                        sig.get("bos_bearish", False),
                        sig.get("choch_bullish", False),
                        sig.get("choch_bearish", False),
                        sig.get("fake_bos", False),
                        sig.get("fake_choch", False),
                        sig.get("trend", "unknown"),
                        sig.get("last_swing_high", 0.0),
                        sig.get("last_swing_low", 0.0),
                        sig.get("confidence", 0.0),
                    )
            except Exception as exc:
                logger.warning("BOS/CHOCH signal computation failed: %s", exc)
                self._last_structure_signals = {}
        return self._last_structure_signals

    # ------------------------------------------------------------------
    # Liquidation price safety check
    # ------------------------------------------------------------------

    def _get_liquidation_price(self) -> float:
        """
        Fetch the current liquidation price from the exchange via the
        testnet executor. Returns 0.0 if unavailable, dry-run, or no
        exchange connection.
        """
        if self.dry_run or not self.testnet_executor:
            return 0.0
        try:
            futures_exec = getattr(self.testnet_executor, '_futures_executor', None)
            if futures_exec and hasattr(futures_exec, 'get_liquidation_price'):
                return futures_exec.get_liquidation_price(self.symbol)
        except Exception as exc:
            logger.debug("Failed to fetch liquidation price: %s", exc)
        return 0.0

    def _check_liquidation_safety(self) -> bool:
        """
        Validate that the SL price triggers BEFORE the liquidation price.

        For LONG:  liq price is BELOW entry → danger if liq >= SL - delta
        For SHORT: liq price is ABOVE entry → danger if liq <= SL + delta

        Uses the liquidation price from the exchange to infer actual direction
        (avoids false alerts from bot/exchange direction desyncs).

        Returns True if safe, False if at risk.
        """
        if self.position == 0 or self.position_price <= 0 or self.sl_price <= 0:
            return True

        liq_price = self._get_liquidation_price()

        # Skip if liquidation is 0 (1x leverage — impossible to liquidate)
        if liq_price <= 0:
            return True

        delta = self.position_price * 0.01  # 1% buffer

        # Determine actual direction from liquidation price vs entry price.
        # LONG: liquidation is below entry (you get liquidated if price drops)
        # SHORT: liquidation is above entry (you get liquidated if price rises)
        # This avoids false alerts when bot direction is desynced from exchange.
        if liq_price < self.position_price:
            # Liq below entry → this is a LONG position
            if liq_price >= self.sl_price - delta:
                buffer = self.sl_price - delta - liq_price
                buffer_pct = buffer / self.position_price * 100 if self.position_price > 0 else 0
                logger.critical(
                    "⚠️ LIQUIDATION RISK: %s LONG liq=$%.2f >= SL-delta=$%.2f "
                    "(SL=$%.2f, entry=$%.2f, buffer=$%.2f / %.2f%%)",
                    self.symbol, liq_price, self.sl_price - delta,
                    self.sl_price, self.position_price, buffer, buffer_pct,
                )
                self._write_liquidation_alert(liq_price, delta)
                return False
        else:
            # Liq above entry → this is a SHORT position
            if liq_price <= self.sl_price + delta:
                buffer = liq_price - (self.sl_price + delta)
                buffer_pct = buffer / self.position_price * 100 if self.position_price > 0 else 0
                logger.critical(
                    "⚠️ LIQUIDATION RISK: %s SHORT liq=$%.2f <= SL+delta=$%.2f "
                    "(SL=$%.2f, entry=$%.2f, buffer=$%.2f / %.2f%%)",
                    self.symbol, liq_price, self.sl_price + delta,
                    self.sl_price, self.position_price, buffer, buffer_pct,
                )
                self._write_liquidation_alert(liq_price, delta)
                return False

        return True

    def _write_liquidation_alert(self, liq_price: float, delta: float) -> None:
        """Write a LIQUIDATION_RISK alert to the shared alerts file for Telegram."""
        try:
            alert_file = Path("logs/htf_pending_alerts.jsonl")
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            direction = "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT"

            if self.position == 1:
                buffer = self.sl_price - liq_price
            else:
                buffer = liq_price - self.sl_price
            buffer_pct = buffer / self.position_price * 100 if self.position_price > 0 else 0

            with open(alert_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "strategy": "htf",
                    "trade": {
                        "type": "LIQUIDATION_RISK",
                        "symbol": self.symbol,
                        "direction": direction,
                        "liquidation_price": liq_price,
                        "sl_price": self.sl_price,
                        "entry_price": self.position_price,
                        "delta": delta,
                        "buffer": buffer,
                        "buffer_pct": buffer_pct,
                    },
                    "signals": {},
                    "position": {
                        "entry_price": self.position_price,
                        "sl_price": self.sl_price,
                        "tp_price": self.tp_price,
                        "units": self.position_units,
                        "direction": direction,
                    },
                }) + "\n")
            logger.info("Liquidation risk alert queued for %s %s", self.symbol, direction)
        except Exception as exc:
            logger.warning("Failed to queue liquidation risk alert: %s", exc)

    # ------------------------------------------------------------------
    # SL/TP check
    # ------------------------------------------------------------------

    def _update_peak_price(self, current_price: float) -> None:
        """Track the most-favorable price since entry (highest for LONG, lowest for SHORT)."""
        if self.position == 0:
            return
        if self.position == 1:  # LONG — track highest
            if current_price > self.peak_price:
                self.peak_price = current_price
        elif self.position == -1:  # SHORT — track lowest
            if self.peak_price == 0.0 or current_price < self.peak_price:
                self.peak_price = current_price

    def _check_sl_tp(self, current_price: float) -> Optional[str]:
        """
        Dynamic SL/TP logic enhanced with BOS/CHOCH market structure signals.

        Layered approach:
          1. Hard TP check (unchanged at configured TP level).
          2. Basic trailing stop (break-even at +1%, lock 50% at +2%).
          3. BOS/CHOCH overlay — **only when the position is currently
             profitable** (unrealized PnL > 0).

        BOS/CHOCH rules (profitable positions only):
          LONG:
            - BOS bullish  → trail SL to last swing low + buffer, extend TP
            - CHOCH bearish → tighten SL to lock 75% of profit
            - Fake BOS bearish / Fake CHOCH bullish → ignore
          SHORT:
            - BOS bearish  → trail SL to last swing high - buffer, extend TP
            - CHOCH bullish → tighten SL to lock 75% of profit
            - Fake BOS bullish / Fake CHOCH bearish → ignore

        CRITICAL RULES:
          - SL only moves *toward* profit (never to a worse position).
          - BOS/CHOCH adjustments are ONLY applied when profitable.

        Returns 'SL' or 'TP' if the current position should be closed, else None.
        """
        if self.position == 0:
            return None

        entry = self.position_price
        if entry <= 0:
            return None

        # --- Update peak price ---
        self._update_peak_price(current_price)

        # --- Compute unrealized profit % ---
        if self.position == 1:  # LONG
            profit_pct = (current_price - entry) / entry
        else:  # SHORT
            profit_pct = (entry - current_price) / entry

        is_profitable = profit_pct > 0

        # --- Hard TP check ---
        if self.position == 1 and self.tp_price > 0 and current_price >= self.tp_price:
            return "TP"
        if self.position == -1 and self.tp_price > 0 and current_price <= self.tp_price:
            return "TP"

        # --- Baseline trailing stop adjustments (always active) ---
        new_sl = self.sl_price
        new_tp = self.tp_price
        adjustment_reason = ""

        if profit_pct >= TRAILING_LOCK_PCT:
            # At +2% profit → lock 50% of profit from entry
            if self.position == 1:
                locked_sl = entry + 0.5 * (self.peak_price - entry)
                if locked_sl > new_sl:
                    new_sl = locked_sl
                    adjustment_reason = "trailing_lock_50pct"
            else:
                locked_sl = entry - 0.5 * (entry - self.peak_price)
                if new_sl <= 0 or locked_sl < new_sl:
                    new_sl = locked_sl
                    adjustment_reason = "trailing_lock_50pct"

        elif profit_pct >= TRAILING_BREAKEVEN_PCT:
            # At +1% profit → move SL to break-even (entry price)
            if self.position == 1:
                if entry > new_sl:
                    new_sl = entry
                    adjustment_reason = "trailing_breakeven"
            else:
                if new_sl <= 0 or entry < new_sl:
                    new_sl = entry
                    adjustment_reason = "trailing_breakeven"

        # --- BOS/CHOCH overlay (ONLY when profitable) ---
        if is_profitable and self._last_structure_signals:
            sig = self._last_structure_signals
            swing_high = sig.get("last_swing_high", 0.0)
            swing_low = sig.get("last_swing_low", 0.0)
            confidence = sig.get("confidence", 0.0)

            # Small buffer to avoid SL sitting exactly on the swing level
            buffer_pct = 0.002  # 0.2%

            if self.position == 1:  # ── LONG ──
                # BOS bullish (trend continuation) → trail SL to swing low
                if sig.get("bos_bullish") and not sig.get("fake_bos") and swing_low > 0:
                    bos_sl = swing_low * (1.0 - buffer_pct)
                    if bos_sl > new_sl:
                        logger.info(
                            "🔄 BOS bullish → trailing SL to swing low $%.2f (was $%.2f, conf=%.2f)",
                            bos_sl, new_sl, confidence,
                        )
                        new_sl = bos_sl
                        adjustment_reason = f"BOS_bullish(conf={confidence:.2f})"
                    # Extend TP to next swing high if it's above current TP
                    if swing_high > 0 and swing_high > new_tp:
                        new_tp = swing_high
                        logger.info(
                            "📈 BOS bullish: TP extended → $%.2f (swing high)",
                            new_tp,
                        )

                # CHOCH bearish (reversal warning) → lock 75% of profit
                if sig.get("choch_bearish") and not sig.get("fake_choch"):
                    choch_sl = entry + 0.75 * (current_price - entry)
                    if choch_sl > new_sl:
                        logger.info(
                            "🔄 CHOCH bearish → tightening SL to lock 75%% profit $%.2f (was $%.2f, conf=%.2f)",
                            choch_sl, new_sl, confidence,
                        )
                        new_sl = choch_sl
                        adjustment_reason = f"CHOCH_bearish(conf={confidence:.2f})"

                # Fake BOS bearish → explicitly ignore (hold position)
                if sig.get("fake_bos") and sig.get("bos_bearish"):
                    logger.info("⚠️ Fake BOS bearish detected — ignoring, holding LONG")

                # Fake CHOCH bullish → ignore (don't over-tighten)
                if sig.get("fake_choch") and sig.get("choch_bullish"):
                    logger.info("⚠️ Fake CHOCH bullish detected — ignoring")

            elif self.position == -1:  # ── SHORT ──
                # BOS bearish (continuation) → trail SL to swing high
                if sig.get("bos_bearish") and not sig.get("fake_bos") and swing_high > 0:
                    bos_sl = swing_high * (1.0 + buffer_pct)
                    if new_sl <= 0 or bos_sl < new_sl:
                        logger.info(
                            "🔄 BOS bearish → trailing SL to swing high $%.2f (was $%.2f, conf=%.2f)",
                            bos_sl, new_sl, confidence,
                        )
                        new_sl = bos_sl
                        adjustment_reason = f"BOS_bearish(conf={confidence:.2f})"
                    # Extend TP to next swing low
                    if swing_low > 0 and (new_tp <= 0 or swing_low < new_tp):
                        new_tp = swing_low
                        logger.info(
                            "📉 BOS bearish: TP extended → $%.2f (swing low)",
                            new_tp,
                        )

                # CHOCH bullish (reversal warning) → lock 75% of profit
                if sig.get("choch_bullish") and not sig.get("fake_choch"):
                    choch_sl = entry - 0.75 * (entry - current_price)
                    if new_sl <= 0 or choch_sl < new_sl:
                        logger.info(
                            "🔄 CHOCH bullish → tightening SL to lock 75%% profit $%.2f (was $%.2f, conf=%.2f)",
                            choch_sl, new_sl, confidence,
                        )
                        new_sl = choch_sl
                        adjustment_reason = f"CHOCH_bullish(conf={confidence:.2f})"

                # Fake BOS bullish → ignore
                if sig.get("fake_bos") and sig.get("bos_bullish"):
                    logger.info("⚠️ Fake BOS bullish detected — ignoring, holding SHORT")

                # Fake CHOCH bearish → ignore
                if sig.get("fake_choch") and sig.get("choch_bearish"):
                    logger.info("⚠️ Fake CHOCH bearish detected — ignoring")

        # --- Safety: SL must never move to a worse position ---
        # Minimum change threshold: 0.01% of price (e.g. BTC=$7, ETH=$0.22)
        # Prevents spam from sub-cent trailing recalculations on every tick.
        MIN_SLTP_CHANGE_PCT = 0.0001
        _min_change = current_price * MIN_SLTP_CHANGE_PCT

        sl_changed = False
        if self.position == 1:
            if new_sl > self.sl_price and (new_sl - self.sl_price) >= _min_change:
                sl_changed = True
            else:
                new_sl = self.sl_price  # revert
        elif self.position == -1:
            if self.sl_price <= 0 or (new_sl > 0 and new_sl < self.sl_price and (self.sl_price - new_sl) >= _min_change):
                sl_changed = True
            else:
                new_sl = self.sl_price  # revert

        tp_changed = new_tp != self.tp_price and abs(new_tp - self.tp_price) >= _min_change

        # --- Apply and log SL/TP changes ---
        if sl_changed or tp_changed:
            old_sl_price = self.sl_price
            old_tp_price = self.tp_price

            if sl_changed:
                logger.info(
                    "🔄 SL adjusted: $%.2f → $%.2f (reason=%s, profit=%.2f%%, peak=$%.2f)",
                    self.sl_price, new_sl, adjustment_reason or "trailing",
                    profit_pct * 100, self.peak_price,
                )
                self.sl_price = new_sl

            if tp_changed:
                logger.info(
                    "🔄 TP adjusted: $%.2f → $%.2f (reason=%s)",
                    self.tp_price, new_tp, adjustment_reason or "structure",
                )
                self.tp_price = new_tp

            self._save_state()

            # Write SL/TP update alerts to the shared alerts file
            self._write_sltp_update_alert(
                sl_changed, tp_changed,
                old_sl_price, self.sl_price,
                old_tp_price, self.tp_price,
                adjustment_reason or "trailing",
            )

            # Sync updated SL/TP to exchange (for testnet bots: both LONG and SHORT)
            if not self.dry_run and self.testnet_executor:
                try:
                    self.testnet_executor.update_sl_tp(
                        self.symbol, self.sl_price, self.tp_price
                    )
                except Exception as _exc:
                    logger.warning("Testnet SL/TP update failed: %s", _exc)

        # --- Liquidation safety check (max once per 5 min to avoid spam) ---
        _now = time.time()
        if not hasattr(self, '_last_liq_check') or (_now - self._last_liq_check) > 300:
            self._check_liquidation_safety()
            self._last_liq_check = _now

        # --- Check if current price hits the (possibly adjusted) SL ---
        if self.position == 1:
            if self.sl_price > 0 and current_price <= self.sl_price:
                return "SL"
        elif self.position == -1:
            if self.sl_price > 0 and current_price >= self.sl_price:
                return "SL"

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
        # Exception: if the model wants to REVERSE direction (e.g. SHORT→LONG)
        # with high confidence (≥0.75), let it flip — the model should decide.
        is_reversal = (
            (self.position == 1 and action == ACTION_SHORT) or
            (self.position == -1 and action == ACTION_LONG)
        )
        if self.position != 0 and (now - self.last_entry_time) < MIN_HOLD_SECONDS:
            if is_reversal and confidence >= 0.70:
                logger.info(
                    "Min hold override: reversal %s→%s with conf=%.2f ≥ 0.70 — allowing flip",
                    "LONG" if self.position == 1 else "SHORT",
                    ACTION_LABELS.get(action, "?"), confidence,
                )
            else:
                remaining = MIN_HOLD_SECONDS - (now - self.last_entry_time)
                logger.info("Min hold: %.0fs remaining — HOLD", remaining)
                return None

        # ── Guard: confidence threshold ──
        if action != ACTION_HOLD and confidence < MIN_CONFIDENCE:
            logger.info("Low confidence %.2f < %.2f — HOLD", confidence, MIN_CONFIDENCE)
            return None

        # ── Guard: ranging regime filter ──
        # In ranging markets (low ADX), require higher confidence to enter
        if action != ACTION_HOLD and self.position == 0 and self.regime_detector is not None and self._last_df is not None:
            try:
                regime_info = self.regime_detector.detect_regime(self._last_df)
                regime_name = regime_info.regime.value
                adx_val = getattr(regime_info, 'trend_strength', None) or 0.0
                if adx_val < RANGING_ADX_THRESHOLD and confidence < RANGING_MIN_CONFIDENCE:
                    logger.info(
                        "🚫 Ranging regime filter: ADX=%.1f < %.1f, conf=%.2f < %.2f — SKIP entry",
                        adx_val, RANGING_ADX_THRESHOLD, confidence, RANGING_MIN_CONFIDENCE,
                    )
                    return None
            except Exception as exc:
                logger.debug("Regime filter check failed: %s", exc)

        # ── Guard: momentum exhaustion filter ──
        # Don't enter when price is extended far from VWAP (likely to revert)
        if action != ACTION_HOLD and self.position == 0 and self._last_df is not None:
            try:
                df = self._last_df
                if len(df) >= 20:
                    closes = df["close"].values[-20:]
                    volumes = df["volume"].values[-20:]
                    # Compute VWAP over last 20 bars
                    typical_price = (df["high"].values[-20:] + df["low"].values[-20:] + closes) / 3.0
                    vwap_20 = float(np.sum(typical_price * volumes) / (np.sum(volumes) + 1e-10))
                    # Compute ATR (14-period) from last 20 bars
                    highs = df["high"].values[-20:]
                    lows = df["low"].values[-20:]
                    tr = np.maximum(highs - lows, np.maximum(
                        np.abs(highs - np.roll(closes, 1)),
                        np.abs(lows - np.roll(closes, 1))
                    ))
                    atr_14 = float(np.mean(tr[-14:]))
                    if atr_14 > 0:
                        extension = abs(current_price - vwap_20) / atr_14
                        if extension > EXHAUSTION_ATR_THRESHOLD:
                            direction_str = "LONG" if action == ACTION_LONG else "SHORT"
                            logger.info(
                                "🚫 Exhaustion filter: price $%.2f is %.1f ATR from VWAP $%.2f — SKIP %s entry",
                                current_price, extension, vwap_20, direction_str,
                            )
                            return None
            except Exception as exc:
                logger.debug("Exhaustion filter check failed: %s", exc)

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
        # ── Fixed-Dollar-Risk Position Sizing ──
        # Step 1: Calculate dollar risk per trade
        risk_pool = self.session_balance * RISK_POOL_PCT
        dollar_risk = risk_pool / RISK_BUDGET_PARTS

        # Step 2: Calculate notional from dollar risk and SL%
        # (regime-adaptive SL is calculated below, use base for sizing)
        sl_pct_for_sizing = STOP_LOSS_PCT
        notional = dollar_risk / sl_pct_for_sizing

        # Step 3: Cap notional to hard max
        notional = min(notional, FIXED_MAX_NOTIONAL)

        # Step 4: Calculate margin = 3× dollar risk (Dvir's rule)
        margin = MARGIN_MULTIPLIER * dollar_risk

        # Step 5: Derive leverage from notional / margin, cap it
        raw_leverage = notional / margin if margin > 0 else 1
        leverage = max(1, min(int(raw_leverage), MAX_LEVERAGE))

        # Step 6: Recalculate actual margin with integer leverage
        actual_margin = notional / leverage

        # Step 7: Validate we have enough balance for the margin
        if actual_margin > self.session_balance * 0.95:
            # Scale down notional to fit
            notional = self.session_balance * 0.90 * leverage
            logger.warning(
                "⚠️ Margin $%.2f exceeds balance — scaled notional to $%.2f",
                actual_margin, notional,
            )
            actual_margin = notional / leverage

        trade_value = notional
        logger.info(
            "📐 %s risk sizing: bal=$%.0f | risk_pool=$%.0f | risk/trade=$%.2f | "
            "notional=$%.0f | margin=$%.2f (%.1f×risk) | leverage=%dx",
            "LONG" if direction == 1 else "SHORT",
            self.session_balance, risk_pool, dollar_risk,
            notional, actual_margin, MARGIN_MULTIPLIER, leverage,
        )

        fee = trade_value * TRADING_FEE
        self.balance -= fee
        self.position = direction
        self.position_price = price
        self.position_units = (trade_value - fee) / (price + 1e-10)
        self.peak_price = price  # Reset peak price for trailing stop tracking
        self.last_entry_time = time.time()
        self.position_entry_time = time.time()  # Phase 1 §3.4: for time-based exit

        # Phase 1 §3.2: Regime-adaptive SL/TP multipliers
        sl_pct = STOP_LOSS_PCT
        tp_pct = TAKE_PROFIT_PCT
        try:
            if self.regime_detector is not None and self._last_df is not None:
                regime_info = self.regime_detector.detect_regime(self._last_df)
                regime_name = regime_info.regime.value
                if regime_name == 'high_volatility':
                    sl_pct *= 1.5   # Was 2.0× — reduced to limit loss per trade
                    tp_pct *= 1.3   # Was 1.5× — scaled proportionally
                    logger.info("📊 HIGH VOL regime: SL ×1.5, TP ×1.3")
                elif regime_name == 'trending_up' and direction == 1:
                    tp_pct *= 1.8   # With-trend LONG: let runners run
                    logger.info("📊 TRENDING_UP regime (LONG): TP ×1.8, SL tight")
                elif regime_name == 'trending_down' and direction == -1:
                    tp_pct *= 1.8   # With-trend SHORT: let runners run
                    logger.info("📊 TRENDING_DOWN regime (SHORT): TP ×1.8, SL tight")
                elif regime_name == 'trending_down' and direction == 1:
                    sl_pct *= 1.3   # Counter-trend LONG: wider SL
                    logger.info("📊 TRENDING_DOWN regime (counter-trend LONG): SL ×1.3")
                elif regime_name == 'trending_up' and direction == -1:
                    sl_pct *= 1.3   # Counter-trend SHORT: wider SL
                    logger.info("📊 TRENDING_UP regime (counter-trend SHORT): SL ×1.3")
                elif regime_name == 'ranging':
                    sl_pct *= 1.2   # Tighter to limit chop loss
                    tp_pct *= 0.8   # Mean-reversion moves are smaller
                    logger.info("📊 RANGING regime: SL ×1.2, TP ×0.8")
        except Exception as exc:
            logger.warning("Regime-adaptive SL/TP failed: %s — using base values", exc)

        self.sl_pct = sl_pct  # Phase 1 §3.3: store for partial TP R-multiple calculations

        if direction == 1:
            self.sl_price = price * (1.0 - sl_pct)
            self.tp_price = price * (1.0 + tp_pct)
            action_str = "OPEN_LONG"
            # Phase 1 §3.3: Partial TP prices (1R and 2R targets)
            self.partial_tp1_price = price * (1.0 + 1.0 * sl_pct)
            self.partial_tp2_price = price * (1.0 + 2.0 * sl_pct)
        else:
            self.sl_price = price * (1.0 + sl_pct)
            self.tp_price = price * (1.0 - tp_pct)
            action_str = "OPEN_SHORT"
            # Phase 1 §3.3: Partial TP prices (1R and 2R targets)
            self.partial_tp1_price = price * (1.0 - 1.0 * sl_pct)
            self.partial_tp2_price = price * (1.0 - 2.0 * sl_pct)

        # Phase 1 §3.3: Reset partial TP tracking for new trade
        self.initial_position_units = self.position_units
        self.partial_tp_level = 0

        # Phase 1 §3.9: Reset MFE/MAE for new trade
        self.mfe_pct = 0.0
        self.mae_pct = 0.0

        # ── Liquidation vs SL validation ──
        # Estimate liquidation distance: liq ≈ entry × (1 - direction/leverage)
        # SL must be closer to entry than liquidation
        liq_distance_pct = 1.0 / leverage if leverage > 0 else 1.0
        sl_distance_pct = sl_pct
        if sl_distance_pct > 0 and liq_distance_pct / sl_distance_pct < LIQ_SL_BUFFER:
            logger.warning(
                "⚠️ Liquidation too close to SL! liq_dist=%.2f%% sl_dist=%.2f%% "
                "ratio=%.2f (need ≥%.1f) — reducing leverage",
                liq_distance_pct * 100, sl_distance_pct * 100,
                liq_distance_pct / sl_distance_pct, LIQ_SL_BUFFER,
            )
            # Recalculate leverage to ensure buffer
            safe_leverage = int(1.0 / (sl_distance_pct * LIQ_SL_BUFFER))
            leverage = max(1, min(safe_leverage, MAX_LEVERAGE))
            logger.info("📐 Adjusted leverage to %dx for safety", leverage)

        trade = {
            "action": action_str,
            "symbol": self.symbol,
            "price": price,
            "units": self.position_units,
            "trade_value": trade_value,
            "confidence": confidence,
            "sl": self.sl_price,
            "tp": self.tp_price,
            "partial_tp1": self.partial_tp1_price,
            "partial_tp2": self.partial_tp2_price,
            "leverage": leverage,
            "dollar_risk": dollar_risk,
            "margin": actual_margin,
            "pnl": 0.0,
            "timestamp": datetime.now().isoformat(),
            "agent": "htf",
        }

        logger.info(
            "📈 %s @ $%.2f | units=%.5f | SL=$%.2f (%.2f%%) | TP=$%.2f (%.2f%%) | "
            "TP1=$%.2f | TP2=$%.2f | conf=%.2f",
            action_str, price, self.position_units,
            self.sl_price, sl_pct * 100,
            self.tp_price, tp_pct * 100,
            self.partial_tp1_price, self.partial_tp2_price, confidence,
        )

        if not self.dry_run:
            self._mirror_testnet(trade)
            # Sync balance AND position size from exchange after opening
            self._sync_balance_from_exchange()
            self._sync_position_from_exchange()
            # Update trade dict with real exchange units BEFORE logging/alerting
            trade["units"] = self.position_units
            trade["trade_value"] = self.position_units * price
            self._log_trade(trade)
        else:
            self._log_trade(trade)
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
            "balance_after": self.balance,
            "realized_pnl_total": self.realized_pnl,
            "initial_balance": self.initial_balance,
            "balance_pnl_pct": ((self.balance - self.initial_balance) / self.initial_balance * 100)
                if self.initial_balance > 0 else 0.0,
        }

        logger.info(
            "📉 %s @ $%.2f | pnl=$%.2f | reason=%s | balance=$%.2f",
            action_str, price, net_pnl, reason, self.balance,
        )

        # Phase 1 §3.9: Log MFE/MAE on close
        logger.info(
            "📊 Trade MFE/MAE: MFE=%+.3f%% MAE=%+.3f%% (entry=$%.2f exit=$%.2f)",
            self.mfe_pct * 100, self.mae_pct * 100, self.position_price, price,
        )
        trade["mfe_pct"] = self.mfe_pct
        trade["mae_pct"] = self.mae_pct

        # Reset position
        self.position = 0
        self.position_price = 0.0
        self.position_units = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.peak_price = 0.0
        # Phase 1: Reset partial TP and MFE/MAE state
        self.mfe_pct = 0.0
        self.mae_pct = 0.0
        self.partial_tp_level = 0
        self.partial_tp1_price = 0.0
        self.partial_tp2_price = 0.0
        self.sl_pct = 0.0
        self.initial_position_units = 0.0
        self.position_entry_time = 0.0

        if not self.dry_run:
            self._log_trade(trade)
            self._mirror_testnet(trade)
            # Sync balance from exchange after closing position
            self._sync_balance_from_exchange()
        else:
            logger.info("[DRY RUN] Close not recorded")

        return trade

    # ------------------------------------------------------------------
    # Testnet mirroring
    # ------------------------------------------------------------------

    def _mirror_testnet(self, trade: Dict) -> None:
        """
        Mirror a bot decision to futures testnet.

        OPEN_LONG / OPEN_SHORT → places real futures order + SL/TP exchange orders.
        ALL CLOSE actions → close the position on the exchange.
        """
        if not self.testnet_executor:
            return
        action = trade.get("action", "")
        symbol = trade.get("symbol", self.symbol)

        # ALL close actions: close the exchange position
        if "CLOSE" in action:
            try:
                from src.api.futures_executor import get_futures_executor
                executor = get_futures_executor()
                if executor:
                    positions = executor.connector.get_positions()
                    for p in positions:
                        if p.get("symbol") == symbol:
                            amt = float(p.get("positionAmt", 0))
                            if amt != 0:
                                side = "SELL" if amt > 0 else "BUY"
                                qty = abs(amt)
                                executor.connector.place_market_order(symbol, side, qty)
                                logger.info(
                                    "✅ Testnet mirror: %s %s %s qty=%.6f",
                                    action, symbol, side, qty,
                                )
                            else:
                                logger.info(
                                    "Testnet mirror: %s %s — already flat on exchange",
                                    action, symbol,
                                )
                            break
                    else:
                        logger.warning("Testnet mirror: %s not found in exchange positions", symbol)
            except Exception as exc:
                logger.warning("Testnet mirror close failed for %s: %s", symbol, exc)
            return
        try:
            result = self.testnet_executor.mirror_trade(trade, {})
            if result and result.get("executed"):
                logger.info(
                    "Testnet mirror: %s @ $%.2f (order=%s sl=%s tp=%s)",
                    action,
                    trade.get("price", 0),
                    result.get("order_id", ""),
                    result.get("sl_order_id", ""),
                    result.get("tp_order_id", ""),
                )
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

            # Cache data for regime detection in _open_position
            self._last_df = df_15m

            # Exchange position sync: detect stale positions (exchange closed
            # but bot still thinks it's in a trade → ghost SL/TP alerts).
            # Runs every iteration when bot thinks it has a position.
            if self.position != 0 and not self.dry_run:
                self._sync_position_from_exchange()
                # If sync reset us to flat, skip the rest of position logic
                if self.position == 0:
                    status["action"] = "SYNC_FLAT"
                    logger.info("Position was stale — synced to FLAT, skipping to next iteration")
                    return status

            # Phase 1 §3.9: Update MFE/MAE every iteration while in position
            if self.position != 0 and self.position_price > 0:
                if self.position == 1:
                    _unrealized_pct = (current_price - self.position_price) / self.position_price
                else:
                    _unrealized_pct = (self.position_price - current_price) / self.position_price
                self.mfe_pct = max(self.mfe_pct, _unrealized_pct)
                self.mae_pct = min(self.mae_pct, _unrealized_pct)

            # Phase 1 §3.3: Partial Take Profit check (40% at 1R, 35% at 2R)
            if self.position != 0 and self.position_price > 0 and self.initial_position_units > 0:
                if self.position == 1 and self.partial_tp_level < 1 and self.partial_tp1_price > 0:
                    if current_price >= self.partial_tp1_price:
                        partial_units = self.initial_position_units * 0.40
                        partial_units = min(partial_units, self.position_units)
                        if partial_units > 0:
                            partial_pnl = (self.partial_tp1_price - self.position_price) * partial_units
                            self.realized_pnl += partial_pnl
                            self.position_units -= partial_units
                            self.partial_tp_level = 1
                            old_sl = self.sl_price
                            self.sl_price = self.position_price  # Move SL to break-even
                            logger.info(
                                "✅ PARTIAL TP1 (LONG): closed %.5f units @ $%.2f | "
                                "PnL=$%+.2f | SL → break-even $%.2f (was $%.2f) | "
                                "remaining=%.5f",
                                partial_units, self.partial_tp1_price, partial_pnl,
                                self.sl_price, old_sl, self.position_units,
                            )
                            self._save_state()
                            # Sync updated SL to exchange
                            if not self.dry_run and self.testnet_executor:
                                try:
                                    self.testnet_executor.update_sl_tp(
                                        self.symbol, self.sl_price, self.tp_price
                                    )
                                except Exception as _exc:
                                    logger.warning("Testnet SL update after TP1 failed: %s", _exc)

                elif self.position == 1 and self.partial_tp_level == 1 and self.partial_tp2_price > 0:
                    if current_price >= self.partial_tp2_price:
                        partial_units = self.initial_position_units * 0.35
                        partial_units = min(partial_units, self.position_units)
                        if partial_units > 0:
                            partial_pnl = (self.partial_tp2_price - self.position_price) * partial_units
                            self.realized_pnl += partial_pnl
                            self.position_units -= partial_units
                            self.partial_tp_level = 2
                            level2_gain = self.partial_tp2_price - self.position_price
                            new_trailing_sl = self.position_price + level2_gain * 0.50
                            if new_trailing_sl > self.sl_price:
                                old_sl = self.sl_price
                                self.sl_price = new_trailing_sl
                                logger.info(
                                    "✅ PARTIAL TP2 (LONG): closed %.5f units @ $%.2f | "
                                    "PnL=$%+.2f | trailing SL locked → $%.2f (was $%.2f) | "
                                    "remaining=%.5f",
                                    partial_units, self.partial_tp2_price, partial_pnl,
                                    self.sl_price, old_sl, self.position_units,
                                )
                            else:
                                logger.info(
                                    "✅ PARTIAL TP2 (LONG): closed %.5f units @ $%.2f | "
                                    "PnL=$%+.2f | remaining=%.5f (SL already ahead)",
                                    partial_units, self.partial_tp2_price, partial_pnl,
                                    self.position_units,
                                )
                            self._save_state()
                            if not self.dry_run and self.testnet_executor:
                                try:
                                    self.testnet_executor.update_sl_tp(
                                        self.symbol, self.sl_price, self.tp_price
                                    )
                                except Exception as _exc:
                                    logger.warning("Testnet SL update after TP2 failed: %s", _exc)

                elif self.position == -1 and self.partial_tp_level < 1 and self.partial_tp1_price > 0:
                    if current_price <= self.partial_tp1_price:
                        partial_units = self.initial_position_units * 0.40
                        partial_units = min(partial_units, self.position_units)
                        if partial_units > 0:
                            partial_pnl = (self.position_price - self.partial_tp1_price) * partial_units
                            self.realized_pnl += partial_pnl
                            self.position_units -= partial_units
                            self.partial_tp_level = 1
                            old_sl = self.sl_price
                            self.sl_price = self.position_price  # Move SL to break-even
                            logger.info(
                                "✅ PARTIAL TP1 (SHORT): closed %.5f units @ $%.2f | "
                                "PnL=$%+.2f | SL → break-even $%.2f (was $%.2f) | "
                                "remaining=%.5f",
                                partial_units, self.partial_tp1_price, partial_pnl,
                                self.sl_price, old_sl, self.position_units,
                            )
                            self._save_state()
                            if not self.dry_run and self.testnet_executor:
                                try:
                                    self.testnet_executor.update_sl_tp(
                                        self.symbol, self.sl_price, self.tp_price
                                    )
                                except Exception as _exc:
                                    logger.warning("Testnet SL update after TP1 failed: %s", _exc)

                elif self.position == -1 and self.partial_tp_level == 1 and self.partial_tp2_price > 0:
                    if current_price <= self.partial_tp2_price:
                        partial_units = self.initial_position_units * 0.35
                        partial_units = min(partial_units, self.position_units)
                        if partial_units > 0:
                            partial_pnl = (self.position_price - self.partial_tp2_price) * partial_units
                            self.realized_pnl += partial_pnl
                            self.position_units -= partial_units
                            self.partial_tp_level = 2
                            level2_gain = self.position_price - self.partial_tp2_price
                            new_trailing_sl = self.position_price - level2_gain * 0.50
                            if new_trailing_sl < self.sl_price:
                                old_sl = self.sl_price
                                self.sl_price = new_trailing_sl
                                logger.info(
                                    "✅ PARTIAL TP2 (SHORT): closed %.5f units @ $%.2f | "
                                    "PnL=$%+.2f | trailing SL locked → $%.2f (was $%.2f) | "
                                    "remaining=%.5f",
                                    partial_units, self.partial_tp2_price, partial_pnl,
                                    self.sl_price, old_sl, self.position_units,
                                )
                            else:
                                logger.info(
                                    "✅ PARTIAL TP2 (SHORT): closed %.5f units @ $%.2f | "
                                    "PnL=$%+.2f | remaining=%.5f (SL already ahead)",
                                    partial_units, self.partial_tp2_price, partial_pnl,
                                    self.position_units,
                                )
                            self._save_state()
                            if not self.dry_run and self.testnet_executor:
                                try:
                                    self.testnet_executor.update_sl_tp(
                                        self.symbol, self.sl_price, self.tp_price
                                    )
                                except Exception as _exc:
                                    logger.warning("Testnet SL update after TP2 failed: %s", _exc)

            # 1b. Update peak price for trailing stop tracking
            self._update_peak_price(current_price)

            # 1c. Compute BOS/CHOCH structure signals on 5m candles (cached for this iteration)
            self._get_structure_signals(trigger=True)

            # 2. Check SL/TP before computing new action (uses cached BOS/CHOCH)
            exit_reason = self._check_sl_tp(current_price)
            if exit_reason and self.position != 0:
                logger.info("SL/TP triggered (%s) @ $%.2f", exit_reason, current_price)
                trade = self._close_position(current_price, exit_reason, 1.0)
                status["action"] = trade.get("action", "CLOSE")
                status["position"] = self.position
                status["balance"] = self.balance
                status["realized_pnl"] = self.realized_pnl
                return status

            # Phase 1 §3.4: Time-based stagnant exit (>6h, PnL between -0.3% and +0.5%)
            if self.position != 0 and self.position_price > 0 and self.position_entry_time > 0:
                _time_in_pos = time.time() - self.position_entry_time
                if _time_in_pos > 21600:  # 6 hours
                    if self.position == 1:
                        _stagnant_pct = (current_price - self.position_price) / self.position_price
                    else:
                        _stagnant_pct = (self.position_price - current_price) / self.position_price
                    if -0.003 <= _stagnant_pct <= 0.005:
                        logger.info(
                            "⏱️ TIME-BASED STAGNANT EXIT: %s in position %.1fh, "
                            "PnL=%+.3f%% (within stagnant band [-0.3%%, +0.5%%]) → closing",
                            self.symbol, _time_in_pos / 3600, _stagnant_pct * 100,
                        )
                        stagnant_trade = self._close_position(current_price, "STAGNANT_EXIT", 1.0)
                        status["action"] = stagnant_trade.get("action", "CLOSE")
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
    # WebSocket price monitor
    # ------------------------------------------------------------------

    def _start_ws_monitor(self) -> None:
        """
        Start a background daemon thread that streams aggTrade prices from
        Binance and triggers SL/TP closes in real-time (between 15-min loops).
        """
        import websocket as _websocket

        ws_symbol = self.symbol.lower()  # e.g. "ethusdt"
        url = f"wss://stream.binance.com:9443/ws/{ws_symbol}@aggTrade"

        def _on_open(ws):
            logger.info("WS price monitor connected to aggTrade stream")

        def _on_error(ws, error):
            logger.warning("WS price monitor error: %s", error)

        def _on_close(ws, code, msg):
            logger.info("WS price monitor closed (code=%s)", code)

        first_price_logged = [False]  # mutable container for nonlocal in Python 2-compat closure

        def _on_message(ws, message):
            try:
                data = json.loads(message)
                price = float(data.get("p", 0))
                if price <= 0:
                    return

                self.current_price = price

                # Phase 1 §3.9: Update MFE/MAE from real-time price ticks
                if self.position != 0 and self.position_price > 0:
                    if self.position == 1:
                        _tick_unrealized = (price - self.position_price) / self.position_price
                    else:
                        _tick_unrealized = (self.position_price - price) / self.position_price
                    if _tick_unrealized > self.mfe_pct:
                        self.mfe_pct = _tick_unrealized
                    if _tick_unrealized < self.mae_pct:
                        self.mae_pct = _tick_unrealized

                if not first_price_logged[0]:
                    logger.info("WS first price tick: $%.2f — stream is live", price)
                    first_price_logged[0] = True

                with self._lock:
                    if self.position == 0:
                        return
                    exit_reason = self._check_sl_tp(price)
                    if exit_reason:
                        logger.info("WS: SL/TP triggered (%s) @ $%.2f", exit_reason, price)
                        self._close_position(price, exit_reason, 1.0)

            except Exception as exc:
                logger.debug("WS message handler error: %s", exc)

        def _run():
            backoff = 1
            while True:
                try:
                    ws = _websocket.WebSocketApp(
                        url,
                        on_open=_on_open,
                        on_message=_on_message,
                        on_error=_on_error,
                        on_close=_on_close,
                    )
                    ws.run_forever()
                except Exception as exc:
                    logger.warning("WS run_forever exception: %s", exc)
                logger.info("WS reconnecting in %ds...", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        t = threading.Thread(target=_run, name="ws-price-monitor", daemon=True)
        t.start()
        logger.info("WebSocket price monitor thread started")

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
        self._start_ws_monitor()

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
    parser.add_argument("--symbol", type=str, default=None,
                        help="Trading symbol (e.g. ETHUSDT). Overrides HTF_SYMBOL env var.")
    parser.add_argument("--once", action="store_true",
                        help="Run a single iteration and exit")
    args = parser.parse_args()

    # Symbol override: CLI flag > env var > default
    if args.symbol:
        global SYMBOL
        SYMBOL = args.symbol

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
