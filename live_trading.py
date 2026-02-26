#!/usr/bin/env python3
"""
Advanced Live Trading System
Dry-run and live trading with the competitive advanced model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import logging
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque
import json

from src.features.ultimate_features import UltimateFeatureEngine
from src.features.correlation_engine import SimulatedDominanceEngine
from src.features.whale_tracker import WhaleTracker
from src.features.regime_detector import MarketRegimeDetector
from src.features.mtf_analyzer import MultiTimeframeAnalyzer
from src.features.risk_manager import AdaptiveRiskManager
from src.features.order_flow import FundingRateAnalyzer, OrderFlowAnalyzer
from src.data.candle_stream import CandleStreamManager
from src.backtest.data_loader import BinanceHistoricalDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveTradingAgent:
    """
    Live trading agent using the advanced model.
    
    Supports:
    - Dry-run mode (simulated trades)
    - Paper trading on Binance Testnet
    - Risk management with stop-loss/take-profit
    - State persistence across restarts
    """
    
    def __init__(
        self,
        model_path: str = './data/models/ultimate_agent.zip',
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        initial_balance: float = 10000.0,
        position_size: float = 0.50,  # Reduced from 75% to limit losses
        stop_loss_pct: float = 0.015,  # Wider 1.5% stop loss to reduce whipsaws
        take_profit_pct: float = 0.025,  # 2.5% take profit for better R:R
        max_daily_loss: float = 0.05,  # 5% daily loss limit (circuit breaker)
        minimum_hold_minutes: int = 30,  # Minimum hold time
        no_flip_on_sl: bool = True,  # Don't flip position after stop loss
        use_trend_filter: bool = True,  # Don't trade against strong trend
        dry_run: bool = True,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss = max_daily_loss
        self.minimum_hold_minutes = minimum_hold_minutes
        self.no_flip_on_sl = no_flip_on_sl
        self.use_trend_filter = use_trend_filter
        self.dry_run = dry_run
        self.last_exit_reason = None  # Track last exit reason for no-flip logic
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = PPO.load(model_path)
        
        # Load VecNormalize stats for observation normalization
        vec_normalize_path = model_path.replace('.zip', '_vec_normalize.pkl')
        self.vec_normalize = None
        if Path(vec_normalize_path).exists():
            import pickle
            with open(vec_normalize_path, 'rb') as f:
                self.vec_normalize = pickle.load(f)
            logger.info(f"Loaded VecNormalize from {vec_normalize_path}")
        else:
            logger.warning(f"VecNormalize not found at {vec_normalize_path}, observations won't be normalized")
        
        # Feature engine - Ultimate features
        self.feature_engine = UltimateFeatureEngine()
        self.dominance_engine = SimulatedDominanceEngine()
        
        # Data fetcher
        self.fetcher = BinanceHistoricalDataFetcher()
        
        # Whale tracker for trend filtering
        self.whale_tracker = WhaleTracker() if use_trend_filter else None
        if self.whale_tracker:
            logger.info("🐋 Whale tracker enabled for trend filtering")
            
        # Market regime detector
        self.regime_detector = MarketRegimeDetector() if use_trend_filter else None
        if self.regime_detector:
            logger.info("📊 Market regime detector enabled")
        
        # Multi-timeframe analyzer
        self.mtf_analyzer = MultiTimeframeAnalyzer(symbol=symbol.replace('/', '')) if use_trend_filter else None
        if self.mtf_analyzer:
            logger.info("📊 Multi-timeframe analyzer enabled (4H/1H/15m)")
        
        # ===== ADVANCED MODULES =====
        # Adaptive risk manager (ATR-based SL/TP, Kelly sizing, trailing stops)
        self.risk_manager = AdaptiveRiskManager(
            base_sl_pct=stop_loss_pct,
            base_tp_pct=take_profit_pct,
            base_position_size=position_size,
            use_kelly=True,
            use_trailing=True
        ) if use_trend_filter else None
        if self.risk_manager:
            logger.info("📊 Adaptive risk manager enabled (ATR SL/TP, Kelly sizing)")
        
        # Funding rate analyzer
        self.funding_analyzer = FundingRateAnalyzer(
            symbol=symbol.replace('/', '')
        ) if use_trend_filter else None
        if self.funding_analyzer:
            logger.info("💰 Funding rate analyzer enabled")
        
        # Order flow analyzer
        self.order_flow = OrderFlowAnalyzer(
            symbol=symbol.replace('/', ''),
            large_order_threshold=50000
        ) if use_trend_filter else None
        if self.order_flow:
            logger.info("📊 Order flow analyzer enabled (CVD, large orders)")
        
        # ===== WEBSOCKET CANDLE STREAM =====
        # Efficient data management: fetch once, stream updates
        self.candle_stream = CandleStreamManager(
            symbol=symbol.replace('/', ''),
            interval=timeframe,
            max_candles=1000,
            on_new_candle=self._on_new_candle
        )
        self._use_websocket = True  # Can be disabled for fallback
        logger.info("🔌 WebSocket candle stream initialized (1000 candles, streams updates)")
        
        # Track highest/lowest price for trailing stops
        self.highest_since_entry = 0.0
        self.lowest_since_entry = float('inf')
        
        # Files for persistence
        self.log_file = Path('./logs/trading_log.json')
        self.state_file = Path('./logs/trading_state.json')
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Trading state - will be loaded from persisted state
        self.balance = initial_balance
        self.position = 0  # -1: short, 0: flat, 1: long
        self.position_price = 0.0
        self.position_size_units = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.daily_pnl = 0.0
        self.trades: List[Dict] = []
        self.circuit_breaker_active = False
        self.last_trade_time: Optional[datetime] = None  # Track last entry time for cooldown
        
        # Data buffer
        self.lookback_window = 48
        self.price_history = deque(maxlen=500)  # Store recent prices for features
        
        # Load persisted state
        self._load_state()
        
    def _load_state(self):
        """Load trading state from the state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    
                self.balance = state.get('balance', self.initial_balance)
                self.position = state.get('position', 0)
                self.position_price = state.get('position_price', 0.0)
                self.position_size_units = state.get('position_size_units', 0.0)
                self.realized_pnl = state.get('realized_pnl', 0.0)
                self.daily_pnl = state.get('daily_pnl', 0.0)
                
                # Load last trade time for cooldown
                last_trade_str = state.get('last_trade_time')
                if last_trade_str:
                    self.last_trade_time = datetime.fromisoformat(last_trade_str)
                
                position_str = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}
                logger.info(f"📂 Loaded state: Position={position_str.get(self.position, 'UNKNOWN')}, "
                           f"Balance=${self.balance:.2f}, P&L=${self.realized_pnl:.2f}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        else:
            logger.info("📂 No previous state found, starting fresh")
    
    def _on_new_candle(self, df: pd.DataFrame):
        """Callback when new candle is received via WebSocket."""
        if not self._use_websocket:
            return
        
        # Store the dataframe for use in trading logic
        self._last_df = df
        
        # Optionally trigger immediate analysis on new candle
        # For now, we just update the stored data
        logger.debug(f"📊 New candle via WebSocket, total: {len(df)}")
            
    def _save_state(self):
        """Save trading state to file for persistence."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'balance': self.balance,
            'position': self.position,
            'position_price': self.position_price,
            'position_size_units': self.position_size_units,
            'realized_pnl': self.realized_pnl,
            'daily_pnl': self.daily_pnl,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
    def fetch_latest_data(self) -> pd.DataFrame:
        """Fetch latest market data for feature computation."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Need 30 days for feature computation
        
        df = self.fetcher.fetch_historical_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        
        return df
        
    def compute_observation(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute observation from market data using Ultimate Feature Engine.
        
        MUST match UltimateTradingEnv._get_observation() exactly:
        observation = [features_row (99 values) | position_info (3 values)] = 102 total
        """
        # Compute ultimate features (same as training environment)
        all_features = self.feature_engine.get_all_features(df)
        
        # Add simulated dominance features
        dominance_features = self.dominance_engine.compute_simulated_dominance(df)
        all_features.update(dominance_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Handle NaN and inf
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Clip extreme values
        for col in features_df.columns:
            if features_df[col].dtype in [np.float64, np.float32]:
                features_df[col] = features_df[col].clip(-10, 10)
        
        features = features_df.values.astype(np.float32)
        
        # Get LAST ROW of features only (matches UltimateTradingEnv._get_observation)
        if len(features) < 1:
            logger.warning("Not enough data for features")
            return None
            
        last_features = features[-1].copy()
        
        # Get current price
        current_price = df.iloc[-1]['close']
        
        # Compute position info (EXACTLY matches UltimateTradingEnv._get_observation)
        if self.position != 0 and self.position_price > 0:
            if self.position == 1:  # Long
                unrealized_pnl = (current_price - self.position_price) / self.position_price
            else:  # Short
                unrealized_pnl = (self.position_price - current_price) / self.position_price
        else:
            unrealized_pnl = 0.0
            
        # Balance ratio (normalized change from initial)
        portfolio_value = self.balance + self.unrealized_pnl
        balance_ratio = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Position info: [position, unrealized_pnl, balance_ratio] - EXACTLY 3 values
        position_info = np.array([
            float(self.position),
            np.clip(unrealized_pnl, -0.5, 0.5),
            np.clip(balance_ratio, -0.5, 0.5),
        ], dtype=np.float32)
        
        # Combine features and position info (same format as training)
        # This produces exactly: 99 features + 3 position = 102 total
        observation = np.concatenate([last_features, position_info]).astype(np.float32)
        
        # Get market features for logging
        market_features = {
            'price': current_price,
            'unrealized_pnl': unrealized_pnl,
            'balance_ratio': balance_ratio,
        }
        
        return observation, current_price, market_features
        
    def get_action(self, observation: np.ndarray) -> int:
        """Get action from model."""
        # Normalize observation if VecNormalize is available
        if self.vec_normalize is not None:
            # VecNormalize expects batch dimension
            obs_batch = observation.reshape(1, -1)
            normalized_obs = self.vec_normalize.normalize_obs(obs_batch)
            action, _ = self.model.predict(normalized_obs, deterministic=True)
        else:
            action, _ = self.model.predict(observation, deterministic=True)
        return int(action.item() if hasattr(action, 'item') else action)
        
    def check_stop_loss_take_profit(self, current_price: float) -> Optional[str]:
        """Check if stop-loss, take-profit, or trailing stop should trigger."""
        if self.position == 0:
            return None
        
        # Use adaptive SL/TP if available, otherwise use base values
        sl_pct = getattr(self, 'current_sl_pct', self.stop_loss_pct)
        tp_pct = getattr(self, 'current_tp_pct', self.take_profit_pct)
        
        # Update highest/lowest since entry for trailing stop
        if self.position == 1:  # Long
            if current_price > self.highest_since_entry:
                self.highest_since_entry = current_price
            pnl_pct = (current_price - self.position_price) / self.position_price
            
            # Trailing stop: if we were up 2%+ and now dropped 1.5% from high
            if self.risk_manager and self.risk_manager.use_trailing:
                unrealized_from_high = (self.highest_since_entry - current_price) / self.highest_since_entry
                was_profitable = (self.highest_since_entry - self.position_price) / self.position_price > tp_pct * 0.5
                if was_profitable and unrealized_from_high > sl_pct:
                    logger.info(f"📊 TRAILING STOP: Price dropped {unrealized_from_high:.2%} from high ${self.highest_since_entry:.2f}")
                    return 'trailing_stop'
                    
        else:  # Short
            if current_price < self.lowest_since_entry:
                self.lowest_since_entry = current_price
            pnl_pct = (self.position_price - current_price) / self.position_price
            
            # Trailing stop for shorts: if we were up and price bounced
            if self.risk_manager and self.risk_manager.use_trailing:
                bounce_from_low = (current_price - self.lowest_since_entry) / self.lowest_since_entry
                was_profitable = (self.position_price - self.lowest_since_entry) / self.position_price > tp_pct * 0.5
                if was_profitable and bounce_from_low > sl_pct:
                    logger.info(f"📊 TRAILING STOP: Price bounced {bounce_from_low:.2%} from low ${self.lowest_since_entry:.2f}")
                    return 'trailing_stop'
            
        if pnl_pct <= -sl_pct:
            return 'stop_loss'
        elif pnl_pct >= tp_pct:
            return 'take_profit'
            
        return None
        
    def check_circuit_breaker(self) -> bool:
        """Check if daily loss limit is breached."""
        portfolio_value = self.balance + self.unrealized_pnl
        daily_loss = (self.initial_balance - portfolio_value) / self.initial_balance
        
        if daily_loss >= self.max_daily_loss:
            self.circuit_breaker_active = True
            logger.warning(f"🚨 CIRCUIT BREAKER ACTIVATED! Daily loss: {daily_loss:.2%}")
            return True
            
        return False
        
    def execute_action(self, action: int, current_price: float, reason: str = 'model'):
        """Execute trading action with cooldown protection."""
        position_str = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}
        action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
        # Check stop-loss/take-profit first (these override cooldown)
        sl_tp = self.check_stop_loss_take_profit(current_price)
        if sl_tp:
            if self.position == 1:
                action = 2  # Exit long
            elif self.position == -1:
                action = 1  # Exit short
            reason = sl_tp
        else:
            # Cooldown check - only for model-driven trades, not SL/TP
            if self.position != 0 and self.last_trade_time:
                time_since_trade = datetime.now() - self.last_trade_time
                min_hold_minutes = self.minimum_hold_minutes
                
                if time_since_trade.total_seconds() < min_hold_minutes * 60:
                    # Still in cooldown - don't flip position
                    remaining = min_hold_minutes - (time_since_trade.total_seconds() / 60)
                    logger.info(f"⏳ Cooldown active: {remaining:.0f} min remaining before next trade allowed")
                    
                    # If we already have a position and action would flip it, skip
                    if (self.position == 1 and action == 2) or (self.position == -1 and action == 1):
                        logger.info(f"⏳ Skipping position flip due to cooldown (currently {position_str[self.position]})")
                        return False
            
        # Execute action
        trade_executed = False
        
        # Handle SL/TP exits - only close, don't open new position
        if reason == 'stop_loss':
            if self.position == 1:  # Close long
                pnl = self._close_position(current_price)
                self._log_trade('CLOSE_LONG', current_price, pnl, reason)
                self.last_exit_reason = 'stop_loss'
                logger.info(f"🛑 STOP LOSS hit - closing LONG @ ${current_price:.2f}, P&L: ${pnl:.2f}")
                if self.no_flip_on_sl:
                    logger.info("⏳ No-flip mode: staying flat, waiting for next model signal")
                    return True
            elif self.position == -1:  # Close short
                pnl = self._close_position(current_price)
                self._log_trade('CLOSE_SHORT', current_price, pnl, reason)
                self.last_exit_reason = 'stop_loss'
                logger.info(f"🛑 STOP LOSS hit - closing SHORT @ ${current_price:.2f}, P&L: ${pnl:.2f}")
                if self.no_flip_on_sl:
                    logger.info("⏳ No-flip mode: staying flat, waiting for next model signal")
                    return True
                    
        elif reason == 'take_profit':
            if self.position == 1:  # Close long
                pnl = self._close_position(current_price)
                self._log_trade('CLOSE_LONG', current_price, pnl, reason)
                self.last_exit_reason = 'take_profit'
                logger.info(f"🎯 TAKE PROFIT hit - closing LONG @ ${current_price:.2f}, P&L: ${pnl:.2f}")
            elif self.position == -1:  # Close short
                pnl = self._close_position(current_price)
                self._log_trade('CLOSE_SHORT', current_price, pnl, reason)
                self.last_exit_reason = 'take_profit'
                logger.info(f"🎯 TAKE PROFIT hit - closing SHORT @ ${current_price:.2f}, P&L: ${pnl:.2f}")
            # After TP we can consider opening new position based on model
            trade_executed = True
        
        # Model-driven trades (only when flat or model says to flip)
        if action == 1 and self.position != 1:  # BUY signal
            if self.position == -1:  # Close short first
                pnl = self._close_position(current_price)
                self._log_trade('CLOSE_SHORT', current_price, pnl, reason)
                self.last_exit_reason = 'model'
                trade_executed = True
                
            # Open long (if currently flat) - check whale trend filter
            if self.position == 0:
                # Whale trend filter: don't go long if whales are bearish
                if self.whale_tracker and self.use_trend_filter:
                    can_trade, whale_reason = self.whale_tracker.should_trade_long(threshold=-0.15)
                    if not can_trade:
                        logger.info(f"🐋 BLOCKED LONG: {whale_reason}")
                        return False
                
                # Regime filter: don't go long in downtrend
                if self.regime_detector and self.use_trend_filter and hasattr(self, '_last_df'):
                    should_trade, regime_reason, size_mult = self.regime_detector.should_trade(self._last_df, "long")
                    if not should_trade:
                        logger.info(f"📊 BLOCKED LONG: {regime_reason}")
                        return False
                
                # MTF confluence filter: require all timeframes to align
                if self.mtf_analyzer and self.use_trend_filter:
                    can_trade, mtf_reason = self.mtf_analyzer.should_trade("long", self._last_df if hasattr(self, '_last_df') else None)
                    if not can_trade:
                        logger.info(f"📊 BLOCKED LONG: {mtf_reason}")
                        return False
                
                # Funding rate filter: don't go long if shorts are rewarded heavily
                if self.funding_analyzer and self.use_trend_filter:
                    can_trade, funding_reason = self.funding_analyzer.should_trade("long")
                    if not can_trade:
                        logger.info(f"💰 BLOCKED LONG: {funding_reason}")
                        return False
                
                # Order flow filter: don't go long if large sellers dominate
                if self.order_flow and self.use_trend_filter and hasattr(self, '_last_df'):
                    can_trade, flow_reason = self.order_flow.should_trade("long", self._last_df)
                    if not can_trade:
                        logger.info(f"📊 BLOCKED LONG: {flow_reason}")
                        return False
                        
                self._open_position(current_price, 1)
                self._log_trade('OPEN_LONG', current_price, 0, reason)
                self.last_trade_time = datetime.now()
                self.last_exit_reason = None
                trade_executed = True
            
        elif action == 2 and self.position != -1:  # SELL signal
            if self.position == 1:  # Close long first
                pnl = self._close_position(current_price)
                self._log_trade('CLOSE_LONG', current_price, pnl, reason)
                self.last_exit_reason = 'model'
                trade_executed = True
                
            # Open short (if currently flat) - check whale trend filter
            if self.position == 0:
                # Whale trend filter: don't go short if whales are bullish
                if self.whale_tracker and self.use_trend_filter:
                    can_trade, whale_reason = self.whale_tracker.should_trade_short(threshold=0.15)
                    if not can_trade:
                        logger.info(f"🐋 BLOCKED SHORT: {whale_reason}")
                        return False
                
                # Regime filter: don't go short in uptrend
                if self.regime_detector and self.use_trend_filter and hasattr(self, '_last_df'):
                    should_trade, regime_reason, size_mult = self.regime_detector.should_trade(self._last_df, "short")
                    if not should_trade:
                        logger.info(f"📊 BLOCKED SHORT: {regime_reason}")
                        return False
                
                # MTF confluence filter: require all timeframes to align
                if self.mtf_analyzer and self.use_trend_filter:
                    can_trade, mtf_reason = self.mtf_analyzer.should_trade("short", self._last_df if hasattr(self, '_last_df') else None)
                    if not can_trade:
                        logger.info(f"📊 BLOCKED SHORT: {mtf_reason}")
                        return False
                
                # Funding rate filter: don't go short if longs are rewarded heavily
                if self.funding_analyzer and self.use_trend_filter:
                    can_trade, funding_reason = self.funding_analyzer.should_trade("short")
                    if not can_trade:
                        logger.info(f"💰 BLOCKED SHORT: {funding_reason}")
                        return False
                
                # Order flow filter: don't go short if large buyers dominate
                if self.order_flow and self.use_trend_filter and hasattr(self, '_last_df'):
                    can_trade, flow_reason = self.order_flow.should_trade("short", self._last_df)
                    if not can_trade:
                        logger.info(f"📊 BLOCKED SHORT: {flow_reason}")
                        return False
                        
                self._open_position(current_price, -1)
                self._log_trade('OPEN_SHORT', current_price, 0, reason)
                self.last_trade_time = datetime.now()
                self.last_exit_reason = None
                trade_executed = True
            
        if trade_executed:
            logger.info(
                f"📊 Trade executed @ ${current_price:.2f} | "
                f"Position: {position_str[self.position]} | "
                f"Balance: ${self.balance:.2f} | "
                f"P&L: ${self.realized_pnl:.2f} | "
                f"Reason: {reason}"
            )
            
        return trade_executed
        
    def _open_position(self, price: float, position_type: int):
        """Open a new position with adaptive risk parameters."""
        # Get adaptive risk parameters if available
        if self.risk_manager and hasattr(self, '_last_df'):
            trade_type = "long" if position_type == 1 else "short"
            risk_params = self.risk_manager.get_risk_parameters(self._last_df, trade_type)
            
            # Use adaptive position size
            position_size_pct = risk_params.position_size
            
            # Store adaptive SL/TP for this trade
            self.current_sl_pct = risk_params.stop_loss_pct
            self.current_tp_pct = risk_params.take_profit_pct
            
            logger.info(
                f"📊 Adaptive Risk: SL={self.current_sl_pct:.2%}, TP={self.current_tp_pct:.2%}, "
                f"Size={position_size_pct:.0%}, R:R={risk_params.risk_reward_ratio:.2f}"
            )
        else:
            position_size_pct = self.position_size
            self.current_sl_pct = self.stop_loss_pct
            self.current_tp_pct = self.take_profit_pct
        
        trade_amount = self.balance * position_size_pct
        fee = trade_amount * 0.0004  # 0.04% fee
        
        self.position = position_type
        self.position_price = price
        self.position_size_units = (trade_amount - fee) / price
        self.balance -= fee
        
        # Reset trailing stop tracking
        self.highest_since_entry = price
        self.lowest_since_entry = price
        
    def _close_position(self, price: float) -> float:
        """Close current position and return P&L."""
        if self.position == 1:  # Long
            pnl = (price - self.position_price) * self.position_size_units
            pnl_pct = (price - self.position_price) / self.position_price
        elif self.position == -1:  # Short
            pnl = (self.position_price - price) * self.position_size_units
            pnl_pct = (self.position_price - price) / self.position_price
        else:
            return 0.0
            
        # Apply fee
        fee = abs(pnl) * 0.0004
        pnl -= fee
        
        self.balance += pnl
        self.realized_pnl += pnl
        self.daily_pnl += pnl
        
        # Record trade for Kelly calculation
        if self.risk_manager:
            self.risk_manager.record_trade(pnl_pct)
        
        # Reset position
        self.position = 0
        self.position_price = 0.0
        self.position_size_units = 0.0
        
        return pnl
        
    def _update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L."""
        if self.position == 0:
            self.unrealized_pnl = 0.0
            return
            
        if self.position == 1:  # Long
            self.unrealized_pnl = (current_price - self.position_price) * self.position_size_units
        else:  # Short
            self.unrealized_pnl = (self.position_price - current_price) * self.position_size_units
            
    def _log_trade(self, action: str, price: float, pnl: float, reason: str):
        """Log trade to file and persist state."""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'price': price,
            'pnl': pnl,
            'balance': self.balance,
            'position': self.position,
            'reason': reason,
        }
        self.trades.append(trade)
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(trade) + '\n')
        
        # Save state to persist position across restarts
        self._save_state()
            
    def run_once(self) -> Dict:
        """Run one trading cycle."""
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active - no trading")
            return {'status': 'circuit_breaker'}
        
        # Use WebSocket stream data if available, otherwise fetch from API
        if self._use_websocket and self.candle_stream.is_running():
            df = self.candle_stream.get_dataframe()
            if len(df) > 0:
                logger.debug(f"📊 Using WebSocket data: {len(df)} candles")
            else:
                logger.info("Fetching latest market data (WebSocket empty)...")
                df = self.fetch_latest_data()
        else:
            logger.info("Fetching latest market data...")
            df = self.fetch_latest_data()
        
        if len(df) < 300:
            logger.warning(f"Not enough data: {len(df)} candles")
            return {'status': 'insufficient_data'}
        
        # Store df for regime detector
        self._last_df = df
            
        # Compute observation
        result = self.compute_observation(df)
        if result is None:
            return {'status': 'observation_failed'}
            
        observation, current_price, market_features = result
        
        # Update unrealized P&L
        self._update_unrealized_pnl(current_price)
        
        # Check circuit breaker
        if self.check_circuit_breaker():
            return {'status': 'circuit_breaker'}
            
        # Get action from model
        action = self.get_action(observation)
        
        # Execute action
        self.execute_action(action, current_price)
        
        # Return status
        portfolio_value = self.balance + self.unrealized_pnl
        position_str = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}
        
        status = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'price': current_price,
            'action': ['HOLD', 'BUY', 'SELL'][action],
            'position': position_str[self.position],
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'return_pct': (portfolio_value - self.initial_balance) / self.initial_balance * 100,
            'trades_count': len(self.trades),
            'market_features': market_features,
        }
        
        return status
        
    def run_loop(self, interval_minutes: int = 60):
        """Run continuous trading loop with WebSocket streaming."""
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        
        logger.info("=" * 60)
        logger.info(f"🚀 ADVANCED TRADING BOT STARTED - {mode} MODE")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Position Size: {self.position_size*100:.0f}%")
        logger.info(f"Stop Loss: {self.stop_loss_pct*100:.1f}%")
        logger.info(f"Take Profit: {self.take_profit_pct*100:.1f}%")
        logger.info(f"Max Daily Loss: {self.max_daily_loss*100:.1f}%")
        logger.info(f"Check Interval: {interval_minutes} minutes")
        logger.info("=" * 60)
        
        # Start WebSocket stream (fetches 1000 candles once, then streams)
        if self._use_websocket:
            try:
                self.candle_stream.start()
                logger.info("🔌 WebSocket stream started - no more repeated API calls!")
            except Exception as e:
                logger.warning(f"WebSocket start failed, falling back to REST API: {e}")
                self._use_websocket = False
        
        try:
            while True:
                try:
                    status = self.run_once()
                    
                    if status['status'] == 'ok':
                        logger.info(
                            f"💰 {status['timestamp']} | "
                            f"Price: ${status['price']:.2f} | "
                            f"Action: {status['action']} | "
                            f"Position: {status['position']} | "
                            f"Portfolio: ${status['portfolio_value']:.2f} | "
                            f"Return: {status['return_pct']:.2f}%"
                        )
                    else:
                        logger.info(f"Status: {status['status']}")
                        
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    
                # Wait for next interval
                logger.info(f"Waiting {interval_minutes} minutes until next check...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Trading bot stopped by user")
            # Stop WebSocket stream
            if self._use_websocket and self.candle_stream:
                self.candle_stream.stop()
            self.print_summary()
            
    def print_summary(self):
        """Print trading summary."""
        portfolio_value = self.balance + self.unrealized_pnl
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance * 100
        
        print("\n" + "=" * 60)
        print("TRADING SESSION SUMMARY")
        print("=" * 60)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance:   ${self.balance:,.2f}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Realized P&L:    ${self.realized_pnl:,.2f}")
        print(f"Unrealized P&L:  ${self.unrealized_pnl:,.2f}")
        print(f"Total Return:    {total_return:.2f}%")
        print(f"Total Trades:    {len(self.trades)}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Advanced Live Trading Bot')
    parser.add_argument('--model', type=str, default='./data/models/ultimate_agent.zip',
                        help='Path to trained model')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in minutes')
    parser.add_argument('--dry-run', action='store_true', default=True, 
                        help='Run in dry-run mode (default)')
    parser.add_argument('--live', action='store_true', help='Run in live mode')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    # Create directories
    Path('./logs').mkdir(parents=True, exist_ok=True)
    
    # Initialize agent
    agent = LiveTradingAgent(
        model_path=args.model,
        symbol=args.symbol,
        initial_balance=args.balance,
        dry_run=not args.live,
    )
    
    if args.once:
        status = agent.run_once()
        print(json.dumps(status, indent=2, default=str))
        agent.print_summary()
    else:
        agent.run_loop(interval_minutes=args.interval)


if __name__ == '__main__':
    main()
