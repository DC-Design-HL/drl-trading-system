#!/usr/bin/env python3
"""
Multi-Asset Live Trading System

Runs trading bots for multiple assets simultaneously.
Each asset uses its own fine-tuned model from transfer learning.

Usage:
    python live_trading_multi.py --assets BTCUSDT ETHUSDT SOLUSDT --dry-run
    python live_trading_multi.py --assets ETHUSDT --interval 10  # Single asset
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data.multi_asset_fetcher import MultiAssetDataFetcher, SUPPORTED_ASSETS
from src.features.multi_asset_features import MultiAssetFeatureEngine
from src.features.whale_tracker import WhaleTracker
from src.features.regime_detector import MarketRegimeDetector
from src.features.mtf_analyzer import MultiTimeframeAnalyzer
from src.features.risk_manager import AdaptiveRiskManager
from src.features.mtf_analyzer import MultiTimeframeAnalyzer
from src.features.risk_manager import AdaptiveRiskManager
from src.features.order_flow import FundingRateAnalyzer, OrderFlowAnalyzer
from src.features.on_chain_whales import OnChainWhaleWatcher
from src.data.storage import get_storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAssetTradingBot:
    """
    Trading bot for a single asset.
    
    Multiple instances run concurrently for multi-asset trading.
    """
    
    MODEL_DIR = Path("./data/models/multi_asset")
    
    def __init__(
        self,
        symbol: str,
        dry_run: bool = True,
        initial_balance: float = 10000,
        position_size: float = 0.5,
    ):
        self.symbol = symbol
        self.dry_run = dry_run
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.last_equity = initial_balance
        self.position_size = position_size
        
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.position_price = 0.0
        self.current_price = 0.0  # Store latest price
        self.position_units = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        
        self.trades: List[Dict] = []
        self.realized_pnl = 0.0
        
        self._running = False
        
        # Load model
        self.model = self._load_model()
        
        # Initialize feature engine (cross-asset enabled to match training)
        # Note: This requires fetching BTC data even if trading other assets
        self.feature_engine = MultiAssetFeatureEngine(include_cross_asset=True)
        
        # If not BTC, we need to provide BTC data to the engine for cross-asset features
        # The fetcher handles this, but we need to ensure it's set in the engine
        if symbol != "BTCUSDT":
             # We'll handle this in get_action by fetching BTC data if needed
             pass
        
        # Initialize filters
        self.whale_tracker = WhaleTracker(symbol=symbol)
        self.funding_analyzer = FundingRateAnalyzer(symbol=symbol.replace("/", ""))
        self.order_flow = OrderFlowAnalyzer(symbol=symbol.replace("/", ""))
        try:
             # We can use a lightweight version or just run it
             # It caches internally for 1 min so it's safe to call often
             if not hasattr(self, 'mtf_analyzer'):
                 self.mtf_analyzer = MultiTimeframeAnalyzer(symbol=self.symbol)
        except Exception as e:
                logger.warning(f"MTF analyzer init failed: {e}")

        self.risk_manager = AdaptiveRiskManager()
        
        logger.info(f"🤖 Bot initialized for {symbol} (dry-run={dry_run})")
    
    def get_market_analysis(self) -> Dict:
        """Get current market analysis data for dashboard."""
        try:
            # Whale
            whale = self.whale_tracker.get_whale_signals() if hasattr(self, 'whale_tracker') else {}
            
            # Funding
            funding = self.funding_analyzer.get_signal() if hasattr(self, 'funding_analyzer') else None
            funding_data = {
                'rate': funding.rate if funding else 0,
                'signal': funding.signal if funding else 'neutral',
                'strength': funding.strength if funding else 0
            }
            
            # Order Flow
            of = self.order_flow.analyze_large_orders() if hasattr(self, 'order_flow') else {}
            
            # MTF Analysis (Run on demand for dashboard state)
            mtf_data = {}
            if hasattr(self, 'symbol'):
                try:
                    # We can use a lightweight version or just run it
                    # It caches internally for 1 min so it's safe to call often
                    if not hasattr(self, 'mtf_analyzer'):
                        self.mtf_analyzer = MultiTimeframeAnalyzer(symbol=self.symbol)
                    
                    mtf_res = self.mtf_analyzer.get_summary()
                    mtf_data = {
                        'bias': 'BULLISH' if mtf_res['aligned'] and mtf_res['direction'] == 'bullish' else 
                                'BEARISH' if mtf_res['aligned'] and mtf_res['direction'] == 'bearish' else 'NEUTRAL',
                        'aligned': mtf_res['aligned'],
                        'reason': mtf_res['recommendation'],
                        '4h': mtf_res['signals'].get('4h', {}).get('direction', 'neutral'),
                        '1h': mtf_res['signals'].get('1h', {}).get('direction', 'neutral'),
                        '15m': mtf_res['signals'].get('15m', {}).get('direction', 'neutral')
                    }
                except Exception as e:
                    logger.warning(f"MTF analysis failed for state: {e}")
            
            return {
                'whale': whale,
                'funding': funding_data,
                'order_flow': of,
                'mtf': mtf_data
            }
        except Exception as e:
            logger.error(f"Error getting analysis for state: {e}")
            return {}

    def _load_model(self) -> PPO:
        """Load the trained model for this asset."""
        asset_name = self.symbol.replace("USDT", "").lower()
        
        # Try asset-specific model in subdirectory
        model_candidates = [
            self.MODEL_DIR / asset_name / "best_model.zip",
            self.MODEL_DIR / asset_name / "final_model.zip",
            self.MODEL_DIR / f"{asset_name}_agent.zip"
        ]
        
        market_model = None
        for path in model_candidates:
            if path.exists():
                logger.info(f"📥 Found model for {self.symbol}: {path}")
                market_model = path
                break
        
        if not market_model:
            # Try base BTC model
            btc_path = self.MODEL_DIR / "base_btc" / "best_model.zip"
            if btc_path.exists():
                 logger.info(f"Using base BTC model for {self.symbol}")
                 market_model = btc_path
        
        if not market_model:
            # Fall back to original ultimate agent
            ultimate_path = Path("./data/models/ultimate_agent.zip")
            if ultimate_path.exists():
                market_model = ultimate_path

        if not market_model:
            logger.warning(f"⚠️ No model found for {self.symbol}. Running in observation mode.")
            return None
        
        logger.info(f"📥 Loading model for {self.symbol}: {market_model}")
        return PPO.load(str(market_model))
    
    def restore_state(self, state: Dict):
        """Restore bot state from saved dictionary."""
        try:
            self.position = state.get('position', 0)
            self.position_price = state.get('price', 0.0)
            self.balance = state.get('balance', self.initial_balance)
            self.realized_pnl = state.get('pnl', 0.0)
            self.sl_price = state.get('sl', 0.0)
            self.tp_price = state.get('tp', 0.0)
            self.position_units = state.get('units', 0.0)
            
            # Fallback for legacy state without units
            if self.position != 0 and self.position_units == 0 and self.position_price > 0:
                logger.warning(f"⚠️ 'units' missing for {self.symbol}, estimating based on balance...")
                # Assuming 50% position size meant trade_value ~= current_balance
                self.position_units = self.balance / self.position_price
            
            logger.info(f"♻️ Restored state for {self.symbol}: Pos={self.position}, Entry=${self.position_price:.2f}, SL=${self.sl_price:.2f}")
        except Exception as e:
            logger.error(f"Failed to restore state for {self.symbol}: {e}")

    def fetch_data(self, days: int = 7, interval: str = "1h") -> pd.DataFrame:
        """Fetch recent data for the asset."""
        fetcher = MultiAssetDataFetcher()
        return fetcher.fetch_asset(self.symbol, interval, days)
    
    def get_action(self, df: pd.DataFrame) -> int:
        """Get trading action from model (0=HOLD, 1=BUY, 2=SELL)."""
        lookback = 30  # Must match training config
        
        if len(df) < lookback:
            logger.warning(f"Not enough data for {self.symbol}: {len(df)} < {lookback}")
            return 0
            
        # Optimization: batch compute
        all_features = self.feature_engine.compute_features_batch(df, self.symbol)
        
        # Take last 'lookback' rows
        window_features = all_features[-lookback:]
        
        flat_history = window_features.flatten()
        
        # 3. Agent State
        unrealized_pnl_ratio = 0.0
        if self.position != 0:
            current_price = df.iloc[-1]['close']
            if self.position == 1:
                u_pnl = (current_price - self.position_price) * self.position_units
            else:
                u_pnl = (self.position_price - current_price) * self.position_units
            unrealized_pnl_ratio = u_pnl / self.initial_balance
            
        balance_ratio = self.balance / self.initial_balance
        
        # Drawdown (simplified)
        drawdown = 0.0 # for now
        
        trade_count_norm = min(len(self.trades) / 100, 1.0)
        
        agent_state = np.array([
            float(self.position),
            unrealized_pnl_ratio,
            balance_ratio,
            drawdown,
            trade_count_norm
        ], dtype=np.float32)
        
        # Concatenate
        observation = np.concatenate([flat_history, agent_state])
        
        # If model is missing, return HOLD (0)
        if self.model is None:
            return 0
            
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
    
    def apply_filters(self, action: int) -> tuple:
        """Apply trading filters (whale, funding, order flow)."""
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        if action == 0:
            return action, "HOLD signal"
        
        # Whale filter
        try:
            whale_signals = self.whale_tracker.get_whale_signals()
            if action == 1 and whale_signals.get('score', 0) < -0.2:
                return 0, f"🐋 Whale bearish (score={whale_signals.get('score', 0):.2f})"
            if action == 2 and whale_signals.get('score', 0) > 0.2:
                return 0, f"🐋 Whale bullish (score={whale_signals.get('score', 0):.2f})"
        except Exception as e:
            logger.warning(f"Whale filter error: {e}")
        
        # Funding filter
        try:
            funding = self.funding_analyzer.get_signal()
            if action == 1 and "short_favored" in funding.signal and funding.strength > 0.5:
                return 0, f"💰 Funding short-favored ({funding.rate:.4%})"
            if action == 2 and "long_favored" in funding.signal and funding.strength > 0.5:
                return 0, f"💰 Funding long-favored ({funding.rate:.4%})"
        except Exception as e:
            logger.warning(f"Funding filter error: {e}")
        
        # Order flow filter
        try:
            of = self.order_flow.analyze_large_orders()
            if action == 1 and of.get('bias') == 'bearish':
                return 0, f"📊 Order flow bearish"
            if action == 2 and of.get('bias') == 'bullish':
                return 0, f"📊 Order flow bullish"
        except Exception as e:
            logger.warning(f"Order flow filter error: {e}")
            
        # MTF Trend Filter (Crucial for avoiding counter-trend losses)
        try:
            # Running this lightweight check every time is safe due to internal caching
            if not hasattr(self, 'mtf_analyzer'):
                 self.mtf_analyzer = MultiTimeframeAnalyzer(symbol=self.symbol)
            
            mtf_res = self.mtf_analyzer.get_summary()
            trend_4h = mtf_res.get('signals', {}).get('4h', {}).get('direction', 'neutral')
            
            # Rule: Don't buy if 4H is Bearish
            if action == 1 and trend_4h == 'bearish':
                return 0, f"📉 MTF Filter: 4H Trend is BEARISH (Block Long)"
            
            # Rule: Don't sell if 4H is Bullish
            if action == 2 and trend_4h == 'bullish':
                return 0, f"📈 MTF Filter: 4H Trend is BULLISH (Block Short)"
                
        except Exception as e:
            logger.warning(f"MTF filter error: {e}")
        
        return action, f"{action_names[action]} passed all filters"
    
    def execute_trade(self, action: int, current_price: float) -> Optional[Dict]:
        """Execute a trade (or simulate in dry-run mode)."""
        if action == 0:
            return None
        
        trade = {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "price": current_price,
            "dry_run": self.dry_run,
        }
        
        if action == 1:  # BUY/LONG
            if self.position == -1:
                # Close short first
                pnl = (self.position_price - current_price) * self.position_units
                self.balance += pnl + self.position_price * self.position_units
                self.realized_pnl += pnl
                trade["action"] = "CLOSE_SHORT"
                trade["pnl"] = pnl
                self.position = 0
                self.position_units = 0
            elif self.position == 0:
                # Open long
                trade_value = self.balance * self.position_size
                self.position_units = trade_value / current_price
                self.position_price = current_price
                self.position = 1
                self.balance -= trade_value
                trade["action"] = "OPEN_LONG"
                trade["units"] = self.position_units
                
                # Calculate SL/TP
                try:
                    df = self.fetch_data(days=3) # Need some history for ATR
                    sl_pct, tp_pct = self.risk_manager.get_adaptive_sl_tp(df, "long")
                    self.sl_price = current_price * (1 - sl_pct)
                    self.tp_price = current_price * (1 + tp_pct)
                    trade["sl"] = self.sl_price
                    trade["tp"] = self.tp_price
                    logger.info(f"🛡️ LONG SL: ${self.sl_price:.2f} (-{sl_pct:.2%}) | TP: ${self.tp_price:.2f} (+{tp_pct:.2%})")
                except Exception as e:
                    logger.error(f"Failed to calc SL/TP: {e}")
                    self.sl_price = current_price * 0.98
                    self.tp_price = current_price * 1.04
            else:
                # Already LONG - redundant
                return None
        
        elif action == 2:  # SELL/SHORT
            if self.position == 1:
                # Close long first
                pnl = (current_price - self.position_price) * self.position_units
                self.balance += self.position_price * self.position_units + pnl
                self.realized_pnl += pnl
                trade["action"] = "CLOSE_LONG"
                trade["pnl"] = pnl
                self.position = 0
                self.position_units = 0
            elif self.position == 0:
                # Open short
                trade_value = self.balance * self.position_size
                self.position_units = trade_value / current_price
                self.position_price = current_price
                self.position = -1
                self.balance -= trade_value
                trade["action"] = "OPEN_SHORT"
                trade["units"] = self.position_units
                
                # Calculate SL/TP
                try:
                    df = self.fetch_data(days=3)
                    sl_pct, tp_pct = self.risk_manager.get_adaptive_sl_tp(df, "short")
                    self.sl_price = current_price * (1 + sl_pct)
                    self.tp_price = current_price * (1 - tp_pct)
                    trade["sl"] = self.sl_price
                    trade["tp"] = self.tp_price
                    logger.info(f"🛡️ SHORT SL: ${self.sl_price:.2f} (-{sl_pct:.2%}) | TP: ${self.tp_price:.2f} (+{tp_pct:.2%})")
                except Exception as e:
                    logger.error(f"Failed to calc SL/TP: {e}")
                    self.sl_price = current_price * 1.02
                    self.tp_price = current_price * 0.96
            else:
                # Already SHORT - redundant
                return None
        
        self.trades.append(trade)
        return trade
    
    def run_iteration(self) -> Dict:
        """Run a single trading iteration."""
        # Fetch data
        df = self.fetch_data(days=7)
        
        if df.empty:
            return {"status": "error", "message": "No data fetched"}
        
        current_price = float(df.iloc[-1]['close'])
        self.current_price = current_price # Update stored price
        
        # Check SL/TP Hits first
        if self.position != 0:
            hit_sl = False
            hit_tp = False
            
            if self.position == 1: # LONG
                if current_price <= self.sl_price and self.sl_price > 0:
                    hit_sl = True
                elif current_price >= self.tp_price and self.tp_price > 0:
                    hit_tp = True
                    
                if hit_sl or hit_tp:
                    trade = self.execute_trade(2, current_price) # Sell to close
                    reason = "STOP_LOSS" if hit_sl else "TAKE_PROFIT"
                    logger.info(f"🛑 {reason} triggered for {self.symbol} @ ${current_price:.2f}")
                    
                    # Return result immediately
                    # Calculate unrealized (now 0)
                    total_equity = self.balance
                    self.last_equity = total_equity
                    
                    return {
                        "symbol": self.symbol,
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price,
                        "raw_action": "HOLD",
                        "filtered_action": "CLOSE_LONG",
                        "reason": reason,
                        "position": 0,
                        "balance": self.balance,
                        "equity": total_equity,
                        "realized_pnl": self.realized_pnl,
                        "unrealized_pnl": 0,
                        "trade": trade,
                        "sl": 0,
                        "tp": 0
                    }

            elif self.position == -1: # SHORT
                if current_price >= self.sl_price and self.sl_price > 0:
                    hit_sl = True
                elif current_price <= self.tp_price and self.tp_price > 0:
                    hit_tp = True
                    
                if hit_sl or hit_tp:
                    trade = self.execute_trade(1, current_price) # Buy to close
                    reason = "STOP_LOSS" if hit_sl else "TAKE_PROFIT"
                    logger.info(f"🛑 {reason} triggered for {self.symbol} @ ${current_price:.2f}")
                    
                    # Return result immediately
                    total_equity = self.balance
                    self.last_equity = total_equity
                    
                    return {
                        "symbol": self.symbol,
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price,
                        "raw_action": "HOLD",
                        "filtered_action": "CLOSE_SHORT",
                        "reason": reason,
                        "position": 0,
                        "balance": self.balance,
                        "equity": total_equity,
                        "realized_pnl": self.realized_pnl,
                        "unrealized_pnl": 0,
                        "trade": trade,
                        "sl": 0,
                        "tp": 0
                    }
        
        # Get model action
        raw_action = self.get_action(df)
        
        # Apply filters
        filtered_action, reason = self.apply_filters(raw_action)
        
        # Execute if action changed
        trade = None
        if filtered_action != 0:
            trade = self.execute_trade(filtered_action, current_price)
        
        # Calculate unrealized P&L
        if self.position == 1:
            unrealized_pnl = (current_price - self.position_price) * self.position_units
        elif self.position == -1:
            unrealized_pnl = (self.position_price - current_price) * self.position_units
        else:
            unrealized_pnl = 0
        
        total_equity = self.balance + abs(self.position) * self.position_units * current_price + unrealized_pnl
        self.last_equity = total_equity
        
        return {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "price": current_price,
            "raw_action": ["HOLD", "BUY", "SELL"][raw_action],
            "filtered_action": ["HOLD", "BUY", "SELL"][filtered_action],
            "reason": reason,
            "position": self.position,
            "balance": self.balance,
            "equity": total_equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "trade": trade,
            "sl": self.sl_price if self.position != 0 else 0,
            "tp": self.tp_price if self.position != 0 else 0
        }

class MultiAssetOrchestrator:
    """
    Orchestrates multiple trading bots running concurrently.
    """
    
    def __init__(
        self,
        symbols: List[str],
        dry_run: bool = True,
        balance_per_asset: float = 10000,
    ):
        self.symbols = symbols
        self.dry_run = dry_run
        
        # Create bots
        self.bots: Dict[str, MultiAssetTradingBot] = {}
        for symbol in symbols:
            self.bots[symbol] = MultiAssetTradingBot(
                symbol=symbol,
                dry_run=dry_run,
                initial_balance=balance_per_asset,
            )
        
        self.storage = get_storage()
        self.load_state()
        
        self._running = False
        
        # Initialize On-Chain Whale Watcher
        self.whale_watcher = OnChainWhaleWatcher()
        self.last_trade_time = 0
        self.last_whale_check = 0
        
        logger.info(f"🚀 Orchestrator initialized for {len(symbols)} assets")
    
    def load_state(self):
        """Load state from storage."""
        try:
            state = self.storage.load_state()
            if not state:
                return
                
            assets = state.get('assets', {})
            for symbol, asset_state in assets.items():
                if symbol in self.bots:
                    self.bots[symbol].restore_state(asset_state)
                    
            logger.info(f"💾 Loaded state for {len(assets)} assets")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def run_single_cycle(self) -> Dict[str, Dict]:
        """Run one trading cycle for all assets."""
        results = {}
        
        for symbol, bot in self.bots.items():
            try:
                result = bot.run_iteration()
                results[symbol] = result
                
                # Log result
                action = result.get('filtered_action', 'HOLD')
                price = result.get('price', 0)
                pnl = result.get('realized_pnl', 0)
                
                # Save to trading log if action taken and trade executed
                if action != 'HOLD' and result.get('trade'):
                    self.log_trade_to_file(symbol, result)
                
                emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
                logger.info(f"{emoji} {symbol}: {action} @ ${price:,.2f} | P&L: ${pnl:,.2f}")
                
            except Exception as e:
                logger.error(f"Error running {symbol}: {e}")
                results[symbol] = {"status": "error", "message": str(e)}
        
        return results
    
    def run_loop(self, interval_minutes: int = 60):
        """Run continuous trading loop."""
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        
        logger.info("=" * 60)
        logger.info(f"🚀 MULTI-ASSET TRADING STARTED - {mode} MODE")
        logger.info(f"   Assets: {', '.join(self.symbols)}")
        logger.info(f"   Interval: {interval_minutes} minutes")
        logger.info("=" * 60)
        
        self._running = True
        self.last_trade_time = 0
        
        logger.info(f"🔄 Loop started: Trading every {interval_minutes}m, Monitoring every 1m")
        
        try:
            while self._running:
                current_time = time.time()
                
                # Check On-Chain Whales (every 2 minutes)
                if current_time - self.last_whale_check > 120:
                    try:
                        alerts = self.whale_watcher.check_all()
                        if alerts:
                            logger.info(f"🐋 {len(alerts)} new whale alerts detected")
                        self.last_whale_check = current_time
                    except Exception as e:
                        logger.error(f"Whale watcher failed: {e}")

                # Check if it's time to trade
                if current_time - self.last_trade_time >= interval_minutes * 60:
                    logger.info(f"⏰ Starting trading cycle (Last: {datetime.fromtimestamp(self.last_trade_time).strftime('%H:%M') if self.last_trade_time else 'Never'})")
                    
                    # Run trading cycle
                    results = self.run_single_cycle()
                    self.last_trade_time = current_time
                    
                    # Print summary
                    total_pnl = sum(
                        r.get('realized_pnl', 0) 
                        for r in results.values() 
                        if isinstance(r, dict)
                    )
                    logger.info(f"📊 Total P&L across all assets: ${total_pnl:,.2f}")
                
                # Save state frequently (every loop) for dashboard updates
                self.save_state()
                
                # Force garbage collection to prevent OOM
                import gc
                gc.collect()
                
                # Sleep for 60s
                time.sleep(60)
            
            # Save state after loop ends
            self.save_state()
                    
        except KeyboardInterrupt:
            logger.info("🛑 Trading stopped by user")
            self._running = False
            self.save_state()
            
    def log_trade_to_file(self, symbol: str, trade_data: dict):
        """Log trade to storage."""
        try:
            # Prepare trade record
            record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "asset": symbol,
                "action": trade_data.get('trade', {}).get('action', trade_data.get('filtered_action', 'HOLD')),
                "price": trade_data.get('trade', {}).get('price', trade_data.get('price', 0)),
                "pnl": trade_data.get('trade', {}).get('pnl', 0),
                "balance": trade_data.get('balance', 0),
                "position": trade_data.get('position', 0),
                "reason": trade_data.get('reason', 'model')
            }
            
            self.storage.log_trade(record)
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            
    def save_state(self):
        """Save current state to JSON for dashboard."""
        state = self.get_portfolio_status()
        state['timestamp'] = datetime.now().isoformat()
        state['active'] = self._running
        state['whale_alerts'] = self.whale_watcher.get_latest_alerts()
        
        # Add per-asset details
        state['assets'] = {}
        for symbol, bot in self.bots.items():
            state['assets'][symbol] = {
                'price': bot.current_price if bot.current_price > 0 else bot.position_price, # Latest price for display
                'entry_price': bot.position_price, # Explicit entry price for P&L
                'position': bot.position,
                'balance': bot.balance,
                'pnl': bot.realized_pnl,
                'trades': bot.trades[-5:] if bot.trades else [], # last 5 trades
                'last_action': bot.trades[-1].get('action', 'NONE') if bot.trades else 'NONE',
                'sl': bot.sl_price,
                'tp': bot.tp_price,
                'units': bot.position_units,
                'equity': bot.last_equity,
                'analysis': bot.get_market_analysis() # Add analysis data
            }
            
        self.storage.save_state(state)
    
    def stop(self):
        """Stop the trading loop."""
        self._running = False
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        total_balance = 0
        total_pnl = 0
        positions = {}
        
        for symbol, bot in self.bots.items():
            total_balance += bot.last_equity
            total_pnl += bot.realized_pnl
            positions[symbol] = {
                "position": bot.position,
                "balance": bot.balance,
                "pnl": bot.realized_pnl,
            }
        
        return {
            "total_balance": total_balance,
            "total_pnl": total_pnl,
            "positions": positions,
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Live Trading")
    
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Assets to trade"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in simulation mode"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (use with caution!)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Trading interval in minutes"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10000,
        help="Starting balance per asset"
    )
    
    args = parser.parse_args()
    
    dry_run = not args.live
    
    if not dry_run:
        logger.warning("⚠️ LIVE TRADING MODE - Real money at risk!")
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Live trading cancelled")
            return
    
    orchestrator = MultiAssetOrchestrator(
        symbols=args.assets,
        dry_run=dry_run,
        balance_per_asset=args.balance,
    )
    
    orchestrator.run_loop(interval_minutes=args.interval)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("crash.log", "w") as f:
            f.write(f"CRASH TIME: {datetime.now().isoformat()}\n")
            f.write(f"ERROR: {e}\n")
            traceback.print_exc(file=f)
        logger.critical(f"🔥 FATAL CRASH: {e}", exc_info=True)
        sys.exit(1)
