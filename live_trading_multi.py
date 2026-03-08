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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Old imports removed: EnsembleOrchestrator, resample_to_higher_tf, compute_mtf_features_at
from src.data.multi_asset_fetcher import MultiAssetDataFetcher, SUPPORTED_ASSETS
from src.features.multi_asset_features import MultiAssetFeatureEngine
from src.features.whale_tracker import WhaleTracker
from src.features.regime_detector import MarketRegimeDetector
from src.features.mtf_analyzer import MultiTimeframeAnalyzer
from src.features.risk_manager import AdaptiveRiskManager
from src.features.order_flow import FundingRateAnalyzer, OrderFlowAnalyzer
from src.features.on_chain_whales import OnChainWhaleWatcher
from src.data.storage import get_storage
from src.models.price_forecaster import TFTForecaster
from src.models.confidence_engine import ConfidenceEngine
from src.api.portfolio_manager import GlobalPortfolioManager

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
        position_size: float = 0.25,  # 25% of balance per trade (was 50%)
        portfolio_manager = None,
    ):
        self.symbol = symbol
        self.dry_run = dry_run
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.last_equity = initial_balance
        self.position_size = position_size
        self.portfolio_manager = portfolio_manager
        
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.position_price = 0.0
        self.current_price = 0.0  # Store latest price
        self.position_units = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.highest_price = 0.0
        self.lowest_price = 0.0
        self.base_trailing_pct = 0.0
        
        self.pending_orders: List[Dict] = []
        
        self.trades: List[Dict] = []
        self.realized_pnl = 0.0
        
        # Anti-overtrading guards
        self.last_loss_time = 0        # Timestamp of last SL hit
        self.last_entry_time = 0       # Timestamp of last position open
        self.COOLDOWN_SECONDS = 1800   # 30 min cooldown after a loss
        self.MIN_HOLD_SECONDS = 14400  # 4 hour minimum hold time (was 2h)
        
        self._running = False
        
        # Load model (Ultimate Agent - whale-fused PPO)
        self.model = self._load_model()
        
        # Initialize feature engine (MUST match UltimateTradingEnv)
        from src.features.ultimate_features import UltimateFeatureEngine
        from src.features.correlation_engine import SimulatedDominanceEngine
        self.feature_engine = UltimateFeatureEngine()
        self.dominance_engine = SimulatedDominanceEngine()
        
        # Initialize market analysis sources
        self.whale_tracker = WhaleTracker(symbol=symbol)
        self.funding_analyzer = FundingRateAnalyzer(symbol=symbol.replace("/", ""))
        self.order_flow = OrderFlowAnalyzer(symbol=symbol.replace("/", ""))
        try:
             if not hasattr(self, 'mtf_analyzer'):
                 self.mtf_analyzer = MultiTimeframeAnalyzer(symbol=self.symbol)
        except Exception as e:
                logger.warning(f"MTF analyzer init failed: {e}")

        # Regime detector (ADX/ATR-based)
        self.regime_detector = MarketRegimeDetector()

        self.risk_manager = AdaptiveRiskManager()
        self.confidence_engine = ConfidenceEngine()
        self.last_confidence = 0.5  # Will be computed by compute_market_score()
        self.last_forecast = None   # Cached TFT forecast for dashboard
        
        # TFT Price Forecaster (Phase 11.1)
        self.tft_forecaster = TFTForecaster(device='cpu')
        if self.tft_forecaster.load_model(symbol=symbol.replace('/', '')):
            logger.info(f"🔮 TFT forecaster loaded for {symbol}")
        else:
            logger.warning(f"⚠️ No TFT model for {symbol}, running without forecaster")
            self.tft_forecaster = None
        
        logger.info(f"🤖 Bot initialized for {symbol} (dry-run={dry_run})")
        
        # Load VecNormalize stats from training (CRITICAL for correct predictions)
        self.vec_normalize = self._load_vec_normalize()
    
    def _load_vec_normalize(self) -> Optional[VecNormalize]:
        """Load VecNormalize stats from training for proper observation normalization."""
        vec_norm_path = Path('./data/models/ultimate_agent_vec_normalize.pkl')
        if vec_norm_path.exists():
            try:
                import gymnasium as gym
                # Create a dummy env for VecNormalize to wrap
                dummy_env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
                vec_norm = VecNormalize.load(str(vec_norm_path), dummy_env)
                vec_norm.training = False  # Don't update running stats
                vec_norm.norm_reward = False
                logger.info(f"✅ VecNormalize stats loaded from {vec_norm_path}")
                return vec_norm
            except Exception as e:
                logger.warning(f"⚠️ Could not load VecNormalize: {e}")
                return None
        logger.warning(f"⚠️ VecNormalize not found at {vec_norm_path}")
        return None
    
    def get_market_analysis(self) -> Dict:
        """Get current market analysis data for dashboard."""
        import time
        current_time = time.time()
        
        # 2-minute cache (was 1 hour — too stale for dashboard)
        if hasattr(self, '_last_analysis_time') and hasattr(self, '_cached_analysis'):
            if current_time - self._last_analysis_time < 120 and self._cached_analysis:
                return self._cached_analysis
                
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
            
            # Order Flow (enhanced 3-layer signal)
            try:
                df_of = self.fetch_data(days=3) if hasattr(self, 'fetch_data') else None
                of = self.order_flow.get_enhanced_signal(df_of) if hasattr(self, 'order_flow') else {}
            except Exception as e:
                logger.warning(f"Order flow for dashboard failed: {e}")
                of = {}
            
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
            
            analysis = {
                'whale': whale,
                'funding': funding_data,
                'order_flow': of,
                'mtf': mtf_data,
                'forecast': getattr(self, 'last_forecast', None),
                'confidence': getattr(self, 'last_confidence', 0.5),
                'regime': getattr(self.model.classifier, 'last_regime_info', None) if hasattr(self, 'model') and hasattr(self.model, 'classifier') else None
            }
            
            self._last_analysis_time = current_time
            self._cached_analysis = analysis
            return analysis
        except Exception as e:
            logger.error(f"Error getting analysis for state: {e}")
            return {}

    def _load_model(self) -> Optional[PPO]:
        """Load the ultimate whale-fused PPO agent."""
        model_path = Path('./data/models/ultimate_agent.zip')
        
        if model_path.exists():
            try:
                model = PPO.load(str(model_path))
                logger.info(f"✅ Ultimate Agent loaded for {self.symbol} from {model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load Ultimate Agent: {e}")
                return None
        
        logger.warning(f"⚠️ Ultimate Agent not found at {model_path}. Bot will HOLD.")
        return None
    
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
        """Get trading action from model (0=HOLD, 1=BUY, 2=SELL).
        
        MUST match UltimateTradingEnv._get_observation() exactly:
        observation = [features_row (99 values) | position_info (3 values)] = 102 total
        """
        if len(df) < 200:
            logger.warning(f"Not enough data for {self.symbol}: {len(df)} < 200")
            return 0
        
        if self.model is None:
            return 0
            
        # Compute features using UltimateFeatureEngine (same as training env)
        all_features = self.feature_engine.get_all_features(df)
        
        # Add simulated dominance features
        dominance_features = self.dominance_engine.compute_simulated_dominance(df)
        all_features.update(dominance_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Clip extreme values
        for col in features_df.columns:
            if features_df[col].dtype in [np.float64, np.float32]:
                features_df[col] = features_df[col].clip(-10, 10)
        
        features = features_df.values.astype(np.float32)
        
        if len(features) < 1:
            return 0
        
        # Get LAST ROW only (matches UltimateTradingEnv._get_observation)
        last_features = features[-1].copy()
        
        # Position info (EXACTLY matches UltimateTradingEnv)
        current_price = df.iloc[-1]['close']
        if self.position != 0 and self.position_price > 0:
            if self.position == 1:
                unrealized_pnl = (current_price - self.position_price) / self.position_price
            else:
                unrealized_pnl = (self.position_price - current_price) / self.position_price
        else:
            unrealized_pnl = 0.0
        
        balance_ratio = (self.balance - self.initial_balance) / self.initial_balance
        
        position_info = np.array([
            float(self.position),
            np.clip(unrealized_pnl, -0.5, 0.5),
            np.clip(balance_ratio, -0.5, 0.5),
        ], dtype=np.float32)
        
        # Combine: [99 features | 3 position] = 102 total
        observation = np.concatenate([last_features, position_info]).astype(np.float32)
        
        # Apply VecNormalize if available (CRITICAL: model was trained with normalized obs)
        if self.vec_normalize is not None:
            try:
                obs_2d = observation.reshape(1, -1)
                observation = self.vec_normalize.normalize_obs(obs_2d).flatten().astype(np.float32)
            except Exception as e:
                logger.warning(f"VecNormalize application failed: {e}")
        
        # PPO.predict returns (action, state)
        action, _ = self.model.predict(observation, deterministic=True)
        action = int(action.item() if hasattr(action, 'item') else action)
        
        # Confidence is computed by compute_market_score(), not hardcoded
        return action
    
    def make_decision(self, composite_score: float, confidence: float, raw_action: int, df: pd.DataFrame) -> tuple:
        """
        3-tier decision system that overrides weak PPO model when market signals are clear.
        
        Tier 1 (confidence >= 0.65): Override PPO — trade based on market signals
        Tier 2 (confidence 0.45-0.65): Only trade if PPO agrees with signals
        Tier 3 (confidence < 0.45): Hold — insufficient signal clarity
        
        Returns:
            (final_action, reason)
        """
        # ── Position Guard: Don't suggest action if already in that direction ──
        if raw_action == 1 and self.position == 1:
            return 0, "Already LONG — ignoring BUY signal"
        if raw_action == 2 and self.position == -1:
            return 0, "Already SHORT — ignoring SELL signal"
        
        # Determine market-preferred direction from composite score
        if composite_score > 0.10:
            market_action = 1  # Market says BUY
        elif composite_score < -0.10:
            market_action = 2  # Market says SELL
        else:
            market_action = 0  # Market is neutral
        
        # Also guard market action against current position
        if market_action == 1 and self.position == 1:
            market_action = 0  # Already LONG, don't suggest another BUY
        if market_action == 2 and self.position == -1:
            market_action = 0  # Already SHORT, don't suggest another SELL
        
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        # --- Tier 1: SIGNAL OVERRIDE (strong signals) ---
        if confidence >= 0.65 and market_action != 0:
            logger.info(
                f"🎯 SIGNAL OVERRIDE for {self.symbol}: {action_names[market_action]} "
                f"(conf={confidence:.2f}, composite={composite_score:+.3f}, PPO={action_names[raw_action]})"
            )
            return market_action, f"🎯 SIGNAL OVERRIDE: {action_names[market_action]} (conf={confidence:.2f})"
        
        # --- Tier 2: CONSENSUS (moderate signals + PPO agrees) ---
        if confidence >= 0.45:
            if raw_action == market_action and market_action != 0:
                logger.info(
                    f"🤝 CONSENSUS for {self.symbol}: {action_names[raw_action]} "
                    f"(PPO + signals agree, conf={confidence:.2f})"
                )
                return raw_action, f"🤝 PPO+Signal agree: {action_names[raw_action]} (conf={confidence:.2f})"
            elif raw_action != 0 and market_action == 0:
                # Market neutral, PPO has opinion — use PPO with caution
                logger.info(
                    f"⚖️ PPO-only for {self.symbol}: {action_names[raw_action]} "
                    f"(market neutral, conf={confidence:.2f})"
                )
                return raw_action, f"⚖️ PPO signal (market neutral, conf={confidence:.2f})"
            else:
                # PPO disagrees with market signals
                logger.info(
                    f"🚫 DISAGREEMENT for {self.symbol}: PPO={action_names[raw_action]}, "
                    f"Market={action_names[market_action]} → HOLD"
                )
                return 0, f"🚫 PPO ({action_names[raw_action]}) disagrees with market ({action_names[market_action]})"
        
        # --- Tier 3: HOLD (weak signals) ---
        logger.info(
            f"😶 LOW CONFIDENCE for {self.symbol}: conf={confidence:.2f}, HOLD"
        )
        return 0, f"😶 Low confidence ({confidence:.2f}), HOLD"
    
    def compute_market_score(self, action: int, df: pd.DataFrame) -> tuple:
        """
        Compute composite market confidence score from all analysis sources.
        
        Replaces the old binary apply_filters() with a unified scoring system.
        Each source provides a directional score [-1, +1] weighted into a composite.
        The composite score determines both trade filtering AND position sizing.
        
        Returns:
            (final_action, confidence, reason)
        """
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        if action == 0:
            return 0, 0.5, "HOLD signal"
        
        scores = {}  # name -> (directional_score, weight)
        details = []  # Human-readable breakdown
        
        # ── 1. Whale Signals (30% weight) ──────────────────────────────
        try:
            whale = self.whale_tracker.get_whale_signals()
            # FIX: The key returned by whale_tracker is 'combined_score', not 'score'
            whale_score = whale.get('combined_score', 0.0)
            scores['whale'] = (np.clip(whale_score, -1, 1), 0.30)
            
            # Squeeze Metric Boost
            squeeze = whale.get('squeeze_status', 'none')
            if squeeze == 'short_squeeze':
                scores['squeeze'] = (1.0, 0.40)  # Massive weight to force LONG
                details.append("🚨 SHORT SQUEEZE(+1.00)")
            elif squeeze == 'long_squeeze':
                scores['squeeze'] = (-1.0, 0.40) # Massive weight to force SHORT
                details.append("🚨 LONG SQUEEZE(-1.00)")
            else:
                details.append(f"🐋 Whale={whale_score:+.2f}")
                
        except Exception as e:
            logger.warning(f"Whale score error: {e}")
            scores['whale'] = (0, 0.30)
        
        # ── 2. Market Regime / ADX (20% weight) ───────────────────────
        try:
            regime_info = self.regime_detector.detect_regime(df)
            regime_score = regime_info.trend_direction  # Already [-1, +1]
            regime_name = regime_info.regime.value.upper()
            scores['regime'] = (regime_score, 0.20)
            details.append(f"📊 Regime={regime_name}({regime_score:+.2f})")
            
            # Hard veto: Catching falling knives/fading strong trends
            if action == 1 and regime_name == 'TRENDING_DOWN':
                whale_s = scores.get('whale', (0, 0))[0]
                if regime_info.confidence > 0.6 or whale_s <= 0:
                    return 0, 0.15, f"📉 Downtrend (ADX={regime_info.trend_strength:.0f}) without whale buying — block LONG"
            if action == 2 and regime_name == 'TRENDING_UP':
                whale_s = scores.get('whale', (0, 0))[0]
                if regime_info.confidence > 0.6 or whale_s >= 0:
                    return 0, 0.15, f"📈 Uptrend (ADX={regime_info.trend_strength:.0f}) without whale selling — block SHORT"
        except Exception as e:
            logger.warning(f"Regime score error: {e}")
            scores['regime'] = (0, 0.20)
        
        # ── 3. TFT Forecast (20% weight) ──────────────────────────────
        tft_score = 0.0
        if self.tft_forecaster:
            try:
                # Use cached forecast if available, else compute
                if self.last_forecast is None:
                    forecast = self.tft_forecaster.forecast(df)
                    self.last_forecast = forecast
                else:
                    forecast = self.last_forecast
                
                consensus = forecast.get('direction_consensus', 0.0)
                conf_4h = forecast.get('confidence_4h', 0.0)
                ret_4h = forecast.get('return_4h', 0.0)
                tft_score = np.clip(consensus, -1, 1)
                scores['tft'] = (tft_score, 0.20)
                details.append(f"🔮 TFT={consensus:+.2f}(c={conf_4h:.1f})")
                
                # Hard veto: TFT strongly disagrees with high confidence
                if action == 1 and consensus < -0.5 and conf_4h > 0.7:
                    return 0, 0.1, f"🔮 TFT strong bearish (consensus={consensus:.2f}, conf={conf_4h:.2f}) — veto BUY"
                if action == 2 and consensus > 0.5 and conf_4h > 0.7:
                    return 0, 0.1, f"🔮 TFT strong bullish (consensus={consensus:.2f}, conf={conf_4h:.2f}) — veto SELL"
            except Exception as e:
                logger.warning(f"TFT score error: {e}")
                scores['tft'] = (0, 0.20)
        else:
            scores['tft'] = (0, 0.20)
        
        # ── 4. Funding Rate (15% weight) ──────────────────────────────
        try:
            funding = self.funding_analyzer.get_signal()
            # Positive funding = longs pay = bearish bias; negative = bullish
            funding_score = np.clip(-funding.rate * 1000, -1, 1)  # Scale rate to [-1,1]
            scores['funding'] = (funding_score, 0.15)
            details.append(f"💰 Fund={funding.rate:+.4%}")
        except Exception as e:
            logger.warning(f"Funding score error: {e}")
            scores['funding'] = (0, 0.15)
        
        # ── 5. Order Flow — Enhanced 3-layer (15% weight) ──────────────
        try:
            of_signal = self.order_flow.get_enhanced_signal(df)
            flow_score = of_signal.get('score', 0.0)
            scores['order_flow'] = (flow_score, 0.15)
            # Rich detail for logging
            cvd_s = of_signal.get('cvd', {}).get('score', 0)
            taker_s = of_signal.get('taker', {}).get('score', 0)
            notable_s = of_signal.get('notable', {}).get('score', 0)
            details.append(f"📊 Flow={flow_score:+.2f}(cvd={cvd_s:+.1f}/tk={taker_s:+.1f}/lg={notable_s:+.1f})")
        except Exception as e:
            logger.warning(f"Order flow score error: {e}")
            scores['order_flow'] = (0, 0.15)
        
        # ── Compute Composite Score ───────────────────────────────────
        total_weight = sum(w for _, w in scores.values())
        composite = sum(s * w for s, w in scores.values()) / total_weight if total_weight > 0 else 0
        self._last_composite_score = composite  # Store for make_decision()
        
        # Convert composite to confidence for the ACTION direction
        # composite > 0 = market is bullish, composite < 0 = market is bearish
        if action == 1:    # BUY — bullish composite helps
            confidence = np.clip((composite + 1) / 2, 0, 1)  # Map [-1,+1] -> [0,1]
        elif action == 2:  # SELL — bearish composite helps
            confidence = np.clip((-composite + 1) / 2, 0, 1)
        else:
            confidence = 0.5
        
        # Log detailed breakdown
        detail_str = " | ".join(details)
        logger.info(
            f"📈 Market Score for {action_names[action]}: "
            f"composite={composite:+.3f}, confidence={confidence:.2f} | {detail_str}"
        )
        
        # ── Decision Gate ─────────────────────────────────────────────
        MIN_CONFIDENCE = 0.35
        
        if confidence < MIN_CONFIDENCE:
            return 0, confidence, (
                f"❌ Blocked {action_names[action]}: confidence={confidence:.2f} < {MIN_CONFIDENCE} "
                f"(composite={composite:+.3f}) | {detail_str}"
            )
            
        # ── REAL VOLUME CONFIRMATION (Prevent falling knife / dead bounce trades)
        whale_s = scores.get('whale', (0, 0))[0]
        flow_s = scores.get('order_flow', (0, 0))[0]
        
        # If confidence is moderate (Tier 2 Consensus), require volume confirmation
        if confidence < 0.65:
            if abs(whale_s) < 0.05 and abs(flow_s) < 0.05:
                return 0, confidence, (
                    f"❌ Blocked {action_names[action]}: No volume confirmation "
                    f"(Whale={whale_s:+.2f}, Flow={flow_s:+.2f}) in Tier 2 | {detail_str}"
                )
            
            # Check alignment: ensure volume isn't actively betting against the trade
            if action == 1 and whale_s <= 0 and flow_s <= 0:
                return 0, confidence, (
                    f"❌ Blocked {action_names[action]}: Volume is bearish or flat "
                    f"(Whale={whale_s:+.2f}, Flow={flow_s:+.2f}) | {detail_str}"
                )
            if action == 2 and whale_s >= 0 and flow_s >= 0:
                return 0, confidence, (
                    f"❌ Blocked {action_names[action]}: Volume is bullish or flat "
                    f"(Whale={whale_s:+.2f}, Flow={flow_s:+.2f}) | {detail_str}"
                )
        
        return action, confidence, (
            f"✅ {action_names[action]} approved: confidence={confidence:.2f} "
            f"(composite={composite:+.3f}) | {detail_str}"
        )
    
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
                self.pending_orders.clear() # Cancel any pending limit orders
                if self.portfolio_manager:
                    self.portfolio_manager.clear_position(self.symbol)
            elif self.position == 0:
                # Open long (Phase 11.6 Confidence Scaling)
                if self.portfolio_manager and not self.portfolio_manager.can_open_position(self.symbol, 1):
                    logger.info(f"🚫 LONG blocked for {self.symbol} by Global Portfolio Manager (Correlation Limit).")
                    return None
                    
                base_trade_value = self.balance * self.position_size
                scaled_trade_value = self.confidence_engine.apply_confidence(base_trade_value, self.last_confidence)
                
                if scaled_trade_value < 10.0:
                    logger.info(f"🚫 LONG blocked: Scaled position size (${scaled_trade_value:.2f}) too small due to low confidence ({self.last_confidence:.2f}).")
                    return None
                
                # Split-Entry Execution (Scale-In)
                market_trade_value = scaled_trade_value * 0.50
                limit_trade_value = scaled_trade_value * 0.50
                limit_price = current_price * 0.995 # 0.5% dip
                
                self.position_units = market_trade_value / current_price
                self.position_price = current_price
                self.position = 1
                self.balance -= market_trade_value
                
                self.pending_orders.append({
                    "type": "LIMIT_LONG",
                    "target_price": limit_price,
                    "value": limit_trade_value
                })
                
                trade["action"] = "OPEN_LONG_SPLIT"
                trade["units"] = self.position_units
                trade["confidence"] = self.last_confidence
                
                if self.portfolio_manager:
                    self.portfolio_manager.register_position(self.symbol, 1)
                
                # Calculate SL/TP for LONG position (structural)
                try:
                    df = self.fetch_data(days=3)
                    sl_pct, tp_pct = self.risk_manager.get_structural_sl_tp(df, "long")
                    
                    # Regime-adaptive adjustments
                    try:
                        regime_info = self.regime_detector.detect_regime(df)
                        regime_name = regime_info.regime.value
                        if regime_name == 'high_volatility':
                            sl_pct *= 1.5  # Wider stops in high vol
                            tp_pct *= 1.5
                            logger.info(f"📊 HIGH VOL regime: widened SL/TP by 1.5x")
                        elif regime_name == 'trending_up':
                            tp_pct *= 1.5  # Let winners run in trend
                            logger.info(f"📊 TRENDING_UP regime: widened TP by 1.5x")
                        elif regime_name == 'ranging':
                            sl_pct *= 1.5  # Wider stops in range to avoid getting chopped by noise
                            logger.info(f"📊 RANGING regime: widened SL by 1.5x to avoid chop")
                    except Exception as e:
                        logger.warning(f"Regime-adaptive SL/TP failed: {e}")
                    
                    self.sl_price = current_price * (1 - sl_pct)   # SL BELOW entry for LONG
                    self.tp_price = current_price * (1 + tp_pct)   # TP ABOVE entry for LONG
                    self.highest_price = current_price
                    self.base_trailing_pct = sl_pct
                    trade["sl"] = self.sl_price
                    trade["tp"] = self.tp_price
                    logger.info(f"🛡️ LONG SL: ${self.sl_price:.2f} (-{sl_pct:.2%}) | TP: ${self.tp_price:.2f} (+{tp_pct:.2%})")
                except Exception as e:
                    logger.error(f"Failed to calc SL/TP: {e}")
                    self.sl_price = current_price * 0.95   # 5% below entry
                    self.tp_price = current_price * 1.10   # 10% above entry
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
                self.pending_orders.clear() # Cancel pending limit orders
                if self.portfolio_manager:
                    self.portfolio_manager.clear_position(self.symbol)
            elif self.position == 0:
                # Open short (Phase 11.6 Confidence Scaling)
                if self.portfolio_manager and not self.portfolio_manager.can_open_position(self.symbol, -1):
                    logger.info(f"🚫 SHORT blocked for {self.symbol} by Global Portfolio Manager (Correlation Limit).")
                    return None
                    
                base_trade_value = self.balance * self.position_size
                scaled_trade_value = self.confidence_engine.apply_confidence(base_trade_value, self.last_confidence)
                
                if scaled_trade_value < 10.0:
                    logger.info(f"🚫 SHORT blocked: Scaled position size (${scaled_trade_value:.2f}) too small due to low confidence ({self.last_confidence:.2f}).")
                    return None
                
                # Split-Entry Execution (Scale-In)
                market_trade_value = scaled_trade_value * 0.50
                limit_trade_value = scaled_trade_value * 0.50
                limit_price = current_price * 1.005 # Catch 0.5% wick up
                
                self.position_units = market_trade_value / current_price
                self.position_price = current_price
                self.position = -1
                self.balance -= market_trade_value
                
                self.pending_orders.append({
                    "type": "LIMIT_SHORT",
                    "target_price": limit_price,
                    "value": limit_trade_value
                })
                
                trade["action"] = "OPEN_SHORT_SPLIT"
                trade["units"] = self.position_units
                trade["confidence"] = self.last_confidence
                
                if self.portfolio_manager:
                    self.portfolio_manager.register_position(self.symbol, -1)
                
                # Calculate SL/TP for SHORT position (structural)
                try:
                    df = self.fetch_data(days=3)
                    sl_pct, tp_pct = self.risk_manager.get_structural_sl_tp(df, "short")
                    
                    # Regime-adaptive adjustments
                    try:
                        regime_info = self.regime_detector.detect_regime(df)
                        regime_name = regime_info.regime.value
                        if regime_name == 'high_volatility':
                            sl_pct *= 1.5
                            tp_pct *= 1.5
                            logger.info(f"📊 HIGH VOL regime: widened SL/TP by 1.5x")
                        elif regime_name == 'trending_down':
                            tp_pct *= 1.5  # Let winners run in downtrend
                            logger.info(f"📊 TRENDING_DOWN regime: widened TP by 1.5x")
                        elif regime_name == 'ranging':
                            sl_pct *= 1.5
                            logger.info(f"📊 RANGING regime: widened SL by 1.5x to avoid chop")
                    except Exception as e:
                        logger.warning(f"Regime-adaptive SL/TP failed: {e}")
                    
                    self.sl_price = current_price * (1 + sl_pct)
                    self.tp_price = current_price * (1 - tp_pct)
                    self.lowest_price = current_price
                    self.base_trailing_pct = sl_pct
                    trade["sl"] = self.sl_price
                    trade["tp"] = self.tp_price
                    logger.info(f"🛡️ SHORT SL: ${self.sl_price:.2f} (-{sl_pct:.2%}) | TP: ${self.tp_price:.2f} (+{tp_pct:.2%})")
                except Exception as e:
                    logger.error(f"Failed to calc SL/TP: {e}")
                    self.sl_price = current_price * 1.05  # 5% above entry
                    self.tp_price = current_price * 0.90  # 10% below entry
            else:
                # Already SHORT - redundant
                return None
        
        self.trades.append(trade)
        return trade
    
    def run_iteration(self) -> Dict:
        """Run a single trading iteration."""
        # Fetch data
        df = self.fetch_data(days=14)
        
        if df.empty:
            return {"status": "error", "message": "No data fetched"}
        
        current_price = float(df.iloc[-1]['close'])
        self.current_price = current_price # Update stored price
        
        # Check and Fill Pending Limit Orders (Split-Entry Scaling)
        if self.pending_orders and self.position != 0:
            filled_orders = []
            for order in self.pending_orders:
                is_filled = False
                
                if order["type"] == "LIMIT_LONG" and self.position == 1:
                    if current_price <= order["target_price"]:
                        is_filled = True
                elif order["type"] == "LIMIT_SHORT" and self.position == -1:
                    if current_price >= order["target_price"]:
                        is_filled = True
                        
                if is_filled:
                    # Execute second half of position
                    units_added = order["value"] / current_price
                    total_units = self.position_units + units_added
                    
                    # Calculate new average entry price
                    total_value = (self.position_units * self.position_price) + (units_added * current_price)
                    self.position_price = total_value / total_units
                    self.position_units = total_units
                    self.balance -= order["value"]
                    
                    logger.info(f"🎯 LIMIT FILLED for {self.symbol}: Added {units_added:.4f} units @ ${current_price:.2f}. New Avg Entry: ${self.position_price:.2f}")
                    filled_orders.append(order)
                    
            # Remove filled orders
            for f_order in filled_orders:
                self.pending_orders.remove(f_order)
        
        # Check SL/TP Hits first
        if self.position != 0:
            hit_sl = False
            hit_tp = False
            hit_trailing = False
            reason = ""
            
            if self.position == 1: # LONG
                self.highest_price = max(self.highest_price, current_price)
                
                # Check normal SL/TP
                if current_price <= self.sl_price and self.sl_price > 0:
                    hit_sl = True
                    reason = "STOP_LOSS"
                elif current_price >= self.tp_price and self.tp_price > 0:
                    hit_tp = True
                    reason = "TAKE_PROFIT"
                else:
                    # Enhanced trailing stop: protect 60% of unrealized gains
                    if self.highest_price > self.position_price:
                        gain = self.highest_price - self.position_price
                        gain_pct = gain / self.position_price
                        
                        # Only start locking in profits if gain is over 1.5% from entry
                        if gain_pct > 0.015:
                            new_trailing_sl = self.position_price + gain * 0.6  # Lock in 60% of peak gain
                            if new_trailing_sl > self.sl_price:
                                old_sl = self.sl_price
                                self.sl_price = new_trailing_sl
                                if abs(new_trailing_sl - old_sl) > 1:  # Only log meaningful moves
                                    logger.info(f"📈 TRAIL UP for {self.symbol}: SL ${old_sl:.2f} → ${self.sl_price:.2f} (60% of ${gain:.2f} gain)")
                    
                    # Check if trailing stop hit
                    if current_price <= self.sl_price and self.sl_price > self.position_price * 0.99:
                        hit_trailing = True
                        reason = "TRAILING_STOP"

                if hit_sl or hit_tp or hit_trailing:
                    trade = self.execute_trade(2, current_price) # Sell to close
                    if hit_sl:
                        self.last_loss_time = time.time()
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
                self.lowest_price = min(self.lowest_price, current_price) if self.lowest_price > 0 else current_price
                
                if current_price >= self.sl_price and self.sl_price > 0:
                    hit_sl = True
                    reason = "STOP_LOSS"
                elif current_price <= self.tp_price and self.tp_price > 0:
                    hit_tp = True
                    reason = "TAKE_PROFIT"
                else:
                    # Enhanced trailing stop for SHORT: protect 60% of unrealized gains
                    if self.lowest_price < self.position_price and self.lowest_price > 0:
                        gain = self.position_price - self.lowest_price
                        gain_pct = gain / self.position_price
                        
                        # Only start locking in profits if gain is over 1.5% from entry
                        if gain_pct > 0.015:
                            new_trailing_sl = self.position_price - gain * 0.6  # Lock in 60% of peak gain
                            if new_trailing_sl < self.sl_price:
                                old_sl = self.sl_price
                                self.sl_price = new_trailing_sl
                                if abs(new_trailing_sl - old_sl) > 0.001:
                                    logger.info(f"📉 TRAIL DOWN for {self.symbol}: SL ${old_sl:.2f} → ${self.sl_price:.2f} (60% of ${gain:.2f} gain)")
                    
                    # Check if trailing stop hit
                    if current_price >= self.sl_price and self.sl_price < self.position_price * 1.01:
                        hit_trailing = True
                        reason = "TRAILING_STOP"
                    
                if hit_sl or hit_tp or hit_trailing:
                    trade = self.execute_trade(1, current_price) # Buy to close
                    if hit_sl:
                        self.last_loss_time = time.time()
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

            # (Removed duplicate SHORT exit block — already handled above at line 566)
        
        # Get model action
        raw_action = self.get_action(df)
        
        # Refresh TFT forecast periodically (every 30 min, not every iteration)
        if self.tft_forecaster:
            import time as _time
            if not hasattr(self, '_last_tft_time') or _time.time() - self._last_tft_time > 1800:
                try:
                    self.last_forecast = self.tft_forecaster.forecast(df)
                    self._last_tft_time = _time.time()
                except Exception as e:
                    logger.warning(f"TFT forecast refresh failed: {e}")
        
        # Compute composite market confidence score
        filtered_action, confidence, reason = self.compute_market_score(raw_action, df)
        self.last_confidence = confidence  # Feeds into ConfidenceEngine for position sizing
        
        # 3-tier decision: signal override / consensus / hold
        final_action, decision_reason = self.make_decision(
            composite_score=getattr(self, '_last_composite_score', 0),
            confidence=confidence,
            raw_action=raw_action,
            df=df
        )
        
        # Use the decision layer's action instead of pure filter output
        if filtered_action != 0 and final_action == 0:
            # Decision layer overrode filter to HOLD
            filtered_action = 0
            reason = decision_reason
        elif filtered_action == 0 and final_action != 0:
            # Decision layer generated a signal override
            filtered_action = final_action
            reason = decision_reason
        
        # --- Anti-overtrading guards ---
        now = time.time()
        
        # Fix 2: Post-loss cooldown — block new entries for 30 min after SL hit
        if filtered_action != 0 and self.position == 0 and self.last_loss_time > 0:
            elapsed_since_loss = now - self.last_loss_time
            if elapsed_since_loss < self.COOLDOWN_SECONDS:
                remaining = int((self.COOLDOWN_SECONDS - elapsed_since_loss) / 60)
                logger.info(f"⏸️ COOLDOWN: Blocking entry for {self.symbol} ({remaining}min remaining after SL hit)")
                filtered_action = 0
                reason = f"Cooldown after SL ({remaining}min left)"
        
        # Fix 4: Minimum hold time — don't exit via model signal within 2 hours
        if filtered_action != 0 and self.position != 0 and self.last_entry_time > 0:
            elapsed_since_entry = now - self.last_entry_time
            if elapsed_since_entry < self.MIN_HOLD_SECONDS:
                remaining = int((self.MIN_HOLD_SECONDS - elapsed_since_entry) / 60)
                logger.info(f"⏳ HOLD: Blocking model exit for {self.symbol} ({remaining}min until min hold expires)")
                filtered_action = 0
                reason = f"Min hold time ({remaining}min left)"
        
        # Execute if action changed
        trade = None
        if filtered_action != 0:
            trade = self.execute_trade(filtered_action, current_price)
            # Track entry time for new positions
            if trade and self.position != 0:
                self.last_entry_time = now
        
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
        
        # Initialize Cross-Asset Correlation Manager
        self.portfolio_manager = GlobalPortfolioManager()
        
        # Create bots
        self.bots: Dict[str, MultiAssetTradingBot] = {}
        for symbol in symbols:
            self.bots[symbol] = MultiAssetTradingBot(
                symbol=symbol,
                dry_run=dry_run,
                initial_balance=balance_per_asset,
                portfolio_manager=self.portfolio_manager,
            )
            # Force GC after each heavy model load
            import gc
            gc.collect()
        
        self.storage = get_storage()
        self.load_state()
        
        self._running = False
        
        # Initialize On-Chain Whale Watcher
        self.whale_watcher = OnChainWhaleWatcher()
        self.last_trade_time = 0
        self.last_whale_check = 0
        
        # Auto-reset if RESET_FLAG file exists (one-time reset trigger)
        reset_flag = Path('logs/RESET_FLAG')
        if reset_flag.exists():
            logger.warning("🔄 RESET_FLAG detected — archiving old trades and resetting portfolio")
            self.reset_portfolio()
            reset_flag.unlink()  # Remove flag after reset
            logger.info("✅ Portfolio reset complete, RESET_FLAG removed")
        
        logger.info(f"🚀 Orchestrator initialized for {len(symbols)} assets")
    
    def reset_portfolio(self):
        """Archive old trades and reset all bot states to initial values."""
        import shutil
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.reset_timestamp = datetime.now().isoformat()
        
        # 1. Archive old trade log
        trade_file = Path('logs/trading_log.json')
        if trade_file.exists():
            archive_path = Path(f'logs/trading_log_archive_{timestamp}.json')
            shutil.copy2(trade_file, archive_path)
            logger.info(f"📦 Archived old trades to {archive_path}")
            
            # Clear the trade log
            trade_file.write_text('')
            logger.info("🗑️ Cleared trading_log.json")
        
        # 1b. Clear MongoDB trades if using mongo storage
        try:
            from src.data.storage import MongoStorage
            if isinstance(self.storage, MongoStorage):
                self.storage.trades_collection.delete_many({})
                logger.info("🗑️ Cleared MongoDB trades collection")
        except Exception as e:
            logger.warning(f"MongoDB trades clear skipped: {e}")
        
        # 2. Archive old state
        state_file = Path('logs/multi_asset_state.json')
        if state_file.exists():
            archive_state = Path(f'logs/state_archive_{timestamp}.json')
            shutil.copy2(state_file, archive_state)
            logger.info(f"📦 Archived old state to {archive_state}")
        
        # 3. Reset all bots to fresh state
        for symbol, bot in self.bots.items():
            bot.balance = bot.initial_balance
            bot.position = 0
            bot.position_price = 0.0
            bot.current_price = 0.0
            bot.position_units = 0.0
            bot.sl_price = 0.0
            bot.tp_price = 0.0
            bot.highest_price = 0.0
            bot.lowest_price = 0.0
            bot.realized_pnl = 0.0
            bot.trades = []
            bot.last_equity = bot.initial_balance
            bot.last_loss_time = 0
            bot.last_entry_time = 0
            logger.info(f"🔄 Reset {symbol}: balance=${bot.initial_balance}, position=FLAT")
        
        # 4. Save the clean state
        self.save_state()
        logger.info("✅ Clean state saved")
    
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
                
                # Check On-Chain Whales (every 1 hour to save proxy bandwidth)
                if current_time - self.last_whale_check > 3600:
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
        state['reset_timestamp'] = getattr(self, 'reset_timestamp', None)
        
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
