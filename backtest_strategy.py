#!/usr/bin/env python3
"""
Strategy Backtester

Replays the EXACT live trading pipeline on historical data to validate
model performance before deployment.

Usage:
    python backtest_strategy.py                     # Backtest all assets
    python backtest_strategy.py --asset BTCUSDT     # Single asset
    python backtest_strategy.py --days 180          # 6 months
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.features.ultimate_features import UltimateFeatureEngine
from src.features.correlation_engine import SimulatedDominanceEngine
from src.features.regime_detector import MarketRegimeDetector
from src.features.risk_manager import AdaptiveRiskManager
from src.backtest.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Replays the exact live trading pipeline on historical data.
    
    Mirrors:
    - Feature computation (UltimateFeatureEngine + SimulatedDominanceEngine)
    - VecNormalize observation scaling
    - PPO model inference
    - Regime-adaptive SL/TP
    - 60% trailing stops
    - Min hold time + cooldown
    """
    
    def __init__(
        self,
        model_path: str = './data/models/ultimate_agent.zip',
        vec_norm_path: str = './data/models/ultimate_agent_vec_normalize.pkl',
        initial_balance: float = 5000.0,
        position_size: float = 0.25,
        min_hold_bars: int = 4,      # 4 hours at 1h timeframe
        cooldown_bars: int = 2,      # 2 hours cooldown after SL
    ):
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.min_hold_bars = min_hold_bars
        self.cooldown_bars = cooldown_bars
        
        # Load model
        self.model = None
        if Path(model_path).exists():
            try:
                self.model = PPO.load(model_path)
                logger.info(f"✅ Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        
        # Load VecNormalize
        self.vec_normalize = None
        if Path(vec_norm_path).exists():
            try:
                import gymnasium as gym
                dummy_env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
                self.vec_normalize = VecNormalize.load(vec_norm_path, dummy_env)
                self.vec_normalize.training = False
                self.vec_normalize.norm_reward = False
                logger.info(f"✅ VecNormalize loaded from {vec_norm_path}")
            except Exception as e:
                logger.warning(f"⚠️ VecNormalize load failed: {e}")
        
        # Feature engines
        self.feature_engine = UltimateFeatureEngine()
        self.dominance_engine = SimulatedDominanceEngine()
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = AdaptiveRiskManager()
    
    def _compute_observation(
        self, df: pd.DataFrame, position: int, 
        position_price: float, balance: float
    ) -> Optional[np.ndarray]:
        """Compute observation vector matching live pipeline."""
        if len(df) < 200:
            return None
        
        try:
            # Features (same as live)
            all_features = self.feature_engine.get_all_features(df)
            dominance_features = self.dominance_engine.compute_simulated_dominance(df)
            all_features.update(dominance_features)
            
            features_df = pd.DataFrame(all_features)
            features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
            
            for col in features_df.columns:
                if features_df[col].dtype in [np.float64, np.float32]:
                    features_df[col] = features_df[col].clip(-10, 10)
            
            features = features_df.values.astype(np.float32)
            if len(features) < 1:
                return None
            
            last_features = features[-1].copy()
            
            # Position info
            current_price = df.iloc[-1]['close']
            if position != 0 and position_price > 0:
                if position == 1:
                    unrealized_pnl = (current_price - position_price) / position_price
                else:
                    unrealized_pnl = (position_price - current_price) / position_price
            else:
                unrealized_pnl = 0.0
            
            balance_ratio = (balance - self.initial_balance) / self.initial_balance
            
            position_info = np.array([
                float(position),
                np.clip(unrealized_pnl, -0.5, 0.5),
                np.clip(balance_ratio, -0.5, 0.5),
            ], dtype=np.float32)
            
            observation = np.concatenate([last_features, position_info]).astype(np.float32)
            
            # Apply VecNormalize
            if self.vec_normalize is not None:
                try:
                    obs_2d = observation.reshape(1, -1)
                    observation = self.vec_normalize.normalize_obs(obs_2d).flatten().astype(np.float32)
                except:
                    pass
            
            return observation
        except Exception as e:
            logger.warning(f"Feature computation failed: {e}")
            return None
    
    def _get_regime_adjustments(self, df: pd.DataFrame, direction: str) -> Tuple[float, float]:
        """Get regime-adaptive SL/TP multipliers."""
        sl_mult, tp_mult = 1.0, 1.0
        try:
            regime_info = self.regime_detector.detect_regime(df)
            regime_name = regime_info.regime.value
            if regime_name == 'high_volatility':
                sl_mult, tp_mult = 1.5, 1.5
            elif regime_name in ('trending_up', 'trending_down'):
                tp_mult = 1.5
            elif regime_name == 'ranging':
                sl_mult, tp_mult = 0.8, 0.8
        except:
            pass
        return sl_mult, tp_mult
    
    def run(self, symbol: str = 'BTC/USDT', days: int = 90) -> Dict:
        """
        Run backtest on historical data.
        
        Returns detailed performance report.
        """
        logger.info(f"{'='*60}")
        logger.info(f"BACKTESTING {symbol} — last {days} days")
        logger.info(f"{'='*60}")
        
        # Load data
        loader = DataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = loader.load(
            symbol=symbol,
            timeframe='1h',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
        )
        
        if df.empty or len(df) < 250:
            logger.error(f"Insufficient data for {symbol}: {len(df)} candles")
            return {"error": f"Insufficient data: {len(df)} candles"}
        
        logger.info(f"Loaded {len(df)} candles ({df.index[0]} to {df.index[-1]})")
        
        # Trading state
        balance = self.initial_balance
        position = 0          # 0=flat, 1=long, -1=short
        position_price = 0.0
        position_units = 0.0
        sl_price = 0.0
        tp_price = 0.0
        highest_price = 0.0
        lowest_price = 0.0
        entry_bar = 0
        last_loss_bar = -100
        
        # Tracking
        trades = []
        equity_curve = []
        max_equity = balance
        
        # Lookback window for features (need 200+ bars)
        start_idx = 200
        
        for i in range(start_idx, len(df)):
            # Current data slice (everything up to current bar)
            current_df = df.iloc[:i+1].copy()
            current_price = float(current_df.iloc[-1]['close'])
            bars_in_position = i - entry_bar if position != 0 else 0
            
            # ─── Check SL/TP ───────────────────────────────────────
            if position != 0:
                exit_trade = False
                exit_reason = ""
                exit_price = current_price
                
                if position == 1:  # LONG
                    highest_price = max(highest_price, current_price)
                    
                    # Enhanced trailing stop (60% of gains)
                    if highest_price > position_price:
                        gain = highest_price - position_price
                        new_sl = position_price + gain * 0.6
                        if new_sl > sl_price:
                            sl_price = new_sl
                    
                    if current_price <= sl_price and sl_price > 0:
                        exit_trade = True
                        exit_reason = "STOP_LOSS" if sl_price <= position_price else "TRAILING_STOP"
                    elif current_price >= tp_price and tp_price > 0:
                        exit_trade = True
                        exit_reason = "TAKE_PROFIT"
                        
                elif position == -1:  # SHORT
                    lowest_price = min(lowest_price, current_price) if lowest_price > 0 else current_price
                    
                    # Enhanced trailing stop (60% of gains)
                    if lowest_price < position_price and lowest_price > 0:
                        gain = position_price - lowest_price
                        new_sl = position_price - gain * 0.6
                        if new_sl < sl_price:
                            sl_price = new_sl
                    
                    if current_price >= sl_price and sl_price > 0:
                        exit_trade = True
                        exit_reason = "STOP_LOSS" if sl_price >= position_price else "TRAILING_STOP"
                    elif current_price <= tp_price and tp_price > 0:
                        exit_trade = True
                        exit_reason = "TAKE_PROFIT"
                
                if exit_trade:
                    # Calculate P&L
                    if position == 1:
                        pnl = (current_price - position_price) * position_units
                    else:
                        pnl = (position_price - current_price) * position_units
                    
                    balance += position_price * position_units + pnl
                    
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': position_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'reason': exit_reason,
                        'bars_held': bars_in_position,
                        'timestamp': str(current_df.index[-1]) if hasattr(current_df.index[-1], 'strftime') else str(i),
                    })
                    
                    if "STOP_LOSS" in exit_reason:
                        last_loss_bar = i
                    
                    position = 0
                    position_units = 0
                    position_price = 0
                    sl_price = 0
                    tp_price = 0
                    highest_price = 0
                    lowest_price = 0
                    continue
            
            # ─── Get Model Action ─────────────────────────────────
            if self.model is None:
                equity_curve.append(balance)
                continue
            
            obs = self._compute_observation(current_df, position, position_price, balance)
            if obs is None:
                equity_curve.append(balance)
                continue
            
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action.item() if hasattr(action, 'item') else action)
            except Exception as e:
                action = 0
            
            # ─── Apply Guards ─────────────────────────────────────
            # Cooldown after loss
            if action != 0 and position == 0 and (i - last_loss_bar) < self.cooldown_bars:
                action = 0
            
            # Min hold time
            if action != 0 and position != 0 and bars_in_position < self.min_hold_bars:
                action = 0
            
            # ─── Execute Trade ────────────────────────────────────
            if action == 1 and position <= 0:  # BUY
                # Close short first
                if position == -1:
                    pnl = (position_price - current_price) * position_units
                    balance += position_price * position_units + pnl
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'direction': 'SHORT',
                        'entry_price': position_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'reason': 'SIGNAL_EXIT',
                        'bars_held': bars_in_position,
                        'timestamp': str(current_df.index[-1]) if hasattr(current_df.index[-1], 'strftime') else str(i),
                    })
                    position = 0
                
                # Open long
                if position == 0:
                    trade_value = balance * self.position_size
                    if trade_value > 10:
                        position_units = trade_value / current_price
                        position_price = current_price
                        position = 1
                        balance -= trade_value
                        entry_bar = i
                        highest_price = current_price
                        
                        # Calculate SL/TP (regime-adaptive)
                        try:
                            sl_pct, tp_pct = self.risk_manager.get_adaptive_sl_tp(current_df, "long")
                            sl_mult, tp_mult = self._get_regime_adjustments(current_df, "long")
                            sl_pct *= sl_mult
                            tp_pct *= tp_mult
                        except:
                            sl_pct, tp_pct = 0.025, 0.05
                        
                        sl_price = current_price * (1 - sl_pct)
                        tp_price = current_price * (1 + tp_pct)
            
            elif action == 2 and position >= 0:  # SELL
                # Close long first
                if position == 1:
                    pnl = (current_price - position_price) * position_units
                    balance += position_price * position_units + pnl
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'direction': 'LONG',
                        'entry_price': position_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'reason': 'SIGNAL_EXIT',
                        'bars_held': bars_in_position,
                        'timestamp': str(current_df.index[-1]) if hasattr(current_df.index[-1], 'strftime') else str(i),
                    })
                    position = 0
                
                # Open short
                if position == 0:
                    trade_value = balance * self.position_size
                    if trade_value > 10:
                        position_units = trade_value / current_price
                        position_price = current_price
                        position = -1
                        balance -= trade_value
                        entry_bar = i
                        lowest_price = current_price
                        
                        # Calculate SL/TP (regime-adaptive)
                        try:
                            sl_pct, tp_pct = self.risk_manager.get_adaptive_sl_tp(current_df, "short")
                            sl_mult, tp_mult = self._get_regime_adjustments(current_df, "short")
                            sl_pct *= sl_mult
                            tp_pct *= tp_mult
                        except:
                            sl_pct, tp_pct = 0.025, 0.05
                        
                        sl_price = current_price * (1 + sl_pct)
                        tp_price = current_price * (1 - tp_pct)
            
            # Track equity
            unrealized = 0
            if position == 1:
                unrealized = (current_price - position_price) * position_units
            elif position == -1:
                unrealized = (position_price - current_price) * position_units
            
            equity = balance + (position_price * position_units if position != 0 else 0) + unrealized
            equity_curve.append(equity)
            max_equity = max(max_equity, equity)
        
        # ─── Close any remaining position ─────────────────────
        if position != 0:
            final_price = float(df.iloc[-1]['close'])
            if position == 1:
                pnl = (final_price - position_price) * position_units
            else:
                pnl = (position_price - final_price) * position_units
            balance += position_price * position_units + pnl
            trades.append({
                'entry_bar': entry_bar,
                'exit_bar': len(df) - 1,
                'direction': 'LONG' if position == 1 else 'SHORT',
                'entry_price': position_price,
                'exit_price': final_price,
                'pnl': pnl,
                'reason': 'END_OF_DATA',
                'bars_held': len(df) - 1 - entry_bar,
                'timestamp': str(df.index[-1]) if hasattr(df.index[-1], 'strftime') else 'end',
            })
        
        # ─── Calculate Metrics ────────────────────────────────
        report = self._calculate_metrics(trades, equity_curve, symbol)
        return report
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float], symbol: str) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'status': 'NO_TRADES',
                'message': 'Model generated no trades',
            }
        
        # Basic metrics
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        total_return = (sum(pnls) / self.initial_balance) * 100
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        # Sharpe Ratio (annualized from hourly)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)  # Annualize from hourly
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Max Drawdown
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                peak = max(peak, eq)
                dd = (peak - eq) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0
        
        # Win/Loss by reason
        reason_stats = {}
        for t in trades:
            reason = t['reason']
            if reason not in reason_stats:
                reason_stats[reason] = {'count': 0, 'total_pnl': 0, 'wins': 0}
            reason_stats[reason]['count'] += 1
            reason_stats[reason]['total_pnl'] += t['pnl']
            if t['pnl'] > 0:
                reason_stats[reason]['wins'] += 1
        
        # Direction analysis
        long_trades = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']
        long_pnl = sum(t['pnl'] for t in long_trades)
        short_pnl = sum(t['pnl'] for t in short_trades)
        long_wins = sum(1 for t in long_trades if t['pnl'] > 0)
        short_wins = sum(1 for t in short_trades if t['pnl'] > 0)
        
        # Average hold time
        avg_hold = np.mean([t['bars_held'] for t in trades]) if trades else 0
        
        # Validation
        passed = sharpe > 0.5 and win_rate > 50 and max_dd < 0.10
        
        report = {
            'symbol': symbol,
            'status': 'PASSED ✅' if passed else 'FAILED ❌',
            'total_trades': len(trades),
            'total_return_pct': round(total_return, 2),
            'total_pnl': round(sum(pnls), 2),
            'win_rate_pct': round(win_rate, 1),
            'wins': len(wins),
            'losses': len(losses),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 3),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'avg_hold_bars': round(avg_hold, 1),
            'long_trades': len(long_trades),
            'long_pnl': round(long_pnl, 2),
            'long_win_rate': round(long_wins / len(long_trades) * 100, 1) if long_trades else 0,
            'short_trades': len(short_trades),
            'short_pnl': round(short_pnl, 2),
            'short_win_rate': round(short_wins / len(short_trades) * 100, 1) if short_trades else 0,
            'exit_reasons': reason_stats,
            'trades': trades[-20:],  # Last 20 trades for inspection
        }
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTEST RESULTS: {symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"  Status:         {report['status']}")
        logger.info(f"  Total Trades:   {report['total_trades']}")
        logger.info(f"  Total Return:   {report['total_return_pct']}%")
        logger.info(f"  Total P&L:      ${report['total_pnl']}")
        logger.info(f"  Win Rate:       {report['win_rate_pct']}%")
        logger.info(f"  Avg Win:        ${report['avg_win']}")
        logger.info(f"  Avg Loss:       ${report['avg_loss']}")
        logger.info(f"  Profit Factor:  {report['profit_factor']}")
        logger.info(f"  Sharpe Ratio:   {report['sharpe_ratio']}")
        logger.info(f"  Max Drawdown:   {report['max_drawdown_pct']}%")
        logger.info(f"  Avg Hold Time:  {report['avg_hold_bars']} bars ({report['avg_hold_bars']:.0f}h)")
        logger.info(f"  LONG:  {report['long_trades']} trades, ${report['long_pnl']} P&L, {report['long_win_rate']}% win")
        logger.info(f"  SHORT: {report['short_trades']} trades, ${report['short_pnl']} P&L, {report['short_win_rate']}% win")
        logger.info(f"{'='*60}")
        
        for reason, stats in reason_stats.items():
            wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
            logger.info(f"  {reason}: {stats['count']} exits, ${stats['total_pnl']:.2f} P&L, {wr:.0f}% win")
        
        return report
    
    def run_all_assets(self, days: int = 90) -> Dict:
        """Run backtest on all supported assets."""
        assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
        results = {}
        
        for asset in assets:
            try:
                results[asset] = self.run(asset, days)
            except Exception as e:
                logger.error(f"Backtest failed for {asset}: {e}")
                results[asset] = {'error': str(e), 'status': 'ERROR'}
        
        # Aggregate results
        total_trades = sum(r.get('total_trades', 0) for r in results.values() if 'total_trades' in r)
        total_pnl = sum(r.get('total_pnl', 0) for r in results.values() if 'total_pnl' in r)
        total_wins = sum(r.get('wins', 0) for r in results.values() if 'wins' in r)
        total_losses = sum(r.get('losses', 0) for r in results.values() if 'losses' in r)
        
        agg_win_rate = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) > 0 else 0
        avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in results.values() if 'sharpe_ratio' in r])
        max_dd = max([r.get('max_drawdown_pct', 0) for r in results.values() if 'max_drawdown_pct' in r], default=0)
        
        passed = avg_sharpe > 0.5 and agg_win_rate > 50 and max_dd < 10
        
        logger.info(f"\n{'='*60}")
        logger.info(f"AGGREGATE BACKTEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"  Overall Status: {'PASSED ✅' if passed else 'FAILED ❌'}")
        logger.info(f"  Total Trades:   {total_trades}")
        logger.info(f"  Total P&L:      ${total_pnl:.2f}")
        logger.info(f"  Win Rate:       {agg_win_rate:.1f}%")
        logger.info(f"  Avg Sharpe:     {avg_sharpe:.3f}")
        logger.info(f"  Max Drawdown:   {max_dd:.2f}%")
        logger.info(f"{'='*60}")
        
        results['_aggregate'] = {
            'status': 'PASSED ✅' if passed else 'FAILED ❌',
            'total_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'win_rate_pct': round(agg_win_rate, 1),
            'avg_sharpe': round(avg_sharpe, 3),
            'max_drawdown_pct': round(max_dd, 2),
        }
        
        # Save report
        report_path = Path('./data/backtest_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Make JSON serializable
        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = {
                    sk: sv for sk, sv in v.items() 
                    if not isinstance(sv, (np.floating, np.integer))
                }
            else:
                serializable[k] = v
        
        with open(report_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"\n📄 Report saved to {report_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Strategy Backtester')
    parser.add_argument('--asset', type=str, default=None, help='Single asset to backtest (e.g., BTCUSDT)')
    parser.add_argument('--days', type=int, default=90, help='Days of history to backtest')
    parser.add_argument('--balance', type=float, default=5000.0, help='Initial balance')
    args = parser.parse_args()
    
    backtester = StrategyBacktester(initial_balance=args.balance)
    
    if args.asset:
        symbol = args.asset.replace('USDT', '/USDT')
        report = backtester.run(symbol, args.days)
    else:
        report = backtester.run_all_assets(args.days)
    
    return report


if __name__ == '__main__':
    main()
