#!/usr/bin/env python3
"""
Ultimate Model V3 - Post-Quick Wins Retraining

Incorporates findings from Phase 1 Quick Wins analysis:
1. EXCLUDE XRP from training (75% loss rate)
2. Enhanced regime-adaptive reward shaping
3. Longer episodes for time-based pattern learning
4. Focus on BTC, ETH, SOL only

Changes from V2:
- Remove XRP from training data (unprofitable)
- Regime-aware reward function (2.0x bonus for HIGH_VOL wins)
- Extended episodes (500 steps → 1000 steps) for time-based learning
- Asymmetric SL penalties (lighter for regime-appropriate stops)

Expected Outcome:
- V2: Sharpe 0.82, Win Rate 49.8%, P&L +$496
- V3 Target: Sharpe >1.5, Win Rate >55%, P&L >$1,500
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import torch.nn as nn

from src.env.ultimate_env import UltimateTradingEnv
from src.backtest.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickWinsEnv(UltimateTradingEnv):
    """
    Enhanced environment incorporating Quick Wins insights:
    - Regime-aware reward shaping
    - Longer episodes for time-based learning
    - SL penalty adjustment based on regime
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        # Override max_steps for longer episodes
        kwargs['max_steps'] = 1000  # Was 500, now 1000 for time-based patterns
        super().__init__(df, **kwargs)
        self.regime_detector = None

    def _calculate_reward(self) -> float:
        """
        Enhanced reward function with regime awareness.

        Quick Wins insights:
        - HIGH_VOL regime: SL less punishing (2.0x SL is normal)
        - RANGING regime: SL less punishing (1.8x SL is normal)
        - Time-based: Reward holding positions longer
        """
        reward = 0.0

        # Base reward from parent class
        if self.position != 0:
            current_price = self.current_price()

            if self.position == 1:  # LONG
                unrealized_pnl = (current_price - self.position_entry_price) / self.position_entry_price
            else:  # SHORT
                unrealized_pnl = (self.position_entry_price - current_price) / self.position_entry_price

            reward = unrealized_pnl * 100  # Scale to reasonable range

        # CLOSED POSITION REWARDS
        if hasattr(self, 'last_trade_pnl') and self.last_trade_pnl is not None:
            pnl_pct = self.last_trade_pnl / self.initial_balance

            # Base reward from trade
            reward = pnl_pct * 1000  # Scale for learning

            # REGIME-AWARE BONUSES (Quick Wins insight)
            if hasattr(self, 'regime') and self.regime:
                if pnl_pct > 0:  # Winning trade
                    if self.regime == 'high_volatility':
                        # 2x bonus for profitable trades in high vol (harder to achieve)
                        reward *= 2.0
                        logger.debug(f"HIGH_VOL win bonus: {pnl_pct:.2%} → reward={reward:.2f}")
                    elif self.regime == 'ranging':
                        # 1.5x bonus for ranging wins (harder to time)
                        reward *= 1.5
                        logger.debug(f"RANGING win bonus: {pnl_pct:.2%} → reward={reward:.2f}")
                else:  # Losing trade
                    # REDUCED PENALTY for regime-appropriate stops
                    if self.regime == 'high_volatility':
                        # HIGH_VOL: 50% reduced penalty (expected wider SL)
                        reward *= 0.5
                        logger.debug(f"HIGH_VOL loss penalty reduced: {pnl_pct:.2%}")
                    elif self.regime == 'ranging':
                        # RANGING: 30% reduced penalty (chop is expected)
                        reward *= 0.7
                        logger.debug(f"RANGING loss penalty reduced: {pnl_pct:.2%}")

            # TIME-BASED BONUS (Quick Wins insight: 64% recover after 12h)
            if hasattr(self, 'position_hold_time') and self.position_hold_time >= 12:
                # Bonus for holding >12 hours (resilient to volatility)
                time_bonus = min(50, self.position_hold_time * 2)  # Up to +50 for 25h holds
                reward += time_bonus
                logger.debug(f"Time-based bonus: {self.position_hold_time}h → +{time_bonus}")

            # DRAWDOWN PENALTY
            if hasattr(self, 'max_drawdown'):
                if self.max_drawdown > 0.20:  # >20% drawdown
                    reward -= 100  # Harsh penalty
                elif self.max_drawdown > 0.15:
                    reward -= 50

        # HOLDING BONUS (encourage staying in winning positions)
        if self.position != 0:
            hold_steps = self.current_step - self.position_entry_step
            if hold_steps >= 12 and unrealized_pnl > 0:  # 12 steps ~ 12 hours
                reward += 10  # Small bonus for patience

        return reward

    def reset(self, seed=None, **kwargs):
        """Track regime on reset."""
        obs, info = super().reset(seed=seed, **kwargs)

        # Detect current regime
        try:
            from src.features.regime_detector import MarketRegimeDetector
            detector = MarketRegimeDetector()
            regime_info = detector.detect_regime(self.df.iloc[:self.current_step + 100])
            self.regime = regime_info.regime.value
        except:
            self.regime = 'unknown'

        return obs, info


class ValidationCallback(BaseCallback):
    """Validate on unseen data and track Sharpe."""

    def __init__(
        self,
        val_env,
        check_freq: int = 50000,
        patience: int = 5,
        min_sharpe: float = 1.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.val_env = val_env
        self.check_freq = check_freq
        self.patience = patience
        self.min_sharpe = min_sharpe
        self.best_sharpe = -np.inf
        self.wait = 0
        self.sharpe_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Run validation episode
            obs = self.val_env.reset()
            done = False
            trades = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.val_env.step(action)
                if info.get('trade'):
                    trades.append(info['trade'])
                if truncated:
                    break

            # Calculate validation Sharpe
            if len(trades) > 5:
                returns = [t['pnl_pct'] for t in trades if 'pnl_pct' in t]
                if len(returns) > 0:
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
                    win_rate = sum(1 for r in returns if r > 0) / len(returns)

                    self.sharpe_history.append({
                        'step': self.n_calls,
                        'sharpe': sharpe,
                        'win_rate': win_rate,
                        'trades': len(trades)
                    })

                    logger.info(
                        f"\n{'='*60}\n"
                        f"VALIDATION @ Step {self.n_calls}\n"
                        f"  Sharpe:   {sharpe:.3f}\n"
                        f"  Win Rate: {win_rate:.1%}\n"
                        f"  Trades:   {len(trades)}\n"
                        f"{'='*60}"
                    )

                    # Early stopping check
                    if sharpe > self.best_sharpe:
                        self.best_sharpe = sharpe
                        self.wait = 0

                        # Save best model
                        self.model.save('./data/models/ultimate_v3_best.zip')
                        logger.info(f"💾 New best model saved (Sharpe={sharpe:.3f})")
                    else:
                        self.wait += 1
                        if self.wait >= self.patience and self.best_sharpe < self.min_sharpe:
                            logger.warning(
                                f"Early stopping: No improvement for {self.patience} checks. "
                                f"Best Sharpe={self.best_sharpe:.3f} < {self.min_sharpe}"
                            )
                            return False

        return True


def load_multi_asset_data(symbols, days=180):
    """Load and concatenate data from multiple assets (EXCLUDE XRP)."""
    loader = DataLoader()
    dfs = []

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    for symbol in symbols:
        logger.info(f"Loading {symbol} data...")
        try:
            # DataLoader expects 'BTC/USDT' format, not 'BTCUSDT'
            formatted_symbol = symbol[:3] + '/' + symbol[3:]  # BTCUSDT -> BTC/USDT

            df = loader.load(
                symbol=formatted_symbol,
                timeframe='1h',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                use_cache=True,
            )
            df['symbol'] = symbol
            dfs.append(df)
            logger.info(f"  Loaded {len(df)} candles for {symbol}")
        except Exception as e:
            logger.error(f"  Failed to load {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Concatenate all assets
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"\nTotal training data: {len(combined_df)} candles across {len(dfs)} assets")
        return combined_df
    else:
        raise ValueError("No data loaded!")


def train_v3_model(timesteps=2000000, learning_rate=3e-4):
    """Train Ultimate V3 model with Quick Wins insights."""

    logger.info("="*80)
    logger.info("TRAINING ULTIMATE V3 - POST-QUICK WINS")
    logger.info("="*80)

    # 1. Load data (EXCLUDE XRP!)
    logger.info("\n1. Loading training data (BTC, ETH, SOL only)...")
    train_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']  # NO XRP!
    train_df = load_multi_asset_data(train_symbols, days=180)

    # 2. Create train/val split (80/20)
    split_idx = int(len(train_df) * 0.8)
    df_train = train_df.iloc[:split_idx].reset_index(drop=True)
    df_val = train_df.iloc[split_idx:].reset_index(drop=True)

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(df_train)} candles")
    logger.info(f"  Val:   {len(df_val)} candles")

    # 3. Create environments
    logger.info("\n2. Creating training environments...")

    def make_env(df):
        def _init():
            env = QuickWinsEnv(
                df=df,
                initial_balance=10000,
                trading_fee=0.001,  # 0.1% trading fee
            )
            return Monitor(env)
        return _init

    # Training environment (vectorized)
    train_env = DummyVecEnv([make_env(df_train)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Validation environment (single)
    val_env = QuickWinsEnv(
        df=df_val,
        initial_balance=10000,
        trading_fee=0.001,
    )

    # 4. Create model
    logger.info("\n3. Initializing PPO model...")

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Smaller than V1 (was 256x256x128)
        activation_fn=nn.ReLU,
    )

    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log='./logs/tensorboard/',
    )

    logger.info(f"  Network: 128x128 (2 layers)")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Timesteps: {timesteps:,}")

    # 5. Setup callbacks
    logger.info("\n4. Setting up callbacks...")

    callbacks = []

    # Validation callback
    val_callback = ValidationCallback(
        val_env=val_env,
        check_freq=50000,
        patience=5,
        min_sharpe=1.0,
        verbose=1,
    )
    callbacks.append(val_callback)

    callback_list = CallbackList(callbacks)

    # 6. Train
    logger.info("\n5. Starting training...")
    logger.info("="*80)

    start_time = datetime.now()

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callback_list,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Training interrupted by user!")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("="*80)
    logger.info(f"✅ Training complete!")
    logger.info(f"  Duration: {duration/3600:.1f} hours")
    logger.info(f"  Best Sharpe: {val_callback.best_sharpe:.3f}")

    # 7. Save final model
    logger.info("\n6. Saving models...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Save final model
    model.save(f'./data/models/ultimate_v3.zip')
    train_env.save(f'./data/models/ultimate_v3_vec_normalize.pkl')
    logger.info(f"  💾 Final model: ultimate_v3.zip")

    # Save backup with timestamp
    model.save(f'./data/models/ultimate_v3_{timestamp}.zip')
    logger.info(f"  💾 Backup: ultimate_v3_{timestamp}.zip")

    # Save training report
    report = {
        'timestamp': timestamp,
        'duration_hours': duration / 3600,
        'total_timesteps': timesteps,
        'assets': train_symbols,
        'train_candles': len(df_train),
        'val_candles': len(df_val),
        'best_sharpe': float(val_callback.best_sharpe),
        'sharpe_history': val_callback.sharpe_history,
        'quick_wins_integrated': True,
        'xrp_excluded': True,
        'regime_aware': True,
        'time_based_learning': True,
    }

    with open(f'./data/models/ultimate_v3_training_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"  📊 Training report: ultimate_v3_training_report.json")

    logger.info("\n" + "="*80)
    logger.info("🎉 V3 TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("  1. Review training report")
    logger.info("  2. Backtest ultimate_v3_best.zip")
    logger.info("  3. Compare vs V2 and Quick Wins only")
    logger.info("  4. Deploy to dev Space if Sharpe >1.5")

    return model, train_env


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=2000000, help='Training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    args = parser.parse_args()

    try:
        model, env = train_v3_model(
            timesteps=args.timesteps,
            learning_rate=args.lr,
        )
        logger.info("\n✅ All done! V3 model ready for validation.")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
