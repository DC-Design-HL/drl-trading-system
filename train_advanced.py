#!/usr/bin/env python3
"""
Advanced Training Script
Competitive training with optimized hyperparameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import logging
import argparse
from datetime import datetime
from pathlib import Path

from src.env.advanced_env import AdvancedTradingEnv, create_advanced_env
from src.backtest.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProfitTrackingCallback(BaseCallback):
    """Track profit and early stop if model is performing badly."""
    
    def __init__(self, check_freq: int = 10000, min_profit: float = -0.05, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_profit = min_profit
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get episode info
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        logger.info(f"New best mean reward: {mean_reward:.2f}")
                        
        return True


def load_real_data(days: int = 365) -> pd.DataFrame:
    """Load real historical data from Binance."""
    loader = DataLoader()
    
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=days)
    
    df = loader.load(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
    )
    
    logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def create_envs(df: pd.DataFrame, n_envs: int = 1):
    """Create training and evaluation environments."""
    
    # Split data: 80% train, 20% eval
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    eval_df = df.iloc[split_idx:]
    
    logger.info(f"Train data: {len(train_df)} candles, Eval data: {len(eval_df)} candles")
    
    # Create training environment
    def make_train_env():
        env = AdvancedTradingEnv(
            df=train_df,
            initial_balance=10000.0,
            lookback_window=48,
            trading_fee=0.0004,
            position_size=0.25,
        )
        env = Monitor(env)
        return env
        
    # Create eval environment
    def make_eval_env():
        env = AdvancedTradingEnv(
            df=eval_df,
            initial_balance=10000.0,
            lookback_window=48,
            trading_fee=0.0004,
            position_size=0.25,
        )
        env = Monitor(env)
        return env
        
    train_env = DummyVecEnv([make_train_env for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_eval_env])
    
    # Normalize observations for better training
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    
    return train_env, eval_env


def train_competitive_model(
    timesteps: int = 500000,
    n_envs: int = 4,
    save_path: str = './data/models/advanced_agent.zip',
):
    """Train a competitive trading model."""
    
    # Load data
    logger.info("Loading real market data...")
    df = load_real_data(days=365)
    
    # Create environments
    logger.info("Creating training environments...")
    train_env, eval_env = create_envs(df, n_envs=n_envs)
    
    # Competitive hyperparameters (tuned for trading)
    model_config = {
        'policy': 'MlpPolicy',
        'learning_rate': 5e-5,  # Lower LR for stability
        'n_steps': 2048,
        'batch_size': 256,  # Larger batch for stability
        'n_epochs': 5,  # Fewer epochs to prevent overfitting
        'gamma': 0.995,  # Higher discount for longer-term thinking
        'gae_lambda': 0.98,
        'clip_range': 0.1,  # Smaller clip for stability
        'clip_range_vf': 0.1,
        'ent_coef': 0.005,  # Low entropy for less random trading
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {
            'net_arch': dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # Deeper networks
        },
    }
    
    logger.info("Creating PPO model...")
    model = PPO(
        env=train_env,
        verbose=1,
        tensorboard_log='./logs/tensorboard_advanced/',
        **model_config
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./data/checkpoints/',
        name_prefix='advanced_agent',
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./data/models/',
        log_path='./logs/eval/',
        eval_freq=25000,
        n_eval_episodes=3,
        deterministic=True,
    )
    
    profit_callback = ProfitTrackingCallback(check_freq=10000)
    
    callbacks = [checkpoint_callback, eval_callback, profit_callback]
    
    # Train!
    logger.info(f"Starting training for {timesteps} timesteps...")
    logger.info(f"Using {n_envs} parallel environments")
    
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    model.save(save_path)
    train_env.save(save_path.replace('.zip', '_vec_normalize.pkl'))
    
    logger.info(f"Model saved to {save_path}")
    
    return model, train_env


def evaluate_model(model_path: str):
    """Evaluate trained model on full dataset."""
    
    logger.info(f"Evaluating model: {model_path}")
    
    # Load data
    df = load_real_data(days=365)
    
    # Create eval environment
    env = AdvancedTradingEnv(
        df=df,
        initial_balance=10000.0,
        lookback_window=48,
    )
    
    # Load model
    model = PPO.load(model_path)
    
    # Run evaluation
    obs, info = env.reset(options={'random_start': False})
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    # Get metrics
    metrics = env.get_episode_metrics()
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total Return:    {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio:   {metrics['sortino_ratio']:.3f}")
    print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    print(f"Win Rate:        {metrics['win_rate']*100:.1f}%")
    print(f"Max Drawdown:    {metrics['max_drawdown']*100:.2f}%")
    print(f"Total Trades:    {metrics['total_trades']}")
    print(f"Avg Trade P&L:   ${metrics['avg_trade_pnl']:.2f}")
    print(f"Avg Hold Time:   {metrics['avg_hold_time']:.1f} hours")
    print("="*60)
    
    # Validation
    passed = metrics['sharpe_ratio'] >= 0.5
    print(f"\nValidation (Sharpe >= 0.5): {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Trading Model Training')
    parser.add_argument('--timesteps', type=int, default=500000, help='Training timesteps')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--evaluate', type=str, default=None, help='Path to model to evaluate')
    
    args = parser.parse_args()
    
    # Create directories
    Path('./data/models').mkdir(parents=True, exist_ok=True)
    Path('./data/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('./logs/eval').mkdir(parents=True, exist_ok=True)
    
    if args.evaluate:
        evaluate_model(args.evaluate)
    else:
        train_competitive_model(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
        )
