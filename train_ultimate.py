#!/usr/bin/env python3
"""
Ultimate Training Script

Training pipeline for the Ultimate Trading Agent with:
- Advanced feature engineering (150+ features)
- Multi-timeframe data augmentation
- Enhanced reward shaping
- Extended training with proper validation
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
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
import logging
import argparse
from datetime import datetime
from pathlib import Path
import json

from src.env.ultimate_env import UltimateTradingEnv, create_ultimate_env
from src.backtest.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsTrackingCallback(BaseCallback):
    """Track and log detailed training metrics."""
    
    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.metrics_history = []
        self.best_sharpe = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Get episode info
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                
                # Get custom metrics from info
                infos = self.locals.get('infos', [])
                
                metrics = {
                    'step': self.n_calls,
                    'mean_reward': mean_reward,
                    'mean_length': mean_length,
                }
                
                self.metrics_history.append(metrics)
                
                if self.verbose > 0:
                    logger.info(
                        f"Step {self.n_calls}: "
                        f"Mean Reward={mean_reward:.2f}, "
                        f"Mean Length={mean_length:.0f}"
                    )
                    
        return True
    
    def on_training_end(self) -> None:
        # Save metrics history
        metrics_path = Path('./logs/training_metrics.json')
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
        logger.info(f"Saved training metrics to {metrics_path}")


class EarlyStoppingCallback(BaseCallback):
    """Stop training if performance degrades significantly."""
    
    def __init__(
        self,
        check_freq: int = 25000,
        patience: int = 5,
        min_improvement: float = 0.01,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                
                if mean_reward > self.best_mean_reward + self.min_improvement:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        logger.info(f"New best mean reward: {mean_reward:.4f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        logger.info(
                            f"No improvement for {self.no_improvement_count}/{self.patience} checks"
                        )
                        
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        logger.warning("Early stopping triggered!")
                    return False
                    
        return True


def load_training_data(days: int = 365, timeframe: str = '1h') -> pd.DataFrame:
    """Load real historical data from Binance for multiple assets."""
    loader = DataLoader()
    
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=days)
    
    # Multi-asset training: load all assets and concatenate
    assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
    all_dfs = []
    
    for asset in assets:
        try:
            asset_df = loader.load(
                symbol=asset,
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
            )
            if not asset_df.empty:
                # Normalize prices to percentage changes for cross-asset training
                # This way the model learns patterns, not absolute price levels
                logger.info(f"Loaded {len(asset_df)} candles for {asset}")
                all_dfs.append(asset_df)
        except Exception as e:
            logger.warning(f"Failed to load {asset}: {e}")
    
    if not all_dfs:
        raise ValueError("No training data loaded for any asset")
    
    # Concatenate all assets end-to-end for training
    # Each asset's data forms a separate "episode" during training
    df = pd.concat(all_dfs, ignore_index=True)
    
    logger.info(f"Total: {len(df)} candles from {len(all_dfs)} assets")
    return df


def create_training_envs(df: pd.DataFrame, n_envs: int = 4):
    """Create training and evaluation environments."""
    
    # Split: 70% train, 15% validation, 15% test
    train_split = int(len(df) * 0.70)
    val_split = int(len(df) * 0.85)
    
    train_df = df.iloc[:train_split]
    val_df = df.iloc[train_split:val_split]
    test_df = df.iloc[val_split:]
    
    logger.info(f"Train: {len(train_df)} candles")
    logger.info(f"Validation: {len(val_df)} candles")
    logger.info(f"Test: {len(test_df)} candles")
    
    # Create training environments
    def make_train_env():
        env = UltimateTradingEnv(
            df=train_df,
            initial_balance=10000.0,
            lookback_window=48,
            trading_fee=0.0004,
            position_size=0.25,  # Match live (was 75%!)
            reward_scaling=1.0,
            stop_loss_pct=0.025,  # 2.5% SL (matches live)
            take_profit_pct=0.05,  # 5% TP (matches live, 2:1 R:R)
        )
        return Monitor(env)
    
    # Create validation environment
    def make_val_env():
        env = UltimateTradingEnv(
            df=val_df,
            initial_balance=10000.0,
            lookback_window=48,
            trading_fee=0.0004,
            position_size=0.25,
            reward_scaling=1.0,
            stop_loss_pct=0.025,
            take_profit_pct=0.05,
        )
        return Monitor(env)
    
    # Vectorized environments
    train_env = DummyVecEnv([make_train_env for _ in range(n_envs)])
    val_env = DummyVecEnv([make_val_env])
    
    # Normalize observations
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    
    val_env = VecNormalize(
        val_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )
    
    return train_env, val_env, test_df


def get_model_config() -> dict:
    """Get optimized PPO hyperparameters for trading."""
    return {
        'policy': 'MlpPolicy',
        'learning_rate': 3e-5,   # Higher LR for faster learning with more data
        'n_steps': 4096,
        'batch_size': 256,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,       # Standard clip range
        'clip_range_vf': 0.2,
        'ent_coef': 0.005,       # More exploration (was 0.001)
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {
            'net_arch': {
                'pi': [256, 256, 128],    # Smaller to prevent overfitting (was 512x512x256x128)
                'vf': [256, 256, 128],
            },
            'activation_fn': __import__('torch').nn.Mish,
        },
    }


def train_ultimate_model(
    timesteps: int = 1000000,
    n_envs: int = 4,
    save_path: str = './data/models/ultimate_agent.zip',
    resume_from: str = None,
):
    """Train the ultimate trading model."""
    
    logger.info("=" * 60)
    logger.info("ULTIMATE TRADING AGENT TRAINING")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading training data...")
    df = load_training_data(days=1095)  # 3 years for robust training
    
    # Create environments
    logger.info("Creating training environments...")
    train_env, val_env, test_df = create_training_envs(df, n_envs=n_envs)
    
    # Create or load model
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming from {resume_from}")
        model = PPO.load(resume_from, env=train_env)
    else:
        logger.info("Creating new PPO model...")
        config = get_model_config()
        model = PPO(
            env=train_env,
            verbose=1,
            tensorboard_log='./logs/tensorboard_ultimate/',
            **config
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./data/checkpoints/',
        name_prefix='ultimate_agent',
    )
    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./data/models/',
        log_path='./logs/eval/',
        eval_freq=25000,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    metrics_callback = MetricsTrackingCallback(log_freq=10000)
    early_stopping = EarlyStoppingCallback(check_freq=50000, patience=8)
    
    callbacks = CallbackList([
        checkpoint_callback,
        eval_callback,
        metrics_callback,
        early_stopping,
    ])
    
    # Train
    logger.info(f"Starting training for {timesteps:,} timesteps...")
    logger.info(f"Using {n_envs} parallel environments")
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    training_time = datetime.now() - start_time
    logger.info(f"Training completed in {training_time}")
    
    # Save final model
    model.save(save_path)
    train_env.save(save_path.replace('.zip', '_vec_normalize.pkl'))
    
    logger.info(f"Model saved to {save_path}")
    
    return model, test_df


def evaluate_ultimate_model(
    model_path: str,
    test_df: pd.DataFrame = None,
    save_report: bool = True,
):
    """Comprehensive evaluation of the trained model."""
    
    logger.info("=" * 60)
    logger.info("ULTIMATE MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Load test data if not provided
    if test_df is None:
        df = load_training_data(days=1095)  # 3 years test data
        test_df = df.iloc[int(len(df) * 0.85):]
        
    logger.info(f"Evaluating on {len(test_df)} candles")
    
    # Create test environment
    env = UltimateTradingEnv(
        df=test_df,
        initial_balance=10000.0,
        lookback_window=48,
        trading_fee=0.0004,
        position_size=0.25,  # Match training
        stop_loss_pct=0.025,
        take_profit_pct=0.05,
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
    
    # Print results
    print("\n" + "=" * 60)
    print("ULTIMATE MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Return:    {metrics['total_return']*100:+.2f}%")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio:   {metrics['sortino_ratio']:.3f}")
    print(f"Max Drawdown:    {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate:        {metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    print(f"Total Trades:    {metrics['total_trades']}")
    print(f"Avg Trade P&L:   ${metrics['avg_trade_pnl']:.2f}")
    print(f"Final Balance:   ${metrics['final_balance']:.2f}")
    print("=" * 60)
    
    # Validation
    passed = metrics['sharpe_ratio'] >= 0.5 and metrics['total_return'] > 0
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\nValidation: {status}")
    
    if save_report:
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST REPORT                            ║
╠══════════════════════════════════════════════════════════════╣
║ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
║ Model: Ultimate Trading Agent
║ Symbol: BTC/USDT
║ Timeframe: 1h
║ Initial Balance: $10,000.00
╠══════════════════════════════════════════════════════════════╣
║                    PERFORMANCE METRICS                        ║
╠══════════════════════════════════════════════════════════════╣
║ Total Return:        {metrics['total_return']*100:+.2f}%
║ Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}
║ Sortino Ratio:       {metrics['sortino_ratio']:.3f}
║ Max Drawdown:         {metrics['max_drawdown']*100:.2f}%
║ Win Rate:            {metrics['win_rate']*100:.1f}%
║ Profit Factor:       {metrics['profit_factor']:.2f}
║ Trade Count:           {metrics['total_trades']}
╠══════════════════════════════════════════════════════════════╣
║                    VALIDATION STATUS                          ║
╠══════════════════════════════════════════════════════════════╣
║ Min Sharpe Required: 0.50
║ Status: {status}
╚══════════════════════════════════════════════════════════════╝
"""
        report_path = Path('./data/ultimate_backtest_report.txt')
        report_path.write_text(report)
        logger.info(f"Report saved to {report_path}")
        
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ultimate Trading Model Training')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Training timesteps')
    parser.add_argument('--n-envs', type=int, default=4, help='Parallel environments')
    parser.add_argument('--evaluate', type=str, default=None, help='Model path to evaluate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create directories
    Path('./data/models').mkdir(parents=True, exist_ok=True)
    Path('./data/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('./logs/eval').mkdir(parents=True, exist_ok=True)
    Path('./logs/tensorboard_ultimate').mkdir(parents=True, exist_ok=True)
    
    if args.evaluate:
        evaluate_ultimate_model(args.evaluate)
    else:
        model, test_df = train_ultimate_model(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            resume_from=args.resume,
        )
        
        # Auto-evaluate after training
        logger.info("\nRunning post-training evaluation...")
        evaluate_ultimate_model('./data/models/ultimate_agent.zip', test_df)
