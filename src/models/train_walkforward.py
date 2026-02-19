#!/usr/bin/env python3
"""
Advanced Training Pipeline (Phase 11.5)

Implements Walk-Forward Cross Validation to prevent overfitting and
simulate realistic deployment conditions.

Pipeline:
1. Load historical data (2+ years)
2. Split into rolling windows (e.g., 6 months train, 2 months test)
3. Train PPO sequentially on each train window
4. Evaluate on each test window
5. Save models for each window and a final combined model
"""

import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.env.mtf_env import MultiTimeframeTradingEnv, create_mtf_env
from src.backtest.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        train_months: int = 6,
        test_months: int = 2,
        total_days: int = 730,  # 2 years
    ):
        self.symbol = symbol
        self.train_months = train_months
        self.test_months = test_months
        self.total_days = total_days
        
        self.model_dir = Path("./data/models/walkforward")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Hyperparams for rapid adaptation
        self.ppo_kwargs = {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 128,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': {
                'net_arch': {'pi': [256, 256], 'vf': [256, 256]}
            }
        }

    def fetch_data(self) -> pd.DataFrame:
        from src.data.multi_asset_fetcher import MultiAssetDataFetcher
        fetcher = MultiAssetDataFetcher()
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=self.total_days)
        
        logger.info(f"Fetching {self.symbol} data from {start_date.date()} to {end_date.date()}...")
        df = fetcher.fetch_asset(
            symbol=self.symbol,
            interval="1h",
            days=self.total_days,
        )
        return df

    def create_windows(self, df: pd.DataFrame):
        windows = []
        if df.empty:
            return windows
            
        start_ts = df['timestamp'].iloc[0]
        end_ts = df['timestamp'].iloc[-1]
        
        current_start = start_ts
        
        while True:
            # Calculate window boundaries
            train_end = current_start + pd.Timedelta(days=30 * self.train_months)
            test_end = train_end + pd.Timedelta(days=30 * self.test_months)
            
            if test_end > end_ts:
                # We've reached the end of the data
                break
                
            train_df = df[(df['timestamp'] >= current_start) & (df['timestamp'] < train_end)].copy()
            test_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < test_end)].copy()
            
            # Need a minimum amount of data to compute features
            if len(train_df) > 500 and len(test_df) > 100:
                windows.append((train_df, test_df))
                
            # Roll forward by 'test_months'
            current_start = current_start + pd.Timedelta(days=30 * self.test_months)
            
        logger.info(f"Created {len(windows)} walk-forward windows.")
        return windows

    def run(self):
        df = self.fetch_data()
        windows = self.create_windows(df)
        
        if not windows:
            logger.error("No windows created. Not enough data.")
            return

        model = None
        vec_env = None
        
        all_test_metrics = []

        for i, (train_df, test_df) in enumerate(windows):
            logger.info("=" * 60)
            logger.info(f"Walk-Forward Window {i+1}/{len(windows)}")
            logger.info(f"Train: {train_df['timestamp'].iloc[0].date()} to {train_df['timestamp'].iloc[-1].date()} ({len(train_df)} bars)")
            logger.info(f"Test : {test_df['timestamp'].iloc[0].date()} to {test_df['timestamp'].iloc[-1].date()} ({len(test_df)} bars)")
            logger.info("=" * 60)
            
            # Create sub-environments
            train_envs = create_mtf_env(train_df, n_envs=4, randomize_start=True)
            eval_envs = create_mtf_env(test_df, n_envs=1, randomize_start=False)
            
            if model is None:
                logger.info("Initializing base model...")
                model = PPO(env=train_envs, verbose=0, **self.ppo_kwargs)
            else:
                logger.info("Continuing training from previous window...")
                model.set_env(train_envs)
                
            eval_callback = EvalCallback(
                eval_envs,
                best_model_save_path=f"./data/models/walkforward/window_{i}",
                log_path=f"./data/models/walkforward/logs_{i}",
                eval_freq=10000,
                deterministic=True,
                render=False
            )
            
            # Curriculum step size - train for ~100k steps per window
            steps_per_window = min(150000, len(train_df) * 10)
            
            logger.info(f"Training for {steps_per_window} steps...")
            model.learn(total_timesteps=steps_per_window, callback=eval_callback, progress_bar=True)
            
            # Evaluate on Test Data
            logger.info("Evaluating on unseen test window...")
            model.set_env(eval_envs)
            
            obs = eval_envs.reset()
            dones = [False]
            
            while not dones[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_envs.step(action)
                
            # Extract final metrics from info
            info = infos[0] if isinstance(infos, list) else infos
            if hasattr(eval_envs, 'envs'):
                 info = eval_envs.envs[0].get_episode_metrics()
            elif isinstance(info, list):
                 info = info[0].get('episode_metrics', {})
                 
            logger.info(f"Test Window {i+1} Return: {info.get('total_return', 0)*100:.2f}% | Sharpe: {info.get('sharpe_ratio', 0):.2f}")
            all_test_metrics.append(info)
            
            # Save continuous model
            model.save(self.model_dir / "continuous_agent.zip")
            train_envs.save(str(self.model_dir / "continuous_agent_vecnorm.pkl"))
            
        # Final Summary
        avg_return = np.mean([m.get('total_return', 0) for m in all_test_metrics]) * 100
        avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in all_test_metrics])
        
        logger.info("=" * 60)
        logger.info("WALK-FORWARD CROSS VALIDATION COMPLETE")
        logger.info(f"Average Out-of-Sample Return: {avg_return:.2f}%")
        logger.info(f"Average Out-of-Sample Sharpe: {avg_sharpe:.2f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=730)
    args = parser.parse_args()
    
    trainer = WalkForwardTrainer(symbol=args.asset, total_days=args.days)
    trainer.run()
