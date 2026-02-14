#!/usr/bin/env python3
"""
Multi-Asset Transfer Learning Trainer

Training strategy:
1. Train base model on BTC (most liquid, cleanest patterns)
2. Fine-tune copies for each alt using transfer learning
3. Store per-asset models in organized directory structure

Usage:
    python train_multi_asset.py --base-epochs 500000 --finetune-epochs 100000
    python train_multi_asset.py --finetune-only ETHUSDT  # Fine-tune single asset
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import shutil

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.multi_asset_fetcher import MultiAssetDataFetcher, SUPPORTED_ASSETS, get_all_supported_symbols
from src.features.multi_asset_features import MultiAssetFeatureEngine
from src.env.trading_env import CryptoTradingEnv, create_env_from_df

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAssetTrainer:
    """
    Trains trading agents for multiple assets using transfer learning.
    
    Strategy:
    1. Train a base model on BTC (highest quality data)
    2. For each alt, copy base model and fine-tune
    3. Save per-asset models to organized directory
    """
    
    MODEL_DIR = Path("./data/models/multi_asset")
    
    # Training configurations per asset type
    TRAINING_CONFIG = {
        "base": {  # BTC base training
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "clip_range": 0.2,
        },
        "finetune": {  # Alt fine-tuning
            "learning_rate": 1e-4,  # Lower LR for fine-tuning
            "n_steps": 1024,
            "batch_size": 32,
            "n_epochs": 5,
            "gamma": 0.99,
            "clip_range": 0.15,  # Tighter clip for stability
        }
    }
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        interval: str = "1h",
        training_days: int = 365,
    ):
        """
        Initialize multi-asset trainer.
        
        Args:
            symbols: List of symbols to train on (default: all supported)
            interval: Candle interval for training
            training_days: Number of days of historical data
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        self.interval = interval
        self.training_days = training_days
        
        self.fetcher = MultiAssetDataFetcher()
        self.feature_engine = None
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # Create model directory
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 MultiAssetTrainer initialized for {len(self.symbols)} assets")
        logger.info(f"   Assets: {', '.join(self.symbols)}")
    
    def fetch_training_data(self):
        """Fetch and cache training data for all assets."""
        logger.info(f"📊 Fetching {self.training_days} days of {self.interval} data...")
        
        self.data_cache = self.fetcher.fetch_multiple(
            self.symbols, 
            self.interval, 
            self.training_days
        )
        
        # Log data stats
        for symbol, df in self.data_cache.items():
            logger.info(f"   {symbol}: {len(df)} candles")
        
        # Initialize feature engine with BTC data
        if "BTCUSDT" in self.data_cache:
            self.feature_engine = MultiAssetFeatureEngine(include_cross_asset=True)
            self.feature_engine.set_btc_data(self.data_cache["BTCUSDT"])
        else:
            self.feature_engine = MultiAssetFeatureEngine(include_cross_asset=False)
    
    def create_env(
        self, 
        symbol: str, 
        is_eval: bool = False
    ) -> CryptoTradingEnv:
        """
        Create trading environment for a specific asset.
        
        Args:
            symbol: Trading pair
            is_eval: If True, use last 20% of data for evaluation
        """
        if symbol not in self.data_cache:
            raise ValueError(f"No data for {symbol}. Run fetch_training_data() first.")
        
        df = self.data_cache[symbol].copy()
        
        # Split data
        split_idx = int(len(df) * 0.8)
        
        if is_eval:
            df = df.iloc[split_idx:].reset_index(drop=True)
        else:
            df = df.iloc[:split_idx].reset_index(drop=True)
        
        # Create environment
        env = CryptoTradingEnv(
            df=df,
            initial_balance=10000,
            position_size=0.5,
            feature_engine=self.feature_engine,
            symbol=symbol,
        )
        
        return env
    
    def train_base_model(
        self,
        total_timesteps: int = 500000,
        save_freq: int = 50000,
    ) -> PPO:
        """
        Train base model on BTC.
        
        This creates the foundation model that will be fine-tuned for alts.
        """
        logger.info("=" * 60)
        logger.info("🎯 PHASE 1: Training BASE model on BTCUSDT")
        logger.info("=" * 60)
        
        # Fetch data if not already done
        if not self.data_cache:
            self.fetch_training_data()
        
        # Create environments
        train_env = self.create_env("BTCUSDT", is_eval=False)
        eval_env = self.create_env("BTCUSDT", is_eval=True)
        
        # Create model
        config = self.TRAINING_CONFIG["base"]
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            clip_range=config["clip_range"],
            verbose=1,
            tensorboard_log="./logs/tensorboard/multi_asset/"
        )
        
        # Callbacks
        base_model_path = self.MODEL_DIR / "base_btc"
        base_model_path.mkdir(parents=True, exist_ok=True)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(base_model_path),
            log_path=str(base_model_path / "logs"),
            eval_freq=save_freq,
            deterministic=True,
            render=False,
        )
        
        # Train
        logger.info(f"🏋️ Training for {total_timesteps:,} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )
        
        # Save final model
        final_path = base_model_path / "final_model.zip"
        model.save(str(final_path))
        logger.info(f"✅ Base model saved: {final_path}")
        
        return model
    
    def finetune_for_asset(
        self,
        symbol: str,
        base_model_path: Optional[str] = None,
        total_timesteps: int = 100000,
    ) -> PPO:
        """
        Fine-tune the base model for a specific asset.
        
        Args:
            symbol: Target asset (e.g., "ETHUSDT")
            base_model_path: Path to base model (default: use saved base)
            total_timesteps: Fine-tuning steps
        """
        logger.info("=" * 60)
        logger.info(f"🎯 Fine-tuning for {symbol}")
        logger.info("=" * 60)
        
        # Load base model
        if base_model_path is None:
            base_model_path = str(self.MODEL_DIR / "base_btc" / "best_model.zip")
            if not Path(base_model_path).exists():
                base_model_path = str(self.MODEL_DIR / "base_btc" / "final_model.zip")
        
        if not Path(base_model_path).exists():
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
        
        # Fetch data if not done
        if not self.data_cache or symbol not in self.data_cache:
            if symbol not in self.symbols:
                self.symbols.append(symbol)
            self.fetch_training_data()
        
        # Create environments for target asset
        train_env = self.create_env(symbol, is_eval=False)
        eval_env = self.create_env(symbol, is_eval=True)
        
        # Load and adapt model
        logger.info(f"📥 Loading base model from {base_model_path}")
        model = PPO.load(base_model_path, env=train_env)
        
        # Update hyperparameters for fine-tuning
        # Note: Cannot change n_steps or batch_size after loading as buffer is already initialized
        config = self.TRAINING_CONFIG["finetune"]
        model.learning_rate = config["learning_rate"]
        model.n_epochs = config["n_epochs"]
        
        # Clip range must be a function (schedule)
        clip_value = config["clip_range"]
        model.clip_range = lambda x: clip_value
        
        # Create output directory
        asset_name = symbol.replace("USDT", "").lower()
        asset_model_path = self.MODEL_DIR / asset_name
        asset_model_path.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(asset_model_path),
            log_path=str(asset_model_path / "logs"),
            eval_freq=10000,
            deterministic=True,
            render=False,
        )
        
        # Fine-tune
        logger.info(f"🏋️ Fine-tuning for {total_timesteps:,} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
            reset_num_timesteps=True,  # Fresh counter for fine-tuning
        )
        
        # Save final model
        final_path = asset_model_path / "final_model.zip"
        model.save(str(final_path))
        
        # Also save as canonical name
        canonical_path = self.MODEL_DIR / f"{asset_name}_agent.zip"
        shutil.copy(str(final_path), str(canonical_path))
        
        logger.info(f"✅ {symbol} model saved: {canonical_path}")
        
        return model
    
    def train_all(
        self,
        base_timesteps: int = 500000,
        finetune_timesteps: int = 100000,
        skip_base: bool = False,
    ):
        """
        Train models for all assets.
        
        Args:
            base_timesteps: Training steps for base BTC model
            finetune_timesteps: Fine-tuning steps per alt
            skip_base: If True, skip base training (use existing model)
        """
        logger.info("=" * 60)
        logger.info("🚀 MULTI-ASSET TRAINING STARTING")
        logger.info(f"   Assets: {', '.join(self.symbols)}")
        logger.info(f"   Base timesteps: {base_timesteps:,}")
        logger.info(f"   Fine-tune timesteps: {finetune_timesteps:,}")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Fetch all data upfront
        self.fetch_training_data()
        
        # Phase 1: Train base model on BTC
        if not skip_base:
            self.train_base_model(total_timesteps=base_timesteps)
        else:
            logger.info("⏭️ Skipping base model training (using existing)")
        
        # Phase 2: Fine-tune for each alt
        alt_symbols = [s for s in self.symbols if s != "BTCUSDT"]
        
        for symbol in alt_symbols:
            self.finetune_for_asset(
                symbol=symbol,
                total_timesteps=finetune_timesteps,
            )
        
        # Summary
        elapsed = datetime.now() - start_time
        logger.info("=" * 60)
        logger.info("✅ MULTI-ASSET TRAINING COMPLETE")
        logger.info(f"   Elapsed time: {elapsed}")
        logger.info(f"   Models saved to: {self.MODEL_DIR}")
        logger.info("=" * 60)
        
        # List saved models
        for model_file in self.MODEL_DIR.glob("*.zip"):
            logger.info(f"   📦 {model_file.name}")
    
    @staticmethod
    def load_model(symbol: str) -> PPO:
        """
        Load a trained model for a specific asset.
        
        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            
        Returns:
            Loaded PPO model
        """
        asset_name = symbol.replace("USDT", "").lower()
        model_path = MultiAssetTrainer.MODEL_DIR / f"{asset_name}_agent.zip"
        
        if not model_path.exists():
            # Try BTC base model
            if symbol == "BTCUSDT":
                model_path = MultiAssetTrainer.MODEL_DIR / "base_btc" / "best_model.zip"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model found for {symbol} at {model_path}")
        
        logger.info(f"📥 Loading model for {symbol}: {model_path}")
        return PPO.load(str(model_path))


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Transfer Learning Trainer")
    
    parser.add_argument(
        "--assets", 
        nargs="+", 
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        help="Assets to train on"
    )
    parser.add_argument(
        "--base-epochs", 
        type=int, 
        default=500000,
        help="Training timesteps for base BTC model"
    )
    parser.add_argument(
        "--finetune-epochs", 
        type=int, 
        default=100000,
        help="Fine-tuning timesteps per alt"
    )
    parser.add_argument(
        "--skip-base", 
        action="store_true",
        help="Skip base model training (use existing)"
    )
    parser.add_argument(
        "--finetune-only", 
        type=str,
        help="Only fine-tune a specific asset (e.g., ETHUSDT)"
    )
    parser.add_argument(
        "--interval", 
        type=str, 
        default="1h",
        help="Candle interval"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=365,
        help="Days of historical data"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultiAssetTrainer(
        symbols=args.assets,
        interval=args.interval,
        training_days=args.days,
    )
    
    if args.finetune_only:
        # Fine-tune single asset
        trainer.fetch_training_data()
        trainer.finetune_for_asset(
            symbol=args.finetune_only,
            total_timesteps=args.finetune_epochs,
        )
    else:
        # Full training
        trainer.train_all(
            base_timesteps=args.base_epochs,
            finetune_timesteps=args.finetune_epochs,
            skip_base=args.skip_base,
        )


if __name__ == "__main__":
    main()
