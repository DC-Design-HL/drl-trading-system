#!/usr/bin/env python3
"""
Regime Specialist Trainer

Trains a specialized PPO agent on a specific market regime:
BULL_TREND, BEAR_TREND, RANGE_CHOP, or HIGH_VOL_BREAKOUT.

The HMM Regime Classifier is used to filter the historical data
so the agent only sees state-transitions relevant to its specialty.

Usage:
    python -m src.models.train_specialist --asset BTCUSDT --regime BULL_TREND --timesteps 500000
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.env.mtf_env import create_mtf_env
from src.models.regime_classifier import RegimeClassifier, REGIME_NAMES
from src.backtest.data_loader import download_binance_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_regime_dataset(
    df: pd.DataFrame,
    regime: str,
    classifier: RegimeClassifier,
    context_window: int = 48,
) -> pd.DataFrame:
    """
    Filter dataframe to only include periods where the requested regime is active.
    
    CRITICAL: We can't just drop rows, because the environment needs contiguous
    time-series data to compute features (like RSI, MACD, etc.) properly.
    
    Instead, we find continuous 'chunks' of the regime. If a chunk is shorter
    than the context window, we discard it.
    """
    logger.info(f"Filtering dataset for {regime} regime...")
    
    indices = classifier.get_regime_filtered_indices(df, regime)
    
    if len(indices) == 0:
        logger.warning(f"No data found for regime {regime}")
        return pd.DataFrame()
        
    # Find contiguous chunks
    chunks = []
    current_chunk = [indices[0]]
    
    min_chunk_length = 6  # Require at least 6 consecutive hours in the regime
    
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current_chunk.append(indices[i])
        else:
            if len(current_chunk) >= min_chunk_length:
                chunks.append(current_chunk)
            current_chunk = [indices[i]]
            
    if len(current_chunk) >= min_chunk_length:
        chunks.append(current_chunk)
        
    # Reassemble safe dataframe chunks
    # We pad the beginning of each chunk with context_window bars from BEFORE the regime started
    # so the environment can compute features right when the regime begins.
    safe_indices = []
    for chunk in chunks:
        start_idx = max(0, chunk[0] - context_window)
        end_idx = chunk[-1]
        safe_indices.extend(list(range(start_idx, end_idx + 1)))
        
    # Remove duplicates but keep order
    safe_indices = sorted(list(set(safe_indices)))
    
    regime_df = df.iloc[safe_indices].copy()
    
    logger.info(
        f"✅ Filtered to {len(regime_df)} bars "
        f"({len(regime_df)/len(df)*100:.1f}%), {len(chunks)} chunks"
    )
    
    return regime_df


def train_specialist(
    symbol: str = 'BTCUSDT',
    regime: str = 'BULL_TREND',
    timesteps: int = 500000,
    days: int = 730,
):
    """Train a regime specialist agent."""
    
    # ── Verify Regime ──
    valid_regimes = list(REGIME_NAMES.values())
    if regime not in valid_regimes:
        raise ValueError(f"Invalid regime {regime}. Must be one of {valid_regimes}")
        
    # ── Fetch Data ──
    clean_symbol = symbol.replace('USDT', '/USDT')
    logger.info(f"📥 Fetching {days} days of data for {clean_symbol}...")
    df = download_binance_data(symbol=clean_symbol, timeframe='1h', days=days)
    
    # ── Load Regime Classifier ──
    classifier = RegimeClassifier()
    if not classifier.load(symbol):
        logger.info(f"No existing classifier found. Training new HMM for {symbol}...")
        classifier.fit(df)
        classifier.save(symbol)
        
    # ── Filter Data ──
    regime_df = create_regime_dataset(df, regime, classifier, context_window=48)
    if len(regime_df) < 1000:
        logger.error(f"Not enough data to train {regime} specialist (only {len(regime_df)} bars)")
        return
        
    # ── Train/Val Split ──
    split_idx = int(len(regime_df) * 0.8)
    train_df = regime_df.iloc[:split_idx]
    val_df = regime_df.iloc[split_idx:]
    
    # ── Environments ──
    # Using MTF Env from Phase 11.2
    logger.info("Setting up Multi-Timeframe Environments...")
    train_env = create_mtf_env(train_df)
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])
    
    val_env = create_mtf_env(val_df)
    val_env = Monitor(val_env)
    val_env = DummyVecEnv([lambda: val_env])
    
    # ── PPO Setup ──
    # We use a slightly smaller network since it's a specialist
    policy_kwargs = dict(net_arch=[dict(pi=[256, 128], vf=[256, 128])])
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto"
    )
    
    # ── Training ──
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Training {regime} Specialist for {symbol}")
    logger.info(f"{'='*60}\n")
    
    model_dir = Path('./data/models/specialists')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"ppo_{symbol.lower()}_{regime.lower()}"
    best_model_dir = model_dir / f"best_{symbol.lower()}_{regime.lower()}"
    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(model_dir / "logs"),
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    
    # Save final model
    model.save(model_path)
    logger.info(f"✅ Final model saved to {model_path}.zip")
    logger.info(f"✅ Best model saved to {best_model_dir}/best_model.zip")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Regime Specialist')
    parser.add_argument('--asset', type=str, default='BTCUSDT')
    parser.add_argument('--regime', type=str, required=True, 
                        help='BULL_TREND, BEAR_TREND, RANGE_CHOP, or HIGH_VOL_BREAKOUT')
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--days', type=int, default=730)
    args = parser.parse_args()
    
    train_specialist(
        symbol=args.asset,
        regime=args.regime.upper(),
        timesteps=args.timesteps,
        days=args.days,
    )
