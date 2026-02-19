#!/usr/bin/env python3
"""
TFT Training Pipeline

Walk-forward training for the Temporal Fusion Transformer:
1. Fetches 2+ years of historical OHLCV data
2. Creates windowed (lookback, n_features) → (n_horizons,) dataset
3. Trains with quantile loss on MPS/CUDA/CPU
4. Evaluates directional accuracy per horizon
5. Saves best model per asset

Usage:
    python -m src.models.train_forecaster --asset BTCUSDT --epochs 100
    python -m src.models.train_forecaster --asset ETHUSDT --epochs 80 --resume
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.price_forecaster import (
    TemporalFusionTransformer,
    TFTFeaturePreprocessor,
    QuantileLoss,
    TFTForecaster,
)
from src.backtest.data_loader import download_binance_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Sliding window dataset for TFT training.

    Each sample:
    - x: (lookback, n_features) — past features
    - y: (n_horizons,) — future returns at each horizon
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lookback: int = 72,
    ):
        self.features = features
        self.targets = targets
        self.lookback = lookback

        # Valid indices: need lookback bars of history AND forward targets to exist
        max_horizon = targets.shape[1]  # number of horizons
        self.valid_indices = []
        for i in range(lookback, len(features)):
            # Check that targets are not all zero (which means we're at the end)
            if i < len(targets) and not np.all(targets[i] == 0):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        x = self.features[actual_idx - self.lookback:actual_idx]
        y = self.targets[actual_idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ─── Data Loading ─────────────────────────────────────────────────────────────

def fetch_training_data(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    days: int = 730,  # 2 years
) -> pd.DataFrame:
    """Fetch historical data for training."""
    logger.info(f"📥 Fetching {days} days of {timeframe} data for {symbol}...")

    # Convert BTCUSDT → BTC/USDT for the loader
    clean_symbol = symbol.replace('USDT', '/USDT')
    df = download_binance_data(
        symbol=clean_symbol,
        timeframe=timeframe,
        days=days,
    )

    if df is None or df.empty:
        raise ValueError(f"Failed to fetch data for {symbol}")

    logger.info(f"✅ Fetched {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    return df


def prepare_datasets(
    df: pd.DataFrame,
    lookback: int = 72,
    horizons: list = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple:
    """
    Prepare train/val/test datasets with walk-forward split.

    Split is chronological (no data leakage):
    [====== train 70% ======][=== val 15% ===][=== test 15% ===]
    """
    horizons = horizons or [1, 4, 12, 24]

    # Compute features and targets
    features = TFTFeaturePreprocessor.prepare_features(df)
    targets = TFTFeaturePreprocessor.prepare_targets(df, horizons)

    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_ds = TimeSeriesDataset(features[:train_end], targets[:train_end], lookback)
    val_ds = TimeSeriesDataset(features[train_end:val_end], targets[train_end:val_end], lookback)
    test_ds = TimeSeriesDataset(features[val_end:], targets[val_end:], lookback)

    logger.info(
        f"📊 Datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} "
        f"(lookback={lookback}, horizons={horizons})"
    )

    return train_ds, val_ds, test_ds


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_tft(
    symbol: str = 'BTCUSDT',
    epochs: int = 100,
    lookback: int = 72,
    hidden_dim: int = 64,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 15,
    days: int = 730,
    resume: bool = False,
    model_dir: str = './data/models/tft',
):
    """Full TFT training pipeline."""

    # ── Device ──
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"🖥️  Device: {device}")

    # ── Data ──
    df = fetch_training_data(symbol, '1h', days)
    train_ds, val_ds, test_ds = prepare_datasets(df, lookback)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ──
    model = TemporalFusionTransformer(
        n_features=TFTFeaturePreprocessor.N_FEATURES,
        hidden_dim=hidden_dim,
    ).to(device)

    # Resume from checkpoint
    model_path = os.path.join(model_dir, f'tft_{symbol.lower()}.pt')
    if resume and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"📂 Resumed from {model_path}")

    # ── Training setup ──
    criterion = QuantileLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🧠 Model parameters: {n_params:,}")

    best_val_loss = float('inf')
    patience_counter = 0

    # ── Training loop ──
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Training TFT for {symbol} ({epochs} epochs)")
    logger.info(f"{'='*60}\n")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions, _ = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= max(n_train_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        val_correct = {h: 0 for h in [0, 1, 2, 3]}
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                predictions, _ = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                n_val_batches += 1

                # Directional accuracy (median predictions)
                median_idx = 1  # 50th percentile
                pred_dir = torch.sign(predictions[:, :, median_idx])
                true_dir = torch.sign(batch_y)

                for h in range(predictions.shape[1]):
                    val_correct[h] += (pred_dir[:, h] == true_dir[:, h]).sum().item()
                val_total += batch_y.shape[0]

        val_loss /= max(n_val_batches, 1)
        scheduler.step()

        # Directional accuracy per horizon
        dir_acc = {h: val_correct[h] / max(val_total, 1) * 100 for h in val_correct}

        # ── Logging ──
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Dir Acc: 1h={dir_acc[0]:.1f}% 4h={dir_acc[1]:.1f}% "
                f"12h={dir_acc[2]:.1f}% 24h={dir_acc[3]:.1f}% | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"  💾 Best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  ⏹️  Early stopping (no improvement for {patience} epochs)")
                break

    # ── Final Evaluation on Test Set ──
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Final Evaluation on Test Set")
    logger.info(f"{'='*60}\n")

    # Load best model
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            predictions, _ = model(batch_x)
            median_preds = predictions[:, :, 1].cpu().numpy()  # 50th percentile
            test_preds.append(median_preds)
            test_targets.append(batch_y.numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)

    horizons = [1, 4, 12, 24]
    for i, h in enumerate(horizons):
        pred_dir = np.sign(test_preds[:, i])
        true_dir = np.sign(test_targets[:, i])

        dir_acc = (pred_dir == true_dir).mean() * 100
        mae = np.mean(np.abs(test_preds[:, i] - test_targets[:, i])) * 100
        corr = np.corrcoef(test_preds[:, i], test_targets[:, i])[0, 1]

        logger.info(
            f"  {h}h horizon: Dir.Acc={dir_acc:.1f}% | MAE={mae:.3f}% | Corr={corr:.3f}"
        )

    # Save evaluation report
    report = {
        'symbol': symbol,
        'epochs_trained': epoch + 1,
        'best_val_loss': float(best_val_loss),
        'test_results': {},
        'n_params': n_params,
        'lookback': lookback,
        'hidden_dim': hidden_dim,
    }

    for i, h in enumerate(horizons):
        pred_dir = np.sign(test_preds[:, i])
        true_dir = np.sign(test_targets[:, i])
        report['test_results'][f'{h}h'] = {
            'directional_accuracy': float((pred_dir == true_dir).mean() * 100),
            'mae': float(np.mean(np.abs(test_preds[:, i] - test_targets[:, i])) * 100),
            'correlation': float(np.corrcoef(test_preds[:, i], test_targets[:, i])[0, 1]),
        }

    import json
    report_path = os.path.join(model_dir, f'tft_{symbol.lower()}_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n✅ Training complete! Model: {model_path}")
    logger.info(f"📄 Report: {report_path}")

    return model_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TFT Price Forecaster')
    parser.add_argument('--asset', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lookback', type=int, default=72, help='Lookback window (hours)')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--days', type=int, default=730, help='Days of data')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    args = parser.parse_args()

    train_tft(
        symbol=args.asset,
        epochs=args.epochs,
        lookback=args.lookback,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        days=args.days,
        resume=args.resume,
    )
