#!/usr/bin/env python3
"""
Train the Whale Behavior LSTM model.

Usage:
    python train_whale_behavior.py                    # Default: 4h label window
    python train_whale_behavior.py --window 1h        # 1-hour prediction
    python train_whale_behavior.py --epochs 100       # More epochs
    python train_whale_behavior.py --batch-size 128   # Larger batches
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("whale_behavior_train")


def main():
    parser = argparse.ArgumentParser(description="Train Whale Behavior LSTM")
    parser.add_argument("--window", type=str, default="4h", help="Label window: 1h, 4h, 12h, 24h")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--seq-length", type=int, default=None, help="Sequence length (default: 20, increase on machines with more RAM)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Whale Behavior LSTM Training")
    logger.info("  Window: %s", args.window)
    logger.info("  Epochs: %d", args.epochs)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Learning rate: %s", args.lr)
    logger.info("  Accum steps: %d (effective batch: %d)", args.accum_steps, args.batch_size * args.accum_steps)
    if args.seq_length:
        logger.info("  Sequence length: %d", args.seq_length)
    logger.info("=" * 60)

    from src.whale_behavior.models.sequence_model import train_model

    # Override SEQ_LENGTH if specified
    if args.seq_length:
        import src.whale_behavior.models.sequence_model as sm
        sm.SEQ_LENGTH = args.seq_length
        logger.info("Overriding SEQ_LENGTH to %d", args.seq_length)

    results = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        label_window=args.window,
        patience=args.patience,
        accum_steps=args.accum_steps,
    )

    if "error" in results:
        logger.error("Training failed: %s", results["error"])
        sys.exit(1)

    logger.info("✅ Training complete!")
    logger.info("Model saved to data/whale_behavior/models/whale_behavior_lstm.pt")


if __name__ == "__main__":
    main()
