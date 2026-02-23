#!/usr/bin/env python3
"""
Train Whale Pattern Models

Collects historical transactions for tracked whale wallets,
cross-correlates with price data, and trains pattern models.

Usage:
    python train_whale_patterns.py                     # All chains, 90 days
    python train_whale_patterns.py --chain ETH         # ETH only
    python train_whale_patterns.py --chain XRP --days 60
    python train_whale_patterns.py --collect-only      # Just collect data
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_whale_patterns")


def collect_data(chains: list, max_pages: int = 5):
    """Collect historical transaction data for whale wallets."""
    from src.features.whale_wallet_collector import WhaleWalletCollector

    collector = WhaleWalletCollector()

    for chain in chains:
        logger.info(f"\n{'='*60}")
        logger.info(f"📥 Collecting {chain} whale wallet transactions...")
        logger.info(f"{'='*60}")
        results = collector.collect_chain(chain, max_pages=max_pages)

        total_txns = sum(
            len(d.get("transactions", [])) for d in results.values()
        )
        logger.info(
            f"✅ {chain}: Collected data for {len(results)} wallets, "
            f"{total_txns} total transactions"
        )


def train_models(chains: list, days: int = 90):
    """Train whale pattern models for specified chains."""
    from src.models.whale_pattern_learner import WhalePatternLearner

    results = {}

    for chain in chains:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 Training whale pattern model for {chain}...")
        logger.info(f"{'='*60}")

        learner = WhalePatternLearner(chain)
        success = learner.train(days=days)

        results[chain] = success
        if success:
            logger.info(f"✅ {chain} model trained successfully!")
        else:
            logger.warning(
                f"⚠️ {chain} model training failed — "
                f"not enough data or price data unavailable"
            )

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 Training Summary:")
    logger.info(f"{'='*60}")
    for chain, success in results.items():
        status = "✅ TRAINED" if success else "❌ FAILED"
        logger.info(f"  {chain}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Train whale pattern models"
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Specific chain to process (ETH, SOL, XRP). Default: all",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of price history to use for training (default: 90)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Max pagination pages per wallet (default: 20)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect data, don't train",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train models (skip collection)",
    )

    args = parser.parse_args()

    chains = [args.chain.upper()] if args.chain else ["ETH", "SOL", "XRP"]

    logger.info("🐋 Whale Pattern Training Pipeline")
    logger.info(f"   Chains: {', '.join(chains)}")
    logger.info(f"   Days: {args.days}")
    logger.info(f"   Max Pages: {args.max_pages}")

    if not args.train_only:
        collect_data(chains, max_pages=args.max_pages)

    if not args.collect_only:
        train_models(chains, days=args.days)

    logger.info("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    main()
