#!/usr/bin/env python3
"""
Whale Behavior Data Collection Pipeline

Usage:
    python collect_whale_behavior.py                  # Collect + label all wallets
    python collect_whale_behavior.py --collect-only   # Just collect (no labeling)
    python collect_whale_behavior.py --label-only     # Just label (assumes data exists)
    python collect_whale_behavior.py --force           # Re-collect everything from scratch
    python collect_whale_behavior.py --wallet "Binance Hot Wallet"  # Single wallet
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("whale_behavior")


def main():
    parser = argparse.ArgumentParser(description="Whale Behavior Data Collection")
    parser.add_argument("--collect-only", action="store_true", help="Only collect, don't label")
    parser.add_argument("--label-only", action="store_true", help="Only label existing data")
    parser.add_argument("--force", action="store_true", help="Re-collect from scratch")
    parser.add_argument("--wallet", type=str, help="Collect a specific wallet by label")
    args = parser.parse_args()

    if not args.label_only:
        logger.info("=" * 60)
        logger.info("Phase 1: Collecting ETH whale wallet histories")
        logger.info("=" * 60)

        from src.whale_behavior.data.eth_collector import EthWhaleHistoryCollector
        collector = EthWhaleHistoryCollector()

        if args.wallet:
            from src.features.whale_wallet_registry import get_wallets_by_chain
            wallets = get_wallets_by_chain("ETH")
            target = next((w for w in wallets if w.label == args.wallet), None)
            if not target:
                logger.error("Wallet not found: %s", args.wallet)
                logger.info("Available wallets: %s", [w.label for w in wallets])
                return
            collector.collect_wallet(target.address, target.label, force=args.force)
        else:
            results = collector.collect_all_wallets(force=args.force)
            logger.info("Collection results: %s", results)

    if not args.collect_only:
        logger.info("=" * 60)
        logger.info("Phase 2: Labeling with price outcomes")
        logger.info("=" * 60)

        from src.whale_behavior.data.price_labeler import PriceLabeler
        labeler = PriceLabeler()

        if args.wallet:
            labeler.label_wallet(args.wallet)
        else:
            results = labeler.label_all_wallets()
            logger.info("Labeling results: %s", results)

    logger.info("✅ Done!")


if __name__ == "__main__":
    main()
