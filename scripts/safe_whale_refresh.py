#!/usr/bin/env python3
"""
Safe whale data refresh — collects + labels with RAM monitoring.
Aborts if RAM exceeds 90%.
"""
import gc
import logging
import os
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("whale_refresh")

RAM_LIMIT_PCT = 90.0


def check_ram(stage: str) -> bool:
    """Returns True if safe to continue, False if over limit."""
    mem = psutil.virtual_memory()
    logger.info("RAM check [%s]: %.1f%% (%dMB / %dMB)", stage, mem.percent, mem.used // (1024*1024), mem.total // (1024*1024))
    if mem.percent >= RAM_LIMIT_PCT:
        logger.error("🛑 RAM at %.1f%% — ABORTING to protect server!", mem.percent)
        return False
    return True


def step1_collect():
    """Pull latest raw transactions from Etherscan."""
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting latest whale transactions from Etherscan")
    logger.info("=" * 60)

    if not check_ram("pre-collect"):
        return False

    from src.whale_behavior.data.eth_collector import EthWhaleHistoryCollector

    collector = EthWhaleHistoryCollector()

    # Collect one wallet at a time with RAM checks
    from src.features.whale_wallet_registry import get_wallets_by_chain
    wallets = get_wallets_by_chain("ETH")

    results = {}
    for wallet in wallets:
        if not wallet.active:
            logger.info("Skipping inactive: %s", wallet.label)
            continue

        if not check_ram(f"before-{wallet.label}"):
            logger.warning("Stopping collection early due to RAM pressure")
            break

        try:
            actions = collector.collect_wallet(wallet.address, wallet.label, force=False)
            results[wallet.label] = len(actions)
            logger.info("✅ %s: %d total actions", wallet.label, len(actions))
        except Exception as e:
            logger.error("❌ %s failed: %s", wallet.label, e)
            results[wallet.label] = -1

        # Free memory between wallets
        gc.collect()
        # Rate limit — Etherscan free tier is 5 req/sec
        time.sleep(0.5)

    logger.info("📊 Collection results: %s", results)
    return True


def step2_label():
    """Re-label all raw data with price outcomes."""
    logger.info("=" * 60)
    logger.info("STEP 2: Labeling all raw data with price outcomes")
    logger.info("=" * 60)

    if not check_ram("pre-label"):
        return False

    from src.whale_behavior.data.price_labeler import PriceLabeler
    from src.whale_behavior.data.eth_collector import EthWhaleHistoryCollector

    labeler = PriceLabeler()

    # Get list of raw data files
    raw_dir = Path("data/whale_behavior/eth")
    labeled_dir = Path("data/whale_behavior/labeled")
    labeled_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in sorted(raw_dir.glob("*.jsonl")):
        wallet_name = raw_file.stem
        
        if not check_ram(f"before-label-{wallet_name}"):
            logger.warning("Stopping labeling early due to RAM pressure")
            break

        logger.info("Labeling %s (%d lines)...", wallet_name, sum(1 for _ in open(raw_file)))

        try:
            labeled = labeler.label_wallet(wallet_name)
            logger.info("✅ %s: %d labeled", wallet_name, len(labeled))

            # Free memory
            del labeled
            gc.collect()

        except Exception as e:
            logger.error("❌ Failed to label %s: %s", wallet_name, e)
            gc.collect()
            continue

    return True


def step3_summary():
    """Print summary of what we have now."""
    logger.info("=" * 60)
    logger.info("STEP 3: Summary")
    logger.info("=" * 60)

    raw_dir = Path("data/whale_behavior/eth")
    labeled_dir = Path("data/whale_behavior/labeled")

    total_raw = 0
    total_labeled = 0

    for raw_file in sorted(raw_dir.glob("*.jsonl")):
        name = raw_file.stem
        raw_count = sum(1 for _ in open(raw_file))
        total_raw += raw_count

        labeled_file = labeled_dir / f"{name}_labeled.jsonl"
        if labeled_file.exists():
            labeled_count = sum(1 for _ in open(labeled_file))
            total_labeled += labeled_count
            logger.info("  %s: raw=%d labeled=%d", name, raw_count, labeled_count)
        else:
            logger.info("  %s: raw=%d labeled=MISSING", name, raw_count)

    logger.info("TOTAL: raw=%d labeled=%d", total_raw, total_labeled)


if __name__ == "__main__":
    logger.info("Starting safe whale data refresh...")
    logger.info("RAM limit: %.0f%%", RAM_LIMIT_PCT)

    if not check_ram("startup"):
        logger.error("Already over RAM limit at startup! Aborting.")
        sys.exit(1)

    ok = step1_collect()
    if ok:
        gc.collect()
        time.sleep(2)  # Let memory settle
        ok = step2_label()

    step3_summary()

    mem = psutil.virtual_memory()
    logger.info("Final RAM: %.1f%% (%dMB)", mem.percent, mem.used // (1024*1024))

    if ok:
        logger.info("✅ All done! Run 'git add data/whale_behavior && git push origin dev' to push to Chen's Mac")
    else:
        logger.warning("⚠️ Completed partially due to RAM pressure")
