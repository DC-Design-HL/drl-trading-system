#!/usr/bin/env python3
"""
Whale Behavior Live Collector — Runs as a systemd service.

Fetches new wallet actions from Etherscan every hour (incremental).
Keeps data/whale_behavior/eth/*.jsonl up to date so the predictor
always has fresh actions for inference.

No AI tokens used — pure API calls.
"""

import logging
import os
import sys
import time
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
logger = logging.getLogger("whale_collector")

COLLECT_INTERVAL = 3600  # 1 hour between collection cycles


def main():
    from src.whale_behavior.data.eth_collector import EthWhaleHistoryCollector

    api_key = os.environ.get("ETHERSCAN_API_KEY", "")
    if not api_key:
        logger.error("ETHERSCAN_API_KEY not set")
        sys.exit(1)

    collector = EthWhaleHistoryCollector(api_key=api_key)

    logger.info("🐋 Whale Behavior Live Collector started")
    logger.info("   Collection interval: %ds (%d min)", COLLECT_INTERVAL, COLLECT_INTERVAL // 60)
    logger.info("   Wallets: %d", len(collector.list_collected_wallets()))

    # Initial collection on startup
    try:
        results = collector.collect_recent()
        total_new = sum(results.values())
        logger.info("✅ Initial collection: %d new actions across %d wallets", total_new, len(results))
    except Exception as e:
        logger.error("Initial collection failed: %s", e)

    # Continuous loop
    while True:
        time.sleep(COLLECT_INTERVAL)
        try:
            results = collector.collect_recent()
            total_new = sum(results.values())
            if total_new > 0:
                logger.info("✅ Collected %d new actions across %d wallets", total_new, len(results))
            else:
                logger.info("No new actions found")
        except Exception as e:
            logger.error("Collection failed: %s", e)


if __name__ == "__main__":
    main()
