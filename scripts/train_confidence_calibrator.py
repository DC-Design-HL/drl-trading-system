#!/usr/bin/env python3
"""
Train the v2 Confidence Calibrator from historical trade data.

Run after collecting enough trades (minimum ~30-50):
    python scripts/train_confidence_calibrator.py

Reads: logs/htf_pending_alerts.jsonl
Saves: data/models/htf/calibration_v2.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s - %(message)s")
logger = logging.getLogger("calibrator_train")

from src.brain.confidence_calibrator import ConfidenceCalibrator


def main():
    calibrator = ConfidenceCalibrator()
    
    logger.info("Training confidence calibrator from trade history...")
    results = calibrator.train_from_trades()
    
    if "error" in results:
        logger.error("Training failed: %s", results)
        return
    
    logger.info("=" * 60)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 60)
    logger.info("  Trades analyzed: %d", results["trades_analyzed"])
    logger.info("  Optimal temperature: %.1f", results["temperature"])
    logger.info("  Calibration error: %.4f", results["calibration_error"])
    logger.info("")
    logger.info("  Win rate by confidence bin:")
    for bin_key, wr in sorted(results["win_rate_bins"].items()):
        logger.info("    %s → %.0f%% win rate", bin_key, wr * 100)
    logger.info("")
    logger.info("  Regime discounts:")
    for regime, discount in sorted(results["regime_discounts"].items()):
        logger.info("    %s → %.2fx", regime, discount)
    
    calibrator.save()
    logger.info("\n✅ Calibration saved to data/models/htf/calibration_v2.json")
    logger.info("   The live bot will pick this up automatically on next restart.")


if __name__ == "__main__":
    main()
