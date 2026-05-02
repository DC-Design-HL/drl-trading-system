#!/usr/bin/env python3
"""Unified backtest CLI — entry point for the btengine framework.

Usage:
    python3 backtest.py --config configs/sweeps/example.yaml --dry-run
    python3 backtest.py --config configs/sweeps/example.yaml

In Phase 1 only --dry-run is fully wired. The simulation lands in M3-M5.

See docs/backtest_redesign_proposal.md for design + migration plan, and
src/btengine/live_constants.py for the production constants this engine
mirrors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make repo root importable so `src.btengine` works regardless of CWD
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.btengine import BacktestConfig, BacktestRunner  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DRL trading system backtest framework (btengine)")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config + probe data; do not simulate")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    cfg = BacktestConfig.from_yaml(args.config)
    runner = BacktestRunner(cfg)

    if args.dry_run:
        report = runner.dry_run()
        print(json.dumps(report, indent=2, default=str))
        if report.get("warnings"):
            print("\nWARNINGS:", file=sys.stderr)
            for w in report["warnings"]:
                print(f"  - {w}", file=sys.stderr)
        return 0

    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
