#!/usr/bin/env python3
"""Hourly orchestrator cron entry point.

Calls run_tick() once and prints the actions summary. Telegram pings
are sent from inside run_tick() when the canary gate fires or a
researcher escalation surfaces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.self_improve.orchestrator import run_tick  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", default="data/trading.db")
    p.add_argument("--json", action="store_true",
                   help="Emit the result as JSON to stdout")
    args = p.parse_args(argv)

    result = run_tick(db_path=args.db)
    if args.json:
        print(json.dumps(result.to_json(), indent=2))
    else:
        print(f"orchestrator tick: advanced={result.n_experiments_advanced} "
              f"proposed={result.n_experiments_proposed} "
              f"actions={len(result.actions_taken)}")
        for a in result.actions_taken:
            print(f"  · {a}")
        if result.error:
            print(f"  ERROR: {result.error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
