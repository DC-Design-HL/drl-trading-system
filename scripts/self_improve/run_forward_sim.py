#!/usr/bin/env python3
"""Run the forward simulator over a window (PROFITABILITY_PLAN.md P2.F).

Reads from data/kline_cache/ — refresh first with
``python3 -m scripts.self_improve.refresh_kline_cache``.

Examples:
    python3 -m scripts.self_improve.run_forward_sim \\
        --start 2026-05-12 --end 2026-06-09

    python3 -m scripts.self_improve.run_forward_sim \\
        --symbol BTCUSDT --start 2026-06-01 --end 2026-06-08 \\
        --struct-min-confidence 0.65 \\
        --output runs/forward/btc_p1floor065.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.forward_sim import (  # noqa: E402
    DEFAULT_BLOCKLIST,
    ForwardSimConfig,
    run_forward_sim,
)
from src.self_improve.kline_cache import SUPPORTED_SYMBOLS  # noqa: E402


def _parse_date(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--start", required=True,
        help="ISO date or datetime, e.g. 2026-05-12 or 2026-05-12T00:00:00",
    )
    ap.add_argument(
        "--end", required=True,
        help="ISO date or datetime (exclusive upper bound)",
    )
    ap.add_argument(
        "--symbol", "-s",
        action="append",
        choices=SUPPORTED_SYMBOLS,
        help="Restrict to these symbols (repeatable). Default: all.",
    )
    ap.add_argument(
        "--struct-min-confidence", type=float, default=0.0,
        help="Override STRUCT_MIN_CONFIDENCE floor (raise-only, P3 mechanics).",
    )
    ap.add_argument(
        "--no-blocklist", action="store_true",
        help="Disable the SYMBOL_SIDE_BLOCKLIST (research only — never deploy).",
    )
    ap.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write the result JSON to this path. Default: stdout summary.",
    )
    ap.add_argument(
        "--label", default="",
        help="Free-form label echoed into the result JSON.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    )

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    cfg = ForwardSimConfig(
        struct_min_confidence=args.struct_min_confidence,
        blocklist=frozenset() if args.no_blocklist else DEFAULT_BLOCKLIST,
    )

    symbols = tuple(args.symbol) if args.symbol else SUPPORTED_SYMBOLS

    result = run_forward_sim(
        symbols=symbols, start=start, end=end,
        config=cfg, label=args.label,
    )

    blob = result.to_json()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(blob, indent=2, default=str))
        print(f"wrote {args.output}")

    # Always print a one-line summary per symbol.
    for sym, sr in blob["per_symbol"].items():
        print(
            f"{sym}: trades={sr['n_trades']:>3} "
            f"net=${sr['net_pnl_usd']:+8.2f} "
            f"decisions={sr['n_decisions']} "
            f"skip(open/trend/struct/block/s5)={sr['skipped_by_open_position']}/"
            f"{sr['skipped_by_trend']}/{sr['skipped_by_struct_floor']}/"
            f"{sr['skipped_by_blocklist']}/{sr['skipped_by_s5_unimplemented']} "
            f"runtime={sr['runtime_seconds']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
