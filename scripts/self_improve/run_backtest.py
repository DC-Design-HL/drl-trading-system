#!/usr/bin/env python3
"""CLI wrapper around the backtest harness.

Examples:

  # Baseline (no overrides) over a date range
  python -m scripts.self_improve.run_backtest \\
      --start 2026-04-21 --end 2026-05-20 \\
      --label baseline-pre-blocklist

  # Apply the May-20 XRP blocklist retroactively
  python -m scripts.self_improve.run_backtest \\
      --start 2026-04-21 --end 2026-05-20 \\
      --label retroactive-xrp-blocklist \\
      --blocklist 'XRPUSDT:LONG,XRPUSDT:SHORT'

  # Raise the confidence floor portfolio-wide
  python -m scripts.self_improve.run_backtest \\
      --start 2026-04-21 --end 2026-05-20 \\
      --min-conf 0.55
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.backtest_harness import (  # noqa: E402
    BacktestRequest,
    run_backtest,
    serialize,
)


def _parse_blocklist(s: str) -> list[list[str]]:
    """'XRPUSDT:LONG,XRPUSDT:SHORT' → [['XRPUSDT','LONG'], ['XRPUSDT','SHORT']]"""
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        sym, _, side = chunk.partition(":")
        out.append([sym.upper(), side.upper()])
    return out


def _parse_per_symbol_conf(s: str) -> dict[str, float]:
    """'XRPUSDT=0.65,BTCUSDT=0.60' → {'XRPUSDT': 0.65, 'BTCUSDT': 0.60}"""
    out = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        sym, _, val = chunk.partition("=")
        out[sym.upper().strip()] = float(val.strip())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", required=True, help="ISO date/datetime, inclusive")
    p.add_argument("--end", required=True, help="ISO date/datetime, inclusive")
    p.add_argument("--db", default="data/trading.db")
    p.add_argument("--label", default="cli-run", help="Human label for the result")
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol filter (e.g. BTCUSDT,ETHUSDT). Default: all.",
    )
    p.add_argument(
        "--capital-base", type=float, default=5000.0,
        help="Capital base used for return-% in metrics (default $5000 = May-1 reset)",
    )
    p.add_argument(
        "--min-conf",
        type=float,
        default=None,
        help="Portfolio-wide confidence floor",
    )
    p.add_argument(
        "--per-symbol-conf",
        default=None,
        help="Per-symbol confidence floor as 'SYM=val,SYM=val'",
    )
    p.add_argument(
        "--blocklist",
        default=None,
        help="Additional blocklist entries as 'SYM:SIDE,SYM:SIDE'",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Path to write JSON result. If omitted, prints to stdout.",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print a short summary to stderr in addition to JSON output",
    )
    args = p.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.min_conf is not None:
        overrides["MIN_CONFIDENCE"] = args.min_conf
    if args.per_symbol_conf:
        overrides["SYMBOL_MIN_CONFIDENCE"] = _parse_per_symbol_conf(
            args.per_symbol_conf
        )
    if args.blocklist:
        overrides["SYMBOL_SIDE_BLOCKLIST_ADD"] = _parse_blocklist(args.blocklist)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",")) if args.symbols else None

    req = BacktestRequest(
        start_date=args.start,
        end_date=args.end,
        config_overrides=overrides,
        symbols=symbols,
        capital_base=args.capital_base,
        mode="replay",
        label=args.label,
        db_path=args.db,
    )

    result = run_backtest(req)
    body = serialize(result)

    if args.out:
        Path(args.out).write_text(body)
        print(f"✅ Result written to {args.out}", file=sys.stderr)
    else:
        print(body)

    if args.summary or args.out:
        m = result.portfolio_metrics
        print(
            f"\n[{result.label}] kept={result.n_kept_pairs}/{result.n_input_pairs} "
            f"blocked={result.n_blocked_pairs}  "
            f"pnl=${m.get('net_pnl_usd', 0):+.2f}  "
            f"WR={m.get('win_rate', 0) * 100:.1f}%  "
            f"PF={m.get('profit_factor', 0):.2f}  "
            f"Sharpe={m.get('sharpe', 0):.2f}  "
            f"DD={m.get('max_drawdown_pct', 0):.2f}%  "
            f"({result.runtime_seconds * 1000:.0f}ms)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
