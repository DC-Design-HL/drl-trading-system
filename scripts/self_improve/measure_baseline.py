#!/usr/bin/env python3
"""Measure the current trailing-30d baseline.

The baseline is the reference point every proposed experiment is
compared against. Captures: per-symbol metrics, portfolio metrics,
current config touchpoints (so future researchers know what knobs the
baseline used).

Writes:
  data/self_improve/baseline_metrics.json
  data/self_improve/baseline_config_fingerprint.json

Re-running overwrites — that's fine, the baseline is meant to be a
moving reference. Past baselines are still recoverable via git history
of the JSON files.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.self_improve.performance_monitor import (  # noqa: E402
    DEFAULT_CAPITAL_BASE,
    DEFAULT_SINCE,
    _filter_days,
    load_closes,
)
from src.self_improve.metrics import summarize  # noqa: E402

# Config touchpoints we treat as part of the baseline fingerprint.
# These match constants the Researcher / Implementer might propose
# changes to.
_FINGERPRINT_CONSTANTS = (
    "MIN_CONFIDENCE",
    "SIGNAL_GATE_AUTONOMOUS",
    "SIGNAL_GATE_MIN_CONFIRMS",
    "RANGING_MIN_CONFIDENCE",
    "RANGING_ADX_THRESHOLD",
    "ADX_GUARD_MIN",
    "USDT_D_THRESHOLD_PCT",
    "USDT_D_LOOKBACK_HOURS",
    "EXT_POS_NEWS_SENTIMENT_THRESHOLD",
    "EXT_POS_NEWS_LOOKBACK_MINUTES",
    "STOP_LOSS_PCT",
    "TAKE_PROFIT_PCT",
    "TRAILING_DISTANCE_PCT",
    "STAGNANT_HOURS",
    "STAGNANT_PCT_MIN",
    "STAGNANT_PCT_MAX",
    "WHIPSAW_COOLDOWN_HOURS",
    "FIXED_MAX_NOTIONAL",
    "COOLDOWN_SECONDS",
    "MIN_HOLD_SECONDS",
    "SYMBOL_SIDE_BLOCKLIST",
    "SYMBOL_MIN_CONFIDENCE",
    "SYMBOL_DIRECTIONAL_CONF",
    # Structure-confidence floors (PROFITABILITY_PLAN.md P1) — these are
    # the live apply surface in STRUCTURE_FIRST_MODE; the legacy
    # MIN_CONFIDENCE family above is inert in that mode.
    "STRUCT_MIN_CONFIDENCE",
    "STRUCT_SYMBOL_MIN_CONFIDENCE",
    "STRUCT_SYMBOL_DIRECTIONAL_CONF",
    "STRUCTURE_FIRST_MODE",
    "REVERSAL_BLOCK_LONG_CANARY_SYMBOLS",
    "SYMBOL_SIZE_SCALING_ENABLED",
    "SYMBOL_SIZE_WR_THRESHOLD",
    "REGIME_TP_SCHEDULE_ENABLED",
)


def fingerprint_config() -> dict[str, object]:
    """Import live_trading_htf and snapshot its tunables."""
    import importlib
    mod = importlib.import_module("live_trading_htf")
    out: dict[str, object] = {}
    for name in _FINGERPRINT_CONSTANTS:
        if not hasattr(mod, name):
            continue
        val = getattr(mod, name)
        out[name] = _jsonable(val)
    return out


def _jsonable(v: object) -> object:
    """Convert sets and other non-JSON natives into JSON-safe values."""
    if isinstance(v, set):
        return sorted(list(v), key=str)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return v


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def measure(
    db_path: Path,
    *,
    since: str = DEFAULT_SINCE,
    capital_base: float = DEFAULT_CAPITAL_BASE,
    now: datetime | None = None,
) -> dict[str, object]:
    now = now or datetime.now(timezone.utc)
    with sqlite3.connect(str(db_path)) as conn:
        closes = load_closes(conn, since=since)

    last_30 = _filter_days(closes, now, 30)
    last_7 = _filter_days(closes, now, 7)

    by_symbol: dict[str, list] = {}
    for t in last_30:
        by_symbol.setdefault(t.symbol, []).append(t)

    return {
        "captured_at": now.isoformat(),
        "window_since": since,
        "capital_base_usd": capital_base,
        "git_head": _git_head(),
        "portfolio_30d": _safe_summary(last_30, capital_base),
        "portfolio_7d": _safe_summary(last_7, capital_base),
        "per_symbol_30d": {
            sym: _safe_summary(rows, capital_base)
            for sym, rows in by_symbol.items()
        },
        "notes": (
            "Baseline metrics are the comparison point used by future "
            "experiments. The Backtester / Risk Officer compares proposed "
            "configs against these numbers. Update by re-running this "
            "script after any significant manual change to the live "
            "config."
        ),
    }


def _safe_summary(trades, capital_base):
    out = summarize(trades, capital_base=capital_base)
    # Make JSON-safe
    import math
    for k, v in list(out.items()):
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            out[k] = None if math.isnan(v) else 9999.0
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="data/trading.db")
    parser.add_argument("--since", default=DEFAULT_SINCE)
    parser.add_argument(
        "--capital-base", type=float, default=DEFAULT_CAPITAL_BASE
    )
    parser.add_argument(
        "--out-dir", default="data/self_improve",
        help="Directory for baseline_metrics.json and baseline_config_fingerprint.json",
    )
    parser.add_argument(
        "--print", action="store_true",
        help="Also print the baseline to stdout after writing",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = measure(
        Path(args.db),
        since=args.since,
        capital_base=args.capital_base,
    )
    config = fingerprint_config()

    metrics_path = out_dir / "baseline_metrics.json"
    config_path = out_dir / "baseline_config_fingerprint.json"
    metrics_path.write_text(json.dumps(baseline, indent=2, default=str))
    config_path.write_text(json.dumps(config, indent=2, default=str))

    if args.print:
        print(json.dumps(baseline, indent=2, default=str))
    print(f"✅ Wrote baseline metrics → {metrics_path}")
    print(f"✅ Wrote config fingerprint → {config_path}  ({len(config)} constants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
