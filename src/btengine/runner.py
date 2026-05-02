"""BacktestRunner — orchestrates one config to one result set.

Phase 1 scope: skeleton + dry-run validation. The actual simulation is
implemented in M3 (sim) and M4 (guards); this module is the conductor
that wires them together.

Lifecycle of a run:
    1. Validate config (raises on hard errors, warns on soft)
    2. Resolve output dir, write `config.resolved.yaml`
    3. Pre-warm kline cache for all (symbol, interval) × window
    4. Build strategy components (entry, guards, exits, sizing)
    5. Walk bars, dispatch to strategy.on_bar(ctx) → Intent
    6. Broker fills, position lifecycle, exit triggers
    7. Write results (trades.parquet, summary.json, equity.csv, blocked.parquet)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import BacktestConfig
from .data import KlineCache

logger = logging.getLogger(__name__)


class BacktestRunner:
    """One config in, one results directory out."""

    def __init__(self, config: BacktestConfig,
                 cache: Optional[KlineCache] = None):
        self.config = config
        self.cache = cache or KlineCache()
        self.warnings: List[str] = []
        self.out_dir = config.resolve_output_dir()

    # ── Lifecycle ──────────────────────────────────────────────────
    def dry_run(self) -> Dict[str, Any]:
        """Validate config + cache reachability without simulating.

        Returns a small summary dict suitable for printing/JSON.
        """
        self.warnings = self.config.validate()
        report = {
            "run_id": self.config.run_id,
            "window": [self.config.window.start.isoformat(),
                       self.config.window.end.isoformat()],
            "symbols": self.config.symbols,
            "intervals": {
                "primary": self.config.intervals.primary,
                "htf": list(self.config.intervals.htf),
            },
            "strategy": self.config.strategy,
            "guards_enabled": list(self.config.guards.enabled),
            "sweep_mode": self.config.sweep.mode,
            "sweep_n_axes": len(self.config.sweep.axes),
            "output_dir": str(self.out_dir),
            "warnings": self.warnings,
            "kline_probe": self._probe_klines(),
        }
        return report

    def run(self) -> Dict[str, Any]:
        """Full simulation — Phase 1 stub. Implemented in M3-M5."""
        raise NotImplementedError(
            "BacktestRunner.run() lands in M3 (sim) + M4 (guards) + M5 (results). "
            "Phase 1 currently supports dry_run() only."
        )

    # ── Internals ──────────────────────────────────────────────────
    def _probe_klines(self) -> Dict[str, Any]:
        """Fetch a TINY slice of bars per (symbol, interval) just to confirm
        connectivity + cache integrity. Doesn't pre-warm the full window."""
        from datetime import timedelta as _td
        # Probe the last 2h of primary interval per symbol
        end = datetime.now(timezone.utc)
        start = end - _td(hours=2)
        out: Dict[str, Any] = {}
        intervals = [self.config.intervals.primary] + list(self.config.intervals.htf)
        for sym in self.config.symbols:
            sym_out: Dict[str, int] = {}
            for iv in intervals:
                t0 = time.time()
                try:
                    df = self.cache.get(sym, iv, start, end)
                    sym_out[iv] = int(len(df))
                    logger.debug("probe %s %s: %d bars in %.2fs",
                                 sym, iv, len(df), time.time() - t0)
                except Exception as exc:
                    logger.warning("probe %s %s failed: %s", sym, iv, exc)
                    sym_out[iv] = -1
            out[sym] = sym_out
        return out

    def write_resolved_config(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "config.resolved.yaml"
        # We dump the raw with overrides applied for sweeps (Phase 2);
        # in Phase 1 .raw is the source-of-truth document.
        with open(path, "w") as f:
            yaml.safe_dump(self.config.raw, f, sort_keys=False)
        return path
