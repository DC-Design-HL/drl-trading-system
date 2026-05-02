"""YAML config loader + schema validation for backtest runs.

Schema (excerpt — see docs/backtest_redesign_proposal.md):

    run_id: my_run
    window: { start: 2026-02-01, end: 2026-04-30 }
    symbols: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT]
    intervals: { primary: 15m, htf: [1h, 4h] }
    seed: 42
    strategy: structure_first_v3
    strategy_overrides: { ... }
    guards:
        enabled: [adx, usdtd, funding_long, whale_neutral, ext_pos_news]
        params: { adx: { min_adx: 20 }, ... }
    sizing: { type: fixed_notional, usd: 3000, max_concurrent: 4 }
    exits: { partial_tp: [[1.0, 0.40], [2.0, 0.35]], ... }
    fees: { taker: 0.0004, maker: 0.0002, slippage_bps: 5 }
    output: { dir: runs/${run_id}, formats: [parquet, json, csv] }
    sweep:
        mode: grid
        axes:
          - { path: guards.params.usdtd.threshold,
              values: [0.50, ..., 1.00] }
        parallel: 1

Validation is intentionally loose (dict-based) for Phase 1 — easier to
extend than pydantic when the schema is still mutating. Promote to
pydantic in Phase 2.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import live_constants as LC


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class WindowSpec:
    start: date
    end: date


@dataclass
class IntervalSpec:
    primary: str = LC.PRIMARY_INTERVAL
    htf: List[str] = field(default_factory=lambda: list(LC.HTF_INTERVALS))


@dataclass
class GuardsSpec:
    enabled: List[str] = field(default_factory=list)
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SizingSpec:
    type: str = "fixed_notional"
    usd: float = float(LC.FIXED_MAX_NOTIONAL)
    max_concurrent: int = 4
    risk_pool_pct: float = LC.RISK_POOL_PCT


@dataclass
class ExitsSpec:
    partial_tp: List[List[float]] = field(default_factory=lambda: [
        [LC.PARTIAL_TP1_R, LC.PARTIAL_TP1_FRACTION],
        [LC.PARTIAL_TP2_R, LC.PARTIAL_TP2_FRACTION],
    ])
    atr_floor: Dict[str, float] = field(default_factory=lambda: {
        "sl_mult": LC.ATR_SL_FLOOR_MULT, "tp_mult": LC.ATR_TP_FLOOR_MULT,
    })
    trailing: Dict[str, float] = field(default_factory=lambda: {
        "activation_pct": LC.TRAILING_ACTIVATE_PCT,
        "distance_pre_tp1": LC.TRAILING_DISTANCE_PRE_TP1,
        "distance_post_tp1": LC.TRAILING_DISTANCE_POST_TP1,
    })
    stagnant_hours: float = LC.STAGNANT_HOURS
    stagnant_pct_min: float = LC.STAGNANT_PCT_MIN
    stagnant_pct_max: float = LC.STAGNANT_PCT_MAX


@dataclass
class FeesSpec:
    taker: float = LC.TRADING_FEE_TAKER
    maker: float = LC.TRADING_FEE_MAKER
    slippage_pct: float = LC.SLIPPAGE_PCT


@dataclass
class OutputSpec:
    dir: str = "runs/${run_id}"
    formats: List[str] = field(default_factory=lambda: ["parquet", "json", "csv"])
    rollups: List[str] = field(default_factory=lambda: [
        "per_symbol", "per_side", "per_month", "per_guard_blocked",
    ])


@dataclass
class SweepAxis:
    path: str               # dotted path, e.g. "guards.params.usdtd.threshold"
    values: List[Any]


@dataclass
class SweepSpec:
    mode: str = "single"    # single | grid | random | walkforward
    axes: List[SweepAxis] = field(default_factory=list)
    parallel: int = 1
    walkforward: Optional[Dict[str, int]] = None  # {"train_days": 60, "test_days": 14, "step_days": 14}


@dataclass
class BacktestConfig:
    run_id: str
    window: WindowSpec
    symbols: List[str]
    intervals: IntervalSpec
    strategy: str
    seed: int = 42
    strategy_overrides: Dict[str, Any] = field(default_factory=dict)
    guards: GuardsSpec = field(default_factory=GuardsSpec)
    sizing: SizingSpec = field(default_factory=SizingSpec)
    exits: ExitsSpec = field(default_factory=ExitsSpec)
    fees: FeesSpec = field(default_factory=FeesSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    sweep: SweepSpec = field(default_factory=SweepSpec)
    raw: Dict[str, Any] = field(default_factory=dict)  # pristine source

    # ── construction ────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "BacktestConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfig":
        raw = copy.deepcopy(data)
        # Required fields
        for req in ("run_id", "window", "symbols", "strategy"):
            if req not in data:
                raise ValueError(f"Config missing required field: {req!r}")
        win = data["window"]
        window = WindowSpec(
            start=_to_date(win["start"]), end=_to_date(win["end"]),
        )
        if window.end <= window.start:
            raise ValueError(f"window.end ({window.end}) must be after start ({window.start})")
        intervals = IntervalSpec(**data.get("intervals", {}))
        guards_raw = data.get("guards", {})
        guards = GuardsSpec(
            enabled=list(guards_raw.get("enabled", [])),
            params=dict(guards_raw.get("params", {})),
        )
        sizing = SizingSpec(**data.get("sizing", {}))
        exits = ExitsSpec(**data.get("exits", {}))
        fees = FeesSpec(**data.get("fees", {}))
        output = OutputSpec(**data.get("output", {}))
        sweep_raw = data.get("sweep", {}) or {}
        axes = [SweepAxis(**a) for a in sweep_raw.get("axes", [])]
        sweep = SweepSpec(
            mode=sweep_raw.get("mode", "single"),
            axes=axes,
            parallel=int(sweep_raw.get("parallel", 1)),
            walkforward=sweep_raw.get("walkforward"),
        )
        return cls(
            run_id=str(data["run_id"]),
            window=window,
            symbols=list(data["symbols"]),
            intervals=intervals,
            seed=int(data.get("seed", 42)),
            strategy=str(data["strategy"]),
            strategy_overrides=dict(data.get("strategy_overrides", {})),
            guards=guards,
            sizing=sizing,
            exits=exits,
            fees=fees,
            output=output,
            sweep=sweep,
            raw=raw,
        )

    # ── helpers ────────────────────────────────────────────────────
    def resolve_output_dir(self) -> Path:
        s = self.output.dir.replace("${run_id}", self.run_id)
        p = Path(s)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p

    def validate(self) -> List[str]:
        """Return list of (non-fatal) warnings about config plausibility."""
        warnings = []
        if not self.symbols:
            raise ValueError("config.symbols is empty")
        valid_intervals = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}
        if self.intervals.primary not in valid_intervals:
            raise ValueError(f"primary interval {self.intervals.primary!r} not supported")
        for h in self.intervals.htf:
            if h not in valid_intervals:
                raise ValueError(f"htf interval {h!r} not supported")
        # Stale-data warning: window includes future
        from datetime import date as _date
        if self.window.end > _date.today():
            warnings.append(
                f"window.end {self.window.end} is in the future — backtest will use up to today only"
            )
        if self.window.start > _date.today():
            warnings.append(
                f"window.start {self.window.start} is in the future — no data to test"
            )
        # Sweep sanity
        if self.sweep.mode == "grid" and not self.sweep.axes:
            raise ValueError("sweep.mode=grid requires non-empty sweep.axes")
        return warnings


# ── small utilities ────────────────────────────────────────────────────
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _to_date(v) -> date:
    if isinstance(v, date):
        return v
    if isinstance(v, str) and _DATE_RE.match(v):
        return date.fromisoformat(v)
    raise ValueError(f"Cannot parse date {v!r} (expected YYYY-MM-DD)")
