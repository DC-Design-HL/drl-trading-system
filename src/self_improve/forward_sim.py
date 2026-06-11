"""Forward simulator for the self-improvement loop (PROFITABILITY_PLAN.md P2).

Walks 5m candles bar-by-bar from the kline cache, reproducing the live
bot's entry path (BOS/CHOCH MarketStructure + structure-first gates +
P1 STRUCT_* floors + blocklist) and the live bot's exit stack (SL/TP,
partial TP1/TP2, trailing, stagnant). Costs include taker fees, funding
accrued at each 8h boundary, and a configurable slippage parameter.

What this gives the autonomous loop that the replay harness does not:

  * Validation of looser thresholds — STRUCT_* floors that would have
    let MORE trades through, not just blocked ones. Replay can only
    drop historical trades; the sim can also invent the ones the live
    bot did not take but a candidate config would.
  * Exit-side tuning — STAGNANT_HOURS, TRAILING_*, SL/TP multipliers.
    Replay cannot re-time exits because it has no per-bar price path
    between OPEN and CLOSE.

Scope of this commit (chunks B + entry-only):

  * Entry replication of `_get_structure_direction` for S1 symbols.
    ETH (S5) is approximated — the OB-proximity and ADX-directional
    filters are still skipped here; the calibration report flags this.
  * Costs / exits arrive in chunks C-E.

Determinism: same kline-cache + same config → identical output. No
network in the sim path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.signals.bos_choch import MarketStructure

from .kline_cache import as_ohlcv_df, read_klines

logger = logging.getLogger(__name__)


# Default lookback windows — match live_trading_htf.FETCH_DAYS=12.
DEFAULT_LOOKBACK_DAYS = 12
DEFAULT_5M_BARS = DEFAULT_LOOKBACK_DAYS * 24 * 12     # 12d × 288 bars/day
DEFAULT_1H_BARS = DEFAULT_LOOKBACK_DAYS * 24
DEFAULT_4H_BARS = DEFAULT_LOOKBACK_DAYS * 6

# Live bot iterates every 15 minutes — match its cadence.
DEFAULT_DECISION_INTERVAL_MIN = 15

# Per-symbol filter config (mirrors live STRUCTURE_SYMBOL_CONFIG). S5
# symbols carry extra filters that this v1 does not replicate fully —
# the calibration report quantifies the resulting drift for those.
STRUCTURE_SYMBOL_CONFIG = {
    "BTCUSDT": "S1",
    "ETHUSDT": "S5",
    "SOLUSDT": "S1",
    "XRPUSDT": "S1",
}

# Baseline blocklist (mirrors live_trading_htf SYMBOL_SIDE_BLOCKLIST at
# this commit). Tests + the runner can override.
DEFAULT_BLOCKLIST: frozenset[tuple[str, str]] = frozenset({
    ("SOLUSDT", "LONG"),
    ("XRPUSDT", "LONG"),
    ("XRPUSDT", "SHORT"),
})


# ─── Config & output dataclasses ───────────────────────────────────────


@dataclass(frozen=True)
class ForwardSimConfig:
    """Tunable knobs the simulator honours.

    These mirror the live constants that the autonomous loop is allowed
    to write to in P3. Each one defaults to the live baseline so an
    out-of-the-box run reproduces actual bot behaviour as closely as the
    sim can.
    """

    # STRUCT_* family — applied via the same _resolve_struct_floor
    # precedence (directional → per-symbol → global).
    struct_min_confidence: float = 0.0
    struct_symbol_min_confidence: dict[str, float] = field(default_factory=dict)
    struct_symbol_directional_conf: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    blocklist: frozenset[tuple[str, str]] = DEFAULT_BLOCKLIST
    # Knobs that arrive in chunks C-E; kept here so the dataclass is the
    # single source of truth from the start.
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.030
    trailing_distance_pct: float = 0.003
    stagnant_hours: float = 6.0
    taker_fee_pct: float = 0.0004
    slippage_bps: float = 1.0
    capital_base: float = 5000.0


@dataclass
class EntryEvent:
    """One simulated would-have-entered event."""

    ts: pd.Timestamp
    symbol: str
    side: str                   # 'LONG' | 'SHORT'
    confidence: float           # structure confidence at decision time
    price: float                # close of the 5m bar that triggered
    trend: str
    last_signal_direction: str


@dataclass
class SymbolForwardResult:
    """Per-symbol output of one forward-sim run."""

    symbol: str
    n_decisions: int            # iterations evaluated
    entries: list[EntryEvent]
    skipped_by_blocklist: int = 0
    skipped_by_struct_floor: int = 0
    skipped_by_trend: int = 0
    skipped_by_s5_unimplemented: int = 0
    runtime_seconds: float = 0.0


@dataclass
class ForwardSimResult:
    """Top-level result; same shape skeleton as BacktestResult."""

    config: ForwardSimConfig
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    per_symbol: dict[str, SymbolForwardResult]
    mode: str = "forward"
    label: str = ""
    git_head: str = ""
    runtime_seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "git_head": self.git_head,
            "start_ts": self.start_ts.isoformat(),
            "end_ts": self.end_ts.isoformat(),
            "runtime_seconds": self.runtime_seconds,
            "config": _config_to_json(self.config),
            "per_symbol": {
                sym: {
                    "symbol": r.symbol,
                    "n_decisions": r.n_decisions,
                    "n_entries": len(r.entries),
                    "skipped_by_blocklist": r.skipped_by_blocklist,
                    "skipped_by_struct_floor": r.skipped_by_struct_floor,
                    "skipped_by_trend": r.skipped_by_trend,
                    "skipped_by_s5_unimplemented": r.skipped_by_s5_unimplemented,
                    "runtime_seconds": r.runtime_seconds,
                    "entries": [
                        {
                            "ts": e.ts.isoformat(),
                            "side": e.side,
                            "confidence": e.confidence,
                            "price": e.price,
                            "trend": e.trend,
                            "last_signal_direction": e.last_signal_direction,
                        }
                        for e in r.entries
                    ],
                }
                for sym, r in self.per_symbol.items()
            },
        }


def _config_to_json(c: ForwardSimConfig) -> dict[str, Any]:
    d = asdict(c)
    d["blocklist"] = sorted([list(t) for t in c.blocklist])
    return d


def _utc_ts(value: datetime) -> pd.Timestamp:
    """Coerce a datetime to a tz-aware UTC pd.Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ─── Pure helpers (mirror live precedence; importable from tests) ──────


def resolve_struct_floor(
    cfg: ForwardSimConfig, symbol: str, side: str,
) -> tuple[Optional[float], str]:
    """Same precedence as live's _resolve_struct_floor.

    directional → per-symbol → global; 0.0/{} means no floor.
    Kept independent of live_trading_htf so the sim has no torch import.
    """
    sym_u, side_u = symbol.upper(), side.upper()
    dir_cfg = cfg.struct_symbol_directional_conf.get(sym_u) or {}
    if side_u in dir_cfg:
        try:
            v = float(dir_cfg[side_u])
            if v > 0.0:
                return v, f"STRUCT_SYMBOL_DIRECTIONAL_CONF[{sym_u}][{side_u}]"
        except (TypeError, ValueError):
            pass
    if sym_u in cfg.struct_symbol_min_confidence:
        try:
            v = float(cfg.struct_symbol_min_confidence[sym_u])
            if v > 0.0:
                return v, f"STRUCT_SYMBOL_MIN_CONFIDENCE[{sym_u}]"
        except (TypeError, ValueError):
            pass
    try:
        v = float(cfg.struct_min_confidence)
        if v > 0.0:
            return v, "STRUCT_MIN_CONFIDENCE"
    except (TypeError, ValueError):
        pass
    return None, ""


def derive_direction(
    sig: dict[str, Any],
) -> tuple[Optional[str], str]:
    """Replicate _get_structure_direction's trend+last_signal gate.

    Returns (side, reason). side is 'LONG' / 'SHORT' / None;
    reason is a short marker for accounting.
    """
    trend = sig.get("trend", "ranging")
    last_dir = sig.get("last_signal_direction", "none")
    if trend == "bullish" and last_dir == "bullish":
        return "LONG", "trend+last_signal_bull"
    if trend == "bearish" and last_dir == "bearish":
        return "SHORT", "trend+last_signal_bear"
    if trend in ("bullish", "bearish") and last_dir != trend:
        return None, "trend_last_signal_disagree"
    return None, "trend_ranging_or_no_signal"


# ─── Main entry: run_forward_sim ───────────────────────────────────────


def run_forward_sim(
    *,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"),
    start: datetime,
    end: datetime,
    config: Optional[ForwardSimConfig] = None,
    decision_interval_min: int = DEFAULT_DECISION_INTERVAL_MIN,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    cache_base: Optional[Path] = None,
    label: str = "",
) -> ForwardSimResult:
    """Run a deterministic forward simulation for the given window."""
    cfg = config or ForwardSimConfig()
    started = time.perf_counter()

    if cache_base is None:
        from .kline_cache import CACHE_DIR
        cache_base = CACHE_DIR

    # We need an extra lookback worth of data before `start` so the
    # earliest decision still has a full 12-day MarketStructure window.
    pad_ns = int(lookback_days * 86_400 * 1_000_000_000)
    fetch_start_ns = int(start.timestamp() * 1_000_000_000) - pad_ns
    end_ns = int(end.timestamp() * 1_000_000_000)

    ms = MarketStructure(swing_lookback=8)
    per_symbol: dict[str, SymbolForwardResult] = {}

    for symbol in symbols:
        sym_started = time.perf_counter()
        df_5m_full = as_ohlcv_df(read_klines(
            symbol, "5m", start=fetch_start_ns, end=end_ns, base=cache_base,
        ))
        df_1h_full = as_ohlcv_df(read_klines(
            symbol, "1h", start=fetch_start_ns, end=end_ns, base=cache_base,
        ))
        df_4h_full = as_ohlcv_df(read_klines(
            symbol, "4h", start=fetch_start_ns, end=end_ns, base=cache_base,
        ))

        if df_5m_full.empty:
            logger.warning("forward_sim: no 5m data for %s — skipping", symbol)
            per_symbol[symbol] = SymbolForwardResult(
                symbol=symbol, n_decisions=0, entries=[],
            )
            continue

        result = _simulate_symbol(
            symbol=symbol,
            df_5m=df_5m_full,
            df_1h=df_1h_full,
            df_4h=df_4h_full,
            ms=ms,
            cfg=cfg,
            start_ts=_utc_ts(start),
            end_ts=_utc_ts(end),
            decision_interval_min=decision_interval_min,
        )
        result.runtime_seconds = time.perf_counter() - sym_started
        per_symbol[symbol] = result

    runtime = time.perf_counter() - started
    return ForwardSimResult(
        config=cfg,
        start_ts=_utc_ts(start),
        end_ts=_utc_ts(end),
        per_symbol=per_symbol,
        label=label,
        runtime_seconds=runtime,
    )


def _simulate_symbol(
    *,
    symbol: str,
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    ms: MarketStructure,
    cfg: ForwardSimConfig,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    decision_interval_min: int,
) -> SymbolForwardResult:
    """Walk one symbol's 5m bars and emit entry events.

    Decisions are evaluated every `decision_interval_min` minutes — by
    default 15, matching the live bot's loop. At each decision bar, the
    MarketStructure window is the previous 12 days of bars (5m / 1h /
    4h), exactly the slice the live bot would have had.
    """
    sym_cfg = STRUCTURE_SYMBOL_CONFIG.get(symbol, "S1")
    entries: list[EntryEvent] = []
    n_decisions = 0
    n_blocklist = 0
    n_struct = 0
    n_trend = 0
    n_s5 = 0

    # Restrict the 5m frame to the decision window and snap to the cadence.
    decisions_5m = df_5m[(df_5m.index >= start_ts) & (df_5m.index <= end_ts)]
    if decisions_5m.empty:
        return SymbolForwardResult(
            symbol=symbol, n_decisions=0, entries=[],
        )

    step = decision_interval_min // 5  # 5m bars between decisions
    if step < 1:
        step = 1

    # Pre-cache 1h/4h frame slicing — pandas index lookups dominate cost
    # otherwise.
    idx_5m_full = df_5m.index
    idx_1h_full = df_1h.index
    idx_4h_full = df_4h.index

    for i in range(0, len(decisions_5m), step):
        bar_ts = decisions_5m.index[i]
        # Lookback windows up to and including bar_ts.
        pos_5m = idx_5m_full.searchsorted(bar_ts, side="right")
        pos_1h = idx_1h_full.searchsorted(bar_ts, side="right")
        pos_4h = idx_4h_full.searchsorted(bar_ts, side="right")

        win_5m = df_5m.iloc[max(0, pos_5m - DEFAULT_5M_BARS):pos_5m]
        win_1h = df_1h.iloc[max(0, pos_1h - DEFAULT_1H_BARS):pos_1h]
        win_4h = df_4h.iloc[max(0, pos_4h - DEFAULT_4H_BARS):pos_4h]
        if len(win_5m) < 100:
            # Not enough data this early into the cache — skip.
            continue

        n_decisions += 1

        sig = ms.get_signals(win_5m, df_1h=win_1h, df_4h=win_4h)
        side, reason = derive_direction(sig)
        if side is None:
            n_trend += 1
            continue

        # S5 extras are NOT replicated in v1 — flag and skip if symbol
        # configured for S5. ETH is the only one today.
        if sym_cfg == "S5":
            n_s5 += 1
            continue

        if (symbol, side) in cfg.blocklist:
            n_blocklist += 1
            continue

        floor, _label = resolve_struct_floor(cfg, symbol, side)
        struct_conf = float(sig.get("confidence", 0.0) or 0.0)
        if floor is not None and struct_conf < floor:
            n_struct += 1
            continue

        price = float(decisions_5m["close"].iloc[i])
        entries.append(EntryEvent(
            ts=bar_ts,
            symbol=symbol,
            side=side,
            confidence=struct_conf,
            price=price,
            trend=sig.get("trend", ""),
            last_signal_direction=sig.get("last_signal_direction", ""),
        ))

    return SymbolForwardResult(
        symbol=symbol,
        n_decisions=n_decisions,
        entries=entries,
        skipped_by_blocklist=n_blocklist,
        skipped_by_struct_floor=n_struct,
        skipped_by_trend=n_trend,
        skipped_by_s5_unimplemented=n_s5,
    )
