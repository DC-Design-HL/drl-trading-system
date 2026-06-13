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

import numpy as np
import pandas as pd

from src.features.regime_detector import MarketRegime, MarketRegimeDetector
from src.signals.bos_choch import MarketStructure
from src.signals.structure_filters import (
    passes_adx_directional,
    passes_exhaustion_filter,
    passes_ob_proximity,
    passes_rsi_guard,
)

from .kline_cache import as_ohlcv_df, read_funding, read_klines

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
    # S5 filter constants (mirror live_trading_htf).
    structure_ob_proximity_pct: float = 0.010
    adx_guard_min: float = 20.0
    # Pre-trade guards (mirror live).
    exhaustion_atr_threshold: float = 3.0
    rsi_guard_ob_threshold: float = 70.0
    rsi_guard_os_threshold: float = 30.0
    # Trend-aware RSI bands (mirror live RSI_GUARD_*_TREND): in a confirmed
    # strong trend the live bot loosens the RSI ceiling/floor so it stops
    # fighting the regime. Applied via the live MarketRegimeDetector below.
    rsi_guard_ob_trend: float = 80.0
    rsi_guard_os_trend: float = 20.0
    rsi_guard_trend_adx_min: float = 25.0
    # Stateful post-close time gates (mirror live_trading_htf
    # COOLDOWN_SECONDS / WHIPSAW_COOLDOWN_HOURS). After a LOSING close the
    # live bot blocks ALL entries for cooldown_seconds and any OPPOSITE-side
    # (whipsaw) entry for whipsaw_cooldown_hours. The sim only ever decides
    # while flat (it holds one position to exit), so live's in-position
    # min-hold guard never applies here and is intentionally not modelled.
    cooldown_seconds: float = 1800.0
    whipsaw_cooldown_hours: float = 2.0
    # Exit + sizing constants — mirror live_trading_htf defaults.
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.030
    trailing_breakeven_pct: float = 0.008
    trailing_distance_pct: float = 0.005
    trailing_distance_post_tp1: float = 0.008
    tp1_r_multiple: float = 1.0
    tp2_r_multiple: float = 2.0
    tp1_fraction: float = 0.40
    tp2_fraction: float = 0.35
    stagnant_hours: float = 6.0
    stagnant_pct_min: float = -0.010
    stagnant_pct_max: float = 0.005
    # Costs (chunks C/E).
    taker_fee_pct: float = 0.0004
    slippage_bps: float = 1.0
    capital_base: float = 5000.0
    # Fixed-dollar-risk sizing (mirrors live live_trading_htf.RISK_POOL_PCT /
    # RISK_BUDGET_PARTS). With capital 5000 → risk_pool 500 → per-trade
    # risk 25, notional = 25 / 0.015 = $1,666 (capped at FIXED_MAX_NOTIONAL).
    risk_pool_pct: float = 0.10
    risk_budget_parts: int = 20
    fixed_max_notional: float = 3000.0
    # Max time to hold a position before forcing exit (safety net for the
    # sim — live has no hard timeout, but a single trade should not run
    # forever in the sim if for some reason no exit triggers).
    max_hold_hours: float = 168.0


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
class TradeResult:
    """Outcome of one simulated trade — entry + exits."""

    symbol: str
    side: str                   # 'LONG' | 'SHORT'
    confidence: float
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: pd.Timestamp
    exit_price: float
    notional_at_entry: float
    realized_pnl_usd: float     # NET of fees + funding + slippage
    gross_pnl_usd: float        # before costs (for attribution)
    fees_usd: float
    funding_usd: float
    slippage_usd: float
    close_reason: str           # 'SL' | 'TP' | 'TRAIL' | 'STAGNANT' | 'MAX_HOLD' | 'EOD'
    partial_tp_hits: int        # 0, 1, or 2
    mfe_pct: float              # best favourable price during the trade
    mae_pct: float              # worst adverse price during the trade
    hold_bars_5m: int


@dataclass
class SymbolForwardResult:
    """Per-symbol output of one forward-sim run."""

    symbol: str
    n_decisions: int            # iterations evaluated
    entries: list[EntryEvent]
    trades: list[TradeResult] = field(default_factory=list)
    skipped_by_blocklist: int = 0
    skipped_by_struct_floor: int = 0
    skipped_by_trend: int = 0
    skipped_by_s5_unimplemented: int = 0
    skipped_by_struct_first_adx: int = 0
    skipped_by_exhaustion: int = 0
    skipped_by_rsi: int = 0
    skipped_by_cooldown: int = 0        # post-loss cooldown (P2.D timing)
    skipped_by_whipsaw: int = 0         # anti-whipsaw reversal (P2.D timing)
    skipped_by_open_position: int = 0   # already in a trade — live can't double up
    runtime_seconds: float = 0.0

    @property
    def net_pnl_usd(self) -> float:
        return float(sum(t.realized_pnl_usd for t in self.trades))


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
                    "n_trades": len(r.trades),
                    "net_pnl_usd": r.net_pnl_usd,
                    "skipped_by_blocklist": r.skipped_by_blocklist,
                    "skipped_by_struct_floor": r.skipped_by_struct_floor,
                    "skipped_by_trend": r.skipped_by_trend,
                    "skipped_by_s5_unimplemented": r.skipped_by_s5_unimplemented,
                    "skipped_by_struct_first_adx": r.skipped_by_struct_first_adx,
                    "skipped_by_exhaustion": r.skipped_by_exhaustion,
                    "skipped_by_rsi": r.skipped_by_rsi,
                    "skipped_by_cooldown": r.skipped_by_cooldown,
                    "skipped_by_whipsaw": r.skipped_by_whipsaw,
                    "skipped_by_open_position": r.skipped_by_open_position,
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
                    "trades": [
                        {
                            "side": t.side,
                            "entry_ts": t.entry_ts.isoformat(),
                            "exit_ts": t.exit_ts.isoformat(),
                            "entry_price": t.entry_price,
                            "exit_price": t.exit_price,
                            "confidence": t.confidence,
                            "notional_at_entry": t.notional_at_entry,
                            "realized_pnl_usd": t.realized_pnl_usd,
                            "gross_pnl_usd": t.gross_pnl_usd,
                            "fees_usd": t.fees_usd,
                            "funding_usd": t.funding_usd,
                            "slippage_usd": t.slippage_usd,
                            "close_reason": t.close_reason,
                            "partial_tp_hits": t.partial_tp_hits,
                            "mfe_pct": t.mfe_pct,
                            "mae_pct": t.mae_pct,
                            "hold_bars_5m": t.hold_bars_5m,
                        }
                        for t in r.trades
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


# ─── Position-exit simulation (bar-by-bar) ────────────────────────────


def _compute_levels(entry: float, side: str,
                    cfg: ForwardSimConfig) -> dict[str, float]:
    """Initial SL / TP / partial-TP price levels for a given entry."""
    if side == "LONG":
        sl = entry * (1.0 - cfg.stop_loss_pct)
        tp = entry * (1.0 + cfg.take_profit_pct)
        tp1 = entry * (1.0 + cfg.tp1_r_multiple * cfg.stop_loss_pct)
        tp2 = entry * (1.0 + cfg.tp2_r_multiple * cfg.stop_loss_pct)
    else:
        sl = entry * (1.0 + cfg.stop_loss_pct)
        tp = entry * (1.0 - cfg.take_profit_pct)
        tp1 = entry * (1.0 - cfg.tp1_r_multiple * cfg.stop_loss_pct)
        tp2 = entry * (1.0 - cfg.tp2_r_multiple * cfg.stop_loss_pct)
    return {"sl": sl, "tp": tp, "tp1": tp1, "tp2": tp2}


def _compute_notional(cfg: ForwardSimConfig) -> float:
    """Fixed-dollar-risk sizing — same shape as live _open_position.

    notional = (capital × pool% / parts) / SL%, capped at FIXED_MAX_NOTIONAL.
    The sim does not compound: every trade uses the same capital_base,
    so PnL is comparable across windows.
    """
    risk_pool = cfg.capital_base * cfg.risk_pool_pct
    dollar_risk = risk_pool / cfg.risk_budget_parts
    notional = dollar_risk / cfg.stop_loss_pct
    return min(notional, cfg.fixed_max_notional)


def _funding_cost(
    notional: float,
    side: str,
    entry_ns: int,
    exit_ns: int,
    funding_ts: Optional[np.ndarray],
    funding_rates: Optional[np.ndarray],
) -> float:
    """Funding PAID by the position over (entry_ns, exit_ns], accrued at
    each 8h funding timestamp the position is held through (P2.E).

    Convention: a LONG PAYS funding when the rate is positive, a SHORT
    RECEIVES it — so the value returned is what the position paid (added
    as a cost, i.e. subtracted from PnL); it is negative when the position
    was a net receiver. Accrued on the ENTRY notional (a documented
    simplification — does not shrink the notional after partial TPs; the
    per-8h amounts are tiny so the error is sub-cent per leg).
    ``funding_ts`` must be ascending nanosecond timestamps.
    """
    if funding_ts is None or len(funding_ts) == 0:
        return 0.0
    lo = int(np.searchsorted(funding_ts, entry_ns, side="right"))
    hi = int(np.searchsorted(funding_ts, exit_ns, side="right"))
    if hi <= lo:
        return 0.0
    rate_sum = float(funding_rates[lo:hi].sum())
    return notional * rate_sum * (1.0 if side == "LONG" else -1.0)


def simulate_position(
    *,
    symbol: str,
    side: str,
    entry_ts: pd.Timestamp,
    entry_price: float,
    confidence: float,
    df_5m: pd.DataFrame,
    cfg: ForwardSimConfig,
    funding_ts: Optional[np.ndarray] = None,
    funding_rates: Optional[np.ndarray] = None,
) -> Optional[TradeResult]:
    """Walk 5m bars forward from entry_ts and apply the live exit stack.

    Exit precedence per bar (mirroring live _check_sl_tp + _manage_position):
      1. Hard SL hit                    → close, reason="SL"
      2. Hard TP hit                    → close, reason="TP"
      3. Partial TP1 hit (first time)   → close `tp1_fraction`, move SL to BE
      4. Partial TP2 hit (after TP1)    → close `tp2_fraction`
      5. Trailing SL (post-breakeven)   → may tighten SL toward peak
      6. Stagnant exit                  → close after stagnant_hours in band
      7. Max hold                       → safety net

    Intrabar rule: if both SL and TP are inside the bar's range, SL fires
    first (conservative). Same convention spec'd in PROFITABILITY_PLAN.md.
    """
    fwd = df_5m[df_5m.index > entry_ts]
    if fwd.empty:
        return None

    levels = _compute_levels(entry_price, side, cfg)
    sl = levels["sl"]
    tp = levels["tp"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    notional = _compute_notional(cfg)
    initial_units = notional / max(entry_price, 1e-9)
    remaining_units = initial_units
    realized_gross = 0.0
    partial_hits = 0
    peak = entry_price  # for trailing
    trough = entry_price
    mfe_pct = 0.0
    mae_pct = 0.0
    stagnant_start: Optional[pd.Timestamp] = None
    max_hold = pd.Timedelta(hours=cfg.max_hold_hours)
    stagnant_window = pd.Timedelta(hours=cfg.stagnant_hours)

    hold_bars = 0
    close_reason: Optional[str] = None
    close_price = entry_price
    close_ts = entry_ts

    for ts, row in fwd.iterrows():
        hold_bars += 1
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # Update MFE / MAE — instantaneous best / worst since entry.
        if side == "LONG":
            if high > peak:
                peak = high
            if low < trough:
                trough = low
            bar_mfe = (high - entry_price) / entry_price
            bar_mae = (low - entry_price) / entry_price
        else:
            if low < peak:
                peak = low
            if high > trough:
                trough = high
            bar_mfe = (entry_price - low) / entry_price
            bar_mae = (entry_price - high) / entry_price
        if bar_mfe > mfe_pct:
            mfe_pct = bar_mfe
        if bar_mae < mae_pct:
            mae_pct = bar_mae

        # 1. SL — checked before TP per intrabar conservatism.
        if side == "LONG" and low <= sl:
            realized_gross += (sl - entry_price) * remaining_units
            close_reason = "SL"
            close_price = sl
            close_ts = ts
            break
        if side == "SHORT" and high >= sl:
            realized_gross += (entry_price - sl) * remaining_units
            close_reason = "SL"
            close_price = sl
            close_ts = ts
            break

        # 2. Hard TP.
        if side == "LONG" and high >= tp:
            realized_gross += (tp - entry_price) * remaining_units
            close_reason = "TP"
            close_price = tp
            close_ts = ts
            break
        if side == "SHORT" and low <= tp:
            realized_gross += (entry_price - tp) * remaining_units
            close_reason = "TP"
            close_price = tp
            close_ts = ts
            break

        # 3. Partial TP1 (first hit).
        if partial_hits == 0:
            hit = (side == "LONG" and high >= tp1) or (
                side == "SHORT" and low <= tp1
            )
            if hit:
                close_units = initial_units * cfg.tp1_fraction
                partial_pnl = (
                    (tp1 - entry_price) if side == "LONG"
                    else (entry_price - tp1)
                ) * close_units
                realized_gross += partial_pnl
                remaining_units -= close_units
                partial_hits = 1
                # Move SL to breakeven.
                sl = entry_price

        # 4. Partial TP2 (after TP1).
        if partial_hits == 1:
            hit = (side == "LONG" and high >= tp2) or (
                side == "SHORT" and low <= tp2
            )
            if hit:
                close_units = initial_units * cfg.tp2_fraction
                partial_pnl = (
                    (tp2 - entry_price) if side == "LONG"
                    else (entry_price - tp2)
                ) * close_units
                realized_gross += partial_pnl
                remaining_units -= close_units
                partial_hits = 2

        # 5. Trailing SL — activates above breakeven_pct.
        if side == "LONG":
            profit_pct = (close - entry_price) / entry_price
        else:
            profit_pct = (entry_price - close) / entry_price
        if profit_pct >= cfg.trailing_breakeven_pct:
            trail_dist = (
                cfg.trailing_distance_post_tp1
                if partial_hits >= 1
                else cfg.trailing_distance_pct
            )
            if side == "LONG":
                trailing_sl = max(peak * (1.0 - trail_dist), entry_price)
                if trailing_sl > sl:
                    sl = trailing_sl
            else:
                trailing_sl = min(peak * (1.0 + trail_dist), entry_price)
                if sl <= 0 or trailing_sl < sl:
                    sl = trailing_sl

        # 6. Stagnant exit — track running window of PnL within band.
        in_band = cfg.stagnant_pct_min <= profit_pct <= cfg.stagnant_pct_max
        if in_band:
            if stagnant_start is None:
                stagnant_start = ts
            elif ts - stagnant_start >= stagnant_window:
                if remaining_units > 0:
                    realized_gross += (
                        (close - entry_price) if side == "LONG"
                        else (entry_price - close)
                    ) * remaining_units
                close_reason = "STAGNANT"
                close_price = close
                close_ts = ts
                break
        else:
            stagnant_start = None

        # 7. Max hold safety net.
        if ts - entry_ts >= max_hold:
            if remaining_units > 0:
                realized_gross += (
                    (close - entry_price) if side == "LONG"
                    else (entry_price - close)
                ) * remaining_units
            close_reason = "MAX_HOLD"
            close_price = close
            close_ts = ts
            break

    if close_reason is None:
        # Data ran out — close at last bar's close (end-of-data).
        last_ts = fwd.index[-1]
        last_close = float(fwd["close"].iloc[-1])
        if remaining_units > 0:
            realized_gross += (
                (last_close - entry_price) if side == "LONG"
                else (entry_price - last_close)
            ) * remaining_units
        close_reason = "EOD"
        close_price = last_close
        close_ts = last_ts

    # Costs: fees on entry + on each closed leg (approximation: sum of
    # closed notionals × fee_pct each side). Slippage as 1 bp of entry +
    # exit notional. Funding accrues per 8h boundary crossed (P2.E).
    entry_fee = notional * cfg.taker_fee_pct
    exit_notional = abs(close_price) * initial_units
    exit_fee = exit_notional * cfg.taker_fee_pct
    fees = entry_fee + exit_fee
    slippage = (notional + exit_notional) * (cfg.slippage_bps / 10000.0)
    funding = _funding_cost(
        notional, side, entry_ts.value, close_ts.value,
        funding_ts, funding_rates,
    )
    realized_net = realized_gross - fees - slippage - funding

    return TradeResult(
        symbol=symbol,
        side=side,
        confidence=confidence,
        entry_ts=entry_ts,
        entry_price=entry_price,
        exit_ts=close_ts,
        exit_price=close_price,
        notional_at_entry=notional,
        realized_pnl_usd=realized_net,
        gross_pnl_usd=realized_gross,
        fees_usd=fees,
        funding_usd=funding,
        slippage_usd=slippage,
        close_reason=close_reason,
        partial_tp_hits=partial_hits,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        hold_bars_5m=hold_bars,
    )


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
    trace: Optional[list] = None,
) -> ForwardSimResult:
    """Run a deterministic forward simulation for the given window.

    If ``trace`` is a list, every evaluated decision appends a record
    ``{"symbol", "ts", "side", "outcome"}`` where ``outcome`` is "entry"
    or the name of the gate that blocked it. Diagnostic-only; does not
    affect the simulation result.
    """
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
    # Same detector the live bot uses (default params) so the sim's
    # structure-first ADX and trend-aware RSI bands match live exactly,
    # instead of the earlier simplified kline approximations.
    regime_detector = MarketRegimeDetector()
    per_symbol: dict[str, SymbolForwardResult] = {}

    for symbol in symbols:
        sym_started = time.perf_counter()
        df_5m_full = as_ohlcv_df(read_klines(
            symbol, "5m", start=fetch_start_ns, end=end_ns, base=cache_base,
        ))
        df_15m_full = as_ohlcv_df(read_klines(
            symbol, "15m", start=fetch_start_ns, end=end_ns, base=cache_base,
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

        # Funding cache (P2.E). Empty when the symbol has no funding parquet
        # (e.g. the synthetic test cache) → funding accrues to 0.
        fund = read_funding(
            symbol, start=fetch_start_ns, end=end_ns, base=cache_base)
        funding_ts = (
            fund["ts"].to_numpy(dtype="int64") if not fund.empty else None)
        funding_rates = (
            fund["funding_rate"].to_numpy(dtype="float64")
            if not fund.empty else None)

        result = _simulate_symbol(
            symbol=symbol,
            df_5m=df_5m_full,
            df_15m=df_15m_full,
            df_1h=df_1h_full,
            df_4h=df_4h_full,
            ms=ms,
            cfg=cfg,
            start_ts=_utc_ts(start),
            end_ts=_utc_ts(end),
            decision_interval_min=decision_interval_min,
            regime_detector=regime_detector,
            funding_ts=funding_ts,
            funding_rates=funding_rates,
            trace=trace,
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


def _post_close_block(
    side: str,
    bar_ts: pd.Timestamp,
    *,
    cooldown_until_ts: Optional[pd.Timestamp],
    last_close_dir: int,
    last_close_pnl: float,
    last_close_ts: Optional[pd.Timestamp],
    whipsaw_cooldown_hours: float,
) -> Optional[str]:
    """Name of the post-close time gate blocking an entry at ``bar_ts``,
    or None if allowed.

    Mirrors live_trading_htf.execute_trade lines ~2925-2962, specialised
    to the sim's always-flat decision point:
      * 'cooldown' — a losing close set a cooldown that has not elapsed
        (live COOLDOWN_SECONDS, blocks all sides);
      * 'whipsaw' — the last close LOST and ``side`` reverses it within
        ``whipsaw_cooldown_hours`` (live WHIPSAW_COOLDOWN_HOURS, opposite
        side only).
    Live's min-hold guard is intentionally omitted: it only fires while in
    a position, and the sim never evaluates an entry while holding one.
    """
    if cooldown_until_ts is not None and bar_ts < cooldown_until_ts:
        return "cooldown"
    if last_close_dir != 0 and last_close_pnl < 0 and last_close_ts is not None:
        would_reverse = (
            (last_close_dir == 1 and side == "SHORT")
            or (last_close_dir == -1 and side == "LONG")
        )
        if would_reverse:
            hours_since = (bar_ts - last_close_ts).total_seconds() / 3600.0
            if hours_since < whipsaw_cooldown_hours:
                return "whipsaw"
    return None


def _simulate_symbol(
    *,
    symbol: str,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    ms: MarketStructure,
    cfg: ForwardSimConfig,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    decision_interval_min: int,
    regime_detector: MarketRegimeDetector,
    funding_ts: Optional[np.ndarray] = None,
    funding_rates: Optional[np.ndarray] = None,
    trace: Optional[list] = None,
) -> SymbolForwardResult:
    """Walk one symbol's 5m bars and emit entry events.

    Decisions are evaluated every `decision_interval_min` minutes — by
    default 15, matching the live bot's loop. At each decision bar, the
    MarketStructure window is the previous 12 days of bars (5m / 1h /
    4h), exactly the slice the live bot would have had.
    """
    sym_cfg = STRUCTURE_SYMBOL_CONFIG.get(symbol, "S1")
    entries: list[EntryEvent] = []
    trades: list[TradeResult] = []
    n_decisions = 0
    n_blocklist = 0
    n_struct = 0
    n_trend = 0
    n_s5_ob = 0       # ETH OB-proximity block (P2.D)
    n_s5_adx = 0      # ETH ADX-directional block (P2.D)
    n_sf_adx = 0      # structure-first ADX hard block (P2.D)
    n_exhaustion = 0  # VWAP/ATR exhaustion (P2.D)
    n_rsi = 0         # RSI band guard (P2.D)
    n_open = 0
    n_cooldown = 0    # post-loss cooldown gate (P2.D timing)
    n_whipsaw = 0     # anti-whipsaw reversal gate (P2.D timing)
    # Track when the current open position is expected to close so we
    # skip decisions inside the trade window — the live bot can only
    # hold one position per symbol at a time.
    next_free_ts: Optional[pd.Timestamp] = None
    # Stateful post-close gates (mirror live last_loss_time / last_close_*).
    # cooldown_until_ts is set only on a LOSING close (never explicitly
    # cleared — a stale past value compares harmlessly, exactly like live's
    # last_loss_time). last_close_* are updated on every close for whipsaw.
    cooldown_until_ts: Optional[pd.Timestamp] = None
    last_close_dir: int = 0          # +1 LONG, -1 SHORT, 0 none yet
    last_close_pnl: float = 0.0
    last_close_ts: Optional[pd.Timestamp] = None

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
    idx_15m_full = df_15m.index if not df_15m.empty else pd.DatetimeIndex([])
    idx_1h_full = df_1h.index
    idx_4h_full = df_4h.index

    def _rec(outcome: str, the_side: Optional[str] = None) -> None:
        # Diagnostic decision trace (no effect when trace is None). bar_ts
        # is read from the enclosing loop scope at call time.
        if trace is not None:
            trace.append({
                "symbol": symbol, "ts": bar_ts.isoformat(),
                "side": the_side or "", "outcome": outcome,
            })

    for i in range(0, len(decisions_5m), step):
        bar_ts = decisions_5m.index[i]
        # Live can only hold one position per symbol — skip decisions
        # that fall inside the currently-simulated trade window.
        if next_free_ts is not None and bar_ts < next_free_ts:
            n_open += 1
            continue

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
            _rec("trend")
            continue

        # S5 filters (live: ETH only). Helpers shared with the live bot
        # in src/signals/structure_filters.py — same code, same answer.
        price = float(decisions_5m["close"].iloc[i])
        if sym_cfg == "S5" and len(idx_15m_full) > 0:
            pos_15m = idx_15m_full.searchsorted(bar_ts, side="right")
            win_15m = df_15m.iloc[max(0, pos_15m - 100):pos_15m]
            if not passes_ob_proximity(
                win_15m,
                direction_long=(side == "LONG"),
                current_price=price,
                proximity_pct=cfg.structure_ob_proximity_pct,
            ):
                n_s5_ob += 1
                _rec("s5_ob", side)
                continue
            if not passes_adx_directional(
                win_15m,
                direction_long=(side == "LONG"),
                adx_guard_min=cfg.adx_guard_min,
            ):
                n_s5_adx += 1
                _rec("s5_adx", side)
                continue

        if (symbol, side) in cfg.blocklist:
            n_blocklist += 1
            _rec("blocklist", side)
            continue

        floor, _label = resolve_struct_floor(cfg, symbol, side)
        struct_conf = float(sig.get("confidence", 0.0) or 0.0)
        if floor is not None and struct_conf < floor:
            n_struct += 1
            _rec("struct_floor", side)
            continue

        # Stateful post-close time gates (cooldown / anti-whipsaw). Live
        # applies these at the top of execute_trade — after the structure
        # decision + blocklist + floor (phase 1) and before the
        # structure-first ADX / exhaustion / RSI guards (phase 2).
        gate = _post_close_block(
            side, bar_ts,
            cooldown_until_ts=cooldown_until_ts,
            last_close_dir=last_close_dir,
            last_close_pnl=last_close_pnl,
            last_close_ts=last_close_ts,
            whipsaw_cooldown_hours=cfg.whipsaw_cooldown_hours,
        )
        if gate == "cooldown":
            n_cooldown += 1
            _rec("cooldown", side)
            continue
        if gate == "whipsaw":
            n_whipsaw += 1
            _rec("whipsaw", side)
            continue

        # Pre-trade guards (P2.D). Apply in the same order as live
        # execute_trade: structure-first ADX → exhaustion → RSI. The 15m
        # window may be empty (early in cache); guards fail-open then.
        #
        # ADX and the RSI trend bands now come from the SAME
        # MarketRegimeDetector the live bot uses (P2.D #1), replacing the
        # earlier simplified kline approximations that wrongly blocked
        # ~16% of live entries. Residuals vs live: the RSI *value* is a
        # kline proxy (live reads it from the API signals bundle, not
        # reachable offline) and the confidence≥0.90 rescue override is
        # not replayed (needs model conf + order-flow/whale/mtf signals).
        if len(idx_15m_full) > 0:
            pos_15m = idx_15m_full.searchsorted(bar_ts, side="right")
            win_15m = df_15m.iloc[max(0, pos_15m - 100):pos_15m]
            regime_info = regime_detector.detect_regime(win_15m)
            adx_val = float(regime_info.trend_strength or 0.0)
            regime = regime_info.regime

            # Structure-first ADX hard block (live: adx_val < ADX_GUARD_MIN).
            if adx_val < cfg.adx_guard_min:
                n_sf_adx += 1
                _rec("struct_first_adx", side)
                continue
            if not passes_exhaustion_filter(
                win_15m,
                current_price=price,
                threshold_atr=cfg.exhaustion_atr_threshold,
            ):
                n_exhaustion += 1
                _rec("exhaustion", side)
                continue

            # Trend-aware RSI bands (mirror live _check_rsi_adx_guard): in a
            # confirmed strong trend matching our direction, loosen the band.
            ob_threshold = cfg.rsi_guard_ob_threshold
            os_threshold = cfg.rsi_guard_os_threshold
            if adx_val >= cfg.rsi_guard_trend_adx_min:
                if side == "LONG" and regime == MarketRegime.TRENDING_UP:
                    ob_threshold = cfg.rsi_guard_ob_trend
                elif side == "SHORT" and regime == MarketRegime.TRENDING_DOWN:
                    os_threshold = cfg.rsi_guard_os_trend
            if not passes_rsi_guard(
                win_15m,
                direction_long=(side == "LONG"),
                ob_threshold=ob_threshold,
                os_threshold=os_threshold,
            ):
                n_rsi += 1
                _rec("rsi", side)
                continue

        # price computed above (needed by S5 OB-proximity helper).
        _rec("entry", side)
        entries.append(EntryEvent(
            ts=bar_ts,
            symbol=symbol,
            side=side,
            confidence=struct_conf,
            price=price,
            trend=sig.get("trend", ""),
            last_signal_direction=sig.get("last_signal_direction", ""),
        ))

        # Simulate the trade through to exit. Returns None only when the
        # entry is on the very last bar of the cache.
        trade = simulate_position(
            symbol=symbol,
            side=side,
            entry_ts=bar_ts,
            entry_price=price,
            confidence=struct_conf,
            df_5m=df_5m,
            cfg=cfg,
            funding_ts=funding_ts,
            funding_rates=funding_rates,
        )
        if trade is not None:
            trades.append(trade)
            next_free_ts = trade.exit_ts
            # Update post-close state (mirror live). Whipsaw needs the last
            # close on every trade; cooldown is armed only on a loss.
            last_close_ts = trade.exit_ts
            last_close_dir = 1 if side == "LONG" else -1
            last_close_pnl = trade.realized_pnl_usd
            if trade.realized_pnl_usd < 0:
                cooldown_until_ts = trade.exit_ts + pd.Timedelta(
                    seconds=cfg.cooldown_seconds
                )

    return SymbolForwardResult(
        symbol=symbol,
        n_decisions=n_decisions,
        entries=entries,
        trades=trades,
        skipped_by_blocklist=n_blocklist,
        skipped_by_struct_floor=n_struct,
        skipped_by_trend=n_trend,
        skipped_by_s5_unimplemented=n_s5_ob + n_s5_adx,
        skipped_by_struct_first_adx=n_sf_adx,
        skipped_by_exhaustion=n_exhaustion,
        skipped_by_rsi=n_rsi,
        skipped_by_cooldown=n_cooldown,
        skipped_by_whipsaw=n_whipsaw,
        skipped_by_open_position=n_open,
    )
