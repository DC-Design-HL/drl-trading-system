"""Pure-function metrics over a list of closed trades.

The functions here take a list of trade-close rows (each at minimum
needs ``timestamp`` and ``pnl``) and return scalar metrics. They are
deliberately independent of SQLite / Pandas / the live bot — so they
can be unit-tested with synthetic data and so the Performance Monitor
can compose them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

# Trading-days-per-year proxy used for Sharpe annualization. Crypto
# trades 24/7, so use 365 instead of the equities convention of 252.
_PERIODS_PER_YEAR_DAILY = 365


@dataclass(frozen=True)
class TradeClose:
    """Minimal view of a close row used by the metrics functions.

    The DB row has more columns; this dataclass is the contract the
    metrics layer requires. The performance_monitor module is
    responsible for adapting rows from `trades` into TradeClose tuples.
    """

    ts: datetime
    symbol: str
    side: str  # 'LONG' | 'SHORT'
    pnl: float
    # PROFITABILITY_PLAN.md P5: estimated funding PAID over the position's
    # life (USD, positive = a cost that reduces net). 0.0 when unknown, so
    # pre-P5 callers that omit it get identical gross behaviour.
    funding_usd: float = 0.0


def estimate_funding_usd(
    *,
    entry_ts: datetime,
    exit_ts: datetime,
    notional: float,
    side: str,
    funding_ts: Sequence[float],
    funding_rates: Sequence[float],
) -> float:
    """Estimate funding PAID over [entry_ts, exit_ts] (PROFITABILITY_PLAN.md P5).

    ``funding_ts`` are funding-boundary epoch SECONDS, ``funding_rates`` the
    aligned 8h rates. A boundary counts if entry_ts < boundary <= exit_ts.
    A LONG pays when the rate is positive (cost > 0); a SHORT receives it
    (cost < 0). Returns the cost in USD: subtract it from gross PnL to get
    funding-aware net. Empty/short inputs → 0.0 (no estimate available)."""
    if notional <= 0 or not len(funding_ts):
        return 0.0
    e = entry_ts.timestamp()
    x = exit_ts.timestamp()
    if x <= e:
        return 0.0
    sign = 1.0 if str(side).upper() == "LONG" else -1.0
    rate_sum = 0.0
    for ts, rate in zip(funding_ts, funding_rates):
        if e < float(ts) <= x:
            rate_sum += float(rate)
    return sign * notional * rate_sum


def parse_ts(text: str) -> datetime:
    """Parse a timestamp from the trades table into an aware UTC datetime."""
    # Accept '...Z' and naive ISO; treat naive as UTC.
    s = text.rstrip("Z")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def filter_window(
    trades: Iterable[TradeClose], *, now: datetime, days: float
) -> list[TradeClose]:
    """Return trades whose close timestamp is within the last `days` of `now`."""
    cutoff = now - timedelta(days=days)
    return [t for t in trades if t.ts >= cutoff]


def net_pnl(trades: Sequence[TradeClose]) -> float:
    return float(sum(t.pnl for t in trades))


def funding_total(trades: Sequence[TradeClose]) -> float:
    """Sum of estimated funding paid across the trades (P5)."""
    return float(sum(getattr(t, "funding_usd", 0.0) for t in trades))


def net_pnl_after_funding(trades: Sequence[TradeClose]) -> float:
    """Net PnL with estimated funding cost subtracted (P5)."""
    return float(sum(t.pnl - getattr(t, "funding_usd", 0.0) for t in trades))


def win_rate(trades: Sequence[TradeClose]) -> float:
    """Fraction of trades with pnl > 0. Returns 0.0 if no trades."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return wins / len(trades)


def profit_factor(trades: Sequence[TradeClose]) -> float:
    """Sum of wins divided by abs(sum of losses).

    Returns ``math.inf`` if there are wins but zero losses, and ``0.0``
    if there are no wins. Returns 0.0 for an empty input.
    """
    if not trades:
        return 0.0
    wins = sum(t.pnl for t in trades if t.pnl > 0)
    losses = -sum(t.pnl for t in trades if t.pnl < 0)
    if losses == 0:
        return math.inf if wins > 0 else 0.0
    return wins / losses


def _daily_pnl_series(trades: Sequence[TradeClose]) -> list[float]:
    """Bucket pnl into UTC calendar days, return list of daily-net values."""
    by_day: dict[str, float] = {}
    for t in trades:
        key = t.ts.astimezone(timezone.utc).date().isoformat()
        by_day[key] = by_day.get(key, 0.0) + t.pnl
    return [by_day[k] for k in sorted(by_day)]


def _std(xs: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1). Returns 0.0 if n<2."""
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def sharpe_ratio(
    trades: Sequence[TradeClose],
    *,
    risk_free_daily: float = 0.0,
    capital_base: float = 5000.0,
) -> float:
    """Annualized Sharpe over daily-net pnl returns.

    Returns 0.0 if fewer than 2 active trading days or zero variance.
    capital_base is used to convert daily-net-pnl into a return
    fraction. Default is the May-1-reset baseline ($5,000) — callers
    can override for accuracy when capital has changed.
    """
    series = _daily_pnl_series(trades)
    if len(series) < 2:
        return 0.0
    returns = [x / capital_base for x in series]
    excess = [r - risk_free_daily for r in returns]
    sd = _std(excess)
    if sd == 0:
        return 0.0
    mean = sum(excess) / len(excess)
    return (mean / sd) * math.sqrt(_PERIODS_PER_YEAR_DAILY)


def sortino_ratio(
    trades: Sequence[TradeClose],
    *,
    risk_free_daily: float = 0.0,
    capital_base: float = 5000.0,
) -> float:
    """Annualized Sortino: like Sharpe but uses downside deviation only."""
    series = _daily_pnl_series(trades)
    if len(series) < 2:
        return 0.0
    returns = [x / capital_base - risk_free_daily for x in series]
    downside = [r for r in returns if r < 0]
    if not downside:
        return math.inf if sum(returns) > 0 else 0.0
    dd_sq = sum(d * d for d in downside) / len(returns)
    dd = math.sqrt(dd_sq)
    if dd == 0:
        return 0.0
    mean = sum(returns) / len(returns)
    return (mean / dd) * math.sqrt(_PERIODS_PER_YEAR_DAILY)


def max_drawdown_pct(
    trades: Sequence[TradeClose], *, capital_base: float = 5000.0
) -> float:
    """Maximum drawdown of the running pnl-equity curve, in percent.

    Equity = capital_base + cumulative pnl, ordered by close timestamp.
    Returns a non-negative number: e.g. 5.0 means peak-to-trough was
    5% of capital_base.
    """
    if not trades:
        return 0.0
    cum = capital_base
    peak = capital_base
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.ts):
        cum += t.pnl
        peak = max(peak, cum)
        dd_pct = (peak - cum) / capital_base * 100.0
        max_dd = max(max_dd, dd_pct)
    return max_dd


def consecutive_losses(trades: Sequence[TradeClose]) -> int:
    """Longest run of consecutive pnl<=0 closes, in chronological order."""
    if not trades:
        return 0
    in_order = sorted(trades, key=lambda x: x.ts)
    longest = 0
    cur = 0
    for t in in_order:
        if t.pnl <= 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest


def trailing_consecutive_losses(trades: Sequence[TradeClose]) -> int:
    """Run of pnl<=0 closes ending at the most recent close.

    This is what T4 (3+ consecutive losses) actually fires on — we want
    "the bot is currently on a losing streak", not "at some point in
    history it was".
    """
    if not trades:
        return 0
    in_order = sorted(trades, key=lambda x: x.ts, reverse=True)
    n = 0
    for t in in_order:
        if t.pnl <= 0:
            n += 1
        else:
            break
    return n


def summarize(
    trades: Sequence[TradeClose], *, capital_base: float = 5000.0
) -> dict[str, Any]:
    """Bundle all the metrics into one dict — convenient for snapshot rows."""
    return {
        "net_pnl_usd": net_pnl(trades),
        # P5: funding-aware net + the funding total it was derived from.
        # net_pnl_usd stays GROSS (price-only) for backcompat; readers that
        # want the true bottom line use net_pnl_after_funding_usd.
        "funding_usd_total": funding_total(trades),
        "net_pnl_after_funding_usd": net_pnl_after_funding(trades),
        "num_closes": len(trades),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "sharpe": sharpe_ratio(trades, capital_base=capital_base),
        "sortino": sortino_ratio(trades, capital_base=capital_base),
        "max_drawdown_pct": max_drawdown_pct(trades, capital_base=capital_base),
    }
