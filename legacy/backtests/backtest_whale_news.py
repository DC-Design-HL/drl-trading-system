#!/usr/bin/env python3
"""
Backtest: News + Whale Filter Analysis
=======================================
Evaluates how news sentiment and whale wallet flow correlate with
trading bot win rates. Tests whale-only, news-only, and combined filters.

Data sources:
  - Trades: data/trading.db (is_testnet=1, CLOSE actions)
  - News: data/trading.db news_events table
  - Whale: data/whale_behavior/eth/*.jsonl (Alchemy WebSocket data)
"""

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).parent
DB_PATH = REPO / "data" / "trading.db"
WHALE_DIR = REPO / "data" / "whale_behavior" / "eth"

SYMBOL_TO_TAGS = {
    "BTCUSDT": {"BTC", "CRYPTO", "ALL"},
    "ETHUSDT": {"ETH", "CRYPTO", "ALL"},
    "SOLUSDT": {"SOL", "CRYPTO", "ALL"},
    "XRPUSDT": {"XRP", "CRYPTO", "ALL"},
}


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    id: int
    timestamp: datetime
    symbol: str
    action: str
    price: float
    pnl: Optional[float]
    data: dict

@dataclass
class NewsEvent:
    id: int
    title: str
    published_at: datetime
    sentiment_score: float
    urgency: int
    assets: str
    source: str

@dataclass
class WhaleTx:
    timestamp: datetime
    action: str      # LARGE_TRANSFER_IN, LARGE_TRANSFER_OUT, CONTRACT_CALL
    value_eth: float
    wallet: str

@dataclass
class TradePair:
    open_trade: Trade
    close_trade: Optional[Trade]
    pnl: float
    is_win: bool


# ── Parsing helpers ──────────────────────────────────────────────────────────

def parse_ts(ts_str: str) -> datetime:
    ts_str = ts_str.strip()
    for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]:
        try:
            dt = datetime.strptime(ts_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def news_matches_symbol(news_assets_json: str, symbol: str) -> bool:
    try:
        tags = set(json.loads(news_assets_json))
    except (json.JSONDecodeError, TypeError):
        return False
    return bool(tags & SYMBOL_TO_TAGS.get(symbol, set()))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_trades_and_news(days: int = 7):
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    trades = []
    for r in db.execute(
        "SELECT * FROM trades WHERE is_testnet=1 AND timestamp >= ? ORDER BY timestamp",
        (cutoff,)
    ):
        data = json.loads(r["data"]) if r["data"] else {}
        price = r["price"] or data.get("price", 0)
        pnl = r["pnl"] if r["pnl"] is not None else data.get("pnl")
        trades.append(Trade(
            id=r["id"], timestamp=parse_ts(r["timestamp"]),
            symbol=r["symbol"], action=r["action"],
            price=price, pnl=pnl, data=data
        ))

    news = []
    for r in db.execute(
        "SELECT * FROM news_events WHERE published_at >= ? ORDER BY published_at",
        (cutoff,)
    ):
        news.append(NewsEvent(
            id=r["id"], title=r["title"],
            published_at=parse_ts(r["published_at"]),
            sentiment_score=r["sentiment_score"] or 0.0,
            urgency=r["urgency"] or 1,
            assets=r["assets"] or "[]",
            source=r["source"],
        ))

    db.close()
    return trades, news


def load_whale_txs(days: int = 7) -> List[WhaleTx]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_epoch = cutoff.timestamp()
    txs = []

    if not WHALE_DIR.exists():
        print(f"  Warning: whale data dir not found: {WHALE_DIR}")
        return txs

    for fpath in sorted(WHALE_DIR.glob("*.jsonl")):
        wallet_name = fpath.stem
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts_val = rec.get("timestamp", 0)
                if isinstance(ts_val, (int, float)) and ts_val > 1e12:
                    ts_val = ts_val / 1000  # ms to s

                if ts_val < cutoff_epoch:
                    continue

                action = rec.get("action", "")
                if action not in ("LARGE_TRANSFER_IN", "LARGE_TRANSFER_OUT"):
                    continue

                value = float(rec.get("value_eth", rec.get("value", 0)))
                try:
                    dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                except (OSError, ValueError):
                    continue

                txs.append(WhaleTx(
                    timestamp=dt, action=action,
                    value_eth=value, wallet=wallet_name
                ))

    txs.sort(key=lambda t: t.timestamp)
    return txs


# ── Pair trades ──────────────────────────────────────────────────────────────

def pair_trades(trades: List[Trade]) -> List[TradePair]:
    open_stacks: dict = {}
    pairs = []
    for t in trades:
        if t.action.startswith("OPEN_"):
            direction = "LONG" if "LONG" in t.action else "SHORT"
            key = (t.symbol, direction)
            open_stacks.setdefault(key, []).append(t)
        elif t.action.startswith("CLOSE_") and "PARTIAL" not in t.action:
            direction = "LONG" if "LONG" in t.action else "SHORT"
            key = (t.symbol, direction)
            stack = open_stacks.get(key, [])
            if stack:
                open_t = stack.pop(0)
                pnl = t.pnl if t.pnl is not None else 0.0
                pairs.append(TradePair(
                    open_trade=open_t, close_trade=t,
                    pnl=pnl, is_win=pnl > 0
                ))
    return pairs


# ── Signal computation ───────────────────────────────────────────────────────

def get_news_sentiment(news: List[NewsEvent], symbol: str, before: datetime,
                       lookback_h: float = 2.0) -> Optional[float]:
    """Average sentiment in lookback window. Returns None if no news."""
    cutoff = before - timedelta(hours=lookback_h)
    matching = [n for n in news
                if cutoff <= n.published_at <= before
                and news_matches_symbol(n.assets, symbol)]
    if not matching:
        return None
    return sum(n.sentiment_score for n in matching) / len(matching)


def get_whale_flow(whale_txs: List[WhaleTx], before: datetime,
                   lookback_h: float = 1.0) -> dict:
    """Compute inflow/outflow in lookback window."""
    cutoff = before - timedelta(hours=lookback_h)
    inflow = 0.0   # LARGE_TRANSFER_IN = to exchange = sell pressure
    outflow = 0.0  # LARGE_TRANSFER_OUT = from exchange = accumulation

    for tx in whale_txs:
        if tx.timestamp < cutoff:
            continue
        if tx.timestamp > before:
            break
        if tx.action == "LARGE_TRANSFER_IN":
            inflow += tx.value_eth
        elif tx.action == "LARGE_TRANSFER_OUT":
            outflow += tx.value_eth

    total = inflow + outflow
    if total == 0:
        regime = "no_data"
    elif outflow > inflow * 1.2:
        regime = "accumulation"
    elif inflow > outflow * 1.2:
        regime = "distribution"
    else:
        regime = "balanced"

    return {"inflow": inflow, "outflow": outflow, "regime": regime}


# ── Filter definitions ───────────────────────────────────────────────────────

def filter_w1(pair, news, whale_txs):
    """W1: Block distribution — only trade during accumulation or balanced."""
    flow = get_whale_flow(whale_txs, pair.open_trade.timestamp)
    if flow["regime"] == "distribution":
        return True, f"whale distribution (in={flow['inflow']:.0f} > out={flow['outflow']:.0f})"
    return False, ""

def filter_w2(pair, news, whale_txs):
    """W2: Only accumulation — strictest whale filter."""
    flow = get_whale_flow(whale_txs, pair.open_trade.timestamp)
    if flow["regime"] != "accumulation":
        return True, f"not accumulation ({flow['regime']})"
    return False, ""

def filter_w3(pair, news, whale_txs):
    """W3: Block LONG during distribution — directional whale filter."""
    flow = get_whale_flow(whale_txs, pair.open_trade.timestamp)
    if flow["regime"] == "distribution" and "LONG" in pair.open_trade.action:
        return True, f"LONG during distribution"
    return False, ""

def filter_n1(pair, news, whale_txs):
    """N1: Block bearish news — sentiment >= -0.2."""
    sent = get_news_sentiment(news, pair.open_trade.symbol, pair.open_trade.timestamp)
    if sent is not None and sent < -0.2:
        return True, f"bearish news (sent={sent:.2f})"
    return False, ""

def filter_n2(pair, news, whale_txs):
    """N2: Only neutral news — -0.2 <= sentiment <= 0.2."""
    sent = get_news_sentiment(news, pair.open_trade.symbol, pair.open_trade.timestamp)
    if sent is not None and abs(sent) > 0.2:
        return True, f"non-neutral news (sent={sent:.2f})"
    return False, ""

def filter_n3(pair, news, whale_txs):
    """N3: Block direction-misaligned — bearish news + LONG, or bullish news + SHORT."""
    sent = get_news_sentiment(news, pair.open_trade.symbol, pair.open_trade.timestamp)
    if sent is None:
        return False, ""
    if "LONG" in pair.open_trade.action and sent < -0.2:
        return True, f"LONG with bearish news ({sent:.2f})"
    if "SHORT" in pair.open_trade.action and sent > 0.2:
        return True, f"SHORT with bullish news ({sent:.2f})"
    return False, ""

def filter_c1(pair, news, whale_txs):
    """C1: Block distrib AND bearish — only blocks when BOTH signals negative."""
    flow = get_whale_flow(whale_txs, pair.open_trade.timestamp)
    sent = get_news_sentiment(news, pair.open_trade.symbol, pair.open_trade.timestamp)
    is_distrib = flow["regime"] == "distribution"
    is_bearish = sent is not None and sent < -0.2
    if is_distrib and is_bearish:
        return True, f"distribution + bearish news (sent={sent:.2f})"
    return False, ""

def filter_c3(pair, news, whale_txs):
    """C3: Block distrib OR bearish — blocks when EITHER signal negative."""
    flow = get_whale_flow(whale_txs, pair.open_trade.timestamp)
    sent = get_news_sentiment(news, pair.open_trade.symbol, pair.open_trade.timestamp)
    is_distrib = flow["regime"] == "distribution"
    is_bearish = sent is not None and sent < -0.2
    if is_distrib or is_bearish:
        reasons = []
        if is_distrib:
            reasons.append("distribution")
        if is_bearish:
            reasons.append(f"bearish({sent:.2f})")
        return True, " + ".join(reasons)
    return False, ""

def filter_c4(pair, news, whale_txs):
    """C4: Accum+any OR balanced+bullish — most restrictive."""
    flow = get_whale_flow(whale_txs, pair.open_trade.timestamp)
    sent = get_news_sentiment(news, pair.open_trade.symbol, pair.open_trade.timestamp)
    regime = flow["regime"]
    is_bullish = sent is not None and sent > 0.2
    # Allow if: accumulation (any news) OR balanced + bullish news
    if regime == "accumulation":
        return False, ""
    if regime == "balanced" and (is_bullish or sent is None):
        return False, ""
    return True, f"not (accum OR balanced+bullish): regime={regime}, sent={sent}"


FILTERS = {
    "W1: Block distrib":       filter_w1,
    "W2: Only accum":          filter_w2,
    "W3: Block LONG distrib":  filter_w3,
    "N1: Block bearish":       filter_n1,
    "N2: Only neutral":        filter_n2,
    "N3: Block misaligned":    filter_n3,
    "C1: Distrib AND bearish": filter_c1,
    "C3: Distrib OR bearish":  filter_c3,
    "C4: Accum+any/bal+bull":  filter_c4,
}


# ── Run analysis ─────────────────────────────────────────────────────────────

def main(days: int = 7):
    print(f"\n{'='*90}")
    print(f"  NEWS + WHALE FILTER ANALYSIS — Last {days} Days")
    print(f"{'='*90}")

    # Load data
    trades, news = load_trades_and_news(days)
    whale_txs = load_whale_txs(days)
    pairs = pair_trades(trades)

    print(f"\n  Data loaded:")
    print(f"    Trades: {len(trades)} raw, {len(pairs)} paired closes")
    print(f"    News events: {len(news)}")
    print(f"    Whale transactions: {len(whale_txs)}")

    if not pairs:
        print("\n  No trade pairs found — nothing to analyze.")
        return

    # Baseline
    base_wins = sum(1 for p in pairs if p.is_win)
    base_losses = len(pairs) - base_wins
    base_wr = base_wins / len(pairs) * 100
    base_pnl = sum(p.pnl for p in pairs)
    base_avg = base_pnl / len(pairs)

    print(f"\n  Baseline: {len(pairs)} trades, {base_wins}W/{base_losses}L, "
          f"{base_wr:.1f}% WR, ${base_pnl:+.2f} total, ${base_avg:+.2f}/trade")

    # Whale flow regime distribution across trades
    regime_stats = {"accumulation": [], "distribution": [], "balanced": [], "no_data": []}
    for p in pairs:
        flow = get_whale_flow(whale_txs, p.open_trade.timestamp)
        regime_stats[flow["regime"]].append(p)

    print(f"\n  Whale flow regime at trade time:")
    for regime, rp in regime_stats.items():
        if not rp:
            continue
        wins = sum(1 for p in rp if p.is_win)
        wr = wins / len(rp) * 100 if rp else 0
        pnl = sum(p.pnl for p in rp)
        print(f"    {regime:15s}: {len(rp):3d} trades, {wins}W/{len(rp)-wins}L, "
              f"{wr:.0f}% WR, ${pnl:+.2f}")

    # Run all filters
    print(f"\n  {'─'*88}")
    print(f"  {'Filter':<28} {'Trades':>6} {'WR':>7} {'$/trade':>9} {'Total PnL':>11} "
          f"{'Blocked':>7} {'Blk W/L':>8} {'Blk PnL':>10}")
    print(f"  {'─'*88}")

    # Baseline row
    print(f"  {'BASELINE':<28} {len(pairs):>6} {base_wr:>6.1f}% ${base_avg:>+8.2f} "
          f"${base_pnl:>+10.2f} {'—':>7} {'—':>8} {'—':>10}")

    filter_results = {}
    for fname, ffn in FILTERS.items():
        kept = []
        blocked = []
        for p in pairs:
            is_blocked, reason = ffn(p, news, whale_txs)
            if is_blocked:
                blocked.append((p, reason))
            else:
                kept.append(p)

        k_wins = sum(1 for p in kept if p.is_win)
        k_wr = k_wins / len(kept) * 100 if kept else 0
        k_pnl = sum(p.pnl for p in kept)
        k_avg = k_pnl / len(kept) if kept else 0

        b_wins = sum(1 for p, _ in blocked if p.is_win)
        b_losses = len(blocked) - b_wins
        b_pnl = sum(p.pnl for p, _ in blocked)

        filter_results[fname] = {
            "kept": kept, "blocked": blocked,
            "wr": k_wr, "pnl": k_pnl, "avg": k_avg,
            "blk_wins": b_wins, "blk_losses": b_losses, "blk_pnl": b_pnl,
        }

        print(f"  {fname:<28} {len(kept):>6} {k_wr:>6.1f}% ${k_avg:>+8.2f} "
              f"${k_pnl:>+10.2f} {len(blocked):>7} "
              f"{b_wins}W/{b_losses}L ${b_pnl:>+9.2f}")

    # Blocked trade details for top filters
    print(f"\n{'='*90}")
    print(f"  BLOCKED TRADE DETAILS")
    print(f"{'='*90}")

    for fname in ["W1: Block distrib", "C1: Distrib AND bearish", "C3: Distrib OR bearish"]:
        fr = filter_results.get(fname, {})
        blocked = fr.get("blocked", [])
        if not blocked:
            print(f"\n  {fname}: No trades blocked")
            continue

        print(f"\n  {fname} — {len(blocked)} blocked trades:")
        for p, reason in blocked:
            tag = "WIN " if p.is_win else "LOSS"
            print(f"    [{tag}] {p.open_trade.action} {p.open_trade.symbol} @ ${p.open_trade.price:.2f} "
                  f"| PnL ${p.pnl:+.2f} | {p.open_trade.timestamp.strftime('%m/%d %H:%M')} "
                  f"| {reason}")

    # Summary / recommendations
    print(f"\n{'='*90}")
    print(f"  SUMMARY & RECOMMENDATIONS")
    print(f"{'='*90}")

    best_wr = max(filter_results.items(), key=lambda x: x[1]["wr"])
    best_pnl = max(filter_results.items(), key=lambda x: x[1]["pnl"])
    best_avg = max(filter_results.items(), key=lambda x: x[1]["avg"])

    print(f"\n  Best WR:      {best_wr[0]} ({best_wr[1]['wr']:.1f}%, "
          f"{len(best_wr[1]['kept'])} trades)")
    print(f"  Best total $: {best_pnl[0]} (${best_pnl[1]['pnl']:+.2f}, "
          f"{len(best_pnl[1]['kept'])} trades)")
    print(f"  Best $/trade: {best_avg[0]} (${best_avg[1]['avg']:+.2f}/trade)")

    # C1 verdict
    c1 = filter_results.get("C1: Distrib AND bearish", {})
    if c1:
        print(f"\n  C1 (production candidate): {len(c1['kept'])} trades, "
              f"{c1['wr']:.1f}% WR, ${c1['pnl']:+.2f}, "
              f"blocked {len(c1['blocked'])} ({c1['blk_wins']}W/{c1['blk_losses']}L, "
              f"${c1['blk_pnl']:+.2f})")
        if c1["blk_losses"] > c1["blk_wins"]:
            print(f"    ✅ C1 filters more losses than wins — safe for production")
        else:
            print(f"    ⚠️  C1 filters more wins than losses — needs review")

    print()


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    main(days)
