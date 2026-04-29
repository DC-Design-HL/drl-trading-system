#!/usr/bin/env python3
"""
May 2 rerun — whale-flow + news filter backtest.

Source memory: project_news_whale_filter_analysis.md, original analysis
2026-04-18 (5 days / 101 trades) found:
  W1 (block distribution):           83% WR, +$412
  N1 (block bearish news):           68% WR, +$441
  C1 (block distrib AND bearish):    69% WR, +$467  ← highest total PnL
The action item was to rerun ~May 2 with ~3 weeks more data and see if
results hold; if so, deploy C1 as a live guard.

This script reproduces that analysis.

Inputs (all real, all already in repo on the server):
  * data/trading.db trades — closed bot trades with realized pnl
  * data/whale_behavior/eth/*.jsonl — raw on-chain whale transactions
    from 10 wallets (Binance/Coinbase/Kraken hot+cold, Robinhood,
    Jump Trading, Galaxy Digital, ETH 2.0 deposit, etc.)
    Direction: 'in' = into wallet, 'out' = out of wallet.
  * data/trading.db news_events — RSS-scored news with sentiment_score,
    urgency, event_type, asset tags.

Whale-flow signal:
  For each trade open, look at the preceding WINDOW_HOURS of whale
  transactions. EXCHANGE wallets (Binance/Coinbase/Kraken) are the
  meaningful flow indicators:
    - net flow IN  = distribution (whales depositing → about to sell) → bearish
    - net flow OUT = accumulation (whales withdrawing → holding)     → bullish
  The other wallets (Robinhood, Galaxy, Jump, ETH 2.0) are tracked
  separately for diagnostic but the primary signal is exchange flow.

News signal:
  News in preceding WINDOW_HOURS, asset-filtered (BTC/ETH/SOL/XRP) plus
  global. Bias = sign of average sentiment_score (>+0.1 bullish,
  <-0.1 bearish, else neutral).

Outputs:
  * Same table format as project_news_whale_filter_analysis.md
  * data/training/whale_news_filter_results.json — machine-readable

Run on server (no special deps beyond stdlib + sqlite):
    python3 scripts/backtest_whale_flow_news_filters.py
        [--window_hours 4]
        [--min_flow_eth 100]      # ignore noise transactions below this
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "trading.db"
WHALE_DIR = REPO / "data" / "whale_behavior" / "eth"
OUT_PATH = REPO / "data" / "training" / "whale_news_filter_results.json"

# Exchange wallets — net flow IN to these = distribution (about to sell).
EXCHANGE_WALLETS = {
    "binance_hot_wallet",
    "binance_cold_wallet",
    "binance_cold_2",
    "binance_reserve",
    "coinbase_institutional",
    "kraken_deposit",
}
# Other wallets (smart money / staking) — tracked separately.
OTHER_WALLETS = {
    "eth_2.0_deposit_contract",
    "galaxy_digital",
    "jump_trading",
    "robinhood",
    "smart_money_whale_1",
}


def parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "+" not in s and "-" not in s[-6:]:
        s += "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def load_closed_trades() -> list[dict]:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""SELECT id, timestamp, symbol, action, price, pnl, reason
                   FROM trades WHERE timestamp >= '2026-04-06' AND is_testnet=1
                   ORDER BY timestamp""")
    open_pos: dict[str, dict] = {}
    rt = []
    for tid, ts, sym, action, price, pnl, reason in cur.fetchall():
        if "OPEN" in action and "PARTIAL" not in action:
            open_pos[sym] = {"open_ts": ts, "side": "LONG" if "LONG" in action else "SHORT",
                             "entry": price, "open_id": tid}
        elif "CLOSE" in action and "PARTIAL" not in action and sym in open_pos:
            o = open_pos.pop(sym)
            rt.append({
                **o,
                "symbol": sym,
                "close_ts": ts,
                "exit": price,
                "pnl": pnl or 0.0,
                "reason": reason or "",
                "win": 1 if (pnl or 0) > 0 else 0,
            })
    conn.close()
    return rt


def load_whale_txs() -> list[dict]:
    """Load all whale txs across the 10 wallets, with wallet attribution."""
    out = []
    if not WHALE_DIR.exists():
        return out
    for f in sorted(WHALE_DIR.glob("*.jsonl")):
        wallet = f.stem
        with open(f) as fp:
            for line in fp:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts = d.get("timestamp")
                if ts is None:
                    continue
                try:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                except Exception:
                    continue
                out.append({
                    "ts": dt,
                    "wallet": wallet,
                    "is_exchange": wallet in EXCHANGE_WALLETS,
                    "value_eth": float(d.get("value_eth") or 0),
                    "direction": (d.get("direction") or "").lower(),
                })
    out.sort(key=lambda x: x["ts"])
    return out


def load_news() -> list[tuple[datetime, float, int, list[str]]]:
    if not DB.exists():
        return []
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""SELECT published_at, sentiment_score, urgency, assets
                   FROM news_events WHERE sentiment_score IS NOT NULL""")
    out = []
    for ts, sent, urgency, assets in cur.fetchall():
        try:
            dt = parse_iso(ts)
        except Exception:
            continue
        try:
            tags = json.loads(assets) if assets else []
        except Exception:
            tags = []
        out.append((dt, float(sent or 0), int(urgency or 1), tags or []))
    conn.close()
    return out


def whale_flow_at(open_ts: datetime, whale_txs: list[dict],
                  window_hours: int, min_flow_eth: float) -> dict:
    """Net exchange flow over the trailing window. Positive net = into
    exchanges = distribution → bearish.
    """
    win_start = open_ts - timedelta(hours=window_hours)
    in_flow = 0.0
    out_flow = 0.0
    for tx in whale_txs:
        if tx["ts"] < win_start or tx["ts"] > open_ts:
            continue
        if not tx["is_exchange"]:
            continue
        if tx["value_eth"] < min_flow_eth:
            continue
        if tx["direction"] == "in":
            in_flow += tx["value_eth"]
        elif tx["direction"] == "out":
            out_flow += tx["value_eth"]
    net_in = in_flow - out_flow
    if net_in > 0:
        regime = "DISTRIBUTION"
    elif net_in < 0:
        regime = "ACCUMULATION"
    else:
        regime = "BALANCED"
    return {"in_flow": in_flow, "out_flow": out_flow, "net_in_eth": net_in, "regime": regime}


def news_bias_at(open_ts: datetime, symbol: str, news: list,
                 window_hours: int) -> dict:
    """Aggregate news bias in trailing window. Asset-filtered to symbol's
    base asset OR untagged global news. Bias = sign of mean sentiment.
    """
    asset = symbol.replace("USDT", "").upper()
    win_start = open_ts - timedelta(hours=window_hours)
    relevant = []
    for ts, sent, urg, tags in news:
        if ts < win_start or ts > open_ts:
            continue
        if tags and not any(asset in (t or "").upper() for t in tags):
            continue
        relevant.append((sent, urg))
    if not relevant:
        return {"count": 0, "avg_sent": 0.0, "max_urgency": 0, "bias": "NEUTRAL"}
    avg = sum(s for s, _ in relevant) / len(relevant)
    bias = "BULLISH" if avg > 0.1 else "BEARISH" if avg < -0.1 else "NEUTRAL"
    return {"count": len(relevant), "avg_sent": avg,
            "max_urgency": max(u for _, u in relevant), "bias": bias}


# ── Filters ─────────────────────────────────────────────────────────────
# Each filter takes a trade-with-context dict and returns True if it
# would BLOCK (skip) the trade.

def w1_block_distribution(t):
    return t["whale_regime"] == "DISTRIBUTION"


def w2_only_accumulation(t):
    return t["whale_regime"] != "ACCUMULATION"


def w3_block_long_during_distrib(t):
    return t["whale_regime"] == "DISTRIBUTION" and t["side"] == "LONG"


def n1_block_bearish_news(t):
    return t["news_bias"] == "BEARISH"


def n2_only_neutral_news(t):
    return t["news_bias"] != "NEUTRAL"


def n3_block_misaligned_news(t):
    if t["news_bias"] == "BULLISH" and t["side"] == "SHORT":
        return True
    if t["news_bias"] == "BEARISH" and t["side"] == "LONG":
        return True
    return False


def c1_block_distrib_and_bearish(t):
    return t["whale_regime"] == "DISTRIBUTION" and t["news_bias"] == "BEARISH"


def c3_block_distrib_or_bearish(t):
    return t["whale_regime"] == "DISTRIBUTION" or t["news_bias"] == "BEARISH"


def c4_accum_any_or_balanced_bullish(t):
    """Allow ONLY: (accumulation regardless of news) OR (balanced + bullish news)."""
    if t["whale_regime"] == "ACCUMULATION":
        return False  # allowed
    if t["whale_regime"] == "BALANCED" and t["news_bias"] == "BULLISH":
        return False  # allowed
    return True  # block


def evaluate(name: str, trades: list[dict], blocker) -> dict:
    blocked = [t for t in trades if blocker(t)]
    allowed = [t for t in trades if not blocker(t)]
    pnl_b = sum(t["pnl"] for t in blocked)
    pnl_a = sum(t["pnl"] for t in allowed)
    n = len(allowed)
    wr = sum(1 for t in allowed if t["pnl"] > 0) / n * 100 if n else 0.0
    avg = pnl_a / n if n else 0.0
    blocked_wr = sum(1 for t in blocked if t["pnl"] > 0) / len(blocked) * 100 if blocked else 0.0
    return {"name": name, "n_blocked": len(blocked), "n_allowed": n,
            "pnl_blocked": pnl_b, "pnl_allowed": pnl_a, "delta": -pnl_b,
            "wr_allowed": wr, "avg_pnl_allowed": avg, "wr_blocked": blocked_wr}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--window_hours", type=int, default=4)
    p.add_argument("--min_flow_eth", type=float, default=100.0,
                   help="Ignore whale txs below this size to filter noise")
    args = p.parse_args(argv)

    print(f"Window: {args.window_hours}h preceding each trade open")
    print(f"Min whale flow size: {args.min_flow_eth} ETH")

    print("Loading closed trades ...")
    trades = load_closed_trades()
    print(f"  {len(trades)} closed round-trips since 2026-04-06")
    base_pnl = sum(t["pnl"] for t in trades)
    base_wr = sum(t["win"] for t in trades) / len(trades) * 100
    base_avg = base_pnl / len(trades)
    print(f"  baseline: WR={base_wr:.1f}%  pnl=${base_pnl:+.2f}  avg=${base_avg:+.2f}/trade")

    print("Loading whale txs from data/whale_behavior/eth/ ...")
    whale_txs = load_whale_txs()
    print(f"  {len(whale_txs)} txs across {len({t['wallet'] for t in whale_txs})} wallets")
    if whale_txs:
        print(f"  range: {whale_txs[0]['ts'].date()} → {whale_txs[-1]['ts'].date()}")
        n_exchange = sum(1 for t in whale_txs if t["is_exchange"])
        n_recent = sum(1 for t in whale_txs if t["ts"] >= datetime(2026, 4, 6, tzinfo=timezone.utc))
        print(f"  {n_exchange} exchange-wallet txs total, {n_recent} txs since 2026-04-06")

    print("Loading news ...")
    news = load_news()
    print(f"  {len(news)} sentiment-scored news events")

    # Annotate each trade with whale-regime and news-bias context
    print("Annotating trades with context ...")
    for t in trades:
        ots = parse_iso(t["open_ts"])
        wf = whale_flow_at(ots, whale_txs, args.window_hours, args.min_flow_eth)
        nb = news_bias_at(ots, t["symbol"], news, args.window_hours)
        t["whale_regime"] = wf["regime"]
        t["whale_net_in_eth"] = wf["net_in_eth"]
        t["news_bias"] = nb["bias"]
        t["news_count"] = nb["count"]

    # Distribution diagnostic
    from collections import Counter
    wd = Counter(t["whale_regime"] for t in trades)
    nd = Counter(t["news_bias"] for t in trades)
    print(f"\n  whale-regime distribution: {dict(wd)}")
    print(f"  news-bias distribution:    {dict(nd)}")

    # Run all filters
    filters = [
        ("baseline (no filter)", lambda t: False),
        ("W1: block distribution",       w1_block_distribution),
        ("W2: only accumulation",        w2_only_accumulation),
        ("W3: block LONG during distrib", w3_block_long_during_distrib),
        ("N1: block bearish news",       n1_block_bearish_news),
        ("N2: only neutral news",        n2_only_neutral_news),
        ("N3: block direction-misaligned news", n3_block_misaligned_news),
        ("C1: block distrib AND bearish news", c1_block_distrib_and_bearish),
        ("C3: block distrib OR bearish news",  c3_block_distrib_or_bearish),
        ("C4: only accum any OR balanced+bullish", c4_accum_any_or_balanced_bullish),
    ]
    results = []
    print(f"\n{'filter':<46} {'allowed':>8} {'blocked':>8} {'allow_WR':>9} {'$/trade':>9} {'total$':>9} {'Δ vs base':>10}")
    print("-" * 110)
    for name, fn in filters:
        r = evaluate(name, trades, fn)
        delta = r["pnl_allowed"] - base_pnl
        print(f"  {r['name']:<44} {r['n_allowed']:>8} {r['n_blocked']:>8} "
              f"{r['wr_allowed']:>8.1f}% ${r['avg_pnl_allowed']:>+7.2f} "
              f"${r['pnl_allowed']:>+7.2f} ${delta:>+8.2f}")
        results.append({**r, "delta_vs_base": delta})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": args.window_hours,
        "min_flow_eth": args.min_flow_eth,
        "n_trades": len(trades),
        "baseline_wr": round(base_wr, 2),
        "baseline_pnl": round(base_pnl, 2),
        "results": results,
    }, indent=2))
    print(f"\nWrote machine-readable results to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
