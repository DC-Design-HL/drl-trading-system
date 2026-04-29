#!/usr/bin/env python3
"""
Backtest: current production logic + new "skip LONG after extreme-positive
news" filter.

Rigorous setup:
  * "Current logic" = the trades that ALREADY survived every deployed
    filter at open time (blocklist + USDT.D + ADX>60 + funding-LONG +
    whale-NEUTRAL + orderbook + anti-whipsaw + RSI + structure-first
    direction). These are the trades the bot actually took.
  * "+ new filter" = of those, which would have been ADDITIONALLY blocked
    by: any news_events row in the trailing 4h with sentiment_score > 0.5,
    asset-matched to the trade symbol (or untagged/global news).
  * Delta = pnl_blocked × -1 (positive delta = losses prevented).

Constraints we honestly disclose:
  * news_events table only covers 2026-04-21 → 2026-04-29 (~8 days).
  * Trades from Apr 6 → Apr 21 have no news context — the filter cannot
    fire for them. So the effective sample is the ~Apr 21+ window.
  * The agent's pure-correlation finding (ext_pos fades, p=2.5e-6 at
    60m, Bonferroni-OK) was on the same 8-day news window, so this
    backtest is testing on the same data the signal was discovered on
    — moderate overfit risk. Validate again after 2 more weeks.

Run:
    python3 scripts/backtest_extreme_pos_news_filter.py
        [--threshold 0.5]      # ext_pos sentiment cutoff
        [--window_hours 4]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "trading.db"


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
            rt.append({**o, "symbol": sym, "close_ts": ts, "exit": price,
                       "pnl": pnl or 0.0, "reason": reason or "",
                       "win": 1 if (pnl or 0) > 0 else 0})
    conn.close()
    return rt


def load_news() -> list[tuple]:
    """Return (datetime_utc, sentiment_score, urgency, asset_tags_list)."""
    if not DB.exists():
        return []
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""SELECT published_at, sentiment_score, urgency, assets, event_type, title
                   FROM news_events WHERE sentiment_score IS NOT NULL""")
    out = []
    for ts, sent, urg, assets, et, title in cur.fetchall():
        try:
            dt = parse_iso(ts)
        except Exception:
            continue
        try:
            tags = json.loads(assets) if assets else []
        except Exception:
            tags = []
        out.append((dt, float(sent or 0), int(urg or 1), tags or [],
                    (et or "").lower(), title or ""))
    conn.close()
    return out


def has_ext_pos_news(open_dt: datetime, symbol: str, news, threshold: float,
                    window_hours: int) -> tuple[bool, dict]:
    """True if any news_event in trailing window has sentiment > threshold AND
    is either asset-tagged to symbol's base or untagged/global.
    """
    asset = symbol.replace("USDT", "").upper()
    win_start = open_dt - timedelta(hours=window_hours)
    matches = []
    for dt, sent, urg, tags, et, title in news:
        if dt < win_start or dt > open_dt:
            continue
        if sent <= threshold:
            continue
        if tags and not any(asset in (t or "").upper() for t in tags):
            continue
        matches.append({"ts": dt.isoformat(), "sentiment": sent,
                        "urgency": urg, "event_type": et, "title": title[:60]})
    return (len(matches) > 0), {"matches": matches, "match_count": len(matches)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sentiment_score cutoff for 'extreme positive'")
    p.add_argument("--window_hours", type=int, default=4)
    args = p.parse_args(argv)

    print(f"Settings: ext_pos sentiment > {args.threshold}, lookback {args.window_hours}h")

    trades = load_closed_trades()
    news = load_news()
    print(f"\n{len(trades)} closed trades; {len(news)} news with sentiment.")
    if news:
        ts_min = min(n[0] for n in news)
        ts_max = max(n[0] for n in news)
        print(f"News window: {ts_min.isoformat()[:19]} → {ts_max.isoformat()[:19]}")

    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]
    print(f"\nBaseline (trades that survived current filters):")
    print(f"  ALL:    n={len(trades)}, wins={sum(t['win'] for t in trades)}, "
          f"WR={sum(t['win'] for t in trades) / len(trades) * 100:.1f}%, "
          f"pnl=${sum(t['pnl'] for t in trades):+.2f}")
    print(f"  LONGs:  n={len(longs)}, wins={sum(t['win'] for t in longs)}, "
          f"WR={sum(t['win'] for t in longs) / max(1, len(longs)) * 100:.1f}%, "
          f"pnl=${sum(t['pnl'] for t in longs):+.2f}")
    print(f"  SHORTs: n={len(shorts)}, wins={sum(t['win'] for t in shorts)}, "
          f"WR={sum(t['win'] for t in shorts) / max(1, len(shorts)) * 100:.1f}%, "
          f"pnl=${sum(t['pnl'] for t in shorts):+.2f}")

    # Apply new filter: only affects LONGs
    blocked_longs = []
    allowed_longs = []
    for t in longs:
        try:
            open_dt = parse_iso(t["open_ts"])
        except Exception:
            allowed_longs.append(t)
            continue
        flag, info = has_ext_pos_news(open_dt, t["symbol"], news,
                                      args.threshold, args.window_hours)
        if flag:
            blocked_longs.append({**t, "_news_match": info})
        else:
            allowed_longs.append(t)

    print(f"\n+ Skip LONG when ext_pos news in trailing {args.window_hours}h "
          f"(sentiment > {args.threshold}):")
    print(f"  LONGs blocked:  {len(blocked_longs)}")
    print(f"  LONGs allowed:  {len(allowed_longs)}")
    if blocked_longs:
        b_pnl = sum(t["pnl"] for t in blocked_longs)
        b_wins = sum(t["win"] for t in blocked_longs)
        a_pnl = sum(t["pnl"] for t in allowed_longs)
        a_wins = sum(t["win"] for t in allowed_longs)
        print(f"    blocked pnl: ${b_pnl:+.2f}  ({b_wins}/{len(blocked_longs)} wins, "
              f"WR {b_wins / len(blocked_longs) * 100:.1f}%)")
        print(f"    allowed pnl: ${a_pnl:+.2f}  ({a_wins}/{len(allowed_longs)} wins, "
              f"WR {a_wins / len(allowed_longs) * 100:.1f}%)")
        print(f"    delta: ${-b_pnl:+.2f}  (positive = losses prevented)")

        # Per-symbol breakdown of blocked
        per_sym = defaultdict(lambda: {"n": 0, "pnl": 0, "wins": 0})
        for t in blocked_longs:
            per_sym[t["symbol"]]["n"] += 1
            per_sym[t["symbol"]]["pnl"] += t["pnl"]
            per_sym[t["symbol"]]["wins"] += t["win"]
        print(f"\n  Per-symbol breakdown of blocked LONGs:")
        for sym, d in sorted(per_sym.items()):
            print(f"    {sym}: n={d['n']}, pnl=${d['pnl']:+.2f}, "
                  f"WR={d['wins'] / d['n'] * 100:.1f}%")

        # Sample of what got blocked
        print(f"\n  Sample of blocked trades (first 5):")
        for t in blocked_longs[:5]:
            m = t["_news_match"]["matches"][0]
            print(f"    {t['open_ts'][:19]}  {t['symbol']:8}  pnl=${t['pnl']:+7.2f}  "
                  f"news: {m['ts'][:19]} sent={m['sentiment']:.2f} ({m['event_type']}) "
                  f"{m['title']}")

    # Effective evaluation window (trades with news coverage)
    if news:
        cutoff = min(n[0] for n in news)
        evaluable = [t for t in longs if parse_iso(t["open_ts"]) >= cutoff]
        print(f"\nEFFECTIVE evaluation window (LONGs after news data starts):")
        print(f"  evaluable LONGs: {len(evaluable)} of {len(longs)} total")
        if evaluable:
            blocked_in_window = [b for b in blocked_longs
                                 if parse_iso(b["open_ts"]) >= cutoff]
            allowed_in_window = [t for t in evaluable if t not in blocked_in_window]
            ev_pnl = sum(t["pnl"] for t in evaluable)
            blk_pnl = sum(t["pnl"] for t in blocked_in_window)
            print(f"  baseline pnl on evaluable LONGs: ${ev_pnl:+.2f}")
            print(f"  blocked: {len(blocked_in_window)} trades, ${blk_pnl:+.2f}")
            print(f"  delta: ${-blk_pnl:+.2f}")
            if blocked_in_window:
                print(f"  block rate: {len(blocked_in_window) / len(evaluable) * 100:.1f}% of LONGs")

    # Summary
    summary = {
        "settings": {"sentiment_threshold": args.threshold,
                     "window_hours": args.window_hours},
        "n_trades_total": len(trades),
        "n_longs": len(longs),
        "n_blocked_longs": len(blocked_longs),
        "blocked_pnl": float(sum(t["pnl"] for t in blocked_longs)) if blocked_longs else 0.0,
        "delta_pnl": float(-sum(t["pnl"] for t in blocked_longs)) if blocked_longs else 0.0,
    }
    out_path = REPO / "data" / "training" / "ext_pos_news_filter_backtest.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
