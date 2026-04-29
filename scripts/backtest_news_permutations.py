#!/usr/bin/env python3
"""
Permutation backtest: every combination of the new ext-pos news filter
with each existing signal-derived filter rule.

Setup mirrors backtest_extreme_pos_news_filter.py:
  * Closed trades that already survived all live production filters
    (302 paired round-trips since 2026-04-06).
  * Signal context for each trade joined from logs/htf_pending_alerts.jsonl.
  * News context joined from data/trading.db news_events.

Filter library (each filter is a callable trade→bool, True = block):

  News-derived (the new family):
    N_extpos     skip LONG when news.sentiment > 0.5 in trailing 4h
    N_extneg     skip when news.sentiment < -0.5 in trailing 4h
    N_anyhighurg skip when news has urgency≥3 in trailing 4h
    N_anynews4h  skip when ≥1 news event in trailing 4h
    N_anynews1h  skip when ≥1 news event in trailing 1h

  Existing-deployed filters (replayed for sanity, baseline=production):
    B_blocklist  skip BTC-SHORT/ETH-LONG/ETH-SHORT/SOL-LONG
    B_funding    skip LONG when funding < 0
    B_whaleneut  skip when whale.direction = NEUTRAL

  Other signal candidates we haven't deployed:
    S_shortconf72 skip SHORT when confidence < 0.72
    S_longconf72  skip LONG  when confidence < 0.72
    S_offhours    skip 21:00-08:00 UTC opens
    S_ob_oppose   skip when ob.bias contradicts direction
    S_mtf_neutral skip when mtf.bias = NEUTRAL
    S_lowadx25    skip when ADX < 25
    S_highadx55   skip when ADX > 55

Then we evaluate:
  * Each filter standalone
  * Each filter combined with N_extpos (pairs)
  * Top-3 pair winners combined with each remaining filter (triples)
  * Anchor everything to delta_pnl vs the no-filter baseline

Output ranked tables. The trade DB is already conditioned on every
production filter at OPEN time, so the "baseline" here is the 306
trades the production system actually took.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from itertools import combinations

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "trading.db"
ALERTS = REPO / "logs" / "htf_pending_alerts.jsonl"


def parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "+" not in s and "-" not in s[-6:]:
        s += "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def load_trades_with_signals() -> list[dict]:
    # round-trip outcomes from DB
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

    # Index alerts by (open_ts, symbol, side)
    sig_index: dict[tuple, dict] = {}
    if ALERTS.exists():
        with open(ALERTS) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("trade", {})
                action = t.get("action", "")
                if not action.startswith("OPEN") or "PARTIAL" in action:
                    continue
                sigs = d.get("signals", {}) or {}
                key = (d["timestamp"][:19], t.get("symbol"),
                       "LONG" if "LONG" in action else "SHORT")
                sig_index[key] = {
                    "confidence": float(t.get("confidence") or 0),
                    "mtf_bias": ((sigs.get("mtf") or {}).get("bias") or "NEUTRAL").upper(),
                    "of_bias": ((sigs.get("order_flow") or {}).get("bias") or "neutral").lower(),
                    "regime_type": (sigs.get("regime") or {}).get("type") or "UNKNOWN",
                    "adx": float((sigs.get("regime") or {}).get("adx") or 0),
                    "ob_bias": ((sigs.get("orderbook") or {}).get("bias") or "neutral").lower(),
                    "funding_rate": float((sigs.get("funding") or {}).get("rate") or 0),
                    "whale_dir": ((sigs.get("whale") or {}).get("direction") or "NEUTRAL").upper(),
                }
    # join
    out = []
    for t in rt:
        key = (t["open_ts"][:19], t["symbol"], t["side"])
        sig = sig_index.get(key)
        if sig is None:
            # fuzzy match within 60s
            cands = [v for k, v in sig_index.items()
                     if k[1] == t["symbol"] and k[2] == t["side"]
                     and abs((parse_iso(k[0]) - parse_iso(t["open_ts"])).total_seconds()) < 60]
            sig = cands[0] if cands else None
        if sig is None:
            continue
        out.append({**t, **sig})
    return out


def load_news() -> list[tuple[datetime, float, int, list[str]]]:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""SELECT published_at, sentiment_score, urgency, assets
                   FROM news_events WHERE sentiment_score IS NOT NULL""")
    out = []
    for ts, sent, urg, assets in cur.fetchall():
        try:
            dt = parse_iso(ts)
        except Exception:
            continue
        try:
            tags = json.loads(assets) if assets else []
        except Exception:
            tags = []
        out.append((dt, float(sent or 0), int(urg or 1), tags or []))
    conn.close()
    return out


def annotate_news(trades: list[dict], news: list[tuple]) -> None:
    """Attach pre-computed news-window summaries to each trade in place."""
    for t in trades:
        try:
            o = parse_iso(t["open_ts"])
        except Exception:
            t["_news_evaluable"] = False
            continue
        asset = t["symbol"].replace("USDT", "").upper()
        for window_h in (1, 4):
            win_start = o - timedelta(hours=window_h)
            relevant = [
                (sent, urg) for dt, sent, urg, tags in news
                if win_start <= dt <= o
                and (not tags or any(asset in (tg or "").upper() for tg in tags))
            ]
            n = len(relevant)
            t[f"_news_{window_h}h_count"] = n
            if relevant:
                t[f"_news_{window_h}h_max_sent"] = max(s for s, _ in relevant)
                t[f"_news_{window_h}h_min_sent"] = min(s for s, _ in relevant)
                t[f"_news_{window_h}h_max_urg"] = max(u for _, u in relevant)
            else:
                t[f"_news_{window_h}h_max_sent"] = 0.0
                t[f"_news_{window_h}h_min_sent"] = 0.0
                t[f"_news_{window_h}h_max_urg"] = 0
        # Evaluable if news data was available at all in trailing 24h
        t["_news_evaluable"] = bool(news) and o >= news[0][0]


# ── Filter functions (True = block) ───────────────────────────────────────

BLOCKLIST = {("BTCUSDT", "SHORT"), ("ETHUSDT", "SHORT"),
             ("ETHUSDT", "LONG"), ("SOLUSDT", "LONG")}


def f_news_extpos(t):
    """skip LONG when sentiment > 0.5 in trailing 4h"""
    return t["side"] == "LONG" and t.get("_news_4h_max_sent", 0) > 0.5


def f_news_extneg(t):
    """skip when sentiment < -0.5 in trailing 4h"""
    return t.get("_news_4h_min_sent", 0) < -0.5


def f_news_anyhighurg(t):
    """skip when urgency >= 3 in trailing 4h"""
    return t.get("_news_4h_max_urg", 0) >= 3


def f_news_anynews4h(t):
    """skip when any news in trailing 4h (paranoid)"""
    return t.get("_news_4h_count", 0) > 0


def f_news_anynews1h(t):
    """skip when any news in trailing 1h"""
    return t.get("_news_1h_count", 0) > 0


def f_blocklist(t):
    return (t["symbol"], t["side"]) in BLOCKLIST


def f_funding_long(t):
    return t["side"] == "LONG" and t.get("funding_rate", 0) < 0


def f_whaleneut(t):
    return t.get("whale_dir") == "NEUTRAL"


def f_short_conf72(t):
    return t["side"] == "SHORT" and t.get("confidence", 0) < 0.72


def f_long_conf72(t):
    return t["side"] == "LONG" and t.get("confidence", 0) < 0.72


def f_offhours(t):
    h = parse_iso(t["open_ts"]).hour
    return h in set(list(range(21, 24)) + list(range(0, 8)))


def f_ob_oppose(t):
    if t["side"] == "LONG" and t.get("ob_bias") == "bearish":
        return True
    if t["side"] == "SHORT" and t.get("ob_bias") == "bullish":
        return True
    return False


def f_mtf_neutral(t):
    return t.get("mtf_bias") == "NEUTRAL"


def f_low_adx_25(t):
    return t.get("adx", 0) < 25


def f_high_adx_55(t):
    return t.get("adx", 0) > 55


FILTERS = {
    "N_extpos": f_news_extpos,
    "N_extneg": f_news_extneg,
    "N_highurg": f_news_anyhighurg,
    "N_any4h": f_news_anynews4h,
    "N_any1h": f_news_anynews1h,
    "B_blocklist": f_blocklist,
    "B_funding": f_funding_long,
    "B_whaleneut": f_whaleneut,
    "S_shortconf72": f_short_conf72,
    "S_longconf72": f_long_conf72,
    "S_offhours": f_offhours,
    "S_ob_oppose": f_ob_oppose,
    "S_mtf_neutral": f_mtf_neutral,
    "S_lowadx25": f_low_adx_25,
    "S_highadx55": f_high_adx_55,
}


def evaluate(name: str, trades: list[dict], blockers: list) -> dict:
    blocked = [t for t in trades if any(b(t) for b in blockers)]
    allowed = [t for t in trades if not any(b(t) for b in blockers)]
    pnl_b = sum(t["pnl"] for t in blocked)
    pnl_a = sum(t["pnl"] for t in allowed)
    n_a = len(allowed)
    wr_a = sum(t["win"] for t in allowed) / n_a * 100 if n_a else 0
    wr_b = sum(t["win"] for t in blocked) / len(blocked) * 100 if blocked else 0
    return {
        "name": name,
        "n_blocked": len(blocked), "n_allowed": n_a,
        "pnl_blocked": pnl_b, "pnl_allowed": pnl_a,
        "delta_pnl": -pnl_b,
        "wr_allowed": wr_a, "wr_blocked": wr_b,
    }


def fmt_row(r):
    return (f"  {r['name']:<40} blk={r['n_blocked']:>3} "
            f"allow={r['n_allowed']:>3} blkd_WR={r['wr_blocked']:>5.1f}% "
            f"allow_WR={r['wr_allowed']:>5.1f}% "
            f"blkd_pnl=${r['pnl_blocked']:>+8.2f} Δ=${r['delta_pnl']:>+8.2f}")


def main():
    trades = load_trades_with_signals()
    news = load_news()
    annotate_news(trades, news)
    # Limit to trades inside news coverage window for fair comparison
    if news:
        cutoff = min(n[0] for n in news)
        evaluable = [t for t in trades
                     if parse_iso(t["open_ts"]) >= cutoff]
    else:
        evaluable = trades
    print(f"Total round-trips with signals: {len(trades)}")
    print(f"Evaluable (news data exists): {len(evaluable)}")
    base = evaluate("baseline (no filter)", evaluable, [])
    print(f"\nBaseline on {base['n_allowed']} evaluable trades: "
          f"pnl=${base['pnl_allowed']:+.2f} WR={base['wr_allowed']:.1f}%\n")

    # ─── Standalone filters ─────────────────────────────────────────────
    print("=" * 100)
    print("STANDALONE FILTERS (single rule, evaluable trades only)")
    print("=" * 100)
    standalone = []
    for name, fn in FILTERS.items():
        r = evaluate(name, evaluable, [fn])
        standalone.append(r)
    standalone.sort(key=lambda x: -x["delta_pnl"])
    for r in standalone:
        print(fmt_row(r))

    # ─── Pairs anchored on N_extpos ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("PAIRS — N_extpos × every other filter (sorted by delta_pnl)")
    print("=" * 100)
    pairs = []
    for name, fn in FILTERS.items():
        if name == "N_extpos":
            continue
        r = evaluate(f"N_extpos + {name}", evaluable,
                     [FILTERS["N_extpos"], fn])
        pairs.append(r)
    pairs.sort(key=lambda x: -x["delta_pnl"])
    for r in pairs:
        print(fmt_row(r))

    # ─── All other pairs (sanity check, no news anchor) ───────────────────
    print("\n" + "=" * 100)
    print("TOP 20 PAIRS overall (any 2 filters, sorted by delta_pnl)")
    print("=" * 100)
    all_pairs = []
    keys = list(FILTERS.keys())
    for a, b in combinations(keys, 2):
        r = evaluate(f"{a} + {b}", evaluable, [FILTERS[a], FILTERS[b]])
        all_pairs.append(r)
    all_pairs.sort(key=lambda x: -x["delta_pnl"])
    for r in all_pairs[:20]:
        print(fmt_row(r))

    # ─── Triples anchored on N_extpos + best partner ─────────────────────
    print("\n" + "=" * 100)
    print("TRIPLES — N_extpos + (best 5 partners) × every remaining filter")
    print("=" * 100)
    best_partners = [r["name"].replace("N_extpos + ", "") for r in pairs[:5]]
    triples = []
    for partner in best_partners:
        for name, fn in FILTERS.items():
            if name in ("N_extpos", partner):
                continue
            r = evaluate(f"N_extpos + {partner} + {name}", evaluable,
                         [FILTERS["N_extpos"], FILTERS[partner], fn])
            triples.append(r)
    triples.sort(key=lambda x: -x["delta_pnl"])
    for r in triples[:15]:
        print(fmt_row(r))

    # ─── Save full results ───────────────────────────────────────────────
    out = REPO / "data" / "training" / "news_permutation_results.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_evaluable": len(evaluable),
        "baseline": base,
        "standalone": standalone,
        "pairs_with_extpos": pairs,
        "all_pairs_top20": all_pairs[:20],
        "triples_with_extpos_top15": triples[:15],
    }, indent=2, default=str))
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
