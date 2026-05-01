#!/usr/bin/env python3
"""
Exit-leakage forensic analysis.

Per project_drl_system_review_apr22.md, the SL exits ARE the edge
(+$774 / 149 trades / 84% WR), but model-driven REVERSE_CLOSE exits
were leaking -$297 combined at the time of review. The asymmetric
REVERSE_CLOSE_LONG canary was deployed Apr 23 for XRP only.

This script:
  1. Reports realized pnl per exit reason since 2026-04-06
  2. Compares pre-canary (Apr 6 → Apr 23) vs post-canary (Apr 23 → today)
     specifically for REVERSE_CLOSE_LONG to validate the deployed fix
  3. Identifies any other systematically losing exit reasons
  4. Computes per-symbol per-reason breakdown to find the worst offenders
  5. Proposes the next-priority exit fix based on dollar magnitude
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "trading.db"

CANARY_DEPLOY_TS = "2026-04-23T09:29:00"  # per memory
WINDOW_START = "2026-04-06"


def load_round_trips() -> list[dict]:
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("""SELECT id, timestamp, symbol, action, price, pnl, reason, data
                   FROM trades WHERE timestamp >= ? AND is_testnet=1
                   ORDER BY timestamp""", (WINDOW_START,))
    open_pos: dict[str, dict] = {}
    out = []
    for tid, ts, sym, action, price, pnl, reason, data in cur.fetchall():
        if "OPEN" in action and "PARTIAL" not in action:
            open_pos[sym] = {
                "open_ts": ts, "side": "LONG" if "LONG" in action else "SHORT",
                "entry": price, "open_id": tid,
            }
        elif "CLOSE" in action and "PARTIAL" not in action and sym in open_pos:
            o = open_pos.pop(sym)
            out.append({
                **o, "symbol": sym, "close_ts": ts, "exit": price,
                "pnl": pnl or 0.0, "reason": reason or "UNKNOWN",
                "win": 1 if (pnl or 0) > 0 else 0,
            })
    conn.close()
    return out


def fmt_table(rows: list[tuple], headers: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [tuple(headers)] + rows) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    out = []
    out.append("  ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers)))
    out.append(sep)
    for r in rows:
        out.append("  ".join(f"{str(c):<{widths[i]}}" for i, c in enumerate(r)))
    return "\n".join(out)


def main():
    rts = load_round_trips()
    print(f"Closed round-trips since {WINDOW_START}: {len(rts)}\n")

    # ── Aggregate by exit reason ────────────────────────────────────────
    by_reason = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0, "longs": 0, "shorts": 0})
    for t in rts:
        r = by_reason[t["reason"]]
        r["n"] += 1
        r["pnl"] += t["pnl"]
        r["wins"] += t["win"]
        if t["side"] == "LONG": r["longs"] += 1
        else: r["shorts"] += 1
    rows = []
    for reason, d in sorted(by_reason.items(), key=lambda x: x[1]["pnl"]):
        rows.append((
            reason,
            d["n"],
            f"{d['wins']/d['n']*100:.1f}%",
            f"${d['pnl']:+.2f}",
            f"${d['pnl']/d['n']:+.2f}",
            f"{d['longs']}/{d['shorts']}",
        ))
    print("EXITS BY REASON (sorted by total pnl, worst first):")
    print(fmt_table(rows, ["reason", "n", "WR", "total_pnl", "$/trade", "L/S"]))

    # ── Side-specific REVERSE_CLOSE breakdown ────────────────────────────
    print("\n\nREVERSE_CLOSE — side breakdown (memory finding: LONG leaks, SHORT works):")
    rc_data = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in rts:
        if "REVERSE_CLOSE" not in t["reason"]:
            continue
        rc_data[t["side"]]["n"] += 1
        rc_data[t["side"]]["pnl"] += t["pnl"]
        rc_data[t["side"]]["wins"] += t["win"]
    rc_rows = []
    for side in ("LONG", "SHORT"):
        d = rc_data[side]
        if d["n"] == 0: continue
        rc_rows.append((
            side, d["n"], f"{d['wins']/d['n']*100:.1f}%",
            f"${d['pnl']:+.2f}", f"${d['pnl']/d['n']:+.2f}",
        ))
    print(fmt_table(rc_rows, ["side", "n", "WR", "total_pnl", "$/trade"]))

    # ── REVERSE_CLOSE_LONG canary validation: pre vs post Apr 23 09:29 ──
    print("\n\nREVERSE_CLOSE_LONG canary (deployed Apr 23 09:29 for XRPUSDT):")
    pre = []
    post = []
    for t in rts:
        if t["reason"] != "REVERSE_CLOSE_LONG":
            continue
        if t["close_ts"] < CANARY_DEPLOY_TS: pre.append(t)
        else: post.append(t)

    def summarize(label, ts):
        if not ts:
            print(f"  {label}: 0 trades")
            return
        pnl = sum(t["pnl"] for t in ts)
        wins = sum(t["win"] for t in ts)
        per_sym = defaultdict(lambda: [0, 0.0])
        for t in ts:
            per_sym[t["symbol"]][0] += 1
            per_sym[t["symbol"]][1] += t["pnl"]
        sym_str = " ".join(f"{s}:{n}/${p:.0f}" for s, (n, p) in per_sym.items())
        print(f"  {label}: {len(ts)} trades, {wins}/{len(ts)} wins ({wins/len(ts)*100:.0f}% WR), "
              f"pnl ${pnl:+.2f} (${pnl/len(ts):+.2f}/trade)  per-symbol: {sym_str}")

    summarize("Pre-canary  (Apr 6 → Apr 23)", pre)
    summarize("Post-canary (Apr 23 → today)", post)

    # If post-canary REVERSE_CLOSE_LONG happened on XRP, that's a problem (canary should block)
    post_xrp = [t for t in post if t["symbol"] == "XRPUSDT"]
    if post_xrp:
        print(f"  ⚠️  {len(post_xrp)} REVERSE_CLOSE_LONG events on XRPUSDT post-deploy — "
              f"check if BTC slope was below the -0.5% gate (allowing fall-through).")

    # ── Per-symbol-side worst losses ────────────────────────────────────
    print("\n\nWORST EXIT-REASON × (SYMBOL, SIDE) COMBINATIONS (top 10 by total loss):")
    worst = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in rts:
        key = (t["reason"], t["symbol"], t["side"])
        worst[key]["n"] += 1
        worst[key]["pnl"] += t["pnl"]
        worst[key]["wins"] += t["win"]
    worst_rows = []
    for (reason, sym, side), d in sorted(worst.items(), key=lambda x: x[1]["pnl"])[:10]:
        worst_rows.append((
            reason, sym, side, d["n"],
            f"{d['wins']/d['n']*100:.0f}%",
            f"${d['pnl']:+.2f}",
            f"${d['pnl']/d['n']:+.2f}",
        ))
    print(fmt_table(worst_rows, ["reason", "symbol", "side", "n", "WR", "total_pnl", "$/trade"]))

    # ── Stagnant exit deep-dive ─────────────────────────────────────────
    print("\n\nSTAGNANT_EXIT deep-dive (current band: [-1.0%, +0.5%], deployed Apr 25):")
    stag = [t for t in rts if t["reason"] == "STAGNANT_EXIT"]
    if stag:
        pnl = sum(t["pnl"] for t in stag)
        wins = sum(t["win"] for t in stag)
        # Distribution of pnl
        sorted_pnls = sorted(t["pnl"] for t in stag)
        print(f"  n={len(stag)}, wins={wins} ({wins/len(stag)*100:.1f}% WR), "
              f"total ${pnl:+.2f}, avg ${pnl/len(stag):+.2f}/trade")
        print(f"  pnl range: min ${sorted_pnls[0]:+.2f}, p25 ${sorted_pnls[len(stag)//4]:+.2f}, "
              f"median ${sorted_pnls[len(stag)//2]:+.2f}, p75 ${sorted_pnls[3*len(stag)//4]:+.2f}, "
              f"max ${sorted_pnls[-1]:+.2f}")
        # Side breakdown
        s_long = [t for t in stag if t["side"] == "LONG"]
        s_short = [t for t in stag if t["side"] == "SHORT"]
        if s_long:
            print(f"  LONG  : n={len(s_long)}, "
                  f"WR {sum(t['win'] for t in s_long)/len(s_long)*100:.1f}%, "
                  f"pnl ${sum(t['pnl'] for t in s_long):+.2f}")
        if s_short:
            print(f"  SHORT : n={len(s_short)}, "
                  f"WR {sum(t['win'] for t in s_short)/len(s_short)*100:.1f}%, "
                  f"pnl ${sum(t['pnl'] for t in s_short):+.2f}")

    # ── SL exits — confirm they're still the edge ───────────────────────
    print("\n\nSL exits — confirm 'SL is the edge' finding (memory: 84% WR, +$774):")
    sl = [t for t in rts if t["reason"] == "SL"]
    if sl:
        wins = sum(t["win"] for t in sl)
        pnl = sum(t["pnl"] for t in sl)
        print(f"  n={len(sl)}, WR {wins/len(sl)*100:.1f}%, total ${pnl:+.2f}, ${pnl/len(sl):+.2f}/trade")
        s_long = [t for t in sl if t["side"] == "LONG"]
        s_short = [t for t in sl if t["side"] == "SHORT"]
        if s_long:
            print(f"  LONG  : n={len(s_long)}, "
                  f"WR {sum(t['win'] for t in s_long)/len(s_long)*100:.1f}%, "
                  f"pnl ${sum(t['pnl'] for t in s_long):+.2f}")
        if s_short:
            print(f"  SHORT : n={len(s_short)}, "
                  f"WR {sum(t['win'] for t in s_short)/len(s_short)*100:.1f}%, "
                  f"pnl ${sum(t['pnl'] for t in s_short):+.2f}")

    # ── Save machine-readable ───────────────────────────────────────────
    out_path = REPO / "data" / "training" / "exit_leakage_analysis.json"
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_round_trips": len(rts),
        "by_reason": dict(by_reason),
        "reverse_close_canary": {
            "pre": [{"sym": t["symbol"], "pnl": t["pnl"], "ts": t["close_ts"]} for t in pre],
            "post": [{"sym": t["symbol"], "pnl": t["pnl"], "ts": t["close_ts"]} for t in post],
        },
        "worst_combos": [
            {"reason": k[0], "symbol": k[1], "side": k[2], **dict(v)}
            for k, v in sorted(worst.items(), key=lambda x: x[1]["pnl"])[:20]
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
