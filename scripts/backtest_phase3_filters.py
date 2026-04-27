#!/usr/bin/env python3
"""
Phase 3 filter backtest — evaluates the four candidates surfaced by the
2026-04-27 deep-dive agents:

  F1  Block LONG when funding rate < 0
  F2  Block any entry when whale.direction == NEUTRAL
  F3  Tighten ADX_GUARD_MIN  20 → 25 (skip entries with ADX < 25)
  F4  Confidence ≥ 0.72 floor for SHORTs (potential replacement for the
      blunt SYMBOL_SIDE_BLOCKLIST)

Each filter is evaluated:
  * standalone (Δ vs no-filter baseline)
  * incremental on top of the currently-deployed filter set
    (SYMBOL_SIDE_BLOCKLIST + USDT.D + ADX>60 + stagnant band)
  * F4 is also evaluated as a replacement for SYMBOL_SIDE_BLOCKLIST

Source: paired OPEN/CLOSE rows from data/trading.db with signal context
joined from logs/htf_pending_alerts.jsonl. Same harness as
backtest_signal_filters.py.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "trading.db"
ALERTS = REPO / "logs" / "htf_pending_alerts.jsonl"

CURRENT_BLOCKLIST = {("BTCUSDT", "SHORT"), ("ETHUSDT", "SHORT"), ("ETHUSDT", "LONG"), ("SOLUSDT", "LONG")}


def load_open_alerts() -> list[dict]:
    out = []
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
            sigs = d.get("signals", {})
            out.append({
                "ts": d["timestamp"][:19],
                "symbol": t.get("symbol"),
                "side": "LONG" if "LONG" in action else "SHORT",
                "confidence": float(t.get("confidence", 0)),
                "regime_type": (sigs.get("regime") or {}).get("type", "UNKNOWN"),
                "adx": (sigs.get("regime") or {}).get("adx") or 0,
                "mtf_bias": ((sigs.get("mtf") or {}).get("bias") or "NEUTRAL").upper(),
                "of_bias": ((sigs.get("order_flow") or {}).get("bias") or "neutral").lower(),
                "of_score": (sigs.get("order_flow") or {}).get("score") or 0.0,
                "ob_bias": ((sigs.get("orderbook") or {}).get("bias") or "neutral").lower(),
                "ob_imbalance": (sigs.get("orderbook") or {}).get("imbalance_10") or 0.0,
                "whale_dir": ((sigs.get("whale") or {}).get("direction") or "NEUTRAL").upper(),
                "funding_rate": (sigs.get("funding") or {}).get("rate") or 0.0,
            })
    return out


def load_closed() -> list[dict]:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""SELECT id, timestamp, symbol, action, price, pnl, reason
                   FROM trades WHERE timestamp >= '2026-04-06' AND is_testnet=1 ORDER BY timestamp""")
    open_pos: dict[str, dict] = {}
    rt = []
    for tid, ts, sym, action, price, pnl, reason in cur.fetchall():
        if "OPEN" in action and "PARTIAL" not in action:
            open_pos[sym] = {"open_ts": ts, "side": "LONG" if "LONG" in action else "SHORT", "entry": price}
        elif "CLOSE" in action and "PARTIAL" not in action and sym in open_pos:
            o = open_pos.pop(sym)
            rt.append({
                "open_ts": o["open_ts"][:19],
                "close_ts": ts[:19],
                "symbol": sym,
                "side": o["side"],
                "entry": o["entry"],
                "exit": price,
                "pnl": pnl or 0.0,
                "reason": reason or "",
            })
    conn.close()
    return rt


def join(rts: list[dict], alerts: list[dict]) -> list[dict]:
    by_key = {(a["ts"], a["symbol"], a["side"]): a for a in alerts}
    joined = []
    for rt in rts:
        a = by_key.get((rt["open_ts"], rt["symbol"], rt["side"]))
        if a is None:
            cand = [
                aa for aa in alerts
                if aa["symbol"] == rt["symbol"] and aa["side"] == rt["side"]
                and abs((datetime.fromisoformat(aa["ts"]) - datetime.fromisoformat(rt["open_ts"])).total_seconds()) < 60
            ]
            a = cand[0] if cand else None
        if a is not None:
            joined.append({**rt, **{f"sig_{k}": v for k, v in a.items() if k not in ("ts", "symbol", "side")}})
    return joined


# ---------- filters ----------
def f_blocklist(t):  # current deployed blocklist
    return (t["symbol"], t["side"]) in CURRENT_BLOCKLIST


def f_funding_long(t):
    """F1: block LONG when funding rate < 0 (paying-shorts regime)."""
    return t["side"] == "LONG" and t.get("sig_funding_rate", 0) < 0


def f_whale_neutral(t):
    """F2: block when whale.direction is NEUTRAL (regime-stress signal)."""
    return t.get("sig_whale_dir", "NEUTRAL") == "NEUTRAL"


def f_adx_min_25(t):
    """F3: tighter ADX_GUARD_MIN — block when ADX < 25 (was < 20)."""
    return t.get("sig_adx", 0) < 25


def f_short_conf_72(t):
    """F4: confidence ≥ 0.72 floor for SHORT trades (replaces blocklist)."""
    return t["side"] == "SHORT" and t.get("sig_confidence", 0) < 0.72


def evaluate(name: str, joined: list[dict], blockers: list[Callable]) -> dict:
    blocked, allowed = [], []
    for t in joined:
        if any(b(t) for b in blockers):
            blocked.append(t)
        else:
            allowed.append(t)
    pnl_blocked = sum(t["pnl"] for t in blocked)
    pnl_allowed = sum(t["pnl"] for t in allowed)
    delta = -pnl_blocked
    wr_allowed = sum(1 for t in allowed if t["pnl"] > 0) / len(allowed) * 100 if allowed else 0
    wr_blocked = sum(1 for t in blocked if t["pnl"] > 0) / len(blocked) * 100 if blocked else 0
    return {
        "name": name,
        "n_blocked": len(blocked), "n_allowed": len(allowed),
        "pnl_blocked": pnl_blocked, "pnl_allowed": pnl_allowed,
        "delta": delta,
        "wr_allowed": wr_allowed, "wr_blocked": wr_blocked,
    }


def main() -> int:
    alerts = load_open_alerts()
    rts = load_closed()
    joined = join(rts, alerts)
    print(f"Joined {len(joined)} round-trips with signals.")
    base_pnl = sum(t["pnl"] for t in joined)
    base_w = sum(1 for t in joined if t["pnl"] > 0)
    print(f"Baseline (no filter): ${base_pnl:+.2f} over {len(joined)} closes, "
          f"WR {base_w / len(joined) * 100:.1f}%\n")

    print(f"{'name':<60} {'blocked':>8} {'allowed':>8} {'pnl_blkd':>10} {'Δ':>9} {'wr_allow':>9} {'wr_blkd':>9}")
    print("-" * 120)

    # --- Standalone vs no-filter baseline ---
    print("STANDALONE (vs. no-filter baseline):")
    for name, b in [
        ("blocklist (currently deployed)", [f_blocklist]),
        ("F1: block LONG when funding<0",  [f_funding_long]),
        ("F2: block whale=NEUTRAL",         [f_whale_neutral]),
        ("F3: ADX_GUARD_MIN 20→25",         [f_adx_min_25]),
        ("F4: SHORT conf>=0.72",            [f_short_conf_72]),
    ]:
        r = evaluate(name, joined, b)
        print(f"  {r['name']:<58} {r['n_blocked']:>8} {r['n_allowed']:>8} "
              f"${r['pnl_blocked']:>+8.2f} ${r['delta']:>+7.2f} {r['wr_allowed']:>8.1f}% {r['wr_blocked']:>8.1f}%")

    # --- Incremental on top of current deployed (blocklist) ---
    print("\nINCREMENTAL on top of currently-deployed blocklist:")
    for name, extra in [
        ("blocklist + F1 (funding<0 LONG)", [f_funding_long]),
        ("blocklist + F2 (whale=NEUTRAL)",  [f_whale_neutral]),
        ("blocklist + F3 (ADX>=25)",        [f_adx_min_25]),
        ("blocklist + F1+F2",               [f_funding_long, f_whale_neutral]),
        ("blocklist + F1+F2+F3",            [f_funding_long, f_whale_neutral, f_adx_min_25]),
    ]:
        r = evaluate(name, joined, [f_blocklist] + extra)
        print(f"  {r['name']:<58} {r['n_blocked']:>8} {r['n_allowed']:>8} "
              f"${r['pnl_blocked']:>+8.2f} ${r['delta']:>+7.2f} {r['wr_allowed']:>8.1f}% {r['wr_blocked']:>8.1f}%")

    # --- F4 replacement test: drop blocklist, use SHORT conf≥0.72 instead ---
    print("\nREPLACEMENT TEST — F4 SHORT conf≥0.72 instead of blocklist:")
    r_blocklist = evaluate("blocklist baseline (deployed)", joined, [f_blocklist])
    r_f4 = evaluate("F4 alone (no blocklist)", joined, [f_short_conf_72])
    r_f4_combo = evaluate("F4 + LONG-only-blocklist (drop SHORT entries from blocklist)",
                          joined,
                          [lambda t: t["side"] == "LONG" and (t["symbol"], "LONG") in CURRENT_BLOCKLIST,
                           f_short_conf_72])
    for r in (r_blocklist, r_f4, r_f4_combo):
        print(f"  {r['name']:<58} {r['n_blocked']:>8} {r['n_allowed']:>8} "
              f"${r['pnl_blocked']:>+8.2f} ${r['delta']:>+7.2f} {r['wr_allowed']:>8.1f}% {r['wr_blocked']:>8.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
