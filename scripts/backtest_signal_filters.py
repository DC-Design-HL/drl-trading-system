#!/usr/bin/env python3
"""
Stacked-filter backtest using ENTRY-CONTEXT signals stored in
logs/htf_pending_alerts.jsonl (which has full signal payloads, unlike
the trades DB which only has price/pnl).

For each closed round-trip we know:
  * Direction (LONG/SHORT)
  * Confidence at open
  * MTF bias and aligned flag
  * Regime type + ADX
  * Order flow bias + score
  * Orderbook bias + imbalance
  * Funding rate
  * Whale direction

We then evaluate each candidate filter (and combinations) by computing:
  * trades blocked / allowed
  * Δ pnl vs no-filter baseline
  * allowed-set win rate
  * worst-case retroactive drawdown impact

Filter candidates tested:
  F1  Counter-trend block  : skip SHORT in TRENDING_UP (ADX>SIGNAL_GATE_REGIME_ADX_MIN)
                             skip LONG  in TRENDING_DOWN (ADX>min)
  F2  MTF alignment        : require MTF aligned=True (or bias matches direction)
  F3  Confidence floor     : skip if confidence < threshold (0.65, 0.70, 0.75)
  F4  Off-hours block      : skip 21:00-08:00 UTC opens
  F5  Orderbook agreement  : require orderbook bias = direction OR neutral
  F6  Order-flow agreement : require OF bias = direction OR neutral

Plus stacked combinations of the most promising.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "trading.db"
ALERTS = REPO / "logs" / "htf_pending_alerts.jsonl"


def load_open_alerts() -> list[dict]:
    """Yield {ts, symbol, action, confidence, signals dict} for every OPEN entry alert."""
    out = []
    with open(ALERTS) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("trade", {})
            action = t.get("action", "")
            if not action.startswith("OPEN"):
                continue
            if "PARTIAL" in action:
                continue
            sigs = d.get("signals", {})
            out.append({
                "ts": d["timestamp"][:19],
                "symbol": t.get("symbol"),
                "side": "LONG" if "LONG" in action else "SHORT",
                "confidence": float(t.get("confidence", 0)),
                "regime_type": (sigs.get("regime", {}) or {}).get("type", "UNKNOWN"),
                "adx": (sigs.get("regime", {}) or {}).get("adx") or 0,
                "mtf_bias": ((sigs.get("mtf", {}) or {}).get("bias") or "NEUTRAL").upper(),
                "mtf_aligned": (sigs.get("mtf", {}) or {}).get("aligned", False),
                "of_bias": ((sigs.get("order_flow", {}) or {}).get("bias") or "neutral").lower(),
                "of_score": (sigs.get("order_flow", {}) or {}).get("score") or 0.0,
                "ob_bias": ((sigs.get("orderbook", {}) or {}).get("bias") or "neutral").lower(),
                "ob_imbalance": (sigs.get("orderbook", {}) or {}).get("imbalance_10") or 0.0,
                "whale_dir": ((sigs.get("whale", {}) or {}).get("direction") or "NEUTRAL").upper(),
                "funding_rate": (sigs.get("funding", {}) or {}).get("rate") or 0.0,
            })
    return out


def load_closed_round_trips() -> list[dict]:
    """Pair OPEN/CLOSE rows from DB to build closed-trade history with pnl."""
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, timestamp, symbol, action, price, pnl, reason "
        "FROM trades WHERE timestamp >= '2026-04-06' AND is_testnet=1 ORDER BY timestamp"
    )
    open_pos: dict[str, dict] = {}
    rt = []
    for tid, ts, sym, action, price, pnl, reason in cur.fetchall():
        if "OPEN" in action and "PARTIAL" not in action:
            open_pos[sym] = {"open_ts": ts, "side": "LONG" if "LONG" in action else "SHORT", "entry": price, "open_id": tid}
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


def join_signals(round_trips: list[dict], alerts: list[dict]) -> list[dict]:
    """Match each round-trip to its OPEN alert by (ts, symbol, side)."""
    by_key = {(a["ts"], a["symbol"], a["side"]): a for a in alerts}
    joined = []
    for rt in round_trips:
        key = (rt["open_ts"], rt["symbol"], rt["side"])
        a = by_key.get(key)
        if a is None:
            # Try fuzzier match — alert ts can be a few seconds off from DB ts
            cand = [
                aa for aa in alerts
                if aa["symbol"] == rt["symbol"]
                and aa["side"] == rt["side"]
                and abs((datetime.fromisoformat(aa["ts"]) - datetime.fromisoformat(rt["open_ts"])).total_seconds()) < 60
            ]
            a = cand[0] if cand else None
        if a is None:
            continue
        joined.append({**rt, **{f"sig_{k}": v for k, v in a.items() if k not in ("ts", "symbol", "side")}})
    return joined


# ---------- filters ----------

def f1_counter_trend(t: dict, adx_min: float = 25.0) -> bool:
    """True = block. SHORT in trending-up or LONG in trending-down."""
    rt = t.get("sig_regime_type", "")
    adx = t.get("sig_adx", 0)
    if adx < adx_min:
        return False  # ranging — don't gate on regime
    if rt == "TRENDING_UP" and t["side"] == "SHORT":
        return True
    if rt == "TRENDING_DOWN" and t["side"] == "LONG":
        return True
    return False


def f2_require_mtf_aligned(t: dict) -> bool:
    """True = block. Require mtf bias agrees with direction."""
    bias = t.get("sig_mtf_bias", "NEUTRAL")
    if bias == "NEUTRAL":
        return False  # let neutral through
    if t["side"] == "LONG" and bias != "BULLISH":
        return True
    if t["side"] == "SHORT" and bias != "BEARISH":
        return True
    return False


def f3_confidence_floor(t: dict, floor: float) -> bool:
    """True = block. Skip if confidence below floor."""
    return t.get("sig_confidence", 0) < floor


def f4_off_hours(t: dict) -> bool:
    """True = block. Skip 21:00-08:00 UTC."""
    h = datetime.fromisoformat(t["open_ts"]).hour
    return h in set(list(range(21, 24)) + list(range(0, 8)))


def f5_orderbook_agrees(t: dict) -> bool:
    """True = block. Require orderbook bias = direction OR neutral."""
    bias = t.get("sig_ob_bias", "neutral")
    if bias == "neutral":
        return False
    if t["side"] == "LONG" and bias != "bullish":
        return True
    if t["side"] == "SHORT" and bias != "bearish":
        return True
    return False


def f6_order_flow_agrees(t: dict) -> bool:
    """True = block. Require OF bias = direction OR neutral."""
    bias = t.get("sig_of_bias", "neutral")
    if bias == "neutral":
        return False
    if t["side"] == "LONG" and bias != "bullish":
        return True
    if t["side"] == "SHORT" and bias != "bearish":
        return True
    return False


def evaluate(name: str, joined: list[dict], blockers: list[callable]) -> dict:
    """Apply all blockers (any True → block). Return stats."""
    blocked = []
    allowed = []
    for t in joined:
        if any(b(t) for b in blockers):
            blocked.append(t)
        else:
            allowed.append(t)
    pnl_blocked = sum(t["pnl"] for t in blocked)
    pnl_allowed = sum(t["pnl"] for t in allowed)
    delta_vs_baseline = -pnl_blocked  # what we save (positive = saved losses)
    wr_allowed = sum(1 for t in allowed if t["pnl"] > 0) / len(allowed) * 100 if allowed else 0
    return {
        "name": name, "n_blocked": len(blocked), "n_allowed": len(allowed),
        "pnl_blocked": pnl_blocked, "pnl_allowed": pnl_allowed,
        "delta": delta_vs_baseline, "wr_allowed": wr_allowed,
    }


def main() -> int:
    alerts = load_open_alerts()
    rts = load_closed_round_trips()
    joined = join_signals(rts, alerts)
    print(f"Round-trips with signals matched: {len(joined)} / {len(rts)} (alerts: {len(alerts)})")

    base_pnl = sum(t["pnl"] for t in joined)
    base_wins = sum(1 for t in joined if t["pnl"] > 0)
    print(f"Baseline: ${base_pnl:+.2f} pnl over {len(joined)} closes, "
          f"{base_wins} wins ({base_wins / len(joined) * 100:.1f}% WR)\n")

    print(f"{'filter':<55} {'blocked':>8} {'allowed':>8} {'pnl_blkd':>10} {'Δ vs no':>10} {'allow_WR':>10}")
    print("-" * 110)

    # Single filters
    rows = [
        evaluate("F1: counter-trend block (ADX>=25)", joined, [f1_counter_trend]),
        evaluate("F2: MTF aligned (bias matches direction)", joined, [f2_require_mtf_aligned]),
        evaluate("F3a: conf >= 0.65", joined, [lambda t: f3_confidence_floor(t, 0.65)]),
        evaluate("F3b: conf >= 0.70", joined, [lambda t: f3_confidence_floor(t, 0.70)]),
        evaluate("F3c: conf >= 0.75", joined, [lambda t: f3_confidence_floor(t, 0.75)]),
        evaluate("F4: off-hours block (21-08 UTC)", joined, [f4_off_hours]),
        evaluate("F5: orderbook agrees", joined, [f5_orderbook_agrees]),
        evaluate("F6: order-flow agrees", joined, [f6_order_flow_agrees]),
    ]
    for r in rows:
        print(f"{r['name']:<55} {r['n_blocked']:>8} {r['n_allowed']:>8} "
              f"${r['pnl_blocked']:>+8.2f} ${r['delta']:>+8.2f} {r['wr_allowed']:>9.1f}%")

    # Stacked combinations (most promising)
    print("\n--- Stacked filters ---")
    stacks = [
        ("F1 + F2", [f1_counter_trend, f2_require_mtf_aligned]),
        ("F1 + F3b (conf>=0.70)", [f1_counter_trend, lambda t: f3_confidence_floor(t, 0.70)]),
        ("F1 + F2 + F3b", [f1_counter_trend, f2_require_mtf_aligned, lambda t: f3_confidence_floor(t, 0.70)]),
        ("F1 + F4", [f1_counter_trend, f4_off_hours]),
        ("F1 + F2 + F4", [f1_counter_trend, f2_require_mtf_aligned, f4_off_hours]),
        ("F1 + F2 + F4 + F3b", [f1_counter_trend, f2_require_mtf_aligned, f4_off_hours, lambda t: f3_confidence_floor(t, 0.70)]),
        ("F1 + F5", [f1_counter_trend, f5_orderbook_agrees]),
        ("F2 + F5 + F6", [f2_require_mtf_aligned, f5_orderbook_agrees, f6_order_flow_agrees]),
    ]
    for name, blockers in stacks:
        r = evaluate(name, joined, blockers)
        print(f"{name:<55} {r['n_blocked']:>8} {r['n_allowed']:>8} "
              f"${r['pnl_blocked']:>+8.2f} ${r['delta']:>+8.2f} {r['wr_allowed']:>9.1f}%")

    # Detailed look at worst-case filters
    print("\n--- Per-symbol baseline pnl ---")
    by_sym: dict[str, list[float]] = defaultdict(list)
    for t in joined:
        by_sym[t["symbol"]].append(t["pnl"])
    for s, ps in by_sym.items():
        wins = sum(1 for p in ps if p > 0)
        print(f"  {s}: n={len(ps)}, sum=${sum(ps):+.2f}, WR={wins / len(ps) * 100:.1f}%")

    print("\n--- Counter-trend (F1) breakdown ---")
    ct = [t for t in joined if f1_counter_trend(t)]
    nct = [t for t in joined if not f1_counter_trend(t)]
    print(f"  counter-trend trades: {len(ct)}, sum pnl ${sum(t['pnl'] for t in ct):+.2f}, "
          f"WR {sum(1 for t in ct if t['pnl'] > 0) / max(1, len(ct)) * 100:.1f}%")
    print(f"  trend-aligned trades: {len(nct)}, sum pnl ${sum(t['pnl'] for t in nct):+.2f}, "
          f"WR {sum(1 for t in nct if t['pnl'] > 0) / max(1, len(nct)) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
