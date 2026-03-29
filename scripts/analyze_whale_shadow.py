#!/usr/bin/env python3
"""
Analyze whale shadow log — correlates whale SELL signals with trade outcomes.

Run after 1-2 weeks of shadow data collection:
    python scripts/analyze_whale_shadow.py

Reads: logs/whale_shadow.jsonl
Output: Correlation analysis between whale signals and trade P&L
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

SHADOW_FILE = Path("logs/whale_shadow.jsonl")


def load_shadow_data():
    if not SHADOW_FILE.exists():
        print(f"❌ No shadow data found at {SHADOW_FILE}")
        print("   Trades need to happen first. Check back after a few days.")
        sys.exit(0)

    records = []
    with open(SHADOW_FILE) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return records


def analyze():
    records = load_shadow_data()
    if not records:
        print("No records to analyze")
        return

    print(f"📊 Whale Shadow Analysis — {len(records)} trade events")
    print(f"   Period: {records[0]['timestamp'][:10]} → {records[-1]['timestamp'][:10]}")
    print("=" * 60)

    # Match OPEN → CLOSE pairs
    open_trades = {}  # key = symbol → last open
    pairs = []

    for r in records:
        action = r.get("action", "")
        symbol = r.get("symbol", "?")

        if "OPEN" in action:
            open_trades[symbol] = r
        elif "CLOSE" in action and symbol in open_trades:
            open_r = open_trades.pop(symbol)
            pnl = r.get("pnl", 0) or 0
            pairs.append({
                "symbol": symbol,
                "direction": "LONG" if "LONG" in open_r.get("action", "") else "SHORT",
                "entry_price": open_r.get("price", 0),
                "exit_price": r.get("price", 0),
                "pnl": pnl,
                "open_whale_sell": open_r.get("whale", {}).get("sell_confidence", 0),
                "open_whale_buy": open_r.get("whale", {}).get("buy_confidence", 0),
                "close_whale_sell": r.get("whale", {}).get("sell_confidence", 0),
                "open_whale_intent": open_r.get("whale", {}).get("intent", "?"),
                "top_sellers_at_open": open_r.get("whale", {}).get("top_sellers", {}),
            })

    if not pairs:
        print(f"\n⚠️ {len(records)} events but no complete OPEN→CLOSE pairs yet.")
        print("   Need more trades. Current events:")
        for r in records[-5:]:
            whale = r.get("whale", {})
            print(f"   {r['timestamp'][:16]} {r.get('action','?'):15s} {r.get('symbol','?')} "
                  f"whale_sell={whale.get('sell_confidence', 0):.0%}")
        return

    print(f"\n📈 {len(pairs)} complete trades analyzed\n")

    # Split by whale sell confidence at entry
    high_sell = [p for p in pairs if p["open_whale_sell"] >= 0.40]
    low_sell = [p for p in pairs if p["open_whale_sell"] < 0.40]

    def stats(trades, label):
        if not trades:
            print(f"  {label}: No trades")
            return
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        avg_pnl = total_pnl / len(trades)
        win_rate = len(wins) / len(trades) * 100

        print(f"  {label}:")
        print(f"    Trades: {len(trades)} | Win rate: {win_rate:.0f}% | Total PnL: ${total_pnl:+.2f} | Avg: ${avg_pnl:+.2f}")

    print("─── BY WHALE SELL CONFIDENCE AT ENTRY ───")
    stats(high_sell, "Whale SELL ≥ 40% (distributing)")
    stats(low_sell, "Whale SELL < 40% (neutral/accumulating)")

    # Split shorts specifically
    shorts = [p for p in pairs if p["direction"] == "SHORT"]
    longs = [p for p in pairs if p["direction"] == "LONG"]

    if shorts:
        print("\n─── SHORTS ONLY ───")
        shorts_high_sell = [p for p in shorts if p["open_whale_sell"] >= 0.40]
        shorts_low_sell = [p for p in shorts if p["open_whale_sell"] < 0.40]
        stats(shorts_high_sell, "SHORT + Whale SELL ≥ 40%")
        stats(shorts_low_sell, "SHORT + Whale SELL < 40%")

    if longs:
        print("\n─── LONGS ONLY ───")
        longs_high_sell = [p for p in longs if p["open_whale_sell"] >= 0.40]
        longs_low_sell = [p for p in longs if p["open_whale_sell"] < 0.40]
        stats(longs_high_sell, "LONG + Whale SELL ≥ 40% (CONFLICT — should these be blocked?)")
        stats(longs_low_sell, "LONG + Whale SELL < 40%")

    # Per-trade detail
    print("\n─── TRADE DETAILS ───")
    print(f"{'Dir':6s} {'Symbol':10s} {'PnL':>8s} {'Whale SELL':>10s} {'Intent':>13s} {'Top Sellers'}")
    print("-" * 80)
    for p in pairs:
        pnl_str = f"${p['pnl']:+.2f}"
        sellers = ", ".join(f"{k}:{v:.0%}" for k, v in
                           sorted(p["top_sellers_at_open"].items(), key=lambda x: x[1], reverse=True)[:2])
        emoji = "✅" if p["pnl"] > 0 else "❌"
        print(f"{emoji} {p['direction']:5s} {p['symbol']:10s} {pnl_str:>8s} {p['open_whale_sell']:>9.0%} "
              f"{p['open_whale_intent']:>13s} {sellers}")


if __name__ == "__main__":
    analyze()
