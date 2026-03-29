#!/usr/bin/env python3
"""
Daily Trade Report — Generates a summary of the last 24h of trading
with signal correlation analysis.

Run: python scripts/daily_trade_report.py
Or:  python scripts/daily_trade_report.py --hours 48  (custom window)

Reads: logs/htf_pending_alerts.jsonl, logs/whale_shadow.jsonl
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

ALERTS_FILE = Path("logs/htf_pending_alerts.jsonl")
WHALE_SHADOW_FILE = Path("logs/whale_shadow.jsonl")
REPORTS_DIR = Path("logs/daily_reports")


def load_trades(hours=24):
    """Load completed OPEN→CLOSE pairs from the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    records = []
    with open(ALERTS_FILE) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                ts_str = r.get("timestamp", "")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        records.append(r)
            except:
                continue

    # Match OPEN → CLOSE pairs
    open_trades = {}
    pairs = []

    for r in records:
        trade = r.get("trade", {})
        action = trade.get("action", "")
        symbol = trade.get("symbol", "?")

        if "OPEN" in action and "PARTIAL" not in action:
            open_trades[symbol] = r
        elif "CLOSE" in action and "PARTIAL" not in action and symbol in open_trades:
            open_r = open_trades.pop(symbol)
            open_trade = open_r.get("trade", {})
            close_trade = trade
            signals = open_r.get("signals", {})
            whale = open_r.get("whale_behavior", {})

            entry = open_trade.get("price", 0)
            exit_p = close_trade.get("exit_price", close_trade.get("price", 0))
            pnl = close_trade.get("pnl", 0) or 0
            direction = "LONG" if "LONG" in open_trade.get("action", "") else "SHORT"
            confidence = open_trade.get("confidence", 0)

            # Signals
            mtf = signals.get("mtf", {})
            of = signals.get("order_flow", {})
            regime = signals.get("regime", {})
            ob = signals.get("orderbook", {})
            gate = signals.get("signal_gate", {})

            pairs.append({
                "symbol": symbol,
                "direction": direction,
                "entry": entry,
                "exit": exit_p,
                "pnl": pnl,
                "confidence": confidence,
                "timestamp": open_r.get("timestamp", "")[:16],
                "close_time": r.get("timestamp", "")[:16],
                "reason": close_trade.get("reason", ""),
                "mtf_bias": mtf.get("bias", "N/A"),
                "of_bias": of.get("bias", "N/A"),
                "of_score": of.get("score", 0),
                "regime": regime.get("type", regime.get("state", "N/A")),
                "regime_adx": regime.get("adx", 0),
                "ob_bias": ob.get("bias", "N/A"),
                "gate_result": gate.get("result", "N/A"),
                "gate_confirms": gate.get("confirmations", 0),
                "whale_sell": whale.get("sell_confidence", 0),
                "whale_buy": whale.get("buy_confidence", 0),
                "whale_intent": whale.get("intent", "N/A"),
                "whale_top_sellers": whale.get("top_sellers", {}),
            })

    return pairs


def generate_report(hours=24):
    """Generate the daily report."""
    pairs = load_trades(hours)
    
    now = datetime.now(timezone.utc)
    report_lines = []
    
    def out(line=""):
        report_lines.append(line)

    out(f"📊 DAILY TRADE REPORT — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    out(f"Period: Last {hours} hours")
    out("=" * 60)

    if not pairs:
        out("\n⚠️ No completed trades in this period.")
        out("Open positions may still be active.")
        return "\n".join(report_lines)

    # Overall stats
    total_pnl = sum(t["pnl"] for t in pairs)
    wins = [t for t in pairs if t["pnl"] > 0]
    losses = [t for t in pairs if t["pnl"] <= 0]
    wr = len(wins) / len(pairs) * 100

    out(f"\n📈 SUMMARY: {len(pairs)} trades | {len(wins)}W/{len(losses)}L ({wr:.0f}%) | PnL: ${total_pnl:+.2f}")

    # Per-symbol breakdown
    for sym in ["BTCUSDT", "ETHUSDT"]:
        sym_trades = [t for t in pairs if t["symbol"] == sym]
        if not sym_trades:
            continue
        sym_pnl = sum(t["pnl"] for t in sym_trades)
        sym_wins = sum(1 for t in sym_trades if t["pnl"] > 0)
        sym_wr = sym_wins / len(sym_trades) * 100

        out(f"\n{'─' * 60}")
        out(f"  {sym}: {len(sym_trades)} trades | {sym_wins}W/{len(sym_trades)-sym_wins}L ({sym_wr:.0f}%) | PnL: ${sym_pnl:+.2f}")
        out(f"{'─' * 60}")

        for t in sym_trades:
            emoji = "✅" if t["pnl"] > 0 else "❌"
            whale_str = ""
            if t["whale_sell"] > 0:
                whale_str = f" | 🐋 sell={t['whale_sell']:.0%}"
                sellers = ", ".join(f"{k}:{v:.0%}" for k, v in 
                    sorted(t["whale_top_sellers"].items(), key=lambda x: x[1], reverse=True)[:2])
                if sellers:
                    whale_str += f" [{sellers}]"

            out(f"  {emoji} {t['timestamp'][5:]} {t['direction']:5s} "
                f"${t['entry']:,.2f}→${t['exit']:,.2f} "
                f"PnL=${t['pnl']:+.2f} conf={t['confidence']:.2f}")
            out(f"     MTF={t['mtf_bias']:8s} OF={t['of_bias']:8s} "
                f"Regime={t['regime']:15s} Gate={t['gate_result']}({t['gate_confirms']})"
                f"{whale_str}")

    # Signal correlation analysis
    out(f"\n{'=' * 60}")
    out("📊 SIGNAL CORRELATION (today)")
    out(f"{'=' * 60}")

    def stats(name, filtered):
        if not filtered:
            return
        w = sum(1 for t in filtered if t["pnl"] > 0)
        p = sum(t["pnl"] for t in filtered)
        wr = w / len(filtered) * 100
        emoji = "🟢" if p > 0 else "🔴"
        out(f"  {emoji} {name:40s} {len(filtered):2d}T | {wr:3.0f}% WR | ${p:+.2f}")

    stats("High conf (≥0.90)", [t for t in pairs if t["confidence"] >= 0.90])
    stats("Low conf (<0.60)", [t for t in pairs if t["confidence"] < 0.60])
    stats("OF=bearish", [t for t in pairs if t["of_bias"] == "bearish"])
    stats("OF=bullish", [t for t in pairs if t["of_bias"] == "bullish"])
    stats("SHORT trades", [t for t in pairs if t["direction"] == "SHORT"])
    stats("LONG trades", [t for t in pairs if t["direction"] == "LONG"])
    
    # Whale signal correlation (if available)
    whale_trades = [t for t in pairs if t["whale_sell"] > 0]
    if whale_trades:
        out(f"\n🐋 WHALE SIGNAL CORRELATION:")
        stats("Whale SELL ≥ 40%", [t for t in whale_trades if t["whale_sell"] >= 0.40])
        stats("Whale SELL < 40%", [t for t in whale_trades if t["whale_sell"] < 0.40])
        stats("SHORT + Whale SELL ≥ 40%", [t for t in whale_trades if t["direction"] == "SHORT" and t["whale_sell"] >= 0.40])
        stats("LONG + Whale SELL ≥ 40%", [t for t in whale_trades if t["direction"] == "LONG" and t["whale_sell"] >= 0.40])

    # Cumulative stats
    out(f"\n{'=' * 60}")
    out("📊 CUMULATIVE (all time)")
    out(f"{'=' * 60}")
    all_pairs = load_trades(hours=9999)
    all_pnl = sum(t["pnl"] for t in all_pairs)
    all_wins = sum(1 for t in all_pairs if t["pnl"] > 0)
    all_wr = all_wins / max(len(all_pairs), 1) * 100
    out(f"  Total: {len(all_pairs)} trades | {all_wins}W/{len(all_pairs)-all_wins}L ({all_wr:.0f}%) | PnL: ${all_pnl:+.2f}")

    for sym in ["BTCUSDT", "ETHUSDT"]:
        st = [t for t in all_pairs if t["symbol"] == sym]
        if st:
            sp = sum(t["pnl"] for t in st)
            sw = sum(1 for t in st if t["pnl"] > 0)
            swr = sw / len(st) * 100
            out(f"  {sym}: {len(st)}T | {sw}W/{len(st)-sw}L ({swr:.0f}%) | ${sp:+.2f}")

    return "\n".join(report_lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Daily Trade Report")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    args = parser.parse_args()

    report = generate_report(args.hours)
    print(report)

    if args.save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        report_file = REPORTS_DIR / f"report_{date_str}.txt"
        report_file.write_text(report)
        print(f"\n💾 Saved to {report_file}")


if __name__ == "__main__":
    main()
