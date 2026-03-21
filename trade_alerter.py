#!/usr/bin/env python3
"""
Trade Alerter — Monitors HTF bot trades and sends alerts.
Watches logs/htf_trades.json for new entries and reports them.
Outputs JSON to stdout for the systemd service to capture.
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

TRADES_FILE = Path("logs/htf_trades.json")
SEEN_FILE = Path("logs/.htf_alerter_offset")
CHECK_INTERVAL = 30  # seconds


def get_last_offset() -> int:
    """Get the last seen file offset."""
    try:
        return int(SEEN_FILE.read_text().strip())
    except Exception:
        return 0


def save_offset(offset: int) -> None:
    """Save current file offset."""
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text(str(offset))


def format_trade_alert(trade: dict) -> str:
    """Format a trade into a readable alert message."""
    action = trade.get("action", "UNKNOWN")
    symbol = trade.get("symbol", "?")
    price = trade.get("price", 0)
    confidence = trade.get("confidence", 0)
    pnl = trade.get("pnl", 0)
    balance = trade.get("balance", 0)
    reason = trade.get("reason", "model")
    timestamp = trade.get("timestamp", "")

    # Determine emoji and direction
    if "OPEN_LONG" in action:
        emoji = "🟢"
        direction = "LONG"
    elif "OPEN_SHORT" in action:
        emoji = "🔴"
        direction = "SHORT"
    elif "CLOSE" in action:
        emoji = "🏁"
        direction = "CLOSE"
        if pnl > 0:
            emoji = "✅"
        elif pnl < 0:
            emoji = "❌"
    else:
        emoji = "📊"
        direction = action

    # Build message
    lines = [
        f"{emoji} **HTF Trade Alert** {emoji}",
        f"",
        f"**{direction}** {symbol}",
        f"💰 Price: ${price:,.2f}",
        f"🎯 Confidence: {confidence*100:.0f}%",
    ]

    if "CLOSE" in action and pnl != 0:
        pnl_emoji = "📈" if pnl > 0 else "📉"
        lines.append(f"{pnl_emoji} P&L: {'+'if pnl>0 else ''}${pnl:,.2f}")

    if reason and reason != "model":
        lines.append(f"📋 Reason: {reason}")

    lines.append(f"💼 Balance: ${balance:,.2f}")

    try:
        ts = datetime.fromisoformat(timestamp)
        lines.append(f"🕐 {ts.strftime('%Y-%m-%d %H:%M UTC')}")
    except Exception:
        pass

    return "\n".join(lines)


def main():
    print(f"[alerter] Watching {TRADES_FILE} for new trades...", flush=True)
    offset = get_last_offset()

    while True:
        try:
            if TRADES_FILE.exists():
                current_size = TRADES_FILE.stat().st_size
                if current_size > offset:
                    with open(TRADES_FILE, "r") as f:
                        f.seek(offset)
                        new_lines = f.read()
                        new_offset = f.tell()

                    for line in new_lines.strip().split("\n"):
                        if not line.strip():
                            continue
                        try:
                            trade = json.loads(line)
                            alert = format_trade_alert(trade)
                            # Output as JSON for the cron/webhook to pick up
                            print(json.dumps({
                                "type": "trade_alert",
                                "message": alert,
                                "trade": trade,
                            }), flush=True)
                        except json.JSONDecodeError:
                            continue

                    save_offset(new_offset)
                    offset = new_offset
                elif current_size < offset:
                    # File was truncated/rotated
                    offset = 0
                    save_offset(0)
        except Exception as e:
            print(f"[alerter] Error: {e}", file=sys.stderr, flush=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
