#!/usr/bin/env python3
"""
Trade Alerter — Watches htf_pending_alerts.jsonl and sends alerts to Telegram.

Zero AI token usage. Pure Python → Telegram Bot API pipeline.
Tracks file offset so it won't re-send old alerts on restart.
"""

import json
import time
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALERTS_FILE = Path("logs/htf_pending_alerts.jsonl")
OFFSET_FILE = Path("logs/.alerter_offset")
CHECK_INTERVAL = 5  # seconds between file checks

# Use dedicated alert bot token (separate from AI bot), fallback to main token
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_ALERT_BOT_TOKEN", "") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries on network failure


# ---------------------------------------------------------------------------
# Offset tracking
# ---------------------------------------------------------------------------

def get_last_offset() -> int:
    """Get the last processed file offset."""
    try:
        return int(OFFSET_FILE.read_text().strip())
    except Exception:
        return 0


def save_offset(offset: int) -> None:
    """Persist the current file offset."""
    OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_FILE.write_text(str(offset))


# ---------------------------------------------------------------------------
# Telegram sender
# ---------------------------------------------------------------------------

def send_telegram(text: str) -> bool:
    """
    Send a message via Telegram Bot API.
    Uses urllib so we have zero external dependencies.
    Returns True on success, False on failure.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[alerter] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", flush=True)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Use plain text (no parse_mode) to avoid Markdown/HTML parsing failures
    # that cause 400 errors and silently drop alerts
    payload = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text.replace("*", "").replace("_", ""),
        "disable_web_page_preview": True,
    }).encode("utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    return True
                body = resp.read().decode()
                print(f"[alerter] Telegram API returned {resp.status}: {body}", flush=True)
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"[alerter] HTTP {e.code}: {body}", flush=True)
            # Don't retry on 4xx client errors (bad token, bad chat id, etc.)
            if 400 <= e.code < 500:
                return False
        except Exception as e:
            print(f"[alerter] Attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    return False


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def _escape_md(text: str) -> str:
    """Escape Markdown special characters for Telegram."""
    # In Markdown mode, we need to be careful with _ * [ ] ( ) ~ ` > # + - = | { } . !
    # But we use them for formatting, so only escape content that might break things
    return str(text).replace("_", "\\_")


def _pct(entry: float, price: float) -> str:
    """Calculate and format percentage change from entry."""
    if entry <= 0:
        return ""
    pct = ((price - entry) / entry) * 100
    sign = "+" if pct >= 0 else ""
    return f" ({sign}{pct:.1f}%)"


def _format_strategy(strategy: str) -> str:
    """Human-friendly strategy name."""
    mapping = {
        "htf": "HTF Standard",
        "partial": "Partial TP",
        "hybrid": "Hybrid",
    }
    return mapping.get(strategy, strategy.title() if strategy else "Unknown")


def format_open_trade(alert: dict) -> str:
    """Format an OPEN_LONG or OPEN_SHORT alert."""
    trade = alert.get("trade", {})
    signals = alert.get("signals", {})
    position = alert.get("position", {})
    strategy = alert.get("strategy", trade.get("strategy", "htf"))

    action = trade.get("action", "")
    symbol = trade.get("symbol", "?")
    price = trade.get("price", 0)
    confidence = trade.get("confidence", 0)
    sl = position.get("sl_price", trade.get("sl", 0))
    tp = position.get("tp_price", trade.get("tp", 0))
    balance = trade.get("trade_value", 0)

    if "LONG" in action:
        emoji = "🟢"
        direction = "OPEN LONG"
    else:
        emoji = "🔴"
        direction = "OPEN SHORT"

    lines = [
        f"{emoji} *{direction}* — {symbol}",
        "",
        f"💰 Entry: ${price:,.2f}",
    ]

    if tp > 0:
        lines.append(f"🎯 TP: ${tp:,.2f}{_pct(price, tp)}")
    if sl > 0:
        lines.append(f"🛑 SL: ${sl:,.2f}{_pct(price, sl)}")

    # Position size in USDT
    units = trade.get("units", position.get("units", 0))
    position_usdt = units * price if units and price else trade.get("trade_value", 0)
    if position_usdt > 0:
        lines.append(f"📐 Size: ${position_usdt:,.2f} USDT ({units:.4f} units)")

    # Risk management info
    leverage = trade.get("leverage", 0)
    dollar_risk = trade.get("dollar_risk", 0)
    margin = trade.get("margin", 0)
    if leverage and leverage > 1:
        lines.append(f"⚡ Leverage: {leverage}x | Risk: ${dollar_risk:,.2f} | Margin: ${margin:,.2f}")

    lines.append(f"📊 Confidence: {confidence * 100:.0f}%")
    lines.append(f"📈 Strategy: {_format_strategy(strategy)}")

    # Market signals
    signal_lines = _format_signals(signals)
    if signal_lines:
        lines.append("")
        lines.extend(signal_lines)

    ts = _format_timestamp(alert.get("timestamp", ""))
    if ts:
        lines.append(f"🕐 {ts}")

    return "\n".join(lines)


def format_close_trade(alert: dict) -> str:
    """Format a CLOSE trade alert."""
    trade = alert.get("trade", {})
    signals = alert.get("signals", {})
    strategy = alert.get("strategy", trade.get("strategy", "htf"))

    action = trade.get("action", "")
    symbol = trade.get("symbol", "?")
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", trade.get("price", 0))
    pnl = trade.get("pnl", 0)
    reason = trade.get("reason", "")
    confidence = trade.get("confidence", 0)

    if pnl >= 0:
        emoji = "✅"
        pnl_label = "PROFIT"
    else:
        emoji = "❌"
        pnl_label = "LOSS"

    # Determine close type
    if "LONG" in action:
        close_dir = "CLOSE LONG"
    elif "SHORT" in action:
        close_dir = "CLOSE SHORT"
    else:
        close_dir = "CLOSE"

    lines = [
        f"{emoji} *{close_dir}* — {symbol} ({pnl_label})",
        "",
        f"💰 Entry: ${entry_price:,.2f}",
        f"🏁 Exit: ${exit_price:,.2f}{_pct(entry_price, exit_price)}",
    ]

    pnl_emoji = "📈" if pnl >= 0 else "📉"
    lines.append(f"{pnl_emoji} Trade PnL: {'+'if pnl >= 0 else ''}${pnl:,.2f}")

    # Total balance PnL (wallet balance vs $5,000 initial deposit)
    balance_after = trade.get("balance_after", 0)
    initial_balance = 5000.0  # Binance testnet starting balance
    if balance_after:
        total_pnl = balance_after - initial_balance
        pct = (total_pnl / initial_balance) * 100
        bal_emoji = "🟢" if total_pnl >= 0 else "🔴"
        lines.append(
            f"{bal_emoji} Balance: ${balance_after:,.2f} "
            f"({'+'if total_pnl >= 0 else ''}${total_pnl:,.2f} / "
            f"{'+'if pct >= 0 else ''}{pct:.2f}%)"
        )

    # Position size in USDT
    units = trade.get("units", 0)
    close_price = exit_price or entry_price
    position_usdt = units * close_price if units and close_price else 0
    if position_usdt > 0:
        lines.append(f"📐 Size: ${position_usdt:,.2f} USDT ({units:.4f} units)")

    if reason:
        lines.append(f"📋 Reason: {reason}")

    lines.append(f"📈 Strategy: {_format_strategy(strategy)}")

    signal_lines = _format_signals(signals)
    if signal_lines:
        lines.append("")
        lines.extend(signal_lines)

    ts = _format_timestamp(alert.get("timestamp", ""))
    if ts:
        lines.append("")
        lines.append(f"🕐 {ts}")

    return "\n".join(lines)


def format_partial_close(alert: dict) -> str:
    """Format a PARTIAL_CLOSE alert."""
    trade = alert.get("trade", {})
    signals = alert.get("signals", {})
    position = alert.get("position", {})
    strategy = alert.get("strategy", trade.get("strategy", "htf"))

    action = trade.get("action", "")
    symbol = trade.get("symbol", "?")
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)
    pnl = trade.get("pnl", 0)
    reason = trade.get("reason", "")
    partial_num = trade.get("partial_exit_num", 0)
    remaining = trade.get("remaining_units", 0)
    original = trade.get("original_units", 0)

    if "LONG" in action:
        direction = "LONG"
    else:
        direction = "SHORT"

    # Calculate remaining %
    remaining_pct = (remaining / original * 100) if original > 0 else 0

    pnl_emoji = "📈" if pnl >= 0 else "📉"

    lines = [
        f"🔶 *PARTIAL CLOSE {partial_num}* — {symbol} {direction}",
        "",
        f"💰 Entry: ${entry_price:,.2f}",
        f"🏁 Exit: ${exit_price:,.2f}{_pct(entry_price, exit_price)}",
        f"{pnl_emoji} PnL: {'+'if pnl >= 0 else ''}${pnl:,.2f}",
        f"📦 Remaining: {remaining_pct:.0f}% of position",
    ]

    # Position size in USDT (closed portion)
    closed_units = trade.get("units", 0)
    close_price = exit_price or entry_price
    closed_usdt = closed_units * close_price if closed_units and close_price else 0
    remaining_usdt = remaining * close_price if remaining and close_price else 0
    if closed_usdt > 0:
        lines.append(f"📐 Closed: ${closed_usdt:,.2f} USDT | Remaining: ${remaining_usdt:,.2f} USDT")

    if reason:
        lines.append(f"📋 Reason: {reason}")

    # Show SL move if provided
    old_sl = trade.get("old_sl", 0)
    new_sl = trade.get("new_sl", 0)
    if old_sl > 0 and new_sl > 0 and old_sl != new_sl:
        lines.append(f"🛑 SL moved: ${old_sl:,.2f} → ${new_sl:,.2f}")

    if trade.get("trailing_active"):
        lines.append("📍 Trailing stop: ACTIVE")

    lines.append(f"📈 Strategy: {_format_strategy(strategy)}")

    ts = _format_timestamp(alert.get("timestamp", ""))
    if ts:
        lines.append("")
        lines.append(f"🕐 {ts}")

    return "\n".join(lines)


def format_sl_tp_update(alert: dict) -> str:
    """Format an SL/TP update alert."""
    trade = alert.get("trade", {})
    position = alert.get("position", {})
    strategy = alert.get("strategy", trade.get("strategy", "htf"))

    symbol = trade.get("symbol", "?")
    direction = position.get("direction", trade.get("direction", "?"))
    update_type = trade.get("update_type", "SL")
    old_price = trade.get("old_price", 0)
    new_price = trade.get("new_price", 0)
    reason = trade.get("reason", "trailing")
    profit_pct = trade.get("profit_pct", None)
    current_price = trade.get("current_price", 0)
    entry_price = position.get("entry_price", 0)

    if update_type == "TP":
        emoji = "🎯"
        label = "TP Updated"
    else:
        emoji = "🔄"
        label = "SL Updated"

    pct_change = ""
    if old_price > 0:
        change = ((new_price - old_price) / old_price) * 100
        sign = "+" if change >= 0 else ""
        pct_change = f" ({sign}{change:.1f}%)"

    lines = [
        f"{emoji} *{label}* — {symbol} {direction}",
        f"${old_price:,.2f} → ${new_price:,.2f}{pct_change}",
        f"📋 Reason: {reason}",
    ]

    # Show entry and current price for context
    if entry_price > 0:
        lines.append(f"💰 Entry: ${entry_price:,.2f}")
    if current_price > 0:
        lines.append(f"📍 Current: ${current_price:,.2f}")

    # Show profit %
    if profit_pct is not None:
        profit_sign = "+" if profit_pct >= 0 else ""
        profit_emoji = "📈" if profit_pct >= 0 else "📉"
        lines.append(f"{profit_emoji} Profit: {profit_sign}{profit_pct:.1f}%")

    lines.append(f"📈 Strategy: {_format_strategy(strategy)}")

    ts = _format_timestamp(alert.get("timestamp", ""))
    if ts:
        lines.append(f"🕐 {ts}")

    return "\n".join(lines)


def format_liquidation_risk(alert: dict) -> str:
    """Format a LIQUIDATION_RISK alert."""
    trade = alert.get("trade", {})
    position = alert.get("position", {})
    strategy = alert.get("strategy", "htf")

    symbol = trade.get("symbol", "?")
    direction = trade.get("direction", position.get("direction", "?"))
    liq_price = trade.get("liquidation_price", 0)
    sl_price = trade.get("sl_price", position.get("sl_price", 0))
    entry_price = trade.get("entry_price", position.get("entry_price", 0))
    buffer = trade.get("buffer", 0)
    buffer_pct = trade.get("buffer_pct", 0)
    simulated = trade.get("simulated", False)

    sim_tag = " (Simulated)" if simulated else ""

    lines = [
        f"⚠️ *LIQUIDATION RISK{sim_tag}* — {symbol} {direction}",
        "",
        f"🔴 Liquidation: ${liq_price:,.2f}",
        f"🛑 SL: ${sl_price:,.2f}",
    ]

    if entry_price > 0:
        lines.append(f"💰 Entry: ${entry_price:,.2f}")

    lines.append(f"📐 Buffer: ${abs(buffer):,.2f} ({abs(buffer_pct):.1f}%)")
    lines.append(f"⚡ Recommended: Tighten SL or reduce position size")
    lines.append(f"📈 Strategy: {_format_strategy(strategy)}")

    ts = _format_timestamp(alert.get("timestamp", ""))
    if ts:
        lines.append(f"🕐 {ts}")

    return "\n".join(lines)


def _format_signals(signals: dict) -> list:
    """Format market signals into message lines."""
    lines = []
    if not signals:
        return lines

    whale = signals.get("whale", {})
    if whale:
        direction = whale.get("direction", "NEUTRAL")
        confidence = whale.get("confidence", 0)
        lines.append(f"🐋 Whale: {direction} ({confidence}%)")

    funding = signals.get("funding", {})
    if funding:
        rate = funding.get("rate", 0)
        bias = funding.get("bias", "neutral")
        bias_display = bias.replace("_", " ").title() if bias else "Neutral"
        lines.append(f"💱 Funding: {bias_display} ({rate})")

    order_flow = signals.get("order_flow", {})
    if order_flow:
        bias = order_flow.get("bias", "neutral").title()
        buys = order_flow.get("large_buys", 0)
        sells = order_flow.get("large_sells", 0)
        lines.append(f"📊 Order Flow: {bias} ({buys}B/{sells}S)")

    regime = signals.get("regime", {})
    if regime:
        state = regime.get("state", "unknown")
        adx = regime.get("adx")
        if adx:
            lines.append(f"📉 Regime: {state} (ADX: {adx})")

    return lines


def _format_timestamp(ts_str: str) -> str:
    """Format ISO timestamp to a readable string."""
    if not ts_str:
        return ""
    try:
        ts = datetime.fromisoformat(ts_str)
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts_str


# ---------------------------------------------------------------------------
# Alert routing
# ---------------------------------------------------------------------------

_last_liq_alert_time: dict = {}  # {symbol: timestamp} — cooldown tracker
_LIQ_ALERT_COOLDOWN = 600  # 10 minutes between liquidation alerts per symbol


def format_alert(alert: dict) -> str:
    """Route an alert to the appropriate formatter based on action type."""
    trade = alert.get("trade", {})
    action = trade.get("action", "")

    # Liquidation risk alerts (with 10-min cooldown per symbol to prevent spam)
    if trade.get("type") == "LIQUIDATION_RISK":
        symbol = trade.get("symbol", "?")
        now = time.time()
        last_sent = _last_liq_alert_time.get(symbol, 0)
        if now - last_sent < _LIQ_ALERT_COOLDOWN:
            print(f"[alerter] Skipping duplicate LIQUIDATION_RISK for {symbol} (cooldown)", flush=True)
            return ""  # Empty string = skip sending
        _last_liq_alert_time[symbol] = now
        return format_liquidation_risk(alert)

    # SL/TP update alerts
    if trade.get("type") == "SL_UPDATE" or trade.get("update_type") == "SL":
        return format_sl_tp_update(alert)
    if trade.get("type") == "TP_UPDATE" or trade.get("update_type") == "TP":
        return format_sl_tp_update(alert)

    # Partial close
    if "PARTIAL_CLOSE" in action:
        return format_partial_close(alert)

    # Full close
    if "CLOSE" in action:
        return format_close_trade(alert)

    # Open
    if "OPEN" in action:
        return format_open_trade(alert)

    # Fallback — just dump key info
    symbol = trade.get("symbol", "?")
    price = trade.get("price", trade.get("exit_price", 0))
    return f"📊 *Trade Alert* — {symbol}\nAction: {action}\nPrice: ${price:,.2f}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[alerter] ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set", flush=True)
        print("[alerter] Set them in the .env file or environment", flush=True)
        sys.exit(1)

    print(f"[alerter] Starting trade alerter", flush=True)
    print(f"[alerter] Watching: {ALERTS_FILE}", flush=True)
    print(f"[alerter] Chat ID: {TELEGRAM_CHAT_ID}", flush=True)

    # Send startup notification
    send_telegram("🤖 *Trade Alert Service Started*\nMonitoring for new trades...")

    offset = get_last_offset()
    print(f"[alerter] Resuming from offset: {offset}", flush=True)

    while True:
        try:
            if not ALERTS_FILE.exists():
                time.sleep(CHECK_INTERVAL)
                continue

            current_size = ALERTS_FILE.stat().st_size

            # File was truncated/rotated — reset offset
            if current_size < offset:
                print("[alerter] File truncated, resetting offset", flush=True)
                offset = 0
                save_offset(0)

            # New data available
            if current_size > offset:
                with open(ALERTS_FILE, "r") as f:
                    f.seek(offset)
                    new_data = f.read()
                    new_offset = f.tell()

                sent_count = 0
                for line in new_data.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        alert = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[alerter] Skipping malformed JSON: {e}", flush=True)
                        continue

                    # Only send testnet (htf) alerts — skip paper trade bots
                    strategy = alert.get("strategy", alert.get("trade", {}).get("strategy", "htf"))
                    if strategy in ("partial", "hybrid"):
                        print(f"[alerter] Skipping paper trade alert ({strategy})", flush=True)
                        continue

                    message = format_alert(alert)
                    if not message:
                        continue  # Skipped by cooldown or filter
                    if send_telegram(message):
                        sent_count += 1
                        action = alert.get("trade", {}).get("action", "?")
                        symbol = alert.get("trade", {}).get("symbol", "?")
                        print(f"[alerter] ✅ Sent: {action} {symbol}", flush=True)
                    else:
                        print(f"[alerter] ⚠️ Failed to send alert", flush=True)

                    # Small delay between messages to avoid Telegram rate limits
                    if sent_count > 0:
                        time.sleep(0.5)

                save_offset(new_offset)
                offset = new_offset

                if sent_count > 0:
                    print(f"[alerter] Processed {sent_count} alerts", flush=True)

        except Exception as e:
            print(f"[alerter] Error: {e}", file=sys.stderr, flush=True)
            time.sleep(CHECK_INTERVAL)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
