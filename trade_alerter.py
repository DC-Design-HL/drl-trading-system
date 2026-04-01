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

# Whale behavior predictor (display only — not used for decisions)
_whale_predictor = None

def _get_whale_signal() -> dict:
    """Get whale behavior model signal. Returns empty dict on failure."""
    global _whale_predictor
    try:
        if _whale_predictor is None:
            from src.whale_behavior.models.predictor import WhaleIntentPredictor
            _whale_predictor = WhaleIntentPredictor()
        return _whale_predictor.get_signal()
    except Exception as e:
        print(f"[alerter] Whale behavior signal unavailable: {e}", flush=True)
        return {}

# Use dedicated alert bot token (separate from AI bot), fallback to main token
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_ALERT_BOT_TOKEN", "") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Additional chat IDs to receive alerts (comma-separated in env, or hardcoded)
TELEGRAM_EXTRA_CHAT_IDS = [
    cid.strip() for cid in os.environ.get("TELEGRAM_EXTRA_CHAT_IDS", "-5233405100").split(",")
    if cid.strip()
]

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

def _send_to_chat(text: str, chat_id: str) -> bool:
    """Send a message to a specific Telegram chat. Returns True on success."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
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
                print(f"[alerter] Telegram API returned {resp.status} for chat {chat_id}: {body}", flush=True)
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"[alerter] HTTP {e.code} for chat {chat_id}: {body}", flush=True)
            if 400 <= e.code < 500:
                return False
        except Exception as e:
            print(f"[alerter] Error sending to chat {chat_id} (attempt {attempt}): {e}", flush=True)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return False


def send_telegram(text: str) -> bool:
    """
    Send a message to all configured Telegram chats.
    Uses urllib so we have zero external dependencies.
    Returns True if at least the primary chat succeeded.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[alerter] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", flush=True)
        return False

    # Send to primary chat
    primary_ok = _send_to_chat(text, TELEGRAM_CHAT_ID)

    # Send to extra chats (best-effort, don't fail if these fail)
    for extra_id in TELEGRAM_EXTRA_CHAT_IDS:
        if extra_id and extra_id != TELEGRAM_CHAT_ID:
            try:
                _send_to_chat(text, extra_id)
            except Exception as e:
                print(f"[alerter] Failed to send to extra chat {extra_id}: {e}", flush=True)

    return primary_ok


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


def _add_rsi_adx_line(lines: list, signals: dict) -> None:
    """Add a dedicated RSI + ADX line to trade alerts for guard visibility."""
    if not signals:
        return

    # RSI from MTF 15m signal
    mtf = signals.get("mtf", {})
    sigs = mtf.get("signals", {})
    rsi_15m = sigs.get("15m", {}).get("rsi")

    # ADX from regime
    regime = signals.get("regime", {})
    adx = regime.get("adx")

    parts = []
    if rsi_15m is not None and isinstance(rsi_15m, (int, float)):
        # Flag extreme RSI
        if rsi_15m > 70:
            parts.append(f"RSI={rsi_15m:.0f} ⚠️OB")
        elif rsi_15m < 30:
            parts.append(f"RSI={rsi_15m:.0f} ⚠️OS")
        else:
            parts.append(f"RSI={rsi_15m:.0f}")

    if adx is not None and isinstance(adx, (int, float)):
        if adx < 15:
            parts.append(f"ADX={adx:.0f} ⚠️RANGING")
        else:
            parts.append(f"ADX={adx:.0f}")

    if parts:
        lines.append(f"🔬 {' | '.join(parts)}")


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

    # RSI + ADX values (key guard indicators)
    _add_rsi_adx_line(lines, signals)

    lines.append(f"📈 Strategy: {_format_strategy(strategy)}")

    # Market signals
    signal_lines = _format_signals(signals)
    if signal_lines:
        lines.append("")
        lines.extend(signal_lines)

    # Whale behavior model signal (display only — sell-focused)
    whale_lines = _format_whale_behavior()
    if whale_lines:
        lines.append("")
        lines.extend(whale_lines)

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

    # Whale behavior model signal (display only — sell-focused)
    whale_lines = _format_whale_behavior()
    if whale_lines:
        lines.append("")
        lines.extend(whale_lines)

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


def _format_whale_behavior() -> list:
    """Format whale behavior model signal — sell-focused."""
    whale_beh = _get_whale_signal()
    if not whale_beh or whale_beh.get("intent") in ("unavailable", "no_data", None):
        return []

    lines = []
    sell_conf = whale_beh.get("sell_confidence", 0)
    buy_conf = whale_beh.get("buy_confidence", 0)
    active = whale_beh.get("active_wallets", 0)
    details = whale_beh.get("wallet_details", {})

    # Find wallets with strong sell signals
    sellers = []
    for name, d in details.items():
        sell_p = d.get("probs", {}).get("SELL", 0)
        if sell_p >= 0.40:
            # Shorten wallet names for readability
            short = name.replace("binance_", "B.").replace("_wallet", "").replace("_", " ").title()
            sellers.append((short, sell_p))
    sellers.sort(key=lambda x: x[1], reverse=True)

    # Overall whale sell pressure indicator
    if sell_conf >= 0.50:
        emoji = "🔴"
        label = "SELL PRESSURE"
    elif sell_conf >= 0.35:
        emoji = "⚠️"
        label = "Mild sell"
    elif buy_conf >= 0.40:
        emoji = "🟢"
        label = "Accumulating"
    else:
        emoji = "🐋"
        label = "Neutral"

    line = f"{emoji} Whale Signal: {label} (sell={sell_conf:.0%} buy={buy_conf:.0%})"
    lines.append(line)

    # Show individual distributing wallets
    if sellers:
        seller_parts = [f"{name} {p:.0%}" for name, p in sellers[:3]]
        lines.append(f"   Distributing: {' | '.join(seller_parts)}")

    return lines


def _format_signals(signals: dict) -> list:
    """Format market signals into message lines — includes all 4 gate signals."""
    lines = []
    if not signals:
        return lines

    # Signal Gate decision (if present)
    gate = signals.get("signal_gate", {})
    if gate:
        result = gate.get("result", "N/A")
        confirms = gate.get("confirmations", 0)
        tier = gate.get("tier", "?")
        if result == "PASS":
            lines.append(f"🚦 Signal Gate: ✅ PASS ({confirms}/4 confirms) — {tier}")
        elif result == "AUTONOMOUS":
            lines.append(f"🚦 Signal Gate: 🟢 AUTONOMOUS (conf ≥ 80%)")
        else:
            lines.append(f"🚦 Signal Gate: ❌ {result} ({confirms}/4 confirms)")

    # 1. MTF Alignment
    mtf = signals.get("mtf", {})
    if mtf:
        bias = mtf.get("bias", "NEUTRAL")
        aligned = mtf.get("aligned", False)
        strength = mtf.get("strength", 0)
        align_icon = "✅" if aligned else "➖"
        sigs = mtf.get("signals", {})
        tf_str = ""
        if sigs:
            parts = []
            for tf in ["15m", "1h", "4h"]:
                s = sigs.get(tf, {})
                if s:
                    d = s.get("direction", "?")[:1].upper()
                    rsi = s.get("rsi", 0)
                    parts.append(f"{tf}:{d}({rsi:.0f})")
            tf_str = " | " + " ".join(parts) if parts else ""
        lines.append(f"📊 MTF: {align_icon} {bias} (str={strength:.0%}){tf_str}")

    # 2. Order Flow
    order_flow = signals.get("order_flow", {})
    if order_flow:
        bias = order_flow.get("bias", "neutral").title()
        score = order_flow.get("score", 0)
        buys = order_flow.get("large_buys", 0)
        sells = order_flow.get("large_sells", 0)
        score_str = f"{score:+.2f}" if isinstance(score, (int, float)) else "N/A"
        lines.append(f"💹 Order Flow: {bias} (score={score_str} | {buys}B/{sells}S)")

    # 3. Regime
    regime = signals.get("regime", {})
    if regime:
        regime_type = regime.get("type", regime.get("state", "unknown"))
        adx = regime.get("adx")
        adx_str = f" ADX={adx:.0f}" if adx else ""
        lines.append(f"📉 Regime: {regime_type}{adx_str}")

    # 4. Orderbook Imbalance
    ob = signals.get("orderbook", {})
    if ob:
        ob_bias = ob.get("bias", "neutral").title()
        imbalance = ob.get("imbalance_10", 0)
        imb_str = f"{imbalance:+.2f}" if isinstance(imbalance, (int, float)) else "N/A"
        lines.append(f"📖 Orderbook: {ob_bias} (imbalance={imb_str})")

    # Funding rate
    funding = signals.get("funding", {})
    if funding:
        rate = funding.get("rate", 0)
        bias = funding.get("bias", "neutral")
        bias_display = bias.replace("_", " ").title() if bias else "Neutral"
        lines.append(f"💱 Funding: {bias_display} ({rate})")

    # Whale signals
    whale = signals.get("whale", {})
    if whale:
        direction = whale.get("direction", "NEUTRAL")
        confidence = whale.get("confidence", 0)
        lines.append(f"🐋 Whale: {direction} ({confidence}%)")

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


# Cooldown tracking for system alerts (prevent spam)
_last_system_alert_time: dict = {}
_SYSTEM_ALERT_COOLDOWN = 300  # 5-minute cooldown per alert type per symbol


def format_system_alert(alert: dict) -> str:
    """Format a connectivity/system alert (WS disconnect, REST API error, etc.)."""
    trade = alert.get("trade", {})
    alert_type = trade.get("type", "UNKNOWN")
    symbol = trade.get("symbol", "?")
    details = trade.get("details", "No details")
    has_position = trade.get("has_open_position", False)
    direction = trade.get("position_direction", "FLAT")
    ts = alert.get("timestamp", "")

    # Cooldown: max 1 alert per type+symbol per 5 minutes
    key = f"{alert_type}:{symbol}"
    now = time.time()
    last_sent = _last_system_alert_time.get(key, 0)
    if now - last_sent < _SYSTEM_ALERT_COOLDOWN:
        return ""  # Skip — cooldown active
    _last_system_alert_time[key] = now

    # Choose emoji and label
    if alert_type == "WS_DISCONNECTED":
        emoji = "🔴"
        label = "WebSocket Disconnected"
    elif alert_type == "WS_ERROR":
        emoji = "⚠️"
        label = "WebSocket Error"
    elif alert_type == "WS_RECONNECTED":
        emoji = "🟢"
        label = "WebSocket Reconnected"
    elif alert_type == "REST_API_ERROR":
        emoji = "🔴"
        label = "REST API Error"
    else:
        emoji = "❗"
        label = alert_type

    lines = [f"{emoji} {label} — {symbol}"]
    lines.append(f"Details: {details}")

    if has_position:
        pos = alert.get("position", {})
        lines.append(f"Position: {direction}")
        entry = pos.get("entry_price", 0)
        sl = pos.get("sl_price", 0)
        tp = pos.get("tp_price", 0)
        if entry:
            lines.append(f"Entry: ${entry:,.2f}")
        if sl:
            lines.append(f"SL: ${sl:,.2f}")
        if tp:
            lines.append(f"TP: ${tp:,.2f}")

        if alert_type in ("WS_DISCONNECTED", "WS_ERROR"):
            lines.append("")
            lines.append("Exchange crashguard SL is active as backup.")

    if ts:
        lines.append(f"Time: {_format_timestamp(ts)}")

    return "\n".join(lines)


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

    # Connectivity/system alerts
    if trade.get("type") in ("WS_ERROR", "WS_DISCONNECTED", "WS_RECONNECTED", "REST_API_ERROR"):
        return format_system_alert(alert)

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
