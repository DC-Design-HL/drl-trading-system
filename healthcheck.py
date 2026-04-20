#!/usr/bin/env python3
"""
Hourly system health check — sends status via @luigiAlertBot to Telegram.
"""
import json, os, time, urllib.request, urllib.parse
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).parent
os.chdir(REPO)

# Load .env
env_path = REPO / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

BOT_TOKEN = os.environ.get("TELEGRAM_ALERT_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        print("[healthcheck] No bot token or chat ID")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }).encode()
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[healthcheck] Telegram send failed: {e}")
        return False


def check_services() -> dict:
    """Check PID liveness from running_services.json."""
    services = {}
    try:
        with open(REPO / "logs" / "running_services.json") as f:
            svc_data = json.load(f)
        for name, info in svc_data.items():
            pid = info.get("pid", 0)
            try:
                os.kill(pid, 0)
                services[name] = {"alive": True, "pid": pid}
            except (ProcessLookupError, PermissionError):
                services[name] = {"alive": False, "pid": pid}
    except Exception as e:
        services["_error"] = str(e)
    return services


def check_api_state() -> dict:
    """Get trading state from API."""
    try:
        url = "http://127.0.0.1:5001/api/state"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def get_real_balance() -> float:
    """Fetch real wallet balance from Binance Futures testnet.
    Returns total_margin_balance (wallet + unrealized PnL), or 0 on failure."""
    try:
        from src.api.futures_executor import FuturesTestnetExecutor
        executor = FuturesTestnetExecutor()
        portfolio = executor.get_portfolio()
        return float(portfolio.get("total_margin_balance", 0))
    except Exception as e:
        print(f"[healthcheck] Real balance fetch failed: {e}")
        return 0.0


def check_log_recency() -> dict:
    """Check how recent each bot's last log entry is."""
    recency = {}
    bot_log = REPO / "logs" / "bots_live.log"
    if bot_log.exists():
        mtime = bot_log.stat().st_mtime
        age_min = (time.time() - mtime) / 60
        recency["bots_live.log"] = f"{age_min:.0f}m ago"
    return recency


def check_trade_staleness() -> dict:
    """Check if any symbol has gone too long without opening a trade."""
    STALE_HOURS = 24
    stale = {}
    bot_log = REPO / "logs" / "bots_live.log"
    if not bot_log.exists():
        return stale

    # Scan log for last OPEN_LONG / OPEN_SHORT per symbol
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    last_trade_ts = {s: None for s in symbols}

    try:
        with open(bot_log) as f:
            for line in f:
                if "OPEN_LONG" in line or "OPEN_SHORT" in line:
                    # Extract timestamp from log line: "2026-04-13 10:07:36,576 ..."
                    ts_str = line[:23].replace(",", ".")
                    for sym in symbols:
                        if sym in line:
                            try:
                                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                                ts = ts.replace(tzinfo=timezone.utc)
                                last_trade_ts[sym] = ts
                            except ValueError:
                                pass

        now = datetime.now(timezone.utc)
        for sym, ts in last_trade_ts.items():
            if ts is None:
                stale[sym] = "no trades found in log"
            else:
                hours_ago = (now - ts).total_seconds() / 3600
                if hours_ago > STALE_HOURS:
                    stale[sym] = f"last trade {hours_ago:.0f}h ago"
    except Exception as e:
        stale["_error"] = str(e)

    return stale


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 1. Services
    services = check_services()
    alive_count = sum(1 for s in services.values() if isinstance(s, dict) and s.get("alive"))
    total_count = len([s for s in services.values() if isinstance(s, dict)])
    all_alive = alive_count == total_count and total_count > 0

    # 2. API state
    state = check_api_state()
    assets = state.get("assets", {})
    # Fetch real balance from Binance (CLAUDE.md rule #4: all data from exchange)
    balance = get_real_balance()
    if balance <= 0:
        # Fallback to API state only if exchange fetch failed
        balance = state.get("total_balance", 0)

    # 3. Log recency
    recency = check_log_recency()

    # 4. Trade staleness
    stale_trades = check_trade_staleness()

    # Build message
    has_stale = bool(stale_trades and any(k != "_error" for k in stale_trades))
    status_icon = "\u2705" if (all_alive and not has_stale) else "\u26a0\ufe0f"
    lines = [
        f"{status_icon} <b>Hourly Health Check</b> — {now}",
        f"Services: {alive_count}/{total_count} alive",
    ]

    # Dead services
    dead = [name for name, info in services.items() if isinstance(info, dict) and not info.get("alive")]
    if dead:
        lines.append(f"\u274c DEAD: {', '.join(dead)}")

    initial_balance = 5000.0
    lifetime_pnl = balance - initial_balance
    pnl_pct = (lifetime_pnl / initial_balance) * 100 if initial_balance else 0
    lines.append(f"Balance: ${balance:,.2f} (lifetime: {'+'if lifetime_pnl >= 0 else ''}${lifetime_pnl:,.2f} / {'+'if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)")
    lines.append("")

    # Positions
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
        asset = assets.get(symbol, {})
        pos = asset.get("position", 0)
        label = asset.get("position_label", "FLAT")
        entry = asset.get("entry_price", 0)
        sl = asset.get("sl_price", 0)
        tp = asset.get("tp_price", 0)
        pnl = asset.get("pnl", 0)

        if pos != 0:
            direction = "LONG" if pos == 1 else "SHORT"
            lines.append(
                f"\u25b6 {symbol}: {direction} @ ${entry:,.2f} | "
                f"SL ${sl:,.2f} | TP ${tp:,.2f} | PnL ${pnl:+,.2f}"
            )
        else:
            lines.append(f"\u25aa {symbol}: FLAT")

    # Log recency
    if recency:
        lines.append("")
        for log_name, age in recency.items():
            lines.append(f"Last log: {age}")

    # Trade staleness alert
    if stale_trades:
        lines.append("")
        lines.append("\u26a0\ufe0f <b>STALE TRADES (no entry in 24h+):</b>")
        for sym, reason in stale_trades.items():
            if sym != "_error":
                lines.append(f"  \u274c {sym}: {reason}")

    lines.append("\nMode: STRUCTURE-FIRST (BOS/CHOCH)")

    message = "\n".join(lines)
    print(message)

    if send_telegram(message):
        print("[healthcheck] Sent to Telegram")
    else:
        print("[healthcheck] Failed to send")


if __name__ == "__main__":
    main()
