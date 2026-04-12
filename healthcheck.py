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


def check_log_recency() -> dict:
    """Check how recent each bot's last log entry is."""
    recency = {}
    bot_log = REPO / "logs" / "bots_live.log"
    if bot_log.exists():
        mtime = bot_log.stat().st_mtime
        age_min = (time.time() - mtime) / 60
        recency["bots_live.log"] = f"{age_min:.0f}m ago"
    return recency


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
    balance = state.get("total_balance", 0)

    # 3. Log recency
    recency = check_log_recency()

    # Build message
    status_icon = "\u2705" if all_alive else "\u26a0\ufe0f"
    lines = [
        f"{status_icon} <b>Hourly Health Check</b> — {now}",
        f"Services: {alive_count}/{total_count} alive",
    ]

    # Dead services
    dead = [name for name, info in services.items() if isinstance(info, dict) and not info.get("alive")]
    if dead:
        lines.append(f"\u274c DEAD: {', '.join(dead)}")

    lines.append(f"Balance: ${balance:,.2f}")
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

    lines.append("\nMode: STRUCTURE-FIRST (BOS/CHOCH)")

    message = "\n".join(lines)
    print(message)

    if send_telegram(message):
        print("[healthcheck] Sent to Telegram")
    else:
        print("[healthcheck] Failed to send")


if __name__ == "__main__":
    main()
