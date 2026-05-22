#!/usr/bin/env python3
"""Cross-bot watchdog for the Telegram MCP bridge.

Pings Chen_Luigi_Bot (the conversation bot used by the Claude Code
Telegram plugin) every 5 minutes via Telegram's getMe endpoint. Also
checks that the plugin's bun process is alive. Posts an alert via
luigiAlertBot (a DIFFERENT bot on a DIFFERENT transport path) when the
conversation bridge transitions:

  up   -> down  : "⚠ Chen_Luigi_Bot bridge appears DOWN ..."
  down -> up    : "✅ Chen_Luigi_Bot bridge BACK UP ..."

Only sends on transitions so it doesn't spam every 5 minutes during an
outage.

State lives in logs/bridge_watchdog_state.json. Cron-driven; safe to run
manually at any time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUN_PID_FILE = Path(
    "/home/claude/packages/327adce6-6ec4-4402-890c-9d12c6e8a471/.telegram/bot.pid"
)
_BUN_ENV_FILE = Path(
    "/home/claude/packages/327adce6-6ec4-4402-890c-9d12c6e8a471/.telegram/.env"
)
_STATE_FILE = _REPO_ROOT / "logs" / "bridge_watchdog_state.json"
_LOG_FILE = _REPO_ROOT / "logs" / "bridge_watchdog.log"

# Default Telegram chat to ping with the alerts — Chen's trading channel.
DEFAULT_ALERT_CHAT_ID = "-5243679323"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _log(line: str) -> None:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{_utc_now().isoformat()}] {line}\n")


def _read_env_file(path: Path) -> dict[str, str]:
    """Lightweight .env parser. Last value wins on duplicates. No quotes."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def _telegram_request(url: str, *, data: dict[str, str] | None = None, timeout: float = 8.0) -> dict[str, Any]:
    """POST or GET against the Telegram bot API. Returns the parsed JSON
    response, or {'ok': False, 'error': ...} on transport failure."""
    encoded = urllib.parse.urlencode(data).encode() if data else None
    try:
        req = urllib.request.Request(url, data=encoded)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _bot_get_me(token: str) -> tuple[bool, str]:
    """Returns (ok, detail). ok=True only if Telegram returns ok=True."""
    if not token:
        return (False, "no token configured")
    resp = _telegram_request(f"https://api.telegram.org/bot{token}/getMe")
    if not resp.get("ok"):
        return (False, str(resp.get("error") or resp.get("description") or resp))
    return (True, resp.get("result", {}).get("username", "<unknown>"))


def _bun_process_alive() -> tuple[bool, str]:
    """Returns (alive, detail). The plugin writes its PID to bot.pid on
    startup; if that PID is gone, the MCP server is down."""
    if not _BUN_PID_FILE.exists():
        return (False, f"PID file missing: {_BUN_PID_FILE}")
    try:
        pid = int(_BUN_PID_FILE.read_text().strip())
    except ValueError:
        return (False, "PID file unreadable")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return (False, f"PID {pid} not running")
    except PermissionError:
        # Process exists but in a different uid — count as alive.
        return (True, f"PID {pid}")
    return (True, f"PID {pid}")


def _load_state() -> dict[str, Any]:
    if not _STATE_FILE.exists():
        return {"status": "unknown", "since": _utc_now().isoformat(), "alerts_sent": 0}
    try:
        return json.loads(_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"status": "unknown", "since": _utc_now().isoformat(), "alerts_sent": 0}


def _write_state(state: dict[str, Any]) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def _send_alert(text: str, *, token: str, chat_id: str) -> tuple[bool, str]:
    if not token:
        return (False, "no alerter token")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = _telegram_request(url, data={"chat_id": chat_id, "text": text})
    if not resp.get("ok"):
        return (False, str(resp.get("description") or resp.get("error") or resp))
    return (True, "sent")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--alerter-token",
        default=os.environ.get("TELEGRAM_ALERT_BOT_TOKEN", ""),
        help="Token for the alerter bot that delivers the up/down message",
    )
    p.add_argument(
        "--chat-id",
        default=os.environ.get("TELEGRAM_CHAT_ID", DEFAULT_ALERT_CHAT_ID),
        help="Telegram chat to ping with up/down notifications",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print health-check details to stdout even on healthy ticks",
    )
    args = p.parse_args(argv)

    # The conversation bot's token lives in the plugin's .env file,
    # NOT in the trading-system .env. Load it separately.
    plugin_env = _read_env_file(_BUN_ENV_FILE)
    conv_token = plugin_env.get("TELEGRAM_BOT_TOKEN", "")

    bot_ok, bot_detail = _bot_get_me(conv_token)
    proc_ok, proc_detail = _bun_process_alive()
    healthy = bot_ok and proc_ok

    state = _load_state()
    prev_status = state.get("status", "unknown")
    now_status = "up" if healthy else "down"

    transitioned = (
        (prev_status in ("down", "unknown") and now_status == "up" and prev_status != "unknown")
        or (prev_status in ("up", "unknown") and now_status == "down")
    )
    # Special-case first observation: don't alert on unknown→up, but DO
    # alert on unknown→down (because that means we're starting in a
    # degraded state, which is worth knowing).
    if prev_status == "unknown" and now_status == "up":
        transitioned = False

    line = (
        f"status={now_status} bot={bot_detail!r} ({'ok' if bot_ok else 'fail'}) "
        f"proc={proc_detail!r} ({'ok' if proc_ok else 'fail'}) prev={prev_status}"
    )
    _log(line)
    if args.verbose or transitioned:
        print(line)

    if transitioned:
        if now_status == "down":
            text = (
                f"⚠ Chen_Luigi_Bot bridge appears DOWN.\n"
                f"  · bot getMe: {'ok' if bot_ok else 'FAIL — ' + bot_detail}\n"
                f"  · bun process: {'ok' if proc_ok else 'FAIL — ' + proc_detail}\n"
                f"  · since {_utc_now().isoformat()}"
            )
        else:
            since = state.get("since")
            since_text = f"down since {since}" if since else "down for unknown duration"
            text = (
                f"✅ Chen_Luigi_Bot bridge BACK UP.\n"
                f"  · {since_text}\n"
                f"  · bot: {bot_detail}, process: {proc_detail}"
            )
        sent_ok, sent_detail = _send_alert(
            text,
            token=args.alerter_token,
            chat_id=args.chat_id,
        )
        _log(f"alert sent={sent_ok} detail={sent_detail!r}")
        state["alerts_sent"] = int(state.get("alerts_sent", 0)) + (1 if sent_ok else 0)

    if now_status != prev_status:
        state["status"] = now_status
        state["since"] = _utc_now().isoformat()
        state["last_check"] = _utc_now().isoformat()
        state["last_detail"] = line
        _write_state(state)
    else:
        # Refresh last_check timestamp so external monitors can verify
        # the watchdog itself is running.
        state["last_check"] = _utc_now().isoformat()
        state["last_detail"] = line
        _write_state(state)

    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
