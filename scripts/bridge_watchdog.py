#!/usr/bin/env python3
"""Cross-bot watchdog for the Telegram MCP bridge.

Pings Chen_Luigi_Bot (the conversation bot used by the Claude Code
Telegram plugin) via Telegram's getMe endpoint and checks that the
plugin's bun process is alive. Posts an alert via luigiAlertBot
(a DIFFERENT bot on a DIFFERENT transport path).

## Background

The Telegram bridge is normally up only while an active interaction is
being processed. The platform service @clawdeploy/agent-chat-app
spawns `claude --print --resume` for each inbound message; that
process owns a bun MCP child which writes the bot.pid; when the
interaction ends, both die and the bridge sits "down" until the next
inbound message arrives. Messages aren't lost — Telegram queues them
on its side — but the watchdog sees a "down" state.

We therefore only alert on **outages longer than DOWN_THRESHOLD_SECONDS**
(default 10 min), which is long enough that a normal wake-up cycle
should have happened. A continuously-down bridge past that point means
the wake-up isn't firing — that's a real problem worth Telegram-pinging.

## State machine

  up               : bridge healthy (getMe ok AND bun PID alive)
  down_pending     : bridge down, hasn't crossed threshold yet (no alert)
  down_alerted     : bridge down past threshold, DOWN alert sent
  unknown          : first run — initialize but never alert

Transitions and side effects:

  unknown            -> up           : silent
  unknown            -> down         : enter down_pending (no alert)
  up                 -> up           : refresh last_check
  up                 -> down         : enter down_pending (no alert; record down_since=now)
  down_pending       -> up           : silent recovery (normal idle cycle)
  down_pending       -> down (>=th)  : send DOWN alert, transition to down_alerted
  down_pending       -> down (<th)   : stay down_pending, just refresh last_check
  down_alerted       -> up           : send BACK UP alert with downtime duration
  down_alerted       -> down         : refresh last_check (already alerted)

## Files

  logs/bridge_watchdog.log         human-readable per-tick log
  logs/bridge_watchdog_state.json  state machine persistence
  logs/bridge_watchdog_cron.log    redirected stdout from cron (transitions only)

Cron-driven; safe to run manually at any time.
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

# Minimum continuous-down duration before a DOWN alert fires. Normal idle
# cycles between platform wake-ups are routinely shorter than this; only
# longer outages indicate a real wake-up failure worth notifying about.
# Override per-run with --down-threshold, or per-env via SELF_IMPROVE_BRIDGE_THRESHOLD_S.
DEFAULT_DOWN_THRESHOLD_SECONDS = 600  # 10 minutes


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
# State machine (pure function — testable without subprocess/network)
# ─────────────────────────────────────────────────────────────────────────


def decide_transition(
    prev_state: dict[str, Any],
    *,
    healthy: bool,
    now: datetime,
    down_threshold_seconds: int,
) -> tuple[dict[str, Any], str | None]:
    """Pure state-machine step. Returns (new_state, alert_kind).

    alert_kind is None when no alert should be sent, "down" when we should
    send the DOWN alert, or "up" when we should send the recovery alert.

    States: 'up', 'down_pending', 'down_alerted', 'unknown'.

    See module docstring for the full transition table.
    """
    prev_status = prev_state.get("status", "unknown")
    new_state = dict(prev_state)
    new_state["last_check"] = now.isoformat()
    alert_kind: str | None = None

    if healthy:
        # Recovery side of the state machine
        if prev_status == "down_alerted":
            alert_kind = "up"
            new_state["status"] = "up"
            new_state["since"] = now.isoformat()
            new_state["down_since"] = None
        elif prev_status in ("down_pending", "down", "unknown"):
            # Silent transition — either first observation healthy, or a
            # normal idle cycle that closed before threshold.
            new_state["status"] = "up"
            new_state["since"] = now.isoformat()
            new_state["down_since"] = None
        else:  # already up
            new_state["status"] = "up"
        return new_state, alert_kind

    # not healthy → some kind of "down"
    if prev_status in ("up", "unknown"):
        # Entered down for the first time — start the clock, no alert yet
        new_state["status"] = "down_pending"
        new_state["down_since"] = now.isoformat()
        new_state["since"] = now.isoformat()
        return new_state, None

    if prev_status == "down_pending":
        # Check if we've now been down long enough to alert
        down_since_str = new_state.get("down_since") or prev_state.get("since")
        try:
            down_since = datetime.fromisoformat(down_since_str) if down_since_str else now
        except (TypeError, ValueError):
            down_since = now
        elapsed = (now - down_since).total_seconds()
        if elapsed >= down_threshold_seconds:
            new_state["status"] = "down_alerted"
            new_state["since"] = now.isoformat()
            alert_kind = "down"
        else:
            # Stay pending, no alert
            new_state["status"] = "down_pending"
        return new_state, alert_kind

    # prev_status == "down_alerted" → still down, already alerted, refresh only
    new_state["status"] = "down_alerted"
    return new_state, None


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


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
        "--down-threshold",
        type=int,
        default=int(os.environ.get(
            "SELF_IMPROVE_BRIDGE_THRESHOLD_S",
            DEFAULT_DOWN_THRESHOLD_SECONDS,
        )),
        help=(
            "Seconds the bridge must be continuously down before a DOWN "
            f"alert fires (default {DEFAULT_DOWN_THRESHOLD_SECONDS})"
        ),
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

    now = _utc_now()
    prev_state = _load_state()
    new_state, alert_kind = decide_transition(
        prev_state,
        healthy=healthy,
        now=now,
        down_threshold_seconds=args.down_threshold,
    )

    line = (
        f"status={new_state['status']} prev={prev_state.get('status', 'unknown')} "
        f"bot={bot_detail!r} ({'ok' if bot_ok else 'fail'}) "
        f"proc={proc_detail!r} ({'ok' if proc_ok else 'fail'})"
    )
    if alert_kind:
        line += f" alert={alert_kind}"
    _log(line)
    if args.verbose or alert_kind:
        print(line)

    if alert_kind == "down":
        down_since_str = (
            new_state.get("down_since")
            or prev_state.get("down_since")
            or prev_state.get("since")
        )
        try:
            down_since = datetime.fromisoformat(down_since_str) if down_since_str else now
        except (TypeError, ValueError):
            down_since = now
        duration = (now - down_since).total_seconds()
        text = (
            f"⚠ Chen_Luigi_Bot bridge appears DOWN for {_format_duration(duration)}.\n"
            f"  · bot getMe: {'ok' if bot_ok else 'FAIL — ' + bot_detail}\n"
            f"  · bun process: {'ok' if proc_ok else 'FAIL — ' + proc_detail}\n"
            f"  · down since {down_since_str}\n"
            f"  · threshold {args.down_threshold}s exceeded"
        )
        sent_ok, sent_detail = _send_alert(
            text, token=args.alerter_token, chat_id=args.chat_id
        )
        _log(f"alert sent={sent_ok} kind=down detail={sent_detail!r}")
        new_state["alerts_sent"] = int(new_state.get("alerts_sent", 0)) + (1 if sent_ok else 0)

    elif alert_kind == "up":
        down_since_str = prev_state.get("down_since") or prev_state.get("since")
        try:
            down_since = datetime.fromisoformat(down_since_str) if down_since_str else now
        except (TypeError, ValueError):
            down_since = now
        duration = (now - down_since).total_seconds()
        text = (
            f"✅ Chen_Luigi_Bot bridge BACK UP after {_format_duration(duration)}.\n"
            f"  · down since {down_since_str}\n"
            f"  · bot: {bot_detail}, process: {proc_detail}"
        )
        sent_ok, sent_detail = _send_alert(
            text, token=args.alerter_token, chat_id=args.chat_id
        )
        _log(f"alert sent={sent_ok} kind=up detail={sent_detail!r}")
        new_state["alerts_sent"] = int(new_state.get("alerts_sent", 0)) + (1 if sent_ok else 0)

    new_state["last_detail"] = line
    _write_state(new_state)
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
