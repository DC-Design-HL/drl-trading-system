#!/usr/bin/env python3
"""Scheduled P3-rollout health check for the DRL autonomous loop.

Durable companion to the in-session Claude monitor: runs from the server
crontab every 6h, independent of any Claude session, and pings Telegram when
something noteworthy happens during the P3 two-sided-apply rollout
(deployed 2026-06-15, feature/autonomous-loop, commit 3409422).

Checks (all read-only — this script never restarts or mutates anything):
  1. Service liveness from logs/running_services.json.
  2. Autonomous-loop state: in-flight experiment stages + the latest decision.
  3. Live override (data/self_improve/active_overrides.json) — the first time
     the loop actually moves a knob, flag it loudly.
  4. Kill switch / armed flags.
  5. Recent bot errors / circuit-breaker trips in logs/bots_live.log.

Messaging policy (Telegram chat = TELEGRAM_CHAT_ID, English — trading channel):
  * ALERT (always send): a dead service, a circuit-breaker trip, a live
    override, a new experiment or stage transition, a blocklist-removal
    escalation, or fresh errors.
  * HEARTBEAT (morning run only, local hour 8): one-line all-clear.
  * Otherwise: stay silent (no message) to avoid noise.

State is kept in logs/rollout_check_state.json so stage transitions are
detected across runs. Best-effort throughout — a check failure is reported,
never raised.

Usage:
  python3 -m scripts.self_improve.rollout_check            # normal (sends)
  python3 -m scripts.self_improve.rollout_check --dry-run  # print, don't send
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PIDS = _REPO / "logs" / "running_services.json"
_DB = _REPO / "data" / "trading.db"
_SI = _REPO / "data" / "self_improve"
_OVERRIDE = _SI / "active_overrides.json"
_ARMED = _SI / "AUTONOMY_ARMED"
_KILL = _SI / "AUTONOMY_DISABLED"
_BOTS_LOG = _REPO / "logs" / "bots_live.log"
_STATE = _REPO / "logs" / "rollout_check_state.json"

_IN_FLIGHT = (
    "proposed", "backtest", "paper", "awaiting_canary_approval", "canary", "live",
)


def _check_services() -> list[str]:
    """Return list of DEAD service names (empty = all alive)."""
    dead: list[str] = []
    try:
        data = json.loads(_PIDS.read_text())
    except Exception:
        return ["(running_services.json unreadable)"]
    for name, v in data.items():
        pid = v.get("pid") if isinstance(v, dict) else None
        if not pid:
            continue
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            dead.append(f"{name}(pid={pid})")
        except Exception:
            pass
    return dead


def _loop_state() -> dict:
    """Read in-flight experiments + the most recent decision from the DB."""
    out: dict = {"experiments": {}, "last_decision": None}
    if not _DB.exists():
        return out
    try:
        con = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True, timeout=10)
        try:
            rows = con.execute(
                "SELECT id, stage FROM experiments WHERE stage IN "
                f"({','.join('?' * len(_IN_FLIGHT))}) ORDER BY id",
                _IN_FLIGHT,
            ).fetchall()
            out["experiments"] = {str(r[0]): r[1] for r in rows}
            d = con.execute(
                "SELECT id, ts, agent, decision_type, summary, outcome "
                "FROM decisions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if d:
                out["last_decision"] = {
                    "id": d[0], "ts": d[1], "agent": d[2],
                    "type": d[3], "summary": d[4], "outcome": d[5],
                }
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"db read failed: {exc}"
    return out


def _override_info() -> str | None:
    if not _OVERRIDE.exists():
        return None
    try:
        p = json.loads(_OVERRIDE.read_text())
        cc = p.get("config_changes", {})
        return f"exp #{p.get('experiment_id')} → {json.dumps(cc, default=str)[:300]}"
    except Exception:
        return "(active_overrides.json present but unreadable)"


def _recent_errors() -> int:
    """Count Traceback / ERROR / circuit-breaker lines in the bot log tail."""
    if not _BOTS_LOG.exists():
        return 0
    try:
        out = subprocess.run(
            ["tail", "-n", "400", str(_BOTS_LOG)],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        return 0
    n = 0
    for line in out.splitlines():
        if "Traceback" in line or "CIRCUIT BREAKER" in line \
                or " ERROR " in line or line.endswith("ERROR"):
            n += 1
    return n


def _load_prev_state() -> dict:
    try:
        return json.loads(_STATE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        _STATE.write_text(json.dumps(state, indent=2, default=str))
    except Exception:
        pass


def _send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_ALERT_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "-5243679323")
    if not token:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        urllib.request.urlopen(url, data=data, timeout=10).read()
        return True
    except Exception:
        return False


def build_report() -> tuple[str | None, dict]:
    """Return (message_or_None, new_state). message=None means stay silent."""
    dead = _check_services()
    loop = _loop_state()
    override = _override_info()
    killed = _KILL.exists()
    armed = _ARMED.exists()
    errors = _recent_errors()

    prev = _load_prev_state()
    prev_exps = prev.get("experiments", {})
    cur_exps = loop.get("experiments", {})
    prev_override = prev.get("override")

    # Detect noteworthy changes vs last run.
    new_or_changed = {
        eid: st for eid, st in cur_exps.items()
        if prev_exps.get(eid) != st
    }
    override_changed = (override or None) != (prev_override or None)

    alert_bits: list[str] = []
    if dead:
        alert_bits.append("🔴 SERVICES DOWN: " + ", ".join(dead))
    if killed:
        alert_bits.append("⏸️ KILL SWITCH active (AUTONOMY_DISABLED present)")
    if override and override_changed:
        alert_bits.append(f"🤖 LIVE OVERRIDE applied — {override}")
    if new_or_changed:
        moves = ", ".join(f"#{e}→{s}" for e, s in new_or_changed.items())
        alert_bits.append(f"📈 loop stage change: {moves}")
    if errors:
        alert_bits.append(f"⚠️ {errors} error/breaker line(s) in bots_live.log tail")
    if loop.get("error"):
        alert_bits.append("⚠️ " + loop["error"])

    new_state = {
        "experiments": cur_exps,
        "override": override,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }

    if alert_bits:
        ld = loop.get("last_decision")
        ld_line = (
            f"\nLast decision: {ld['type']} by {ld['agent']} — "
            f"{(ld['summary'] or '')[:80]} ({ld['outcome']})"
            if ld else ""
        )
        msg = (
            "🛰️ P3 rollout check — ATTENTION\n"
            + "\n".join(f"• {b}" for b in alert_bits)
            + ld_line
        )
        return msg, new_state

    # Nothing noteworthy (and no dead service — that would have alerted above).
    # Heartbeat only on the morning run, to confirm the monitor is alive.
    if datetime.now().hour == 8:
        stage_txt = (
            ", ".join(f"#{e}:{s}" for e, s in cur_exps.items())
            if cur_exps else "idle (no experiment in flight)"
        )
        arm_txt = "armed" if armed else "disarmed"
        msg = (
            "🛰️ P3 rollout — all nominal. All services up. "
            f"Loop {arm_txt}; {stage_txt}. No live override."
        )
        return msg, new_state

    return None, new_state


def main() -> int:
    dry = "--dry-run" in sys.argv
    msg, new_state = build_report()
    if dry:
        print("=== rollout_check dry-run ===")
        print("message:", repr(msg))
        print("state:", json.dumps(new_state, indent=2, default=str))
        return 0
    if msg:
        _send_telegram(msg)
    _save_state(new_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
