#!/usr/bin/env python3
"""Nightly Reviewer cron entry point.

Invoked once per day at 03:00 UTC via crontab. Reads the last 24h of
closes, writes a post-mortem to logs/decisions/post-mortems/, optionally
runs the LLM pattern pass if ANTHROPIC_API_KEY is set, and Telegram-
pings if a trailing loss streak ≥ 3 is found.

Sundays: extends the window to 7 days for a weekly retrospective.
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.reviewer import run_review  # noqa: E402


def _telegram_post(token: str, chat_id: str, text: str) -> None:
    """Lightweight Telegram sendMessage. Best-effort — failures are logged
    but don't crash the cron."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "disable_notification": "false"}
    ).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if not json.loads(body).get("ok"):
                print(f"[reviewer_cron] telegram returned: {body}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[reviewer_cron] telegram post failed: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", default="data/trading.db")
    p.add_argument("--out-dir", default="logs/decisions/post-mortems")
    p.add_argument("--window-hours", type=float, default=None,
                   help="Override window (default: 24h Mon-Sat, 168h Sun)")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip the LLM pattern pass even if API key is set")
    p.add_argument("--no-telegram", action="store_true",
                   help="Don't post to Telegram regardless of flags fired")
    args = p.parse_args(argv)

    now = datetime.now(timezone.utc)
    window = args.window_hours
    if window is None:
        # Sunday = weekly retrospective
        window = 24 * 7 if now.weekday() == 6 else 24.0

    result = run_review(
        db_path=args.db,
        out_dir=args.out_dir,
        window_hours=window,
        now=now,
        enable_llm=not args.no_llm,
    )

    print(f"✅ Post-mortem written → {result.markdown_path}")
    print(f"   {result.telegram_digest}")

    # Telegram digest — always send for the daily run (and the Sunday
    # weekly), unless --no-telegram. Loss-streak warnings get a fresh
    # ping via the digest's emoji prefix.
    if not args.no_telegram:
        token = os.environ.get("TELEGRAM_ALERT_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "-5243679323")
        if token and result.telegram_digest:
            _telegram_post(token, chat_id, result.telegram_digest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
