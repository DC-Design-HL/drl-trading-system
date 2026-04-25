#!/usr/bin/env python3
"""
News Alerter — reads news_pending_alerts.jsonl and sends Telegram messages.
Mirrors the trade_alerter.py pattern: stdlib-only, tail-file loop.

Tier 3 (FLASH): bold header, immediate send
Tier 2 (NOTABLE): standard format
"""
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Self-anchor CWD so every relative path (data/*, logs/*) lands in the repo
# regardless of caller. Supersedes the earlier partial _LOG_DIR anchor below.
os.chdir(Path(__file__).resolve().parent)

from dotenv import load_dotenv
load_dotenv()

# Absolute paths anchored to this file — cron/watchdog invoke us with
# CWD=$HOME, so relative "logs/..." paths used to crash on startup.
_REPO_DIR = Path(__file__).parent
_LOG_DIR = _REPO_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NEWS-ALERT] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_DIR / "news_alerter.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)

ALERTS_FILE = _LOG_DIR / "news_pending_alerts.jsonl"
POLL_INTERVAL = 5  # seconds

BOT_TOKEN = os.getenv("TELEGRAM_ALERT_BOT_TOKEN", "")
# News alerts go to dedicated news channel only (NEWS_TELEGRAM_CHAT_IDS)
# Falls back to main chat IDs if not set
_news_ids = os.getenv("NEWS_TELEGRAM_CHAT_IDS", "")
_main_ids = os.getenv("TELEGRAM_CHAT_IDS", os.getenv("TELEGRAM_CHAT_ID", ""))
CHAT_IDS_RAW = _news_ids if _news_ids else _main_ids
CHAT_IDS = [cid.strip() for cid in CHAT_IDS_RAW.split(",") if cid.strip()]


SENTIMENT_LABEL = {
    range(-100, -60): "🔴 VERY BEARISH",
    range(-60, -25):  "🟠 BEARISH",
    range(-25, 25):   "⚪ NEUTRAL",
    range(25, 60):    "🟢 BULLISH",
    range(60, 101):   "🟢 VERY BULLISH",
}

def sentiment_label(score: float) -> str:
    pct = int(score * 100)
    for r, label in SENTIMENT_LABEL.items():
        if pct in r:
            return label
    return "⚪ NEUTRAL"


def format_alert(alert: dict) -> str:
    urgency = alert.get("urgency", 1)
    event_type = (alert.get("event_type") or "other").upper()
    score = alert.get("sentiment_score", 0.0) or 0.0
    conf = alert.get("confidence", 0.0) or 0.0
    assets = alert.get("assets", []) or []
    title = alert.get("title", "")[:120]
    source = alert.get("source", "?").replace("rss:", "")
    reasoning = alert.get("reasoning", "")[:100]

    ts = alert.get("published_at", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        ts_fmt = dt.strftime("%H:%M UTC")
    except Exception:
        ts_fmt = ts[:16]

    assets_str = " ".join(assets) if assets else "CRYPTO"
    method = alert.get("scorer_method", "keyword")
    method_tag = "🤖 GPT" if method == "gpt" else "🔑 keyword"

    if urgency == 3:
        header = "🚨 [TIER 3 FLASH EVENT]"
    else:
        header = "📰 [NEWS ALERT T2]"

    lines = [
        header,
        f"{event_type} | urgency: {urgency}/3",
        f"Sentiment: {score:+.2f} {sentiment_label(score)} | conf: {conf:.2f} {method_tag}",
        f"Assets: {assets_str}",
        f"Source: {source} | {ts_fmt}",
        "",
        f'"{title}"',
    ]
    if reasoning:
        lines.append(f"\n💬 {reasoning}")
    lines.append("\nPhase 1: MONITOR ONLY — no trade action")

    return "\n".join(lines)


def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_IDS:
        logger.warning("Telegram not configured — skipping alert send")
        return
    for chat_id in CHAT_IDS:
        try:
            payload = json.dumps({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "",
            }).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                if not result.get("ok"):
                    logger.error("Telegram API error: %s", result)
        except Exception as e:
            logger.error("Telegram send failed (chat %s): %s", chat_id, e)


def tail_alerts():
    """Tail the alerts JSONL file and process new lines."""
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERTS_FILE.touch(exist_ok=True)

    processed_lines = 0
    # On startup, skip existing lines (don't re-alert on restart)
    with open(ALERTS_FILE, "r") as f:
        processed_lines = sum(1 for _ in f)
    logger.info("Skipping %d existing alert lines on startup", processed_lines)

    while True:
        try:
            with open(ALERTS_FILE, "r") as f:
                lines = f.readlines()

            if len(lines) > processed_lines:
                new_lines = lines[processed_lines:]
                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        alert = json.loads(line)
                        if alert.get("type") != "news":
                            continue
                        urgency = alert.get("urgency", 1)
                        if urgency < 2:
                            continue
                        text = format_alert(alert)
                        logger.info(
                            "Sending Tier %d alert: %s",
                            urgency,
                            alert.get("title", "")[:60],
                        )
                        send_telegram(text)
                    except json.JSONDecodeError:
                        logger.warning("Malformed alert line: %s", line[:80])
                    except Exception as e:
                        logger.error("Alert processing error: %s", e)
                processed_lines = len(lines)

        except Exception as e:
            logger.error("File read error: %s", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    logger.info("📢 News Alerter starting")
    if not BOT_TOKEN:
        logger.warning("TELEGRAM_ALERT_BOT_TOKEN not set — alerts will not be sent")
    tail_alerts()
