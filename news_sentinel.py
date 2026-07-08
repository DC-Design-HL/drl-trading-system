#!/usr/bin/env python3
"""
News Sentinel — Phase 1 background monitoring daemon.

Polls RSS feeds and Reddit, scores sentiment via GPT-4o-mini,
classifies urgency, persists to SQLite, and writes alerts to
logs/news_pending_alerts.jsonl for news_alerter.py to deliver.

Phase 1: monitor-only. Does NOT influence trade decisions.
Phase 2 (future): /api/market endpoint will expose rolling sentiment
                  to the signal gate as confirmation signal #5.

Usage:
    python3 news_sentinel.py
    python3 news_sentinel.py --dry-run   # fetch + score but don't write alerts or DB

Memory target: < 80MB RSS
CPU target:    < 5% average (mostly I/O wait)
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# Self-anchor CWD (defense-in-depth — see live_trading_all.py).
os.chdir(Path(__file__).resolve().parent)
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.news.fetchers import fetch_rss_feeds, fetch_reddit
from src.news.processor import process_item
from src.news.sentiment import GPTScorer, score_keyword
from src.news.classifier import classify
from src.news.storage import NewsStorage

# Absolute paths anchored to this file, so the service works regardless
# of the caller's CWD. (cron / watchdog invoke us with CWD=$HOME, which
# previously caused FileNotFoundError: '/home/claude/logs/news_sentinel.log'.)
_REPO_DIR = Path(__file__).parent
_LOG_DIR = _REPO_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NEWS] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_DIR / "news_sentinel.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Timing config ──
RSS_POLL_INTERVAL    = 90    # seconds between RSS polls
REDDIT_POLL_INTERVAL = 180   # seconds between Reddit polls
DB_CLEANUP_INTERVAL  = 3600  # seconds between DB cleanup runs
DB_CLEANUP_DAYS      = 3650  # days to retain events (was 7 — raised 2026-07-08 to
                             # stop purging the news corpus; we need history to train
                             # the news-impact model. ~45 events/day → trivial storage.)

# ── Alert output ──
ALERTS_FILE = _LOG_DIR / "news_pending_alerts.jsonl"


class NewsSentinel:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.storage = NewsStorage()
        self.scorer = GPTScorer()
        self._seen_hashes: set = set()   # in-memory dedup (url_hash + title_hash)
        self._item_queue: deque = deque(maxlen=200)
        self._stop_event = threading.Event()

        ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        logger.info("📰 News Sentinel starting (dry_run=%s)", dry_run)

    # ── Deduplication ──

    def _is_seen(self, url_hash: str, title_hash: str) -> bool:
        if url_hash in self._seen_hashes or title_hash in self._seen_hashes:
            return True
        # Also check DB (catches restarts)
        if self.storage.is_duplicate(url_hash, title_hash):
            self._seen_hashes.add(url_hash)
            self._seen_hashes.add(title_hash)
            return True
        return False

    def _mark_seen(self, url_hash: str, title_hash: str):
        self._seen_hashes.add(url_hash)
        self._seen_hashes.add(title_hash)
        # Keep cache bounded
        if len(self._seen_hashes) > 4000:
            # Clear half — simple eviction
            items = list(self._seen_hashes)
            self._seen_hashes = set(items[2000:])

    # ── Fetch threads ──

    def _rss_poll_loop(self):
        while not self._stop_event.is_set():
            try:
                items = fetch_rss_feeds()
                new_count = 0
                for item in items:
                    processed = process_item(item)
                    if processed is None:
                        continue
                    if self._is_seen(processed["url_hash"], processed["title_hash"]):
                        continue
                    self._item_queue.append(processed)
                    new_count += 1
                if new_count:
                    logger.info("RSS: %d new items queued", new_count)
            except Exception as e:
                logger.error("RSS poll error: %s", e)
            self._stop_event.wait(RSS_POLL_INTERVAL)

    def _reddit_poll_loop(self):
        while not self._stop_event.is_set():
            try:
                items = fetch_reddit()
                new_count = 0
                for item in items:
                    processed = process_item(item)
                    if processed is None:
                        continue
                    if self._is_seen(processed["url_hash"], processed["title_hash"]):
                        continue
                    self._item_queue.append(processed)
                    new_count += 1
                if new_count:
                    logger.info("Reddit: %d new items queued", new_count)
            except Exception as e:
                logger.error("Reddit poll error: %s", e)
            self._stop_event.wait(REDDIT_POLL_INTERVAL)

    # ── Processor ──

    def _process_queue(self):
        """Drain the queue: score, classify, persist, maybe alert."""
        while self._item_queue:
            item = self._item_queue.popleft()

            try:
                # Score sentiment
                scored = self.scorer.score(
                    title=item.get("title", ""),
                    body_snippet=item.get("body_snippet", ""),
                    source=item.get("source", ""),
                )
                item.update(scored)

                # Merge GPT asset suggestions with our keyword extraction
                gpt_assets = item.pop("assets_gpt", []) or []
                keyword_assets = item.get("assets", [])
                merged_assets = list(set(keyword_assets + gpt_assets)) or ["ALL"]
                item["assets"] = merged_assets

                # Classify urgency
                item = classify(item)
                item["fetched_at"] = datetime.now(timezone.utc).isoformat()
                item["processed_at"] = datetime.now(timezone.utc).isoformat()

                logger.info(
                    "📰 [U%d] [%s] %s | score=%.2f conf=%.2f | %s",
                    item["urgency"],
                    item.get("event_type", "other").upper()[:8],
                    item.get("source", "?"),
                    item.get("sentiment_score", 0),
                    item.get("confidence", 0),
                    item.get("title", "")[:80],
                )

                if self.dry_run:
                    self._mark_seen(item["url_hash"], item["title_hash"])
                    continue

                # Persist to DB
                event_id = self.storage.save_event(item)
                if event_id:
                    item["_db_id"] = event_id
                    self._mark_seen(item["url_hash"], item["title_hash"])

                    # Write alert for Tier 2+ events
                    if item["urgency"] >= 2:
                        self._write_alert(item, event_id)

            except Exception as e:
                logger.error("Process error for '%s': %s", item.get("title", "?")[:50], e)

    def _write_alert(self, item: Dict, event_id: int):
        """Append a JSONL alert entry for news_alerter.py to consume."""
        try:
            alert = {
                "type": "news",
                "event_id": event_id,
                "urgency": item["urgency"],
                "event_type": item.get("event_type", "other"),
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "sentiment_score": item.get("sentiment_score", 0),
                "confidence": item.get("confidence", 0),
                "assets": item.get("assets", []),
                "reasoning": item.get("reasoning", ""),
                "published_at": item.get("published_at", ""),
                "scorer_method": item.get("scorer_method", "keyword"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(ALERTS_FILE, "a") as f:
                f.write(json.dumps(alert) + "\n")
        except Exception as e:
            logger.error("Alert write failed: %s", e)

    # ── DB cleanup ──

    def _cleanup_loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(DB_CLEANUP_INTERVAL)
            try:
                deleted = self.storage.cleanup_old_events(days=DB_CLEANUP_DAYS)
                if deleted:
                    logger.info("DB cleanup: removed %d old news events", deleted)
            except Exception as e:
                logger.error("DB cleanup error: %s", e)

    # ── Main loop ──

    def run(self):
        threads = [
            threading.Thread(target=self._rss_poll_loop, name="rss-poller", daemon=True),
            threading.Thread(target=self._reddit_poll_loop, name="reddit-poller", daemon=True),
            threading.Thread(target=self._cleanup_loop, name="db-cleanup", daemon=True),
        ]
        for t in threads:
            t.start()

        logger.info("✅ News Sentinel running — RSS every %ds, Reddit every %ds", RSS_POLL_INTERVAL, REDDIT_POLL_INTERVAL)

        try:
            while True:
                self._process_queue()
                time.sleep(5)  # drain queue every 5s
        except KeyboardInterrupt:
            logger.info("Stopping News Sentinel...")
            self._stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="News Sentinel daemon")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and score but don't write to DB or alerts file")
    args = parser.parse_args()

    sentinel = NewsSentinel(dry_run=args.dry_run)
    sentinel.run()


if __name__ == "__main__":
    main()
