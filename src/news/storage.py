"""
News events DB read/write methods.
Uses the same SQLite connection pattern as SQLiteStorage.
"""
import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NewsStorage:
    """Thin SQLite wrapper for news_events table."""

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self._local = threading.local()
        # Ensure schema exists (migration-safe — storage.py _init_schema handles table creation)
        logger.info("📰 NewsStorage ready: %s", self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path), timeout=10, check_same_thread=False
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    # ── Deduplication helpers ──

    @staticmethod
    def url_hash(url: str) -> str:
        return hashlib.md5(url.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def title_hash(title: str) -> str:
        normalized = title.lower().strip()[:100]
        return hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()

    def is_duplicate(self, url_hash: str, title_hash: str) -> bool:
        """Return True if this url or title was seen in the last 24 hours."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT id FROM news_events
               WHERE (url_hash = ? OR title_hash = ?)
               AND fetched_at >= datetime('now', '-24 hours')
               LIMIT 1""",
            (url_hash, title_hash),
        ).fetchone()
        return row is not None

    # ── Write ──

    def save_event(self, event: Dict) -> Optional[int]:
        """
        Persist a news event. Returns the inserted row id, or None on failure.
        Event dict keys match the news_events columns.
        """
        try:
            conn = self._get_conn()
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """INSERT INTO news_events
                   (url_hash, title_hash, source, title, body_snippet,
                    published_at, fetched_at, processed_at,
                    sentiment_score, confidence, urgency, event_type,
                    assets, reasoning, alerted, scorer_method, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.get("url_hash", ""),
                    event.get("title_hash", ""),
                    event.get("source", "unknown"),
                    event.get("title", ""),
                    event.get("body_snippet"),
                    event.get("published_at", now),
                    event.get("fetched_at", now),
                    event.get("processed_at"),
                    event.get("sentiment_score"),
                    event.get("confidence"),
                    event.get("urgency", 1),
                    event.get("event_type"),
                    json.dumps(event.get("assets", [])),
                    event.get("reasoning"),
                    int(event.get("alerted", False)),
                    event.get("scorer_method", "keyword"),
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error("NewsStorage.save_event failed: %s", e)
            return None

    def mark_alerted(self, event_id: int) -> None:
        try:
            conn = self._get_conn()
            conn.execute("UPDATE news_events SET alerted = 1 WHERE id = ?", (event_id,))
            conn.commit()
        except Exception as e:
            logger.error("NewsStorage.mark_alerted failed: %s", e)

    # ── Read ──

    def get_recent(self, hours: int = 4, min_urgency: int = 1, limit: int = 50) -> List[Dict]:
        """Fetch recent events ordered newest-first."""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT * FROM news_events
                   WHERE fetched_at >= datetime('now', ? || ' hours')
                   AND urgency >= ?
                   ORDER BY id DESC LIMIT ?""",
                (f"-{hours}", min_urgency, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error("NewsStorage.get_recent failed: %s", e)
            return []

    def get_unalerted(self, min_urgency: int = 2) -> List[Dict]:
        """Fetch events that haven't been sent as Telegram alerts yet."""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT * FROM news_events
                   WHERE alerted = 0 AND urgency >= ?
                   AND fetched_at >= datetime('now', '-6 hours')
                   ORDER BY urgency DESC, id ASC""",
                (min_urgency,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error("NewsStorage.get_unalerted failed: %s", e)
            return []

    def get_rolling_sentiment(self, hours: int = 4, assets: Optional[List[str]] = None) -> Dict:
        """
        Compute weighted rolling sentiment average for the dashboard / Phase 2 signal.
        Weights: urgency 3 = 3x, urgency 2 = 2x, urgency 1 = 1x.
        """
        try:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT sentiment_score, confidence, urgency, assets
                   FROM news_events
                   WHERE fetched_at >= datetime('now', ? || ' hours')
                   AND sentiment_score IS NOT NULL
                   AND confidence IS NOT NULL""",
                (f"-{hours}",),
            ).fetchall()

            if not rows:
                return {"score": 0.0, "confidence": 0.0, "urgency_max": 1, "event_count": 0}

            total_weight = 0.0
            weighted_score = 0.0
            weighted_conf = 0.0
            urgency_max = 1

            for r in rows:
                row_assets = json.loads(r["assets"] or "[]")
                if assets and not any(a in row_assets or "ALL" in row_assets for a in assets):
                    continue
                w = float(r["urgency"] or 1)
                weighted_score += (r["sentiment_score"] or 0.0) * w
                weighted_conf += (r["confidence"] or 0.0) * w
                total_weight += w
                if (r["urgency"] or 1) > urgency_max:
                    urgency_max = r["urgency"]

            if total_weight == 0:
                return {"score": 0.0, "confidence": 0.0, "urgency_max": 1, "event_count": 0}

            return {
                "score": round(weighted_score / total_weight, 3),
                "confidence": round(weighted_conf / total_weight, 3),
                "urgency_max": urgency_max,
                "event_count": len(rows),
            }
        except Exception as e:
            logger.error("NewsStorage.get_rolling_sentiment failed: %s", e)
            return {"score": 0.0, "confidence": 0.0, "urgency_max": 1, "event_count": 0}

    def cleanup_old_events(self, days: int = 7) -> int:
        """Delete events older than `days` days. Returns number of rows deleted."""
        try:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM news_events WHERE created_at < datetime('now', ? || ' days')",
                (f"-{days}",),
            )
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error("NewsStorage.cleanup_old_events failed: %s", e)
            return 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        d = dict(row)
        if d.get("assets"):
            try:
                d["assets"] = json.loads(d["assets"])
            except (json.JSONDecodeError, TypeError):
                d["assets"] = []
        return d
