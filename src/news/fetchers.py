"""
News data fetchers: RSS feeds and Reddit.
Each fetcher returns a list of raw item dicts with standard keys:
  url, title, body_snippet, published_at, source
"""
import logging
import re
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    ("rss:coindesk",       "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("rss:cointelegraph",  "https://cointelegraph.com/rss"),
    ("rss:theblock",       "https://www.theblock.co/rss.xml"),
    ("rss:decrypt",        "https://decrypt.co/feed"),
    ("rss:reuters",        "https://feeds.reuters.com/reuters/businessNews"),
]

REDDIT_SUBREDDITS = ["CryptoCurrency", "Bitcoin", "ethfinance", "solana"]


def _parse_published(entry) -> str:
    """Parse feedparser entry published date to ISO8601 UTC string."""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            return dt.isoformat()
        if hasattr(entry, "published") and entry.published:
            dt = parsedate_to_datetime(entry.published)
            return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    return datetime.now(timezone.utc).isoformat()


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_rss_feeds(timeout: int = 10) -> List[Dict]:
    """Fetch all configured RSS feeds. Returns list of raw items."""
    try:
        import feedparser
    except ImportError:
        logger.error("feedparser not installed — run: pip install feedparser")
        return []

    items = []
    for source_name, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "DRL-TradingBot/1.0"})
            for entry in (feed.entries or [])[:20]:  # cap at 20 per feed
                raw_body = ""
                if hasattr(entry, "summary"):
                    raw_body = _strip_html(entry.summary)
                elif hasattr(entry, "content") and entry.content:
                    raw_body = _strip_html(entry.content[0].get("value", ""))

                items.append({
                    "url": getattr(entry, "link", ""),
                    "title": getattr(entry, "title", "").strip(),
                    "body_snippet": raw_body[:280],
                    "published_at": _parse_published(entry),
                    "source": source_name,
                })
        except Exception as e:
            logger.warning("RSS fetch failed for %s: %s", source_name, e)

    logger.debug("RSS: fetched %d items from %d feeds", len(items), len(RSS_FEEDS))
    return items


def fetch_reddit(limit_per_sub: int = 10) -> List[Dict]:
    """Fetch top new posts from crypto subreddits via praw."""
    try:
        import praw
        from dotenv import load_dotenv
        import os
        load_dotenv()
    except ImportError:
        logger.warning("praw not installed — Reddit fetcher disabled")
        return []

    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        logger.debug("Reddit credentials not set — skipping Reddit fetch")
        return []

    items = []
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="DRL-TradingBot/1.0 (news sentinel)",
            check_for_async=False,
        )
        for sub_name in REDDIT_SUBREDDITS:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.new(limit=limit_per_sub):
                    # Skip low-engagement posts (likely noise)
                    if post.score < 10:
                        continue
                    published = datetime.fromtimestamp(
                        post.created_utc, tz=timezone.utc
                    ).isoformat()
                    items.append({
                        "url": f"https://reddit.com{post.permalink}",
                        "title": post.title.strip(),
                        "body_snippet": (post.selftext or "")[:280],
                        "published_at": published,
                        "source": f"reddit:{sub_name.lower()}",
                    })
            except Exception as e:
                logger.warning("Reddit fetch failed for r/%s: %s", sub_name, e)
    except Exception as e:
        logger.warning("Reddit client init failed: %s", e)

    logger.debug("Reddit: fetched %d items", len(items))
    return items
