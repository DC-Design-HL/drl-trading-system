"""
Sentiment scoring for news items.
Primary: Groq API (free tier, llama-3.1-8b-instant) — 30 req/min, 14,400/day
Fallback: keyword-based scorer (no external calls)
"""
import json
import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a crypto market sentiment analyst. Given a news headline and optional body text, output ONLY valid JSON (no markdown, no explanation) with these fields:
- sentiment: float from -1.0 (extremely bearish) to +1.0 (extremely bullish) for crypto markets
- confidence: float 0.0 to 1.0 (how certain you are)
- urgency: int 1 (background noise), 2 (notable), or 3 (high-impact flash event)
- event_type: one of: regulatory, macro, geopolitical, influencer, exchange, hack, adoption, other
- assets: list of tickers affected, using only values from [BTC, ETH, SOL, XRP, ALL]
- reasoning: one sentence max

Urgency 3 criteria (only use sparingly): imminent regulatory ban/approval, major exchange hack/insolvency, head-of-state crypto statement, systemic risk event, war/sanctions escalation with direct crypto impact."""

BULLISH_KEYWORDS = [
    "approval", "approved", "adopt", "adoption", "partnership", "rally", "breakout",
    "etf approved", "legal tender", "bought", "purchased", "accumulate", "bullish",
    "surge", "soar", "record high", "all-time high", "institutional", "launch",
    "wins", "victory", "cleared", "dismissed", "not guilty",
]
BEARISH_KEYWORDS = [
    "ban", "banned", "hack", "hacked", "exploit", "crash", "bankrupt", "insolvency",
    "sec sues", "arrest", "sanctions", "war", "tariff", "inflation", "rate hike",
    "investigation", "fine", "penalty", "shutdown", "restrict", "illegal",
    "plunge", "collapse", "stolen", "loses", "lost", "down", "drop",
]


def score_keyword(text: str) -> Dict:
    """Fast keyword-based scorer. Used as fallback when OpenAI is unavailable."""
    lower = text.lower()
    bullish = sum(1 for kw in BULLISH_KEYWORDS if kw in lower)
    bearish = sum(1 for kw in BEARISH_KEYWORDS if kw in lower)
    total = bullish + bearish
    if total == 0:
        score = 0.0
    else:
        score = (bullish - bearish) / total
    return {
        "sentiment_score": round(score, 2),
        "confidence": 0.30,  # low confidence — keyword matching is crude
        "urgency": 1,
        "event_type": "other",
        "reasoning": f"keyword: +{bullish}b/-{bearish}br",
        "scorer_method": "keyword",
    }


class GPTScorer:
    """
    Groq-based sentiment scorer (free tier: 30 req/min, 14,400/day).
    Uses llama-3.1-8b-instant — fast, accurate, free.
    Falls back to keyword scorer on any failure.
    """

    MAX_ITEMS_PER_MINUTE = 25   # stay under 30/min limit with headroom
    _call_timestamps: list = []

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
            except ImportError:
                logger.error("groq package not installed — run: pip install groq")
                raise
        return self._client

    def _within_rate_limit(self) -> bool:
        import time
        now = time.time()
        # Keep only timestamps in the last 60s
        GPTScorer._call_timestamps = [t for t in GPTScorer._call_timestamps if now - t < 60]
        if len(GPTScorer._call_timestamps) >= self.MAX_ITEMS_PER_MINUTE:
            logger.warning("Groq rate limit: %d calls in last 60s — falling back to keyword", len(GPTScorer._call_timestamps))
            return False
        return True

    def score(self, title: str, body_snippet: str = "", source: str = "") -> Dict:
        """Score a single item. Falls back to keyword scorer on any failure."""
        if not self._api_key or not self._within_rate_limit():
            return score_keyword(f"{title} {body_snippet}")

        try:
            client = self._get_client()
            user_msg = f"HEADLINE: {title}"
            if body_snippet:
                user_msg += f"\nBODY: {body_snippet[:200]}"
            if source:
                user_msg += f"\nSOURCE: {source}"

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=200,
                timeout=10,
            )
            import time
            GPTScorer._call_timestamps.append(time.time())

            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)

            return {
                "sentiment_score": float(parsed.get("sentiment", 0.0)),
                "confidence": float(parsed.get("confidence", 0.5)),
                "urgency": int(parsed.get("urgency", 1)),
                "event_type": str(parsed.get("event_type", "other")),
                "assets_gpt": parsed.get("assets", []),
                "reasoning": str(parsed.get("reasoning", ""))[:200],
                "scorer_method": "groq",
            }

        except json.JSONDecodeError as e:
            logger.warning("Groq returned non-JSON: %s — falling back", e)
            return score_keyword(f"{title} {body_snippet}")
        except Exception as e:
            logger.warning("Groq score failed: %s — falling back", e)
            return score_keyword(f"{title} {body_snippet}")
