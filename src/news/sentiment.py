"""
Sentiment scoring for news items.
Primary: GPT-4o-mini via OpenAI API
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
    """GPT-4o-mini sentiment scorer. One instance shared across the process."""

    MAX_ITEMS_PER_HOUR = 100
    _call_count = 0
    _window_start = 0.0

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                logger.error("openai package not installed")
                raise
        return self._client

    def _within_rate_limit(self) -> bool:
        import time
        now = time.time()
        if now - self._window_start > 3600:
            GPTScorer._window_start = now
            GPTScorer._call_count = 0
        if GPTScorer._call_count >= self.MAX_ITEMS_PER_HOUR:
            logger.warning("GPT rate limit reached (%d/hr) — falling back to keyword scorer", self.MAX_ITEMS_PER_HOUR)
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
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=200,
                timeout=10,
            )
            GPTScorer._call_count += 1

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
                "assets_gpt": parsed.get("assets", []),  # GPT's asset suggestion
                "reasoning": str(parsed.get("reasoning", ""))[:200],
                "scorer_method": "gpt",
            }

        except json.JSONDecodeError as e:
            logger.warning("GPT returned non-JSON: %s — falling back", e)
            return score_keyword(f"{title} {body_snippet}")
        except Exception as e:
            logger.warning("GPT score failed: %s — falling back", e)
            return score_keyword(f"{title} {body_snippet}")
