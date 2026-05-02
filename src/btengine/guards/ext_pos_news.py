"""ExtPosNewsGuard — block LONG after extreme-positive news.

Live's EXT_POS_NEWS_GUARD: if any news scored sentiment > threshold in
the trailing N hours, block LONG entries. The "fade" hypothesis was
validated 2026-04-30 (Bonferroni-significant on forward returns).

Backtest path: ctx.extras['recent_max_news_sentiment'] is the highest
sentiment score in the trailing EXT_POS_NEWS_LOOKBACK_HOURS window for
this symbol. If absent (no news data for backtest window), fail-open.
"""

from __future__ import annotations

from .. import live_constants as LC
from ..strategy.base import Guard, GuardResult, Intent


class ExtPosNewsGuard(Guard):
    name = "ext_pos_news"

    def __init__(self,
                 sentiment_threshold: float = LC.EXT_POS_NEWS_SENTIMENT_THRESHOLD):
        self.threshold = float(sentiment_threshold)

    def __call__(self, intent: Intent, ctx) -> GuardResult:
        if intent.action != "OPEN_LONG":
            return GuardResult.allow()
        recent_max = ctx.extras.get("recent_max_news_sentiment")
        if recent_max is None:
            return GuardResult.allow()
        if float(recent_max) > self.threshold:
            return GuardResult.block(
                f"ext_pos_news: recent max sentiment {recent_max:.2f} > {self.threshold}",
                recent_max_sentiment=float(recent_max),
            )
        return GuardResult.allow()
