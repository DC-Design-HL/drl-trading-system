"""
Multi-Source Crypto News Aggregator

Aggregates news and sentiment from multiple crypto-specific sources:
- CryptoPanic (50% weight) - Real-time crypto news aggregation
- Reddit (30% weight) - r/CryptoCurrency, r/Bitcoin community sentiment
- CryptoCompare (20% weight) - Verified crypto news API

Provides superior crypto-specific sentiment vs general news APIs.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("praw not available - install with: pip install praw")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("textblob not available")


class CryptoPanicClient:
    """
    CryptoPanic API client - Primary crypto news source.

    Free tier: Unlimited with public feed
    API: https://cryptopanic.com/developers/api/

    Features:
    - Real-time crypto news aggregation
    - Community voting (positive/negative/important)
    - Multi-source aggregation (500+ sources)
    - Currency-specific filtering
    """

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token or os.environ.get('CRYPTOPANIC_TOKEN', '')
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    def get_news(
        self,
        currencies: str = "BTC",
        filter_type: str = "hot",  # hot, rising, bullish, bearish
        max_results: int = 50
    ) -> List[Dict]:
        """
        Fetch crypto news from CryptoPanic.

        Args:
            currencies: Comma-separated currency codes (BTC, ETH, SOL, XRP)
            filter_type: Filter type (hot, rising, bullish, bearish)
            max_results: Maximum posts to return

        Returns:
            List of news posts with voting data
        """
        cache_key = f"{currencies}_{filter_type}"

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data

        params = {
            'currencies': currencies,
            'filter': filter_type,
        }

        if self.auth_token:
            params['auth_token'] = self.auth_token
        else:
            params['public'] = 'true'

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])[:max_results]

                self.cache[cache_key] = (time.time(), results)
                logger.info(f"📰 CryptoPanic: {len(results)} posts for {currencies}")
                return results
            else:
                logger.warning(f"CryptoPanic error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"CryptoPanic error: {e}")
            return []

    def calculate_sentiment(self, posts: List[Dict]) -> Dict:
        """
        Calculate sentiment from CryptoPanic posts using voting data.

        Args:
            posts: List of CryptoPanic posts

        Returns:
            Sentiment analysis with score -1 to +1
        """
        if not posts:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'post_count': 0,
                'bullish_count': 0,
                'bearish_count': 0
            }

        sentiments = []
        bullish_count = 0
        bearish_count = 0

        for post in posts:
            votes = post.get('votes', {})
            positive = votes.get('positive', 0)
            negative = votes.get('negative', 0)
            important = votes.get('important', 0)

            total_votes = positive + negative + max(important, 1)

            # Calculate sentiment from votes
            if total_votes > 0:
                vote_sentiment = (positive - negative) / total_votes
                sentiments.append(vote_sentiment)

                # Count bullish/bearish
                if vote_sentiment > 0.2:
                    bullish_count += 1
                elif vote_sentiment < -0.2:
                    bearish_count += 1

        # Aggregate sentiment
        avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0
        sentiment_std = float(np.std(sentiments)) if len(sentiments) > 1 else 0.0

        # Confidence based on agreement
        confidence = 1.0 - min(sentiment_std, 1.0)

        return {
            'sentiment': avg_sentiment,
            'confidence': confidence,
            'post_count': len(posts),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'recent_posts': [p.get('title', '') for p in posts[:5]]
        }


class RedditClient:
    """
    Reddit API client for crypto community sentiment.

    Free tier: Unlimited (rate limit: 60 requests/min)
    API: https://www.reddit.com/dev/api

    Features:
    - Community sentiment (r/CryptoCurrency, r/Bitcoin, etc.)
    - Upvote ratios
    - Comment sentiment analysis
    - Hot/Rising/Top filtering
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "DRL-Trading-Bot/1.0"
    ):
        self.client_id = client_id or os.environ.get('REDDIT_CLIENT_ID', '')
        self.client_secret = client_secret or os.environ.get('REDDIT_CLIENT_SECRET', '')
        self.user_agent = user_agent

        self.reddit = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

        # Initialize PRAW if credentials available
        if PRAW_AVAILABLE and self.client_id and self.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("✅ Reddit client initialized")
            except Exception as e:
                logger.warning(f"Reddit initialization failed: {e}")
        else:
            logger.warning("Reddit credentials not set - using fallback")

    def get_posts(
        self,
        symbol: str = "BTC",
        subreddits: List[str] = None,
        time_filter: str = "day",
        limit: int = 50
    ) -> List[Dict]:
        """
        Get Reddit posts about a cryptocurrency.

        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)
            subreddits: List of subreddits to search
            time_filter: Time filter (hour, day, week, month)
            limit: Maximum posts to retrieve

        Returns:
            List of post dictionaries
        """
        if subreddits is None:
            subreddits = ['CryptoCurrency', 'Bitcoin', 'CryptoMarkets']

        cache_key = f"{symbol}_{time_filter}"

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data

        posts = []

        if not self.reddit:
            logger.warning("Reddit client not available")
            return posts

        try:
            # Search each subreddit
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Get hot posts mentioning the symbol
                for post in subreddit.search(symbol, time_filter=time_filter, limit=limit):
                    posts.append({
                        'title': post.title,
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments,
                        'created_utc': post.created_utc,
                        'url': post.url,
                        'subreddit': subreddit_name
                    })

            self.cache[cache_key] = (time.time(), posts)
            logger.info(f"🔴 Reddit: {len(posts)} posts for {symbol}")
            return posts

        except Exception as e:
            logger.error(f"Reddit error: {e}")
            return []

    def calculate_sentiment(self, posts: List[Dict]) -> Dict:
        """
        Calculate sentiment from Reddit posts.

        Uses upvote ratio and score as sentiment indicators.

        Args:
            posts: List of Reddit posts

        Returns:
            Sentiment analysis
        """
        if not posts:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'post_count': 0,
                'avg_upvote_ratio': 0.0,
                'avg_score': 0
            }

        sentiments = []
        upvote_ratios = []
        scores = []

        for post in posts:
            # Upvote ratio as sentiment (0.5 = neutral, >0.5 = positive, <0.5 = negative)
            upvote_ratio = post.get('upvote_ratio', 0.5)
            upvote_ratios.append(upvote_ratio)

            # Convert to -1 to +1 scale
            sentiment = (upvote_ratio - 0.5) * 2  # 0.5 → 0, 1.0 → +1, 0.0 → -1

            # Weight by score (popular posts matter more)
            score = post.get('score', 0)
            scores.append(score)

            sentiments.append(sentiment)

        # Weighted average by score
        total_score = sum(scores)
        if total_score > 0:
            weighted_sentiment = sum(s * sc for s, sc in zip(sentiments, scores)) / total_score
        else:
            weighted_sentiment = float(np.mean(sentiments)) if sentiments else 0.0

        # Confidence based on post count and agreement
        confidence = min(len(posts) / 20, 1.0)  # More posts = higher confidence

        return {
            'sentiment': float(np.clip(weighted_sentiment, -1, 1)),
            'confidence': confidence,
            'post_count': len(posts),
            'avg_upvote_ratio': float(np.mean(upvote_ratios)) if upvote_ratios else 0.5,
            'avg_score': int(np.mean(scores)) if scores else 0,
            'top_posts': [p.get('title', '') for p in sorted(posts, key=lambda x: x.get('score', 0), reverse=True)[:3]]
        }


class CryptoCompareClient:
    """
    CryptoCompare News API client.

    Free tier: 100,000 calls/month (3,333/day)
    API: https://min-api.cryptocompare.com/

    Features:
    - High-quality crypto news
    - Sentiment classification
    - Category filtering
    - Source credibility scoring
    """

    BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('CRYPTOCOMPARE_API_KEY', '')
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    def get_news(
        self,
        categories: str = "BTC",
        max_results: int = 50
    ) -> List[Dict]:
        """
        Get crypto news from CryptoCompare.

        Args:
            categories: Categories to filter (BTC, ETH, Trading, etc.)
            max_results: Maximum articles to return

        Returns:
            List of news articles
        """
        cache_key = f"{categories}"

        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data

        params = {
            'categories': categories,
        }

        if self.api_key:
            params['api_key'] = self.api_key

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                articles_data = data.get('Data', [])

                # Ensure articles_data is a list before slicing
                if isinstance(articles_data, list):
                    articles = articles_data[:max_results]
                else:
                    logger.warning(f"CryptoCompare returned unexpected data type: {type(articles_data)}")
                    return []

                self.cache[cache_key] = (time.time(), articles)
                logger.info(f"📊 CryptoCompare: {len(articles)} articles for {categories}")
                return articles
            else:
                logger.warning(f"CryptoCompare error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"CryptoCompare error: {e}")
            return []

    def calculate_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Calculate sentiment from CryptoCompare articles.

        Args:
            articles: List of articles

        Returns:
            Sentiment analysis
        """
        if not articles:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'article_count': 0
            }

        sentiments = []

        for article in articles:
            title = article.get('title', '')
            body = article.get('body', '')

            # Use TextBlob for sentiment if available
            if TEXTBLOB_AVAILABLE:
                text = f"{title} {body[:200]}"  # Limit body length
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                sentiments.append(sentiment)

        if not sentiments:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'article_count': len(articles)
            }

        avg_sentiment = float(np.mean(sentiments))
        sentiment_std = float(np.std(sentiments)) if len(sentiments) > 1 else 0.0
        confidence = 1.0 - min(sentiment_std, 1.0)

        return {
            'sentiment': avg_sentiment,
            'confidence': confidence,
            'article_count': len(articles),
            'recent_articles': [a.get('title', '') for a in articles[:5]]
        }


class CryptoNewsAggregator:
    """
    Multi-source crypto news aggregator.

    Combines:
    - CryptoPanic (50% weight) - Real-time crypto news
    - Reddit (30% weight) - Community sentiment
    - CryptoCompare (20% weight) - Verified news

    Provides comprehensive crypto-specific sentiment analysis.
    """

    # High-impact keywords (same as before)
    HIGH_IMPACT_KEYWORDS = {
        'bearish': [
            'hack', 'hacked', 'exploit', 'exploited', 'stolen', 'theft',
            'security breach', 'vulnerability', 'scam', 'fraud',
            'SEC', 'lawsuit', 'sued', 'charges', 'investigation',
            'ban', 'banned', 'regulation', 'crackdown', 'illegal',
            'bankrupt', 'bankruptcy', 'insolvent', 'collapse',
            'shut down', 'shutdown', 'ceased operations',
            'dumping', 'crash', 'plunge', 'tank', 'bloodbath'
        ],
        'bullish': [
            'ETF approved', 'ETF approval', 'approved ETF',
            'adoption', 'partnership', 'integrates', 'integration',
            'MicroStrategy', 'Michael Saylor', 'buying', 'accumulating',
            'bullish', 'rally', 'surge', 'moon', 'pump',
            'all-time high', 'ATH', 'breakthrough', 'milestone',
            'institutional', 'Wall Street', 'Goldman Sachs', 'BlackRock',
            'halvening', 'halving', 'upgrade', 'launch'
        ]
    }

    def __init__(self, symbol: str = "BTC"):
        """
        Initialize multi-source aggregator.

        Args:
            symbol: Crypto symbol (BTC, ETH, SOL, XRP)
        """
        self.symbol = symbol

        # Initialize clients
        self.cryptopanic = CryptoPanicClient()
        self.reddit = RedditClient()
        self.cryptocompare = CryptoCompareClient()

        # Symbol mappings
        self.symbol_map = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'SOL': 'Solana',
            'XRP': 'Ripple'
        }

        # Sentiment history
        self.sentiment_history: deque = deque(maxlen=24)

        logger.info(f"📰 CryptoNewsAggregator initialized for {symbol}")

    def get_aggregated_sentiment(self, hours: int = 4) -> Dict:
        """
        Get aggregated sentiment from all sources.

        Args:
            hours: Look back hours (for filtering)

        Returns:
            Comprehensive sentiment analysis
        """
        crypto_name = self.symbol_map.get(self.symbol, self.symbol)

        # Source 1: CryptoPanic (50% weight)
        cryptopanic_data = {'sentiment': 0.0, 'confidence': 0.0, 'post_count': 0}
        try:
            posts = self.cryptopanic.get_news(currencies=self.symbol, filter_type='hot')
            if posts:
                cryptopanic_data = self.cryptopanic.calculate_sentiment(posts)
        except Exception as e:
            logger.warning(f"CryptoPanic failed: {e}")

        # Source 2: Reddit (30% weight)
        reddit_data = {'sentiment': 0.0, 'confidence': 0.0, 'post_count': 0}
        try:
            reddit_posts = self.reddit.get_posts(symbol=crypto_name, time_filter='day')
            if reddit_posts:
                reddit_data = self.reddit.calculate_sentiment(reddit_posts)
        except Exception as e:
            logger.warning(f"Reddit failed: {e}")

        # Source 3: CryptoCompare (20% weight)
        cryptocompare_data = {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
        try:
            articles = self.cryptocompare.get_news(categories=self.symbol)
            if articles:
                cryptocompare_data = self.cryptocompare.calculate_sentiment(articles)
        except Exception as e:
            logger.warning(f"CryptoCompare failed: {e}")

        # Weighted aggregation
        weights = {
            'cryptopanic': 0.50,
            'reddit': 0.30,
            'cryptocompare': 0.20
        }

        # Calculate weighted sentiment
        total_weight = 0
        weighted_sentiment = 0

        if cryptopanic_data['confidence'] > 0:
            weighted_sentiment += cryptopanic_data['sentiment'] * weights['cryptopanic']
            total_weight += weights['cryptopanic']

        if reddit_data['confidence'] > 0:
            weighted_sentiment += reddit_data['sentiment'] * weights['reddit']
            total_weight += weights['reddit']

        if cryptocompare_data['confidence'] > 0:
            weighted_sentiment += cryptocompare_data['sentiment'] * weights['cryptocompare']
            total_weight += weights['cryptocompare']

        final_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0

        # Aggregate confidence (average of source confidences)
        confidences = [
            cryptopanic_data['confidence'],
            reddit_data['confidence'],
            cryptocompare_data['confidence']
        ]
        avg_confidence = float(np.mean([c for c in confidences if c > 0])) if any(confidences) else 0.0

        # Detect high-impact events from CryptoPanic
        high_impact_events = self._detect_high_impact_events(
            cryptopanic_data.get('recent_posts', []) +
            reddit_data.get('top_posts', []) +
            cryptocompare_data.get('recent_articles', [])
        )

        # Track history
        self.sentiment_history.append({
            'timestamp': datetime.now().isoformat(),
            'sentiment': final_sentiment,
            'confidence': avg_confidence
        })

        # Calculate trend
        trend = self._calculate_trend()

        result = {
            'sentiment': float(np.clip(final_sentiment, -1, 1)),
            'confidence': avg_confidence,
            'trend': trend,

            # Source breakdown
            'sources': {
                'cryptopanic': cryptopanic_data,
                'reddit': reddit_data,
                'cryptocompare': cryptocompare_data
            },

            # Aggregated counts
            'total_sources': sum([
                1 if cryptopanic_data['post_count'] > 0 else 0,
                1 if reddit_data['post_count'] > 0 else 0,
                1 if cryptocompare_data['article_count'] > 0 else 0
            ]),

            # High-impact events
            'high_impact_events': high_impact_events,
            'bearish_events': [e for e in high_impact_events if e['type'] == 'bearish'],
            'bullish_events': [e for e in high_impact_events if e['type'] == 'bullish'],
        }

        # Log summary
        sentiment_emoji = "🟢" if final_sentiment > 0.2 else ("🔴" if final_sentiment < -0.2 else "⚪")
        logger.info(
            f"📰 Aggregated sentiment [{self.symbol}]: {sentiment_emoji} {final_sentiment:+.2f} "
            f"(confidence={avg_confidence:.2f}, sources={result['total_sources']}/3, "
            f"trend={trend})"
        )

        return result

    def _detect_high_impact_events(self, texts: List[str]) -> List[Dict]:
        """Detect high-impact events from text."""
        events = []

        for text in texts:
            text_lower = text.lower()

            # Check bearish keywords
            for keyword in self.HIGH_IMPACT_KEYWORDS['bearish']:
                if keyword in text_lower:
                    events.append({
                        'text': text,
                        'keyword': keyword,
                        'type': 'bearish'
                    })
                    break

            # Check bullish keywords
            for keyword in self.HIGH_IMPACT_KEYWORDS['bullish']:
                if keyword in text_lower:
                    events.append({
                        'text': text,
                        'keyword': keyword,
                        'type': 'bullish'
                    })
                    break

        return events

    def _calculate_trend(self) -> str:
        """Calculate sentiment trend from history."""
        if len(self.sentiment_history) < 6:
            return 'unknown'

        recent = list(self.sentiment_history)[-3:]
        previous = list(self.sentiment_history)[-6:-3]

        recent_avg = np.mean([h['sentiment'] for h in recent])
        previous_avg = np.mean([h['sentiment'] for h in previous])

        change = recent_avg - previous_avg

        if change > 0.15:
            return 'improving'
        elif change < -0.15:
            return 'deteriorating'
        else:
            return 'stable'

    def should_trade(self, trade_type: str, threshold: float = 0.5) -> Tuple[bool, str]:
        """
        Check if sentiment allows the trade.

        Args:
            trade_type: 'long' or 'short'
            threshold: Minimum absolute sentiment to block trades

        Returns:
            Tuple of (should_trade, reason)
        """
        sentiment_data = self.get_aggregated_sentiment()

        sentiment = sentiment_data['sentiment']
        confidence = sentiment_data['confidence']

        # Only block if high confidence
        if confidence < 0.5:
            return True, f"Low news confidence ({confidence:.2f}) - allowing trade"

        # Block LONG if strong negative sentiment
        if trade_type == "long" and sentiment < -threshold:
            bearish_count = len(sentiment_data['bearish_events'])
            sources = sentiment_data['total_sources']
            return False, f"Strong negative sentiment ({sentiment:.2f}, {bearish_count} bearish events, {sources}/3 sources agree)"

        # Block SHORT if strong positive sentiment
        if trade_type == "short" and sentiment > threshold:
            bullish_count = len(sentiment_data['bullish_events'])
            sources = sentiment_data['total_sources']
            return False, f"Strong positive sentiment ({sentiment:.2f}, {bullish_count} bullish events, {sources}/3 sources agree)"

        return True, f"News OK: sentiment={sentiment:+.2f}, confidence={confidence:.2f}, sources={sentiment_data['total_sources']}/3"

    def get_emergency_signal(self) -> Dict:
        """
        Detect emergency situations from news.

        Returns:
            Emergency status and recommended action
        """
        sentiment_data = self.get_aggregated_sentiment(hours=2)

        bearish_events = sentiment_data['bearish_events']

        # Check for critical keywords
        critical_keywords = ['hack', 'exploit', 'stolen', 'security breach', 'bankrupt']

        for event in bearish_events:
            if any(kw in event['keyword'] for kw in critical_keywords):
                return {
                    'emergency': True,
                    'severity': 'critical',
                    'action': 'close_all_positions',
                    'reason': f"CRITICAL: {event['text']}"
                }

        # Check for severe negative sentiment across all sources
        if sentiment_data['sentiment'] < -0.8 and sentiment_data['confidence'] > 0.7:
            if sentiment_data['total_sources'] >= 2:  # At least 2 sources agree
                return {
                    'emergency': True,
                    'severity': 'high',
                    'action': 'pause_trading',
                    'reason': f"Severe negative sentiment: {sentiment_data['sentiment']:.2f} ({sentiment_data['total_sources']}/3 sources)",
                    'duration_hours': 4
                }

        return {
            'emergency': False,
            'severity': 'none',
            'action': 'none',
            'reason': 'No emergency detected'
        }


# Singleton instance
_aggregator_instance: Optional[CryptoNewsAggregator] = None


def get_crypto_news_aggregator(symbol: str = "BTC") -> CryptoNewsAggregator:
    """Get or create singleton aggregator instance."""
    global _aggregator_instance
    if _aggregator_instance is None or _aggregator_instance.symbol != symbol:
        _aggregator_instance = CryptoNewsAggregator(symbol=symbol)
    return _aggregator_instance


if __name__ == "__main__":
    # Test the aggregator
    logging.basicConfig(level=logging.INFO)

    print("📰 Testing CryptoNewsAggregator...")

    aggregator = CryptoNewsAggregator(symbol="BTC")
    sentiment = aggregator.get_aggregated_sentiment()

    print("\n📊 Aggregated Sentiment:")
    print(f"  Sentiment: {sentiment['sentiment']:+.3f}")
    print(f"  Confidence: {sentiment['confidence']:.3f}")
    print(f"  Sources: {sentiment['total_sources']}/3")
    print(f"  Trend: {sentiment['trend']}")

    print("\n📰 Source Breakdown:")
    for source, data in sentiment['sources'].items():
        print(f"\n  {source.upper()}:")
        print(f"    Sentiment: {data.get('sentiment', 0):+.3f}")
        print(f"    Confidence: {data.get('confidence', 0):.3f}")
        print(f"    Items: {data.get('post_count', 0) or data.get('article_count', 0)}")
