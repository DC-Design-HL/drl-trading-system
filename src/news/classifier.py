"""
Urgency tier classification and event taxonomy for news items.
Applies urgency boosts based on event type keywords, then maps to final tier.
"""
from typing import Dict, List

# Keywords that boost urgency for each event type
EVENT_TYPE_CONFIG = {
    "regulatory":   {"keywords": ["sec", "cftc", "ban", "banned", "legal", "court", "ruling", "law", "regulation", "regulator"], "urgency_boost": 1},
    "macro":        {"keywords": ["fed", "fomc", "rate hike", "rate cut", "inflation", "cpi", "gdp", "recession", "treasury"], "urgency_boost": 0},
    "geopolitical": {"keywords": ["war", "sanctions", "iran", "russia", "china", "tariff", "trump", "executive order", "military"], "urgency_boost": 1},
    "influencer":   {"keywords": ["elon", "trump", "musk", "michael saylor", "cathie wood", "jack dorsey"], "urgency_boost": 2},
    "exchange":     {"keywords": ["binance", "coinbase", "kraken", "bybit", "okx", "halts", "withdrawals", "suspended", "insolvent"], "urgency_boost": 1},
    "hack":         {"keywords": ["hack", "hacked", "exploit", "stolen", "bridge attack", "reentrancy", "rug pull", "phishing"], "urgency_boost": 2},
    "adoption":     {"keywords": ["etf approved", "legal tender", "institutional", "accumulate", "buys bitcoin", "spot etf"], "urgency_boost": -1},
}


def classify(item: Dict) -> Dict:
    """
    Determine final event_type and adjust urgency based on content.
    Returns updated item dict with event_type and urgency finalized.
    """
    text = (item.get("normalized_text") or "").lower()
    sentiment_score = item.get("sentiment_score", 0.0) or 0.0
    confidence = item.get("confidence", 0.0) or 0.0
    base_urgency = item.get("urgency", 1)

    # Determine event_type from content if GPT didn't provide a good one
    event_type = item.get("event_type", "other")
    if event_type in ("other", None, ""):
        for etype, cfg in EVENT_TYPE_CONFIG.items():
            if any(kw in text for kw in cfg["keywords"]):
                event_type = etype
                break

    # Apply urgency boost based on event type
    boost = EVENT_TYPE_CONFIG.get(event_type, {}).get("urgency_boost", 0)
    adjusted_urgency = min(3, max(1, base_urgency + boost))

    # Downgrade urgency if confidence is low
    if confidence < 0.5 and adjusted_urgency == 3:
        adjusted_urgency = 2

    # Ensure Tier 3 requires strong sentiment signal
    if adjusted_urgency == 3 and abs(sentiment_score) < 0.5:
        adjusted_urgency = 2

    return {
        **item,
        "event_type": event_type,
        "urgency": adjusted_urgency,
    }
