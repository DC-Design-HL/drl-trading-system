"""
News item processor: normalization, asset extraction, relevance filtering.
"""
import re
from typing import Dict, List, Optional, Set

# Asset keyword mapping
ASSET_KEYWORDS: Dict[str, List[str]] = {
    "BTC": ["bitcoin", "btc", "satoshi", "lightning network", "sats"],
    "ETH": ["ethereum", "eth", "ether", "vitalik", "eip-", "gas fees", "defi", "dapp", "smart contract"],
    "SOL": ["solana", "sol", "anatoly", "saga phone"],
    "XRP": ["ripple", "xrp", "brad garlinghouse", "sec v ripple", "xrpl"],
}

# Global crypto keywords — if matched without a specific asset, maps to ALL
GLOBAL_CRYPTO_KEYWORDS: List[str] = [
    "crypto", "cryptocurrency", "blockchain", "defi", "nft", "cbdc", "stablecoin",
    "binance", "coinbase", "kraken", "ftx", "exchange", "wallet",
    "sec", "cftc", "fed rate", "fomc", "interest rate",
    "trump", "elon musk", "michael saylor", "cathie wood",
    "tariff", "sanctions", "iran", "china ban", "geopolitical",
    "recession", "inflation", "cpi", "gdp", "treasury",
    "etf", "spot etf", "institutional", "blackrock", "fidelity",
    "hack", "exploit", "stolen", "bridge attack", "rug pull",
    "ban", "regulate", "law", "legal", "court", "ruling",
]

# Domains to skip (noise/clickbait)
BLOCKED_DOMAINS: Set[str] = {
    "bitcointalk.org", "steemit.com", "publish0x.com",
    "hive.blog", "medium.com/tag/crypto",
}


def normalize_text(title: str, body: str = "") -> str:
    """Merge title + body snippet, strip HTML artifacts and URLs."""
    text = (title or "").strip()
    if body:
        text += " " + (body or "")[:280].strip()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_assets(text: str) -> List[str]:
    """Return list of asset tickers mentioned in text. 'ALL' if only global keywords match."""
    lower = text.lower()
    matched = []
    for ticker, keywords in ASSET_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            matched.append(ticker)
    if not matched:
        if any(kw in lower for kw in GLOBAL_CRYPTO_KEYWORDS):
            matched = ["ALL"]
    return matched


def is_relevant(item: Dict) -> bool:
    """Return True if the item is relevant to crypto markets."""
    title = item.get("title", "")
    body = item.get("body_snippet", "")
    if not title:
        return False

    # Block known noise domains
    url = item.get("url", "").lower()
    if any(domain in url for domain in BLOCKED_DOMAINS):
        return False

    # Check relevance via keywords
    text = normalize_text(title, body).lower()
    has_asset = any(
        any(kw in text for kw in kws) for kws in ASSET_KEYWORDS.values()
    )
    has_global = any(kw in text for kw in GLOBAL_CRYPTO_KEYWORDS)
    return has_asset or has_global


def process_item(item: Dict) -> Optional[Dict]:
    """
    Enrich a raw fetched item with normalized text, asset extraction, and hashes.
    Returns None if item fails relevance filter.
    """
    from src.news.storage import NewsStorage

    if not is_relevant(item):
        return None

    title = item.get("title", "")
    body = item.get("body_snippet", "")
    normalized = normalize_text(title, body)
    assets = extract_assets(normalized)

    return {
        **item,
        "url_hash": NewsStorage.url_hash(item.get("url", title)),
        "title_hash": NewsStorage.title_hash(title),
        "normalized_text": normalized,
        "assets": assets,
    }
