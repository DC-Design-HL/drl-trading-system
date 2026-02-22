"""
Whale Wallet Registry

Curated list of known whale/smart-money wallet addresses per chain.
These are non-exchange wallets that historically move markets.

Sources:
- Etherscan labeled accounts
- XRPL rich list (excluding Ripple escrow + exchanges)
- Solana explorer known wallets
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class WhaleWallet:
    """Represents a tracked whale wallet."""
    address: str
    label: str
    chain: str          # "ETH", "SOL", "XRP"
    wallet_type: str    # "accumulator", "trader", "institution", "fund"
    notes: str = ""


# ─────────────────────────────────────────────
# Ethereum Wallets (tracked via Etherscan API)
# ─────────────────────────────────────────────
ETH_WHALES: List[WhaleWallet] = [
    WhaleWallet(
        address="0x28C6c06298d514Db089934071355E5743bf21d60",
        label="Binance Hot Wallet",
        chain="ETH",
        wallet_type="exchange",
        notes="Binance main hot wallet — large outflows = accumulation by users"
    ),
    WhaleWallet(
        address="0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549",
        label="Binance Cold Wallet",
        chain="ETH",
        wallet_type="exchange",
        notes="Binance cold storage — movements signal large rebalances"
    ),
    WhaleWallet(
        address="0xF977814e90dA44bFA03b6295A0616a897441aceC",
        label="Binance Reserve",
        chain="ETH",
        wallet_type="exchange",
        notes="Binance reserve wallet"
    ),
    WhaleWallet(
        address="0x56Eddb7aa87536c09CCc2793473599fD21A8b17F",
        label="Smart Money Whale 1",
        chain="ETH",
        wallet_type="accumulator",
        notes="Known ETH accumulator — historically buys before major rallies"
    ),
    WhaleWallet(
        address="0x8103683202aa8DA10536036EDef04CDd865C225E",
        label="Galaxy Digital",
        chain="ETH",
        wallet_type="institution",
        notes="Galaxy Digital institutional wallet"
    ),
]


# ─────────────────────────────────────────────
# Solana Wallets (tracked via Solana RPC)
# ─────────────────────────────────────────────
SOL_WHALES: List[WhaleWallet] = [
    WhaleWallet(
        address="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvu6Kg",
        label="Binance SOL Hot Wallet",
        chain="SOL",
        wallet_type="exchange",
        notes="Binance main SOL hot wallet"
    ),
    WhaleWallet(
        address="9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
        label="Binance SOL Cold Wallet",
        chain="SOL",
        wallet_type="exchange",
        notes="Binance SOL cold storage"
    ),
    WhaleWallet(
        address="ASTyfSima4LLAdDgoFGkgqoKowG1LZFDr9fAQrg7iaJZ",
        label="Solana Foundation Stake",
        chain="SOL",
        wallet_type="institution",
        notes="Solana Foundation staking authority — unstaking signals selling pressure"
    ),
]


# ─────────────────────────────────────────────
# XRP Wallets (tracked via XRPL RPC)
# ─────────────────────────────────────────────
XRP_WHALES: List[WhaleWallet] = [
    WhaleWallet(
        address="rDsbeomae4FXwgQTJp9Rs64Qg9vDiTCdBv",
        label="Binance XRP Hot Wallet",
        chain="XRP",
        wallet_type="exchange",
        notes="Binance main XRP hot wallet"
    ),
    WhaleWallet(
        address="rEb8TK3gBgk5auZkwc6sHnwrGVJH8DuaLh",
        label="Binance XRP Cold Wallet",
        chain="XRP",
        wallet_type="exchange",
        notes="Binance XRP cold wallet — large movements = rebalance"
    ),
    WhaleWallet(
        address="rLNaPoKeeBjZe2qs6x52yVPKpg8oT9Gkgb",
        label="XRP Whale Accumulator",
        chain="XRP",
        wallet_type="accumulator",
        notes="Non-exchange large XRP holder — movements often precede price action"
    ),
    WhaleWallet(
        address="rLHzPsX6oXkzU2qL12kHCH8G8cnZv1rBJh",
        label="Ripple Distribution",
        chain="XRP",
        wallet_type="institution",
        notes="Ripple programmatic sales wallet"
    ),
]


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def get_wallets_by_chain(chain: str) -> List[WhaleWallet]:
    """Get all tracked wallets for a specific chain."""
    chain = chain.upper()
    if chain == "ETH":
        return ETH_WHALES
    elif chain == "SOL":
        return SOL_WHALES
    elif chain == "XRP":
        return XRP_WHALES
    else:
        return []


def get_all_wallets() -> Dict[str, List[WhaleWallet]]:
    """Get all tracked wallets grouped by chain."""
    return {
        "ETH": ETH_WHALES,
        "SOL": SOL_WHALES,
        "XRP": XRP_WHALES,
    }


def get_wallet_addresses(chain: str) -> List[str]:
    """Get just the addresses for a chain."""
    return [w.address for w in get_wallets_by_chain(chain)]
