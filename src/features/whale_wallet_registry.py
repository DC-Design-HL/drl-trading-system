"""
Whale Wallet Registry

Curated list of known whale/smart-money wallet addresses per chain.
Includes exchange hot/cold wallets and known institutional wallets.

Sources:
- Etherscan labeled accounts
- XRPL rich list (excluding Ripple escrow)
- Solana explorer known wallets
- Arkham Intelligence labels
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class WhaleWallet:
    """Represents a tracked whale wallet."""
    address: str
    label: str
    chain: str          # "ETH", "SOL", "XRP"
    wallet_type: str    # "exchange", "accumulator", "institution", "fund"
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
    # --- NEW wallets ---
    WhaleWallet(
        address="0xA090e606E30bD747d4E6245a1517EbE430F0057e",
        label="Coinbase Institutional",
        chain="ETH",
        wallet_type="exchange",
        notes="Coinbase institutional/prime custody wallet"
    ),
    WhaleWallet(
        address="0x40B38765696e3d5d8d9d834D8AaD4bB6e418E489",
        label="Robinhood",
        chain="ETH",
        wallet_type="exchange",
        notes="Robinhood ETH custody — retail flow proxy"
    ),
    WhaleWallet(
        address="0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8",
        label="Binance Cold 2",
        chain="ETH",
        wallet_type="exchange",
        notes="Binance second cold wallet — large reserve movements"
    ),
]


# ─────────────────────────────────────────────
# Solana Wallets (tracked via Helius/RPC)
# ─────────────────────────────────────────────
SOL_WHALES: List[WhaleWallet] = [
    WhaleWallet(
        address="2AQdpHJ2JpcEgPiATUXjQxA8QmafFegfQwSLWSprPicm",
        label="Coinbase Commerce",
        chain="SOL",
        wallet_type="exchange",
        notes="Coinbase Commerce SOL wallet — active on public RPC"
    ),
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
    # --- NEW wallets ---
    WhaleWallet(
        address="GJRs4FwHtemZ5ZE9x3FNvJ8TMwitKTh21yxdRPqn7yVQ",
        label="Kraken SOL",
        chain="SOL",
        wallet_type="exchange",
        notes="Kraken exchange SOL hot wallet"
    ),
    WhaleWallet(
        address="FWznbcNXWQuHTawe9RxvQ2LdCENssh12dsXAowFrgr2e",
        label="Phantom Treasury",
        chain="SOL",
        wallet_type="institution",
        notes="Phantom wallet treasury — ecosystem health signal"
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
        notes="Non-exchange large XRP holder"
    ),
    WhaleWallet(
        address="rLHzPsX6oXkzU2qL12kHCH8G8cnZv1rBJh",
        label="Ripple Distribution",
        chain="XRP",
        wallet_type="institution",
        notes="Ripple programmatic sales wallet"
    ),
    # --- NEW wallets ---
    WhaleWallet(
        address="rN7nDp64EKhECn3uMTfWKnJYKkHo2vjrPN",
        label="Upbit XRP Hot",
        chain="XRP",
        wallet_type="exchange",
        notes="Upbit exchange — largest XRP exchange holder (~6B XRP)"
    ),
    WhaleWallet(
        address="rKRDibMbAaMMoUGRyNFabRboMFWEvFciJR",
        label="Bitstamp XRP",
        chain="XRP",
        wallet_type="exchange",
        notes="Bitstamp XRP hot wallet — European exchange flow signal"
    ),
    WhaleWallet(
        address="rU2mEJSLqBRkYLVTv55rFTgQajkLTnT6mA",
        label="Bithumb XRP Hot",
        chain="XRP",
        wallet_type="exchange",
        notes="Bithumb exchange — 2nd largest XRP exchange holder (~1.7B XRP)"
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
