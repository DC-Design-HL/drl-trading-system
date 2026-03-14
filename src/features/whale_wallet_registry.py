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
    active: bool = True  # Set False to disable broken wallets (0 transactions)


# ─────────────────────────────────────────────
# Bitcoin Wallets (tracked via Mempool/RPC in future)
# ─────────────────────────────────────────────
BTC_WHALES: List[WhaleWallet] = [
    WhaleWallet(
        address="bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
        label="Bitfinex Cold Storage",
        chain="BTC",
        wallet_type="exchange",
        notes="Bitfinex multi-sig cold wallet"
    ),
    WhaleWallet(
        address="34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
        label="Binance BTC Cold",
        chain="BTC",
        wallet_type="exchange",
        notes="Binance largest BTC cold wallet"
    ),
    WhaleWallet(
        address="bc1qjasf9z3h7w3jspkhtgatgpyvvzgpa2wwd2lr0eh5tx44reyn2k7sfc27a4",
        label="US Government (Silk Road)",
        chain="BTC",
        wallet_type="institution",
        notes="US Gov seized assets — large market impact on movement"
    ),
    WhaleWallet(
        address="1GQyBvJqR77hLqN78Uhw3L5x32oWuvW5w2",
        label="Mt. Gox Estate",
        chain="BTC",
        wallet_type="institution",
        notes="Mt. Gox trustee bankruptcy wallet"
    ),
    WhaleWallet(
        address="1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ",
        label="MicroStrategy",
        chain="BTC",
        wallet_type="accumulator",
        notes="MicroStrategy proxy tracking address"
    ),
    WhaleWallet(
        address="385cR5DM96n1HvBDMzLHPYcw89fZAXHTYe",
        label="Grayscale Bitcoin Trust",
        chain="BTC",
        wallet_type="institution",
        notes="GBTC primary custody wallet"
    ),
    WhaleWallet(
        address="1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",
        label="Binance BTC Hot",
        chain="BTC",
        wallet_type="exchange",
        notes="Binance massive BTC hot wallet"
    ),
    WhaleWallet(
        address="bc1ql49ydapnjafl5t2c9zptw3wk9dfu6qwk0p6jxtk",
        label="Robinhood BTC",
        chain="BTC",
        wallet_type="exchange",
        notes="Robinhood proxy tracking address"
    ),
]

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
    WhaleWallet(
        address="0x00000000219ab540356cBB839Cbe05303d7705Fa",
        label="ETH 2.0 Deposit Contract",
        chain="ETH",
        wallet_type="accumulator",
        notes="Network staking proxy — massive accumulation signal"
    ),
    WhaleWallet(
        address="0x4F4a11D88A196d11D2f1dD45138f2cdbb4dE2831",
        label="Wintermute Fast Hot",
        chain="ETH",
        wallet_type="exchange",
        notes="Wintermute market maker highly active hot wallet"
    ),
    WhaleWallet(
        address="0x1b3cb81e51011b549d78bf720b0d924ac763a7c2",
        label="Jump Trading",
        chain="ETH",
        wallet_type="institution",
        notes="Jump Crypto market making operations"
    ),
    WhaleWallet(
        address="0x39C6b3e42d6A679d7A0dA8cAC14723bd3A03070C",
        label="Justin Sun (Accumulator)",
        chain="ETH",
        wallet_type="accumulator",
        notes="Tron founder Justin Sun personal accumulation wallet"
    ),
    WhaleWallet(
        address="0x97ecb3a25d2cbae9d6174a1e941f173456d3ec62",
        label="FTX Bankruptcy Estate",
        chain="ETH",
        wallet_type="institution",
        notes="FTX liquidation tracker — huge outflow selling pressure signals"
    ),
    WhaleWallet(
        address="0xcEe284F754E854890e311e3280bbf64426eb862b",
        label="Arbitrum Bridge",
        chain="ETH",
        wallet_type="institution",
        notes="Massive lockup proxy for Arbitrum network"
    ),
    WhaleWallet(
        address="0x2910543af39abA0Cd09dBb2D50200b3E800A63D2",
        label="Kraken Deposit",
        chain="ETH",
        wallet_type="exchange",
        notes="Kraken ETH Deposit consolidation"
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
    # --- DISABLED: Data collection failures (0 transactions) ---
    # Reactivate after fixing Solana RPC/API issues
    WhaleWallet(
        address="GJRs4FwHtemZ5ZE9x3FNvJ8TMwitKTh21yxdRPqn7yVQ",
        label="Kraken SOL",
        chain="SOL",
        wallet_type="exchange",
        notes="Kraken exchange SOL hot wallet - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="FWznbcNXWQuHTawe9RxvQ2LdCENssh12dsXAowFrgr2e",
        label="Phantom Treasury",
        chain="SOL",
        wallet_type="institution",
        notes="Phantom wallet treasury - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="6b4aypBhH337qSzzkbeoHWzTLt4DjGjAwT8B4rGk8nJd",
        label="FTX Estate SOL",
        chain="SOL",
        wallet_type="institution",
        notes="FTX Bankruptcy Estate - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="CuieVDEDtLo7FsnA9MtQMRp6oKhQcw7cT9E29227QDBh",
        label="Alameda Research (SOL)",
        chain="SOL",
        wallet_type="institution",
        notes="Legacy Alameda Research - DATA COLLECTION FAILED",
        active=False
    ),
    # DUPLICATE ADDRESS REMOVED - Same as line 222 (Binance SOL Cold)
    # WhaleWallet(address="9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM", ...),
    WhaleWallet(
        address="27pVcSMMmEHTyZzUURmryS2yH8jT3uJkE7k2f7m1vU1B",
        label="Jump Crypto SOL",
        chain="SOL",
        wallet_type="institution",
        notes="Jump Crypto SOL liquidity provider - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="8EwG1y2Z9jC8H3Q7X9D5N1T4R6M3K7D8J4Q3T6N2H9",
        label="OKX Hot (SOL)",
        chain="SOL",
        wallet_type="exchange",
        notes="OKX active SOL withdrawal handler - INVALID ADDRESS FORMAT",
        active=False
    ),
    WhaleWallet(
        address="5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1",
        label="Raydium Authority",
        chain="SOL",
        wallet_type="institution",
        notes="Massive SOL DEX router - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="3vxheE5CEeG7tQpD7ZtL1mWeXF5B8M63U4hD1D1WegE",
        label="Upbit SOL",
        chain="SOL",
        wallet_type="exchange",
        notes="Upbit SOL dominant exchange - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvu6Kg",
        label="Binance SOL Hot (INACTIVE)",
        chain="SOL",
        wallet_type="exchange",
        notes="Binance main SOL hot wallet - DATA COLLECTION FAILED",
        active=False
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
    # --- ACTIVE wallets (verified with transaction data) ---
    WhaleWallet(
        address="rU2mEJSLqBRkYLVTv55rFTgQajkLTnT6mA",
        label="Bithumb XRP Hot",
        chain="XRP",
        wallet_type="exchange",
        notes="Bithumb exchange — 4,001 transactions",
        active=True
    ),
    WhaleWallet(
        address="rw2ciyaNshpHe7bCHo4bRWq6pqqynnWKQg",
        label="Kraken XRP Hot",
        chain="XRP",
        wallet_type="exchange",
        notes="Kraken exchange XRP Hot Wallet — 2,994 transactions",
        active=True
    ),
    WhaleWallet(
        address="rMdG3ju8pgyVh29ELPWaDuA74CpWW6Fxns",
        label="Bitfinex XRP Hot",
        chain="XRP",
        wallet_type="exchange",
        notes="Bitfinex XRP routing wallet — 2,958 transactions",
        active=True
    ),
    # --- DISABLED: Data collection failures (0 transactions) ---
    WhaleWallet(
        address="rN7nDp64EKhECn3uMTfWKnJYKkHo2vjrPN",
        label="Upbit XRP Hot",
        chain="XRP",
        wallet_type="exchange",
        notes="Upbit exchange largest holder - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="rKRDibMbAaMMoUGRyNFabRboMFWEvFciJR",
        label="Bitstamp XRP",
        chain="XRP",
        wallet_type="exchange",
        notes="Bitstamp XRP hot wallet - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="rNxp4h8apvRis6mJf9Sh8C6iRxfeFNhx3",
        label="KuCoin XRP Hot",
        chain="XRP",
        wallet_type="exchange",
        notes="KuCoin main XRP withdrawal - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="rTicJi7HWR7TuxMQn7tXjsjH6RqwzL1L4E",
        label="Wintermute XRP",
        chain="XRP",
        wallet_type="exchange",
        notes="Wintermute market maker - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="rKiCet8SdvWxPeB8U4b2gA2YF4hXGkKw7T",
        label="Ripple OTC",
        chain="XRP",
        wallet_type="institution",
        notes="Ripple OTC distribution - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="rJb5KsHsDnc14QvA3P9A5h5D3H4x7R8gMv",
        label="Binance XRP Cold 2",
        chain="XRP",
        wallet_type="exchange",
        notes="Binance Cold Storage backup - DATA COLLECTION FAILED",
        active=False
    ),
    WhaleWallet(
        address="rLNaPoKeeBjZe2qs6x52yVPKpg8oT9Gkgb",
        label="XRP Whale Accumulator",
        chain="XRP",
        wallet_type="accumulator",
        notes="Non-exchange large XRP holder - DATA COLLECTION FAILED",
        active=False
    ),
]


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def get_wallets_by_chain(chain: str, active_only: bool = True) -> List[WhaleWallet]:
    """Get all tracked wallets for a specific chain.

    Args:
        chain: Chain name (ETH, SOL, XRP, BTC)
        active_only: If True, only return wallets with active=True (default)
    """
    chain = chain.upper()
    if chain == "ETH":
        wallets = ETH_WHALES
    elif chain == "SOL":
        wallets = SOL_WHALES
    elif chain == "XRP":
        wallets = XRP_WHALES
    elif chain == "BTC":
        wallets = BTC_WHALES
    else:
        return []

    if active_only:
        return [w for w in wallets if w.active]
    return wallets


def get_all_wallets() -> Dict[str, List[WhaleWallet]]:
    """Get all tracked wallets grouped by chain."""
    return {
        "ETH": ETH_WHALES,
        "SOL": SOL_WHALES,
        "XRP": XRP_WHALES,
        "BTC": BTC_WHALES,
    }


def get_wallet_addresses(chain: str) -> List[str]:
    """Get just the addresses for a chain."""
    return [w.address for w in get_wallets_by_chain(chain)]

def get_address_context(address: str, chain: str) -> str:
    """Return the context of an address (e.g., 'exchange', 'institution', 'unknown')."""
    if not address:
        return "unknown"
    address = address.lower()
    wallets = get_wallets_by_chain(chain)
    for w in wallets:
        if w.address.lower() == address:
            return w.wallet_type
    return "unknown"
