"""
Whale Wallet Transaction Collector

Fetches historical and recent transactions for tracked whale wallets
across ETH, SOL, and XRP chains using free APIs.

Storage: JSON files in data/whale_wallets/{chain}/{address}.json
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.features.whale_wallet_registry import (
    WhaleWallet,
    get_wallets_by_chain,
    get_all_wallets,
    get_address_context,
)

logger = logging.getLogger(__name__)

# Base directory for whale wallet data
WHALE_DATA_DIR = Path("data/whale_wallets")


class BaseCollector:
    """Base class for chain-specific collectors."""

    def __init__(self, chain: str):
        self.chain = chain
        self.data_dir = WHALE_DATA_DIR / chain.lower()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_wallet_data(self, address: str) -> Dict:
        """Load existing wallet data from disk."""
        filepath = self.data_dir / f"{address}.json"
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "address": address,
            "chain": self.chain,
            "transactions": [],
            "last_updated": None,
        }

    def _save_wallet_data(self, address: str, data: Dict):
        """Save wallet data to disk."""
        filepath = self.data_dir / f"{address}.json"
        data["last_updated"] = datetime.utcnow().isoformat()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(
            f"💾 Saved {len(data.get('transactions', []))} txns for "
            f"{self.chain}:{address[:10]}..."
        )

    def _deduplicate(self, transactions: List[Dict]) -> List[Dict]:
        """Remove duplicate transactions by hash."""
        seen = set()
        unique = []
        for tx in transactions:
            tx_hash = tx.get("hash", "")
            if tx_hash and tx_hash not in seen:
                seen.add(tx_hash)
                unique.append(tx)
            elif not tx_hash:
                unique.append(tx)
        return sorted(unique, key=lambda x: x.get("timestamp", 0))


class EthereumCollector(BaseCollector):
    """Collect ETH whale transactions via Etherscan API."""

    BASE_URL = "https://api.etherscan.io/v2/api"
    RATE_LIMIT = 0.25  # 4 req/s (within free tier of 5/s)

    def __init__(self):
        super().__init__("ETH")
        self.api_key = os.environ.get("ETHERSCAN_API_KEY", "")
        self.last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def collect_wallet(
        self, wallet: WhaleWallet, max_pages: int = 5
    ) -> Dict:
        """
        Fetch transaction history for an ETH wallet.

        Uses Etherscan txlist API (up to 10k records per call).
        """
        if not self.api_key:
            logger.warning("No ETHERSCAN_API_KEY set, skipping ETH collection")
            return self._load_wallet_data(wallet.address)

        data = self._load_wallet_data(wallet.address)
        data["label"] = wallet.label
        data["wallet_type"] = wallet.wallet_type
        existing_hashes = {
            tx["hash"] for tx in data["transactions"] if "hash" in tx
        }

        all_new_txns = []

        for page in range(1, max_pages + 1):
            self._rate_limit()
            try:
                params = {
                    "chainid": "1",
                    "module": "account",
                    "action": "txlist",
                    "address": wallet.address,
                    "startblock": 0,
                    "endblock": 99999999,
                    "page": page,
                    "offset": 1000,  # records per page
                    "sort": "desc",  # newest first
                    "apikey": self.api_key,
                }

                resp = requests.get(self.BASE_URL, params=params, timeout=15)
                if resp.status_code != 200:
                    logger.warning(f"Etherscan HTTP {resp.status_code}")
                    break

                result = resp.json()
                if result.get("status") != "1":
                    # No more results or error
                    break

                txns = result.get("result", [])
                if not txns:
                    break

                for tx in txns:
                    tx_hash = tx.get("hash", "")
                    if tx_hash in existing_hashes:
                        continue

                    value_wei = int(tx.get("value", "0"))
                    value_eth = value_wei / 1e18

                    # Determine direction relative to our tracked wallet
                    from_addr = tx.get("from", "").lower()
                    to_addr = tx.get("to", "").lower()
                    wallet_lower = wallet.address.lower()

                    if from_addr == wallet_lower:
                        direction = "out"
                        counterparty = to_addr
                    elif to_addr == wallet_lower:
                        direction = "in"
                        counterparty = from_addr
                    else:
                        direction = "unknown"
                        counterparty = ""

                    context = get_address_context(counterparty, self.chain)

                    all_new_txns.append(
                        {
                            "hash": tx_hash,
                            "timestamp": int(tx.get("timeStamp", 0)),
                            "block": int(tx.get("blockNumber", 0)),
                            "direction": direction,
                            "value": value_eth,
                            "counterparty": counterparty,
                            "context": context,
                            "gas_used": int(tx.get("gasUsed", 0)),
                        }
                    )

                logger.info(
                    f"📥 ETH page {page}: {len(txns)} txns for "
                    f"{wallet.label} ({wallet.address[:10]}...)"
                )

                # If fewer than requested, no more pages
                if len(txns) < 1000:
                    break

            except Exception as e:
                logger.error(f"ETH collector error: {e}")
                break

        if all_new_txns:
            data["transactions"].extend(all_new_txns)
            data["transactions"] = self._deduplicate(data["transactions"])
            logger.info(
                f"✅ ETH: {len(all_new_txns)} new txns for {wallet.label}"
            )

        self._save_wallet_data(wallet.address, data)
        return data


class SolanaCollector(BaseCollector):
    """Collect SOL whale transactions via Helius enhanced API (or fallback to public RPC)."""

    RATE_LIMIT = 0.5  # 2 req/s for Helius free tier

    def __init__(self):
        super().__init__("SOL")
        self.api_key = os.environ.get("HELIUS_API_KEY", "")
        self.last_request_time = 0

        if self.api_key:
            self.base_url = f"https://api.helius.xyz/v0"
            logger.info("🔑 Helius API key found — using enhanced SOL data")
        else:
            self.base_url = None
            logger.warning("⚠️ No HELIUS_API_KEY — SOL collection will be limited")

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def collect_wallet(
        self, wallet: WhaleWallet, max_sigs: int = 50
    ) -> Dict:
        """
        Fetch transaction history for a SOL wallet using Helius.

        Helius provides parsed transactions with native SOL transfers,
        making it much faster than raw RPC (1 call = up to 100 txns).
        """
        if self.api_key:
            return self._collect_via_helius(wallet, max_sigs)
        else:
            return self._collect_via_rpc(wallet, max_sigs)

    def _collect_via_helius(
        self, wallet: WhaleWallet, max_sigs: int = 100
    ) -> Dict:
        """Fetch transactions using Helius enhanced transactions API."""
        data = self._load_wallet_data(wallet.address)
        data["label"] = wallet.label
        data["wallet_type"] = wallet.wallet_type
        existing_hashes = {
            tx["hash"] for tx in data["transactions"] if "hash" in tx
        }

        all_new_txns = []
        before_sig = ""  # pagination cursor

        # Helius /addresses/{address}/transactions returns parsed data
        pages = 0
        max_pages = max(1, max_sigs // 100)

        while pages < max_pages:
            self._rate_limit()
            pages += 1

            try:
                url = (
                    f"{self.base_url}/addresses/{wallet.address}/transactions"
                    f"?api-key={self.api_key}&limit=100"
                )
                if before_sig:
                    url += f"&before={before_sig}"

                resp = requests.get(url, timeout=15)

                if resp.status_code == 429:
                    logger.warning("Helius rate limit hit, waiting...")
                    time.sleep(5)
                    continue

                if resp.status_code != 200:
                    logger.warning(
                        f"Helius HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    break

                txns = resp.json()
                if not txns or not isinstance(txns, list):
                    break

                for tx in txns:
                    tx_sig = tx.get("signature", "")
                    if tx_sig in existing_hashes:
                        continue

                    timestamp = tx.get("timestamp", 0)
                    tx_type = tx.get("type", "UNKNOWN")

                    # Parse native SOL transfers
                    native_transfers = tx.get("nativeTransfers", [])
                    sol_change = 0.0
                    counterparty = ""

                    for nt in native_transfers:
                        from_addr = nt.get("fromUserAccount", "")
                        to_addr = nt.get("toUserAccount", "")
                        amount_lamports = nt.get("amount", 0)
                        amount_sol = amount_lamports / 1e9

                        if to_addr == wallet.address:
                            sol_change += amount_sol
                            if from_addr and from_addr != wallet.address:
                                counterparty = from_addr
                        elif from_addr == wallet.address:
                            sol_change -= amount_sol
                            if to_addr and to_addr != wallet.address:
                                counterparty = to_addr

                    # Skip transactions with no SOL movement
                    if abs(sol_change) < 0.001:
                        continue

                    direction = "in" if sol_change > 0 else "out"
                    context = get_address_context(counterparty, self.chain)

                    all_new_txns.append({
                        "hash": tx_sig,
                        "timestamp": timestamp,
                        "direction": direction,
                        "value": abs(sol_change),
                        "tx_type": tx_type,
                        "counterparty": counterparty,
                        "context": context,
                    })

                # Pagination: use last signature as cursor
                if len(txns) < 100:
                    break
                before_sig = txns[-1].get("signature", "")
                if not before_sig:
                    break

                logger.info(
                    f"📥 SOL page {pages}: {len(txns)} txns for "
                    f"{wallet.label}"
                )

            except Exception as e:
                logger.error(f"Helius collector error: {e}")
                break

        if all_new_txns:
            data["transactions"].extend(all_new_txns)
            data["transactions"] = self._deduplicate(data["transactions"])
            logger.info(
                f"✅ SOL: {len(all_new_txns)} new txns for {wallet.label}"
            )

        self._save_wallet_data(wallet.address, data)
        return data

    def _collect_via_rpc(
        self, wallet: WhaleWallet, max_sigs: int = 20
    ) -> Dict:
        """Fallback: fetch via public Solana RPC (slow, limited)."""
        data = self._load_wallet_data(wallet.address)
        data["label"] = wallet.label
        data["wallet_type"] = wallet.wallet_type

        logger.warning(
            f"SOL fallback RPC for {wallet.label} — "
            f"set HELIUS_API_KEY for better results"
        )

        # Minimal RPC collection (just signatures + timestamps)
        self._rate_limit()
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [wallet.address, {"limit": max_sigs}],
            }
            resp = requests.post(
                "https://api.mainnet-beta.solana.com",
                json=payload,
                timeout=15,
            )
            if resp.status_code == 200:
                result = resp.json().get("result", [])
                existing = {
                    tx["hash"] for tx in data["transactions"] if "hash" in tx
                }
                new_txns = []
                for sig_info in result:
                    sig = sig_info.get("signature", "")
                    if sig in existing:
                        continue
                    new_txns.append({
                        "hash": sig,
                        "timestamp": sig_info.get("blockTime", 0),
                        "direction": "unknown",
                        "value": 0,
                        "context": "unknown",
                    })
                if new_txns:
                    data["transactions"].extend(new_txns)
                    data["transactions"] = self._deduplicate(
                        data["transactions"]
                    )
        except Exception as e:
            logger.error(f"SOL RPC fallback error: {e}")

        self._save_wallet_data(wallet.address, data)
        return data


class XRPLCollector(BaseCollector):
    """Collect XRP whale transactions via XRPL RPC."""

    RPC_URL = "https://s1.ripple.com:51234/"
    RATE_LIMIT = 0.5  # 2 req/s

    def __init__(self):
        super().__init__("XRP")
        self.last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def collect_wallet(
        self, wallet: WhaleWallet, max_pages: int = 5, limit: int = 200
    ) -> Dict:
        """
        Fetch transaction history for an XRP wallet.

        Uses XRPL account_tx method (free, full history).
        """
        data = self._load_wallet_data(wallet.address)
        data["label"] = wallet.label
        data["wallet_type"] = wallet.wallet_type
        existing_hashes = {
            tx["hash"] for tx in data["transactions"] if "hash" in tx
        }

        marker = None
        all_new_txns = []

        for page in range(max_pages):
            self._rate_limit()
            try:
                params = {
                    "account": wallet.address,
                    "limit": limit,
                    "forward": False,  # newest first — we need recent data
                }

                # Use marker for pagination
                if marker:
                    params["marker"] = marker

                # If we have existing data, only get newer transactions
                if data["transactions"] and not marker:
                    # With forward=False we get newest first, so we'll
                    # just deduplicate against existing hashes
                    pass

                payload = {
                    "method": "account_tx",
                    "params": [params],
                }

                resp = requests.post(
                    self.RPC_URL, json=payload, timeout=15
                )
                if resp.status_code != 200:
                    logger.warning(f"XRPL HTTP {resp.status_code}")
                    break

                result = resp.json().get("result", {})

                if result.get("status") != "success":
                    logger.warning(
                        f"XRPL error: {result.get('error_message', 'unknown')}"
                    )
                    break

                txns = result.get("transactions", [])
                if not txns:
                    break

                for tx_wrapper in txns:
                    tx = tx_wrapper.get("tx", tx_wrapper.get("tx_json", {}))
                    meta = tx_wrapper.get("meta", {})
                    tx_hash = tx.get("hash", "")

                    if tx_hash in existing_hashes:
                        continue

                    # Only process Payment transactions
                    if tx.get("TransactionType") != "Payment":
                        continue

                    # Parse XRP amount (string = drops)
                    amount = tx.get("Amount")
                    if not isinstance(amount, str):
                        # Issued currency — skip for now
                        continue

                    xrp_value = float(amount) / 1_000_000

                    # Determine direction
                    account = tx.get("Account", "")
                    destination = tx.get("Destination", "")

                    if account == wallet.address:
                        direction = "out"
                        counterparty = destination
                    elif destination == wallet.address:
                        direction = "in"
                        counterparty = account
                    else:
                        direction = "unknown"
                        counterparty = ""

                    context = get_address_context(counterparty, self.chain)

                    # Parse timestamp (ripple epoch = 946684800)
                    ripple_time = tx.get("date", 0)
                    unix_time = ripple_time + 946684800 if ripple_time else 0

                    all_new_txns.append(
                        {
                            "hash": tx_hash,
                            "timestamp": unix_time,
                            "ledger": tx_wrapper.get(
                                "ledger_index",
                                tx.get("ledger_index", 0),
                            ),
                            "direction": direction,
                            "value": xrp_value,
                            "counterparty": counterparty,
                            "context": context,
                        }
                    )

                logger.info(
                    f"📥 XRP page {page + 1}: {len(txns)} txns for "
                    f"{wallet.label}"
                )

                # Check for pagination marker
                marker = result.get("marker")
                if not marker:
                    break

            except Exception as e:
                logger.error(f"XRP collector error: {e}")
                break

        if all_new_txns:
            data["transactions"].extend(all_new_txns)
            data["transactions"] = self._deduplicate(data["transactions"])
            logger.info(
                f"✅ XRP: {len(all_new_txns)} new txns for {wallet.label}"
            )

        self._save_wallet_data(wallet.address, data)
        return data


# ─────────────────────────────────────────────
# Unified Collector
# ─────────────────────────────────────────────

class WhaleWalletCollector:
    """
    Unified collector that fetches transactions for all
    tracked whale wallets across all chains.
    """

    def __init__(self):
        self.eth_collector = EthereumCollector()
        self.sol_collector = SolanaCollector()
        self.xrp_collector = XRPLCollector()

    def collect_chain(self, chain: str, max_pages: int = 5) -> Dict[str, Dict]:
        """Collect transactions for all wallets on a specific chain."""
        wallets = get_wallets_by_chain(chain)
        results = {}

        chain_upper = chain.upper()
        if chain_upper == "ETH":
            collector = self.eth_collector
        elif chain_upper == "SOL":
            collector = self.sol_collector
        elif chain_upper == "XRP":
            collector = self.xrp_collector
        else:
            logger.warning(f"Unknown chain: {chain}")
            return results

        for wallet in wallets:
            logger.info(
                f"🐋 Collecting {chain_upper} wallet: "
                f"{wallet.label} ({wallet.address[:12]}...)"
            )
            try:
                if chain_upper == "SOL":
                    # SOL uses max_sigs, not max_pages
                    data = collector.collect_wallet(wallet, max_sigs=max_pages * 10)
                else:
                    data = collector.collect_wallet(wallet, max_pages=max_pages)
                results[wallet.address] = data
            except Exception as e:
                logger.error(f"Failed to collect {wallet.label}: {e}")

        return results

    def collect_all(self, max_pages: int = 5) -> Dict[str, Dict[str, Dict]]:
        """Collect transactions for all wallets on all chains."""
        all_results = {}
        for chain in ["ETH", "SOL", "XRP"]:
            logger.info(f"\n{'='*50}")
            logger.info(f"🔍 Collecting {chain} whale wallets...")
            logger.info(f"{'='*50}")
            all_results[chain] = self.collect_chain(chain, max_pages=max_pages)
        return all_results

    def get_wallet_data(self, chain: str, address: str) -> Optional[Dict]:
        """Load cached wallet data from disk."""
        chain_lower = chain.lower()
        filepath = WHALE_DATA_DIR / chain_lower / f"{address}.json"
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        return None

    def get_all_cached_data(self) -> Dict[str, List[Dict]]:
        """Load all cached wallet data organized by chain."""
        result = {}
        for chain in ["ETH", "SOL", "XRP"]:
            chain_lower = chain.lower()
            chain_dir = WHALE_DATA_DIR / chain_lower
            if not chain_dir.exists():
                result[chain] = []
                continue

            wallets = []
            for filepath in chain_dir.glob("*.json"):
                try:
                    with open(filepath, "r") as f:
                        wallets.append(json.load(f))
                except (json.JSONDecodeError, IOError):
                    pass
            result[chain] = wallets

        return result
