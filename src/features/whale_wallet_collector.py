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

        # Determine start block from last known transaction
        start_block = 0
        if data["transactions"]:
            last_block = max(
                int(tx.get("block", 0)) for tx in data["transactions"]
            )
            start_block = last_block + 1

        all_new_txns = []

        for page in range(1, max_pages + 1):
            self._rate_limit()
            try:
                params = {
                    "chainid": "1",
                    "module": "account",
                    "action": "txlist",
                    "address": wallet.address,
                    "startblock": start_block,
                    "endblock": 99999999,
                    "page": page,
                    "offset": 1000,  # records per page
                    "sort": "asc",
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

                    all_new_txns.append(
                        {
                            "hash": tx_hash,
                            "timestamp": int(tx.get("timeStamp", 0)),
                            "block": int(tx.get("blockNumber", 0)),
                            "direction": direction,
                            "value": value_eth,
                            "counterparty": counterparty,
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
    """Collect SOL whale transactions via Solana RPC."""

    RPC_URL = "https://api.mainnet-beta.solana.com"
    RATE_LIMIT = 1.0  # 1 req/s to be safe on public RPC

    def __init__(self):
        super().__init__("SOL")
        self.last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def _rpc_call(self, method: str, params: list) -> Optional[Dict]:
        """Make a JSON-RPC call to Solana."""
        self._rate_limit()
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params,
            }
            resp = requests.post(self.RPC_URL, json=payload, timeout=15)
            if resp.status_code == 200:
                result = resp.json()
                if "error" in result:
                    logger.warning(f"Solana RPC error: {result['error']}")
                    return None
                return result.get("result")
        except Exception as e:
            logger.error(f"Solana RPC error: {e}")
        return None

    def collect_wallet(
        self, wallet: WhaleWallet, max_sigs: int = 50
    ) -> Dict:
        """
        Fetch recent transaction signatures for a SOL wallet.

        Uses getSignaturesForAddress (public, rate-limited).
        Only gets recent data — meant to be run incrementally.
        """
        data = self._load_wallet_data(wallet.address)
        data["label"] = wallet.label
        data["wallet_type"] = wallet.wallet_type
        existing_hashes = {
            tx["hash"] for tx in data["transactions"] if "hash" in tx
        }

        # Get recent signatures
        sigs_result = self._rpc_call(
            "getSignaturesForAddress",
            [wallet.address, {"limit": max_sigs}],
        )

        if not sigs_result:
            self._save_wallet_data(wallet.address, data)
            return data

        new_txns = []
        for sig_info in sigs_result:
            sig = sig_info.get("signature", "")
            if sig in existing_hashes:
                continue

            block_time = sig_info.get("blockTime", 0)

            # Get transaction details
            tx_result = self._rpc_call(
                "getTransaction",
                [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
            )

            if not tx_result:
                continue

            # Parse SOL transfers from the transaction
            meta = tx_result.get("meta", {})
            pre_balances = meta.get("preBalances", [])
            post_balances = meta.get("postBalances", [])

            account_keys = (
                tx_result.get("transaction", {})
                .get("message", {})
                .get("accountKeys", [])
            )

            # Find our wallet's balance change
            sol_change = 0.0
            wallet_lower = wallet.address.lower()
            for i, key_info in enumerate(account_keys):
                # accountKeys can be strings or objects
                if isinstance(key_info, dict):
                    addr = key_info.get("pubkey", "")
                else:
                    addr = str(key_info)

                if addr.lower() == wallet_lower:
                    if i < len(pre_balances) and i < len(post_balances):
                        lamport_change = post_balances[i] - pre_balances[i]
                        sol_change = lamport_change / 1e9  # lamports to SOL
                    break

            direction = "in" if sol_change > 0 else "out" if sol_change < 0 else "unknown"

            new_txns.append(
                {
                    "hash": sig,
                    "timestamp": block_time or int(time.time()),
                    "direction": direction,
                    "value": abs(sol_change),
                    "counterparty": "",  # Hard to extract reliably from SOL
                }
            )

        if new_txns:
            data["transactions"].extend(new_txns)
            data["transactions"] = self._deduplicate(data["transactions"])
            logger.info(
                f"✅ SOL: {len(new_txns)} new txns for {wallet.label}"
            )

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
