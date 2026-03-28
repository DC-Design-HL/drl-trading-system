"""
ETH Whale Wallet Historical Data Collector

Pulls complete transaction history for each tracked ETH whale wallet from
Etherscan, classifies actions, and stores structured timelines for model training.

Does NOT touch the existing whale tracking system.

Usage:
    collector = EthWhaleHistoryCollector()
    collector.collect_all_wallets()  # Full historical pull
    collector.collect_recent()       # Incremental update (last 24h)
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ── Known address labels for action classification ──────────────────────
# Maps addresses to their type so we can classify transaction intent
KNOWN_EXCHANGES = {
    "0x28c6c06298d514db089934071355e5743bf21d60": "binance",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "binance",
    "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8": "binance",
    "0xf977814e90da44bfa03b6295a0616a897441acec": "binance",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "binance",
    "0xa090e606e30bd747d4e6245a1517ebe430f0057e": "coinbase",
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "coinbase",
    "0x503828976d22510aad0201ac7ec88293211d23da": "coinbase",
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "kraken",
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": "kraken",
    "0x40b38765696e3d5d8d9d834d8aad4bb6e418e489": "robinhood",
    "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506": "sushiswap_router",
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "uniswap_v2_router",
    "0xe592427a0aece92de3edee1f18e0157c05861564": "uniswap_v3_router",
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_v3_router2",
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": "uniswap_universal",
    "0x1111111254eeb25477b68fb85ed929f73a960582": "1inch_router",
    "0x00000000219ab540356cbb839cbe05303d7705fa": "eth2_deposit",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "weth_contract",
}

KNOWN_STAKING = {
    "0x00000000219ab540356cbb839cbe05303d7705fa",  # ETH 2.0 Deposit
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",  # Lido stETH
    "0xa1290d69c65a6fe4df752f95823fae25cb99e5a7",  # RocketPool
}

KNOWN_DEX_ROUTERS = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 Router2
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Uniswap Universal
    "0x1111111254eeb25477b68fb85ed929f73a960582",  # 1inch
    "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",  # SushiSwap
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff",  # 0x Exchange
}

# Where to store collected data
DATA_DIR = Path("data/whale_behavior/eth")


class EthWhaleHistoryCollector:
    """
    Collects and classifies full transaction history for ETH whale wallets.

    For each wallet, produces a structured timeline:
    [
        {
            "timestamp": 1711234567,
            "action": "EXCHANGE_DEPOSIT",
            "value_eth": 150.5,
            "value_usd": 452000.0,  # approximate
            "to_type": "binance",
            "tx_hash": "0x...",
            "gas_used": 21000,
            "block": 19500000,
        },
        ...
    ]
    """

    ETHERSCAN_URL = "https://api.etherscan.io/v2/api"
    RATE_LIMIT_DELAY = 0.25  # 4 req/sec (conservative for free tier: 5/sec)
    PAGE_SIZE = 1000  # Max transactions per page

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ETHERSCAN_API_KEY", "")
        if not self.api_key:
            raise ValueError("ETHERSCAN_API_KEY required")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Etherscan API ─────────────────────────────────────────────────

    def _etherscan_get(self, params: Dict) -> Dict:
        """Make a rate-limited Etherscan API call."""
        params["apikey"] = self.api_key
        params["chainid"] = "1"
        time.sleep(self.RATE_LIMIT_DELAY)
        try:
            resp = requests.get(self.ETHERSCAN_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Etherscan API error: %s", exc)
            return {"status": "0", "result": []}

    def fetch_tx_history(
        self, address: str, start_block: int = 0, end_block: int = 99999999
    ) -> List[Dict]:
        """
        Fetch complete normal transaction history for an address.
        Handles pagination automatically.
        """
        address = address.lower()
        all_txns = []
        page = 1

        while True:
            data = self._etherscan_get({
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": str(start_block),
                "endblock": str(end_block),
                "page": str(page),
                "offset": str(self.PAGE_SIZE),
                "sort": "asc",
            })

            results = data.get("result", [])
            if not isinstance(results, list) or len(results) == 0:
                break

            all_txns.extend(results)
            logger.info(
                "  Page %d: %d txns (total: %d)",
                page, len(results), len(all_txns),
            )

            if len(results) < self.PAGE_SIZE:
                break  # Last page
            page += 1

        return all_txns

    def fetch_internal_txns(
        self, address: str, start_block: int = 0, end_block: int = 99999999
    ) -> List[Dict]:
        """Fetch internal (contract) transactions."""
        address = address.lower()
        all_txns = []
        page = 1

        while True:
            data = self._etherscan_get({
                "module": "account",
                "action": "txlistinternal",
                "address": address,
                "startblock": str(start_block),
                "endblock": str(end_block),
                "page": str(page),
                "offset": str(self.PAGE_SIZE),
                "sort": "asc",
            })

            results = data.get("result", [])
            if not isinstance(results, list) or len(results) == 0:
                break

            all_txns.extend(results)
            if len(results) < self.PAGE_SIZE:
                break
            page += 1

        return all_txns

    def fetch_erc20_transfers(
        self, address: str, start_block: int = 0, end_block: int = 99999999
    ) -> List[Dict]:
        """Fetch ERC-20 token transfers (USDT, USDC, etc.)."""
        address = address.lower()
        all_txns = []
        page = 1

        while True:
            data = self._etherscan_get({
                "module": "account",
                "action": "tokentx",
                "address": address,
                "startblock": str(start_block),
                "endblock": str(end_block),
                "page": str(page),
                "offset": str(self.PAGE_SIZE),
                "sort": "asc",
            })

            results = data.get("result", [])
            if not isinstance(results, list) or len(results) == 0:
                break

            all_txns.extend(results)
            if len(results) < self.PAGE_SIZE:
                break
            page += 1

        return all_txns

    # ── Action Classification ─────────────────────────────────────────

    def classify_action(
        self, tx: Dict, wallet_address: str, tx_type: str = "normal"
    ) -> Dict:
        """
        Classify a transaction into a behavioral action.

        Returns a structured action record.
        """
        wallet = wallet_address.lower()
        from_addr = tx.get("from", "").lower()
        to_addr = tx.get("to", "").lower()
        value_wei = int(tx.get("value", "0") or "0")
        value_eth = value_wei / 1e18
        timestamp = int(tx.get("timeStamp", "0"))
        block = int(tx.get("blockNumber", "0"))
        gas_used = int(tx.get("gasUsed", "0") or "0")
        func_name = tx.get("functionName", "") or ""
        tx_hash = tx.get("hash", "")

        is_outgoing = from_addr == wallet
        is_incoming = to_addr == wallet
        counterparty = to_addr if is_outgoing else from_addr

        # Classify action based on counterparty and direction
        action = "UNKNOWN"
        to_type = KNOWN_EXCHANGES.get(counterparty, "")

        if tx_type == "erc20":
            # ERC-20 token transfer
            token_symbol = tx.get("tokenSymbol", "").upper()
            token_decimal = int(tx.get("tokenDecimal", "18") or "18")
            token_value = int(tx.get("value", "0") or "0") / (10 ** token_decimal)

            if is_outgoing:
                if counterparty in KNOWN_EXCHANGES.values() or to_type:
                    action = "TOKEN_TO_EXCHANGE"
                elif counterparty in KNOWN_DEX_ROUTERS:
                    action = "DEX_SWAP"
                else:
                    action = "TOKEN_TRANSFER_OUT"
            else:
                if from_addr in KNOWN_EXCHANGES or KNOWN_EXCHANGES.get(from_addr):
                    action = "TOKEN_FROM_EXCHANGE"
                elif from_addr in KNOWN_DEX_ROUTERS:
                    action = "DEX_SWAP_RECEIVED"
                else:
                    action = "TOKEN_TRANSFER_IN"

            return {
                "timestamp": timestamp,
                "action": action,
                "value_eth": 0.0,
                "token_symbol": token_symbol,
                "token_value": token_value,
                "to_type": to_type or self._classify_address(counterparty),
                "tx_hash": tx_hash,
                "gas_used": gas_used,
                "block": block,
                "direction": "out" if is_outgoing else "in",
            }

        # Normal / Internal ETH transactions
        if is_outgoing:
            if to_type or counterparty in KNOWN_EXCHANGES:
                action = "EXCHANGE_DEPOSIT"
            elif counterparty in KNOWN_STAKING:
                action = "STAKING_DEPOSIT"
            elif counterparty in KNOWN_DEX_ROUTERS:
                action = "DEX_INTERACTION"
            elif "swap" in func_name.lower():
                action = "DEX_SWAP"
            elif value_eth > 0:
                action = "LARGE_TRANSFER_OUT"
            else:
                action = "CONTRACT_CALL"
        elif is_incoming:
            from_type = KNOWN_EXCHANGES.get(from_addr, "")
            if from_type or from_addr in KNOWN_EXCHANGES:
                action = "EXCHANGE_WITHDRAWAL"
                to_type = from_type
            elif from_addr in KNOWN_STAKING:
                action = "STAKING_WITHDRAWAL"
            elif from_addr in KNOWN_DEX_ROUTERS:
                action = "DEX_RECEIVED"
            elif value_eth > 0:
                action = "LARGE_TRANSFER_IN"
            else:
                action = "CONTRACT_RECEIVED"

        return {
            "timestamp": timestamp,
            "action": action,
            "value_eth": value_eth,
            "to_type": to_type or self._classify_address(counterparty),
            "tx_hash": tx_hash,
            "gas_used": gas_used,
            "block": block,
            "direction": "out" if is_outgoing else "in",
        }

    def _classify_address(self, address: str) -> str:
        """Best-effort classification of an unknown address."""
        addr = address.lower()
        if addr in KNOWN_EXCHANGES:
            return KNOWN_EXCHANGES[addr]
        if addr in KNOWN_STAKING:
            return "staking"
        if addr in KNOWN_DEX_ROUTERS:
            return "dex"
        return "unknown"

    # ── Collection Pipeline ───────────────────────────────────────────

    def collect_wallet(
        self,
        address: str,
        label: str,
        start_block: int = 0,
        force: bool = False,
    ) -> List[Dict]:
        """
        Collect and classify full history for one wallet.
        Saves to data/whale_behavior/eth/<label>.jsonl

        If the file already exists and force=False, loads from cache
        and only fetches new transactions since the last known block.
        """
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        out_file = DATA_DIR / f"{safe_label}.jsonl"

        # Load existing data for incremental update
        existing = []
        last_block = start_block
        if out_file.exists() and not force:
            with open(out_file) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        existing.append(rec)
                        if rec.get("block", 0) > last_block:
                            last_block = rec["block"]
                    except json.JSONDecodeError:
                        continue
            if existing:
                logger.info(
                    "  Loaded %d existing records for %s (last block: %d)",
                    len(existing), label, last_block,
                )
                last_block += 1  # Start from next block

        logger.info("📥 Collecting %s (%s) from block %d...", label, address[:15], last_block)

        # Fetch normal transactions
        normal_txns = self.fetch_tx_history(address, start_block=last_block)
        logger.info("  Normal txns: %d", len(normal_txns))

        # Fetch internal transactions
        internal_txns = self.fetch_internal_txns(address, start_block=last_block)
        logger.info("  Internal txns: %d", len(internal_txns))

        # Fetch ERC-20 transfers
        erc20_txns = self.fetch_erc20_transfers(address, start_block=last_block)
        logger.info("  ERC-20 txns: %d", len(erc20_txns))

        # Classify all transactions
        new_actions = []
        seen_hashes = {rec.get("tx_hash") for rec in existing}

        for tx in normal_txns:
            if tx.get("hash") in seen_hashes:
                continue
            action = self.classify_action(tx, address, "normal")
            new_actions.append(action)
            seen_hashes.add(tx.get("hash"))

        for tx in internal_txns:
            h = tx.get("hash", "") + "_internal"
            if h in seen_hashes:
                continue
            action = self.classify_action(tx, address, "internal")
            action["tx_hash"] = h
            new_actions.append(action)
            seen_hashes.add(h)

        for tx in erc20_txns:
            h = tx.get("hash", "") + "_erc20_" + tx.get("tokenSymbol", "")
            if h in seen_hashes:
                continue
            action = self.classify_action(tx, address, "erc20")
            action["tx_hash"] = h
            new_actions.append(action)
            seen_hashes.add(h)

        # Sort by timestamp
        new_actions.sort(key=lambda x: x["timestamp"])

        # Append to file
        if new_actions:
            with open(out_file, "a") as f:
                for action in new_actions:
                    f.write(json.dumps(action) + "\n")

        total = len(existing) + len(new_actions)
        logger.info(
            "✅ %s: %d new actions collected (total: %d)",
            label, len(new_actions), total,
        )

        return existing + new_actions

    def collect_all_wallets(self, force: bool = False) -> Dict[str, int]:
        """
        Collect history for all ETH whale wallets in the registry.
        Returns dict of {label: action_count}.
        """
        from src.features.whale_wallet_registry import get_wallets_by_chain

        wallets = get_wallets_by_chain("ETH")
        results = {}

        for wallet in wallets:
            if not wallet.active:
                logger.info("Skipping inactive wallet: %s", wallet.label)
                continue

            try:
                actions = self.collect_wallet(
                    wallet.address, wallet.label, force=force,
                )
                results[wallet.label] = len(actions)
            except Exception as exc:
                logger.error("Failed to collect %s: %s", wallet.label, exc)
                results[wallet.label] = -1

        logger.info("📊 Collection complete: %s", results)
        return results

    def collect_recent(self, hours: int = 24) -> Dict[str, int]:
        """
        Incremental update — collect only recent transactions.
        Uses the existing data files to determine where to start.
        """
        logger.info("🔄 Incremental collection (last %dh)...", hours)
        return self.collect_all_wallets(force=False)

    # ── Data Loading ──────────────────────────────────────────────────

    @staticmethod
    def load_wallet_timeline(label: str) -> List[Dict]:
        """Load the action timeline for a specific wallet."""
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        filepath = DATA_DIR / f"{safe_label}.jsonl"
        if not filepath.exists():
            return []
        actions = []
        with open(filepath) as f:
            for line in f:
                try:
                    actions.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return actions

    @staticmethod
    def list_collected_wallets() -> List[Tuple[str, int]]:
        """List all collected wallets and their action counts."""
        results = []
        if not DATA_DIR.exists():
            return results
        for f in sorted(DATA_DIR.glob("*.jsonl")):
            count = sum(1 for _ in open(f))
            results.append((f.stem, count))
        return results
