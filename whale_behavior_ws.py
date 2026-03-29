#!/usr/bin/env python3
"""
Whale Behavior WebSocket Collector — Real-time whale transaction tracking.

Subscribes to Alchemy WebSocket for all tracked whale wallet addresses.
When a whale makes a move, instantly classifies and appends to the live
data files so the predictor always has the freshest signal.

Zero AI tokens. Pure WebSocket + Etherscan classification.
"""

import json
import logging
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Set

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import websocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("whale_ws")

# ── Configuration ──────────────────────────────────────────────────────

ALCHEMY_API_KEY = os.environ.get("ALCHEMY_API_KEY", "")
ALCHEMY_WS_URL = f"wss://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

DATA_DIR = Path("data/whale_behavior/eth")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Known address labels for classification (from eth_collector.py)
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
}

KNOWN_DEX_ROUTERS = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 Router2
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Uniswap Universal
    "0x1111111254eeb25477b68fb85ed929f73a960582",  # 1inch
    "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",  # SushiSwap
}

KNOWN_STAKING = {
    "0x00000000219ab540356cbb839cbe05303d7705fa",  # ETH 2.0 Deposit
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",  # Lido stETH
    "0xa1290d69c65a6fe4df752f95823fae25cb99e5a7",  # RocketPool
}

# Reconnection settings
RECONNECT_DELAY = 5
MAX_RECONNECT_DELAY = 120
PING_INTERVAL = 30
PING_TIMEOUT = 10


class WhaleWebSocketCollector:
    def __init__(self):
        self.wallet_addresses: Dict[str, str] = {}  # address_lower → label
        self.wallet_labels: Dict[str, str] = {}      # label → safe_filename
        self.ws = None
        self.reconnect_delay = RECONNECT_DELAY
        self.subscription_ids: Set[str] = set()
        self.tx_count = 0
        self._load_wallets()

    def _load_wallets(self):
        """Load tracked wallets from registry."""
        try:
            from src.features.whale_wallet_registry import get_wallets_by_chain
            wallets = get_wallets_by_chain("ETH")
            for w in wallets:
                if w.active:
                    addr = w.address.lower()
                    self.wallet_addresses[addr] = w.label
                    safe = w.label.lower().replace(" ", "_").replace("/", "_")
                    self.wallet_labels[w.label] = safe
            logger.info("Loaded %d wallet addresses to track", len(self.wallet_addresses))
        except Exception as e:
            logger.error("Failed to load wallets: %s", e)
            sys.exit(1)

    def _classify_action(self, from_addr: str, to_addr: str, value_eth: float,
                          wallet_addr: str, input_data: str) -> Dict:
        """Classify a transaction into an action type."""
        from_lower = from_addr.lower()
        to_lower = to_addr.lower()
        wallet_lower = wallet_addr.lower()

        is_outgoing = from_lower == wallet_lower
        is_incoming = to_lower == wallet_lower
        direction = "out" if is_outgoing else "in"
        counterparty = to_lower if is_outgoing else from_lower

        # Determine counterparty type
        to_type = "unknown"
        if counterparty in KNOWN_EXCHANGES:
            to_type = KNOWN_EXCHANGES[counterparty]
        elif counterparty in KNOWN_DEX_ROUTERS:
            to_type = "dex"
        elif counterparty in KNOWN_STAKING:
            to_type = "staking"

        # Classify action
        if is_outgoing:
            if to_type in ("binance", "coinbase", "kraken", "robinhood"):
                action = "EXCHANGE_DEPOSIT"
            elif to_type == "dex" or counterparty in KNOWN_DEX_ROUTERS:
                action = "DEX_SWAP"
            elif to_type == "staking" or counterparty in KNOWN_STAKING:
                action = "STAKING_DEPOSIT"
            elif input_data and len(input_data) > 10:
                action = "CONTRACT_CALL"
            elif value_eth >= 10:
                action = "LARGE_TRANSFER_OUT"
            else:
                action = "LARGE_TRANSFER_OUT"
        else:
            if to_type in ("binance", "coinbase", "kraken", "robinhood"):
                action = "EXCHANGE_WITHDRAWAL"
            elif to_type == "dex" or counterparty in KNOWN_DEX_ROUTERS:
                action = "DEX_RECEIVED"
            elif to_type == "staking" or counterparty in KNOWN_STAKING:
                action = "STAKING_WITHDRAWAL"
            elif value_eth >= 10:
                action = "LARGE_TRANSFER_IN"
            else:
                action = "LARGE_TRANSFER_IN"

        return {
            "timestamp": int(time.time()),
            "action": action,
            "value_eth": value_eth,
            "to_type": to_type,
            "direction": direction,
            "tx_hash": "",  # filled by caller
            "gas_used": 21000,  # default, updated if available
            "block": 0,  # filled by caller
        }

    def _append_action(self, label: str, action: Dict):
        """Append a classified action to the wallet's live data file."""
        safe = self.wallet_labels.get(label, label.lower().replace(" ", "_"))
        filepath = DATA_DIR / f"{safe}.jsonl"
        with open(filepath, "a") as f:
            f.write(json.dumps(action) + "\n")

    def _process_transaction(self, tx: Dict):
        """Process a raw transaction from the WebSocket."""
        from_addr = tx.get("from", "").lower()
        to_addr = tx.get("to", "").lower()
        tx_hash = tx.get("hash", "")
        block_hex = tx.get("blockNumber", "0x0")
        value_hex = tx.get("value", "0x0")
        gas_hex = tx.get("gas", "0x5208")
        input_data = tx.get("input", "0x")

        try:
            value_wei = int(value_hex, 16)
            value_eth = value_wei / 1e18
            block = int(block_hex, 16) if block_hex else 0
            gas = int(gas_hex, 16) if gas_hex else 21000
        except (ValueError, TypeError):
            return

        # Check if from or to is a tracked wallet
        wallet_addr = None
        label = None

        if from_addr in self.wallet_addresses:
            wallet_addr = from_addr
            label = self.wallet_addresses[from_addr]
        elif to_addr in self.wallet_addresses:
            wallet_addr = to_addr
            label = self.wallet_addresses[to_addr]
        else:
            return  # Not our wallet

        # Classify and store
        action = self._classify_action(from_addr, to_addr, value_eth,
                                         wallet_addr, input_data)
        action["tx_hash"] = tx_hash
        action["block"] = block
        action["gas_used"] = gas

        self._append_action(label, action)
        self.tx_count += 1

        # Log all whale transactions (most are token transfers with 0 ETH)
        if value_eth >= 1.0:
            logger.info("🐋 %s: %s %.2f ETH (to_type=%s) tx=%s",
                        label, action["action"], value_eth,
                        action["to_type"], tx_hash[:16])
        elif self.tx_count % 10 == 0:
            # Log every 10th small tx to avoid spam
            logger.info("📊 %s: %s (to_type=%s) [%d total txs]",
                        label, action["action"], action["to_type"], self.tx_count)

    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Subscription confirmation
            if "id" in data and "result" in data:
                sub_id = data["result"]
                self.subscription_ids.add(sub_id)
                logger.info("Subscription confirmed: %s (id=%s)", sub_id, data["id"])
                return

            # Transaction notification (alchemy_minedTransactions format)
            if "params" in data:
                result = data["params"].get("result", {})
                # Alchemy wraps the tx in result.transaction
                tx = result.get("transaction", result)
                if isinstance(tx, dict) and "hash" in tx:
                    self._process_transaction(tx)

        except Exception as e:
            logger.error("Message processing error: %s", e)

    def _on_error(self, ws, error):
        logger.error("WebSocket error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning("WebSocket closed (code=%s, msg=%s). Reconnecting in %ds...",
                       close_status_code, close_msg, self.reconnect_delay)

    def _on_open(self, ws):
        """Subscribe to pending transactions for all tracked wallets."""
        logger.info("WebSocket connected to Alchemy")
        self.reconnect_delay = RECONNECT_DELAY  # Reset delay on success

        addresses = list(self.wallet_addresses.keys())

        # Subscribe to alchemy_pendingTransactions filtered by our addresses
        # We need both from and to filters, so subscribe twice
        subscribe_from = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_subscribe",
            "params": [
                "alchemy_minedTransactions",
                {
                    "addresses": [
                        {"from": addr} for addr in addresses
                    ],
                    "includeRemoved": False,
                    "hashesOnly": False,
                }
            ]
        }

        subscribe_to = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "eth_subscribe",
            "params": [
                "alchemy_minedTransactions",
                {
                    "addresses": [
                        {"to": addr} for addr in addresses
                    ],
                    "includeRemoved": False,
                    "hashesOnly": False,
                }
            ]
        }

        ws.send(json.dumps(subscribe_from))
        ws.send(json.dumps(subscribe_to))
        logger.info("Subscribed to mined transactions for %d wallets (from + to)", len(addresses))

    def _on_ping(self, ws, data):
        pass

    def run(self):
        """Main loop with auto-reconnect."""
        if not ALCHEMY_API_KEY:
            logger.error("ALCHEMY_API_KEY not set")
            sys.exit(1)

        logger.info("🐋 Whale Behavior WebSocket Collector starting")
        logger.info("   Tracking %d wallets", len(self.wallet_addresses))
        logger.info("   Data dir: %s", DATA_DIR)

        while True:
            try:
                self.ws = websocket.WebSocketApp(
                    ALCHEMY_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_ping=self._on_ping,
                )
                self.ws.run_forever(
                    ping_interval=PING_INTERVAL,
                    ping_timeout=PING_TIMEOUT,
                )
            except Exception as e:
                logger.error("WebSocket run error: %s", e)

            # Reconnect with exponential backoff
            logger.info("Reconnecting in %ds... (total txs processed: %d)",
                        self.reconnect_delay, self.tx_count)
            time.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 2, MAX_RECONNECT_DELAY)


def main():
    collector = WhaleWebSocketCollector()
    collector.run()


if __name__ == "__main__":
    main()
