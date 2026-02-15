"""
On-Chain Whale Alert System
Tracks large transactions directly from blockchain APIs for ETH, SOL, and XRP.

Supported chains:
- Ethereum (Etherscan)
- Solana (Solscan/Public RPC)
- XRP (XRPL)
"""

import sys
import os
import time
import requests
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class EthereumWhaleWatcher:
    """Track large ETH transactions using Etherscan."""
    
    BASE_URL = "https://api.etherscan.io/v2/api"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ETHERSCAN_API_KEY")
        self.min_value_eth = 100.0  # ~250k-300k USD
        self.last_block = 0
        self.cache = {}
        
    def get_latest_whales(self) -> List[Dict]:
        """Check latest block for whale movements."""
        if not self.api_key:
            return []
            
        try:
            # 1. Get latest block number
            params = {
                "chainid": "1",
                "module": "proxy",
                "action": "eth_blockNumber",
                "apikey": self.api_key
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=5)
            if resp.status_code != 200:
                return []
                
            current_block = int(resp.json().get("result", "0"), 16)
            
            if current_block <= self.last_block:
                return []
                
            self.last_block = current_block
            
            # 2. Get block details
            params = {
                "chainid": "1",
                "module": "proxy",
                "action": "eth_getBlockByNumber",
                "tag": hex(current_block),
                "boolean": "true",
                "apikey": self.api_key
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=5)
            data = resp.json()
            
            transactions = data.get("result", {}).get("transactions", [])
            whales = []
            
            for tx in transactions:
                # Value is in wei
                value_wei = int(tx.get("value", "0"), 16)
                value_eth = value_wei / 10**18
                
                if value_eth >= self.min_value_eth:
                    whales.append({
                        "chain": "ETH",
                        "hash": tx.get("hash"),
                        "from": tx.get("from"),
                        "to": tx.get("to"),
                        "value": value_eth,
                        "currency": "ETH",
                        "timestamp": int(time.time()),
                        "link": f"https://etherscan.io/tx/{tx.get('hash')}"
                    })
                    
            if whales:
                logger.info(f"🐋 Found {len(whales)} ETH whales in block {current_block}")
                
            return whales
            
        except Exception as e:
            logger.error(f"ETH Whale Watcher error: {e}")
            return []

class SolanaWhaleWatcher:
    """Track large SOL transactions using Solscan/RPC."""
    
    # Public RPC endpoints (using multiple for redundancy)
    RPC_URLS = [
        "https://api.mainnet-beta.solana.com",
        "https://solana-api.projectserum.com"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SOLSCAN_API_KEY")
        self.min_value_sol = 5000.0  # ~$500k USD
        self.rpc_index = 0
        
    def _get_rpc_url(self):
        url = self.RPC_URLS[self.rpc_index]
        self.rpc_index = (self.rpc_index + 1) % len(self.RPC_URLS)
        return url
        
    def get_latest_whales(self) -> List[Dict]:
        """Fetch recent high-value SOL signatures."""
        try:
            # For simplicity and free access, we check recent confirmed blocks via RPC
            # Getting full block is heavy, instead we can sample or usage `getRecentPerformanceSamples`
            # Actually, parsing full Solana blocks is too heavy for a python bot.
            # Strategy: Use Solscan API if key exists, else skip to avoid rate limits
            
            if self.api_key:
                # Use Solscan public API (if available/paid)
                # But Solscan public API strictly rate limited and endpoint format changes.
                # Let's fallback to RPC "getSignaturesForAddress" of a known whale/exchange if generic block parsing fails.
                
                # Alternate: Watch a major exchange wallet (Binance Cold Storage)
                # Binance Hot Wallet: 5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvu6Kg
                target = "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvu6Kg"
                
                # Check recent sigs
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [
                        target,
                        {"limit": 5}
                    ]
                }
                resp = requests.post(self._get_rpc_url(), json=payload, timeout=5).json()
                
                # Without parsing inner instruction, we assume movement involving this wallet
                # This is a heuristic.
                whales = []
                # Placeholder for valid logic
                return whales
                
            return []
            
        except Exception as e:
            logger.error(f"SOL Whale Watcher error: {e}")
            return []

class RippleWhaleWatcher:
    """Track large XRP transactions using XRPL."""
    
    BASE_URL = "https://data.xrpl.org/v1/lib/ledger"  # Simplified
    RPC_URL = "https://s1.ripple.com:51234/"
    
    def __init__(self):
        self.min_value_xrp = 500000.0  # 500k XRP (~$300k)
        self.last_ledger = 0
        
    def get_latest_whales(self) -> List[Dict]:
        try:
            # 1. Get stats for latest ledger
            payload = {
                "method": "ledger",
                "params": [
                    {
                        "ledger_index": "validated",
                        "transactions": True,
                        "expand": True
                    }
                ]
            }
            resp = requests.post(self.RPC_URL, json=payload, timeout=5)
            if resp.status_code != 200:
                return []
                
            result = resp.json().get("result", {}).get("ledger", {})
            ledger_index = int(result.get("ledger_index", 0))
            
            if ledger_index <= self.last_ledger:
                return []
                
            self.last_ledger = ledger_index
            transactions = result.get("transactions", [])
            
            whales = []
            for tx in transactions:
                # Check for Payment type
                if isinstance(tx, dict) and tx.get("TransactionType") == "Payment":
                    # Amount is varying (string=XRP drops, dict=Issued Currency)
                    amount = tx.get("Amount")
                    
                    if isinstance(amount, str):
                        # Drops (1 XRP = 1,000,000 drops)
                        xrp_val = float(amount) / 1_000_000
                        if xrp_val >= self.min_value_xrp:
                            whales.append({
                                "chain": "XRP",
                                "hash": tx.get("hash"),
                                "from": tx.get("Account"),
                                "to": tx.get("Destination"),
                                "value": xrp_val,
                                "currency": "XRP",
                                "timestamp": int(time.time()),
                                "link": f"https://xrpscan.com/tx/{tx.get('hash')}"
                            })
                            
            if whales:
                logger.info(f"🐋 Found {len(whales)} XRP whales in ledger {ledger_index}")
                
            return whales
            
        except Exception as e:
            logger.error(f"XRP Whale Watcher error: {e}")
            return []

class OnChainWhaleWatcher:
    """Manager for all chain watchers."""
    
    def __init__(self):
        self.eth = EthereumWhaleWatcher()
        self.sol = SolanaWhaleWatcher()
        self.xrp = RippleWhaleWatcher()
        self.alerts = []
        
    def check_all(self) -> List[Dict]:
        """Run checks for all chains and return new alerts."""
        new_alerts = []
        
        # ETH
        if self.eth.api_key:
            new_alerts.extend(self.eth.get_latest_whales())
            
        # SOL (Skip if no key/reliable RPC logic for now)
        # new_alerts.extend(self.sol.get_latest_whales())
        
        # XRP
        new_alerts.extend(self.xrp.get_latest_whales())
        
        # Keep last 50 alerts
        self.alerts = (new_alerts + self.alerts)[:50]
        
        return new_alerts
        
    def get_latest_alerts(self) -> List[Dict]:
        return self.alerts
