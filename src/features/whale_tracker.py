"""
Free Whale Tracking Engine

Tracks whale activity using free APIs:
1. Whale Alert API - Large BTC transfers (free tier: 10 req/min)
2. Binance WebSocket - Real-time liquidations
3. Blockchain.com - Mempool analysis

Provides signals to improve trend prediction.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
from collections import deque

import requests
import numpy as np
import websocket

logger = logging.getLogger(__name__)


class WhaleAlertClient:
    """
    Fetch large BTC transfers from Whale Alert free API.
    Free tier: 10 requests per minute.
    """
    
    BASE_URL = "https://api.whale-alert.io/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Whale Alert client.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
        self.last_request_time = 0
        self.min_request_interval = 6.0  # 10 requests per minute = 6 seconds between requests
        self.cache = {}
        self.cache_ttl = 60  # Cache for 60 seconds
        
    def _rate_limit(self):
        """Respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
        
    def get_recent_transactions(
        self,
        min_value: int = 1000000,  # $1M minimum
        currency: str = "btc",
        limit: int = 100
    ) -> List[Dict]:
        """
        Get recent large transactions.
        
        Args:
            min_value: Minimum USD value of transactions
            currency: Currency to track
            limit: Maximum number of transactions
            
        Returns:
            List of transaction dictionaries
        """
        cache_key = f"{currency}_{min_value}_{limit}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        self._rate_limit()
        
        # Calculate time range (last 1 hour)
        end_time = int(time.time())
        start_time = end_time - 3600  # 1 hour ago
        
        params = {
            "start": start_time,
            "end": end_time,
            "min_value": min_value,
            "currency": currency,
            "limit": limit,
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            response = requests.get(
                f"{self.BASE_URL}/transactions",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                transactions = data.get("transactions", [])
                self.cache[cache_key] = (time.time(), transactions)
                logger.info(f"Fetched {len(transactions)} whale transactions")
                return transactions
            else:
                logger.warning(f"Whale Alert API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching whale transactions: {e}")
            return []
            
    def analyze_flow(self, transactions: List[Dict]) -> Dict:
        """
        Analyze transaction flow to determine whale sentiment.
        
        Returns:
            Dictionary with flow analysis:
            - exchange_inflow: Total USD flowing to exchanges (bearish)
            - exchange_outflow: Total USD flowing from exchanges (bullish)
            - flow_score: -1 (bearish) to +1 (bullish)
        """
        exchange_inflow = 0  # To exchange = selling pressure
        exchange_outflow = 0  # From exchange = accumulation
        unknown_flow = 0
        
        exchange_keywords = ['binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 
                            'okex', 'bybit', 'kucoin', 'ftx', 'exchange']
        
        for tx in transactions:
            amount_usd = tx.get('amount_usd', 0)
            from_owner = tx.get('from', {}).get('owner', '').lower()
            to_owner = tx.get('to', {}).get('owner', '').lower()
            from_type = tx.get('from', {}).get('owner_type', '').lower()
            to_type = tx.get('to', {}).get('owner_type', '').lower()
            
            # Check if going TO exchange (bearish)
            if to_type == 'exchange' or any(ex in to_owner for ex in exchange_keywords):
                exchange_inflow += amount_usd
            # Check if coming FROM exchange (bullish)
            elif from_type == 'exchange' or any(ex in from_owner for ex in exchange_keywords):
                exchange_outflow += amount_usd
            else:
                unknown_flow += amount_usd
                
        total_flow = exchange_inflow + exchange_outflow
        
        if total_flow > 0:
            # Positive = outflow dominant (bullish), Negative = inflow dominant (bearish)
            flow_score = (exchange_outflow - exchange_inflow) / total_flow
        else:
            flow_score = 0.0
            
        return {
            'exchange_inflow': exchange_inflow,
            'exchange_outflow': exchange_outflow,
            'unknown_flow': unknown_flow,
            'flow_score': np.clip(flow_score, -1, 1),
            'transaction_count': len(transactions),
            'total_volume': total_flow + unknown_flow
        }


class BSCScanClient:
    """
    Track large stablecoin transfers on Binance Smart Chain.
    Stablecoin movements often precede BTC price moves.
    
    NOTE: BSCScan V1 API deprecated, V2 requires paid plan.
    This client will fail gracefully if API not available.
    """
    
    # Original BSCScan API (deprecated but may still work for some users)
    BASE_URL = "https://api.bscscan.com/api"
    
    # Major stablecoin addresses on BSC
    USDT_ADDRESS = "0x55d398326f99059ff775485246999027b3197955"
    USDC_ADDRESS = "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d"
    BUSD_ADDRESS = "0xe9e7cea3dedca5984780bafc599bd69add087d56"
    
    # Known exchange hot wallets on BSC
    EXCHANGE_ADDRESSES = {
        "0x8894e0a0c962cb723c1976a4421c95949be2d4e3": "binance",
        "0xe2fc31f816a9b94326492132018c3aebc4a93ae1": "binance",
        "0x3c783c21a0383057d128bae431894a5c19f9cf06": "binance",
        "0xf977814e90da44bfa03b6295a0616a897441acec": "binance",
        "0x28c6c06298d514db089934071355e5743bf21d60": "binance",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize BSCScan client.
        
        Args:
            api_key: BSCScan/Etherscan API key (optional but recommended)
        """
        self.api_key = api_key or os.environ.get('BSCSCAN_API_KEY', '')
        self.cache = {}
        self.cache_ttl = 60  # 60 second cache
        self.last_request_time = 0
        self.min_request_interval = 0.25  # 4 requests/sec max
        
    def _rate_limit(self):
        """Respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
        
    def get_token_transfers(
        self,
        token_address: str,
        min_value: float = 1000000,  # $1M minimum
        blocks: int = 1000  # ~50 minutes of blocks
    ) -> List[Dict]:
        """
        Get recent large token transfers.
        
        Args:
            token_address: Token contract address
            min_value: Minimum USD value
            blocks: Number of blocks to look back
            
        Returns:
            List of transfer dictionaries
        """
        if not self.api_key:
            logger.warning("BSCScan API key not set, skipping stablecoin tracking")
            return []
            
        cache_key = f"{token_address}_{min_value}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
                
        self._rate_limit()
        
        try:
            params = {
                "module": "account",
                "action": "tokentx",
                "contractaddress": token_address,
                "page": 1,
                "offset": 100,
                "sort": "desc",
                "apikey": self.api_key
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1':
                    transfers = data.get('result', [])
                    
                    # Filter by minimum value (assuming 18/6 decimals)
                    large_transfers = []
                    for tx in transfers:
                        decimals = int(tx.get('tokenDecimal', 18))
                        value = float(tx.get('value', 0)) / (10 ** decimals)
                        if value >= min_value:
                            tx['value_usd'] = value  # Stablecoins = 1:1 USD
                            large_transfers.append(tx)
                            
                    self.cache[cache_key] = (time.time(), large_transfers)
                    logger.info(f"BSCScan: Found {len(large_transfers)} large transfers")
                    return large_transfers
                else:
                    logger.warning(f"BSCScan API error: {data.get('message')}")
                    
        except Exception as e:
            logger.error(f"BSCScan error: {e}")
            
        return []
        
    def analyze_stablecoin_flow(self) -> Dict:
        """
        Analyze stablecoin flows to/from exchanges.
        
        Returns:
            Dictionary with flow analysis:
            - exchange_inflow: Stablecoins moving TO exchanges (bullish - ready to buy)
            - exchange_outflow: Stablecoins leaving exchanges (bearish - already bought)
            - flow_score: -1 (bearish) to +1 (bullish)
        """
        exchange_inflow = 0  # To exchange = buying power coming in (bullish)
        exchange_outflow = 0  # From exchange = buying power leaving (neutral/bearish)
        
        # Get USDT, USDC, BUSD transfers
        for token_addr in [self.USDT_ADDRESS, self.USDC_ADDRESS]:
            transfers = self.get_token_transfers(token_addr, min_value=500000)
            
            for tx in transfers:
                value = tx.get('value_usd', 0)
                from_addr = tx.get('from', '').lower()
                to_addr = tx.get('to', '').lower()
                
                # Going TO exchange (bullish - capital ready to buy)
                if to_addr in self.EXCHANGE_ADDRESSES:
                    exchange_inflow += value
                # Coming FROM exchange (neutral)
                elif from_addr in self.EXCHANGE_ADDRESSES:
                    exchange_outflow += value
                    
        total = exchange_inflow + exchange_outflow
        
        if total > 0:
            # Positive = inflow dominant (bullish - buying power)
            flow_score = (exchange_inflow - exchange_outflow) / total
        else:
            flow_score = 0.0
            
        return {
            'stablecoin_inflow': exchange_inflow,
            'stablecoin_outflow': exchange_outflow,
            'stablecoin_flow_score': np.clip(flow_score, -1, 1),
            'total_volume': total
        }


class BinanceLiquidationTracker:
    """
    Track Binance futures liquidations using REST API.
    Uses public endpoints - no API key needed.
    """
    
    BASE_URL = os.environ.get("BINANCE_FUTURES_URL", "https://data-api.binance.vision")
    
    def __init__(self, symbol: str = "BTCUSDT", window_minutes: int = 60):
        """
        Initialize liquidation tracker.
        
        Args:
            symbol: Trading pair symbol (e.g. BTCUSDT)
            window_minutes: Time window to track liquidations
        """
        self.symbol = symbol
        self.window_minutes = window_minutes
        self.cache = {}
        self.cache_ttl = 300  # 300 second cache (up from 30s to conserve proxy bandwidth)
        
    def start(self):
        """No-op for REST API version."""
        logger.info("Binance liquidation tracker ready (REST API mode)")
        
    def stop(self):
        """No-op for REST API version."""
        pass
        
    def _get_request_config(self, endpoint: str) -> Tuple[str, Optional[Dict]]:
        """Get URL and proxy config."""
        proxy = os.environ.get("BINANCE_PROXY")
        
        if proxy:
            # Use Futures API with Proxy
            base_url = "https://fapi.binance.com"
            proxies = {
                "http": proxy, 
                "https": proxy
            }
            return f"{base_url}{endpoint}", proxies
        else:
            # Fallback to Public Data API
            base_url = "https://data-api.binance.vision" 
            return f"{base_url}{endpoint}", None

    def _get_okx_data(self) -> Dict:
        """Fetch fallback data from OKX."""
        try:
            # OKX Symbol format: BTC-USDT-SWAP
            okx_symbol = self.symbol.replace("USDT", "-USDT-SWAP")
            ccy = self.symbol.replace("USDT", "")
            
            # 1. Funding Rate
            funding_url = "https://www.okx.com/api/v5/public/funding-rate"
            f_resp = requests.get(funding_url, params={"instId": okx_symbol}, timeout=5)
            funding_rate = 0.0
            if f_resp.status_code == 200:
                data = f_resp.json().get('data', [])
                if data:
                    funding_rate = float(data[0].get('fundingRate', 0))

            # 2. Long/Short Ratio
            ls_url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
            ls_resp = requests.get(ls_url, params={"ccy": ccy, "period": "1H"}, timeout=5)
            long_ratio = 0.5
            if ls_resp.status_code == 200:
                data = ls_resp.json().get('data', [])
                if data:
                    # OKX returns [ [timestamp, ratio], ... ] or objects?
                    # OKX can return objects or lists depending on endpoint version
                    item = data[0]
                    if isinstance(item, list):
                        # [timestamp, ratio] format
                        long_ratio = float(item[1])
                    elif isinstance(item, dict):
                         # {"ratio": "...", ...} format
                        long_ratio = float(item.get('longAccount', item.get('ratio', 0.5)))

            return {
                'funding_rate': funding_rate,
                'long_ratio': long_ratio,
                'short_ratio': 1 - long_ratio,
                'open_interest': 0, # Hard to map 1:1 without more calls, keep 0 for now
                'imbalance': (0.5 - long_ratio) * 2
            }
        except Exception as e:
            logger.warning(f"OKX Fallback failed: {e}")
            return {
                'funding_rate': 0, 'long_ratio': 0.5, 'short_ratio': 0.5, 
                'open_interest': 0, 'imbalance': 0
            }

    def get_liquidation_stats(self) -> Dict:
        """
        Get liquidation statistics using funding rate and open interest as proxy.
        """
        # Check cache
        if 'stats' in self.cache:
            cached_time, cached_data = self.cache['stats']
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
                
        try:
            # Check if we should use OKX Fallback
            endpoint = "/fapi/v1/fundingRate"
            url, proxies = self._get_request_config(endpoint)
            
            # If no proxy and attempting to hit Futures on data-api -> Use OKX
            if not proxies and "data-api" in url:
                okx_data = self._get_okx_data()
                
                # Calculate sentiment based on OKX data
                funding_sentiment = -np.sign(okx_data['funding_rate']) * min(abs(okx_data['funding_rate']) * 100, 1)
                imbalance = okx_data['imbalance']
                
                stats = {
                    'long_ratio': okx_data['long_ratio'],
                    'short_ratio': okx_data['short_ratio'],
                    'funding_rate': okx_data['funding_rate'],
                    'open_interest': 0,
                    'imbalance': np.clip(imbalance + funding_sentiment * 0.3, -1, 1),
                    'long_liquidations': 0,
                    'short_liquidations': 0,
                    'total_liquidations': 0,
                    'count': 0
                }
                self.cache['stats'] = (time.time(), stats)
                return stats

            # ... Original Binance Logic below ...
            # Get funding rate (positive = longs pay shorts = crowded long)
            response = requests.get(
                url,
                params={"symbol": self.symbol, "limit": 1},
                proxies=proxies,
                timeout=10
            )
            
            funding_rate = 0.0
            if response.status_code == 200:
                data = response.json()
                if data:
                    funding_rate = float(data[0].get('fundingRate', 0))
                    
            # Get open interest
            oi_endpoint = "/fapi/v1/openInterest"
            oi_url, _ = self._get_request_config(oi_endpoint)
            
            response = requests.get(
                oi_url,
                params={"symbol": self.symbol},
                proxies=proxies,
                timeout=10
            )
            
            open_interest = 0.0
            if response.status_code == 200:
                data = response.json()
                open_interest = float(data.get('openInterest', 0))
                
            # Get long/short ratio
            ls_endpoint = "/futures/data/globalLongShortAccountRatio"
            ls_url, _ = self._get_request_config(ls_endpoint)
            
            response = requests.get(
                ls_url,
                params={"symbol": self.symbol, "period": "1h", "limit": 1},
                proxies=proxies,
                timeout=10
            )
            
            long_ratio = 0.5
            if response.status_code == 200:
                data = response.json()
                if data:
                    long_ratio = float(data[0].get('longAccount', 0.5))
                    
            # Calculate imbalance: positive = more shorts (bullish squeeze potential)
            # negative = more longs (bearish squeeze potential)
            imbalance = (0.5 - long_ratio) * 2  # Scale to -1 to 1
            
            # Funding rate sentiment: high positive = crowded longs (bearish)
            # High negative = crowded shorts (bullish)
            funding_sentiment = -np.sign(funding_rate) * min(abs(funding_rate) * 100, 1)
            
            stats = {
                'long_ratio': long_ratio,
                'short_ratio': 1 - long_ratio,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'imbalance': np.clip(imbalance + funding_sentiment * 0.3, -1, 1),
                'long_liquidations': 0,  # Not available via REST
                'short_liquidations': 0,
                'total_liquidations': 0,
                'count': 0
            }
            
            self.cache['stats'] = (time.time(), stats)
            logger.info(f"Binance sentiment: L/S ratio={long_ratio:.2%}, funding={funding_rate:.4%}")
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching Binance stats: {e}")
            return {
                'long_ratio': 0.5,
                'short_ratio': 0.5,
                'funding_rate': 0,
                'open_interest': 0,
                'imbalance': 0,
                'long_liquidations': 0,
                'short_liquidations': 0,
                'total_liquidations': 0,
                'count': 0
            }


class MempoolAnalyzer:
    """
    Analyze Bitcoin mempool for large pending transactions.
    Uses blockchain.com free API.
    """
    
    BASE_URL = "https://blockchain.info"
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 30  # 30 second cache
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Rate limit
        
    def _rate_limit(self):
        """Respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
        
    def get_mempool_stats(self) -> Dict:
        """
        Get mempool statistics.
        
        Returns:
            Dictionary with mempool analysis
        """
        # Check cache
        if 'stats' in self.cache:
            cached_time, cached_data = self.cache['stats']
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
                
        self._rate_limit()
        
        try:
            # Get unconfirmed transactions count
            response = requests.get(
                f"{self.BASE_URL}/q/unconfirmedcount",
                timeout=10
            )
            unconfirmed_count = int(response.text) if response.status_code == 200 else 0
            
            # Get mempool size
            response = requests.get(
                f"{self.BASE_URL}/charts/mempool-size?format=json&timespan=1day",
                timeout=10
            )
            
            mempool_data = {}
            if response.status_code == 200:
                data = response.json()
                values = data.get('values', [])
                if values:
                    current_size = values[-1].get('y', 0)
                    avg_size = np.mean([v.get('y', 0) for v in values[-24:]])  # Last 24 data points
                    mempool_data = {
                        'current_size': current_size,
                        'avg_size': avg_size,
                        'congestion_ratio': current_size / avg_size if avg_size > 0 else 1.0
                    }
                    
            stats = {
                'unconfirmed_count': unconfirmed_count,
                'mempool_size': mempool_data.get('current_size', 0),
                'congestion_ratio': mempool_data.get('congestion_ratio', 1.0),
                'urgency_score': self._calculate_urgency(mempool_data.get('congestion_ratio', 1.0))
            }
            
            self.cache['stats'] = (time.time(), stats)
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching mempool stats: {e}")
            return {
                'unconfirmed_count': 0,
                'mempool_size': 0,
                'congestion_ratio': 1.0,
                'urgency_score': 0.0
            }
            
    def _calculate_urgency(self, congestion_ratio: float) -> float:
        """
        Calculate urgency score based on congestion.
        High congestion often means panic selling.
        
        Returns:
            -1 (low urgency) to +1 (high urgency/panic)
        """
        # Normalize congestion ratio to -1 to 1 scale
        # > 1.5x average = high urgency, < 0.5x = low urgency
        if congestion_ratio >= 2.0:
            return 1.0
        elif congestion_ratio >= 1.5:
            return 0.5
        elif congestion_ratio <= 0.5:
            return -0.5
        else:
            return (congestion_ratio - 1.0)  # Linear between 0.5 and 1.5


class BinanceOITracker:
    """
    Track Binance Open Interest changes.
    Rising OI with rising price = bullish (smart money accumulating)
    Rising OI with falling price = bearish (smart money shorting)
    Free API - no key needed.
    """
    
    def __init__(self, symbol: str = "BTCUSDT", window_minutes: int = 60):
        """
        Initialize OI tracker.
        
        Args:
            symbol: Trading pair symbol
            window_minutes: Analysis window
        """
        self.symbol = symbol
        self.window_minutes = window_minutes
        self.cache = {}
        self.cache_ttl = 300  # 300 second cache (up from 60s to conserve proxy bandwidth)
    
    def _get_request_config(self, endpoint: str) -> Tuple[str, Optional[Dict]]:
        """Get URL and proxy config."""
        proxy = os.environ.get("BINANCE_PROXY")
        
        if proxy:
            # Use Futures API with Proxy
            base_url = "https://fapi.binance.com"
            proxies = {
                "http": proxy, 
                "https": proxy
            }
            return f"{base_url}{endpoint}", proxies
        else:
            # Fallback to Public Data API
            base_url = "https://data-api.binance.vision" 
            return f"{base_url}{endpoint}", None

    def get_oi_signal(self) -> Dict:
        """
        Get Open Interest trend signal.
        
        Returns:
            Dictionary with OI analysis
        """
        # Check cache
        if 'signal' in self.cache:
            cached_time, cached_data = self.cache['signal']
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
                
        try:
            # Get current OI
            endpoint = "/fapi/v1/openInterest"
            url, proxies = self._get_request_config(endpoint)
            
            # If no proxy and trying to hit futures endpoint on data-api, fail fast
            if not proxies and "data-api" in url:
                return {'oi': 0, 'price': 0, 'signal': 0}

            response = requests.get(
                url,
                params={"symbol": self.symbol},
                proxies=proxies,
                timeout=10
            )
            
            current_oi = 0.0
            if response.status_code == 200:
                data = response.json()
                current_oi = float(data.get('openInterest', 0))
                
            # Get current price
            # data-api supports ticker/price, so we can try it even without proxy
            price_endpoint = "/fapi/v1/ticker/price"
            price_url, _ = self._get_request_config(price_endpoint)
            
            # If data-api, fallback to spot ticker if fapi ticker fails?
            # Actually data-api DOES have /api/v3/ticker/price.
            if not proxies and "data-api" in price_url:
                 price_url = "https://data-api.binance.vision/api/v3/ticker/price"
            
            response = requests.get(
                price_url,
                params={"symbol": self.symbol},
                proxies=proxies,
                timeout=10
            )
            
            current_price = 0.0
            if response.status_code == 200:
                data = response.json()
                current_price = float(data.get('price', 0))
                
            # Store in history
            self.oi_history.append({
                'oi': current_oi,
                'price': current_price,
                'time': time.time()
            })
            
            # Calculate signal based on OI change vs price change
            signal = 0.0
            if len(self.oi_history) >= 2:
                prev = self.oi_history[-2]
                oi_change = (current_oi - prev['oi']) / prev['oi'] if prev['oi'] > 0 else 0
                price_change = (current_price - prev['price']) / prev['price'] if prev['price'] > 0 else 0
                
                # OI up + Price up = bullish (smart money long)
                # OI up + Price down = bearish (smart money short)
                # OI down = neutral (taking profits)
                if oi_change > 0.001:  # OI increased >0.1%
                    signal = 1.0 if price_change > 0 else -1.0
                elif oi_change < -0.001:  # OI decreased
                    signal = 0.0  # Neutral - profit taking
                else:
                    signal = 0.0
                    
            result = {
                'oi': current_oi,
                'price': current_price,
                'signal': np.clip(signal, -1, 1)
            }
            
            self.cache['signal'] = (time.time(), result)
            logger.info(f"📊 OI Tracker: OI={current_oi:,.0f}, signal={signal:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"OI Tracker error: {e}")
            return {'oi': 0, 'price': 0, 'signal': 0}


class BinanceTopTraderClient:
    """
    Track top trader positions on Binance Futures.
    Uses free Binance API - no key needed.
    
    Renamed from Blockchair to reflect actual source (Binance).
    """
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.cache = {}
        self.cache_ttl = 300  # 300 second cache (up from 60s to conserve proxy bandwidth)
        
    def _get_request_config(self, endpoint: str) -> Tuple[str, Optional[Dict]]:
        """Get URL and proxy config."""
        proxy = os.environ.get("BINANCE_PROXY")
        
        if proxy:
            # Use Futures API with Proxy
            base_url = "https://fapi.binance.com"
            proxies = {
                "http": proxy, 
                "https": proxy
            }
            return f"{base_url}{endpoint}", proxies
        else:
            # Fallback to Public Data API
            base_url = "https://data-api.binance.vision" 
            return f"{base_url}{endpoint}", None
        
    def get_large_transactions(self, min_btc: float = 100) -> Dict:
        """
        Get top trader positioning signal.
        Uses Binance top trader long/short ratio.
        
        Returns:
            Dictionary with top trader analysis
        """
        # Check cache
        if 'top_trader' in self.cache:
            cached_time, cached_data = self.cache['top_trader']
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
        
        try:
            # Get top trader positions
            endpoint = "/futures/data/topLongShortPositionRatio"
            url, proxies = self._get_request_config(endpoint)
            
            # If no proxy and trying to hit futures endpoint on data-api, fail fast
            if not proxies and "data-api" in url:
                 return {'signal': 0, 'long_ratio': 0.5, 'short_ratio': 0.5}

            response = requests.get(
                url,
                params={"symbol": self.symbol, "period": "1h", "limit": 1},
                proxies=proxies,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    latest = data[-1]
                    long_ratio = float(latest.get('longAccount', 0.5))
                    short_ratio = float(latest.get('shortAccount', 0.5))
                    
                    # Signal: more longs = bearish (crowded), more shorts = bullish (contrarian)
                    # Top traders being long-heavy is actually bullish (smart money)
                    # But extreme positioning is a warning sign
                    signal = 0.0
                    if long_ratio > 0.65:  # >65% long = very crowded
                        signal = -0.3  # Bearish
                    elif long_ratio > 0.55:
                        signal = (long_ratio - 0.5) * 2  # Slight bullish (following smart money)
                    elif short_ratio > 0.55:
                        signal = -((short_ratio - 0.5) * 2)  # Slight bearish
                    elif short_ratio > 0.65:
                        signal = 0.3  # Contrarian bullish
                        
                    result = {
                        'long_ratio': long_ratio,
                        'short_ratio': short_ratio,
                        'signal': np.clip(signal, -1, 1)
                    }
                    
                    self.cache['top_trader'] = (time.time(), result)
                    logger.info(f"� Top Traders: {long_ratio:.1%} long, {short_ratio:.1%} short → signal={signal:.2f}")
                    return result
                    
        except Exception as e:
            logger.error(f"Top Trader error: {e}")
            
        return {'long_ratio': 0.5, 'short_ratio': 0.5, 'signal': 0}



class FearGreedTracker:
    """
    Track Fear & Greed Index from alternative.me.
    Free API - no key needed.
    
    Interpretation:
    - Extreme Fear (<25) = Bullish contrarian signal
    - Fear (25-45) = Slight bullish
    - Neutral (45-55) = Neutral
    - Greed (55-75) = Slight bearish
    - Extreme Greed (>75) = Bearish contrarian signal
    """
    
    API_URL = "https://api.alternative.me/fng/"
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache (updates daily anyway)
        
    def get_fear_greed_signal(self) -> Dict:
        """
        Get Fear & Greed Index as a trading signal.
        
        Returns:
            Dictionary with F&G analysis
        """
        # Check cache
        if 'fng' in self.cache:
            cached_time, cached_data = self.cache['fng']
            if time.time() - cached_time < self.cache_ttl:
                return cached_data
                
        try:
            response = requests.get(
                self.API_URL,
                params={"limit": 1},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                fng_data = data.get('data', [{}])[0]
                
                value = int(fng_data.get('value', 50))
                classification = fng_data.get('value_classification', 'Neutral')
                
                # Convert to trading signal
                # Extreme fear = buy opportunity = bullish
                # Extreme greed = sell opportunity = bearish
                if value <= 25:
                    signal = 0.8  # Strong bullish
                elif value <= 40:
                    signal = 0.3  # Slight bullish
                elif value >= 75:
                    signal = -0.8  # Strong bearish
                elif value >= 60:
                    signal = -0.3  # Slight bearish
                else:
                    signal = 0.0  # Neutral
                    
                result = {
                    'value': value,
                    'classification': classification,
                    'signal': signal
                }
                
                self.cache['fng'] = (time.time(), result)
                logger.info(f"😰 Fear & Greed: {value} ({classification}) → signal={signal:.2f}")
                return result
                
        except Exception as e:
            logger.error(f"Fear & Greed error: {e}")
            
        return {'value': 50, 'classification': 'Neutral', 'signal': 0}


class WhaleTracker:
    """
    High-confidence whale tracking combining multiple data sources.
    Only trades when multiple signals agree (confluence).
    """
    
    def __init__(
        self, 
        symbol: str = "BTCUSDT",
        whale_alert_api_key: Optional[str] = None,
        bscscan_api_key: Optional[str] = None,
        min_confidence: float = 0.5,
        min_signal_agreement: int = 2
    ):
        """
        Initialize whale tracker with all data sources.
        
        Args:
            whale_alert_api_key: Optional Whale Alert API key
            bscscan_api_key: Optional BSCScan API key
            min_confidence: Minimum confidence score to allow trades (0-1)
            min_signal_agreement: Minimum number of signals that must agree
        """
        # Initialize FREE data sources (no API key needed)
        self.liquidation_tracker = BinanceLiquidationTracker(symbol=symbol)  # L/S ratio
        self.oi_tracker = BinanceOITracker(symbol=symbol)  # Open Interest
        self.blockchair = BinanceTopTraderClient(symbol=symbol)  # Top Traders (was Blockchair)
        self.fear_greed = FearGreedTracker()  # Sentiment (Global, so no symbol)
        
        # Optional paid sources (if keys provided)
        self.whale_alert = WhaleAlertClient(api_key=whale_alert_api_key)
        self.bscscan = BSCScanClient(api_key=bscscan_api_key)
        
        # Real-time WebSocket Stream (NEW)
        from src.data.whale_stream import BinanceWhaleStream

        self.whale_stream = BinanceWhaleStream(symbol=symbol)

        self.mempool_analyzer = MempoolAnalyzer()
        
        # Whale Pattern Predictor (Phase 12 — learned wallet patterns)
        self.whale_pattern_predictor = None
        try:
            from src.features.whale_pattern_predictor import WhalePatternPredictor
            self.whale_pattern_predictor = WhalePatternPredictor()
            logger.info(f"🧠 Whale pattern predictor loaded for {symbol}")
        except Exception as e:
            logger.warning(f"⚠️ Whale pattern predictor not available: {e}")
        
        self.symbol = symbol  # Store for pattern predictor lookups
        
        # High-confidence settings
        self.min_confidence = min_confidence
        self.min_signal_agreement = min_signal_agreement
        
        # Start liquidation tracker
        self.liquidation_tracker.start()
        
        # Cache for combined signals
        self.signal_cache = {}
        self.signal_cache_ttl = 30  # 30 second cache
        
        # Signal history for debugging
        self.signal_history = deque(maxlen=100)
        
        logger.info(f"🐋 WhaleTracker initialized with 4+ FREE sources (min_confidence={min_confidence}, min_agreement={min_signal_agreement})")

    def start_stream(self):
        """Start the real-time whale stream."""
        if hasattr(self, 'whale_stream'):
            self.whale_stream.start()

    def stop_stream(self):
        """Stop the real-time whale stream."""
        if hasattr(self, 'whale_stream'):
            self.whale_stream.stop()
        
    def get_whale_signals(self) -> Dict:
        """
        Get combined whale signals from all sources with confidence scoring.
        
        Returns:
            Dictionary with:
            - Individual signal scores (-1 to +1)
            - combined_score: Weighted average
            - confidence: How many signals agree (0-1)
            - signal_agreement: Count of agreeing signals
            - high_confidence: Boolean if meets threshold
        """
        # Check cache
        if 'signals' in self.signal_cache:
            cached_time, cached_data = self.signal_cache['signals']
            if time.time() - cached_time < self.signal_cache_ttl:
                return cached_data
        
        # Collect all signals from FREE sources
        signals_raw = {}
        
        if hasattr(self, 'whale_stream'):
            try:
                flow_metrics = self.whale_stream.get_metrics()
                net_flow = flow_metrics.get('net_flow', 0)
                # Normalize: >$1M net flow is strong signal
                signals_raw['flow'] = np.clip(net_flow / 1000000.0, -1.0, 1.0)
                # Store raw metrics for dashboard
                self.last_flow_metrics = flow_metrics
            except Exception as e:
                logger.error(f"Error getting flow metrics: {e}")
                signals_raw['flow'] = 0.0
        else:
            signals_raw['flow'] = 0.0
        


        # 1. Binance L/S Ratio (25% weight - most reliable)
        try:
            liq_stats = self.liquidation_tracker.get_liquidation_stats()
            signals_raw['binance_ls'] = liq_stats.get('imbalance', 0)
            logger.info(f"📊 Binance L/S: {liq_stats.get('long_ratio', 0.5):.1%}, Funding: {liq_stats.get('funding_rate', 0):.4%}")
        except Exception as e:
            logger.error(f"Error getting Binance L/S: {e}")
            signals_raw['binance_ls'] = None
            
        # 2. Binance Open Interest Changes (25% weight)
        try:
            oi_data = self.oi_tracker.get_oi_signal()
            signals_raw['oi_trend'] = oi_data.get('signal', 0)
        except Exception as e:
            logger.error(f"Error getting OI signal: {e}")
            signals_raw['oi_trend'] = None
            
        # 3. Blockchair Large Transactions (25% weight)
        try:
            blockchair_data = self.blockchair.get_large_transactions()
            signals_raw['large_txns'] = blockchair_data.get('signal', 0)
        except Exception as e:
            logger.error(f"Error getting Blockchair: {e}")
            signals_raw['large_txns'] = None
            
        # 4. Fear & Greed Index (25% weight - contrarian)
        try:
            fng_data = self.fear_greed.get_fear_greed_signal()
            signals_raw['fear_greed'] = fng_data.get('signal', 0)
        except Exception as e:
            logger.error(f"Error getting Fear & Greed: {e}")
            signals_raw['fear_greed'] = None
            
        # 5. Whale Pattern Predictor (Phase 12 — learned wallet patterns)
        if self.whale_pattern_predictor:
            try:
                wp_data = self.whale_pattern_predictor.get_signal(self.symbol)
                signals_raw['whale_patterns'] = wp_data.get('signal', 0)
                logger.info(f"🧠 Whale Pattern: signal={wp_data.get('signal', 0):.3f}, confidence={wp_data.get('confidence', 0):.3f}")
            except Exception as e:
                logger.error(f"Error getting whale patterns: {e}")
                signals_raw['whale_patterns'] = None
        else:
            signals_raw['whale_patterns'] = None
            
        # Calculate weighted score
        weights = {
            'flow': 0.25,            # Real-time whale buying/selling
            'binance_ls': 0.15,      # Smart money positioning (imbalance)
            'oi_trend': 0.10,        # Smart money interest
            'large_txns': 0.10,      # Whale activity
            'fear_greed': 0.10,      # Market sentiment
            'whale_patterns': 0.30,  # Learned whale wallet patterns
        }
        
        total_weight = 0
        weighted_sum = 0
        signal_directions = []  # Track bullish/bearish/neutral for each
        
        for signal_name, weight in weights.items():
            value = signals_raw.get(signal_name)
            if value is not None:
                weighted_sum += value * weight
                total_weight += weight
                
                # Track signal direction for agreement
                if value > 0.15:
                    signal_directions.append('bullish')
                elif value < -0.15:
                    signal_directions.append('bearish')
                else:
                    signal_directions.append('neutral')
                    
        combined_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate signal agreement
        bullish_count = signal_directions.count('bullish')
        bearish_count = signal_directions.count('bearish')
        total_signals = len(signal_directions)
        
        # Agreement = how many signals point the same direction
        max_agreement = max(bullish_count, bearish_count) if total_signals > 0 else 0
        signal_agreement = max_agreement
        
        # Confidence = agreement / total signals
        confidence = max_agreement / total_signals if total_signals > 0 else 0
        
        # High confidence only if enough signals agree
        high_confidence = (
            confidence >= self.min_confidence and 
            signal_agreement >= self.min_signal_agreement
        )
        
        # Determine recommendation
        if combined_score > 0.15 and bullish_count >= bearish_count:
            recommendation = 'bullish'
        elif combined_score < -0.15 and bearish_count >= bullish_count:
            recommendation = 'bearish'
        else:
            recommendation = 'neutral'
            
        signals = {
            # Individual signals (4 FREE sources + 1 Real-time)
            'flow': signals_raw.get('flow', 0) or 0,
            'binance_ls': signals_raw.get('binance_ls', 0) or 0,
            'oi_trend': signals_raw.get('oi_trend', 0) or 0,
            'large_txns': signals_raw.get('large_txns', 0) or 0,
            'fear_greed': signals_raw.get('fear_greed', 0) or 0,
            # Aggregated
            'combined_score': np.clip(combined_score, -1, 1),
            'confidence': confidence,
            'signal_agreement': signal_agreement,
            'total_signals': total_signals,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'high_confidence': high_confidence,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'flow_metrics': getattr(self, 'last_flow_metrics', {})  # Expose real-time flow data
        }
        
        self.signal_cache['signals'] = (time.time(), signals)
        self.signal_history.append(signals)
        
        # Detailed logging
        direction_str = f"🟢{bullish_count} 🔴{bearish_count} ⚪{total_signals - bullish_count - bearish_count}"
        confidence_str = "✅ HIGH" if high_confidence else "⚠️ LOW"
        
        logger.info(
            f"🐋 Whale signals: score={combined_score:.2f}, "
            f"signals={direction_str}, confidence={confidence:.0%} {confidence_str} → {recommendation.upper()}"
        )
        
        return signals
        
    def should_trade_long(self, threshold: float = -0.2, require_high_confidence: bool = True) -> Tuple[bool, str]:
        """
        Check if conditions are favorable for long trades.
        
        Args:
            threshold: Minimum combined score to allow long trades
            require_high_confidence: If True, also require high confidence
            
        Returns:
            Tuple of (should_trade, reason)
        """
        signals = self.get_whale_signals()
        combined = signals['combined_score']
        high_conf = signals['high_confidence']
        
        # Check confidence first
        if require_high_confidence and not high_conf:
            return True, f"Low confidence ({signals['confidence']:.0%}) - allowing trade"
            
        # High confidence + bearish = block long
        if combined < threshold and high_conf:
            return False, f"HIGH CONF bearish ({combined:.2f}) - blocking LONG"
            
        return True, f"Whale OK: score={combined:.2f}, conf={signals['confidence']:.0%}"
        
    def should_trade_short(self, threshold: float = 0.2, require_high_confidence: bool = True) -> Tuple[bool, str]:
        """
        Check if conditions are favorable for short trades.
        
        Args:
            threshold: Maximum combined score to allow short trades
            require_high_confidence: If True, also require high confidence
            
        Returns:
            Tuple of (should_trade, reason)
        """
        signals = self.get_whale_signals()
        combined = signals['combined_score']
        high_conf = signals['high_confidence']
        
        # Check confidence first
        if require_high_confidence and not high_conf:
            return True, f"Low confidence ({signals['confidence']:.0%}) - allowing trade"
            
        # High confidence + bullish = block short
        if combined > threshold and high_conf:
            return False, f"HIGH CONF bullish ({combined:.2f}) - blocking SHORT"
            
        return True, f"Whale OK: score={combined:.2f}, conf={signals['confidence']:.0%}"
    
    def get_trade_recommendation(self) -> Tuple[str, float, str]:
        """
        Get a trade recommendation based on whale signals.
        
        Returns:
            Tuple of (action, confidence, reason)
            action: 'long', 'short', or 'hold'
        """
        signals = self.get_whale_signals()
        
        if not signals['high_confidence']:
            return 'hold', signals['confidence'], "Low confidence - no clear signal"
            
        if signals['recommendation'] == 'bullish' and signals['combined_score'] > 0.3:
            return 'long', signals['confidence'], f"High confidence bullish ({signals['combined_score']:.2f})"
        elif signals['recommendation'] == 'bearish' and signals['combined_score'] < -0.3:
            return 'short', signals['confidence'], f"High confidence bearish ({signals['combined_score']:.2f})"
        else:
            return 'hold', signals['confidence'], f"Neutral ({signals['combined_score']:.2f})"
        
    def stop(self):
        """Stop all background processes."""
        self.liquidation_tracker.stop()
        logger.info("WhaleTracker stopped")


# Test the tracker
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🐋 Testing Whale Tracker...")
    tracker = WhaleTracker()
    
    # Wait a moment for liquidation WebSocket to connect
    time.sleep(2)
    
    # Get signals
    signals = tracker.get_whale_signals()
    
    print("\n📊 Whale Signals:")
    for key, value in signals.items():
        print(f"  {key}: {value}")
        
    # Test trade recommendations
    can_long, reason = tracker.should_trade_long()
    print(f"\n📈 Can go LONG: {can_long} - {reason}")
    
    can_short, reason = tracker.should_trade_short()
    print(f"📉 Can go SHORT: {can_short} - {reason}")
    
    tracker.stop()
