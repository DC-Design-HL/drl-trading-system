"""
HTTP Proxy for Binance API
Works around DNS/network restrictions by using requests library with custom adapters
"""

import os
import requests
import hashlib
import hmac
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BinanceHTTPProxy:
    """
    Direct HTTP proxy for Binance API that bypasses ccxt's networking.
    Uses Python requests library which may have better compatibility with
    containerized environments like HuggingFace Spaces.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://testnet.binance.vision"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': api_key,
            'Content-Type': 'application/json',
        })

        # Check if using Cloudflare proxy
        proxy_url = os.getenv('BINANCE_TESTNET_PROXY_URL', '').strip()
        if proxy_url:
            self.base_url = proxy_url.rstrip('/')
            logger.info(f"🌐 Using HTTP proxy: {self.base_url}")
        else:
            logger.info(f"🌐 Using direct connection: {self.base_url}")

    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def get(self, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """
        Make GET request to Binance API

        Args:
            endpoint: API endpoint (e.g., '/api/v3/time')
            params: Query parameters
            signed: Whether to sign the request

        Returns:
            JSON response as dict
        """
        params = params or {}

        if signed:
            # Add timestamp and recvWindow
            params['timestamp'] = int(datetime.now().timestamp() * 1000)
            params['recvWindow'] = 60000  # 60 seconds

            # Generate signature
            signature = self._sign_request(params)
            params['signature'] = signature

        url = f"{self.base_url}{endpoint}"

        try:
            logger.debug(f"GET {url} with params: {params}")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            raise

    def post(self, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """
        Make POST request to Binance API

        Args:
            endpoint: API endpoint
            params: Query parameters
            signed: Whether to sign the request

        Returns:
            JSON response as dict
        """
        params = params or {}

        if signed:
            # Add timestamp and recvWindow
            params['timestamp'] = int(datetime.now().timestamp() * 1000)
            params['recvWindow'] = 60000  # 60 seconds

            # Generate signature
            signature = self._sign_request(params)
            params['signature'] = signature

        url = f"{self.base_url}{endpoint}"

        try:
            logger.debug(f"POST {url} with params: {params}")
            response = self.session.post(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            raise

    def test_connectivity(self) -> bool:
        """Test API connectivity by fetching server time"""
        try:
            result = self.get('/api/v3/time')
            server_time = result.get('serverTime')
            logger.info(f"✅ Server time: {server_time}")
            return True
        except Exception as e:
            logger.error(f"❌ Connectivity test failed: {e}")
            return False

    def get_account(self) -> Dict:
        """Get account information (authenticated)"""
        return self.get('/api/v3/account', signed=True)

    def get_balance(self, asset: str = 'USDT') -> float:
        """Get balance for specific asset"""
        try:
            account = self.get_account()
            balances = account.get('balances', [])

            for balance in balances:
                if balance.get('asset') == asset:
                    return float(balance.get('free', 0.0))
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def get_all_balances(self) -> Dict[str, Dict[str, float]]:
        """Get all non-zero balances"""
        try:
            account = self.get_account()
            balances = account.get('balances', [])

            non_zero = {}
            for balance in balances:
                asset = balance.get('asset')
                free = float(balance.get('free', 0))
                locked = float(balance.get('locked', 0))
                total = free + locked

                if total > 0:
                    non_zero[asset] = {
                        'free': free,
                        'used': locked,
                        'total': total
                    }

            return non_zero
        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            raise

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24hr ticker price change statistics

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Ticker data
        """
        # Convert symbol format: 'BTC/USDT' -> 'BTCUSDT'
        binance_symbol = symbol.replace('/', '')

        try:
            result = self.get('/api/v3/ticker/24hr', params={'symbol': binance_symbol})
            return {
                'symbol': symbol,
                'last': float(result.get('lastPrice', 0)),
                'bid': float(result.get('bidPrice', 0)),
                'ask': float(result.get('askPrice', 0)),
                'high': float(result.get('highPrice', 0)),
                'low': float(result.get('lowPrice', 0)),
                'volume': float(result.get('volume', 0)),
            }
        except Exception as e:
            logger.error(f"Failed to get ticker: {e}")
            raise

    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """
        Place market order

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Amount in base currency

        Returns:
            Order response or None if failed
        """
        # Convert symbol format: 'BTC/USDT' -> 'BTCUSDT'
        binance_symbol = symbol.replace('/', '')

        params = {
            'symbol': binance_symbol,
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': amount,
        }

        try:
            result = self.post('/api/v3/order', params=params, signed=True)
            logger.info(f"✅ Order placed: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """
        Get open orders

        Args:
            symbol: Optional trading pair filter

        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            # Convert symbol format: 'BTC/USDT' -> 'BTCUSDT'
            params['symbol'] = symbol.replace('/', '')

        try:
            return self.get('/api/v3/openOrders', params=params, signed=True)
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
