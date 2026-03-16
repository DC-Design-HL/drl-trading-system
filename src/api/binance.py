"""
Binance Connector
Handles connectivity to Binance Testnet via ccxt.
"""

import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import logging
import time

logger = logging.getLogger(__name__)


class BinanceConnector:
    """
    Connector for Binance Spot Testnet using ccxt.
    
    Handles:
    - OHLCV data fetching
    - Order placement and management
    - Account balance queries
    - Rate limiting
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        rate_limit: bool = True,
    ):
        """
        Initialize the Binance connector.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet (default True)
            rate_limit: Whether to enable rate limiting
        """
        self.testnet = testnet

        # Check if using Cloudflare Workers proxy to bypass geo-restrictions
        proxy_url = os.getenv('BINANCE_TESTNET_PROXY_URL', '').strip() if testnet else None

        # Initialize ccxt Binance
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': rate_limit,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'fetchCurrencies': False,  # Disable fetching currency info (uses sapi)
                'fetchMarkets': True,  # Keep market fetching (uses spot API)
            }
        }

        # If using proxy, set custom hostname
        if proxy_url:
            proxy_url = proxy_url.rstrip('/')
            # Extract hostname from proxy URL (e.g., frosty-lake-46b0.chen470.workers.dev)
            proxy_hostname = proxy_url.replace('https://', '').replace('http://', '')
            config['hostname'] = proxy_hostname
            logger.info(f"🌐 Configuring Cloudflare Workers proxy: {proxy_url}")

        # Create exchange
        self.exchange = ccxt.binance(config)

        if testnet:
            if proxy_url:
                # Use Cloudflare Workers proxy
                # ccxt expects urls['api'] to be a dict with public/private keys
                # OR we can use the simpler approach: just replace the hostname
                proxy_hostname = proxy_url.replace('https://', '').replace('http://', '').rstrip('/')

                # First, set to testnet URLs
                self.exchange.urls['api'] = self.exchange.urls['test']

                # Then replace the hostname in all testnet URLs with our proxy
                for key in self.exchange.urls['api']:
                    if isinstance(self.exchange.urls['api'][key], str):
                        # Replace testnet.binance.vision with our proxy hostname
                        self.exchange.urls['api'][key] = self.exchange.urls['api'][key].replace(
                            'testnet.binance.vision',
                            proxy_hostname
                        )

                logger.info(f"✅ Using Cloudflare Workers proxy: {proxy_url}")
                logger.info(f"   Public API: {self.exchange.urls['api'].get('public', 'N/A')}")
                logger.info(f"   Private API: {self.exchange.urls['api'].get('private', 'N/A')}")
            else:
                # No proxy - try direct connection (may be geo-restricted)
                # Auto-detect which testnet to use based on environment variable
                # Default to legacy testnet (testnet.binance.vision)
                use_legacy_testnet = os.getenv('USE_LEGACY_TESTNET', 'true').lower() == 'true'

                if use_legacy_testnet:
                    # Legacy testnet (testnet.binance.vision) - use CCXT's built-in test URLs
                    self.exchange.urls['api'] = self.exchange.urls['test']
                    logger.info("Using legacy Binance testnet (testnet.binance.vision)")
                else:
                    # New demo API (demo-api.binance.com) - use CCXT's built-in demo URLs
                    self.exchange.urls['api'] = self.exchange.urls['demo']

                    # Disable sapi endpoints (not supported on demo API)
                    # Redirect sapi to regular API to prevent errors
                    demo_spot_url = 'https://demo-api.binance.com/api/v3'
                    self.exchange.urls['api']['sapi'] = demo_spot_url
                    self.exchange.urls['api']['sapiV2'] = demo_spot_url
                    self.exchange.urls['api']['sapiV3'] = demo_spot_url
                    self.exchange.urls['api']['sapiV4'] = demo_spot_url

                    logger.info("Using new Binance Demo API (demo-api.binance.com)")

                logger.warning("⚠️ No proxy configured - direct connection may fail due to geo-restrictions")

        # Timeframe mapping
        self.timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
        }
        
    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            since: Start datetime for data
            limit: Maximum number of candles

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert symbol format: 'BTC/USDT' -> 'BTCUSDT'
            market_symbol = symbol.replace('/', '')

            # Prepare parameters
            params = {
                'symbol': market_symbol,
                'interval': timeframe,
                'limit': limit,
            }

            if since:
                params['startTime'] = int(since.timestamp() * 1000)

            # Use basic spot API klines endpoint
            ohlcv = self.exchange.public_get_klines(params)

            # Parse response
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore']
            )

            # Convert types and select relevant columns
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])

            # Select and set index
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            raise
            
    def fetch_historical_data(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with pagination.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            DataFrame with all historical data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
            
        all_data = []
        current_date = start_date
        
        # Calculate time delta per candle
        tf_seconds = self.timeframe_map.get(timeframe, 3600)
        candles_per_request = 1000
        time_per_request = timedelta(seconds=tf_seconds * candles_per_request)
        
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        
        while current_date < end_date:
            try:
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_date,
                    limit=candles_per_request,
                )
                
                if len(df) == 0:
                    break
                    
                all_data.append(df)
                
                # Move to next batch
                current_date = df.index[-1].to_pydatetime() + timedelta(seconds=tf_seconds)
                
                # Rate limiting - be nice to the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                time.sleep(5)
                continue
                
        if not all_data:
            return pd.DataFrame()
            
        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep='first')]
        result = result[(result.index >= start_date) & (result.index <= end_date)]
        result = result.sort_index()
        
        logger.info(f"Fetched {len(result)} candles")
        return result
        
    def _fetch_account_info(self) -> Dict:
        """
        Fetch account information using basic spot API.
        Uses direct API call to avoid CCXT's sapi/margin dependencies.
        """
        try:
            # Use CCXT's private_get_account method which directly calls /api/v3/account
            account = self.exchange.private_get_account()
            return account
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            # Re-raise the exception so it can be handled upstream with full error details
            raise

    def get_balance(self, currency: str = 'USDT') -> float:
        """
        Get account balance for a currency.

        Args:
            currency: Currency symbol (e.g., 'USDT', 'BTC')

        Returns:
            Available balance
        """
        try:
            # Use basic spot API account endpoint
            account = self._fetch_account_info()
            balances = account.get('balances', [])

            for balance in balances:
                if balance.get('asset') == currency:
                    return float(balance.get('free', 0.0))
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def get_all_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Get all non-zero balances.

        Returns:
            Dictionary of currency -> {free, used, total}
        """
        try:
            # Use basic spot API account endpoint
            account = self._fetch_account_info()
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
            logger.error(f"Error fetching balances: {e}")
            return {}
            
    def get_ticker(self, symbol: str = 'BTC/USDT') -> Dict[str, Any]:
        """
        Get current ticker data.

        Args:
            symbol: Trading pair

        Returns:
            Ticker data including last price, bid, ask, etc.
        """
        try:
            # Convert symbol format: 'BTC/USDT' -> 'BTCUSDT'
            market_symbol = symbol.replace('/', '')

            # Use basic spot API ticker endpoint
            ticker = self.exchange.public_get_ticker_24hr({'symbol': market_symbol})

            return {
                'symbol': symbol,
                'last': float(ticker['lastPrice']),
                'bid': float(ticker['bidPrice']),
                'ask': float(ticker['askPrice']),
                'high': float(ticker['highPrice']),
                'low': float(ticker['lowPrice']),
                'volume': float(ticker['volume']),
                'timestamp': datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return {}
            
    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> Optional[Dict]:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Amount in base currency
            
        Returns:
            Order details or None if failed
        """
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
            )
            logger.info(f"Market order placed: {side} {amount} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
            
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> Optional[Dict]:
        """
        Place a limit order.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Amount in base currency
            price: Limit price
            
        Returns:
            Order details or None if failed
        """
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
            )
            logger.info(f"Limit order placed: {side} {amount} {symbol} @ {price}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
            
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair
            
        Returns:
            True if cancelled, False otherwise
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
            
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get all open orders.

        Args:
            symbol: Optional trading pair to filter

        Returns:
            List of open orders
        """
        try:
            # Use basic spot API open orders endpoint
            params = {}
            if symbol:
                # Convert symbol format: 'BTC/USDT' -> 'BTCUSDT'
                params['symbol'] = symbol.replace('/', '')

            orders = self.exchange.private_get_openorders(params)
            return orders
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []
            
    def test_connectivity(self) -> bool:
        """Test API connectivity."""
        try:
            self.exchange.fetch_time()
            return True
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            return False


def create_connector(config: Dict) -> BinanceConnector:
    """Factory function to create connector from config."""
    import os
    
    # Get API keys from environment or config
    api_key = os.environ.get('BINANCE_TESTNET_API_KEY') or config.get('api_key')
    api_secret = os.environ.get('BINANCE_TESTNET_API_SECRET') or config.get('api_secret')
    
    return BinanceConnector(
        api_key=api_key,
        api_secret=api_secret,
        testnet=config.get('testnet', True),
    )
