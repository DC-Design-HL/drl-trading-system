"""
Binance Connector
Handles connectivity to Binance Testnet via ccxt.
"""

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
        
        # Initialize ccxt Binance
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': rate_limit,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        }
        
        if testnet:
            config['urls'] = {
                'api': {
                    'public': 'https://testnet.binance.vision/api/v3',
                    'private': 'https://testnet.binance.vision/api/v3',
                },
            }
            # Use sandbox mode for testnet
            self.exchange = ccxt.binance(config)
            self.exchange.set_sandbox_mode(True)
        else:
            self.exchange = ccxt.binance(config)
            
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
        since_ts = None
        if since:
            since_ts = int(since.timestamp() * 1000)
            
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=limit,
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
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
        
    def get_balance(self, currency: str = 'USDT') -> float:
        """
        Get account balance for a currency.
        
        Args:
            currency: Currency symbol (e.g., 'USDT', 'BTC')
            
        Returns:
            Available balance
        """
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get(currency, {}).get('free', 0.0))
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
            balance = self.exchange.fetch_balance()
            non_zero = {}
            for currency, amounts in balance.items():
                if isinstance(amounts, dict) and amounts.get('total', 0) > 0:
                    non_zero[currency] = amounts
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
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['baseVolume'],
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
            orders = self.exchange.fetch_open_orders(symbol)
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
