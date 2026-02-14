"""
Data Loader
Fetches and caches historical OHLCV data from Binance public API.
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging
import time

logger = logging.getLogger(__name__)


class BinanceHistoricalDataFetcher:
    """
    Fetches historical klines data from Binance public API.
    No authentication required.
    """
    
    BASE_URL = os.environ.get("BINANCE_API_URL", "https://data-api.binance.vision/api/v3/klines")
    
    # Mapping from our timeframe format to Binance's interval format
    TIMEFRAME_MAP = {
        '1m': '1m',
        '3m': '3m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '8h': '8h',
        '12h': '12h',
        '1d': '1d',
        '3d': '3d',
        '1w': '1w',
        '1M': '1M',
    }
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Binance historical data fetcher.
        
        Args:
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List]:
        """
        Fetch klines (candlestick) data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1h', '4h', '1d')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of klines to return (max 1000)
            
        Returns:
            List of klines data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000),
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
                    
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a date range.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT' or 'BTCUSDT')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            start_date: Start datetime
            end_date: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert symbol format (BTC/USDT -> BTCUSDT)
        binance_symbol = symbol.replace('/', '')
        
        # Get Binance interval
        interval = self.TIMEFRAME_MAP.get(timeframe, '1h')
        
        # Convert dates to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        all_klines = []
        current_start = start_ms
        
        logger.info(f"Fetching historical data for {binance_symbol} from {start_date} to {end_date}")
        
        while current_start < end_ms:
            klines = self.fetch_klines(
                symbol=binance_symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ms,
                limit=1000,
            )
            
            if not klines:
                break
                
            all_klines.extend(klines)
            
            # Update start time for next batch
            # Last kline's close time + 1ms
            current_start = klines[-1][6] + 1
            
            # Rate limiting - Binance allows 1200 requests/minute
            time.sleep(0.1)
            
            logger.debug(f"Fetched {len(klines)} klines, total: {len(all_klines)}")
            
        if not all_klines:
            logger.warning("No data fetched from Binance")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('timestamp')
        
        # Select and convert numeric columns
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        logger.info(f"Fetched {len(df)} klines from Binance")
        
        return df


class DataLoader:
    """
    Loads and caches historical OHLCV data.
    
    Supports:
    - Fetching from Binance public API (real data)
    - Loading from local cache
    - Generating synthetic data for testing (fallback)
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/historical",
        connector: Optional['BinanceConnector'] = None,
    ):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory for cached data files
            connector: Optional BinanceConnector (not used for historical data)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.connector = connector
        self.binance_fetcher = BinanceHistoricalDataFetcher()
        
    def load(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for the specified parameters.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            use_cache: Whether to use cached data
            force_download: Force download even if cache exists
            
        Returns:
            DataFrame with OHLCV data
        """
        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=365)
        end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        
        # Check cache first
        cache_file = self._get_cache_path(symbol, timeframe, start, end)
        
        if use_cache and cache_file.exists() and not force_download:
            logger.info(f"Loading from cache: {cache_file}")
            return self._load_from_cache(cache_file)
            
        # Fetch from Binance public API
        try:
            logger.info(f"Downloading from Binance: {symbol} {timeframe}")
            df = self.binance_fetcher.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start,
                end_date=end,
            )
            
            if len(df) > 0:
                # Cache the data
                self._save_to_cache(df, cache_file)
                return df
            else:
                logger.warning("No data received from Binance, falling back to synthetic data")
                return self._generate_synthetic_data(start, end, timeframe)
                
        except Exception as e:
            logger.error(f"Failed to fetch from Binance: {e}")
            logger.warning("Falling back to synthetic data")
            return self._generate_synthetic_data(start, end, timeframe)
        
    def _get_cache_path(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Path:
        """Generate cache file path."""
        symbol_clean = symbol.replace('/', '_')
        filename = f"{symbol_clean}_{timeframe}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        return self.cache_dir / filename
        
    def _load_from_cache(self, path: Path) -> pd.DataFrame:
        """Load data from cached CSV file."""
        df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
        return df
        
    def _save_to_cache(self, df: pd.DataFrame, path: Path):
        """Save data to cache."""
        df.to_csv(path)
        logger.info(f"Data cached to: {path}")
        
    def _generate_synthetic_data(
        self,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing.
        
        Uses geometric Brownian motion to simulate price movement.
        """
        import numpy as np
        
        # Parse timeframe to minutes
        tf_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
        }.get(timeframe, 60)
        
        # Generate timestamps
        n_candles = int((end - start).total_seconds() / (tf_minutes * 60))
        timestamps = pd.date_range(start=start, periods=n_candles, freq=f'{tf_minutes}min')
        
        # Generate prices using GBM
        initial_price = 40000  # Starting price (BTC-like)
        mu = 0.0001  # Drift
        sigma = 0.02  # Volatility
        
        # Generate log returns
        returns = np.random.normal(mu, sigma, n_candles)
        log_prices = np.log(initial_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)
        
        # Generate OHLC from close
        high_mult = 1 + np.abs(np.random.normal(0, 0.005, n_candles))
        low_mult = 1 - np.abs(np.random.normal(0, 0.005, n_candles))
        open_mult = 1 + np.random.normal(0, 0.002, n_candles)
        
        df = pd.DataFrame({
            'open': close_prices * open_mult,
            'high': close_prices * high_mult,
            'low': close_prices * low_mult,
            'close': close_prices,
            'volume': np.random.uniform(100, 10000, n_candles),
        }, index=timestamps)
        
        df.index.name = 'timestamp'
        
        # Ensure high is highest, low is lowest
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        logger.info(f"Generated {len(df)} synthetic candles")
        return df
        
    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared")


def download_binance_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    days: int = 365,
    output_dir: str = './data/historical',
) -> pd.DataFrame:
    """
    Convenience function to download Binance historical data.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
        days: Number of days of historical data
        output_dir: Directory to save data
        
    Returns:
        DataFrame with OHLCV data
    """
    loader = DataLoader(cache_dir=output_dir)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return loader.load(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        force_download=True,
    )
