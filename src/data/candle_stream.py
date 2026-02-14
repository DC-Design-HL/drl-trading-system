"""
WebSocket Candle Stream Manager

Efficiently manages candle data:
- Fetches historical candles once at startup (1000 candles)
- Streams new candles via WebSocket
- Maintains in-memory DataFrame
- Reduces API calls from 288/day to ~1/day
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Optional, Callable, Dict
import pandas as pd
import numpy as np
import websocket
import requests
import logging

logger = logging.getLogger(__name__)


class CandleStreamManager:
    """
    Manages candle data with WebSocket streaming.
    
    Usage:
        manager = CandleStreamManager("BTCUSDT", "1h")
        manager.start()
        
        # Get current data anytime
        df = manager.get_dataframe()
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        max_candles: int = 1000,
        on_new_candle: Optional[Callable] = None,
    ):
        self.symbol = symbol.upper()
        self.interval = interval
        self.max_candles = max_candles
        self.on_new_candle = on_new_candle
        
        # Data storage
        self._df: Optional[pd.DataFrame] = None
        self._current_candle: Optional[Dict] = None
        self._lock = threading.Lock()
        
        # WebSocket
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_delay = 5
        
        logger.info(f"📊 CandleStreamManager initialized for {symbol} {interval}")
    
    def _fetch_historical(self) -> pd.DataFrame:
        """Fetch historical candles from REST API (one-time)."""
        url = os.environ.get("BINANCE_FUTURES_URL", "https://data-api.binance.vision") + "/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.max_candles
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.set_index('timestamp')
            
            logger.info(f"📊 Fetched {len(df)} historical candles (one-time)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise
    
    def _on_ws_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            if 'k' not in data:
                return
            
            kline = data['k']
            
            candle = {
                'timestamp': pd.Timestamp(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'is_closed': kline['x']
            }
            
            with self._lock:
                self._current_candle = candle
                
                # If candle is closed, add to DataFrame
                if candle['is_closed']:
                    self._add_candle(candle)
                    
                    if self.on_new_candle:
                        self.on_new_candle(self._df.copy())
                    
                    logger.info(
                        f"📊 New candle: {candle['timestamp']} | "
                        f"O:{candle['open']:.2f} H:{candle['high']:.2f} "
                        f"L:{candle['low']:.2f} C:{candle['close']:.2f}"
                    )
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _add_candle(self, candle: Dict):
        """Add a completed candle to the DataFrame."""
        new_row = pd.DataFrame([{
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }], index=[candle['timestamp']])
        
        self._df = pd.concat([self._df, new_row])
        
        # Keep only max_candles
        if len(self._df) > self.max_candles:
            self._df = self._df.iloc[-self.max_candles:]
    
    def _on_ws_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        if self._running:
            logger.info(f"Reconnecting in {self._reconnect_delay}s...")
            time.sleep(self._reconnect_delay)
            self._connect_websocket()
    
    def _on_ws_open(self, ws):
        """Handle WebSocket open."""
        logger.info(f"🔌 WebSocket connected for {self.symbol} {self.interval}")
    
    def _connect_websocket(self):
        """Connect to Binance WebSocket."""
        import ssl
        
        stream = f"{self.symbol.lower()}@kline_{self.interval}"
        url = f"wss://fstream.binance.com/ws/{stream}"
        
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
            on_open=self._on_ws_open
        )
        
        # Use default SSL context to handle certificate verification
        self._ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    
    def start(self):
        """Start the candle stream manager."""
        if self._running:
            return
        
        # Fetch historical data first
        self._df = self._fetch_historical()
        
        # Start WebSocket in background thread
        self._running = True
        self._ws_thread = threading.Thread(target=self._connect_websocket, daemon=True)
        self._ws_thread.start()
        
        logger.info("🚀 CandleStreamManager started")
    
    def stop(self):
        """Stop the candle stream manager."""
        self._running = False
        if self._ws:
            self._ws.close()
        logger.info("🛑 CandleStreamManager stopped")
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get current DataFrame with all candles.
        
        Returns:
            DataFrame with OHLCV data, includes current (incomplete) candle
        """
        with self._lock:
            if self._df is None:
                return pd.DataFrame()
            
            df = self._df.copy()
            
            # Optionally include current incomplete candle
            if self._current_candle and not self._current_candle.get('is_closed', True):
                current = self._current_candle
                current_row = pd.DataFrame([{
                    'open': current['open'],
                    'high': current['high'],
                    'low': current['low'],
                    'close': current['close'],
                    'volume': current['volume']
                }], index=[current['timestamp']])
                
                # Update last row if same timestamp, otherwise append
                if current['timestamp'] in df.index:
                    df.loc[current['timestamp']] = current_row.iloc[0]
                else:
                    df = pd.concat([df, current_row])
            
            return df
    
    def get_latest_price(self) -> float:
        """Get the latest close price."""
        with self._lock:
            if self._current_candle:
                return self._current_candle['close']
            elif self._df is not None and len(self._df) > 0:
                return self._df['close'].iloc[-1]
            return 0.0
    
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._running and self._ws_thread is not None and self._ws_thread.is_alive()


# Convenience function for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def on_candle(df):
        print(f"New candle! Total: {len(df)}")
    
    manager = CandleStreamManager("BTCUSDT", "1m", on_new_candle=on_candle)
    manager.start()
    
    try:
        while True:
            time.sleep(10)
            df = manager.get_dataframe()
            print(f"Current price: ${manager.get_latest_price():.2f}, Candles: {len(df)}")
    except KeyboardInterrupt:
        manager.stop()
