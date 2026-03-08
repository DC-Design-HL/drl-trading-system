import json
import logging
import threading
import time
import os
import requests
from collections import deque
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)

try:
    import websocket
except ImportError:
    websocket = None
    logger.error("❌ 'websocket-client' not found. Whale Stream disabled.")

class BinanceWhaleStream:
    """
    Real-time Whale Trade Tracker using Binance WebSocket (aggTrade).
    
    Tracks:
    - Large Buy Orders (Taker Buy) > threshold
    - Large Sell Orders (Taker Sell) > threshold
    - Net Whale Flow (Buy Vol - Sell Vol)
    """
    
    BASE_URL = "wss://stream.binance.com:9443/ws"
    
    def __init__(self, symbol: str = "BTCUSDT", min_value_usd: float = 100000.0, window_seconds: int = 60):
        """
        Initialize Whale Stream.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            min_value_usd: Minimum trade value to be considered a 'whale' trade
            window_seconds: Rolling window size for flow calculation
        """
        self.symbol = symbol.lower()
        self.min_value_usd = min_value_usd
        self.window_seconds = window_seconds
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self.wst: Optional[threading.Thread] = None
        self.running = False
        
        # Data storage
        self.trades = deque()  # Store (timestamp, value, is_buyer_maker)
        self.lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            'buy_vol': 0.0,
            'sell_vol': 0.0,
            'net_flow': 0.0,
            'buy_count': 0,
            'sell_count': 0,
            'last_whale_trade': None
        }

    def start(self):
        """Start the WebSocket connection in a background thread."""
        if self.running:
            return
            
        if websocket is None:
            logger.warning("❌ Cannot start Whale Stream: websocket module missing. Whale features disabled.")
            return

        self.running = True
        
        # Calculate dynamic threshold based on recent market volume
        self._calculate_dynamic_threshold()
        
        stream_url = f"{self.BASE_URL}/{self.symbol}@aggTrade"
        
        logger.info(f"🐳 Starting Whale Stream for {self.symbol} (Threshold: ${self.min_value_usd:,.0f})")
        
        self.ws = websocket.WebSocketApp(
            stream_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()

    def _calculate_dynamic_threshold(self):
        """
        Dynamically calculate the whale threshold based on 24h volume.
        Rule: A whale trade should be ~0.2% of hourly volume (which is ~0.008% of 24h volume),
        with a hard floor of the base min_value_usd.
        """
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            response = requests.get(url, params={"symbol": self.symbol.upper()}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                quote_volume = float(data.get('quoteVolume', 0))
                
                # 0.0001 (0.01%) of 24h volume is approx 0.24% of avg hourly volume
                dynamic_threshold = quote_volume * 0.0001
                
                old_threshold = self.min_value_usd
                self.min_value_usd = max(old_threshold, dynamic_threshold)
                
                if self.min_value_usd > old_threshold:
                    logger.info(f"📈 High Volume Market: Dynamic whale threshold raised to ${self.min_value_usd:,.0f} (from ${old_threshold:,.0f})")
            else:
                logger.warning(f"Failed to fetch 24h volume for dynamic threshold. Sticking to ${self.min_value_usd:,.0f}.")
        except Exception as e:
            logger.warning(f"Error calculating dynamic threshold: {e}. Sticking to ${self.min_value_usd:,.0f}.")

    def stop(self):
        """Stop the WebSocket connection."""
        self.running = False
        if self.ws:
            self.ws.close()
        if self.wst:
            self.wst.join(timeout=2)
            
    def _on_open(self, ws):
        logger.info(f"🐳 Whale Stream Connected: {self.symbol}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("🐳 Whale Stream Closed")
        if self.running:
            logger.info("Reconnecting in 5 seconds...")
            time.sleep(5)
            self.start()

    def _on_error(self, ws, error):
        logger.error(f"Whale Stream Error: {error}")
        if "451" in str(error) or "403" in str(error):
             logger.warning("❌ WSS Geo-Blocked! Relying on alternative indicators.")

    def _on_message(self, ws, message):
        """Process incoming trade message."""
        try:
            data = json.loads(message)
            # aggTrade format:
            # {
            #   "e": "aggTrade",
            #   "p": "4235.4",  // Price
            #   "q": "2.5",     // Quantity
            #   "T": 123456785, // Trade time
            #   "m": true       // Is buyer the market maker? (True=Sell, False=Buy)
            # }
            
            price = float(data['p'])
            quantity = float(data['q'])
            value_usd = price * quantity
            timestamp = data['T'] / 1000.0
            is_buyer_maker = data['m']  # If True, buyer is maker -> Taker Sell
            
            # Filter for whale trades
            if value_usd >= self.min_value_usd:
                side = "SELL" if is_buyer_maker else "BUY"
                
                with self.lock:
                    # Add to deque
                    self.trades.append({
                        'time': timestamp,
                        'value': value_usd,
                        'side': side,
                        'price': price
                    })
                    
                    # Update snapshot of last trade
                    self.metrics['last_whale_trade'] = {
                        'time': timestamp,
                        'side': side,
                        'value': value_usd,
                        'price': price
                    }
                    
                    # Log significant events
                    icon = "🟢" if side == "BUY" else "🔴"
                    logger.info(f"{icon} WHALE {side}: ${value_usd:,.0f} @ {price}")
                    
                self._cleanup_old_trades()
                self._update_metrics()
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _cleanup_old_trades(self):
        """Remove trades older than the window."""
        current_time = time.time()
        cutoff = current_time - self.window_seconds
        
        with self.lock:
            while self.trades and self.trades[0]['time'] < cutoff:
                self.trades.popleft()

    def _update_metrics(self):
        """Recalculate flow metrics from current trade window."""
        buy_vol = 0.0
        sell_vol = 0.0
        buy_count = 0
        sell_count = 0
        
        with self.lock:
            for trade in self.trades:
                if trade['side'] == 'BUY':
                    buy_vol += trade['value']
                    buy_count += 1
                else:
                    sell_vol += trade['value']
                    sell_count += 1
                    
            self.metrics['buy_vol'] = buy_vol
            self.metrics['sell_vol'] = sell_vol
            self.metrics['net_flow'] = buy_vol - sell_vol
            self.metrics['buy_count'] = buy_count
            self.metrics['sell_count'] = sell_count

    def get_metrics(self) -> Dict:
        """Get current whale flow metrics."""
        self._cleanup_old_trades() # Ensure data is fresh
        with self.lock:
            return self.metrics.copy()
