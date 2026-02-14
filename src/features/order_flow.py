"""
Funding Rate and Order Flow Analyzer

Features:
- Real-time funding rate analysis
- Order flow detection from trade stream
- CVD (Cumulative Volume Delta)
- Large order detection
"""

import numpy as np
import pandas as pd
import requests
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from collections import deque
import logging
import os
import time

logger = logging.getLogger(__name__)


@dataclass
class FundingSignal:
    """Funding rate signal."""
    rate: float  # Current funding rate
    predicted_rate: float  # Predicted next funding
    signal: str  # "long_favored", "short_favored", "neutral"
    strength: float  # Signal strength 0-1
    payout_direction: str  # Who pays: "longs_pay", "shorts_pay"


@dataclass
class OrderFlowSignal:
    """Order flow analysis signal."""
    cvd: float  # Cumulative Volume Delta
    cvd_trend: str  # "bullish", "bearish", "neutral"
    large_orders_bias: str  # Direction of large orders
    buy_pressure: float  # 0-1, ratio of aggressive buys
    signal_strength: float


class FundingRateAnalyzer:
    """
    Analyze funding rates for trade timing.
    
    Strategy: 
    - When longs pay (positive funding), shorts are favored
    - When shorts pay (negative funding), longs are favored
    - Extreme funding often precedes reversals
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        extreme_threshold: float = 0.0003,  # 0.03% is considered extreme
        cache_duration: int = 60,  # Cache for 60 seconds
    ):
        self.symbol = symbol
        self.extreme_threshold = extreme_threshold
        self.cache_duration = cache_duration
        self._last_fetch = 0
        self._cached_data = None
        
        logger.info(f"💰 FundingRateAnalyzer initialized for {symbol}")
    
    def _get_request_config(self, endpoint: str) -> Tuple[str, Optional[Dict]]:
        """
        Get URL and proxy config based on environment.
        
        Returns:
            Tuple of (url, proxies)
        """
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
            # Fallback to Public Data API (Spot only)
            # Note: Funding Rate endpoints don't exist on data-api.binance.vision
            # so this will likely fail or return 404 if we try to access fapi endpoints here
            # We handle this gracefully in the caller
            base_url = "https://data-api.binance.vision" 
            return f"{base_url}{endpoint}", None

    def _get_okx_funding(self) -> float:
        """Fetch funding rate from OKX as fallback."""
        try:
            # OKX Symbol format: BTC-USDT-SWAP
            okx_symbol = self.symbol.replace("USDT", "-USDT-SWAP")
            url = "https://www.okx.com/api/v5/public/funding-rate"
            response = requests.get(url, params={"instId": okx_symbol}, timeout=5)
            
            if response.status_code == 200:
                data = response.json().get('data', [])
                if data:
                    return float(data[0].get('fundingRate', 0))
            return 0.0
        except Exception:
            return 0.0

    def _fetch_funding_rate(self) -> Dict:
        """Fetch current and predicted funding rate from Binance."""
        now = time.time()
        
        # Use cache if fresh
        if self._cached_data and (now - self._last_fetch) < self.cache_duration:
            return self._cached_data
        
        try:
            # Current funding rate
            endpoint = "/fapi/v1/premiumIndex"
            url, proxies = self._get_request_config(endpoint)
            
            # If no proxy and attempting to hit Futures endpoint via data-api, 
            # Use OKX Fallback
            if not proxies and "data-api" in url:
                funding_rate = self._get_okx_funding()
                
                # OKX doesn't give prediction directly without history, so just use current
                self._cached_data = {
                    'rate': funding_rate,
                    'predicted': funding_rate,
                    'mark_price': 0, # Not critical for simply tracking rate sentiment
                    'index_price': 0,
                    'next_funding_time': 0,
                    'history': [],
                }
                self._last_fetch = now
                return self._cached_data

            response = requests.get(url, params={"symbol": self.symbol}, proxies=proxies, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            funding_rate = float(data.get('lastFundingRate', 0))
            mark_price = float(data.get('markPrice', 0))
            index_price = float(data.get('indexPrice', 0))
            next_funding_time = int(data.get('nextFundingTime', 0))
            
            # Funding rate history for prediction
            hist_endpoint = "/fapi/v1/fundingRate"
            hist_url, _ = self._get_request_config(hist_endpoint)
            
            hist_response = requests.get(
                hist_url, 
                params={"symbol": self.symbol, "limit": 10},
                proxies=proxies,
                timeout=10
            )
            hist_response.raise_for_status()
            history = hist_response.json()
            
            # Calculate average recent funding
            recent_rates = [float(h['fundingRate']) for h in history]
            predicted_rate = np.mean(recent_rates) if recent_rates else funding_rate
            
            self._cached_data = {
                'rate': funding_rate,
                'predicted': predicted_rate,
                'mark_price': mark_price,
                'index_price': index_price,
                'next_funding_time': next_funding_time,
                'history': recent_rates,
            }
            self._last_fetch = now
            
            return self._cached_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate (proxy={bool(os.environ.get('BINANCE_PROXY'))}): {e}")
            return {'rate': 0, 'predicted': 0, 'history': []}
    
    def get_signal(self) -> FundingSignal:
        """
        Get funding rate signal for trade filtering.
        
        Returns:
            FundingSignal with recommendation
        """
        data = self._fetch_funding_rate()
        
        rate = data.get('rate', 0)
        predicted = data.get('predicted', rate)
        
        # Determine who pays
        if rate > 0:
            payout = "longs_pay"
        elif rate < 0:
            payout = "shorts_pay"
        else:
            payout = "neutral"
        
        # Determine signal
        if rate > self.extreme_threshold:
            signal = "short_favored"  # Longs are paying a lot, market may reverse down
            strength = min(abs(rate) / (self.extreme_threshold * 3), 1.0)
        elif rate < -self.extreme_threshold:
            signal = "long_favored"  # Shorts are paying a lot, market may reverse up
            strength = min(abs(rate) / (self.extreme_threshold * 3), 1.0)
        else:
            # Moderate funding - slight favor to the side getting paid
            if rate > 0:
                signal = "slight_short_favored"
            elif rate < 0:
                signal = "slight_long_favored"
            else:
                signal = "neutral"
            strength = abs(rate) / self.extreme_threshold if self.extreme_threshold else 0
        
        result = FundingSignal(
            rate=rate,
            predicted_rate=predicted,
            signal=signal,
            strength=strength,
            payout_direction=payout
        )
        
        logger.info(
            f"💰 Funding: {rate:.4%} → {signal} (strength={strength:.2f})"
        )
        
        return result
    
    def should_trade(self, trade_type: str) -> Tuple[bool, str]:
        """
        Check if funding rate favors the trade.
        
        Args:
            trade_type: "long" or "short"
            
        Returns:
            Tuple of (should_trade, reason)
        """
        signal = self.get_signal()
        
        # Strong signals block opposing trades
        if trade_type == "long" and signal.signal == "short_favored" and signal.strength > 0.5:
            return False, f"Funding opposes LONG: {signal.rate:.4%} (longs paying heavily)"
        
        if trade_type == "short" and signal.signal == "long_favored" and signal.strength > 0.5:
            return False, f"Funding opposes SHORT: {signal.rate:.4%} (shorts paying heavily)"
        
        # Favorable funding adds confidence
        if trade_type == "long" and "long_favored" in signal.signal:
            return True, f"Funding favors LONG: {signal.rate:.4%} (shorts paying)"
        
        if trade_type == "short" and "short_favored" in signal.signal:
            return True, f"Funding favors SHORT: {signal.rate:.4%} (longs paying)"
        
        return True, f"Funding neutral: {signal.rate:.4%}"


class OrderFlowAnalyzer:
    """
    Analyze order flow for trade signals.
    
    Features:
    - CVD (Cumulative Volume Delta) - difference between buy and sell volume
    - Large order detection
    - Buy/sell pressure analysis
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        large_order_threshold: float = 50000,  # $50k+ is a large order
        lookback_minutes: int = 60,
    ):
        self.symbol = symbol
        self.large_order_threshold = large_order_threshold
        self.lookback_minutes = lookback_minutes
        
        # Trade buffer
        self.recent_trades: deque = deque(maxlen=10000)
        
        logger.info(f"📊 OrderFlowAnalyzer initialized (large orders > ${large_order_threshold:,.0f})")
    
    def _get_request_config(self) -> Tuple[str, Optional[Dict]]:
        """Get URL and proxy config."""
        proxy = os.environ.get("BINANCE_PROXY")
        
        if proxy:
            # Use Futures API (fapi) with Proxy
            return "https://fapi.binance.com/fapi/v1/trades", {
                "http": proxy, 
                "https": proxy
            }
        else:
            # Fallback to Spot API (api/v3) without Proxy
            return "https://data-api.binance.vision/api/v3/trades", None

    def _fetch_recent_trades(self) -> List[Dict]:
        """Fetch recent trades from Binance."""
        try:
            url, proxies = self._get_request_config()
            
            response = requests.get(
                url,
                params={"symbol": self.symbol, "limit": 1000},
                proxies=proxies,
                timeout=10
            )
            response.raise_for_status()
            trades = response.json()
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []
    
    def calculate_cvd(self, df: pd.DataFrame = None) -> float:
        """
        Calculate Cumulative Volume Delta from OHLCV data.
        
        Uses candle body position to estimate buy/sell volume:
        - Close > Open = buying pressure (bullish candle)
        - Close < Open = selling pressure (bearish candle)
        """
        if df is None or len(df) < 10:
            return 0
        
        # Use last 20 candles for CVD
        recent = df.tail(20)
        
        # Calculate volume delta per candle
        # Positive = buy volume, Negative = sell volume
        body = recent['close'] - recent['open']
        range_size = recent['high'] - recent['low']
        
        # Volume weighted by body size relative to range
        body_ratio = body / (range_size + 1e-10)
        volume_delta = body_ratio * recent['volume']
        
        cvd = volume_delta.sum()
        
        return cvd
    
    def analyze_large_orders(self) -> Dict:
        """
        Analyze recent trades for large order patterns.
        """
        trades = self._fetch_recent_trades()
        
        if not trades:
            return {
                'large_buys': 0,
                'large_sells': 0,
                'bias': 'neutral',
                'total_large_volume': 0
            }
        
        large_buys = 0
        large_sells = 0
        large_buy_vol = 0
        large_sell_vol = 0
        
        for trade in trades:
            price = float(trade.get('price', 0))
            qty = float(trade.get('qty', 0))
            is_maker = trade.get('isBuyerMaker', False)
            
            value = price * qty
            
            if value >= self.large_order_threshold:
                if is_maker:
                    large_sells += 1
                    large_sell_vol += value
                else:
                    large_buys += 1
                    large_buy_vol += value
        
        total = large_buys + large_sells
        if total > 0:
            if large_buy_vol > large_sell_vol * 1.2:
                bias = 'bullish'
            elif large_sell_vol > large_buy_vol * 1.2:
                bias = 'bearish'
            else:
                bias = 'neutral'
        else:
            bias = 'neutral'
        
        return {
            'large_buys': large_buys,
            'large_sells': large_sells,
            'large_buy_volume': large_buy_vol,
            'large_sell_volume': large_sell_vol,
            'bias': bias,
            'total_large_volume': large_buy_vol + large_sell_vol
        }
    
    def get_signal(self, df: pd.DataFrame = None) -> OrderFlowSignal:
        """
        Get order flow signal.
        """
        # Calculate CVD
        cvd = self.calculate_cvd(df)
        
        # Normalize CVD
        if df is not None and len(df) > 0:
            avg_volume = df['volume'].tail(20).mean()
            cvd_normalized = cvd / (avg_volume + 1e-10)
        else:
            cvd_normalized = 0
        
        # CVD trend
        if cvd_normalized > 2:
            cvd_trend = 'bullish'
        elif cvd_normalized < -2:
            cvd_trend = 'bearish'
        else:
            cvd_trend = 'neutral'
        
        # Large orders
        large_orders = self.analyze_large_orders()
        
        # Combined signal strength
        if cvd_trend == large_orders['bias']:
            strength = 0.8
        elif large_orders['bias'] == 'neutral':
            strength = 0.5
        else:
            strength = 0.3  # Conflicting signals
        
        # Buy pressure (from candle analysis)
        if df is not None and len(df) > 0:
            recent = df.tail(10)
            bullish_candles = (recent['close'] > recent['open']).sum()
            buy_pressure = bullish_candles / len(recent)
        else:
            buy_pressure = 0.5
        
        result = OrderFlowSignal(
            cvd=cvd,
            cvd_trend=cvd_trend,
            large_orders_bias=large_orders['bias'],
            buy_pressure=buy_pressure,
            signal_strength=strength
        )
        
        logger.info(
            f"📊 Order Flow: CVD={cvd_normalized:.2f} ({cvd_trend}), "
            f"Large Orders={large_orders['bias']}, Buy Pressure={buy_pressure:.0%}"
        )
        
        return result
    
    def should_trade(self, trade_type: str, df: pd.DataFrame = None) -> Tuple[bool, str]:
        """Check if order flow supports the trade."""
        signal = self.get_signal(df)
        
        # Strong opposing flow blocks trade
        if trade_type == "long" and signal.cvd_trend == "bearish" and signal.signal_strength > 0.6:
            return False, f"Order flow opposing LONG: CVD bearish, large sells"
        
        if trade_type == "short" and signal.cvd_trend == "bullish" and signal.signal_strength > 0.6:
            return False, f"Order flow opposing SHORT: CVD bullish, large buys"
        
        return True, f"Order flow: {signal.cvd_trend}, pressure={signal.buy_pressure:.0%}"
