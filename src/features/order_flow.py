"""
Funding Rate and Order Flow Analyzer

Features:
- Real-time funding rate analysis
- 3-layer order flow signal (CVD + Taker Ratio + Notable Orders)
- CVD (Cumulative Volume Delta) from OHLCV
- Large/Notable order detection with $5K threshold
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
    rate: float          # Current funding rate
    predicted_rate: float  # Predicted next funding
    signal: str          # "long_favored", "short_favored", "neutral"
    strength: float      # Signal strength 0-1
    payout_direction: str  # Who pays: "longs_pay", "shorts_pay"


@dataclass
class OrderFlowSignal:
    """Order flow analysis signal."""
    cvd: float              # Cumulative Volume Delta
    cvd_trend: str          # "bullish", "bearish", "neutral"
    large_orders_bias: str  # Direction of large orders
    buy_pressure: float     # 0-1, ratio of aggressive buys
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
        cache_duration: int = 60,           # Cache for 60 seconds
    ):
        self.symbol = symbol
        self.extreme_threshold = extreme_threshold
        self.cache_duration = cache_duration
        self._last_fetch = 0
        self._cached_data = None
        
        # Open Interest tracking for liquidation cascade detection
        self._oi_history: deque = deque(maxlen=24)  # Rolling 24 samples (~24 hours at 1h interval)
        self._last_oi_fetch = 0
        self._cached_oi = 0.0

        logger.info(f"💰 FundingRateAnalyzer initialized for {symbol}")

    def _fetch_funding_rate(self) -> Dict:
        """Fetch current funding rate exclusively from OKX (no proxy needed)."""
        now = time.time()

        # Use cache if fresh
        if self._cached_data and (now - self._last_fetch) < self.cache_duration:
            return self._cached_data

        rate = self._get_okx_funding()

        self._cached_data = {
            'rate': rate,
            'predicted': rate,
            'mark_price': 0,
            'index_price': 0,
            'next_funding_time': 0,
            'history': [rate],
        }
        self._last_fetch = now
        return self._cached_data

    def _get_okx_funding(self) -> float:
        """Fetch funding rate from OKX (no geo-blocking)."""
        try:
            # Map symbol to OKX format (e.g. BTCUSDT -> BTC-USDT-SWAP)
            base = self.symbol.replace('USDT', '')
            okx_inst = f"{base}-USDT-SWAP"
            url = "https://www.okx.com/api/v5/public/funding-rate"
            response = requests.get(
                url,
                params={"instId": okx_inst},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('data'):
                rate = float(data['data'][0].get('fundingRate', 0))
                logger.info(f"💰 OKX Funding Rate for {okx_inst}: {rate:.6f}")
                return rate
        except Exception as e:
            logger.warning(f"OKX funding rate fetch failed: {e}")
        
        return 0.0

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
            signal = "short_favored"  # Longs paying a lot → market may reverse down
            strength = min(abs(rate) / (self.extreme_threshold * 3), 1.0)
        elif rate < -self.extreme_threshold:
            signal = "long_favored"   # Shorts paying a lot → market may reverse up
            strength = min(abs(rate) / (self.extreme_threshold * 3), 1.0)
        else:
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

        logger.info(f"💰 Funding: {rate:.4%} → {signal} (strength={strength:.2f})")
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

        if trade_type == "long" and signal.signal == "short_favored" and signal.strength > 0.5:
            return False, f"Funding opposes LONG: {signal.rate:.4%} (longs paying heavily)"

        if trade_type == "short" and signal.signal == "long_favored" and signal.strength > 0.5:
            return False, f"Funding opposes SHORT: {signal.rate:.4%} (shorts paying heavily)"

        if trade_type == "long" and "long_favored" in signal.signal:
            return True, f"Funding favors LONG: {signal.rate:.4%} (shorts paying)"

        if trade_type == "short" and "short_favored" in signal.signal:
            return True, f"Funding favors SHORT: {signal.rate:.4%} (longs paying)"

        return True, f"Funding neutral: {signal.rate:.4%}"

    def _fetch_open_interest(self) -> float:
        """Fetch current Open Interest from OKX."""
        now = time.time()
        if self._cached_oi > 0 and (now - self._last_oi_fetch) < self.cache_duration:
            return self._cached_oi
            
        try:
            base = self.symbol.replace('USDT', '')
            okx_inst = f"{base}-USDT-SWAP"
            url = "https://www.okx.com/api/v5/public/open-interest"
            response = requests.get(
                url,
                params={"instId": okx_inst},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('data'):
                oi = float(data['data'][0].get('oi', 0))
                self._cached_oi = oi
                self._last_oi_fetch = now
                
                # Track rolling history
                self._oi_history.append(oi)
                logger.info(f"📈 OKX Open Interest for {okx_inst}: {oi:,.0f} contracts")
                return oi
        except Exception as e:
            logger.warning(f"OKX OI fetch failed: {e}")
        
        return self._cached_oi if self._cached_oi > 0 else 0.0

    def get_liquidation_danger(self) -> Dict:
        """
        Compute a Liquidation Danger Score.
        
        Combines:
        - Funding Rate (how greedy is retail?)
        - Open Interest spike (how loaded is the system?)
        
        Returns dict with:
          - danger_score: 0.0 (safe) to 1.0 (extremely dangerous)
          - long_danger: True if conditions favor a long squeeze
          - short_danger: True if conditions favor a short squeeze
          - reason: Human-readable explanation
        """
        funding_data = self._fetch_funding_rate()
        rate = funding_data.get('rate', 0)
        
        oi = self._fetch_open_interest()
        
        # Calculate OI spike ratio (current vs rolling average)
        oi_spike = 0.0
        if len(self._oi_history) >= 3:
            avg_oi = sum(list(self._oi_history)[:-1]) / (len(self._oi_history) - 1)
            if avg_oi > 0:
                oi_spike = (oi - avg_oi) / avg_oi  # e.g. 0.10 = 10% above average
        
        # Funding danger: how extreme is the funding rate?
        funding_danger = min(abs(rate) / 0.001, 1.0)  # Normalize: 0.1% funding = max danger
        
        # OI danger: how much has OI spiked?
        oi_danger = min(max(oi_spike, 0) / 0.15, 1.0)  # 15% OI spike = max danger
        
        # Combined danger score (weighted blend)
        danger_score = 0.6 * funding_danger + 0.4 * oi_danger
        danger_score = min(danger_score, 1.0)
        
        # Determine direction of danger
        long_danger = rate > 0.00015 and danger_score > 0.5  # Greedy longs → long squeeze risk
        short_danger = rate < -0.00015 and danger_score > 0.5  # Greedy shorts → short squeeze risk
        
        reason_parts = []
        if funding_danger > 0.3:
            reason_parts.append(f"Funding={rate:.4%}")
        if oi_danger > 0.3:
            reason_parts.append(f"OI_Spike={oi_spike:+.1%}")
        reason = ", ".join(reason_parts) if reason_parts else "Normal"
        
        logger.info(
            f"⚠️ Liquidation Danger [{self.symbol}]: score={danger_score:.2f} "
            f"(funding={funding_danger:.2f}, oi={oi_danger:.2f}) "
            f"long_danger={long_danger}, short_danger={short_danger}"
        )
        
        return {
            'danger_score': danger_score,
            'long_danger': long_danger,
            'short_danger': short_danger,
            'funding_rate': rate,
            'oi': oi,
            'oi_spike': oi_spike,
            'reason': reason,
        }


class OrderFlowAnalyzer:
    """
    Enhanced Order Flow Analyzer with 3-layer signal.

    Layer 1: CVD from OHLCV (most reliable, always available) — 50% weight
    Layer 2: Taker buy/sell ratio from recent trades — 30% weight
    Layer 3: Notable orders ($5K+ threshold) — 20% weight

    Combined signal provides [-1, +1] directional score.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        notable_order_threshold: float = 5000,  # $5K (was $50K)
        lookback_minutes: int = 60,
    ):
        self.symbol = symbol
        self.notable_order_threshold = notable_order_threshold
        # Keep backward compat attribute name
        self.large_order_threshold = notable_order_threshold
        self.lookback_minutes = lookback_minutes

        # Trade buffer
        self.recent_trades: deque = deque(maxlen=10000)

        # Cache
        self._cache_time = 0
        self._cache_result = None
        self._cache_ttl = 30  # 30 seconds

        logger.info(f"📊 OrderFlowAnalyzer initialized (notable orders > ${notable_order_threshold:,.0f})")

    def _fetch_recent_trades(self) -> List[Dict]:
        """Fetch recent trades from Binance Spot API (with OKX fallback)."""
        try:
            url = "https://data-api.binance.vision/api/v3/trades"
            response = requests.get(
                url,
                params={"symbol": self.symbol, "limit": 1000},
                timeout=10
            )
            response.raise_for_status()
            trades = response.json()

            if not trades:
                return self._get_okx_trades()

            return trades

        except Exception as e:
            logger.warning(f"Failed to fetch Binance Spot trades: {e}. Trying OKX fallback...")
            return self._get_okx_trades()

    def _get_okx_trades(self) -> List[Dict]:
        """Fetch recent trades from OKX (no geo-blocking)."""
        try:
            base = self.symbol.replace('USDT', '')
            okx_inst = f"{base}-USDT"
            url = "https://www.okx.com/api/v5/market/trades"
            response = requests.get(
                url,
                params={"instId": okx_inst, "limit": "500"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if data.get('data'):
                trades = []
                for t in data['data']:
                    trades.append({
                        'price': t.get('px', '0'),
                        'qty': t.get('sz', '0'),
                        'isBuyerMaker': t.get('side', 'buy') == 'sell',
                    })
                logger.info(f"📊 OKX trades fetched: {len(trades)} trades for {okx_inst}")
                return trades
        except Exception as e:
            logger.warning(f"OKX trades fetch failed: {e}")

        return []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 1: CVD from OHLCV (50% of signal weight)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def calculate_cvd(self, df: pd.DataFrame = None) -> Dict:
        """
        Calculate Cumulative Volume Delta from OHLCV data.

        Uses candle body position to estimate buy/sell volume:
        - Close > Open = buying pressure (bullish candle)
        - Close < Open = selling pressure (bearish candle)

        Returns dict with raw CVD, normalized score [-1,+1], and trend.
        """
        if df is None or len(df) < 10:
            return {'cvd': 0, 'score': 0.0, 'trend': 'neutral'}

        recent = df.tail(20)
        body = recent['close'] - recent['open']
        range_size = recent['high'] - recent['low']
        body_ratio = body / (range_size + 1e-10)
        volume_delta = body_ratio * recent['volume']

        cvd_raw = volume_delta.sum()

        # Normalize CVD against average volume for a [-1, +1] score
        avg_volume = recent['volume'].mean()
        cvd_normalized = cvd_raw / (avg_volume * 5 + 1e-10)
        cvd_score = float(np.clip(cvd_normalized, -1, 1))

        # Short-term CVD trend (last 5 vs previous 5 candles)
        if len(recent) >= 10:
            recent_5 = volume_delta.tail(5).sum()
            prev_5 = volume_delta.tail(10).head(5).sum()
            if recent_5 > prev_5 * 1.2:
                trend = 'bullish'
            elif recent_5 < prev_5 * 0.8:
                trend = 'bearish'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'

        return {
            'cvd': float(cvd_raw),
            'score': cvd_score,
            'trend': trend,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 2: Taker Buy/Sell Ratio (30% of signal weight)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def calculate_taker_ratio(self, trades: List[Dict] = None) -> Dict:
        """
        Calculate taker buy/sell volume ratio from ALL recent trades.
        No threshold filtering — uses total volume to measure
        aggregate buy vs sell pressure.
        """
        if trades is None:
            trades = self._fetch_recent_trades()

        if not trades:
            return {
                'buy_volume': 0, 'sell_volume': 0,
                'ratio': 0.5, 'score': 0.0, 'total_trades': 0
            }

        buy_vol = 0
        sell_vol = 0

        for trade in trades:
            price = float(trade.get('price', 0))
            qty = float(trade.get('qty', 0))
            is_maker = trade.get('isBuyerMaker', False)
            value = price * qty

            if is_maker:
                sell_vol += value
            else:
                buy_vol += value

        total = buy_vol + sell_vol
        ratio = buy_vol / total if total > 0 else 0.5

        # Convert ratio to [-1, +1]: 0.5 = neutral, >0.5 = bullish, <0.5 = bearish
        score = float(np.clip((ratio - 0.5) * 4, -1, 1))

        return {
            'buy_volume': round(buy_vol, 2),
            'sell_volume': round(sell_vol, 2),
            'ratio': round(ratio, 3),
            'score': score,
            'total_trades': len(trades),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 3: Notable Orders (20% of signal weight)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def analyze_large_orders(self, trades: List[Dict] = None) -> Dict:
        """
        Analyze recent trades for notable order patterns.
        Uses $5K threshold (down from $50K) to capture meaningful activity.
        """
        if trades is None:
            trades = self._fetch_recent_trades()

        if not trades:
            return {
                'large_buys': 0, 'large_sells': 0,
                'large_buy_volume': 0, 'large_sell_volume': 0,
                'bias': 'neutral', 'total_large_volume': 0, 'score': 0.0
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

            if value >= self.notable_order_threshold:
                if is_maker:
                    large_sells += 1
                    large_sell_vol += value
                else:
                    large_buys += 1
                    large_buy_vol += value

        total_vol = large_buy_vol + large_sell_vol
        if total_vol > 0:
            score = float(np.clip((large_buy_vol - large_sell_vol) / total_vol, -1, 1))
            if large_buy_vol > large_sell_vol * 1.2:
                bias = 'bullish'
            elif large_sell_vol > large_buy_vol * 1.2:
                bias = 'bearish'
            else:
                bias = 'neutral'
        else:
            score = 0.0
            bias = 'neutral'

        return {
            'large_buys': large_buys,
            'large_sells': large_sells,
            'large_buy_volume': round(large_buy_vol, 2),
            'large_sell_volume': round(large_sell_vol, 2),
            'bias': bias,
            'total_large_volume': round(total_vol, 2),
            'score': score,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Combined: Enhanced Signal (all 4 layers blended)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_enhanced_signal(self, df: pd.DataFrame = None) -> Dict:
        """
        Get blended order flow signal from all 4 layers.

        Weights:
          - CVD (OHLCV):          40% — most reliable, always available
          - Taker ratio:          25% — aggregate buy/sell pressure
          - Notable orders:       15% — institutional activity
          - Order Book Imbalance: 20% — bid/ask depth ratio (short-term predictor)

        Returns dict with composite score [-1,+1] and layer details.
        """
        # Check cache
        now = time.time()
        if self._cache_result and (now - self._cache_time) < self._cache_ttl:
            return self._cache_result

        # Fetch trades once, share across layers 2 & 3
        trades = self._fetch_recent_trades()

        # Layer 1: CVD from OHLCV
        cvd_data = self.calculate_cvd(df)

        # Layer 2: Taker ratio from all trades
        taker_data = self.calculate_taker_ratio(trades)

        # Layer 3: Notable orders ($5K+)
        notable_data = self.analyze_large_orders(trades)

        # Layer 4: Order Book Imbalance (bid/ask depth ratio)
        ob_data = None
        ob_score = 0.0
        try:
            from src.features.orderbook_imbalance import get_orderbook_imbalance
            ob_data = get_orderbook_imbalance(self.symbol)
            if ob_data:
                ob_score = ob_data.get('score', 0.0)
        except Exception as e:
            logger.warning(f"Order book imbalance unavailable: {e}")

        # Blend: 40% CVD + 25% taker + 15% notable + 20% OB imbalance
        composite_score = (
            0.40 * cvd_data['score'] +
            0.25 * taker_data['score'] +
            0.15 * notable_data['score'] +
            0.20 * ob_score
        )
        composite_score = float(np.clip(composite_score, -1, 1))

        # Determine overall bias
        if composite_score > 0.15:
            bias = 'bullish'
        elif composite_score < -0.15:
            bias = 'bearish'
        else:
            bias = 'neutral'

        result = {
            'score': composite_score,
            'bias': bias,
            # Layer details for dashboard
            'cvd': cvd_data,
            'taker': taker_data,
            'notable': notable_data,
            'orderbook': ob_data or {'score': 0, 'bias': 'neutral'},
            # Backward compat for dashboard
            'large_buys': notable_data['large_buys'],
            'large_sells': notable_data['large_sells'],
            'large_buy_volume': notable_data['large_buy_volume'],
            'large_sell_volume': notable_data['large_sell_volume'],
            'total_large_volume': notable_data['total_large_volume'],
        }

        ob_str = f"OB={ob_score:+.2f}({ob_data['bias'] if ob_data else 'N/A'})" if ob_data else "OB=N/A"
        logger.info(
            f"📊 Order Flow [{self.symbol}]: score={composite_score:+.2f} ({bias}) | "
            f"CVD={cvd_data['score']:+.2f}({cvd_data['trend']}) | "
            f"Taker={taker_data['score']:+.2f}(buy={taker_data['ratio']:.1%}) | "
            f"Notable={notable_data['score']:+.2f}(B:{notable_data['large_buys']}/S:{notable_data['large_sells']}) | "
            f"{ob_str}"
        )

        # Cache result
        self._cache_time = now
        self._cache_result = result

        return result

    def detect_cvd_divergence(self, df: pd.DataFrame = None) -> Dict:
        """
        Detect CVD Divergence: price going up while volume delta goes down.
        This signals hidden selling (whales limit-selling into retail buying)
        and often precedes a sudden dump.
        
        Returns:
            dict with divergence_detected (bool), direction, and details.
        """
        if df is None or len(df) < 20:
            return {'divergence_detected': False, 'direction': 'none', 'price_trend': 'unknown', 'cvd_trend': 'unknown'}
        
        recent = df.tail(20)
        
        # Price trend: compare last 5 candles avg vs previous 5
        price_recent = recent['close'].tail(5).mean()
        price_prev = recent['close'].tail(10).head(5).mean()
        price_change = (price_recent - price_prev) / price_prev
        
        if price_change > 0.002:  # 0.2% up
            price_trend = 'bullish'
        elif price_change < -0.002:
            price_trend = 'bearish'
        else:
            price_trend = 'neutral'
        
        # CVD trend: same window
        body = recent['close'] - recent['open']
        range_size = recent['high'] - recent['low']
        body_ratio = body / (range_size + 1e-10)
        volume_delta = body_ratio * recent['volume']
        
        cvd_recent = volume_delta.tail(5).sum()
        cvd_prev = volume_delta.tail(10).head(5).sum()
        
        if cvd_recent > cvd_prev * 1.1:
            cvd_trend = 'bullish'
        elif cvd_recent < cvd_prev * 0.9:
            cvd_trend = 'bearish'
        else:
            cvd_trend = 'neutral'
        
        # Divergence: price up but CVD down (bearish divergence → dump incoming)
        bearish_divergence = price_trend == 'bullish' and cvd_trend == 'bearish'
        # Divergence: price down but CVD up (bullish divergence → pump incoming)
        bullish_divergence = price_trend == 'bearish' and cvd_trend == 'bullish'
        
        divergence_detected = bearish_divergence or bullish_divergence
        direction = 'bearish' if bearish_divergence else ('bullish' if bullish_divergence else 'none')
        
        if divergence_detected:
            logger.warning(
                f"⚡ CVD DIVERGENCE [{self.symbol}]: {direction.upper()} "
                f"(price={price_trend}, cvd={cvd_trend}, Δprice={price_change:+.2%})"
            )
        
        return {
            'divergence_detected': divergence_detected,
            'direction': direction,
            'price_trend': price_trend,
            'cvd_trend': cvd_trend,
            'price_change': price_change,
        }

    def get_signal(self, df: pd.DataFrame = None) -> OrderFlowSignal:
        """Get order flow signal (backward-compatible wrapper)."""
        enhanced = self.get_enhanced_signal(df)

        return OrderFlowSignal(
            cvd=enhanced['cvd']['cvd'],
            cvd_trend=enhanced['cvd']['trend'],
            large_orders_bias=enhanced['notable']['bias'],
            buy_pressure=enhanced['taker']['ratio'],
            signal_strength=abs(enhanced['score']),
        )

    def should_trade(self, trade_type: str, df: pd.DataFrame = None) -> Tuple[bool, str]:
        """Check if order flow supports the trade."""
        enhanced = self.get_enhanced_signal(df)
        score = enhanced['score']

        if trade_type == "long" and score < -0.4:
            return False, f"Order flow opposing LONG: score={score:+.2f} (bearish)"

        if trade_type == "short" and score > 0.4:
            return False, f"Order flow opposing SHORT: score={score:+.2f} (bullish)"

        return True, f"Order flow: {enhanced['bias']}, score={score:+.2f}"
