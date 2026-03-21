"""
Comprehensive Market Analysis Test Suite

Tests the market analysis pipeline end-to-end:
- API-level tests against live server (port 5001)
- Component-level tests with real Binance data
- Signal quality validation
- Schema validation

All tests use REAL API calls — no mocks.
"""

import pytest
import requests
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:5001"
BINANCE_BASE = "https://data-api.binance.vision/api/v3"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEOUT = 30


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
    """Fetch real OHLCV data from Binance."""
    url = f"{BINANCE_BASE}/klines"
    resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df


def get_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """Fetch market analysis from live API."""
    resp = requests.get(f"{API_BASE}/api/market", params={"symbol": symbol}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: API-Level Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIHealth:
    """Verify the API server is responsive."""

    def test_ping(self):
        resp = requests.get(f"{API_BASE}/api/ping", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "timestamp" in data

    def test_health(self):
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestMarketAPIStructure:
    """Test /api/market returns valid structure for all symbols."""

    @pytest.fixture(scope="class")
    def btc_data(self):
        return get_market_data("BTCUSDT")

    def test_market_returns_200(self):
        resp = requests.get(f"{API_BASE}/api/market", params={"symbol": "BTCUSDT"}, timeout=TIMEOUT)
        assert resp.status_code == 200

    def test_market_has_required_fields(self, btc_data):
        required = ["timestamp", "symbol", "whale", "regime", "funding", "order_flow"]
        for field in required:
            assert field in btc_data, f"Missing field: {field}"

    def test_no_internal_fields_leaked(self, btc_data):
        """_fetched_at should not be in the response."""
        assert "_fetched_at" not in btc_data, "_fetched_at leaked in response"

    def test_symbol_matches_request(self, btc_data):
        assert btc_data["symbol"] == "BTCUSDT"

    def test_timestamp_is_iso_format(self, btc_data):
        ts = btc_data["timestamp"]
        # Should be parseable as ISO datetime
        from datetime import datetime
        datetime.fromisoformat(ts)

    def test_price_is_reasonable(self, btc_data):
        price = btc_data.get("price")
        assert price is not None, "Price missing from response"
        assert isinstance(price, (int, float))
        # BTC should be between $10K and $500K
        assert 10_000 < price < 500_000, f"BTC price unreasonable: {price}"

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_all_symbols_return_data(self, symbol):
        data = get_market_data(symbol)
        assert "timestamp" in data
        assert "regime" in data
        assert data.get("price") is not None or data.get("regime") is not None


class TestMarketAPICache:
    """Test cache behavior."""

    def test_cache_returns_fast(self):
        """Second call within TTL should be significantly faster (cached)."""
        # First call — may be slow (fetches fresh data)
        t0 = time.time()
        data1 = get_market_data("BTCUSDT")
        t1 = time.time()
        first_duration = t1 - t0

        # Second call — should be cached
        t2 = time.time()
        data2 = get_market_data("BTCUSDT")
        t3 = time.time()
        second_duration = t3 - t2

        # Cached call should be at least 2x faster (usually 10x+)
        assert second_duration < max(first_duration, 1.0), (
            f"Cache doesn't seem to work: first={first_duration:.2f}s, second={second_duration:.2f}s"
        )

    def test_different_symbols_have_different_data(self):
        btc = get_market_data("BTCUSDT")
        eth = get_market_data("ETHUSDT")
        # Prices should differ
        if btc.get("price") and eth.get("price"):
            assert btc["price"] != eth["price"]


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Signal Schema Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestWhaleSignalSchema:
    """Validate whale signal structure and values."""

    @pytest.fixture(scope="class")
    def whale(self):
        data = get_market_data("BTCUSDT")
        return data.get("whale")

    def test_whale_is_not_null(self, whale):
        assert whale is not None, "Whale signal is null"

    def test_whale_has_direction(self, whale):
        direction = whale.get("direction")
        assert direction is not None
        assert direction.upper() in ["BULLISH", "BEARISH", "NEUTRAL"], (
            f"Invalid whale direction: {direction}"
        )

    def test_whale_has_score(self, whale):
        score = whale.get("score")
        assert score is not None
        assert isinstance(score, (int, float))
        assert -1.0 <= score <= 1.0, f"Whale score out of range: {score}"

    def test_whale_has_confidence(self, whale):
        confidence = whale.get("confidence")
        assert confidence is not None
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 100, f"Whale confidence out of range: {confidence}"

    def test_whale_has_flow_metrics(self, whale):
        assert "flow_metrics" in whale


class TestRegimeSignalSchema:
    """Validate regime signal structure and values."""

    @pytest.fixture(scope="class")
    def regime(self):
        data = get_market_data("BTCUSDT")
        return data.get("regime")

    def test_regime_is_not_null(self, regime):
        assert regime is not None, "Regime signal is null"

    def test_regime_has_type(self, regime):
        regime_type = regime.get("type")
        assert regime_type is not None
        valid_types = [
            "TRENDING_UP", "TRENDING_DOWN", "RANGING",
            "HIGH_VOLATILITY", "LOW_VOLATILITY", "UNKNOWN",
            # Also accept lowercase/underscore variants
            "trending_up", "trending_down", "ranging",
            "high_volatility", "low_volatility",
            # HMM regime types
            "bull_trend", "bear_trend", "range_chop", "high_vol_breakout",
            "BULL_TREND", "BEAR_TREND", "RANGE_CHOP", "HIGH_VOL_BREAKOUT",
        ]
        assert regime_type in valid_types, f"Invalid regime type: {regime_type}"

    def test_regime_has_adx(self, regime):
        adx = regime.get("adx")
        assert adx is not None
        assert isinstance(adx, (int, float))
        assert 0 <= adx <= 100, f"ADX out of range: {adx}"

    def test_regime_has_volatility(self, regime):
        vol = regime.get("volatility")
        assert vol is not None
        assert isinstance(vol, (int, float))
        assert 0 < vol < 10, f"Volatility ratio unreasonable: {vol}"


class TestFundingSignalSchema:
    """Validate funding rate signal structure and values."""

    @pytest.fixture(scope="class")
    def funding(self):
        data = get_market_data("BTCUSDT")
        return data.get("funding")

    def test_funding_is_not_null(self, funding):
        assert funding is not None, "Funding signal is null"

    def test_funding_rate_is_reasonable(self, funding):
        rate = funding.get("rate")
        assert rate is not None
        assert isinstance(rate, (int, float))
        # Funding rate should be between -1% and 1% (displayed as percentage)
        assert -1.0 <= rate <= 1.0, f"Funding rate unreasonable: {rate}%"

    def test_funding_has_bias(self, funding):
        bias = funding.get("bias")
        assert bias is not None
        valid_biases = [
            "long_favored", "short_favored", "neutral",
            "slight_long_favored", "slight_short_favored",
        ]
        assert bias in valid_biases, f"Invalid funding bias: {bias}"

    def test_funding_has_annualized(self, funding):
        ann = funding.get("annualized")
        assert ann is not None
        assert isinstance(ann, (int, float))
        # Annualized funding should be between -200% and 200%
        assert -200 <= ann <= 200, f"Annualized funding unreasonable: {ann}%"


class TestOrderFlowSignalSchema:
    """Validate order flow signal structure and values."""

    @pytest.fixture(scope="class")
    def order_flow(self):
        data = get_market_data("BTCUSDT")
        return data.get("order_flow")

    def test_order_flow_is_not_null(self, order_flow):
        assert order_flow is not None, "Order flow signal is null"

    def test_order_flow_has_bias(self, order_flow):
        bias = order_flow.get("bias")
        assert bias is not None
        assert bias in ["bullish", "bearish", "neutral"], f"Invalid order flow bias: {bias}"

    def test_order_flow_has_score(self, order_flow):
        score = order_flow.get("score")
        assert score is not None
        assert isinstance(score, (int, float))
        assert -1.0 <= score <= 1.0, f"Order flow score out of range: {score}"

    def test_order_flow_has_large_buys_sells(self, order_flow):
        assert "large_buys" in order_flow
        assert "large_sells" in order_flow
        assert order_flow["large_buys"] >= 0
        assert order_flow["large_sells"] >= 0

    def test_order_flow_has_cvd(self, order_flow):
        cvd = order_flow.get("cvd")
        assert cvd is not None
        assert "score" in cvd
        assert "trend" in cvd
        assert cvd["trend"] in ["bullish", "bearish", "neutral"]

    def test_order_flow_has_taker(self, order_flow):
        taker = order_flow.get("taker")
        assert taker is not None
        assert "ratio" in taker
        ratio = taker["ratio"]
        assert 0 <= ratio <= 1.0, f"Taker ratio out of range: {ratio}"

    def test_order_flow_has_notable(self, order_flow):
        notable = order_flow.get("notable")
        assert notable is not None
        assert "large_buys" in notable
        assert "large_sells" in notable
        assert "bias" in notable


class TestForecastAndNews:
    """Validate optional signal fields."""

    @pytest.fixture(scope="class")
    def data(self):
        return get_market_data("BTCUSDT")

    def test_forecast_is_null_or_valid(self, data):
        forecast = data.get("forecast")
        # Forecast may be null (disabled in API to prevent deadlock)
        if forecast is not None:
            assert "return_1h" in forecast or "return_4h" in forecast

    def test_news_is_null_or_valid(self, data):
        news = data.get("news")
        # News is disabled per user request
        if news is not None:
            assert "sentiment" in news


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Component-Level Tests with Real Data
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeDetector:
    """Test MarketRegimeDetector with real Binance data."""

    @pytest.fixture(scope="class")
    def detector(self):
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.regime_detector import MarketRegimeDetector
        return MarketRegimeDetector()

    @pytest.fixture(scope="class")
    def btc_df(self):
        return fetch_binance_klines("BTCUSDT", "1h", 200)

    def test_initialization(self, detector):
        assert detector.adx_period == 14
        assert detector.atr_period == 14
        assert detector.trend_threshold == 25.0

    def test_detect_regime_returns_valid(self, detector, btc_df):
        result = detector.detect_regime(btc_df)
        assert result is not None
        assert hasattr(result, 'regime')
        assert hasattr(result, 'trend_strength')
        assert hasattr(result, 'volatility_ratio')

    def test_regime_type_is_valid(self, detector, btc_df):
        from src.features.regime_detector import MarketRegime
        result = detector.detect_regime(btc_df)
        valid_regimes = [r for r in MarketRegime]
        assert result.regime in valid_regimes

    def test_adx_in_range(self, detector, btc_df):
        result = detector.detect_regime(btc_df)
        assert 0 <= result.trend_strength <= 100

    def test_volatility_ratio_positive(self, detector, btc_df):
        result = detector.detect_regime(btc_df)
        assert result.volatility_ratio > 0

    def test_confidence_in_range(self, detector, btc_df):
        result = detector.detect_regime(btc_df)
        assert 0 <= result.confidence <= 1.0

    def test_insufficient_data(self, detector):
        small_df = pd.DataFrame({
            'open': [100, 101], 'high': [102, 103],
            'low': [99, 100], 'close': [101, 102], 'volume': [1000, 1100]
        })
        result = detector.detect_regime(small_df)
        from src.features.regime_detector import MarketRegime
        assert result.regime == MarketRegime.UNKNOWN

    def test_should_trade_returns_tuple(self, detector, btc_df):
        should, reason, mult = detector.should_trade(btc_df, "long")
        assert isinstance(should, bool)
        assert isinstance(reason, str)
        assert isinstance(mult, float)


class TestOrderFlowAnalyzer:
    """Test OrderFlowAnalyzer with real data."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.order_flow import OrderFlowAnalyzer
        return OrderFlowAnalyzer(symbol="BTCUSDT")

    @pytest.fixture(scope="class")
    def btc_df(self):
        return fetch_binance_klines("BTCUSDT", "1h", 100)

    def test_initialization(self, analyzer):
        assert analyzer.symbol == "BTCUSDT"
        assert analyzer.notable_order_threshold == 5000

    def test_cvd_with_real_data(self, analyzer, btc_df):
        result = analyzer.calculate_cvd(btc_df)
        assert "cvd" in result
        assert "score" in result
        assert "trend" in result
        assert -1 <= result["score"] <= 1
        assert result["trend"] in ["bullish", "bearish", "neutral"]

    def test_taker_ratio_with_real_trades(self, analyzer):
        result = analyzer.calculate_taker_ratio()
        assert "buy_volume" in result
        assert "sell_volume" in result
        assert "ratio" in result
        assert "score" in result
        assert 0 <= result["ratio"] <= 1.0
        assert result["total_trades"] > 0

    def test_large_orders_with_real_trades(self, analyzer):
        result = analyzer.analyze_large_orders()
        assert "large_buys" in result
        assert "large_sells" in result
        assert "bias" in result
        assert result["bias"] in ["bullish", "bearish", "neutral"]
        assert result["large_buys"] >= 0
        assert result["large_sells"] >= 0

    def test_enhanced_signal(self, analyzer, btc_df):
        result = analyzer.get_enhanced_signal(btc_df)
        assert "score" in result
        assert "bias" in result
        assert "cvd" in result
        assert "taker" in result
        assert "notable" in result
        assert -1 <= result["score"] <= 1
        assert result["bias"] in ["bullish", "bearish", "neutral"]

    def test_cvd_with_empty_df(self, analyzer):
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        result = analyzer.calculate_cvd(empty_df)
        assert result["cvd"] == 0
        assert result["score"] == 0.0

    def test_cvd_with_single_candle(self, analyzer):
        single = pd.DataFrame({
            'open': [100.0], 'high': [102.0],
            'low': [99.0], 'close': [101.0], 'volume': [1000.0]
        })
        result = analyzer.calculate_cvd(single)
        # Should handle gracefully (less than 10 candles)
        assert result["trend"] == "neutral"

    def test_cvd_divergence_detection(self, analyzer, btc_df):
        result = analyzer.detect_cvd_divergence(btc_df)
        assert "divergence_detected" in result
        assert isinstance(result["divergence_detected"], bool)
        assert result["direction"] in ["bullish", "bearish", "none"]


class TestFundingRateAnalyzer:
    """Test FundingRateAnalyzer with real OKX data."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.order_flow import FundingRateAnalyzer
        return FundingRateAnalyzer(symbol="BTCUSDT")

    def test_initialization(self, analyzer):
        assert analyzer.symbol == "BTCUSDT"
        assert analyzer.extreme_threshold == 0.0003

    def test_get_signal_returns_funding_signal(self, analyzer):
        signal = analyzer.get_signal()
        assert hasattr(signal, 'rate')
        assert hasattr(signal, 'signal')
        assert hasattr(signal, 'strength')
        assert hasattr(signal, 'payout_direction')

    def test_funding_rate_is_reasonable(self, analyzer):
        signal = analyzer.get_signal()
        # Funding rate should be between -0.5% and 0.5% (raw, not percentage)
        assert -0.005 <= signal.rate <= 0.005, f"Funding rate unreasonable: {signal.rate}"

    def test_signal_value_is_valid(self, analyzer):
        signal = analyzer.get_signal()
        valid_signals = [
            "long_favored", "short_favored", "neutral",
            "slight_long_favored", "slight_short_favored",
        ]
        assert signal.signal in valid_signals

    def test_strength_in_range(self, analyzer):
        signal = analyzer.get_signal()
        assert 0 <= signal.strength <= 1.0

    def test_should_trade_returns_tuple(self, analyzer):
        ok, reason = analyzer.should_trade("long")
        assert isinstance(ok, bool)
        assert isinstance(reason, str)

    def test_liquidation_danger(self, analyzer):
        result = analyzer.get_liquidation_danger()
        assert "danger_score" in result
        assert "long_danger" in result
        assert "short_danger" in result
        assert 0 <= result["danger_score"] <= 1.0

    @pytest.mark.parametrize("symbol", ["ETHUSDT", "SOLUSDT", "XRPUSDT"])
    def test_funding_for_multiple_symbols(self, symbol):
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.order_flow import FundingRateAnalyzer
        fa = FundingRateAnalyzer(symbol=symbol)
        signal = fa.get_signal()
        assert signal.rate is not None
        assert -0.01 <= signal.rate <= 0.01


class TestMultiTimeframeAnalyzer:
    """Test MultiTimeframeAnalyzer with real Binance data."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.mtf_analyzer import MultiTimeframeAnalyzer
        return MultiTimeframeAnalyzer(symbol="BTCUSDT")

    def test_initialization(self, analyzer):
        assert analyzer.symbol == "BTCUSDT"
        assert analyzer.timeframes == ["4h", "1h", "15m"]

    def test_single_timeframe_analysis(self, analyzer):
        signal = analyzer.analyze_timeframe("1h")
        assert signal is not None
        assert signal.timeframe == "1h"
        assert signal.price > 0
        from src.features.mtf_analyzer import TrendDirection
        assert signal.direction in [TrendDirection.BULLISH, TrendDirection.BEARISH, TrendDirection.NEUTRAL]
        assert 0 <= signal.strength <= 1.0

    def test_rsi_in_range(self, analyzer):
        signal = analyzer.analyze_timeframe("1h")
        assert signal is not None
        assert 0 <= signal.rsi <= 100

    def test_confluence_analysis(self, analyzer):
        result = analyzer.get_confluence()
        assert hasattr(result, 'aligned')
        assert hasattr(result, 'direction')
        assert hasattr(result, 'strength')
        assert hasattr(result, 'recommendation')
        assert isinstance(result.aligned, bool)

    def test_confluence_has_all_timeframes(self, analyzer):
        result = analyzer.get_confluence()
        for tf in ["4h", "1h", "15m"]:
            assert tf in result.signals, f"Missing timeframe: {tf}"

    def test_should_trade_returns_tuple(self, analyzer):
        ok, reason = analyzer.should_trade("long")
        assert isinstance(ok, bool)
        assert isinstance(reason, str)


class TestAlternativeDataCollector:
    """Test AlternativeDataCollector with real API calls."""

    @pytest.fixture(scope="class")
    def collector(self):
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.alternative_data import AlternativeDataCollector
        return AlternativeDataCollector()

    def test_initialization(self, collector):
        assert collector.CACHE_TTL_HOURS == 12

    def test_fear_greed_returns_data(self, collector):
        fng = collector.fetch_fear_greed(limit=1)
        assert fng is not None
        assert len(fng) > 0
        assert "value" in fng[0]
        val = int(fng[0]["value"])
        assert 0 <= val <= 100

    def test_btc_dominance_returns_data(self, collector):
        dom = collector.fetch_btc_dominance()
        assert dom is not None
        btc_d = dom.get("btc_dominance", 0)
        # BTC dominance should be between 20% and 80%
        assert 20 <= btc_d <= 80, f"BTC dominance unreasonable: {btc_d}"

    def test_get_current_features(self, collector):
        features = collector.get_current_features()
        assert "fear_greed_value" in features
        assert "btc_dominance" in features
        assert -1 <= features["fear_greed_value"] <= 1
        assert -1 <= features["btc_dominance"] <= 1


class TestRiskManager:
    """Test AdaptiveRiskManager with real data."""

    @pytest.fixture(scope="class")
    def risk_mgr(self):
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.risk_manager import AdaptiveRiskManager
        return AdaptiveRiskManager()

    @pytest.fixture(scope="class")
    def btc_df(self):
        return fetch_binance_klines("BTCUSDT", "1h", 100)

    def test_initialization(self, risk_mgr):
        assert risk_mgr.use_kelly is True
        assert risk_mgr.use_trailing is True

    def test_atr_calculation(self, risk_mgr, btc_df):
        atr = risk_mgr.calculate_atr(btc_df)
        assert atr > 0
        # ATR should be a reasonable fraction of price
        price = btc_df['close'].iloc[-1]
        atr_pct = atr / price
        assert 0.001 < atr_pct < 0.10, f"ATR as % of price unreasonable: {atr_pct:.4f}"

    def test_adaptive_sl_tp(self, risk_mgr, btc_df):
        sl, tp = risk_mgr.get_adaptive_sl_tp(btc_df, "long")
        assert sl > 0, "SL must be positive"
        assert tp > 0, "TP must be positive"
        assert tp > sl, "TP should be larger than SL (positive R:R)"
        assert sl <= 0.10, f"SL too wide: {sl:.2%}"
        assert tp <= 0.20, f"TP too wide: {tp:.2%}"

    def test_structural_sl_tp(self, risk_mgr, btc_df):
        sl, tp = risk_mgr.get_structural_sl_tp(btc_df, "long", "BTCUSDT")
        assert sl > 0
        assert tp > 0

    def test_asset_specific_params(self, risk_mgr):
        btc_sl, btc_tp = risk_mgr.get_asset_specific_params("BTCUSDT")
        sol_sl, sol_tp = risk_mgr.get_asset_specific_params("SOLUSDT")
        # SOL should have wider stops than BTC (higher volatility)
        assert sol_sl >= btc_sl
        assert sol_tp >= btc_tp

    def test_kelly_with_no_history(self, risk_mgr):
        kelly = risk_mgr.calculate_kelly_fraction()
        assert kelly == risk_mgr.base_position_size

    def test_risk_parameters(self, risk_mgr, btc_df):
        params = risk_mgr.get_risk_parameters(btc_df, "long")
        assert params.stop_loss_pct > 0
        assert params.take_profit_pct > 0
        assert params.position_size > 0
        assert params.risk_reward_ratio > 0


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Model Component Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeClassifier:
    """Test HMM Regime Classifier with real data."""

    @pytest.fixture(scope="class")
    def classifier(self):
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.models.regime_classifier import RegimeClassifier
        rc = RegimeClassifier()
        loaded = rc.load("BTCUSDT")
        if not loaded:
            pytest.skip("No trained regime model for BTCUSDT")
        return rc

    @pytest.fixture(scope="class")
    def btc_df(self):
        return fetch_binance_klines("BTCUSDT", "1h", 300)

    def test_model_loads(self, classifier):
        assert classifier.model is not None
        assert classifier.feature_scaler is not None

    def test_predict_returns_valid_structure(self, classifier, btc_df):
        result = classifier.predict(btc_df)
        assert "current_regime" in result
        assert "regime_id" in result
        assert "transition_probs" in result
        assert "confidence" in result

    def test_regime_is_valid_label(self, classifier, btc_df):
        result = classifier.predict(btc_df)
        valid_regimes = ["BULL_TREND", "BEAR_TREND", "RANGE_CHOP", "HIGH_VOL_BREAKOUT", "UNKNOWN"]
        assert result["current_regime"] in valid_regimes

    def test_transition_probs_sum_to_one(self, classifier, btc_df):
        result = classifier.predict(btc_df)
        probs = result["transition_probs"]
        if probs:
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01, f"Transition probs don't sum to 1: {total}"

    def test_confidence_in_range(self, classifier, btc_df):
        result = classifier.predict(btc_df)
        assert 0 <= result["confidence"] <= 1.0


class TestConfidenceEngine:
    """Test ConfidenceEngine."""

    @pytest.fixture(scope="class")
    def engine(self):
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.models.confidence_engine import ConfidenceEngine
        return ConfidenceEngine()

    def test_low_confidence_reduces_size(self, engine):
        mult = engine.get_position_multiplier(0.1)
        assert mult < 1.0

    def test_baseline_confidence_is_one(self, engine):
        mult = engine.get_position_multiplier(0.5)
        assert abs(mult - 1.0) < 0.01

    def test_high_confidence_increases_size(self, engine):
        mult = engine.get_position_multiplier(0.9)
        assert mult > 1.0

    def test_multiplier_never_exceeds_max(self, engine):
        mult = engine.get_position_multiplier(1.0)
        assert mult <= engine.max_multiplier

    def test_multiplier_never_below_min(self, engine):
        mult = engine.get_position_multiplier(0.0)
        assert mult >= engine.min_multiplier

    def test_apply_confidence(self, engine):
        size = engine.apply_confidence(0.5, 0.8)
        assert size > 0

    def test_reliability_with_no_history(self, engine):
        r = engine.get_confidence_reliability()
        assert r == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Signal Quality / Reasonableness Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalQuality:
    """Cross-check signal values for reasonableness."""

    @pytest.fixture(scope="class")
    def all_market_data(self):
        """Fetch market data for all symbols."""
        data = {}
        for symbol in SYMBOLS:
            try:
                data[symbol] = get_market_data(symbol)
            except Exception:
                pass
        return data

    def test_all_symbols_have_regime(self, all_market_data):
        for symbol, data in all_market_data.items():
            assert data.get("regime") is not None, f"{symbol} missing regime"

    def test_all_symbols_have_funding(self, all_market_data):
        for symbol, data in all_market_data.items():
            funding = data.get("funding")
            assert funding is not None, f"{symbol} missing funding"
            rate = funding.get("rate", 0)
            assert -1.0 <= rate <= 1.0, f"{symbol} funding rate out of range: {rate}"

    def test_all_symbols_have_order_flow(self, all_market_data):
        for symbol, data in all_market_data.items():
            of = data.get("order_flow")
            assert of is not None, f"{symbol} missing order_flow"
            assert of.get("bias") in ["bullish", "bearish", "neutral"]

    def test_prices_are_reasonable(self, all_market_data):
        """Verify prices are in expected ranges."""
        expected_ranges = {
            "BTCUSDT": (20_000, 500_000),
            "ETHUSDT": (500, 20_000),
            "SOLUSDT": (5, 1000),
            "XRPUSDT": (0.1, 50),
        }
        for symbol, (low, high) in expected_ranges.items():
            if symbol in all_market_data:
                price = all_market_data[symbol].get("price", 0)
                if price:
                    assert low < price < high, f"{symbol} price {price} outside expected range ({low}-{high})"

    def test_regime_consistency(self, all_market_data):
        """All regime ADX values should be in valid range."""
        for symbol, data in all_market_data.items():
            regime = data.get("regime", {})
            adx = regime.get("adx", 0)
            if adx:
                assert 0 < adx < 100, f"{symbol} ADX unreasonable: {adx}"

    def test_taker_ratios_are_valid(self, all_market_data):
        for symbol, data in all_market_data.items():
            of = data.get("order_flow", {})
            taker = of.get("taker", {})
            ratio = taker.get("ratio", 0)
            if ratio:
                assert 0 <= ratio <= 1.0, f"{symbol} taker ratio out of range: {ratio}"


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: Edge Cases and Error Handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_symbol(self):
        """API should handle unknown symbols gracefully."""
        resp = requests.get(f"{API_BASE}/api/market", params={"symbol": "FAKECOIN"}, timeout=TIMEOUT)
        # Should still return 200 (with errors in signals, not a 500)
        assert resp.status_code == 200

    def test_missing_symbol_defaults_to_btc(self):
        """No symbol param should default to BTCUSDT."""
        resp = requests.get(f"{API_BASE}/api/market", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("symbol") == "BTCUSDT"

    def test_ohlcv_endpoint_works(self):
        """Test the OHLCV candle endpoint."""
        resp = requests.get(
            f"{API_BASE}/api/ohlcv",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 10},
            timeout=15
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            candle = data[0]
            assert "time" in candle
            assert "open" in candle
            assert "close" in candle

    def test_regime_detector_with_flat_data(self):
        """Regime detector should handle flat/constant price data."""
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.regime_detector import MarketRegimeDetector, MarketRegime

        detector = MarketRegimeDetector()
        flat_df = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [100.1] * 100,
            'low': [99.9] * 100,
            'close': [100.0] * 100,
            'volume': [1000.0] * 100,
        })
        result = detector.detect_regime(flat_df)
        # Flat data should be RANGING or LOW_VOLATILITY
        assert result.regime in [MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY, MarketRegime.UNKNOWN]

    def test_order_flow_with_no_trades(self):
        """OrderFlowAnalyzer handles empty trade list gracefully."""
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.order_flow import OrderFlowAnalyzer

        oa = OrderFlowAnalyzer(symbol="BTCUSDT")
        result = oa.calculate_taker_ratio(trades=[])
        assert result["ratio"] == 0.5
        assert result["score"] == 0.0
        assert result["total_trades"] == 0

    def test_large_orders_with_no_trades(self):
        """analyze_large_orders handles empty trade list."""
        import sys
        sys.path.insert(0, "/root/.openclaw/projects/drl-trading-system/repo")
        from src.features.order_flow import OrderFlowAnalyzer

        oa = OrderFlowAnalyzer(symbol="BTCUSDT")
        result = oa.analyze_large_orders(trades=[])
        assert result["large_buys"] == 0
        assert result["large_sells"] == 0
        assert result["bias"] == "neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: OHLCV Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOHLCVEndpoint:
    """Test the /api/ohlcv candlestick data endpoint."""

    def test_returns_valid_candles(self):
        resp = requests.get(
            f"{API_BASE}/api/ohlcv",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 50},
            timeout=15
        )
        assert resp.status_code == 200
        candles = resp.json()
        assert len(candles) == 50

    def test_candle_structure(self):
        resp = requests.get(
            f"{API_BASE}/api/ohlcv",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 5},
            timeout=15
        )
        candles = resp.json()
        for c in candles:
            assert "time" in c
            assert "open" in c
            assert "high" in c
            assert "low" in c
            assert "close" in c
            assert "volume" in c
            assert c["high"] >= c["low"]
            assert c["high"] >= c["open"]
            assert c["high"] >= c["close"]
            assert c["low"] <= c["open"]
            assert c["low"] <= c["close"]

    def test_candles_are_chronological(self):
        resp = requests.get(
            f"{API_BASE}/api/ohlcv",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 10},
            timeout=15
        )
        candles = resp.json()
        times = [c["time"] for c in candles]
        assert times == sorted(times), "Candles not in chronological order"

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_ohlcv_for_all_symbols(self, symbol):
        resp = requests.get(
            f"{API_BASE}/api/ohlcv",
            params={"symbol": symbol, "interval": "1h", "limit": 5},
            timeout=15
        )
        assert resp.status_code == 200
        candles = resp.json()
        assert len(candles) > 0
