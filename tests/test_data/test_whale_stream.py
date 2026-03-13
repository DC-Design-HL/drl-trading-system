"""
Tests for BinanceWhaleStream

Test Coverage:
- Initialization and configuration
- WebSocket connection management
- Trade filtering and classification
- Metrics calculation
- Window-based data cleanup
- Error handling
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from collections import deque
from src.data.whale_stream import BinanceWhaleStream


@pytest.mark.unit
class TestBinanceWhaleStream:
    """Test suite for BinanceWhaleStream."""

    def test_initialization(self):
        """Test whale stream initializes correctly."""
        stream = BinanceWhaleStream(
            symbol="BTCUSDT",
            min_value_usd=100000.0,
            window_seconds=60
        )

        assert stream.symbol == "btcusdt"  # Lowercased
        assert stream.min_value_usd == 100000.0
        assert stream.window_seconds == 60
        assert stream.running is False
        assert isinstance(stream.trades, deque)
        assert stream.metrics['buy_vol'] == 0.0
        assert stream.metrics['sell_vol'] == 0.0

    def test_initialization_defaults(self):
        """Test default initialization parameters."""
        stream = BinanceWhaleStream()

        assert stream.symbol == "btcusdt"
        assert stream.min_value_usd == 100000.0
        assert stream.window_seconds == 60

    @patch('src.data.whale_stream.websocket')
    @patch('requests.get')
    def test_calculate_dynamic_threshold(self, mock_get, mock_websocket):
        """Test dynamic threshold calculation."""
        # Mock 24h volume response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'quoteVolume': '1000000000.0'  # 1 billion USD volume
        }
        mock_get.return_value = mock_response

        stream = BinanceWhaleStream(min_value_usd=50000.0)
        stream._calculate_dynamic_threshold()

        # Dynamic threshold should be 0.0001 of 24h volume = 100k
        assert stream.min_value_usd >= 100000.0

    @patch('src.data.whale_stream.websocket')
    @patch('requests.get')
    def test_calculate_dynamic_threshold_api_error(self, mock_get, mock_websocket):
        """Test dynamic threshold calculation when API fails."""
        mock_get.side_effect = Exception("API Error")

        stream = BinanceWhaleStream(min_value_usd=50000.0)
        stream._calculate_dynamic_threshold()

        # Should keep original threshold
        assert stream.min_value_usd == 50000.0

    @patch('src.data.whale_stream.websocket')
    def test_start_without_websocket_module(self, mock_websocket):
        """Test start behavior when websocket module is missing."""
        # Simulate missing websocket module
        with patch('src.data.whale_stream.websocket', None):
            stream = BinanceWhaleStream()
            stream.start()

            # Should not start
            assert stream.running is False

    @patch('src.data.whale_stream.websocket')
    @patch('requests.get')
    def test_start_creates_websocket(self, mock_get, mock_websocket):
        """Test that start creates WebSocket connection."""
        # Mock 24h volume
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'quoteVolume': '1000000000.0'}
        mock_get.return_value = mock_response

        # Mock WebSocketApp
        mock_ws = MagicMock()
        mock_websocket.WebSocketApp = MagicMock(return_value=mock_ws)

        stream = BinanceWhaleStream()
        stream.start()

        # Verify WebSocket created
        assert stream.running is True
        assert mock_websocket.WebSocketApp.called

    def test_on_message_whale_buy(self):
        """Test processing whale buy trade."""
        stream = BinanceWhaleStream(min_value_usd=100000.0)

        # Simulate whale buy message
        message = json.dumps({
            'e': 'aggTrade',
            'p': '50000.0',    # Price
            'q': '3.0',        # Quantity (150k USD)
            'T': int(time.time() * 1000),
            'm': False         # Buyer is taker = BUY
        })

        stream._on_message(None, message)

        # Check metrics
        assert len(stream.trades) == 1
        assert stream.metrics['buy_vol'] > 0
        assert stream.metrics['buy_count'] == 1
        assert stream.metrics['sell_count'] == 0
        assert stream.metrics['last_whale_trade'] is not None
        assert stream.metrics['last_whale_trade']['side'] == 'BUY'

    def test_on_message_whale_sell(self):
        """Test processing whale sell trade."""
        stream = BinanceWhaleStream(min_value_usd=100000.0)

        # Simulate whale sell message
        message = json.dumps({
            'e': 'aggTrade',
            'p': '50000.0',    # Price
            'q': '3.0',        # Quantity (150k USD)
            'T': int(time.time() * 1000),
            'm': True          # Buyer is maker = SELL
        })

        stream._on_message(None, message)

        # Check metrics
        assert len(stream.trades) == 1
        assert stream.metrics['sell_vol'] > 0
        assert stream.metrics['sell_count'] == 1
        assert stream.metrics['buy_count'] == 0
        assert stream.metrics['last_whale_trade']['side'] == 'SELL'

    def test_on_message_small_trade_filtered(self):
        """Test that small trades are filtered out."""
        stream = BinanceWhaleStream(min_value_usd=100000.0)

        # Simulate small trade (only 5k USD)
        message = json.dumps({
            'e': 'aggTrade',
            'p': '50000.0',
            'q': '0.1',        # Only 5k USD
            'T': int(time.time() * 1000),
            'm': False
        })

        stream._on_message(None, message)

        # Should be filtered
        assert len(stream.trades) == 0
        assert stream.metrics['buy_count'] == 0

    def test_on_message_invalid_json(self):
        """Test handling of invalid JSON message."""
        stream = BinanceWhaleStream()

        # Should not crash
        stream._on_message(None, "invalid json")

        assert len(stream.trades) == 0

    def test_cleanup_old_trades(self):
        """Test cleanup of old trades outside window."""
        stream = BinanceWhaleStream(window_seconds=60)

        # Add old trade (2 minutes ago)
        old_time = time.time() - 120
        stream.trades.append({
            'time': old_time,
            'value': 100000.0,
            'side': 'BUY',
            'price': 50000.0
        })

        # Add recent trade
        recent_time = time.time() - 30
        stream.trades.append({
            'time': recent_time,
            'value': 100000.0,
            'side': 'BUY',
            'price': 50000.0
        })

        # Cleanup
        stream._cleanup_old_trades()

        # Only recent trade should remain
        assert len(stream.trades) == 1
        assert stream.trades[0]['time'] == recent_time

    def test_update_metrics(self):
        """Test metrics calculation."""
        stream = BinanceWhaleStream()

        # Add multiple trades
        current_time = time.time()
        stream.trades.append({
            'time': current_time,
            'value': 100000.0,
            'side': 'BUY',
            'price': 50000.0
        })
        stream.trades.append({
            'time': current_time,
            'value': 150000.0,
            'side': 'BUY',
            'price': 50000.0
        })
        stream.trades.append({
            'time': current_time,
            'value': 80000.0,
            'side': 'SELL',
            'price': 50000.0
        })

        stream._update_metrics()

        # Check calculations
        assert stream.metrics['buy_vol'] == 250000.0
        assert stream.metrics['sell_vol'] == 80000.0
        assert stream.metrics['net_flow'] == 170000.0  # Buy - Sell
        assert stream.metrics['buy_count'] == 2
        assert stream.metrics['sell_count'] == 1

    def test_get_metrics(self):
        """Test getting current metrics."""
        stream = BinanceWhaleStream()

        # Add a trade
        current_time = time.time()
        stream.trades.append({
            'time': current_time,
            'value': 100000.0,
            'side': 'BUY',
            'price': 50000.0
        })
        stream._update_metrics()

        # Get metrics
        metrics = stream.get_metrics()

        assert isinstance(metrics, dict)
        assert 'buy_vol' in metrics
        assert 'sell_vol' in metrics
        assert 'net_flow' in metrics
        assert 'buy_count' in metrics
        assert 'sell_count' in metrics
        assert 'last_whale_trade' in metrics

    def test_get_metrics_includes_cleanup(self):
        """Test that get_metrics triggers cleanup."""
        stream = BinanceWhaleStream(window_seconds=60)

        # Add old trade (2 minutes ago, outside window)
        old_time = time.time() - 120
        stream.trades.append({
            'time': old_time,
            'value': 100000.0,
            'side': 'BUY',
            'price': 50000.0
        })

        # Get metrics (should trigger cleanup and update metrics)
        metrics = stream.get_metrics()

        # Old trade should be cleaned after get_metrics call
        assert len(stream.trades) == 0
        # Metrics should reflect empty state (after cleanup)
        assert metrics['buy_vol'] == 0.0

    def test_stop(self):
        """Test stopping the stream."""
        stream = BinanceWhaleStream()
        stream.running = True

        # Mock WebSocket
        stream.ws = MagicMock()
        stream.wst = MagicMock()

        stream.stop()

        assert stream.running is False
        stream.ws.close.assert_called_once()

    def test_net_flow_calculation(self):
        """Test net flow calculation with mixed trades."""
        stream = BinanceWhaleStream()

        current_time = time.time()

        # Scenario: More buying pressure
        stream.trades.extend([
            {'time': current_time, 'value': 200000.0, 'side': 'BUY', 'price': 50000.0},
            {'time': current_time, 'value': 300000.0, 'side': 'BUY', 'price': 50000.0},
            {'time': current_time, 'value': 100000.0, 'side': 'SELL', 'price': 50000.0},
        ])

        stream._update_metrics()

        # Net flow should be positive (bullish)
        assert stream.metrics['net_flow'] == 400000.0  # 500k buy - 100k sell
        assert stream.metrics['buy_vol'] > stream.metrics['sell_vol']

    def test_multiple_symbols(self):
        """Test that different streams handle different symbols."""
        btc_stream = BinanceWhaleStream(symbol="BTCUSDT")
        eth_stream = BinanceWhaleStream(symbol="ETHUSDT")

        assert btc_stream.symbol == "btcusdt"
        assert eth_stream.symbol == "ethusdt"
        assert btc_stream.trades is not eth_stream.trades

    @patch('src.data.whale_stream.websocket')
    @patch('requests.get')
    def test_on_open_callback(self, mock_get, mock_websocket):
        """Test WebSocket on_open callback."""
        stream = BinanceWhaleStream()

        # Should not raise
        stream._on_open(None)

    @patch('src.data.whale_stream.websocket')
    def test_on_close_callback(self, mock_websocket):
        """Test WebSocket on_close callback."""
        stream = BinanceWhaleStream()
        stream.running = False

        # Should not raise
        stream._on_close(None, None, None)

    @patch('src.data.whale_stream.websocket')
    def test_on_error_callback(self, mock_websocket):
        """Test WebSocket on_error callback."""
        stream = BinanceWhaleStream()

        # Should not raise
        stream._on_error(None, "Test error")

    def test_thread_safety(self):
        """Test that lock protects concurrent access."""
        stream = BinanceWhaleStream()

        # Simulate concurrent access
        import threading

        def add_trade():
            current_time = time.time()
            stream.trades.append({
                'time': current_time,
                'value': 100000.0,
                'side': 'BUY',
                'price': 50000.0
            })
            stream._update_metrics()

        threads = [threading.Thread(target=add_trade) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 10 trades
        assert len(stream.trades) == 10
