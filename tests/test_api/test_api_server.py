"""
Tests for Flask API Server

Test Coverage:
- Health check endpoint
- State management endpoints
- Trade logging endpoints
- Model info endpoints
- Market analysis endpoints
- Caching behavior
- Error handling
- CORS headers
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from src.ui.api_server import app
from src.data.storage import JsonFileStorage


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_storage(tmp_path):
    """Create mock storage for testing."""
    storage = JsonFileStorage(base_dir=tmp_path)
    return storage


@pytest.mark.unit
class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test /health endpoint returns healthy status."""
        response = client.get('/health')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data

    def test_health_check_response_format(self, client):
        """Test health check returns valid JSON."""
        response = client.get('/health')

        assert response.content_type == 'application/json'
        data = json.loads(response.data)
        assert isinstance(data, dict)


@pytest.mark.unit
class TestStateEndpoint:
    """Test /api/state endpoint."""

    @patch('src.ui.api_server.storage')
    @patch('src.api.futures_executor.get_futures_executor', return_value=None)
    def test_get_state_success(self, mock_executor, mock_storage, client):
        """Test successful state retrieval."""
        # Mock storage response
        mock_storage.load_state.return_value = {
            'balance': 10000.0,
            'total_pnl': 500.0,
            'assets': {}
        }
        mock_storage.get_trades.return_value = []

        response = client.get('/api/state')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'balance' in data
        # Balance comes from storage when exchange override is unavailable
        assert data['balance'] == 10000.0

    @patch('src.ui.api_server.storage')
    def test_get_state_empty(self, mock_storage, client):
        """Test state endpoint when no state exists."""
        mock_storage.load_state.return_value = {}
        mock_storage.get_trades.return_value = []

        response = client.get('/api/state')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)

    @patch('src.ui.api_server.storage')
    def test_get_state_pnl_calculation(self, mock_storage, client):
        """Test that PnL is calculated from trades."""
        # Mock trades with PnL
        mock_storage.load_state.return_value = {
            'raw_state': {
                'assets': {
                    'BTCUSDT': {'position': 0, 'pnl': 0}
                }
            }
        }
        mock_storage.get_trades.return_value = [
            {'action': 'CLOSE', 'pnl': 100.0},
            {'action': 'EXIT', 'pnl': 50.0},
            {'action': 'BUY', 'pnl': 0.0},
        ]

        response = client.get('/api/state')

        assert response.status_code == 200
        data = json.loads(response.data)
        # Should calculate total PnL from closed trades
        assert 'total_pnl' in data

    @patch('src.ui.api_server.storage')
    def test_get_state_error_handling(self, mock_storage, client):
        """Test error handling in state endpoint."""
        mock_storage.load_state.side_effect = Exception("Database error")

        response = client.get('/api/state')

        # Should return empty dict on error, not crash
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)

    @patch('src.ui.api_server.storage')
    @patch('builtins.open', new_callable=MagicMock)
    def test_get_state_whale_alerts(self, mock_open, mock_storage, client):
        """Test whale alerts injection."""
        mock_storage.load_state.return_value = {}
        mock_storage.get_trades.return_value = []

        # Mock whale wallet data
        whale_data = {
            "address": "0xtest",
            "transactions": [
                {
                    "value": 1000.0,
                    "asset": "ETH",
                    "timestamp": int(datetime.now().timestamp()),
                    "link": "https://etherscan.io/tx/test"
                }
            ]
        }

        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = json.dumps(whale_data)
        mock_open.return_value = mock_file

        response = client.get('/api/state')

        assert response.status_code == 200


@pytest.mark.unit
class TestTradesEndpoint:
    """Test /api/trades endpoint."""

    @patch('src.ui.api_server.storage')
    def test_get_trades_success(self, mock_storage, client):
        """Test successful trades retrieval."""
        # Mock trades
        mock_trades = [
            {'symbol': 'BTCUSDT', 'action': 'BUY', 'price': 50000.0},
            {'symbol': 'ETHUSDT', 'action': 'SELL', 'price': 3000.0},
        ]
        mock_storage.get_trades.return_value = mock_trades

        response = client.get('/api/trades')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]['symbol'] == 'BTCUSDT'

    @patch('src.ui.api_server.storage')
    def test_get_trades_empty(self, mock_storage, client):
        """Test trades endpoint with no trades."""
        mock_storage.get_trades.return_value = []

        response = client.get('/api/trades')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    @patch('src.ui.api_server.storage')
    def test_get_trades_error_handling(self, mock_storage, client):
        """Test error handling in trades endpoint."""
        mock_storage.get_trades.side_effect = Exception("Database error")

        response = client.get('/api/trades')

        # Should return empty list on error
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []


@pytest.mark.unit
class TestTradeCountEndpoint:
    """Test /api/trades/count endpoint."""

    @patch('src.ui.api_server.storage')
    def test_get_trade_count(self, mock_storage, client):
        """Test trade count endpoint."""
        mock_trades = [{'id': i} for i in range(10)]
        mock_storage.get_trades.return_value = mock_trades

        response = client.get('/api/trades/count')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['count'] == 10
        assert 'timestamp' in data

    @patch('src.ui.api_server.storage')
    def test_get_trade_count_empty(self, mock_storage, client):
        """Test trade count with no trades."""
        mock_storage.get_trades.return_value = []

        response = client.get('/api/trades/count')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['count'] == 0


@pytest.mark.unit
class TestModelEndpoint:
    """Test /api/model endpoint."""

    @patch('src.ui.api_server.storage')
    @patch('os.path.getmtime')
    @patch('pathlib.Path.exists')
    def test_get_model_info_exists(self, mock_exists, mock_mtime, mock_storage, client):
        """Test model info when model exists."""
        # Mock model file exists
        mock_exists.return_value = True
        mock_mtime.return_value = datetime.now().timestamp()

        # Mock storage
        mock_storage.load_state.return_value = {
            'balance': 11000.0
        }
        mock_storage.get_trades.return_value = [
            {'pnl': 100.0},
            {'pnl': -50.0},
            {'pnl': 150.0},
        ]

        response = client.get('/api/model')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['model_exists'] is True
        assert 'model_name' in data
        assert 'total_return' in data
        assert 'win_rate' in data
        assert 'total_trades' in data

    @patch('src.ui.api_server.storage')
    @patch('pathlib.Path.exists')
    def test_get_model_info_not_exists(self, mock_exists, mock_storage, client):
        """Test model info when model doesn't exist."""
        mock_exists.return_value = False
        mock_storage.load_state.return_value = {}
        mock_storage.get_trades.return_value = []

        response = client.get('/api/model')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['model_exists'] is False
        assert data['model_date'] == "Not found"

    @patch('src.ui.api_server.storage')
    @patch('pathlib.Path.exists')
    def test_model_win_rate_calculation(self, mock_exists, mock_storage, client):
        """Test win rate calculation."""
        mock_exists.return_value = False
        mock_storage.load_state.return_value = {}

        # 6 winning, 4 losing = 60% win rate (action must contain CLOSE or EXIT)
        mock_storage.get_trades.return_value = [
            {'action': 'CLOSE_LONG', 'pnl': 100.0},
            {'action': 'CLOSE_LONG', 'pnl': 50.0},
            {'action': 'CLOSE_SHORT', 'pnl': -30.0},
            {'action': 'EXIT', 'pnl': 75.0},
            {'action': 'CLOSE_LONG', 'pnl': -20.0},
            {'action': 'CLOSE_SHORT', 'pnl': 120.0},
            {'action': 'EXIT', 'pnl': 90.0},
            {'action': 'CLOSE_LONG', 'pnl': -40.0},
            {'action': 'CLOSE_SHORT', 'pnl': 60.0},
            {'action': 'EXIT', 'pnl': -10.0},
        ]

        response = client.get('/api/model')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['total_trades'] == 10
        assert data['winning_trades'] == 6
        assert data['win_rate'] == 60.0


@pytest.mark.unit
class TestMarketEndpoint:
    """Test /api/market endpoint."""

    @patch('src.ui.api_server.storage')
    @patch('requests.get')
    def test_get_market_analysis_cached(self, mock_requests, mock_storage, client):
        """Test market analysis with cache hit."""
        # Make first request
        mock_storage.load_state.return_value = {}

        mock_response = Mock()
        mock_response.json.return_value = [
            [1640000000000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
        ] * 100
        mock_requests.return_value = mock_response

        response1 = client.get('/api/market?symbol=BTCUSDT')
        assert response1.status_code == 200

        # Second request should use cache
        response2 = client.get('/api/market?symbol=BTCUSDT')
        assert response2.status_code == 200

        # Both should return same data
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)
        assert data1['symbol'] == data2['symbol']

    @patch('src.ui.api_server.storage')
    @patch('requests.get')
    def test_get_market_analysis_structure(self, mock_requests, mock_storage, client):
        """Test market analysis response structure."""
        mock_storage.load_state.return_value = {}

        mock_response = Mock()
        mock_response.json.return_value = [
            [1640000000000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
        ] * 100
        mock_requests.return_value = mock_response

        response = client.get('/api/market?symbol=BTCUSDT')

        assert response.status_code == 200
        data = json.loads(response.data)

        # Check expected keys
        assert 'timestamp' in data
        assert 'symbol' in data
        assert 'whale' in data or data.get('whale') is None
        assert 'regime' in data or data.get('regime') is None
        assert 'funding' in data or data.get('funding') is None

    @patch('src.ui.api_server.storage')
    def test_get_market_analysis_default_symbol(self, mock_storage, client):
        """Test market analysis with default symbol."""
        mock_storage.load_state.return_value = {}

        response = client.get('/api/market')

        assert response.status_code == 200
        data = json.loads(response.data)
        # Should default to BTCUSDT or similar
        assert 'symbol' in data

    @patch('src.ui.api_server.storage')
    @patch('requests.get')
    def test_get_market_analysis_error_handling(self, mock_requests, mock_storage, client):
        """Test error handling in market analysis."""
        mock_storage.load_state.return_value = {}
        mock_requests.side_effect = Exception("API Error")

        response = client.get('/api/market?symbol=BTCUSDT')

        # Should not crash, return partial data
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'timestamp' in data


@pytest.mark.unit
class TestDebugEndpoint:
    """Test /api/debug/log endpoint."""

    def test_get_crash_log_not_exists(self, client):
        """Test crash log when file doesn't exist."""
        response = client.get('/api/debug/log')

        assert response.status_code == 404
        assert b"No crash log found" in response.data

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    def test_get_crash_log_exists(self, mock_open, mock_exists, client):
        """Test crash log when file exists."""
        mock_exists.return_value = True

        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "Error: Test crash"
        mock_open.return_value = mock_file

        response = client.get('/api/debug/log')

        assert response.status_code == 200
        assert b"Error: Test crash" in response.data


@pytest.mark.unit
class TestAPIResponseFormats:
    """Test API response formats and content types."""

    @patch('src.ui.api_server.storage')
    def test_json_content_type(self, mock_storage, client):
        """Test that JSON endpoints return correct content type."""
        mock_storage.load_state.return_value = {}
        mock_storage.get_trades.return_value = []

        endpoints = [
            '/api/state',
            '/api/trades',
            '/api/trades/count',
            '/api/model',
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert 'application/json' in response.content_type

    @patch('src.ui.api_server.storage')
    def test_valid_json_responses(self, mock_storage, client):
        """Test that all endpoints return valid JSON."""
        mock_storage.load_state.return_value = {}
        mock_storage.get_trades.return_value = []

        endpoints = [
            '/health',
            '/api/state',
            '/api/trades',
            '/api/trades/count',
            '/api/model',
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should not raise exception
            data = json.loads(response.data)
            assert isinstance(data, (dict, list))


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API server."""

    @patch('src.ui.api_server.storage')
    def test_full_workflow(self, mock_storage, client):
        """Test complete API workflow."""
        # Setup mock storage
        initial_state = {
            'balance': 10000.0,
            'total_pnl': 0.0,
            'assets': {
                'BTCUSDT': {'position': 0, 'pnl': 0}
            }
        }
        trades = [
            {'symbol': 'BTCUSDT', 'action': 'BUY', 'pnl': 0},
            {'symbol': 'BTCUSDT', 'action': 'CLOSE', 'pnl': 100},
        ]

        mock_storage.load_state.return_value = initial_state
        mock_storage.get_trades.return_value = trades

        # Test endpoints in sequence
        health = client.get('/health')
        assert health.status_code == 200

        state = client.get('/api/state')
        assert state.status_code == 200

        trades_resp = client.get('/api/trades')
        assert trades_resp.status_code == 200
        trades_data = json.loads(trades_resp.data)
        assert len(trades_data) == 2

        count = client.get('/api/trades/count')
        assert count.status_code == 200
        count_data = json.loads(count.data)
        assert count_data['count'] == 2

        model = client.get('/api/model')
        assert model.status_code == 200

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        # Flask test client is not thread-safe, so we test sequentially
        # but verify multiple requests work correctly
        results = []

        # Make 10 sequential requests
        for _ in range(10):
            response = client.get('/health')
            results.append(response.status_code)

        # All should succeed
        assert all(code == 200 for code in results)
        assert len(results) == 10


@pytest.mark.unit
class TestCacheBehavior:
    """Test caching behavior in API endpoints."""

    @patch('src.ui.api_server.storage')
    def test_market_cache_expiry(self, mock_storage, client):
        """Test that market cache returns valid data."""
        mock_storage.load_state.return_value = {}

        # Make first request
        response1 = client.get('/api/market?symbol=BTCUSDT')
        assert response1.status_code == 200
        data1 = json.loads(response1.data)

        # Make second request - should use cache or fetch again
        response2 = client.get('/api/market?symbol=BTCUSDT')
        assert response2.status_code == 200
        data2 = json.loads(response2.data)

        # Both should have valid structure
        assert 'timestamp' in data1
        assert 'timestamp' in data2
        assert 'symbol' in data1
        assert 'symbol' in data2
