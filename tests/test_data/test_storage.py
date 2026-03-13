"""
Tests for Storage Backends

Test Coverage:
- JSON file storage operations
- MongoDB storage operations
- Storage interface compliance
- Factory function behavior
- Error handling
- Data persistence
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from src.data.storage import (
    StorageInterface,
    JsonFileStorage,
    MongoStorage,
    get_storage,
)


@pytest.mark.unit
class TestJsonFileStorage:
    """Test suite for JSON file storage backend."""

    def test_initialization(self, tmp_path):
        """Test JsonFileStorage initialization."""
        storage = JsonFileStorage(base_dir=tmp_path)

        assert storage.base_dir == tmp_path
        assert storage.state_file == tmp_path / "multi_asset_state.json"
        assert storage.trade_file == tmp_path / "trading_log.json"
        assert storage.base_dir.exists()

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state."""
        storage = JsonFileStorage(base_dir=tmp_path)

        # Save state
        test_state = {
            "balance": 10000.0,
            "position": 0,
            "assets": {
                "BTCUSDT": {
                    "balance": 5000.0,
                    "position": 0.1,
                }
            }
        }
        storage.save_state(test_state)

        # Load state
        loaded_state = storage.load_state()

        assert loaded_state == test_state
        assert loaded_state["balance"] == 10000.0
        assert loaded_state["assets"]["BTCUSDT"]["position"] == 0.1

    def test_load_state_nonexistent_file(self, tmp_path):
        """Test loading state when file doesn't exist."""
        storage = JsonFileStorage(base_dir=tmp_path)

        # Should return empty dict
        state = storage.load_state()
        assert state == {}

    def test_log_trade(self, tmp_path):
        """Test logging a trade."""
        storage = JsonFileStorage(base_dir=tmp_path)

        # Log a trade
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "BTCUSDT",
            "action": "BUY",
            "price": 50000.0,
            "quantity": 0.1,
            "pnl": 0.0,
        }
        storage.log_trade(trade)

        # Verify file exists
        assert storage.trade_file.exists()

        # Read and verify
        with open(storage.trade_file, 'r') as f:
            line = f.readline()
            logged_trade = json.loads(line)
            assert logged_trade == trade

    def test_log_multiple_trades(self, tmp_path):
        """Test logging multiple trades."""
        storage = JsonFileStorage(base_dir=tmp_path)

        # Log multiple trades
        trades = [
            {"symbol": "BTCUSDT", "action": "BUY", "price": 50000.0},
            {"symbol": "ETHUSDT", "action": "SELL", "price": 3000.0},
            {"symbol": "SOLUSDT", "action": "BUY", "price": 100.0},
        ]

        for trade in trades:
            storage.log_trade(trade)

        # Get trades
        loaded_trades = storage.get_trades(limit=10)

        assert len(loaded_trades) == 3
        assert loaded_trades[0]["symbol"] == "BTCUSDT"
        assert loaded_trades[1]["symbol"] == "ETHUSDT"
        assert loaded_trades[2]["symbol"] == "SOLUSDT"

    def test_get_trades_with_limit(self, tmp_path):
        """Test getting trades with limit."""
        storage = JsonFileStorage(base_dir=tmp_path)

        # Log 10 trades
        for i in range(10):
            trade = {"id": i, "symbol": "BTCUSDT"}
            storage.log_trade(trade)

        # Get last 5
        trades = storage.get_trades(limit=5)

        assert len(trades) == 5
        # Should get the last 5 trades (5-9)
        assert trades[0]["id"] == 5
        assert trades[-1]["id"] == 9

    def test_get_trades_empty_file(self, tmp_path):
        """Test getting trades when file doesn't exist."""
        storage = JsonFileStorage(base_dir=tmp_path)

        trades = storage.get_trades()
        assert trades == []

    def test_save_state_error_handling(self, tmp_path):
        """Test error handling when saving state fails."""
        storage = JsonFileStorage(base_dir=tmp_path)

        # Create a state with non-serializable object
        bad_state = {
            "value": object()  # Can't be JSON serialized
        }

        # Should not raise, just log error
        storage.save_state(bad_state)

    def test_concurrent_access(self, tmp_path):
        """Test multiple storage instances can coexist."""
        storage1 = JsonFileStorage(base_dir=tmp_path)
        storage2 = JsonFileStorage(base_dir=tmp_path)

        # Save from storage1
        state1 = {"balance": 5000.0}
        storage1.save_state(state1)

        # Load from storage2
        state2 = storage2.load_state()

        assert state1 == state2


@pytest.mark.unit
class TestMongoStorage:
    """Test suite for MongoDB storage backend."""

    @patch('pymongo.MongoClient')
    def test_initialization_success(self, mock_client):
        """Test MongoDB initialization with valid connection."""
        # Mock MongoDB client
        mock_db = MagicMock()
        mock_client.return_value.get_database.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        with patch.dict('os.environ', {'MONGO_URI': 'mongodb://test:27017'}):
            storage = MongoStorage()

            assert storage.uri == 'mongodb://test:27017'
            assert storage.db is not None
            mock_client.return_value.admin.command.assert_called_with('ping')

    def test_initialization_no_uri(self):
        """Test MongoDB initialization without URI."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="MONGO_URI"):
                MongoStorage()

    @patch('pymongo.MongoClient')
    def test_save_state(self, mock_client):
        """Test saving state to MongoDB."""
        # Setup mock
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.get_collection.return_value = mock_collection
        mock_client.return_value.get_database.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        with patch.dict('os.environ', {'MONGO_URI': 'mongodb://test:27017'}):
            storage = MongoStorage()

            # Save state
            test_state = {"balance": 10000.0, "position": 0}
            storage.save_state(test_state)

            # Verify update_one was called
            mock_collection.update_one.assert_called_once()
            call_args = mock_collection.update_one.call_args
            assert call_args[0][0] == {"_id": "current_state"}
            assert call_args[1]["upsert"] is True

    @patch('pymongo.MongoClient')
    def test_load_state(self, mock_client):
        """Test loading state from MongoDB."""
        # Setup mock
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = {
            "_id": "current_state",
            "balance": 10000.0,
            "position": 0
        }
        mock_db.get_collection.return_value = mock_collection
        mock_client.return_value.get_database.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        with patch.dict('os.environ', {'MONGO_URI': 'mongodb://test:27017'}):
            storage = MongoStorage()

            # Load state
            state = storage.load_state()

            assert state == {"balance": 10000.0, "position": 0}
            assert "_id" not in state

    @patch('pymongo.MongoClient')
    def test_load_state_empty(self, mock_client):
        """Test loading state when no state exists."""
        # Setup mock
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        mock_db.get_collection.return_value = mock_collection
        mock_client.return_value.get_database.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        with patch.dict('os.environ', {'MONGO_URI': 'mongodb://test:27017'}):
            storage = MongoStorage()

            state = storage.load_state()
            assert state == {}

    @patch('pymongo.MongoClient')
    def test_log_trade(self, mock_client):
        """Test logging trade to MongoDB."""
        # Setup mock
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.get_collection.return_value = mock_collection
        mock_client.return_value.get_database.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        with patch.dict('os.environ', {'MONGO_URI': 'mongodb://test:27017'}):
            storage = MongoStorage()

            # Get trades collection
            trades_collection = storage.trades_collection

            # Log trade
            trade = {"symbol": "BTCUSDT", "action": "BUY"}
            storage.log_trade(trade)

            # Verify insert_one was called
            trades_collection.insert_one.assert_called_once_with(trade)

    @patch('pymongo.MongoClient')
    def test_get_trades(self, mock_client):
        """Test getting trades from MongoDB."""
        # Setup mock
        mock_db = MagicMock()
        mock_collection = MagicMock()

        # Mock cursor with trades
        mock_trades = [
            {"_id": "1", "symbol": "BTCUSDT", "timestamp": 100},
            {"_id": "2", "symbol": "ETHUSDT", "timestamp": 200},
        ]
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_trades))
        mock_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor

        mock_db.get_collection.return_value = mock_collection
        mock_client.return_value.get_database.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        with patch.dict('os.environ', {'MONGO_URI': 'mongodb://test:27017'}):
            storage = MongoStorage()

            trades = storage.get_trades(limit=10)

            # Verify _id removed
            assert len(trades) == 2
            for trade in trades:
                assert "_id" not in trade

    @patch('pymongo.MongoClient')
    def test_environment_based_database_selection(self, mock_client):
        """Test that database name changes based on environment."""
        mock_db = MagicMock()
        mock_client.return_value.get_database.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        # Test production
        with patch.dict('os.environ', {
            'MONGO_URI': 'mongodb://test:27017',
            'ENVIRONMENT': 'production'
        }):
            storage = MongoStorage()
            mock_client.return_value.get_database.assert_called_with("trading_system")

        # Reset mock
        mock_client.reset_mock()
        mock_client.return_value.get_database.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        # Test development
        with patch.dict('os.environ', {
            'MONGO_URI': 'mongodb://test:27017',
            'ENVIRONMENT': 'development'
        }):
            storage = MongoStorage()
            mock_client.return_value.get_database.assert_called_with("trading_system_development")


@pytest.mark.unit
class TestStorageFactory:
    """Test storage factory function."""

    @patch('src.data.storage.MongoStorage')
    def test_get_storage_mongo(self, mock_mongo):
        """Test getting MongoDB storage."""
        mock_instance = MagicMock()
        mock_mongo.return_value = mock_instance

        with patch.dict('os.environ', {'STORAGE_TYPE': 'mongo'}):
            storage = get_storage()

            mock_mongo.assert_called_once()

    def test_get_storage_json(self, tmp_path):
        """Test getting JSON storage."""
        with patch.dict('os.environ', {'STORAGE_TYPE': 'json'}):
            storage = get_storage()

            assert isinstance(storage, JsonFileStorage)

    @patch('src.data.storage.MongoStorage')
    def test_get_storage_mongo_fallback(self, mock_mongo):
        """Test fallback to JSON when MongoDB fails."""
        # Make MongoDB fail
        mock_mongo.side_effect = Exception("Connection failed")

        with patch.dict('os.environ', {'STORAGE_TYPE': 'mongo'}):
            storage = get_storage()

            # Should fallback to JSON
            assert isinstance(storage, JsonFileStorage)

    def test_get_storage_default(self):
        """Test default storage type."""
        with patch.dict('os.environ', {}, clear=True):
            storage = get_storage()

            # Default should be JSON
            assert isinstance(storage, JsonFileStorage)


@pytest.mark.unit
class TestStorageInterface:
    """Test that implementations comply with interface."""

    def test_json_implements_interface(self, tmp_path):
        """Test JsonFileStorage implements StorageInterface."""
        storage = JsonFileStorage(base_dir=tmp_path)

        assert isinstance(storage, StorageInterface)
        assert hasattr(storage, 'save_state')
        assert hasattr(storage, 'load_state')
        assert hasattr(storage, 'log_trade')
        assert hasattr(storage, 'get_trades')

    @patch('pymongo.MongoClient')
    def test_mongo_implements_interface(self, mock_client):
        """Test MongoStorage implements StorageInterface."""
        mock_db = MagicMock()
        mock_client.return_value.get_database.return_value = mock_db
        mock_client.return_value.admin.command.return_value = True

        with patch.dict('os.environ', {'MONGO_URI': 'mongodb://test:27017'}):
            storage = MongoStorage()

            assert isinstance(storage, StorageInterface)
            assert hasattr(storage, 'save_state')
            assert hasattr(storage, 'load_state')
            assert hasattr(storage, 'log_trade')
            assert hasattr(storage, 'get_trades')


@pytest.mark.integration
@pytest.mark.requires_api
class TestStorageIntegration:
    """Integration tests for storage (requires real MongoDB)."""

    def test_json_storage_full_workflow(self, tmp_path):
        """Test complete workflow with JSON storage."""
        storage = JsonFileStorage(base_dir=tmp_path)

        # Save initial state
        initial_state = {
            "balance": 10000.0,
            "total_pnl": 0.0,
            "assets": {}
        }
        storage.save_state(initial_state)

        # Log some trades
        trades = [
            {"action": "BUY", "symbol": "BTCUSDT", "pnl": 100.0},
            {"action": "SELL", "symbol": "BTCUSDT", "pnl": 50.0},
        ]
        for trade in trades:
            storage.log_trade(trade)

        # Update state
        updated_state = storage.load_state()
        updated_state["balance"] = 10150.0
        updated_state["total_pnl"] = 150.0
        storage.save_state(updated_state)

        # Verify final state
        final_state = storage.load_state()
        assert final_state["balance"] == 10150.0
        assert final_state["total_pnl"] == 150.0

        # Verify trades
        logged_trades = storage.get_trades()
        assert len(logged_trades) == 2
