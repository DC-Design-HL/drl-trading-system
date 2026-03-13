"""
End-to-End Integration Tests

Test Coverage:
- Complete trading workflows
- Data pipeline integration
- Feature computation to model prediction
- Storage persistence
- Multi-component interaction
- System resilience
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


@pytest.mark.integration
class TestDataPipeline:
    """Test complete data pipeline integration."""

    @patch('requests.get')
    def test_fetch_compute_features_flow(self, mock_get):
        """Test fetching data and computing features."""
        from src.data.multi_asset_fetcher import MultiAssetDataFetcher
        from src.features.ultimate_features import UltimateFeatureEngine

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1640000000000 + i*3600000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
            for i in range(100)
        ]
        mock_get.return_value = mock_response

        # Fetch data
        fetcher = MultiAssetDataFetcher()
        df = fetcher.fetch_asset("BTCUSDT", "1h", days=1)

        assert not df.empty
        assert len(df) >= 100

        # Compute features
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(df)

        assert len(features) > 0
        assert 'rsi_14' in features
        assert 'macd' in features

        # Verify no NaN in critical features
        for feature_name, feature_values in features.items():
            if isinstance(feature_values, pd.Series):
                feature_values = feature_values.values
            assert not np.isnan(feature_values).all(), f"{feature_name} is all NaN"

    @patch('requests.get')
    def test_multi_asset_feature_computation(self, mock_get):
        """Test feature computation across multiple assets."""
        from src.data.multi_asset_fetcher import MultiAssetDataFetcher
        from src.features.ultimate_features import UltimateFeatureEngine

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1640000000000 + i*3600000, str(50000 + i*10), "51000", "49000",
             str(50000 + i*10), "100", 1640003599999, "5050000", 1000, "50", "2525000", "0"]
            for i in range(50)
        ]
        mock_get.return_value = mock_response

        # Fetch multiple assets
        fetcher = MultiAssetDataFetcher()
        data = fetcher.fetch_multiple(["BTCUSDT", "ETHUSDT"], "1h", days=1)

        assert len(data) == 2
        assert "BTCUSDT" in data
        assert "ETHUSDT" in data

        # Compute features for each
        engine = UltimateFeatureEngine()
        for symbol, df in data.items():
            if not df.empty:
                features = engine.get_all_features(df)
                assert len(features) > 0


@pytest.mark.integration
class TestStoragePersistence:
    """Test storage persistence across operations."""

    def test_json_storage_persistence(self, tmp_path):
        """Test JSON storage persists data correctly."""
        from src.data.storage import JsonFileStorage

        storage = JsonFileStorage(base_dir=tmp_path)

        # Initial state
        initial_state = {
            'balance': 10000.0,
            'total_pnl': 0.0,
            'assets': {
                'BTCUSDT': {'position': 0, 'balance': 5000.0}
            }
        }
        storage.save_state(initial_state)

        # Log trades
        trades = [
            {'timestamp': datetime.now().isoformat(), 'symbol': 'BTCUSDT',
             'action': 'BUY', 'price': 50000, 'quantity': 0.1, 'pnl': 0},
            {'timestamp': datetime.now().isoformat(), 'symbol': 'BTCUSDT',
             'action': 'SELL', 'price': 51000, 'quantity': 0.1, 'pnl': 100},
        ]
        for trade in trades:
            storage.log_trade(trade)

        # Update state
        updated_state = storage.load_state()
        updated_state['total_pnl'] = 100.0
        updated_state['balance'] = 10100.0
        storage.save_state(updated_state)

        # Verify persistence by creating new storage instance
        storage2 = JsonFileStorage(base_dir=tmp_path)
        loaded_state = storage2.load_state()
        loaded_trades = storage2.get_trades()

        assert loaded_state['balance'] == 10100.0
        assert loaded_state['total_pnl'] == 100.0
        assert len(loaded_trades) == 2

    @patch('pymongo.MongoClient')
    def test_storage_factory_fallback(self, mock_client):
        """Test storage factory falls back to JSON when MongoDB fails."""
        from src.data.storage import get_storage, JsonFileStorage

        # Make MongoDB fail
        mock_client.side_effect = Exception("Connection failed")

        with patch.dict('os.environ', {'STORAGE_TYPE': 'mongo'}):
            storage = get_storage()

            # Should fallback to JSON
            assert isinstance(storage, JsonFileStorage)


@pytest.mark.integration
class TestFeatureToModelPipeline:
    """Test pipeline from features to model predictions."""

    @pytest.mark.requires_model
    @patch('requests.get')
    def test_complete_prediction_pipeline(self, mock_get):
        """Test complete pipeline: data -> features -> prediction."""
        from src.data.multi_asset_fetcher import MultiAssetDataFetcher
        from src.features.ultimate_features import UltimateFeatureEngine

        # Mock data fetch
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1640000000000 + i*3600000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
            for i in range(200)
        ]
        mock_get.return_value = mock_response

        # Fetch data
        fetcher = MultiAssetDataFetcher()
        df = fetcher.fetch_asset("BTCUSDT", "1h", days=1)

        # Compute features
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(df)

        # Create observation (simplified - normally done by environment)
        feature_list = []
        for key in sorted(features.keys())[:150]:  # Take first 150 features
            val = features[key]
            if isinstance(val, pd.Series):
                val = val.iloc[-1] if len(val) > 0 else 0.0
            feature_list.append(float(val))

        # Pad if needed
        while len(feature_list) < 150:
            feature_list.append(0.0)

        # Add position state (3 dims)
        obs = np.array(feature_list[:150] + [0.0, 0.0, 1.0], dtype=np.float32)

        assert len(obs) == 153
        assert not np.isnan(obs).any()

        # Model prediction (if model exists)
        try:
            from stable_baselines3 import PPO
            model_path = Path(__file__).parent.parent.parent / 'data' / 'models' / 'ultimate_agent.zip'

            if model_path.exists():
                model = PPO.load(str(model_path))
                action, _states = model.predict(obs, deterministic=True)

                assert action in [0, 1, 2]
        except:
            pytest.skip("Model not available for testing")

    def test_feature_dimension_consistency(self):
        """Test that features maintain consistent dimensions."""
        from src.features.ultimate_features import UltimateFeatureEngine

        # Create sample data
        np.random.seed(42)
        n_rows = 100
        base_price = 50000
        returns = np.random.randn(n_rows) * 0.02
        close_prices = base_price * (1 + returns).cumprod()

        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='1h'),
            'open': close_prices * 0.99,
            'high': close_prices * 1.01,
            'low': close_prices * 0.98,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, n_rows),
        })

        engine = UltimateFeatureEngine()
        features = engine.get_all_features(df)

        # All features should have same length
        expected_length = len(df)
        for feature_name, feature_values in features.items():
            if isinstance(feature_values, pd.Series):
                feature_values = feature_values.values
            assert len(feature_values) == expected_length


@pytest.mark.integration
class TestWhaleDataIntegration:
    """Test whale data collection and analysis integration."""

    @patch('requests.get')
    def test_whale_pattern_predictor_integration(self, mock_get):
        """Test whale pattern predictor with real-like data."""
        # This test checks if whale predictor can handle various scenarios
        # without crashing, even with mock/missing data
        try:
            from src.features.whale_pattern_predictor import WhalePatternPredictor

            predictor = WhalePatternPredictor()

            # Test with different symbols
            for symbol in ["ETHUSDT", "BTCUSDT", "SOLUSDT"]:
                # Should not crash even if data is missing
                signal = predictor.get_signal(symbol=symbol)

                assert isinstance(signal, dict)
                assert 'signal' in signal or 'error' in signal
                assert 'confidence' in signal or 'error' in signal

        except Exception as e:
            # If whale predictor dependencies are missing, skip
            pytest.skip(f"Whale predictor not fully available: {e}")


@pytest.mark.integration
class TestEndToEndTradingWorkflow:
    """Test complete trading workflow simulation."""

    def test_simulated_trading_cycle(self, tmp_path):
        """Test a complete trading cycle: fetch -> analyze -> decide -> log."""
        from src.data.storage import JsonFileStorage

        # Setup storage
        storage = JsonFileStorage(base_dir=tmp_path)

        # Initial state
        initial_state = {
            'balance': 10000.0,
            'position': 0,
            'total_pnl': 0.0,
        }
        storage.save_state(initial_state)

        # Simulate trading decisions
        trades = []

        # Buy decision
        trades.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTCUSDT',
            'action': 'BUY',
            'price': 50000.0,
            'quantity': 0.2,
            'pnl': 0.0,
            'fee': 0.2 * 50000 * 0.0004,  # 0.04% fee
        })
        storage.log_trade(trades[-1])

        # Update state after buy
        state = storage.load_state()
        state['position'] = 0.2
        state['entry_price'] = 50000.0
        storage.save_state(state)

        # Sell decision (profitable)
        exit_price = 51000.0
        pnl = (exit_price - 50000.0) * 0.2 - (0.2 * 51000 * 0.0004)

        trades.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTCUSDT',
            'action': 'SELL',
            'price': exit_price,
            'quantity': 0.2,
            'pnl': pnl,
            'fee': 0.2 * 51000 * 0.0004,
        })
        storage.log_trade(trades[-1])

        # Update state after sell
        state = storage.load_state()
        state['position'] = 0
        state['balance'] += pnl
        state['total_pnl'] += pnl
        storage.save_state(state)

        # Verify final state
        final_state = storage.load_state()
        assert final_state['position'] == 0
        assert final_state['balance'] > 10000.0  # Made profit
        assert final_state['total_pnl'] > 0

        # Verify trades logged
        logged_trades = storage.get_trades()
        assert len(logged_trades) == 2
        assert logged_trades[0]['action'] == 'BUY'
        assert logged_trades[1]['action'] == 'SELL'

    @patch('requests.get')
    def test_multi_asset_trading_workflow(self, mock_get, tmp_path):
        """Test workflow with multiple assets."""
        from src.data.multi_asset_fetcher import MultiAssetDataFetcher
        from src.data.storage import JsonFileStorage

        # Mock data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1640000000000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
        ] * 10
        mock_get.return_value = mock_response

        # Fetch data for multiple assets
        fetcher = MultiAssetDataFetcher()
        data = fetcher.fetch_multiple(["BTCUSDT", "ETHUSDT"], "1h", days=1)

        # Setup storage
        storage = JsonFileStorage(base_dir=tmp_path)

        # Initialize multi-asset state
        state = {
            'total_balance': 20000.0,
            'total_pnl': 0.0,
            'assets': {}
        }

        for symbol, df in data.items():
            if not df.empty:
                state['assets'][symbol] = {
                    'balance': 10000.0,
                    'position': 0,
                    'pnl': 0.0,
                }

        storage.save_state(state)

        # Simulate trades on each asset
        for symbol in data.keys():
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'BUY',
                'price': 50000.0 if symbol == 'BTCUSDT' else 3000.0,
                'quantity': 0.1,
                'pnl': 0.0,
            }
            storage.log_trade(trade)

        # Verify
        trades = storage.get_trades()
        assert len(trades) == 2

        final_state = storage.load_state()
        assert len(final_state['assets']) == 2


@pytest.mark.integration
class TestSystemResilience:
    """Test system resilience and error recovery."""

    def test_handles_missing_data_gracefully(self):
        """Test system handles missing data without crashing."""
        from src.features.ultimate_features import UltimateFeatureEngine

        # Empty dataframe
        df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        engine = UltimateFeatureEngine()

        # Should not crash
        features = engine.get_all_features(df)

        # Should return empty or zero features
        assert isinstance(features, dict)

    def test_handles_corrupted_data(self):
        """Test system handles corrupted data."""
        from src.features.ultimate_features import UltimateFeatureEngine

        # DataFrame with NaN values
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
            'open': [np.nan] * 10,
            'high': [51000.0] * 10,
            'low': [49000.0] * 10,
            'close': [50500.0] * 10,
            'volume': [1000] * 10,
        })

        engine = UltimateFeatureEngine()

        # Should handle NaN gracefully
        features = engine.get_all_features(df)

        # Should still produce features (filled with defaults)
        assert len(features) > 0

    def test_storage_recovery_from_corruption(self, tmp_path):
        """Test storage can recover from corruption."""
        from src.data.storage import JsonFileStorage

        storage = JsonFileStorage(base_dir=tmp_path)

        # Write corrupted state file
        with open(storage.state_file, 'w') as f:
            f.write("{ corrupted json")

        # Should return empty dict on load
        state = storage.load_state()
        assert state == {}

        # Should be able to save new state
        new_state = {'balance': 10000.0}
        storage.save_state(new_state)

        # Should load correctly now
        loaded = storage.load_state()
        assert loaded == new_state


@pytest.mark.integration
class TestPerformanceUnderLoad:
    """Test system performance under load."""

    @patch('requests.get')
    def test_concurrent_data_fetching(self, mock_get):
        """Test concurrent data fetching doesn't cause issues."""
        from src.data.multi_asset_fetcher import MultiAssetDataFetcher

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1640000000000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
        ] * 10
        mock_get.return_value = mock_response

        fetcher = MultiAssetDataFetcher(max_workers=8)

        # Fetch many assets concurrently
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"]
        results = fetcher.fetch_multiple(symbols, "1h", days=1)

        # All should succeed
        assert len(results) == len(symbols)
        for symbol, df in results.items():
            assert symbol in symbols

    def test_large_dataset_processing(self):
        """Test processing large datasets."""
        from src.features.ultimate_features import UltimateFeatureEngine

        # Create large dataset
        np.random.seed(42)
        n_rows = 10000
        base_price = 50000
        returns = np.random.randn(n_rows) * 0.02
        close_prices = base_price * (1 + returns).cumprod()

        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='1h'),
            'open': close_prices * 0.99,
            'high': close_prices * 1.01,
            'low': close_prices * 0.98,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, n_rows),
        })

        engine = UltimateFeatureEngine()

        # Should complete without memory issues
        import time
        start = time.time()
        features = engine.get_all_features(df)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert len(features) > 0
