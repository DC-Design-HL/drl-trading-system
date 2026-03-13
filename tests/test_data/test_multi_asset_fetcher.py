"""
Tests for MultiAssetDataFetcher

Test Coverage:
- Asset configuration management
- Single asset data fetching
- Multiple asset parallel fetching
- Combined dataset creation
- Asset embedding generation
- Error handling and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.data.multi_asset_fetcher import (
    MultiAssetDataFetcher,
    AssetConfig,
    SUPPORTED_ASSETS,
    get_asset_embedding,
    get_all_supported_symbols,
)


@pytest.mark.unit
class TestAssetConfig:
    """Test suite for AssetConfig dataclass."""

    def test_asset_config_creation(self):
        """Test creating AssetConfig instance."""
        config = AssetConfig(
            symbol="TESTUSDT",
            name="Test Coin",
            asset_id=99,
            base_volatility=1.5,
            liquidity_score=0.8,
            btc_correlation=0.7,
        )

        assert config.symbol == "TESTUSDT"
        assert config.name == "Test Coin"
        assert config.asset_id == 99
        assert config.base_volatility == 1.5
        assert config.liquidity_score == 0.8
        assert config.btc_correlation == 0.7

    def test_to_features(self):
        """Test conversion to feature vector."""
        config = AssetConfig(
            symbol="BTCUSDT",
            name="Bitcoin",
            asset_id=0,
            base_volatility=1.0,
            liquidity_score=1.0,
            btc_correlation=1.0,
        )

        features = config.to_features()

        assert isinstance(features, np.ndarray)
        assert len(features) == 4
        assert features[0] == 0.0  # asset_id / 10
        assert features[1] == 1.0  # base_volatility
        assert features[2] == 1.0  # liquidity_score
        assert features[3] == 1.0  # btc_correlation

    def test_supported_assets_configuration(self):
        """Test that all supported assets are properly configured."""
        assert len(SUPPORTED_ASSETS) >= 6
        assert "BTCUSDT" in SUPPORTED_ASSETS
        assert "ETHUSDT" in SUPPORTED_ASSETS
        assert "SOLUSDT" in SUPPORTED_ASSETS
        assert "XRPUSDT" in SUPPORTED_ASSETS

        # Check BTC configuration (baseline)
        btc = SUPPORTED_ASSETS["BTCUSDT"]
        assert btc.asset_id == 0
        assert btc.base_volatility == 1.0
        assert btc.liquidity_score == 1.0
        assert btc.btc_correlation == 1.0

    def test_asset_volatility_ordering(self):
        """Test that more volatile assets have higher base_volatility."""
        btc_vol = SUPPORTED_ASSETS["BTCUSDT"].base_volatility
        eth_vol = SUPPORTED_ASSETS["ETHUSDT"].base_volatility
        sol_vol = SUPPORTED_ASSETS["SOLUSDT"].base_volatility
        doge_vol = SUPPORTED_ASSETS["DOGEUSDT"].base_volatility

        # BTC is baseline
        assert btc_vol == 1.0

        # SOL and DOGE should be more volatile than BTC
        assert sol_vol > btc_vol
        assert doge_vol > btc_vol


@pytest.mark.unit
class TestMultiAssetDataFetcher:
    """Test suite for MultiAssetDataFetcher."""

    def test_initialization(self):
        """Test fetcher initializes correctly."""
        fetcher = MultiAssetDataFetcher()
        assert fetcher is not None
        assert fetcher.max_workers == 4

    def test_initialization_custom_params(self):
        """Test fetcher with custom parameters."""
        fetcher = MultiAssetDataFetcher(
            base_url="https://custom.api.com",
            max_workers=8,
        )
        assert fetcher.base_url == "https://custom.api.com"
        assert fetcher.max_workers == 8

    def test_get_asset_config(self):
        """Test getting asset configuration."""
        fetcher = MultiAssetDataFetcher()
        config = fetcher.get_asset_config("BTCUSDT")

        assert isinstance(config, AssetConfig)
        assert config.symbol == "BTCUSDT"
        assert config.asset_id == 0

    def test_get_asset_config_invalid_symbol(self):
        """Test error handling for invalid symbol."""
        fetcher = MultiAssetDataFetcher()

        with pytest.raises(ValueError, match="Unsupported asset"):
            fetcher.get_asset_config("INVALIDUSDT")

    @patch('requests.get')
    def test_fetch_asset_success(self, mock_get):
        """Test successful data fetching for a single asset."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [
                1640000000000,  # timestamp
                "50000.0",      # open
                "51000.0",      # high
                "49000.0",      # low
                "50500.0",      # close
                "100.5",        # volume
                1640003599999,  # close_time
                "5050000.0",    # quote_volume
                1000,           # trades
                "50.0",         # taker_buy_base
                "2525000.0",    # taker_buy_quote
                "0"             # ignore
            ],
        ] * 10  # 10 candles

        mock_get.return_value = mock_response

        fetcher = MultiAssetDataFetcher()
        df = fetcher.fetch_asset("BTCUSDT", "1h", days=1)

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) >= 10

        # Check required columns
        required_cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'symbol', 'asset_id', 'base_volatility', 'liquidity_score',
            'btc_correlation'
        ]
        for col in required_cols:
            assert col in df.columns

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        assert pd.api.types.is_float_dtype(df['close'])
        assert pd.api.types.is_float_dtype(df['volume'])

        # Check asset metadata
        assert df['symbol'].iloc[0] == "BTCUSDT"
        assert df['asset_id'].iloc[0] == 0

    @patch('requests.get')
    def test_fetch_asset_empty_response(self, mock_get):
        """Test handling of empty API response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        fetcher = MultiAssetDataFetcher()
        df = fetcher.fetch_asset("BTCUSDT", "1h", days=1)

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch('requests.get')
    def test_fetch_asset_api_error(self, mock_get):
        """Test handling of API errors."""
        mock_get.side_effect = Exception("API Error")

        fetcher = MultiAssetDataFetcher()
        df = fetcher.fetch_asset("BTCUSDT", "1h", days=1)

        # Should return empty DataFrame on error
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch('requests.get')
    def test_fetch_multiple_assets(self, mock_get):
        """Test parallel fetching of multiple assets."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1640000000000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
        ] * 5
        mock_get.return_value = mock_response

        fetcher = MultiAssetDataFetcher()
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        results = fetcher.fetch_multiple(symbols, "1h", days=1)

        assert isinstance(results, dict)
        assert len(results) == 3

        for symbol in symbols:
            assert symbol in results
            assert isinstance(results[symbol], pd.DataFrame)

    @patch('requests.get')
    def test_fetch_all_supported(self, mock_get):
        """Test fetching all supported assets."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1640000000000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
        ] * 5
        mock_get.return_value = mock_response

        fetcher = MultiAssetDataFetcher()
        results = fetcher.fetch_all_supported("1h", days=1)

        assert isinstance(results, dict)
        assert len(results) == len(SUPPORTED_ASSETS)

    @patch('requests.get')
    def test_create_combined_dataset(self, mock_get):
        """Test creating combined dataset from multiple assets."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1640000000000 + i*3600000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
            for i in range(10)
        ]
        mock_get.return_value = mock_response

        fetcher = MultiAssetDataFetcher()
        combined = fetcher.create_combined_dataset(
            ["BTCUSDT", "ETHUSDT"],
            "1h",
            days=1
        )

        assert isinstance(combined, pd.DataFrame)
        assert not combined.empty
        assert len(combined) >= 20  # At least 10 rows per asset

        # Check that multiple symbols are present
        symbols = combined['symbol'].unique()
        assert len(symbols) >= 2

        # Check sorting
        for symbol in symbols:
            symbol_data = combined[combined['symbol'] == symbol]
            timestamps = symbol_data['timestamp'].values
            assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))

    @patch('requests.get')
    def test_interval_mapping(self, mock_get):
        """Test that different intervals are handled correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1640000000000, "50000", "51000", "49000", "50500", "100",
             1640003599999, "5050000", 1000, "50", "2525000", "0"]
        ]
        mock_get.return_value = mock_response

        fetcher = MultiAssetDataFetcher()

        intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
        for interval in intervals:
            df = fetcher.fetch_asset("BTCUSDT", interval, days=1)
            # Should not raise error
            assert isinstance(df, pd.DataFrame)


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_asset_embedding(self):
        """Test getting asset embedding."""
        embedding = get_asset_embedding("BTCUSDT")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 4
        assert not np.isnan(embedding).any()

    def test_get_asset_embedding_invalid(self):
        """Test error handling for invalid symbol."""
        with pytest.raises(ValueError, match="Unknown asset"):
            get_asset_embedding("INVALIDUSDT")

    def test_get_all_supported_symbols(self):
        """Test getting all supported symbols."""
        symbols = get_all_supported_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) >= 6
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols


@pytest.mark.integration
class TestDataFetcherIntegration:
    """Integration tests (require network access)."""

    @pytest.mark.requires_api
    def test_real_fetch_btc(self):
        """Test real API call for BTC data."""
        fetcher = MultiAssetDataFetcher()
        df = fetcher.fetch_asset("BTCUSDT", "1h", days=1, limit=100)

        # If API is available, we should get data
        if not df.empty:
            assert len(df) > 0
            assert 'close' in df.columns
            assert df['close'].iloc[-1] > 0

    @pytest.mark.requires_api
    def test_real_fetch_multiple(self):
        """Test real API call for multiple assets."""
        fetcher = MultiAssetDataFetcher()
        results = fetcher.fetch_multiple(
            ["BTCUSDT", "ETHUSDT"],
            "1h",
            days=1
        )

        # If API is available, check results
        if any(not df.empty for df in results.values()):
            assert len(results) == 2
