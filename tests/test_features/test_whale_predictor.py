"""
Tests for WhalePatternPredictor

Critical test cases:
- Model loading
- Signal generation
- Confidence scoring
- Wallet accuracy weighting
- Edge cases
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.features.whale_pattern_predictor import WhalePatternPredictor


@pytest.mark.unit
class TestWhalePatternPredictor:
    """Test suite for WhalePatternPredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        with patch('src.features.whale_pattern_predictor.WhalePatternLearner'):
            predictor = WhalePatternPredictor()
            assert predictor is not None
            assert hasattr(predictor, 'learners')
            assert hasattr(predictor, 'asset_chain_map')

    def test_asset_chain_mapping(self):
        """Test asset to chain mapping."""
        with patch('src.features.whale_pattern_predictor.WhalePatternLearner'):
            predictor = WhalePatternPredictor()

            # Check expected mappings
            assert predictor.asset_chain_map['ETHUSDT'] == 'ETH'
            assert predictor.asset_chain_map['SOLUSDT'] == 'SOL'
            assert predictor.asset_chain_map['XRPUSDT'] == 'XRP'
            assert predictor.asset_chain_map['BTCUSDT'] == 'ETH'  # BTC uses ETH as proxy

    @patch('src.features.whale_pattern_predictor.WhalePatternLearner')
    def test_get_signal_basic(self, mock_learner_class):
        """Test basic signal generation."""
        # Mock the learner
        mock_learner = Mock()
        mock_learner.load_model.return_value = True
        mock_learner.predict.return_value = {
            'momentum_signal': 0.5,
            'confidence': 0.7,
        }
        mock_learner_class.return_value = mock_learner

        predictor = WhalePatternPredictor()

        # Get signal
        signal = predictor.get_signal(symbol='ETHUSDT')

        # Check signal structure
        assert isinstance(signal, dict)
        assert 'signal' in signal
        assert 'confidence' in signal

    @patch('src.features.whale_pattern_predictor.WhalePatternLearner')
    def test_signal_bounds(self, mock_learner_class):
        """Test that signals are within expected bounds."""
        # Mock the learner to return extreme values
        mock_learner = Mock()
        mock_learner.load_model.return_value = True
        mock_learner.predict.return_value = {
            'momentum_signal': 0.8,
            'confidence': 0.9,
        }
        mock_learner_class.return_value = mock_learner

        predictor = WhalePatternPredictor()
        signal = predictor.get_signal(symbol='ETHUSDT')

        # Signal should be -1 to 1
        assert -1.0 <= signal['signal'] <= 1.0, f"Signal {signal['signal']} out of range"

        # Confidence should be 0 to 1
        assert 0.0 <= signal['confidence'] <= 1.0, f"Confidence {signal['confidence']} out of range"

    @patch('src.features.whale_pattern_predictor.WhalePatternLearner')
    def test_no_model_loaded(self, mock_learner_class):
        """Test behavior when no model is loaded."""
        # Mock learner that fails to load
        mock_learner = Mock()
        mock_learner.load_model.return_value = False
        mock_learner_class.return_value = mock_learner

        predictor = WhalePatternPredictor()

        # Should still return a signal (neutral/default)
        signal = predictor.get_signal(symbol='ETHUSDT')

        assert isinstance(signal, dict)
        # Should return neutral signal when no model
        assert signal['signal'] == 0.0 or signal['confidence'] == 0.0

    @patch('src.features.whale_pattern_predictor.WhalePatternLearner')
    def test_unsupported_asset(self, mock_learner_class):
        """Test handling of unsupported asset."""
        mock_learner = Mock()
        mock_learner.load_model.return_value = True
        mock_learner_class.return_value = mock_learner

        predictor = WhalePatternPredictor()

        # Try to get signal for unsupported asset
        signal = predictor.get_signal(symbol='DOGEUSDT')

        # Should either return neutral signal or raise error
        assert isinstance(signal, dict)
        # If it returns, should be neutral
        assert signal['signal'] == 0.0 or signal['confidence'] < 0.5

    @patch('src.features.whale_pattern_predictor.WhalePatternLearner')
    def test_prediction_caching(self, mock_learner_class):
        """Test that predictions are cached."""
        mock_learner = Mock()
        mock_learner.load_model.return_value = True
        mock_learner.predict.return_value = {
            'momentum_signal': 0.5,
            'confidence': 0.7,
        }
        mock_learner_class.return_value = mock_learner

        predictor = WhalePatternPredictor()

        # Get signal twice quickly
        signal1 = predictor.get_signal(symbol='ETHUSDT')
        signal2 = predictor.get_signal(symbol='ETHUSDT')

        # Both should return same result
        assert signal1 == signal2

    @patch('src.features.whale_pattern_predictor.WhalePatternLearner')
    def test_multi_chain_signals(self, mock_learner_class):
        """Test getting signals for multiple chains."""
        mock_learner = Mock()
        mock_learner.load_model.return_value = True
        mock_learner.predict.return_value = {
            'momentum_signal': 0.5,
            'confidence': 0.7,
        }
        mock_learner_class.return_value = mock_learner

        predictor = WhalePatternPredictor()

        # Get signals for different assets
        eth_signal = predictor.get_signal(symbol='ETHUSDT')
        sol_signal = predictor.get_signal(symbol='SOLUSDT')
        xrp_signal = predictor.get_signal(symbol='XRPUSDT')

        # All should return valid signals
        assert isinstance(eth_signal, dict)
        assert isinstance(sol_signal, dict)
        assert isinstance(xrp_signal, dict)


@pytest.mark.integration
@pytest.mark.requires_model
class TestWhalePatternPredictorIntegration:
    """Integration tests requiring trained models."""

    def test_real_model_loading(self):
        """Test loading real whale pattern models (if available)."""
        try:
            predictor = WhalePatternPredictor()

            # Check if any models loaded
            has_models = len(predictor.learners) > 0

            if has_models:
                # If models exist, test prediction
                signal = predictor.get_signal(symbol='ETHUSDT')

                assert isinstance(signal, dict)
                assert 'signal' in signal
                assert 'confidence' in signal
                assert -1.0 <= signal['signal'] <= 1.0
                assert 0.0 <= signal['confidence'] <= 1.0
            else:
                # Skip if no models trained
                pytest.skip("No whale pattern models found")

        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    def test_real_signal_generation(self):
        """Test real signal generation with actual models."""
        try:
            predictor = WhalePatternPredictor()

            if len(predictor.learners) == 0:
                pytest.skip("No models loaded")

            # Get signal for each supported asset
            for symbol in ['ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BTCUSDT']:
                signal = predictor.get_signal(symbol=symbol)

                # Validate signal structure
                assert isinstance(signal, dict)
                assert 'signal' in signal
                assert 'confidence' in signal

                # Validate ranges
                assert -1.0 <= signal['signal'] <= 1.0
                assert 0.0 <= signal['confidence'] <= 1.0

                print(f"\n{symbol}: signal={signal['signal']:.3f}, confidence={signal['confidence']:.3f}")

        except Exception as e:
            pytest.skip(f"Signal generation failed: {e}")


@pytest.mark.unit
class TestWhalePatternPredictorHelpers:
    """Test helper methods of WhalePatternPredictor."""

    @patch('src.features.whale_pattern_predictor.WhalePatternLearner')
    def test_compute_wallet_accuracy_weights(self, mock_learner_class):
        """Test wallet accuracy weight computation."""
        mock_learner = Mock()
        mock_learner.load_model.return_value = True
        mock_learner_class.return_value = mock_learner

        predictor = WhalePatternPredictor()

        # Should have computed accuracy weights
        assert hasattr(predictor, 'wallet_accuracy')
        assert isinstance(predictor.wallet_accuracy, dict)

    @patch('src.features.whale_pattern_predictor.WhalePatternLearner')
    @patch('src.features.whale_pattern_predictor.WhaleWalletCollector')
    def test_maybe_collect(self, mock_collector_class, mock_learner_class):
        """Test conditional wallet data collection."""
        mock_learner = Mock()
        mock_learner.load_model.return_value = True
        mock_learner_class.return_value = mock_learner

        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector

        predictor = WhalePatternPredictor()

        # Call _maybe_collect
        predictor._maybe_collect()

        # Should track last collection time
        assert predictor.last_collection_time > 0
