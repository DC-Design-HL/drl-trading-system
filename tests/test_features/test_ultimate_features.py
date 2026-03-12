"""
Tests for UltimateFeatureEngine

Critical test cases:
- Feature computation correctness
- Wyckoff phase detection
- SMC feature extraction
- No NaN/inf in outputs
- Edge case handling
"""

import pytest
import numpy as np
import pandas as pd
from src.features.ultimate_features import (
    UltimateFeatureEngine,
    WyckoffAnalyzer,
    SMCAnalyzer,
)


@pytest.mark.unit
class TestUltimateFeatureEngine:
    """Test suite for UltimateFeatureEngine."""

    def test_initialization(self):
        """Test that engine initializes without errors."""
        engine = UltimateFeatureEngine()
        assert engine is not None

    def test_feature_computation_basic(self, sample_ohlcv_data):
        """Test basic feature computation."""
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(sample_ohlcv_data)

        # Check that features were computed
        assert isinstance(features, dict)
        assert len(features) > 0

        # Check for expected features
        assert 'rsi_14' in features
        assert 'macd' in features
        assert 'bb_upper' in features
        assert 'atr_14' in features

    def test_no_nan_inf_in_features(self, sample_ohlcv_data):
        """Test that features don't produce NaN or inf values."""
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(sample_ohlcv_data)

        for feature_name, feature_values in features.items():
            # Convert to numpy array if Series
            if isinstance(feature_values, pd.Series):
                feature_values = feature_values.values

            # Check for NaN (after fillna in engine, should be 0)
            nan_count = np.isnan(feature_values).sum()
            assert nan_count == 0, f"Feature {feature_name} has {nan_count} NaN values"

            # Check for inf
            inf_count = np.isinf(feature_values).sum()
            assert inf_count == 0, f"Feature {feature_name} has {inf_count} inf values"

    def test_feature_dimensions(self, sample_ohlcv_data):
        """Test that features have correct dimensions."""
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(sample_ohlcv_data)

        # All features should have same length as input data
        expected_length = len(sample_ohlcv_data)
        for feature_name, feature_values in features.items():
            if isinstance(feature_values, pd.Series):
                feature_values = feature_values.values
            actual_length = len(feature_values)
            assert actual_length == expected_length, \
                f"Feature {feature_name} has length {actual_length}, expected {expected_length}"

    def test_rsi_bounds(self, sample_ohlcv_data):
        """Test that RSI is bounded between 0 and 100."""
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(sample_ohlcv_data)

        rsi = features['rsi_14']
        if isinstance(rsi, pd.Series):
            rsi = rsi.values

        # Remove NaN values for checking
        rsi_valid = rsi[~np.isnan(rsi)]

        assert np.all(rsi_valid >= 0), "RSI has values < 0"
        assert np.all(rsi_valid <= 100), "RSI has values > 100"

    def test_macd_computation(self, sample_ohlcv_data):
        """Test MACD computation."""
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(sample_ohlcv_data)

        assert 'macd' in features
        assert 'macd_signal' in features
        assert 'macd_hist' in features

        # MACD histogram should be MACD - Signal
        macd = features['macd']
        signal = features['macd_signal']
        hist = features['macd_hist']

        if isinstance(macd, pd.Series):
            macd = macd.values
            signal = signal.values
            hist = hist.values

        # Check relationship (allowing for small numerical errors)
        valid_idx = ~(np.isnan(macd) | np.isnan(signal) | np.isnan(hist))
        if valid_idx.any():
            expected_hist = macd[valid_idx] - signal[valid_idx]
            actual_hist = hist[valid_idx]
            np.testing.assert_allclose(actual_hist, expected_hist, rtol=1e-5)

    def test_empty_dataframe_handling(self, empty_dataframe):
        """Test handling of empty DataFrame."""
        engine = UltimateFeatureEngine()

        # Should either return empty features or raise informative error
        try:
            features = engine.get_all_features(empty_dataframe)
            # If it returns, check it's empty or has appropriate shape
            assert len(features) == 0 or all(len(v) == 0 for v in features.values())
        except ValueError as e:
            # Acceptable to raise ValueError for empty data
            assert "empty" in str(e).lower() or "insufficient" in str(e).lower()

    def test_single_row_dataframe(self, single_row_dataframe):
        """Test handling of single-row DataFrame."""
        engine = UltimateFeatureEngine()

        # Should handle gracefully (may have NaN for indicators requiring history)
        try:
            features = engine.get_all_features(single_row_dataframe)
            assert isinstance(features, dict)
            # Features should have length 1
            for feature_name, feature_values in features.items():
                if isinstance(feature_values, pd.Series):
                    assert len(feature_values) == 1
        except ValueError as e:
            # Acceptable to require minimum data
            assert "insufficient" in str(e).lower()

    def test_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands computation."""
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(sample_ohlcv_data)

        assert 'bb_upper' in features
        assert 'bb_middle' in features
        assert 'bb_lower' in features

        upper = features['bb_upper']
        middle = features['bb_middle']
        lower = features['bb_lower']

        if isinstance(upper, pd.Series):
            upper = upper.values
            middle = middle.values
            lower = lower.values

        # Upper should be >= middle >= lower (where not NaN)
        valid_idx = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        if valid_idx.any():
            assert np.all(upper[valid_idx] >= middle[valid_idx]), "BB upper < middle"
            assert np.all(middle[valid_idx] >= lower[valid_idx]), "BB middle < lower"


@pytest.mark.unit
class TestWyckoffAnalyzer:
    """Test suite for WyckoffAnalyzer."""

    def test_initialization(self):
        """Test Wyckoff analyzer initialization."""
        analyzer = WyckoffAnalyzer()
        assert analyzer is not None
        assert analyzer.lookback == 50  # Default

    def test_climax_detection(self, sample_ohlcv_data):
        """Test selling/buying climax detection."""
        analyzer = WyckoffAnalyzer()
        climax = analyzer.detect_climax(sample_ohlcv_data)

        assert isinstance(climax, pd.Series)
        assert len(climax) == len(sample_ohlcv_data)

        # Climax values should be -1, 0, or 1
        unique_values = set(climax.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_spring_upthrust_detection(self, sample_ohlcv_data):
        """Test spring and upthrust detection."""
        analyzer = WyckoffAnalyzer()
        spring, upthrust = analyzer.detect_spring_upthrust(sample_ohlcv_data)

        assert isinstance(spring, pd.Series)
        assert isinstance(upthrust, pd.Series)
        assert len(spring) == len(sample_ohlcv_data)
        assert len(upthrust) == len(sample_ohlcv_data)

        # Should be binary (0 or 1)
        assert set(spring.unique()).issubset({0.0, 1.0})
        assert set(upthrust.unique()).issubset({0.0, 1.0})

    def test_phase_detection(self, sample_ohlcv_data):
        """Test Wyckoff phase classification."""
        analyzer = WyckoffAnalyzer()
        phase = analyzer.detect_phase(sample_ohlcv_data)

        assert isinstance(phase, pd.Series)
        assert len(phase) == len(sample_ohlcv_data)

        # Phase should be 0-4 (Unknown, Accumulation, Markup, Distribution, Markdown)
        unique_phases = set(phase.unique())
        assert unique_phases.issubset({0, 1, 2, 3, 4})


@pytest.mark.unit
class TestSMCAnalyzer:
    """Test suite for Smart Money Concepts analyzer."""

    def test_initialization(self):
        """Test SMC analyzer initialization."""
        analyzer = SMCAnalyzer()
        assert analyzer is not None

    def test_order_block_detection(self, sample_ohlcv_data):
        """Test order block detection."""
        analyzer = SMCAnalyzer()
        bullish_ob, bearish_ob = analyzer.detect_order_blocks(sample_ohlcv_data)

        assert isinstance(bullish_ob, pd.Series)
        assert isinstance(bearish_ob, pd.Series)
        assert len(bullish_ob) == len(sample_ohlcv_data)
        assert len(bearish_ob) == len(sample_ohlcv_data)

        # Should be binary
        assert set(bullish_ob.unique()).issubset({0.0, 1.0, False, True})
        assert set(bearish_ob.unique()).issubset({0.0, 1.0, False, True})

    def test_fair_value_gap(self, sample_ohlcv_data):
        """Test Fair Value Gap detection."""
        analyzer = SMCAnalyzer()
        fvg = analyzer.detect_fair_value_gap(sample_ohlcv_data)

        assert isinstance(fvg, pd.Series)
        assert len(fvg) == len(sample_ohlcv_data)

        # FVG should be numeric
        assert fvg.dtype in [np.float32, np.float64, float]

    def test_break_of_structure(self, sample_ohlcv_data):
        """Test Break of Structure detection."""
        analyzer = SMCAnalyzer()
        bos = analyzer.detect_break_of_structure(sample_ohlcv_data)

        assert isinstance(bos, pd.Series)
        assert len(bos) == len(sample_ohlcv_data)

        # BOS should be -1 (bearish), 0 (none), or 1 (bullish)
        unique_values = set(bos.unique())
        assert unique_values.issubset({-1, 0, 1, -1.0, 0.0, 1.0})


@pytest.mark.integration
class TestFeatureEngineIntegration:
    """Integration tests for full feature pipeline."""

    def test_large_dataset_processing(self, large_ohlcv_data):
        """Test processing large dataset (1000 rows)."""
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(large_ohlcv_data)

        assert len(features) > 0
        # Check all features have correct length
        for feature_name, feature_values in features.items():
            if isinstance(feature_values, pd.Series):
                assert len(feature_values) == len(large_ohlcv_data)

    def test_feature_count(self, sample_ohlcv_data):
        """Test that we generate expected number of features."""
        engine = UltimateFeatureEngine()
        features = engine.get_all_features(sample_ohlcv_data)

        # Should have 100+ features (targeting 150+)
        assert len(features) >= 100, f"Only {len(features)} features generated, expected 100+"

    @pytest.mark.slow
    def test_performance_benchmark(self, large_ohlcv_data):
        """Benchmark feature computation performance."""
        import time

        engine = UltimateFeatureEngine()

        start = time.time()
        features = engine.get_all_features(large_ohlcv_data)
        elapsed = time.time() - start

        # Should compute 1000 rows in < 1 second (this is generous)
        assert elapsed < 1.0, f"Feature computation took {elapsed:.2f}s, too slow"

        print(f"\nPerformance: {len(large_ohlcv_data)} rows in {elapsed*1000:.2f}ms")
        print(f"Features generated: {len(features)}")
