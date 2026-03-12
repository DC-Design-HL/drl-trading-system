"""
Pytest configuration and shared fixtures for DRL Trading System tests.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.

    Returns 100 rows of synthetic price data with realistic characteristics.
    """
    np.random.seed(42)
    n_rows = 100

    # Generate synthetic price data
    base_price = 50000
    returns = np.random.randn(n_rows) * 0.02  # 2% volatility
    close_prices = base_price * (1 + returns).cumprod()

    # Generate OHLC from close
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_rows) * 0.01))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_rows) * 0.01))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    # Generate volume
    volume = np.random.randint(1000, 10000, n_rows)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='1h'),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
    })

    return df


@pytest.fixture
def large_ohlcv_data() -> pd.DataFrame:
    """
    Generate large OHLCV dataset (1000 rows) for performance testing.
    """
    np.random.seed(42)
    n_rows = 1000

    base_price = 50000
    returns = np.random.randn(n_rows) * 0.02
    close_prices = base_price * (1 + returns).cumprod()

    high_prices = close_prices * (1 + np.abs(np.random.randn(n_rows) * 0.01))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_rows) * 0.01))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    volume = np.random.randint(1000, 10000, n_rows)

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_rows, freq='1h'),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
    })

    return df


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Empty DataFrame with correct columns for edge case testing."""
    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


@pytest.fixture
def single_row_dataframe() -> pd.DataFrame:
    """Single-row DataFrame for edge case testing."""
    return pd.DataFrame({
        'timestamp': [datetime.now()],
        'open': [50000.0],
        'high': [51000.0],
        'low': [49000.0],
        'close': [50500.0],
        'volume': [1000],
    })


@pytest.fixture
def dataframe_with_nans() -> pd.DataFrame:
    """DataFrame with NaN values for testing NaN handling."""
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
        'open': [50000.0] * 10,
        'high': [51000.0] * 5 + [np.nan] * 5,
        'low': [49000.0] * 10,
        'close': [50500.0] * 10,
        'volume': [1000] * 10,
    })
    return df


# ============================================================================
# WHALE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_whale_wallets() -> Dict[str, Any]:
    """Sample whale wallet data for testing."""
    return {
        "ETH": [
            "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8",  # Binance
            "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance 2
        ],
        "SOL": [
            "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvu6Kg",
            "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
        ],
        "XRP": [
            "rLNaPoKeeBjZe2qs6x52yVPKpg8oT9Gkgb",
            "rU2mEJSLqBRkYLVTv55rFTgQajkLTnT6mA",
        ]
    }


@pytest.fixture
def sample_whale_transactions() -> list:
    """Sample whale transaction data."""
    return [
        {
            "timestamp": int(datetime.now().timestamp()) - 3600,
            "value": 1000000,
            "type": "out",
            "to_exchange": True,
        },
        {
            "timestamp": int(datetime.now().timestamp()) - 1800,
            "value": 500000,
            "type": "in",
            "to_exchange": False,
        },
    ]


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def mock_observation() -> np.ndarray:
    """
    Generate a mock observation for model testing.

    Ultimate agent expects 153 dimensions:
    - 150 features
    - 3 position state (position, unrealized_pnl, balance_ratio)
    """
    np.random.seed(42)
    return np.random.randn(153).astype(np.float32)


@pytest.fixture
def mock_vec_normalize_stats() -> Dict[str, np.ndarray]:
    """Mock VecNormalize statistics."""
    return {
        'obs_mean': np.zeros(153),
        'obs_var': np.ones(153),
        'ret_mean': 0.0,
        'ret_var': 1.0,
    }


# ============================================================================
# TRADING FIXTURES
# ============================================================================

@pytest.fixture
def trading_config() -> Dict[str, Any]:
    """Default trading configuration."""
    return {
        'initial_balance': 10000.0,
        'position_size': 0.25,
        'stop_loss_pct': 0.025,
        'take_profit_pct': 0.05,
        'trading_fee': 0.0004,
        'max_position': 1,
        'min_hold_seconds': 14400,
        'cooldown_seconds': 1800,
    }


@pytest.fixture
def risk_config() -> Dict[str, Any]:
    """Risk management configuration."""
    return {
        'max_daily_loss_pct': 0.05,
        'max_drawdown_pct': 0.20,
        'cooldown_hours': 24.0,
    }


# ============================================================================
# PYTEST MARKERS
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower, dependencies)")
    config.addinivalue_line("markers", "slow: Slow tests (skip in quick runs)")
    config.addinivalue_line("markers", "requires_model: Tests that require trained models")
    config.addinivalue_line("markers", "requires_api: Tests that require API keys/network")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def assert_no_nan_inf(df: pd.DataFrame, column_name: str = None):
    """
    Assert that a DataFrame or column has no NaN or inf values.

    Args:
        df: DataFrame to check
        column_name: Optional column name to check specifically
    """
    if column_name:
        assert not df[column_name].isna().any(), f"Column {column_name} contains NaN values"
        assert not np.isinf(df[column_name]).any(), f"Column {column_name} contains inf values"
    else:
        for col in df.columns:
            assert not df[col].isna().any(), f"Column {col} contains NaN values"
            assert not np.isinf(df[col]).any(), f"Column {col} contains inf values"


def assert_valid_signal(signal: float, min_val: float = -1.0, max_val: float = 1.0):
    """Assert that a signal is within valid range and not NaN/inf."""
    assert not np.isnan(signal), "Signal is NaN"
    assert not np.isinf(signal), "Signal is inf"
    assert min_val <= signal <= max_val, f"Signal {signal} outside range [{min_val}, {max_val}]"


def assert_valid_probability(prob: float):
    """Assert that a probability is in [0, 1] and not NaN/inf."""
    assert not np.isnan(prob), "Probability is NaN"
    assert not np.isinf(prob), "Probability is inf"
    assert 0.0 <= prob <= 1.0, f"Probability {prob} outside range [0, 1]"
