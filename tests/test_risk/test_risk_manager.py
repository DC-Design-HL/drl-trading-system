"""
Tests for Risk Manager and Circuit Breaker

Critical test cases:
- Circuit breaker triggering
- Position sizing limits
- Daily metrics tracking
- Mode transitions
- Stop loss/take profit calculations
"""

import pytest
from datetime import datetime, date
from src.api.risk_manager import (
    CircuitBreaker,
    TradingMode,
    DailyMetrics,
)


@pytest.mark.unit
class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    def test_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.20,
            cooldown_hours=24.0,
        )

        assert cb.max_daily_loss_pct == 0.05
        assert cb.max_drawdown_pct == 0.20
        assert cb.cooldown_hours == 24.0
        assert cb.is_tripped is False
        assert cb.trip_time is None

    def test_initialize_with_balance(self):
        """Test initialization with starting balance."""
        cb = CircuitBreaker()
        cb.initialize(balance=10000.0)

        assert cb.initial_balance == 10000.0
        assert cb.peak_balance == 10000.0
        assert cb.daily_metrics is not None
        assert cb.daily_metrics.start_balance == 10000.0

    def test_circuit_breaker_triggers_on_daily_loss(self):
        """Test that circuit breaker triggers at 5% daily loss."""
        cb = CircuitBreaker(max_daily_loss_pct=0.05)
        cb.initialize(balance=10000.0)

        # Simulate 6% loss
        current_balance = 9400.0

        # Update should trigger circuit breaker
        cb.update(balance=current_balance, trade_pnl=0)

        # Check if tripped
        should_trip = cb.check_circuit_breaker(current_balance)

        # Should trip at >5% loss
        assert should_trip is True
        assert cb.is_tripped is True

    def test_circuit_breaker_does_not_trigger_on_small_loss(self):
        """Test that circuit breaker doesn't trigger on small loss."""
        cb = CircuitBreaker(max_daily_loss_pct=0.05)
        cb.initialize(balance=10000.0)

        # Simulate 3% loss (below threshold)
        current_balance = 9700.0

        cb.update(balance=current_balance, trade_pnl=0)
        should_trip = cb.check_circuit_breaker(current_balance)

        assert should_trip is False
        assert cb.is_tripped is False

    def test_max_drawdown_enforcement(self):
        """Test maximum drawdown enforcement."""
        cb = CircuitBreaker(max_drawdown_pct=0.20)
        cb.initialize(balance=10000.0)

        # Simulate reaching peak
        cb.update(balance=12000.0, trade_pnl=0)
        assert cb.peak_balance == 12000.0

        # Simulate 25% drawdown from peak
        current_balance = 9000.0  # 25% down from 12000
        cb.update(balance=current_balance, trade_pnl=0)

        should_trip = cb.check_circuit_breaker(current_balance)

        # Should trip on >20% drawdown
        assert should_trip is True

    def test_trip_reason_recorded(self):
        """Test that trip reason is recorded."""
        cb = CircuitBreaker(max_daily_loss_pct=0.05)
        cb.initialize(balance=10000.0)

        # Trigger circuit breaker
        cb.update(balance=9400.0, trade_pnl=0)
        cb.check_circuit_breaker(9400.0)

        assert cb.is_tripped is True
        assert cb.trip_reason is not None
        assert "loss" in cb.trip_reason.lower() or "drawdown" in cb.trip_reason.lower()

    def test_trip_time_recorded(self):
        """Test that trip time is recorded."""
        cb = CircuitBreaker(max_daily_loss_pct=0.05)
        cb.initialize(balance=10000.0)

        before_trip = datetime.now()
        cb.update(balance=9400.0, trade_pnl=0)
        cb.check_circuit_breaker(9400.0)
        after_trip = datetime.now()

        assert cb.is_tripped is True
        assert cb.trip_time is not None
        assert before_trip <= cb.trip_time <= after_trip

    def test_on_trip_callback(self):
        """Test that on_trip callback is called."""
        callback_called = {'count': 0, 'data': None}

        def on_trip_callback(data):
            callback_called['count'] += 1
            callback_called['data'] = data

        cb = CircuitBreaker(max_daily_loss_pct=0.05, on_trip=on_trip_callback)
        cb.initialize(balance=10000.0)

        # Trigger circuit breaker
        cb.update(balance=9400.0, trade_pnl=0)
        cb.check_circuit_breaker(9400.0)

        # Callback should have been called
        assert callback_called['count'] == 1
        assert callback_called['data'] is not None


@pytest.mark.unit
class TestDailyMetrics:
    """Test suite for DailyMetrics."""

    def test_initialization(self):
        """Test daily metrics initialization."""
        today = date.today()
        metrics = DailyMetrics(
            date=today,
            start_balance=10000.0,
            current_balance=10500.0,
            high_balance=10800.0,
            low_balance=9900.0,
        )

        assert metrics.date == today
        assert metrics.start_balance == 10000.0
        assert metrics.current_balance == 10500.0

    def test_daily_return_calculation(self):
        """Test daily return calculation."""
        metrics = DailyMetrics(
            date=date.today(),
            start_balance=10000.0,
            current_balance=10500.0,
            high_balance=10500.0,
            low_balance=10000.0,
        )

        # Return should be 5%
        assert metrics.daily_return == 0.05

    def test_daily_drawdown_calculation(self):
        """Test daily drawdown calculation."""
        metrics = DailyMetrics(
            date=date.today(),
            start_balance=10000.0,
            current_balance=9500.0,
            high_balance=10500.0,
            low_balance=9500.0,
        )

        # Drawdown should be (10500 - 9500) / 10500 = 9.52%
        expected_dd = (10500 - 9500) / 10500
        assert abs(metrics.daily_drawdown - expected_dd) < 0.001

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        metrics = DailyMetrics(
            date=date.today(),
            start_balance=10000.0,
            current_balance=10000.0,
            high_balance=10000.0,
            low_balance=10000.0,
            trade_count=10,
            winning_trades=7,
            losing_trades=3,
        )

        # Win rate should be 70%
        assert metrics.win_rate == 0.7

    def test_win_rate_with_no_trades(self):
        """Test win rate with zero trades."""
        metrics = DailyMetrics(
            date=date.today(),
            start_balance=10000.0,
            current_balance=10000.0,
            high_balance=10000.0,
            low_balance=10000.0,
            trade_count=0,
        )

        # Win rate should be 0 with no trades
        assert metrics.win_rate == 0.0


@pytest.mark.unit
class TestTradingMode:
    """Test TradingMode enum."""

    def test_modes_exist(self):
        """Test that all expected modes exist."""
        assert TradingMode.ACTIVE
        assert TradingMode.PAUSED
        assert TradingMode.CIRCUIT_BREAKER
        assert TradingMode.RETRAINING

    def test_mode_values(self):
        """Test mode string values."""
        assert TradingMode.ACTIVE.value == "active"
        assert TradingMode.PAUSED.value == "paused"
        assert TradingMode.CIRCUIT_BREAKER.value == "circuit_breaker"
        assert TradingMode.RETRAINING.value == "retraining"


@pytest.mark.integration
class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker."""

    def test_full_trading_day_simulation(self):
        """Simulate a full trading day with multiple trades."""
        cb = CircuitBreaker(max_daily_loss_pct=0.05)
        cb.initialize(balance=10000.0)

        # Trade 1: Small win
        cb.update(balance=10100.0, trade_pnl=100.0)
        assert cb.is_tripped is False

        # Trade 2: Small loss
        cb.update(balance=10050.0, trade_pnl=-50.0)
        assert cb.is_tripped is False

        # Trade 3: Large loss (triggers breaker)
        cb.update(balance=9400.0, trade_pnl=-650.0)
        should_trip = cb.check_circuit_breaker(9400.0)

        assert should_trip is True
        assert cb.is_tripped is True

        # Further trades should be blocked
        # (This would be enforced by the trading bot, not the CB itself)

    def test_peak_tracking_across_trades(self):
        """Test that peak balance is tracked correctly."""
        cb = CircuitBreaker()
        cb.initialize(balance=10000.0)

        # Series of winning trades
        cb.update(balance=10500.0, trade_pnl=500.0)
        assert cb.peak_balance == 10500.0

        cb.update(balance=11000.0, trade_pnl=500.0)
        assert cb.peak_balance == 11000.0

        # Losing trade (peak shouldn't change)
        cb.update(balance=10800.0, trade_pnl=-200.0)
        assert cb.peak_balance == 11000.0

    def test_daily_metrics_reset(self):
        """Test that daily metrics reset properly."""
        cb = CircuitBreaker()
        cb.initialize(balance=10000.0)

        # Record some activity
        cb.update(balance=10500.0, trade_pnl=500.0)

        # Manually reset daily metrics (simulating new day)
        cb._reset_daily_metrics(balance=10500.0)

        # Metrics should be fresh
        assert cb.daily_metrics.start_balance == 10500.0
        assert cb.daily_metrics.trade_count == 0
        assert cb.daily_metrics.total_pnl == 0.0


@pytest.mark.unit
class TestRiskManagerEdgeCases:
    """Test edge cases for risk manager."""

    def test_zero_balance(self):
        """Test handling of zero balance."""
        cb = CircuitBreaker()

        # Should handle zero balance gracefully
        try:
            cb.initialize(balance=0.0)
            # If it doesn't raise, check state
            assert cb.initial_balance == 0.0
        except ValueError:
            # Acceptable to reject zero balance
            pass

    def test_negative_balance(self):
        """Test handling of negative balance."""
        cb = CircuitBreaker()

        # Should reject or handle negative balance
        try:
            cb.initialize(balance=-1000.0)
            # Should either raise or clamp to zero
            assert cb.initial_balance >= 0.0
        except ValueError:
            # Acceptable to reject negative balance
            pass

    def test_extremely_high_loss_threshold(self):
        """Test with unrealistic loss threshold."""
        cb = CircuitBreaker(max_daily_loss_pct=0.99)
        cb.initialize(balance=10000.0)

        # Even 50% loss shouldn't trip
        cb.update(balance=5000.0, trade_pnl=0)
        should_trip = cb.check_circuit_breaker(5000.0)

        assert should_trip is False

    def test_extremely_low_loss_threshold(self):
        """Test with very tight loss threshold."""
        cb = CircuitBreaker(max_daily_loss_pct=0.01)  # 1% threshold
        cb.initialize(balance=10000.0)

        # Even 2% loss should trip
        cb.update(balance=9800.0, trade_pnl=0)
        should_trip = cb.check_circuit_breaker(9800.0)

        assert should_trip is True
