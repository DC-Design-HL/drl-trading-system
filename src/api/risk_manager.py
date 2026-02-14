"""
Risk Manager Module
Circuit breaker and position sizing logic.
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, Callable
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode states."""
    ACTIVE = "active"           # Normal trading
    PAUSED = "paused"           # Manually paused
    CIRCUIT_BREAKER = "circuit_breaker"  # Stopped due to losses
    RETRAINING = "retraining"   # In retraining mode


@dataclass
class DailyMetrics:
    """Daily trading metrics."""
    date: date
    start_balance: float
    current_balance: float
    high_balance: float
    low_balance: float
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    
    @property
    def daily_return(self) -> float:
        return (self.current_balance - self.start_balance) / self.start_balance
        
    @property
    def daily_drawdown(self) -> float:
        return (self.high_balance - self.current_balance) / self.high_balance
        
    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.winning_trades / self.trade_count


class CircuitBreaker:
    """
    Circuit breaker that stops trading when losses exceed threshold.
    
    If the agent loses more than max_daily_loss_pct of the balance
    in a single day, trading stops and retraining mode is triggered.
    """
    
    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.20,
        cooldown_hours: float = 24.0,
        on_trip: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            max_daily_loss_pct: Maximum daily loss as fraction (0.05 = 5%)
            max_drawdown_pct: Maximum drawdown as fraction (0.20 = 20%)
            cooldown_hours: Hours to wait after circuit breaker triggers
            on_trip: Callback when circuit breaker trips
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_hours = cooldown_hours
        self.on_trip = on_trip
        
        # State
        self.is_tripped = False
        self.trip_time: Optional[datetime] = None
        self.trip_reason: Optional[str] = None
        
        # Daily tracking
        self.daily_metrics: Optional[DailyMetrics] = None
        
        # Overall tracking
        self.initial_balance: Optional[float] = None
        self.peak_balance: float = 0.0
        
    def initialize(self, balance: float):
        """Initialize with starting balance."""
        self.initial_balance = balance
        self.peak_balance = balance
        self._reset_daily_metrics(balance)
        
    def _reset_daily_metrics(self, balance: float):
        """Reset daily metrics for a new day."""
        self.daily_metrics = DailyMetrics(
            date=date.today(),
            start_balance=balance,
            current_balance=balance,
            high_balance=balance,
            low_balance=balance,
        )
        
    def check(self, current_balance: float) -> bool:
        """
        Check if trading should continue.
        
        Args:
            current_balance: Current portfolio value
            
        Returns:
            True if safe to trade, False if circuit breaker tripped
        """
        if self.is_tripped:
            # Check if cooldown has passed
            if self._cooldown_expired():
                self._reset_after_cooldown(current_balance)
                return True
            return False
            
        # Check for new day
        if self.daily_metrics is None or self.daily_metrics.date != date.today():
            self._reset_daily_metrics(current_balance)
            
        # Update metrics
        self.daily_metrics.current_balance = current_balance
        self.daily_metrics.high_balance = max(
            self.daily_metrics.high_balance, current_balance
        )
        self.daily_metrics.low_balance = min(
            self.daily_metrics.low_balance, current_balance
        )
        
        # Update peak balance
        self.peak_balance = max(self.peak_balance, current_balance)
        
        # Check daily loss
        daily_loss = -self.daily_metrics.daily_return
        if daily_loss >= self.max_daily_loss_pct:
            self._trip(
                reason=f"Daily loss exceeded {self.max_daily_loss_pct:.1%}: {daily_loss:.2%}",
                balance=current_balance,
            )
            return False
            
        # Check overall drawdown
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        if drawdown >= self.max_drawdown_pct:
            self._trip(
                reason=f"Drawdown exceeded {self.max_drawdown_pct:.1%}: {drawdown:.2%}",
                balance=current_balance,
            )
            return False
            
        return True
        
    def _trip(self, reason: str, balance: float):
        """Trip the circuit breaker."""
        self.is_tripped = True
        self.trip_time = datetime.now()
        self.trip_reason = reason
        
        logger.warning(f"⚠️ CIRCUIT BREAKER TRIPPED: {reason}")
        
        trip_info = {
            'time': self.trip_time.isoformat(),
            'reason': reason,
            'balance': balance,
            'initial_balance': self.initial_balance,
            'daily_metrics': {
                'start_balance': self.daily_metrics.start_balance,
                'current_balance': self.daily_metrics.current_balance,
                'daily_return': self.daily_metrics.daily_return,
                'trade_count': self.daily_metrics.trade_count,
            },
        }
        
        if self.on_trip:
            self.on_trip(trip_info)
            
    def _cooldown_expired(self) -> bool:
        """Check if cooldown period has passed."""
        if self.trip_time is None:
            return True
        elapsed = (datetime.now() - self.trip_time).total_seconds() / 3600
        return elapsed >= self.cooldown_hours
        
    def _reset_after_cooldown(self, balance: float):
        """Reset circuit breaker after cooldown."""
        logger.info("Circuit breaker cooldown expired, resuming trading")
        self.is_tripped = False
        self.trip_time = None
        self.trip_reason = None
        self._reset_daily_metrics(balance)
        
    def record_trade(self, pnl: float):
        """Record a completed trade."""
        if self.daily_metrics:
            self.daily_metrics.trade_count += 1
            self.daily_metrics.total_pnl += pnl
            if pnl > 0:
                self.daily_metrics.winning_trades += 1
            else:
                self.daily_metrics.losing_trades += 1
                
    def force_reset(self, balance: float):
        """Force reset the circuit breaker."""
        self.is_tripped = False
        self.trip_time = None
        self.trip_reason = None
        self.peak_balance = balance
        self._reset_daily_metrics(balance)
        logger.info("Circuit breaker force reset")
        
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'is_tripped': self.is_tripped,
            'trip_time': self.trip_time.isoformat() if self.trip_time else None,
            'trip_reason': self.trip_reason,
            'cooldown_remaining': self._get_cooldown_remaining(),
            'daily_metrics': {
                'date': str(self.daily_metrics.date) if self.daily_metrics else None,
                'daily_return': self.daily_metrics.daily_return if self.daily_metrics else 0,
                'daily_drawdown': self.daily_metrics.daily_drawdown if self.daily_metrics else 0,
                'trade_count': self.daily_metrics.trade_count if self.daily_metrics else 0,
                'win_rate': self.daily_metrics.win_rate if self.daily_metrics else 0,
            },
            'peak_balance': self.peak_balance,
            'thresholds': {
                'max_daily_loss': self.max_daily_loss_pct,
                'max_drawdown': self.max_drawdown_pct,
            },
        }
        
    def _get_cooldown_remaining(self) -> Optional[str]:
        """Get remaining cooldown time."""
        if not self.is_tripped or self.trip_time is None:
            return None
        elapsed = (datetime.now() - self.trip_time).total_seconds() / 3600
        remaining = max(0, self.cooldown_hours - elapsed)
        if remaining <= 0:
            return "Ready to resume"
        hours = int(remaining)
        minutes = int((remaining - hours) * 60)
        return f"{hours}h {minutes}m"


class RiskManager:
    """
    Comprehensive risk management for the trading system.
    
    Handles:
    - Position sizing
    - Stop loss / take profit
    - Circuit breaker coordination
    """
    
    def __init__(
        self,
        initial_balance: float,
        max_position_size: float = 0.1,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.20,
    ):
        """
        Initialize the risk manager.
        
        Args:
            initial_balance: Starting balance
            max_position_size: Max position as fraction of balance
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_daily_loss_pct: Max daily loss for circuit breaker
            max_drawdown_pct: Max drawdown for circuit breaker
        """
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            max_daily_loss_pct=max_daily_loss_pct,
            max_drawdown_pct=max_drawdown_pct,
            on_trip=self._on_circuit_breaker_trip,
        )
        self.circuit_breaker.initialize(initial_balance)
        
        # Mode
        self.mode = TradingMode.ACTIVE
        
        # Position tracking
        self.current_position: Optional[Dict] = None
        
    def _on_circuit_breaker_trip(self, trip_info: Dict):
        """Handle circuit breaker trip."""
        self.mode = TradingMode.CIRCUIT_BREAKER
        logger.warning(f"Entering circuit breaker mode: {trip_info['reason']}")
        
    def calculate_position_size(
        self,
        balance: float,
        price: float,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            balance: Current balance
            price: Current price
            volatility: Optional volatility for adaptive sizing
            
        Returns:
            Position size in base currency units
        """
        # Base position size
        position_value = balance * self.max_position_size
        
        # Adjust for volatility if provided
        if volatility is not None and volatility > 0:
            # Reduce position size in high volatility
            volatility_factor = min(1.0, 0.02 / volatility)
            position_value *= volatility_factor
            
        # Convert to units
        position_units = position_value / price
        
        return position_units
        
    def calculate_stop_loss(self, entry_price: float, is_long: bool) -> float:
        """Calculate stop loss price."""
        if is_long:
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
            
    def calculate_take_profit(self, entry_price: float, is_long: bool) -> float:
        """Calculate take profit price."""
        if is_long:
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
            
    def can_trade(self, current_balance: float) -> bool:
        """Check if trading is allowed."""
        if self.mode != TradingMode.ACTIVE:
            return False
        return self.circuit_breaker.check(current_balance)
        
    def enter_position(
        self,
        side: str,
        price: float,
        amount: float,
    ) -> Dict[str, Any]:
        """
        Record entering a position.
        
        Returns position info with stop loss and take profit levels.
        """
        is_long = side == 'buy'
        
        self.current_position = {
            'side': side,
            'entry_price': price,
            'amount': amount,
            'entry_time': datetime.now(),
            'stop_loss': self.calculate_stop_loss(price, is_long),
            'take_profit': self.calculate_take_profit(price, is_long),
        }
        
        return self.current_position
        
    def exit_position(self, exit_price: float) -> float:
        """
        Record exiting a position.
        
        Returns P&L.
        """
        if self.current_position is None:
            return 0.0
            
        entry = self.current_position['entry_price']
        amount = self.current_position['amount']
        is_long = self.current_position['side'] == 'buy'
        
        if is_long:
            pnl = (exit_price - entry) * amount
        else:
            pnl = (entry - exit_price) * amount
            
        self.circuit_breaker.record_trade(pnl)
        self.current_position = None
        
        return pnl
        
    def check_stop_loss_take_profit(self, current_price: float) -> Optional[str]:
        """
        Check if stop loss or take profit should trigger.
        
        Returns 'stop_loss', 'take_profit', or None.
        """
        if self.current_position is None:
            return None
            
        is_long = self.current_position['side'] == 'buy'
        sl = self.current_position['stop_loss']
        tp = self.current_position['take_profit']
        
        if is_long:
            if current_price <= sl:
                return 'stop_loss'
            if current_price >= tp:
                return 'take_profit'
        else:
            if current_price >= sl:
                return 'stop_loss'
            if current_price <= tp:
                return 'take_profit'
                
        return None
        
    def set_mode(self, mode: TradingMode):
        """Set trading mode."""
        self.mode = mode
        logger.info(f"Trading mode set to: {mode.value}")
        
    def get_status(self) -> Dict[str, Any]:
        """Get full risk manager status."""
        return {
            'mode': self.mode.value,
            'circuit_breaker': self.circuit_breaker.get_status(),
            'current_position': self.current_position,
            'thresholds': {
                'max_position_size': self.max_position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
            },
        }
