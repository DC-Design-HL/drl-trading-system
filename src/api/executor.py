"""
Order Executor
Handles order execution with risk management integration.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass

from .binance import BinanceConnector
from .risk_manager import RiskManager, TradingMode

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status states."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class Order:
    """Order record."""
    id: str
    symbol: str
    side: str
    type: str
    amount: float
    price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    filled_price: Optional[float] = None
    pnl: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'type': self.type,
            'amount': self.amount,
            'price': self.price,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'filled_price': self.filled_price,
            'pnl': self.pnl,
        }


class OrderExecutor:
    """
    Executes orders with risk management checks.
    
    Integrates with:
    - BinanceConnector for order placement
    - RiskManager for position sizing and circuit breaker
    """
    
    def __init__(
        self,
        connector: BinanceConnector,
        risk_manager: RiskManager,
        symbol: str = 'BTC/USDT',
        dry_run: bool = False,
    ):
        """
        Initialize the order executor.
        
        Args:
            connector: Binance exchange connector
            risk_manager: Risk management module
            symbol: Default trading pair
            dry_run: If True, don't actually place orders
        """
        self.connector = connector
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.dry_run = dry_run
        
        # Order history
        self.orders: List[Order] = []
        self.order_counter = 0
        
    def execute_signal(
        self,
        action: int,
        current_price: float,
        current_balance: float,
        volatility: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Execute a trading signal from the agent.
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            current_price: Current market price
            current_balance: Current account balance
            volatility: Optional volatility for position sizing
            
        Returns:
            Order if placed, None otherwise
        """
        # Check if we can trade
        if not self.risk_manager.can_trade(current_balance):
            logger.warning("Trading not allowed - circuit breaker or mode restriction")
            return None
            
        # Action 0 = hold
        if action == 0:
            # Check for stop loss / take profit on existing position
            trigger = self.risk_manager.check_stop_loss_take_profit(current_price)
            if trigger:
                return self._close_position(current_price, reason=trigger)
            return None
            
        # Check if we need to close existing position first
        if self.risk_manager.current_position is not None:
            current_side = self.risk_manager.current_position['side']
            
            # If same direction, do nothing
            if (action == 1 and current_side == 'buy') or \
               (action == 2 and current_side == 'sell'):
                return None
                
            # Close opposite position
            self._close_position(current_price, reason='signal_reversal')
            
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            balance=current_balance,
            price=current_price,
            volatility=volatility,
        )
        
        if position_size <= 0:
            return None
            
        # Determine side
        side = 'buy' if action == 1 else 'sell'
        
        # Place order
        order = self._place_order(
            side=side,
            amount=position_size,
            price=current_price,
        )
        
        if order and order.status == OrderStatus.FILLED:
            # Record position with risk manager
            self.risk_manager.enter_position(
                side=side,
                price=order.filled_price or current_price,
                amount=position_size,
            )
            
        return order
        
    def _place_order(
        self,
        side: str,
        amount: float,
        price: float,
        order_type: str = 'market',
    ) -> Order:
        """
        Place an order on the exchange.
        
        Args:
            side: 'buy' or 'sell'
            amount: Amount in base currency
            price: Current price (for market orders, this is reference)
            order_type: 'market' or 'limit'
            
        Returns:
            Order record
        """
        self.order_counter += 1
        order_id = f"order_{self.order_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        order = Order(
            id=order_id,
            symbol=self.symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
        )
        
        if self.dry_run:
            # Simulate filled
            order.status = OrderStatus.FILLED
            order.filled_price = price
            logger.info(f"[DRY RUN] Order simulated: {side} {amount:.6f} @ {price:.2f}")
        else:
            # Actual order placement
            try:
                if order_type == 'market':
                    result = self.connector.place_market_order(
                        symbol=self.symbol,
                        side=side,
                        amount=amount,
                    )
                else:
                    result = self.connector.place_limit_order(
                        symbol=self.symbol,
                        side=side,
                        amount=amount,
                        price=price,
                    )
                    
                if result:
                    order.status = OrderStatus.FILLED
                    order.filled_price = float(result.get('price', price))
                    order.id = result.get('id', order_id)
                else:
                    order.status = OrderStatus.FAILED
                    
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                order.status = OrderStatus.FAILED
                
        self.orders.append(order)
        return order
        
    def _close_position(
        self,
        current_price: float,
        reason: str = 'signal',
    ) -> Optional[Order]:
        """
        Close the current position.
        
        Args:
            current_price: Current market price
            reason: Reason for closing
            
        Returns:
            Close order if placed
        """
        position = self.risk_manager.current_position
        if position is None:
            return None
            
        # Opposite side to close
        close_side = 'sell' if position['side'] == 'buy' else 'buy'
        
        order = self._place_order(
            side=close_side,
            amount=position['amount'],
            price=current_price,
        )
        
        if order.status == OrderStatus.FILLED:
            pnl = self.risk_manager.exit_position(
                exit_price=order.filled_price or current_price
            )
            order.pnl = pnl
            
            logger.info(
                f"Position closed ({reason}): {close_side} @ "
                f"{order.filled_price:.2f}, P&L: ${pnl:+.2f}"
            )
            
        return order
        
    def close_all_positions(self, current_price: float) -> List[Order]:
        """Close all open positions."""
        orders = []
        if self.risk_manager.current_position:
            order = self._close_position(current_price, reason='close_all')
            if order:
                orders.append(order)
        return orders
        
    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get recent order history."""
        return [order.to_dict() for order in self.orders[-limit:]]
        
    def get_open_position(self) -> Optional[Dict]:
        """Get current open position."""
        return self.risk_manager.current_position
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        filled_orders = [o for o in self.orders if o.status == OrderStatus.FILLED]
        winning = [o for o in filled_orders if o.pnl and o.pnl > 0]
        losing = [o for o in filled_orders if o.pnl and o.pnl < 0]
        
        total_pnl = sum(o.pnl for o in filled_orders if o.pnl) or 0
        
        return {
            'total_orders': len(self.orders),
            'filled_orders': len(filled_orders),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / max(1, len(winning) + len(losing)),
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / max(1, len([o for o in filled_orders if o.pnl])),
            'dry_run': self.dry_run,
        }
