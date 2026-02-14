"""
Advanced Risk Management Module

Features:
- Adaptive SL/TP based on ATR volatility
- Kelly Criterion position sizing
- Trailing stops
- Dynamic risk per trade
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Dynamic risk parameters."""
    stop_loss_pct: float
    take_profit_pct: float
    position_size: float
    trailing_stop_pct: Optional[float]
    risk_reward_ratio: float


class AdaptiveRiskManager:
    """
    Advanced risk management with dynamic SL/TP and position sizing.
    
    Key features:
    - ATR-based stop loss and take profit
    - Kelly Criterion for optimal position sizing
    - Trailing stops for maximizing winners
    """
    
    def __init__(
        self,
        base_sl_pct: float = 0.015,  # 1.5% base stop loss
        base_tp_pct: float = 0.025,  # 2.5% base take profit
        base_position_size: float = 0.5,  # 50% base position
        min_sl_pct: float = 0.008,  # Minimum 0.8% SL
        max_sl_pct: float = 0.03,   # Maximum 3% SL
        min_tp_pct: float = 0.015,  # Minimum 1.5% TP
        max_tp_pct: float = 0.06,   # Maximum 6% TP
        min_position_size: float = 0.1,  # Minimum 10% position
        max_position_size: float = 0.75,  # Maximum 75% position
        atr_period: int = 14,
        use_kelly: bool = True,
        use_trailing: bool = True,
    ):
        self.base_sl_pct = base_sl_pct
        self.base_tp_pct = base_tp_pct
        self.base_position_size = base_position_size
        self.min_sl_pct = min_sl_pct
        self.max_sl_pct = max_sl_pct
        self.min_tp_pct = min_tp_pct
        self.max_tp_pct = max_tp_pct
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.atr_period = atr_period
        self.use_kelly = use_kelly
        self.use_trailing = use_trailing
        
        # Trade history for Kelly calculation
        self.trade_history: list = []
        
        logger.info(f"📊 AdaptiveRiskManager initialized (Kelly={use_kelly}, Trailing={use_trailing})")
    
    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> float:
        """Calculate Average True Range."""
        period = period or self.atr_period
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(tr[-period:])
        
        return atr
    
    def get_adaptive_sl_tp(self, df: pd.DataFrame, trade_type: str = "long") -> Tuple[float, float]:
        """
        Calculate adaptive SL/TP based on current volatility (ATR).
        
        Higher volatility = wider stops
        Lower volatility = tighter stops
        """
        current_price = df['close'].iloc[-1]
        atr = self.calculate_atr(df)
        
        # ATR as percentage of price
        atr_pct = atr / current_price
        
        # Calculate "normal" ATR (average over last 50 periods)
        historical_atrs = []
        for i in range(50, len(df), 10):
            historical_atrs.append(self.calculate_atr(df.iloc[:i]))
        
        if historical_atrs:
            avg_atr = np.mean(historical_atrs)
            volatility_ratio = atr / (avg_atr + 1e-10)
        else:
            volatility_ratio = 1.0
        
        # Adjust SL/TP based on volatility
        # Higher volatility = wider stops (multiply by ratio)
        adjusted_sl = self.base_sl_pct * volatility_ratio
        adjusted_tp = self.base_tp_pct * volatility_ratio
        
        # Clamp to min/max
        sl_pct = np.clip(adjusted_sl, self.min_sl_pct, self.max_sl_pct)
        tp_pct = np.clip(adjusted_tp, self.min_tp_pct, self.max_tp_pct)
        
        logger.info(
            f"📊 Adaptive SL/TP: ATR={atr:.2f} ({atr_pct:.2%}), "
            f"Vol Ratio={volatility_ratio:.2f} → SL={sl_pct:.2%}, TP={tp_pct:.2%}"
        )
        
        return sl_pct, tp_pct
    
    def calculate_kelly_fraction(self) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly = (Win% * Avg_Win / Avg_Loss) - (1 - Win%) / (Avg_Win / Avg_Loss)
        Or simplified: Kelly = Win% - (Loss% / Win/Loss Ratio)
        """
        if len(self.trade_history) < 10:
            # Not enough history, use base position size
            return self.base_position_size
        
        # Calculate statistics
        wins = [t for t in self.trade_history if t > 0]
        losses = [t for t in self.trade_history if t < 0]
        
        if not wins or not losses:
            return self.base_position_size
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # Kelly formula
        if avg_loss == 0:
            return self.max_position_size
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Use half-Kelly for safety (more conservative)
        half_kelly = kelly * 0.5
        
        # Clamp to reasonable range
        position_size = np.clip(half_kelly, self.min_position_size, self.max_position_size)
        
        logger.info(
            f"📊 Kelly Sizing: Win%={win_rate:.1%}, W/L Ratio={win_loss_ratio:.2f}, "
            f"Kelly={kelly:.2%}, Half-Kelly={position_size:.2%}"
        )
        
        return position_size
    
    def record_trade(self, pnl_pct: float):
        """Record trade result for Kelly calculation."""
        self.trade_history.append(pnl_pct)
        # Keep only last 50 trades for recency
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]
    
    def get_trailing_stop(
        self, 
        entry_price: float, 
        current_price: float, 
        highest_price: float,
        trade_type: str = "long",
        base_trailing_pct: float = 0.015
    ) -> Tuple[float, bool]:
        """
        Calculate trailing stop price and whether it's triggered.
        
        Args:
            entry_price: Position entry price
            current_price: Current market price
            highest_price: Highest price since entry (for longs)
            trade_type: "long" or "short"
            base_trailing_pct: Trailing stop percentage
            
        Returns:
            Tuple of (trailing_stop_price, is_triggered)
        """
        if trade_type == "long":
            # For longs, trail from the highest price
            trailing_stop = highest_price * (1 - base_trailing_pct)
            triggered = current_price <= trailing_stop
        else:
            # For shorts, trail from the lowest price
            trailing_stop = highest_price * (1 + base_trailing_pct)  # highest_price is actually lowest for shorts
            triggered = current_price >= trailing_stop
        
        return trailing_stop, triggered
    
    def get_risk_parameters(self, df: pd.DataFrame, trade_type: str = "long") -> RiskParameters:
        """
        Get all risk parameters for a trade.
        
        Returns complete RiskParameters with adaptive values.
        """
        # Adaptive SL/TP
        sl_pct, tp_pct = self.get_adaptive_sl_tp(df, trade_type)
        
        # Kelly position sizing
        if self.use_kelly:
            position_size = self.calculate_kelly_fraction()
        else:
            position_size = self.base_position_size
        
        # Trailing stop
        trailing_pct = sl_pct if self.use_trailing else None
        
        return RiskParameters(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            position_size=position_size,
            trailing_stop_pct=trailing_pct,
            risk_reward_ratio=tp_pct / sl_pct
        )
    
    def get_summary(self) -> Dict:
        """Get summary of current risk parameters."""
        if len(self.trade_history) >= 10:
            wins = [t for t in self.trade_history if t > 0]
            losses = [t for t in self.trade_history if t < 0]
            win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0
            kelly = self.calculate_kelly_fraction()
        else:
            win_rate = 0
            kelly = self.base_position_size
        
        return {
            'trades_recorded': len(self.trade_history),
            'win_rate': win_rate,
            'kelly_fraction': kelly,
            'use_kelly': self.use_kelly,
            'use_trailing': self.use_trailing,
        }
