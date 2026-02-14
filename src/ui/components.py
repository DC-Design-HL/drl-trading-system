"""
Dashboard Components
Reusable UI components for the Streamlit dashboard.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime


def render_header(
    symbol: str = "BTC/USDT",
    price: float = 0.0,
    change_24h: float = 0.0,
    connection_status: bool = True,
):
    """
    Render the dashboard header with live price and status.
    """
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <h1 style='margin: 0; padding: 0;'>
            🤖 DRL Trading System
        </h1>
        """, unsafe_allow_html=True)
        
    with col2:
        change_color = "#26a69a" if change_24h >= 0 else "#ef5350"
        st.markdown(f"""
        <div style='text-align: center; padding: 10px;'>
            <span style='font-size: 24px; font-weight: bold;'>{symbol}</span><br>
            <span style='font-size: 32px; color: white;'>${price:,.2f}</span><br>
            <span style='color: {change_color}; font-size: 16px;'>
                {'+' if change_24h >= 0 else ''}{change_24h:.2f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        status_color = "#26a69a" if connection_status else "#ef5350"
        status_text = "🟢 Connected" if connection_status else "🔴 Disconnected"
        st.markdown(f"""
        <div style='text-align: right; padding: 20px;'>
            <span style='color: {status_color}; font-size: 14px;'>{status_text}</span><br>
            <span style='color: #888; font-size: 12px;'>Binance Testnet</span>
        </div>
        """, unsafe_allow_html=True)


def render_confidence_gauge(confidence: float):
    """
    Render the agent confidence gauge.
    
    Args:
        confidence: Confidence percentage (0-100)
    """
    # Determine color based on confidence
    if confidence >= 70:
        color = "#26a69a"
        label = "High"
    elif confidence >= 40:
        color = "#ffc107"
        label = "Medium"
    else:
        color = "#ef5350"
        label = "Low"
        
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #1e222d 0%, #131722 100%);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a2e39;
    '>
        <div style='color: #888; margin-bottom: 10px;'>Agent Confidence</div>
        <div style='
            background: #0d0d0d;
            border-radius: 50%;
            width: 120px;
            height: 120px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 4px solid {color};
        '>
            <span style='font-size: 28px; color: {color}; font-weight: bold;'>
                {confidence:.0f}%
            </span>
        </div>
        <div style='color: {color}; margin-top: 10px;'>{label} Confidence</div>
    </div>
    """, unsafe_allow_html=True)


def render_pnl_card(
    total_balance: float,
    daily_pnl: float,
    daily_roi: float,
    initial_balance: float = 10000.0,
):
    """
    Render the P&L summary card.
    """
    total_pnl = total_balance - initial_balance
    total_roi = (total_pnl / initial_balance) * 100
    
    pnl_color = "#26a69a" if total_pnl >= 0 else "#ef5350"
    daily_color = "#26a69a" if daily_pnl >= 0 else "#ef5350"
    
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #1e222d 0%, #131722 100%);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2a2e39;
    '>
        <div style='color: #888; margin-bottom: 15px;'>Portfolio Value</div>
        <div style='font-size: 32px; font-weight: bold; color: white;'>
            ${total_balance:,.2f}
        </div>
        
        <div style='display: flex; justify-content: space-between; margin-top: 20px;'>
            <div>
                <div style='color: #888; font-size: 12px;'>Total P&L</div>
                <div style='color: {pnl_color}; font-size: 18px; font-weight: bold;'>
                    {'+' if total_pnl >= 0 else ''}${total_pnl:,.2f}
                </div>
                <div style='color: {pnl_color}; font-size: 14px;'>
                    ({'+' if total_roi >= 0 else ''}{total_roi:.2f}%)
                </div>
            </div>
            <div style='text-align: right;'>
                <div style='color: #888; font-size: 12px;'>Daily P&L</div>
                <div style='color: {daily_color}; font-size: 18px; font-weight: bold;'>
                    {'+' if daily_pnl >= 0 else ''}${daily_pnl:,.2f}
                </div>
                <div style='color: {daily_color}; font-size: 14px;'>
                    ({'+' if daily_roi >= 0 else ''}{daily_roi:.2f}%)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_position_card(position: Optional[Dict[str, Any]]):
    """
    Render the current position card.
    """
    if position is None:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1e222d 0%, #131722 100%);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #2a2e39;
        '>
            <div style='color: #888;'>Current Position</div>
            <div style='font-size: 24px; color: #555; margin-top: 10px;'>No Position</div>
        </div>
        """, unsafe_allow_html=True)
        return
        
    side = position.get('side', 'unknown')
    is_long = side == 'buy'
    color = "#26a69a" if is_long else "#ef5350"
    icon = "📈" if is_long else "📉"
    
    entry_price = position.get('entry_price', 0)
    amount = position.get('amount', 0)
    unrealized_pnl = position.get('unrealized_pnl', 0)
    
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #1e222d 0%, #131722 100%);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid {color};
    '>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div style='color: #888;'>Current Position</div>
            <div style='
                background: {color};
                padding: 4px 12px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            '>{icon} {'LONG' if is_long else 'SHORT'}</div>
        </div>
        
        <div style='margin-top: 15px;'>
            <div style='display: flex; justify-content: space-between;'>
                <span style='color: #888;'>Entry Price:</span>
                <span style='color: white;'>${entry_price:,.2f}</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                <span style='color: #888;'>Size:</span>
                <span style='color: white;'>{amount:.6f}</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                <span style='color: #888;'>Unrealized P&L:</span>
                <span style='color: {"#26a69a" if unrealized_pnl >= 0 else "#ef5350"};'>
                    {'+' if unrealized_pnl >= 0 else ''}${unrealized_pnl:,.2f}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_trade_history(trades: List[Dict], limit: int = 10):
    """
    Render the trade history table.
    """
    st.markdown("""
    <div style='color: #888; margin-bottom: 10px;'>Recent Trades</div>
    """, unsafe_allow_html=True)
    
    if not trades:
        st.info("No trades yet")
        return
        
    # Prepare data for display
    display_trades = trades[-limit:][::-1]  # Most recent first
    
    for trade in display_trades:
        side = trade.get('side', trade.get('position', 0))
        if isinstance(side, int):
            side = 'LONG' if side == 1 else 'SHORT'
        else:
            side = side.upper()
            
        is_long = side == 'LONG' or side == 'BUY'
        color = "#26a69a" if is_long else "#ef5350"
        
        pnl = trade.get('pnl', 0)
        pnl_color = "#26a69a" if pnl >= 0 else "#ef5350"
        
        entry = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        
        st.markdown(f"""
        <div style='
            background: #1e222d;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        '>
            <div>
                <span style='color: {color}; font-weight: bold;'>{side}</span>
                <span style='color: #888; font-size: 12px; margin-left: 10px;'>
                    ${entry:,.2f} → ${exit_price:,.2f}
                </span>
            </div>
            <span style='color: {pnl_color}; font-weight: bold;'>
                {'+' if pnl >= 0 else ''}${pnl:,.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_circuit_breaker_status(status: Dict[str, Any]):
    """
    Render the circuit breaker status indicator.
    """
    is_tripped = status.get('is_tripped', False)
    
    if is_tripped:
        st.error(f"""
        ⚠️ **CIRCUIT BREAKER ACTIVATED**
        
        Reason: {status.get('trip_reason', 'Unknown')}
        
        Cooldown: {status.get('cooldown_remaining', 'Unknown')}
        """)
    else:
        daily_metrics = status.get('daily_metrics', {})
        daily_loss = -daily_metrics.get('daily_return', 0) * 100
        max_loss = status.get('thresholds', {}).get('max_daily_loss', 0.05) * 100
        
        progress = min(daily_loss / max_loss, 1.0) if max_loss > 0 else 0
        
        if progress < 0.5:
            status_color = "#26a69a"
        elif progress < 0.8:
            status_color = "#ffc107"
        else:
            status_color = "#ef5350"
            
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #1e222d 0%, #131722 100%);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #2a2e39;
        '>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: #888;'>Risk Monitor</span>
                <span style='color: {status_color};'>●</span>
            </div>
            <div style='margin-top: 10px;'>
                <div style='
                    background: #0d0d0d;
                    height: 6px;
                    border-radius: 3px;
                    overflow: hidden;
                '>
                    <div style='
                        background: {status_color};
                        width: {progress * 100}%;
                        height: 100%;
                    '></div>
                </div>
                <div style='
                    display: flex;
                    justify-content: space-between;
                    margin-top: 5px;
                    font-size: 12px;
                '>
                    <span style='color: #888;'>Daily Loss: {daily_loss:.2f}%</span>
                    <span style='color: #888;'>Max: {max_loss:.1f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_training_status(status: Dict[str, Any]):
    """
    Render the self-improvement training status.
    """
    is_finetuning = status.get('is_finetuning', False)
    finetune_count = status.get('finetune_count', 0)
    next_finetune = status.get('next_finetune_in', 'Unknown')
    buffer_size = status.get('buffer_size', 0)
    
    if is_finetuning:
        st.info("🔄 Fine-tuning in progress...")
    else:
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #1e222d 0%, #131722 100%);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #2a2e39;
        '>
            <div style='color: #888; margin-bottom: 10px;'>Self-Improvement</div>
            <div style='display: flex; justify-content: space-between;'>
                <span style='color: #888;'>Fine-tunes:</span>
                <span style='color: white;'>{finetune_count}</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                <span style='color: #888;'>Buffer Size:</span>
                <span style='color: white;'>{buffer_size}</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                <span style='color: #888;'>Next Fine-tune:</span>
                <span style='color: #26a69a;'>{next_finetune}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
