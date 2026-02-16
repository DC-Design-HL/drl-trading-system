"""
DRL Trading System - Streamlit Dashboard
Real-time monitoring with TradingView charts, WebSocket live data, and timeframe switching.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.data_loader import DataLoader, BinanceHistoricalDataFetcher
from src.data.storage import get_storage, JsonFileStorage

# Initialize storage with caching
@st.cache_resource
def get_app_storage():
    return get_storage()

storage = get_app_storage()

# Page configuration
st.set_page_config(
    page_title="DRL Trading System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for TradingView dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #131722;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #888;
    }
    .stTabs [aria-selected="true"] {
        color: white;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #131722 100%);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2a2e39;
        margin-bottom: 15px;
    }
    .metric-label {
        color: #888;
        font-size: 12px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    .metric-delta-positive {
        color: #26a69a;
    }
    .metric-delta-negative {
        color: #ef5350;
    }
    /* Timeframe buttons */
    .timeframe-btn {
        background: #1e222d;
        border: 1px solid #2a2e39;
        color: #888;
        padding: 5px 12px;
        margin: 2px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
    }
    .timeframe-btn.active {
        background: #2962FF;
        color: white;
        border-color: #2962FF;
    }
    .timeframe-btn:hover {
        background: #363a45;
    }
</style>
""", unsafe_allow_html=True)


# Timeframe options
TIMEFRAMES = {
    '1m': {'binance': '1m', 'label': '1m', 'days': 1},
    '5m': {'binance': '5m', 'label': '5m', 'days': 2},
    '15m': {'binance': '15m', 'label': '15m', 'days': 5},
    '30m': {'binance': '30m', 'label': '30m', 'days': 7},
    '1h': {'binance': '1h', 'label': '1H', 'days': 14},
    '4h': {'binance': '4h', 'label': '4H', 'days': 30},
    '1d': {'binance': '1d', 'label': '1D', 'days': 180},
}


def load_trading_log(symbol: str = None) -> list:
    """Load real trading data from updated storage."""
    try:
        all_trades = storage.get_trades(limit=500)
        
        if not symbol:
            return all_trades
            
        filtered_trades = []
        for trade in all_trades:
            # Filter by symbol
            trade_symbol = trade.get('symbol', trade.get('asset', 'BTC/USDT'))
            s1 = symbol.replace('/', '').upper()
            s2 = trade_symbol.replace('/', '').upper()
            if s1 in s2 or s2 in s1:
                filtered_trades.append(trade)
        return filtered_trades
    except Exception as e:
        st.error(f"Failed to load trades: {e}")
        return []


def check_pid_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False

def check_process_running(process_name_substr: str) -> bool:
    """Check if a process is running by parsing ps aux output."""
    try:
        import subprocess
        # Run ps aux
        res = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if res.returncode != 0:
            return False
            
        # Check if process name is in output
        for line in res.stdout.splitlines():
            if process_name_substr in line and "grep" not in line:
                return True
        return False
    except:
        return False

def get_last_logs(log_path: Path, lines: int = 50) -> str:
    """Read last N lines of a log file."""
    if not log_path.exists():
        return f"Log file not found: {log_path}"
    try:
        # Use simple file reading for portability
        content = log_path.read_text().splitlines()
        return "\n".join(content[-lines:])
    except Exception as e:
        return f"Error reading logs: {e}"




def get_trading_state(selected_asset: str = None) -> dict:
    """Get current trading state from storage."""
    try:
        # Load state through storage interface
        state = storage.load_state()
        
        if not state:
            return {
                'balance': 0, 
                'realized_pnl': 0, 
                'multi_asset': True,
                'available_assets': []
            }

        # If specific asset selected, return its details mixed with global
        if selected_asset and 'assets' in state and selected_asset in state['assets']:
            asset_state = state['assets'][selected_asset]
            asset_trades = load_trading_log(symbol=selected_asset)
            return {
                'balance': state.get('total_balance', 0),
                'asset_balance': asset_state.get('balance', 0),
                'position': asset_state.get('position', 0),
                'realized_pnl': state.get('total_pnl', 0),
                'asset_pnl': asset_state.get('pnl', 0),
                'trades': asset_trades,
                'total_trades': len([t for t in asset_trades if 'OPEN' in t.get('action', '')]),
                'position_price': asset_state.get('price', 0),
                'position_size_units': asset_state.get('units', 0),
                'price': asset_state.get('price', 0),
                'timestamp': state.get('timestamp'),
                'multi_asset': True,
                'available_assets': list(state.get('assets', {}).keys()),
                'raw_state': state,
                'sl': asset_state.get('sl', 0),
                'tp': asset_state.get('tp', 0)
            }
        
        # Default: return list of assets if no specific one selected (or global view)
        return {
            'balance': state.get('total_balance', 0),
            'realized_pnl': state.get('total_pnl', 0),
            'multi_asset': True,
            'available_assets': list(state.get('assets', {}).keys()),
            'raw_state': state 
        }
    except Exception as e:
        # Fallback for empty init
        return {'balance': 0, 'realized_pnl': 0, 'multi_asset': True}


def create_tradingview_chart_with_websocket(df: pd.DataFrame, trades: list, timeframe: str = '1h', symbol: str = 'BTC/USDT') -> str:
    """Create TradingView Lightweight Charts HTML with WebSocket live updates."""
    if df.empty:
        return "<div style='color: #888; text-align: center; padding: 50px;'>No market data available</div>"
    
    # Convert data to the format expected by Lightweight Charts
    candlestick_data = []
    for idx, row in df.iterrows():
        candlestick_data.append({
            'time': int(idx.timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
        })
    
    volume_data = []
    for idx, row in df.iterrows():
        color = '#26a69a80' if row['close'] >= row['open'] else '#ef535080'
        volume_data.append({
            'time': int(idx.timestamp()),
            'value': float(row['volume']),
            'color': color,
        })
    
    # Create markers for trades
    markers = []
    for trade in trades:
        if 'price' in trade and 'timestamp' in trade:
            try:
                ts = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                action = trade.get('action', '')
                reason = trade.get('reason', 'model')
                
                if 'OPEN_LONG' in action:
                    markers.append({
                        'time': int(ts.timestamp()),
                        'position': 'belowBar',
                        'color': '#26a69a',
                        'shape': 'arrowUp',
                        'text': 'LONG',
                    })
                elif 'OPEN_SHORT' in action:
                    markers.append({
                        'time': int(ts.timestamp()),
                        'position': 'aboveBar',
                        'color': '#ef5350',
                        'shape': 'arrowDown',
                        'text': 'SHORT',
                    })
                elif 'CLOSE' in action:
                    # Differentiate exit types
                    if reason == 'stop_loss':
                        markers.append({
                            'time': int(ts.timestamp()),
                            'position': 'aboveBar',
                            'color': '#ff4444',
                            'shape': 'square',
                            'text': 'SL',
                        })
                    elif reason == 'take_profit':
                        markers.append({
                            'time': int(ts.timestamp()),
                            'position': 'aboveBar',
                            'color': '#00ff88',
                            'shape': 'square',
                            'text': 'TP',
                        })
                    else:
                        markers.append({
                            'time': int(ts.timestamp()),
                            'position': 'aboveBar',
                            'color': '#ffc107',
                            'shape': 'circle',
                            'text': 'EXIT',
                        })
            except:
                pass
    
    # Get OHLC for display
    last_candle = df.iloc[-1]
    tf_label = TIMEFRAMES.get(timeframe, {}).get('label', timeframe.upper())

    # WebSocket stream name for Binance
    # Symbol needs to be lowercase and without /
    clean_symbol = symbol.replace('/', '').lower()
    ws_stream = f"{clean_symbol}@kline_{timeframe}"
    chart_id = f"chart_{clean_symbol}_{timeframe}"
    
    html = f"""
    <div id="tv-chart-container" style="width: 100%; height: 550px; position: relative; background: #131722;">
        <!-- OHLC and Price Display -->
        <div id="chart-header" style="
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 100;
            font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif;
        ">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="color: white; font-size: 16px; font-weight: bold;">{symbol}</span>
                <span style="color: #888; font-size: 13px;">{tf_label}</span>
                <span id="live-indicator" style="
                    display: inline-flex;
                    align-items: center;
                    gap: 5px;
                    color: #26a69a;
                    font-size: 11px;
                ">
                    <span style="
                        width: 8px;
                        height: 8px;
                        background: #26a69a;
                        border-radius: 50%;
                        animation: pulse 2s infinite;
                    "></span>
                    LIVE
                </span>
            </div>
            <div id="ohlc-display" style="
                margin-top: 5px;
                font-size: 12px;
                color: #d1d4dc;
            ">
                <span style="color: #888;">O</span> <span id="o-val">{last_candle['open']:.2f}</span>
                <span style="color: #888; margin-left: 10px;">H</span> <span id="h-val">{last_candle['high']:.2f}</span>
                <span style="color: #888; margin-left: 10px;">L</span> <span id="l-val">{last_candle['low']:.2f}</span>
                <span style="color: #888; margin-left: 10px;">C</span> <span id="c-val">{last_candle['close']:.2f}</span>
                <span id="change-val" style="margin-left: 15px;"></span>
            </div>
        </div>
        
        <!-- Current Price Label -->
        <div id="current-price" style="
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
            text-align: right;
            font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif;
        ">
            <div id="price-value" style="font-size: 28px; font-weight: bold; color: white;">
                ${last_candle['close']:,.2f}
            </div>
            <div id="price-change" style="font-size: 14px; color: #26a69a;"></div>
        </div>
        
        <div id="{chart_id}" style="width: 100%; height: 550px;"></div>
    </div>
    
    <style>
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
    </style>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script>
        (function() {{
            const container = document.getElementById('{chart_id}');
            
            const chart = LightweightCharts.createChart(container, {{
                width: container.clientWidth,
                height: 550,
                layout: {{
                    background: {{ type: 'solid', color: '#131722' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: '#1e222d' }},
                    horzLines: {{ color: '#1e222d' }},
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                    vertLine: {{
                        color: '#758696',
                        width: 1,
                        style: LightweightCharts.LineStyle.Dashed,
                        labelBackgroundColor: '#2962FF',
                    }},
                    horzLine: {{
                        color: '#758696',
                        width: 1,
                        style: LightweightCharts.LineStyle.Dashed,
                        labelBackgroundColor: '#2962FF',
                    }},
                }},
                rightPriceScale: {{
                    borderColor: '#2a2e39',
                    scaleMargins: {{
                        top: 0.1,
                        bottom: 0.2,
                    }},
                }},
                timeScale: {{
                    borderColor: '#2a2e39',
                    timeVisible: true,
                    secondsVisible: false,
                }},
            }});
            
            // Candlestick series
            const candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderDownColor: '#ef5350',
                borderUpColor: '#26a69a',
                wickDownColor: '#ef5350',
                wickUpColor: '#26a69a',
            }});
            
            let candleData = {json.dumps(candlestick_data)};
            candlestickSeries.setData(candleData);
            
            // Add markers for trades
            const markers = {json.dumps(markers)};
            if (markers.length > 0) {{
                candlestickSeries.setMarkers(markers);
            }}
            
            // Volume series
            const volumeSeries = chart.addHistogramSeries({{
                priceFormat: {{
                    type: 'volume',
                }},
                priceScaleId: 'volume',
            }});
            
            chart.priceScale('volume').applyOptions({{
                scaleMargins: {{
                    top: 0.85,
                    bottom: 0,
                }},
            }});
            
            let volumeData = {json.dumps(volume_data)};
            volumeSeries.setData(volumeData);
            
            // Update OHLC on crosshair move
            chart.subscribeCrosshairMove((param) => {{
                if (param.time) {{
                    const data = param.seriesData.get(candlestickSeries);
                    if (data) {{
                        document.getElementById('o-val').textContent = data.open.toFixed(2);
                        document.getElementById('h-val').textContent = data.high.toFixed(2);
                        document.getElementById('l-val').textContent = data.low.toFixed(2);
                        document.getElementById('c-val').textContent = data.close.toFixed(2);
                        
                        const change = ((data.close - data.open) / data.open * 100).toFixed(2);
                        const changeEl = document.getElementById('change-val');
                        changeEl.textContent = (change >= 0 ? '+' : '') + change + '%';
                        changeEl.style.color = change >= 0 ? '#26a69a' : '#ef5350';
                    }}
                }}
            }});
            
            // Fit content
            chart.timeScale().fitContent();
            
            // Resize handler
            new ResizeObserver(entries => {{
                chart.applyOptions({{ width: entries[0].contentRect.width }});
            }}).observe(container);
            
            // WebSocket for live updates
            let ws;
            let reconnectInterval = 5000;
            let lastCandle = candleData[candleData.length - 1];
            
            function connectWebSocket() {{
                ws = new WebSocket('wss://stream.binance.com:9443/ws/{ws_stream}');
                
                ws.onopen = function() {{
                    console.log('WebSocket connected');
                    document.getElementById('live-indicator').style.display = 'inline-flex';
                }};
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    const kline = data.k;
                    
                    const candle = {{
                        time: Math.floor(kline.t / 1000),
                        open: parseFloat(kline.o),
                        high: parseFloat(kline.h),
                        low: parseFloat(kline.l),
                        close: parseFloat(kline.c),
                    }};
                    
                    // Update or add candle
                    candlestickSeries.update(candle);
                    
                    // Update volume
                    const volColor = candle.close >= candle.open ? '#26a69a80' : '#ef535080';
                    volumeSeries.update({{
                        time: candle.time,
                        value: parseFloat(kline.v),
                        color: volColor,
                    }});
                    
                    // Update price display
                    const priceEl = document.getElementById('price-value');
                    const changeEl = document.getElementById('price-change');
                    
                    priceEl.textContent = '$' + candle.close.toLocaleString('en-US', {{
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    }});
                    
                    // Calculate 24h change (approximation from last candle)
                    if (lastCandle) {{
                        const change = ((candle.close - lastCandle.open) / lastCandle.open * 100);
                        changeEl.textContent = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';
                        changeEl.style.color = change >= 0 ? '#26a69a' : '#ef5350';
                        priceEl.style.color = change >= 0 ? '#26a69a' : '#ef5350';
                    }}
                    
                    // Share price with sidebar via localStorage
                    localStorage.setItem('btc_live_price', candle.close.toFixed(2));
                    
                    // Update OHLC display for current candle
                    document.getElementById('o-val').textContent = candle.open.toFixed(2);
                    document.getElementById('h-val').textContent = candle.high.toFixed(2);
                    document.getElementById('l-val').textContent = candle.low.toFixed(2);
                    document.getElementById('c-val').textContent = candle.close.toFixed(2);
                }};
                
                ws.onerror = function(error) {{
                    console.error('WebSocket error:', error);
                }};
                
                ws.onclose = function() {{
                    console.log('WebSocket closed, reconnecting...');
                    document.getElementById('live-indicator').style.display = 'none';
                    setTimeout(connectWebSocket, reconnectInterval);
                }};
            }}
            
            connectWebSocket();
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', function() {{
                if (ws) ws.close();
            }});
        }})();
    </script>
    """
    return html


def render_position_card(state: dict, current_price: float):
    """Render current position card."""
    position = state.get('position', 0)
    
    # SL/TP percentages (match live trading config)
    SL_PCT = 0.015  # 1.5% (matches live_trading.py)
    TP_PCT = 0.025  # 2.5% (matches live_trading.py)
    
    if position == 0:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Current Position</div>
            <div style="font-size: 24px; color: #555; margin-top: 10px;">No Position</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        is_long = position == 1
        color = "#26a69a" if is_long else "#ef5350"
        side = "LONG" if is_long else "SHORT"
        icon = "📈" if is_long else "📉"
        
        # Get entry price from state
        entry_price = state.get('position_price', current_price)
        
        # Get SL/TP from state (preferred) or calculate
        sl_price = state.get('sl', 0)
        tp_price = state.get('tp', 0)
        
        if sl_price == 0 or tp_price == 0:
            # Fallback to estimation
            if is_long:
                sl_price = entry_price * (1 - SL_PCT)
                tp_price = entry_price * (1 + TP_PCT)
            else:
                sl_price = entry_price * (1 + SL_PCT)
                tp_price = entry_price * (1 - TP_PCT)
        
        # Calculate Unrealized PnL
        if is_long:
            unrealized_pnl = (current_price - entry_price) * state.get('position_size_units', 0)
        else:
            unrealized_pnl = (entry_price - current_price) * state.get('position_size_units', 0)
        
        pnl_color = "#26a69a" if unrealized_pnl >= 0 else "#ef5350"
        pnl_sign = "+" if unrealized_pnl >= 0 else ""
        
        st.markdown(f"""
        <div class="metric-card" style="border: 1px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">Current Position</span>
                <span style="
                    background: {color};
                    padding: 4px 12px;
                    border-radius: 4px;
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                ">{icon} {side}</span>
            </div>
            <div style="margin-top: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #888;">Entry Price:</span>
                    <span style="color: white;">${entry_price:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #888;">Current Price:</span>
                    <span id="sidebar-current-price" style="color: white;">${current_price:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #888;">Unrealized P&L:</span>
                    <span id="sidebar-pnl" style="color: {pnl_color}; font-weight: bold;">{pnl_sign}${unrealized_pnl:,.2f}</span>
                </div>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #ef5350;">🛑 Stop Loss:</span>
                        <span style="color: #ef5350;">${sl_price:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #26a69a;">🎯 Take Profit:</span>
                        <span style="color: #26a69a;">${tp_price:,.2f}</span>
                    </div>
                </div>
            </div>
        </div>
        <script>
            // Real-time price update from WebSocket via localStorage
            const entryPrice = {entry_price};
            const positionUnits = {state.get('position_size_units', 0)};
            const isLong = {'true' if is_long else 'false'};
            
            function updateSidebarPrice() {{
                const livePrice = parseFloat(localStorage.getItem('btc_live_price'));
                if (livePrice && livePrice > 0) {{
                    // Update current price
                    const priceEl = document.getElementById('sidebar-current-price');
                    if (priceEl) priceEl.textContent = '$' + livePrice.toLocaleString('en-US', {{minimumFractionDigits: 2}});
                    
                    // Update unrealized P&L
                    let pnl = isLong ? (livePrice - entryPrice) * positionUnits : (entryPrice - livePrice) * positionUnits;
                    const pnlEl = document.getElementById('sidebar-pnl');
                    if (pnlEl) {{
                        pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
                        pnlEl.style.color = pnl >= 0 ? '#26a69a' : '#ef5350';
                    }}
                }}
            }}
            
            // Update every 500ms
            setInterval(updateSidebarPrice, 500);
            updateSidebarPrice();
        </script>
        """, unsafe_allow_html=True)


def render_trade_history(trades: list):
    """Render real trade history."""
    st.markdown('<div class="metric-label">Recent Trades</div>', unsafe_allow_html=True)
    
    action_trades = [t for t in trades if 'action' in t and t['action'] != 'HOLD']
    
    if not action_trades:
        st.info("No trades yet")
        return
    
    for trade in reversed(action_trades[-10:]):
        action = trade.get('action', '')
        price = trade.get('price', 0)
        pnl = trade.get('pnl', 0)
        timestamp = trade.get('timestamp', '')
        reason = trade.get('reason', 'model')
        
        try:
            ts = datetime.fromisoformat(timestamp)
            time_str = ts.strftime('%m/%d %H:%M')
        except:
            time_str = ''
        
        # Determine display based on action and reason
        if 'OPEN_LONG' in action:
            color = "#26a69a"
            side = "LONG"
        elif 'OPEN_SHORT' in action:
            color = "#ef5350"
            side = "SHORT"
        elif 'CLOSE' in action:
            if reason == 'stop_loss':
                color = "#ff4444"
                side = "SL"
            elif reason == 'take_profit':
                color = "#00ff88"
                side = "TP"
            else:
                color = "#ffc107"
                side = "EXIT"
        else:
            color = "#888"
            side = action
        
        pnl_color = "#26a69a" if pnl >= 0 else "#ef5350"
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_display = f"{pnl_sign}${pnl:,.2f}" if pnl != 0 else ""
        
        st.markdown(f"""
        <div style="
            background: #1e222d;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <span style="color: {color}; font-weight: bold;">{side}</span>
                <span style="color: #888; font-size: 12px; margin-left: 10px;">${price:,.2f}</span>
                <span style="color: #555; font-size: 10px; margin-left: 10px;">{time_str}</span>
            </div>
            <span style="color: {pnl_color}; font-weight: bold;">{pnl_display}</span>
        </div>
        """, unsafe_allow_html=True)


def load_real_market_data(symbol: str = 'BTC/USDT', timeframe: str = '1h') -> pd.DataFrame:
    """Load real market data from Binance."""
    try:
        fetcher = BinanceHistoricalDataFetcher()
        end_date = datetime.now()
        days = TIMEFRAMES.get(timeframe, {}).get('days', 7)
        start_date = end_date - timedelta(days=days)
        
        # Ensure symbol format is correct (e.g. BTC/USDT)
        if "USDT" in symbol and "/" not in symbol:
            symbol = symbol.replace("USDT", "/USDT")
            
        df = fetcher.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        return df
    except Exception as e:
        st.error(f"Failed to fetch market data for {symbol}: {e}")
        return pd.DataFrame()



@st.fragment(run_every=15)
def render_sidebar_metrics_fragment():
    """Render sidebar portfolio metrics with auto-refresh."""
    import requests
    import logging
    logger = logging.getLogger(__name__)
    try:
        # Fetch State
        try:
            state_resp = requests.get('http://127.0.0.1:5001/api/state', timeout=1)
            if state_resp.status_code == 200:
                api_state = state_resp.json()
                # Update session state with API data (optional, but good for other parts)
                if 'balance' in api_state:
                     st.session_state.portfolio_balance = api_state.get('balance', 0)
                     st.session_state.realized_pnl = api_state.get('realized_pnl', 0)
            
            # Render
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value">${st.session_state.get('portfolio_balance', 10000):,.2f}</div>
                <div class="metric-delta" style="color: {'#26a69a' if st.session_state.get('realized_pnl', 0) >= 0 else '#ef5350'}">
                    P&L: {'+' if st.session_state.get('realized_pnl', 0) >= 0 else ''}${st.session_state.get('realized_pnl', 0):,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"<div style='color: #ef5350'>Connection Error</div>", unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Sidebar data fetch error: {e}")

@st.fragment(run_every=15)
def render_market_analysis_fragment(symbol: str):
    """Render market analysis panel with auto-refresh."""
    import requests
    import logging
    logger = logging.getLogger(__name__)
    
    st.markdown("### 📊 Market Analysis")
    
    # Fetch Market Analysis for current asset
    market_data = {}
    try:
        api_symbol = symbol.replace('/', '').upper()
        market_resp = requests.get(f'http://127.0.0.1:5001/api/market?symbol={api_symbol}', timeout=5)
        if market_resp.status_code == 200:
            market_data = market_resp.json()
    except Exception as e:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Market Analysis</div>
            <div style="color: #ef5350; font-size: 12px;">Unable to load (API server offline?)</div>
            <div style="color: #888; font-size: 10px; margin-top:5px;">Error: {str(e)}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Whale Tracker
    whale = market_data.get('whale', {})
    if whale:
        if whale.get('error'):
             st.markdown(f"""
             <div class="metric-card">
                 <div class="metric-label">🐋 Whale Signals</div>
                 <div style="color: #ef5350; font-size: 12px;">Data Error</div>
                 <div style="color: #888; font-size: 10px;">{whale.get('error')}</div>
             </div>
             """, unsafe_allow_html=True)
        else:
            whale_color = "#26a69a" if whale.get('score', 0) > 0 else "#ef5350" if whale.get('score', 0) < 0 else "#888"
            whale_emoji = "🟢" if whale.get('score', 0) > 0.1 else "🔴" if whale.get('score', 0) < -0.1 else "⚪"
            
            # Format Flow Metrics
            flow_metrics = whale.get('flow_metrics', {})
            net_flow = flow_metrics.get('net_flow', 0)
            flow_color = "#26a69a" if net_flow > 0 else "#ef5350"
            flow_sign = "+" if net_flow > 0 else "-"
            
            # Format to K or M
            if abs(net_flow) > 1000000:
                flow_str = f"{flow_sign}${abs(net_flow)/1000000:.1f}M"
            elif abs(net_flow) > 1000:
                flow_str = f"{flow_sign}${abs(net_flow)/1000:.0f}K"
            else:
                flow_str = "$0"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🐋 Whale Signals</div>
                <div style="color: {whale_color}; font-size: 14px;">{whale_emoji} {whale.get('direction', 'NEUTRAL')}</div>
                <div style="color: #888; font-size: 11px;">
                    Score: {whale.get('score', 0):.2f} | Conf: {whale.get('confidence', 0)}%<br>
                    Flow (1m): <span style="color: {flow_color}; font-weight: bold;">{flow_str}</span><br>
                    🟢{whale.get('bullish', 0)} 🔴{whale.get('bearish', 0)} ⚪{whale.get('neutral', 0)}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Funding
    funding_data = market_data.get('funding', {})
    funding = funding_data.get('data', {}) # structure varies, being safe
    if funding_data and not funding_data.get('error'):
         # Extract funding rate
         rate = funding_data.get('rate', 0)
         funding_color = "#26a69a" if rate > 0.0001 else "#ef5350" if rate < -0.0001 else "#888"
         
         st.markdown(f"""
         <div class="metric-card">
             <div class="metric-label">💰 Funding Rate</div>
             <div style="color: {funding_color}; font-size: 14px;">{rate:.4f}%</div>
             <div style="color: #888; font-size: 11px;">
                 Bias: {funding_data.get('bias', 'neutral')} | APR: {funding_data.get('annualized', 0):.1f}%
             </div>
         </div>
         """, unsafe_allow_html=True)
    
    # Order Flow
    order_flow = market_data.get('order_flow', {})
    if order_flow and not order_flow.get('error'):
        of_bias = order_flow.get('bias', 'neutral')
        of_color = "#26a69a" if of_bias == 'bullish' else "#ef5350" if of_bias == 'bearish' else "#888"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Order Flow</div>
            <div style="color: {of_color}; font-size: 14px;">{of_bias.upper()}</div>
            <div style="color: #888; font-size: 11px;">
                Buys: {order_flow.get('large_buys', 0)} | Sells: {order_flow.get('large_sells', 0)}
            </div>
        </div>
        """, unsafe_allow_html=True)

@st.fragment(run_every=15)
def render_position_fragment(symbol: str):
    """Render current position and portfolio status with auto-refresh."""
    import requests
    import os
    from datetime import datetime
    import logging
    logger = logging.getLogger(__name__)
    
    # 1. Fetch Trading State
    state = {}
    try:
        state_resp = requests.get('http://127.0.0.1:5001/api/state', timeout=1)
        if state_resp.status_code == 200:
            state = state_resp.json()
    except Exception as e:
        logger.error(f"State fetch error: {e}")

    # 2. Fetch Live Price (Fast, from API or Fallback)
    current_price = 0.0
    try:
        # Try to get price from market API first (faster)
        clean_symbol = symbol.replace('/', '').upper()
        market_resp = requests.get(f'http://127.0.0.1:5001/api/market?symbol={clean_symbol}', timeout=2)
        if market_resp.status_code == 200:
            m_data = market_resp.json()
            if 'price' in m_data:
                current_price = float(m_data['price'])
                
        # Fallback if API didn't return price
        if current_price == 0:
            live_data = load_real_market_data(symbol, '1m')
            if not live_data.empty:
                current_price = float(live_data.iloc[-1]['close'])
            else:
                 live_1h = load_real_market_data(symbol, '1h')
                 if not live_1h.empty:
                     current_price = float(live_1h.iloc[-1]['close'])
    except Exception as e:
        logger.error(f"Price fetch error: {e}")

    # 3. Render Portfolio Value & P&L
    # Handle both top-level keys (from app.py compatibility) and nested structure
    balance = state.get('total_balance', state.get('balance', 10000))
    total_pnl = state.get('total_pnl', state.get('realized_pnl', 0))
    pnl_class = "metric-delta-positive" if total_pnl >= 0 else "metric-delta-negative"
    pnl_sign = "+" if total_pnl >= 0 else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Portfolio Value</div>
        <div class="metric-value">${balance:,.2f}</div>
        <div class="{pnl_class}">P&L: {pnl_sign}${total_pnl:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Render Position Card
    # Extract specific asset state from global state
    asset_state = {}
    if 'assets' in state:
        # Try exact match or cleaned match
        clean_symbol = symbol.replace('/', '').upper()
        # Check standard symbol, cleaned, or slashed
        if symbol in state['assets']:
            asset_state = state['assets'][symbol]
        elif clean_symbol in state['assets']:
            asset_state = state['assets'][clean_symbol]
    
    # If not found, fall back to global state (in case API returns single asset state)
    if not asset_state and 'position' in state:
        asset_state = state
        
    render_position_card(asset_state, current_price)
    
    # 5. Render Trade History
    # Filter trades for this symbol if they are mixed
    trades = state.get('trades', [])
    if not trades and asset_state:
         trades = asset_state.get('trades', [])
         
    render_trade_history(trades)


@st.fragment(run_every=15)
def render_agent_status_fragment():
    """Render active agent status and model info with auto-refresh."""
    import requests
    import os
    from datetime import datetime
    import logging
    logger = logging.getLogger(__name__)
    
    # Fetch State
    state = {}
    try:
        state_resp = requests.get('http://127.0.0.1:5001/api/state', timeout=1)
        if state_resp.status_code == 200:
            state = state_resp.json()
    except Exception as e:
        pass
        
    # Model Info
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'data' / 'models' / 'ultimate_agent.zip'
    model_exists = model_path.exists()
    
    # Calculate actual win rate from trades
    trades = state.get('trades', [])
    if trades:
        winning = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total = len(trades)
        win_rate = (winning / total * 100) if total > 0 else 0
    else:
        win_rate = 0
    
    # Calculate return from balance
    balance = state.get('balance', 10000)
    total_return = ((balance - 10000) / 10000) * 100
    
    # Get model last modified time
    if model_exists:
        try:
            model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
            model_date = model_mtime.strftime("%Y-%m-%d")
        except:
            model_date = "Unknown"
    else:
        model_date = "Not found"
    
    return_color = "#26a69a" if total_return >= 0 else "#ef5350"
    return_sign = "+" if total_return >= 0 else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Model</div>
        <div style="color: white; font-size: 14px; margin-top: 5px;">Ultimate Agent (PPO)</div>
        <div style="color: {return_color}; font-size: 12px;">{return_sign}{total_return:.2f}% Return | {win_rate:.1f}% Win Rate</div>
        <div style="color: #888; font-size: 11px;">Trades: {len(trades)} | Model: {model_date}</div>
        <div style="color: {'#26a69a' if model_exists else '#ef5350'}; font-size: 11px;">{'✓ Model loaded' if model_exists else '✗ Model not found'}</div>
    </div>
    """, unsafe_allow_html=True)




def on_asset_change():
    """Callback for asset selection change."""
    # Clear stale market analysis to trigger fresh fetch in fragments
    st.session_state.market_analysis = None
    # Optional: Reset other asset-specific state if needed

def main():
    """Main application entry point."""
    
    # Initialize session state for timeframe
    if 'timeframe' not in st.session_state:
        st.session_state.timeframe = '1h'
    
    # Initialize session state for selected asset
    if 'selected_asset' not in st.session_state:
        st.session_state.selected_asset = 'BTCUSDT'
        
    # Check for multi-asset state to populate selector
    state_preview = get_trading_state()
    available_assets = state_preview.get('available_assets', ['BTCUSDT'])
    
    # Initialize session state for auto-refresh (kept for toggle state only)
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Asset Selector
        if len(available_assets) > 1:
            st.session_state.selected_asset = st.selectbox(
                "Select Asset",
                available_assets,
                index=available_assets.index(st.session_state.selected_asset) if st.session_state.selected_asset in available_assets else 0,
                on_change=on_asset_change
            )
        else:
            st.markdown(f"**Asset:** {st.session_state.selected_asset}")
            
        st.divider()
        
        st.markdown("### 🐞 Debug")
        if st.checkbox("Show Crash Log"):
            log_path = project_root / "crash.log"
            if log_path.exists():
                st.error("⚠️ Crash Log Found")
                with open(log_path, "r") as f:
                    st.text_area("Log Content", f.read(), height=300)
            else:
                st.success("✅ No crash log found")
                
        if st.checkbox("Show Process Log (Stdout/Stderr)"):
            proc_log = project_root / "process.log"
            if proc_log.exists():
                with open(proc_log, "r") as f:
                    st.text_area("Process Output", f.read(), height=300)
            else:
                st.warning("⚠️ process.log not found (yet)")

        if st.checkbox("Show API Server Log"):
            api_log = project_root / "api_server.log"
            if api_log.exists():
                with open(api_log, "r") as f:
                    st.text_area("API Server Output", f.read(), height=300)
            else:
                st.warning("⚠️ api_server.log not found (yet)")
                
        # Debug Storage Path
        if isinstance(storage, JsonFileStorage):
            st.code(f"Storage Path: {storage.state_file.absolute()}")
            st.code(f"Exists: {storage.state_file.exists()}")
            if storage.state_file.exists():
                st.code(f"Size: {storage.state_file.stat().st_size} bytes")
                
        if st.checkbox("Show System Inspector"):
            st.markdown("#### 🕵️ System Inspector")
            
            if st.button("List Processes (ps aux)"):
                try:
                    import subprocess
                    # Use 'ps aux' for more details, or 'ps -ef'
                    res = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                    st.code(res.stdout if res.returncode == 0 else res.stderr)
                except Exception as e:
                    st.error(f"Failed to run ps: {e}")
            
            if st.button("List Files (ls -R)"):
                try:
                    import subprocess
                    res = subprocess.run(['ls', '-R'], capture_output=True, text=True)
                    st.code(res.stdout if res.returncode == 0 else res.stderr)
                except Exception as e:
                    st.error(f"Failed to run ls: {e}")
            
            if st.button("Check Connectivity (ping google.com)"):
                try:
                    import subprocess
                    res = subprocess.run(['ping', '-c', '3', 'google.com'], capture_output=True, text=True)
                    st.code(res.stdout if res.returncode == 0 else res.stderr)
                except Exception as e:
                    st.error(f"Ping failed: {e}")

        st.markdown("### 🔑 API Status")
        eth_key = os.environ.get("ETHERSCAN_API_KEY")
        sol_key = os.environ.get("SOLSCAN_API_KEY")
        xrp_key = os.environ.get("XRPSCAN_API_KEY")
        
        st.caption(f"ETH: {'✅ Set' if eth_key else '❌ Missing'}")
        st.caption(f"SOL: {'✅ Set' if sol_key else '❌ Missing'}")
        st.caption(f"XRP: {'✅ Set' if xrp_key else '⚪ Optional (Public)'}")

        
    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"# 🤖 DRL Trading System - {st.session_state.selected_asset}")
    
    with col2:
        refresh_status = "🔄 Auto (10s)" if st.session_state.auto_refresh else "⏸️ Paused"
        st.markdown(f"""
        <div style="text-align: right; padding-top: 10px;">
            <span style="color: #26a69a; font-size: 14px;">🟢 Connected</span><br>
            <span style="color: #888; font-size: 12px;">{refresh_status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.toggle("Auto Refresh", value=st.session_state.auto_refresh)
    
    # Data fetching is now handled inside fragments (render_sidebar_metrics_fragment, render_market_analysis_fragment)
    pass
    
    # Render Sidebar Metrics using Fragment
    with st.sidebar:
        render_sidebar_metrics_fragment()
    
    st.divider()
    
    # Main layout
    col_main, col_sidebar = st.columns([3, 1])
    
    with col_main:
        # Timeframe selector
        st.markdown("#### Select Timeframe")
        tf_cols = st.columns(7)
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        for i, tf in enumerate(timeframes):
            with tf_cols[i]:
                label = TIMEFRAMES[tf]['label']
                if st.button(label, key=f"tf_{tf}", use_container_width=True,
                           type="primary" if st.session_state.timeframe == tf else "secondary"):
                    st.session_state.timeframe = tf
                    st.rerun()
        
        # Load data for selected timeframe and asset
        with st.spinner(f"Loading {st.session_state.selected_asset} {st.session_state.timeframe} data..."):
            df = load_real_market_data(st.session_state.selected_asset, st.session_state.timeframe)
            state = get_trading_state(st.session_state.selected_asset)
        
        current_price = float(df.iloc[-1]['close']) if not df.empty else 0
        
        # Tabs
        tab_chart, tab_portfolio, tab_performance, tab_whales, tab_backtest = st.tabs([
            "📊 Live Chart", "📋 Portfolio Details", "📈 Performance", "🐋 On-Chain Whales", "🔬 Backtest"
        ])
        
        with tab_chart:
            # TradingView Chart with WebSocket
            trades = state.get('trades', [])
            chart_html = create_tradingview_chart_with_websocket(df, trades, st.session_state.timeframe, st.session_state.selected_asset)
            # Append timestamp comment to force re-render since components.html doesn't support key
            # Create a placeholder for the chart to force re-rendering
            chart_placeholder = st.empty()
            
            # Append timestamp comment to force re-render since components.html doesn't support key
            current_time = time.time()
            chart_html += f"<!-- {current_time} -->"
            
            with chart_placeholder:
                components.html(chart_html, height=600)
                


            # Info about trade markers
            num_trades = len([t for t in trades if 'OPEN' in t.get('action', '')])
            st.caption(f"📍 {num_trades} trade signals on chart • Switch timeframes to see trades at different intervals")
            
            # Trading Controls Section
            st.markdown("---")
            st.markdown("### 🎮 Trading Controls")
            
            # Bot status check
            import subprocess
            bot_running = False
            try:
                result = subprocess.run(['pgrep', '-f', 'live_trading.py'], capture_output=True, text=True)
                bot_running = result.returncode == 0
            except:
                pass
            
            # Status indicator
            if bot_running:
                st.success("🟢 **Trading Bot is RUNNING** (Dry-Run Mode)")
            else:
                st.warning("🟠 **Trading Bot is STOPPED**")
            
            # Control buttons row
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
            
            with ctrl_col1:
                if not bot_running:
                    if st.button("▶️ Start Trading", key="start_trading", use_container_width=True, type="primary"):
                        try:
                            subprocess.Popen(
                                ['./venv/bin/python', 'live_trading.py', '--dry-run', '--interval', '5'],
                                cwd=str(project_root),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            st.success("✓ Trading bot started!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to start: {e}")
                else:
                    if st.button("⏹️ Stop Trading", key="stop_trading", use_container_width=True, type="secondary"):
                        try:
                            subprocess.run(['pkill', '-f', 'live_trading.py'], check=False)
                            st.info("✓ Trading bot stopped")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to stop: {e}")
            
            with ctrl_col2:
                if st.button("📈 Open Long", key="open_long", use_container_width=True):
                    # Manual long entry
                    trade = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'OPEN_LONG',
                        'price': current_price,
                        'pnl': 0,
                        'balance': state.get('balance', 10000),
                        'position': 1,
                        'reason': 'manual',
                        'symbol': st.session_state.selected_asset,
                        'asset': st.session_state.selected_asset
                    }
                    storage.log_trade(trade)
                    st.success(f"✓ Opened LONG @ ${current_price:,.2f}")
                    time.sleep(0.5)
                    st.rerun()
            
            with ctrl_col3:
                if st.button("📉 Open Short", key="open_short", use_container_width=True):
                    # Manual short entry
                    trade = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'OPEN_SHORT',
                        'price': current_price,
                        'pnl': 0,
                        'balance': state.get('balance', 10000),
                        'position': -1,
                        'reason': 'manual',
                        'symbol': st.session_state.selected_asset,
                        'asset': st.session_state.selected_asset
                    }
                    storage.log_trade(trade)
                    st.success(f"✓ Opened SHORT @ ${current_price:,.2f}")
                    time.sleep(0.5)
                    st.rerun()
            
            with ctrl_col4:
                if st.button("🚪 Close Position", key="close_position", use_container_width=True):
                    # Close current position
                    position = state.get('position', 0)
                    if position != 0:
                        action = 'CLOSE_LONG' if position == 1 else 'CLOSE_SHORT'
                        trade = {
                            'timestamp': datetime.now().isoformat(),
                            'action': action,
                            'price': current_price,
                            'pnl': 0,  # Would calculate actual P&L in production
                            'balance': state.get('balance', 10000),
                            'position': 0,
                            'reason': 'manual',
                            'symbol': st.session_state.selected_asset,
                            'asset': st.session_state.selected_asset
                        }
                        storage.log_trade(trade)
                        st.success(f"✓ Closed position @ ${current_price:,.2f}")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.info("No position to close")
            
            # Quick actions row
            st.markdown("")
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("🔄 Refresh Data", key="refresh_data", use_container_width=True):
                    st.rerun()
            
            with action_col2:
                if st.button("🗑️ Clear Trade Log", key="clear_log", use_container_width=True):
                    log_file = project_root / 'logs' / 'trading_log.json'
                    state_file = project_root / 'logs' / 'trading_state.json'
                    log_file.write_text('')
                    if state_file.exists():
                        state_file.unlink()
                    st.success("✓ Trade log cleared")
                    time.sleep(0.5)
                    st.rerun()
        
        with tab_portfolio:
            st.markdown("### 📋 Multi-Asset Portfolio Overview")
            
            # Metric Row
            m1, m2, m3, m4 = st.columns(4)
            total_balance = state.get('balance', 0)
            total_pnl = state.get('realized_pnl', 0)
            active_assets_count = len(state.get('available_assets', []))
            
            m1.metric("Total Portfolio Value", f"${total_balance:,.2f}")
            m2.metric("Total Realized P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            m3.metric("Active Assets", active_assets_count)
            is_online = check_process_running("live_trading_multi.py")
            m4.metric("System Status", "🟢 Online" if is_online else "🔴 Offline")
            
            if not is_online:
                st.error("⚠️ System Verification Failed")
                with st.expander("📝 Debug Logs (Process & API)", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.text("Trading Bot Log:")
                        st.code(get_last_logs(project_root / "process.log"), language="text")
                    with c2:
                        st.text("API Server Log:")
                        st.code(get_last_logs(project_root / "api_server.log"), language="text")

            st.divider()
            
            # Asset Table
            if state.get('multi_asset') and state.get('raw_state', {}).get('assets'):
                assets_data = []
                raw_assets = state['raw_state']['assets']
                
                for symbol, data in raw_assets.items():
                    # Calculate metrics
                    position_amt = data.get('position', 0)
                    pos_str = "LONG 🟢" if position_amt > 0 else "SHORT 🔴" if position_amt < 0 else "FLAT ⚪"
                    
                    trades_count = len(data.get('trades', []))
                    last_action = data.get('last_action', 'NONE')
                    
                    assets_data.append({
                        "Asset": symbol,
                        "Price": data.get('price', 0),
                        "Position": pos_str,
                        "Equity": data.get('balance', 0), # This is actually equity per live_trading_multi.py update
                        "PnL ($)": data.get('pnl', 0),
                        "Trades": trades_count,
                        "Last Action": last_action
                    })
                
                if assets_data:
                    df_portfolio = pd.DataFrame(assets_data)
                    
                    # Formatting
                    st.dataframe(
                        df_portfolio,
                        column_config={
                            "Price": st.column_config.NumberColumn(format="$%.2f"),
                            "Equity": st.column_config.NumberColumn(format="$%.2f"),
                            "PnL ($)": st.column_config.NumberColumn(format="$%.2f"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("No asset data available yet.")
            else:
                st.info("Multi-Asset mode not active or data not available.")

        with tab_performance:
            total_pnl = state.get('realized_pnl', 0)
            total_trades = state.get('total_trades', 0)
            balance = state.get('balance', 10000)
            total_return = ((balance - 10000) / 10000) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Return",
                    value=f"{total_return:.2f}%",
                    delta=f"${total_pnl:.2f}"
                )
            with col2:
                st.metric(
                    label="Portfolio Value",
                    value=f"${balance:,.2f}",
                )
            with col3:
                st.metric(
                    label="Total Trades",
                    value=f"{total_trades}",
                )
            with col4:
                st.metric(
                    label="Realized P&L",
                    value=f"${total_pnl:,.2f}",
                )
            
            backtest_file = project_root / 'data' / 'backtest_report.txt'
            if backtest_file.exists():
                st.markdown("### Backtest Results")
                with open(backtest_file, 'r') as f:
                    st.code(f.read())
        
        with tab_whales:
            st.markdown("### 🐋 On-Chain Whale Alerts")
            
            whale_alerts = state.get('whale_alerts', [])
            
            if whale_alerts:
                col1, col2, col3 = st.columns(3)
                col1.metric("Recent Alerts", len(whale_alerts))
                
                eth_whales = len([w for w in whale_alerts if w.get('chain') == 'ETH'])
                xrp_whales = len([w for w in whale_alerts if w.get('chain') == 'XRP'])
                
                col2.metric("ETH Whales", eth_whales)
                col3.metric("XRP Whales", xrp_whales)
                
                st.divider()
                
                for alert in whale_alerts:
                    chain = alert.get('chain', 'UNKNOWN')
                    value = alert.get('value', 0)
                    currency = alert.get('currency', '')
                    link = alert.get('link', '#')
                    ts = alert.get('timestamp', 0)
                    time_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    
                    color = "#627EEA" if chain == 'ETH' else "#23292F"
                    if chain == 'XRP': color = "#00AAE4"
                    if chain == 'SOL': color = "#14F195"
                    
                    st.markdown(f'''
                    <div style="
                        background: #1e222d;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 10px;
                        border-left: 4px solid {color};
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <div>
                            <span style="color: {color}; font-weight: bold; font-size: 14px;">{chain}</span>
                            <span style="color: white; font-size: 16px; font-weight: bold; margin-left: 15px;">
                                {value:,.0f} {currency}
                            </span>
                            <span style="color: #888; font-size: 12px; margin-left: 15px;">
                                {time_str}
                            </span>
                        </div>
                        <div>
                            <a href="{link}" target="_blank" style="
                                background: #2a2e39;
                                color: white;
                                text-decoration: none;
                                padding: 5px 15px;
                                border-radius: 4px;
                                font-size: 12px;
                            ">View TX ↗</a>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("🌊 No whale alerts detected yet. Monitoring blockchain for large movements...")
        
        with tab_backtest:
            st.markdown("### Run Backtest")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now()
                )
            
            if st.button("🚀 Run Backtest", key="run_backtest"):
                st.info("To run backtest, execute in terminal:")
                st.code("python train_advanced.py --evaluate ./data/models/advanced_agent.zip")
    
    with col_sidebar:
        st.markdown("### 🎯 Agent Status")
        
        # Load state for sidebar
        state = get_trading_state(st.session_state.selected_asset)
        
        # Fetch real-time price using 1m data for accuracy
        # Position & Portfolio Fragment (Live 15s updates)
        render_position_fragment(st.session_state.selected_asset)
        
        # Market Analysis Fragment (Live 15s updates)
        render_market_analysis_fragment(st.session_state.selected_asset)
        
        # Agent Status Fragment (Live 15s updates)
        render_agent_status_fragment()

    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #888; font-size: 12px;">
        DRL Trading System v2.1 | Advanced PPO Agent | 
        <span style="color: #26a69a;">●</span> WebSocket Live Data |
        Deployed: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
