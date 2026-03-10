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

# Custom CSS — Premium Dark Theme (matches Live Portfolio aesthetic)
st.markdown("""
<style>
    /* ═══ Foundation ═══ */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* ═══ Sidebar ═══ */
    div[data-testid="stSidebarContent"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
    div[data-testid="stSidebarContent"] .stMarkdown h3 {
        color: #8b949e;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* ═══ Metric Cards (native st.metric) ═══ */
    div[data-testid="stMetric"] {
        background: #151b23;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px 18px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #fff !important;
        font-weight: 700;
    }
    div[data-testid="stMetricDelta"] svg { display: none; }
    
    /* ═══ Custom metric-card class (sidebar panels) ═══ */
    .metric-card {
        background: #151b23;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .metric-label {
        color: #8b949e;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #fff;
    }
    .metric-delta-positive { color: #00e676; }
    .metric-delta-negative { color: #ff5252; }
    
    /* ═══ Tabs ═══ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b949e;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-size: 13px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e6edf3;
        background-color: rgba(255,255,255,0.04);
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        font-weight: 600;
        border-bottom: 2px solid #00e676;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #00e676 !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    /* ═══ Buttons ═══ */
    .stButton > button {
        background: #151b23;
        border: 1px solid #21262d;
        color: #e6edf3;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: #1c2333;
        border-color: #388bfd;
        color: #fff;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: #1a6b3c;
        border-color: #1a6b3c;
        color: #00e676;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #217a45;
        border-color: #00e676;
    }
    
    /* ═══ Inputs, Selects, Date Pickers ═══ */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stDateInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #151b23 !important;
        border-color: #21262d !important;
        color: #e6edf3 !important;
    }
    
    /* ═══ Text Areas ═══ */
    .stTextArea textarea {
        background-color: #151b23 !important;
        border-color: #21262d !important;
        color: #e6edf3 !important;
        border-radius: 6px;
    }
    
    /* ═══ Code Blocks ═══ */
    .stCodeBlock, code, pre {
        background-color: #151b23 !important;
        border: 1px solid #21262d;
        border-radius: 6px;
    }
    
    /* ═══ Expanders ═══ */
    .streamlit-expanderHeader {
        background: #151b23;
        border: 1px solid #21262d;
        border-radius: 6px;
        color: #e6edf3;
    }
    details {
        background: #151b23;
        border: 1px solid #21262d;
        border-radius: 8px;
    }
    
    /* ═══ Dividers ═══ */
    hr {
        border-color: #21262d !important;
    }
    
    /* ═══ Checkboxes & Toggles ═══ */
    .stCheckbox label span {
        color: #8b949e;
    }
    
    /* ═══ Dataframes ═══ */
    .stDataFrame {
        border: 1px solid #21262d;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* ═══ Alerts ═══ */
    .stAlert {
        background: #151b23;
        border: 1px solid #21262d;
        border-radius: 8px;
    }
    
    /* ═══ Caption ═══ */
    .stCaption {
        color: #8b949e !important;
    }
    
    /* ═══ Scrollbar ═══ */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #21262d;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #30363d;
    }
    
    /* ═══ Hide defaults ═══ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ═══ Timeframe buttons (custom) ═══ */
    .timeframe-btn {
        background: #151b23;
        border: 1px solid #21262d;
        color: #8b949e;
        padding: 5px 12px;
        margin: 2px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 12px;
    }
    .timeframe-btn.active {
        background: #1a6b3c;
        color: #00e676;
        border-color: #1a6b3c;
    }
    .timeframe-btn:hover {
        background: #1c2333;
        border-color: #388bfd;
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
    """Load real trading data from updated storage, filtered by reset timestamp."""
    try:
        all_trades = storage.get_trades(limit=500)
        
        # Filter by reset_timestamp if available (hide pre-reset trades)
        try:
            state = storage.load_state()
            reset_ts = state.get('reset_timestamp')
            if reset_ts:
                reset_dt = datetime.fromisoformat(reset_ts.replace('Z', '+00:00'))
                filtered_by_time = []
                for trade in all_trades:
                    try:
                        trade_ts = trade.get('timestamp', '')
                        trade_dt = datetime.fromisoformat(trade_ts.replace('Z', '+00:00'))
                        if trade_dt >= reset_dt:
                            filtered_by_time.append(trade)
                    except:
                        filtered_by_time.append(trade)  # Include if can't parse
                all_trades = filtered_by_time
        except:
            pass
        
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
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Current Position</div>
            <div style="font-size: 24px; color: #555; margin-top: 10px;">No Position (FLAT)</div>
            <div style="font-size: 12px; color: #888; margin-top: 5px;">Current Price: ${current_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # DEBUG: Inspect state to find units key
        # st.write(f"Debug State for P&L: {state}")
        # logger.info(f"Debug State for P&L: {state}") 
        pass
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
        units = state.get('position_size_units', state.get('position_units', state.get('units', 0)))
        if is_long:
            unrealized_pnl = (current_price - entry_price) * units
        else:
            unrealized_pnl = (entry_price - current_price) * units
        
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
        market_resp = requests.get(f'http://127.0.0.1:5001/api/market?symbol={api_symbol}', timeout=15)
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
    
    # Order Flow (enhanced 3-layer)
    order_flow = market_data.get('order_flow', {})
    if order_flow and not order_flow.get('error'):
        of_bias = order_flow.get('bias', 'neutral')
        of_score = order_flow.get('score', 0)
        of_color = "#26a69a" if of_bias == 'bullish' else "#ef5350" if of_bias == 'bearish' else "#888"
        
        # Layer details
        cvd_data = order_flow.get('cvd', {})
        taker_data = order_flow.get('taker', {})
        notable_data = order_flow.get('notable', {})
        
        cvd_trend = cvd_data.get('trend', 'n/a')
        taker_ratio = taker_data.get('ratio', 0.5)
        notable_buys = notable_data.get('large_buys', order_flow.get('large_buys', 0))
        notable_sells = notable_data.get('large_sells', order_flow.get('large_sells', 0))
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Order Flow</div>
            <div style="color: {of_color}; font-size: 14px;">{of_bias.upper()} ({of_score:+.2f})</div>
            <div style="color: #888; font-size: 11px;">
                CVD: {cvd_trend} | Taker Buy: {taker_ratio:.0%}<br/>
                Notable: B:{notable_buys} / S:{notable_sells}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # HMM Regime
    regime_data = market_data.get('regime', {})
    if regime_data and not regime_data.get('error'):
        r_type = regime_data.get('type', 'UNKNOWN')
        # Colors: Green for Bull, Red for Bear, Orange for Breakout, Blue for Range
        r_color = "#26a69a" if "BULL" in r_type else "#ef5350" if "BEAR" in r_type else "#ffa726" if "BREAKOUT" in r_type else "#42a5f5"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👑 Market Regime (HMM)</div>
            <div style="color: {r_color}; font-size: 14px; font-weight: bold;">{r_type.replace('_', ' ')}</div>
            <div style="color: #888; font-size: 11px;">
                ADX: {regime_data.get('adx', 0)} | Volatility: {regime_data.get('volatility', 1.0)}x
            </div>
        </div>
        """, unsafe_allow_html=True)

    # TFT Forecast
    forecast = market_data.get('forecast')
    if forecast:
        ret_4h = forecast.get('return_4h', 0)
        fc_color = "#26a69a" if ret_4h > 0 else "#ef5350" if ret_4h < 0 else "#888"
        fc_sign = "+" if ret_4h > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚀 AI Price Forecast (TFT)</div>
            <div style="color: {fc_color}; font-size: 14px;">4h: {fc_sign}{ret_4h}% | 12h: {forecast.get('return_12h', 0)}%</div>
            <div style="color: #888; font-size: 11px;">
                Consensus: {forecast.get('consensus', 0):.2f} | Confidence: {forecast.get('confidence', 0):.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Ensemble Confidence Engine
    confidence = market_data.get('ensemble_confidence')
    if confidence is not None:
        conf_pct = min(100, max(0, int(confidence * 100)))
        # Map 0-1.0 to 0.25x - 2.0x for UI display (matching the ConfidenceEngine logic roughly)
        mult = 0.25 + 1.75 * confidence if confidence < 0.5 else 1.0 + 1.0 * (confidence - 0.5) * 2  # Approximate for UI
        c_color = "#26a69a" if confidence > 0.6 else "#ffa726" if confidence > 0.35 else "#ef5350"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🧠 Ensemble Agreement</div>
            <div style="color: {c_color}; font-size: 14px;">{conf_pct}% Alignment</div>
            <div style="color: #888; font-size: 11px;">
                Position Size Multiplier: ~{mult:.1f}x
            </div>
            
            <!-- Progress Bar -->
            <div style="width: 100%; background-color: #333; height: 4px; border-radius: 2px; margin-top: 5px;">
                <div style="width: {conf_pct}%; background-color: {c_color}; height: 100%; border-radius: 2px;"></div>
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
        market_resp = requests.get(f'http://127.0.0.1:5001/api/market?symbol={clean_symbol}', timeout=5)
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

    # 3. Fetch ALL Trades early to calculate perfectly mathematically synced global Portfolio Value
    all_trades = []
    try:
        trades_resp = requests.get('http://127.0.0.1:5001/api/trades', timeout=2)
        if trades_resp.status_code == 200:
            all_trades = trades_resp.json()
    except Exception as e:
        logger.error(f"Trades fetch error: {e}")
        
    realized_pnl_total = sum(t.get('pnl', 0) for t in all_trades if 'CLOSE' in t.get('action', '').upper() or 'EXIT' in t.get('action', '').upper())
    
    open_pnl_total = 0.0
    raw_assets = state.get('raw_state', {}).get('assets', {})
    for sym, asset_data in raw_assets.items():
        if asset_data.get('position', 0) != 0:
            open_pnl_total += asset_data.get('pnl', 0)
            
    initial_capital = max(len(raw_assets), 1) * 5000 if raw_assets else 20000
    
    if all_trades or raw_assets:
        total_pnl = realized_pnl_total + open_pnl_total
        balance = initial_capital + total_pnl
    else:
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
        if symbol in state['assets']:
            asset_state = state['assets'][symbol]
        elif clean_symbol in state['assets']:
            asset_state = state['assets'][clean_symbol]
    
    # If not found, fall back to global state (in case API returns single asset state)
    if not asset_state and 'position' in state:
        asset_state = state
        
    # NORMALIZE STATE: Ensure position_price is set for P&L calc
    # Agent state usually has 'price' as entry price
    if asset_state:
        if 'position_price' not in asset_state and 'price' in asset_state:
            asset_state['position_price'] = asset_state['price']
        if 'position_price' not in asset_state and 'entry_price' in asset_state:
            asset_state['position_price'] = asset_state['entry_price']
            
    render_position_card(asset_state, current_price)
    
    # 5. Render Trade History
    # Filter trades for current symbol (already fetched above)
    clean_symbol = symbol.replace('/', '').upper()
    trades = [t for t in all_trades if t.get('symbol', '').replace('/', '').upper() == clean_symbol]
        
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
    
    # Fetch trades explicitly for mathematical accuracy
    all_trades = state.get('trades', [])
    try:
        trades_resp = requests.get('http://127.0.0.1:5001/api/trades', timeout=1)
        if trades_resp.status_code == 200:
            all_trades = trades_resp.json()
    except Exception as e:
        pass
        
    # Calculate real mathematical return
    realized_pnl = sum(t.get('pnl', 0) for t in all_trades if 'CLOSE' in t.get('action', '').upper() or 'EXIT' in t.get('action', '').upper())
    raw_assets = state.get('raw_state', {}).get('assets', {})
    open_pnl = sum(a.get('pnl', 0) for a in raw_assets.values() if a.get('position', 0) != 0)
    total_pnl = realized_pnl + open_pnl
    initial_capital = max(len(raw_assets), 1) * 5000 if raw_assets else 20000
    
    total_return = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
    
    # Calculate actual win rate from trades
    closed_trades = [t for t in all_trades if 'CLOSE' in t.get('action', '').upper() or 'EXIT' in t.get('action', '').upper()]
    if closed_trades:
        winning = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
        total = len(closed_trades)
        win_rate = (winning / total * 100) if total > 0 else 0
    else:
        win_rate = 0
    
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
        <div style="color: #888; font-size: 11px;">Trades: {len(all_trades)} | Model: {model_date}</div>
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
            <span style="color: #00e676; font-size: 14px;">🟢 Connected</span><br>
            <span style="color: #8b949e; font-size: 12px;">{refresh_status}</span>
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
        tab_chart, tab_live_portfolio, tab_performance, tab_whales, tab_backtest = st.tabs([
            "📊 Live Chart", "💼 Live Portfolio", "📈 Performance", "🐋 On-Chain Whales", "🔬 Backtest"
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
                # Check for either script
                result = subprocess.run(['pgrep', '-f', 'live_trading'], capture_output=True, text=True)
                bot_running = result.returncode == 0
            except:
                pass
            
            # Status indicator
            if bot_running:
                st.success("🟢 **Trading Bot is RUNNING** (Multi-Asset Mode)")
            else:
                st.warning("🟠 **Trading Bot is STOPPED**")
            
            # Control buttons row
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
            
            with ctrl_col1:
                if not bot_running:
                    if st.button("▶️ Start Trading", key="start_trading", use_container_width=True, type="primary"):
                        try:
                            # Start Multi-Asset Bot
                            # Log to process.log for UI visibility
                            with open(project_root / "process.log", "a") as log_file:
                                subprocess.Popen(
                                    ['./venv/bin/python', '-u', 'live_trading_multi.py', 
                                     '--assets', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 
                                     '--balance', '5000'],
                                    cwd=str(project_root),
                                    stdout=log_file,
                                    stderr=log_file,
                                )
                            st.success("✓ Multi-Asset Bot started!")
                            time.sleep(2) # Give it time to start
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to start: {e}")
                else:
                    if st.button("⏹️ Stop Trading", key="stop_trading", use_container_width=True, type="secondary"):
                        try:
                            # Kill any trading script
                            subprocess.run(['pkill', '-f', 'live_trading'], check=False)
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
        
        with tab_live_portfolio:
            # ─── Compute portfolio metrics from trade data ───
            all_trades_lp = []
            try:
                all_trades_lp = storage.get_trades(limit=1000)
                # Apply reset filter
                try:
                    lp_state = storage.load_state()
                    reset_ts = lp_state.get('reset_timestamp')
                    if reset_ts:
                        reset_dt = datetime.fromisoformat(reset_ts.replace('Z', '+00:00'))
                        all_trades_lp = [t for t in all_trades_lp if datetime.fromisoformat(t.get('timestamp', '2020-01-01').replace('Z', '+00:00')) >= reset_dt]
                except:
                    pass
            except:
                pass
            
            # Separate by symbol and compute per-asset metrics
            assets_by_symbol = {}
            for t in all_trades_lp:
                sym = t.get('symbol', t.get('asset', 'UNKNOWN'))
                sym = sym.replace('/', '').upper()
                if sym not in assets_by_symbol:
                    assets_by_symbol[sym] = []
                assets_by_symbol[sym].append(t)
            
            # Compute closed P&L, open P&L, win rate
            realized_pnl_total = 0.0
            open_pnl_total = 0.0
            total_closed_trades = 0
            total_winning_trades = 0
            total_open_trades = 0
            equity_points = [0.0]  # Start at 0%
            
            asset_rows = []
            
            raw_state = state.get('raw_state', {})
            raw_assets = raw_state.get('assets', {})
            
            for sym, trades_list in assets_by_symbol.items():
                sorted_trades = sorted(trades_list, key=lambda x: x.get('timestamp', ''))
                
                sym_realized = 0.0
                sym_open_pnl = 0.0
                sym_wins = 0
                sym_closed = 0
                sym_open = 0
                sym_best = None
                sym_worst = None
                sym_status = 'FLAT'
                
                for t in sorted_trades:
                    action = t.get('action', '').upper()
                    pnl = t.get('pnl', 0) or 0
                    
                    if 'CLOSE' in action or 'EXIT' in action:
                        sym_realized += pnl
                        sym_closed += 1
                        if pnl > 0:
                            sym_wins += 1
                        equity_points.append(equity_points[-1] + pnl)
                        # Track best/worst
                        if sym_best is None or pnl > sym_best:
                            sym_best = pnl
                        if sym_worst is None or pnl < sym_worst:
                            sym_worst = pnl
                    elif 'OPEN_LONG' in action:
                        sym_status = 'LONG'
                        sym_open += 1
                    elif 'OPEN_SHORT' in action:
                        sym_status = 'SHORT'
                        sym_open += 1
                
                # Check current state for open P&L
                if sym in raw_assets:
                    asset_data = raw_assets[sym]
                    if asset_data.get('position', 0) != 0:
                        sym_open_pnl = asset_data.get('pnl', 0) or 0
                        sym_status = 'LONG' if asset_data.get('position', 0) > 0 else 'SHORT'
                    else:
                        sym_status = 'FLAT'
                
                realized_pnl_total += sym_realized
                open_pnl_total += sym_open_pnl
                total_closed_trades += sym_closed
                total_winning_trades += sym_wins
                if sym_status != 'FLAT':
                    total_open_trades += 1
                
                # Format display symbol
                display_sym = sym
                if sym.endswith('USDT'):
                    display_sym = sym[:-4] + ' /USDT'
                
                # Get price / equity from raw state
                sym_price = raw_assets.get(sym, {}).get('price', 0)
                # Calculate True Equity mathematically instead of relying on historically corrupted bot.balance
                sym_equity = 5000 + sym_realized + sym_open_pnl
                
                asset_rows.append({
                    'symbol': display_sym,
                    'raw_symbol': sym,
                    'status': sym_status,
                    'price': sym_price,
                    'equity': sym_equity,
                    'pnl': sym_realized + sym_open_pnl,
                    'trades': sym_closed + (1 if sym_status != 'FLAT' else 0),
                    'open_trades': 1 if sym_status != 'FLAT' else 0,
                    'win_rate': (sym_wins / sym_closed * 100) if sym_closed > 0 else 0,
                    'wins': sym_wins,
                    'closed': sym_closed,
                    'best': sym_best,
                    'worst': sym_worst,
                })
            
            # Overall metrics
            initial_capital = max(len(raw_assets), 1) * 5000
            realized_pct = (realized_pnl_total / initial_capital) * 100 if initial_capital > 0 else 0
            open_pct = (open_pnl_total / initial_capital) * 100 if initial_capital > 0 else 0
            overall_win_rate = (total_winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
            total_trades_count = total_closed_trades + total_open_trades
            
            # Calculate Total Balance derived strictly from asset equity sums to ensure consistency
            lp_total_balance = sum([row['equity'] for row in asset_rows]) if asset_rows else initial_capital
            lp_active_assets_count = len(asset_rows) if asset_rows else len(state.get('available_assets', []))
            lp_grand_total_pnl = lp_total_balance - initial_capital
            
            # System status
            is_online = check_process_running("live_trading_multi.py")
            status_dot = '🟢' if is_online else '🔴'
            status_text = 'Online' if is_online else 'Offline'
            status_color = '#00e676' if is_online else '#ff5252'
            
            # Color helpers
            def pnl_color(val):
                return '#00e676' if val >= 0 else '#ff5252'
            
            def pnl_sign(val):
                return '+' if val >= 0 else ''
            
            # ─── Build the Live Portfolio HTML ───
            # Equity curve data for SVG chart (pure inline, no CDN)
            eq_pct = [(p / initial_capital) * 100 if initial_capital > 0 else 0 for p in equity_points]
            
            # Build SVG polyline points
            svg_w = 900
            svg_h = 160
            n_points = len(eq_pct)
            eq_min_val = min(eq_pct) if eq_pct else 0
            eq_max_val = max(eq_pct) if eq_pct else 0
            eq_range_val = max(abs(eq_min_val), abs(eq_max_val), 0.01)
            padding_y = 20  # vertical padding
            
            svg_points = []
            svg_fill_points = []
            for i, val in enumerate(eq_pct):
                x = (i / max(n_points - 1, 1)) * svg_w
                # Map value from [-range, +range] to [svg_h - padding, padding]
                y = svg_h - padding_y - ((val + eq_range_val) / (2 * eq_range_val)) * (svg_h - 2 * padding_y)
                svg_points.append(f"{x:.1f},{y:.1f}")
                svg_fill_points.append(f"{x:.1f},{y:.1f}")
            
            polyline_str = ' '.join(svg_points)
            # Close the fill polygon at bottom
            fill_points = svg_fill_points.copy()
            if fill_points:
                fill_points.append(f"{svg_w:.1f},{svg_h - padding_y:.1f}")
                fill_points.append(f"0,{svg_h - padding_y:.1f}")
            fill_str = ' '.join(fill_points)
            
            last_eq = eq_pct[-1] if eq_pct else 0
            line_color = '#00e676' if last_eq >= 0 else '#ff5252'
            fill_color_start = 'rgba(0,230,118,0.3)' if last_eq >= 0 else 'rgba(255,82,82,0.3)'
            fill_color_end = 'rgba(0,230,118,0.0)' if last_eq >= 0 else 'rgba(255,82,82,0.0)'
            
            # Zero line Y position
            zero_y = svg_h - padding_y - ((0 + eq_range_val) / (2 * eq_range_val)) * (svg_h - 2 * padding_y)
            
            # Last point for dot
            last_x = svg_w if n_points <= 1 else ((n_points - 1) / max(n_points - 1, 1)) * svg_w
            last_y = svg_h - padding_y - ((last_eq + eq_range_val) / (2 * eq_range_val)) * (svg_h - 2 * padding_y)
            
            # Y-axis labels
            top_label = f"+{eq_range_val:.1f}%"
            bot_label = f"-{eq_range_val:.1f}%"
            
            svg_chart = f'''
                <svg width="100%" viewBox="0 0 {svg_w} {svg_h}" preserveAspectRatio="none" style="display:block;">
                    <defs>
                        <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stop-color="{fill_color_start}"/>
                            <stop offset="100%" stop-color="{fill_color_end}"/>
                        </linearGradient>
                    </defs>
                    <!-- Zero line -->
                    <line x1="0" y1="{zero_y:.1f}" x2="{svg_w}" y2="{zero_y:.1f}" stroke="rgba(255,255,255,0.08)" stroke-width="1" stroke-dasharray="4,4"/>
                    <!-- Fill area -->
                    <polygon points="{fill_str}" fill="url(#eqGrad)"/>
                    <!-- Line -->
                    <polyline points="{polyline_str}" fill="none" stroke="{line_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <!-- Last point dot -->
                    <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="{line_color}" stroke="#fff" stroke-width="1.5"/>
                    <!-- Labels -->
                    <text x="{svg_w - 5}" y="{padding_y + 4}" fill="#8b949e" font-size="10" text-anchor="end" font-family="monospace">{top_label}</text>
                    <text x="{svg_w - 5}" y="{svg_h - padding_y + 12}" fill="#8b949e" font-size="10" text-anchor="end" font-family="monospace">{bot_label}</text>
                    <text x="5" y="{zero_y - 4:.1f}" fill="#555" font-size="9" font-family="monospace">0%</text>
                </svg>
            '''
            
            # Build asset rows HTML
            asset_rows_html = ''
            for row in asset_rows:
                # Status badge
                if row['status'] == 'LONG':
                    status_html = '<span style="background:#1b3a26;color:#00e676;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;">● LONG</span>'
                elif row['status'] == 'SHORT':
                    status_html = '<span style="background:#3a1b1b;color:#ff5252;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;">● SHORT</span>'
                else:
                    status_html = '<span style="background:#2a2e39;color:#888;padding:3px 10px;border-radius:4px;font-size:11px;">● —</span>'
                
                # PNL
                pnl_val = row['pnl']
                pnl_pct = (pnl_val / (initial_capital / max(len(asset_rows), 1))) * 100 if initial_capital > 0 else 0
                pnl_html = f'<span style="color:{pnl_color(pnl_val)};font-weight:600;">{pnl_sign(pnl_val)}{pnl_pct:.2f}%</span>'
                pnl_dollar_html = f'<span style="color:{pnl_color(pnl_val)};font-weight:600;font-family:monospace;">{pnl_sign(pnl_val)}${abs(pnl_val):,.2f}</span>'
                
                # Trades
                trades_str = str(row['trades'])
                if row['open_trades'] > 0:
                    trades_str += f' <span style="color:#888;">(+{row["open_trades"]})</span>'
                
                # Win rate bar
                wr = row['win_rate']
                bar_color = '#00e676' if wr >= 50 else '#ff9800' if wr > 0 else '#555'
                wr_html = f'''
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="flex:1;background:#1a1e2a;border-radius:4px;height:8px;overflow:hidden;min-width:60px;">
                            <div style="width:{wr}%;height:100%;background:{bar_color};border-radius:4px;"></div>
                        </div>
                        <span style="color:#ccc;font-size:12px;min-width:35px;">{wr:.0f}%</span>
                    </div>
                '''
                
                # Best
                if row['best'] is not None:
                    best_pct = (row['best'] / (initial_capital / max(len(asset_rows), 1))) * 100
                    best_html = f'<span style="color:#00e676;">{pnl_sign(best_pct)}{best_pct:.2f}%</span>'
                else:
                    best_html = '<span style="color:#555;">—</span>'
                
                # Worst
                if row['worst'] is not None:
                    worst_pct = (row['worst'] / (initial_capital / max(len(asset_rows), 1))) * 100
                    worst_html = f'<span style="color:#ff5252;">{worst_pct:.2f}%</span>'
                else:
                    worst_html = '<span style="color:#555;">—</span>'
                
                asset_rows_html += f'''
                <tr style="border-bottom:1px solid #1a1e2a;">
                    <td style="padding:14px 16px;font-weight:600;color:#fff;font-size:13px;">
                        {row['symbol']}
                    </td>
                    <td style="padding:14px 16px;">{status_html}</td>
                    <td style="padding:14px 16px;text-align:right;color:#ccc;font-size:13px;font-family:monospace;">${row['price']:,.2f}</td>
                    <td style="padding:14px 16px;text-align:right;color:#ccc;font-size:13px;font-family:monospace;">${row['equity']:,.2f}</td>
                    <td style="padding:14px 16px;">{pnl_html}</td>
                    <td style="padding:14px 16px;">{pnl_dollar_html}</td>
                    <td style="padding:14px 16px;color:#ccc;font-size:13px;">{trades_str}</td>
                    <td style="padding:14px 16px;min-width:100px;">{wr_html}</td>
                    <td style="padding:14px 16px;">{best_html}</td>
                    <td style="padding:14px 16px;">{worst_html}</td>
                </tr>
                '''
            
            if not asset_rows_html:
                asset_rows_html = '''
                <tr>
                    <td colspan="10" style="padding:30px;text-align:center;color:#555;font-size:14px;">
                        No trades recorded yet. Start the trading bot to see portfolio data.
                    </td>
                </tr>
                '''
            
            portfolio_html = f'''
            <div style="
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0d1117;
                color: #fff;
                padding: 0;
            ">
                <!-- Header -->
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
                    <span style="font-size:22px;font-weight:700;color:#fff;">Live Portfolio</span>
                    <span style="
                        background: #1a6b3c;
                        color: #00e676;
                        padding: 3px 10px;
                        border-radius: 4px;
                        font-size: 10px;
                        font-weight: 700;
                        letter-spacing: 1px;
                        text-transform: uppercase;
                    ">LIVE TRADING</span>
                    <span style="
                        background: {'#1b3a26' if is_online else '#3a1b1b'};
                        color: {status_color};
                        padding: 3px 10px;
                        border-radius: 4px;
                        font-size: 10px;
                        font-weight: 700;
                        letter-spacing: 1px;
                        margin-left: 4px;
                    ">{status_dot} {status_text}</span>
                </div>
                
                <!-- Metric Cards Row 1 -->
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">
                    <div style="background:#151b23;border:1px solid #21262d;border-radius:8px;padding:18px 20px;">
                        <div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Realized PNL</div>
                        <div style="font-size:26px;font-weight:700;color:{pnl_color(realized_pnl_total)};">{pnl_sign(realized_pct)}{realized_pct:.2f}%</div>
                    </div>
                    <div style="background:#151b23;border:1px solid #21262d;border-radius:8px;padding:18px 20px;">
                        <div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Open PNL</div>
                        <div style="font-size:26px;font-weight:700;color:{pnl_color(open_pnl_total)};">{pnl_sign(open_pct)}{open_pct:.2f}%</div>
                    </div>
                    <div style="background:#151b23;border:1px solid #21262d;border-radius:8px;padding:18px 20px;">
                        <div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Win Rate</div>
                        <div style="font-size:26px;font-weight:700;color:#fff;">{overall_win_rate:.0f}%</div>
                        <div style="color:#8b949e;font-size:11px;">{total_winning_trades}W / {total_closed_trades - total_winning_trades}L</div>
                    </div>
                    <div style="background:#151b23;border:1px solid #21262d;border-radius:8px;padding:18px 20px;">
                        <div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Trades</div>
                        <div style="font-size:26px;font-weight:700;color:#fff;">{total_trades_count}</div>
                        <div style="color:#8b949e;font-size:11px;">{total_open_trades} open · {total_closed_trades} closed</div>
                    </div>
                </div>
                
                <!-- Metric Cards Row 2 (Dollar Values) -->
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px;">
                    <div style="background:#151b23;border:1px solid #21262d;border-radius:8px;padding:14px 20px;">
                        <div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Portfolio Value</div>
                        <div style="font-size:22px;font-weight:700;color:#fff;">${lp_total_balance:,.2f}</div>
                    </div>
                    <div style="background:#151b23;border:1px solid #21262d;border-radius:8px;padding:14px 20px;">
                        <div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Total P&L</div>
                        <div style="font-size:22px;font-weight:700;color:{pnl_color(lp_grand_total_pnl)};">{pnl_sign(lp_grand_total_pnl)}${abs(lp_grand_total_pnl):,.2f}</div>
                    </div>
                    <div style="background:#151b23;border:1px solid #21262d;border-radius:8px;padding:14px 20px;">
                        <div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Active Assets</div>
                        <div style="font-size:22px;font-weight:700;color:#fff;">{lp_active_assets_count}</div>
                    </div>
                </div>
                
                <!-- Equity Curve (Pure SVG — no external deps) -->
                <div style="background:#151b23;border:1px solid #21262d;border-radius:8px;padding:20px;margin-bottom:24px;">
                    <div style="font-size:15px;font-weight:600;color:#fff;margin-bottom:2px;">Equity Curve</div>
                    <div style="color:#8b949e;font-size:11px;margin-bottom:12px;">Cumulative P&L from closed trades</div>
                    {svg_chart}
                </div>
                
                <!-- Asset Table -->
                <div style="background:#151b23;border:1px solid #21262d;border-radius:8px;overflow:hidden;">
                    <table style="width:100%;border-collapse:collapse;">
                        <thead>
                            <tr style="border-bottom:1px solid #21262d;">
                                <th style="padding:12px 16px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Asset</th>
                                <th style="padding:12px 16px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Status</th>
                                <th style="padding:12px 16px;text-align:right;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Price</th>
                                <th style="padding:12px 16px;text-align:right;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Equity</th>
                                <th style="padding:12px 16px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">PNL (%)</th>
                                <th style="padding:12px 16px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">PNL ($)</th>
                                <th style="padding:12px 16px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Trades</th>
                                <th style="padding:12px 16px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Win Rate</th>
                                <th style="padding:12px 16px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Best</th>
                                <th style="padding:12px 16px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Worst</th>
                            </tr>
                        </thead>
                        <tbody>
                            {asset_rows_html}
                        </tbody>
                    </table>
                </div>
                
                <!-- Footer -->
                <div style="text-align:center;color:#555;font-size:11px;margin-top:16px;">
                    DRL Trading System · Signals from PPO + Composite Scoring · Connected to OKX
                </div>
            </div>
            '''
            
            components.html(portfolio_html, height=950, scrolling=True)

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
                        background: #151b23;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 10px;
                        border-left: 4px solid {color};
                        border: 1px solid #21262d;
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
                                background: #21262d;
                                color: #e6edf3;
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
        <span style="color: #00e676;">●</span> WebSocket Live Data |
        Deployed: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
