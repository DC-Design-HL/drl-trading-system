"""
DRL Trading System - Streamlit Dashboard
Real-time monitoring with TradingView charts, WebSocket live data, and timeframe switching.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import bisect
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.backtest.data_loader import DataLoader, BinanceHistoricalDataFetcher
    _HAS_BACKTEST = True
except ImportError:
    _HAS_BACKTEST = False
from src.data.storage import get_storage, JsonFileStorage
from src.ui.design_system import (
    GLOBAL_CSS, BG_PRIMARY, BG_CARD, BG_CARD_ALT, ACCENT, SUCCESS, DANGER,
    WARNING, TEXT_PRIMARY, TEXT_MUTED, BORDER, SUCCESS_DIM, DANGER_DIM,
    metric_card, status_badge, pnl_text, section_header, styled_table,
    card_container, loading_card, error_card, position_badge, metric_row,
    progress_bar, _pnl_color, _pnl_sign, _format_number, _esc,
)

# True when running as a client-only HF Space (API_SERVER_URL points at remote server)
IS_CLIENT_MODE = bool(os.environ.get('API_SERVER_URL'))

# API server URL — configurable for remote (local server) or local deployments
def get_api_url() -> str:
    """Return the base URL of the Flask API server.

    Set API_SERVER_URL env var to point at a remote local server
    (e.g. https://abc123.ngrok.io). Defaults to localhost:5001.
    """
    return os.environ.get('API_SERVER_URL', 'http://127.0.0.1:5001').rstrip('/')

# Page configuration — MUST be first Streamlit command
st.set_page_config(
    page_title="DRL Trading System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize storage with caching (must be after set_page_config)
@st.cache_resource
def get_app_storage():
    return get_storage()

storage = get_app_storage()

# Inject design system global CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


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
    """Load real trading data — via API in client mode, local storage otherwise."""
    import requests as _r

    def _filter_by_symbol(trades, symbol):
        if not symbol:
            return trades
        s1 = symbol.replace('/', '').upper()
        return [t for t in trades if s1 in t.get('symbol', t.get('asset', '')).replace('/', '').upper()
                or t.get('symbol', t.get('asset', '')).replace('/', '').upper() in s1]

    if IS_CLIENT_MODE:
        try:
            resp = _r.get(f'{get_api_url()}/api/trades', timeout=10)
            if resp.ok:
                return _filter_by_symbol(resp.json(), symbol)
        except Exception:
            pass
        return []

    # Local storage mode
    try:
        all_trades = storage.get_trades(limit=1000)

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
                        filtered_by_time.append(trade)
                all_trades = filtered_by_time
        except:
            pass

        return _filter_by_symbol(all_trades, symbol)
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
    """Get current trading state — via API in client mode, local storage otherwise."""
    import requests as _r

    _empty = {'balance': 0, 'realized_pnl': 0, 'multi_asset': True,
              'whale_alerts': [], 'assets': {}, 'available_assets': []}

    if IS_CLIENT_MODE:
        try:
            state_resp = _r.get(f'{get_api_url()}/api/state', timeout=10)
            state = state_resp.json() if state_resp.ok else {}
            trades_resp = _r.get(f'{get_api_url()}/api/trades', timeout=10)
            all_trades = trades_resp.json() if trades_resp.ok else []

            raw_assets = state.get('assets', {})

            if selected_asset:
                s1 = selected_asset.replace('/', '').upper()
                asset_trades = [t for t in all_trades
                                if s1 in t.get('symbol', t.get('asset', '')).replace('/', '').upper()]
                asset_state = raw_assets.get(selected_asset, raw_assets.get(s1, {}))
                return {
                    'balance': state.get('balance', state.get('total_balance', 0)),
                    'total_balance': state.get('total_balance', state.get('balance', 0)),
                    'asset_balance': asset_state.get('balance', 0),
                    'position': asset_state.get('position', 0),
                    'realized_pnl': state.get('realized_pnl', state.get('total_pnl', 0)),
                    'total_pnl': state.get('total_pnl', state.get('realized_pnl', 0)),
                    'asset_pnl': asset_state.get('pnl', 0),
                    'trades': asset_trades,
                    'total_trades': len([t for t in asset_trades if 'OPEN' in t.get('action', '')]),
                    'position_price': asset_state.get('price', 0),
                    'position_size_units': asset_state.get('units', 0),
                    'price': asset_state.get('price', 0),
                    'timestamp': state.get('timestamp'),
                    'multi_asset': True,
                    'available_assets': state.get('available_assets') or list(raw_assets.keys()) or ['BTCUSDT'],
                    'whale_alerts': state.get('whale_alerts', []),
                    'raw_state': state,
                    'assets': raw_assets,
                    'sl': asset_state.get('sl', 0),
                    'tp': asset_state.get('tp', 0),
                }
            else:
                return {
                    'balance': state.get('balance', state.get('total_balance', 0)),
                    'total_balance': state.get('total_balance', state.get('balance', 0)),
                    'realized_pnl': state.get('realized_pnl', state.get('total_pnl', 0)),
                    'total_pnl': state.get('total_pnl', state.get('realized_pnl', 0)),
                    'multi_asset': True,
                    'available_assets': state.get('available_assets') or list(raw_assets.keys()) or ['BTCUSDT'],
                    'whale_alerts': state.get('whale_alerts', []),
                    'raw_state': state,
                    'assets': raw_assets,
                }
        except Exception:
            pass
        return _empty

    # ── Local storage mode (server-side only) ──────────────────────────────
    try:
        state = storage.load_state()

        if not state:
            return {**_empty}

        # If specific asset selected, return its details mixed with global
        if selected_asset and 'assets' in state and selected_asset in state['assets']:
            asset_state = state['assets'][selected_asset]
            asset_trades = load_trading_log(symbol=selected_asset)

            all_trades = load_trading_log()
            realized_pnl = sum(t.get('pnl', 0) for t in all_trades
                               if 'CLOSE' in t.get('action', '').upper() or 'EXIT' in t.get('action', '').upper())
            raw_assets = state.get('assets', {})
            open_pnl = sum(a.get('pnl', 0) for a in raw_assets.values() if a.get('position', 0) != 0)
            total_pnl = realized_pnl + open_pnl
            total_balance = state.get('total_balance', state.get('balance'))

            whale_alerts = _load_whale_alerts_local()
            state['whale_alerts'] = whale_alerts

            return {
                'balance': total_balance,
                'total_balance': total_balance,
                'asset_balance': asset_state.get('balance', 0),
                'position': asset_state.get('position', 0),
                'realized_pnl': total_pnl,
                'total_pnl': total_pnl,
                'asset_pnl': asset_state.get('pnl', 0),
                'trades': asset_trades,
                'total_trades': len([t for t in asset_trades if 'OPEN' in t.get('action', '')]),
                'position_price': asset_state.get('price', 0),
                'position_size_units': asset_state.get('units', 0),
                'price': asset_state.get('price', 0),
                'timestamp': state.get('timestamp'),
                'multi_asset': True,
                'available_assets': state.get('available_assets') or list(state.get('assets', {}).keys()) or ['BTCUSDT'],
                'whale_alerts': whale_alerts,
                'raw_state': state,
                'assets': raw_assets,
                'sl': asset_state.get('sl', 0),
                'tp': asset_state.get('tp', 0),
            }

        # Global view
        all_trades = load_trading_log()
        realized_pnl = sum(t.get('pnl', 0) for t in all_trades
                           if 'CLOSE' in t.get('action', '').upper() or 'EXIT' in t.get('action', '').upper())
        raw_assets = state.get('assets', {})
        open_pnl = sum(a.get('pnl', 0) for a in raw_assets.values() if a.get('position', 0) != 0)
        total_pnl = realized_pnl + open_pnl
        total_balance = state.get('total_balance', state.get('balance'))

        whale_alerts = _load_whale_alerts_local()
        state['whale_alerts'] = whale_alerts

        return {
            'balance': total_balance,
            'total_balance': total_balance,
            'realized_pnl': total_pnl,
            'total_pnl': total_pnl,
            'multi_asset': True,
            'available_assets': state.get('available_assets') or list(state.get('assets', {}).keys()) or ['BTCUSDT'],
            'whale_alerts': whale_alerts,
            'raw_state': state,
            'assets': raw_assets,
        }
    except Exception:
        return {**_empty}


def _load_whale_alerts_local() -> list:
    """Load whale alerts from local wallet files (server-side only). Returns [] on HF."""
    import json as _json, time as _time
    whale_alerts = []
    try:
        whale_dir = Path(__file__).parent.parent.parent / "data" / "whale_wallets"
        if not whale_dir.exists():
            return []
        try:
            from src.features.whale_wallet_registry import get_wallets_by_chain as _gwbc
        except ImportError:
            return []
        for chain_dir in whale_dir.iterdir():
            if not chain_dir.is_dir():
                continue
            chain = chain_dir.name.upper()
            for wallet_file in chain_dir.glob("*.json"):
                try:
                    with open(wallet_file, "r") as f:
                        w_data = _json.load(f)
                    addr = w_data.get("address", "")
                    chain_wallets = _gwbc(chain)
                    wallet = next((w for w in chain_wallets if w.address.lower() == addr.lower()), None)
                    w_label = wallet.label if wallet else f"Unknown {chain} Whale"
                    w_type = wallet.wallet_type if wallet else "unknown"
                    price_map = {'BTC': 70000, 'ETH': 3500, 'SOL': 150, 'XRP': 0.6}
                    for tx in w_data.get("transactions", [])[-10:]:
                        val = float(tx.get('value', 0))
                        if val * price_map.get(chain, 1) > 50000:
                            whale_alerts.append({
                                'chain': chain, 'value': val, 'currency': tx.get('asset', chain),
                                'timestamp': tx.get('timestamp', int(_time.time())),
                                'link': tx.get('link', '#'),
                                'wallet_label': w_label, 'wallet_type': w_type, 'wallet_address': addr,
                            })
                except Exception:
                    pass
        whale_alerts = sorted(whale_alerts, key=lambda x: x.get('timestamp', 0), reverse=True)[:50]
    except Exception:
        pass
    return whale_alerts


def _classify_exit_reason(close_trade: dict, all_trades: list) -> str:
    """Classify a CLOSE trade as EXIT(SL), EXIT(TP), or EXIT.

    Strategy:
    1. Check explicit 'reason' field (handles 'SL', 'stop_loss', 'TP', 'take_profit')
    2. If no reason, find the matching OPEN trade and compare fill price vs SL/TP
    3. Fallback to 'EXIT' when reason cannot be determined
    """
    reason = str(close_trade.get('reason', '')).lower().strip()

    # 1. Explicit reason field
    if reason in ('sl', 'stop_loss', 'stoploss'):
        return 'EXIT(SL)'
    if reason in ('tp', 'take_profit', 'takeprofit'):
        return 'EXIT(TP)'

    # 2. Infer from fill price vs SL/TP on the matching OPEN trade
    fill_price = float(close_trade.get('price', 0) or close_trade.get('exit_price', 0) or close_trade.get('filled_price', 0) or 0)
    if fill_price <= 0:
        return 'EXIT'

    # Check if the close trade itself has SL/TP data (some formats include it)
    sl_price = float(close_trade.get('sl', 0) or close_trade.get('sl_price', 0) or 0)
    tp_price = float(close_trade.get('tp', 0) or close_trade.get('tp_price', 0) or 0)

    # If not on the close trade, find the preceding OPEN trade for this symbol
    if sl_price <= 0 and tp_price <= 0:
        close_sym = (close_trade.get('symbol', '') or close_trade.get('asset', '')).replace('/', '').upper()
        close_ts = close_trade.get('timestamp', '')
        for t in reversed(all_trades):
            t_action = t.get('action', '')
            if 'OPEN' not in t_action:
                continue
            t_sym = (t.get('symbol', '') or t.get('asset', '')).replace('/', '').upper()
            t_ts = t.get('timestamp', '')
            if t_sym == close_sym and t_ts < close_ts:
                sl_price = float(t.get('sl', 0) or t.get('sl_price', 0) or 0)
                tp_price = float(t.get('tp', 0) or t.get('tp_price', 0) or 0)
                break

    if sl_price <= 0 and tp_price <= 0:
        return 'EXIT'

    # Compare fill price proximity to SL vs TP (within 0.3% tolerance)
    TOLERANCE = 0.003
    sl_dist = abs(fill_price - sl_price) / fill_price if sl_price > 0 else float('inf')
    tp_dist = abs(fill_price - tp_price) / fill_price if tp_price > 0 else float('inf')

    if sl_dist < TOLERANCE and sl_dist <= tp_dist:
        return 'EXIT(SL)'
    if tp_dist < TOLERANCE and tp_dist < sl_dist:
        return 'EXIT(TP)'

    # If PnL is available, use it as a secondary heuristic
    pnl = float(close_trade.get('pnl', 0) or close_trade.get('realizedPnl', 0) or 0)
    if sl_price > 0 and tp_price > 0:
        if pnl < 0 and sl_dist < tp_dist:
            return 'EXIT(SL)'
        if pnl > 0 and tp_dist < sl_dist:
            return 'EXIT(TP)'

    return 'EXIT'


def create_tradingview_chart_with_websocket(df: pd.DataFrame, trades: list, timeframe: str = '1h', symbol: str = 'BTC/USDT', market_structure: dict = None) -> str:
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
    
    # Build sorted list of candle open-times so we can snap trade markers
    # to the nearest candle.  lightweight-charts v4 REQUIRES that every
    # marker time matches an existing candle time, otherwise the marker is
    # silently dropped.  It also requires all markers sorted ascending.
    candle_times = sorted(c['time'] for c in candlestick_data)

    def _snap_to_candle(unix_ts: int) -> int | None:
        """Return the candle open-time closest to *unix_ts*, or None if
        the trade falls outside the chart's visible range."""
        if not candle_times:
            return None
        idx = bisect.bisect_left(candle_times, unix_ts)
        # Pick whichever neighbour is closest
        candidates = []
        if idx < len(candle_times):
            candidates.append(candle_times[idx])
        if idx > 0:
            candidates.append(candle_times[idx - 1])
        best = min(candidates, key=lambda ct: abs(ct - unix_ts))
        return best

    # Create markers for trades
    # ── Phase 1: build one raw marker per trade ────────────────────────
    _raw_markers: list[dict] = []
    for trade in trades:
        # CLOSE trades may use 'exit_price' instead of 'price'
        has_price = 'price' in trade or 'exit_price' in trade or 'filled_price' in trade
        if has_price and 'timestamp' in trade:
            try:
                ts = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                raw_time = int(ts.timestamp())
                snapped = _snap_to_candle(raw_time)
                if snapped is None:
                    continue
                action = trade.get('action', '')
                # Defensive: reason can be None, missing, or a string
                reason = str(trade.get('reason') or '').lower()

                if 'OPEN_LONG' in action:
                    _raw_markers.append({
                        'time': snapped,
                        'position': 'belowBar',
                        'color': '#26a69a',
                        'shape': 'arrowUp',
                        'text': 'LONG',
                        '_kind': 'entry',
                    })
                elif 'OPEN_SHORT' in action:
                    _raw_markers.append({
                        'time': snapped,
                        'position': 'aboveBar',
                        'color': '#ef5350',
                        'shape': 'arrowDown',
                        'text': 'SHORT',
                        '_kind': 'entry',
                    })
                elif 'CLOSE' in action:
                    # Determine exit reason from explicit reason field or
                    # infer from fill price vs SL/TP prices
                    exit_label = _classify_exit_reason(trade, trades)

                    if exit_label == 'EXIT(SL)':
                        _raw_markers.append({
                            'time': snapped,
                            'position': 'aboveBar',
                            'color': '#ff4444',
                            'shape': 'square',
                            'text': 'EXIT(SL)',
                            '_kind': 'exit',
                        })
                    elif exit_label == 'EXIT(TP)':
                        _raw_markers.append({
                            'time': snapped,
                            'position': 'belowBar',
                            'color': '#00e676',
                            'shape': 'square',
                            'text': 'EXIT(TP)',
                            '_kind': 'exit',
                        })
                    else:
                        _raw_markers.append({
                            'time': snapped,
                            'position': 'aboveBar',
                            'color': '#ffc107',
                            'shape': 'square',
                            'text': 'EXIT',
                            '_kind': 'exit',
                        })
            except Exception:
                continue

    # ── Phase 2: split into entry markers + exit markers ─────────────
    # lightweight-charts only allows ONE marker per time per series.
    # FIX: put entry markers on the candlestick series and exit markers
    # on a separate invisible LineSeries.  Both render independently.
    from collections import defaultdict as _defaultdict

    # Aggregate per-candle for each kind
    _entry_by_candle: dict[int, list] = _defaultdict(list)
    _exit_by_candle: dict[int, list] = _defaultdict(list)
    for m in _raw_markers:
        if m['_kind'] == 'exit':
            _exit_by_candle[m['time']].append(m)
        else:
            _entry_by_candle[m['time']].append(m)

    # Build entry markers (one per candle on candlestickSeries)
    entry_markers: list[dict] = []
    for candle_time in sorted(_entry_by_candle):
        entries = _entry_by_candle[candle_time]
        n_long = sum(1 for e in entries if e['text'] == 'LONG')
        n_short = sum(1 for e in entries if e['text'] == 'SHORT')
        if n_long and n_short:
            label = f'L×{n_long}+S×{n_short}' if (n_long + n_short) > 1 else entries[0]['text']
        elif n_long:
            label = f'LONG×{n_long}' if n_long > 1 else 'LONG'
        elif n_short:
            label = f'SHORT×{n_short}' if n_short > 1 else 'SHORT'
        else:
            label = entries[0]['text']
        is_short = 'SHORT' in label
        entry_markers.append({
            'time': candle_time,
            'position': 'aboveBar' if is_short else 'belowBar',
            'color': '#ef5350' if is_short else '#26a69a',
            'shape': 'arrowDown' if is_short else 'arrowUp',
            'text': label,
        })

    # Build exit markers (one per candle on separate exitSeries)
    exit_markers: list[dict] = []
    for candle_time in sorted(_exit_by_candle):
        exits = _exit_by_candle[candle_time]
        n_sl = sum(1 for e in exits if e['text'] == 'EXIT(SL)')
        n_tp = sum(1 for e in exits if e['text'] == 'EXIT(TP)')
        n_plain = sum(1 for e in exits if e['text'] == 'EXIT')
        total = len(exits)
        if total == 1:
            label = exits[0]['text']
        elif n_sl or n_tp:
            parts = []
            if n_sl:
                parts.append(f'SL×{n_sl}')
            if n_tp:
                parts.append(f'TP×{n_tp}')
            if n_plain:
                parts.append(f'×{n_plain}')
            label = 'EXIT(' + '+'.join(parts) + ')'
        else:
            label = f'EXIT×{total}'
        color = '#ff4444' if n_sl else ('#00e676' if n_tp else '#ffc107')
        exit_markers.append({
            'time': candle_time,
            'position': 'aboveBar',
            'color': color,
            'shape': 'square',
            'text': label,
        })

    # lightweight-charts requires markers sorted by time ascending;
    # unsorted markers cause ALL markers to be silently dropped.
    entry_markers.sort(key=lambda m: m['time'])
    exit_markers.sort(key=lambda m: m['time'])
    
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
            
            // Add ENTRY markers on candlestick series
            const entryMarkers = {json.dumps(entry_markers)};
            if (entryMarkers.length > 0) {{
                entryMarkers.sort((a, b) => a.time - b.time);
                console.log('[chart] Setting', entryMarkers.length, 'entry markers:', entryMarkers.map(m => m.text));
                candlestickSeries.setMarkers(entryMarkers);
            }}

            // Add EXIT markers on a separate invisible LineSeries
            // This avoids lightweight-charts' one-marker-per-time-per-series limit
            const exitMarkers = {json.dumps(exit_markers)};
            let exitSeries = null;
            if (exitMarkers.length > 0) {{
                exitSeries = chart.addLineSeries({{
                    color: 'transparent',
                    lineWidth: 0,
                    lastValueVisible: false,
                    priceLineVisible: false,
                    crosshairMarkerVisible: false,
                }});
                // Feed the exit series the same candle close prices so markers
                // attach at valid time points (series needs data at marker times)
                exitSeries.setData(candleData.map(c => ({{ time: c.time, value: c.close }})));
                exitMarkers.sort((a, b) => a.time - b.time);
                console.log('[chart] Setting', exitMarkers.length, 'exit markers:', exitMarkers.map(m => m.text));
                exitSeries.setMarkers(exitMarkers);
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

            // Track whether user is hovering over a specific candle
            let isHoveringCandle = false;
            let hoverTimeout = null;

            // Update OHLC on crosshair move
            chart.subscribeCrosshairMove((param) => {{
                if (param.time) {{
                    const data = param.seriesData.get(candlestickSeries);
                    if (data) {{
                        // User is hovering over a candle
                        isHoveringCandle = true;

                        // Clear any existing timeout
                        if (hoverTimeout) {{
                            clearTimeout(hoverTimeout);
                        }}

                        // Reset hover flag after 2 seconds of inactivity
                        hoverTimeout = setTimeout(() => {{
                            isHoveringCandle = false;
                        }}, 2000);

                        document.getElementById('o-val').textContent = data.open.toFixed(2);
                        document.getElementById('h-val').textContent = data.high.toFixed(2);
                        document.getElementById('l-val').textContent = data.low.toFixed(2);
                        document.getElementById('c-val').textContent = data.close.toFixed(2);

                        const change = ((data.close - data.open) / data.open * 100).toFixed(2);
                        const changeEl = document.getElementById('change-val');
                        changeEl.textContent = (change >= 0 ? '+' : '') + change + '%';
                        changeEl.style.color = change >= 0 ? '#26a69a' : '#ef5350';
                    }}
                }} else {{
                    // User moved cursor away from chart
                    isHoveringCandle = false;
                    if (hoverTimeout) {{
                        clearTimeout(hoverTimeout);
                        hoverTimeout = null;
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
                ws = new WebSocket('wss://data-stream.binance.vision/ws/{ws_stream}');
                
                ws.onopen = function() {{
                    console.log('WebSocket connected to Binance Vision cluster successfully');
                    document.getElementById('live-indicator').style.display = 'inline-flex';
                }};
                
                ws.onclose = function(event) {{
                    console.log('WebSocket closed: code=' + event.code + ', reason=' + event.reason);
                    document.getElementById('live-indicator').style.display = 'none';
                    setTimeout(connectWebSocket, reconnectInterval);
                }};
                
                ws.onerror = function(err) {{
                    console.error('WebSocket encountered an error:', err);
                    ws.close();
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
                    
                    // Keep exit marker series in sync with candle data
                    if (exitSeries) {{
                        exitSeries.update({{ time: candle.time, value: candle.close }});
                    }}
                    
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
                    localStorage.setItem('{clean_symbol}_live_price', candle.close.toFixed(2));

                    // Update OHLC display for current candle (only if user is not hovering over a historical candle)
                    if (!isHoveringCandle) {{
                        document.getElementById('o-val').textContent = candle.open.toFixed(2);
                        document.getElementById('h-val').textContent = candle.high.toFixed(2);
                        document.getElementById('l-val').textContent = candle.low.toFixed(2);
                        document.getElementById('c-val').textContent = candle.close.toFixed(2);

                        // Update change display as well
                        const change = ((candle.close - candle.open) / candle.open * 100).toFixed(2);
                        const changeEl = document.getElementById('change-val');
                        changeEl.textContent = (change >= 0 ? '+' : '') + change + '%';
                        changeEl.style.color = change >= 0 ? '#26a69a' : '#ef5350';
                    }}
                }};
            }}
            
            connectWebSocket();

            // ── Market Structure Overlay (BOS/CHOCH/Swing Points) ──
            const msData = {json.dumps(market_structure) if market_structure else 'null'};
            if (msData) {{ try {{
                console.log('[BOS/CHOCH] msData received:', JSON.stringify({{
                    highs: (msData.swing_highs || []).length,
                    lows: (msData.swing_lows || []).length,
                    bos: (msData.bos_signals || []).length,
                    choch: (msData.choch_signals || []).length,
                }}));

                // --- Swing High structure line (green, connecting highs) ---
                const swingHighLine = (msData.swing_highs || [])
                    .map(sh => ({{ time: sh.time, value: sh.price }}))
                    .sort((a, b) => a.time - b.time);

                if (swingHighLine.length >= 2) {{
                    const shSeries = chart.addLineSeries({{
                        color: '#00e676',
                        lineWidth: 2,
                        lineStyle: 2,  // Dotted (numeric enum: 0=Solid, 1=Dotted, 2=Dashed, 3=LargeDashed, 4=SparseDotted)
                        lastValueVisible: false,
                        priceLineVisible: false,
                        crosshairMarkerVisible: false,
                    }});
                    shSeries.setData(swingHighLine);
                    console.log('[BOS/CHOCH] Swing high line drawn:', swingHighLine.length, 'points');
                }}

                // --- Swing Low structure line (red, connecting lows) ---
                const swingLowLine = (msData.swing_lows || [])
                    .map(sl => ({{ time: sl.time, value: sl.price }}))
                    .sort((a, b) => a.time - b.time);

                if (swingLowLine.length >= 2) {{
                    const slSeries = chart.addLineSeries({{
                        color: '#ff5252',
                        lineWidth: 2,
                        lineStyle: 2,  // Dashed
                        lastValueVisible: false,
                        priceLineVisible: false,
                        crosshairMarkerVisible: false,
                    }});
                    slSeries.setData(swingLowLine);
                    console.log('[BOS/CHOCH] Swing low line drawn:', swingLowLine.length, 'points');
                }}

                // --- BOS signals as horizontal lines with markers ---
                const bosSignals = msData.bos_signals || [];
                bosSignals.forEach((bos, idx) => {{
                    const isFake = bos.is_fake;
                    const isBullish = bos.direction === 'bullish';
                    const lineColor = isBullish
                        ? (isFake ? 'rgba(0, 230, 118, 0.3)' : 'rgba(0, 230, 118, 0.7)')
                        : (isFake ? 'rgba(255, 82, 82, 0.3)' : 'rgba(255, 82, 82, 0.7)');
                    const lineStyle = isFake ? 1 : 2;  // Dotted for fake, Dashed for real
                    const label = isBullish
                        ? (isFake ? 'BOS↑(fake)' : 'BOS↑')
                        : (isFake ? 'BOS↓(fake)' : 'BOS↓');

                    // Draw a short horizontal line at the BOS level
                    // spanning from the signal time back a few candles
                    const sigTime = bos.time;
                    const candleTimes = candleData.map(c => c.time);
                    const sigIdx = candleTimes.indexOf(sigTime);
                    const startIdx = Math.max(0, sigIdx - 8);
                    const endIdx = Math.min(candleTimes.length - 1, sigIdx + 2);

                    if (sigIdx >= 0 && startIdx < endIdx) {{
                        const bosLine = chart.addLineSeries({{
                            color: lineColor,
                            lineWidth: 2,
                            lineStyle: lineStyle,
                            lastValueVisible: false,
                            priceLineVisible: false,
                            crosshairMarkerVisible: false,
                        }});
                        bosLine.setData([
                            {{ time: candleTimes[startIdx], value: bos.level }},
                            {{ time: candleTimes[endIdx], value: bos.level }},
                        ]);
                        // Add label as marker on this line
                        bosLine.setMarkers([{{
                            time: candleTimes[Math.min(sigIdx, endIdx)],
                            position: isBullish ? 'aboveBar' : 'belowBar',
                            color: lineColor,
                            shape: 'square',
                            text: label,
                        }}]);
                    }}
                }});

                // --- CHOCH signals as horizontal lines with markers ---
                const chochSignals = msData.choch_signals || [];
                chochSignals.forEach((choch, idx) => {{
                    const isFake = choch.is_fake;
                    const isBullish = choch.direction === 'bullish';
                    const lineColor = isBullish
                        ? (isFake ? 'rgba(66, 165, 245, 0.3)' : 'rgba(66, 165, 245, 0.8)')
                        : (isFake ? 'rgba(255, 152, 0, 0.3)' : 'rgba(255, 152, 0, 0.8)');
                    const lineStyle = isFake ? 1 : 2;  // 1=Dotted, 2=Dashed
                    const label = isBullish
                        ? (isFake ? 'CHOCH↑(fake)' : 'CHOCH↑')
                        : (isFake ? 'CHOCH↓(fake)' : 'CHOCH↓');

                    const sigTime = choch.time;
                    const candleTimes = candleData.map(c => c.time);
                    const sigIdx = candleTimes.indexOf(sigTime);
                    const startIdx = Math.max(0, sigIdx - 8);
                    const endIdx = Math.min(candleTimes.length - 1, sigIdx + 2);

                    if (sigIdx >= 0 && startIdx < endIdx) {{
                        const chochLine = chart.addLineSeries({{
                            color: lineColor,
                            lineWidth: 2,
                            lineStyle: lineStyle,
                            lastValueVisible: false,
                            priceLineVisible: false,
                            crosshairMarkerVisible: false,
                        }});
                        chochLine.setData([
                            {{ time: candleTimes[startIdx], value: choch.level }},
                            {{ time: candleTimes[endIdx], value: choch.level }},
                        ]);
                        chochLine.setMarkers([{{
                            time: candleTimes[Math.min(sigIdx, endIdx)],
                            position: isBullish ? 'aboveBar' : 'belowBar',
                            color: lineColor,
                            shape: 'square',
                            text: label,
                        }}]);
                    }}
                }});

                // Log summary
                console.log('[chart] Market structure loaded:',
                    'swingH=' + (msData.swing_highs || []).length,
                    'swingL=' + (msData.swing_lows || []).length,
                    'BOS=' + bosSignals.length,
                    'CHOCH=' + chochSignals.length,
                    'trend=' + msData.trend,
                    'conf=' + msData.confidence);
            }} catch(msErr) {{
                console.error('[BOS/CHOCH] Error rendering market structure:', msErr);
            }}
            }}

            // Cleanup on page unload
            window.addEventListener('beforeunload', function() {{
                if (ws) ws.close();
            }});
        }})();
    </script>
    """
    return html


def render_position_card(state: dict, current_price: float, symbol: str = 'BTC/USDT'):
    """Render current position card."""
    position = state.get('position', 0)
    clean_symbol = symbol.replace('/', '').lower()  # For localStorage key
    
    # SL/TP percentages (match live trading config)
    SL_PCT = 0.015  # 1.5% (matches live_trading.py)
    TP_PCT = 0.025  # 2.5% (matches live_trading.py)
    
    if position == 0:
        st.markdown(card_container(
            f'<div style="text-align:center;">'
            f'<div class="ds-card-label">Current Position</div>'
            f'<div style="font-size:24px;color:{TEXT_MUTED};margin-top:10px;">No Position (FLAT)</div>'
            f'<div style="font-size:12px;color:{TEXT_MUTED};margin-top:5px;font-family:monospace;">Current Price: ${current_price:,.2f}</div>'
            f'</div>'
        ), unsafe_allow_html=True)
    else:
        # DEBUG: Inspect state to find units key
        # st.write(f"Debug State for P&L: {state}")
        # logger.info(f"Debug State for P&L: {state}") 
        pass
        is_long = position == 1
        color = SUCCESS if is_long else DANGER
        side = "LONG" if is_long else "SHORT"
        icon = "📈" if is_long else "📉"
        
        # Get entry price from state - check multiple field names for compatibility
        # Priority: position_price > entry_price > price (last trade price fallback)
        entry_price = state.get('position_price') or state.get('entry_price') or state.get('price', current_price)

        # Validation: Entry price must be reasonable (within 50% of current price)
        if entry_price > 0 and current_price > 0:
            price_diff_pct = abs(entry_price - current_price) / current_price
            if price_diff_pct > 0.5:  # More than 50% difference is suspicious
                logger.warning(f"Entry price ${entry_price:,.2f} is {price_diff_pct*100:.1f}% different from current ${current_price:,.2f} - using current price")
                entry_price = current_price
        
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
        
        pnl_color = SUCCESS if unrealized_pnl >= 0 else DANGER
        pnl_sign = "+" if unrealized_pnl >= 0 else ""
        
        st.markdown(card_container(
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span class="ds-card-label" style="margin-bottom:0;">Current Position</span>'
            f'{status_badge(f"{icon} {side}", color)}'
            f'</div>'
            f'<div style="margin-top:15px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            f'<span style="color:{TEXT_MUTED};font-size:12px;">Entry Price:</span>'
            f'<span class="mono" style="color:{TEXT_PRIMARY};">${entry_price:,.2f}</span>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            f'<span style="color:{TEXT_MUTED};font-size:12px;">Current Price:</span>'
            f'<span id="sidebar-current-price" class="mono" style="color:{TEXT_PRIMARY};">${current_price:,.2f}</span>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            f'<span style="color:{TEXT_MUTED};font-size:12px;">Unrealized P&L:</span>'
            f'<span id="sidebar-pnl" class="mono" style="color:{pnl_color};font-weight:600;">{pnl_sign}${unrealized_pnl:,.2f}</span>'
            f'</div>'
            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid {BORDER};">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            f'<span style="color:{DANGER};font-size:12px;">🛑 Stop Loss:</span>'
            f'<span class="mono" style="color:{DANGER};">${sl_price:,.2f}</span>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span style="color:{SUCCESS};font-size:12px;">🎯 Take Profit:</span>'
            f'<span class="mono" style="color:{SUCCESS};">${tp_price:,.2f}</span>'
            f'</div></div></div>'
        ), unsafe_allow_html=True)
        # Real-time price update script (reads WebSocket price from localStorage)
        st.markdown(f"""
        <script>
            // Real-time price update from WebSocket via localStorage
            const entryPrice = {entry_price};
            const positionUnits = {state.get('position_size_units', 0)};
            const isLong = {'true' if is_long else 'false'};
            
            function updateSidebarPrice() {{
                const livePrice = parseFloat(localStorage.getItem('{clean_symbol}_live_price'));
                if (livePrice && livePrice > 0) {{
                    // Update current price
                    const priceEl = document.getElementById('sidebar-current-price');
                    if (priceEl) priceEl.textContent = '$' + livePrice.toLocaleString('en-US', {{minimumFractionDigits: 2}});
                    
                    // Update unrealized P&L
                    let pnl = isLong ? (livePrice - entryPrice) * positionUnits : (entryPrice - livePrice) * positionUnits;
                    const pnlEl = document.getElementById('sidebar-pnl');
                    if (pnlEl) {{
                        pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
                        pnlEl.style.color = pnl >= 0 ? '{SUCCESS}' : '{DANGER}';
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
    st.markdown(f'<div class="ds-card-label">Recent Trades</div>', unsafe_allow_html=True)
    
    action_trades = [t for t in trades if 'action' in t and t['action'] != 'HOLD']
    
    if not action_trades:
        st.markdown(card_container(
            f'<div style="text-align:center;padding:12px;color:{TEXT_MUTED};font-size:13px;">No trades yet</div>'
        ), unsafe_allow_html=True)
        return
    
    trade_rows_html = ""
    for trade in reversed(action_trades[-10:]):
        action = trade.get('action', '')
        price = trade.get('price', 0)
        pnl = trade.get('pnl', 0)
        timestamp = trade.get('timestamp', '')
        
        try:
            ts = datetime.fromisoformat(timestamp)
            time_str = ts.strftime('%m/%d %H:%M')
        except Exception:
            time_str = ''
        
        # Determine display based on action and exit classification
        if 'OPEN_LONG' in action:
            badge_color = SUCCESS
            side = "LONG"
        elif 'OPEN_SHORT' in action:
            badge_color = DANGER
            side = "SHORT"
        elif 'CLOSE' in action:
            exit_label = _classify_exit_reason(trade, trades)
            if exit_label == 'EXIT(SL)':
                badge_color = DANGER
                side = "EXIT(SL)"
            elif exit_label == 'EXIT(TP)':
                badge_color = SUCCESS
                side = "EXIT(TP)"
            else:
                badge_color = WARNING
                side = "EXIT"
        else:
            badge_color = TEXT_MUTED
            side = action
        
        t_pnl_color = SUCCESS if pnl >= 0 else DANGER
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_display = f"{pnl_sign}${pnl:,.2f}" if pnl != 0 else ""
        
        trade_rows_html += (
            f'<div style="background:{BG_CARD_ALT};border-radius:6px;padding:10px 12px;'
            f'margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;'
            f'border:1px solid {BORDER};">'
            f'<div>'
            f'{status_badge(side, badge_color)}'
            f' <span class="mono" style="color:{TEXT_PRIMARY};font-size:12px;margin-left:8px;">${price:,.2f}</span>'
            f' <span style="color:{TEXT_MUTED};font-size:10px;margin-left:8px;">{time_str}</span>'
            f'</div>'
            f'<span class="mono" style="color:{t_pnl_color};font-weight:600;font-size:12px;">{pnl_display}</span>'
            f'</div>'
        )
    
    st.markdown(card_container(trade_rows_html), unsafe_allow_html=True)


def load_real_market_data(symbol: str = 'BTC/USDT', timeframe: str = '1h') -> pd.DataFrame:
    """Load OHLCV candlestick data — via /api/ohlcv or direct Binance public API."""
    import requests as _mkt_requests
    import logging as _log
    _logger = _log.getLogger(__name__)

    clean_symbol = symbol.replace("/", "")

    def _parse_ohlcv_list(data: list) -> pd.DataFrame:
        """Parse list of {time,open,high,low,close,volume} dicts into DataFrame."""
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df['time'], unit='s')
        df.index.name = None
        return df[['open', 'high', 'low', 'close', 'volume']]

    # Primary: /api/ohlcv via local Flask server
    api_url = get_api_url()
    try:
        resp = _mkt_requests.get(
            f'{api_url}/api/ohlcv',
            params={'symbol': clean_symbol, 'interval': timeframe, 'limit': 500},
            timeout=10
        )
        if resp.ok:
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                return _parse_ohlcv_list(data)
            else:
                _logger.warning(f"load_real_market_data: empty/invalid response from {api_url} for {clean_symbol} {timeframe}: {str(data)[:200]}")
        else:
            _logger.warning(f"load_real_market_data: HTTP {resp.status_code} from {api_url}/api/ohlcv for {clean_symbol} {timeframe}")
    except Exception as e:
        _logger.warning(f"load_real_market_data: Flask API unavailable ({api_url}): {e}")

    # Fallback: Direct Binance public API (no auth required, works on HF)
    try:
        _logger.info(f"load_real_market_data: trying direct Binance API for {clean_symbol} {timeframe}")
        binance_url = os.environ.get("BINANCE_FUTURES_URL", "https://data-api.binance.vision")
        resp = _mkt_requests.get(
            f"{binance_url}/api/v3/klines",
            params={'symbol': clean_symbol, 'interval': timeframe, 'limit': 500},
            timeout=15
        )
        if resp.ok:
            raw = resp.json()
            if isinstance(raw, list) and len(raw) > 0 and not (isinstance(raw, dict) and raw.get('code')):
                candles = [
                    {
                        'time': int(row[0]) // 1000,
                        'open': float(row[1]),
                        'high': float(row[2]),
                        'low': float(row[3]),
                        'close': float(row[4]),
                        'volume': float(row[5]),
                    }
                    for row in raw
                ]
                _logger.info(f"load_real_market_data: direct Binance returned {len(candles)} candles for {clean_symbol} {timeframe}")
                return _parse_ohlcv_list(candles)
            else:
                _logger.warning(f"load_real_market_data: Binance direct API returned unexpected data: {str(raw)[:200]}")
        else:
            _logger.warning(f"load_real_market_data: Binance direct API HTTP {resp.status_code} for {clean_symbol} {timeframe}")
    except Exception as e:
        _logger.error(f"load_real_market_data: direct Binance fallback failed: {e}")

    # Final fallback: BinanceHistoricalDataFetcher if backtest module available (local server only)
    if _HAS_BACKTEST:
        try:
            fetcher = BinanceHistoricalDataFetcher()
            end_date = datetime.now()
            days = TIMEFRAMES.get(timeframe, {}).get('days', 7)
            start_date = end_date - timedelta(days=days)
            if "USDT" in symbol and "/" not in symbol:
                symbol = symbol.replace("USDT", "/USDT")
            df = fetcher.fetch_historical_data(
                symbol=symbol, timeframe=timeframe,
                start_date=start_date, end_date=end_date,
            )
            return df
        except Exception as e:
            _logger.error(f"load_real_market_data: BinanceHistoricalDataFetcher failed: {e}")
            return pd.DataFrame()
    return pd.DataFrame()



@st.fragment(run_every=60)
def render_sidebar_metrics_fragment():
    """Render sidebar portfolio metrics with auto-refresh."""
    import requests
    import logging
    logger = logging.getLogger(__name__)
    try:
        try:
            state_resp = requests.get(f'{get_api_url()}/api/state', timeout=5)
            if state_resp.status_code == 200:
                api_state = state_resp.json()
                if 'balance' in api_state:
                     st.session_state.portfolio_balance = api_state.get('balance', 0)
                     st.session_state.total_pnl = api_state.get('total_pnl', 0)

            bal = st.session_state.get('portfolio_balance')
            pnl = float(st.session_state.get('total_pnl') or 0)
            bal_str = f"${bal:,.2f}" if bal is not None else "—"
            st.markdown(
                metric_card("Portfolio Value", bal_str, delta=pnl, icon="💰"),
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(error_card("Connection Error", str(e)), unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Sidebar data fetch error: {e}")

@st.fragment(run_every=120)
def render_market_analysis_fragment(symbol: str):
    """Render market analysis panel with auto-refresh."""
    import requests
    import logging
    logger = logging.getLogger(__name__)
    
    st.markdown(section_header("Market Analysis", "📊"), unsafe_allow_html=True)
    
    # Fetch Market Analysis for current asset
    market_data = {}
    try:
        api_symbol = symbol.replace('/', '').upper()
        market_resp = requests.get(f'{get_api_url()}/api/market?symbol={api_symbol}', timeout=15)
        if market_resp.status_code == 200:
            market_data = market_resp.json()
        else:
            st.markdown(error_card(
                f"API error (HTTP {market_resp.status_code})",
                "Server returned non-200 for /api/market"
            ), unsafe_allow_html=True)
            return
    except Exception as e:
        st.markdown(error_card(
            "Unable to load (API server offline?)",
            str(e)
        ), unsafe_allow_html=True)
        return

    # Whale Tracker
    whale = market_data.get('whale', {})
    if whale:
        if whale.get('error'):
            st.markdown(error_card("🐋 Whale Signals — Data Error", whale.get('error')), unsafe_allow_html=True)
        else:
            w_score = whale.get('score', 0)
            w_direction = whale.get('direction', 'NEUTRAL')
            if w_score > 0:
                w_badge_color = SUCCESS
            elif w_score < 0:
                w_badge_color = DANGER
            else:
                w_badge_color = TEXT_MUTED
            
            # Format Flow Metrics
            flow_metrics = whale.get('flow_metrics', {})
            net_flow = flow_metrics.get('net_flow', 0)
            flow_color = SUCCESS if net_flow > 0 else DANGER
            flow_sign = "+" if net_flow > 0 else "-"
            
            # Format to K or M
            if abs(net_flow) > 1000000:
                flow_str = f"{flow_sign}${abs(net_flow)/1000000:.1f}M"
            elif abs(net_flow) > 1000:
                flow_str = f"{flow_sign}${abs(net_flow)/1000:.0f}K"
            else:
                flow_str = "$0"
            
            st.markdown(card_container(
                f'<div class="ds-card-label">🐋 Whale Signals</div>'
                f'<div style="margin:6px 0;">{status_badge(w_direction, w_badge_color)}</div>'
                f'<div style="color:{TEXT_MUTED};font-size:11px;line-height:1.6;">'
                f'Score: <span class="mono" style="color:{TEXT_PRIMARY};">{w_score:.2f}</span>'
                f' | Conf: <span class="mono" style="color:{TEXT_PRIMARY};">{whale.get("confidence", 0)}%</span><br>'
                f'Flow (1m): <span class="mono" style="color:{flow_color};font-weight:600;">{flow_str}</span><br>'
                f'🟢{whale.get("bullish", 0)} 🔴{whale.get("bearish", 0)} ⚪{whale.get("neutral", 0)}'
                f'</div>'
            ), unsafe_allow_html=True)
    
    # Funding
    funding_data = market_data.get('funding', {})
    funding = funding_data.get('data', {}) # structure varies, being safe
    if funding_data and not funding_data.get('error'):
         # Extract funding rate
         rate = funding_data.get('rate', 0)
         rate_color = SUCCESS if rate > 0.0001 else DANGER if rate < -0.0001 else TEXT_MUTED
         bias = funding_data.get('bias', 'neutral')
         bias_color = SUCCESS if bias == 'bullish' else DANGER if bias == 'bearish' else TEXT_MUTED
         
         st.markdown(card_container(
             f'<div class="ds-card-label">💰 Funding Rate</div>'
             f'<div style="margin:6px 0;font-size:18px;font-weight:700;">'
             f'<span class="mono" style="color:{rate_color};">{rate:.4f}%</span></div>'
             f'<div style="color:{TEXT_MUTED};font-size:11px;">'
             f'Bias: {status_badge(bias.upper(), bias_color)}'
             f' | APR: <span class="mono" style="color:{TEXT_PRIMARY};">{funding_data.get("annualized", 0):.1f}%</span>'
             f'</div>'
         ), unsafe_allow_html=True)
    
    # Order Flow (enhanced 3-layer)
    order_flow = market_data.get('order_flow', {})
    if order_flow and not order_flow.get('error'):
        of_bias = order_flow.get('bias', 'neutral')
        of_score = order_flow.get('score', 0)
        of_badge_color = SUCCESS if of_bias == 'bullish' else DANGER if of_bias == 'bearish' else TEXT_MUTED
        
        # Layer details
        cvd_data = order_flow.get('cvd', {})
        taker_data = order_flow.get('taker', {})
        notable_data = order_flow.get('notable', {})
        
        cvd_trend = cvd_data.get('trend', 'n/a')
        taker_ratio = taker_data.get('ratio', 0.5)
        notable_buys = notable_data.get('large_buys', order_flow.get('large_buys', 0))
        notable_sells = notable_data.get('large_sells', order_flow.get('large_sells', 0))
        
        st.markdown(card_container(
            f'<div class="ds-card-label">📊 Order Flow</div>'
            f'<div style="margin:6px 0;">'
            f'{status_badge(of_bias.upper(), of_badge_color)}'
            f' <span class="mono" style="color:{TEXT_PRIMARY};font-size:13px;">({(of_score or 0):+.2f})</span>'
            f'</div>'
            f'<div style="color:{TEXT_MUTED};font-size:11px;line-height:1.6;">'
            f'CVD: <span style="color:{TEXT_PRIMARY};">{_esc(cvd_trend)}</span>'
            f' | Taker Buy: <span class="mono" style="color:{TEXT_PRIMARY};">{taker_ratio:.0%}</span><br/>'
            f'Notable: <span style="color:{SUCCESS};">B:{notable_buys}</span>'
            f' / <span style="color:{DANGER};">S:{notable_sells}</span>'
            f'</div>'
        ), unsafe_allow_html=True)

    # News Sentiment - DISABLED (not reliable, removed per user request)
    # Commented out - news sentiment disabled in trading logic
    # news_data = market_data.get('news')
    # if news_data is not None and isinstance(news_data, dict):
    #     news_sentiment = news_data.get('sentiment', 0)
    #     news_conf = news_data.get('confidence', 0)
    #     news_trend = news_data.get('trend', 'unknown')
    #     news_sources = news_data.get('sources', 0)
    #
    #     # Sentiment color and emoji
    #     news_color = "#26a69a" if news_sentiment > 0.1 else "#ef5350" if news_sentiment < -0.1 else "#888"
    #     news_emoji = "🟢" if news_sentiment > 0.1 else "🔴" if news_sentiment < -0.1 else "⚪"
    #     sentiment_label = "Bullish" if news_sentiment > 0.1 else "Bearish" if news_sentiment < -0.1 else "Neutral"
    #
    #     # Trend indicator
    #     trend_emoji = "📈" if news_trend == "improving" else "📉" if news_trend == "deteriorating" else "➡️"
    #
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <div class="metric-label">📰 News Sentiment</div>
    #         <div style="color: {news_color}; font-size: 14px;">{news_emoji} {sentiment_label} ({news_sentiment:+.2f})</div>
    #         <div style="color: #888; font-size: 11px;">
    #             Confidence: {news_conf:.0%} | Trend: {trend_emoji} {news_trend}<br/>
    #             Sources: {news_sources}/3 (CryptoCompare)
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)
    # else:
    #     # Show placeholder when news data is not available yet
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <div class="metric-label">📰 News Sentiment</div>
    #         <div style="color: #888; font-size: 12px;">Loading...</div>
    #         <div style="color: #666; font-size: 10px;">
    #             Waiting for first news fetch (takes ~1-2 min)
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    # HMM Regime
    regime_data = market_data.get('regime', {})
    if regime_data and not regime_data.get('error'):
        r_type = regime_data.get('type', 'UNKNOWN')
        # Colors: Green for Bull, Red for Bear, Orange for Breakout, Blue for Range
        r_color = SUCCESS if "BULL" in r_type else DANGER if "BEAR" in r_type else WARNING if "BREAKOUT" in r_type else ACCENT
        st.markdown(card_container(
            f'<div class="ds-card-label">👑 Market Regime (HMM)</div>'
            f'<div style="margin:6px 0;">{status_badge(r_type.replace("_", " "), r_color)}</div>'
            f'<div style="color:{TEXT_MUTED};font-size:11px;">'
            f'ADX: <span class="mono" style="color:{TEXT_PRIMARY};">{regime_data.get("adx", 0)}</span>'
            f' | Volatility: <span class="mono" style="color:{TEXT_PRIMARY};">{regime_data.get("volatility", 1.0)}x</span>'
            f'</div>'
        ), unsafe_allow_html=True)

    # TFT Forecast
    forecast = market_data.get('forecast')
    if forecast:
        ret_4h = forecast.get('return_4h', 0)
        ret_12h = forecast.get('return_12h', 0)
        fc_color_4h = SUCCESS if ret_4h > 0 else DANGER if ret_4h < 0 else TEXT_MUTED
        fc_color_12h = SUCCESS if ret_12h > 0 else DANGER if ret_12h < 0 else TEXT_MUTED
        fc_sign_4h = "+" if ret_4h > 0 else ""
        fc_sign_12h = "+" if ret_12h > 0 else ""
        st.markdown(card_container(
            f'<div class="ds-card-label">🚀 AI Price Forecast (TFT)</div>'
            f'<div style="margin:6px 0;font-size:14px;">'
            f'4h: <span class="mono" style="color:{fc_color_4h};font-weight:600;">{fc_sign_4h}{ret_4h}%</span>'
            f' | 12h: <span class="mono" style="color:{fc_color_12h};font-weight:600;">{fc_sign_12h}{ret_12h}%</span>'
            f'</div>'
            f'<div style="color:{TEXT_MUTED};font-size:11px;">'
            f'Consensus: <span class="mono" style="color:{TEXT_PRIMARY};">{forecast.get("consensus", 0):.2f}</span>'
            f' | Confidence: <span class="mono" style="color:{TEXT_PRIMARY};">{forecast.get("confidence", 0):.2f}</span>'
            f'</div>'
        ), unsafe_allow_html=True)

    # Ensemble Confidence Engine
    confidence = market_data.get('ensemble_confidence')
    if confidence is not None:
        conf_pct = min(100, max(0, int(confidence * 100)))
        # Map 0-1.0 to 0.25x - 2.0x for UI display (matching the ConfidenceEngine logic roughly)
        mult = 0.25 + 1.75 * confidence if confidence < 0.5 else 1.0 + 1.0 * (confidence - 0.5) * 2  # Approximate for UI
        c_color = SUCCESS if confidence > 0.6 else WARNING if confidence > 0.35 else DANGER
        
        st.markdown(card_container(
            f'<div class="ds-card-label">🧠 Ensemble Agreement</div>'
            f'<div style="margin:6px 0;font-size:18px;font-weight:700;">'
            f'<span class="mono" style="color:{c_color};">{conf_pct}% Alignment</span></div>'
            f'<div style="color:{TEXT_MUTED};font-size:11px;margin-bottom:8px;">'
            f'Position Size Multiplier: <span class="mono" style="color:{TEXT_PRIMARY};">~{mult:.1f}x</span>'
            f'</div>'
            f'{progress_bar(conf_pct, 100, c_color)}'
        ), unsafe_allow_html=True)

@st.fragment(run_every=30)
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
        state_resp = requests.get(f'{get_api_url()}/api/state', timeout=5)
        if state_resp.status_code == 200:
            state = state_resp.json()
    except Exception as e:
        logger.error(f"State fetch error: {e}")

    # 2. Fetch Live Price (Fast, from API or Fallback)
    current_price = 0.0
    try:
        # Try to get price from market API first (faster)
        clean_symbol = symbol.replace('/', '').upper()
        market_resp = requests.get(f'{get_api_url()}/api/market?symbol={clean_symbol}', timeout=5)
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
        trades_resp = requests.get(f'{get_api_url()}/api/trades', timeout=5)
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
            
    if all_trades or raw_assets:
        total_pnl = realized_pnl_total + open_pnl_total
    else:
        total_pnl = state.get('total_pnl', state.get('realized_pnl', 0))
    balance = state.get('total_balance', state.get('balance'))
    bal_str = f"${balance:,.2f}" if balance is not None else "—"
    st.markdown(
        metric_card("Portfolio Value", bal_str, delta=total_pnl, icon="💰"),
        unsafe_allow_html=True,
    )
    
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
    # CRITICAL: entry_price is the actual entry price, price is the current price
    # Must prioritize entry_price over price to avoid showing current price as entry
    if asset_state:
        if 'position_price' not in asset_state and 'entry_price' in asset_state:
            asset_state['position_price'] = asset_state['entry_price']
        elif 'position_price' not in asset_state and 'price' in asset_state:
            # Only use 'price' if entry_price is not available (legacy compatibility)
            asset_state['position_price'] = asset_state['price']
            
    render_position_card(asset_state, current_price, symbol)
    
    # 5. Render Trade History
    # Filter trades for current symbol (already fetched above)
    clean_symbol = symbol.replace('/', '').upper()
    trades = [t for t in all_trades if t.get('symbol', '').replace('/', '').upper() == clean_symbol]
        
    render_trade_history(trades)


@st.fragment(run_every=60)
def render_agent_status_fragment():
    """Render active agent status and model info with auto-refresh."""
    import requests
    import os
    from datetime import datetime
    import logging
    logger = logging.getLogger(__name__)

    # In client mode, use /api/model which returns pre-computed model stats
    if IS_CLIENT_MODE:
        try:
            model_resp = requests.get(f'{get_api_url()}/api/model', timeout=5)
            if model_resp.status_code == 200:
                model_info = model_resp.json()
                total_return = model_info.get('total_return', 0)
                win_rate = model_info.get('win_rate', 0)
                total_trades = model_info.get('total_trades', 0)
                model_date = model_info.get('model_date', 'Remote')
                model_exists = model_info.get('model_exists', True)
            else:
                total_return, win_rate, total_trades = 0, 0, 0
                model_date, model_exists = 'API error', False
        except Exception:
            total_return, win_rate, total_trades = 0, 0, 0
            model_date, model_exists = 'Connecting...', False
    else:
        # Local mode: check filesystem and compute from trades
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / 'data' / 'models' / 'ultimate_agent.zip'
        model_exists = model_path.exists()

        state = {}
        try:
            state_resp = requests.get(f'{get_api_url()}/api/state', timeout=5)
            if state_resp.status_code == 200:
                state = state_resp.json()
        except Exception:
            pass

        all_trades = state.get('trades', [])
        try:
            trades_resp = requests.get(f'{get_api_url()}/api/trades', timeout=5)
            if trades_resp.status_code == 200:
                all_trades = trades_resp.json()
        except Exception:
            pass

        realized_pnl = sum(t.get('pnl', 0) for t in all_trades if 'CLOSE' in t.get('action', '').upper() or 'EXIT' in t.get('action', '').upper())
        raw_assets = state.get('raw_state', {}).get('assets', {})

        open_pnl = 0.0
        for sym, asset_data in raw_assets.items():
            if asset_data.get('position', 0) != 0:
                current_price = asset_data.get('price', 0)
                units = asset_data.get('units', 0)
                position = asset_data.get('position', 0)
                entry_price = 0
                sym_trades = [t for t in all_trades if t.get('symbol', '').upper() == sym.upper() or t.get('asset', '').upper() == sym.upper()]
                for t in reversed(sorted(sym_trades, key=lambda x: x.get('timestamp', ''))):
                    if 'OPEN' in t.get('action', '').upper():
                        entry_price = t.get('price', 0)
                        break
                if entry_price > 0 and units > 0 and current_price > 0:
                    if position > 0:
                        open_pnl += (current_price - entry_price) * units
                    else:
                        open_pnl += (entry_price - current_price) * units

        total_pnl = realized_pnl + open_pnl
        total_return = None  # Cannot compute without knowing real initial capital

        closed_trades = [t for t in all_trades if 'CLOSE' in t.get('action', '').upper() or 'EXIT' in t.get('action', '').upper()]
        if closed_trades:
            winning = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
            win_rate = (winning / len(closed_trades) * 100)
        else:
            win_rate = 0
        total_trades = len(all_trades)

        if model_exists:
            try:
                model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                model_date = model_mtime.strftime("%Y-%m-%d")
            except Exception:
                model_date = "Unknown"
        else:
            model_date = "Not found"

    return_str = f"{'+' if total_return >= 0 else ''}{total_return:.2f}% Return" if total_return is not None else "N/A Return"
    return_color = SUCCESS if (total_return or 0) >= 0 else DANGER
    model_status_color = SUCCESS if model_exists else DANGER
    model_status_text = "✓ Model loaded" if model_exists else "✗ Model not found"

    st.markdown(card_container(
        f'<div class="ds-card-label">🤖 Active Model</div>'
        f'<div style="color:{TEXT_PRIMARY};font-size:14px;margin-top:5px;font-weight:600;">Ultimate Agent (PPO)</div>'
        f'<div style="color:{return_color};font-size:12px;font-family:monospace;">{return_str} | {win_rate:.1f}% Win Rate</div>'
        f'<div style="color:{TEXT_MUTED};font-size:11px;">Trades: {total_trades} | Model: {model_date}</div>'
        f'<div style="color:{model_status_color};font-size:11px;">{model_status_text}</div>'
    ), unsafe_allow_html=True)




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
                
        # Storage path is server-side only; omitted from client UI
                
        # Database Reset (Dev Only)
        env = os.getenv("ENVIRONMENT", "production").lower()
        if env in ["dev", "development"]:
            st.divider()
            st.markdown(f'<div class="ds-card-label" style="margin-top:8px;">🔄 Database Reset</div>', unsafe_allow_html=True)
            st.markdown(card_container(
                f'<div style="color:{DANGER};font-size:12px;font-weight:600;">⚠️ This will clear all trades and positions!</div>'
            ), unsafe_allow_html=True)

            if st.button("🗑️ Reset All Trades", type="primary"):
                try:
                    import subprocess
                    reset_script = project_root / "reset_all_storage.py"
                    if reset_script.exists():
                        result = subprocess.run(
                            [sys.executable, str(reset_script)],
                            capture_output=True,
                            text=True,
                            cwd=str(project_root)
                        )
                        if result.returncode == 0:
                            st.success("✅ Database reset successful!")
                            st.code(result.stdout)
                            st.info("🔄 Refresh the page to see changes")
                        else:
                            st.error(f"❌ Reset failed: {result.stderr}")
                    else:
                        st.error(f"❌ Reset script not found at {reset_script}")
                except Exception as e:
                    st.error(f"❌ Error running reset: {e}")

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

        st.markdown(f'<div class="ds-card-label" style="margin-top:16px;">🔑 API Status</div>', unsafe_allow_html=True)
        eth_key = os.environ.get("ETHERSCAN_API_KEY")
        sol_key = os.environ.get("SOLSCAN_API_KEY")
        xrp_key = os.environ.get("XRPSCAN_API_KEY")
        
        api_status_html = (
            f'<div style="display:flex;flex-direction:column;gap:6px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:{TEXT_MUTED};font-size:12px;">ETH</span>'
            f'{status_badge("SET", SUCCESS) if eth_key else status_badge("MISSING", DANGER)}'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:{TEXT_MUTED};font-size:12px;">SOL</span>'
            f'{status_badge("SET", SUCCESS) if sol_key else status_badge("MISSING", DANGER)}'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:{TEXT_MUTED};font-size:12px;">XRP</span>'
            f'{status_badge("SET", SUCCESS) if xrp_key else status_badge("OPTIONAL", TEXT_MUTED)}'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_container(api_status_html), unsafe_allow_html=True)

        
    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"# 🤖 DRL Trading System - {st.session_state.selected_asset}")
    
    with col2:
        refresh_status = "🔄 Auto (10s)" if st.session_state.auto_refresh else "⏸️ Paused"
        st.markdown(f"""
        <div style="text-align: right; padding-top: 10px;">
            <span style="color: {SUCCESS}; font-size: 14px;">🟢 Connected</span><br>
            <span style="color: {TEXT_MUTED}; font-size: 12px;">{refresh_status}</span>
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
        tab_chart, tab_live_portfolio, tab_performance, tab_whales, tab_testnet, tab_htf, tab_backtest = st.tabs([
            "📊 Live Chart", "💼 Live Portfolio", "📈 Performance", "🐋 On-Chain Whales", "🧪 Testnet", "🔮 HTF Agent", "🔬 Backtest"
        ])
        
        with tab_chart:
            # TradingView Chart with WebSocket
            # Fetch trades from testnet trade history (exchange fills) — state.get('trades') is empty
            trades = state.get('trades', [])
            if not trades:
                try:
                    _api = get_api_url()
                    _sym_clean = st.session_state.selected_asset.replace('/', '').upper()
                    _trades_resp = requests.get(f'{_api}/api/testnet/trades?symbol={_sym_clean}&limit=100', timeout=10)
                    if _trades_resp.ok:
                        trades = _trades_resp.json().get('trades', [])
                except Exception:
                    trades = []
            # Fetch market structure (BOS/CHOCH) signals for chart overlay
            _ms_data = None
            try:
                _api_ms = get_api_url()
                _sym_ms = st.session_state.selected_asset.replace('/', '').upper()
                _tf_ms = st.session_state.timeframe
                _ms_resp = requests.get(
                    f'{_api_ms}/api/market-structure?symbol={_sym_ms}&timeframe={_tf_ms}&limit=500',
                    timeout=10
                )
                if _ms_resp.ok:
                    _ms_data = _ms_resp.json()
                    if _ms_data.get('error'):
                        _ms_data = None
            except Exception:
                _ms_data = None

            chart_html = create_tradingview_chart_with_websocket(df, trades, st.session_state.timeframe, st.session_state.selected_asset, market_structure=_ms_data)
            # Append timestamp comment to force re-render since components.html doesn't support key
            # Create a placeholder for the chart to force re-rendering
            chart_placeholder = st.empty()
            
            # Append timestamp comment to force re-render since components.html doesn't support key
            current_time = time.time()
            chart_html += f"<!-- {current_time} -->"
            
            with chart_placeholder:
                components.html(chart_html, height=600)
                


            # Info about trade markers
            num_opens = len([t for t in trades if 'OPEN' in t.get('action', '')])
            num_closes = len([t for t in trades if 'CLOSE' in t.get('action', '')])
            _ms_info = ""
            if _ms_data:
                _n_sh = len(_ms_data.get('swing_highs', []))
                _n_sl = len(_ms_data.get('swing_lows', []))
                _n_bos = len(_ms_data.get('bos_signals', []))
                _n_choch = len(_ms_data.get('choch_signals', []))
                _ms_trend = _ms_data.get('trend', 'ranging')
                _ms_info = f" | 🏗 Structure: {_n_sh}H/{_n_sl}L swings, {_n_bos} BOS, {_n_choch} CHOCH, trend={_ms_trend}"
            st.caption(f"📍 {num_opens} entries + {num_closes} exits on chart • 🟢 LONG ▲ • 🔴 SHORT ▼ • EXIT(SL) ■ • EXIT(TP) ■ • EXIT ●{_ms_info}")
            
            # Trading Controls Section
            st.markdown("---")
            st.markdown("### 🎮 Trading Controls")

            if IS_CLIENT_MODE:
                st.info("🌐 **Client Mode** — Trading bot is managed on the remote server. Use the server dashboard to start/stop the bot or place manual trades.")
                if st.button("🔄 Refresh Data", key="refresh_data", use_container_width=True):
                    st.rerun()
            else:
                # Bot status check (server-side only)
                import subprocess
                bot_running = False
                try:
                    result = subprocess.run(['pgrep', '-f', 'live_trading'], capture_output=True, text=True)
                    bot_running = result.returncode == 0
                except Exception:
                    pass

                if bot_running:
                    st.success("🟢 **Trading Bot is RUNNING** (Multi-Asset Mode)")
                else:
                    st.warning("🟠 **Trading Bot is STOPPED**")

                ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

                with ctrl_col1:
                    if not bot_running:
                        if st.button("▶️ Start Trading", key="start_trading", use_container_width=True, type="primary"):
                            try:
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
                                time.sleep(2)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to start: {e}")
                    else:
                        if st.button("⏹️ Stop Trading", key="stop_trading", use_container_width=True, type="secondary"):
                            try:
                                subprocess.run(['pkill', '-f', 'live_trading'], check=False)
                                st.info("✓ Trading bot stopped")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to stop: {e}")

                with ctrl_col2:
                    if st.button("📈 Open Long", key="open_long", use_container_width=True):
                        trade = {
                            'timestamp': datetime.now().isoformat(),
                            'action': 'OPEN_LONG',
                            'price': current_price,
                            'pnl': 0,
                            'balance': state.get('balance'),
                            'position': 1,
                            'reason': 'manual',
                            'symbol': st.session_state.selected_asset,
                            'asset': st.session_state.selected_asset,
                        }
                        storage.log_trade(trade)
                        st.success(f"✓ Opened LONG @ ${current_price:,.2f}")
                        time.sleep(0.5)
                        st.rerun()

                with ctrl_col3:
                    if st.button("📉 Open Short", key="open_short", use_container_width=True):
                        trade = {
                            'timestamp': datetime.now().isoformat(),
                            'action': 'OPEN_SHORT',
                            'price': current_price,
                            'pnl': 0,
                            'balance': state.get('balance'),
                            'position': -1,
                            'reason': 'manual',
                            'symbol': st.session_state.selected_asset,
                            'asset': st.session_state.selected_asset,
                        }
                        storage.log_trade(trade)
                        st.success(f"✓ Opened SHORT @ ${current_price:,.2f}")
                        time.sleep(0.5)
                        st.rerun()

                with ctrl_col4:
                    if st.button("🚪 Close Position", key="close_position", use_container_width=True):
                        position = state.get('position', 0)
                        if position != 0:
                            action = 'CLOSE_LONG' if position == 1 else 'CLOSE_SHORT'
                            trade = {
                                'timestamp': datetime.now().isoformat(),
                                'action': action,
                                'price': current_price,
                                'pnl': 0,
                                'balance': state.get('balance'),
                                'position': 0,
                                'reason': 'manual',
                                'symbol': st.session_state.selected_asset,
                                'asset': st.session_state.selected_asset,
                            }
                            storage.log_trade(trade)
                            st.success(f"✓ Closed position @ ${current_price:,.2f}")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.info("No position to close")

                st.markdown("")
                action_col1, action_col2 = st.columns(2)

                with action_col1:
                    if st.button("🔄 Refresh Data", key="refresh_data", use_container_width=True):
                        st.rerun()

                with action_col2:
                    if st.button("🗑️ Clear Trade Log", key="clear_log", use_container_width=True):
                        try:
                            log_file = project_root / 'logs' / 'trading_log.json'
                            state_file = project_root / 'logs' / 'trading_state.json'
                            log_file.write_text('')
                            if state_file.exists():
                                state_file.unlink()
                            st.success("✓ Trade log cleared")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to clear log: {e}")
        
        with tab_live_portfolio:
            # ─── Compute portfolio metrics from trade data ───
            all_trades_lp = []
            try:
                # Use API in client mode, local storage otherwise
                all_trades_lp = load_trading_log()
                if not IS_CLIENT_MODE:
                    # Apply reset filter (local mode only — API already filters)
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

            # FIX: State structure is state['assets'], not state['raw_state']['assets']
            raw_assets = state.get('assets', {})
            
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

                # FIX: Determine final status from last trade (if it was a CLOSE, position is FLAT)
                if sorted_trades:
                    last_trade = sorted_trades[-1]
                    last_action = last_trade.get('action', '').upper()
                    if 'CLOSE' in last_action or 'EXIT' in last_action:
                        sym_status = 'FLAT'

                # Check current state for position status (this overrides trade-based status)
                if sym in raw_assets:
                    asset_data = raw_assets[sym]
                    if asset_data.get('position', 0) != 0:
                        # Calculate unrealized P&L from entry price vs current price
                        current_price = asset_data.get('price', 0)
                        units = asset_data.get('units', 0)
                        position = asset_data.get('position', 0)

                        # Find entry price from last OPEN trade
                        entry_price = 0
                        for t in reversed(sorted_trades):
                            if 'OPEN' in t.get('action', '').upper():
                                entry_price = t.get('price', 0)
                                break

                        # Calculate unrealized P&L
                        if entry_price > 0 and units > 0 and current_price > 0:
                            if position > 0:  # LONG
                                sym_open_pnl = (current_price - entry_price) * units
                            else:  # SHORT
                                sym_open_pnl = (entry_price - current_price) * units

                        sym_status = 'LONG' if position > 0 else 'SHORT'
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
                # Use actual balance from state file (not accumulated trade log PnL)
                sym_state_balance = raw_assets.get(sym, {}).get('balance', 0)
                sym_state_rpnl = raw_assets.get(sym, {}).get('realized_pnl', 0)
                if sym_state_balance > 0:
                    sym_equity = sym_state_balance + sym_open_pnl
                    sym_realized = sym_state_rpnl
                else:
                    sym_equity = 5000 + sym_realized + sym_open_pnl
                
                asset_rows.append({
                    'symbol': display_sym,
                    'raw_symbol': sym,
                    'status': sym_status,
                    'price': sym_price,
                    'equity': sym_equity,
                    'pnl': sym_realized + sym_open_pnl,
                    'realized': sym_realized,
                    'open_pnl': sym_open_pnl,
                    'trades': sym_closed + (1 if sym_status != 'FLAT' else 0),
                    'open_trades': 1 if sym_status != 'FLAT' else 0,
                    'win_rate': (sym_wins / sym_closed * 100) if sym_closed > 0 else 0,
                    'wins': sym_wins,
                    'closed': sym_closed,
                    'best': sym_best,
                    'worst': sym_worst,
                })
            
            # Overall metrics — use state file data (not accumulated trade log PnL which inflates)
            # Recalculate totals from asset_rows which now prefer state file balances
            realized_pnl_total = sum(r.get('realized', 0) for r in asset_rows)
            open_pnl_total = sum(r.get('open_pnl', 0) for r in asset_rows)
            lp_grand_total_pnl = realized_pnl_total + open_pnl_total
            lp_total_balance = sum(r.get('equity', 0) for r in asset_rows) if asset_rows else state.get('total_balance', state.get('balance'))
            overall_win_rate = (total_winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
            total_trades_count = total_closed_trades + total_open_trades
            
            lp_active_assets_count = len(asset_rows) if asset_rows else len(state.get('available_assets', []))
            
            # System status: check if API server is reachable and any bot process is running
            if IS_CLIENT_MODE:
                # Client mode: server reachable = online (we got state data)
                is_online = bool(state and state.get('available_assets'))
            else:
                # Server mode: check if any trading bot process is running
                is_online = (check_process_running("live_trading_multi.py") or
                             check_process_running("live_trading_htf.py") or
                             check_process_running("api_server"))
            status_dot = '🟢' if is_online else '🔴'
            status_text = 'Online' if is_online else 'Offline'
            status_color = SUCCESS if is_online else DANGER
            
            # ─── Render Live Portfolio with Design System ───

            # Header with badges
            st.markdown(
                section_header("Live Portfolio", icon="💼")
                + " " + status_badge("LIVE TRADING", SUCCESS)
                + " " + status_badge(f"{status_dot} {status_text}", status_color),
                unsafe_allow_html=True,
            )

            # ── Metric Row 1: Realized, Open PnL, Win Rate, Trades ──
            r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1])
            with r1c1:
                _rpnl_val = f'{_pnl_sign(realized_pnl_total)}${abs(realized_pnl_total):,.2f}'
                st.markdown(metric_card(
                    label="Realized PNL",
                    value=_rpnl_val,
                    value_color=_pnl_color(realized_pnl_total),
                ), unsafe_allow_html=True)
            with r1c2:
                _opnl_val = f'{_pnl_sign(open_pnl_total)}${abs(open_pnl_total):,.2f}'
                st.markdown(metric_card(
                    label="Open PNL",
                    value=_opnl_val,
                    value_color=_pnl_color(open_pnl_total),
                ), unsafe_allow_html=True)
            with r1c3:
                _wr_sub = f'<div style="color:{TEXT_MUTED};font-size:11px;margin-top:4px;">{total_winning_trades}W / {total_closed_trades - total_winning_trades}L</div>'
                st.markdown(metric_card(
                    label="Win Rate",
                    value=f'{overall_win_rate:.0f}%',
                ) + _wr_sub, unsafe_allow_html=True)
            with r1c4:
                _tr_sub = f'<div style="color:{TEXT_MUTED};font-size:11px;margin-top:4px;">{total_open_trades} open · {total_closed_trades} closed</div>'
                st.markdown(metric_card(
                    label="Trades",
                    value=str(total_trades_count),
                ) + _tr_sub, unsafe_allow_html=True)

            # ── Metric Row 2: Portfolio Value, Total P&L, Active Assets ──
            r2c1, r2c2, r2c3 = st.columns([1, 1, 1])
            with r2c1:
                _pv = f'${lp_total_balance:,.2f}' if lp_total_balance is not None else '—'
                st.markdown(metric_card(label="Portfolio Value", value=_pv), unsafe_allow_html=True)
            with r2c2:
                _tpnl_val = f'{_pnl_sign(lp_grand_total_pnl)}${abs(lp_grand_total_pnl):,.2f}'
                st.markdown(metric_card(
                    label="Total P&L",
                    value=_tpnl_val,
                    value_color=_pnl_color(lp_grand_total_pnl),
                ), unsafe_allow_html=True)
            with r2c3:
                st.markdown(metric_card(label="Active Assets", value=str(lp_active_assets_count)), unsafe_allow_html=True)

            # ── Equity Curve (Pure SVG — no external deps) ──
            eq_pct = list(equity_points)
            svg_w, svg_h = 900, 160
            n_points = len(eq_pct)
            eq_min_val = min(eq_pct) if eq_pct else 0
            eq_max_val = max(eq_pct) if eq_pct else 0
            eq_range_val = max(abs(eq_min_val), abs(eq_max_val), 0.01)
            padding_y = 20

            svg_points = []
            svg_fill_points = []
            for i, val in enumerate(eq_pct):
                x = (i / max(n_points - 1, 1)) * svg_w
                y = svg_h - padding_y - ((val + eq_range_val) / (2 * eq_range_val)) * (svg_h - 2 * padding_y)
                svg_points.append(f"{x:.1f},{y:.1f}")
                svg_fill_points.append(f"{x:.1f},{y:.1f}")

            polyline_str = ' '.join(svg_points)
            fill_points = svg_fill_points.copy()
            if fill_points:
                fill_points.append(f"{svg_w:.1f},{svg_h - padding_y:.1f}")
                fill_points.append(f"0,{svg_h - padding_y:.1f}")
            fill_str = ' '.join(fill_points)

            last_eq = eq_pct[-1] if eq_pct else 0
            eq_line_color = SUCCESS if last_eq >= 0 else DANGER
            fill_color_start = f'rgba(16,185,129,0.3)' if last_eq >= 0 else f'rgba(239,68,68,0.3)'
            fill_color_end = f'rgba(16,185,129,0.0)' if last_eq >= 0 else f'rgba(239,68,68,0.0)'
            zero_y = svg_h - padding_y - ((0 + eq_range_val) / (2 * eq_range_val)) * (svg_h - 2 * padding_y)
            last_x = svg_w if n_points <= 1 else ((n_points - 1) / max(n_points - 1, 1)) * svg_w
            last_y = svg_h - padding_y - ((last_eq + eq_range_val) / (2 * eq_range_val)) * (svg_h - 2 * padding_y)
            top_label = f"+{eq_range_val:.1f}%"
            bot_label = f"-{eq_range_val:.1f}%"

            svg_chart = f'''<svg width="100%" viewBox="0 0 {svg_w} {svg_h}" preserveAspectRatio="none" style="display:block;">
                <defs><linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="{fill_color_start}"/>
                    <stop offset="100%" stop-color="{fill_color_end}"/>
                </linearGradient></defs>
                <line x1="0" y1="{zero_y:.1f}" x2="{svg_w}" y2="{zero_y:.1f}" stroke="rgba(255,255,255,0.08)" stroke-width="1" stroke-dasharray="4,4"/>
                <polygon points="{fill_str}" fill="url(#eqGrad)"/>
                <polyline points="{polyline_str}" fill="none" stroke="{eq_line_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="{eq_line_color}" stroke="#fff" stroke-width="1.5"/>
                <text x="{svg_w - 5}" y="{padding_y + 4}" fill="{TEXT_MUTED}" font-size="10" text-anchor="end" font-family="monospace">{top_label}</text>
                <text x="{svg_w - 5}" y="{svg_h - padding_y + 12}" fill="{TEXT_MUTED}" font-size="10" text-anchor="end" font-family="monospace">{bot_label}</text>
                <text x="5" y="{zero_y - 4:.1f}" fill="{BORDER}" font-size="9" font-family="monospace">0%</text>
            </svg>'''

            eq_inner = (
                f'<div style="font-size:15px;font-weight:600;color:{TEXT_PRIMARY};margin-bottom:2px;">Equity Curve</div>'
                f'<div style="color:{TEXT_MUTED};font-size:11px;margin-bottom:12px;">Cumulative P&amp;L from closed trades</div>'
                + svg_chart
            )
            st.markdown(card_container(eq_inner), unsafe_allow_html=True)

            # ── Asset Table using styled_table() ──
            table_headers = ["Asset", "Status", "Price", "Equity", "PNL ($)", "Trades", "Win Rate", "Best", "Worst"]
            table_rows = []
            for row in asset_rows:
                # Status badge via design system
                if row['status'] == 'LONG':
                    row_status = status_badge("● LONG", SUCCESS)
                elif row['status'] == 'SHORT':
                    row_status = status_badge("● SHORT", DANGER)
                else:
                    row_status = status_badge("● FLAT", TEXT_MUTED)

                # PnL via design system
                row_pnl = pnl_text(row['pnl'])

                # Trades
                trades_str = str(row['trades'])
                if row['open_trades'] > 0:
                    trades_str += f' <span style="color:{TEXT_MUTED};">(+{row["open_trades"]})</span>'

                # Win rate with progress bar
                wr = row['win_rate']
                wr_color = SUCCESS if wr >= 50 else WARNING if wr > 0 else TEXT_MUTED
                wr_cell = (
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<div style="flex:1;min-width:60px;">{progress_bar(wr, 100, wr_color)}</div>'
                    f'<span style="color:{TEXT_PRIMARY};font-size:12px;min-width:35px;">{wr:.0f}%</span>'
                    f'</div>'
                )

                # Best / Worst
                best_cell = pnl_text(row['best']) if row['best'] is not None else f'<span style="color:{TEXT_MUTED};">—</span>'
                worst_cell = pnl_text(row['worst']) if row['worst'] is not None else f'<span style="color:{TEXT_MUTED};">—</span>'

                table_rows.append([
                    f'<span style="font-weight:600;color:{TEXT_PRIMARY};">{_esc(row["symbol"])}</span>',
                    row_status,
                    f'<span class="mono" style="color:{TEXT_PRIMARY};">${row["price"]:,.2f}</span>',
                    f'<span class="mono" style="color:{TEXT_PRIMARY};">${row["equity"]:,.2f}</span>',
                    row_pnl,
                    trades_str,
                    wr_cell,
                    best_cell,
                    worst_cell,
                ])

            if table_rows:
                st.markdown(styled_table(table_headers, table_rows), unsafe_allow_html=True)
            else:
                st.markdown(card_container(
                    f'<div style="text-align:center;padding:30px;color:{TEXT_MUTED};font-size:14px;">'
                    f'No trades recorded yet. Start the trading bot to see portfolio data.</div>'
                ), unsafe_allow_html=True)

            # Footer
            st.markdown(
                f'<div style="text-align:center;color:{TEXT_MUTED};font-size:11px;margin-top:16px;">'
                f'DRL Trading System · Signals from PPO + Composite Scoring · Connected to OKX</div>',
                unsafe_allow_html=True,
            )

        with tab_performance:
            # ─── Fetch REAL trade data from API ───
            perf_trades = []
            try:
                perf_trades = load_trading_log()
            except Exception:
                pass

            # ─── Compute metrics from closed trades ───
            STARTING_BALANCE = 20000.0
            closed_trades = []
            for t in perf_trades:
                action = t.get('action', '').upper()
                if 'CLOSE' in action or 'EXIT' in action:
                    closed_trades.append(t)

            closed_trades.sort(key=lambda x: x.get('timestamp', ''))

            total_closed = len(closed_trades)
            pnl_list = [t.get('pnl', 0) or 0 for t in closed_trades]
            total_pnl_perf = sum(pnl_list)
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]
            win_count = len(wins)
            loss_count = len(losses)

            # Win Rate
            win_rate_perf = (win_count / total_closed * 100) if total_closed > 0 else 0

            # Total Return
            total_return_pct = (total_pnl_perf / STARTING_BALANCE * 100) if STARTING_BALANCE > 0 else 0

            # Best / Worst trade
            best_trade_pnl = max(pnl_list) if pnl_list else 0
            worst_trade_pnl = min(pnl_list) if pnl_list else 0
            best_trade_sym = ''
            worst_trade_sym = ''
            for t in closed_trades:
                p = t.get('pnl', 0) or 0
                sym = t.get('symbol', t.get('asset', ''))
                if sym.endswith('USDT'):
                    sym = sym[:-4]
                if p == best_trade_pnl and not best_trade_sym:
                    best_trade_sym = sym
                if p == worst_trade_pnl and not worst_trade_sym:
                    worst_trade_sym = sym

            # Profit Factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)

            # Sharpe Ratio from individual trade returns
            sharpe_perf = None
            if len(pnl_list) >= 2:
                trade_returns = [p / STARTING_BALANCE for p in pnl_list]
                mean_ret = sum(trade_returns) / len(trade_returns)
                var_ret = sum((r - mean_ret) ** 2 for r in trade_returns) / (len(trade_returns) - 1)
                std_ret = var_ret ** 0.5
                if std_ret > 0:
                    sharpe_perf = (mean_ret / std_ret) * (252 ** 0.5)  # Annualized

            # Max Drawdown from equity curve
            equity_curve = [0.0]
            for p in pnl_list:
                equity_curve.append(equity_curve[-1] + p)
            peak = equity_curve[0]
            max_dd = 0.0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd
            max_dd_pct = (max_dd / STARTING_BALANCE * 100) if STARTING_BALANCE > 0 else 0

            # ─── Render Performance Tab ───
            st.markdown(
                section_header("Trading Performance", icon="📈"),
                unsafe_allow_html=True,
            )

            # ── Row 1: Total Return, Sharpe, Max Drawdown ──
            p1c1, p1c2, p1c3 = st.columns(3)
            with p1c1:
                _ret_color = _pnl_color(total_return_pct)
                _ret_val = f'{_pnl_sign(total_return_pct)}{abs(total_return_pct):.2f}%'
                st.markdown(metric_card(
                    label="Total Return",
                    value=_ret_val,
                    icon="📊",
                    value_color=_ret_color,
                ), unsafe_allow_html=True)
            with p1c2:
                _sharpe_val = f'{sharpe_perf:.2f}' if sharpe_perf is not None else '—'
                st.markdown(metric_card(
                    label="Sharpe Ratio",
                    value=_sharpe_val,
                    icon="📐",
                ), unsafe_allow_html=True)
            with p1c3:
                st.markdown(metric_card(
                    label="Max Drawdown",
                    value=f'-{max_dd_pct:.2f}%',
                    icon="📉",
                ), unsafe_allow_html=True)

            # ── Row 2: Total Trades, Win Rate, Profit Factor ──
            p2c1, p2c2, p2c3 = st.columns(3)
            with p2c1:
                _tr_sub = f'<div style="color:{TEXT_MUTED};font-size:11px;margin-top:4px;">{win_count}W / {loss_count}L</div>'
                st.markdown(metric_card(
                    label="Total Trades",
                    value=str(total_closed),
                    icon="🔄",
                ) + _tr_sub, unsafe_allow_html=True)
            with p2c2:
                _wr_color = SUCCESS if win_rate_perf >= 50 else DANGER
                st.markdown(metric_card(
                    label="Win Rate",
                    value=f'{win_rate_perf:.1f}%',
                    icon="🎯",
                    value_color=_wr_color,
                ), unsafe_allow_html=True)
            with p2c3:
                _pf_val = f'{profit_factor:.2f}' if profit_factor != float('inf') else '∞'
                st.markdown(metric_card(
                    label="Profit Factor",
                    value=_pf_val,
                    icon="⚖️",
                ), unsafe_allow_html=True)

            # ── Row 3: Total P&L, Best Trade, Worst Trade ──
            p3c1, p3c2, p3c3 = st.columns(3)
            with p3c1:
                _tpnl_color = _pnl_color(total_pnl_perf)
                _tpnl_val = f'{_pnl_sign(total_pnl_perf)}${abs(total_pnl_perf):,.2f}'
                st.markdown(metric_card(
                    label="Total P&L",
                    value=_tpnl_val,
                    icon="💰",
                    value_color=_tpnl_color,
                ), unsafe_allow_html=True)
            with p3c2:
                _best_label = f'Best Trade ({best_trade_sym})' if best_trade_sym else 'Best Trade'
                st.markdown(metric_card(
                    label=_best_label,
                    value=f'{_pnl_sign(best_trade_pnl)}${abs(best_trade_pnl):,.2f}' if pnl_list else '—',
                    icon="🏆",
                    value_color=SUCCESS if pnl_list else None,
                ), unsafe_allow_html=True)
            with p3c3:
                _worst_label = f'Worst Trade ({worst_trade_sym})' if worst_trade_sym else 'Worst Trade'
                st.markdown(metric_card(
                    label=_worst_label,
                    value=f'{_pnl_sign(worst_trade_pnl)}${abs(worst_trade_pnl):,.2f}' if pnl_list else '—',
                    icon="💔",
                    value_color=DANGER if pnl_list else None,
                ), unsafe_allow_html=True)

            # ── Equity Curve (SVG) ──
            if len(equity_curve) > 1:
                st.markdown(
                    section_header("Equity Curve", icon="📈"),
                    unsafe_allow_html=True,
                )
                eq_svg_w, eq_svg_h = 900, 180
                eq_n = len(equity_curve)
                eq_min = min(equity_curve)
                eq_max = max(equity_curve)
                eq_range = max(abs(eq_min), abs(eq_max), 0.01)
                eq_pad = 20

                svg_pts = []
                svg_fill = []
                for i, val in enumerate(equity_curve):
                    x = (i / max(eq_n - 1, 1)) * eq_svg_w
                    y = eq_svg_h - eq_pad - ((val + eq_range) / (2 * eq_range)) * (eq_svg_h - 2 * eq_pad)
                    svg_pts.append(f"{x:.1f},{y:.1f}")
                    svg_fill.append(f"{x:.1f},{y:.1f}")

                poly_str = ' '.join(svg_pts)
                fill_pts = svg_fill.copy()
                if fill_pts:
                    fill_pts.append(f"{eq_svg_w:.1f},{eq_svg_h - eq_pad:.1f}")
                    fill_pts.append(f"0,{eq_svg_h - eq_pad:.1f}")
                fill_str_eq = ' '.join(fill_pts)

                last_eq_val = equity_curve[-1]
                eq_color = SUCCESS if last_eq_val >= 0 else DANGER
                eq_fill_start = 'rgba(16,185,129,0.3)' if last_eq_val >= 0 else 'rgba(239,68,68,0.3)'
                eq_fill_end = 'rgba(16,185,129,0.0)' if last_eq_val >= 0 else 'rgba(239,68,68,0.0)'
                zero_y_eq = eq_svg_h - eq_pad - ((0 + eq_range) / (2 * eq_range)) * (eq_svg_h - 2 * eq_pad)

                eq_svg = f'''<svg width="100%" viewBox="0 0 {eq_svg_w} {eq_svg_h}" preserveAspectRatio="none" style="display:block;">
                    <defs><linearGradient id="perfEqGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="{eq_fill_start}"/>
                        <stop offset="100%" stop-color="{eq_fill_end}"/>
                    </linearGradient></defs>
                    <line x1="0" y1="{zero_y_eq:.1f}" x2="{eq_svg_w}" y2="{zero_y_eq:.1f}" stroke="rgba(255,255,255,0.08)" stroke-width="1" stroke-dasharray="4,4"/>
                    <polygon points="{fill_str_eq}" fill="url(#perfEqGrad)"/>
                    <polyline points="{poly_str}" fill="none" stroke="{eq_color}" stroke-width="2.5"/>
                </svg>'''
                st.markdown(card_container(eq_svg), unsafe_allow_html=True)

            # ── Trade History Table ──
            if closed_trades:
                st.markdown(
                    section_header("Closed Trades", icon="📋"),
                    unsafe_allow_html=True,
                )
                trade_rows = []
                for t in closed_trades:
                    sym = t.get('symbol', t.get('asset', ''))
                    if sym.endswith('USDT'):
                        sym = sym[:-4] + '/USDT'
                    ts = t.get('timestamp', '')
                    if 'T' in ts:
                        ts = ts.split('T')[0] + ' ' + ts.split('T')[1][:8]
                    pnl_val = t.get('pnl', 0) or 0
                    price = t.get('price', 0) or 0
                    reason = t.get('reason', '—')
                    trade_rows.append([
                        ts,
                        sym,
                        f'${price:,.2f}',
                        pnl_text(pnl_val),
                        reason,
                    ])
                st.markdown(
                    styled_table(
                        headers=["Time", "Symbol", "Price", "P&L", "Reason"],
                        rows=trade_rows,
                    ),
                    unsafe_allow_html=True,
                )

            # ── Walk-Forward Validation Results ──
            wf_file = project_root / 'data' / 'models' / 'htf_walkforward_50pct_v2' / 'walk_forward_summary.json'
            if wf_file.exists():
                st.markdown(
                    section_header("Walk-Forward Validation", icon="🔬"),
                    unsafe_allow_html=True,
                )
                try:
                    with open(wf_file, 'r') as f:
                        wf = json.load(f)
                    wf_sharpe = wf.get('oos_sharpe_mean', 0)
                    wf_return = wf.get('oos_return_mean_pct', 0)
                    wf_dd = wf.get('oos_drawdown_mean_pct', 0)
                    wf_pos_folds = wf.get('positive_fold_pct', 0)
                    wf_total_folds = wf.get('total_folds', 0)
                    wf_verdict = wf.get('overfit_verdict', '—')

                    wfc1, wfc2, wfc3, wfc4 = st.columns(4)
                    with wfc1:
                        st.markdown(metric_card(
                            label="OOS Sharpe",
                            value=f'{wf_sharpe:.2f}',
                            icon="📐",
                        ), unsafe_allow_html=True)
                    with wfc2:
                        _wfr_color = _pnl_color(wf_return)
                        st.markdown(metric_card(
                            label="OOS Return",
                            value=f'{_pnl_sign(wf_return)}{abs(wf_return):.1f}%',
                            icon="📊",
                            value_color=_wfr_color,
                        ), unsafe_allow_html=True)
                    with wfc3:
                        st.markdown(metric_card(
                            label="Mean Drawdown",
                            value=f'-{wf_dd:.1f}%',
                            icon="📉",
                        ), unsafe_allow_html=True)
                    with wfc4:
                        _folds_color = SUCCESS if wf_pos_folds >= 75 else (WARNING if wf_pos_folds >= 50 else DANGER)
                        st.markdown(metric_card(
                            label="Positive Folds",
                            value=f'{wf_pos_folds:.0f}%',
                            icon="✅",
                            value_color=_folds_color,
                        ), unsafe_allow_html=True)

                    # Verdict badge
                    verdict_color = SUCCESS if 'EXCELLENT' in wf_verdict.upper() else (WARNING if 'ACCEPTABLE' in wf_verdict.upper() else DANGER)
                    st.markdown(
                        card_container(
                            f'<div style="text-align:center;padding:8px 0;">'
                            f'<span style="color:{TEXT_MUTED};font-size:11px;text-transform:uppercase;letter-spacing:0.8px;">Overfit Verdict</span><br/>'
                            f'<span style="color:{verdict_color};font-size:16px;font-weight:700;">{_esc(wf_verdict)}</span>'
                            f'</div>'
                        ),
                        unsafe_allow_html=True,
                    )

                    # Per-fold table
                    per_fold = wf.get('per_fold', [])
                    if per_fold:
                        fold_rows = []
                        for fold in per_fold:
                            f_sharpe = fold.get('oos_sharpe', 0)
                            f_ret = fold.get('oos_return_pct', 0)
                            f_dd = fold.get('oos_drawdown_pct', 0)
                            f_wr = fold.get('oos_win_rate', 0) * 100
                            f_trades = int(fold.get('oos_trades', 0))
                            f_period = fold.get('test_period', '—')
                            fold_rows.append([
                                f"Fold {fold.get('fold', '?')}",
                                f_period,
                                f'{f_sharpe:.2f}',
                                pnl_text(f_ret) if f_ret != 0 else '0.00%',
                                f'-{f_dd:.1f}%',
                                f'{f_wr:.0f}%',
                                str(f_trades),
                            ])
                        st.markdown(
                            styled_table(
                                headers=["Fold", "Period", "Sharpe", "Return", "Drawdown", "Win Rate", "Trades"],
                                rows=fold_rows,
                            ),
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass
        
        with tab_whales:
            st.markdown("### 🐋 On-Chain Whale Analytics")
            
            whale_alerts = state.get('whale_alerts', [])
            
            if whale_alerts:
                import pandas as pd
                import plotly.express as px
                
                df = pd.DataFrame(whale_alerts)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Approximate USD prices for aggregation (since we only have raw crypto values)
                # This allows us to "tell the story" of total USD economic volume moved
                price_map = {'BTC': 70000, 'ETH': 3500, 'SOL': 150, 'XRP': 0.6}
                df['usd_value'] = df.apply(lambda row: row['value'] * price_map.get(row['chain'], 1), axis=1)
                
                total_usd = df['usd_value'].sum()
                top_chain = df.groupby('chain')['usd_value'].sum().idxmax() if not df.empty else "N/A"
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Recent Alerts", len(df))
                col2.metric("Trailing Vol (USD)", f"${total_usd/1e6:.1f}M")
                col3.metric("Most Active Chain", top_chain)
                
                eth_whales = len(df[df['chain'] == 'ETH'])
                xrp_whales = len(df[df['chain'] == 'XRP'])
                sol_whales = len(df[df['chain'] == 'SOL'])
                btc_whales = len(df[df['chain'] == 'BTC'])
                col4.metric("Network Activity", f"BTC:{btc_whales} ETH:{eth_whales} SOL:{sol_whales} XRP:{xrp_whales}")
                
                st.divider()
                
                chart_col, table_col = st.columns([1.2, 1])
                
                with chart_col:
                    st.markdown("#### 📊 Whale Volume by Chain (USD)")
                    # Group by chain and enforce order
                    chain_vol = df.groupby('chain')['usd_value'].sum().reset_index()
                    all_chains = pd.DataFrame({'chain': ['BTC', 'ETH', 'SOL', 'XRP']})
                    chain_vol = pd.merge(all_chains, chain_vol, on='chain', how='left').fillna(0)
                    
                    fig = px.bar(
                        chain_vol, x='chain', y='usd_value', 
                        color='chain', text_auto='.2s',
                        color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA', 'SOL': '#14F195', 'XRP': '#00AAE4'},
                        labels={'usd_value': 'Estimated USD Volume', 'chain': 'Network'}
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#8b949e'), height=250, margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                        xaxis={'categoryorder':'array', 'categoryarray':['BTC','ETH','SOL','XRP']}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### 🏛 Volume by Entity Type")
                    if 'wallet_type' in df.columns:
                        type_vol = df.groupby('wallet_type')['usd_value'].sum().reset_index()
                        fig_type = px.pie(
                            type_vol, values='usd_value', names='wallet_type', hole=0.4,
                            color_discrete_sequence=['#F7931A', '#627EEA', '#14F195', '#00AAE4', '#888888']
                        )
                        fig_type.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#8b949e'), height=250, margin=dict(l=0, r=0, t=30, b=0),
                            showlegend=True
                        )
                        st.plotly_chart(fig_type, use_container_width=True)
                    
                with table_col:
                    st.markdown("#### 📝 Latest Transaction Feed")
                    # Format table
                    cols_to_keep = ['datetime', 'chain', 'wallet_label', 'wallet_type', 'value', 'usd_value', 'link']
                    exist_cols = [c for c in cols_to_keep if c in df.columns]
                    display_df = df[exist_cols].copy()
                    display_df = display_df.sort_values('datetime', ascending=False)
                    display_df['datetime'] = display_df['datetime'].dt.strftime('%H:%M:%S')
                    display_df['value'] = display_df.apply(lambda r: f"{r['value']:,.0f} {r['chain']}" if r['value'] >= 1000 else (f"{r['value']:,.2f} {r['chain']}" if r['value'] >= 1 else f"{r['value']:,.4f} {r['chain']}"), axis=1)
                    display_df['usd_value'] = display_df['usd_value'].apply(lambda x: f"${x/1e6:,.1f}M" if x >= 1e6 else f"${x/1000:,.0f}k")
                    
                    rename_map = {
                        'datetime': 'Time', 
                        'chain': 'Net',
                        'wallet_label': 'Entity',
                        'wallet_type': 'Type',
                        'value': 'Amount', 
                        'usd_value': 'Est. USD',
                        'link': 'Explorer'
                    }
                    display_df.rename(columns=rename_map, inplace=True)
                    
                    try:
                        st.dataframe(
                            display_df,
                            column_config={
                                "Explorer": st.column_config.LinkColumn("Explorer", display_text="View TX ↗")
                            },
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                    except Exception:
                        # Fallback for older Streamlit versions without column_config
                        st.dataframe(display_df.drop(columns=['Explorer']), use_container_width=True, height=400)
                
                st.divider()
                st.markdown("#### 🤖 AI Momentum Predictions")
                st.caption("Real-time directional predictions based on institutional flow and wallet behavioral analysis.")
                
                pred_cols = st.columns(4)
                idx = 0
                for chain in ['BTC', 'ETH', 'SOL', 'XRP']:
                    chain_df = df[df['chain'] == chain]
                    signal, reason, color = "🟡 STANDBY", f"Insufficient whale data for {chain}.", "#888888"
                    
                    if not chain_df.empty and 'wallet_type' in chain_df.columns:
                        c_vol = chain_df['usd_value'].sum()
                        if c_vol > 0:
                            types_vol = chain_df.groupby('wallet_type')['usd_value'].sum()
                            acc_vol = types_vol.get('accumulator', 0)
                            exc_vol = types_vol.get('exchange', 0)
                            
                            if acc_vol / c_vol > 0.5:
                                signal, color = "🟢 BULLISH", "#14F195"
                                reason = f"Supply Shock: {acc_vol/c_vol*100:.0f}% of volume moving to Accumulators."
                            elif exc_vol / c_vol > 0.6:
                                signal, color = "🔴 BEARISH", "#FF4B4B"
                                reason = f"Sell Wall: {exc_vol/c_vol*100:.0f}% of volume flowing into Exchanges."
                            else:
                                signal, color = "🟡 STANDBY", "#F7931A"
                                reason = "Mixed flows. No clear imbalance."
                                
                    with pred_cols[idx]:
                        st.markdown(f'''
                        <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; border-left: 4px solid {color}; height: 140px; overflow: hidden;">
                            <h4 style="margin: 0; padding: 0; color: #E2E8F0;">{chain}</h4>
                            <h5 style="margin: 5px 0 10px 0; color: {color};">{signal}</h5>
                            <p style="margin: 0; font-size: 0.85em; color: #94A3B8; line-height: 1.4; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;">{reason}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    idx += 1
            else:
                st.info("🌊 No whale alerts detected yet. Monitoring blockchain for large movements...")

        with tab_testnet:
            st.markdown("### 🧪 Binance Testnet Trading")
            st.markdown("Real orders on Binance Testnet — bot decisions mirrored live.")

            # All testnet calls go through the API server (client-mode compatible)
            import requests as _tn_requests

            _api = get_api_url()

            # ── Fetch all data in parallel ─────────────────────────────────
            tn_data, tn_positions_data, tn_pnl_data, tn_trades_data = {}, {}, {}, {}
            try:
                tn_resp = _tn_requests.get(f'{_api}/api/testnet/status', timeout=15)
                tn_data = tn_resp.json() if tn_resp.status_code == 200 else {}
            except Exception as _e:
                st.error(f"❌ Cannot reach API server: {_e}")

            try:
                _pos_resp = _tn_requests.get(f'{_api}/api/testnet/positions', timeout=20)
                tn_positions_data = _pos_resp.json() if _pos_resp.status_code == 200 else {}
            except Exception:
                tn_positions_data = {}

            try:
                _pnl_resp = _tn_requests.get(f'{_api}/api/testnet/pnl', timeout=20)
                tn_pnl_data = _pnl_resp.json() if _pnl_resp.status_code == 200 else {}
            except Exception:
                tn_pnl_data = {}

            try:
                _trades_resp = _tn_requests.get(f'{_api}/api/testnet/trades?limit=200', timeout=15)
                tn_trades_data = _trades_resp.json() if _trades_resp.status_code == 200 else {}
            except Exception:
                tn_trades_data = {}

            # ── Connection status ──────────────────────────────────────────
            if not tn_data.get('configured', True) or (tn_data.get('error') and not tn_data.get('connected')):
                st.error(f"⚠️ {tn_data.get('error', 'Testnet not configured on server')}")
                st.info("Set `BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_API_SECRET` in server environment.")
            else:
                _key_pfx = tn_data.get('api_key_prefix', '')
                _connected = tn_data.get('connected', False)
                status_cols = st.columns([2, 2, 2])
                with status_cols[0]:
                    if _connected:
                        st.success(f"✅ Connected to Binance Testnet")
                    else:
                        st.warning("⚠️ Testnet connection failed")
                with status_cols[1]:
                    if _key_pfx:
                        st.info(f"🔑 Key: `{_key_pfx}`")
                with status_cols[2]:
                    mirror_active = bool(tn_data.get('connected'))
                    st.info(f"🤖 Auto-Mirror: {'ON (set TESTNET_MIRROR=true)' if mirror_active else 'Enable via TESTNET_MIRROR=true'}")

                # ── PNL Summary metrics ────────────────────────────────────
                st.markdown("---")
                st.markdown("### 💰 Portfolio & PNL Summary")

                portfolio_value = float(tn_data.get('portfolio_value', 0) or 0)
                usdt_balance = float(tn_data.get('usdt_balance', 0) or 0)
                realized_pnl = float(tn_pnl_data.get('realized_pnl', 0) or 0)
                unrealized_pnl = float(tn_pnl_data.get('unrealized_pnl', 0) or 0)
                total_pnl = float(tn_pnl_data.get('total_pnl', 0) or 0)
                total_trades = int(tn_pnl_data.get('total_trades', 0) or 0)
                closed_trades = int(tn_pnl_data.get('closed_trades', 0) or 0)
                win_rate = float(tn_pnl_data.get('win_rate', 0) or 0)
                winning_trades = int(tn_pnl_data.get('winning_trades', 0) or 0)
                total_balance = float(tn_pnl_data.get('total_balance', 0) or 0)
                initial_balance = float(tn_pnl_data.get('initial_balance', 5000) or 5000)
                balance_pnl_usdt = float(tn_pnl_data.get('balance_pnl_usdt', 0) or 0)
                balance_pnl_pct = float(tn_pnl_data.get('balance_pnl_pct', 0) or 0)

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        "💰 USDT Balance",
                        f"${total_balance:,.2f}" if total_balance else "—",
                    )
                with m2:
                    st.metric(
                        "📊 Total PNL (from $5,000)",
                        f"${balance_pnl_usdt:+,.2f}" if total_balance else "—",
                        delta=f"{balance_pnl_pct:+.2f}%" if total_balance else None,
                        delta_color="normal",
                    )
                with m3:
                    st.metric(
                        "📈 Realized PNL",
                        f"${realized_pnl:+,.2f}" if realized_pnl is not None else "—",
                        delta=f"${unrealized_pnl:+,.2f} unrealized" if unrealized_pnl else None,
                    )
                with m4:
                    wr_str = f"{win_rate * 100:.1f}%" if win_rate is not None else "—"
                    st.metric(
                        "🎯 Win Rate",
                        wr_str,
                        delta=f"{winning_trades}/{closed_trades} closed" if closed_trades > 0 else None,
                    )

                # ── Bot-Mirrored Open Positions ────────────────────────────
                st.markdown("---")
                st.markdown("### 📊 Open Positions (Bot-Mirrored)")

                bot_positions = tn_positions_data.get('positions', [])
                if bot_positions:
                    pos_rows = []
                    for p in bot_positions:
                        sym = p.get('symbol', '')
                        side = p.get('side', '')
                        entry = float(p.get('entry_price', 0) or 0)
                        curr = float(p.get('current_price', 0) or 0)
                        amt = float(p.get('amount', 0) or 0)
                        upnl = float(p.get('unrealized_pnl', 0) or 0)
                        upnl_pct = float(p.get('unrealized_pnl_pct', 0) or 0)
                        sl_p = float(p.get('sl_price', 0) or p.get('sl', 0) or 0)
                        tp_p = float(p.get('tp_price', 0) or p.get('tp', 0) or 0)
                        conf = float(p.get('confidence', 0) or 0)
                        sim = bool(p.get('simulated', False))
                        side_display = f"{side} {'(sim)' if sim else ''}"
                        pos_rows.append({
                            'Symbol': sym,
                            'Side': side_display,
                            'Entry': f"${entry:,.4f}" if entry else "—",
                            'Current': f"${curr:,.4f}" if curr else "—",
                            'Amount': f"{amt:.6f}",
                            'Unreal. PNL': f"${upnl:+,.4f} ({upnl_pct:+.2f}%)" if curr else "—",
                            'SL': f"${sl_p:,.4f}" if sl_p else "—",
                            'TP': f"${tp_p:,.4f}" if tp_p else "—",
                            'Confidence': f"{conf:.2f}" if conf else "—",
                        })
                    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No open bot-mirrored positions.")

                # Spot wallet positions from status endpoint
                spot_positions = tn_data.get('positions', [])
                if spot_positions:
                    st.markdown("**Spot Wallet Holdings:**")
                    spot_rows = [{
                        'Asset': p.get('asset', ''),
                        'Amount': f"{float(p.get('amount', 0) or 0):.6f}",
                        'Price': f"${float(p.get('price', 0) or 0):,.2f}",
                        'Value (USDT)': f"${float(p.get('value_usdt', 0) or 0):,.2f}",
                    } for p in spot_positions]
                    st.dataframe(pd.DataFrame(spot_rows), use_container_width=True, hide_index=True)

                # ── Equity Curve ───────────────────────────────────────────
                equity_curve = tn_pnl_data.get('equity_curve', [])
                if equity_curve:
                    st.markdown("---")
                    st.markdown("### 📈 Equity Curve (Cumulative PNL)")
                    try:
                        eq_df = pd.DataFrame(equity_curve)
                        # Timestamps may be ISO-8601 strings or raw ms epoch ints.
                        # Try ISO first; if mostly NaT, fall back to unit='ms'.
                        raw_ts = eq_df['timestamp'].copy()
                        eq_df['timestamp'] = pd.to_datetime(raw_ts, errors='coerce')
                        if eq_df['timestamp'].isna().sum() > len(eq_df) * 0.5:
                            eq_df['timestamp'] = pd.to_datetime(
                                pd.to_numeric(raw_ts, errors='coerce'),
                                unit='ms', errors='coerce',
                            )
                        eq_df = eq_df.dropna(subset=['timestamp'])
                        if not eq_df.empty:
                            import plotly.graph_objects as go
                            fig_eq = go.Figure()
                            fig_eq.add_trace(go.Scatter(
                                x=eq_df['timestamp'],
                                y=eq_df['cumulative_pnl'],
                                mode='lines+markers',
                                name='Cumulative PNL',
                                line=dict(color='#00e676', width=2),
                                marker=dict(size=6),
                                hovertemplate=(
                                    '<b>%{x}</b><br>'
                                    'Cumulative PNL: $%{y:,.4f}<br>'
                                    '<extra></extra>'
                                ),
                            ))
                            fig_eq.add_hline(y=0, line_dash='dash', line_color='#666')
                            fig_eq.update_layout(
                                height=280,
                                margin=dict(l=0, r=0, t=20, b=0),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#E2E8F0'),
                                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickprefix='$'),
                            )
                            st.plotly_chart(fig_eq, use_container_width=True)
                    except Exception as _eq_e:
                        st.warning(f"Equity curve render failed: {_eq_e}")

                # ── Trade History ──────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📋 Trade History (Testnet Executions)")

                all_trades = tn_trades_data.get('trades', [])
                if all_trades:
                    trade_rows = []
                    for t in reversed(all_trades):  # newest first
                        # Timestamp: normalised 'timestamp' (ISO-8601) or
                        # raw 'time' (ms epoch) from Binance
                        ts_raw = t.get('timestamp', '')
                        if ts_raw:
                            ts = str(ts_raw)[:19].replace('T', ' ')
                        else:
                            raw_time = t.get('time', 0)
                            if raw_time:
                                try:
                                    from datetime import datetime as _dt, timezone as _tz
                                    ts = _dt.fromtimestamp(int(raw_time) / 1000, tz=_tz.utc).strftime('%Y-%m-%d %H:%M:%S')
                                except Exception:
                                    ts = '—'
                            else:
                                ts = '—'

                        sym = t.get('symbol', '—')

                        # Action: normalised 'action' or derive from side + realizedPnl
                        action = t.get('action', '')
                        if not action or action == '—':
                            side = t.get('side', '').upper()
                            rpnl = float(t.get('realizedPnl', 0) or 0)
                            if side == 'BUY':
                                action = 'CLOSE_SHORT' if rpnl != 0 else 'OPEN_LONG'
                            elif side == 'SELL':
                                action = 'CLOSE_LONG' if rpnl != 0 else 'OPEN_SHORT'
                            else:
                                action = side or '—'

                        # Price, Amount, PNL — normalised fields or raw Binance fields
                        price_v = float(t.get('filled_price') or t.get('price', 0) or 0)
                        amt = float(t.get('amount') or t.get('qty', 0) or 0)
                        pnl_v = t.get('pnl')
                        if pnl_v is None:
                            rpnl_raw = t.get('realizedPnl')
                            if rpnl_raw is not None and float(rpnl_raw) != 0:
                                pnl_v = float(rpnl_raw)
                        pnl_str = f"${float(pnl_v):+,.4f}" if pnl_v is not None else "—"

                        oid = str(t.get('order_id') or t.get('orderId', '') or '—')[:16]
                        trade_rows.append({
                            'Time': ts,
                            'Symbol': sym,
                            'Action': action,
                            'Price': f"${price_v:,.4f}" if price_v else "—",
                            'Amount': f"{amt:.6f}" if amt else "—",
                            'PNL': pnl_str,
                            'Order ID': oid,
                        })
                    st.dataframe(
                        pd.DataFrame(trade_rows),
                        use_container_width=True,
                        hide_index=True,
                        height=320,
                    )
                else:
                    st.info("No testnet trades recorded yet. Enable `TESTNET_MIRROR=true` to auto-mirror bot decisions.")

                # ── Live Order Book (open orders) ──────────────────────────
                st.markdown("---")
                st.markdown("### 📖 Live Open Orders")

                ord_col1, ord_col2 = st.columns([3, 1])
                with ord_col2:
                    if st.button("🔄 Refresh Orders", key="testnet_refresh_orders", use_container_width=True):
                        st.rerun()

                try:
                    orders_resp = _tn_requests.get(f'{_api}/api/testnet/orders', timeout=15)
                    open_orders = orders_resp.json().get('orders', []) if orders_resp.status_code == 200 else []
                    if open_orders:
                        ord_rows = []
                        for o in open_orders:
                            ord_rows.append({
                                'Order ID': str(o.get('orderId', o.get('id', '—')))[:16],
                                'Symbol': o.get('symbol', '—'),
                                'Side': o.get('side', '—'),
                                'Type': o.get('type', '—'),
                                'Price': f"${float(o.get('price', 0) or 0):,.4f}",
                                'Qty': f"{float(o.get('origQty', o.get('amount', 0)) or 0):.6f}",
                                'Status': o.get('status', '—'),
                            })
                        st.dataframe(pd.DataFrame(ord_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No open orders on testnet.")
                except Exception as _oe:
                    st.warning(f"Could not fetch open orders: {_oe}")

                # ── Manual Trading Controls ────────────────────────────────
                st.markdown("---")
                st.markdown("### 🎮 Manual Trading Controls")

                col1, col2 = st.columns(2)
                with col1:
                    trade_symbol = st.selectbox(
                        "Select Pair", ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT'],
                        key="testnet_symbol"
                    )
                with col2:
                    trade_amount = st.number_input(
                        "Amount (USDT)", min_value=10.0,
                        max_value=float(usdt_balance) if usdt_balance > 10 else 10000.0,
                        value=100.0, step=10.0, key="testnet_amount"
                    )

                btn_cols = st.columns(3)

                if btn_cols[0].button("🟢 BUY (Market)", key="testnet_buy", use_container_width=True):
                    try:
                        order_resp = _tn_requests.post(
                            f'{_api}/api/testnet/order',
                            json={'symbol': trade_symbol, 'side': 'buy', 'amount_usdt': trade_amount},
                            timeout=20
                        )
                        result = order_resp.json()
                        if result.get('success'):
                            _amt = float(result.get('amount', 0) or 0)
                            _pr = float(result.get('price', 0) or 0)
                            st.success(f"✅ BUY: {_amt:.6f} {trade_symbol.split('/')[0]} @ ${_pr:,.2f}")
                            st.rerun()
                        else:
                            st.error(f"❌ Order failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Order failed: {e}")

                if btn_cols[1].button("🔴 SELL (Market)", key="testnet_sell", use_container_width=True):
                    try:
                        order_resp = _tn_requests.post(
                            f'{_api}/api/testnet/order',
                            json={'symbol': trade_symbol, 'side': 'sell', 'amount_usdt': 0},
                            timeout=20
                        )
                        result = order_resp.json()
                        if result.get('success'):
                            _amt = float(result.get('amount', 0) or 0)
                            _sym = trade_symbol.split('/')[0]
                            st.success(f"✅ SELL: {_amt:.6f} {_sym}")
                            st.rerun()
                        else:
                            st.error(f"❌ Order failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Order failed: {e}")

                if btn_cols[2].button("🧪 Mirror Bot Trade", key="testnet_mirror_btn", use_container_width=True):
                    try:
                        _sym_raw = trade_symbol.replace('/', '')
                        exec_resp = _tn_requests.post(
                            f'{_api}/api/testnet/execute',
                            json={'action': 'OPEN_LONG_SPLIT', 'symbol': f"{_sym_raw}USDT" if 'USDT' not in _sym_raw else _sym_raw, 'confidence': 0.65},
                            timeout=25
                        )
                        result = exec_resp.json()
                        if result.get('success'):
                            t = result.get('trade', {}) or {}
                            _pr = float(t.get('price', 0) or 0)
                            st.success(f"✅ Testnet mirror executed: OPEN_LONG @ ${_pr:,.2f}")
                            st.rerun()
                        else:
                            st.error(f"❌ Mirror failed: {result.get('error', 'Unknown')}")
                    except Exception as e:
                        st.error(f"❌ Mirror failed: {e}")

                st.markdown("---")
                _ic1, _ic2 = st.columns(2)
                with _ic1:
                    st.info("""
                    **Bot Auto-Mirror (TESTNET_MIRROR=true):**
                    - Set env var to enable real-time mirroring
                    - Every bot decision → real testnet order
                    - LONG = real BUY order (50% market + 50% limit)
                    - SHORT = conceptual (spot testnet only)
                    - Trades logged to `logs/testnet_trades.json`
                    """)
                with _ic2:
                    st.warning("""
                    **Testnet Notes:**
                    - Zero real money risk (testnet.binance.vision)
                    - Testnet funds reset periodically
                    - SHORT positions tracked conceptually (spot exchange)
                    - SL/TP managed by bot logic (no exchange OCO orders)
                    """)

        with tab_htf:
            st.markdown("### 🔮 HTF Agent — Hierarchical Multi-Timeframe Trader")
            st.caption("4-timeframe cascade: 1D → 4H → 1H → 15M | PPO | Walk-forward validated (Avg Sharpe 3.85, +14.8%/2mo)")

            api_base = get_api_url()

            # ── Status ──
            try:
                htf_status_resp = __import__('requests').get(f"{api_base}/api/htf/status", timeout=8)
                htf_status = htf_status_resp.json() if htf_status_resp.ok else {}
            except Exception:
                htf_status = {}

            if not htf_status.get('running'):
                st.warning(
                    "**HTF bot is not running.** Start it with:\n"
                    "```bash\npython live_trading_htf.py --interval 15\n```\n"
                    "Add `--live` to enable real execution. Default is dry-run (paper trading)."
                )
            else:
                dry_tag = " *(dry-run)*" if htf_status.get('dry_run') else " *(LIVE)*"
                col1, col2, col3, col4 = st.columns(4)
                pos_label = htf_status.get('position_label', 'FLAT')
                pos_color = {"LONG": "#00e676", "SHORT": "#ff5252", "FLAT": "#8b949e"}.get(pos_label, "#8b949e")

                col1.metric("Position", pos_label)
                col2.metric("Balance", f"${htf_status.get('balance', 0):,.2f}" if htf_status.get('balance') else "—")
                col3.metric("Realized PnL", f"${htf_status.get('realized_pnl', 0):+,.2f}")
                col4.metric("Unrealized PnL", f"${htf_status.get('unrealized_pnl', 0):+,.2f}")

                # Position details
                if htf_status.get('position', 0) != 0:
                    st.markdown(f"""
                    <div style="background:#151b23;border:1px solid {pos_color};border-radius:8px;padding:14px 18px;margin:8px 0;">
                    <b style="color:{pos_color};">{pos_label}</b> &nbsp;|&nbsp;
                    Entry: <b>${htf_status.get('position_price', 0):,.2f}</b> &nbsp;|&nbsp;
                    SL: <b style="color:#ff5252;">${htf_status.get('sl_price', 0):,.2f}</b> &nbsp;|&nbsp;
                    TP: <b style="color:#00e676;">${htf_status.get('tp_price', 0):,.2f}</b> &nbsp;|&nbsp;
                    Units: <b>{htf_status.get('position_units', 0):.5f}</b>
                    </div>
                    """, unsafe_allow_html=True)

                agent_cols = st.columns(3)
                agent_cols[0].info(f"**Win Rate:** {htf_status.get('win_rate', 0)*100:.1f}%")
                agent_cols[1].info(f"**Trades:** {htf_status.get('trade_count', 0)}")
                agent_cols[2].info(f"**Mode:** HTF PPO{dry_tag}")

                model_path = htf_status.get('model_path') or 'Not loaded'
                st.caption(f"Model: `{Path(model_path).name if model_path else '—'}` | "
                           f"Started: {htf_status.get('start_time', '—')[:19] if htf_status.get('start_time') else '—'}")

            st.markdown("---")

            # ── Performance Metrics ──
            st.markdown("#### 📈 Performance Metrics")
            try:
                perf_resp = __import__('requests').get(f"{api_base}/api/htf/performance", timeout=8)
                perf = perf_resp.json() if perf_resp.ok else {}
            except Exception:
                perf = {}

            if perf and not perf.get('error') and perf.get('total_trades', 0) > 0:
                pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                pm1.metric("Total Trades", perf.get('total_trades', 0))
                pm2.metric("Win Rate", f"{perf.get('win_rate', 0)*100:.1f}%")
                pm3.metric("Total PnL", f"${perf.get('total_pnl', 0):+,.2f}")
                pm4.metric("Sharpe Ratio", f"{perf.get('sharpe', 0):.2f}")
                pm5.metric("Max Drawdown", f"{perf.get('max_drawdown', 0):.1f}%")

                pm6, pm7, pm8 = st.columns(3)
                pm6.metric("Return", f"{perf.get('return_pct', 0):+.1f}%")
                pm7.metric("Best Trade", f"${perf.get('best_trade', 0):+,.2f}")
                pm8.metric("Worst Trade", f"${perf.get('worst_trade', 0):+,.2f}")
            else:
                st.info(perf.get('message', 'No closed trades yet — metrics will appear after first completed trade.'))

            st.markdown("---")

            # ── Trade History ──
            st.markdown("#### 📋 Trade History")
            try:
                trades_resp = __import__('requests').get(f"{api_base}/api/htf/trades?limit=100", timeout=8)
                htf_trades = trades_resp.json().get('trades', []) if trades_resp.ok else []
            except Exception:
                htf_trades = []

            if htf_trades:
                close_trades = [t for t in reversed(htf_trades) if 'CLOSE' in t.get('action', '').upper()]
                open_trades = [t for t in reversed(htf_trades) if 'OPEN' in t.get('action', '').upper()]

                if close_trades:
                    rows = []
                    for t in close_trades[:50]:
                        pnl = t.get('pnl', 0)
                        rows.append({
                            'Time': t.get('timestamp', '')[:19],
                            'Action': t.get('action', ''),
                            'Entry': f"${t.get('entry_price', 0):,.2f}",
                            'Exit': f"${t.get('exit_price', 0):,.2f}",
                            'PnL': f"${pnl:+,.2f}",
                            'Reason': t.get('reason', ''),
                        })
                    df_trades = pd.DataFrame(rows)

                    def _color_pnl(val):
                        if isinstance(val, str) and val.startswith('$'):
                            try:
                                v = float(val.replace('$', '').replace(',', '').replace('+', ''))
                                return 'color: #00e676' if v > 0 else 'color: #ff5252'
                            except Exception:
                                pass
                        return ''

                    st.dataframe(
                        df_trades.style.applymap(_color_pnl, subset=['PnL']),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No closed trades yet.")

                if open_trades:
                    st.markdown("**Open Trades**")
                    for t in open_trades[:5]:
                        st.markdown(
                            f"- `{t.get('action','')}` @ **${t.get('price', 0):,.2f}** "
                            f"| conf: {t.get('confidence', 0):.2f} "
                            f"| {t.get('timestamp', '')[:19]}"
                        )
            else:
                st.info("No HTF trades recorded yet. Bot will begin trading on next cycle.")

            st.markdown("---")

            # ── Architecture Info ──
            with st.expander("🏗 HTF Agent Architecture"):
                st.markdown("""
                **Observation Space:** 117 dimensions across 4 timeframes
                | Block | Dims | Features |
                |-------|------|---------|
                | 1D    | 20   | Macro trend, regime, HTF structure |
                | 4H    | 25   | Swing structure, Smart Money Concepts (BOS, CHoCH, OB, FVG) |
                | 1H    | 30   | Momentum, RSI divergence, MACD, Stochastic |
                | 15M   | 35   | Micro entry triggers, candle patterns, Wyckoff phase |
                | Align | 4    | Cross-TF cascade hierarchy signals |
                | Pos   | 3    | Position, unrealized PnL, balance ratio |

                **Agent:** PPO with `[512, 256, 128]` network, VecNormalize, curriculum training
                **Training:** Walk-forward validation (8 folds, 50% position size)
                **Validated:** Avg Sharpe 3.85 · +14.8% / 2 months · Max Drawdown 5.95%
                **Risk:** SL 1.5% · TP 3.0% · Fee 0.04% · Min hold 1h · Cooldown 30min after loss
                """)

        with tab_backtest:
            st.markdown("### 🔬 Backtest")
            if IS_CLIENT_MODE:
                st.info("🌐 **Backtest is not available in client mode.** Run the trading server locally and access backtesting from the server dashboard.")
            else:
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
    <div style="text-align: center; color: {TEXT_MUTED}; font-size: 12px;">
        DRL Trading System v2.1 | Advanced PPO Agent |
        <span style="color: {SUCCESS};">●</span> WebSocket Live Data |
        Deployed: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
