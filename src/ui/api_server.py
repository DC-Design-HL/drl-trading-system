"""
API endpoints for live streaming data to the dashboard.
This provides JSON endpoints that JavaScript can poll for updates.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
@app.route('/health')
def health_check():
    """Health check endpoint for system status."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.storage import get_storage

# Initialize storage
storage = get_storage()
WHALE_TRACKERS = {}

# Cache for /api/market
_MARKET_CACHE: dict = {}
MARKET_CACHE_TTL = 30  # 30 seconds cache (restored from 1 hour since we removed proxy)


@app.route('/api/state')
def get_state():
    """Get current trading state."""
    try:
        state = storage.load_state()
        
        # MATHEMATICAL OVERRIDE: Calculate true global equity from trade history
        # bypassing the historically corrupted bot balance tracker.
        try:
            all_trades = storage.get_trades(limit=1000)
            realized_pnl = sum(t.get('pnl', 0) for t in all_trades if 'CLOSE' in t.get('action', '').upper() or 'EXIT' in t.get('action', '').upper())
            
            raw_assets = state.get('raw_state', {}).get('assets', {})
            open_pnl = sum(a.get('pnl', 0) for a in raw_assets.values() if a.get('position', 0) != 0)
            
            initial_capital = max(len(raw_assets), 1) * 5000 if raw_assets else 20000
            
            if all_trades or raw_assets:
                state['total_pnl'] = realized_pnl + open_pnl
                state['realized_pnl'] = realized_pnl + open_pnl
                state['total_balance'] = initial_capital + state['total_pnl']
                state['balance'] = state['total_balance']
        except Exception as math_err:
            logger.error(f"Failed mathematical override: {math_err}")
            
        # Normalize keys for frontend compatibility
        if 'total_balance' in state and 'balance' not in state:
            state['balance'] = state['total_balance']
        if 'total_pnl' in state and 'realized_pnl' not in state:
            state['realized_pnl'] = state['total_pnl']
            
        return jsonify(state)
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return jsonify({})

@app.route('/api/trades')
def get_trades():
    """Get recent trades."""
    try:
        trades = storage.get_trades(limit=1000)
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Failed to load trades: {e}")
        return jsonify([])

@app.route('/api/trades/count')
def get_trade_count():
    """Get trade count for change detection."""
    try:
        # Simple count estimation via list length
        trades = storage.get_trades(limit=1000)
        count = len(trades)
        return jsonify({'count': count, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Failed to count trades: {e}")
        return jsonify({'count': 0, 'timestamp': datetime.now().isoformat()})

@app.route('/api/model')
def get_model_info():
    """Get active model information."""
    import os
    
    model_path = PROJECT_ROOT / 'data' / 'models' / 'ultimate_agent.zip'
    
    # Model info
    model_exists = model_path.exists()
    if model_exists:
        model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        model_date = model_mtime.strftime("%Y-%m-%d")
    else:
        model_date = "Not found"
    
    # State info via storage
    state = {}
    try:
        state = storage.load_state()
    except Exception as e:
        logger.error(f"Failed to load state for model info: {e}")
    
    # Handle multi-asset state structure
    balance = state.get('total_balance', state.get('balance', 10000))
    total_return = ((balance - 10000) / 10000) * 100
    
    # Trade stats via storage
    trades = []
    try:
        trades = storage.get_trades(limit=1000)
    except Exception as e:
        logger.error(f"Failed to load trades for model info: {e}")
    
    winning = sum(1 for t in trades if t.get('pnl', 0) > 0)
    total = len(trades)
    win_rate = (winning / total * 100) if total > 0 else 0
    
    return jsonify({
        'model_name': 'Ultimate Agent (PPO)',
        'model_exists': model_exists,
        'model_date': model_date,
        'total_return': round(total_return, 2),
        'win_rate': round(win_rate, 1),
        'total_trades': total,
        'winning_trades': winning,
        'balance': balance,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/market')
def get_market_analysis():
    """Get comprehensive market analysis from all analyzers."""
    import sys
    import os
    import time
    from flask import request
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Get symbol from query params, default to BTC/USDT
    symbol = request.args.get('symbol', 'BTCUSDT')
    clean_symbol = symbol.replace('/', '').upper()

    # ── 60-second cache to limit proxy bandwidth ──────────────────────────
    global _MARKET_CACHE
    cached = _MARKET_CACHE.get(clean_symbol)
    if cached and (time.time() - cached['_fetched_at']) < MARKET_CACHE_TTL:
        return jsonify(cached)
    # ─────────────────────────────────────────────────────────────────────

    result = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'whale': None,
        'regime': None,
        'mtf': None,
        'funding': None,
        'order_flow': None,
        'forecast': None
    }
    
    # Try to load from state first (Consistency with Bot)
    state_file = PROJECT_ROOT / 'logs' / 'trading_state.json'
    state_analysis = None
    
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                assets = state.get('assets', {})
                # Try explicit symbol or clean symbol
                asset_data = assets.get(symbol, assets.get(clean_symbol))
                if asset_data:
                    state_analysis = asset_data.get('analysis')
        except Exception as e:
            logger.error(f"Failed to load state for market analysis: {e}")

    # Use state analysis if available
    if state_analysis:
        # Whale
        whale_data = state_analysis.get('whale', {})
        if whale_data:
            result['whale'] = {
                'score': round(whale_data.get('score', 0), 2),
                'direction': whale_data.get('direction', 'NEUTRAL'),
                'confidence': int(whale_data.get('confidence', 0) * 100),
                'bullish': whale_data.get('bullish_signals', 0),
                'bearish': whale_data.get('bearish_signals', 0),
                'neutral': whale_data.get('neutral_signals', 0),
                'flow_metrics': whale_data.get('flow_metrics', {})
            }
            
        # Funding
        funding_data = state_analysis.get('funding', {})
        if funding_data:
            result['funding'] = {
                'rate': round(funding_data.get('rate', 0) * 100, 4), # rate is usually float
                'bias': funding_data.get('signal', 'neutral'),
                'annualized': round(funding_data.get('rate', 0) * 3 * 365 * 100, 1)
            }
            
        # Order Flow
        of_data = state_analysis.get('order_flow', {})
        if of_data:
            result['order_flow'] = {
                'large_buys': of_data.get('large_buys', 0),
                'large_sells': of_data.get('large_sells', 0),
                'bias': of_data.get('bias', 'neutral'),
                'net_flow': of_data.get('large_buy_volume', 0) - of_data.get('large_sell_volume', 0)
            }
            
        # Forecast
        forecast_data = state_analysis.get('forecast')
        if forecast_data:
            result['forecast'] = {
                'return_1h': round(forecast_data.get('return_1h', 0) * 100, 3),
                'return_4h': round(forecast_data.get('return_4h', 0) * 100, 3),
                'return_12h': round(forecast_data.get('return_12h', 0) * 100, 3),
                'return_24h': round(forecast_data.get('return_24h', 0) * 100, 3),
                'confidence': round(forecast_data.get('confidence_4h', 0), 2),
                'consensus': round(forecast_data.get('direction_consensus', 0), 2),
            }
            
        # Confidence Engine
        confidence = state_analysis.get('confidence')
        if confidence is not None:
             # Adding at root since it applies to the whole ensemble
             result['ensemble_confidence'] = round(confidence, 2)
            
        # Regime (New Phase 11.3 addition, HMM model)
        regime_data = state_analysis.get('regime')
        if regime_data:
             result['regime'] = regime_data

    # Fallbacks and Regime (Regime is not in state yet, calculate it)
    # ... (Keep existing regime calculation as it's fast) ...
    
    if not result['whale']:
        try:
            # Whale Tracker - Use persistent instance for WebSocket
            from src.features.whale_tracker import WhaleTracker
            
            # Global store for active trackers (defined at module level)
            global WHALE_TRACKERS
            
            if clean_symbol not in WHALE_TRACKERS:
                logger.info(f"Initializing new WhaleTracker for {clean_symbol}")
                tracker = WhaleTracker(symbol=clean_symbol, enable_ml=False)
                # DO NOT start live websocket stream inside the API server memory space
                # as it leaks sockets and hangs threads. We only use it for REST fetch here.
                # tracker.start_stream()
                WHALE_TRACKERS[clean_symbol] = tracker
            
            whale = WHALE_TRACKERS[clean_symbol]
            signals = whale.get_whale_signals() 
            
            result['whale'] = {
                'score': round(signals.get('score', 0), 2),
                'direction': signals.get('direction', 'NEUTRAL'),
                'confidence': int(signals.get('confidence', 0) * 100),
                'bullish': signals.get('bullish_signals', 0),
                'bearish': signals.get('bearish_signals', 0),
                'neutral': signals.get('neutral_signals', 0),
                'flow_metrics': signals.get('flow_metrics', {})
            }
        except Exception as e:
            result['whale'] = {'error': str(e)}
    
    try:
        # Fetch current market data for regime detection (fast REST call)
        import requests as req
        import os as _os
        url = _os.environ.get("BINANCE_FUTURES_URL", "https://data-api.binance.vision") + "/api/v3/klines"
        params = {"symbol": clean_symbol, "interval": "1h", "limit": 100}
        resp = req.get(url, params=params, timeout=10)
        data = resp.json()
        
        import pandas as pd
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Add current price to result
        if not df.empty:
            result['price'] = float(df.iloc[-1]['close'])
        
        # Regime Detector
        from src.features.regime_detector import MarketRegimeDetector
        regime = MarketRegimeDetector()
        regime_result = regime.detect_regime(df)
        
        regime_type = getattr(regime_result, 'regime', 'UNKNOWN')
        result['regime'] = {
            'type': str(regime_type.name) if hasattr(regime_type, 'name') else str(regime_type),
            'adx': round(getattr(regime_result, 'adx', 0), 1),
            'direction': str(getattr(regime_result, 'direction', 'NEUTRAL')),
            'volatility': round(getattr(regime_result, 'volatility_multiplier', 1), 2)
        }
    except Exception as e:
        logger.error(f"Market data fetch error: {e}")
        result['regime'] = {'error': str(e)}
    
    if not result['mtf'] and state_analysis:
        mtf_data = state_analysis.get('mtf', {})
        if mtf_data:
             result['mtf'] = {
                'bias': mtf_data.get('bias', 'NEUTRAL'),
                'aligned': mtf_data.get('aligned', False),
                'reason': mtf_data.get('reason', 'Syncing...'),
                '4h': mtf_data.get('4h', 'neutral'),
                '1h': mtf_data.get('1h', 'neutral'),
                '15m': mtf_data.get('15m', 'neutral')
            }

    # API Server should NOT perform heavy analysis on the fly.
    # It must rely on the live_trading_multi.py to populate the state.
    # If the state is missing data, we return empty/null to indicate "System Syncing".
    
    if not result['mtf']:
        result['mtf'] = {'reason': 'Syncing...', 'aligned': False, 'bias': 'NEUTRAL'}
        
    if not result['funding']:
        try:
            from src.features.order_flow import FundingRateAnalyzer
            fa = FundingRateAnalyzer(symbol=clean_symbol)
            sig = fa.get_signal()
            result['funding'] = {
                'rate': round(sig.rate * 100, 4),
                'bias': sig.signal,
                'annualized': round(sig.rate * 3 * 365 * 100, 1)
            }
        except Exception as e:
            logger.error(f"Funding fallback error: {e}")
            result['funding'] = {'rate': 0, 'bias': 'neutral', 'annualized': 0}
        
    if not result['order_flow']:
        try:
            from src.features.order_flow import OrderFlowAnalyzer
            oa = OrderFlowAnalyzer(symbol=clean_symbol)
            of = oa.analyze_large_orders()
            result['order_flow'] = {
                'large_buys': of.get('large_buys', 0),
                'large_sells': of.get('large_sells', 0),
                'bias': of.get('bias', 'neutral'),
                'net_flow': of.get('large_buy_volume', 0) - of.get('large_sell_volume', 0)
            }
        except Exception as e:
            logger.error(f"OrderFlow fallback error: {e}")
            result['order_flow'] = {'bias': 'neutral', 'net_flow': 0, 'large_buys': 0, 'large_sells': 0}

    # API Server should NOT perform heavy ML analysis on the fly.
    # TFT Forecast (Phase 11.1) requires massive PyTorch model loading that hangs the main thread.
    if not result['forecast']:
        # Return empty forecast instead of loading PyTorch on CPU to prevent API deadlock
        pass
        
    if not result['mtf']:
        result['mtf'] = {'reason': 'Syncing...', 'aligned': False, 'bias': 'NEUTRAL'}

    # Store result in 60s cache before returning
    result['_fetched_at'] = time.time()
    _MARKET_CACHE[clean_symbol] = result
    response = {k: v for k, v in result.items() if k != '_fetched_at'}
    return jsonify(response)


@app.route('/api/debug/log')
def get_crash_log():
    """Get crash log if exists."""
    log_file = PROJECT_ROOT / 'crash.log'
    if log_file.exists():
        with open(log_file, 'r') as f:
            return f.read(), 200, {'Content-Type': 'text/plain'}
    return "No crash log found. Bot might be running or log not written.", 404

if __name__ == '__main__':
    # threaded=True is CRITICAL to prevent single requests (like Market Analysis fallback)
    # from locking up the entire dashboard and causing 'Trades: 0' sidebars
    app.run(port=5001, debug=False, threaded=True)
