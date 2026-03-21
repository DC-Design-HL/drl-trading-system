"""
API endpoints for live streaming data to the dashboard.
This provides JSON endpoints that JavaScript can poll for updates.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from pathlib import Path
from datetime import datetime
import time

# Load .env for local development (no-op if vars already set)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / '.env', override=False)
except ImportError:
    pass

app = Flask(__name__)
CORS(app, origins="*")

@app.route('/api/ping')
def ping():
    """Lightweight connectivity check for remote clients."""
    return jsonify({"ok": True, "timestamp": datetime.now().isoformat()})

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

            if all_trades or raw_assets:
                state['total_pnl'] = realized_pnl + open_pnl
                state['realized_pnl'] = realized_pnl + open_pnl
                # Do not override balance — use real value from stored state
        except Exception as math_err:
            logger.error(f"Failed mathematical override: {math_err}")
            
        # Normalize keys for frontend compatibility
        if 'total_balance' in state and 'balance' not in state:
            state['balance'] = state['total_balance']
        if 'total_pnl' in state and 'realized_pnl' not in state:
            state['realized_pnl'] = state['total_pnl']

        # Expose available assets for the frontend selector
        # Always include all configured trading assets so user can switch
        configured_assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
        raw_assets_keys = list(state.get('raw_state', {}).get('assets', {}).keys())
        if not raw_assets_keys:
            raw_assets_keys = list(state.get('assets', {}).keys())
        # Merge: configured assets + any additional ones from state
        all_assets = list(dict.fromkeys(configured_assets + raw_assets_keys))
        state['available_assets'] = all_assets

        # Inject Recent On-Chain Whale Alerts from tracked wallets
        try:
            whale_alerts = []
            whale_dir = PROJECT_ROOT / "data" / "whale_wallets"
            if whale_dir.exists():
                from src.features.whale_wallet_registry import get_wallets_by_chain
                for chain_dir in whale_dir.iterdir():
                    if chain_dir.is_dir():
                        chain = chain_dir.name.upper()
                        for wallet_file in chain_dir.glob("*.json"):
                            try:
                                with open(wallet_file, "r") as f:
                                    w_data = json.load(f)
                                    addr = w_data.get("address", "")
                                    
                                    # Cross-reference live address with registry identities
                                    chain_wallets = get_wallets_by_chain(chain)
                                    wallet = next((w for w in chain_wallets if w.address.lower() == addr.lower()), None)
                                    w_label = wallet.label if wallet else f"Unknown {chain} Whale"
                                    w_type = wallet.wallet_type if wallet else "unknown"
                                    
                                    for tx in w_data.get("transactions", [])[-10:]:
                                        val = float(tx.get('value', 0))
                                        
                                        # Strict $50k USD minimum threshold
                                        price_map = {'BTC': 70000, 'ETH': 3500, 'SOL': 150, 'XRP': 0.6}
                                        usd_val = val * price_map.get(chain, 1)
                                        
                                        if usd_val > 50000: 
                                            alert = {
                                                'chain': chain,
                                                'value': val,
                                                'currency': tx.get('asset', chain),
                                                'timestamp': tx.get('timestamp', int(time.time())),
                                                'link': tx.get('link', '#'),
                                                'wallet_label': w_label,
                                                'wallet_type': w_type,
                                                'wallet_address': addr
                                            }
                                            whale_alerts.append(alert)
                            except:
                                pass
            
            # Sort globally by timestamp descending and take top 50 alerts
            if whale_alerts:
                whale_alerts = sorted(whale_alerts, key=lambda x: x.get('timestamp', 0), reverse=True)[:50]
                state['whale_alerts'] = whale_alerts
        except Exception as e:
            logger.error(f"Failed to load whale alerts: {e}")
            
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
    balance = state.get('total_balance', state.get('balance'))
    total_return = None  # Cannot compute without knowing real initial capital
    
    # Trade stats via storage
    trades = []
    try:
        trades = storage.get_trades(limit=1000)
    except Exception as e:
        logger.error(f"Failed to load trades for model info: {e}")
    
    # Only count CLOSED trades for win rate (OPEN trades have pnl=0, not relevant)
    closed_trades = [t for t in trades if 'CLOSE' in t.get('action', '').upper() or 'EXIT' in t.get('action', '').upper()]
    winning = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
    total_closed = len(closed_trades)
    win_rate = (winning / total_closed * 100) if total_closed > 0 else 0
    
    # Calculate total return from realized PnL
    initial_capital = 20000.0
    realized_pnl = sum(t.get('pnl', 0) for t in closed_trades)
    total_return = (realized_pnl / initial_capital * 100) if initial_capital > 0 else None
    
    return jsonify({
        'model_name': 'Ultimate Agent (PPO)',
        'model_exists': model_exists,
        'model_date': model_date,
        'total_return': round(total_return, 2) if total_return is not None else None,
        'win_rate': round(win_rate, 1),
        'total_trades': total_closed,
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
        response = {k: v for k, v in cached.items() if k != '_fetched_at'}
        return jsonify(response)
    # ─────────────────────────────────────────────────────────────────────

    result = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'whale': None,
        'regime': None,
        'mtf': None,
        'funding': None,
        'order_flow': None,
        'forecast': None,
        'news': None,
        'stablecoin': None,
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
                'net_flow': of_data.get('large_buy_volume', 0) - of_data.get('large_sell_volume', 0),
                'score': of_data.get('score', 0),
                # Layer details (enhanced signal)
                'cvd': of_data.get('cvd', {}),
                'taker': of_data.get('taker', {}),
                'notable': of_data.get('notable', {})
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

        # News Sentiment - DISABLED (not reliable, removed per user request)
        # Commented out - news sentiment disabled in trading logic
        # news_data = state_analysis.get('news', {})
        # if news_data:
        #     result['news'] = {
        #         'sentiment': round(news_data.get('sentiment', 0), 3),
        #         'confidence': round(news_data.get('confidence', 0), 3),
        #         'trend': news_data.get('trend', 'unknown'),
        #         'sources': news_data.get('sources', 0)
        #     }

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
                'score': round(signals.get('combined_score', signals.get('score', 0)), 2),
                'direction': signals.get('recommendation', signals.get('direction', 'NEUTRAL')).upper(),
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
        
        # Regime Detector (ADX-based)
        from src.features.regime_detector import MarketRegimeDetector, get_hmm_prediction
        regime = MarketRegimeDetector()
        regime_result = regime.detect_regime(df)
        
        regime_type = getattr(regime_result, 'regime', 'UNKNOWN')
        result['regime'] = {
            'type': str(regime_type.name) if hasattr(regime_type, 'name') else str(regime_type),
            'adx': round(getattr(regime_result, 'trend_strength', 0), 1),
            'direction': str(getattr(regime_result, 'trend_direction', 'NEUTRAL')),
            'volatility': round(getattr(regime_result, 'volatility_ratio', 1), 2)
        }
        
        # HMM Regime Prediction (P1.1 — forward-looking transition probabilities)
        try:
            hmm_data = get_hmm_prediction(df, symbol=clean_symbol)
            if hmm_data:
                result['regime']['hmm_regime'] = hmm_data.get('hmm_regime', 'UNKNOWN')
                result['regime']['transition_probs'] = hmm_data.get('transition_probs', {})
                result['regime']['regime_confidence'] = round(hmm_data.get('regime_confidence', 0), 4)
        except Exception as hmm_err:
            logger.warning(f"HMM regime prediction failed: {hmm_err}")
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
    
    # MTF Fallback (P1.4): When MTF is still "Syncing...", compute it live
    if not result['mtf'] or result['mtf'].get('reason') == 'Syncing...':
        try:
            from src.features.mtf_analyzer import MultiTimeframeAnalyzer
            mtf_analyzer = MultiTimeframeAnalyzer(symbol=clean_symbol)
            # Pass the 1H df we already fetched above (if available)
            mtf_result = mtf_analyzer.get_confluence(primary_df=df if 'df' in dir() else None)
            
            signal_summary = {}
            for tf, sig in mtf_result.signals.items():
                signal_summary[tf] = {
                    'direction': sig.direction.value,
                    'strength': round(sig.strength, 3),
                    'rsi': round(sig.rsi, 1),
                }
            
            result['mtf'] = {
                'aligned': mtf_result.aligned,
                'bias': mtf_result.direction.value.upper(),
                'strength': round(mtf_result.strength, 3),
                'reason': mtf_result.recommendation,
                'signals': signal_summary,
                '4h': mtf_result.signals.get('4h', None) and mtf_result.signals['4h'].direction.value or 'neutral',
                '1h': mtf_result.signals.get('1h', None) and mtf_result.signals['1h'].direction.value or 'neutral',
                '15m': mtf_result.signals.get('15m', None) and mtf_result.signals['15m'].direction.value or 'neutral',
            }
            logger.info(f"📊 MTF fallback computed for {clean_symbol}: {result['mtf']['bias']}")
        except Exception as mtf_err:
            logger.error(f"MTF fallback error: {mtf_err}")
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
            # Use enhanced signal instead of just large orders
            enhanced = oa.get_enhanced_signal(df)
            result['order_flow'] = {
                'large_buys': enhanced.get('large_buys', 0),
                'large_sells': enhanced.get('large_sells', 0),
                'bias': enhanced.get('bias', 'neutral'),
                'net_flow': enhanced.get('large_buy_volume', 0) - enhanced.get('large_sell_volume', 0),
                'score': enhanced.get('score', 0),
                # Layer details
                'cvd': enhanced.get('cvd', {}),
                'taker': enhanced.get('taker', {}),
                'notable': enhanced.get('notable', {}),
                'orderbook': enhanced.get('orderbook', {}),
            }
        except Exception as e:
            logger.error(f"OrderFlow fallback error: {e}")
            result['order_flow'] = {
                'bias': 'neutral', 'net_flow': 0, 'large_buys': 0, 'large_sells': 0,
                'score': 0, 'cvd': {}, 'taker': {'ratio': 0.5}, 'notable': {},
                'orderbook': {'score': 0, 'bias': 'neutral'}
            }

    # API Server should NOT perform heavy ML analysis on the fly.
    # TFT Forecast (Phase 11.1) requires massive PyTorch model loading that hangs the main thread.
    if not result['forecast']:
        # Return empty forecast instead of loading PyTorch on CPU to prevent API deadlock
        pass

    # Stablecoin Supply Signal (P1.5)
    try:
        from src.features.alternative_data import AlternativeDataCollector
        alt_collector = AlternativeDataCollector()
        stablecoin_data = alt_collector.fetch_stablecoin_supply()
        if stablecoin_data:
            result['stablecoin'] = {
                'total_supply': stablecoin_data.get('total_supply', 0),
                'change_7d_pct': stablecoin_data.get('change_7d_pct', 0),
                'signal': stablecoin_data.get('signal', 'neutral'),
            }
        else:
            result['stablecoin'] = None
    except Exception as sc_err:
        logger.error(f"Stablecoin supply error: {sc_err}")
        result['stablecoin'] = None

    # Store result in 60s cache before returning
    result['_fetched_at'] = time.time()
    _MARKET_CACHE[clean_symbol] = result
    response = {k: v for k, v in result.items() if k != '_fetched_at'}
    return jsonify(response)


@app.route('/api/ohlcv')
def get_ohlcv():
    """Get OHLCV candlestick data from Binance for the dashboard chart."""
    import requests as _req
    import os as _os
    from flask import request as flask_req

    symbol = flask_req.args.get('symbol', 'BTCUSDT').upper().replace('/', '')
    interval = flask_req.args.get('interval', '1h')
    limit = min(int(flask_req.args.get('limit', 500)), 1000)

    try:
        base_url = _os.environ.get("BINANCE_FUTURES_URL", "https://data-api.binance.vision")
        url = base_url + "/api/v3/klines"
        resp = _req.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
        data = resp.json()

        if isinstance(data, dict) and data.get('code'):
            logger.error(f"Binance klines error for {symbol}: {data}")
            return jsonify([])

        candles = []
        for row in data:
            candles.append({
                'time': int(row[0]) // 1000,  # ms → seconds
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
                'volume': float(row[5]),
            })
        return jsonify(candles)
    except Exception as e:
        logger.error(f"OHLCV fetch error for {symbol}: {e}")
        return jsonify([])


@app.route('/api/debug/log')
def get_crash_log():
    """Get crash log if exists."""
    log_file = PROJECT_ROOT / 'crash.log'
    if log_file.exists():
        with open(log_file, 'r') as f:
            return f.read(), 200, {'Content-Type': 'text/plain'}
    return "No crash log found. Bot might be running or log not written.", 404


@app.route('/api/testnet/status')
def get_testnet_status():
    """Get Binance testnet account status, balances and positions."""
    import os
    try:
        api_key = os.getenv('BINANCE_TESTNET_API_KEY', '').strip()
        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', '').strip()

        if not api_key or not api_secret:
            return jsonify({'error': 'Testnet API keys not configured on server', 'configured': False})

        from src.api.binance import BinanceConnector
        testnet = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=True)

        connectivity = testnet.test_connectivity()
        balances = testnet.get_all_balances() or {}

        portfolio_value = 0.0
        positions_data = []
        usdt_balance = 0.0

        # Binance spot testnet only supports a limited set of trading pairs.
        # Attempting to fetch tickers for unsupported symbols returns "Invalid symbol"
        # and can cause slow timeouts if repeated. Limit to known testnet pairs.
        TESTNET_QUOTE_USDT = {'BTC', 'ETH', 'BNB', 'LTC', 'TRX', 'XRP', 'SOL', 'ADA', 'DOGE'}

        for currency, amounts in balances.items():
            total = float(amounts.get('total', 0))
            if total > 0:
                if currency == 'USDT':
                    portfolio_value += total
                    usdt_balance = float(amounts.get('free', 0))
                elif currency in TESTNET_QUOTE_USDT:
                    try:
                        # Use slash format — BinanceConnector.get_ticker strips the slash internally
                        ticker = testnet.get_ticker(f"{currency}/USDT")
                        price = float(ticker.get('last', 0))
                        if price > 0:
                            value_usdt = total * price
                            portfolio_value += value_usdt
                            positions_data.append({
                                'asset': currency,
                                'amount': total,
                                'price': price,
                                'value_usdt': value_usdt
                            })
                    except Exception:
                        pass
                # Skip currencies not in the testnet whitelist to avoid Invalid symbol hangs

        return jsonify({
            'configured': True,
            'connected': bool(connectivity),
            'api_key_prefix': f"{api_key[:8]}...{api_key[-4:]}",
            'portfolio_value': portfolio_value,
            'usdt_balance': usdt_balance,
            'pnl_pct': None,
            'pnl_usdt': None,
            'positions': positions_data,
            'balance_count': len(balances)
        })
    except Exception as e:
        logger.error(f"Testnet status error: {e}")
        return jsonify({'error': str(e), 'configured': True, 'connected': False})


@app.route('/api/testnet/order', methods=['POST'])
def place_testnet_order():
    """Place a market order on Binance testnet."""
    import os
    from flask import request as flask_request
    try:
        api_key = os.getenv('BINANCE_TESTNET_API_KEY', '').strip()
        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', '').strip()

        if not api_key or not api_secret:
            return jsonify({'success': False, 'error': 'Testnet API keys not configured on server'}), 400

        data = flask_request.get_json() or {}
        symbol = data.get('symbol', 'BTC/USDT')
        side = data.get('side', 'buy')
        amount_usdt = float(data.get('amount_usdt', 100))

        from src.api.binance import BinanceConnector
        testnet = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=True)

        ticker = testnet.get_ticker(symbol)
        current_price = float(ticker['last'])

        if side == 'buy':
            amount_base = amount_usdt / current_price
        else:
            base_currency = symbol.split('/')[0]
            balances = testnet.get_all_balances() or {}
            amount_base = float(balances.get(base_currency, {}).get('free', 0))
            if amount_base == 0:
                return jsonify({'success': False, 'error': f'No {base_currency} to sell'}), 400

        order = testnet.place_market_order(symbol=symbol, side=side, amount=amount_base)

        return jsonify({
            'success': bool(order),
            'order': order,
            'price': current_price,
            'amount': amount_base,
            'symbol': symbol,
            'side': side
        })
    except Exception as e:
        logger.error(f"Testnet order error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/testnet/orders')
def get_testnet_orders():
    """Get open orders on Binance testnet."""
    import os
    try:
        api_key = os.getenv('BINANCE_TESTNET_API_KEY', '').strip()
        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', '').strip()

        if not api_key or not api_secret:
            return jsonify({'error': 'Testnet API keys not configured on server', 'orders': []})

        from src.api.binance import BinanceConnector
        testnet = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=True)
        open_orders = testnet.get_open_orders()

        return jsonify({'orders': open_orders or []})
    except Exception as e:
        logger.error(f"Testnet orders error: {e}")
        return jsonify({'error': str(e), 'orders': []})


@app.route('/api/testnet/trades')
def get_testnet_trades():
    """Get testnet trade history (bot-mirrored real orders)."""
    try:
        from src.api.testnet_executor import get_testnet_executor
        executor = get_testnet_executor()
        if not executor:
            return jsonify({'trades': [], 'error': 'Testnet not configured'})
        limit = int(request.args.get('limit', 200))
        trades = executor.get_trades(limit=limit)
        return jsonify({'trades': trades, 'total': len(trades)})
    except Exception as e:
        logger.error(f"GET /api/testnet/trades error: {e}")
        return jsonify({'trades': [], 'error': str(e)})


@app.route('/api/testnet/positions')
def get_testnet_positions():
    """Get current open positions on testnet with live prices and unrealized PNL."""
    try:
        from src.api.testnet_executor import get_testnet_executor
        executor = get_testnet_executor()
        if not executor:
            return jsonify({'positions': [], 'error': 'Testnet not configured'})
        positions = executor.get_current_positions()
        return jsonify({'positions': positions})
    except Exception as e:
        logger.error(f"GET /api/testnet/positions error: {e}")
        return jsonify({'positions': [], 'error': str(e)})


@app.route('/api/testnet/pnl')
def get_testnet_pnl():
    """Get testnet PNL summary: realized, unrealized, win rate, equity curve."""
    try:
        from src.api.testnet_executor import get_testnet_executor
        executor = get_testnet_executor()
        if not executor:
            return jsonify({'error': 'Testnet not configured', 'realized_pnl': 0, 'unrealized_pnl': 0})
        summary = executor.get_pnl_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"GET /api/testnet/pnl error: {e}")
        return jsonify({'error': str(e), 'realized_pnl': 0, 'unrealized_pnl': 0})


@app.route('/api/testnet/execute', methods=['POST'])
def execute_testnet_trade():
    """Manually trigger a testnet trade (for testing). Body: {action, symbol, price, confidence, sl, tp}."""
    import os
    from flask import request as flask_request
    try:
        data = flask_request.get_json() or {}
        action = data.get('action', 'OPEN_LONG_SPLIT')
        symbol = data.get('symbol', 'BTCUSDT')
        confidence = float(data.get('confidence', 0.6))
        sl = float(data.get('sl', 0))
        tp = float(data.get('tp', 0))

        from src.api.testnet_executor import get_testnet_executor
        executor = get_testnet_executor()
        if not executor:
            return jsonify({'success': False, 'error': 'Testnet not configured'}), 400

        # Get current price from connector
        from src.api.binance import BinanceConnector
        api_key = os.getenv('BINANCE_TESTNET_API_KEY', '').strip()
        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', '').strip()
        connector = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=True)

        ccxt_symbol = symbol if '/' in symbol else symbol[:-4] + '/USDT'
        ticker = connector.get_ticker(ccxt_symbol)
        price = float(ticker.get('last', data.get('price', 0)))

        # Build a synthetic bot_trade dict
        bot_trade = {
            'action': action,
            'symbol': symbol,
            'price': price,
            'confidence': confidence,
            'sl': sl if sl > 0 else price * 0.95,
            'tp': tp if tp > 0 else price * 1.05,
            'units': 0,
            'pnl': 0,
        }

        record = executor.mirror_trade(bot_trade, {})
        return jsonify({'success': bool(record and record.get('executed')), 'trade': record})
    except Exception as e:
        logger.error(f"POST /api/testnet/execute error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400


# ---------------------------------------------------------------------------
# HTF Agent endpoints
# ---------------------------------------------------------------------------

HTF_STATE_FILE = PROJECT_ROOT / "logs" / "htf_trading_state.json"
HTF_TRADES_FILE = PROJECT_ROOT / "logs" / "htf_trades.json"


def _load_htf_state() -> dict:
    """Load HTF bot state from disk."""
    if HTF_STATE_FILE.exists():
        try:
            return json.loads(HTF_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _load_htf_trades(limit: int = 200) -> list:
    """Load HTF trade history from line-delimited JSON."""
    if not HTF_TRADES_FILE.exists():
        return []
    trades = []
    try:
        with open(HTF_TRADES_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
    except Exception as e:
        logger.error(f"HTF trades read error: {e}")
    return trades[-limit:]


@app.route('/api/htf/status')
def get_htf_status():
    """Return HTF agent status: position, last signal, model info, feature summary."""
    try:
        state = _load_htf_state()
        if not state:
            return jsonify({
                'running': False,
                'symbol': 'BTCUSDT',
                'position': 0,
                'position_label': 'FLAT',
                'balance': None,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'message': 'HTF bot not running or no state file found',
            })

        # Compute unrealized PnL from current price if position is open
        position = int(state.get('position', 0))
        position_price = float(state.get('position_price', 0.0))
        position_units = float(state.get('position_units', 0.0))
        sl_price = float(state.get('sl_price', 0.0))
        tp_price = float(state.get('tp_price', 0.0))
        realized_pnl = float(state.get('realized_pnl', 0.0))

        # Try to get current price for unrealized PnL
        unrealized_pnl = 0.0
        current_price = 0.0
        try:
            from src.api.binance import BinanceConnector
            connector = BinanceConnector()
            ticker = connector.get_ticker('BTC/USDT')
            current_price = float(ticker.get('last', 0))
            if position != 0 and position_price > 0 and position_units > 0:
                if position == 1:
                    unrealized_pnl = (current_price - position_price) * position_units
                else:
                    unrealized_pnl = (position_price - current_price) * position_units
        except Exception:
            pass

        trades = _load_htf_trades(limit=500)
        close_trades = [t for t in trades if 'CLOSE' in t.get('action', '').upper()]
        wins = [t for t in close_trades if t.get('pnl', 0) > 0]
        win_rate = len(wins) / len(close_trades) if close_trades else 0.0

        return jsonify({
            'running': True,
            'symbol': state.get('symbol', 'BTCUSDT'),
            'position': position,
            'position_label': {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(position, 'FLAT'),
            'position_price': position_price,
            'position_units': position_units,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'current_price': current_price,
            'balance': float(state.get('balance', 0)),
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl,
            'win_rate': win_rate,
            'trade_count': len(close_trades),
            'model_path': state.get('model_path'),
            'start_time': state.get('start_time'),
            'updated_at': state.get('updated_at'),
            'dry_run': state.get('dry_run', True),
        })
    except Exception as e:
        logger.error(f"GET /api/htf/status error: {e}")
        return jsonify({'error': str(e), 'running': False}), 500


@app.route('/api/htf/trades')
def get_htf_trades():
    """Return HTF agent trade history."""
    try:
        limit = int(request.args.get('limit', 200))
        trades = _load_htf_trades(limit=limit)
        return jsonify({'trades': trades, 'total': len(trades)})
    except Exception as e:
        logger.error(f"GET /api/htf/trades error: {e}")
        return jsonify({'trades': [], 'error': str(e)})


@app.route('/api/htf/performance')
def get_htf_performance():
    """Return HTF performance metrics: Sharpe ratio, return, drawdown, win rate."""
    try:
        trades = _load_htf_trades(limit=1000)
        state = _load_htf_state()

        close_trades = [t for t in trades if 'CLOSE' in t.get('action', '').upper()]
        pnls = [float(t.get('pnl', 0)) for t in close_trades]

        if not pnls:
            return jsonify({
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'return_pct': 0.0,
                'message': 'No closed trades yet',
            })

        wins = [p for p in pnls if p > 0]
        win_rate = len(wins) / len(pnls)
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls)

        # Sharpe ratio (daily, assuming each trade ~4h average)
        if len(pnls) > 1:
            mean_r = float(np.mean(pnls))
            std_r = float(np.std(pnls))
            sharpe = (mean_r / (std_r + 1e-10)) * (6 ** 0.5)  # annualise ~6 trades/day
        else:
            sharpe = 0.0

        # Max drawdown via equity curve
        initial_balance = float(state.get('balance', 10000)) - total_pnl
        equity = initial_balance
        peak = equity
        max_dd = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            dd = (peak - equity) / (peak + 1e-10)
            max_dd = max(max_dd, dd)

        return_pct = (total_pnl / (initial_balance + 1e-10)) * 100

        return jsonify({
            'total_trades': len(close_trades),
            'wins': len(wins),
            'losses': len(pnls) - len(wins),
            'win_rate': round(win_rate, 4),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'best_trade': round(max(pnls), 2),
            'worst_trade': round(min(pnls), 2),
            'sharpe': round(sharpe, 3),
            'max_drawdown': round(max_dd * 100, 2),
            'return_pct': round(return_pct, 2),
            'start_time': state.get('start_time'),
        })
    except Exception as e:
        logger.error(f"GET /api/htf/performance error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # threaded=True is CRITICAL to prevent single requests (like Market Analysis fallback)
    # from locking up the entire dashboard and causing 'Trades: 0' sidebars
    app.run(port=5001, debug=False, threaded=True)
