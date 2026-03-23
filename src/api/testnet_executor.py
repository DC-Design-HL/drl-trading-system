"""
Testnet Trade Executor
Mirrors bot trading decisions to Binance Testnet with real order execution.

LONG/SHORT opens → delegated to FuturesTestnetExecutor (demo-fapi.binance.com).
Exchange handles all exits autonomously via SL/TP orders — bot does NOT close.
Stores bot decision audit log in logs/testnet_trades.json (line-delimited JSON).
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from .binance import BinanceConnector
from .futures_executor import FuturesTestnetExecutor, get_futures_executor

logger = logging.getLogger(__name__)

TESTNET_TRADES_FILE = Path('logs/testnet_trades.json')

# Minimum trade value in USDT
MIN_TRADE_VALUE_USDT = 10.0

# Position size as fraction of testnet balance (mirrors bot's position_size=0.25)
POSITION_SIZE = 0.25


def _get_amount_precision(symbol: str) -> int:
    """Return decimal places for base currency amount."""
    s = symbol.upper()
    if 'BTC' in s:
        return 5
    if 'ETH' in s:
        return 4
    if 'SOL' in s:
        return 2
    if 'XRP' in s:
        return 0
    return 4


def _get_price_precision(symbol: str) -> int:
    """Return decimal places for price."""
    s = symbol.upper()
    if 'BTC' in s:
        return 2
    if 'ETH' in s:
        return 2
    if 'SOL' in s:
        return 3
    if 'XRP' in s:
        return 4
    return 2


def _extract_filled_price(order: Dict, reference_price: float) -> float:
    """
    Extract the actual average filled price from an order response.
    Handles both ccxt normalized responses and raw Binance API responses.
    """
    # ccxt normalized 'average' field
    avg = order.get('average')
    if avg and float(avg) > 0:
        return float(avg)

    # Raw Binance API response: 'fills' array with price/qty per fill
    fills = order.get('fills') or []
    if fills:
        total_qty = sum(float(f.get('qty', f.get('amount', 0))) for f in fills)
        if total_qty > 0:
            weighted = sum(
                float(f.get('price', 0)) * float(f.get('qty', f.get('amount', 0)))
                for f in fills
            )
            return weighted / total_qty

    # ccxt info.fills (raw response nested inside ccxt wrapper)
    info_fills = (order.get('info') or {}).get('fills', [])
    if info_fills:
        total_qty = sum(float(f.get('qty', 0)) for f in info_fills)
        if total_qty > 0:
            weighted = sum(float(f.get('price', 0)) * float(f.get('qty', 0)) for f in info_fills)
            return weighted / total_qty

    # Raw Binance 'price' field (set for LIMIT orders, may be 0 for MARKET)
    raw_price = order.get('price')
    if raw_price and float(raw_price) > 0:
        return float(raw_price)

    return reference_price


def _to_ccxt_symbol(symbol: str) -> str:
    """Convert BTCUSDT → BTC/USDT for ccxt."""
    if '/' in symbol:
        return symbol
    if symbol.endswith('USDT'):
        base = symbol[:-4]
        return f"{base}/USDT"
    return symbol


class TestnetExecutor:
    """
    Executes real orders on Binance Testnet mirroring bot decisions.

    When BINANCE_FUTURES_API_KEY is set:
      - OPEN_LONG / OPEN_SHORT → delegates to FuturesTestnetExecutor (real shorts)
      - CLOSE_LONG / CLOSE_SHORT → no-op (exchange exits via SL/TP orders)
      - update_sl_tp() → cancels/replaces exchange SL order (trailing SL)

    Fallback (spot testnet only, no futures keys):
      - OPEN_LONG → real BUY spot orders (market 50% + limit 50%)
      - CLOSE_LONG → real SELL spot orders
      - OPEN/CLOSE SHORT → removed (no simulated shorts)
    """

    def __init__(self):
        api_key = os.getenv('BINANCE_TESTNET_API_KEY', '').strip()
        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET', '').strip()

        if not api_key or not api_secret:
            raise ValueError("BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET not set")

        self.connector = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )

        # Futures executor (preferred — supports real shorts + autonomous exits)
        self._futures_executor = None
        try:
            self._futures_executor = get_futures_executor()
            if self._futures_executor:
                logger.info(
                    "TestnetExecutor: futures executor active "
                    "(demo-fapi) — real longs/shorts, autonomous SL/TP exits"
                )
        except Exception as exc:
            logger.warning("FuturesTestnetExecutor not available: %s", exc)

        # In-memory position tracking (symbol → position dict)
        self._positions: Dict[str, Dict] = {}
        self._load_positions_from_trades()

    # ── Position loading ──────────────────────────────────────────────────────

    def _load_positions_from_trades(self):
        """Reconstruct current open positions from trade history."""
        trades = self.get_trades(limit=10000)
        self._positions = {}
        for t in trades:
            symbol = t.get('symbol', '')
            action = t.get('action', '')
            if not symbol:
                continue
            if 'OPEN_LONG' in action:
                self._positions[symbol] = {
                    'side': 'LONG',
                    'entry_price': t.get('filled_price') or t.get('price', 0),
                    'amount': t.get('amount', 0),
                    'sl': t.get('sl', 0),
                    'tp': t.get('tp', 0),
                    'timestamp': t.get('timestamp'),
                    'confidence': t.get('confidence', 0),
                    'order_id': t.get('order_id', ''),
                    'simulated': False,
                    # OCO list ID is not persisted across restarts — exchange orders
                    # must be managed manually after a bot restart.
                    'oco_order_list_id': t.get('oco_order_list_id'),
                }
            elif 'CLOSE_LONG' in action or action in ('STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP'):
                self._positions.pop(symbol, None)
            elif 'OPEN_SHORT' in action:
                self._positions[symbol] = {
                    'side': 'SHORT',
                    'entry_price': t.get('price', 0),
                    'amount': t.get('amount', 0),
                    'sl': t.get('sl', 0),
                    'tp': t.get('tp', 0),
                    'timestamp': t.get('timestamp'),
                    'confidence': t.get('confidence', 0),
                    'order_id': t.get('order_id', ''),
                    'simulated': True,  # spot testnet — conceptual short
                }
            elif 'CLOSE_SHORT' in action:
                self._positions.pop(symbol, None)

    # ── Public API ────────────────────────────────────────────────────────────

    def mirror_trade(self, bot_trade: Dict, bot_result: Dict) -> Optional[Dict]:
        """
        Mirror a bot trade decision to Binance Testnet.

        Args:
            bot_trade:  dict returned by MultiAssetTradingBot.execute_trade()
            bot_result: dict returned by MultiAssetTradingBot.run_iteration()

        Returns:
            Testnet trade record saved to file, or None if no action taken.
        """
        if not bot_trade:
            return None

        action = bot_trade.get('action', '')
        symbol = bot_trade.get('symbol', '')
        current_price = float(bot_trade.get('price', 0) or bot_trade.get('exit_price', 0) or 0)
        sl = float(bot_trade.get('sl', 0) or 0)
        tp = float(bot_trade.get('tp', 0) or 0)
        confidence = float(bot_trade.get('confidence', 0) or 0)
        units = float(bot_trade.get('units', 0) or 0)
        pnl = float(bot_trade.get('pnl', 0) or 0)

        if not symbol or current_price <= 0:
            return None

        ccxt_symbol = _to_ccxt_symbol(symbol)

        record: Dict[str, Any] = {
            'symbol': symbol,
            'ccxt_symbol': ccxt_symbol,
            'action': action,
            'price': current_price,
            'filled_price': None,
            'amount': 0.0,
            'side': None,
            'sl': sl,
            'tp': tp,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'order_id': None,
            'executed': False,
            'error': None,
            'pnl': None,
            'dry_run': False,
        }

        try:
            if self._futures_executor:
                # ── Futures path ──────────────────────────────────────────
                # OPEN actions → real futures order + SL/TP exchange orders
                # CLOSE actions → no-op (exchange exits autonomously via SL/TP)
                if 'OPEN_LONG' in action:
                    record = self._execute_futures_open(
                        record, symbol, 'LONG', current_price, confidence,
                        sl, tp, units,
                    )
                elif 'OPEN_SHORT' in action:
                    record = self._execute_futures_open(
                        record, symbol, 'SHORT', current_price, confidence,
                        sl, tp, units,
                    )
                elif 'CLOSE_LONG' in action or 'CLOSE_SHORT' in action:
                    # Exchange handles exit autonomously via SL/TP orders.
                    # Record in audit log but do NOT send a close to the exchange.
                    record['executed'] = True
                    record['note'] = (
                        'Exit handled autonomously by exchange SL/TP orders. '
                        'No close order sent.'
                    )
                    record['pnl'] = pnl
                    self._positions.pop(symbol, None)
                    logger.info(
                        "🧪 TESTNET %s (futures no-op): exchange exits autonomously",
                        action,
                    )
                else:
                    logger.warning(
                        "TestnetExecutor: unknown action '%s' for %s", action, symbol
                    )
                    return None
            else:
                # ── Spot fallback ─────────────────────────────────────────
                if 'OPEN_LONG' in action:
                    record = self._execute_open_long(
                        record, ccxt_symbol, current_price, confidence, sl, tp
                    )
                elif 'CLOSE_LONG' in action:
                    record = self._execute_close_long(
                        record, ccxt_symbol, current_price, pnl
                    )
                elif 'OPEN_SHORT' in action or 'CLOSE_SHORT' in action:
                    # Spot testnet cannot short — record as skipped
                    record['executed'] = False
                    record['note'] = (
                        'Short positions require futures testnet. '
                        'Set BINANCE_FUTURES_API_KEY to enable real shorts.'
                    )
                    logger.warning(
                        "TestnetExecutor: SHORT action skipped — futures keys not configured"
                    )
                else:
                    logger.warning(
                        "TestnetExecutor: unknown action '%s' for %s", action, symbol
                    )
                    return None
        except Exception as exc:
            logger.error(
                "TestnetExecutor error (%s %s): %s", action, symbol, exc, exc_info=True
            )
            record['error'] = str(exc)

        self._save_trade(record)
        return record

    def get_current_positions(self) -> List[Dict]:
        """
        Return open positions with live price + unrealized PnL.
        When futures executor is active: data comes directly from exchange
        (entry price, mark price, unrealized PnL all exchange-reported).
        Fallback: local position tracking + spot ticker prices.
        """
        if self._futures_executor:
            try:
                return self._futures_executor.get_positions()
            except Exception as exc:
                logger.error("Futures get_positions failed: %s", exc)
                return []

        result = []
        for symbol, pos in list(self._positions.items()):
            try:
                ccxt_symbol = _to_ccxt_symbol(symbol)
                ticker = self.connector.get_ticker(ccxt_symbol)
                current_price = float(ticker.get('last', 0) or 0)
                entry_price = float(pos.get('entry_price', 0) or 0)
                amount = float(pos.get('amount', 0) or 0)
                side = pos.get('side', 'LONG')

                if entry_price > 0 and current_price > 0:
                    if side == 'LONG':
                        upnl = (current_price - entry_price) * amount
                        upnl_pct = (current_price - entry_price) / entry_price * 100
                    else:
                        upnl = (entry_price - current_price) * amount
                        upnl_pct = (entry_price - current_price) / entry_price * 100
                else:
                    upnl = 0.0
                    upnl_pct = 0.0

                result.append({
                    'symbol': symbol,
                    'side': side,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'amount': amount,
                    'sl': float(pos.get('sl', 0) or 0),
                    'tp': float(pos.get('tp', 0) or 0),
                    'unrealized_pnl': upnl,
                    'unrealized_pnl_pct': upnl_pct,
                    'confidence': float(pos.get('confidence', 0) or 0),
                    'timestamp': pos.get('timestamp'),
                    'order_id': pos.get('order_id', ''),
                    'simulated': bool(pos.get('simulated', False)),
                })
            except Exception as exc:
                logger.error(f"TestnetExecutor: position fetch failed for {symbol}: {exc}")

        return result

    def get_trades(self, limit: int = 100) -> List[Dict]:
        """Return testnet trade history (oldest first, capped at limit)."""
        if not TESTNET_TRADES_FILE.exists():
            return []
        trades = []
        try:
            with open(TESTNET_TRADES_FILE, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as exc:
            logger.error(f"TestnetExecutor: failed to read trades file: {exc}")
        return trades[-limit:]

    def get_pnl_summary(self) -> Dict:
        """
        PnL summary.
        When futures executor is active: all values from exchange APIs directly.
        Fallback: computed from local audit log (spot testnet only).
        """
        if self._futures_executor:
            try:
                return self._futures_executor.get_pnl_summary()
            except Exception as exc:
                logger.error("Futures get_pnl_summary failed: %s", exc)
                return {
                    'realized_pnl': 0.0, 'unrealized_pnl': 0.0,
                    'total_pnl': 0.0, 'total_trades': 0,
                    'closed_trades': 0, 'winning_trades': 0,
                    'win_rate': 0.0, 'equity_curve': [], 'error': str(exc),
                }

        trades = self.get_trades(limit=10000)

        realized_pnl = sum(float(t.get('pnl', 0) or 0) for t in trades)

        positions = self.get_current_positions()
        unrealized_pnl = sum(float(p.get('unrealized_pnl', 0) or 0) for p in positions)

        closed_trades = [t for t in trades if t.get('pnl') is not None and t.get('pnl') != 0]
        winning = [t for t in closed_trades if float(t.get('pnl', 0) or 0) > 0]
        win_rate = len(winning) / max(1, len(closed_trades))

        # Equity curve: cumulative realized PNL per trade timestamp
        equity_curve = []
        cumulative = 0.0
        for t in trades:
            pnl_val = float(t.get('pnl', 0) or 0)
            if pnl_val != 0:
                cumulative += pnl_val
                equity_curve.append({
                    'timestamp': t.get('timestamp', ''),
                    'cumulative_pnl': round(cumulative, 4),
                    'trade_pnl': round(pnl_val, 4),
                    'symbol': t.get('symbol', ''),
                })

        return {
            'realized_pnl': round(realized_pnl, 4),
            'unrealized_pnl': round(unrealized_pnl, 4),
            'total_pnl': round(realized_pnl + unrealized_pnl, 4),
            'total_trades': len(trades),
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning),
            'win_rate': round(win_rate, 4),
            'equity_curve': equity_curve,
        }

    # ── Private execution helpers ─────────────────────────────────────────────

    def _execute_futures_open(
        self, record: Dict, symbol: str, side: str, price: float,
        confidence: float, sl: float, tp: float, units: float,
    ) -> Dict:
        """Delegate open to FuturesTestnetExecutor. Records result in audit log."""
        # Compute USDT amount from units × price, or from bot balance fraction
        usdt_amount = units * price if units > 0 else price * POSITION_SIZE

        try:
            account = self._futures_executor.get_portfolio()
            avail = account.get('available_balance', 0.0)
            if avail > 0:
                usdt_amount = avail * POSITION_SIZE
        except Exception:
            pass  # fall back to units × price

        conf_scale = max(0.5, min(1.0, confidence)) if confidence > 0 else 0.75
        usdt_amount *= conf_scale

        if usdt_amount < MIN_TRADE_VALUE_USDT:
            record['error'] = f"Trade value ${usdt_amount:.2f} below minimum ${MIN_TRADE_VALUE_USDT}"
            return record

        if side == 'LONG':
            result = self._futures_executor.open_long(symbol, usdt_amount, sl=sl, tp=tp)
        else:
            result = self._futures_executor.open_short(symbol, usdt_amount, sl=sl, tp=tp)

        record['executed'] = result.get('executed', False)
        record['order_id'] = str(result.get('order_id') or '')
        record['sl_order_id'] = result.get('sl_order_id')
        record['tp_order_id'] = result.get('tp_order_id')
        record['quantity'] = result.get('quantity', 0)
        record['mark_price'] = result.get('mark_price', price)
        record['filled_price'] = result.get('mark_price', price)
        record['amount'] = result.get('quantity', 0)
        record['side'] = side

        if result.get('error'):
            record['error'] = result['error']

        if record['executed']:
            self._positions[symbol] = {
                'side': side,
                'entry_price': record['filled_price'],
                'amount': record['amount'],
                'sl': sl,
                'tp': tp,
                'timestamp': record['timestamp'],
                'confidence': confidence,
                'order_id': record['order_id'],
                'simulated': False,
            }

        return record

    def _execute_open_long(
        self, record: Dict, ccxt_symbol: str, price: float,
        confidence: float, sl: float, tp: float
    ) -> Dict:
        """Open LONG: 50% market buy + 50% limit buy at -0.5%."""
        usdt_balance = self.connector.get_balance('USDT')
        if usdt_balance <= 0:
            record['error'] = "No USDT balance available"
            return record

        base_value = usdt_balance * POSITION_SIZE
        # Scale by confidence (clamp to [0.5, 1.0] to avoid zero trades)
        conf_scale = max(0.5, min(1.0, confidence)) if confidence > 0 else 0.75
        scaled_value = base_value * conf_scale

        if scaled_value < MIN_TRADE_VALUE_USDT:
            record['error'] = f"Trade value ${scaled_value:.2f} below minimum ${MIN_TRADE_VALUE_USDT}"
            return record

        prec = _get_amount_precision(ccxt_symbol)
        pprec = _get_price_precision(ccxt_symbol)

        # 50% market order
        market_value = scaled_value * 0.50
        market_amount = round(market_value / price, prec)
        if market_amount <= 0:
            record['error'] = "Calculated market amount is zero"
            return record

        market_order = self.connector.place_market_order(
            symbol=ccxt_symbol, side='buy', amount=market_amount
        )
        if not market_order:
            record['error'] = "Market BUY order failed"
            return record

        filled_price = _extract_filled_price(market_order, price)
        record['order_id'] = str(
            market_order.get('orderId') or market_order.get('id') or ''
        )
        record['amount'] = market_amount
        record['filled_price'] = filled_price
        record['side'] = 'BUY'
        record['executed'] = True

        # 50% limit order at 0.5% dip
        limit_price = round(price * 0.995, pprec)
        limit_amount = round((scaled_value * 0.50) / limit_price, prec)
        if limit_amount > 0:
            limit_order = self.connector.place_limit_order(
                symbol=ccxt_symbol, side='buy', amount=limit_amount, price=limit_price
            )
            if limit_order:
                record['limit_order_id'] = str(
                    limit_order.get('orderId') or limit_order.get('id') or ''
                )
                record['limit_price'] = limit_price
                record['limit_amount'] = limit_amount

        # Track position
        self._positions[record['symbol']] = {
            'side': 'LONG',
            'entry_price': filled_price,
            'amount': market_amount,
            'sl': sl,
            'tp': tp,
            'timestamp': record['timestamp'],
            'confidence': confidence,
            'order_id': record['order_id'],
            'simulated': False,
            'oco_order_list_id': None,
        }

        # Place exchange-side OCO order (TP + SL) to catch flash wicks between polls
        if sl > 0 and tp > 0:
            oco_list_id = self._place_oco_for_long(
                ccxt_symbol, market_amount, sl, tp
            )
            if oco_list_id is not None:
                self._positions[record['symbol']]['oco_order_list_id'] = oco_list_id
                record['oco_order_list_id'] = oco_list_id

        logger.info(
            f"🧪 TESTNET LONG opened: {ccxt_symbol} market {market_amount} @ "
            f"${filled_price:,.2f} | SL=${sl:,.2f} TP=${tp:,.2f}"
        )
        return record

    def _execute_close_long(
        self, record: Dict, ccxt_symbol: str, price: float, pnl: float
    ) -> Dict:
        """Close LONG: cancel any OCO, then sell all held base currency at market."""
        # Cancel outstanding OCO order before closing to avoid double-fill
        pos = self._positions.get(record['symbol'], {})
        oco_list_id = pos.get('oco_order_list_id')
        if oco_list_id is not None:
            self.connector.cancel_order_list(ccxt_symbol, oco_list_id)

        base_currency = ccxt_symbol.split('/')[0]
        balance = self.connector.get_balance(base_currency)

        if balance <= 0:
            record['error'] = f"No {base_currency} balance to sell"
            # Still clear position tracking
            self._positions.pop(record['symbol'], None)
            return record

        prec = _get_amount_precision(ccxt_symbol)
        sell_amount = round(balance, prec)

        order = self.connector.place_market_order(
            symbol=ccxt_symbol, side='sell', amount=sell_amount
        )
        if not order:
            record['error'] = "Market SELL order failed"
            return record

        filled_price = _extract_filled_price(order, price)
        record['order_id'] = str(order.get('orderId') or order.get('id') or '')
        record['amount'] = sell_amount
        record['filled_price'] = filled_price
        record['side'] = 'SELL'
        record['executed'] = True
        record['pnl'] = pnl

        self._positions.pop(record['symbol'], None)

        logger.info(
            f"🧪 TESTNET LONG closed: {ccxt_symbol} sold {sell_amount} @ "
            f"${filled_price:,.2f} | PNL=${pnl:+.2f}"
        )
        return record

    def _execute_open_short(
        self, record: Dict, ccxt_symbol: str, price: float,
        confidence: float, sl: float, tp: float
    ) -> Dict:
        """
        Open SHORT (conceptual on spot testnet).
        If base currency is held, sell it; record position as simulated short.
        """
        base_currency = ccxt_symbol.split('/')[0]
        balance = self.connector.get_balance(base_currency)

        record['side'] = 'SHORT_SIMULATED'
        record['executed'] = True
        record['note'] = (
            'Spot testnet cannot truly short. '
            'Sold any held base currency; position tracked conceptually.'
        )

        if balance > 0:
            prec = _get_amount_precision(ccxt_symbol)
            sell_amount = round(balance, prec)
            order = self.connector.place_market_order(
                symbol=ccxt_symbol, side='sell', amount=sell_amount
            )
            if order:
                filled_price = _extract_filled_price(order, price)
                record['order_id'] = str(order.get('orderId') or order.get('id') or '')
                record['amount'] = sell_amount
                record['filled_price'] = filled_price
                logger.info(
                    f"🧪 TESTNET SHORT (sim): sold {sell_amount} {base_currency} @ "
                    f"${filled_price:,.2f} to open conceptual short"
                )
            else:
                record['error'] = "Sell order for short failed"

        # Track as conceptual short
        self._positions[record['symbol']] = {
            'side': 'SHORT',
            'entry_price': price,
            'amount': float(record.get('amount', 0)),
            'sl': sl,
            'tp': tp,
            'timestamp': record['timestamp'],
            'confidence': confidence,
            'order_id': record.get('order_id', ''),
            'simulated': True,
        }
        return record

    def _place_oco_for_long(
        self, ccxt_symbol: str, amount: float, sl: float, tp: float
    ) -> Optional[int]:
        """
        Place a SELL OCO order to protect a LONG position.
        Returns the orderListId on success, or None on failure.
        """
        pprec = _get_price_precision(ccxt_symbol)
        prec = _get_amount_precision(ccxt_symbol)

        tp_price = round(tp, pprec)
        stop_price = round(sl, pprec)
        # Limit price slightly below stop trigger as slippage buffer
        stop_limit_price = round(sl * 0.998, pprec)
        sell_amount = round(amount, prec)

        try:
            resp = self.connector.place_oco_order(
                symbol=ccxt_symbol,
                side='sell',
                amount=sell_amount,
                price=tp_price,
                stop_price=stop_price,
                stop_limit_price=stop_limit_price,
            )
            if resp:
                oco_list_id = resp.get('orderListId')
                logger.info(
                    f"🧪 OCO placed: {ccxt_symbol} sell {sell_amount} "
                    f"TP=${tp_price} SL_stop=${stop_price} SL_lmt=${stop_limit_price} "
                    f"listId={oco_list_id}"
                )
                return oco_list_id
        except Exception as exc:
            logger.warning(f"OCO placement failed for {ccxt_symbol}: {exc}")
        return None

    def update_sl_tp(self, symbol: str, new_sl: float, new_tp: float) -> None:
        """
        Update exchange-side SL order after trailing SL adjustment.

        Futures path: cancel old STOP_MARKET order, place new one at new_sl.
        Spot fallback: cancel old OCO, place new OCO.
        """
        pos = self._positions.get(symbol)

        if self._futures_executor:
            side = (pos or {}).get('side', 'LONG')
            old_sl = (pos or {}).get('sl', 0)
            old_tp = (pos or {}).get('tp', 0)
            # Only update orders that actually changed (avoid duplicate cancels/placements)
            if new_sl > 0 and abs(new_sl - old_sl) > 0.01:
                self._futures_executor.update_sl(symbol, side, new_sl)
            if new_tp > 0 and abs(new_tp - old_tp) > 0.01:
                self._futures_executor.update_tp(symbol, side, new_tp)
            if pos:
                pos['sl'] = new_sl
                pos['tp'] = new_tp
            return

        # Spot fallback (OCO)
        if not pos or pos.get('simulated'):
            return

        ccxt_symbol = _to_ccxt_symbol(symbol)
        old_list_id = pos.get('oco_order_list_id')

        if old_list_id is not None:
            self.connector.cancel_order_list(ccxt_symbol, old_list_id)
            pos['oco_order_list_id'] = None

        amount = float(pos.get('amount', 0))
        if amount > 0 and new_sl > 0 and new_tp > 0:
            new_list_id = self._place_oco_for_long(ccxt_symbol, amount, new_sl, new_tp)
            pos['oco_order_list_id'] = new_list_id
            pos['sl'] = new_sl
            pos['tp'] = new_tp

    def _execute_close_short(
        self, record: Dict, ccxt_symbol: str, price: float, pnl: float
    ) -> Dict:
        """Close SHORT (conceptual on spot testnet)."""
        record['side'] = 'CLOSE_SHORT'
        record['executed'] = True
        record['pnl'] = pnl
        record['note'] = 'Conceptual short closed (spot testnet — no real short was held)'
        self._positions.pop(record['symbol'], None)
        logger.info(f"🧪 TESTNET SHORT (sim) closed: {ccxt_symbol} | PNL=${pnl:+.2f}")
        return record

    # ── Storage ───────────────────────────────────────────────────────────────

    def _save_trade(self, trade: Dict):
        """Append a trade record to the testnet trades log file."""
        try:
            TESTNET_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TESTNET_TRADES_FILE, 'a') as fh:
                fh.write(json.dumps(trade) + '\n')
        except Exception as exc:
            logger.error(f"TestnetExecutor: failed to save trade: {exc}")


def get_testnet_executor() -> Optional[TestnetExecutor]:
    """Factory: create TestnetExecutor if API keys are configured."""
    try:
        return TestnetExecutor()
    except Exception as exc:
        logger.warning(f"TestnetExecutor unavailable: {exc}")
        return None
