"""
Testnet Trade Executor
Mirrors bot trading decisions to Binance Testnet with real order execution.
Stores results in logs/testnet_trades.json (line-delimited JSON).
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from .binance import BinanceConnector

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

    LONG positions → real BUY spot orders (market 50% + limit 50%)
    CLOSE LONG     → real SELL spot orders
    SHORT/CLOSE SHORT → conceptual (spot testnet has no shorting);
                        any held base currency is sold if present
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
        current_price = float(bot_trade.get('price', 0) or 0)
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
            if 'OPEN_LONG' in action:
                record = self._execute_open_long(
                    record, ccxt_symbol, current_price, confidence, sl, tp
                )
            elif 'CLOSE_LONG' in action:
                record = self._execute_close_long(record, ccxt_symbol, current_price, pnl)
            elif 'OPEN_SHORT' in action:
                record = self._execute_open_short(
                    record, ccxt_symbol, current_price, confidence, sl, tp
                )
            elif 'CLOSE_SHORT' in action:
                record = self._execute_close_short(record, ccxt_symbol, current_price, pnl)
            else:
                logger.warning(f"TestnetExecutor: unknown action '{action}' for {symbol}")
                return None
        except Exception as exc:
            logger.error(f"TestnetExecutor error ({action} {symbol}): {exc}", exc_info=True)
            record['error'] = str(exc)

        self._save_trade(record)
        return record

    def get_current_positions(self) -> List[Dict]:
        """Return open positions enriched with live price + unrealized PNL."""
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
        """Compute PNL summary + equity curve data from trade history."""
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
        }

        logger.info(
            f"🧪 TESTNET LONG opened: {ccxt_symbol} market {market_amount} @ "
            f"${filled_price:,.2f} | SL=${sl:,.2f} TP=${tp:,.2f}"
        )
        return record

    def _execute_close_long(
        self, record: Dict, ccxt_symbol: str, price: float, pnl: float
    ) -> Dict:
        """Close LONG: sell all held base currency at market."""
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
