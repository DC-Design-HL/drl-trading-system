"""
Futures Testnet Executor

Executes real long/short trades on Binance Futures paper trading (demo-fapi).
All financial data (PnL, balances, positions) comes directly from the exchange.
Zero local PnL calculation.

Design principles:
  - Bot OPENS positions and places SL/TP exchange orders.
  - Exchange autonomously exits positions when SL or TP is triggered.
  - Bot does NOT send close orders — the exchange is the exit authority.
  - SL/TP are placed as STOP_MARKET / TAKE_PROFIT_MARKET with closePosition=true.
  - Trailing SL: cancel old SL order, place new one at updated price.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .binance_futures import BinanceFuturesConnector

logger = logging.getLogger(__name__)

DEFAULT_LEVERAGE = 1
POSITION_SIZE = 0.25  # Fraction of available balance used per trade


class FuturesTestnetExecutor:
    """
    Manages long/short futures positions on Binance paper trading.

    Tracks SL/TP order IDs per symbol so they can be updated (trailing SL).
    """

    def __init__(self, connector: Optional[BinanceFuturesConnector] = None):
        if connector is not None:
            self.connector = connector
        else:
            api_key = os.getenv("BINANCE_FUTURES_API_KEY", "").strip()
            api_secret = os.getenv("BINANCE_FUTURES_API_SECRET", "").strip()
            if not api_key or not api_secret:
                raise ValueError(
                    "BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET not set"
                )
            self.connector = BinanceFuturesConnector(
                api_key=api_key, api_secret=api_secret
            )

        # symbol → exchange order_id for the live SL/TP orders
        self._sl_orders: Dict[str, int] = {}
        self._tp_orders: Dict[str, int] = {}

        # Re-populate order tracking from exchange on startup so that trailing
        # SL/TP updates after a bot restart cancel the *existing* orders rather
        # than stacking new ones on top.
        self._sync_order_tracking()

    # ── Startup sync ──────────────────────────────────────────────────────────

    def _sync_order_tracking(self) -> None:
        """
        Re-populate _sl_orders / _tp_orders from exchange open orders.

        Called once on construction so that after a bot restart the executor
        knows which SL/TP orders already exist on the exchange.  Any open
        STOP_MARKET (reduceOnly / closePosition) order is treated as an SL;
        any LIMIT reduceOnly order is treated as a TP (demo-fapi fallback).

        Failures are silently swallowed so they never block construction.
        """
        try:
            open_orders = self.connector.get_open_orders()
        except Exception as exc:
            logger.warning("_sync_order_tracking: could not fetch open orders: %s", exc)
            return

        for o in open_orders:
            sym = o.get("symbol", "")
            if not sym:
                continue
            oid = o.get("orderId")
            if oid is None:
                continue
            order_type = o.get("type", "")
            reduce_only = o.get("reduceOnly", False)
            close_position = o.get("closePosition", False)

            if order_type == "STOP_MARKET" and (reduce_only or close_position):
                self._sl_orders[sym] = oid
            elif order_type == "LIMIT" and reduce_only:
                # TP fallback order placed when TAKE_PROFIT_MARKET is unsupported
                self._tp_orders[sym] = oid

        if self._sl_orders or self._tp_orders:
            logger.info(
                "_sync_order_tracking: loaded SL=%s TP=%s from exchange",
                dict(self._sl_orders),
                dict(self._tp_orders),
            )

    def ensure_tp_order(self, symbol: str, side: str, tp_price: float) -> Optional[int]:
        """
        Ensure a TP order exists for the position.  If one is already tracked
        (from _sync_order_tracking or a prior open) this is a no-op.  Otherwise
        places a new LIMIT reduceOnly TP order (demo-fapi fallback).

        Returns the TP order ID, or None on failure.
        """
        sym = symbol.upper().replace("/", "")
        if sym in self._tp_orders:
            return self._tp_orders[sym]

        order_side = "SELL" if side.upper() == "LONG" else "BUY"
        try:
            tp_order = self.connector.place_take_profit_order(
                sym, order_side, tp_price, close_position=True
            )
            tp_oid = tp_order.get("orderId")
            if tp_oid is not None:
                self._tp_orders[sym] = tp_oid
                logger.info("ensure_tp_order: placed TP for %s @ %.2f (orderId=%s)", sym, tp_price, tp_oid)
            return tp_oid
        except Exception as exc:
            logger.error("ensure_tp_order: failed for %s: %s", sym, exc)
            return None

    # ── Open positions ────────────────────────────────────────────────────────

    def open_long(
        self,
        symbol: str,
        usdt_amount: float,
        sl: float,
        tp: float,
        leverage: int = DEFAULT_LEVERAGE,
    ) -> Dict[str, Any]:
        """
        Open a LONG position.

        Steps:
          1. Set leverage.
          2. Market BUY (quantity = usdt_amount / mark_price).
          3. STOP_MARKET SELL  → SL (closePosition=true).
          4. TAKE_PROFIT_MARKET SELL → TP (closePosition=true).

        Returns a result dict with executed=True/False plus order IDs.
        """
        sym = symbol.upper().replace("/", "")
        result: Dict[str, Any] = {
            "symbol": sym,
            "side": "LONG",
            "executed": False,
            "error": None,
            "order_id": None,
            "sl_order_id": None,
            "tp_order_id": None,
        }
        try:
            self.connector.set_leverage(sym, leverage)

            mark_price = self.connector.get_mark_price(sym)
            if mark_price <= 0:
                result["error"] = f"Invalid mark price: {mark_price}"
                return result

            qty_prec = self.connector.get_qty_precision(sym)
            quantity = round(usdt_amount / mark_price, qty_prec)
            if quantity <= 0:
                result["error"] = (
                    f"Calculated quantity is zero "
                    f"(usdt={usdt_amount}, mark={mark_price})"
                )
                return result

            order = self.connector.place_market_order(sym, "BUY", quantity)
            result["order_id"] = order.get("orderId")
            result["quantity"] = quantity
            result["mark_price"] = mark_price
            result["executed"] = True

            # ── SL placement (bot-side monitoring on demo-fapi) ───────────
            if sl > 0:
                try:
                    sl_order = self.connector.place_stop_loss_order(
                        sym, "SELL", sl, close_position=True
                    )
                    sl_oid = sl_order.get("orderId")
                    if sl_oid is not None:
                        self._sl_orders[sym] = sl_oid
                        logger.info("SL exchange order placed for %s: orderId=%s @ $%.2f", sym, sl_oid, sl)
                    else:
                        logger.info(
                            "SL for %s @ $%.2f will be handled by bot-side WebSocket monitoring "
                            "(STOP_MARKET not supported on demo-fapi)", sym, sl,
                        )
                    result["sl_order_id"] = sl_oid
                except Exception as exc:
                    logger.warning("SL placement failed for %s: %s", sym, exc)
                    result["sl_error"] = str(exc)

            # ── TP placement (CRITICAL — must succeed or close position) ──
            if tp > 0:
                tp_oid = None
                tp_error = None
                try:
                    tp_order = self.connector.place_take_profit_order(
                        sym, "SELL", tp, close_position=True
                    )
                    tp_oid = tp_order.get("orderId")
                except Exception as exc:
                    tp_error = str(exc)

                if tp_oid is not None:
                    self._tp_orders[sym] = tp_oid
                    result["tp_order_id"] = tp_oid
                    logger.info("TP order placed for %s: orderId=%s @ $%.2f", sym, tp_oid, tp)
                else:
                    # TP FAILED — position is unprotected, close immediately
                    logger.error(
                        "❌ TP placement FAILED for LONG %s — closing position immediately. Error: %s",
                        sym, tp_error,
                    )
                    try:
                        self.connector.place_market_order(sym, "SELL", quantity)
                        logger.info("Emergency close of LONG %s completed (qty=%s)", sym, quantity)
                    except Exception as close_exc:
                        logger.error("Emergency close ALSO failed for %s: %s", sym, close_exc)
                    # Cancel any SL order we placed
                    if sym in self._sl_orders:
                        try:
                            self.connector.cancel_order(sym, self._sl_orders.pop(sym))
                        except Exception:
                            pass
                    result["executed"] = False
                    result["error"] = f"TP placement failed, position auto-closed: {tp_error}"
                    return result

            logger.info(
                "🚀 FUTURES LONG opened: %s qty=%s mark=$%.2f "
                "SL=$%.2f TP=$%.2f lev=%dx",
                sym, quantity, mark_price, sl, tp, leverage,
            )
        except Exception as exc:
            logger.error("open_long failed for %s: %s", sym, exc, exc_info=True)
            result["error"] = str(exc)
        return result

    def open_short(
        self,
        symbol: str,
        usdt_amount: float,
        sl: float,
        tp: float,
        leverage: int = DEFAULT_LEVERAGE,
    ) -> Dict[str, Any]:
        """
        Open a SHORT position.

        Steps:
          1. Set leverage.
          2. Market SELL (quantity = usdt_amount / mark_price).
          3. STOP_MARKET BUY  → SL (closePosition=true).
          4. TAKE_PROFIT_MARKET BUY → TP (closePosition=true).

        Returns a result dict with executed=True/False plus order IDs.
        """
        sym = symbol.upper().replace("/", "")
        result: Dict[str, Any] = {
            "symbol": sym,
            "side": "SHORT",
            "executed": False,
            "error": None,
            "order_id": None,
            "sl_order_id": None,
            "tp_order_id": None,
        }
        try:
            self.connector.set_leverage(sym, leverage)

            mark_price = self.connector.get_mark_price(sym)
            if mark_price <= 0:
                result["error"] = f"Invalid mark price: {mark_price}"
                return result

            qty_prec = self.connector.get_qty_precision(sym)
            quantity = round(usdt_amount / mark_price, qty_prec)
            if quantity <= 0:
                result["error"] = (
                    f"Calculated quantity is zero "
                    f"(usdt={usdt_amount}, mark={mark_price})"
                )
                return result

            order = self.connector.place_market_order(sym, "SELL", quantity)
            result["order_id"] = order.get("orderId")
            result["quantity"] = quantity
            result["mark_price"] = mark_price
            result["executed"] = True

            # ── SL placement (bot-side monitoring on demo-fapi) ───────────
            if sl > 0:
                try:
                    sl_order = self.connector.place_stop_loss_order(
                        sym, "BUY", sl, close_position=True
                    )
                    sl_oid = sl_order.get("orderId")
                    if sl_oid is not None:
                        self._sl_orders[sym] = sl_oid
                        logger.info("SL exchange order placed for %s: orderId=%s @ $%.2f", sym, sl_oid, sl)
                    else:
                        logger.info(
                            "SL for %s @ $%.2f will be handled by bot-side WebSocket monitoring "
                            "(STOP_MARKET not supported on demo-fapi)", sym, sl,
                        )
                    result["sl_order_id"] = sl_oid
                except Exception as exc:
                    logger.warning("SL placement failed for %s: %s", sym, exc)
                    result["sl_error"] = str(exc)

            # ── TP placement (CRITICAL — must succeed or close position) ──
            if tp > 0:
                tp_oid = None
                tp_error = None
                try:
                    tp_order = self.connector.place_take_profit_order(
                        sym, "BUY", tp, close_position=True
                    )
                    tp_oid = tp_order.get("orderId")
                except Exception as exc:
                    tp_error = str(exc)

                if tp_oid is not None:
                    self._tp_orders[sym] = tp_oid
                    result["tp_order_id"] = tp_oid
                    logger.info("TP order placed for %s: orderId=%s @ $%.2f", sym, tp_oid, tp)
                else:
                    # TP FAILED — position is unprotected, close immediately
                    logger.error(
                        "❌ TP placement FAILED for SHORT %s — closing position immediately. Error: %s",
                        sym, tp_error,
                    )
                    try:
                        self.connector.place_market_order(sym, "BUY", quantity)
                        logger.info("Emergency close of SHORT %s completed (qty=%s)", sym, quantity)
                    except Exception as close_exc:
                        logger.error("Emergency close ALSO failed for %s: %s", sym, close_exc)
                    # Cancel any SL order we placed
                    if sym in self._sl_orders:
                        try:
                            self.connector.cancel_order(sym, self._sl_orders.pop(sym))
                        except Exception:
                            pass
                    result["executed"] = False
                    result["error"] = f"TP placement failed, position auto-closed: {tp_error}"
                    return result

            logger.info(
                "🚀 FUTURES SHORT opened: %s qty=%s mark=$%.2f "
                "SL=$%.2f TP=$%.2f lev=%dx",
                sym, quantity, mark_price, sl, tp, leverage,
            )
        except Exception as exc:
            logger.error("open_short failed for %s: %s", sym, exc, exc_info=True)
            result["error"] = str(exc)
        return result

    # ── SL/TP management ──────────────────────────────────────────────────────

    def update_sl(self, symbol: str, side: str, new_sl: float) -> bool:
        """
        Replace the live SL order with a new stop price (trailing SL).

        side: 'LONG' → close side is 'SELL'; 'SHORT' → close side is 'BUY'.
        Returns True on success, False on failure.
        """
        sym = symbol.upper().replace("/", "")
        order_side = "SELL" if side.upper() == "LONG" else "BUY"

        old_id = self._sl_orders.pop(sym, None)
        if old_id is not None:
            try:
                self.connector.cancel_order(sym, old_id)
            except Exception as exc:
                logger.warning(
                    "Could not cancel old SL %s for %s: %s", old_id, sym, exc
                )

        try:
            sl_order = self.connector.place_stop_loss_order(
                sym, order_side, new_sl, close_position=True
            )
            new_id = sl_order.get("orderId")
            if new_id is not None:
                self._sl_orders[sym] = new_id
            logger.info("🔄 SL updated: %s → $%.2f (orderId=%s)", sym, new_sl, new_id)
            return True
        except Exception as exc:
            logger.error("Failed to place new SL for %s: %s", sym, exc)
            return False

    def update_tp(self, symbol: str, side: str, new_tp: float) -> bool:
        """
        Replace the live TP order with a new stop price.

        side: 'LONG' → close side is 'SELL'; 'SHORT' → close side is 'BUY'.
        Returns True on success, False on failure.
        """
        sym = symbol.upper().replace("/", "")
        order_side = "SELL" if side.upper() == "LONG" else "BUY"

        old_id = self._tp_orders.pop(sym, None)
        if old_id is not None:
            try:
                self.connector.cancel_order(sym, old_id)
            except Exception as exc:
                logger.warning(
                    "Could not cancel old TP %s for %s: %s", old_id, sym, exc
                )

        try:
            tp_order = self.connector.place_take_profit_order(
                sym, order_side, new_tp, close_position=True
            )
            self._tp_orders[sym] = tp_order.get("orderId")
            logger.info("🔄 TP updated: %s → $%.2f", sym, new_tp)
            return True
        except Exception as exc:
            logger.error("Failed to place new TP for %s: %s", sym, exc)
            return False

    # ── Portfolio / position data ─────────────────────────────────────────────

    def sync_positions(self) -> List[Dict]:
        """
        Fetch open positions from exchange and clean up stale order-ID tracking
        for symbols that are now flat.
        Returns the list of open position dicts.
        """
        positions = self.connector.get_positions()
        open_symbols = {p["symbol"] for p in positions}
        for sym in list(self._sl_orders):
            if sym not in open_symbols:
                self._sl_orders.pop(sym, None)
        for sym in list(self._tp_orders):
            if sym not in open_symbols:
                self._tp_orders.pop(sym, None)
        return positions

    def get_portfolio(self) -> Dict[str, float]:
        """
        Return wallet/margin summary directly from /fapi/v2/account.
        All values are exchange-reported — no local calculation.
        """
        account = self.connector.get_account()
        return {
            "total_wallet_balance": float(
                account.get("totalWalletBalance", 0)
            ),
            "available_balance": float(account.get("availableBalance", 0)),
            "total_unrealized_profit": float(
                account.get("totalUnrealizedProfit", 0)
            ),
            "total_margin_balance": float(
                account.get("totalMarginBalance", 0)
            ),
            "total_position_initial_margin": float(
                account.get("totalPositionInitialMargin", 0)
            ),
        }

    def _load_state_and_trades(self) -> tuple:
        """Load state files and last trades for SL/TP/confidence enrichment."""
        import json as _json
        import os

        base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
        state_files = {
            "BTCUSDT": os.path.join(base, "htf_trading_state.json"),
            "ETHUSDT": os.path.join(base, "htf_trading_state_ETHUSDT.json"),
        }
        states: Dict[str, Dict] = {}
        for sym, path in state_files.items():
            try:
                with open(path, "r") as f:
                    states[sym] = _json.load(f)
            except Exception:
                pass

        # Load last confidence per symbol from htf_trades.json (JSONL)
        confidence_map: Dict[str, float] = {}
        trades_path = os.path.join(base, "htf_trades.json")
        try:
            with open(trades_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = _json.loads(line)
                        s = t.get("symbol", "")
                        if "confidence" in t and s:
                            confidence_map[s] = t["confidence"]
                    except Exception:
                        pass
        except Exception:
            pass

        return states, confidence_map

    def get_positions(self) -> List[Dict]:
        """
        Return open positions normalised to a shape compatible with the UI.

        Exchange fields (entryPrice, markPrice, unRealizedProfit, etc.) are
        passed through unchanged — no local recalculation.
        """
        raw = self.connector.get_positions()

        # Enrich with SL/TP prices and confidence
        states, confidence_map = self._load_state_and_trades()

        # Fetch open orders once for TP price lookup
        open_orders: Dict[int, float] = {}
        try:
            for sym_key in ["BTCUSDT", "ETHUSDT"]:
                for order in self.connector.get_open_orders(sym_key):
                    oid = order.get("orderId")
                    price = float(order.get("price", 0))
                    if oid and price:
                        open_orders[int(oid)] = price
        except Exception as exc:
            logger.warning("Failed to fetch open orders for TP enrichment: %s", exc)

        result = []
        for p in raw:
            amt = float(p.get("positionAmt", 0))
            if amt == 0:
                continue
            sym = p.get("symbol", "")

            # TP price: from open orders by tp_order_id, fallback to state file
            tp_order_id = self._tp_orders.get(sym)
            tp_price = None
            if tp_order_id and int(tp_order_id) in open_orders:
                tp_price = open_orders[int(tp_order_id)]
            if tp_price is None and sym in states:
                tp_price = states[sym].get("tp_price")

            # SL price: from state file
            sl_price = states.get(sym, {}).get("sl_price")

            # Confidence: from trades log
            confidence = confidence_map.get(sym)

            result.append(
                {
                    "symbol": sym,
                    "side": "LONG" if amt > 0 else "SHORT",
                    "amount": abs(amt),
                    "entry_price": float(p.get("entryPrice", 0)),
                    "current_price": float(p.get("markPrice", 0)),
                    "unrealized_pnl": float(p.get("unRealizedProfit", 0)),
                    "liquidation_price": float(p.get("liquidationPrice", 0)),
                    "leverage": int(float(p.get("leverage", 1))),
                    "sl_order_id": self._sl_orders.get(sym),
                    "tp_order_id": tp_order_id,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "confidence": confidence,
                    # UI backward-compat fields
                    "unrealized_pnl_pct": (
                        float(p.get("unRealizedProfit", 0))
                        / max(
                            float(p.get("entryPrice", 1)) * abs(amt),
                            1e-9,
                        )
                        * 100
                    ),
                    "simulated": False,
                }
            )
        return result

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """
        Fetch trade fill history from exchange.
        If symbol is None, attempts to fetch for all currently/recently tracked symbols.
        """
        if symbol:
            return self.connector.get_trade_history(symbol.upper(), limit=limit)

        # Best-effort: query all symbols we have (or have had) active orders for
        symbols = list(
            set(list(self._sl_orders.keys()) + list(self._tp_orders.keys()))
        )
        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT"]

        trades: List[Dict] = []
        for sym in symbols:
            try:
                trades.extend(
                    self.connector.get_trade_history(sym, limit=limit)
                )
            except Exception as exc:
                logger.warning("Trade history failed for %s: %s", sym, exc)

        trades.sort(key=lambda t: int(t.get("time", 0)))
        return trades

    def get_pnl_summary(
        self, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        PnL summary sourced entirely from exchange APIs.

        realized_pnl    — sum of realizedPnl from /fapi/v1/userTrades
        unrealized_pnl  — totalUnrealizedProfit from /fapi/v2/account
        win_rate        — fraction of closing fills with positive realizedPnl
        equity_curve    — cumulative realized PnL per closing fill
        """
        # Unrealized from account endpoint
        try:
            account = self.connector.get_account()
            unrealized_pnl = float(account.get("totalUnrealizedProfit", 0))
        except Exception as exc:
            logger.warning("Account fetch failed: %s", exc)
            unrealized_pnl = 0.0

        # Realized from trade history
        trades = self.get_trade_history(symbol=symbol, limit=1000)
        realized_pnl = sum(float(t.get("realizedPnl", 0)) for t in trades)

        # Win rate: only count fills that actually closed a position (PnL != 0)
        closing = [t for t in trades if float(t.get("realizedPnl", 0)) != 0]
        winning = [t for t in closing if float(t.get("realizedPnl", 0)) > 0]
        win_rate = len(winning) / max(1, len(closing))

        # Equity curve: cumulative per closing fill, ordered by time
        equity_curve = []
        cumulative = 0.0
        for t in trades:
            pnl_val = float(t.get("realizedPnl", 0))
            if pnl_val != 0:
                cumulative += pnl_val
                equity_curve.append(
                    {
                        "timestamp": t.get("time", ""),
                        "cumulative_pnl": round(cumulative, 4),
                        "trade_pnl": round(pnl_val, 4),
                        "symbol": t.get("symbol", ""),
                    }
                )

        return {
            "realized_pnl": round(realized_pnl, 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "total_pnl": round(realized_pnl + unrealized_pnl, 4),
            "total_trades": len(trades),
            "closed_trades": len(closing),
            "winning_trades": len(winning),
            "win_rate": round(win_rate, 4),
            "equity_curve": equity_curve,
        }


def get_futures_executor() -> Optional[FuturesTestnetExecutor]:
    """Factory: return FuturesTestnetExecutor if API keys are configured, else None."""
    try:
        return FuturesTestnetExecutor()
    except Exception as exc:
        logger.warning("FuturesTestnetExecutor unavailable: %s", exc)
        return None
