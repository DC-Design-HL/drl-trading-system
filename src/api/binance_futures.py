"""
Binance USDM Futures REST API Connector

Direct HMAC-SHA256 signed requests — no ccxt dependency.
Compatible with demo-fapi.binance.com (paper trading) and testnet.binancefuture.com.

Environment variables:
  BINANCE_FUTURES_API_KEY     — futures API key
  BINANCE_FUTURES_API_SECRET  — futures API secret
  BINANCE_FUTURES_BASE_URL    — base URL (default: https://demo-fapi.binance.com)
"""

import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://demo-fapi.binance.com"
DEFAULT_RECV_WINDOW = 60000


class BinanceFuturesConnector:
    """
    Direct REST connector for Binance USDM Futures API.

    All signed requests use HMAC-SHA256 authentication. Parameters are sent
    as a query string; the signature is appended as the last parameter.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
        recv_window: int = DEFAULT_RECV_WINDOW,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = (
            base_url
            or os.getenv("BINANCE_FUTURES_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.recv_window = recv_window

        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self.api_key})

        # Cache symbol precision info fetched from exchangeInfo
        self._symbol_cache: Dict[str, Dict] = {}

    # ── Signing & HTTP helpers ────────────────────────────────────────────────

    def _sign_query(self, params: Dict) -> str:
        """Build a URL-encoded query string with appended HMAC-SHA256 signature."""
        query = urlencode(params)
        sig = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query}&signature={sig}"

    def _get(
        self,
        path: str,
        params: Optional[Dict] = None,
        signed: bool = False,
    ) -> Any:
        p = dict(params or {})
        if signed:
            p["timestamp"] = int(time.time() * 1000)
            p["recvWindow"] = self.recv_window
            url = f"{self.base_url}{path}?{self._sign_query(p)}"
            resp = self._session.get(url, timeout=15)
        else:
            resp = self._session.get(
                f"{self.base_url}{path}", params=p or None, timeout=15
            )
        self._raise_for_status(resp)
        return resp.json()

    def _post(self, path: str, params: Optional[Dict] = None) -> Any:
        p = dict(params or {})
        p["timestamp"] = int(time.time() * 1000)
        p["recvWindow"] = self.recv_window
        url = f"{self.base_url}{path}?{self._sign_query(p)}"
        resp = self._session.post(url, timeout=15)
        self._raise_for_status(resp)
        return resp.json()

    def _delete(self, path: str, params: Optional[Dict] = None) -> Any:
        p = dict(params or {})
        p["timestamp"] = int(time.time() * 1000)
        p["recvWindow"] = self.recv_window
        url = f"{self.base_url}{path}?{self._sign_query(p)}"
        resp = self._session.delete(url, timeout=15)
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise requests.HTTPError(
                f"Binance Futures API {resp.status_code}: {detail}",
                response=resp,
            )

    # ── Symbol precision ──────────────────────────────────────────────────────

    def _fetch_symbol_info(self, symbol: str) -> Dict:
        """Fetch and cache precision info for *symbol* from /fapi/v1/exchangeInfo."""
        sym = symbol.upper()
        if sym in self._symbol_cache:
            return self._symbol_cache[sym]
        try:
            info = self._get("/fapi/v1/exchangeInfo")
            for s in info.get("symbols", []):
                sname = s["symbol"]
                # Binance provides pricePrecision / quantityPrecision directly
                # Extract tick size and step size from filters
                tick_size = None
                step_size = None
                for f in s.get("filters", []):
                    if f.get("filterType") == "PRICE_FILTER":
                        tick_size = float(f.get("tickSize", 0))
                    elif f.get("filterType") == "LOT_SIZE":
                        step_size = float(f.get("stepSize", 0))
                self._symbol_cache[sname] = {
                    "qty_precision": int(s.get("quantityPrecision", 3)),
                    "price_precision": int(s.get("pricePrecision", 2)),
                    "tick_size": tick_size or 0.01,
                    "step_size": step_size or 0.001,
                }
        except Exception as exc:
            logger.warning("Could not fetch futures exchangeInfo: %s", exc)
        return self._symbol_cache.get(
            sym, {"qty_precision": 3, "price_precision": 2}
        )

    def get_tick_size(self, symbol: str) -> float:
        return self._fetch_symbol_info(symbol)["tick_size"]

    def get_step_size(self, symbol: str) -> float:
        return self._fetch_symbol_info(symbol)["step_size"]

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to the nearest valid tick size for the symbol."""
        tick = self.get_tick_size(symbol)
        return round(round(price / tick) * tick, self.get_price_precision(symbol))

    def round_qty(self, symbol: str, qty: float) -> float:
        """Round quantity to the nearest valid step size for the symbol."""
        step = self.get_step_size(symbol)
        return round(round(qty / step) * step, self.get_qty_precision(symbol))

    def get_qty_precision(self, symbol: str) -> int:
        return self._fetch_symbol_info(symbol)["qty_precision"]

    def get_price_precision(self, symbol: str) -> int:
        return self._fetch_symbol_info(symbol)["price_precision"]

    # ── Public API ────────────────────────────────────────────────────────────

    def ping(self) -> Dict:
        """Test API connectivity. Returns {} on success."""
        return self._get("/fapi/v1/ping")

    def get_account(self) -> Dict:
        """
        Full account snapshot.
        Key fields: totalWalletBalance, availableBalance, totalUnrealizedProfit,
                    totalMarginBalance, totalPositionInitialMargin, assets, positions.
        """
        return self._get("/fapi/v2/account", signed=True)

    def get_positions(self) -> List[Dict]:
        """
        All open positions (positionAmt != 0) from /fapi/v2/positionRisk.
        Each record includes: symbol, positionAmt, entryPrice, markPrice,
        unRealizedProfit, liquidationPrice, leverage, positionSide.
        """
        raw = self._get("/fapi/v2/positionRisk", signed=True)
        return [p for p in raw if float(p.get("positionAmt", 0)) != 0]

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Fetch the open position for a single symbol, or None if flat.
        """
        raw = self._get(
            "/fapi/v2/positionRisk",
            params={"symbol": symbol.upper()},
            signed=True,
        )
        for p in raw:
            if float(p.get("positionAmt", 0)) != 0:
                return p
        return None

    def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> Dict:
        """
        Place a MARKET order.
        side: 'BUY' (open long / close short) or 'SELL' (open short / close long).
        quantity: base asset amount.
        """
        sym = symbol.upper()
        qty_prec = self.get_qty_precision(sym)
        return self._post(
            "/fapi/v1/order",
            {
                "symbol": sym,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": round(quantity, qty_prec),
            },
        )

    def place_stop_loss_order(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        quantity: Optional[float] = None,
        close_position: bool = True,
    ) -> Dict:
        """
        Place a STOP_MARKET stop loss order.

        Strategy (in order):
          1. Try the Algo Order API (POST /fapi/v1/algoOrder) — the new
             required endpoint since Binance migrated conditional orders.
             Returns algoId instead of orderId.
          2. Fall back to the legacy order API (POST /fapi/v1/order) in case
             the Algo endpoint is not yet available on this environment.
          3. On -4120 (STOP_ORDER_SWITCH_ALGO) from legacy, return a sentinel
             so the caller can rely on bot-side WebSocket monitoring.

        side: 'SELL' to close a long, 'BUY' to close a short.
        close_position=True: exchange closes the entire position automatically.
        workingType=MARK_PRICE: trigger on mark price (avoids wick triggers).
        """
        sym = symbol.upper()

        # ── Try Algo Order API first ──────────────────────────────────
        try:
            result = self.place_stop_loss_algo(
                sym, side, stop_price,
                quantity=quantity,
                close_position=close_position,
            )
            algo_id = result.get("algoId")
            logger.info(
                "SL STOP_MARKET placed via Algo API: %s side=%s trigger=$%.2f algoId=%s",
                sym, side, stop_price, algo_id,
            )
            # Normalize: add orderId pointing to algoId for backward compat
            result["orderId"] = algo_id
            result["_algo_order"] = True
            return result
        except Exception as algo_exc:
            logger.warning(
                "Algo order SL failed for %s (will try legacy): %s", sym, algo_exc
            )

        # ── Fallback: legacy order API ────────────────────────────────
        params: Dict[str, Any] = {
            "symbol": sym,
            "side": side.upper(),
            "type": "STOP_MARKET",
            "stopPrice": self.round_price(sym, stop_price),
            "workingType": "MARK_PRICE",
        }
        if close_position:
            params["closePosition"] = "true"
        elif quantity is not None:
            qty_prec = self.get_qty_precision(sym)
            params["quantity"] = round(quantity, qty_prec)
            params["reduceOnly"] = "true"
        try:
            return self._post("/fapi/v1/order", params)
        except Exception as exc:
            if "-4120" in str(exc):
                # Neither algo nor legacy worked — return sentinel.
                logger.info(
                    "STOP_MARKET not supported via either API path. "
                    "Bot-side monitoring is the SL authority. Symbol=%s stop=$%.2f",
                    sym, stop_price,
                )
                return {
                    "orderId": None,
                    "type": "STOP_MARKET",
                    "status": "TESTNET_NOT_SUPPORTED",
                    "note": "Conditional orders not supported; "
                            "bot-side monitoring handles SL",
                }
            raise

    def place_take_profit_order(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        quantity: Optional[float] = None,
        close_position: bool = True,
    ) -> Dict:
        """
        Place a TAKE_PROFIT_MARKET order (take profit).

        Strategy (in order):
          1. Try the Algo Order API (POST /fapi/v1/algoOrder) — the new
             required endpoint since Binance migrated conditional orders.
          2. Fall back to the legacy order API (POST /fapi/v1/order).
          3. On -4120, fall back to LIMIT reduceOnly which correctly simulates
             TP behavior for both LONG and SHORT positions.
        """
        sym = symbol.upper()

        # ── Try Algo Order API first ──────────────────────────────────
        try:
            result = self.place_take_profit_algo(
                sym, side, stop_price,
                quantity=quantity,
                close_position=close_position,
            )
            algo_id = result.get("algoId")
            logger.info(
                "TP TAKE_PROFIT_MARKET placed via Algo API: %s side=%s trigger=$%.2f algoId=%s",
                sym, side, stop_price, algo_id,
            )
            # Normalize: add orderId pointing to algoId for backward compat
            result["orderId"] = algo_id
            result["_algo_order"] = True
            return result
        except Exception as algo_exc:
            logger.warning(
                "Algo order TP failed for %s (will try legacy): %s", sym, algo_exc
            )

        # ── Fallback: legacy order API ────────────────────────────────
        params: Dict[str, Any] = {
            "symbol": sym,
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": self.round_price(sym, stop_price),
            "workingType": "MARK_PRICE",
        }
        if close_position:
            params["closePosition"] = "true"
        elif quantity is not None:
            qty_prec = self.get_qty_precision(sym)
            params["quantity"] = round(quantity, qty_prec)
            params["reduceOnly"] = "true"
        try:
            return self._post("/fapi/v1/order", params)
        except Exception as exc:
            if "-4120" in str(exc):
                # Last resort: use LIMIT reduceOnly as TP
                logger.info(
                    "TAKE_PROFIT_MARKET not supported — using LIMIT reduceOnly at $%.2f for %s",
                    stop_price, sym,
                )
                qty_prec = self.get_qty_precision(sym)
                if quantity is None:
                    # Get position quantity from exchange
                    pos = self.get_position(sym)
                    if pos:
                        quantity = abs(float(pos.get("positionAmt", 0)))
                    else:
                        raise
                limit_params: Dict[str, Any] = {
                    "symbol": sym,
                    "side": side.upper(),
                    "type": "LIMIT",
                    "price": self.round_price(sym, stop_price),
                    "quantity": self.round_qty(sym, quantity),
                    "timeInForce": "GTC",
                    "reduceOnly": "true",
                }
                return self._post("/fapi/v1/order", limit_params)
            raise

    def cancel_order(self, symbol: str, order_id: int, is_algo: bool = False) -> Dict:
        """
        Cancel a specific order by ID.

        If is_algo=True, cancels via the Algo Order API (DELETE /fapi/v1/algoOrder).
        Otherwise uses the standard order API (DELETE /fapi/v1/order).

        If standard cancel fails with a 'not found' type error, automatically
        retries as an algo order cancel.
        """
        if is_algo:
            return self.cancel_algo_order(algo_id=order_id)

        try:
            return self._delete(
                "/fapi/v1/order",
                {"symbol": symbol.upper(), "orderId": order_id},
            )
        except Exception as exc:
            # If the order is not found in regular orders, try algo orders
            err_str = str(exc)
            if "-2011" in err_str or "Unknown order" in err_str:
                logger.info(
                    "Order %s not found in regular orders for %s, trying algo cancel...",
                    order_id, symbol,
                )
                try:
                    return self.cancel_algo_order(algo_id=order_id)
                except Exception as algo_exc:
                    logger.warning("Algo cancel also failed for %s: %s", order_id, algo_exc)
                    raise exc  # Re-raise original error
            raise

    def cancel_all_orders(self, symbol: str) -> Dict:
        """
        Cancel all open orders for a symbol — both standard and algo orders.
        """
        results = {}

        # Cancel standard orders
        try:
            results["standard"] = self._delete(
                "/fapi/v1/allOpenOrders", {"symbol": symbol.upper()}
            )
        except Exception as exc:
            logger.warning("cancel_all standard orders failed for %s: %s", symbol, exc)
            results["standard_error"] = str(exc)

        # Cancel algo orders
        try:
            results["algo"] = self.cancel_all_algo_orders(symbol)
        except Exception as exc:
            logger.warning("cancel_all algo orders failed for %s: %s", symbol, exc)
            results["algo_error"] = str(exc)

        return results

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Fetch all open orders, optionally filtered by symbol."""
        params: Dict = {}
        if symbol:
            params["symbol"] = symbol.upper().replace("/", "")
        return self._get("/fapi/v1/openOrders", params=params, signed=True)

    def get_trade_history(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Fetch trade fill history for symbol from /fapi/v1/userTrades.
        Each record includes: symbol, orderId, side, price, qty, realizedPnl,
                               commission, time.
        """
        return self._get(
            "/fapi/v1/userTrades",
            params={"symbol": symbol.upper(), "limit": min(limit, 1000)},
            signed=True,
        )

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set cross-margin leverage for symbol. Returns {symbol, leverage, maxNotionalValue}."""
        return self._post(
            "/fapi/v1/leverage",
            {"symbol": symbol.upper(), "leverage": int(leverage)},
        )

    def get_ticker(self, symbol: str) -> Dict:
        """Fetch latest price ticker. Returns {symbol, price, time}."""
        return self._get(
            "/fapi/v1/ticker/price", params={"symbol": symbol.upper().replace("/", "")}
        )

    # ── Algo Order API ─────────────────────────────────────────────────────
    #
    # Binance migrated conditional orders (STOP_MARKET, TAKE_PROFIT_MARKET,
    # STOP, TAKE_PROFIT, TRAILING_STOP_MARKET) to the Algo Order API.
    # The old /fapi/v1/order endpoint returns error -4120 for these types.
    # These methods use POST/DELETE/GET /fapi/v1/algoOrder and related endpoints.

    def place_algo_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        trigger_price: float,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        close_position: bool = False,
        reduce_only: bool = False,
        working_type: str = "MARK_PRICE",
        price_protect: bool = False,
        position_side: str = "BOTH",
        time_in_force: str = "GTC",
        client_algo_id: Optional[str] = None,
        callback_rate: Optional[float] = None,
        activate_price: Optional[float] = None,
    ) -> Dict:
        """
        Place an algo order via POST /fapi/v1/algoOrder.

        Used for conditional orders: STOP_MARKET, TAKE_PROFIT_MARKET,
        STOP, TAKE_PROFIT, TRAILING_STOP_MARKET.

        Args:
            symbol:         Trading pair (e.g. 'BTCUSDT').
            side:           'BUY' or 'SELL'.
            order_type:     STOP_MARKET, TAKE_PROFIT_MARKET, STOP, TAKE_PROFIT,
                            or TRAILING_STOP_MARKET.
            trigger_price:  Price at which the order triggers.
            quantity:        Order quantity. Cannot be sent with closePosition=true.
            price:          Limit price (for STOP / TAKE_PROFIT limit orders).
            close_position: If true, closes the entire position on trigger.
            reduce_only:    If true, only reduces position. Cannot combine with
                            closePosition or Hedge Mode.
            working_type:   MARK_PRICE or CONTRACT_PRICE (default MARK_PRICE).
            price_protect:  Enable price protection (default False).
            position_side:  BOTH (One-way) or LONG/SHORT (Hedge Mode).
            time_in_force:  GTC, IOC, FOK, GTX (default GTC).
            client_algo_id: Custom client order ID (auto-generated if omitted).
            callback_rate:  For TRAILING_STOP_MARKET, 0.1–10 (1 = 1%).
            activate_price: For TRAILING_STOP_MARKET, activation price.

        Returns:
            Algo order response dict with algoId, clientAlgoId, algoStatus, etc.
        """
        sym = symbol.upper()
        params: Dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": sym,
            "side": side.upper(),
            "type": order_type.upper(),
            "triggerPrice": self.round_price(sym, trigger_price),
            "workingType": working_type,
            "positionSide": position_side.upper(),
            "timeInForce": time_in_force,
        }

        if close_position:
            params["closePosition"] = "true"
        elif quantity is not None:
            params["quantity"] = self.round_qty(sym, quantity)
            if reduce_only:
                params["reduceOnly"] = "true"

        if price is not None:
            params["price"] = self.round_price(sym, price)

        if price_protect:
            params["priceProtect"] = "TRUE"

        if client_algo_id is not None:
            params["clientAlgoId"] = client_algo_id

        if callback_rate is not None:
            params["callbackRate"] = callback_rate

        if activate_price is not None:
            params["activatePrice"] = self.round_price(sym, activate_price)

        return self._post("/fapi/v1/algoOrder", params)

    def cancel_algo_order(
        self,
        algo_id: Optional[int] = None,
        client_algo_id: Optional[str] = None,
    ) -> Dict:
        """
        Cancel an active algo order via DELETE /fapi/v1/algoOrder.

        Either algo_id or client_algo_id must be provided.

        Returns:
            Response dict with algoId, clientAlgoId, code, msg.
        """
        params: Dict[str, Any] = {}
        if algo_id is not None:
            params["algoId"] = algo_id
        if client_algo_id is not None:
            params["clientAlgoId"] = client_algo_id
        if not params:
            raise ValueError("Either algo_id or client_algo_id must be provided")
        return self._delete("/fapi/v1/algoOrder", params)

    def cancel_all_algo_orders(self, symbol: str) -> Dict:
        """
        Cancel all open algo orders for a symbol via DELETE /fapi/v1/algoOpenOrders.

        Returns:
            Response dict with code and msg.
        """
        return self._delete(
            "/fapi/v1/algoOpenOrders", {"symbol": symbol.upper()}
        )

    def get_open_algo_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all open algo orders via GET /fapi/v1/openAlgoOrders.

        Weight: 1 for a single symbol, 40 without symbol filter.

        Returns:
            List of open algo order dicts.
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._get("/fapi/v1/openAlgoOrders", params=params, signed=True)

    def get_algo_order(
        self,
        algo_id: Optional[int] = None,
        client_algo_id: Optional[str] = None,
    ) -> Dict:
        """
        Query a specific algo order via GET /fapi/v1/algoOrder.

        Either algo_id or client_algo_id must be provided.

        Returns:
            Algo order detail dict.
        """
        params: Dict[str, Any] = {}
        if algo_id is not None:
            params["algoId"] = algo_id
        if client_algo_id is not None:
            params["clientAlgoId"] = client_algo_id
        if not params:
            raise ValueError("Either algo_id or client_algo_id must be provided")
        return self._get("/fapi/v1/algoOrder", params=params, signed=True)

    def get_all_algo_orders(
        self,
        symbol: str,
        algo_id: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        page: Optional[int] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """
        Get all algo orders (active, CANCELED, TRIGGERED, FINISHED)
        via GET /fapi/v1/allAlgoOrders.

        Weight: 5.

        Returns:
            List of algo order dicts.
        """
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": min(limit, 1000),
        }
        if algo_id is not None:
            params["algoId"] = algo_id
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if page is not None:
            params["page"] = page
        return self._get("/fapi/v1/allAlgoOrders", params=params, signed=True)

    # ── Algo-based SL/TP convenience methods ─────────────────────────────────

    def place_stop_loss_algo(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        quantity: Optional[float] = None,
        close_position: bool = True,
    ) -> Dict:
        """
        Place a STOP_MARKET algo order (stop loss) via the Algo Order API.

        side: 'SELL' to close a long, 'BUY' to close a short.
        close_position=True: exchange closes the entire position automatically.
        workingType=MARK_PRICE: trigger on mark price (avoids wick triggers).
        """
        return self.place_algo_order(
            symbol=symbol,
            side=side,
            order_type="STOP_MARKET",
            trigger_price=stop_price,
            quantity=quantity,
            close_position=close_position,
            working_type="MARK_PRICE",
        )

    def place_take_profit_algo(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        quantity: Optional[float] = None,
        close_position: bool = True,
    ) -> Dict:
        """
        Place a TAKE_PROFIT_MARKET algo order (take profit) via the Algo Order API.

        side: 'SELL' to close a long, 'BUY' to close a short.
        close_position=True: exchange closes the entire position automatically.
        workingType=MARK_PRICE: trigger on mark price (avoids wick triggers).
        """
        return self.place_algo_order(
            symbol=symbol,
            side=side,
            order_type="TAKE_PROFIT_MARKET",
            trigger_price=stop_price,
            quantity=quantity,
            close_position=close_position,
            working_type="MARK_PRICE",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_mark_price(self, symbol: str) -> float:
        """
        Fetch current mark price for sizing calculations.
        Falls back to last trade price if premium index call fails.
        """
        try:
            data = self._get(
                "/fapi/v1/premiumIndex", params={"symbol": symbol.upper()}
            )
            return float(data.get("markPrice", 0))
        except Exception:
            ticker = self.get_ticker(symbol)
            return float(ticker.get("price", 0))
