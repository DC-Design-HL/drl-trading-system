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
                self._symbol_cache[sname] = {
                    "qty_precision": int(s.get("quantityPrecision", 3)),
                    "price_precision": int(s.get("pricePrecision", 2)),
                }
        except Exception as exc:
            logger.warning("Could not fetch futures exchangeInfo: %s", exc)
        return self._symbol_cache.get(
            sym, {"qty_precision": 3, "price_precision": 2}
        )

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
        Place a STOP_MARKET order (stop loss).
        side: 'SELL' to close a long, 'BUY' to close a short.
        close_position=True: exchange closes the entire position automatically.
        workingType=MARK_PRICE: trigger on mark price (avoids wick triggers).
        """
        sym = symbol.upper()
        price_prec = self.get_price_precision(sym)
        params: Dict[str, Any] = {
            "symbol": sym,
            "side": side.upper(),
            "type": "STOP_MARKET",
            "stopPrice": round(stop_price, price_prec),
            "workingType": "MARK_PRICE",
        }
        if close_position:
            params["closePosition"] = "true"
        elif quantity is not None:
            qty_prec = self.get_qty_precision(sym)
            params["quantity"] = round(quantity, qty_prec)
            params["reduceOnly"] = "true"
        return self._post("/fapi/v1/order", params)

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
        side: 'SELL' to close a long, 'BUY' to close a short.
        close_position=True: exchange closes the entire position automatically.
        workingType=MARK_PRICE: trigger on mark price.
        """
        sym = symbol.upper()
        price_prec = self.get_price_precision(sym)
        params: Dict[str, Any] = {
            "symbol": sym,
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": round(stop_price, price_prec),
            "workingType": "MARK_PRICE",
        }
        if close_position:
            params["closePosition"] = "true"
        elif quantity is not None:
            qty_prec = self.get_qty_precision(sym)
            params["quantity"] = round(quantity, qty_prec)
            params["reduceOnly"] = "true"
        return self._post("/fapi/v1/order", params)

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel a specific order by ID."""
        return self._delete(
            "/fapi/v1/order",
            {"symbol": symbol.upper(), "orderId": order_id},
        )

    def cancel_all_orders(self, symbol: str) -> Dict:
        """Cancel all open orders for a symbol."""
        return self._delete(
            "/fapi/v1/allOpenOrders", {"symbol": symbol.upper()}
        )

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
