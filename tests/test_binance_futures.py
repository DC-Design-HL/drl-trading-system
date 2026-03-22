"""
Tests for BinanceFuturesConnector — 100% method coverage with mocked HTTP.
All network calls are intercepted; no real API keys required.
"""

import hashlib
import hmac
import json
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import requests

from src.api.binance_futures import BinanceFuturesConnector, DEFAULT_BASE_URL, DEFAULT_RECV_WINDOW


# ── Helpers ───────────────────────────────────────────────────────────────────

FAKE_KEY = "testkey0123456789"
FAKE_SECRET = "testsecret0123456789abcdef"
FAKE_URL = "https://demo-fapi.binance.com"


def make_connector(**kwargs) -> BinanceFuturesConnector:
    return BinanceFuturesConnector(
        api_key=kwargs.get("api_key", FAKE_KEY),
        api_secret=kwargs.get("api_secret", FAKE_SECRET),
        base_url=kwargs.get("base_url", FAKE_URL),
    )


def mock_response(data, status_code=200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


def mock_error_response(data, status_code=400):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = False
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


# ── Construction & signing ────────────────────────────────────────────────────

class TestConstruction:
    def test_defaults(self):
        c = make_connector()
        assert c.api_key == FAKE_KEY
        assert c.api_secret == FAKE_SECRET
        assert c.base_url == FAKE_URL
        assert c.recv_window == DEFAULT_RECV_WINDOW

    def test_custom_base_url(self):
        c = make_connector(base_url="https://testnet.binancefuture.com/")
        assert c.base_url == "https://testnet.binancefuture.com"  # trailing slash stripped

    def test_session_has_api_key_header(self):
        c = make_connector()
        assert c._session.headers.get("X-MBX-APIKEY") == FAKE_KEY

    def test_env_var_base_url(self, monkeypatch):
        monkeypatch.setenv("BINANCE_FUTURES_BASE_URL", "https://custom.example.com")
        c = BinanceFuturesConnector(api_key=FAKE_KEY, api_secret=FAKE_SECRET, base_url=None)
        assert c.base_url == "https://custom.example.com"


class TestSigning:
    def test_sign_query_produces_valid_hmac(self):
        c = make_connector()
        params = {"symbol": "BTCUSDT", "timestamp": 1700000000000, "recvWindow": 6000}
        qs = c._sign_query(params)
        # Manually compute expected signature
        base = "symbol=BTCUSDT&timestamp=1700000000000&recvWindow=6000"
        expected_sig = hmac.new(
            FAKE_SECRET.encode(), base.encode(), hashlib.sha256
        ).hexdigest()
        assert f"&signature={expected_sig}" in qs

    def test_sign_query_appends_signature_last(self):
        c = make_connector()
        params = {"a": "1", "b": "2", "timestamp": 123, "recvWindow": 6000}
        qs = c._sign_query(params)
        assert qs.endswith(f"&signature={qs.split('signature=')[1]}")


class TestRaiseForStatus:
    def test_ok_response_passes(self):
        c = make_connector()
        resp = mock_response({"ok": True}, status_code=200)
        c._raise_for_status(resp)  # should not raise

    def test_error_response_raises_http_error(self):
        c = make_connector()
        resp = mock_error_response({"code": -1121, "msg": "Invalid symbol"}, status_code=400)
        with pytest.raises(requests.HTTPError):
            c._raise_for_status(resp)

    def test_error_response_with_non_json_body(self):
        c = make_connector()
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 500
        resp.ok = False
        resp.json.side_effect = Exception("not json")
        resp.text = "Internal Server Error"
        with pytest.raises(requests.HTTPError):
            c._raise_for_status(resp)


# ── Symbol precision ──────────────────────────────────────────────────────────

class TestSymbolPrecision:
    EXCHANGE_INFO = {
        "symbols": [
            {"symbol": "BTCUSDT", "quantityPrecision": 3, "pricePrecision": 1},
            {"symbol": "ETHUSDT", "quantityPrecision": 3, "pricePrecision": 2},
        ]
    }

    def test_fetch_and_cache(self):
        c = make_connector()
        with patch.object(c, "_get", return_value=self.EXCHANGE_INFO) as mock_get:
            qty = c.get_qty_precision("BTCUSDT")
            price = c.get_price_precision("BTCUSDT")

        assert qty == 3
        assert price == 1
        # Second call uses cache
        with patch.object(c, "_get") as mock_get2:
            c.get_qty_precision("BTCUSDT")
            mock_get2.assert_not_called()

    def test_unknown_symbol_falls_back_to_defaults(self):
        c = make_connector()
        with patch.object(c, "_get", return_value={"symbols": []}):
            qty = c.get_qty_precision("UNKNOWNUSDT")
            price = c.get_price_precision("UNKNOWNUSDT")
        assert qty == 3
        assert price == 2

    def test_exchange_info_fetch_failure_returns_defaults(self):
        c = make_connector()
        with patch.object(c, "_get", side_effect=Exception("network error")):
            qty = c.get_qty_precision("BTCUSDT")
        assert qty == 3


# ── Public endpoints ──────────────────────────────────────────────────────────

class TestPing:
    def test_ping_success(self):
        c = make_connector()
        with patch.object(c._session, "get", return_value=mock_response({})):
            result = c.ping()
        assert result == {}

    def test_ping_network_error_raises(self):
        c = make_connector()
        with patch.object(c._session, "get", side_effect=requests.ConnectionError):
            with pytest.raises(requests.ConnectionError):
                c.ping()


class TestGetTicker:
    def test_returns_price(self):
        c = make_connector()
        payload = {"symbol": "BTCUSDT", "price": "84000.5", "time": 1700000000000}
        with patch.object(c._session, "get", return_value=mock_response(payload)):
            result = c.get_ticker("BTCUSDT")
        assert result == payload

    def test_slash_symbol_normalised(self):
        c = make_connector()
        payload = {"symbol": "ETHUSDT", "price": "3000.0", "time": 123}
        captured = {}

        def fake_get(url, **kwargs):
            captured["params"] = kwargs.get("params") or {}
            return mock_response(payload)

        with patch.object(c._session, "get", side_effect=fake_get):
            c.get_ticker("ETH/USDT")

        # Slash should be stripped before sending to the API
        assert captured["params"].get("symbol") == "ETHUSDT"


# ── Signed account endpoints ──────────────────────────────────────────────────

class TestGetAccount:
    def test_returns_account_dict(self):
        c = make_connector()
        account = {
            "totalWalletBalance": "5000.0",
            "availableBalance": "4000.0",
            "totalUnrealizedProfit": "50.0",
        }
        with patch.object(c._session, "get", return_value=mock_response(account)):
            result = c.get_account()
        assert result["totalWalletBalance"] == "5000.0"

    def test_api_error_raises(self):
        c = make_connector()
        with patch.object(
            c._session, "get",
            return_value=mock_error_response({"code": -2014}, 401)
        ):
            with pytest.raises(requests.HTTPError):
                c.get_account()


class TestGetPositions:
    def test_filters_zero_positions(self):
        c = make_connector()
        raw = [
            {"symbol": "BTCUSDT", "positionAmt": "0.100", "entryPrice": "84000"},
            {"symbol": "ETHUSDT", "positionAmt": "0.000", "entryPrice": "0"},
        ]
        with patch.object(c._session, "get", return_value=mock_response(raw)):
            result = c.get_positions()
        assert len(result) == 1
        assert result[0]["symbol"] == "BTCUSDT"

    def test_returns_empty_on_all_flat(self):
        c = make_connector()
        with patch.object(c._session, "get", return_value=mock_response([
            {"symbol": "BTCUSDT", "positionAmt": "0"}
        ])):
            result = c.get_positions()
        assert result == []


class TestGetPosition:
    def test_returns_matching_position(self):
        c = make_connector()
        raw = [{"symbol": "BTCUSDT", "positionAmt": "0.05", "entryPrice": "83000"}]
        with patch.object(c._session, "get", return_value=mock_response(raw)):
            result = c.get_position("BTCUSDT")
        assert result is not None
        assert result["symbol"] == "BTCUSDT"

    def test_returns_none_when_flat(self):
        c = make_connector()
        with patch.object(c._session, "get", return_value=mock_response([
            {"symbol": "BTCUSDT", "positionAmt": "0"}
        ])):
            result = c.get_position("BTCUSDT")
        assert result is None


# ── Order placement ───────────────────────────────────────────────────────────

class TestPlaceMarketOrder:
    ORDER_RESP = {"orderId": 111, "symbol": "BTCUSDT", "side": "BUY", "status": "FILLED"}

    def test_buy_market(self):
        c = make_connector()
        with patch.object(c, "get_qty_precision", return_value=3):
            with patch.object(c._session, "post", return_value=mock_response(self.ORDER_RESP)):
                result = c.place_market_order("BTCUSDT", "BUY", 0.05678)
        assert result["orderId"] == 111

    def test_sell_market(self):
        c = make_connector()
        with patch.object(c, "get_qty_precision", return_value=3):
            with patch.object(c._session, "post", return_value=mock_response(
                {**self.ORDER_RESP, "side": "SELL"}
            )):
                result = c.place_market_order("BTCUSDT", "sell", 0.05)
        assert result["orderId"] == 111

    def test_quantity_rounded_to_precision(self):
        c = make_connector()
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return mock_response(self.ORDER_RESP)

        with patch.object(c, "get_qty_precision", return_value=3):
            with patch.object(c._session, "post", side_effect=fake_post):
                c.place_market_order("BTCUSDT", "BUY", 0.123456789)

        assert "0.123" in captured["url"]

    def test_api_error_raises(self):
        c = make_connector()
        with patch.object(c, "get_qty_precision", return_value=3):
            with patch.object(c._session, "post", return_value=mock_error_response({}, 400)):
                with pytest.raises(requests.HTTPError):
                    c.place_market_order("BTCUSDT", "BUY", 0.01)


class TestPlaceStopLossOrder:
    SL_RESP = {"orderId": 222, "type": "STOP_MARKET", "side": "SELL"}

    def test_long_sl_uses_sell_side(self):
        c = make_connector()
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return mock_response(self.SL_RESP)

        with patch.object(c, "get_price_precision", return_value=1):
            with patch.object(c, "get_qty_precision", return_value=3):
                with patch.object(c._session, "post", side_effect=fake_post):
                    result = c.place_stop_loss_order("BTCUSDT", "SELL", 80000.0)

        assert result["orderId"] == 222
        assert "SELL" in captured["url"]
        assert "STOP_MARKET" in captured["url"]
        assert "closePosition=true" in captured["url"]

    def test_short_sl_uses_buy_side(self):
        c = make_connector()
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return mock_response({**self.SL_RESP, "side": "BUY"})

        with patch.object(c, "get_price_precision", return_value=1):
            with patch.object(c, "get_qty_precision", return_value=3):
                with patch.object(c._session, "post", side_effect=fake_post):
                    c.place_stop_loss_order("BTCUSDT", "BUY", 90000.0)

        assert "BUY" in captured["url"]

    def test_with_explicit_quantity_no_close_position(self):
        c = make_connector()
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return mock_response(self.SL_RESP)

        with patch.object(c, "get_price_precision", return_value=1):
            with patch.object(c, "get_qty_precision", return_value=3):
                with patch.object(c._session, "post", side_effect=fake_post):
                    c.place_stop_loss_order("BTCUSDT", "SELL", 80000.0, quantity=0.1, close_position=False)

        assert "closePosition" not in captured["url"]
        assert "reduceOnly=true" in captured["url"]

    def test_demo_testnet_4120_returns_sentinel_not_raises(self):
        """On demo-fapi, -4120 should return a sentinel dict (orderId=None), not raise."""
        c = make_connector()
        error_resp = mock_error_response({"code": -4120, "msg": "Order type not supported"}, 400)

        with patch.object(c, "get_price_precision", return_value=1):
            with patch.object(c, "get_qty_precision", return_value=3):
                with patch.object(c._session, "post", return_value=error_resp):
                    result = c.place_stop_loss_order("BTCUSDT", "SELL", 80000.0)

        assert result["orderId"] is None
        assert result["status"] == "TESTNET_NOT_SUPPORTED"

    def test_non_4120_error_still_raises(self):
        """Non-4120 errors should still propagate."""
        c = make_connector()
        with patch.object(c, "get_price_precision", return_value=1):
            with patch.object(c, "get_qty_precision", return_value=3):
                with patch.object(c._session, "post",
                                  return_value=mock_error_response({"code": -2010}, 400)):
                    with pytest.raises(requests.HTTPError):
                        c.place_stop_loss_order("BTCUSDT", "SELL", 80000.0)


class TestPlaceTakeProfitOrder:
    TP_RESP = {"orderId": 333, "type": "TAKE_PROFIT_MARKET", "side": "SELL"}

    def test_long_tp(self):
        c = make_connector()
        with patch.object(c, "get_price_precision", return_value=1):
            with patch.object(c, "get_qty_precision", return_value=3):
                with patch.object(c._session, "post", return_value=mock_response(self.TP_RESP)):
                    result = c.place_take_profit_order("BTCUSDT", "SELL", 90000.0)
        assert result["orderId"] == 333

    def test_short_tp_buy_side(self):
        c = make_connector()
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return mock_response({**self.TP_RESP, "side": "BUY"})

        with patch.object(c, "get_price_precision", return_value=1):
            with patch.object(c, "get_qty_precision", return_value=3):
                with patch.object(c._session, "post", side_effect=fake_post):
                    c.place_take_profit_order("BTCUSDT", "BUY", 70000.0)

        assert "BUY" in captured["url"]
        assert "TAKE_PROFIT_MARKET" in captured["url"]

    def test_with_quantity_no_close_position(self):
        c = make_connector()
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return mock_response(self.TP_RESP)

        with patch.object(c, "get_price_precision", return_value=1):
            with patch.object(c, "get_qty_precision", return_value=3):
                with patch.object(c._session, "post", side_effect=fake_post):
                    c.place_take_profit_order("BTCUSDT", "SELL", 90000.0, quantity=0.05, close_position=False)

        assert "closePosition" not in captured["url"]


class TestCancelOrder:
    def test_cancel_success(self):
        c = make_connector()
        resp = {"orderId": 111, "status": "CANCELED"}
        with patch.object(c._session, "delete", return_value=mock_response(resp)):
            result = c.cancel_order("BTCUSDT", 111)
        assert result["orderId"] == 111

    def test_cancel_api_error_raises(self):
        c = make_connector()
        with patch.object(
            c._session, "delete",
            return_value=mock_error_response({"code": -2011, "msg": "Unknown order"}, 400)
        ):
            with pytest.raises(requests.HTTPError):
                c.cancel_order("BTCUSDT", 999)


class TestCancelAllOrders:
    def test_cancel_all(self):
        c = make_connector()
        resp = {"code": 200, "msg": "The operation of cancel all open order is done."}
        with patch.object(c._session, "delete", return_value=mock_response(resp)):
            result = c.cancel_all_orders("BTCUSDT")
        assert result["code"] == 200

    def test_error_raises(self):
        c = make_connector()
        with patch.object(
            c._session, "delete",
            return_value=mock_error_response({}, 400)
        ):
            with pytest.raises(requests.HTTPError):
                c.cancel_all_orders("BTCUSDT")


class TestGetOpenOrders:
    def test_all_symbols(self):
        c = make_connector()
        orders = [{"orderId": 1, "symbol": "BTCUSDT"}, {"orderId": 2, "symbol": "ETHUSDT"}]
        with patch.object(c._session, "get", return_value=mock_response(orders)):
            result = c.get_open_orders()
        assert len(result) == 2

    def test_filtered_by_symbol(self):
        c = make_connector()
        orders = [{"orderId": 1, "symbol": "BTCUSDT"}]
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            return mock_response(orders)

        with patch.object(c._session, "get", side_effect=fake_get):
            result = c.get_open_orders("btcusdt")

        assert "BTCUSDT" in captured["url"]

    def test_slash_symbol_normalised(self):
        c = make_connector()
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            return mock_response([])

        with patch.object(c._session, "get", side_effect=fake_get):
            c.get_open_orders("BTC/USDT")

        # Slash should be stripped before signing, not URL-encoded as %2F
        assert "BTC%2FUSDT" not in captured["url"]
        assert "BTCUSDT" in captured["url"]


class TestGetTradeHistory:
    def test_returns_list(self):
        c = make_connector()
        trades = [{"id": 1, "symbol": "BTCUSDT", "realizedPnl": "10.5"}]
        with patch.object(c._session, "get", return_value=mock_response(trades)):
            result = c.get_trade_history("BTCUSDT", limit=100)
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_limit_capped_at_1000(self):
        c = make_connector()
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            return mock_response([])

        with patch.object(c._session, "get", side_effect=fake_get):
            c.get_trade_history("BTCUSDT", limit=9999)

        assert "limit=1000" in captured["url"]


class TestSetLeverage:
    def test_set_leverage(self):
        c = make_connector()
        resp = {"symbol": "BTCUSDT", "leverage": 5, "maxNotionalValue": "1000000"}
        with patch.object(c._session, "post", return_value=mock_response(resp)):
            result = c.set_leverage("BTCUSDT", 5)
        assert result["leverage"] == 5

    def test_error_raises(self):
        c = make_connector()
        with patch.object(c._session, "post", return_value=mock_error_response({}, 400)):
            with pytest.raises(requests.HTTPError):
                c.set_leverage("BTCUSDT", 125)


class TestGetMarkPrice:
    def test_from_premium_index(self):
        c = make_connector()
        with patch.object(c._session, "get", return_value=mock_response(
            {"symbol": "BTCUSDT", "markPrice": "84123.45"}
        )):
            result = c.get_mark_price("BTCUSDT")
        assert result == 84123.45

    def test_falls_back_to_ticker_on_failure(self):
        c = make_connector()
        call_count = 0

        def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "premiumIndex" in url:
                raise requests.ConnectionError("timeout")
            return mock_response({"symbol": "BTCUSDT", "price": "84000.0"})

        with patch.object(c._session, "get", side_effect=fake_get):
            result = c.get_mark_price("BTCUSDT")

        assert result == 84000.0
        assert call_count == 2  # premiumIndex failed, ticker succeeded
