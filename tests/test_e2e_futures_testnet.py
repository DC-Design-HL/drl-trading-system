"""
E2E Integration Tests — Real Binance Futures demo-fapi.binance.com

These tests HIT THE REAL API. They use minimum order sizes and always clean up.
Run with: python3 -m pytest tests/test_e2e_futures_testnet.py -v

Requires BINANCE_FUTURES_API_KEY and BINANCE_FUTURES_API_SECRET in .env
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.binance_futures import BinanceFuturesConnector
from src.api.futures_executor import FuturesTestnetExecutor

API_KEY = os.getenv("BINANCE_FUTURES_API_KEY", "")
API_SECRET = os.getenv("BINANCE_FUTURES_API_SECRET", "")
BASE_URL = os.getenv("BINANCE_FUTURES_BASE_URL", "https://demo-fapi.binance.com")

HAS_CREDS = bool(API_KEY and API_SECRET)
skip_no_creds = pytest.mark.skipif(not HAS_CREDS, reason="No Binance Futures credentials")

SYMBOL = "BTCUSDT"
MIN_QTY = 0.001  # Minimum for BTCUSDT


@pytest.fixture
def conn():
    return BinanceFuturesConnector(api_key=API_KEY, api_secret=API_SECRET, base_url=BASE_URL)


@pytest.fixture
def executor(conn):
    return FuturesTestnetExecutor(connector=conn)


def cleanup_position(conn, symbol=SYMBOL):
    """Close any open position and cancel all orders for symbol."""
    try:
        conn.cancel_all_orders(symbol)
    except Exception:
        pass
    try:
        pos = conn.get_position(symbol)
        if pos:
            amt = float(pos.get("positionAmt", 0))
            if amt > 0:
                conn.place_market_order(symbol, "SELL", abs(amt))
            elif amt < 0:
                conn.place_market_order(symbol, "BUY", abs(amt))
    except Exception:
        pass


# ── Test 1: Connectivity ──────────────────────────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_connectivity(conn):
    result = conn.ping()
    assert isinstance(result, dict)


# ── Test 2: Account Balance ───────────────────────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_account_balance(conn):
    account = conn.get_account()
    balance = float(account.get("totalWalletBalance", 0))
    assert balance > 0, f"Wallet balance should be > 0, got {balance}"


# ── Test 3: Place and Cancel Limit Order ──────────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_place_and_cancel_limit_order(conn):
    # Place a LIMIT BUY far below market so it won't fill
    mark = conn.get_mark_price(SYMBOL)
    price_prec = conn.get_price_precision(SYMBOL)
    limit_price = conn.round_price(SYMBOL, mark * 0.5)  # 50% below market
    # Ensure notional >= 100 USDT
    qty = max(MIN_QTY, round(110 / limit_price, conn.get_qty_precision(SYMBOL)))

    order = None
    try:
        order = conn._post("/fapi/v1/order", {
            "symbol": SYMBOL,
            "side": "BUY",
            "type": "LIMIT",
            "price": limit_price,
            "quantity": qty,
            "timeInForce": "GTC",
        })
        order_id = order["orderId"]
        assert order_id is not None

        # Verify it's in open orders
        time.sleep(0.5)
        open_orders = conn.get_open_orders(SYMBOL)
        ids = [o["orderId"] for o in open_orders]
        assert order_id in ids, f"Order {order_id} not found in open orders"

        # Cancel
        conn.cancel_order(SYMBOL, order_id)
        time.sleep(0.5)

        # Verify gone
        open_orders = conn.get_open_orders(SYMBOL)
        ids = [o["orderId"] for o in open_orders]
        assert order_id not in ids
    finally:
        if order and order.get("orderId"):
            try:
                conn.cancel_order(SYMBOL, order["orderId"])
            except Exception:
                pass


# ── Test 4: Open Long with TP ────────────────────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_open_long_with_tp(conn):
    cleanup_position(conn)
    try:
        conn.set_leverage(SYMBOL, 1)
        mark = conn.get_mark_price(SYMBOL)
        price_prec = conn.get_price_precision(SYMBOL)
        qty_prec = conn.get_qty_precision(SYMBOL)

        # Ensure minimum notional
        qty = max(MIN_QTY, round(110 / mark, qty_prec))

        # Open long
        order = conn.place_market_order(SYMBOL, "BUY", qty)
        assert order.get("orderId") is not None

        time.sleep(1)

        # Place TP (LIMIT SELL above market with reduceOnly)
        tp_price = conn.round_price(SYMBOL, mark * 1.10)  # 10% above
        tp_order = conn._post("/fapi/v1/order", {
            "symbol": SYMBOL,
            "side": "SELL",
            "type": "LIMIT",
            "price": tp_price,
            "quantity": qty,
            "timeInForce": "GTC",
            "reduceOnly": "true",
        })
        tp_oid = tp_order.get("orderId")
        assert tp_oid is not None

        time.sleep(0.5)

        # Verify position exists
        pos = conn.get_position(SYMBOL)
        assert pos is not None
        assert float(pos["positionAmt"]) > 0

        # Verify TP order exists
        open_orders = conn.get_open_orders(SYMBOL)
        tp_ids = [o["orderId"] for o in open_orders if o.get("reduceOnly")]
        assert tp_oid in tp_ids
    finally:
        cleanup_position(conn)


# ── Test 5: Open Short with TP ───────────────────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_open_short_with_tp(conn):
    cleanup_position(conn)
    try:
        conn.set_leverage(SYMBOL, 1)
        mark = conn.get_mark_price(SYMBOL)
        price_prec = conn.get_price_precision(SYMBOL)
        qty_prec = conn.get_qty_precision(SYMBOL)

        qty = max(MIN_QTY, round(110 / mark, qty_prec))

        # Open short
        order = conn.place_market_order(SYMBOL, "SELL", qty)
        assert order.get("orderId") is not None

        time.sleep(1)

        # Place TP (LIMIT BUY below market with reduceOnly)
        tp_price = conn.round_price(SYMBOL, mark * 0.90)  # 10% below
        tp_order = conn._post("/fapi/v1/order", {
            "symbol": SYMBOL,
            "side": "BUY",
            "type": "LIMIT",
            "price": tp_price,
            "quantity": qty,
            "timeInForce": "GTC",
            "reduceOnly": "true",
        })
        tp_oid = tp_order.get("orderId")
        assert tp_oid is not None

        time.sleep(0.5)

        # Verify position
        pos = conn.get_position(SYMBOL)
        assert pos is not None
        assert float(pos["positionAmt"]) < 0

        # Verify TP order
        open_orders = conn.get_open_orders(SYMBOL)
        tp_ids = [o["orderId"] for o in open_orders if o.get("reduceOnly")]
        assert tp_oid in tp_ids
    finally:
        cleanup_position(conn)


# ── Test 6: STOP_MARKET Rejected ─────────────────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_stop_market_rejected(conn):
    """Verify STOP_MARKET returns -4120 on demo-fapi (documents the limitation)."""
    mark = conn.get_mark_price(SYMBOL)
    stop_price = conn.round_price(SYMBOL, mark * 0.95)

    from requests import HTTPError
    with pytest.raises(HTTPError, match="-4120"):
        conn._post("/fapi/v1/order", {
            "symbol": SYMBOL,
            "side": "SELL",
            "type": "STOP_MARKET",
            "stopPrice": stop_price,
            "closePosition": "true",
            "workingType": "MARK_PRICE",
        })


# ── Test 7: TAKE_PROFIT_MARKET Rejected ──────────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_take_profit_market_rejected(conn):
    """Verify TAKE_PROFIT_MARKET returns -4120 on demo-fapi."""
    mark = conn.get_mark_price(SYMBOL)
    tp_price = conn.round_price(SYMBOL, mark * 1.05)

    from requests import HTTPError
    with pytest.raises(HTTPError, match="-4120"):
        conn._post("/fapi/v1/order", {
            "symbol": SYMBOL,
            "side": "SELL",
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": tp_price,
            "closePosition": "true",
            "workingType": "MARK_PRICE",
        })


# ── Test 8: Full open_long() Flow ────────────────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_full_open_long_flow(conn, executor):
    cleanup_position(conn)
    try:
        mark = conn.get_mark_price(SYMBOL)
        sl_price = conn.round_price(SYMBOL, mark * 0.95)
        tp_price = conn.round_price(SYMBOL, mark * 1.10)

        result = executor.open_long(
            symbol=SYMBOL,
            usdt_amount=150.0,  # slightly above minimum notional
            sl=sl_price,
            tp=tp_price,
            leverage=1,
        )

        assert result["executed"] is True, f"open_long failed: {result.get('error')}"
        assert result["order_id"] is not None
        # TP should have a valid order ID (LIMIT fallback on demo-fapi)
        assert result["tp_order_id"] is not None, "TP order must have a valid orderId"
        # SL will be None on demo-fapi (bot monitoring)
        # That's expected — just verify it's in the result
        assert "sl_order_id" in result

        time.sleep(1)

        # Verify position on exchange
        pos = conn.get_position(SYMBOL)
        assert pos is not None, "Position should exist on exchange"
        assert float(pos["positionAmt"]) > 0

        # Verify TP order on exchange
        open_orders = conn.get_open_orders(SYMBOL)
        tp_ids = [o["orderId"] for o in open_orders]
        assert result["tp_order_id"] in tp_ids, "TP order should be on exchange"
    finally:
        cleanup_position(conn)


# ── Test 9: Full open_short() Flow ───────────────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_full_open_short_flow(conn, executor):
    cleanup_position(conn)
    try:
        mark = conn.get_mark_price(SYMBOL)
        sl_price = conn.round_price(SYMBOL, mark * 1.05)
        tp_price = conn.round_price(SYMBOL, mark * 0.90)

        result = executor.open_short(
            symbol=SYMBOL,
            usdt_amount=150.0,
            sl=sl_price,
            tp=tp_price,
            leverage=1,
        )

        assert result["executed"] is True, f"open_short failed: {result.get('error')}"
        assert result["order_id"] is not None
        assert result["tp_order_id"] is not None, "TP order must have a valid orderId"
        assert "sl_order_id" in result

        time.sleep(1)

        pos = conn.get_position(SYMBOL)
        assert pos is not None
        assert float(pos["positionAmt"]) < 0

        open_orders = conn.get_open_orders(SYMBOL)
        tp_ids = [o["orderId"] for o in open_orders]
        assert result["tp_order_id"] in tp_ids
    finally:
        cleanup_position(conn)


# ── Test 10: TP Failure Auto-Closes Position ─────────────────────────────

@pytest.mark.integration
@skip_no_creds
def test_tp_failure_closes_position(conn, executor):
    """If TP placement fails, the position should be auto-closed."""
    cleanup_position(conn)
    try:
        mark = conn.get_mark_price(SYMBOL)
        sl_price = conn.round_price(SYMBOL, mark * 0.95)
        tp_price = conn.round_price(SYMBOL, mark * 1.10)

        # Mock place_take_profit_order to raise an exception
        with patch.object(
            conn, "place_take_profit_order",
            side_effect=Exception("Simulated TP failure"),
        ):
            result = executor.open_long(
                symbol=SYMBOL,
                usdt_amount=150.0,
                sl=sl_price,
                tp=tp_price,
                leverage=1,
            )

        # Should report failure
        assert result["executed"] is False
        assert "TP placement failed" in (result.get("error") or "")

        time.sleep(1)

        # Position should be closed (auto-closed by the executor)
        pos = conn.get_position(SYMBOL)
        assert pos is None, f"Position should be closed but found: {pos}"
    finally:
        cleanup_position(conn)
