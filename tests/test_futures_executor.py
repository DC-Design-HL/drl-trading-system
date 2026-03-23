"""
Tests for FuturesTestnetExecutor — 100% method coverage with mocked connector.
"""

from unittest.mock import MagicMock, patch, call

import pytest

from src.api.futures_executor import FuturesTestnetExecutor, get_futures_executor
from src.api.binance_futures import BinanceFuturesConnector


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_executor():
    """Return (executor, mock_connector)."""
    connector = MagicMock(spec=BinanceFuturesConnector)
    connector.get_mark_price.return_value = 84000.0
    connector.get_qty_precision.return_value = 3
    connector.get_price_precision.return_value = 1
    connector.set_leverage.return_value = {"leverage": 1}
    connector.place_market_order.return_value = {"orderId": 101, "status": "FILLED"}
    connector.place_stop_loss_order.return_value = {"orderId": 201}
    connector.place_take_profit_order.return_value = {"orderId": 301}
    connector.cancel_order.return_value = {"orderId": 201, "status": "CANCELED"}
    connector.get_account.return_value = {
        "totalWalletBalance": "5000.0",
        "availableBalance": "4000.0",
        "totalUnrealizedProfit": "50.0",
        "totalMarginBalance": "5050.0",
        "totalPositionInitialMargin": "1000.0",
    }
    connector.get_positions.return_value = []
    connector.get_open_orders.return_value = []
    connector.get_open_algo_orders.return_value = []
    connector.get_trade_history.return_value = []
    executor = FuturesTestnetExecutor(connector=connector)
    return executor, connector


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_accepts_connector_directly(self):
        connector = MagicMock(spec=BinanceFuturesConnector)
        ex = FuturesTestnetExecutor(connector=connector)
        assert ex.connector is connector

    def test_creates_connector_from_env(self, monkeypatch):
        monkeypatch.setenv("BINANCE_FUTURES_API_KEY", "key123")
        monkeypatch.setenv("BINANCE_FUTURES_API_SECRET", "secret456")
        with patch("src.api.futures_executor.BinanceFuturesConnector") as MockConn:
            MockConn.return_value = MagicMock()
            ex = FuturesTestnetExecutor()
        MockConn.assert_called_once_with(api_key="key123", api_secret="secret456")

    def test_raises_without_env_keys(self, monkeypatch):
        monkeypatch.delenv("BINANCE_FUTURES_API_KEY", raising=False)
        monkeypatch.delenv("BINANCE_FUTURES_API_SECRET", raising=False)
        with pytest.raises(ValueError, match="not set"):
            FuturesTestnetExecutor()

    def test_initial_order_tracking_empty(self):
        ex, _ = make_executor()
        assert ex._sl_orders == {}
        assert ex._tp_orders == {}


# ── open_long ─────────────────────────────────────────────────────────────────

class TestOpenLong:
    def test_success(self):
        ex, conn = make_executor()
        result = ex.open_long("BTCUSDT", usdt_amount=1000, sl=80000, tp=90000)

        assert result["executed"] is True
        assert result["side"] == "LONG"
        assert result["order_id"] == 101
        assert result["sl_order_id"] == 201
        assert result["tp_order_id"] == 301
        assert result["error"] is None

        conn.set_leverage.assert_called_once_with("BTCUSDT", 1)
        conn.place_market_order.assert_called_once()
        conn.place_stop_loss_order.assert_called_once()
        conn.place_take_profit_order.assert_called_once()

    def test_sl_tp_stored(self):
        ex, conn = make_executor()
        ex.open_long("BTCUSDT", 1000, sl=80000, tp=90000)
        assert ex._sl_orders["BTCUSDT"] == 201
        assert ex._tp_orders["BTCUSDT"] == 301

    def test_sl_sentinel_none_not_stored(self):
        """When place_stop_loss_order returns orderId=None (demo-fapi), don't store None."""
        ex, conn = make_executor()
        conn.place_stop_loss_order.return_value = {
            "orderId": None, "status": "TESTNET_NOT_SUPPORTED"
        }
        result = ex.open_long("BTCUSDT", 1000, sl=80000, tp=90000)
        assert result["executed"] is True
        assert result["sl_order_id"] is None
        assert "BTCUSDT" not in ex._sl_orders  # None should not be stored

    def test_no_sl_skips_sl_order(self):
        ex, conn = make_executor()
        result = ex.open_long("BTCUSDT", 1000, sl=0, tp=90000)
        conn.place_stop_loss_order.assert_not_called()
        assert result["sl_order_id"] is None

    def test_no_tp_skips_tp_order(self):
        ex, conn = make_executor()
        result = ex.open_long("BTCUSDT", 1000, sl=80000, tp=0)
        conn.place_take_profit_order.assert_not_called()
        assert result["tp_order_id"] is None

    def test_zero_mark_price_returns_error(self):
        ex, conn = make_executor()
        conn.get_mark_price.return_value = 0.0
        result = ex.open_long("BTCUSDT", 1000, sl=80000, tp=90000)
        assert result["executed"] is False
        assert "Invalid mark price" in result["error"]

    def test_zero_quantity_returns_error(self):
        ex, conn = make_executor()
        conn.get_mark_price.return_value = 84000.0
        conn.get_qty_precision.return_value = 0
        result = ex.open_long("BTCUSDT", usdt_amount=0.001, sl=80000, tp=90000)
        assert result["executed"] is False

    def test_sl_failure_logged_but_continues(self):
        ex, conn = make_executor()
        conn.place_stop_loss_order.side_effect = Exception("SL failed")
        result = ex.open_long("BTCUSDT", 1000, sl=80000, tp=90000)
        assert result["executed"] is True
        assert "sl_error" in result
        assert ex._sl_orders.get("BTCUSDT") is None

    def test_tp_failure_closes_position(self):
        ex, conn = make_executor()
        conn.place_take_profit_order.side_effect = Exception("TP failed")
        result = ex.open_long("BTCUSDT", 1000, sl=80000, tp=90000)
        assert result["executed"] is False
        assert "TP placement failed" in result["error"]
        # Should have placed a closing MARKET SELL
        calls = conn.place_market_order.call_args_list
        assert len(calls) == 2  # open BUY + emergency close SELL
        assert calls[1][0][1] == "SELL"

    def test_market_order_failure_returns_error(self):
        ex, conn = make_executor()
        conn.place_market_order.side_effect = Exception("exchange down")
        result = ex.open_long("BTCUSDT", 1000, sl=80000, tp=90000)
        assert result["executed"] is False
        assert "exchange down" in result["error"]

    def test_custom_leverage(self):
        ex, conn = make_executor()
        ex.open_long("BTCUSDT", 1000, sl=80000, tp=90000, leverage=5)
        conn.set_leverage.assert_called_once_with("BTCUSDT", 5)

    def test_slash_symbol_normalised(self):
        ex, conn = make_executor()
        ex.open_long("BTC/USDT", 1000, sl=80000, tp=90000)
        conn.get_mark_price.assert_called_with("BTCUSDT")


# ── open_short ────────────────────────────────────────────────────────────────

class TestOpenShort:
    def test_success(self):
        ex, conn = make_executor()
        result = ex.open_short("BTCUSDT", usdt_amount=1000, sl=90000, tp=75000)

        assert result["executed"] is True
        assert result["side"] == "SHORT"
        conn.place_stop_loss_order.assert_called_once_with(
            "BTCUSDT", "BUY", 90000, close_position=True
        )
        conn.place_take_profit_order.assert_called_once_with(
            "BTCUSDT", "BUY", 75000, close_position=True
        )
        conn.place_market_order.assert_called_once_with("BTCUSDT", "SELL", pytest.approx(0.012, abs=0.001))

    def test_sl_tp_stored(self):
        ex, conn = make_executor()
        ex.open_short("BTCUSDT", 1000, sl=90000, tp=75000)
        assert ex._sl_orders["BTCUSDT"] == 201
        assert ex._tp_orders["BTCUSDT"] == 301

    def test_sl_sentinel_none_not_stored(self):
        """When place_stop_loss_order returns sentinel on demo-fapi, don't store None."""
        ex, conn = make_executor()
        conn.place_stop_loss_order.return_value = {
            "orderId": None, "status": "TESTNET_NOT_SUPPORTED"
        }
        result = ex.open_short("BTCUSDT", 1000, sl=90000, tp=75000)
        assert result["executed"] is True
        assert result["sl_order_id"] is None
        assert "BTCUSDT" not in ex._sl_orders

    def test_zero_mark_price_returns_error(self):
        ex, conn = make_executor()
        conn.get_mark_price.return_value = 0.0
        result = ex.open_short("BTCUSDT", 1000, sl=90000, tp=75000)
        assert result["executed"] is False

    def test_sl_failure_continues(self):
        ex, conn = make_executor()
        conn.place_stop_loss_order.side_effect = Exception("SL failed")
        result = ex.open_short("BTCUSDT", 1000, sl=90000, tp=75000)
        assert result["executed"] is True
        assert "sl_error" in result

    def test_market_order_failure_returns_error(self):
        ex, conn = make_executor()
        conn.place_market_order.side_effect = Exception("no liquidity")
        result = ex.open_short("BTCUSDT", 1000, sl=90000, tp=75000)
        assert result["executed"] is False


# ── update_sl ─────────────────────────────────────────────────────────────────

class TestUpdateSL:
    def test_long_cancels_old_places_new(self):
        ex, conn = make_executor()
        ex._sl_orders["BTCUSDT"] = 201
        result = ex.update_sl("BTCUSDT", "LONG", 81000.0)

        assert result is True
        conn.cancel_order.assert_called_once_with("BTCUSDT", 201, is_algo=False)
        conn.place_stop_loss_order.assert_called_once_with(
            "BTCUSDT", "SELL", 81000.0, close_position=True
        )

    def test_short_uses_buy_side(self):
        ex, conn = make_executor()
        ex.update_sl("BTCUSDT", "SHORT", 88000.0)
        conn.place_stop_loss_order.assert_called_once_with(
            "BTCUSDT", "BUY", 88000.0, close_position=True
        )

    def test_no_existing_sl_still_places_new(self):
        ex, conn = make_executor()
        result = ex.update_sl("BTCUSDT", "LONG", 80000.0)
        conn.cancel_order.assert_not_called()
        assert result is True

    def test_cancel_failure_still_places_new(self):
        ex, conn = make_executor()
        ex._sl_orders["BTCUSDT"] = 201
        conn.cancel_order.side_effect = Exception("already cancelled")
        result = ex.update_sl("BTCUSDT", "LONG", 82000.0)
        conn.place_stop_loss_order.assert_called_once()
        assert result is True

    def test_placement_failure_returns_false(self):
        ex, conn = make_executor()
        conn.place_stop_loss_order.side_effect = Exception("order rejected")
        result = ex.update_sl("BTCUSDT", "LONG", 80000.0)
        assert result is False

    def test_sl_sentinel_update_returns_true_no_store(self):
        """update_sl with sentinel (demo-fapi) returns True but doesn't store None."""
        ex, conn = make_executor()
        conn.place_stop_loss_order.return_value = {
            "orderId": None, "status": "TESTNET_NOT_SUPPORTED"
        }
        result = ex.update_sl("BTCUSDT", "LONG", 82000.0)
        assert result is True
        assert "BTCUSDT" not in ex._sl_orders


# ── update_tp ─────────────────────────────────────────────────────────────────

class TestUpdateTP:
    def test_long_cancels_old_places_new(self):
        ex, conn = make_executor()
        ex._tp_orders["BTCUSDT"] = 301
        result = ex.update_tp("BTCUSDT", "LONG", 92000.0)

        assert result is True
        conn.cancel_order.assert_called_once_with("BTCUSDT", 301, is_algo=False)
        conn.place_take_profit_order.assert_called_once_with(
            "BTCUSDT", "SELL", 92000.0, close_position=True
        )

    def test_short_uses_buy_side(self):
        ex, conn = make_executor()
        ex.update_tp("BTCUSDT", "SHORT", 70000.0)
        conn.place_take_profit_order.assert_called_once_with(
            "BTCUSDT", "BUY", 70000.0, close_position=True
        )

    def test_placement_failure_returns_false(self):
        ex, conn = make_executor()
        conn.place_take_profit_order.side_effect = Exception("rejected")
        result = ex.update_tp("BTCUSDT", "LONG", 90000.0)
        assert result is False


# ── sync_positions ────────────────────────────────────────────────────────────

class TestSyncPositions:
    def test_clears_stale_order_tracking(self):
        ex, conn = make_executor()
        ex._sl_orders["XRPUSDT"] = 999
        ex._tp_orders["XRPUSDT"] = 888
        conn.get_positions.return_value = [{"symbol": "BTCUSDT", "positionAmt": "0.05"}]
        result = ex.sync_positions()

        assert len(result) == 1
        assert "XRPUSDT" not in ex._sl_orders
        assert "XRPUSDT" not in ex._tp_orders

    def test_keeps_active_symbol_orders(self):
        ex, conn = make_executor()
        ex._sl_orders["BTCUSDT"] = 201
        conn.get_positions.return_value = [{"symbol": "BTCUSDT", "positionAmt": "0.05"}]
        ex.sync_positions()
        assert "BTCUSDT" in ex._sl_orders


# ── get_portfolio ─────────────────────────────────────────────────────────────

class TestGetPortfolio:
    def test_returns_exchange_reported_values(self):
        ex, conn = make_executor()
        result = ex.get_portfolio()

        assert result["total_wallet_balance"] == 5000.0
        assert result["available_balance"] == 4000.0
        assert result["total_unrealized_profit"] == 50.0
        assert result["total_margin_balance"] == 5050.0
        assert result["total_position_initial_margin"] == 1000.0

    def test_missing_fields_default_to_zero(self):
        ex, conn = make_executor()
        conn.get_account.return_value = {}
        result = ex.get_portfolio()
        assert result["total_wallet_balance"] == 0.0


# ── get_positions ─────────────────────────────────────────────────────────────

class TestGetPositions:
    def test_normalises_long_position(self):
        ex, conn = make_executor()
        conn.get_positions.return_value = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.050",
                "entryPrice": "83000.0",
                "markPrice": "84000.0",
                "unRealizedProfit": "50.0",
                "liquidationPrice": "60000.0",
                "leverage": "5",
            }
        ]
        result = ex.get_positions()

        assert len(result) == 1
        p = result[0]
        assert p["symbol"] == "BTCUSDT"
        assert p["side"] == "LONG"
        assert p["amount"] == 0.05
        assert p["entry_price"] == 83000.0
        assert p["current_price"] == 84000.0
        assert p["unrealized_pnl"] == 50.0
        assert p["liquidation_price"] == 60000.0
        assert p["leverage"] == 5
        assert p["simulated"] is False

    def test_normalises_short_position(self):
        ex, conn = make_executor()
        conn.get_positions.return_value = [
            {
                "symbol": "ETHUSDT",
                "positionAmt": "-1.000",
                "entryPrice": "3000.0",
                "markPrice": "2950.0",
                "unRealizedProfit": "50.0",
                "liquidationPrice": "5000.0",
                "leverage": "2",
            }
        ]
        result = ex.get_positions()

        assert result[0]["side"] == "SHORT"
        assert result[0]["amount"] == 1.0

    def test_includes_sl_tp_order_ids(self):
        ex, conn = make_executor()
        ex._sl_orders["BTCUSDT"] = 201
        ex._tp_orders["BTCUSDT"] = 301
        conn.get_positions.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.05", "entryPrice": "83000",
             "markPrice": "84000", "unRealizedProfit": "50", "liquidationPrice": "60000",
             "leverage": "1"}
        ]
        result = ex.get_positions()

        assert result[0]["sl_order_id"] == 201
        assert result[0]["tp_order_id"] == 301


# ── get_trade_history ─────────────────────────────────────────────────────────

class TestGetTradeHistory:
    def test_single_symbol(self):
        ex, conn = make_executor()
        trades = [{"id": 1, "symbol": "BTCUSDT", "realizedPnl": "5.0"}]
        conn.get_trade_history.return_value = trades
        result = ex.get_trade_history("BTCUSDT", limit=100)
        assert result == trades
        conn.get_trade_history.assert_called_once_with("BTCUSDT", limit=100)

    def test_no_symbol_uses_tracked_symbols(self):
        ex, conn = make_executor()
        ex._sl_orders["BTCUSDT"] = 201
        ex._tp_orders["ETHUSDT"] = 302
        conn.get_trade_history.return_value = []
        ex.get_trade_history()
        called_symbols = {c[0][0] for c in conn.get_trade_history.call_args_list}
        assert called_symbols == {"BTCUSDT", "ETHUSDT"}

    def test_no_symbol_and_no_tracked_falls_back_to_defaults(self):
        ex, conn = make_executor()
        conn.get_trade_history.return_value = []
        ex.get_trade_history()
        called_symbols = {c[0][0] for c in conn.get_trade_history.call_args_list}
        assert "BTCUSDT" in called_symbols

    def test_results_sorted_by_time(self):
        ex, conn = make_executor()
        ex._sl_orders["BTCUSDT"] = 1
        ex._sl_orders["ETHUSDT"] = 2
        conn.get_trade_history.side_effect = [
            [{"time": 200, "symbol": "BTCUSDT", "realizedPnl": "1"}],
            [{"time": 100, "symbol": "ETHUSDT", "realizedPnl": "2"}],
        ]
        result = ex.get_trade_history()
        assert result[0]["time"] == 100
        assert result[1]["time"] == 200

    def test_symbol_fetch_failure_skipped(self):
        ex, conn = make_executor()
        ex._sl_orders["BTCUSDT"] = 1
        ex._sl_orders["BROKEN"] = 2
        def side_effect(sym, limit):
            if sym == "BROKEN":
                raise Exception("bad symbol")
            return [{"time": 100, "realizedPnl": "1"}]
        conn.get_trade_history.side_effect = side_effect
        result = ex.get_trade_history()
        assert len(result) == 1


# ── get_pnl_summary ───────────────────────────────────────────────────────────

class TestGetPnlSummary:
    def test_all_from_exchange(self):
        ex, conn = make_executor()
        # Use explicit symbol so only one symbol is queried (avoids default 2-symbol doubling)
        conn.get_trade_history.return_value = [
            {"time": 1, "symbol": "BTCUSDT", "realizedPnl": "100.0"},
            {"time": 2, "symbol": "BTCUSDT", "realizedPnl": "-30.0"},
            {"time": 3, "symbol": "BTCUSDT", "realizedPnl": "0.0"},
        ]
        result = ex.get_pnl_summary(symbol="BTCUSDT")

        assert result["realized_pnl"] == pytest.approx(70.0)
        assert result["unrealized_pnl"] == pytest.approx(50.0)
        assert result["total_pnl"] == pytest.approx(120.0)
        assert result["total_trades"] == 3
        assert result["closed_trades"] == 2
        assert result["winning_trades"] == 1
        assert result["win_rate"] == pytest.approx(0.5)

    def test_equity_curve_cumulative(self):
        ex, conn = make_executor()
        conn.get_trade_history.return_value = [
            {"time": 1, "symbol": "BTCUSDT", "realizedPnl": "50.0"},
            {"time": 2, "symbol": "BTCUSDT", "realizedPnl": "25.0"},
        ]
        result = ex.get_pnl_summary(symbol="BTCUSDT")
        curve = result["equity_curve"]
        assert len(curve) == 2
        assert curve[0]["cumulative_pnl"] == 50.0
        assert curve[1]["cumulative_pnl"] == 75.0

    def test_account_fetch_failure_uses_zero_unrealized(self):
        ex, conn = make_executor()
        conn.get_account.side_effect = Exception("connection error")
        conn.get_trade_history.return_value = []
        result = ex.get_pnl_summary(symbol="BTCUSDT")
        assert result["unrealized_pnl"] == 0.0

    def test_zero_trades_safe(self):
        ex, conn = make_executor()
        conn.get_trade_history.return_value = []
        result = ex.get_pnl_summary(symbol="BTCUSDT")
        assert result["win_rate"] == 0.0
        assert result["equity_curve"] == []

    def test_single_symbol_filter(self):
        ex, conn = make_executor()
        conn.get_trade_history.return_value = [
            {"time": 1, "symbol": "BTCUSDT", "realizedPnl": "10.0"}
        ]
        result = ex.get_pnl_summary(symbol="BTCUSDT")
        conn.get_trade_history.assert_called_once_with("BTCUSDT", limit=1000)
        assert result["realized_pnl"] == 10.0


# ── _sync_order_tracking ──────────────────────────────────────────────────────

class TestSyncOrderTracking:
    def test_loads_limit_reduconly_as_tp(self):
        ex, conn = make_executor()
        conn.get_open_orders.return_value = [
            {"symbol": "BTCUSDT", "orderId": 999, "type": "LIMIT", "reduceOnly": True},
        ]
        ex._sync_order_tracking()
        assert ex._tp_orders["BTCUSDT"] == 999
        assert "BTCUSDT" not in ex._sl_orders

    def test_loads_stop_market_closepositon_as_sl(self):
        ex, conn = make_executor()
        conn.get_open_orders.return_value = [
            {"symbol": "ETHUSDT", "orderId": 555, "type": "STOP_MARKET", "reduceOnly": False, "closePosition": True},
        ]
        ex._sync_order_tracking()
        assert ex._sl_orders["ETHUSDT"] == 555
        assert "ETHUSDT" not in ex._tp_orders

    def test_loads_stop_market_reduconly_as_sl(self):
        ex, conn = make_executor()
        conn.get_open_orders.return_value = [
            {"symbol": "BTCUSDT", "orderId": 777, "type": "STOP_MARKET", "reduceOnly": True, "closePosition": False},
        ]
        ex._sync_order_tracking()
        assert ex._sl_orders["BTCUSDT"] == 777

    def test_ignores_non_reduconly_limit_orders(self):
        ex, conn = make_executor()
        conn.get_open_orders.return_value = [
            {"symbol": "BTCUSDT", "orderId": 888, "type": "LIMIT", "reduceOnly": False},
        ]
        ex._sync_order_tracking()
        assert "BTCUSDT" not in ex._tp_orders

    def test_exchange_failure_does_not_raise(self):
        ex, conn = make_executor()
        conn.get_open_orders.side_effect = Exception("network error")
        # Should not raise
        ex._sync_order_tracking()
        assert ex._sl_orders == {}
        assert ex._tp_orders == {}

    def test_called_on_construction(self):
        connector = MagicMock(spec=__import__('src.api.binance_futures', fromlist=['BinanceFuturesConnector']).BinanceFuturesConnector)
        connector.get_open_orders.return_value = [
            {"symbol": "BTCUSDT", "orderId": 12345, "type": "LIMIT", "reduceOnly": True},
        ]
        connector.get_open_algo_orders.return_value = []
        ex = FuturesTestnetExecutor(connector=connector)
        assert ex._tp_orders["BTCUSDT"] == 12345


# ── ensure_tp_order ───────────────────────────────────────────────────────────

class TestEnsureTpOrder:
    def test_noop_when_already_tracked(self):
        ex, conn = make_executor()
        ex._tp_orders["BTCUSDT"] = 300
        result = ex.ensure_tp_order("BTCUSDT", "LONG", 90000)
        conn.place_take_profit_order.assert_not_called()
        assert result == 300

    def test_places_tp_when_missing(self):
        ex, conn = make_executor()
        conn.place_take_profit_order.return_value = {"orderId": 400}
        result = ex.ensure_tp_order("BTCUSDT", "LONG", 90000)
        conn.place_take_profit_order.assert_called_once_with(
            "BTCUSDT", "SELL", 90000, close_position=True
        )
        assert result == 400
        assert ex._tp_orders["BTCUSDT"] == 400

    def test_short_uses_buy_side(self):
        ex, conn = make_executor()
        conn.place_take_profit_order.return_value = {"orderId": 401}
        ex.ensure_tp_order("ETHUSDT", "SHORT", 1800)
        conn.place_take_profit_order.assert_called_once_with(
            "ETHUSDT", "BUY", 1800, close_position=True
        )

    def test_exchange_failure_returns_none(self):
        ex, conn = make_executor()
        conn.place_take_profit_order.side_effect = Exception("API error")
        result = ex.ensure_tp_order("BTCUSDT", "LONG", 90000)
        assert result is None
        assert "BTCUSDT" not in ex._tp_orders

    def test_sentinel_none_not_stored(self):
        ex, conn = make_executor()
        conn.place_take_profit_order.return_value = {"orderId": None, "status": "TESTNET_NOT_SUPPORTED"}
        result = ex.ensure_tp_order("BTCUSDT", "LONG", 90000)
        assert result is None
        assert "BTCUSDT" not in ex._tp_orders


# ── Factory function ──────────────────────────────────────────────────────────

class TestFactory:
    def test_returns_executor_when_keys_set(self, monkeypatch):
        monkeypatch.setenv("BINANCE_FUTURES_API_KEY", "k")
        monkeypatch.setenv("BINANCE_FUTURES_API_SECRET", "s")
        with patch("src.api.futures_executor.BinanceFuturesConnector"):
            ex = get_futures_executor()
        assert ex is not None

    def test_returns_none_when_keys_missing(self, monkeypatch):
        monkeypatch.delenv("BINANCE_FUTURES_API_KEY", raising=False)
        monkeypatch.delenv("BINANCE_FUTURES_API_SECRET", raising=False)
        ex = get_futures_executor()
        assert ex is None
