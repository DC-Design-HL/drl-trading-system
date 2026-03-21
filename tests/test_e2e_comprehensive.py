"""
DRL Trading System — Comprehensive E2E + Component Tests
Tests API endpoints (live server on :5001), design_system components, and app imports.
"""

import sys
import os
from pathlib import Path

import pytest

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

API_BASE = "http://127.0.0.1:5001"


@pytest.fixture(scope="session")
def api_session():
    """Shared requests session for API tests."""
    import requests
    sess = requests.Session()
    sess.headers["Accept"] = "application/json"
    return sess


@pytest.fixture(scope="session")
def api_available(api_session):
    """Check whether the API server is reachable at all."""
    try:
        resp = api_session.get(f"{API_BASE}/api/state", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
# 1. API Endpoint Tests — all hit the live server
# ═══════════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    """Test every documented API endpoint returns 200 with valid JSON."""

    ENDPOINTS_200 = [
        "/api/state",
        "/api/market",
        "/api/model",
        "/api/testnet/trades",
        "/api/testnet/pnl",
        "/api/htf/status",
        "/api/htf/trades",
        "/api/htf/performance",
    ]

    @pytest.mark.parametrize("endpoint", ENDPOINTS_200)
    def test_endpoint_returns_200(self, api_session, api_available, endpoint):
        if not api_available:
            pytest.skip("API server not reachable on port 5001")
        resp = api_session.get(f"{API_BASE}{endpoint}", timeout=15)
        assert resp.status_code == 200, f"{endpoint} returned {resp.status_code}: {resp.text[:200]}"

    @pytest.mark.parametrize("endpoint", ENDPOINTS_200)
    def test_endpoint_returns_json(self, api_session, api_available, endpoint):
        if not api_available:
            pytest.skip("API server not reachable on port 5001")
        resp = api_session.get(f"{API_BASE}{endpoint}", timeout=15)
        data = resp.json()
        assert isinstance(data, (dict, list)), f"{endpoint} returned non-JSON-dict/list: {type(data)}"

    def test_invalid_endpoint_returns_404(self, api_session, api_available):
        if not api_available:
            pytest.skip("API server not reachable on port 5001")
        resp = api_session.get(f"{API_BASE}/api/this-does-not-exist-12345", timeout=5)
        assert resp.status_code == 404

    # ── Response key checks ──

    def test_state_has_expected_keys(self, api_session, api_available):
        if not api_available:
            pytest.skip("API server not reachable")
        data = api_session.get(f"{API_BASE}/api/state", timeout=10).json()
        for key in ("assets", "total_balance", "available_assets"):
            assert key in data, f"/api/state missing key '{key}'"

    def test_market_has_expected_keys(self, api_session, api_available):
        if not api_available:
            pytest.skip("API server not reachable")
        data = api_session.get(f"{API_BASE}/api/market", timeout=15).json()
        # Should have at least some of these
        found = [k for k in ("price", "symbol", "order_flow", "regime", "funding") if k in data]
        assert len(found) >= 2, f"/api/market only has keys: {list(data.keys())}"

    def test_model_has_expected_keys(self, api_session, api_available):
        if not api_available:
            pytest.skip("API server not reachable")
        data = api_session.get(f"{API_BASE}/api/model", timeout=10).json()
        for key in ("model_exists", "total_trades", "win_rate"):
            assert key in data, f"/api/model missing key '{key}'"

    def test_testnet_trades_has_trades_key(self, api_session, api_available):
        if not api_available:
            pytest.skip("API server not reachable")
        data = api_session.get(f"{API_BASE}/api/testnet/trades", timeout=10).json()
        assert "trades" in data, f"/api/testnet/trades missing 'trades' key"

    def test_testnet_pnl_has_expected_keys(self, api_session, api_available):
        if not api_available:
            pytest.skip("API server not reachable")
        data = api_session.get(f"{API_BASE}/api/testnet/pnl", timeout=10).json()
        for key in ("realized_pnl", "total_pnl", "win_rate"):
            assert key in data, f"/api/testnet/pnl missing key '{key}'"

    def test_htf_status_has_expected_keys(self, api_session, api_available):
        if not api_available:
            pytest.skip("API server not reachable")
        data = api_session.get(f"{API_BASE}/api/htf/status", timeout=10).json()
        for key in ("running", "position", "position_label"):
            assert key in data, f"/api/htf/status missing key '{key}'"

    def test_htf_trades_has_trades_key(self, api_session, api_available):
        if not api_available:
            pytest.skip("API server not reachable")
        data = api_session.get(f"{API_BASE}/api/htf/trades", timeout=10).json()
        assert "trades" in data, f"/api/htf/trades missing 'trades' key"

    def test_htf_performance_has_expected_keys(self, api_session, api_available):
        if not api_available:
            pytest.skip("API server not reachable")
        data = api_session.get(f"{API_BASE}/api/htf/performance", timeout=10).json()
        for key in ("total_trades", "win_rate", "total_pnl"):
            assert key in data, f"/api/htf/performance missing key '{key}'"


# ═══════════════════════════════════════════════════════════════════
# 2. Design System Component Tests
# ═══════════════════════════════════════════════════════════════════

class TestDesignSystemImport:
    """Verify design_system.py imports without error and exposes expected symbols."""

    def test_import_design_system(self):
        from src.ui.design_system import (
            BG_PRIMARY, BG_CARD, ACCENT, SUCCESS, DANGER, WARNING,
            TEXT_PRIMARY, TEXT_MUTED, BORDER,
            metric_card, status_badge, pnl_text, section_header,
            styled_table, card_container, GLOBAL_CSS,
        )
        # All should be non-empty strings
        for name, val in [
            ("BG_PRIMARY", BG_PRIMARY),
            ("BG_CARD", BG_CARD),
            ("ACCENT", ACCENT),
            ("SUCCESS", SUCCESS),
            ("DANGER", DANGER),
            ("WARNING", WARNING),
            ("TEXT_PRIMARY", TEXT_PRIMARY),
            ("TEXT_MUTED", TEXT_MUTED),
            ("BORDER", BORDER),
            ("GLOBAL_CSS", GLOBAL_CSS),
        ]:
            assert isinstance(val, str) and len(val) > 0, f"{name} is empty or not a string"


class TestMetricCard:
    """Test metric_card() with various inputs."""

    def test_normal_values(self):
        from src.ui.design_system import metric_card
        html = metric_card("Balance", "$5,000.00", delta=25.50, icon="💰")
        assert "Balance" in html
        assert "$5,000.00" in html
        assert "💰" in html
        assert "25.50" in html

    def test_none_value(self):
        from src.ui.design_system import metric_card
        html = metric_card("Balance", None)
        assert "—" in html
        assert "Balance" in html

    def test_negative_delta(self):
        from src.ui.design_system import metric_card
        html = metric_card("PnL", "$4,900", delta=-100.0)
        assert "100.00" in html
        # Should contain DANGER color (#EF4444)
        assert "#EF4444" in html or "ef4444" in html.lower()

    def test_zero_delta(self):
        from src.ui.design_system import metric_card
        html = metric_card("Value", "0", delta=0)
        assert "0" in html

    def test_no_delta(self):
        from src.ui.design_system import metric_card
        html = metric_card("Label", "Value")
        assert "Label" in html
        assert "Value" in html

    def test_none_label(self):
        from src.ui.design_system import metric_card
        html = metric_card(None, "test")
        # Should not crash; renders "—" for None label
        assert "—" in html

    def test_numeric_value_formatted(self):
        from src.ui.design_system import metric_card
        html = metric_card("Test", 1234.56)
        # Should contain formatted number
        assert "1,234.56" in html


class TestPnlText:
    """Test pnl_text() with various inputs."""

    def test_positive(self):
        from src.ui.design_system import pnl_text
        html = pnl_text(150.25)
        assert "+$150.25" in html
        assert "#10B981" in html or "10b981" in html.lower()  # SUCCESS color

    def test_negative(self):
        from src.ui.design_system import pnl_text
        html = pnl_text(-42.10)
        assert "$42.10" in html
        assert "#EF4444" in html or "ef4444" in html.lower()  # DANGER color

    def test_zero(self):
        from src.ui.design_system import pnl_text
        html = pnl_text(0)
        assert "$0.00" in html

    def test_none(self):
        from src.ui.design_system import pnl_text
        html = pnl_text(None)
        assert "—" in html

    def test_string_input(self):
        from src.ui.design_system import pnl_text
        # Should handle gracefully
        html = pnl_text("not-a-number")
        assert "—" in html


class TestStatusBadge:
    """Test status_badge() output."""

    def test_normal(self):
        from src.ui.design_system import status_badge, SUCCESS
        html = status_badge("LONG", SUCCESS)
        assert "LONG" in html
        assert SUCCESS.lower() in html.lower()

    def test_empty_text(self):
        from src.ui.design_system import status_badge
        html = status_badge("")
        assert "—" in html

    def test_default_color(self):
        from src.ui.design_system import status_badge, TEXT_MUTED
        html = status_badge("FLAT")
        assert TEXT_MUTED.lower() in html.lower()


class TestSectionHeader:
    """Test section_header() output."""

    def test_with_icon(self):
        from src.ui.design_system import section_header
        html = section_header("Portfolio", "📊")
        assert "Portfolio" in html
        assert "📊" in html

    def test_without_icon(self):
        from src.ui.design_system import section_header
        html = section_header("Trades")
        assert "Trades" in html

    def test_none_title(self):
        from src.ui.design_system import section_header
        html = section_header(None)
        # Should not crash
        assert isinstance(html, str)


class TestStyledTable:
    """Test styled_table() with various row counts."""

    def test_empty_data(self):
        from src.ui.design_system import styled_table
        html = styled_table(["A", "B"], [])
        assert "No data available" in html

    def test_single_row(self):
        from src.ui.design_system import styled_table
        html = styled_table(["Name", "Value"], [["BTC", "$70,000"]])
        assert "BTC" in html
        assert "$70,000" in html

    def test_many_rows(self):
        from src.ui.design_system import styled_table
        rows = [[f"Row{i}", f"Val{i}"] for i in range(20)]
        html = styled_table(["Col1", "Col2"], rows)
        assert "Row0" in html
        assert "Row19" in html

    def test_no_headers(self):
        from src.ui.design_system import styled_table
        html = styled_table([], [[1, 2]])
        assert html == ""

    def test_html_pass_through(self):
        """Cells containing HTML tags should be passed through unescaped."""
        from src.ui.design_system import styled_table
        cell_html = '<span style="color:green;">OK</span>'
        html = styled_table(["Status"], [[cell_html]])
        assert 'color:green' in html

    def test_none_cell(self):
        from src.ui.design_system import styled_table
        html = styled_table(["A"], [[None]])
        assert "—" in html


class TestCardContainer:
    """Test card_container() wrapper."""

    def test_wraps_content(self):
        from src.ui.design_system import card_container
        html = card_container("<p>Hello</p>")
        assert "<p>Hello</p>" in html
        assert "border-radius" in html

    def test_none_content(self):
        from src.ui.design_system import card_container
        html = card_container(None)
        assert isinstance(html, str)
        assert "border-radius" in html


class TestHelperComponents:
    """Test additional helper components."""

    def test_loading_card(self):
        from src.ui.design_system import loading_card
        html = loading_card("Fetching data...")
        assert "Fetching data..." in html
        assert "⏳" in html

    def test_error_card(self):
        from src.ui.design_system import error_card
        html = error_card("Connection failed", "timeout after 5s")
        assert "Connection failed" in html
        assert "timeout after 5s" in html

    def test_position_badge_flat(self):
        from src.ui.design_system import position_badge
        html = position_badge(0)
        assert "FLAT" in html

    def test_position_badge_long(self):
        from src.ui.design_system import position_badge
        html = position_badge(1)
        assert "LONG" in html

    def test_position_badge_short(self):
        from src.ui.design_system import position_badge
        html = position_badge(-1)
        assert "SHORT" in html

    def test_position_badge_none(self):
        from src.ui.design_system import position_badge
        html = position_badge(None)
        assert "FLAT" in html

    def test_metric_row(self):
        from src.ui.design_system import metric_row
        html = metric_row([
            {"label": "A", "value": "100"},
            {"label": "B", "value": "200", "delta": 10},
        ])
        assert "A" in html
        assert "B" in html
        assert "grid" in html

    def test_metric_row_empty(self):
        from src.ui.design_system import metric_row
        html = metric_row([])
        assert html == ""

    def test_progress_bar(self):
        from src.ui.design_system import progress_bar
        html = progress_bar(75, 100)
        assert "75.0%" in html

    def test_progress_bar_zero(self):
        from src.ui.design_system import progress_bar
        html = progress_bar(0, 100)
        assert "0.0%" in html

    def test_progress_bar_over_max(self):
        from src.ui.design_system import progress_bar
        html = progress_bar(200, 100)
        assert "100.0%" in html


# ═══════════════════════════════════════════════════════════════════
# 3. App Import Tests
# ═══════════════════════════════════════════════════════════════════

class TestAppImports:
    """Verify core modules import without syntax errors."""

    def test_design_system_imports(self):
        """design_system.py should import cleanly."""
        import importlib
        mod = importlib.import_module("src.ui.design_system")
        assert hasattr(mod, "metric_card")
        assert hasattr(mod, "GLOBAL_CSS")
        assert hasattr(mod, "styled_table")

    def test_design_system_compiles(self):
        """design_system.py should have zero syntax errors."""
        import py_compile
        py_compile.compile(
            str(PROJECT_ROOT / "src" / "ui" / "design_system.py"),
            doraise=True,
        )

    def test_app_compiles(self):
        """app.py should have zero syntax errors."""
        import py_compile
        py_compile.compile(
            str(PROJECT_ROOT / "src" / "ui" / "app.py"),
            doraise=True,
        )

    def test_this_file_compiles(self):
        """This test file itself should compile cleanly."""
        import py_compile
        py_compile.compile(__file__, doraise=True)


# ═══════════════════════════════════════════════════════════════════
# 4. Color Palette Consistency
# ═══════════════════════════════════════════════════════════════════

class TestColorPalette:
    """Ensure color constants are valid hex codes."""

    @pytest.mark.parametrize("color_name", [
        "BG_PRIMARY", "BG_CARD", "ACCENT", "SUCCESS", "DANGER",
        "WARNING", "TEXT_PRIMARY", "TEXT_MUTED", "BORDER",
    ])
    def test_color_is_valid_hex(self, color_name):
        import re
        from src.ui import design_system as ds
        color = getattr(ds, color_name)
        assert re.match(r'^#[0-9A-Fa-f]{6}$', color), f"{color_name}={color} is not a valid #RRGGBB hex"
