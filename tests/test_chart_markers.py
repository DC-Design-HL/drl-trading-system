"""Tests for TradingView chart marker generation.

Verifies that the Live Chart produces correct entry and exit markers
from trade data returned by /api/testnet/trades.

Key invariants tested:
  - EXIT markers are generated for CLOSE_LONG / CLOSE_SHORT trades
  - All marker times match an existing candle time (snap-to-candle)
  - Markers are sorted ascending by time (lightweight-charts requirement)
  - Marker shapes/colors match the expected type
  - Per-candle deduplication produces at most 2 markers per candle
"""
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bisect
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

API_BASE = os.environ.get('API_URL', 'http://127.0.0.1:5001')


def _fetch_trades(symbol: str = 'BTCUSDT', limit: int = 100) -> list:
    """Fetch trades from the live API (skip if unreachable)."""
    import requests
    try:
        resp = requests.get(
            f'{API_BASE}/api/testnet/trades',
            params={'symbol': symbol, 'limit': limit},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get('trades', [])
    except Exception:
        pytest.skip('API server not reachable')


def _fetch_candles(symbol: str = 'BTCUSDT', interval: str = '1h', limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV candle data from the live API."""
    import requests
    try:
        resp = requests.get(
            f'{API_BASE}/api/ohlcv',
            params={'symbol': symbol, 'interval': interval, 'limit': limit},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df['time'], unit='s')
        df.index.name = None
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        pytest.skip('API server not reachable')


def _generate_html(df: pd.DataFrame, trades: list) -> str:
    """Generate chart HTML using the real function."""
    from src.ui.app import create_tradingview_chart_with_websocket
    return create_tradingview_chart_with_websocket(df, trades, '1h', 'BTC/USDT')


def _extract_markers(html: str) -> list:
    """Parse the markers JSON from the rendered HTML."""
    match = re.search(r'const markers = (\[.*?\]);\s*\n', html, re.DOTALL)
    assert match, 'Could not find markers JSON in HTML output'
    return json.loads(match.group(1))


def _extract_candle_times(html: str) -> set:
    """Parse candle times from the rendered HTML."""
    match = re.search(r'let candleData = (\[.*?\]);\s*\n', html, re.DOTALL)
    assert match, 'Could not find candleData JSON in HTML output'
    candles = json.loads(match.group(1))
    return {c['time'] for c in candles}


# ---------------------------------------------------------------------------
# Synthetic test data (for unit tests that don't need the live API)
# ---------------------------------------------------------------------------

def _make_synthetic_trades():
    """Create a minimal set of trades with known OPEN and CLOSE actions."""
    base_ts = 1700000000
    return [
        {'price': '100', 'timestamp': '2023-11-14T22:13:20+00:00', 'action': 'OPEN_LONG', 'symbol': 'BTCUSDT'},
        {'price': '105', 'timestamp': '2023-11-14T23:13:20+00:00', 'action': 'CLOSE_LONG', 'symbol': 'BTCUSDT', 'realizedPnl': '5'},
        {'price': '110', 'timestamp': '2023-11-15T00:13:20+00:00', 'action': 'OPEN_SHORT', 'symbol': 'BTCUSDT'},
        {'price': '108', 'timestamp': '2023-11-15T01:13:20+00:00', 'action': 'CLOSE_SHORT', 'symbol': 'BTCUSDT', 'realizedPnl': '2'},
        # Two trades in same hour (should get deduplicated)
        {'price': '112', 'timestamp': '2023-11-15T02:00:05+00:00', 'action': 'OPEN_LONG', 'symbol': 'BTCUSDT'},
        {'price': '111', 'timestamp': '2023-11-15T02:00:30+00:00', 'action': 'CLOSE_LONG', 'symbol': 'BTCUSDT', 'realizedPnl': '-1'},
    ]


def _make_synthetic_df():
    """Create a minimal DataFrame covering the synthetic trade times."""
    times = list(range(1699995600, 1700013600, 3600))  # 5 candles, 1h apart
    data = []
    for t in times:
        data.append({
            'time': t,
            'open': 100.0,
            'high': 120.0,
            'low': 95.0,
            'close': 110.0,
            'volume': 1000.0,
        })
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df['time'], unit='s')
    df.index.name = None
    return df[['open', 'high', 'low', 'close', 'volume']]


# ---------------------------------------------------------------------------
# Tests using synthetic data (always run)
# ---------------------------------------------------------------------------

class TestMarkerGenerationSynthetic:
    """Unit tests using synthetic data — no API dependency."""

    def test_exit_markers_generated(self):
        """CLOSE trades produce EXIT markers in the HTML output."""
        df = _make_synthetic_df()
        trades = _make_synthetic_trades()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)

        exit_markers = [m for m in markers if 'EXIT' in m.get('text', '')]
        assert len(exit_markers) >= 2, (
            f'Expected at least 2 exit markers, got {len(exit_markers)}: '
            f'{[m["text"] for m in markers]}'
        )

    def test_entry_markers_generated(self):
        """OPEN trades produce LONG/SHORT markers."""
        df = _make_synthetic_df()
        trades = _make_synthetic_trades()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)

        entry_markers = [m for m in markers if m.get('text', '').startswith(('LONG', 'SHORT', 'L×', 'S×'))]
        assert len(entry_markers) >= 1, f'No entry markers found: {markers}'

    def test_markers_sorted_ascending(self):
        """All markers are sorted by time ascending (lightweight-charts requirement)."""
        df = _make_synthetic_df()
        trades = _make_synthetic_trades()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)

        times = [m['time'] for m in markers]
        assert times == sorted(times), f'Markers not sorted: {times}'

    def test_markers_snapped_to_candles(self):
        """All marker times match an existing candle time."""
        df = _make_synthetic_df()
        trades = _make_synthetic_trades()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)
        candle_times = _extract_candle_times(html)

        for m in markers:
            assert m['time'] in candle_times, (
                f'Marker time {m["time"]} (text={m["text"]}) not in candle data'
            )

    def test_marker_shapes_and_colors(self):
        """Each marker type has correct shape and color."""
        df = _make_synthetic_df()
        trades = _make_synthetic_trades()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)

        for m in markers:
            text = m['text']
            if text.startswith('LONG') or text.startswith('L×'):
                assert m['shape'] == 'arrowUp', f'LONG marker has wrong shape: {m}'
                assert m['color'] == '#26a69a', f'LONG marker has wrong color: {m}'
                assert m['position'] == 'belowBar', f'LONG marker has wrong position: {m}'
            elif text.startswith('SHORT') or text.startswith('S×'):
                assert m['shape'] == 'arrowDown', f'SHORT marker has wrong shape: {m}'
                assert m['color'] == '#ef5350', f'SHORT marker has wrong color: {m}'
                assert m['position'] == 'aboveBar', f'SHORT marker has wrong position: {m}'
            elif 'EXIT(SL)' in text or 'SL' in text:
                assert m['shape'] == 'square', f'EXIT(SL) marker has wrong shape: {m}'
                assert m['color'] == '#ff4444', f'EXIT(SL) marker has wrong color: {m}'
            elif 'EXIT(TP)' in text or 'TP' in text:
                assert m['shape'] == 'square', f'EXIT(TP) marker has wrong shape: {m}'
                assert m['color'] == '#00e676', f'EXIT(TP) marker has wrong color: {m}'
            elif 'EXIT' in text:
                assert m['shape'] == 'square', f'EXIT marker has wrong shape: {m}'

    def test_deduplication_limits_markers_per_candle(self):
        """After deduplication, at most 2 markers per candle time (1 entry + 1 exit)."""
        df = _make_synthetic_df()
        trades = _make_synthetic_trades()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)

        time_counts = Counter(m['time'] for m in markers)
        for t, count in time_counts.items():
            assert count <= 2, (
                f'Candle time {t} has {count} markers (max should be 2): '
                f'{[m for m in markers if m["time"] == t]}'
            )

    def test_no_internal_fields_leaked(self):
        """Internal fields like _kind must not appear in the final markers."""
        df = _make_synthetic_df()
        trades = _make_synthetic_trades()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)

        for m in markers:
            assert '_kind' not in m, f'Internal _kind field leaked: {m}'

    def test_none_reason_does_not_crash(self):
        """Trades with reason=None should not throw AttributeError."""
        df = _make_synthetic_df()
        trades = [
            {'price': '100', 'timestamp': '2023-11-14T22:13:20+00:00', 'action': 'OPEN_LONG', 'reason': None},
            {'price': '105', 'timestamp': '2023-11-14T23:13:20+00:00', 'action': 'CLOSE_LONG', 'reason': None, 'realizedPnl': '-1'},
        ]
        html = _generate_html(df, trades)
        markers = _extract_markers(html)
        exit_markers = [m for m in markers if 'EXIT' in m.get('text', '')]
        assert len(exit_markers) >= 1, f'EXIT marker missing when reason=None: {markers}'

    def test_empty_trades_no_markers(self):
        """No trades → no markers."""
        df = _make_synthetic_df()
        html = _generate_html(df, [])
        markers = _extract_markers(html)
        assert markers == [], f'Expected no markers, got {markers}'

    def test_empty_dataframe_returns_placeholder(self):
        """Empty DataFrame should return a placeholder div, not crash."""
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        df.index = pd.DatetimeIndex([])
        html = _generate_html(df, [])
        assert 'No market data' in html


# ---------------------------------------------------------------------------
# Tests using live API data (skipped if API unreachable)
# ---------------------------------------------------------------------------

class TestMarkerGenerationLive:
    """Integration tests hitting the real API."""

    def test_trade_data_has_close_actions(self):
        """API returns trades with CLOSE_LONG or CLOSE_SHORT actions."""
        trades = _fetch_trades()
        close_trades = [t for t in trades if 'CLOSE' in t.get('action', '')]
        assert len(close_trades) > 0, (
            f'No CLOSE trades in API response! Actions found: '
            f'{set(t.get("action", "") for t in trades)}'
        )

    def test_exit_markers_in_rendered_html(self):
        """Full pipeline: API → chart HTML → exit markers present."""
        trades = _fetch_trades()
        df = _fetch_candles()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)

        exit_markers = [m for m in markers if 'EXIT' in m.get('text', '')]
        assert len(exit_markers) > 0, (
            f'No EXIT markers in rendered HTML! '
            f'Markers: {[m["text"] for m in markers]}'
        )

    def test_live_markers_sorted(self):
        """Markers from real data are sorted ascending."""
        trades = _fetch_trades()
        df = _fetch_candles()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)

        times = [m['time'] for m in markers]
        assert times == sorted(times)

    def test_live_markers_snapped(self):
        """Markers from real data snap to valid candle times."""
        trades = _fetch_trades()
        df = _fetch_candles()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)
        candle_times = _extract_candle_times(html)

        for m in markers:
            assert m['time'] in candle_times

    def test_live_deduplication(self):
        """Live markers have at most 2 per candle."""
        trades = _fetch_trades()
        df = _fetch_candles()
        html = _generate_html(df, trades)
        markers = _extract_markers(html)

        time_counts = Counter(m['time'] for m in markers)
        for t, count in time_counts.items():
            assert count <= 2, f'Candle {t} has {count} markers'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
