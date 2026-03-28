"""
Metrics Collector — Lightweight in-memory metrics for the trading system.

Tracks:
  - API call counts (inbound/outbound) per endpoint
  - Error counts by type
  - Service health status
  - Trade statistics

Uses a thread-safe in-memory store with periodic MongoDB persistence.
No Prometheus/Grafana needed — served via our existing Flask API.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class APICallMetric:
    """Tracks API calls for a specific endpoint."""
    total_calls: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    last_call_time: float = 0.0
    last_error: str = ""
    last_error_time: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0
        return self.total_latency_ms / self.total_calls

    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0
        return self.error_count / self.total_calls


@dataclass
class ErrorMetric:
    """Tracks a specific error type."""
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    sample_message: str = ""


@dataclass
class ServiceHealth:
    """Health status for a service/integration."""
    name: str = ""
    status: str = "unknown"  # "healthy", "degraded", "down", "unknown"
    last_check: float = 0.0
    last_success: float = 0.0
    last_error: str = ""
    response_time_ms: float = 0.0
    consecutive_failures: int = 0


class MetricsCollector:
    """
    Thread-safe metrics collector for the trading system.
    Singleton pattern — use get_collector() to access.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._api_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._health_lock = threading.Lock()
        self._trade_lock = threading.Lock()

        # API call tracking: {direction: {endpoint: APICallMetric}}
        # direction = "outbound" (us → Binance/MongoDB) or "inbound" (UI → our API)
        self._api_calls: Dict[str, Dict[str, APICallMetric]] = {
            "outbound": defaultdict(APICallMetric),
            "inbound": defaultdict(APICallMetric),
        }

        # Error tracking: {error_type: ErrorMetric}
        self._errors: Dict[str, ErrorMetric] = defaultdict(ErrorMetric)

        # Service health: {service_name: ServiceHealth}
        self._services: Dict[str, ServiceHealth] = {}

        # Trade stats (rolling window)
        self._trade_stats = {
            "testnet": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "trades_today": 0,
                "pnl_today": 0.0,
                "recent_trades": [],  # Last 50 trades
            },
            "paper": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "trades_today": 0,
                "pnl_today": 0.0,
                "recent_trades": [],
            },
        }

        # Startup time
        self._start_time = time.time()

        # Rolling window for rate calculations (per-minute buckets)
        self._minute_buckets: Dict[str, Dict[int, int]] = {
            "outbound": defaultdict(int),
            "inbound": defaultdict(int),
        }

    # ── API Call Tracking ─────────────────────────────────────────────

    def record_api_call(
        self,
        direction: str,
        endpoint: str,
        success: bool = True,
        latency_ms: float = 0.0,
        error: str = "",
    ) -> None:
        """
        Record an API call.

        direction: "outbound" (us → external) or "inbound" (external → us)
        endpoint: e.g. "binance/fapi/v1/order", "mongo/trades", "api/status"
        """
        now = time.time()
        minute_bucket = int(now // 60)

        with self._api_lock:
            m = self._api_calls[direction][endpoint]
            m.total_calls += 1
            m.total_latency_ms += latency_ms
            m.last_call_time = now

            if success:
                m.success_count += 1
            else:
                m.error_count += 1
                m.last_error = error
                m.last_error_time = now

            # Update per-minute bucket for rate calculation
            self._minute_buckets[direction][minute_bucket] += 1

    def get_api_stats(self) -> Dict:
        """Get all API call statistics."""
        with self._api_lock:
            result = {}
            for direction in ["outbound", "inbound"]:
                result[direction] = {}
                for endpoint, metric in self._api_calls[direction].items():
                    result[direction][endpoint] = {
                        "total_calls": metric.total_calls,
                        "success": metric.success_count,
                        "errors": metric.error_count,
                        "error_rate": f"{metric.error_rate:.1%}",
                        "avg_latency_ms": round(metric.avg_latency_ms, 1),
                        "last_call": datetime.fromtimestamp(metric.last_call_time).isoformat()
                        if metric.last_call_time > 0 else None,
                        "last_error": metric.last_error or None,
                    }

            # Calculate rates (calls per minute, last 5 minutes)
            now_minute = int(time.time() // 60)
            for direction in ["outbound", "inbound"]:
                recent_total = sum(
                    self._minute_buckets[direction].get(now_minute - i, 0)
                    for i in range(5)
                )
                result[f"{direction}_rate_per_min"] = round(recent_total / 5, 1)

            return result

    # ── Error Tracking ────────────────────────────────────────────────

    def record_error(self, error_type: str, message: str = "") -> None:
        """Record an error occurrence."""
        now = time.time()
        with self._error_lock:
            e = self._errors[error_type]
            e.count += 1
            if e.first_seen == 0:
                e.first_seen = now
            e.last_seen = now
            if message:
                e.sample_message = message[:200]  # Truncate long messages

    def get_error_stats(self) -> Dict:
        """Get error statistics."""
        with self._error_lock:
            result = {}
            for error_type, metric in sorted(
                self._errors.items(), key=lambda x: x[1].count, reverse=True
            ):
                result[error_type] = {
                    "count": metric.count,
                    "first_seen": datetime.fromtimestamp(metric.first_seen).isoformat()
                    if metric.first_seen > 0 else None,
                    "last_seen": datetime.fromtimestamp(metric.last_seen).isoformat()
                    if metric.last_seen > 0 else None,
                    "sample": metric.sample_message or None,
                }
            return result

    # ── Service Health ────────────────────────────────────────────────

    def update_service_health(
        self,
        name: str,
        status: str,
        response_time_ms: float = 0.0,
        error: str = "",
    ) -> None:
        """Update health status for a service."""
        now = time.time()
        with self._health_lock:
            if name not in self._services:
                self._services[name] = ServiceHealth(name=name)
            s = self._services[name]
            s.status = status
            s.last_check = now
            s.response_time_ms = response_time_ms
            if status == "healthy":
                s.last_success = now
                s.consecutive_failures = 0
            else:
                s.consecutive_failures += 1
                if error:
                    s.last_error = error

    def get_health_stats(self) -> Dict:
        """Get health status for all services."""
        with self._health_lock:
            result = {}
            for name, s in self._services.items():
                result[name] = {
                    "status": s.status,
                    "response_time_ms": round(s.response_time_ms, 1),
                    "last_check": datetime.fromtimestamp(s.last_check).isoformat()
                    if s.last_check > 0 else None,
                    "last_success": datetime.fromtimestamp(s.last_success).isoformat()
                    if s.last_success > 0 else None,
                    "last_error": s.last_error or None,
                    "consecutive_failures": s.consecutive_failures,
                }
            return result

    # ── Trade Statistics ──────────────────────────────────────────────

    def record_trade(
        self,
        mode: str,  # "testnet" or "paper"
        symbol: str,
        action: str,
        pnl: float,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        direction: str = "",
        reason: str = "",
    ) -> None:
        """Record a completed trade."""
        now = time.time()
        today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()

        with self._trade_lock:
            stats = self._trade_stats.get(mode, self._trade_stats["paper"])
            stats["total_trades"] += 1
            stats["total_pnl"] += pnl

            if pnl > 0:
                stats["wins"] += 1
            elif pnl < 0:
                stats["losses"] += 1

            if pnl > stats["best_trade"]:
                stats["best_trade"] = pnl
            if pnl < stats["worst_trade"]:
                stats["worst_trade"] = pnl

            # Today's stats
            stats["trades_today"] += 1
            stats["pnl_today"] += pnl

            # Recent trades (keep last 50)
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": action,
                "direction": direction,
                "pnl": round(pnl, 2),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "reason": reason,
            }
            stats["recent_trades"].append(trade_record)
            if len(stats["recent_trades"]) > 50:
                stats["recent_trades"] = stats["recent_trades"][-50:]

    def get_trade_stats(self) -> Dict:
        """Get trade statistics for all modes."""
        with self._trade_lock:
            result = {}
            for mode in ["testnet", "paper"]:
                stats = self._trade_stats[mode]
                total = stats["total_trades"]
                wins = stats["wins"]
                result[mode] = {
                    "total_trades": total,
                    "wins": wins,
                    "losses": stats["losses"],
                    "win_rate": f"{(wins/total*100):.1f}%" if total > 0 else "N/A",
                    "total_pnl": round(stats["total_pnl"], 2),
                    "best_trade": round(stats["best_trade"], 2),
                    "worst_trade": round(stats["worst_trade"], 2),
                    "avg_pnl": round(stats["total_pnl"] / total, 2) if total > 0 else 0,
                    "trades_today": stats["trades_today"],
                    "pnl_today": round(stats["pnl_today"], 2),
                    "recent_trades": stats["recent_trades"][-10:],  # Last 10 for API
                }
            return result

    # ── System Overview ───────────────────────────────────────────────

    def get_overview(self) -> Dict:
        """Get a complete system overview."""
        uptime_seconds = time.time() - self._start_time
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)

        return {
            "uptime": f"{hours}h {minutes}m",
            "uptime_seconds": round(uptime_seconds),
            "api": self.get_api_stats(),
            "errors": self.get_error_stats(),
            "health": self.get_health_stats(),
            "trades": self.get_trade_stats(),
        }


def get_collector() -> MetricsCollector:
    """Get the singleton MetricsCollector instance."""
    return MetricsCollector()
