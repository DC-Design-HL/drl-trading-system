"""
Health Checker — Periodically checks connectivity to all external services.

Runs in a background thread and updates MetricsCollector health status.
"""

import logging
import os
import threading
import time
from typing import Optional

import requests

from .collector import get_collector

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 60  # Check every 60 seconds


def _check_binance_rest() -> tuple:
    """Ping Binance Futures REST API."""
    url = os.getenv("BINANCE_FUTURES_BASE_URL", "https://demo-fapi.binance.com")
    t0 = time.time()
    try:
        resp = requests.get(f"{url}/fapi/v1/ping", timeout=10)
        latency = (time.time() - t0) * 1000
        if resp.ok:
            return "healthy", latency, ""
        return "degraded", latency, f"HTTP {resp.status_code}"
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return "down", latency, str(exc)


def _check_binance_ws() -> tuple:
    """Check if Binance WebSocket stream is reachable."""
    t0 = time.time()
    try:
        # Quick TCP connect test to the WS endpoint
        import socket
        sock = socket.create_connection(("stream.binance.com", 9443), timeout=10)
        sock.close()
        latency = (time.time() - t0) * 1000
        return "healthy", latency, ""
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return "down", latency, str(exc)


def _check_binance_spot() -> tuple:
    """Ping Binance Spot API (used for OHLCV data)."""
    base_url = os.getenv("BINANCE_FUTURES_URL", "https://data-api.binance.vision")
    t0 = time.time()
    try:
        resp = requests.get(f"{base_url}/api/v3/ping", timeout=10)
        latency = (time.time() - t0) * 1000
        if resp.ok:
            return "healthy", latency, ""
        return "degraded", latency, f"HTTP {resp.status_code}"
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return "down", latency, str(exc)


def _check_mongodb() -> tuple:
    """Check MongoDB Atlas connectivity."""
    mongo_uri = os.getenv("MONGO_URI", "")
    if not mongo_uri:
        return "unknown", 0, "MONGO_URI not set"
    t0 = time.time()
    try:
        import certifi
        from pymongo import MongoClient
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=10000,
            tlsCAFile=certifi.where(),
        )
        client.admin.command("ping")
        latency = (time.time() - t0) * 1000
        client.close()
        return "healthy", latency, ""
    except ImportError:
        # certifi not available — try without explicit CA
        try:
            from pymongo import MongoClient
            client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=10000,
                tlsAllowInvalidCertificates=True,
            )
            client.admin.command("ping")
            latency = (time.time() - t0) * 1000
            client.close()
            return "healthy", latency, ""
        except Exception as exc:
            latency = (time.time() - t0) * 1000
            return "down", latency, str(exc)
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return "down", latency, str(exc)


def _check_cloudflare_tunnel() -> tuple:
    """Check Cloudflare tunnel connectivity."""
    t0 = time.time()
    try:
        import subprocess
        result = subprocess.run(
            ["systemctl", "is-active", "drl-trading-tunnel.service"],
            capture_output=True, text=True, timeout=5,
        )
        latency = (time.time() - t0) * 1000
        if result.stdout.strip() == "active":
            return "healthy", latency, ""
        return "down", latency, f"Service status: {result.stdout.strip()}"
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return "down", latency, str(exc)


def _check_local_api_server() -> tuple:
    """Check our local Flask API server."""
    t0 = time.time()
    try:
        resp = requests.get("http://127.0.0.1:5001/api/ping", timeout=10)
        latency = (time.time() - t0) * 1000
        if resp.ok:
            return "healthy", latency, ""
        return "degraded", latency, f"HTTP {resp.status_code}"
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return "down", latency, str(exc)


def _check_trading_bots() -> dict:
    """Check trading bot services."""
    import subprocess
    bots = {
        "htf-btc": "drl-htf-agent.service",
        "htf-eth": "drl-htf-eth.service",
        "htf-partial": "drl-htf-partial.service",
        "htf-hybrid": "drl-htf-hybrid.service",
    }
    results = {}
    for name, service in bots.items():
        t0 = time.time()
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service],
                capture_output=True, text=True, timeout=5,
            )
            latency = (time.time() - t0) * 1000
            status = "healthy" if result.stdout.strip() == "active" else "down"
            results[name] = (status, latency, "")
        except Exception as exc:
            results[name] = ("down", 0, str(exc))
    return results


def run_health_checks() -> dict:
    """Run all health checks and update the collector. Returns results."""
    collector = get_collector()
    results = {}

    # External services
    checks = {
        "Binance REST (Futures)": _check_binance_rest,
        "Binance WebSocket": _check_binance_ws,
        "Binance Spot (OHLCV)": _check_binance_spot,
        "MongoDB Atlas": _check_mongodb,
        "Cloudflare Tunnel": _check_cloudflare_tunnel,
        "API Server (local)": _check_local_api_server,
    }

    for name, check_fn in checks.items():
        try:
            status, latency, error = check_fn()
            collector.update_service_health(name, status, latency, error)
            results[name] = {"status": status, "latency_ms": round(latency, 1), "error": error}
        except Exception as exc:
            collector.update_service_health(name, "down", 0, str(exc))
            results[name] = {"status": "down", "error": str(exc)}

    # Trading bots
    bot_results = _check_trading_bots()
    for name, (status, latency, error) in bot_results.items():
        full_name = f"Bot: {name}"
        collector.update_service_health(full_name, status, latency, error)
        results[full_name] = {"status": status, "latency_ms": round(latency, 1), "error": error}

    return results


def start_health_checker() -> threading.Thread:
    """Start the background health checker thread."""

    def _loop():
        logger.info("Health checker started (interval=%ds)", CHECK_INTERVAL)
        while True:
            try:
                results = run_health_checks()
                # Count healthy vs down
                healthy = sum(1 for r in results.values() if r.get("status") == "healthy")
                total = len(results)
                down = [name for name, r in results.items() if r.get("status") == "down"]
                if down:
                    logger.warning(
                        "Health check: %d/%d healthy, DOWN: %s",
                        healthy, total, ", ".join(down),
                    )
                else:
                    logger.debug("Health check: %d/%d healthy", healthy, total)
            except Exception as exc:
                logger.error("Health checker error: %s", exc)

            time.sleep(CHECK_INTERVAL)

    t = threading.Thread(target=_loop, name="health-checker", daemon=True)
    t.start()
    return t
