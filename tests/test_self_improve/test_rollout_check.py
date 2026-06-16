"""Tests for the durable P3-rollout health-check (scripts/self_improve/rollout_check.py).

Focus on the two pieces with real logic: the fresh/non-benign error filter
(so the 6h monitor never cries wolf on the known recurring whale-model load
failure) and the build_report messaging policy (alert / morning heartbeat /
silent).
"""

from __future__ import annotations

from datetime import datetime

import scripts.self_improve.rollout_check as rc


_NOW = datetime(2026, 6, 16, 6, 40, 0)

_LOG = """\
2026-06-16 06:18:12,566 [src.whale_behavior.models.predictor] ERROR - Failed to load whale behavior model: state_dict
2026-06-15 21:34:33,343 [src.api.futures_executor] ERROR - open_short failed for SOLUSDT: Binance Futures API 400: -2019
2026-06-16 06:30:00,000 [htf_live] ERROR - genuinely broke in the loop
2026-06-16 00:00:00,000 [htf_live] ERROR - old error outside the 6.5h window
2026-06-16 06:31:00,000 [orchestrator] CIRCUIT BREAKER tripped exp #7
"""


def _write_log(tmp_path):
    p = tmp_path / "bots_live.log"
    p.write_text(_LOG)
    return p


def test_recent_errors_filters_benign_and_old(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "_BOTS_LOG", _write_log(tmp_path))
    hits = rc._recent_errors(now=_NOW)
    # Keeps: the genuine loop error + the circuit-breaker line.
    # Drops: whale-model load (benign), Binance 4xx (benign), old (>6.5h).
    assert len(hits) == 2
    joined = "\n".join(hits)
    assert "genuinely broke" in joined
    assert "CIRCUIT BREAKER" in joined
    assert "whale behavior" not in joined
    assert "Binance Futures API" not in joined
    assert "old error" not in joined


def test_recent_errors_empty_when_only_benign(tmp_path, monkeypatch):
    p = tmp_path / "bots_live.log"
    p.write_text(
        "2026-06-16 06:18:12,566 [predictor] ERROR - Failed to load whale behavior model: x\n"
        "2026-06-16 06:20:00,000 [futures_executor] ERROR - open_short failed: Binance Futures API 400: -2019\n"
    )
    monkeypatch.setattr(rc, "_BOTS_LOG", p)
    assert rc._recent_errors(now=_NOW) == []


def _patch_clean(monkeypatch, *, services=None, loop=None, override=None,
                 errors=None, killed=False, armed=True, prev=None):
    monkeypatch.setattr(rc, "_check_services", lambda: services or [])
    monkeypatch.setattr(
        rc, "_loop_state",
        lambda: loop or {"experiments": {}, "last_decision": None})
    monkeypatch.setattr(rc, "_override_info", lambda: override)
    monkeypatch.setattr(rc, "_recent_errors", lambda *a, **k: errors or [])
    monkeypatch.setattr(rc, "_load_prev_state", lambda: prev or {})

    # Kill/armed flags are read via Path.exists — swap the module attrs for
    # fakes with a controllable .exists().
    class _Flag:
        def __init__(self, present): self._p = present
        def exists(self): return self._p
    monkeypatch.setattr(rc, "_KILL", _Flag(killed))
    monkeypatch.setattr(rc, "_ARMED", _Flag(armed))


def test_build_report_silent_when_all_nominal_offpeak(monkeypatch):
    _patch_clean(monkeypatch)
    monkeypatch.setattr(rc, "datetime", _frozen(2026, 6, 16, 14, 0))  # 14:00
    msg, _ = rc.build_report()
    assert msg is None


def test_build_report_heartbeat_on_morning(monkeypatch):
    _patch_clean(monkeypatch)
    monkeypatch.setattr(rc, "datetime", _frozen(2026, 6, 16, 8, 5))
    msg, _ = rc.build_report()
    assert msg and "all nominal" in msg and "armed" in msg


def test_build_report_alerts_on_stage_change(monkeypatch):
    _patch_clean(
        monkeypatch,
        loop={"experiments": {"7": "paper"}, "last_decision": None},
        prev={"experiments": {}, "override": None},
    )
    monkeypatch.setattr(rc, "datetime", _frozen(2026, 6, 16, 14, 0))
    msg, _ = rc.build_report()
    assert msg and "ATTENTION" in msg and "#7→paper" in msg


def test_build_report_alerts_on_live_override(monkeypatch):
    _patch_clean(
        monkeypatch,
        loop={"experiments": {"7": "canary"}, "last_decision": None},
        override='exp #7 → {"TRAILING_DISTANCE_PCT": 0.008}',
        prev={"experiments": {"7": "canary"}, "override": None},
    )
    monkeypatch.setattr(rc, "datetime", _frozen(2026, 6, 16, 14, 0))
    msg, _ = rc.build_report()
    assert msg and "LIVE OVERRIDE" in msg and "TRAILING_DISTANCE_PCT" in msg


def test_build_report_alerts_on_dead_service(monkeypatch):
    _patch_clean(monkeypatch, services=["bots(pid=123)"])
    monkeypatch.setattr(rc, "datetime", _frozen(2026, 6, 16, 8, 5))  # even at heartbeat slot
    msg, _ = rc.build_report()
    assert msg and "SERVICES DOWN" in msg


def _frozen(*args):
    """A datetime subclass whose .now() returns a fixed instant."""
    fixed = datetime(*args)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    return _DT
