"""State-machine tests for scripts/bridge_watchdog.py.

We test the pure decide_transition() function so no subprocess /
network / Telegram calls happen during the test.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bridge_watchdog import decide_transition  # noqa: E402

UTC = timezone.utc
NOW = datetime(2026, 5, 22, 16, 0, 0, tzinfo=UTC)
THRESHOLD = 600  # 10 min


# ─────────────────────────────────────────────────────────────────────────
# unknown -> ...
# ─────────────────────────────────────────────────────────────────────────


def test_unknown_to_up_is_silent() -> None:
    """First observation healthy should NOT fire an alert."""
    state, alert = decide_transition(
        {"status": "unknown"},
        healthy=True, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "up"
    assert alert is None


def test_unknown_to_down_starts_pending_no_alert() -> None:
    """First observation degraded enters down_pending; threshold timer
    starts now; no alert until the clock crosses threshold."""
    state, alert = decide_transition(
        {"status": "unknown"},
        healthy=False, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "down_pending"
    assert state["down_since"] == NOW.isoformat()
    assert alert is None


# ─────────────────────────────────────────────────────────────────────────
# up -> ...
# ─────────────────────────────────────────────────────────────────────────


def test_up_to_up_no_alert() -> None:
    state, alert = decide_transition(
        {"status": "up", "since": NOW.isoformat()},
        healthy=True, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "up"
    assert alert is None


def test_up_to_down_starts_pending_no_alert() -> None:
    """The IMPORTANT case for fixing the noise: going down doesn't
    alert immediately. Only after the threshold."""
    state, alert = decide_transition(
        {"status": "up", "since": NOW.isoformat()},
        healthy=False, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "down_pending"
    assert state["down_since"] == NOW.isoformat()
    assert alert is None


# ─────────────────────────────────────────────────────────────────────────
# down_pending -> ...
# ─────────────────────────────────────────────────────────────────────────


def test_down_pending_to_up_silent_recovery() -> None:
    """Normal idle cycle: bridge went down briefly, came back before
    threshold. Must NOT alert (this is the source of the previous noise)."""
    state, alert = decide_transition(
        {"status": "down_pending", "down_since": (NOW - timedelta(seconds=120)).isoformat()},
        healthy=True, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "up"
    assert alert is None
    assert state["down_since"] is None  # clock reset


def test_down_pending_under_threshold_stays_pending() -> None:
    """Still down, but hasn't been long enough yet. No alert; clock
    keeps running; state remains down_pending."""
    state, alert = decide_transition(
        {"status": "down_pending", "down_since": (NOW - timedelta(seconds=300)).isoformat()},
        healthy=False, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "down_pending"
    assert alert is None


def test_down_pending_crosses_threshold_fires_down_alert() -> None:
    """Down >= threshold → transition to down_alerted, fire DOWN alert."""
    state, alert = decide_transition(
        {"status": "down_pending", "down_since": (NOW - timedelta(seconds=THRESHOLD + 5)).isoformat()},
        healthy=False, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "down_alerted"
    assert alert == "down"


def test_down_pending_at_exact_threshold_fires() -> None:
    """Boundary: elapsed == threshold should fire (>=, not >)."""
    state, alert = decide_transition(
        {"status": "down_pending", "down_since": (NOW - timedelta(seconds=THRESHOLD)).isoformat()},
        healthy=False, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "down_alerted"
    assert alert == "down"


# ─────────────────────────────────────────────────────────────────────────
# down_alerted -> ...
# ─────────────────────────────────────────────────────────────────────────


def test_down_alerted_to_up_fires_recovery() -> None:
    """Recovery only sends the BACK UP alert if we'd sent a DOWN alert."""
    state, alert = decide_transition(
        {
            "status": "down_alerted",
            "down_since": (NOW - timedelta(minutes=30)).isoformat(),
        },
        healthy=True, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "up"
    assert alert == "up"
    assert state["down_since"] is None


def test_down_alerted_stays_no_alert() -> None:
    """Still down past threshold, already alerted — refresh only."""
    state, alert = decide_transition(
        {
            "status": "down_alerted",
            "down_since": (NOW - timedelta(minutes=30)).isoformat(),
        },
        healthy=False, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "down_alerted"
    assert alert is None


# ─────────────────────────────────────────────────────────────────────────
# Cross-cutting
# ─────────────────────────────────────────────────────────────────────────


def test_last_check_updated_every_call() -> None:
    """Every transition (or non-transition) refreshes last_check so an
    external monitor can verify the watchdog itself is running."""
    state, _ = decide_transition(
        {"status": "up", "since": NOW.isoformat(), "last_check": "1970-01-01T00:00:00+00:00"},
        healthy=True, now=NOW, down_threshold_seconds=THRESHOLD,
    )
    assert state["last_check"] == NOW.isoformat()


def test_full_idle_cycle_silent_end_to_end() -> None:
    """Simulate the typical noisy scenario we're fixing: up → brief
    down → up, all under threshold. Should produce ZERO alerts.

    This is the test that proves the noise-reduction goal."""
    t0 = NOW
    state = {"status": "up", "since": t0.isoformat()}

    # Tick 1: bridge dropped (idle gap starts)
    state, alert1 = decide_transition(
        state, healthy=False, now=t0 + timedelta(seconds=60),
        down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "down_pending"
    assert alert1 is None

    # Tick 2: still down (3 min in — under threshold)
    state, alert2 = decide_transition(
        state, healthy=False, now=t0 + timedelta(seconds=180),
        down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "down_pending"
    assert alert2 is None

    # Tick 3: bridge came back (5 min in — well under threshold)
    state, alert3 = decide_transition(
        state, healthy=True, now=t0 + timedelta(seconds=300),
        down_threshold_seconds=THRESHOLD,
    )
    assert state["status"] == "up"
    assert alert3 is None

    # No alerts across the full idle cycle — the goal.


def test_real_outage_pings_then_recovers() -> None:
    """The other side of the threshold: a genuine outage past 10 min
    should fire DOWN exactly once, then BACK UP when it recovers."""
    t0 = NOW
    state = {"status": "up", "since": t0.isoformat()}
    alerts: list[str | None] = []

    # Down at t=60s
    state, a = decide_transition(
        state, healthy=False, now=t0 + timedelta(seconds=60),
        down_threshold_seconds=THRESHOLD,
    )
    alerts.append(a)

    # Still down at t=5 min — under threshold
    state, a = decide_transition(
        state, healthy=False, now=t0 + timedelta(seconds=300),
        down_threshold_seconds=THRESHOLD,
    )
    alerts.append(a)

    # Still down at t=11 min — over threshold, expect DOWN alert
    state, a = decide_transition(
        state, healthy=False, now=t0 + timedelta(seconds=660),
        down_threshold_seconds=THRESHOLD,
    )
    alerts.append(a)
    assert a == "down"
    assert state["status"] == "down_alerted"

    # Still down at t=15 min — already alerted, no new alert
    state, a = decide_transition(
        state, healthy=False, now=t0 + timedelta(seconds=900),
        down_threshold_seconds=THRESHOLD,
    )
    alerts.append(a)

    # Recovers at t=20 min — expect BACK UP alert
    state, a = decide_transition(
        state, healthy=True, now=t0 + timedelta(seconds=1200),
        down_threshold_seconds=THRESHOLD,
    )
    alerts.append(a)
    assert a == "up"
    assert state["status"] == "up"

    assert alerts == [None, None, "down", None, "up"]
