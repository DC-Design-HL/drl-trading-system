"""
Regression test for the 2026-04-24 CWD bug.

Cron/watchdog invokes services with CWD=$HOME, and our launcher script
(`start_services.sh`) plus each long-running Python entry point must
self-anchor to the repo so relative paths (`data/models`, `data/trading.db`,
`logs/...`) resolve correctly.

Failure modes this guards against:
  * Bots silently running in HOLD-only mode because `find_best_htf_model`
    looks at a non-existent `/home/claude/data/models`.
  * Alerter / news services writing state files to a shadow tree under
    /home/claude, so the dashboard / rest of the stack never sees the data.

These tests are text-level (no child process spawn) so they stay fast and
do not touch the running production cluster.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_start_services_cd_anchor() -> None:
    """start_services.sh must `cd "$REPO"` before any setsid child spawn."""
    content = (REPO / "start_services.sh").read_text()
    assert 'cd "$REPO"' in content, (
        "start_services.sh is missing the `cd \"$REPO\"` anchor — cron-triggered "
        "restarts will run with CWD=$HOME and every child will write to /home/claude/. "
        "See scripts/merge_cwd_shadow_trades.py for the incident context."
    )
    # And it must happen BEFORE the first actual setsid spawn line.
    # Walk the file line by line so comments mentioning "setsid" don't fool us.
    lines = content.splitlines()
    cd_line = next(i for i, l in enumerate(lines) if 'cd "$REPO"' in l and not l.lstrip().startswith("#"))
    setsid_line = next(i for i, l in enumerate(lines) if l.lstrip().startswith("setsid"))
    assert cd_line < setsid_line, (
        f"`cd \"$REPO\"` (line {cd_line + 1}) must appear before the first setsid spawn "
        f"(line {setsid_line + 1}) — children inherit the wrong CWD otherwise."
    )


def test_restart_ui_cd_anchor() -> None:
    """restart_ui.sh must `cd "$REPO"` too (same bug class)."""
    content = (REPO / "restart_ui.sh").read_text()
    assert 'cd "$REPO"' in content


_SELF_ANCHOR_NEEDLE = 'os.chdir(Path(__file__).resolve().parent)'
_SCRIPTS_REQUIRING_SELF_ANCHOR = [
    "live_trading_all.py",
    "trade_alerter.py",
    "start_local_server.py",
    "news_sentinel.py",
    "news_alerter.py",
    "whale_behavior_ws.py",
]


def test_python_entry_points_self_anchor() -> None:
    """Each long-running Python entry point self-anchors CWD as belt-and-suspenders.

    Even if the shell launcher regresses, every script re-anchors itself
    on import so no relative-path operation can escape the repo.
    """
    missing: list[str] = []
    for script in _SCRIPTS_REQUIRING_SELF_ANCHOR:
        text = (REPO / script).read_text()
        if _SELF_ANCHOR_NEEDLE not in text:
            missing.append(script)
    assert not missing, (
        f"These scripts are missing `{_SELF_ANCHOR_NEEDLE}` near the top: "
        f"{missing}. Add it immediately after the imports so relative paths "
        f"always resolve to the repo regardless of caller CWD."
    )
