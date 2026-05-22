"""Migration tests — verify schema creation against a throwaway DB."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.self_improve.migrate import migrate, verify
from src.self_improve.schema import ALL_TABLES


def test_migrate_creates_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    with sqlite3.connect(str(db)) as conn:
        migrate(conn)
        status = verify(conn)
    assert status == {t: True for t in ALL_TABLES}


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    """Running migrate twice should not raise — CREATE … IF NOT EXISTS."""
    db = tmp_path / "test.db"
    with sqlite3.connect(str(db)) as conn:
        migrate(conn)
        migrate(conn)  # must not raise
        status = verify(conn)
    assert all(status.values())


def test_decisions_fk_to_experiments(tmp_path: Path) -> None:
    """A decisions row with a missing experiment_id should be rejected."""
    db = tmp_path / "test.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        migrate(conn)
        # Insert an experiment, then a decision referencing it — OK.
        conn.execute(
            "INSERT INTO experiments(ts_created, proposal, stage) "
            "VALUES (?, ?, ?)",
            ("2026-05-22T00:00:00Z", "test", "proposed"),
        )
        exp_id = conn.execute(
            "SELECT id FROM experiments WHERE proposal='test'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO decisions(ts, agent, decision_type, summary, "
            "rationale, experiment_id) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "2026-05-22T00:00:01Z",
                "test-agent",
                "config_change",
                "test summary",
                "test rationale",
                exp_id,
            ),
        )

        # Now try with a bogus experiment_id — FK should reject.
        try:
            conn.execute(
                "INSERT INTO decisions(ts, agent, decision_type, summary, "
                "rationale, experiment_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "2026-05-22T00:00:02Z",
                    "test-agent",
                    "config_change",
                    "bad",
                    "bad",
                    99999,
                ),
            )
            raised = False
        except sqlite3.IntegrityError:
            raised = True
        assert raised, "FK constraint did not reject bogus experiment_id"


def test_dry_run_does_not_touch_db(tmp_path: Path) -> None:
    """--dry-run should not create any tables."""
    db = tmp_path / "test.db"
    with sqlite3.connect(str(db)) as conn:
        ran = migrate(conn, dry_run=True)
        assert ran, "Dry run should return the statements it would run"
        status = verify(conn)
    assert all(v is False for v in status.values()), (
        "Dry run created tables — should not have"
    )


def test_indexes_created(tmp_path: Path) -> None:
    """Spot-check that the named indexes were created."""
    db = tmp_path / "test.db"
    with sqlite3.connect(str(db)) as conn:
        migrate(conn)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
    names = {r[0] for r in rows}
    for expected in (
        "idx_decisions_ts",
        "idx_decisions_outcome",
        "idx_metrics_ts_window",
        "idx_metrics_symbol",
        "idx_experiments_stage",
        "idx_agent_runs_ts",
        "idx_agent_runs_agent",
    ):
        assert expected in names, f"missing index: {expected}"
