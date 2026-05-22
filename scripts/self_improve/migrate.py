#!/usr/bin/env python3
"""Migrate data/trading.db to include the self-improvement-plane tables.

Idempotent: re-running adds nothing if the tables already exist. Safe to
run while the live bot is connected — SQLite handles concurrent reads
and the CREATEs are additive (no existing rows touched).

Usage:
    python -m scripts.self_improve.migrate              # apply to data/trading.db
    python -m scripts.self_improve.migrate --dry-run    # print SQL, don't execute
    python -m scripts.self_improve.migrate --db PATH    # custom DB path
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Allow running this script directly without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.schema import (  # noqa: E402
    ALL_TABLES,
    DDL_AGENT_RUNS,
    DDL_DECISIONS,
    DDL_EXPERIMENTS,
    DDL_METRICS_SNAPSHOTS,
    INDEXES,
)

_DDL_IN_ORDER = (
    # experiments must exist before decisions (FK reference)
    DDL_EXPERIMENTS,
    DDL_DECISIONS,
    DDL_METRICS_SNAPSHOTS,
    DDL_AGENT_RUNS,
)


def migrate(conn: sqlite3.Connection, *, dry_run: bool = False) -> list[str]:
    """Apply the self-improvement schema. Returns the list of statements run.

    Idempotent: CREATE TABLE/INDEX IF NOT EXISTS. Tables can be created
    in any order from the user's perspective, but DDL_EXPERIMENTS comes
    first because decisions.experiment_id FK-references it.
    """
    statements = [*(_DDL_IN_ORDER), *INDEXES]
    if dry_run:
        return statements
    cur = conn.cursor()
    for sql in statements:
        cur.execute(sql)
    conn.commit()
    return statements


def verify(conn: sqlite3.Connection) -> dict[str, bool]:
    """Return {table_name: exists} for every table we expect."""
    cur = conn.cursor()
    existing = {
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    return {t: t in existing for t in ALL_TABLES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="data/trading.db", help="SQLite DB path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SQL that would be run, but don't execute",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists() and not args.dry_run:
        print(f"❌ DB not found at {db_path}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"-- DRY RUN against {db_path} --")
        for sql in migrate(sqlite3.connect(":memory:"), dry_run=True):
            print(sql.strip(), ";\n", sep="")
        return 0

    with sqlite3.connect(str(db_path)) as conn:
        # Foreign-key enforcement is off by default in SQLite; turn it on.
        conn.execute("PRAGMA foreign_keys = ON")
        ran = migrate(conn)
        status = verify(conn)

    print(f"✅ Migration complete against {db_path}")
    print(f"   Statements executed: {len(ran)}")
    for table, exists in status.items():
        mark = "✓" if exists else "✗"
        print(f"   [{mark}] {table}")

    if not all(status.values()):
        print("❌ Some tables missing after migration", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
