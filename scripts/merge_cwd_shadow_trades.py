#!/usr/bin/env python3
"""
Merge the 2026-04-24 CWD-shadow trades into the repo DB.

Context: from 2026-04-22 20:28 to 2026-04-24 08:50 the bot ran with CWD=$HOME
(watchdog/cron launch), so `data/trading.db` writes landed at
`/home/claude/data/trading.db` instead of the repo tree. This merges those
trades back into the canonical DB.

Safe to re-run: dedupes by (timestamp, symbol, action, price) so repeated
invocations are idempotent.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPO_DB = REPO / "data" / "trading.db"
SHADOW_DB = Path("/home/claude/data/trading.db")


def merge() -> int:
    if not SHADOW_DB.exists():
        print(f"No shadow DB at {SHADOW_DB} — nothing to merge.")
        return 0
    if not REPO_DB.exists():
        print(f"Repo DB missing at {REPO_DB} — aborting.", file=sys.stderr)
        return 2

    shadow = sqlite3.connect(SHADOW_DB)
    shadow.row_factory = sqlite3.Row
    repo = sqlite3.connect(REPO_DB)
    repo.row_factory = sqlite3.Row

    shadow_rows = shadow.execute(
        "SELECT timestamp, symbol, action, data, price, pnl, confidence, reason, created_at, is_testnet "
        "FROM trades ORDER BY id"
    ).fetchall()
    print(f"Shadow DB has {len(shadow_rows)} trade(s) to consider.")

    existing_keys = {
        (r["timestamp"], r["symbol"], r["action"], r["price"])
        for r in repo.execute(
            "SELECT timestamp, symbol, action, price FROM trades WHERE timestamp >= '2026-04-22'"
        ).fetchall()
    }

    to_insert = []
    skipped_dup = 0
    for r in shadow_rows:
        key = (r["timestamp"], r["symbol"], r["action"], r["price"])
        if key in existing_keys:
            skipped_dup += 1
            continue
        data_obj = json.loads(r["data"]) if r["data"] else {}
        data_obj["_reconciled_from"] = "cwd_shadow_20260424"
        to_insert.append(
            (
                r["timestamp"],
                r["symbol"],
                r["action"],
                json.dumps(data_obj),
                r["price"],
                r["pnl"],
                r["confidence"],
                r["reason"],
                r["created_at"],
                r["is_testnet"],
            )
        )

    repo.executemany(
        "INSERT INTO trades (timestamp, symbol, action, data, price, pnl, confidence, reason, created_at, is_testnet) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        to_insert,
    )
    repo.commit()

    print(f"Inserted {len(to_insert)} row(s); skipped {skipped_dup} duplicate(s).")
    shadow.close()
    repo.close()
    return 0


if __name__ == "__main__":
    sys.exit(merge())
