"""Test the decisions exporter against a synthetic decisions table."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.self_improve.export_decisions import export, render
from scripts.self_improve.migrate import migrate


def _seed(db: Path) -> None:
    with sqlite3.connect(str(db)) as conn:
        migrate(conn)
        conn.execute(
            "INSERT INTO experiments(ts_created, proposal, stage, branch) "
            "VALUES (?, ?, ?, ?)",
            ("2026-05-22T00:00:00Z", "tweak ADX min", "paper", "auto/exp-1"),
        )
        conn.execute(
            """
            INSERT INTO decisions(ts, agent, decision_type, summary,
                                  rationale, trigger_metric, trigger_value,
                                  experiment_id, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-05-22T00:00:05Z",
                "researcher",
                "strategy_propose",
                "Raise ADX guard min from 20 to 22",
                "BTC PF dropped to 0.65 over last 20 closes; the ranging-"
                "regime band is too wide, pushing marginal entries through.",
                "profit_factor_last_20_BTCUSDT",
                0.65,
                1,
                "approved",
            ),
        )


def test_render_empty() -> None:
    out = render([])
    assert "No decisions recorded yet" in out


def test_export_writes_markdown(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    out_md = tmp_path / "decisions.md"
    _seed(db)

    stats = export(db, out_md)
    assert stats["entries"] == 1
    content = out_md.read_text(encoding="utf-8")
    assert "Raise ADX guard min from 20 to 22" in content
    assert "researcher" in content
    assert "✅" in content  # approved glyph
    assert "auto/exp-1" in content


def test_export_empty_table(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    out_md = tmp_path / "decisions.md"
    with sqlite3.connect(str(db)) as conn:
        migrate(conn)

    stats = export(db, out_md)
    assert stats["entries"] == 0
    assert "No decisions recorded yet" in out_md.read_text()
