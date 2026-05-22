"""Implementer tests — surgical regex edits + test generation + git ops.

Each test runs against a tmp scratch repo with a synthetic
live_trading_htf.py so we don't disturb the real one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.self_improve.implementer import (
    EDITABLE_CONSTANTS,
    _add_to_blocklist,
    _format_value_for_python,
    _replace_constant,
    _validate,
    implement,
)
from src.self_improve.risk_officer import Proposal


# ─────────────────────────────────────────────────────────────────────────
# Pure-function tests (no git, no files)
# ─────────────────────────────────────────────────────────────────────────


def test_format_simple_types() -> None:
    assert _format_value_for_python(0.55) == "0.55"
    assert _format_value_for_python(42) == "42"
    assert _format_value_for_python(True) == "True"
    assert _format_value_for_python("hi") == "'hi'"


def test_format_dict_inline() -> None:
    out = _format_value_for_python({"XRPUSDT": 0.65})
    assert out == "{'XRPUSDT': 0.65}"


def test_format_set_sorted() -> None:
    out = _format_value_for_python({("Z", "L"), ("A", "S")})
    # Sorted by JSON repr key, so A comes first
    assert out.startswith("{(") and "('A', 'S')" in out and "('Z', 'L')" in out


def test_replace_constant_simple_int() -> None:
    src = "FOO = 10\nBAR = 20\n"
    new, ok = _replace_constant(src, "FOO", 99)
    assert ok
    assert new == "FOO = 99\nBAR = 20\n"


def test_replace_constant_with_type_annotation() -> None:
    src = "X: float = 0.5\n"
    new, ok = _replace_constant(src, "X", 0.8)
    assert ok
    assert "X = 0.8" in new


def test_replace_constant_multiline_set() -> None:
    src = (
        "PREFIX = 1\n"
        "MYSET: set = {\n"
        '    ("A", "L"),\n'
        '    ("B", "S"),\n'
        "}\n"
        "SUFFIX = 99\n"
    )
    new, ok = _replace_constant(src, "MYSET", {("C", "S")})
    assert ok
    assert "MYSET = " in new
    assert "('C', 'S')" in new
    assert "PREFIX = 1" in new
    assert "SUFFIX = 99" in new


def test_replace_constant_missing_returns_unchanged() -> None:
    src = "FOO = 1\n"
    new, ok = _replace_constant(src, "NOPE", 2)
    assert not ok
    assert new == src


def test_add_to_blocklist_appends() -> None:
    src = (
        "SYMBOL_SIDE_BLOCKLIST: set = {\n"
        '    ("SOLUSDT", "LONG"),\n'
        "}\n"
    )
    new, ok, total = _add_to_blocklist(src, [["XRPUSDT", "LONG"]])
    assert ok
    assert ("SOLUSDT", "LONG") in total
    assert ("XRPUSDT", "LONG") in total
    assert "XRPUSDT" in new


def test_add_to_blocklist_no_op_when_already_present() -> None:
    src = 'SYMBOL_SIDE_BLOCKLIST: set = {("SOLUSDT", "LONG")}\n'
    new, ok, _ = _add_to_blocklist(src, [["SOLUSDT", "LONG"]])
    assert not ok  # nothing new to add
    assert new == src


# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────


def test_validate_rejects_non_editable_constant() -> None:
    p = Proposal(
        description="bad",
        config_changes={"FIXED_MAX_NOTIONAL": 500.0},
    )
    err = _validate(p)
    assert "EDITABLE_CONSTANTS" in err


def test_validate_accepts_editable() -> None:
    p = Proposal(
        description="ok",
        config_changes={"MIN_CONFIDENCE": 0.55},
    )
    assert _validate(p) == ""


def test_validate_accepts_blocklist_add() -> None:
    p = Proposal(
        description="add XRP to blocklist",
        config_changes={"SYMBOL_SIDE_BLOCKLIST_ADD": [("XRPUSDT", "LONG")]},
    )
    assert _validate(p) == ""


def test_validate_rejects_empty() -> None:
    p = Proposal(description="empty", config_changes={})
    err = _validate(p)
    assert "no config_changes" in err


# ─────────────────────────────────────────────────────────────────────────
# End-to-end implement() against a scratch git repo
# ─────────────────────────────────────────────────────────────────────────


def _make_scratch_repo(tmp_path: Path) -> Path:
    """Build a minimal git repo with a stub live_trading_htf.py."""
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / "live_trading_htf.py").write_text(
        "MIN_CONFIDENCE = 0.45\n"
        'SYMBOL_MIN_CONFIDENCE = {"ETHUSDT": 0.80}\n'
        "SYMBOL_SIDE_BLOCKLIST: set = {\n"
        '    ("SOLUSDT", "LONG"),\n'
        "}\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_self_improve").mkdir()
    (tmp_path / "tests" / "test_self_improve" / "__init__.py").write_text("")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=tmp_path, check=True)
    # Create the base branch the Implementer expects
    subprocess.run(["git", "branch", "feature/autonomous-loop"], cwd=tmp_path, check=True)
    subprocess.run(["git", "checkout", "feature/autonomous-loop"], cwd=tmp_path, check=True)
    return tmp_path


def test_implement_creates_branch_and_commit(tmp_path: Path) -> None:
    repo = _make_scratch_repo(tmp_path)
    p = Proposal(
        description="Raise XRP confidence floor to 0.65",
        config_changes={"SYMBOL_MIN_CONFIDENCE": {"XRPUSDT": 0.65, "ETHUSDT": 0.80}},
        category="config_tune",
        rationale="XRP entries cluster low",
        expected_impact="block low-conf XRP",
    )
    result = implement(p, experiment_id=42, repo=repo, skip_tests=True)
    assert result.ok, f"implement failed: {result.error}\n{result.test_output_tail}"
    assert result.branch == "auto/experiment-42"
    assert result.commit_sha
    # Check the file was edited
    new_src = (repo / "live_trading_htf.py").read_text()
    assert "XRPUSDT" in new_src
    assert "0.65" in new_src
    # Test file was added
    assert (repo / result.test_file_added).exists()


def test_implement_blocklist_add(tmp_path: Path) -> None:
    repo = _make_scratch_repo(tmp_path)
    p = Proposal(
        description="Block XRP both sides",
        config_changes={
            "SYMBOL_SIDE_BLOCKLIST_ADD": [
                ["XRPUSDT", "LONG"], ["XRPUSDT", "SHORT"]
            ],
        },
        category="blocklist_change",
        rationale="XRP no edge in current regime",
    )
    result = implement(p, experiment_id=7, repo=repo, skip_tests=True)
    assert result.ok, f"implement failed: {result.error}"
    new_src = (repo / "live_trading_htf.py").read_text()
    # Old entry preserved
    assert '("SOLUSDT", "LONG")' in new_src
    # New entries added
    assert '("XRPUSDT", "LONG")' in new_src
    assert '("XRPUSDT", "SHORT")' in new_src


def test_implement_refuses_non_editable(tmp_path: Path) -> None:
    repo = _make_scratch_repo(tmp_path)
    p = Proposal(
        description="evil",
        config_changes={"FIXED_MAX_NOTIONAL": 10000.0},
    )
    result = implement(p, experiment_id=99, repo=repo, skip_tests=True)
    assert not result.ok
    assert "EDITABLE_CONSTANTS" in result.rejected_reason


def test_implement_refuses_dirty_tree(tmp_path: Path) -> None:
    repo = _make_scratch_repo(tmp_path)
    # Introduce a dirty change in source
    (repo / "live_trading_htf.py").write_text("DIRTY = 1\n")
    p = Proposal(
        description="ok",
        config_changes={"MIN_CONFIDENCE": 0.55},
    )
    result = implement(p, experiment_id=1, repo=repo, skip_tests=True)
    assert not result.ok
    assert "dirty" in result.error.lower() or "unrelated" in result.error.lower()


def test_editable_constants_includes_expected() -> None:
    """Spot-check that the allowlist matches what the Researcher can produce."""
    expected = {
        "MIN_CONFIDENCE", "SYMBOL_MIN_CONFIDENCE", "STAGNANT_HOURS",
        "STAGNANT_PCT_MIN", "STAGNANT_PCT_MAX",
        "WHIPSAW_COOLDOWN_HOURS", "COOLDOWN_SECONDS",
        "RANGING_MIN_CONFIDENCE",
    }
    assert expected <= EDITABLE_CONSTANTS
