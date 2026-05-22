"""Implementer — converts an approved Proposal into a git branch with the
config change committed and a regression test added.

M4 scope is **config-only**:

  * Only changes to numeric constants and small literal collections
    (sets, dicts) are supported
  * The Implementer regex-edits live_trading_htf.py with surgical
    precision — no AST manipulation, no fancy refactors
  * Every change must come with a test in tests/test_self_improve/
    so future regressions surface
  * Always creates a fresh branch `auto/experiment-<id>`
  * NEVER merges to dev — that's a manual step gated through canary
    promotion in M5 (and even then, the merge is PR-style, never
    silent)

M5+ will add a more sophisticated mode that lets the LLM write Python
code via patches. For M4 we keep the surface tiny so the failure modes
are obvious.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .risk_officer import Proposal

# Constants the Implementer is allowed to edit. Anything outside this
# allowlist auto-rejects (a defense-in-depth in addition to
# FORBIDDEN_AREAS in researcher.py).
EDITABLE_CONSTANTS = {
    "MIN_CONFIDENCE",
    "SYMBOL_MIN_CONFIDENCE",
    "SYMBOL_DIRECTIONAL_CONF",
    "SYMBOL_SIDE_BLOCKLIST",
    "RANGING_MIN_CONFIDENCE",
    "RANGING_ADX_THRESHOLD",
    "USDT_D_THRESHOLD_PCT",
    "USDT_D_LOOKBACK_HOURS",
    "EXT_POS_NEWS_SENTIMENT_THRESHOLD",
    "EXT_POS_NEWS_LOOKBACK_MINUTES",
    "STAGNANT_HOURS",
    "STAGNANT_PCT_MIN",
    "STAGNANT_PCT_MAX",
    "WHIPSAW_COOLDOWN_HOURS",
    "COOLDOWN_SECONDS",
    "MIN_HOLD_SECONDS",
    "TRAILING_DISTANCE_PCT",
    "TRAILING_DISTANCE_POST_TP1",
    "ADX_GUARD_MIN",
    "REVERSAL_BLOCK_LONG_CANARY_SYMBOLS",
    "REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT",
    "SYMBOL_SIZE_WR_THRESHOLD",
    "SYMBOL_SIZE_DOWNSCALE_FACTOR",
    "SYMBOL_SIZE_LOOKBACK_TRADES",
}


@dataclass
class ImplementerResult:
    """Outcome of an Implementer run."""

    ok: bool
    branch: str = ""
    commit_sha: str = ""
    diff_summary: str = ""
    test_file_added: str = ""
    test_output_tail: str = ""
    error: str = ""
    rejected_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "diff_summary": self.diff_summary,
            "test_file_added": self.test_file_added,
            "test_output_excerpt": self.test_output_tail[-1500:],
            "error": self.error,
            "rejected_reason": self.rejected_reason,
        }


# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────


def _validate(proposal: Proposal) -> str:
    """Return an error string if the proposal can't be implemented in M4,
    else empty string."""
    if not proposal.config_changes:
        return "proposal has no config_changes — nothing to implement"
    for name in proposal.config_changes:
        if name == "SYMBOL_SIDE_BLOCKLIST_ADD":
            # Special: this isn't a constant the Implementer rewrites
            # outright — it's an ADD to the existing set. Allowed.
            continue
        if name not in EDITABLE_CONSTANTS:
            return (
                f"constant {name!r} is not in EDITABLE_CONSTANTS — the "
                f"Implementer refuses to touch it in M4 (config-only mode)"
            )
    return ""


# ─────────────────────────────────────────────────────────────────────────
# File edits — surgical regex against live_trading_htf.py
# ─────────────────────────────────────────────────────────────────────────


def _format_value_for_python(v: Any) -> str:
    """Render a Python literal for embedding in source.

    Floats are rendered with trailing zero stripped. Sets use sorted
    order so the diff is stable. Dicts are pretty-printed if multi-line.
    """
    import json as _json
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Avoid scientific notation for readability
        s = f"{v}"
        return s
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, (set, frozenset)):
        items = sorted(v, key=lambda x: _json.dumps(x, sort_keys=True, default=str))
        if not items:
            return "set()"
        inner = ", ".join(_format_value_for_python(i) for i in items)
        return "{" + inner + "}"
    if isinstance(v, tuple):
        items = [_format_value_for_python(x) for x in v]
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
    if isinstance(v, list):
        items = [_format_value_for_python(x) for x in v]
        return "[" + ", ".join(items) + "]"
    if isinstance(v, dict):
        items = []
        for k, val in v.items():
            items.append(f"{_format_value_for_python(k)}: {_format_value_for_python(val)}")
        return "{" + ", ".join(items) + "}"
    return repr(v)


def _replace_constant(source: str, name: str, new_value: Any) -> tuple[str, bool]:
    """Replace `NAME = <anything>` (single-line or `set()` block) with the
    new value. Returns (new_source, changed)."""
    # Single-line assignment: `NAME = something`
    pat = re.compile(
        rf"^([ \t]*){re.escape(name)}\s*(?::\s*[A-Za-z_\[\], ]+)?\s*=\s*[^\n]+",
        re.MULTILINE,
    )
    rendered = _format_value_for_python(new_value)
    replacement = rf"\g<1>{name} = {rendered}"
    new_source, n = pat.subn(replacement, source, count=1)
    if n:
        return new_source, True

    # Multi-line set/dict/tuple literal: `NAME: set = {` ... `}`
    # We look for `NAME` followed by optional annotation, '=', then the
    # opening bracket, and replace through the matching closing bracket
    # at the same indent level. Use a balanced-bracket walker.
    m = re.search(
        rf"^([ \t]*){re.escape(name)}\s*(?::\s*[A-Za-z_\[\], ]+)?\s*=\s*([\(\{{\[])",
        source,
        re.MULTILINE,
    )
    if not m:
        return source, False
    indent, opener = m.group(1), m.group(2)
    closer = {"(": ")", "{": "}", "[": "]"}[opener]
    start = m.start()
    bracket_start = m.end() - 1
    depth = 0
    i = bracket_start
    while i < len(source):
        ch = source[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                end = i + 1
                replacement = f"{indent}{name} = {rendered}"
                return source[:start] + replacement + source[end:], True
        i += 1
    return source, False


def _add_to_blocklist(source: str, additions: Iterable) -> tuple[str, bool, set]:
    """For SYMBOL_SIDE_BLOCKLIST_ADD — read current literal, add new
    entries, rewrite. Returns (new_source, changed, the new total set)."""
    m = re.search(
        r"^([ \t]*)SYMBOL_SIDE_BLOCKLIST\s*(?::\s*[A-Za-z_\[\], ]+)?\s*=\s*\{([^}]*)\}",
        source,
        re.MULTILINE | re.DOTALL,
    )
    if not m:
        return source, False, set()
    indent = m.group(1)
    inner = m.group(2)

    # Parse existing entries: tuples of the form ("XXX", "YYY")
    existing: set[tuple[str, str]] = set()
    for sym, side in re.findall(r"\(\s*[\"\']([^\"\']+)[\"\']\s*,\s*[\"\']([^\"\']+)[\"\']\s*\)", inner):
        existing.add((sym.upper(), side.upper()))

    additions_set: set[tuple[str, str]] = set()
    for a in additions:
        if isinstance(a, (list, tuple)) and len(a) == 2:
            additions_set.add((str(a[0]).upper(), str(a[1]).upper()))

    new_total = existing | additions_set
    if not (additions_set - existing):
        return source, False, new_total

    # Rewrite. Keep one entry per line for diff readability.
    sorted_entries = sorted(new_total)
    body = ",\n    ".join(
        f'("{s}", "{side}")' for s, side in sorted_entries
    )
    rendered = "{\n    " + body + ",\n}"
    new_source = source[:m.start()] + f"{indent}SYMBOL_SIDE_BLOCKLIST: set = {rendered}" + source[m.end():]
    return new_source, True, new_total


# ─────────────────────────────────────────────────────────────────────────
# Test generation
# ─────────────────────────────────────────────────────────────────────────


def _render_test_file(*, experiment_id: int, proposal: Proposal) -> str:
    """Generate a regression test that asserts the new constant values
    are present after the change. This is the minimum bar — future
    versions can ask the LLM to write richer tests."""
    description_safe = proposal.description.replace('"""', '"\\"\\"')
    rationale_safe = (proposal.rationale or "").replace('"""', '"\\"\\"')
    asserts: list[str] = []
    for name, value in proposal.config_changes.items():
        if name == "SYMBOL_SIDE_BLOCKLIST_ADD":
            for s, side in value:
                asserts.append(
                    f'    assert ("{s}", "{side}") in mod.SYMBOL_SIDE_BLOCKLIST'
                )
            continue
        rendered = _format_value_for_python(value)
        # For sets/dicts, equality compare. For floats, allow tiny epsilon.
        if isinstance(value, float):
            asserts.append(
                f"    assert abs(mod.{name} - ({rendered})) < 1e-9"
            )
        elif isinstance(value, (set, frozenset)):
            # The new set should be a SUBSET of the live one (we may have
            # added entries through SYMBOL_SIDE_BLOCKLIST_ADD too)
            asserts.append(
                f"    assert mod.{name} == {rendered}"
            )
        else:
            asserts.append(f"    assert mod.{name} == {rendered}")

    asserts_block = "\n".join(asserts) if asserts else "    pass"

    return f'''"""Auto-generated regression test for experiment #{experiment_id}.

Generated by src.self_improve.implementer. Do not edit by hand.

Proposal: {description_safe}

Rationale: {rationale_safe}
"""

from __future__ import annotations

import importlib


def test_experiment_{experiment_id}_constants_applied() -> None:
    mod = importlib.import_module("live_trading_htf")
{asserts_block}
'''


# ─────────────────────────────────────────────────────────────────────────
# Git operations
# ─────────────────────────────────────────────────────────────────────────


def _git(*args: str, cwd: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def _ensure_clean_for_branching(repo: Path) -> str:
    """Quick check that the working tree is clean enough to branch off.
    Returns an error string if dirty, empty string otherwise."""
    r = _git("status", "--porcelain", cwd=repo)
    if r.returncode != 0:
        return f"git status failed: {r.stderr}"
    # Allow modifications in data/ (runtime artifacts) — only block on
    # tracked source-code changes.
    dirty_source: list[str] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Untracked or modified line — keep only paths
        path = line[3:] if len(line) > 3 else line
        if path.startswith("data/") or path.startswith("logs/"):
            continue
        # Allow our auto-edit target if it's already been edited (e.g.
        # by a manual operator) — but the implementer should still
        # warn the operator.
        dirty_source.append(line)
    if dirty_source:
        return "working tree has unrelated source changes; refusing to branch: " + "; ".join(dirty_source[:5])
    return ""


# ─────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────


def implement(
    proposal: Proposal,
    *,
    experiment_id: int,
    repo: str | Path = ".",
    base_branch: str = "feature/autonomous-loop",
    target_file: str = "live_trading_htf.py",
    skip_tests: bool = False,
) -> ImplementerResult:
    """Implement the proposal: edit target_file, add a test, commit on
    a new branch. Returns ImplementerResult.

    Does NOT push. Does NOT merge. The orchestrator decides what to
    do with the branch.
    """
    repo_path = Path(repo).resolve()

    # 1. Validate scope
    err = _validate(proposal)
    if err:
        return ImplementerResult(ok=False, rejected_reason=err)

    # 2. Make sure base branch exists and check it out cleanly. We don't
    # do destructive cleanup — if the tree is dirty, refuse.
    dirty = _ensure_clean_for_branching(repo_path)
    if dirty:
        return ImplementerResult(ok=False, error=dirty)

    # 3. Switch to base branch
    co = _git("checkout", base_branch, cwd=repo_path)
    if co.returncode != 0:
        return ImplementerResult(
            ok=False,
            error=f"git checkout {base_branch} failed: {co.stderr}",
        )

    # 4. Create the experiment branch
    branch = f"auto/experiment-{experiment_id}"
    nb = _git("checkout", "-B", branch, cwd=repo_path)
    if nb.returncode != 0:
        return ImplementerResult(
            ok=False,
            error=f"git checkout -B {branch} failed: {nb.stderr}",
        )

    # 5. Apply config changes to target file
    target_path = repo_path / target_file
    if not target_path.exists():
        return ImplementerResult(
            ok=False,
            error=f"target file {target_path} does not exist",
        )
    source = target_path.read_text()
    changes_applied: list[str] = []
    for name, value in proposal.config_changes.items():
        if name == "SYMBOL_SIDE_BLOCKLIST_ADD":
            new_src, changed, total = _add_to_blocklist(source, value)
            if not changed:
                return ImplementerResult(
                    ok=False,
                    error=f"SYMBOL_SIDE_BLOCKLIST_ADD: no edit applied (additions already present?)",
                )
            source = new_src
            changes_applied.append(
                f"SYMBOL_SIDE_BLOCKLIST += {value} (now {len(total)} entries)"
            )
            continue

        new_src, changed = _replace_constant(source, name, value)
        if not changed:
            return ImplementerResult(
                ok=False,
                error=f"could not find constant {name} in {target_file}",
            )
        source = new_src
        changes_applied.append(f"{name} = {_format_value_for_python(value)}")

    target_path.write_text(source)

    # 6. Generate test file
    test_name = f"test_experiment_{experiment_id}.py"
    test_path = repo_path / "tests" / "test_self_improve" / test_name
    test_path.write_text(
        _render_test_file(experiment_id=experiment_id, proposal=proposal)
    )

    # 7. Run pytest on the new test (and the broader self-improve suite)
    test_output_tail = ""
    if not skip_tests:
        pyt = subprocess.run(
            ["python3", "-m", "pytest",
             f"tests/test_self_improve/{test_name}",
             "tests/test_self_improve/test_metrics.py",
             "tests/test_self_improve/test_triggers.py",
             "-q"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        test_output_tail = (pyt.stdout or "") + "\n" + (pyt.stderr or "")
        if pyt.returncode != 0:
            return ImplementerResult(
                ok=False,
                branch=branch,
                error=f"pytest failed on the implemented change",
                diff_summary="; ".join(changes_applied),
                test_file_added=str(test_path.relative_to(repo_path)),
                test_output_tail=test_output_tail,
            )

    # 8. Stage and commit
    _git("add", str(target_path.relative_to(repo_path)),
         str(test_path.relative_to(repo_path)), cwd=repo_path)
    commit_msg = (
        f"experiment {experiment_id}: {proposal.description}\n\n"
        f"Auto-generated by the self-improvement Implementer.\n\n"
        f"Category: {proposal.category}\n"
        f"Rationale: {proposal.rationale}\n"
        f"Expected impact: {proposal.expected_impact}\n\n"
        f"Changes: {'; '.join(changes_applied)}\n\n"
        f"Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>\n"
    )
    ci = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )
    if ci.returncode != 0:
        return ImplementerResult(
            ok=False,
            branch=branch,
            error=f"git commit failed: {ci.stderr or ci.stdout}",
            diff_summary="; ".join(changes_applied),
            test_file_added=str(test_path.relative_to(repo_path)),
            test_output_tail=test_output_tail,
        )

    # 9. Get the SHA
    sha = _git("rev-parse", "HEAD", cwd=repo_path).stdout.strip()

    return ImplementerResult(
        ok=True,
        branch=branch,
        commit_sha=sha,
        diff_summary="; ".join(changes_applied),
        test_file_added=str(test_path.relative_to(repo_path)),
        test_output_tail=test_output_tail,
    )
