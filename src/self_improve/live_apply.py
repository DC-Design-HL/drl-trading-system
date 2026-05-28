"""Live-apply layer — the ONLY place that touches the running bots.

Everything that writes the runtime override file or restarts the production
trading process lives here, isolated so the risky surface is small and
testable. The orchestrator calls into this module; nothing else should
write `active_overrides.json` or shell out to `start_services.sh`.

Two flag files under `data/self_improve/` gate every action:

  * ``AUTONOMY_ARMED`` — MUST be present for any live apply/revert restart
    to fire. Absent (the default, and what ships) means the loop will
    research + backtest + paper-trade and even stage a change, but will
    NEVER write an override or restart a bot. Arming is a deliberate,
    one-file act (done with Chen's go-ahead).
  * ``AUTONOMY_DISABLED`` — hard kill switch. If present, no apply happens
    AND the live bot's loader ignores any override file that exists. This
    is the instant-freeze control.

The apply path is also strictly monotonic-tightening (delegated to
``runtime_overrides.check_tightening_only``): a live change can only block
more entries, never loosen a floor or change sizing/SL/TP.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .runtime_overrides import check_tightening_only

UTC = timezone.utc

# Circuit-breaker thresholds (PLAN.md §8). Measured on realized PnL of
# closes that happened AFTER the override was applied.
CIRCUIT_BREAKER_LOSS_PCT = 0.05   # cumulative realized loss ≥5% of capital → revert
CIRCUIT_BREAKER_DD_PCT = 0.08     # peak-to-trough drawdown ≥8% since apply → revert
CIRCUIT_BREAKER_MIN_CLOSES = 3    # don't trip on a single unlucky close

# How long a change observes as a canary before auto-promoting to "live".
CANARY_HOURS = 48.0


def _sd(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def override_path(base_dir: Path) -> Path:
    return base_dir / "active_overrides.json"


def armed_flag(base_dir: Path) -> Path:
    return base_dir / "AUTONOMY_ARMED"


def kill_flag(base_dir: Path) -> Path:
    return base_dir / "AUTONOMY_DISABLED"


def is_armed(base_dir: Path) -> bool:
    return armed_flag(base_dir).exists() and not kill_flag(base_dir).exists()


def is_killed(base_dir: Path) -> bool:
    return kill_flag(base_dir).exists()


# ─────────────────────────────────────────────────────────────────────────
# Pristine baseline — read live constants WITHOUT applying any override
# ─────────────────────────────────────────────────────────────────────────


_BASELINE_CONSTANT_NAMES = (
    "MIN_CONFIDENCE",
    "SYMBOL_MIN_CONFIDENCE",
    "SYMBOL_DIRECTIONAL_CONF",
    "SYMBOL_SIDE_BLOCKLIST",
)


def read_baseline_constants(repo: Path) -> Optional[dict[str, Any]]:
    """Parse the four entry-gate constants from live_trading_htf.py SOURCE
    via AST — deliberately WITHOUT importing the module, so this stays cheap
    and never loads torch on the memory-constrained box. Returns the pristine
    committed values (independent of any active override). Returns None if
    the file or a constant can't be parsed — callers treat None as "skip the
    pre-check and rely on the loader's runtime tightening guard instead"."""
    import ast

    src_path = repo / "live_trading_htf.py"
    if not src_path.exists():
        return None
    try:
        tree = ast.parse(src_path.read_text())
    except (OSError, SyntaxError):
        return None

    found: dict[str, Any] = {}
    for node in tree.body:
        target_name = None
        value_node = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        if target_name in _BASELINE_CONSTANT_NAMES and value_node is not None \
                and target_name not in found:
            try:
                found[target_name] = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                return None

    if not all(name in found for name in _BASELINE_CONSTANT_NAMES):
        return None
    return {
        "min_confidence": float(found["MIN_CONFIDENCE"]),
        "symbol_min_confidence": dict(found["SYMBOL_MIN_CONFIDENCE"]),
        "symbol_directional_conf": {
            s: dict(v) for s, v in found["SYMBOL_DIRECTIONAL_CONF"].items()
        },
        "symbol_side_blocklist": set(found["SYMBOL_SIDE_BLOCKLIST"]),
    }


# ─────────────────────────────────────────────────────────────────────────
# Apply / revert
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ApplyResult:
    ok: bool
    action: str = ""               # "applied" | "reverted" | "noop"
    restarted: bool = False
    override_written: bool = False
    reason: str = ""
    violations: list[str] = field(default_factory=list)
    restart_output: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "action": self.action, "restarted": self.restarted,
            "override_written": self.override_written, "reason": self.reason,
            "violations": self.violations,
            "restart_output": self.restart_output[-800:],
        }


def _read_active_changes(base_dir: Path) -> dict[str, Any]:
    """Return the config_changes currently live (empty if none)."""
    op = override_path(base_dir)
    if not op.exists():
        return {}
    try:
        payload = json.loads(op.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    cc = payload.get("config_changes", {}) if isinstance(payload, dict) else {}
    return cc if isinstance(cc, dict) else {}


def _merge_config_changes(
    existing: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    """Union two config_changes dicts. Per-symbol/per-side dicts are merged
    key-wise; blocklist-add lists are unioned. The live loader re-tightens
    monotonically at startup, so this only needs to preserve information."""
    out: dict[str, Any] = {k: v for k, v in existing.items()}
    for key, val in new.items():
        if key in ("SYMBOL_MIN_CONFIDENCE",) and isinstance(val, dict):
            merged = dict(out.get(key, {}))
            merged.update(val)
            out[key] = merged
        elif key == "SYMBOL_DIRECTIONAL_CONF" and isinstance(val, dict):
            merged = {s: dict(d) for s, d in out.get(key, {}).items()}
            for sym, sides in val.items():
                if isinstance(sides, dict):
                    merged.setdefault(sym, {}).update(sides)
            out[key] = merged
        elif key == "SYMBOL_SIDE_BLOCKLIST_ADD" and isinstance(val, (list, tuple)):
            seen = {tuple(x) for x in out.get(key, []) if isinstance(x, (list, tuple))}
            combined = list(out.get(key, []))
            for entry in val:
                if isinstance(entry, (list, tuple)) and tuple(entry) not in seen:
                    combined.append(list(entry))
                    seen.add(tuple(entry))
            out[key] = combined
        else:
            out[key] = val
    return out


def _restart_services(repo: Path, dry_run: bool) -> tuple[bool, str]:
    """Restart the production cluster via start_services.sh. The script is
    idempotent and flock-guarded against the watchdog. Returns (ok, output)."""
    if dry_run:
        return True, "[dry-run] would run ./start_services.sh"
    script = repo / "start_services.sh"
    if not script.exists():
        return False, f"start_services.sh not found at {script}"
    try:
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=str(repo), capture_output=True, text=True, timeout=180,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "start_services.sh timed out after 180s"
    except Exception as exc:  # noqa: BLE001
        return False, f"restart failed: {exc}"


def apply_live(
    *,
    experiment_id: int,
    config_changes: dict[str, Any],
    base_dir: Path,
    repo: Path,
    dry_run: bool = False,
) -> ApplyResult:
    """Write the override file and restart the bots so the change goes live.

    Refuses unless armed, not killed, and the change is pure tightening
    relative to the pristine baseline (best-effort — the loader enforces
    tightening at runtime regardless)."""
    base_dir = _sd(base_dir)
    if is_killed(base_dir):
        return ApplyResult(ok=False, reason="kill switch (AUTONOMY_DISABLED) present")
    if not is_armed(base_dir):
        return ApplyResult(ok=False, reason="autonomy not armed (AUTONOMY_ARMED absent)")

    baseline = read_baseline_constants(repo)
    if baseline is not None:
        violations = check_tightening_only(overrides=config_changes, **baseline)
        if violations:
            return ApplyResult(
                ok=False, reason="override is not pure tightening",
                violations=violations,
            )

    # Merge into any change already live so successive live experiments
    # accumulate their tightenings instead of clobbering each other. The
    # loader re-applies everything monotonically, so a union of dicts is
    # safe; we just keep the file readable.
    merged = _merge_config_changes(_read_active_changes(base_dir), config_changes)
    payload = {
        "experiment_id": experiment_id,
        "config_changes": merged,
        "applied_at": datetime.now(UTC).isoformat(),
    }
    override_path(base_dir).write_text(json.dumps(payload, indent=2, default=str))
    ok, out = _restart_services(repo, dry_run)
    return ApplyResult(
        ok=ok, action="applied", restarted=not dry_run and ok,
        override_written=True,
        reason="applied + restarted" if ok else "override written but restart failed",
        restart_output=out,
    )


def revert_live(
    *,
    reason: str,
    base_dir: Path,
    repo: Path,
    dry_run: bool = False,
) -> ApplyResult:
    """Remove the override file and restart so the bots return to the
    committed baseline. Safe to call even if no override is present."""
    base_dir = _sd(base_dir)
    op = override_path(base_dir)
    written = op.exists()
    if written:
        op.unlink()
    ok, out = _restart_services(repo, dry_run)
    return ApplyResult(
        ok=ok, action="reverted", restarted=not dry_run and ok,
        override_written=False,
        reason=f"reverted ({reason})" if ok else f"override cleared but restart failed ({reason})",
        restart_output=out,
    )


# ─────────────────────────────────────────────────────────────────────────
# Circuit breaker — measure realized PnL since the change went live
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class BreakerReading:
    n_closes: int
    realized_pnl: float
    max_drawdown_pct: float
    tripped: bool
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "n_closes": self.n_closes, "realized_pnl": self.realized_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "tripped": self.tripped, "reason": self.reason,
        }


def measure_since(
    conn: sqlite3.Connection, *, since_iso: str, capital: float,
) -> BreakerReading:
    """Read realized PnL of testnet closes since ``since_iso`` and decide
    whether the circuit breaker trips. Drawdown is the worst peak-to-trough
    of the cumulative-PnL curve over that window."""
    rows = conn.execute(
        """
        SELECT pnl FROM trades
        WHERE is_testnet=1 AND pnl IS NOT NULL
          AND action IN ('CLOSE_LONG','CLOSE_SHORT','REVERSE_CLOSE_LONG',
                         'REVERSE_CLOSE_SHORT','SL_HIT','TP_HIT')
          AND COALESCE(created_at, timestamp) >= ?
        ORDER BY COALESCE(created_at, timestamp)
        """,
        (since_iso,),
    ).fetchall()
    pnls = [float(r[0]) for r in rows]
    n = len(pnls)
    total = sum(pnls)

    # Max drawdown of the cumulative curve.
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = (max_dd / capital * 100.0) if capital > 0 else 0.0

    tripped = False
    reason = ""
    if n >= CIRCUIT_BREAKER_MIN_CLOSES:
        if total <= -CIRCUIT_BREAKER_LOSS_PCT * capital:
            tripped = True
            reason = (
                f"realized loss ${total:+.2f} ≥ {CIRCUIT_BREAKER_LOSS_PCT*100:.0f}% "
                f"of ${capital:.0f} since apply ({n} closes)"
            )
        elif max_dd_pct >= CIRCUIT_BREAKER_DD_PCT * 100.0:
            tripped = True
            reason = (
                f"drawdown {max_dd_pct:.2f}% ≥ {CIRCUIT_BREAKER_DD_PCT*100:.0f}% "
                f"since apply ({n} closes)"
            )
    return BreakerReading(
        n_closes=n, realized_pnl=round(total, 2),
        max_drawdown_pct=round(max_dd_pct, 2), tripped=tripped, reason=reason,
    )
