"""Pure logic for autonomous-loop runtime config overrides.

This module is deliberately stdlib-only so it can be unit-tested without
importing the heavy `live_trading_htf` module (torch, stable-baselines3,
etc.). `live_trading_htf` imports `tighten_overrides` and applies the
result to its module globals once at process startup; the autonomous-loop
apply step uses `check_tightening_only` as a pre-write guard.

Safety invariant enforced here (see live_trading_htf for the prose):

  * Only the 4 entry-suppression gates are recognized — sizing, SL/TP and
    risk-per-trade are never expressible through this path.
  * Overrides are MONOTONIC TIGHTENING ONLY: a confidence floor may be
    raised, a blocklist entry may be added; a floor may never be lowered
    and a block may never be removed. The worst a corrupt or hostile
    override file can do is make the bot block more entries than its
    committed baseline.
"""

from __future__ import annotations

from typing import Any

APPLYABLE_KEYS: frozenset[str] = frozenset({
    "MIN_CONFIDENCE",
    "SYMBOL_MIN_CONFIDENCE",
    "SYMBOL_DIRECTIONAL_CONF",
    "SYMBOL_SIDE_BLOCKLIST_ADD",
    "SYMBOL_SIDE_BLOCKLIST",  # alias of SYMBOL_SIDE_BLOCKLIST_ADD
})


def _coerce_overrides(payload: Any) -> dict[str, Any]:
    """Accept both {"config_changes": {...}} and a bare overrides dict."""
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("config_changes", payload)
    return inner if isinstance(inner, dict) else {}


def tighten_overrides(
    *,
    overrides: dict[str, Any],
    min_confidence: float,
    symbol_min_confidence: dict[str, float],
    symbol_directional_conf: dict[str, dict[str, float]],
    symbol_side_blocklist: set[tuple[str, str]],
) -> dict[str, Any]:
    """Compute the monotonically-tightened constants given an override dict.

    Inputs are treated as read-only; the returned values are fresh copies
    with only tightening changes applied. The caller assigns them to the
    live globals. Returns a dict with the new values plus ``applied`` and
    ``skipped`` lists describing every decision (for logging / audit).
    """
    overrides = _coerce_overrides(overrides)

    new_min_conf = float(min_confidence)
    new_sym_min = dict(symbol_min_confidence)
    new_dir = {s: dict(v) for s, v in symbol_directional_conf.items()}
    new_block = set(symbol_side_blocklist)

    applied: list[str] = []
    skipped: list[str] = []

    # A floor change is one of: raise (apply), equal restatement (benign
    # no-op), or lower (loosening — refused). Only loosening is a violation.
    def _floor_skip(label: str, v: float, cur: float) -> str:
        if v == cur:
            return f"{label} {v:.3f} equals baseline (no-op)"
        return f"{label} {v:.3f} below baseline {cur:.3f} (would loosen)"

    # 1. Global MIN_CONFIDENCE — raise only.
    if "MIN_CONFIDENCE" in overrides:
        try:
            v = float(overrides["MIN_CONFIDENCE"])
            if v > new_min_conf:
                applied.append(f"MIN_CONFIDENCE {new_min_conf:.3f}→{v:.3f}")
                new_min_conf = v
            else:
                skipped.append(_floor_skip("MIN_CONFIDENCE", v, new_min_conf))
        except (TypeError, ValueError):
            skipped.append("MIN_CONFIDENCE not numeric")

    # 2. SYMBOL_MIN_CONFIDENCE — per symbol, merge + raise only.
    if "SYMBOL_MIN_CONFIDENCE" in overrides:
        cfg = overrides["SYMBOL_MIN_CONFIDENCE"]
        if isinstance(cfg, dict):
            for sym, val in cfg.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    skipped.append(f"SYMBOL_MIN_CONFIDENCE[{sym}] not numeric")
                    continue
                cur = new_sym_min.get(sym)
                if cur is None or v > cur:
                    new_sym_min[sym] = v
                    applied.append(
                        f"SYMBOL_MIN_CONFIDENCE[{sym}] "
                        f"{'∅' if cur is None else f'{cur:.3f}'}→{v:.3f}"
                    )
                else:
                    skipped.append(_floor_skip(f"SYMBOL_MIN_CONFIDENCE[{sym}]", v, cur))
        else:
            skipped.append("SYMBOL_MIN_CONFIDENCE not a dict")

    # 3. SYMBOL_DIRECTIONAL_CONF — per (symbol, side), merge + raise only.
    if "SYMBOL_DIRECTIONAL_CONF" in overrides:
        cfg = overrides["SYMBOL_DIRECTIONAL_CONF"]
        if isinstance(cfg, dict):
            for sym, sides in cfg.items():
                if not isinstance(sides, dict):
                    skipped.append(f"SYMBOL_DIRECTIONAL_CONF[{sym}] not a dict")
                    continue
                for side, val in sides.items():
                    side_u = str(side).upper()
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        skipped.append(
                            f"SYMBOL_DIRECTIONAL_CONF[{sym}][{side_u}] not numeric"
                        )
                        continue
                    cur = new_dir.get(sym, {}).get(side_u)
                    if cur is None or v > cur:
                        new_dir.setdefault(sym, {})[side_u] = v
                        applied.append(
                            f"SYMBOL_DIRECTIONAL_CONF[{sym}][{side_u}] "
                            f"{'∅' if cur is None else f'{cur:.3f}'}→{v:.3f}"
                        )
                    else:
                        skipped.append(
                            _floor_skip(
                                f"SYMBOL_DIRECTIONAL_CONF[{sym}][{side_u}]", v, cur
                            )
                        )
        else:
            skipped.append("SYMBOL_DIRECTIONAL_CONF not a dict")

    # 4. SYMBOL_SIDE_BLOCKLIST_ADD (or SYMBOL_SIDE_BLOCKLIST alias) — union only.
    _blocklist_src_key = (
        "SYMBOL_SIDE_BLOCKLIST_ADD" if "SYMBOL_SIDE_BLOCKLIST_ADD" in overrides
        else ("SYMBOL_SIDE_BLOCKLIST" if "SYMBOL_SIDE_BLOCKLIST" in overrides else None)
    )
    if _blocklist_src_key is not None:
        adds = overrides[_blocklist_src_key]
        if isinstance(adds, (list, tuple)):
            for entry in adds:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    pair = (str(entry[0]).upper(), str(entry[1]).upper())
                    if pair not in new_block:
                        new_block.add(pair)
                        applied.append(f"SYMBOL_SIDE_BLOCKLIST+{pair}")
                else:
                    skipped.append(f"{_blocklist_src_key} bad entry {entry!r}")
        else:
            skipped.append(f"{_blocklist_src_key} not a list")

    # Unknown keys — schema-drift signal (mirrors backtest harness).
    for key in sorted(set(overrides) - APPLYABLE_KEYS):
        skipped.append(f"unknown key {key!r} (not an entry-suppression gate)")

    return {
        "min_confidence": new_min_conf,
        "symbol_min_confidence": new_sym_min,
        "symbol_directional_conf": new_dir,
        "symbol_side_blocklist": new_block,
        "applied": applied,
        "skipped": skipped,
    }


def check_tightening_only(
    *,
    overrides: dict[str, Any],
    min_confidence: float,
    symbol_min_confidence: dict[str, float],
    symbol_directional_conf: dict[str, dict[str, float]],
    symbol_side_blocklist: set[tuple[str, str]],
) -> list[str]:
    """Pre-write guard for the orchestrator apply step.

    Returns a list of violations: override entries that would LOOSEN the
    live config (lower a floor, or that are unknown keys). Empty list means
    the override is safe to apply (pure tightening). This lets the
    orchestrator refuse to even write an override file that isn't strictly
    more conservative than the running baseline.
    """
    result = tighten_overrides(
        overrides=overrides,
        min_confidence=min_confidence,
        symbol_min_confidence=symbol_min_confidence,
        symbol_directional_conf=symbol_directional_conf,
        symbol_side_blocklist=symbol_side_blocklist,
    )
    violations: list[str] = []
    for s in result["skipped"]:
        # Only actual loosening or unknown (unvalidated) keys disqualify.
        # An "equals baseline (no-op)" restatement is benign.
        if "would loosen" in s or "unknown key" in s or "not numeric" in s \
                or "not a dict" in s or "not a list" in s or "bad entry" in s:
            violations.append(s)
    # Something must actually change, or there's nothing to deploy live.
    if not result["applied"] and not violations:
        violations.append("override has no applicable tightening effect")
    return violations
