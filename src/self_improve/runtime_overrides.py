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

from .safety_envelopes import (
    BLOCKLIST_REMOVE_KEY,
    ENVELOPES,
    check_envelope,
    is_envelope_key,
)

APPLYABLE_KEYS: frozenset[str] = frozenset({
    # Legacy PPO-model-confidence floors. Inert while
    # STRUCTURE_FIRST_MODE=True in live_trading_htf (execute_trade skips
    # them); kept applyable in case structure-first is ever switched off.
    "MIN_CONFIDENCE",
    "SYMBOL_MIN_CONFIDENCE",
    "SYMBOL_DIRECTIONAL_CONF",
    # Structure-confidence floors (PROFITABILITY_PLAN.md P1) — the live
    # apply surface in structure-first mode. Same mechanics as the legacy
    # twins above: raise-only.
    "STRUCT_MIN_CONFIDENCE",
    "STRUCT_SYMBOL_MIN_CONFIDENCE",
    "STRUCT_SYMBOL_DIRECTIONAL_CONF",
    # Blocklist additions (always applyable). Removal is Chen-only.
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
    struct_min_confidence: float = 0.0,
    struct_symbol_min_confidence: dict[str, float] | None = None,
    struct_symbol_directional_conf: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Compute the monotonically-tightened constants given an override dict.

    Inputs are treated as read-only; the returned values are fresh copies
    with only tightening changes applied. The caller assigns them to the
    live globals. Returns a dict with the new values plus ``applied`` and
    ``skipped`` lists describing every decision (for logging / audit).

    The ``struct_*`` parameters are PROFITABILITY_PLAN.md P1's parallel
    floor surface — structure-signal confidence, not PPO model confidence.
    Default 0.0 / {} = no-op (callers that don't yet supply them get
    identical pre-P1 behavior for the legacy keys).
    """
    overrides = _coerce_overrides(overrides)

    new_min_conf = float(min_confidence)
    new_sym_min = dict(symbol_min_confidence)
    new_dir = {s: dict(v) for s, v in symbol_directional_conf.items()}
    new_block = set(symbol_side_blocklist)
    new_struct_min = float(struct_min_confidence)
    new_struct_sym = dict(struct_symbol_min_confidence or {})
    new_struct_dir = {
        s: dict(v) for s, v in (struct_symbol_directional_conf or {}).items()
    }

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

    # 5. STRUCT_MIN_CONFIDENCE — raise only. Same shape as MIN_CONFIDENCE.
    if "STRUCT_MIN_CONFIDENCE" in overrides:
        try:
            v = float(overrides["STRUCT_MIN_CONFIDENCE"])
            if v > new_struct_min:
                applied.append(
                    f"STRUCT_MIN_CONFIDENCE {new_struct_min:.3f}→{v:.3f}"
                )
                new_struct_min = v
            else:
                skipped.append(
                    _floor_skip("STRUCT_MIN_CONFIDENCE", v, new_struct_min)
                )
        except (TypeError, ValueError):
            skipped.append("STRUCT_MIN_CONFIDENCE not numeric")

    # 6. STRUCT_SYMBOL_MIN_CONFIDENCE — per symbol, merge + raise only.
    if "STRUCT_SYMBOL_MIN_CONFIDENCE" in overrides:
        cfg = overrides["STRUCT_SYMBOL_MIN_CONFIDENCE"]
        if isinstance(cfg, dict):
            for sym, val in cfg.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    skipped.append(
                        f"STRUCT_SYMBOL_MIN_CONFIDENCE[{sym}] not numeric"
                    )
                    continue
                cur = new_struct_sym.get(sym)
                if cur is None or v > cur:
                    new_struct_sym[sym] = v
                    applied.append(
                        f"STRUCT_SYMBOL_MIN_CONFIDENCE[{sym}] "
                        f"{'∅' if cur is None else f'{cur:.3f}'}→{v:.3f}"
                    )
                else:
                    skipped.append(
                        _floor_skip(
                            f"STRUCT_SYMBOL_MIN_CONFIDENCE[{sym}]", v, cur
                        )
                    )
        else:
            skipped.append("STRUCT_SYMBOL_MIN_CONFIDENCE not a dict")

    # 7. STRUCT_SYMBOL_DIRECTIONAL_CONF — per (symbol, side), merge + raise.
    if "STRUCT_SYMBOL_DIRECTIONAL_CONF" in overrides:
        cfg = overrides["STRUCT_SYMBOL_DIRECTIONAL_CONF"]
        if isinstance(cfg, dict):
            for sym, sides in cfg.items():
                if not isinstance(sides, dict):
                    skipped.append(
                        f"STRUCT_SYMBOL_DIRECTIONAL_CONF[{sym}] not a dict"
                    )
                    continue
                for side, val in sides.items():
                    side_u = str(side).upper()
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        skipped.append(
                            f"STRUCT_SYMBOL_DIRECTIONAL_CONF[{sym}][{side_u}]"
                            f" not numeric"
                        )
                        continue
                    cur = new_struct_dir.get(sym, {}).get(side_u)
                    if cur is None or v > cur:
                        new_struct_dir.setdefault(sym, {})[side_u] = v
                        applied.append(
                            f"STRUCT_SYMBOL_DIRECTIONAL_CONF[{sym}][{side_u}] "
                            f"{'∅' if cur is None else f'{cur:.3f}'}→{v:.3f}"
                        )
                    else:
                        skipped.append(
                            _floor_skip(
                                f"STRUCT_SYMBOL_DIRECTIONAL_CONF[{sym}][{side_u}]",
                                v,
                                cur,
                            )
                        )
        else:
            skipped.append("STRUCT_SYMBOL_DIRECTIONAL_CONF not a dict")

    # Unknown keys — schema-drift signal (mirrors backtest harness).
    for key in sorted(set(overrides) - APPLYABLE_KEYS):
        skipped.append(f"unknown key {key!r} (not an entry-suppression gate)")

    return {
        "min_confidence": new_min_conf,
        "symbol_min_confidence": new_sym_min,
        "symbol_directional_conf": new_dir,
        "symbol_side_blocklist": new_block,
        "struct_min_confidence": new_struct_min,
        "struct_symbol_min_confidence": new_struct_sym,
        "struct_symbol_directional_conf": new_struct_dir,
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
    struct_min_confidence: float = 0.0,
    struct_symbol_min_confidence: dict[str, float] | None = None,
    struct_symbol_directional_conf: dict[str, dict[str, float]] | None = None,
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
        struct_min_confidence=struct_min_confidence,
        struct_symbol_min_confidence=struct_symbol_min_confidence,
        struct_symbol_directional_conf=struct_symbol_directional_conf,
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


# ─────────────────────────────────────────────────────────────────────────
# PROFITABILITY_PLAN.md P3 — two-sided bounded apply surface
#
# Envelope keys (safety_envelopes.ENVELOPES) may move in EITHER direction so
# long as every numeric leaf stays inside the key's [min, max]. This
# SUPERSEDES the monotonic-tightening rule above for those keys: a STRUCT_*
# floor may be lowered within its envelope because forward-sim + paper +
# canary + circuit breaker validate the change. Keys NOT in ENVELOPES keep
# the legacy raise-only / add-only semantics. Blocklist REMOVAL is never
# applyable here — it is escalated to Chen.
# ─────────────────────────────────────────────────────────────────────────

# Envelope keys whose live value is a per-symbol dict and is MERGED into the
# current value rather than replaced wholesale.
_ENVELOPE_PER_SYMBOL_KEYS: frozenset[str] = frozenset({
    "STRUCT_SYMBOL_MIN_CONFIDENCE",
})
# Envelope keys whose live value is a nested per-symbol-per-side dict.
_ENVELOPE_NESTED_KEYS: frozenset[str] = frozenset({
    "STRUCT_SYMBOL_DIRECTIONAL_CONF",
})


def apply_envelopes(
    *, overrides: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Two-sided, range-checked apply of envelope keys for the live loader.

    ``current`` maps each envelope key (== the live global's name) to its
    current value: a scalar, a per-symbol dict, or a nested per-symbol-per-
    side dict. For every envelope key present in ``overrides``:

      * the WHOLE proposed value is range-checked via ``check_envelope`` —
        if any numeric leaf falls outside [min, max] the key is SKIPPED
        (logged) and the current value is preserved (corrupt/out-of-range
        file is ignored at load, satisfying the P3 re-validation rule);
      * scalars replace the current value; per-symbol / nested dicts are
        MERGED into the current value (so successive applies accumulate);
      * direction is irrelevant — a value inside the envelope applies
        whether it raises or lowers the current value.

    Returns ``{"values": {key: new_value}, "applied": [...], "skipped": [...]}``
    where ``values`` holds only the envelope keys that changed. Inputs are
    treated as read-only.
    """
    overrides = _coerce_overrides(overrides)
    values: dict[str, Any] = {}
    applied: list[str] = []
    skipped: list[str] = []

    for key in ENVELOPES:
        if key not in overrides:
            continue
        proposed = overrides[key]
        chk = check_envelope(key, proposed)
        if not chk.ok:
            skipped.append(chk.reason)
            continue
        cur = current.get(key)

        if key in _ENVELOPE_PER_SYMBOL_KEYS and isinstance(proposed, dict):
            merged = dict(cur or {})
            for sym, val in proposed.items():
                su = str(sym).upper()
                merged[su] = float(val)
                applied.append(f"{key}[{su}]→{float(val):.4f}")
            values[key] = merged

        elif key in _ENVELOPE_NESTED_KEYS and isinstance(proposed, dict):
            merged = {s: dict(d) for s, d in (cur or {}).items()}
            for sym, sides in proposed.items():
                if not isinstance(sides, dict):
                    skipped.append(f"{key}[{sym}] not a dict")
                    continue
                su = str(sym).upper()
                for side, val in sides.items():
                    sd = str(side).upper()
                    merged.setdefault(su, {})[sd] = float(val)
                    applied.append(f"{key}[{su}][{sd}]→{float(val):.4f}")
            values[key] = merged

        else:
            # Scalar knob.
            newv = float(proposed)  # check_envelope already proved numeric
            try:
                curf = float(cur) if cur is not None else None
            except (TypeError, ValueError):
                curf = None
            if curf is not None and newv == curf:
                skipped.append(f"{key} {newv:g} equals current (no-op)")
                continue
            values[key] = newv
            applied.append(
                f"{key} {('∅' if curf is None else f'{curf:g}')}→{newv:g}"
            )

    return {"values": values, "applied": applied, "skipped": skipped}


def check_apply_allowed(
    *,
    overrides: dict[str, Any],
    baseline_legacy: dict[str, Any] | None = None,
) -> list[str]:
    """Apply-time guard for the P3 two-sided surface (used by live_apply).

    Returns a list of violations; an empty list means the override is safe
    to apply. The guard partitions ``overrides`` into:

      * ``BLOCKLIST_REMOVE_KEY`` — never auto-applyable (Chen-only); always
        a violation here (the orchestrator escalates it to Telegram instead);
      * envelope keys — range-checked via ``check_envelope`` (two-sided);
      * legacy keys (floors / blocklist-add) — checked with the existing
        monotonic-tightening guard when ``baseline_legacy`` is available;
      * anything else — an unknown key, rejected.

    ``baseline_legacy`` is the pristine legacy-constant dict from
    ``live_apply.read_baseline_constants`` (keys: min_confidence,
    symbol_min_confidence, symbol_directional_conf, symbol_side_blocklist).
    When it is None (source parse failed) the tighten sub-check is skipped
    and the loader's runtime tightening guard is relied upon instead — but
    the envelope range check, blocklist-removal block, and unknown-key
    detection still run.
    """
    ov = _coerce_overrides(overrides)
    if not ov:
        return ["override is empty — nothing to apply"]

    violations: list[str] = []
    if BLOCKLIST_REMOVE_KEY in ov:
        violations.append(
            f"{BLOCKLIST_REMOVE_KEY} is Chen-only — cannot be auto-applied "
            f"(must be escalated to Telegram)"
        )

    env_keys = {k for k in ov if is_envelope_key(k)}
    for key in sorted(env_keys):
        chk = check_envelope(key, ov[key])
        if not chk.ok:
            violations.append(chk.reason)

    non_env = {
        k: v for k, v in ov.items()
        if k not in env_keys and k != BLOCKLIST_REMOVE_KEY
    }
    # Unknown-key detection, independent of baseline availability.
    for key in sorted(non_env):
        if key not in APPLYABLE_KEYS:
            violations.append(
                f"unknown key {key!r} (not an envelope key or "
                f"entry-suppression gate)"
            )

    # Legacy tighten check for the recognised floor/blocklist keys.
    if baseline_legacy is not None and non_env:
        legacy = check_tightening_only(overrides=non_env, **baseline_legacy)
        for v in legacy:
            if "unknown key" in v:
                continue  # already reported above
            # If envelopes carry the change, a no-op legacy set is fine.
            if v == "override has no applicable tightening effect" and env_keys:
                continue
            violations.append(v)

    return violations
