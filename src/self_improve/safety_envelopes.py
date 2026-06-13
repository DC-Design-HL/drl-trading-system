"""Safety envelopes for the autonomous apply surface (PROFITABILITY_PLAN.md P3).

Single source of truth for which config knobs the self-improvement loop may
move and the HARD [min, max] range each may move within. This turns the loop
from a loss-suppressor into a bounded optimiser — it may move an envelope key
in EITHER direction, but only inside its range.

Safety model (do not relax without Chen's sign-off):
  * A key absent from ``ENVELOPES`` is NOT applyable. Sizing, leverage,
    SL/TP geometry, per-trade notional, and halt controls are structurally
    absent here — the loop can never touch them.
  * Blocklist ADD is allowed via a separate path (tighten-only semantics).
    Blocklist REMOVAL stays Chen-only: the researcher may propose it, the
    orchestrator must escalate it to Telegram for a YES/NO.
  * The apply-time guard checks RANGE, not direction. For envelope keys this
    supersedes the old monotonic-tightening rule (lowering a STRUCT_* floor
    within range is allowed because forward-sim + paper + canary + circuit
    breaker validate it).

Each entry maps a key to ``(min, max, validation_engine)`` where the engine
is which simulator validates a proposed change to that key:
  * "replay"  — replay harness can express it (entry-side thresholds);
  * "forward" — needs the forward simulator (exit / timing knobs).

NOTE: the ranges below are PROPOSALS pending Chen's approval on Telegram;
P3 must not merge until that table is confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass

VALIDATION_REPLAY = "replay"
VALIDATION_FORWARD = "forward"

# key -> (min, max, validation_engine)
ENVELOPES: dict[str, tuple[float, float, str]] = {
    "STRUCT_MIN_CONFIDENCE":          (0.0,   0.95,  VALIDATION_REPLAY),
    "STRUCT_SYMBOL_MIN_CONFIDENCE":   (0.0,   0.95,  VALIDATION_REPLAY),
    "STRUCT_SYMBOL_DIRECTIONAL_CONF": (0.0,   0.95,  VALIDATION_REPLAY),
    "TRAILING_DISTANCE_PCT":          (0.003, 0.010, VALIDATION_FORWARD),
    "TRAILING_BREAKEVEN_PCT":         (0.005, 0.015, VALIDATION_FORWARD),
    "STAGNANT_HOURS":                 (4.0,   12.0,  VALIDATION_FORWARD),
    "COOLDOWN_SECONDS":               (900,   7200,  VALIDATION_FORWARD),
    "MIN_HOLD_SECONDS":               (1800,  7200,  VALIDATION_FORWARD),
    "WHIPSAW_COOLDOWN_HOURS":         (1.0,   6.0,   VALIDATION_FORWARD),
    "ADX_GUARD_MIN":                  (15.0,  30.0,  VALIDATION_FORWARD),
    "EXHAUSTION_ATR_THRESHOLD":       (2.0,   4.0,   VALIDATION_FORWARD),
}

# Key that carries blocklist additions (tighten-only; not range-checked here).
BLOCKLIST_ADD_KEY = "SYMBOL_SIDE_BLOCKLIST_ADD"
# Key that would REMOVE from the blocklist — never auto-applied.
BLOCKLIST_REMOVE_KEY = "SYMBOL_SIDE_BLOCKLIST_REMOVE"


@dataclass(frozen=True)
class EnvelopeCheck:
    ok: bool
    reason: str


def is_envelope_key(key: str) -> bool:
    return key in ENVELOPES


def validation_engine(key: str) -> str:
    """Which simulator validates a change to ``key``. Raises on unknown key."""
    return ENVELOPES[key][2]


def _leaf_values(value: object) -> list[float]:
    """Flatten a scalar / per-symbol dict / per-symbol-per-side dict into the
    list of numeric leaves to range-check. Raises ValueError on a non-numeric
    leaf so a malformed proposal is rejected rather than silently passed."""
    if isinstance(value, dict):
        leaves: list[float] = []
        for v in value.values():
            leaves.extend(_leaf_values(v))
        return leaves
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        raise ValueError("boolean is not a valid envelope value")
    if isinstance(value, (int, float)):
        return [float(value)]
    raise ValueError(f"non-numeric envelope value: {value!r}")


def check_envelope(key: str, value: object) -> EnvelopeCheck:
    """Range-check a proposed apply of ``key`` -> ``value``.

    Accepts a scalar or a (nested) per-symbol dict; every numeric leaf must
    lie within the key's [min, max]. Returns an EnvelopeCheck — callers
    REJECT the apply when ``ok`` is False (and must never apply unknown keys).
    """
    if key not in ENVELOPES:
        return EnvelopeCheck(
            False, f"key '{key}' is not in ENVELOPES — not applyable")
    lo, hi, _engine = ENVELOPES[key]
    try:
        leaves = _leaf_values(value)
    except ValueError as exc:
        return EnvelopeCheck(False, f"{key}: {exc}")
    if not leaves:
        return EnvelopeCheck(False, f"{key}: no numeric value to check")
    for v in leaves:
        if v < lo or v > hi:
            return EnvelopeCheck(
                False,
                f"{key}={v} outside envelope [{lo}, {hi}]")
    return EnvelopeCheck(True, f"{key} within [{lo}, {hi}]")


def allowed_areas_text() -> str:
    """Researcher-prompt 'allowed areas' block generated FROM the envelope
    table — the single source of truth, so the prompt can never drift from
    what the apply guard actually permits (the bug class fixed in 08512ea).
    """
    lines = [
        "You may propose changes ONLY to these keys, within the stated "
        "range (either direction inside the range is allowed):",
    ]
    for key in sorted(ENVELOPES):
        lo, hi, engine = ENVELOPES[key]
        lines.append(f"  - {key}: [{lo}, {hi}]  (validated by {engine})")
    lines.append(
        "You may also propose ADDING a (symbol, side) to the blocklist. "
        "REMOVING from the blocklist is not yours to apply — propose it and "
        "it will be escalated to Chen for a yes/no. Any other key "
        "(sizing, leverage, SL/TP, notional, halts) is off-limits and will "
        "be rejected at apply time.")
    return "\n".join(lines)
