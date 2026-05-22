"""Anthropic API client wrapper for the self-improvement agents.

Responsibilities:
  * Wrap the anthropic SDK behind a small typed interface
  * Track every call in the `agent_runs` table (cost audit)
  * Enforce a per-day budget (PLAN.md §12 — defaults to $15 soft cap,
    $30 hard cap; configurable via env vars)
  * Gracefully degrade when ANTHROPIC_API_KEY is missing — returns a
    StubResponse with `degraded=True` so callers can fall back to
    deterministic-only behavior without crashing
  * Use prompt caching automatically on the system block so the agent
    personas (loaded from .agent/agents/*.md) only pay full input
    cost on first call of a session

The Risk Officer, Reviewer, and (later) Researcher all go through this
client. Tests inject a stub via the LLM_CLIENT_FACTORY env var so they
never hit the network.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Default models per PLAN.md §3.2
MODEL_HAIKU = "claude-haiku-4-5"
MODEL_OPUS = "claude-opus-4-7"

# Per-day budget caps (PLAN.md §12, default $15 soft / $30 hard).
# Used by `check_budget()` before any LLM call.
SOFT_CAP_USD_DEFAULT = 15.0
HARD_CAP_USD_DEFAULT = 30.0


# Pricing as of 2026 — used to estimate per-call cost from token counts.
# These are USD per million tokens. Caller does not need to think about
# these; the cost is computed automatically.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    MODEL_HAIKU: {"input": 1.0,  "output": 5.0,  "cache_write": 1.25, "cache_read": 0.10},
    MODEL_OPUS:  {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
}


@dataclass(frozen=True)
class LLMResponse:
    """Returned by every LLMClient.call. JSON-friendly for storage."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    duration_s: float
    degraded: bool = False           # True if API key missing / fallback used
    stop_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BudgetState:
    """Snapshot of today's cumulative spend (USD). Computed from agent_runs."""

    spend_usd: float
    soft_cap_usd: float
    hard_cap_usd: float
    n_calls: int

    @property
    def over_soft(self) -> bool:
        return self.spend_usd >= self.soft_cap_usd

    @property
    def over_hard(self) -> bool:
        return self.spend_usd >= self.hard_cap_usd


@dataclass
class CallContext:
    """Optional metadata to attach to the agent_runs row for this call.

    Caller (Risk Officer / Reviewer / Researcher) supplies what it knows.
    """

    agent: str
    decision_id: Optional[int] = None
    context_summary: str = ""


def _today_iso() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _calc_cost_usd(model: str, usage: dict[str, int]) -> float:
    """Cost calculation from usage dict (anthropic SDK shape)."""
    rates = PRICING_USD_PER_MTOK.get(model)
    if rates is None:
        # Unknown model — bill at Opus rates to be conservative
        rates = PRICING_USD_PER_MTOK[MODEL_OPUS]
    return (
        usage.get("input_tokens", 0) / 1_000_000 * rates["input"]
        + usage.get("output_tokens", 0) / 1_000_000 * rates["output"]
        + usage.get("cache_creation_input_tokens", 0) / 1_000_000 * rates["cache_write"]
        + usage.get("cache_read_input_tokens", 0) / 1_000_000 * rates["cache_read"]
    )


class BudgetExceeded(RuntimeError):
    """Raised when an LLM call would exceed the hard-cap budget."""


class LLMClient:
    """Single point of truth for outbound Claude API calls.

    Construct one per process (cheap). Each call() writes one row to
    `agent_runs` for audit.
    """

    def __init__(
        self,
        *,
        db_path: str | Path = "data/trading.db",
        api_key: Optional[str] = None,
        soft_cap_usd: Optional[float] = None,
        hard_cap_usd: Optional[float] = None,
    ):
        self.db_path = Path(db_path)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self._soft = (
            float(soft_cap_usd)
            if soft_cap_usd is not None
            else float(os.environ.get("SELF_IMPROVE_SOFT_CAP_USD", SOFT_CAP_USD_DEFAULT))
        )
        self._hard = (
            float(hard_cap_usd)
            if hard_cap_usd is not None
            else float(os.environ.get("SELF_IMPROVE_HARD_CAP_USD", HARD_CAP_USD_DEFAULT))
        )
        self._anthropic = None  # lazy import

    # ── public ──────────────────────────────────────────────────────────

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def budget_state(self) -> BudgetState:
        """Today's spend so far (UTC day) and the cap configuration."""
        if not self.db_path.exists():
            return BudgetState(0.0, self._soft, self._hard, 0)
        today = _today_iso()
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n,
                       COALESCE(
                         SUM(
                           CASE
                             WHEN json_valid(output_summary)
                              AND json_extract(output_summary, '$.cost_usd') IS NOT NULL
                             THEN CAST(json_extract(output_summary, '$.cost_usd') AS REAL)
                             ELSE 0
                           END
                         ), 0) AS spend
                FROM agent_runs
                WHERE substr(ts, 1, 10) = ?
                """,
                (today,),
            ).fetchone()
            n = int(row[0] or 0)
            spend = float(row[1] or 0.0)
        return BudgetState(spend, self._soft, self._hard, n)

    def check_budget(self, *, raise_on_hard: bool = True) -> BudgetState:
        """Return current budget. Raise BudgetExceeded if over hard cap
        (and raise_on_hard=True)."""
        state = self.budget_state()
        if raise_on_hard and state.over_hard:
            raise BudgetExceeded(
                f"daily LLM budget exceeded: spend=${state.spend_usd:.2f} "
                f">= hard cap ${state.hard_cap_usd:.2f}; "
                f"calls today={state.n_calls}"
            )
        return state

    def call(
        self,
        *,
        ctx: CallContext,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        cache_system: bool = True,
        timeout: float = 60.0,
    ) -> LLMResponse:
        """Make one Claude API call.

        Returns an LLMResponse. If ANTHROPIC_API_KEY is missing or any
        exception occurs, returns a degraded LLMResponse with the error
        text in `error`. The caller (Risk Officer / Reviewer) decides
        how to handle the degraded case — typically by falling back to
        the deterministic-only verdict.
        """
        start = time.perf_counter()

        if not self._api_key:
            resp = LLMResponse(
                text="",
                model=model,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=0.0,
                duration_s=time.perf_counter() - start,
                degraded=True,
                error="ANTHROPIC_API_KEY not set — running deterministic-only",
            )
            self._record_run(ctx, model, resp)
            return resp

        # Enforce hard cap before the call. Soft cap is informational.
        try:
            self.check_budget(raise_on_hard=True)
        except BudgetExceeded as exc:
            resp = LLMResponse(
                text="",
                model=model,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=0.0,
                duration_s=time.perf_counter() - start,
                degraded=True,
                error=str(exc),
            )
            self._record_run(ctx, model, resp)
            return resp

        try:
            client = self._client()
            system_block: Any
            if cache_system:
                system_block = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                system_block = system

            api_resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_block,
                messages=[{"role": "user", "content": user}],
                timeout=timeout,
            )

            usage = {
                "input_tokens": getattr(api_resp.usage, "input_tokens", 0),
                "output_tokens": getattr(api_resp.usage, "output_tokens", 0),
                "cache_creation_input_tokens": getattr(
                    api_resp.usage, "cache_creation_input_tokens", 0
                ) or 0,
                "cache_read_input_tokens": getattr(
                    api_resp.usage, "cache_read_input_tokens", 0
                ) or 0,
            }
            cost = _calc_cost_usd(model, usage)
            text = ""
            for block in api_resp.content:
                if getattr(block, "type", None) == "text":
                    text += getattr(block, "text", "")

            resp = LLMResponse(
                text=text,
                model=model,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cache_read_tokens=usage["cache_read_input_tokens"],
                cache_write_tokens=usage["cache_creation_input_tokens"],
                cost_usd=cost,
                duration_s=time.perf_counter() - start,
                degraded=False,
                stop_reason=getattr(api_resp, "stop_reason", None),
            )
        except Exception as exc:  # noqa: BLE001 — network failures fall through
            resp = LLMResponse(
                text="",
                model=model,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=0.0,
                duration_s=time.perf_counter() - start,
                degraded=True,
                error=f"{type(exc).__name__}: {exc}",
            )

        self._record_run(ctx, model, resp)
        return resp

    # ── internal ────────────────────────────────────────────────────────

    def _client(self):
        if self._anthropic is None:
            import anthropic  # local import — keeps import-time light
            self._anthropic = anthropic.Anthropic(api_key=self._api_key)
        return self._anthropic

    def _record_run(self, ctx: CallContext, model: str, resp: LLMResponse) -> None:
        """Append one row to agent_runs with the audit data."""
        if not self.db_path.exists():
            # Tests may use a non-existent path — no-op rather than crash.
            return
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_runs(
                        ts, agent, model, duration_s, input_tokens,
                        output_tokens, decision_id, context_summary,
                        output_summary, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _utc_iso(),
                        ctx.agent,
                        model,
                        resp.duration_s,
                        resp.input_tokens,
                        resp.output_tokens,
                        ctx.decision_id,
                        ctx.context_summary[:5000],
                        json.dumps({
                            "cost_usd": resp.cost_usd,
                            "cache_read_tokens": resp.cache_read_tokens,
                            "cache_write_tokens": resp.cache_write_tokens,
                            "degraded": resp.degraded,
                            "stop_reason": resp.stop_reason,
                            "text_excerpt": resp.text[:500],
                        }),
                        resp.error,
                    ),
                )
        except Exception:
            # The audit row failing must NEVER crash the call path.
            pass


# ─────────────────────────────────────────────────────────────────────────
# Test injection point
# ─────────────────────────────────────────────────────────────────────────


_CLIENT_FACTORY: Callable[[], LLMClient] | None = None


def set_client_factory(factory: Callable[[], LLMClient] | None) -> None:
    """Install a custom factory so tests can substitute a stub client."""
    global _CLIENT_FACTORY
    _CLIENT_FACTORY = factory


def default_client(db_path: str | Path = "data/trading.db") -> LLMClient:
    """Return a singleton-style client; respects the test injection point."""
    if _CLIENT_FACTORY is not None:
        return _CLIENT_FACTORY()
    return LLMClient(db_path=db_path)


def load_persona(role: str) -> str:
    """Load the persona text from `.agent/agents/<role>.md`.

    Falls back to an empty string if the file isn't present — the
    caller decides whether that's a hard error.
    """
    candidates = [
        Path(".agent/agents") / f"{role}.md",
        Path(".agent/agents") / f"{role}.py",  # historical: some roles were .py
    ]
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8")
    return ""
